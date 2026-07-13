/*
 * Gemma4-optimized paged attention decode kernel.
 *
 * Online softmax, multi-warp split-KV within each CTA.
 * Grid: (num_q_heads, num_seqs). No cross-CTA splits — all work done in one CTA.
 * Block: NUM_WARPS * WARP_SIZE. Each warp handles a disjoint KV range and
 * produces a partial (acc, M, L). A final serial reduce across warps merges
 * them in shared memory — one __syncthreads at the end, not per-tile.
 *
 * NHD KV cache: (num_blocks, block_size, num_kv_heads, head_size)
 */
#pragma once

#include "../../attention/attention_dtypes.h"
#include "../../attention/attention_generic.cuh"
#include "../../cuda_compat.h"

#ifndef USE_ROCM
  #include "../../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#else
  #include <hip/hip_bf16.h>
  #include "../../quantization/w8a8/fp8/amd/quant_utils.cuh"
typedef __hip_bfloat16 __nv_bfloat16;
#endif

#include <float.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>

#define GEMMA_CDIV(a, b) (((a) + (b) - 1) / (b))

namespace vllm {
namespace gemma {

using namespace nvcuda;
static constexpr float LOG2E = 1.4426950408889634f;

// ---- sm80 mma.sync helpers for the lean tensor-core decode (GEMMA_DECODE_MMA).
// Validated layouts in /tmp/mma_attn_test.cu (full attention, maxerr 3.7e-4).
__device__ __forceinline__ uint32_t mma_smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mma_ldm_x4(uint32_t* r, const void* p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(mma_smem_addr(p)));
}
__device__ __forceinline__ void mma_ldm_x2(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(mma_smem_addr(p)));
}
__device__ __forceinline__ void mma_ldm_x4t(uint32_t* r, const void* p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(mma_smem_addr(p)));
}
// D[16x8] += A[16x16] * B[16x8].  A row-major (MxK), B col-major (KxN), f32 acc.
__device__ __forceinline__ void mma_m16n8k16(float* d, const uint32_t* a,
                                             const uint32_t* b,
                                             const float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

inline __device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
}

// Warp-wide argmax (value, index); ties -> lower index. Used by top-k select.
inline __device__ void warp_argmax(float& val, int& idx) {
#pragma unroll
  for (int o = WARP_SIZE / 2; o >= 1; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, val, o);
    const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (ov > val || (ov == val && oi < idx)) { val = ov; idx = oi; }
  }
}

// Lossy top-k block SELECTION (P2, Quest-style) — EXACT scoring variant.
// One CTA per (kv_head, seq); grid (num_kv_heads, num_seqs). Computes, per KV
// block, score = max over {tokens in block} x {GROUP query heads} of q.k, then
// writes the top-`num_sel` block indices (sink + recent window forced in) to
// selected_tiles[seq, kv_head, :]. The decode kernel then walks that list.
// This reads K (the QK part) -> a quality reference; the maintained-bounds
// scoring (no full-K read = the speed win) is the follow-up, same output API.
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int GROUP, int NUM_WARPS>
__global__ void gemma_topk_select_kernel(
    int* __restrict__ selected_tiles,           // [num_seqs,num_kv_heads,num_sel]
    const scalar_t* __restrict__ q,             // [num_seqs,num_q_heads,HEAD_SIZE]
    const cache_t* __restrict__ k_cache,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int64_t kv_stride_block,
    const int64_t kv_stride_slot,
    const int64_t kv_stride_head,
    const int num_sel, const int sink_tiles, const int win_tiles) {
  const int kv_head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int num_kv_heads = gridDim.x;
  const int seq_len = seq_lens[seq_idx];
  const int n_blocks = GEMMA_CDIV(seq_len, BLOCK_SIZE);
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  constexpr int EPT = HEAD_SIZE / WARP_SIZE;
  static_assert(HEAD_SIZE % WARP_SIZE == 0);
  const int dim_start = lane * EPT;

  extern __shared__ float s_scores[];           // [n_blocks]

  // Load this lane's EPT dims of all GROUP query heads of this kv_head.
  float q_regs[GROUP][EPT];
#pragma unroll
  for (int g = 0; g < GROUP; g++) {
    const scalar_t* qp = q + (int64_t)seq_idx * q_stride
                           + (int64_t)(kv_head * GROUP + g) * HEAD_SIZE
                           + dim_start;
#pragma unroll
    for (int e = 0; e < EPT; e++) q_regs[g][e] = static_cast<float>(qp[e]);
  }

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // Phase 1: per-block score (warp-strided over blocks).
  for (int blk = warp_idx; blk < n_blocks; blk += NUM_WARPS) {
    const int64_t phys = block_table[blk];
    const int ntok = min(BLOCK_SIZE, seq_len - blk * BLOCK_SIZE);
    float bmax = -FLT_MAX;
    for (int slot = 0; slot < ntok; slot++) {
      const cache_t* kp = k_cache + phys * kv_stride_block
                                   + slot * kv_stride_slot
                                   + (int64_t)kv_head * kv_stride_head
                                   + dim_start;
      float kv[EPT];
#pragma unroll
      for (int e = 0; e < EPT; e++) kv[e] = static_cast<float>(kp[e]);
#pragma unroll
      for (int g = 0; g < GROUP; g++) {
        float qk = 0.f;
#pragma unroll
        for (int e = 0; e < EPT; e++) qk += q_regs[g][e] * kv[e];
        qk = warp_reduce_sum(qk);
        bmax = fmaxf(bmax, qk);
      }
    }
    if (lane == 0) s_scores[blk] = bmax * scale;
  }
  __syncthreads();

  // Force sinks [0,sink_tiles) and recent window [n_blocks-win_tiles,n_blocks).
  for (int t = threadIdx.x; t < n_blocks; t += blockDim.x)
    if (t < sink_tiles || t >= n_blocks - win_tiles) s_scores[t] = FLT_MAX;
  __syncthreads();

  // Phase 2: pick top-`num_sel` (warp 0). Caller gates num_sel <= n_blocks; the
  // pad branch only fires for degenerate short seqs (then ~full attention).
  if (warp_idx == 0) {
    int* outp = selected_tiles
              + ((int64_t)seq_idx * num_kv_heads + kv_head) * num_sel;
    const int cnt = (num_sel < n_blocks) ? num_sel : n_blocks;
    for (int i = 0; i < cnt; i++) {
      float bv = -FLT_MAX;
      int bi = n_blocks;  // sentinel (> any valid index for tie-break)
      for (int t = lane; t < n_blocks; t += WARP_SIZE) {
        const float v = s_scores[t];
        if (v > bv || (v == bv && t < bi)) { bv = v; bi = t; }
      }
      warp_argmax(bv, bi);
      if (lane == 0) { outp[i] = bi; s_scores[bi] = -FLT_MAX; }
      __syncwarp();
    }
    if (lane == 0)
      for (int i = cnt; i < num_sel; i++) outp[i] = (cnt > 0) ? outp[cnt - 1] : 0;
  }
}

// Maintain per-block channel min/max key bounds (for the bounds-scoring top-k =
// the SPEED path: scoring reads these small bounds, NOT the full K). Recomputes
// only the TOUCHED blocks (passed as uniq_blocks/ntoks) from the cache after a
// KV write -> correct, race-free, and recycling-safe (freed/reused blocks are
// recomputed from offset 0 on their first new write). Layout
// block_bounds[num_blocks, 2(min,max), num_kv_heads, HEAD_SIZE], stored in the
// cache dtype (bf16) — lossless (min/max of bf16 keys are bf16) and halves the
// scoring read vs fp32.
template <typename cache_t, int HEAD_SIZE>
__global__ void gemma_update_kv_bounds_kernel(
    cache_t* __restrict__ block_bounds,
    const cache_t* __restrict__ k_cache,
    const int* __restrict__ uniq_blocks,   // [M] physical block ids touched
    const int* __restrict__ ntoks,         // [M] valid tokens in each block
    const int num_kv_heads,
    const int64_t kv_stride_block,
    const int64_t kv_stride_slot,
    const int64_t kv_stride_head) {
  const int m = blockIdx.x;
  const int phys = uniq_blocks[m];
  const int ntok = ntoks[m];
  const int plane = num_kv_heads * HEAD_SIZE;   // min/max plane stride
  const int total = plane;                      // channels = nkv * HEAD_SIZE
  const int64_t bb_block = (int64_t)2 * plane;
  for (int c = threadIdx.x; c < total; c += blockDim.x) {
    const int kvh = c / HEAD_SIZE;
    const int d = c % HEAD_SIZE;
    const cache_t* base = k_cache + (int64_t)phys * kv_stride_block
                                   + (int64_t)kvh * kv_stride_head + d;
    float vmin = FLT_MAX, vmax = -FLT_MAX;
    for (int s = 0; s < ntok; s++) {
      const float v = static_cast<float>(base[(int64_t)s * kv_stride_slot]);
      vmin = fminf(vmin, v);
      vmax = fmaxf(vmax, v);
    }
    cache_t* bb = block_bounds + (int64_t)phys * bb_block + c;
    bb[0] = static_cast<cache_t>(vmin);          // min plane
    bb[plane] = static_cast<cache_t>(vmax);      // max plane
  }
}

// Bounds-scoring top-k SELECTION (the SPEED path). Same output as
// gemma_topk_select_kernel but the per-block score is the Quest upper bound
// sum_d (q_d>0 ? q_d*max_d : q_d*min_d) read from the maintained bounds (no
// full-K read). Validated to select identically to exact scoring (synthetic).
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int GROUP,
          int NUM_WARPS>
__global__ void gemma_topk_select_bounds_kernel(
    int* __restrict__ selected_tiles,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ block_bounds,  // [num_blocks,2,nkv,HEAD_SIZE]
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int num_kv_heads,
    const int num_sel, const int sink_tiles, const int win_tiles) {
  const int kv_head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  const int n_blocks = GEMMA_CDIV(seq_len, BLOCK_SIZE);
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  constexpr int EPT = HEAD_SIZE / WARP_SIZE;
  static_assert(HEAD_SIZE % WARP_SIZE == 0);
  const int dim_start = lane * EPT;
  const int plane = num_kv_heads * HEAD_SIZE;
  const int64_t bb_block = (int64_t)2 * plane;

  extern __shared__ float s_scores[];

  float q_regs[GROUP][EPT];
#pragma unroll
  for (int g = 0; g < GROUP; g++) {
    const scalar_t* qp = q + (int64_t)seq_idx * q_stride
                           + (int64_t)(kv_head * GROUP + g) * HEAD_SIZE
                           + dim_start;
#pragma unroll
    for (int e = 0; e < EPT; e++) q_regs[g][e] = static_cast<float>(qp[e]);
  }

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  for (int blk = warp_idx; blk < n_blocks; blk += NUM_WARPS) {
    const int64_t phys = block_table[blk];
    const scalar_t* bmin = block_bounds + phys * bb_block
                         + (int64_t)kv_head * HEAD_SIZE + dim_start;
    const scalar_t* bmax = bmin + plane;
    float bmn[EPT], bmx[EPT];
#pragma unroll
    for (int e = 0; e < EPT; e++) {
      bmn[e] = static_cast<float>(bmin[e]);
      bmx[e] = static_cast<float>(bmax[e]);
    }
    float best = -FLT_MAX;
#pragma unroll
    for (int g = 0; g < GROUP; g++) {
      float part = 0.f;
#pragma unroll
      for (int e = 0; e < EPT; e++) {
        const float qd = q_regs[g][e];
        part += (qd > 0.f) ? qd * bmx[e] : qd * bmn[e];
      }
      part = warp_reduce_sum(part);
      best = fmaxf(best, part);
    }
    if (lane == 0) s_scores[blk] = best * scale;
  }
  __syncthreads();

  for (int t = threadIdx.x; t < n_blocks; t += blockDim.x)
    if (t < sink_tiles || t >= n_blocks - win_tiles) s_scores[t] = FLT_MAX;
  __syncthreads();

  // Phase 2: top-`num_sel` selection (warp 0; the smem scan is cheap relative
  // to the bounds read, so parallelizing it across warps doesn't pay for the
  // extra per-round barriers).
  if (warp_idx == 0) {
    int* outp = selected_tiles
              + ((int64_t)seq_idx * num_kv_heads + kv_head) * num_sel;
    const int cnt = (num_sel < n_blocks) ? num_sel : n_blocks;
    for (int i = 0; i < cnt; i++) {
      float bv = -FLT_MAX;
      int bi = n_blocks;
      for (int t = lane; t < n_blocks; t += WARP_SIZE) {
        const float v = s_scores[t];
        if (v > bv || (v == bv && t < bi)) { bv = v; bi = t; }
      }
      warp_argmax(bv, bi);
      if (lane == 0) { outp[i] = bi; s_scores[bi] = -FLT_MAX; }
      __syncwarp();
    }
    if (lane == 0)
      for (int i = cnt; i < num_sel; i++) outp[i] = (cnt > 0) ? outp[cnt - 1] : 0;
  }
}

// 128-bit (uint4) vectorized load of N contiguous cache_t elements into buf.
// Falls back to scalar when N is not a multiple of the 16-byte vector width.
template <typename cache_t, int N>
__device__ __forceinline__ void gemma_vec_load(
    const cache_t* __restrict__ src, cache_t* __restrict__ buf) {
  constexpr int VEC = 16 / sizeof(cache_t);
  if constexpr (N >= VEC && N % VEC == 0) {
#pragma unroll
    for (int c = 0; c < N / VEC; c++)
      reinterpret_cast<uint4*>(buf)[c] =
          reinterpret_cast<const uint4*>(src)[c];
  } else {
#pragma unroll
    for (int e = 0; e < N; e++) buf[e] = src[e];
  }
}

// Cooperatively issue cp.async (16B) loads of one KV block into a shared-memory
// buffer. Each KV element is fetched once; async so it overlaps compute.
template <typename cache_t, int ACTUAL_HEAD_SIZE, bool K_EQ_V>
__device__ __forceinline__ void gemma_stage_async(
    cache_t* __restrict__ smem_k, cache_t* __restrict__ smem_v,
    const cache_t* __restrict__ k_cache, const cache_t* __restrict__ v_cache,
    int64_t phys, int kv_head_idx, int end_tok,
    int64_t kv_stride_block, int64_t kv_stride_slot, int64_t kv_stride_head,
    int tid, int num_threads) {
  constexpr int VEC = 16 / sizeof(cache_t);
  const int vps = ACTUAL_HEAD_SIZE / VEC;
  const int n_vec = end_tok * vps;
  for (int iv = tid; iv < n_vec; iv += num_threads) {
    const int slot = iv / vps;
    const int dv = (iv - slot * vps) * VEC;
    const int64_t base = phys * kv_stride_block + slot * kv_stride_slot
                         + kv_head_idx * kv_stride_head + dv;
    __pipeline_memcpy_async(smem_k + slot * ACTUAL_HEAD_SIZE + dv,
                            k_cache + base, 16);
    if constexpr (!K_EQ_V)
      __pipeline_memcpy_async(smem_v + slot * ACTUAL_HEAD_SIZE + dv,
                              v_cache + base, 16);
  }
}

// Online-softmax over one staged KV block (in shared memory), updating the
// running (M, L, acc) for this warp's query head. Shared by both GQA kernels.
template <typename cache_t, int ACTUAL_HEAD_SIZE, int ELEMS_PER_THREAD,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool K_EQ_V>
__device__ __forceinline__ void gemma_block_compute(
    const cache_t* __restrict__ smem_k, const cache_t* __restrict__ smem_v,
    int end_tok, int dim_start, const float* __restrict__ q_regs,
    float ks, float vs, float scale_log2,
    float& M, float& L, float* __restrict__ acc) {
  for (int slot = 0; slot < end_tok; slot++) {
    const cache_t* ksm = smem_k + slot * ACTUAL_HEAD_SIZE + dim_start;
    cache_t kbuf[ELEMS_PER_THREAD];
    gemma_vec_load<cache_t, ELEMS_PER_THREAD>(ksm, kbuf);
    float kv_regs[ELEMS_PER_THREAD];
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++)
      kv_regs[e] = (KV_DTYPE == Fp8KVCacheDataType::kAuto)
                       ? static_cast<float>(kbuf[e])
                       : static_cast<float>(kbuf[e]) * ks;
    float qk = 0.f;
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) qk += q_regs[e] * kv_regs[e];
    qk = warp_reduce_sum(qk) * scale_log2;
    const float new_max = fmaxf(M, qk);
    const float alpha = exp2f(M - new_max);
    const float p = exp2f(qk - new_max);
    L = L * alpha + p;
    if constexpr (K_EQ_V) {
#pragma unroll
      for (int e = 0; e < ELEMS_PER_THREAD; e++)
        acc[e] = acc[e] * alpha + p * kv_regs[e];
    } else {
      const cache_t* vsm = smem_v + slot * ACTUAL_HEAD_SIZE + dim_start;
      cache_t vbuf[ELEMS_PER_THREAD];
      gemma_vec_load<cache_t, ELEMS_PER_THREAD>(vsm, vbuf);
#pragma unroll
      for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        float vf = (KV_DTYPE == Fp8KVCacheDataType::kAuto)
                       ? static_cast<float>(vbuf[e])
                       : static_cast<float>(vbuf[e]) * vs;
        acc[e] = acc[e] * alpha + p * vf;
      }
    }
    M = new_max;
  }
}

// Grid: (num_q_heads, num_seqs)
// Block: NUM_WARPS * WARP_SIZE
//
// Each warp independently processes a disjoint slice of KV blocks
// using online softmax. At the end, warp 0 reduces all partials
// via shared memory.
template <typename scalar_t, typename cache_t,
          int HEAD_SIZE, int ACTUAL_HEAD_SIZE,
          int BLOCK_SIZE, int NUM_WARPS,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          bool K_EQ_V, bool USE_SLIDING_WINDOW>
__global__ void gemma_flash_decode_kernel(
    scalar_t* __restrict__ out,          // [num_seqs, num_q_heads, HEAD_SIZE]
    const scalar_t* __restrict__ q,      // [num_seqs, num_q_heads, HEAD_SIZE]
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int64_t kv_stride_block,
    const int64_t kv_stride_slot,
    const int64_t kv_stride_head,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    const int sliding_window) {

  const int q_head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int num_q_heads = gridDim.x;
  const int num_queries_per_kv = num_q_heads / num_kv_heads;
  const int kv_head_idx = q_head_idx / num_queries_per_kv;

  const int seq_len = seq_lens[seq_idx];
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  const int num_seq_blocks = GEMMA_CDIV(seq_len, BLOCK_SIZE);
  int start_block = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int first = seq_len - sliding_window;
    start_block = (first > 0) ? (first / BLOCK_SIZE) : 0;
  }
  const int total_blocks = num_seq_blocks - start_block;

  // Split KV blocks across warps.
  const int blocks_per_warp = GEMMA_CDIV(total_blocks, NUM_WARPS);
  const int warp_start = start_block + warp_idx * blocks_per_warp;
  const int warp_end_raw = warp_start + blocks_per_warp;
  const int warp_end = (warp_end_raw < num_seq_blocks) ? warp_end_raw : num_seq_blocks;

  constexpr int ELEMS_PER_THREAD = ACTUAL_HEAD_SIZE / WARP_SIZE;
  static_assert(ACTUAL_HEAD_SIZE % WARP_SIZE == 0);
  const int dim_start = lane * ELEMS_PER_THREAD;

  // Load Q.
  const scalar_t* q_ptr = q + seq_idx * q_stride
                             + q_head_idx * HEAD_SIZE + dim_start;
  float q_regs[ELEMS_PER_THREAD];
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++)
    q_regs[e] = static_cast<float>(q_ptr[e]);

  float ks = 1.f, vs = 1.f;
  if constexpr (KV_DTYPE != Fp8KVCacheDataType::kAuto) {
    ks = *k_scale_ptr;
    vs = *v_scale_ptr;
  }

  const float scale_log2 = scale * LOG2E;

  // Per-warp online softmax.
  float M = -FLT_MAX;
  float L = 0.f;
  float acc[ELEMS_PER_THREAD];
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++) acc[e] = 0.f;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  for (int blk = warp_start; blk < warp_end; blk++) {
    const int64_t phys = static_cast<int64_t>(block_table[blk]);
    const int end_tok = (blk * BLOCK_SIZE + BLOCK_SIZE < seq_len)
                          ? BLOCK_SIZE : (seq_len - blk * BLOCK_SIZE);

    for (int slot = 0; slot < end_tok; slot++) {
      const cache_t* k_ptr = k_cache + phys * kv_stride_block
                                      + slot * kv_stride_slot
                                      + kv_head_idx * kv_stride_head
                                      + dim_start;
      float kv_regs[ELEMS_PER_THREAD];
#pragma unroll
      for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto)
          kv_regs[e] = static_cast<float>(k_ptr[e]);
        else
          kv_regs[e] = static_cast<float>(k_ptr[e]) * ks;
      }

      float qk = 0.f;
#pragma unroll
      for (int e = 0; e < ELEMS_PER_THREAD; e++)
        qk += q_regs[e] * kv_regs[e];
      qk = warp_reduce_sum(qk) * scale_log2;

      const float new_max = fmaxf(M, qk);
      const float alpha = exp2f(M - new_max);
      const float p = exp2f(qk - new_max);
      L = L * alpha + p;
#pragma unroll
      for (int e = 0; e < ELEMS_PER_THREAD; e++)
        acc[e] *= alpha;

      if constexpr (K_EQ_V) {
#pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++)
          acc[e] += p * kv_regs[e];
      } else {
        const cache_t* v_ptr = v_cache + phys * kv_stride_block
                                        + slot * kv_stride_slot
                                        + kv_head_idx * kv_stride_head
                                        + dim_start;
#pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
          float v_val;
          if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto)
            v_val = static_cast<float>(v_ptr[e]);
          else
            v_val = static_cast<float>(v_ptr[e]) * vs;
          acc[e] += p * v_val;
        }
      }
      M = new_max;
    }
  }

  // ---- Reduce across warps via shared memory ----
  // Each warp writes its (M, L, acc[]) to smem. Warp 0 merges.
  __shared__ float smem_M[NUM_WARPS];
  __shared__ float smem_L[NUM_WARPS];
  __shared__ float smem_acc[NUM_WARPS][ACTUAL_HEAD_SIZE];

  if (lane == 0) {
    smem_M[warp_idx] = M;
    smem_L[warp_idx] = L;
  }
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++)
    smem_acc[warp_idx][dim_start + e] = acc[e];

  __syncthreads();

  // Warp 0 does the final reduce and writes output.
  if (warp_idx == 0) {
    // Start from warp 0's values.
    float final_M = smem_M[0];
    float final_L = smem_L[0];
    float final_acc[ELEMS_PER_THREAD];
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++)
      final_acc[e] = smem_acc[0][dim_start + e];

    // Merge in other warps.
    for (int w = 1; w < NUM_WARPS; w++) {
      float w_M = smem_M[w];
      float w_L = smem_L[w];
      if (w_M <= -FLT_MAX) continue;

      float new_max = fmaxf(final_M, w_M);
      float scale_old = exp2f(final_M - new_max);
      float scale_new = exp2f(w_M - new_max);
      final_L = final_L * scale_old + w_L * scale_new;
#pragma unroll
      for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        final_acc[e] = final_acc[e] * scale_old
                     + smem_acc[w][dim_start + e] * scale_new;
      }
      final_M = new_max;
    }

    // Normalize and write.
    float inv_L = (final_L > 0.f) ? (1.f / final_L) : 0.f;
    scalar_t* out_ptr = out + seq_idx * num_q_heads * HEAD_SIZE
                            + q_head_idx * HEAD_SIZE + dim_start;
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++)
      from_float(out_ptr[e], final_acc[e] * inv_L);
  }
}

// ---------------------------------------------------------------------------
// GQA-group-reuse decode kernel (prototype).
//
// One CTA per (kv_head, seq); GQA_GROUP warps, one query head per warp.
// Each KV block is staged once into shared memory by the whole CTA, then every
// warp computes its query head against the staged block. So each KV element is
// read from DRAM ONCE per (kv_head, seq, block) and reused across all GQA_GROUP
// query heads — vs the baseline kernel which reloads KV once per (q_head, seq).
// Because every warp performs the full KV scan, no cross-warp merge is needed.
//
// Grid:  (num_kv_heads, num_seqs)
// Block: GQA_GROUP * WARP_SIZE
// Smem:  BLOCK_SIZE * ACTUAL_HEAD_SIZE * sizeof(cache_t) * (K_EQ_V ? 1 : 2)
// ---------------------------------------------------------------------------
template <typename scalar_t, typename cache_t,
          int HEAD_SIZE, int ACTUAL_HEAD_SIZE,
          int BLOCK_SIZE, int GQA_GROUP,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          bool K_EQ_V, bool USE_SLIDING_WINDOW>
__global__ void gemma_gqa_decode_kernel(
    scalar_t* __restrict__ out,          // [num_seqs, num_q_heads, HEAD_SIZE]
    const scalar_t* __restrict__ q,      // [num_seqs, num_q_heads, HEAD_SIZE]
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int64_t kv_stride_block,
    const int64_t kv_stride_slot,
    const int64_t kv_stride_head,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    const int sliding_window) {

  const int kv_head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int num_q_heads = gridDim.x * GQA_GROUP;
  const int warp_idx = threadIdx.x / WARP_SIZE;   // query head within group
  const int lane = threadIdx.x % WARP_SIZE;
  const int q_head_idx = kv_head_idx * GQA_GROUP + warp_idx;

  const int seq_len = seq_lens[seq_idx];
  const int num_seq_blocks = GEMMA_CDIV(seq_len, BLOCK_SIZE);
  int start_block = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int first = seq_len - sliding_window;
    start_block = (first > 0) ? (first / BLOCK_SIZE) : 0;
  }

  constexpr int ELEMS_PER_THREAD = ACTUAL_HEAD_SIZE / WARP_SIZE;
  static_assert(ACTUAL_HEAD_SIZE % WARP_SIZE == 0);
  const int dim_start = lane * ELEMS_PER_THREAD;

  // Load this warp's query head into registers.
  const scalar_t* q_ptr = q + seq_idx * q_stride
                            + q_head_idx * HEAD_SIZE + dim_start;
  float q_regs[ELEMS_PER_THREAD];
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++)
    q_regs[e] = static_cast<float>(q_ptr[e]);

  float ks = 1.f, vs = 1.f;
  if constexpr (KV_DTYPE != Fp8KVCacheDataType::kAuto) {
    ks = *k_scale_ptr;
    vs = *v_scale_ptr;
  }
  const float scale_log2 = scale * LOG2E;

  // Shared memory: stage one KV block. V aliases K when K_EQ_V.
  extern __shared__ char gqa_smem_raw[];
  cache_t* smem_base = reinterpret_cast<cache_t*>(gqa_smem_raw);
  constexpr int KBLK = BLOCK_SIZE * ACTUAL_HEAD_SIZE;
  constexpr int BUFSZ = (K_EQ_V ? 1 : 2) * KBLK;  // elems per double-buffer slot
  cache_t* buf_k[2] = {smem_base, smem_base + BUFSZ};
  cache_t* buf_v[2] = {
      K_EQ_V ? smem_base : (smem_base + KBLK),
      K_EQ_V ? (smem_base + BUFSZ) : (smem_base + BUFSZ + KBLK)};

  float M = -FLT_MAX;
  float L = 0.f;
  float acc[ELEMS_PER_THREAD];
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++) acc[e] = 0.f;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int num_threads = GQA_GROUP * WARP_SIZE;
  const int tid = threadIdx.x;

#define GQA_ENDTOK(blk) \
  (((blk) * BLOCK_SIZE + BLOCK_SIZE < seq_len) ? BLOCK_SIZE \
                                               : (seq_len - (blk) * BLOCK_SIZE))

  constexpr int STG_VEC = 16 / sizeof(cache_t);
  if constexpr (ACTUAL_HEAD_SIZE % STG_VEC == 0) {
    // Double-buffered cp.async staging: prefetch next block while computing.
    const int lo = start_block, hi = num_seq_blocks;
    if (lo < hi) {
      gemma_stage_async<cache_t, ACTUAL_HEAD_SIZE, K_EQ_V>(
          buf_k[0], buf_v[0], k_cache, v_cache,
          static_cast<int64_t>(block_table[lo]), kv_head_idx, GQA_ENDTOK(lo),
          kv_stride_block, kv_stride_slot, kv_stride_head, tid, num_threads);
      __pipeline_commit();
      for (int i = lo; i < hi; i++) {
        const int cur = (i - lo) & 1;
        if (i + 1 < hi) {
          const int nb = (i + 1 - lo) & 1;
          gemma_stage_async<cache_t, ACTUAL_HEAD_SIZE, K_EQ_V>(
              buf_k[nb], buf_v[nb], k_cache, v_cache,
              static_cast<int64_t>(block_table[i + 1]), kv_head_idx,
              GQA_ENDTOK(i + 1), kv_stride_block, kv_stride_slot,
              kv_stride_head, tid, num_threads);
          __pipeline_commit();
          __pipeline_wait_prior(1);
        } else {
          __pipeline_wait_prior(0);
        }
        __syncthreads();
        gemma_block_compute<cache_t, ACTUAL_HEAD_SIZE, ELEMS_PER_THREAD,
                            KV_DTYPE, K_EQ_V>(
            buf_k[cur], buf_v[cur], GQA_ENDTOK(i), dim_start, q_regs, ks, vs,
            scale_log2, M, L, acc);
        __syncthreads();
      }
    }
  } else {
    // Synchronous single-buffer fallback (non-16B-aligned head dims).
    for (int blk = start_block; blk < num_seq_blocks; blk++) {
      const int64_t phys = static_cast<int64_t>(block_table[blk]);
      const int end_tok = GQA_ENDTOK(blk);
      const int n_elem = end_tok * ACTUAL_HEAD_SIZE;
      for (int i = tid; i < n_elem; i += num_threads) {
        const int slot = i / ACTUAL_HEAD_SIZE;
        const int d = i % ACTUAL_HEAD_SIZE;
        const int64_t base = phys * kv_stride_block + slot * kv_stride_slot
                              + kv_head_idx * kv_stride_head + d;
        buf_k[0][i] = k_cache[base];
        if constexpr (!K_EQ_V) buf_v[0][i] = v_cache[base];
      }
      __syncthreads();
      gemma_block_compute<cache_t, ACTUAL_HEAD_SIZE, ELEMS_PER_THREAD,
                          KV_DTYPE, K_EQ_V>(
          buf_k[0], buf_v[0], end_tok, dim_start, q_regs, ks, vs, scale_log2,
          M, L, acc);
      __syncthreads();
    }
  }
#undef GQA_ENDTOK

  // Each warp owns a full query head; normalize and write directly.
  float inv_L = (L > 0.f) ? (1.f / L) : 0.f;
  scalar_t* out_ptr = out + seq_idx * num_q_heads * HEAD_SIZE
                          + q_head_idx * HEAD_SIZE + dim_start;
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++)
    from_float(out_ptr[e], acc[e] * inv_L);
}

// ---------------------------------------------------------------------------
// GQA-reuse SPLIT-KV decode (prototype, phase 1).
//
// Same warp-per-head + smem-staged-KV scheme as gemma_gqa_decode_kernel, but
// each CTA processes only this split's slice of the KV blocks and writes a
// PARTIAL (M, L, acc/L) per query head. A separate combine kernel merges the
// num_splits partials. This restores SM occupancy at small batch (the grid
// gains a split dimension) while keeping GQA KV-reuse.
//
// Grid:  (num_kv_heads, num_seqs, num_splits)
// Block: GQA_GROUP * WARP_SIZE
// Partial buffers (classic vLLM-v2 layout, partition == split):
//   max_logits/exp_sums: [num_seqs, num_q_heads, max_parts]
//   tmp_out:             [num_seqs, num_q_heads, max_parts, HEAD_SIZE]
// ---------------------------------------------------------------------------
template <typename scalar_t, typename cache_t,
          int HEAD_SIZE, int ACTUAL_HEAD_SIZE,
          int BLOCK_SIZE, int GQA_GROUP,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          bool K_EQ_V, bool USE_SLIDING_WINDOW>
__global__ void gemma_gqa_split_decode_kernel(
    scalar_t* __restrict__ tmp_out,        // [num_seqs, num_q_heads, max_parts, HEAD_SIZE]
    float* __restrict__ exp_sums,          // [num_seqs, num_q_heads, max_parts]
    float* __restrict__ max_logits,        // [num_seqs, num_q_heads, max_parts]
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int64_t kv_stride_block,
    const int64_t kv_stride_slot,
    const int64_t kv_stride_head,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    const int sliding_window,
    const int num_splits,
    const int max_parts) {

  const int kv_head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int num_q_heads = gridDim.x * GQA_GROUP;
  const int warp_idx = threadIdx.x / WARP_SIZE;   // query head within group
  const int lane = threadIdx.x % WARP_SIZE;
  const int q_head_idx = kv_head_idx * GQA_GROUP + warp_idx;

  const int seq_len = seq_lens[seq_idx];
  const int num_seq_blocks = GEMMA_CDIV(seq_len, BLOCK_SIZE);
  int start_block = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int first = seq_len - sliding_window;
    start_block = (first > 0) ? (first / BLOCK_SIZE) : 0;
  }
  const int total_blocks = num_seq_blocks - start_block;
  const int blocks_per_split = GEMMA_CDIV(total_blocks, num_splits);
  const int split_start = start_block + split_idx * blocks_per_split;
  int split_end = split_start + blocks_per_split;
  if (split_end > num_seq_blocks) split_end = num_seq_blocks;

  constexpr int ELEMS_PER_THREAD = ACTUAL_HEAD_SIZE / WARP_SIZE;
  static_assert(ACTUAL_HEAD_SIZE % WARP_SIZE == 0);
  const int dim_start = lane * ELEMS_PER_THREAD;

  // partition offset for this (seq, q_head, split).
  const int64_t part_idx =
      (static_cast<int64_t>(seq_idx) * num_q_heads + q_head_idx) * max_parts
      + split_idx;

  // Empty split: write neutral partial (M=-inf, L=0, O=0) so combine ignores it.
  if (split_start >= split_end) {
    if (lane == 0) {
      max_logits[part_idx] = -FLT_MAX;
      exp_sums[part_idx] = 0.f;
    }
    scalar_t* to = tmp_out + part_idx * HEAD_SIZE + dim_start;
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) from_float(to[e], 0.f);
    return;
  }

  const scalar_t* q_ptr = q + seq_idx * q_stride
                            + q_head_idx * HEAD_SIZE + dim_start;
  float q_regs[ELEMS_PER_THREAD];
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++)
    q_regs[e] = static_cast<float>(q_ptr[e]);

  float ks = 1.f, vs = 1.f;
  if constexpr (KV_DTYPE != Fp8KVCacheDataType::kAuto) {
    ks = *k_scale_ptr;
    vs = *v_scale_ptr;
  }
  const float scale_log2 = scale * LOG2E;

  extern __shared__ char gqa_split_smem_raw[];
  cache_t* smem_base = reinterpret_cast<cache_t*>(gqa_split_smem_raw);
  constexpr int KBLK = BLOCK_SIZE * ACTUAL_HEAD_SIZE;
  constexpr int BUFSZ = (K_EQ_V ? 1 : 2) * KBLK;
  cache_t* buf_k[2] = {smem_base, smem_base + BUFSZ};
  cache_t* buf_v[2] = {
      K_EQ_V ? smem_base : (smem_base + KBLK),
      K_EQ_V ? (smem_base + BUFSZ) : (smem_base + BUFSZ + KBLK)};

  float M = -FLT_MAX;
  float L = 0.f;
  float acc[ELEMS_PER_THREAD];
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++) acc[e] = 0.f;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int num_threads = GQA_GROUP * WARP_SIZE;
  const int tid = threadIdx.x;

#define GQA_ENDTOK(blk) \
  (((blk) * BLOCK_SIZE + BLOCK_SIZE < seq_len) ? BLOCK_SIZE \
                                               : (seq_len - (blk) * BLOCK_SIZE))

  constexpr int STG_VEC = 16 / sizeof(cache_t);
  if constexpr (ACTUAL_HEAD_SIZE % STG_VEC == 0) {
    const int lo = split_start, hi = split_end;
    if (lo < hi) {
      gemma_stage_async<cache_t, ACTUAL_HEAD_SIZE, K_EQ_V>(
          buf_k[0], buf_v[0], k_cache, v_cache,
          static_cast<int64_t>(block_table[lo]), kv_head_idx, GQA_ENDTOK(lo),
          kv_stride_block, kv_stride_slot, kv_stride_head, tid, num_threads);
      __pipeline_commit();
      for (int i = lo; i < hi; i++) {
        const int cur = (i - lo) & 1;
        if (i + 1 < hi) {
          const int nb = (i + 1 - lo) & 1;
          gemma_stage_async<cache_t, ACTUAL_HEAD_SIZE, K_EQ_V>(
              buf_k[nb], buf_v[nb], k_cache, v_cache,
              static_cast<int64_t>(block_table[i + 1]), kv_head_idx,
              GQA_ENDTOK(i + 1), kv_stride_block, kv_stride_slot,
              kv_stride_head, tid, num_threads);
          __pipeline_commit();
          __pipeline_wait_prior(1);
        } else {
          __pipeline_wait_prior(0);
        }
        __syncthreads();
        gemma_block_compute<cache_t, ACTUAL_HEAD_SIZE, ELEMS_PER_THREAD,
                            KV_DTYPE, K_EQ_V>(
            buf_k[cur], buf_v[cur], GQA_ENDTOK(i), dim_start, q_regs, ks, vs,
            scale_log2, M, L, acc);
        __syncthreads();
      }
    }
  } else {
    for (int blk = split_start; blk < split_end; blk++) {
      const int64_t phys = static_cast<int64_t>(block_table[blk]);
      const int end_tok = GQA_ENDTOK(blk);
      const int n_elem = end_tok * ACTUAL_HEAD_SIZE;
      for (int i = tid; i < n_elem; i += num_threads) {
        const int slot = i / ACTUAL_HEAD_SIZE;
        const int d = i % ACTUAL_HEAD_SIZE;
        const int64_t base = phys * kv_stride_block + slot * kv_stride_slot
                              + kv_head_idx * kv_stride_head + d;
        buf_k[0][i] = k_cache[base];
        if constexpr (!K_EQ_V) buf_v[0][i] = v_cache[base];
      }
      __syncthreads();
      gemma_block_compute<cache_t, ACTUAL_HEAD_SIZE, ELEMS_PER_THREAD,
                          KV_DTYPE, K_EQ_V>(
          buf_k[0], buf_v[0], end_tok, dim_start, q_regs, ks, vs, scale_log2,
          M, L, acc);
      __syncthreads();
    }
  }
#undef GQA_ENDTOK

  // Write this split's partial: O = acc / L (normalized), plus (M, L).
  float inv_L = (L > 0.f) ? (1.f / L) : 0.f;
  if (lane == 0) {
    max_logits[part_idx] = M;
    exp_sums[part_idx] = L;
  }
  scalar_t* to = tmp_out + part_idx * HEAD_SIZE + dim_start;
#pragma unroll
  for (int e = 0; e < ELEMS_PER_THREAD; e++)
    from_float(to[e], acc[e] * inv_L);
}

// ---------------------------------------------------------------------------
// Bandwidth-saturating tensor-core decode (Phase 1).
//
// Decode is HBM-bandwidth-bound; the scalar kernel was compute-bound (per-token
// warp-shuffle reduction => SM busy doing math, not issuing loads) so HBM sat at
// ~14-21%. This kernel combines: (1) LIGHT compute -- GQA-pack the group's query
// heads into a 16-row mma tile and use wmma for QK/PV (no per-token shuffle);
// (2) OVERLAP -- a double-buffered cp.async pipeline so tile t+1 streams while
// tile t computes; (3) the split-KV heuristic supplies enough waves. Goal: drive
// %HBM-roofline toward Triton's ~82% while reading HALF the bytes (k_eq_v), i.e.
// ~2x Triton at high batch. Same online-softmax math as the prefill v2 kernel,
// BLOCK_M=16 (the G group-heads at the last position; no causal triangle).
//
// SPLIT=false: grid (num_kv_heads, num_seqs) -> final out.
// SPLIT=true:  grid (num_kv_heads, num_seqs, num_splits) -> partials + combine.
// bf16 only (no fp8 dequant here); fp8 stays on the scalar path.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SM90 1D bulk-TMA helpers (P1): cp.async.bulk global->shared with mbarrier
// completion. Address-based (no tensor map), so paged KV at block_size=16
// works. Bodies compile to no-ops below sm90; callers gate at runtime
// (use_bulk is only set on sm90 devices).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ds_mbar_init(uint64_t* bar, uint32_t count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
#endif
}
// Order prior generic-proxy smem accesses (zero-fill, wmma reads of the stage
// being reused) before subsequent async-proxy (DMA) writes.
__device__ __forceinline__ void ds_fence_proxy_async() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}
__device__ __forceinline__ void ds_mbar_arrive_expect_tx(uint64_t* bar,
                                                         uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   a),
               "r"(bytes)
               : "memory");
#endif
}
__device__ __forceinline__ void ds_bulk_g2s(void* dst, const void* src,
                                            uint32_t bytes, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  const uint32_t m = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(d),
      "l"(src), "r"(bytes), "r"(m)
      : "memory");
#endif
}
__device__ __forceinline__ void ds_mbar_wait(uint64_t* bar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "DS_MBW%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra DS_MBW%=;\n\t}" ::"r"(a),
      "r"(parity)
      : "memory");
#endif
}

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_N,
          int NUM_WARPS, int GQA_GROUP, bool K_EQ_V, bool USE_SLIDING_WINDOW,
          bool SPLIT, int MIN_CTA = 1, bool USE_BULK = false, int STAGES = 2>
__global__ void __launch_bounds__(NUM_WARPS * 32, MIN_CTA)
gemma_decode_stream_kernel(
    scalar_t* __restrict__ out_or_tmp,   // SPLIT ? tmp_out partials : final out
    float* __restrict__ exp_sums,        // [num_seqs,num_q_heads,max_parts] (SPLIT)
    float* __restrict__ max_logits,      // (SPLIT)
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq, const int page_size, const int q_stride,
    const int64_t kv_stride_block, const int64_t kv_stride_slot,
    const int64_t kv_stride_head, const int sliding_window,
    const int num_splits, const int max_parts,
    float* __restrict__ lse_out = nullptr) {  // [num_q_heads,num_seqs] natural-log
  constexpr int BLOCK_M = 16;
  constexpr int MT = BLOCK_M / 16;          // == 1
  constexpr int NT = BLOCK_N / 16;          // QK N tiles
  constexpr int DT = HEAD_SIZE / 16;        // head tiles
  constexpr int HNT_W = DT / NUM_WARPS;     // head tiles per warp
  constexpr int VEC = 16 / sizeof(cache_t);
  static_assert(DT % NUM_WARPS == 0, "head tiles must split across warps");
  static_assert(MT * NT <= NUM_WARPS, "QK tiles must fit in warps");
  static_assert(BLOCK_N == 16 || BLOCK_N % 32 == 0,
                "warp-softmax assumes BLOCK_N == 16 or a multiple of 32");

  const int kv_head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int split_idx = SPLIT ? blockIdx.z : 0;
  const int nsplits = SPLIT ? num_splits : 1;
  const int num_q_heads = gridDim.x * GQA_GROUP;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int nthreads = NUM_WARPS * 32;
  const int seq_len = seq_lens[seq_idx];

#define DSTREAM_PART(qh) \
  (((int64_t)(seq_idx) * num_q_heads + (qh)) * max_parts + split_idx)

  int kv_begin = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int lo = seq_len - sliding_window;
    kv_begin = (lo > 0) ? (lo / BLOCK_N) * BLOCK_N : 0;
  }
  const int n_tiles = GEMMA_CDIV(seq_len - kv_begin, BLOCK_N);
  const int tiles_per_split = GEMMA_CDIV(n_tiles, nsplits);
  const int tile_lo = split_idx * tiles_per_split;
  int tile_hi = tile_lo + tiles_per_split;
  if (tile_hi > n_tiles) tile_hi = n_tiles;
  const int hvps = HEAD_SIZE / VEC;

  if (SPLIT && tile_lo >= tile_hi) {
    for (int g = warp; g < GQA_GROUP; g += NUM_WARPS)
      if (lane == 0) {
        const int qh = kv_head * GQA_GROUP + g;
        max_logits[DSTREAM_PART(qh)] = -FLT_MAX;
        exp_sums[DSTREAM_PART(qh)] = 0.f;
      }
    for (int iv = tid; iv < GQA_GROUP * hvps; iv += nthreads) {
      const int r = iv / hvps, dv = (iv - r * hvps) * VEC;
      const int qh = kv_head * GQA_GROUP + r;
      *reinterpret_cast<uint4*>(out_or_tmp + DSTREAM_PART(qh) * HEAD_SIZE + dv) =
          uint4{0, 0, 0, 0};
    }
    return;
  }

  constexpr int SPAD = 8;
  constexpr int LDH = HEAD_SIZE + SPAD;
  constexpr int LDN = BLOCK_N + SPAD;
  constexpr int KTILE = BLOCK_N * LDH;          // elems per staged K tile
  // V handling (3 cases):
  //  - k_eq_v: V == K, reuse the staged K tile (no extra smem).
  //  - !k_eq_v, hd512 (V_GMEM): V read straight from global in PV. Each BLOCK_N
  //    tile == one paged block (BLOCK_N <= page_size), so V is wmma-loadable with
  //    ldm=kv_stride_slot. Avoids a 2nd hd512 smem stage that would drop us to
  //    1 CTA/SM -> keeps 3 CTA/SM.
  //  - !k_eq_v, hd256 (V_SMEM): stage V to smem (prefetched). hd256's V tile is
  //    small enough to keep 3 CTA/SM, and smem V avoids exposing load latency.
  constexpr bool V_GMEM = (!K_EQ_V) && (HEAD_SIZE >= 512);
  constexpr bool V_SMEM = (!K_EQ_V) && !V_GMEM;
  constexpr int VBUF = V_SMEM ? KTILE : 0;
  constexpr int STAGE = KTILE + VBUF;
  static_assert(STAGES == 2 || STAGES == 3, "ring supports 2 or 3 stages");
  static_assert(!(USE_BULK && STAGES != 2), "bulk path is 2-stage only");

  // QK k-split (bigtile only): with BLOCK_N>=32 the S n-tiles occupy only
  // NT of the NUM_WARPS warps during QK while the rest idle. Split the
  // head-dim contraction QK_KS ways across the idle warps; each extra way
  // stores a partial S to sSk and softmax sums them during its read (no
  // extra barrier). BLOCK_N==16 keeps QK_KS=1 -> legacy path bit-identical.
  constexpr int QK_KS =
      (BLOCK_N >= 32 && (NUM_WARPS % (MT * NT)) == 0 &&
       (NUM_WARPS / (MT * NT)) > 1 && (DT % (NUM_WARPS / (MT * NT))) == 0)
          ? (NUM_WARPS / (MT * NT))
          : 1;

  extern __shared__ char ds_smem[];
  cache_t* sQ = reinterpret_cast<cache_t*>(ds_smem);   // [16, LDH]
  cache_t* sKV = sQ + BLOCK_M * LDH;                    // STAGES-deep K(+V) ring
  cache_t* sP = sKV + STAGES * STAGE;                   // [16, LDN]
  float* sS = reinterpret_cast<float*>(sP + BLOCK_M * LDN);  // [16, LDN]
  float* sM = sS + BLOCK_M * LDN;
  float* sL = sM + BLOCK_M;
  float* sA = sL + BLOCK_M;
  float* sSk = sA + BLOCK_M;  // [(QK_KS-1) x 16, LDN] k-split S partials
#define DS_KBUF(s) (sKV + (s) * STAGE)
#define DS_VBUF(s) (DS_KBUF(s) + KTILE)

  // P1 bulk-TMA: one mbarrier per pipeline stage tracks that stage's
  // cp.async.bulk transaction bytes. Static smem coexists with the dynamic
  // blob. Only used when use_bulk (sm90).
  __shared__ __align__(8) uint64_t ds_mbar[2];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> Ofrag[MT][HNT_W];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int j = 0; j < HNT_W; j++) wmma::fill_fragment(Ofrag[m][j], 0.0f);

  for (int r = tid; r < BLOCK_M; r += nthreads) { sM[r] = -FLT_MAX; sL[r] = 0.f; }
  // Zero KV stages once so partial-tile tails (n >= n_tok) never feed NaN to QK.
  for (int i = tid; i < STAGES * STAGE; i += nthreads)
    sKV[i] = static_cast<cache_t>(0);

  // Stage Q: G group heads at the last position into rows 0..G-1, pad rest 0.
  for (int iv = tid; iv < BLOCK_M * hvps; iv += nthreads) {
    const int r = iv / hvps, dv = (iv - r * hvps) * VEC;
    if (r < GQA_GROUP) {
      const int qh = kv_head * GQA_GROUP + r;
      const scalar_t* gq = q + (int64_t)seq_idx * q_stride + qh * HEAD_SIZE + dv;
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) =
          *reinterpret_cast<const uint4*>(gq);
    } else {
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) = uint4{0, 0, 0, 0};
    }
  }

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const float scale_log2 = scale * LOG2E;
  const int q_abs = seq_len - 1;
  const int ntiles_local = tile_hi - tile_lo;

  // Async-stage tile (tile_lo + ti) into pipeline stage buffer `s`.
  // Default path: per-(token, 16B-chunk) paged gather via cp.async, one
  // pending group committed per call.
  // use_bulk (P1, sm90): one cp.async.bulk per token row (HEAD_SIZE*2 bytes,
  // the contiguous unit of the NHD cache), issued by tid 0, tracked by the
  // stage's mbarrier. The tile never crosses a page (BLOCK_N <= page_size,
  // _kv0 is a multiple of BLOCK_N), so one block_table lookup per tile.
#define DS_STAGE(ti, s)                                                        \
  do {                                                                         \
    const int _kv0 = kv_begin + (ti) * BLOCK_N;                                \
    const int _ntok = min(BLOCK_N, seq_len - _kv0);                            \
    if constexpr (USE_BULK) {                                                  \
      if (tid == 0) {                                                          \
        ds_fence_proxy_async();                                                \
        ds_mbar_arrive_expect_tx(                                              \
            &ds_mbar[s], (uint32_t)_ntok * HEAD_SIZE * sizeof(cache_t) *       \
                             (V_SMEM ? 2 : 1));                                \
        for (int n = 0; n < _ntok; n++) {                                      \
          const int _tok = _kv0 + n;  /* per-token page: tile may span pages   \
                                         when BLOCK_N > page_size */           \
          const int64_t off =                                                  \
              (int64_t)block_table[_tok / page_size] * kv_stride_block +       \
              (int64_t)(_tok % page_size) * kv_stride_slot +                   \
              kv_head * kv_stride_head;                                        \
          ds_bulk_g2s(DS_KBUF(s) + n * LDH, k_cache + off,                     \
                      HEAD_SIZE * sizeof(cache_t), &ds_mbar[s]);               \
          if (V_SMEM)                                                          \
            ds_bulk_g2s(DS_VBUF(s) + n * LDH, v_cache + off,                   \
                        HEAD_SIZE * sizeof(cache_t), &ds_mbar[s]);             \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      for (int iv = tid; iv < BLOCK_N * hvps; iv += nthreads) {                \
        const int n = iv / hvps, dv = (iv - n * hvps) * VEC;                   \
        if (n < _ntok) {                                                       \
          const int tok = _kv0 + n;                                            \
          const int64_t phys = block_table[tok / page_size];                   \
          const int slot = tok % page_size;                                    \
          const int64_t off = phys * kv_stride_block + slot * kv_stride_slot   \
                              + kv_head * kv_stride_head + dv;                 \
          __pipeline_memcpy_async(DS_KBUF(s) + n * LDH + dv, k_cache + off,    \
                                  16);                                         \
          if (V_SMEM)                                                          \
            __pipeline_memcpy_async(DS_VBUF(s) + n * LDH + dv, v_cache + off,  \
                                    16);                                       \
        }                                                                      \
      }                                                                        \
      __pipeline_commit();                                                     \
    }                                                                          \
  } while (0)

  if constexpr (USE_BULK) {
    if (tid == 0) {
      ds_mbar_init(&ds_mbar[0], 1);  // one arrive per phase (tid 0's expect_tx)
      ds_mbar_init(&ds_mbar[1], 1);
    }
  }
  uint32_t bulk_phase = 0;  // bit s = expected parity of ds_mbar[s]
  (void)bulk_phase;

  __syncthreads();              // Q + zeroed KV + mbarrier init visible
  // Prologue: fill STAGES-1 ring slots (1 for the classic double buffer,
  // 2 for the 3-stage ring).
  DS_STAGE(tile_lo, 0);
  if constexpr (STAGES >= 3) {
    if (ntiles_local > 1) DS_STAGE(tile_lo + 1, 1);
  }

  for (int t = 0; t < ntiles_local; t++) {
    const int cur = (STAGES == 2) ? (t & 1) : (t % STAGES);
    if constexpr (STAGES == 2) {
      // Classic depth-1 schedule: issue t+1 BEFORE waiting on t (the issue
      // overlaps the wait); the end-of-loop barrier protects stage reuse.
      if (t + 1 < ntiles_local) {
        DS_STAGE(tile_lo + t + 1, (t + 1) & 1);
        if constexpr (!USE_BULK) __pipeline_wait_prior(1);
      } else if constexpr (!USE_BULK) {
        __pipeline_wait_prior(0);
      }
      if constexpr (USE_BULK) {
        ds_mbar_wait(&ds_mbar[cur], (bulk_phase >> cur) & 1u);
        bulk_phase ^= (1u << cur);
      }
    } else {
      // 3-stage ring, depth-2, issue-at-top: issue t+2 into stage
      // (t+2)%3 == (t-1)%3 BEFORE waiting on t, so the issue cost overlaps
      // the wait (the first 3-stage variant issued post-barrier and
      // regressed ~4%: the issue landed on the critical path). Safe: the
      // end-of-loop barrier of iteration t-1 was passed by ALL threads
      // before any thread entered iteration t, so t-1's stage reads are
      // complete before this overwrite is issued.
      if (t + STAGES - 1 < ntiles_local)
        DS_STAGE(tile_lo + t + STAGES - 1, (t + STAGES - 1) % STAGES);
      {
        const int rem = ntiles_local - 1 - t;  // tiles still outstanding
        __pipeline_wait_prior(rem >= STAGES - 1 ? STAGES - 1 : rem);
      }
    }
    __syncthreads();

    const int kv0 = kv_begin + (tile_lo + t) * BLOCK_N;
    const int n_tok = min(BLOCK_N, seq_len - kv0);
    cache_t* kbuf = DS_KBUF(cur);
    cache_t* vbuf = V_SMEM ? DS_VBUF(cur) : kbuf;  // staged V (hd256 !k_eq_v)
    // V page coords for V_GMEM (this tile is one paged block: BLOCK_N<=page_size).
    const int64_t v_phys = V_GMEM ? (int64_t)block_table[kv0 / page_size] : 0;
    const int v_slot0 = V_GMEM ? (kv0 % page_size) : 0;

    // QK: S[16, BLOCK_N] = sQ @ kbuf^T (col-major K), full-head contraction.
    // (Split-K across warps was tried to balance this single-warp phase but was
    // net-negative: the partial-reduction scratch dropped 3->2 CTA/SM and the
    // occupancy loss outweighed the removed barrier. Occupancy wins here.)
    if (warp < MT * NT * QK_KS) {
      const int kh = warp / (MT * NT);   // k-split slice owned by this warp
      const int wnt = warp % (MT * NT);
      const int mt = wnt / NT, nt = wnt % NT;
      constexpr int KCH = DT / QK_KS;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.0f);
#pragma unroll
      for (int kk = 0; kk < KCH; kk++) {
        const int kt = kh * KCH + kk;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, cache_t, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, cache_t, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, sQ + mt * 16 * LDH + kt * 16, LDH);
        wmma::load_matrix_sync(fb, kbuf + nt * 16 * LDH + kt * 16, LDH);
        wmma::mma_sync(s, fa, fb, s);
      }
      float* dst = (kh == 0)
                       ? (sS + mt * 16 * LDN + nt * 16)
                       : (sSk + (kh - 1) * BLOCK_M * LDN + mt * 16 * LDN +
                          nt * 16);
      wmma::store_matrix_sync(dst, s, LDN, wmma::mem_row_major);
    }
    __syncthreads();

    // online softmax (parallel, one warp per row). Mask: token-valid + sliding
    // lower bound only (decode query is the last position -> no causal).
    constexpr int CPL = (BLOCK_N + 31) / 32;  // cols/lane (1 for BLOCK_N<=32)
    for (int r = warp; r < BLOCK_M; r += NUM_WARPS) {
      float sv[CPL];
      float rmax = -FLT_MAX;
#pragma unroll
      for (int cc = 0; cc < CPL; ++cc) {
        const int c = lane + cc * 32;
        const int k_abs = kv0 + c;
        bool valid = (c < BLOCK_N) && (c < n_tok);  // c<BLOCK_N masks BN<32
        if (USE_SLIDING_WINDOW && sliding_window > 0)
          valid = valid && (k_abs > q_abs - sliding_window);
        float sval = sS[r * LDN + c];
        if constexpr (QK_KS > 1) {
#pragma unroll
          for (int j = 0; j < QK_KS - 1; j++)
            sval += sSk[j * BLOCK_M * LDN + r * LDN + c];
        }
        sv[cc] = valid ? sval : -FLT_MAX;
        rmax = fmaxf(rmax, sv[cc]);
      }
#pragma unroll
      for (int o = 16; o >= 1; o >>= 1)
        rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
      const float m_old = sM[r];
      const float m_new = fmaxf(m_old, rmax);
      const float alpha =
          (m_old <= -FLT_MAX) ? 0.f : exp2f((m_old - m_new) * scale_log2);
      float rsum = 0.f;
#pragma unroll
      for (int cc = 0; cc < CPL; ++cc) {
        const int c = lane + cc * 32;
        const float p =
            (m_new <= -FLT_MAX) ? 0.f : exp2f((sv[cc] - m_new) * scale_log2);
        if (c < BLOCK_N) sP[r * LDN + c] = static_cast<cache_t>(p);
        rsum += p;  // invalid lanes contribute exp2(-inf)=0
      }
#pragma unroll
      for (int o = 16; o >= 1; o >>= 1)
        rsum += __shfl_xor_sync(0xffffffffu, rsum, o);
      if (lane == 0) {
        sM[r] = m_new;
        sL[r] = sL[r] * alpha + rsum;
        sA[r] = alpha;
      }
    }
    __syncthreads();

    // PV: rescale O frags by per-row alpha (SM80 acc layout), then P@V.
    const int gid = lane / 4;
#pragma unroll
    for (int m = 0; m < MT; m++) {
      const float a_lo = sA[m * 16 + gid];
      const float a_hi = sA[m * 16 + gid + 8];
#pragma unroll
      for (int j = 0; j < HNT_W; j++) {
        Ofrag[m][j].x[0] *= a_lo; Ofrag[m][j].x[1] *= a_lo;
        Ofrag[m][j].x[2] *= a_hi; Ofrag[m][j].x[3] *= a_hi;
        Ofrag[m][j].x[4] *= a_lo; Ofrag[m][j].x[5] *= a_lo;
        Ofrag[m][j].x[6] *= a_hi; Ofrag[m][j].x[7] *= a_hi;
        const int ht = warp * HNT_W + j;
#pragma unroll
        for (int kt = 0; kt < NT; kt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, cache_t, wmma::row_major> fp;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, cache_t, wmma::row_major> fv;
          wmma::load_matrix_sync(fp, sP + m * 16 * LDN + kt * 16, LDN);
          if constexpr (V_GMEM) {
            // V straight from global: tokens [kt*16..] of this paged block are
            // rows (stride kv_stride_slot), head dims [ht*16..] are cols.
            const cache_t* gv = v_cache + v_phys * kv_stride_block
                + (int64_t)(v_slot0 + kt * 16) * kv_stride_slot
                + kv_head * kv_stride_head + ht * 16;
            wmma::load_matrix_sync(fv, gv, kv_stride_slot);
          } else {  // k_eq_v (V==K) or V_SMEM (staged V): both in smem
            wmma::load_matrix_sync(fv, vbuf + kt * 16 * LDH + ht * 16, LDH);
          }
          wmma::mma_sync(Ofrag[m][j], fp, fv, Ofrag[m][j]);
        }
      }
    }
    __syncthreads();  // stage-reuse guard (issue-at-top relies on it)
  }

  // epilogue: store O frags to reused smem (sQ as f32, stride LDH), /L.
  float* sOt = reinterpret_cast<float*>(sQ);  // [16, LDH] f32 (fits sQ region)
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int j = 0; j < HNT_W; j++) {
      const int ht = warp * HNT_W + j;
      wmma::store_matrix_sync(sOt + m * 16 * LDH + ht * 16, Ofrag[m][j], LDH,
                              wmma::mem_row_major);
    }
  __syncthreads();

  if constexpr (SPLIT) {
    for (int g = warp; g < GQA_GROUP; g += NUM_WARPS)
      if (lane == 0) {
        const int qh = kv_head * GQA_GROUP + g;
        max_logits[DSTREAM_PART(qh)] = sM[g] * scale_log2;  // base-2 units
        exp_sums[DSTREAM_PART(qh)] = sL[g];
      }
    for (int iv = tid; iv < GQA_GROUP * hvps; iv += nthreads) {
      const int r = iv / hvps, dv = (iv - r * hvps) * VEC;
      const int qh = kv_head * GQA_GROUP + r;
      const float inv = (sL[r] > 0.f) ? (1.f / sL[r]) : 0.f;
      scalar_t tmp[VEC];
#pragma unroll
      for (int e = 0; e < VEC; e++)
        from_float(tmp[e], sOt[r * LDH + dv + e] * inv);
      *reinterpret_cast<uint4*>(out_or_tmp + DSTREAM_PART(qh) * HEAD_SIZE + dv) =
          *reinterpret_cast<uint4*>(tmp);
    }
  } else {
    for (int iv = tid; iv < GQA_GROUP * hvps; iv += nthreads) {
      const int r = iv / hvps, dv = (iv - r * hvps) * VEC;
      const int qh = kv_head * GQA_GROUP + r;
      const float inv = (sL[r] > 0.f) ? (1.f / sL[r]) : 0.f;
      scalar_t tmp[VEC];
#pragma unroll
      for (int e = 0; e < VEC; e++)
        from_float(tmp[e], sOt[r * LDH + dv + e] * inv);
      scalar_t* go = out_or_tmp + (int64_t)seq_idx * num_q_heads * HEAD_SIZE
                       + qh * HEAD_SIZE + dv;
      *reinterpret_cast<uint4*>(go) = *reinterpret_cast<uint4*>(tmp);
    }
    // Natural-log LSE per (q_head, seq) for cascade merge. sM is the raw running
    // max and sL = sum exp((s_i - sM) * scale), so LSE = sM*scale + ln(sL).
    if (lse_out != nullptr) {
      const int num_seqs = gridDim.y;
      for (int r = warp; r < GQA_GROUP; r += NUM_WARPS)
        if (lane == 0) {
          const int qh = kv_head * GQA_GROUP + r;
          const float l = sL[r];
          lse_out[(int64_t)qh * num_seqs + seq_idx] =
              sM[r] * scale + logf(l > 0.f ? l : 1e-30f);
        }
    }
  }
#undef DS_STAGE
#undef DS_KBUF
#undef DS_VBUF
#undef DSTREAM_PART
}

// ---------------------------------------------------------------------------
// Lean tensor-core (mma.sync) flash-decode. k_eq_v + full (non-sliding) layers,
// non-split. Removes BOTH walls: tensor cores kill the SIMT shuffle-compute
// ceiling, and (unlike wmma) the only big smem buffer is the K tile -> Q is
// built directly into registers (no 16KB sQ), scores/O stay register/tiny-smem,
// so occupancy stays ~SIMT-level. 8 warps: split-K QK -> atomic smem-S reduce ->
// online softmax (warp0) -> hd-split PV (each warp owns HEAD/8 of O). bf16.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_N,
          int GQA_GROUP, int MIN_CTA = 1>
__global__ void __launch_bounds__(256, MIN_CTA)
gemma_decode_mma_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache, const int num_kv_heads,
    const float scale, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, const int max_num_blocks_per_seq,
    const int page_size, const int q_stride, const int64_t kv_stride_block,
    const int64_t kv_stride_slot, const int64_t kv_stride_head,
    float* __restrict__ lse_out = nullptr) {
  constexpr int NWARP = 8;
  constexpr int KCH = HEAD_SIZE / 16;      // QK k-chunks
  constexpr int KPW = KCH / NWARP;         // k-chunks per warp (QK split-K)
  constexpr int HDPW = HEAD_SIZE / NWARP;  // hd output owned per warp (PV split)
  constexpr int NPV = HDPW / 8;            // PV n-tiles per warp
  constexpr int NTILE = BLOCK_N / 8;       // QK n-tiles
  constexpr int U4 = HEAD_SIZE / 8;        // uint4 per token row (smem load)

  const int kv_head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int num_q_heads = gridDim.x * GQA_GROUP;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int group = lane >> 2, tg = lane & 3;
  const int seq_len = seq_lens[seq_idx];

  extern __shared__ char mma_smem[];
  cache_t* sQ = reinterpret_cast<cache_t*>(mma_smem);            // [GQA_GROUP,HD]
  cache_t* sK = sQ + GQA_GROUP * HEAD_SIZE;                      // [2,BN,HD]
  // Per-warp partial S (non-atomic reduce): the softmax sums across warps.
  float* sSp = reinterpret_cast<float*>(sK + 2 * BLOCK_N * HEAD_SIZE);  // [NWARP,16,16]
  float* sM = sSp + NWARP * 16 * 16;                             // [16]
  float* sL = sM + 16;                                           // [16]
  float* sAl = sL + 16;                                          // [16]
  cache_t* sP = reinterpret_cast<cache_t*>(sAl + 16);           // [16,16]

  // Q -> smem (real heads only). q row for (seq, qh=kv_head*GQA_GROUP+h).
  for (int i = tid; i < GQA_GROUP * HEAD_SIZE; i += 256) {
    const int h = i / HEAD_SIZE, d = i % HEAD_SIZE;
    sQ[i] = q[(int64_t)seq_idx * q_stride +
              (kv_head * GQA_GROUP + h) * HEAD_SIZE + d];
  }
  for (int i = tid; i < 16; i += 256) { sM[i] = -FLT_MAX; sL[i] = 0.f; }
  __syncthreads();

  // O accumulator: warp owns hd-chunk [warp*HDPW, +HDPW), NPV n-tiles, C-frag.
  float oacc[NPV][4];
#pragma unroll
  for (int n = 0; n < NPV; n++)
    oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;

  const int* bt = block_tables + seq_idx * max_num_blocks_per_seq;
  const int n_tiles = GEMMA_CDIV(seq_len, BLOCK_N);

#define MMA_ISSUE(BUF, KV0)                                                    \
  do {                                                                         \
    const int nt_ = min(BLOCK_N, seq_len - (KV0));                            \
    for (int i = tid; i < BLOCK_N * U4; i += 256) {                           \
      const int tok = i / U4, dv = (i % U4) * 8;                              \
      if (tok < nt_) {                                                         \
        const int g = (KV0) + tok;                                           \
        const int64_t phys = bt[g / page_size];                             \
        const cache_t* gk = k_cache + phys * kv_stride_block                 \
            + (int64_t)(g % page_size) * kv_stride_slot                      \
            + kv_head * kv_stride_head + dv;                                  \
        __pipeline_memcpy_async(                                             \
            sK + (BUF) * BLOCK_N * HEAD_SIZE + tok * HEAD_SIZE + dv, gk, 16); \
      }                                                                       \
    }                                                                         \
    __pipeline_commit();                                                     \
  } while (0)

  int buf = 0;
  MMA_ISSUE(0, 0);
  for (int ti = 0; ti < n_tiles; ti++) {
    const int kv0 = ti * BLOCK_N;
    const int ntok = min(BLOCK_N, seq_len - kv0);
    if (ti + 1 < n_tiles) { MMA_ISSUE(buf ^ 1, kv0 + BLOCK_N); __pipeline_wait_prior(1); }
    else __pipeline_wait_prior(0);
    __syncthreads();
    cache_t* kbuf = sK + buf * BLOCK_N * HEAD_SIZE;

    // QK split-K: this warp does k-chunks [warp*KPW, +KPW).
    float spart[NTILE][4];
#pragma unroll
    for (int n = 0; n < NTILE; n++)
      spart[n][0] = spart[n][1] = spart[n][2] = spart[n][3] = 0.f;
#pragma unroll
    for (int kc = 0; kc < KPW; kc++) {
      const int c = warp * KPW + kc;
      uint32_t qa[4] = {0, 0, 0, 0};
      if (group < GQA_GROUP) {
        const cache_t* qr = sQ + group * HEAD_SIZE + c * 16;
        qa[0] = *reinterpret_cast<const uint32_t*>(qr + 2 * tg);
        qa[2] = *reinterpret_cast<const uint32_t*>(qr + 2 * tg + 8);
      }
#pragma unroll
      for (int n = 0; n < NTILE; n++) {
        uint32_t kb[2];
        mma_ldm_x2(kb, kbuf + (n * 8 + lane % 8) * HEAD_SIZE + c * 16
                            + (lane / 8) * 8);
        mma_m16n8k16(spart[n], qa, kb, spart[n]);
      }
    }
    // Non-atomic reduce: each warp writes its partial S to its own region.
#pragma unroll
    for (int n = 0; n < NTILE; n++) {
      const int b = warp * 256;
      sSp[b + group * 16 + n * 8 + 2 * tg] = spart[n][0];
      sSp[b + group * 16 + n * 8 + 2 * tg + 1] = spart[n][1];
      sSp[b + (group + 8) * 16 + n * 8 + 2 * tg] = spart[n][2];
      sSp[b + (group + 8) * 16 + n * 8 + 2 * tg + 1] = spart[n][3];
    }
    __syncthreads();

    // online softmax (warp 0 only), summing the per-warp partials across warps.
    if (warp == 0 && group < GQA_GROUP) {
      float s16[16];
#pragma unroll
      for (int c = 0; c < 16; c++) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < NWARP; w++) v += sSp[w * 256 + group * 16 + c];
        s16[c] = v * scale;
      }
      float tmax = -FLT_MAX;
      for (int c = 0; c < ntok; c++) tmax = fmaxf(tmax, s16[c]);
      const float m_old = sM[group];
      const float m_new = fmaxf(m_old, tmax);
      const float al = (m_old <= -FLT_MAX) ? 0.f : __expf(m_old - m_new);
      float ssum = 0.f;
#pragma unroll
      for (int c = 0; c < 16; c++) {
        float p = (c < ntok) ? __expf(s16[c] - m_new) : 0.f;
        from_float(sP[group * 16 + c], p);
        ssum += p;
      }
      if (tg == 0) {
        sM[group] = m_new;
        sL[group] = sL[group] * al + ssum;
        sAl[group] = al;
      }
    }
    // zero sP for padded rows (>=GQA_GROUP) once
    if (warp == 0 && group >= GQA_GROUP)
      for (int c = tg; c < 16; c += 4) from_float(sP[group * 16 + c], 0.f);
    __syncthreads();

    // rescale O by alpha (row group), then PV mma.
    const float al = (group < GQA_GROUP) ? sAl[group] : 0.f;
#pragma unroll
    for (int n = 0; n < NPV; n++) { oacc[n][0] *= al; oacc[n][1] *= al; }
    uint32_t pa[4];
    mma_ldm_x4(pa, sP + (lane % 16) * 16 + (lane / 16) * 8);
#pragma unroll
    for (int b = 0; b < NPV / 2; b++) {       // one ldm_x4t covers 16 hd = 2 mmas
      const int hd = warp * HDPW + b * 16;
      uint32_t vb[4];
      mma_ldm_x4t(vb, kbuf + (lane % 16) * HEAD_SIZE + hd + (lane / 16) * 8);
      mma_m16n8k16(oacc[2 * b], pa, &vb[0], oacc[2 * b]);      // hd b*16+0..7
      mma_m16n8k16(oacc[2 * b + 1], pa, &vb[2], oacc[2 * b + 1]);  // +8..15
    }
    __syncthreads();
    buf ^= 1;
  }
#undef MMA_ISSUE

  // normalize + write (warp owns hd-chunk). O row = group (real if <GQA_GROUP).
  if (group < GQA_GROUP) {
    const float inv = (sL[group] > 0.f) ? (1.f / sL[group]) : 0.f;
    const int qh = kv_head * GQA_GROUP + group;
    scalar_t* go = out + (int64_t)seq_idx * num_q_heads * HEAD_SIZE
                     + qh * HEAD_SIZE;
#pragma unroll
    for (int n = 0; n < NPV; n++) {
      const int hd = warp * HDPW + (n / 2) * 16 + (n % 2) * 8;
      from_float(go[hd + 2 * tg], oacc[n][0] * inv);
      from_float(go[hd + 2 * tg + 1], oacc[n][1] * inv);
    }
    if (lse_out != nullptr && warp == 0 && tg == 0)
      lse_out[(int64_t)qh * gridDim.y + seq_idx] =
          sM[group] + logf(sL[group] > 0.f ? sL[group] : 1e-30f);
  }
}

// ---------------------------------------------------------------------------
// Gate D: fused mma.sync decode (register softmax). Differences vs the wmma
// stream kernel: S never touches smem (each warp owns an 8-column slice of
// S[16, BLOCK_N] in mma.sync c-frags with a DOCUMENTED thread layout, so the
// online softmax runs in registers; only 16 f32 of row-stats per warp cross
// through smem); QK runs on ALL warps (one n8 slice each); 3 CTA barriers per
// tile instead of 4-5. O rescale needs no data movement at all: the same
// lane owns the same rows in the S and O fragments. Split partials use the
// stream kernel's exact base-2 conventions -> combine v2 reused verbatim.
// GROUP <= 16, BLOCK_N == 8 * NWARP (64), bf16, k_eq_v or V_SMEM staged.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int GQA_GROUP,
          bool K_EQ_V, bool USE_SLIDING_WINDOW, bool SPLIT, int NWARP_T = 8>
__global__ void __launch_bounds__(NWARP_T * 32, 1)
gemma_decode_fused_kernel(
    scalar_t* __restrict__ out_or_tmp,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq, const int page_size, const int q_stride,
    const int64_t kv_stride_block, const int64_t kv_stride_slot,
    const int64_t kv_stride_head, const int sliding_window,
    const int num_splits, const int max_parts,
    float* __restrict__ lse_out = nullptr) {
  constexpr int NWARP = NWARP_T;            // 8 (BN64) or 4 (BN32, 2-3 CTA/SM)
  constexpr int BLOCK_N = 8 * NWARP;        // one n8 S-slice per warp
  constexpr int KCH = HEAD_SIZE / 16;       // QK k-chunks
  constexpr int HDPW = HEAD_SIZE / NWARP;   // O head-slice per warp
  constexpr int NPV = HDPW / 8;             // O n8 tiles per warp
  constexpr int SPAD = 8;
  constexpr int LDH = HEAD_SIZE + SPAD;
  constexpr int LDN = BLOCK_N + SPAD;
  constexpr int VEC = 16 / sizeof(cache_t);
  constexpr bool V_SMEM = !K_EQ_V;          // decode: stage V unless V==K
  constexpr int KTILE = BLOCK_N * LDH;
  constexpr int STAGE = KTILE * (V_SMEM ? 2 : 1);
  // 2-stage double buffer (3-stage measured -3%: depth is not the decode
  // constraint; modulo indexing + fatter footprint cost more than they buy).
  constexpr int NSTG = 2;
  static_assert(GQA_GROUP <= 16 && HEAD_SIZE % (16 * NWARP) == 0);

  const int kv_head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int split_idx = SPLIT ? blockIdx.z : 0;
  const int nsplits = SPLIT ? num_splits : 1;
  const int num_q_heads = gridDim.x * GQA_GROUP;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int group = lane >> 2, tg = lane & 3;   // c-frag: rows {group, group+8}
  const int seq_len = seq_lens[seq_idx];
  const int nthreads = NWARP * 32;

#define DFUSED_PART(qh) \
  (((int64_t)(seq_idx) * num_q_heads + (qh)) * max_parts + split_idx)

  int kv_begin = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int lo = seq_len - sliding_window;
    kv_begin = (lo > 0) ? (lo / BLOCK_N) * BLOCK_N : 0;
  }
  const int n_tiles = GEMMA_CDIV(seq_len - kv_begin, BLOCK_N);
  const int tiles_per_split = GEMMA_CDIV(n_tiles, nsplits);
  const int tile_lo = split_idx * tiles_per_split;
  int tile_hi = tile_lo + tiles_per_split;
  if (tile_hi > n_tiles) tile_hi = n_tiles;
  const int hvps = HEAD_SIZE / VEC;

  if (SPLIT && tile_lo >= tile_hi) {
    for (int g = warp; g < GQA_GROUP; g += NWARP)
      if (lane == 0) {
        const int qh = kv_head * GQA_GROUP + g;
        max_logits[DFUSED_PART(qh)] = -FLT_MAX;
        exp_sums[DFUSED_PART(qh)] = 0.f;
      }
    for (int iv = tid; iv < GQA_GROUP * hvps; iv += nthreads) {
      const int r = iv / hvps, dv = (iv - r * hvps) * VEC;
      const int qh = kv_head * GQA_GROUP + r;
      *reinterpret_cast<uint4*>(out_or_tmp + DFUSED_PART(qh) * HEAD_SIZE + dv) =
          uint4{0, 0, 0, 0};
    }
    return;
  }

  extern __shared__ char df_smem[];
  cache_t* sQ = reinterpret_cast<cache_t*>(df_smem);        // [16, LDH]
  cache_t* sKV = sQ + 16 * LDH;                             // NSTG-stage ring
  cache_t* sP = sKV + NSTG * STAGE;                         // [16, LDN] bf16
  float* sWm = reinterpret_cast<float*>(sP + 16 * LDN);     // [NWARP, 16]
  float* sWl = sWm + NWARP * 16;                            // [NWARP, 16]
#define DF_KBUF(s) (sKV + (s) * STAGE)
#define DF_VBUF(s) (DF_KBUF(s) + KTILE)

  // Q rows 0..GQA_GROUP-1 real, rest zero (M=16 pad).
  for (int iv = tid; iv < 16 * hvps; iv += nthreads) {
    const int r = iv / hvps, dv = (iv - r * hvps) * VEC;
    if (r < GQA_GROUP) {
      const scalar_t* gq =
          q + (int64_t)seq_idx * q_stride + (kv_head * GQA_GROUP + r) * HEAD_SIZE + dv;
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) =
          *reinterpret_cast<const uint4*>(gq);
    } else {
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) = uint4{0, 0, 0, 0};
    }
  }

  // Zero the ring once so tail-pad rows stay finite for the mma (lazy
  // per-tile pad zeroing measured SLOWER and subtly racy — keep this).
  for (int i = tid; i < NSTG * STAGE; i += nthreads)
    sKV[i] = static_cast<cache_t>(0);

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const float scale_log2 = scale * LOG2E;
  const int q_abs = seq_len - 1;
  const int ntiles_local = tile_hi - tile_lo;

  // Q fragments cached in registers for the WHOLE loop: this thread only
  // ever consumes rows {group, group+8} at cols {2tg, 2tg+1} (+8) of each
  // k16 chunk. Re-reading them from smem cost ~128KB of smem traffic per
  // tile. KCH*4 uint32 = 128 regs at hd512.
  uint32_t qreg[KCH][4];
  __syncthreads();  // sQ populated
  {
    const cache_t* q0 = sQ + group * LDH;
    const cache_t* q1 = sQ + (group + 8) * LDH;
#pragma unroll
    for (int c = 0; c < KCH; c++) {
      qreg[c][0] = *reinterpret_cast<const uint32_t*>(q0 + c * 16 + 2 * tg);
      qreg[c][1] = *reinterpret_cast<const uint32_t*>(q1 + c * 16 + 2 * tg);
      qreg[c][2] = *reinterpret_cast<const uint32_t*>(q0 + c * 16 + 2 * tg + 8);
      qreg[c][3] = *reinterpret_cast<const uint32_t*>(q1 + c * 16 + 2 * tg + 8);
    }
  }

  // Per-thread online state for its two c-frag rows (quad-redundant).
  float m_run0 = -FLT_MAX, m_run1 = -FLT_MAX;
  float l_run0 = 0.f, l_run1 = 0.f;
  float oacc[NPV][4];
#pragma unroll
  for (int n = 0; n < NPV; n++)
    oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;

#define DF_STAGE(ti, st)                                                       \
  do {                                                                         \
    const int _kv0 = kv_begin + (ti) * BLOCK_N;                                \
    const int _ntok = min(BLOCK_N, seq_len - _kv0);                            \
    for (int iv = tid; iv < BLOCK_N * hvps; iv += nthreads) {                  \
      const int n = iv / hvps, dv = (iv - n * hvps) * VEC;                     \
      if (n < _ntok) {                                                         \
        const int tok = _kv0 + n;                                              \
        const int64_t phys = block_table[tok / page_size];                     \
        const int64_t off = phys * kv_stride_block +                           \
                            (tok % page_size) * kv_stride_slot +               \
                            kv_head * kv_stride_head + dv;                     \
        __pipeline_memcpy_async(DF_KBUF(st) + n * LDH + dv, k_cache + off,     \
                                16);                                           \
        if (V_SMEM)                                                            \
          __pipeline_memcpy_async(DF_VBUF(st) + n * LDH + dv, v_cache + off,   \
                                  16);                                         \
      }                                                                        \
    }                                                                          \
    __pipeline_commit();                                                       \
  } while (0)

  __syncthreads();
  DF_STAGE(tile_lo, 0);
  if (NSTG >= 3 && ntiles_local > 1) DF_STAGE(tile_lo + 1, 1);

  for (int t = 0; t < ntiles_local; t++) {
    const int cur = (NSTG == 2) ? (t & 1) : (t % NSTG);
    {
      const int rem = ntiles_local - 1 - t;  // tiles still outstanding
      __pipeline_wait_prior(rem >= NSTG - 1 ? NSTG - 1 : rem);
    }
    __syncthreads();  // B1: staged K(+V) visible CTA-wide

    const int kv0 = kv_begin + (tile_lo + t) * BLOCK_N;
    const int n_tok = min(BLOCK_N, seq_len - kv0);
    cache_t* kbuf = DF_KBUF(cur);
    cache_t* vbuf = V_SMEM ? DF_VBUF(cur) : kbuf;

    // ---- QK: this warp's n8 slice, Q from registers ----
    float sacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < KCH; c++) {
      uint32_t kb[2];
      mma_ldm_x2(kb, kbuf + (warp * 8 + (lane & 7)) * LDH + c * 16 +
                         ((lane >> 3) & 1) * 8);
      mma_m16n8k16(sacc, qreg[c], kb, sacc);
    }

    // ---- masking + slice row-max (registers) ----
    const int col0 = kv0 + warp * 8 + 2 * tg;      // this thread's 2 columns
    const bool v0 = (warp * 8 + 2 * tg) < n_tok;
    const bool v1 = (warp * 8 + 2 * tg + 1) < n_tok;
    bool s0 = v0, s1 = v1;
    if (USE_SLIDING_WINDOW && sliding_window > 0) {
      s0 = s0 && (col0 > q_abs - sliding_window);
      s1 = s1 && (col0 + 1 > q_abs - sliding_window);
    }
    if (!s0) { sacc[0] = -FLT_MAX; sacc[2] = -FLT_MAX; }
    if (!s1) { sacc[1] = -FLT_MAX; sacc[3] = -FLT_MAX; }
    float sm0 = fmaxf(sacc[0], sacc[1]);
    float sm1 = fmaxf(sacc[2], sacc[3]);
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      sm0 = fmaxf(sm0, __shfl_xor_sync(0xffffffffu, sm0, o));
      sm1 = fmaxf(sm1, __shfl_xor_sync(0xffffffffu, sm1, o));
    }
    if (tg == 0) {
      sWm[warp * 16 + group] = sm0;
      sWm[warp * 16 + group + 8] = sm1;
    }
    __syncthreads();  // B2: stats visible; PV(t-1) provably complete

    // Prefetch tile t+NSTG-1 now: overlaps softmax+PV. Its ring slot was
    // last read at iteration t-1, and B2 (all threads past QK(t)) certifies
    // iteration t-1 fully complete.
    if (t + NSTG - 1 < ntiles_local)
      DF_STAGE(tile_lo + t + NSTG - 1, (t + NSTG - 1) % NSTG);

    // ---- global row stats + exp + P + O rescale (registers) ----
    float tm0 = -FLT_MAX, tm1 = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < NWARP; w++) {
      tm0 = fmaxf(tm0, sWm[w * 16 + group]);
      tm1 = fmaxf(tm1, sWm[w * 16 + group + 8]);
    }
    const float mn0 = fmaxf(m_run0, tm0);
    const float mn1 = fmaxf(m_run1, tm1);
    const float al0 =
        (m_run0 <= -FLT_MAX) ? 0.f : exp2f((m_run0 - mn0) * scale_log2);
    const float al1 =
        (m_run1 <= -FLT_MAX) ? 0.f : exp2f((m_run1 - mn1) * scale_log2);
    float p0 = s0 && mn0 > -FLT_MAX ? exp2f((sacc[0] - mn0) * scale_log2) : 0.f;
    float p1 = s1 && mn0 > -FLT_MAX ? exp2f((sacc[1] - mn0) * scale_log2) : 0.f;
    float p2 = s0 && mn1 > -FLT_MAX ? exp2f((sacc[2] - mn1) * scale_log2) : 0.f;
    float p3 = s1 && mn1 > -FLT_MAX ? exp2f((sacc[3] - mn1) * scale_log2) : 0.f;
    m_run0 = mn0; m_run1 = mn1;
    // P -> smem (bf16 pairs; this warp's 8 columns).
    {
      __nv_bfloat162* d0 = reinterpret_cast<__nv_bfloat162*>(
          sP + group * LDN + warp * 8 + 2 * tg);
      __nv_bfloat162* d1 = reinterpret_cast<__nv_bfloat162*>(
          sP + (group + 8) * LDN + warp * 8 + 2 * tg);
      *d0 = __floats2bfloat162_rn(p0, p1);
      *d1 = __floats2bfloat162_rn(p2, p3);
    }
    // slice row-sums -> smem
    float ss0 = p0 + p1, ss1 = p2 + p3;
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      ss0 += __shfl_xor_sync(0xffffffffu, ss0, o);
      ss1 += __shfl_xor_sync(0xffffffffu, ss1, o);
    }
    if (tg == 0) {
      sWl[warp * 16 + group] = ss0;
      sWl[warp * 16 + group + 8] = ss1;
    }
    // O rescale: same lanes own the same rows in O c-frags.
#pragma unroll
    for (int n = 0; n < NPV; n++) {
      oacc[n][0] *= al0; oacc[n][1] *= al0;
      oacc[n][2] *= al1; oacc[n][3] *= al1;
    }
    __syncthreads();  // B3: sP + sWl visible

    // ---- PV: O[:, warp's HDPW slice] += P[16, BLOCK_N] @ V ----
#pragma unroll
    for (int kk = 0; kk < BLOCK_N / 16; kk++) {
      uint32_t pa[4];
      mma_ldm_x4(pa, sP + (lane & 15) * LDN + kk * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int b = 0; b < NPV / 2; b++) {
        const int hd = warp * HDPW + b * 16;
        uint32_t vb[4];
        mma_ldm_x4t(vb, vbuf + (kk * 16 + (lane & 15)) * LDH + hd +
                            (lane >> 4) * 8);
        mma_m16n8k16(oacc[2 * b], pa, &vb[0], oacc[2 * b]);
        mma_m16n8k16(oacc[2 * b + 1], pa, &vb[2], oacc[2 * b + 1]);
      }
    }
    // l update (quad-redundant, reads this tile's sWl).
    {
      float tl0 = 0.f, tl1 = 0.f;
#pragma unroll
      for (int w = 0; w < NWARP; w++) {
        tl0 += sWl[w * 16 + group];
        tl1 += sWl[w * 16 + group + 8];
      }
      l_run0 = l_run0 * al0 + tl0;
      l_run1 = l_run1 * al1 + tl1;
    }
  }
#undef DF_STAGE

  // ---- epilogue: per-thread c-frag writes (rows group / group+8) ----
  const float inv0 = (l_run0 > 0.f) ? (1.f / l_run0) : 0.f;
  const float inv1 = (l_run1 > 0.f) ? (1.f / l_run1) : 0.f;
  if (SPLIT) {
    if (warp == 0 && tg == 0) {
      if (group < GQA_GROUP) {
        const int qh = kv_head * GQA_GROUP + group;
        max_logits[DFUSED_PART(qh)] = m_run0 * scale_log2;
        exp_sums[DFUSED_PART(qh)] = l_run0;
      }
      if (group + 8 < GQA_GROUP) {
        const int qh = kv_head * GQA_GROUP + group + 8;
        max_logits[DFUSED_PART(qh)] = m_run1 * scale_log2;
        exp_sums[DFUSED_PART(qh)] = l_run1;
      }
    }
#pragma unroll
    for (int n = 0; n < NPV; n++) {
      const int hd = warp * HDPW + n * 8 + 2 * tg;
      if (group < GQA_GROUP) {
        const int qh = kv_head * GQA_GROUP + group;
        scalar_t* go = out_or_tmp + DFUSED_PART(qh) * HEAD_SIZE + hd;
        from_float(go[0], oacc[n][0] * inv0);
        from_float(go[1], oacc[n][1] * inv0);
      }
      if (group + 8 < GQA_GROUP) {
        const int qh = kv_head * GQA_GROUP + group + 8;
        scalar_t* go = out_or_tmp + DFUSED_PART(qh) * HEAD_SIZE + hd;
        from_float(go[0], oacc[n][2] * inv1);
        from_float(go[1], oacc[n][3] * inv1);
      }
    }
  } else {
#pragma unroll
    for (int n = 0; n < NPV; n++) {
      const int hd = warp * HDPW + n * 8 + 2 * tg;
      if (group < GQA_GROUP) {
        const int qh = kv_head * GQA_GROUP + group;
        scalar_t* go = out_or_tmp + (int64_t)seq_idx * num_q_heads * HEAD_SIZE +
                       qh * HEAD_SIZE + hd;
        from_float(go[0], oacc[n][0] * inv0);
        from_float(go[1], oacc[n][1] * inv0);
      }
      if (group + 8 < GQA_GROUP) {
        const int qh = kv_head * GQA_GROUP + group + 8;
        scalar_t* go = out_or_tmp + (int64_t)seq_idx * num_q_heads * HEAD_SIZE +
                       qh * HEAD_SIZE + hd;
        from_float(go[0], oacc[n][2] * inv1);
        from_float(go[1], oacc[n][3] * inv1);
      }
    }
    if (lse_out != nullptr && warp == 0 && tg == 0) {
      const int num_seqs = gridDim.y;
      if (group < GQA_GROUP) {
        const int qh = kv_head * GQA_GROUP + group;
        lse_out[(int64_t)qh * num_seqs + seq_idx] =
            m_run0 * scale + logf(l_run0 > 0.f ? l_run0 : 1e-30f);
      }
      if (group + 8 < GQA_GROUP) {
        const int qh = kv_head * GQA_GROUP + group + 8;
        lse_out[(int64_t)qh * num_seqs + seq_idx] =
            m_run1 * scale + logf(l_run1 > 0.f ? l_run1 : 1e-30f);
      }
    }
  }
#undef DFUSED_PART
}

// ---------------------------------------------------------------------------
// Bandwidth-first SIMT decode (no tensor cores). The wmma stream kernel is
// occupancy-limited (79 reg + 52KB smem -> 3 CTA/SM -> 37.5% occ -> 40% DRAM,
// latency-bound). This kernel is lean: O accumulated in registers (no wmma
// fragments), only a small single-buffered K tile in smem (shared by the GQA
// group, k_eq_v reuses it as V), so many CTAs fit per SM -> latency hidden ->
// HBM saturated. One warp per query head; the warp's 32 lanes own a head slice
// (EPL = HEAD_SIZE/32 contiguous dims) for coalesced loads + register O.
// k_eq_v only (the 2x-over-Triton target = full hd512 layers). bf16 only.
// Matches the stream kernel's base-2 online-softmax convention so the existing
// split-reduce + LSE path is reused verbatim.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_N,
          int BDY, int BDZ, bool USE_SLIDING_WINDOW, bool SPLIT, int MIN_CTA = 1>
__global__ void __launch_bounds__(BDY * BDZ * 32, MIN_CTA)
gemma_decode_simt_kernel(
    scalar_t* __restrict__ out_or_tmp,   // SPLIT ? tmp_out partials : final out
    float* __restrict__ exp_sums,        // [num_seqs,num_q_heads,max_parts]
    float* __restrict__ max_logits,
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq, const int page_size, const int q_stride,
    const int64_t kv_stride_block, const int64_t kv_stride_slot,
    const int64_t kv_stride_head, const int sliding_window,
    const int num_splits, const int max_parts,
    float* __restrict__ lse_out = nullptr,
    const int full_sink = 0, const int full_window = 0,  // lossy sink+window
    // Lossy top-k: per-(seq,kv_head) list of selected tile indices to attend
    // [num_seqs, num_kv_heads, num_sel] int32 (nullptr -> use sink+window/full).
    const int* __restrict__ selected_tiles = nullptr, const int num_sel = 0) {
  constexpr int EPL = HEAD_SIZE / 32;     // head dims per lane
  constexpr int VEC = 16 / sizeof(cache_t);  // bf16 -> 8 per uint4
  const int kv_head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int split_idx = SPLIT ? blockIdx.z : 0;
  const int nsplits = SPLIT ? num_splits : 1;
  const int num_q_heads = gridDim.x * BDY;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int ty = warp % BDY;               // query head within the group
  const int tz = warp / BDY;               // bdz: KV-split group (occupancy)
  const int qh = kv_head * BDY + ty;
  const int seq_len = seq_lens[seq_idx];
  const int nthreads = BDY * BDZ * 32;
  constexpr int TPZ = BLOCK_N / BDZ;       // tile tokens this tz-group handles
  const float scale_log2 = scale * LOG2E;

#define DSIMT_PART(h) \
  (((int64_t)(seq_idx) * num_q_heads + (h)) * max_parts + split_idx)

  // Sliding-window lower bound (tile-aligned), then split the tile range.
  int kv_begin = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int lo = seq_len - sliding_window;
    kv_begin = (lo > 0) ? (lo / BLOCK_N) * BLOCK_N : 0;
  }
  // Effective tile sequence. Sliding layers keep their lossless windowed range
  // [kv_begin, L). Full layers may use LOSSY sink+window: attend only
  // [0, full_sink) U [L-full_window, L). Degenerate (overlap / short seq) ->
  // full attention (lossless). The map (SIMT_KV0) is transparent to the compute
  // (decode has no causal mask; only the tile boundary `ntok` matters).
  const int eff_begin_tile = kv_begin / BLOCK_N;
  int n_tiles = GEMMA_CDIV(seq_len - kv_begin, BLOCK_N);
  int sink_tiles = 0, win_start_tile = 0;
  bool sw = (!USE_SLIDING_WINDOW) && (full_sink > 0) && (full_window > 0);
  if (sw) {
    const int n_full = GEMMA_CDIV(seq_len, BLOCK_N);
    sink_tiles = GEMMA_CDIV(full_sink, BLOCK_N);
    win_start_tile =
        (seq_len > full_window) ? ((seq_len - full_window) / BLOCK_N) : 0;
    if (win_start_tile <= sink_tiles) {
      sw = false;  // ranges touch -> full attention (lossless)
    } else {
      n_tiles = sink_tiles + (n_full - win_start_tile);
    }
  }
  // Top-k (P2): an explicit per-(seq,kv_head) selected-tile list overrides the
  // sink+window/full map. n_tiles becomes the selected count.
  if (selected_tiles != nullptr) { n_tiles = num_sel; sw = false; }
  const int tiles_per_split = GEMMA_CDIV(n_tiles, nsplits);
  const int tile_lo = split_idx * tiles_per_split;
  int tile_hi = tile_lo + tiles_per_split;
  if (tile_hi > n_tiles) tile_hi = n_tiles;

  if (SPLIT && tile_lo >= tile_hi) {     // empty split -> neutral partial
    if (lane == 0) {
      max_logits[DSIMT_PART(qh)] = -FLT_MAX;
      exp_sums[DSIMT_PART(qh)] = 0.f;
    }
    scalar_t* to = out_or_tmp + DSIMT_PART(qh) * HEAD_SIZE + lane * EPL;
#pragma unroll
    for (int e = 0; e < EPL; e++) from_float(to[e], 0.f);
    return;
  }
  // Map a logical tile index -> physical first-token: explicit selected-tile
  // list (top-k) if provided, else the sink+window/full map.
#define SIMT_KV0(LT)                                                          \
  ((selected_tiles != nullptr)                                                \
       ? (selected_tiles[((int64_t)seq_idx * gridDim.x + kv_head) * num_sel   \
                         + (LT)] * BLOCK_N)                                    \
       : ((sw ? ((LT) < sink_tiles                                            \
                     ? (LT)                                                    \
                     : (win_start_tile + ((LT) - sink_tiles)))                \
              : (eff_begin_tile + (LT))) * BLOCK_N))

  // Q into registers (lane owns dims [lane*EPL, +EPL)); O accumulator in regs.
  float q_reg[EPL], o_reg[EPL];
  const scalar_t* qp =
      q + (int64_t)seq_idx * q_stride + qh * HEAD_SIZE + lane * EPL;
#pragma unroll
  for (int e = 0; e < EPL; e++) {
    q_reg[e] = static_cast<float>(qp[e]);
    o_reg[e] = 0.f;
  }
  float m = -FLT_MAX, l = 0.f;

  // smem holds ONLY the K tile (BLOCK_N tokens) -> the GQA group's BDY query-head
  // warps read it once (GQA reuse), k_eq_v reuses it as V. Q/scores/O stay in
  // registers (small smem -> high occupancy, unlike the wmma kernel's 52KB).
  // Tile-parallel softmax: BLOCK_N independent dots, then ONE online-softmax
  // update per tile -> breaks the per-token serial recurrence.
  // Double-buffered: smem holds 2 K tiles so tile t+1 streams (cp.async) while
  // tile t computes -> HBM latency hidden (the wmma kernel's edge, now here too).
  extern __shared__ char simt_smem[];
  cache_t* sK = reinterpret_cast<cache_t*>(simt_smem);  // [2][BLOCK_N, HEAD_SIZE]
  const int* bt = block_tables + seq_idx * max_num_blocks_per_seq;
  constexpr int U4_PER_TOK = HEAD_SIZE / VEC;
  constexpr int TILE_ELT = BLOCK_N * HEAD_SIZE;  // elements per K buffer
  const int koff = lane * EPL;

#define SIMT_ISSUE(BUF, KV0)                                                    \
  do {                                                                          \
    const int nt_ = min(BLOCK_N, seq_len - (KV0));                             \
    for (int i = threadIdx.x; i < BLOCK_N * U4_PER_TOK; i += nthreads) {       \
      const int tok = i / U4_PER_TOK;                                          \
      const int d = (i % U4_PER_TOK) * VEC;                                    \
      if (tok < nt_) {                                                          \
        const int g = (KV0) + tok;                                            \
        const int64_t phys = bt[g / page_size];                              \
        const cache_t* gk = k_cache + phys * kv_stride_block                  \
            + (int64_t)(g % page_size) * kv_stride_slot                       \
            + kv_head * kv_stride_head + d;                                    \
        __pipeline_memcpy_async(                                              \
            sK + (BUF) * TILE_ELT + tok * HEAD_SIZE + d, gk, 16);             \
      }                                                                        \
    }                                                                          \
    __pipeline_commit();                                                      \
  } while (0)

  int buf = 0;
  SIMT_ISSUE(0, SIMT_KV0(tile_lo));
  for (int lt = tile_lo; lt < tile_hi; lt++) {
    const int kv0 = SIMT_KV0(lt);
    const int ntok = min(BLOCK_N, seq_len - kv0);
    if (lt + 1 < tile_hi) {
      SIMT_ISSUE(buf ^ 1, SIMT_KV0(lt + 1));
      __pipeline_wait_prior(1);  // tile t ready; tile t+1 stays in flight
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    const cache_t* kbuf = sK + buf * TILE_ELT;

    // This bdz-group owns tile tokens [tz*TPZ, tz*TPZ+TPZ). TPZ independent dots
    // (lane owns a head slice; warp-reduce per token), then ONE softmax update.
    float s[TPZ];
    float tmax = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < TPZ; j++) {
      const int t = tz * TPZ + j;
      float dot = -FLT_MAX;
      if (t < ntok) {
        const cache_t* kt = kbuf + t * HEAD_SIZE + koff;
        cache_t kv[EPL];  // vectorized smem read (uint4) -> registers
#pragma unroll
        for (int u = 0; u < EPL / VEC; u++)
          *reinterpret_cast<uint4*>(&kv[u * VEC]) =
              *reinterpret_cast<const uint4*>(kt + u * VEC);
        dot = 0.f;
#pragma unroll
        for (int e = 0; e < EPL; e++)
          dot += q_reg[e] * static_cast<float>(kv[e]);
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
          dot += __shfl_xor_sync(0xffffffffu, dot, o);
      }
      s[j] = dot;
      tmax = fmaxf(tmax, dot);
    }
    const float m_new = fmaxf(m, tmax);
    const float alpha = (m <= -FLT_MAX) ? 0.f : exp2f((m - m_new) * scale_log2);
    float dsum = 0.f;
#pragma unroll
    for (int j = 0; j < TPZ; j++) {
      s[j] = (s[j] <= -FLT_MAX) ? 0.f : exp2f((s[j] - m_new) * scale_log2);
      dsum += s[j];
    }
    l = l * alpha + dsum;
#pragma unroll
    for (int e = 0; e < EPL; e++) o_reg[e] *= alpha;
#pragma unroll
    for (int j = 0; j < TPZ; j++) {
      if (s[j] != 0.f) {
        const cache_t* kt = kbuf + (tz * TPZ + j) * HEAD_SIZE + koff;
        cache_t kv[EPL];
#pragma unroll
        for (int u = 0; u < EPL / VEC; u++)
          *reinterpret_cast<uint4*>(&kv[u * VEC]) =
              *reinterpret_cast<const uint4*>(kt + u * VEC);
#pragma unroll
        for (int e = 0; e < EPL; e++)
          o_reg[e] += s[j] * static_cast<float>(kv[e]);
      }
    }
    m = m_new;
    __syncthreads();
    buf ^= 1;
  }
#undef SIMT_ISSUE
#undef SIMT_KV0

  if constexpr (BDZ == 1) {
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;
    if (SPLIT) {
      if (lane == 0) {
        max_logits[DSIMT_PART(qh)] = m * scale_log2;
        exp_sums[DSIMT_PART(qh)] = l;
      }
      scalar_t* to = out_or_tmp + DSIMT_PART(qh) * HEAD_SIZE + lane * EPL;
#pragma unroll
      for (int e = 0; e < EPL; e++) from_float(to[e], o_reg[e] * inv);
    } else {
      scalar_t* go = out_or_tmp + (int64_t)seq_idx * num_q_heads * HEAD_SIZE
                       + qh * HEAD_SIZE + lane * EPL;
#pragma unroll
      for (int e = 0; e < EPL; e++) from_float(go[e], o_reg[e] * inv);
      if (lse_out != nullptr && lane == 0)
        lse_out[(int64_t)qh * gridDim.y + seq_idx] =
            m * scale + logf(l > 0.f ? l : 1e-30f);
    }
  } else {
    // Combine the BDZ partials per head via smem (reuse the K-tile region).
    float* csM = reinterpret_cast<float*>(simt_smem);          // [BDY*BDZ]
    float* csL = csM + BDY * BDZ;                              // [BDY*BDZ]
    float* csO = csL + BDY * BDZ;                              // [BDY*BDZ*HEAD]
    __syncthreads();  // K-tile readers done -> safe to reuse smem as combine buf
    const int slot = ty * BDZ + tz;
    if (lane == 0) { csM[slot] = m; csL[slot] = l; }
#pragma unroll
    for (int e = 0; e < EPL; e++) csO[slot * HEAD_SIZE + koff + e] = o_reg[e];
    __syncthreads();
    if (tz == 0) {
      float Mf = -FLT_MAX;
#pragma unroll
      for (int z = 0; z < BDZ; z++) Mf = fmaxf(Mf, csM[ty * BDZ + z]);
      float Lf = 0.f, acc[EPL];
#pragma unroll
      for (int e = 0; e < EPL; e++) acc[e] = 0.f;
#pragma unroll
      for (int z = 0; z < BDZ; z++) {
        const float mz = csM[ty * BDZ + z];
        if (mz <= -FLT_MAX) continue;
        const float w = exp2f((mz - Mf) * scale_log2);
        Lf += csL[ty * BDZ + z] * w;
#pragma unroll
        for (int e = 0; e < EPL; e++)
          acc[e] += w * csO[(ty * BDZ + z) * HEAD_SIZE + koff + e];
      }
      const float inv = (Lf > 0.f) ? (1.f / Lf) : 0.f;
      if (SPLIT) {
        if (lane == 0) {
          max_logits[DSIMT_PART(qh)] = Mf * scale_log2;
          exp_sums[DSIMT_PART(qh)] = Lf;
        }
        scalar_t* to = out_or_tmp + DSIMT_PART(qh) * HEAD_SIZE + lane * EPL;
#pragma unroll
        for (int e = 0; e < EPL; e++) from_float(to[e], acc[e] * inv);
      } else {
        scalar_t* go = out_or_tmp + (int64_t)seq_idx * num_q_heads * HEAD_SIZE
                         + qh * HEAD_SIZE + lane * EPL;
#pragma unroll
        for (int e = 0; e < EPL; e++) from_float(go[e], acc[e] * inv);
        if (lse_out != nullptr && lane == 0)
          lse_out[(int64_t)qh * gridDim.y + seq_idx] =
              Mf * scale + logf(Lf > 0.f ? Lf : 1e-30f);
      }
    }
  }
#undef DSIMT_PART
}

// ---------------------------------------------------------------------------
// Split-KV combine (prototype, phase 2): merge num_splits partials per
// (seq, q_head) via the numerically-stable base-2 LSE recurrence.
// Grid: (num_q_heads, num_seqs). Block: WARP_SIZE (head dim across lanes).
// ---------------------------------------------------------------------------
template <typename scalar_t, int HEAD_SIZE>
__global__ void gemma_split_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_q_heads, HEAD_SIZE]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_q_heads, max_parts, HEAD_SIZE]
    const float* __restrict__ exp_sums,    // [num_seqs, num_q_heads, max_parts] (L)
    const float* __restrict__ max_logits,  // [num_seqs, num_q_heads, max_parts] (M)
    const int num_splits,
    const int max_parts,
    float* __restrict__ lse_out = nullptr) {  // [num_q_heads,num_seqs] natural-log
  const int q_head = blockIdx.x;
  const int seq = blockIdx.y;
  const int num_q_heads = gridDim.x;
  const int lane = threadIdx.x;  // 0..WARP_SIZE-1
  constexpr int ELEMS = HEAD_SIZE / WARP_SIZE;
  static_assert(HEAD_SIZE % WARP_SIZE == 0);
  const int dim_start = lane * ELEMS;

  const int64_t base = static_cast<int64_t>(seq) * num_q_heads + q_head;
  const float* m_ptr = max_logits + base * max_parts;
  const float* l_ptr = exp_sums + base * max_parts;

  float M_g = -FLT_MAX;
  for (int s = 0; s < num_splits; s++) M_g = fmaxf(M_g, m_ptr[s]);

  float denom = 0.f;
  for (int s = 0; s < num_splits; s++) {
    if (m_ptr[s] > -FLT_MAX) denom += l_ptr[s] * exp2f(m_ptr[s] - M_g);
  }
  const float inv = (denom > 0.f) ? (1.f / denom) : 0.f;

  // Natural-log LSE for cascade merge. m_ptr/M_g are base-2 exponent units and
  // denom = sum L_i * exp2(m_i - M_g), so LSE = M_g*ln2 + ln(denom).
  if (lse_out != nullptr && lane == 0) {
    const int num_seqs = gridDim.y;
    lse_out[(int64_t)q_head * num_seqs + seq] =
        M_g * 0.69314718055994531f + logf(denom > 0.f ? denom : 1e-30f);
  }

  float acc[ELEMS];
#pragma unroll
  for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

  const scalar_t* o_base = tmp_out + base * max_parts * HEAD_SIZE;
  for (int s = 0; s < num_splits; s++) {
    if (!(m_ptr[s] > -FLT_MAX)) continue;
    const float w = l_ptr[s] * exp2f(m_ptr[s] - M_g) * inv;
    if (w == 0.f) continue;
    const scalar_t* o = o_base + static_cast<int64_t>(s) * HEAD_SIZE + dim_start;
#pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] += w * static_cast<float>(o[e]);
  }

  scalar_t* out_ptr = out + base * HEAD_SIZE + dim_start;
#pragma unroll
  for (int e = 0; e < ELEMS; e++) from_float(out_ptr[e], acc[e]);
}

// ---------------------------------------------------------------------------
// Split-KV combine v2 (Session 9). The v1 kernel runs ONE WARP per
// (q_head, seq) — 64 warps total at b=4 — with 32B-strided lane loads:
// hopelessly under-parallelized and 2x sector over-fetch (measured 16.9us
// hd512 @33 splits vs FA4's 4.7us). v2: grid (q_heads, seqs, HEAD/128) so
// every SM gets work at low batch, 64 threads per CTA, each thread owning
// one dim-PAIR (uint32 loads, consecutive threads -> consecutive dims =
// fully coalesced), m/l staged once through smem. The per-dim float
// sequence (M_g fmax order, denom order, per-split FMA order) is IDENTICAL
// to v1 -> bitwise-identical output. Requires num_splits <= 256 (host
// falls back to v1 otherwise).
// ---------------------------------------------------------------------------
template <typename scalar_t, int HEAD_SIZE>
__global__ void gemma_split_reduce_v2_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_q_heads, HEAD_SIZE]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_q_heads, max_parts, HEAD_SIZE]
    const float* __restrict__ exp_sums,    // [num_seqs, num_q_heads, max_parts] (L)
    const float* __restrict__ max_logits,  // [num_seqs, num_q_heads, max_parts] (M)
    const int num_splits,
    const int max_parts,
    float* __restrict__ lse_out = nullptr) {  // [num_q_heads,num_seqs] natural-log
  const int q_head = blockIdx.x;
  const int seq = blockIdx.y;
  const int num_q_heads = gridDim.x;
  const int tid = threadIdx.x;  // 0..63
  constexpr int NTHREADS = 64;          // one dim-pair each: 128 dims per CTA
  constexpr int MAX_SPLITS_V2 = 256;
  static_assert(HEAD_SIZE % 128 == 0);
  // This CTA's dim-pair: blockIdx.z selects the 128-dim chunk.
  const int pair = blockIdx.z * NTHREADS + tid;

  const int64_t base = static_cast<int64_t>(seq) * num_q_heads + q_head;
  const float* m_ptr = max_logits + base * max_parts;
  const float* l_ptr = exp_sums + base * max_parts;

  __shared__ float s_m[MAX_SPLITS_V2];
  __shared__ float s_l[MAX_SPLITS_V2];
  for (int s = tid; s < num_splits; s += NTHREADS) {
    s_m[s] = m_ptr[s];
    s_l[s] = l_ptr[s];
  }
  __syncthreads();

  // Every thread derives M_g/denom redundantly from smem — same float order
  // as v1 (s = 0..num_splits-1), so weights are bit-identical.
  float M_g = -FLT_MAX;
  for (int s = 0; s < num_splits; s++) M_g = fmaxf(M_g, s_m[s]);
  float denom = 0.f;
  for (int s = 0; s < num_splits; s++) {
    if (s_m[s] > -FLT_MAX) denom += s_l[s] * exp2f(s_m[s] - M_g);
  }
  const float inv = (denom > 0.f) ? (1.f / denom) : 0.f;

  if (lse_out != nullptr && tid == 0 && blockIdx.z == 0) {
    const int num_seqs = gridDim.y;
    lse_out[(int64_t)q_head * num_seqs + seq] =
        M_g * 0.69314718055994531f + logf(denom > 0.f ? denom : 1e-30f);
  }

  // Stage the weights once (reuses s_l), then run the O accumulation
  // BRANCHLESS + unrolled: with the conditional loads of the naive form the
  // compiler cannot issue iterations ahead and the loop degenerates into a
  // serialized dependent-load chain (~11us for 33 splits). Unconditional
  // loads are safe (tmp_out is fully allocated and empty splits wrote O=0,
  // w=0) and w=0 contributions add +0.0f — v1-bitwise output preserved.
  __syncthreads();
  for (int s = tid; s < num_splits; s += NTHREADS) {
    s_l[s] = (s_m[s] > -FLT_MAX) ? s_l[s] * exp2f(s_m[s] - M_g) * inv : 0.f;
  }
  __syncthreads();

  float acc0 = 0.f, acc1 = 0.f;
  const scalar_t* o_base = tmp_out + base * max_parts * HEAD_SIZE + 2 * pair;
#pragma unroll 4
  for (int s = 0; s < num_splits; s++) {
    scalar_t o2[2];
    *reinterpret_cast<uint32_t*>(o2) = *reinterpret_cast<const uint32_t*>(
        o_base + static_cast<int64_t>(s) * HEAD_SIZE);
    const float w = s_l[s];
    acc0 += w * static_cast<float>(o2[0]);
    acc1 += w * static_cast<float>(o2[1]);
  }
  scalar_t r2[2];
  from_float(r2[0], acc0);
  from_float(r2[1], acc1);
  *reinterpret_cast<uint32_t*>(out + base * HEAD_SIZE + 2 * pair) =
      *reinterpret_cast<uint32_t*>(r2);
}

}  // namespace gemma
}  // namespace vllm

#undef GEMMA_CDIV
