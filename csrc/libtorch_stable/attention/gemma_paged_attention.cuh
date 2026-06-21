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

inline __device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
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
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_N,
          int NUM_WARPS, int GQA_GROUP, bool K_EQ_V, bool USE_SLIDING_WINDOW,
          bool SPLIT, int MIN_CTA = 1>
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

  extern __shared__ char ds_smem[];
  cache_t* sQ = reinterpret_cast<cache_t*>(ds_smem);   // [16, LDH]
  cache_t* sKV = sQ + BLOCK_M * LDH;                    // 2 pipeline stages of K(+V)
  cache_t* sP = sKV + 2 * STAGE;                        // [16, LDN]
  float* sS = reinterpret_cast<float*>(sP + BLOCK_M * LDN);  // [16, LDN]
  float* sM = sS + BLOCK_M * LDN;
  float* sL = sM + BLOCK_M;
  float* sA = sL + BLOCK_M;
#define DS_KBUF(s) (sKV + (s) * STAGE)
#define DS_VBUF(s) (DS_KBUF(s) + KTILE)

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> Ofrag[MT][HNT_W];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int j = 0; j < HNT_W; j++) wmma::fill_fragment(Ofrag[m][j], 0.0f);

  for (int r = tid; r < BLOCK_M; r += nthreads) { sM[r] = -FLT_MAX; sL[r] = 0.f; }
  // Zero KV stages once so partial-tile tails (n >= n_tok) never feed NaN to QK.
  for (int i = tid; i < 2 * STAGE; i += nthreads) sKV[i] = static_cast<cache_t>(0);

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

  // Async-stage tile (tile_lo + ti) into pipeline stage buffer `s` (per-token
  // paged gather; 16B cp.async). One pending group committed per call.
#define DS_STAGE(ti, s)                                                        \
  do {                                                                         \
    const int _kv0 = kv_begin + (ti) * BLOCK_N;                                \
    const int _ntok = min(BLOCK_N, seq_len - _kv0);                            \
    for (int iv = tid; iv < BLOCK_N * hvps; iv += nthreads) {                  \
      const int n = iv / hvps, dv = (iv - n * hvps) * VEC;                     \
      if (n < _ntok) {                                                         \
        const int tok = _kv0 + n;                                             \
        const int64_t phys = block_table[tok / page_size];                    \
        const int slot = tok % page_size;                                     \
        const int64_t off = phys * kv_stride_block + slot * kv_stride_slot     \
                            + kv_head * kv_stride_head + dv;                   \
        __pipeline_memcpy_async(DS_KBUF(s) + n * LDH + dv, k_cache + off, 16); \
        if (V_SMEM)                                                            \
          __pipeline_memcpy_async(DS_VBUF(s) + n * LDH + dv, v_cache + off,    \
                                  16);                                         \
      }                                                                        \
    }                                                                          \
    __pipeline_commit();                                                       \
  } while (0)

  __syncthreads();              // Q + zeroed KV visible before async loads land
  DS_STAGE(tile_lo, 0);         // prefetch first tile

  for (int t = 0; t < ntiles_local; t++) {
    const int cur = t & 1;
    if (t + 1 < ntiles_local) {
      DS_STAGE(tile_lo + t + 1, (t + 1) & 1);
      __pipeline_wait_prior(1);  // keep next tile's load in flight
    } else {
      __pipeline_wait_prior(0);
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
    if (warp < MT * NT) {
      const int mt = warp / NT, nt = warp % NT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.0f);
#pragma unroll
      for (int kt = 0; kt < DT; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, cache_t, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, cache_t, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, sQ + mt * 16 * LDH + kt * 16, LDH);
        wmma::load_matrix_sync(fb, kbuf + nt * 16 * LDH + kt * 16, LDH);
        wmma::mma_sync(s, fa, fb, s);
      }
      wmma::store_matrix_sync(sS + mt * 16 * LDN + nt * 16, s, LDN,
                              wmma::mem_row_major);
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
        sv[cc] = valid ? sS[r * LDN + c] : -FLT_MAX;
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
    __syncthreads();
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

}  // namespace gemma
}  // namespace vllm

#undef GEMMA_CDIV
