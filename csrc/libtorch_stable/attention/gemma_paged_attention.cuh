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

#define GEMMA_CDIV(a, b) (((a) + (b) - 1) / (b))

namespace vllm {
namespace gemma {

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
    const int max_parts) {
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
