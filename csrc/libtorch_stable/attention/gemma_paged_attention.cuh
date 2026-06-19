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

}  // namespace gemma
}  // namespace vllm

#undef GEMMA_CDIV
