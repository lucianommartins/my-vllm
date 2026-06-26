/*
 * Gemma4-optimized paged attention — SM90 (Hopper) decode kernels.
 *
 * P1b: Wider-tile SIMT decode exploiting H100's 228KB smem budget.
 * BLOCK_N=32 (vs 16 on A100) halves the pipeline iterations, block-table
 * lookups, and barrier overhead per byte of KV loaded. Same register-
 * resident O / scalar-FMA compute (the right fit for vector-matrix decode).
 *
 * Key design decisions vs SM80 SIMT:
 *   - BLOCK_N=32: 2-stage smem = 2×32×HEAD×2B = 64KB → 3 CTAs/SM in 228KB
 *   - TPZ = BLOCK_N/BDZ = 32/BDZ: more tokens per bdz-group per tile
 *   - MIN_CTA tuned for H100 smem/register tradeoffs
 *   - k_eq_v: K tile reused as V (half HBM bytes, same as SM80)
 */
#pragma once

#include "../../attention/attention_dtypes.h"
#include "../../attention/attention_generic.cuh"
#include "../../cuda_compat.h"

#ifndef USE_ROCM
  #include "../../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#else
  #include "../../quantization/w8a8/fp8/amd/quant_utils.cuh"
#endif

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <float.h>

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

#define SM90_CDIV(a, b) (((a) + (b) - 1) / (b))

namespace vllm {
namespace gemma {
namespace sm90 {

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_N,
          int BDY, int BDZ, bool USE_SLIDING_WINDOW, bool SPLIT,
          int MIN_CTA = 1>
__global__ void __launch_bounds__(BDY * BDZ * 32, MIN_CTA)
gemma_decode_sm90_kernel(
    scalar_t* __restrict__ out_or_tmp,
    float* __restrict__ exp_sums,
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
    const int full_sink = 0, const int full_window = 0) {

  constexpr int EPL = HEAD_SIZE / 32;
  constexpr int VEC = 16 / sizeof(cache_t);
  const int kv_head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int split_idx = SPLIT ? blockIdx.z : 0;
  const int nsplits = SPLIT ? num_splits : 1;
  const int num_q_heads = gridDim.x * BDY;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int ty = warp % BDY;
  const int tz = warp / BDY;
  const int qh = kv_head * BDY + ty;
  const int seq_len = seq_lens[seq_idx];
  const int nthreads = BDY * BDZ * 32;
  constexpr int TPZ = BLOCK_N / BDZ;
  const float scale_log2 = scale * LOG2E;

#define DSM90_PART(h) \
  (((int64_t)(seq_idx) * num_q_heads + (h)) * max_parts + split_idx)

  int kv_begin = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int lo = seq_len - sliding_window;
    kv_begin = (lo > 0) ? (lo / BLOCK_N) * BLOCK_N : 0;
  }
  const int eff_begin_tile = kv_begin / BLOCK_N;
  int n_tiles = SM90_CDIV(seq_len - kv_begin, BLOCK_N);
  int sink_tiles = 0, win_start_tile = 0;
  bool sw = (!USE_SLIDING_WINDOW) && (full_sink > 0) && (full_window > 0);
  if (sw) {
    const int n_full = SM90_CDIV(seq_len, BLOCK_N);
    sink_tiles = SM90_CDIV(full_sink, BLOCK_N);
    win_start_tile =
        (seq_len > full_window) ? ((seq_len - full_window) / BLOCK_N) : 0;
    if (win_start_tile <= sink_tiles) {
      sw = false;
    } else {
      n_tiles = sink_tiles + (n_full - win_start_tile);
    }
  }
  const int tiles_per_split = SM90_CDIV(n_tiles, nsplits);
  const int tile_lo = split_idx * tiles_per_split;
  int tile_hi = tile_lo + tiles_per_split;
  if (tile_hi > n_tiles) tile_hi = n_tiles;

  if (SPLIT && tile_lo >= tile_hi) {
    if (lane == 0) {
      max_logits[DSM90_PART(qh)] = -FLT_MAX;
      exp_sums[DSM90_PART(qh)] = 0.f;
    }
    scalar_t* to = out_or_tmp + DSM90_PART(qh) * HEAD_SIZE + lane * EPL;
#pragma unroll
    for (int e = 0; e < EPL; e++) from_float(to[e], 0.f);
    return;
  }

#define SM90_KV0(LT) \
  ((sw ? ((LT) < sink_tiles \
              ? (LT) : (win_start_tile + ((LT) - sink_tiles))) \
       : (eff_begin_tile + (LT))) * BLOCK_N)

  float q_reg[EPL], o_reg[EPL];
  const scalar_t* qp =
      q + (int64_t)seq_idx * q_stride + qh * HEAD_SIZE + lane * EPL;
#pragma unroll
  for (int e = 0; e < EPL; e++) {
    q_reg[e] = static_cast<float>(qp[e]);
    o_reg[e] = 0.f;
  }
  float m = -FLT_MAX, l = 0.f;

  extern __shared__ char sm90_decode_smem[];
  cache_t* sK = reinterpret_cast<cache_t*>(sm90_decode_smem);
  const int* bt = block_tables + seq_idx * max_num_blocks_per_seq;
  constexpr int U4_PER_TOK = HEAD_SIZE / VEC;
  constexpr int TILE_ELT = BLOCK_N * HEAD_SIZE;
  const int koff = lane * EPL;

  // cp.async tile load — handles multi-page tiles (BLOCK_N > page_size).
#define SM90_ISSUE(BUF, KV0) \
  do { \
    const int nt_ = min(BLOCK_N, seq_len - (KV0)); \
    for (int i = threadIdx.x; i < BLOCK_N * U4_PER_TOK; i += nthreads) { \
      const int tok = i / U4_PER_TOK; \
      const int d = (i % U4_PER_TOK) * VEC; \
      if (tok < nt_) { \
        const int g = (KV0) + tok; \
        const int64_t phys = bt[g / page_size]; \
        const cache_t* gk = k_cache + phys * kv_stride_block \
            + (int64_t)(g % page_size) * kv_stride_slot \
            + kv_head * kv_stride_head + d; \
        __pipeline_memcpy_async( \
            sK + (BUF) * TILE_ELT + tok * HEAD_SIZE + d, gk, 16); \
      } \
    } \
    __pipeline_commit(); \
  } while (0)

  int buf = 0;
  SM90_ISSUE(0, SM90_KV0(tile_lo));
  for (int lt = tile_lo; lt < tile_hi; lt++) {
    const int kv0 = SM90_KV0(lt);
    const int ntok = min(BLOCK_N, seq_len - kv0);
    if (lt + 1 < tile_hi) {
      SM90_ISSUE(buf ^ 1, SM90_KV0(lt + 1));
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    const cache_t* kbuf = sK + buf * TILE_ELT;

    float s[TPZ];
    float tmax = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < TPZ; j++) {
      const int t = tz * TPZ + j;
      float dot = -FLT_MAX;
      if (t < ntok) {
        const cache_t* kt = kbuf + t * HEAD_SIZE + koff;
        cache_t kv[EPL];
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
    const float alpha =
        (m <= -FLT_MAX) ? 0.f : exp2f((m - m_new) * scale_log2);
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
#undef SM90_ISSUE
#undef SM90_KV0

  // --- Epilogue (identical to SM80 SIMT) ---
  if constexpr (BDZ == 1) {
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;
    if (SPLIT) {
      if (lane == 0) {
        max_logits[DSM90_PART(qh)] = m * scale_log2;
        exp_sums[DSM90_PART(qh)] = l;
      }
      scalar_t* to = out_or_tmp + DSM90_PART(qh) * HEAD_SIZE + lane * EPL;
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
    float* csM = reinterpret_cast<float*>(sm90_decode_smem);
    float* csL = csM + BDY * BDZ;
    float* csO = csL + BDY * BDZ;
    __syncthreads();
    const int slot = ty * BDZ + tz;
    if (lane == 0) { csM[slot] = m; csL[slot] = l; }
#pragma unroll
    for (int e = 0; e < EPL; e++)
      csO[slot * HEAD_SIZE + koff + e] = o_reg[e];
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
          max_logits[DSM90_PART(qh)] = Mf * scale_log2;
          exp_sums[DSM90_PART(qh)] = Lf;
        }
        scalar_t* to = out_or_tmp + DSM90_PART(qh) * HEAD_SIZE + lane * EPL;
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
#undef DSM90_PART
}

}  // namespace sm90
}  // namespace gemma
}  // namespace vllm

#undef SM90_CDIV
