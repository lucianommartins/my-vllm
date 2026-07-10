/*
 * SM90 paged decode attention kernel for Gemma4 — CUDA-graph-safe.
 *
 * Register-resident Q/O + double-buffered cp.async paged KV loads.
 * BLOCK_N=32 with 4 CTA/SM target on H100 (50% occupancy).
 * Non-k_eq_v: separate V load after QK into second smem buffer.
 * k_eq_v: V=K, skip V load (halves bandwidth).
 * No split-KV: single-pass online softmax.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <float.h>

static constexpr float TMA_LOG2E = 1.4426950408889634f;

template <int HEAD_SIZE, int BLOCK_N, int BDY, int BDZ, bool K_EQ_V,
          bool USE_SLIDING_WINDOW, int MIN_CTA = 4>
__global__ void __launch_bounds__(BDY * BDZ * 32, MIN_CTA)
gemma_decode_tma_simt_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ page_table,
    const int* __restrict__ seq_lens,
    const int num_kv_heads,
    const float scale,
    const int q_stride,
    const int max_blocks_per_seq,
    const int64_t kv_stride_block,
    const int64_t kv_stride_slot,
    const int64_t kv_stride_head,
    const int64_t v_stride_block,
    const int64_t v_stride_slot,
    const int64_t v_stride_head,
    const int page_size,
    const int sliding_window,
    float* __restrict__ lse_out) {

  static_assert(HEAD_SIZE == 256 || HEAD_SIZE == 512);
  constexpr int EPL = HEAD_SIZE / 32;
  constexpr int VEC = 16 / sizeof(__nv_bfloat16);
  constexpr int TPZ = BLOCK_N / BDZ;
  constexpr int TILE_ELT = BLOCK_N * HEAD_SIZE;
  constexpr int U4_PER_TOK = HEAD_SIZE / VEC;
  const int nthreads = BDY * BDZ * 32;

  const int kv_head = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int ty = warp % BDY;
  const int tz = warp / BDY;
  const int qh = kv_head * BDY + ty;
  const int num_q_heads = gridDim.x * BDY;
  const int seq_len = seq_lens[seq_idx];
  const float scale_log2 = scale * TMA_LOG2E;
  const int koff = lane * EPL;

  if (seq_len <= 0) return;

  // Q → registers
  float q_reg[EPL], o_reg[EPL];
  const __nv_bfloat16* qp =
      q + (int64_t)seq_idx * q_stride + qh * HEAD_SIZE + lane * EPL;
#pragma unroll
  for (int e = 0; e < EPL; e++) {
    q_reg[e] = __bfloat162float(qp[e]);
    o_reg[e] = 0.f;
  }
  float m = -FLT_MAX, l = 0.f;

  // Double-buffered K smem [2][BLOCK_N * HEAD_SIZE]
  extern __shared__ char tma_smem_[];
  __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(tma_smem_);

  const int* bt = page_table + seq_idx * max_blocks_per_seq;

  // Sliding window
  int kv_begin = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int lo = seq_len - sliding_window;
    kv_begin = (lo > 0) ? (lo / BLOCK_N) * BLOCK_N : 0;
  }
  const int n_tiles = (seq_len - kv_begin + BLOCK_N - 1) / BLOCK_N;

  // Issue K load macro (paged cp.async, double-buffered)
#define ISSUE_K(BUF, KV0)                                                     \
  do {                                                                         \
    const int nt_ = min(BLOCK_N, seq_len - (KV0));                            \
    for (int i = threadIdx.x; i < BLOCK_N * U4_PER_TOK; i += nthreads) {     \
      const int tok = i / U4_PER_TOK;                                         \
      const int d = (i % U4_PER_TOK) * VEC;                                   \
      if (tok < nt_) {                                                         \
        const int g = (KV0) + tok;                                             \
        const int64_t phys = bt[g / page_size];                               \
        const __nv_bfloat16* gk = k_cache + phys * kv_stride_block            \
            + (int64_t)(g % page_size) * kv_stride_slot                       \
            + kv_head * kv_stride_head + d;                                    \
        __pipeline_memcpy_async(                                               \
            sK + (BUF) * TILE_ELT + tok * HEAD_SIZE + d, gk, 16);             \
      }                                                                        \
    }                                                                          \
    __pipeline_commit();                                                       \
  } while (0)

  // Prefetch first tile
  int buf = 0;
  ISSUE_K(0, kv_begin);

  for (int tile = 0; tile < n_tiles; tile++) {
    const int kv0 = kv_begin + tile * BLOCK_N;
    const int ntok = min(BLOCK_N, seq_len - kv0);

    // Prefetch next tile while computing current
    if (tile + 1 < n_tiles) {
      ISSUE_K(buf ^ 1, kv_begin + (tile + 1) * BLOCK_N);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();

    const __nv_bfloat16* kbuf = sK + buf * TILE_ELT;

    // --- QK ---
    float s[TPZ];
    float tmax = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < TPZ; j++) {
      const int t = tz * TPZ + j;
      float dot = -FLT_MAX;
      if (t < ntok) {
        const __nv_bfloat16* kt = kbuf + t * HEAD_SIZE + koff;
        __nv_bfloat16 kv[EPL];
#pragma unroll
        for (int u = 0; u < EPL / VEC; u++)
          *reinterpret_cast<uint4*>(&kv[u * VEC]) =
              *reinterpret_cast<const uint4*>(kt + u * VEC);
        dot = 0.f;
#pragma unroll
        for (int e = 0; e < EPL; e++)
          dot += q_reg[e] * __bfloat162float(kv[e]);
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
          dot += __shfl_xor_sync(0xffffffffu, dot, o);
      }
      s[j] = dot;
      tmax = fmaxf(tmax, dot);
    }

    // --- Online softmax ---
    const float m_new = fmaxf(m, tmax);
    const float alpha =
        (m <= -FLT_MAX) ? 0.f : exp2f((m - m_new) * scale_log2);
    float dsum = 0.f;
#pragma unroll
    for (int j = 0; j < TPZ; j++) {
      s[j] = (s[j] <= -FLT_MAX)
                 ? 0.f
                 : exp2f((s[j] - m_new) * scale_log2);
      dsum += s[j];
    }
    l = l * alpha + dsum;
#pragma unroll
    for (int e = 0; e < EPL; e++) o_reg[e] *= alpha;

    // --- PV: for k_eq_v, reuse K as V. For non-k_eq_v, load V. ---
    if constexpr (K_EQ_V) {
#pragma unroll
      for (int j = 0; j < TPZ; j++) {
        if (s[j] != 0.f) {
          const __nv_bfloat16* vt =
              kbuf + (tz * TPZ + j) * HEAD_SIZE + koff;
          __nv_bfloat16 vv[EPL];
#pragma unroll
          for (int u = 0; u < EPL / VEC; u++)
            *reinterpret_cast<uint4*>(&vv[u * VEC]) =
                *reinterpret_cast<const uint4*>(vt + u * VEC);
#pragma unroll
          for (int e = 0; e < EPL; e++)
            o_reg[e] += s[j] * __bfloat162float(vv[e]);
        }
      }
    } else {
      // V ≠ K: load V to the OTHER buffer (not the one with K being prefetched)
      // We use buffer "buf" which held K and is no longer needed for K prefetch
      // since the next K goes to buf^1
      __syncthreads();
      __nv_bfloat16* vbuf_ptr = sK + buf * TILE_ELT;
      for (int i = threadIdx.x; i < BLOCK_N * U4_PER_TOK; i += nthreads) {
        const int tok = i / U4_PER_TOK;
        const int d = (i % U4_PER_TOK) * VEC;
        if (tok < ntok) {
          const int g = kv0 + tok;
          const int64_t phys = bt[g / page_size];
          const __nv_bfloat16* gv = v_cache + phys * v_stride_block
              + (int64_t)(g % page_size) * v_stride_slot
              + kv_head * v_stride_head + d;
          __pipeline_memcpy_async(vbuf_ptr + tok * HEAD_SIZE + d, gv, 16);
        }
      }
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();

#pragma unroll
      for (int j = 0; j < TPZ; j++) {
        if (s[j] != 0.f) {
          const __nv_bfloat16* vt =
              vbuf_ptr + (tz * TPZ + j) * HEAD_SIZE + koff;
          __nv_bfloat16 vv[EPL];
#pragma unroll
          for (int u = 0; u < EPL / VEC; u++)
            *reinterpret_cast<uint4*>(&vv[u * VEC]) =
                *reinterpret_cast<const uint4*>(vt + u * VEC);
#pragma unroll
          for (int e = 0; e < EPL; e++)
            o_reg[e] += s[j] * __bfloat162float(vv[e]);
        }
      }
    }
    m = m_new;
    __syncthreads();
    buf ^= 1;
  }
#undef ISSUE_K

  // --- BDZ cross-group reduction ---
  if constexpr (BDZ > 1) {
    float* csM = reinterpret_cast<float*>(tma_smem_);
    float* csL = csM + BDY * BDZ;
    float* csO = csL + BDY * BDZ;
    csM[warp] = m;
    csL[warp] = l;
    for (int e = 0; e < EPL; e++)
      csO[warp * HEAD_SIZE + lane * EPL + e] = o_reg[e];
    __syncthreads();

    if (tz == 0) {
      float rm = csM[ty], rl = csL[ty];
      float ro[EPL];
      for (int e = 0; e < EPL; e++)
        ro[e] = csO[ty * HEAD_SIZE + lane * EPL + e];
      for (int z = 1; z < BDZ; z++) {
        float zm = csM[z * BDY + ty], zl = csL[z * BDY + ty];
        float m2 = fmaxf(rm, zm);
        float a1 = (rm <= -FLT_MAX) ? 0.f
                                     : exp2f((rm - m2) * scale_log2);
        float a2 = (zm <= -FLT_MAX) ? 0.f
                                     : exp2f((zm - m2) * scale_log2);
        rl = rl * a1 + zl * a2;
        for (int e = 0; e < EPL; e++) {
          float zo = csO[(z * BDY + ty) * HEAD_SIZE + lane * EPL + e];
          ro[e] = ro[e] * a1 + zo * a2;
        }
        rm = m2;
      }
      const float inv = (rl > 0.f) ? (1.f / rl) : 0.f;
      __nv_bfloat16* op = out + (int64_t)seq_idx * q_stride
                          + qh * HEAD_SIZE + lane * EPL;
      for (int e = 0; e < EPL; e++)
        op[e] = __float2bfloat16(ro[e] * inv);
      if (lse_out && lane == 0)
        lse_out[seq_idx * num_q_heads + qh] =
            m + __log2f(rl) / scale_log2;
    }
  } else {
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;
    __nv_bfloat16* op = out + (int64_t)seq_idx * q_stride
                        + qh * HEAD_SIZE + lane * EPL;
    for (int e = 0; e < EPL; e++)
      op[e] = __float2bfloat16(o_reg[e] * inv);
    if (lse_out && lane == 0)
      lse_out[seq_idx * num_q_heads + qh] =
          m + __log2f(l) / scale_log2;
  }
}
