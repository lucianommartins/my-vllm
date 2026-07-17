/*
 * Gemma4 fused prefill (context) attention — SM80 mma.sync + register softmax.
 *
 * Port of the Gate-D fused DECODE dataflow (gemma_paged_attention.cuh,
 * gemma_decode_fused_kernel) to prefill: the kernel that beat the wmma stream
 * decode on every measured cell, applied to the 18%-MFU wmma prefill v2.
 *
 * vs gemma_prefill_kernel_v2 (wmma):
 *   - BM=16 (one M tile): S[16, BLOCK_N] lives in mma.sync c-frags — the
 *     online softmax runs in REGISTERS; only [NWARP,16] f32 row-stats cross
 *     smem. No sS round-trip, no S store_matrix_sync.
 *   - QK on ALL warps (one n8 slice each) vs 4/16 warps; per-warp KCH-chunk
 *     accumulation instead of one ILP=1 serial 64-HMMA chain.
 *   - 2-stage cp.async ring on K(+V): tile t+1 streams while tile t computes,
 *     vs fully synchronous uint4 staging.
 *   - 3 CTA barriers/tile vs 4-5.
 *   - Register-O head-split across warps (HDPW = HEAD/NWARP per warp): dodges
 *     the O[16,512] 255-reg wall that blocked FA2-class softmax at BM=32
 *     (32 f32/lane at hd512/NW8, exactly like the decode kernel).
 *   - record640-native (GEMMA_CACHE_V3): K-hat via the record address
 *     permutation at stage time; V's 128 rotated channels overwritten from
 *     the record's rotor-original columns after QK (plain loads, no extra
 *     barrier) — same contract as the fused decode + SM80 record prefill v2.
 *
 * Rows = 16 consecutive query tokens of ONE (q_head, seq); grid mirrors v2:
 * (ceil(max_q_len/16), num_q_heads, num_seqs). Causal / sliding-window /
 * mm-prefix(bidirectional image spans) / non_causal(cascade) masks are
 * applied per c-frag element in registers. LSE output for cascade merge.
 */
#pragma once

#include "../../attention/attention_dtypes.h"
#include "../../cuda_compat.h"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <float.h>

namespace vllm {
namespace gemma_prefill {

#ifndef GEMMA_PF_LOG2E
#define GEMMA_PF_LOG2E 1.4426950408889634f
#endif

// sm80 mma.sync/ldmatrix helpers (same layouts as the decode kernel;
// duplicated here to keep this header self-contained).
__device__ __forceinline__ uint32_t pf_smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void pf_ldm_x4(uint32_t* r, const void* p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(pf_smem_addr(p)));
}
__device__ __forceinline__ void pf_ldm_x2(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(pf_smem_addr(p)));
}
__device__ __forceinline__ void pf_ldm_x4t(uint32_t* r, const void* p) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(pf_smem_addr(p)));
}
__device__ __forceinline__ void pf_mma_16816(float* d, const uint32_t* a,
                                             const uint32_t* b,
                                             const float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

#define PF_FUSED_CDIV(a, b) (((a) + (b) - 1) / (b))

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int NWARP_T,
          int GQA_GROUP, bool K_EQ_V, bool USE_SLIDING_WINDOW,
          bool USE_MM_PREFIX = false, int MIN_CTA = 1>
__global__ void __launch_bounds__(NWARP_T * 32, MIN_CTA)
gemma_prefill_fused_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache, const cache_t* __restrict__ v_cache,
    const float scale, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, const int* __restrict__ cu_seqlens_q,
    const int max_num_blocks_per_seq, const int page_size, const int q_stride,
    const int64_t kv_stride_block, const int64_t kv_stride_slot,
    const int64_t kv_stride_head, const int sliding_window,
    const int* __restrict__ mm_prefix_ranges, const int max_mm_ranges,
    const bool non_causal = false, float* __restrict__ lse_out = nullptr,
    const int num_tokens = 0, const bool record640 = false) {
  constexpr int NWARP = NWARP_T;
  constexpr int BLOCK_N = 8 * NWARP;        // one n8 S-slice per warp
  constexpr int KCH = HEAD_SIZE / 16;       // QK k-chunks
  constexpr int HDPW = HEAD_SIZE / NWARP;   // O head-slice per warp
  constexpr int NPV = HDPW / 8;             // O n8 tiles per warp
  constexpr int SPAD = 8;
  constexpr int LDH = HEAD_SIZE + SPAD;
  constexpr int LDN = BLOCK_N + SPAD;
  constexpr int VEC = 16 / sizeof(cache_t);
  constexpr bool V_SMEM = !K_EQ_V;          // stage true V unless V==K/record
  constexpr int KTILE = BLOCK_N * LDH;
  constexpr int STAGE = KTILE * (V_SMEM ? 2 : 1);
  constexpr int NSTG = 2;                   // matches decode: depth 3 lost
  constexpr int MM_RANGE_CAP = 32;
  static_assert(HEAD_SIZE % 16 == 0 && HEAD_SIZE % NWARP == 0);
  static_assert(HDPW % 8 == 0, "head slice per warp must be n8-tileable");

  const int q_block = blockIdx.x;
  const int q_head = blockIdx.y;
  const int seq_idx = blockIdx.z;
  const int num_q_heads = gridDim.y;
  const int kv_head = q_head / GQA_GROUP;

  const int q_start = cu_seqlens_q[seq_idx];
  const int q_len = cu_seqlens_q[seq_idx + 1] - q_start;
  const int row0 = q_block * 16;
  if (row0 >= q_len) return;

  const int seq_len = seq_lens[seq_idx];
  const int context = seq_len - q_len;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int group = lane >> 2, tg = lane & 3;   // c-frag rows {group, group+8}
  const int nthreads = NWARP * 32;
  const int hvps = HEAD_SIZE / VEC;
  const float scale_log2 = scale * GEMMA_PF_LOG2E;

  extern __shared__ char pff_smem[];
  cache_t* sQ = reinterpret_cast<cache_t*>(pff_smem);       // [16, LDH]
  cache_t* sKV = sQ + 16 * LDH;                             // NSTG-stage ring
  float* sWm = reinterpret_cast<float*>(sKV + NSTG * STAGE); // [NWARP,16]
  float* sWl = sWm + NWARP * 16;                            // [NWARP,16]
  cache_t* sP = reinterpret_cast<cache_t*>(sWl + NWARP * 16); // [16, LDN]
  int* sMM = reinterpret_cast<int*>(sP + 16 * LDN);         // [CAP*2]
#define PFF_KBUF(s) (sKV + (s) * STAGE)
#define PFF_VBUF(s) (PFF_KBUF(s) + KTILE)

  // ---- stage Q rows (16 consecutive query tokens; zero-pad the tail) ----
  for (int iv = tid; iv < 16 * hvps; iv += nthreads) {
    const int r = iv / hvps, dv = (iv - r * hvps) * VEC;
    const int qr = row0 + r;
    if (qr < q_len) {
      const scalar_t* gq = q + (int64_t)(q_start + qr) * q_stride +
                           q_head * HEAD_SIZE + dv;
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) =
          *reinterpret_cast<const uint4*>(gq);
    } else {
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) = uint4{0, 0, 0, 0};
    }
  }

  // Zero the ring once (tail-pad rows must stay finite for the mma).
  for (int i = tid; i < NSTG * STAGE; i += nthreads)
    sKV[i] = static_cast<cache_t>(0);

  // mm-prefix spans -> smem (compiled out for text-only / full layers).
  int nr = 0;
  bool has_mm = false;
  if constexpr (USE_MM_PREFIX) {
    nr = (max_mm_ranges < MM_RANGE_CAP) ? max_mm_ranges : MM_RANGE_CAP;
    for (int i = tid; i < nr; i += nthreads) {
      sMM[2 * i] = mm_prefix_ranges[(seq_idx * max_mm_ranges + i) * 2];
      sMM[2 * i + 1] = mm_prefix_ranges[(seq_idx * max_mm_ranges + i) * 2 + 1];
    }
  }

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int last_q_abs = context + min(row0 + 15, q_len - 1);
  int kv_end = non_causal ? seq_len : min(last_q_abs + 1, seq_len);
  int kv_begin = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int lo = context + row0 - sliding_window + 1;
    kv_begin = (lo > 0) ? (lo / BLOCK_N) * BLOCK_N : 0;
  }
  __syncthreads();  // sQ + ring-zero + sMM visible
  if constexpr (USE_MM_PREFIX) {
    for (int i = 0; i < nr; i++) has_mm |= (sMM[2 * i] < sMM[2 * i + 1]);
    if (has_mm) {
      const int ext = (sliding_window > 0) ? sliding_window : seq_len;
      const int e2 = last_q_abs + 1 + ext;
      kv_end = (e2 < seq_len) ? e2 : seq_len;
    }
  }
  const int n_tiles = PF_FUSED_CDIV(kv_end - kv_begin, BLOCK_N);
  if (n_tiles <= 0) return;

  // Q fragments in registers for the whole KV loop (KCH*4 u32).
  uint32_t qreg[KCH][4];
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

  float m_run0 = -FLT_MAX, m_run1 = -FLT_MAX;
  float l_run0 = 0.f, l_run1 = 0.f;
  float oacc[NPV][4];
#pragma unroll
  for (int n = 0; n < NPV; n++)
    oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;

  // Per-thread row identities (c-frag rows group / group+8 = q tokens).
  const int qr0 = row0 + group, qr1 = row0 + group + 8;
  const bool rv0 = qr0 < q_len, rv1 = qr1 < q_len;
  const int qa0 = context + qr0, qa1 = context + qr1;

  // cp.async tile stage — K-hat via record permutation when record640;
  // true-V plane staged alongside when V_SMEM. Same contract as decode.
#define PFF_STAGE(ti, st)                                                      \
  do {                                                                         \
    const int _kv0 = kv_begin + (ti) * BLOCK_N;                                \
    const int _ntok = min(BLOCK_N, seq_len - _kv0);                            \
    for (int iv = tid; iv < BLOCK_N * hvps; iv += nthreads) {                  \
      const int n = iv / hvps, dv = (iv - n * hvps) * VEC;                     \
      if (n < _ntok) {                                                         \
        const int tok = _kv0 + n;                                              \
        const int64_t phys = block_table[tok / page_size];                     \
        const int sdv = !record640 ? dv                                        \
                        : (dv < 64)    ? dv + 512                              \
                        : (dv < 256)   ? dv - 64                               \
                        : (dv < 320)   ? dv + 320                              \
                                       : dv - 128;                             \
        const int64_t off = phys * kv_stride_block +                           \
                            (int64_t)(tok % page_size) * kv_stride_slot +      \
                            kv_head * kv_stride_head;                      \
        __pipeline_memcpy_async(PFF_KBUF(st) + n * LDH + dv,                   \
                                k_cache + off + sdv, 16);                      \
        if (V_SMEM)                                                            \
          __pipeline_memcpy_async(PFF_VBUF(st) + n * LDH + dv,                 \
                                  v_cache + off + dv, 16);                     \
      }                                                                        \
    }                                                                          \
    __pipeline_commit();                                                       \
  } while (0)

  PFF_STAGE(0, 0);

  for (int t = 0; t < n_tiles; t++) {
    const int cur = (NSTG == 2) ? (t & 1) : (t % NSTG);
    {
      const int rem = n_tiles - 1 - t;
      __pipeline_wait_prior(rem >= NSTG - 2 ? NSTG - 2 : rem);
    }
    __syncthreads();  // B1: staged K(+V) visible

    if (t + NSTG - 1 < n_tiles) PFF_STAGE(t + NSTG - 1, (t + NSTG - 1) % NSTG);

    const int kv0 = kv_begin + t * BLOCK_N;
    const int n_tok = min(BLOCK_N, seq_len - kv0);
    cache_t* kbuf = PFF_KBUF(cur);
    cache_t* vbuf = V_SMEM ? PFF_VBUF(cur) : kbuf;

    // ---- QK: this warp's n8 slice, Q from registers ----
    float sacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < KCH; c++) {
      uint32_t kb[2];
      pf_ldm_x2(kb, kbuf + (warp * 8 + (lane & 7)) * LDH + c * 16 +
                        ((lane >> 3) & 1) * 8);
      pf_mma_16816(sacc, qreg[c], kb, sacc);
    }

    // ---- masking (causal / sliding / mm-prefix / non_causal) ----
    const int col0 = kv0 + warp * 8 + 2 * tg;
    const bool v0 = (warp * 8 + 2 * tg) < n_tok;
    const bool v1 = (warp * 8 + 2 * tg + 1) < n_tok;
    bool e00 = rv0 && v0 && (non_causal || col0 <= qa0);
    bool e01 = rv0 && v1 && (non_causal || col0 + 1 <= qa0);
    bool e10 = rv1 && v0 && (non_causal || col0 <= qa1);
    bool e11 = rv1 && v1 && (non_causal || col0 + 1 <= qa1);
    if (USE_SLIDING_WINDOW && sliding_window > 0 && !non_causal) {
      e00 = e00 && (col0 > qa0 - sliding_window);
      e01 = e01 && (col0 + 1 > qa0 - sliding_window);
      e10 = e10 && (col0 > qa1 - sliding_window);
      e11 = e11 && (col0 + 1 > qa1 - sliding_window);
    }
    if constexpr (USE_MM_PREFIX) {
      if (has_mm) {
#pragma unroll 1
        for (int i = 0; i < nr; i++) {
          const int s = sMM[2 * i], e = sMM[2 * i + 1];
          if (s < e) {
            const bool q0in = qa0 >= s && qa0 <= e;
            const bool q1in = qa1 >= s && qa1 <= e;
            e00 = e00 || (rv0 && v0 && q0in && col0 >= s && col0 <= e);
            e01 = e01 || (rv0 && v1 && q0in && col0 + 1 >= s && col0 + 1 <= e);
            e10 = e10 || (rv1 && v0 && q1in && col0 >= s && col0 <= e);
            e11 = e11 || (rv1 && v1 && q1in && col0 + 1 >= s && col0 + 1 <= e);
          }
        }
      }
    }
    if (!e00) sacc[0] = -FLT_MAX;
    if (!e01) sacc[1] = -FLT_MAX;
    if (!e10) sacc[2] = -FLT_MAX;
    if (!e11) sacc[3] = -FLT_MAX;
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
    __syncthreads();  // B2: stats visible; QK reads of kbuf complete

    // ---- record640: rebuild true V's 128 rotated channels in place ----
    if constexpr (K_EQ_V) {
      if (record640) {
        constexpr int NV16 = 128 / VEC;
        for (int i = tid; i < BLOCK_N * NV16; i += nthreads) {
          const int n = i / NV16, k = i - n * NV16;
          if (n < n_tok) {
            const int dv =
                (k < NV16 / 2) ? k * VEC : 256 + (k - NV16 / 2) * VEC;
            const int src = (dv < 64) ? dv + 384 : dv + 192;
            const int tok = kv0 + n;
            const int64_t phys = block_table[tok / page_size];
            *reinterpret_cast<uint4*>(kbuf + n * LDH + dv) =
                *reinterpret_cast<const uint4*>(
                    k_cache + phys * kv_stride_block +
                    (int64_t)(tok % page_size) * kv_stride_slot +
                    kv_head * kv_stride_head + src);
          }
        }
      }
    }

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
    float p0 = e00 && mn0 > -FLT_MAX ? exp2f((sacc[0] - mn0) * scale_log2) : 0.f;
    float p1 = e01 && mn0 > -FLT_MAX ? exp2f((sacc[1] - mn0) * scale_log2) : 0.f;
    float p2 = e10 && mn1 > -FLT_MAX ? exp2f((sacc[2] - mn1) * scale_log2) : 0.f;
    float p3 = e11 && mn1 > -FLT_MAX ? exp2f((sacc[3] - mn1) * scale_log2) : 0.f;
    m_run0 = mn0; m_run1 = mn1;
    {
      __nv_bfloat162* d0 = reinterpret_cast<__nv_bfloat162*>(
          sP + group * LDN + warp * 8 + 2 * tg);
      __nv_bfloat162* d1 = reinterpret_cast<__nv_bfloat162*>(
          sP + (group + 8) * LDN + warp * 8 + 2 * tg);
      *d0 = __floats2bfloat162_rn(p0, p1);
      *d1 = __floats2bfloat162_rn(p2, p3);
    }
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
#pragma unroll
    for (int n = 0; n < NPV; n++) {
      oacc[n][0] *= al0; oacc[n][1] *= al0;
      oacc[n][2] *= al1; oacc[n][3] *= al1;
    }
    __syncthreads();  // B3: sP + sWl + V-overwrite visible

    // ---- PV: O[:, warp's HDPW slice] += P[16, BLOCK_N] @ V ----
#pragma unroll
    for (int kk = 0; kk < BLOCK_N / 16; kk++) {
      uint32_t pa[4];
      pf_ldm_x4(pa, sP + (lane & 15) * LDN + kk * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int b = 0; b < NPV / 2; b++) {
        const int hd = warp * HDPW + b * 16;
        uint32_t vb[4];
        pf_ldm_x4t(vb, vbuf + (kk * 16 + (lane & 15)) * LDH + hd +
                           (lane >> 4) * 8);
        pf_mma_16816(oacc[2 * b], pa, &vb[0], oacc[2 * b]);
        pf_mma_16816(oacc[2 * b + 1], pa, &vb[2], oacc[2 * b + 1]);
      }
    }
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
#undef PFF_STAGE

  // ---- epilogue: per-thread c-frag rows -> gmem; optional LSE ----
  const float inv0 = (l_run0 > 0.f) ? (1.f / l_run0) : 0.f;
  const float inv1 = (l_run1 > 0.f) ? (1.f / l_run1) : 0.f;
#pragma unroll
  for (int n = 0; n < NPV; n++) {
    const int hd = warp * HDPW + n * 8 + 2 * tg;
    if (rv0) {
      scalar_t* go = out + (int64_t)(q_start + qr0) * num_q_heads * HEAD_SIZE +
                     q_head * HEAD_SIZE + hd;
      from_float(go[0], oacc[n][0] * inv0);
      from_float(go[1], oacc[n][1] * inv0);
    }
    if (rv1) {
      scalar_t* go = out + (int64_t)(q_start + qr1) * num_q_heads * HEAD_SIZE +
                     q_head * HEAD_SIZE + hd;
      from_float(go[0], oacc[n][2] * inv1);
      from_float(go[1], oacc[n][3] * inv1);
    }
  }
  if (lse_out != nullptr && warp == 0 && tg == 0) {
    if (rv0)
      lse_out[(int64_t)q_head * num_tokens + (q_start + qr0)] =
          m_run0 * scale + logf(l_run0 > 0.f ? l_run0 : 1e-30f);
    if (rv1)
      lse_out[(int64_t)q_head * num_tokens + (q_start + qr1)] =
          m_run1 * scale + logf(l_run1 > 0.f ? l_run1 : 1e-30f);
  }
#undef PFF_KBUF
#undef PFF_VBUF
}

// ===========================================================================
// fused2: BM=32 two-M-tile-pass register-softmax prefill (hd512 k_eq_v v1).
//
// The BM=16 port (above) lost 13-17% to halved K-tile arithmetic intensity.
// This keeps v2's BM=32 intensity: each staged K tile serves TWO M=16 passes
// with the Gate-D register machinery per pass. hd512-specific budget:
//   - Q for ONE pass resident (KCH=32 -> 128 regs); pass-order ALTERNATES per
//     tile so only one qreg reload happens per tile (~131KB smem reads).
//   - O both passes resident: 2 x NPV x 4 = 64 f32 (NWARP=8, HDPW=64).
//   - smem ~106KB (sQ 33.3K + single sKV 66.6K + sP[2] 4.6K + stats 4K)
//     -> 1 CTA/SM; staging is exposed (v1 accepts; half-tile ring = follow-up).
//   - 4 barriers/tile: B0 pv-done (buffer reuse), B1 staged, B2 both-QK done,
//     B3 both-P + V visible. Tokamax causal-split: interior tiles skip all
//     mask evaluation.
//   - k_eq_v only (PV reuses the K tile; record640 strip overwrite after B2,
//     cp.async overlapped with the softmax phase). E-series hd512 stays on v2.
// ===========================================================================
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int NWARP_T,
          int GQA_GROUP, int MIN_CTA = 1>
__global__ void __launch_bounds__(NWARP_T * 32, MIN_CTA)
gemma_prefill_fused2_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache, const cache_t* __restrict__ v_cache,
    const float scale, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, const int* __restrict__ cu_seqlens_q,
    const int max_num_blocks_per_seq, const int page_size, const int q_stride,
    const int64_t kv_stride_block, const int64_t kv_stride_slot,
    const int64_t kv_stride_head, const int sliding_window,
    const int* __restrict__ mm_prefix_ranges, const int max_mm_ranges,
    const bool non_causal = false, float* __restrict__ lse_out = nullptr,
    const int num_tokens = 0, const bool record640 = false) {
  constexpr int NWARP = NWARP_T;
  constexpr int BLOCK_N = 8 * NWARP;        // one n8 S-slice per warp per pass
  constexpr int KCH = HEAD_SIZE / 16;
  constexpr int HDPW = HEAD_SIZE / NWARP;
  constexpr int NPV = HDPW / 8;
  constexpr int SPAD = 8;
  constexpr int LDH = HEAD_SIZE + SPAD;
  constexpr int LDN = BLOCK_N + SPAD;
  constexpr int VEC = 16 / sizeof(cache_t);
  static_assert(HEAD_SIZE % 16 == 0 && HDPW % 8 == 0);

  const int q_block = blockIdx.x;             // 32 query rows per CTA
  const int q_head = blockIdx.y;
  const int seq_idx = blockIdx.z;
  const int num_q_heads = gridDim.y;
  const int kv_head = q_head / GQA_GROUP;

  const int q_start = cu_seqlens_q[seq_idx];
  const int q_len = cu_seqlens_q[seq_idx + 1] - q_start;
  const int row0 = q_block * 32;
  if (row0 >= q_len) return;

  const int seq_len = seq_lens[seq_idx];
  const int context = seq_len - q_len;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int group = lane >> 2, tg = lane & 3;
  const int nthreads = NWARP * 32;
  const int hvps = HEAD_SIZE / VEC;
  const float scale_log2 = scale * GEMMA_PF_LOG2E;

  extern __shared__ char pf2_smem[];
  cache_t* sQ = reinterpret_cast<cache_t*>(pf2_smem);        // [32, LDH]
  cache_t* sKV = sQ + 32 * LDH;                              // [BN, LDH]
  float* sWm = reinterpret_cast<float*>(sKV + BLOCK_N * LDH); // [2][NWARP,16]
  float* sWl = sWm + 2 * NWARP * 16;                         // [2][NWARP,16]
  cache_t* sP = reinterpret_cast<cache_t*>(sWl + 2 * NWARP * 16); // [2][16,LDN]

  // ---- stage all 32 Q rows (cp.async; zero-pad tail) ----
  for (int iv = tid; iv < 32 * hvps; iv += nthreads) {
    const int r = iv / hvps, dv = (iv - r * hvps) * VEC;
    const int qr = row0 + r;
    if (qr < q_len) {
      __pipeline_memcpy_async(
          sQ + r * LDH + dv,
          q + (int64_t)(q_start + qr) * q_stride + q_head * HEAD_SIZE + dv, 16);
    } else {
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) = uint4{0, 0, 0, 0};
    }
  }
  __pipeline_commit();

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int last_q_abs = context + min(row0 + 31, q_len - 1);
  const int kv_end = non_causal ? seq_len : min(last_q_abs + 1, seq_len);
  const int n_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;  // kv_begin = 0 (full)

  // Per-pass per-thread state. Pass p covers rows row0 + p*16 + {group,
  // group+8}; only ONE pass's Q is register-resident at a time.
  float m_run[2][2], l_run[2][2];
  float oacc[2][NPV][4];
#pragma unroll
  for (int p = 0; p < 2; p++) {
    m_run[p][0] = m_run[p][1] = -FLT_MAX;
    l_run[p][0] = l_run[p][1] = 0.f;
#pragma unroll
    for (int n = 0; n < NPV; n++)
      oacc[p][n][0] = oacc[p][n][1] = oacc[p][n][2] = oacc[p][n][3] = 0.f;
  }
  const int qr_of[2][2] = {{row0 + group, row0 + group + 8},
                           {row0 + 16 + group, row0 + 16 + group + 8}};

  uint32_t qreg[KCH][4];
  int resident = 0;  // which pass's Q is in qreg
#define PF2_PULL_Q(P)                                                          \
  do {                                                                         \
    const cache_t* q0 = sQ + ((P) * 16 + group) * LDH;                         \
    const cache_t* q1 = sQ + ((P) * 16 + group + 8) * LDH;                     \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < KCH; c++) {                                            \
      qreg[c][0] = *reinterpret_cast<const uint32_t*>(q0 + c * 16 + 2 * tg);   \
      qreg[c][1] = *reinterpret_cast<const uint32_t*>(q1 + c * 16 + 2 * tg);   \
      qreg[c][2] =                                                             \
          *reinterpret_cast<const uint32_t*>(q0 + c * 16 + 2 * tg + 8);        \
      qreg[c][3] =                                                             \
          *reinterpret_cast<const uint32_t*>(q1 + c * 16 + 2 * tg + 8);        \
    }                                                                          \
  } while (0)

  __pipeline_wait_prior(0);
  __syncthreads();  // sQ visible
  PF2_PULL_Q(0);

  float sacc2[2][4];  // masked S fragments for both passes, this tile

  for (int t = 0; t < n_tiles; t++) {
    const int kv0 = t * BLOCK_N;
    const int n_tok = min(BLOCK_N, seq_len - kv0);

    __syncthreads();  // B0: prior tile's PV reads of sKV complete

    // ---- stage K tile (record sdv permutation; zero-pad tail) ----
    for (int iv = tid; iv < BLOCK_N * hvps; iv += nthreads) {
      const int n = iv / hvps, dv = (iv - n * hvps) * VEC;
      if (n < n_tok) {
        const int tok = kv0 + n;
        const int64_t phys = block_table[tok / page_size];
        const int sdv = !record640 ? dv
                        : (dv < 64)    ? dv + 512
                        : (dv < 256)   ? dv - 64
                        : (dv < 320)   ? dv + 320
                                       : dv - 128;
        __pipeline_memcpy_async(
            sKV + n * LDH + dv,
            k_cache + phys * kv_stride_block +
                (int64_t)(tok % page_size) * kv_stride_slot +
                kv_head * kv_stride_head + sdv,
            16);
      } else if (t == 0) {  // pad rows written once; later tiles reuse
        *reinterpret_cast<uint4*>(sKV + n * LDH + dv) = uint4{0, 0, 0, 0};
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();  // B1: staged K visible

    // Tokamax causal-split: interior tile = every column valid for every row.
    const bool interior =
        non_causal ? (n_tok == BLOCK_N)
                   : (n_tok == BLOCK_N && kv0 + BLOCK_N - 1 <= context + row0);

    // ---- QK both passes; pass order alternates so 1 qreg reload/tile ----
#pragma unroll
    for (int pp = 0; pp < 2; pp++) {
      const int p = (resident == 0) ? pp : 1 - pp;
      if (p != resident) { PF2_PULL_Q(p); resident = p; }
      float sacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < KCH; c++) {
        uint32_t kb[2];
        pf_ldm_x2(kb, sKV + (warp * 8 + (lane & 7)) * LDH + c * 16 +
                          ((lane >> 3) & 1) * 8);
        pf_mma_16816(sacc, qreg[c], kb, sacc);
      }
      if (!interior) {
        const int col0 = kv0 + warp * 8 + 2 * tg;
        const bool v0 = (warp * 8 + 2 * tg) < n_tok;
        const bool v1 = (warp * 8 + 2 * tg + 1) < n_tok;
        const bool rv0 = qr_of[p][0] < q_len, rv1 = qr_of[p][1] < q_len;
        const int qa0 = context + qr_of[p][0], qa1 = context + qr_of[p][1];
        if (!(rv0 && v0 && (non_causal || col0 <= qa0))) sacc[0] = -FLT_MAX;
        if (!(rv0 && v1 && (non_causal || col0 + 1 <= qa0))) sacc[1] = -FLT_MAX;
        if (!(rv1 && v0 && (non_causal || col0 <= qa1))) sacc[2] = -FLT_MAX;
        if (!(rv1 && v1 && (non_causal || col0 + 1 <= qa1))) sacc[3] = -FLT_MAX;
      }
      float sm0 = fmaxf(sacc[0], sacc[1]);
      float sm1 = fmaxf(sacc[2], sacc[3]);
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        sm0 = fmaxf(sm0, __shfl_xor_sync(0xffffffffu, sm0, o));
        sm1 = fmaxf(sm1, __shfl_xor_sync(0xffffffffu, sm1, o));
      }
      if (tg == 0) {
        sWm[p * NWARP * 16 + warp * 16 + group] = sm0;
        sWm[p * NWARP * 16 + warp * 16 + group + 8] = sm1;
      }
      sacc2[p][0] = sacc[0]; sacc2[p][1] = sacc[1];
      sacc2[p][2] = sacc[2]; sacc2[p][3] = sacc[3];
    }
    __syncthreads();  // B2: both passes' QK done + slice maxima visible

    // ---- record640: rebuild true V's 128 rotated channels (cp.async,
    // overlapped with the softmax phase below) ----
    if (record640) {
      constexpr int NV16 = 128 / VEC;
      for (int i = tid; i < BLOCK_N * NV16; i += nthreads) {
        const int n = i / NV16, k = i - n * NV16;
        if (n < n_tok) {
          const int dv = (k < NV16 / 2) ? k * VEC : 256 + (k - NV16 / 2) * VEC;
          const int src = (dv < 64) ? dv + 384 : dv + 192;
          const int tok = kv0 + n;
          const int64_t phys = block_table[tok / page_size];
          __pipeline_memcpy_async(
              sKV + n * LDH + dv,
              k_cache + phys * kv_stride_block +
                  (int64_t)(tok % page_size) * kv_stride_slot +
                  kv_head * kv_stride_head + src,
              16);
        }
      }
      __pipeline_commit();
    }

    // ---- softmax both passes (registers); P -> sP[p] ----
#pragma unroll
    for (int p = 0; p < 2; p++) {
      float tm0 = -FLT_MAX, tm1 = -FLT_MAX;
#pragma unroll
      for (int w = 0; w < NWARP; w++) {
        tm0 = fmaxf(tm0, sWm[p * NWARP * 16 + w * 16 + group]);
        tm1 = fmaxf(tm1, sWm[p * NWARP * 16 + w * 16 + group + 8]);
      }
      const float mn0 = fmaxf(m_run[p][0], tm0);
      const float mn1 = fmaxf(m_run[p][1], tm1);
      const float al0 =
          (m_run[p][0] <= -FLT_MAX) ? 0.f
                                    : exp2f((m_run[p][0] - mn0) * scale_log2);
      const float al1 =
          (m_run[p][1] <= -FLT_MAX) ? 0.f
                                    : exp2f((m_run[p][1] - mn1) * scale_log2);
      const float* sa = sacc2[p];
      const float p0 = (sa[0] > -FLT_MAX && mn0 > -FLT_MAX)
                           ? exp2f((sa[0] - mn0) * scale_log2) : 0.f;
      const float p1 = (sa[1] > -FLT_MAX && mn0 > -FLT_MAX)
                           ? exp2f((sa[1] - mn0) * scale_log2) : 0.f;
      const float p2 = (sa[2] > -FLT_MAX && mn1 > -FLT_MAX)
                           ? exp2f((sa[2] - mn1) * scale_log2) : 0.f;
      const float p3 = (sa[3] > -FLT_MAX && mn1 > -FLT_MAX)
                           ? exp2f((sa[3] - mn1) * scale_log2) : 0.f;
      m_run[p][0] = mn0; m_run[p][1] = mn1;
      cache_t* sPp = sP + p * 16 * LDN;
      *reinterpret_cast<__nv_bfloat162*>(
          sPp + group * LDN + warp * 8 + 2 * tg) = __floats2bfloat162_rn(p0, p1);
      *reinterpret_cast<__nv_bfloat162*>(
          sPp + (group + 8) * LDN + warp * 8 + 2 * tg) =
          __floats2bfloat162_rn(p2, p3);
      float ss0 = p0 + p1, ss1 = p2 + p3;
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        ss0 += __shfl_xor_sync(0xffffffffu, ss0, o);
        ss1 += __shfl_xor_sync(0xffffffffu, ss1, o);
      }
      if (tg == 0) {
        sWl[p * NWARP * 16 + warp * 16 + group] = ss0;
        sWl[p * NWARP * 16 + warp * 16 + group + 8] = ss1;
      }
#pragma unroll
      for (int n = 0; n < NPV; n++) {
        oacc[p][n][0] *= al0; oacc[p][n][1] *= al0;
        oacc[p][n][2] *= al1; oacc[p][n][3] *= al1;
      }
      // stash alphas for the l update after PV (per-pass, per-row)
      sacc2[p][0] = al0; sacc2[p][1] = al1;  // reuse: S no longer needed
    }
    if (record640) __pipeline_wait_prior(0);
    __syncthreads();  // B3: both sP halves + V(strip) visible

    // ---- PV both passes: oacc[p] += P[p] @ V ----
#pragma unroll
    for (int p = 0; p < 2; p++) {
      const cache_t* sPp = sP + p * 16 * LDN;
#pragma unroll
      for (int kk = 0; kk < BLOCK_N / 16; kk++) {
        uint32_t pa[4];
        pf_ldm_x4(pa, sPp + (lane & 15) * LDN + kk * 16 + (lane >> 4) * 8);
#pragma unroll
        for (int b = 0; b < NPV / 2; b++) {
          const int hd = warp * HDPW + b * 16;
          uint32_t vb[4];
          pf_ldm_x4t(vb, sKV + (kk * 16 + (lane & 15)) * LDH + hd +
                             (lane >> 4) * 8);
          pf_mma_16816(oacc[p][2 * b], pa, &vb[0], oacc[p][2 * b]);
          pf_mma_16816(oacc[p][2 * b + 1], pa, &vb[2], oacc[p][2 * b + 1]);
        }
      }
      float tl0 = 0.f, tl1 = 0.f;
#pragma unroll
      for (int w = 0; w < NWARP; w++) {
        tl0 += sWl[p * NWARP * 16 + w * 16 + group];
        tl1 += sWl[p * NWARP * 16 + w * 16 + group + 8];
      }
      l_run[p][0] = l_run[p][0] * sacc2[p][0] + tl0;
      l_run[p][1] = l_run[p][1] * sacc2[p][1] + tl1;
    }
  }
#undef PF2_PULL_Q

  // ---- epilogue: both passes' c-frag rows -> gmem; optional LSE ----
#pragma unroll
  for (int p = 0; p < 2; p++) {
    const float inv0 = (l_run[p][0] > 0.f) ? (1.f / l_run[p][0]) : 0.f;
    const float inv1 = (l_run[p][1] > 0.f) ? (1.f / l_run[p][1]) : 0.f;
    const bool rv0 = qr_of[p][0] < q_len, rv1 = qr_of[p][1] < q_len;
#pragma unroll
    for (int n = 0; n < NPV; n++) {
      const int hd = warp * HDPW + n * 8 + 2 * tg;
      if (rv0) {
        scalar_t* go = out +
            (int64_t)(q_start + qr_of[p][0]) * num_q_heads * HEAD_SIZE +
            q_head * HEAD_SIZE + hd;
        from_float(go[0], oacc[p][n][0] * inv0);
        from_float(go[1], oacc[p][n][1] * inv0);
      }
      if (rv1) {
        scalar_t* go = out +
            (int64_t)(q_start + qr_of[p][1]) * num_q_heads * HEAD_SIZE +
            q_head * HEAD_SIZE + hd;
        from_float(go[0], oacc[p][n][2] * inv1);
        from_float(go[1], oacc[p][n][3] * inv1);
      }
    }
    if (lse_out != nullptr && warp == 0 && tg == 0) {
      if (rv0)
        lse_out[(int64_t)q_head * num_tokens + (q_start + qr_of[p][0])] =
            m_run[p][0] * scale +
            logf(l_run[p][0] > 0.f ? l_run[p][0] : 1e-30f);
      if (rv1)
        lse_out[(int64_t)q_head * num_tokens + (q_start + qr_of[p][1])] =
            m_run[p][1] * scale +
            logf(l_run[p][1] > 0.f ? l_run[p][1] : 1e-30f);
    }
  }
}

#undef PF_FUSED_CDIV

}  // namespace gemma_prefill
}  // namespace vllm
