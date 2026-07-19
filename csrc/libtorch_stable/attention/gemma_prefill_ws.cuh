/*
 * Gemma4 warp-specialized prefill (context) attention — SM80, hd512 k_eq_v.
 *
 * 3a / S1 (docs/gemma_prefill_3a_design.md VERIFIED REVISION §3'-§6'): a
 * producer/consumer restructuring of gemma_prefill_kernel_v2 at 1 CTA/SM.
 *
 *   - NW=20 (640 threads), __launch_bounds__(640,1) -> 1 CTA/SM.
 *   - CONSUMERS = warps 0..15: run v2's compute phases BIT-IDENTICALLY (4-warp
 *     QK, warp-softmax, register-O PV, epilogue). All compute-warp indexing
 *     (HNT_W, O-fragment ownership, softmax row map, PV head-tile map) is
 *     computed on 16 consumers, exactly as v2's NUM_WARPS=16.
 *   - PRODUCERS = warps 16..19: run a 3-stage BN32 cp.async K-ring, streaming
 *     K-hat 2 stages ahead (same sdv record permutation + 16B row-major chunks
 *     as v2). They stage K only; they carry no O and skip all compute, but DO
 *     participate in every __syncthreads().
 *   - record640 V-mutation is issued by CONSUMERS into the current ring buffer
 *     after B_qk (identical src permutation to v2), overlapping softmax.
 *
 * S1 does NOT change arithmetic (no k-split QK yet — that is S2), so the output
 * is bit-identical to v2. Purpose: isolate whether deterministic intra-CTA
 * staging-hiding at 1 CTA/SM repays the -1.5..-2.2% lost vs v2's 2 CTA/SM.
 *
 * Barrier protocol (per KV tile t; 4 handoffs = 2 syncthreads + 2 named pairs;
 * §5's 5th handoff, B_red, is a k-split-reduce barrier that only exists in S2):
 *   1. T_stage (named bar 1+t%3): producer issues K(t+2), tail-aware
 *      wait_group so stage t retires, then bar.arrive; consumer bar.sync before
 *      the first QK ldmatrix of stage t (cp.async visibility rule: issuer did
 *      the wait_group, the named bar publishes to consumers).
 *   2. B_qk (__syncthreads): after QK, before the V-mutation / softmax.
 *   3. B_sm (__syncthreads): after softmax + wait_prior(0), before PV
 *      (publishes sP AND the overwritten V bytes together).
 *   4. buffer-free (named bar 4+t%3): consumers bar.arrive after the last PV
 *      read of stage t; producers bar.sync before re-staging that buffer
 *      (ring depth 3 -> buffer t%3 reused at t+3; WAR guard, zero ring slack).
 * The shared syncthreads bound producer/consumer drift to <1 tile, which keeps
 * the named-barrier generations aligned (no cross-generation false-release).
 */
#pragma once

#include "../../attention/attention_dtypes.h"
#include "../../cuda_compat.h"

#include <cuda_pipeline.h>
#include <mma.h>
#include <float.h>

namespace vllm {
namespace gemma_prefill {

using namespace nvcuda;
#ifndef GEMMA_WS_LOG2E
#define GEMMA_WS_LOG2E 1.4426950408889634f
#endif

// SM80 named barriers (ids 1..15, thread count a multiple of 32). The "memory"
// clobber prevents the compiler from reordering shared-memory accesses across
// the barrier; the barrier itself provides the cross-thread visibility fence
// required after a producer's cp.async.wait_group (3b spec §2 item 4).
__device__ __forceinline__ void ws_bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void ws_bar_arrive(int id, int count) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// ===========================================================================
// gemma_prefill_ws_kernel — warp-specialized v2 (hd512 k_eq_v full-attention).
// NUM_CONSUMERS compute warps + NUM_PRODUCERS staging warps; NSTG-stage K ring.
// ===========================================================================
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_M,
          int BLOCK_N, int NUM_CONSUMERS, int NUM_PRODUCERS, int NSTG,
          int GQA_GROUP, bool K_EQ_V, bool USE_SLIDING_WINDOW,
          bool USE_MM_PREFIX = false>
__global__ void __launch_bounds__((NUM_CONSUMERS + NUM_PRODUCERS) * 32, 1)
    gemma_prefill_ws_kernel(
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
  constexpr int NUM_WARPS = NUM_CONSUMERS + NUM_PRODUCERS;
  constexpr int MT = BLOCK_M / 16;                 // M tiles
  constexpr int NT = BLOCK_N / 16;                 // QK N tiles
  constexpr int DT = HEAD_SIZE / 16;               // total head tiles
  constexpr int HNT_W = DT / NUM_CONSUMERS;        // head tiles per consumer warp
  constexpr int VEC = 16 / sizeof(cache_t);
  constexpr int MM_RANGE_CAP = 32;
  static_assert(DT % NUM_CONSUMERS == 0, "head tiles must split across consumers");
  static_assert(MT * NT <= NUM_CONSUMERS, "QK tiles must fit in consumer warps");

  const int q_block = blockIdx.x;
  const int q_head = blockIdx.y;
  const int seq_idx = blockIdx.z;
  const int kv_head = q_head / GQA_GROUP;
  const int num_q_heads = gridDim.y;

  const int q_start = cu_seqlens_q[seq_idx];
  const int q_len = cu_seqlens_q[seq_idx + 1] - q_start;
  const int row0 = q_block * BLOCK_M;
  if (row0 >= q_len) return;

  const int seq_len = seq_lens[seq_idx];
  const int context = seq_len - q_len;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int nthreads = NUM_WARPS * 32;
  const bool is_consumer = warp < NUM_CONSUMERS;

  constexpr int SPAD = 8;
  constexpr int LDH = HEAD_SIZE + SPAD;   // smem row stride for Q/KV (and O-tmp)
  constexpr int LDN = BLOCK_N + SPAD;     // smem row stride for S/P

  extern __shared__ char pfws_smem_raw[];
  cache_t* sQ = reinterpret_cast<cache_t*>(pfws_smem_raw);        // [BM, LDH]
  cache_t* sKV = sQ + BLOCK_M * LDH;                              // NSTG*[BN, LDH]
  cache_t* sP = sKV + NSTG * BLOCK_N * LDH;                       // [BM, LDN]
  float* sS = reinterpret_cast<float*>(sP + BLOCK_M * LDN);       // [BM, LDN]
  float* sM = sS + BLOCK_M * LDN;                                 // [BM]
  float* sL = sM + BLOCK_M;                                       // [BM]
  float* sA = sL + BLOCK_M;                                       // [BM]
  int* sMM = reinterpret_cast<int*>(sA + BLOCK_M);               // [MM_RANGE_CAP*2]
#define WS_KBUF(s) (sKV + (s) * BLOCK_N * LDH)

  // O accumulator fragments (register-resident across the KV loop). Producers
  // allocate them too (uniform code) but never use them; they cost a handful of
  // regs well within the 1-CTA/SM budget.
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> Ofrag[MT][HNT_W];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int j = 0; j < HNT_W; j++) wmma::fill_fragment(Ofrag[m][j], 0.0f);

  for (int r = tid; r < BLOCK_M; r += nthreads) { sM[r] = -FLT_MAX; sL[r] = 0.f; }

  // Q stage via cp.async (all threads); pad rows zero-filled (cp.async can't
  // synthesize zeros).
  const int hvps = HEAD_SIZE / VEC;
  for (int iv = tid; iv < BLOCK_M * hvps; iv += nthreads) {
    const int r = iv / hvps;
    const int dv = (iv - r * hvps) * VEC;
    const int qr = row0 + r;
    if (qr < q_len) {
      const scalar_t* gq = q + (int64_t)(q_start + qr) * q_stride
                             + q_head * HEAD_SIZE + dv;
      __pipeline_memcpy_async(sQ + r * LDH + dv, gq, 16);
    } else {
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) = uint4{0, 0, 0, 0};
    }
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  int nr = 0;
  bool has_mm = false;
  if constexpr (USE_MM_PREFIX) {
    nr = (max_mm_ranges < MM_RANGE_CAP) ? max_mm_ranges : MM_RANGE_CAP;
    for (int i = tid; i < nr; i += nthreads) {
      sMM[2 * i]     = mm_prefix_ranges[(seq_idx * max_mm_ranges + i) * 2];
      sMM[2 * i + 1] = mm_prefix_ranges[(seq_idx * max_mm_ranges + i) * 2 + 1];
    }
    if (nr > 0) __syncthreads();
    for (int i = 0; i < nr; i++) has_mm |= (sMM[2 * i] < sMM[2 * i + 1]);
  }

  const float scale_log2 = scale * GEMMA_WS_LOG2E;
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int last_q_abs = context + min(row0 + BLOCK_M - 1, q_len - 1);
  int kv_end = non_causal ? seq_len : min(last_q_abs + 1, seq_len);
  int kv_begin = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int first_q_abs = context + row0;
    int lo = first_q_abs - sliding_window + 1;
    kv_begin = (lo > 0) ? (lo / BLOCK_N) * BLOCK_N : 0;
  }
  if constexpr (USE_MM_PREFIX) {
    if (has_mm) {
      const int ext = (sliding_window > 0) ? sliding_window : seq_len;
      const int e2 = last_q_abs + 1 + ext;
      kv_end = (e2 < seq_len) ? e2 : seq_len;
    }
  }
  const int n_tiles = (kv_end - kv_begin + BLOCK_N - 1) / BLOCK_N;
  // n_tiles is CTA-uniform, so this guard is barrier-safe; n_tiles==0 (no KV)
  // falls straight through to the epilogue with O/L == 0.

  // Producer K-hat staging of tile `ti` into ring buffer `st` (record sdv
  // permutation; zero-pad the tail; producers only).
#define WS_STAGE_K(ti, st)                                                     \
  do {                                                                         \
    const int _kv0 = kv_begin + (ti) * BLOCK_N;                                \
    const int _ntok = min(BLOCK_N, seq_len - _kv0);                            \
    for (int iv = ptid; iv < BLOCK_N * hvps; iv += pthreads) {                 \
      const int n = iv / hvps;                                                 \
      const int dv = (iv - n * hvps) * VEC;                                    \
      if (n < _ntok) {                                                         \
        const int tok = _kv0 + n;                                             \
        const int64_t phys = block_table[tok / page_size];                    \
        const int slot = tok % page_size;                                     \
        const int sdv = !record640 ? dv                                       \
                        : (dv < 64)  ? dv + 512                               \
                        : (dv < 256) ? dv - 64                                \
                        : (dv < 320) ? dv + 320                               \
                                     : dv - 128;                              \
        __pipeline_memcpy_async(WS_KBUF(st) + n * LDH + dv,                    \
                                k_cache + phys * kv_stride_block               \
                                    + slot * kv_stride_slot                    \
                                    + kv_head * kv_stride_head + sdv,          \
                                16);                                           \
      } else {                                                                 \
        *reinterpret_cast<uint4*>(WS_KBUF(st) + n * LDH + dv) =                \
            uint4{0, 0, 0, 0};                                                 \
      }                                                                        \
    }                                                                          \
    __pipeline_commit();                                                       \
  } while (0)

  if (n_tiles > 0) {
    const int ptid = tid - NUM_CONSUMERS * 32;   // producer-local tid
    constexpr int pthreads = NUM_PRODUCERS * 32;

    // Producer prologue: stage the first NSTG-1 tiles (stages 0..NSTG-2).
    if (!is_consumer) {
#pragma unroll
      for (int s = 0; s < NSTG - 1; s++)
        if (s < n_tiles) WS_STAGE_K(s, s);
    }

    static_assert(BLOCK_N % 32 == 0, "warp-softmax assumes BLOCK_N % 32 == 0");
    constexpr int CPL = BLOCK_N / 32;

    for (int t = 0; t < n_tiles; t++) {
      const int kv0 = kv_begin + t * BLOCK_N;
      const int n_tok = min(BLOCK_N, seq_len - kv0);
      const int cur = t % NSTG;

      // ---- T_stage: producers fetch stage t+NSTG-1 and publish stage t ----
      if (!is_consumer) {
        const int fetch = t + NSTG - 1;
        if (fetch < n_tiles) {
          if (fetch >= NSTG) ws_bar_sync(4 + (fetch % NSTG), nthreads);  // WAR
          WS_STAGE_K(fetch, fetch % NSTG);
        }
        const int committed = min(t + NSTG, n_tiles);  // groups committed so far
        __pipeline_wait_prior(committed - 1 - t);       // retire stage t
        ws_bar_arrive(1 + cur, nthreads);
      } else {
        ws_bar_sync(1 + cur, nthreads);  // wait stage t K-hat visible
      }

      // ---- QK: consumer warps 0..MT*NT-1 own one S[16x16] tile each ----
      if (is_consumer && warp < MT * NT) {
        const int mt = warp / NT;
        const int nt = warp % NT;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
        wmma::fill_fragment(s, 0.0f);
#pragma unroll
        for (int kt = 0; kt < DT; kt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, cache_t, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, cache_t, wmma::col_major> fb;
          wmma::load_matrix_sync(fa, sQ + mt * 16 * LDH + kt * 16, LDH);
          wmma::load_matrix_sync(fb, WS_KBUF(cur) + nt * 16 * LDH + kt * 16, LDH);
          wmma::mma_sync(s, fa, fb, s);
        }
        wmma::store_matrix_sync(sS + mt * 16 * LDN + nt * 16, s, LDN,
                                wmma::mem_row_major);
      }
      __syncthreads();  // B_qk: all QK reads of stage `cur` complete

      // ---- record640: consumers rebuild true V's 128 rotated channels in the
      // current ring buffer (cp.async, overlaps softmax). Consumers are tids
      // 0..NUM_CONSUMERS*32-1, identical distribution to v2's tid loop. ----
      if constexpr (K_EQ_V) {
        if (record640 && is_consumer) {
          constexpr int NV16 = 128 / VEC;
          const int cons_threads = NUM_CONSUMERS * 32;
          for (int i = tid; i < BLOCK_N * NV16; i += cons_threads) {
            const int n = i / NV16;
            const int k = i - n * NV16;
            if (n < n_tok) {
              const int dv =
                  (k < NV16 / 2) ? k * VEC : 256 + (k - NV16 / 2) * VEC;
              const int src = (dv < 64) ? dv + 384 : dv + 192;
              const int tok = kv0 + n;
              const int64_t phys = block_table[tok / page_size];
              const int slot = tok % page_size;
              __pipeline_memcpy_async(
                  WS_KBUF(cur) + n * LDH + dv,
                  k_cache + phys * kv_stride_block + slot * kv_stride_slot +
                      kv_head * kv_stride_head + src,
                  16);
            }
          }
          __pipeline_commit();
        }
      }

      // ---- mask + online softmax: one consumer warp per row (v2-identical) ---
      if (is_consumer) {
        for (int r = warp; r < BLOCK_M; r += NUM_CONSUMERS) {
          const int qr = row0 + r;
          const int q_abs = context + qr;
          float sv[CPL];
          float rmax = -FLT_MAX;
#pragma unroll
          for (int cc = 0; cc < CPL; ++cc) {
            const int c = lane + cc * 32;
            const int k_abs = kv0 + c;
            bool keep = non_causal ? true : (k_abs <= q_abs);
            if (USE_SLIDING_WINDOW && sliding_window > 0)
              keep = keep && (k_abs > q_abs - sliding_window);
            if constexpr (USE_MM_PREFIX) {
              if (has_mm) {
#pragma unroll 1
                for (int i = 0; i < nr; i++) {
                  const int s = sMM[2 * i], e = sMM[2 * i + 1];
                  if (s < e && q_abs >= s && q_abs <= e && k_abs >= s &&
                      k_abs <= e) {
                    keep = true;
                    break;
                  }
                }
              }
            }
            bool valid = (qr < q_len) && (c < n_tok) && keep;
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
            sP[r * LDN + c] = static_cast<cache_t>(p);
            rsum += p;
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
        if constexpr (K_EQ_V) {
          if (record640) __pipeline_wait_prior(0);  // V bytes landed
        }
      }
      __syncthreads();  // B_sm: sP + rebuilt V visible before PV

      // ---- PV: consumers rescale O frags by per-row alpha, then P@V ----
      if (is_consumer) {
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
              wmma::load_matrix_sync(fv, WS_KBUF(cur) + kt * 16 * LDH + ht * 16, LDH);
              wmma::mma_sync(Ofrag[m][j], fp, fv, Ofrag[m][j]);
            }
          }
        }
      }

      // ---- buffer-free: publish that stage `cur` is done being read ----
      if (is_consumer) {
        ws_bar_arrive(4 + cur, nthreads);
      }
    }
  }
#undef WS_STAGE_K

  // store O frags to reused smem (sQ + ring[0] as f32, stride LDH), /L.
  float* sOt = reinterpret_cast<float*>(sQ);  // spans sQ + first K stage
  __syncthreads();
  if (is_consumer) {
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
      for (int j = 0; j < HNT_W; j++) {
        const int ht = warp * HNT_W + j;
        wmma::store_matrix_sync(sOt + m * 16 * LDH + ht * 16, Ofrag[m][j], LDH,
                                wmma::mem_row_major);
      }
  }
  __syncthreads();
  for (int iv = tid; iv < BLOCK_M * hvps; iv += nthreads) {
    const int r = iv / hvps;
    const int dv = (iv - r * hvps) * VEC;
    const int qr = row0 + r;
    if (qr >= q_len) continue;
    const float inv = (sL[r] > 0.f) ? (1.f / sL[r]) : 0.f;
    scalar_t tmp[VEC];
#pragma unroll
    for (int e = 0; e < VEC; e++)
      from_float(tmp[e], sOt[r * LDH + dv + e] * inv);
    scalar_t* go = out + (int64_t)(q_start + qr) * num_q_heads * HEAD_SIZE
                     + q_head * HEAD_SIZE + dv;
    *reinterpret_cast<uint4*>(go) = *reinterpret_cast<uint4*>(tmp);
  }
  if (lse_out != nullptr) {
    if (is_consumer) {
      for (int r = warp; r < BLOCK_M; r += NUM_CONSUMERS) {
        const int qr = row0 + r;
        if (qr < q_len && lane == 0) {
          const float l = sL[r];
          lse_out[(int64_t)q_head * num_tokens + (q_start + qr)] =
              sM[r] * scale + logf(l > 0.f ? l : 1e-30f);
        }
      }
    }
  }
#undef WS_KBUF
}

}  // namespace gemma_prefill
}  // namespace vllm
