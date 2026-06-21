/*
 * Gemma4-optimized prefill (context) attention — tensor-core (wmma) flash
 * attention forward, specialized for hd=512 full-attention layers where neither
 * FA2/FA3 (cap 256) nor Triton/FSA tune well.
 *
 * One CTA per (q_block, q_head, seq). BM query rows attend causally to the
 * paged KV cache. O[BM x HEAD] f32 lives in shared memory (the hd=512
 * accumulator is too big for registers); Q is staged once; K/V share one
 * staging buffer (QK completes before PV; for k_eq_v the K buffer is reused as
 * V). Online softmax (base-2) with a smem round-trip for S/P.
 *
 * NHD paged KV cache: (num_blocks, page_size, num_kv_heads, head_size)
 */
#pragma once

#include "../../attention/attention_dtypes.h"
#include "../../cuda_compat.h"

#include <mma.h>
#include <float.h>

namespace vllm {
namespace gemma_prefill {

using namespace nvcuda;
static constexpr float LOG2E = 1.4426950408889634f;

// Grid:  (cdiv(max_q_len, BLOCK_M), num_q_heads, num_seqs)
// Block: NUM_WARPS * 32
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_M,
          int BLOCK_N, int NUM_WARPS, int GQA_GROUP, bool K_EQ_V,
          bool USE_SLIDING_WINDOW>
__global__ void gemma_prefill_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, num_q_heads, HEAD_SIZE]
    const scalar_t* __restrict__ q,      // [num_tokens, num_q_heads, HEAD_SIZE]
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,      // context length per seq (== key len)
    const int* __restrict__ cu_seqlens_q,  // [num_seqs + 1] query offsets
    const int max_num_blocks_per_seq,
    const int page_size,
    const int q_stride,
    const int64_t kv_stride_block,
    const int64_t kv_stride_slot,
    const int64_t kv_stride_head,
    const int sliding_window) {
  constexpr int MT = BLOCK_M / 16;   // M wmma tiles
  constexpr int NT = BLOCK_N / 16;   // N wmma tiles (QK)
  constexpr int DT = HEAD_SIZE / 16; // head wmma tiles
  constexpr int VEC = 16 / sizeof(cache_t);

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
  const int nthreads = NUM_WARPS * 32;

  extern __shared__ char pf_smem_raw[];
  cache_t* sQ = reinterpret_cast<cache_t*>(pf_smem_raw);          // [BM, HEAD]
  cache_t* sKV = sQ + BLOCK_M * HEAD_SIZE;                        // [BN, HEAD]
  cache_t* sP = sKV + BLOCK_N * HEAD_SIZE;                        // [BM, BN]
  float* sS = reinterpret_cast<float*>(sP + BLOCK_M * BLOCK_N);   // [BM, BN]
  float* sO = sS + BLOCK_M * BLOCK_N;                             // [BM, HEAD]
  float* sM = sO + BLOCK_M * HEAD_SIZE;                           // [BM]
  float* sL = sM + BLOCK_M;                                       // [BM]
  float* sA = sL + BLOCK_M;                                       // [BM] alpha

  for (int i = tid; i < BLOCK_M * HEAD_SIZE; i += nthreads) sO[i] = 0.f;
  for (int r = tid; r < BLOCK_M; r += nthreads) { sM[r] = -FLT_MAX; sL[r] = 0.f; }

  // stage Q for this block
  const int hvps = HEAD_SIZE / VEC;
  for (int iv = tid; iv < BLOCK_M * hvps; iv += nthreads) {
    const int r = iv / hvps;
    const int dv = (iv - r * hvps) * VEC;
    const int qr = row0 + r;
    if (qr < q_len) {
      const scalar_t* gq = q + (int64_t)(q_start + qr) * q_stride
                             + q_head * HEAD_SIZE + dv;
      *reinterpret_cast<uint4*>(sQ + r * HEAD_SIZE + dv) =
          *reinterpret_cast<const uint4*>(gq);
    } else {
      *reinterpret_cast<uint4*>(sQ + r * HEAD_SIZE + dv) = uint4{0, 0, 0, 0};
    }
  }
  __syncthreads();

  const float scale_log2 = scale * LOG2E;
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  const int last_q_abs = context + min(row0 + BLOCK_M - 1, q_len - 1);
  int kv_end = min(last_q_abs + 1, seq_len);
  int kv_begin = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int first_q_abs = context + row0;
    int lo = first_q_abs - sliding_window + 1;
    kv_begin = (lo > 0) ? (lo / BLOCK_N) * BLOCK_N : 0;
  }

  for (int kv0 = kv_begin; kv0 < kv_end; kv0 += BLOCK_N) {
    const int n_tok = min(BLOCK_N, seq_len - kv0);

    // stage K block into sKV
    for (int iv = tid; iv < BLOCK_N * hvps; iv += nthreads) {
      const int n = iv / hvps;
      const int dv = (iv - n * hvps) * VEC;
      if (n < n_tok) {
        const int tok = kv0 + n;
        const int64_t phys = block_table[tok / page_size];
        const int slot = tok % page_size;
        const cache_t* gk = k_cache + phys * kv_stride_block
                              + slot * kv_stride_slot
                              + kv_head * kv_stride_head + dv;
        *reinterpret_cast<uint4*>(sKV + n * HEAD_SIZE + dv) =
            *reinterpret_cast<const uint4*>(gk);
      } else {
        *reinterpret_cast<uint4*>(sKV + n * HEAD_SIZE + dv) = uint4{0, 0, 0, 0};
      }
    }
    __syncthreads();

    // ---- QK: S = Q @ K^T (4 warps cover MT x NT tiles) ----
    {
      const int mt = warp % MT;
      const int nt = warp / MT;
      if (nt < NT) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int kt = 0; kt < DT; kt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, cache_t, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, cache_t, wmma::col_major> fb;
          wmma::load_matrix_sync(fa, sQ + mt * 16 * HEAD_SIZE + kt * 16, HEAD_SIZE);
          // K stored [BN, HEAD] row-major; col_major load over (k,n) gives K^T.
          wmma::load_matrix_sync(fb, sKV + nt * 16 * HEAD_SIZE + kt * 16, HEAD_SIZE);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(sS + mt * 16 * BLOCK_N + nt * 16, acc, BLOCK_N,
                                wmma::mem_row_major);
      }
    }
    __syncthreads();

    // ---- mask + online softmax (one row per active thread) ----
    for (int r = tid; r < BLOCK_M; r += nthreads) {
      const int qr = row0 + r;
      const int q_abs = context + qr;
      float rmax = -FLT_MAX;
#pragma unroll 1
      for (int c = 0; c < BLOCK_N; c++) {
        const int k_abs = kv0 + c;
        bool valid = (qr < q_len) && (c < n_tok) && (k_abs <= q_abs);
        if (USE_SLIDING_WINDOW && sliding_window > 0)
          valid = valid && (k_abs > q_abs - sliding_window);
        float s = valid ? sS[r * BLOCK_N + c] : -FLT_MAX;
        sS[r * BLOCK_N + c] = s;
        rmax = fmaxf(rmax, s);
      }
      const float m_old = sM[r];
      const float m_new = fmaxf(m_old, rmax);
      const float alpha = (m_old <= -FLT_MAX) ? 0.f
                                              : exp2f((m_old - m_new) * scale_log2);
      float rsum = 0.f;
#pragma unroll 1
      for (int c = 0; c < BLOCK_N; c++) {
        float p = (m_new <= -FLT_MAX) ? 0.f
                                      : exp2f((sS[r * BLOCK_N + c] - m_new) * scale_log2);
        sP[r * BLOCK_N + c] = static_cast<cache_t>(p);
        rsum += p;
      }
      sM[r] = m_new;
      sL[r] = sL[r] * alpha + rsum;
      sA[r] = alpha;
    }
    __syncthreads();

    // ---- rescale O by alpha ----
    for (int i = tid; i < BLOCK_M * HEAD_SIZE; i += nthreads) {
      sO[i] *= sA[i / HEAD_SIZE];
    }

    // ---- stage V (reuse sKV for k_eq_v) ----
    if constexpr (!K_EQ_V) {
      __syncthreads();
      for (int iv = tid; iv < BLOCK_N * hvps; iv += nthreads) {
        const int n = iv / hvps;
        const int dv = (iv - n * hvps) * VEC;
        if (n < n_tok) {
          const int tok = kv0 + n;
          const int64_t phys = block_table[tok / page_size];
          const int slot = tok % page_size;
          const cache_t* gv = v_cache + phys * kv_stride_block
                                + slot * kv_stride_slot
                                + kv_head * kv_stride_head + dv;
          *reinterpret_cast<uint4*>(sKV + n * HEAD_SIZE + dv) =
              *reinterpret_cast<const uint4*>(gv);
        } else {
          *reinterpret_cast<uint4*>(sKV + n * HEAD_SIZE + dv) = uint4{0, 0, 0, 0};
        }
      }
    }
    __syncthreads();

    // ---- PV: O += P @ V  (tiles distributed round-robin across warps) ----
    constexpr int N_OTILES = MT * DT;
    for (int t = warp; t < N_OTILES; t += NUM_WARPS) {
      const int mt = t / DT;
      const int ht = t % DT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::load_matrix_sync(acc, sO + mt * 16 * HEAD_SIZE + ht * 16, HEAD_SIZE,
                             wmma::mem_row_major);
#pragma unroll
      for (int kt = 0; kt < NT; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, cache_t, wmma::row_major> fp;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, cache_t, wmma::row_major> fv;
        wmma::load_matrix_sync(fp, sP + mt * 16 * BLOCK_N + kt * 16, BLOCK_N);
        wmma::load_matrix_sync(fv, sKV + kt * 16 * HEAD_SIZE + ht * 16, HEAD_SIZE);
        wmma::mma_sync(acc, fp, fv, acc);
      }
      wmma::store_matrix_sync(sO + mt * 16 * HEAD_SIZE + ht * 16, acc, HEAD_SIZE,
                              wmma::mem_row_major);
    }
    __syncthreads();
  }

  // ---- epilogue: O /= L, write ----
  for (int iv = tid; iv < BLOCK_M * hvps; iv += nthreads) {
    const int r = iv / hvps;
    const int dv = (iv - r * hvps) * VEC;
    const int qr = row0 + r;
    if (qr >= q_len) continue;
    const float inv = (sL[r] > 0.f) ? (1.f / sL[r]) : 0.f;
    scalar_t tmp[VEC];
#pragma unroll
    for (int e = 0; e < VEC; e++)
      from_float(tmp[e], sO[r * HEAD_SIZE + dv + e] * inv);
    scalar_t* go = out + (int64_t)(q_start + qr) * num_q_heads * HEAD_SIZE
                     + q_head * HEAD_SIZE + dv;
    *reinterpret_cast<uint4*>(go) = *reinterpret_cast<uint4*>(tmp);
  }
}

// ===========================================================================
// v2: register-resident O accumulator, head-dim split across warps.
//
// O[BM x HEAD] is held in wmma accumulator fragments in REGISTERS across the
// whole KV loop (no per-block O smem round-trip). NUM_WARPS warps each own a
// HEAD/NUM_WARPS-wide head slice of O (HNT_W = HEAD/16/NUM_WARPS n-tiles).
// QK is computed by the first MT*NT warps (full-head contraction per S tile);
// softmax runs in smem; PV rescales each warp's O frags by the per-row alpha
// (using the SM80 wmma 16x16 f32 accumulator layout) and accumulates P@V.
//
// smem (no sO): sQ[BM,HEAD] sKV[BN,HEAD] sP[BM,BN] sS[BM,BN] sM[BM] sL[BM]
// sA[BM]; the epilogue reuses sQ/sKV as an f32 O staging area.
// ===========================================================================
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_M,
          int BLOCK_N, int NUM_WARPS, int GQA_GROUP, bool K_EQ_V,
          bool USE_SLIDING_WINDOW, int MIN_CTA = 1, bool USE_MM_PREFIX = false>
__global__ void __launch_bounds__(NUM_WARPS * 32, MIN_CTA)
    gemma_prefill_kernel_v2(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache, const cache_t* __restrict__ v_cache,
    const float scale, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, const int* __restrict__ cu_seqlens_q,
    const int max_num_blocks_per_seq, const int page_size, const int q_stride,
    const int64_t kv_stride_block, const int64_t kv_stride_slot,
    const int64_t kv_stride_head, const int sliding_window,
    // Multimodal bidirectional ("mm-prefix") image-token spans, int32
    // [num_seqs, max_mm_ranges, 2] (absolute [start,end] inclusive; valid iff
    // start<end). max_mm_ranges==0 -> no mm-prefix (text-only / full layers).
    const int* __restrict__ mm_prefix_ranges, const int max_mm_ranges) {
  constexpr int MT = BLOCK_M / 16;            // M tiles
  constexpr int NT = BLOCK_N / 16;            // QK N tiles
  constexpr int DT = HEAD_SIZE / 16;          // total head tiles
  constexpr int HNT_W = DT / NUM_WARPS;       // head tiles owned per warp
  constexpr int VEC = 16 / sizeof(cache_t);
  constexpr int MM_RANGE_CAP = 32;            // max image spans/seq we honor
  static_assert(DT % NUM_WARPS == 0, "head tiles must split across warps");
  static_assert(MT * NT <= NUM_WARPS, "QK tiles must fit in warps");

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

  // Padded smem leading dims to avoid wmma bank conflicts (ld multiple of 32
  // banks -> all rows hit the same banks). +8 staggers rows across banks.
  constexpr int SPAD = 8;
  constexpr int LDH = HEAD_SIZE + SPAD;   // smem row stride for Q/KV (and O-tmp)
  constexpr int LDN = BLOCK_N + SPAD;     // smem row stride for S/P

  extern __shared__ char pf_smem_raw[];
  cache_t* sQ = reinterpret_cast<cache_t*>(pf_smem_raw);          // [BM, LDH]
  cache_t* sKV = sQ + BLOCK_M * LDH;                              // [BN, LDH]
  cache_t* sP = sKV + BLOCK_N * LDH;                              // [BM, LDN]
  float* sS = reinterpret_cast<float*>(sP + BLOCK_M * LDN);       // [BM, LDN]
  float* sM = sS + BLOCK_M * LDN;                                 // [BM]
  float* sL = sM + BLOCK_M;                                       // [BM]
  float* sA = sL + BLOCK_M;                                       // [BM]
  int* sMM = reinterpret_cast<int*>(sA + BLOCK_M);               // [MM_RANGE_CAP*2]

  // O accumulator fragments (register-resident across the KV loop).
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> Ofrag[MT][HNT_W];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int j = 0; j < HNT_W; j++) wmma::fill_fragment(Ofrag[m][j], 0.0f);

  for (int r = tid; r < BLOCK_M; r += nthreads) { sM[r] = -FLT_MAX; sL[r] = 0.f; }

  const int hvps = HEAD_SIZE / VEC;
  for (int iv = tid; iv < BLOCK_M * hvps; iv += nthreads) {
    const int r = iv / hvps;
    const int dv = (iv - r * hvps) * VEC;
    const int qr = row0 + r;
    if (qr < q_len) {
      const scalar_t* gq = q + (int64_t)(q_start + qr) * q_stride
                             + q_head * HEAD_SIZE + dv;
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) =
          *reinterpret_cast<const uint4*>(gq);
    } else {
      *reinterpret_cast<uint4*>(sQ + r * LDH + dv) = uint4{0, 0, 0, 0};
    }
  }
  __syncthreads();

  // Preload this seq's mm-prefix image spans into smem (shared by every (q,k)
  // mask check below). Compiled out entirely for text-only / full-attn layers
  // (USE_MM_PREFIX=false) -> zero overhead. nr = #spans; has_mm = seq has a span.
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

  const float scale_log2 = scale * LOG2E;
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int last_q_abs = context + min(row0 + BLOCK_M - 1, q_len - 1);
  int kv_end = min(last_q_abs + 1, seq_len);
  int kv_begin = 0;
  if (USE_SLIDING_WINDOW && sliding_window > 0) {
    const int first_q_abs = context + row0;
    int lo = first_q_abs - sliding_window + 1;
    kv_begin = (lo > 0) ? (lo / BLOCK_N) * BLOCK_N : 0;
  }
  // Bidirectional image spans (<= sliding_window wide) can sit AHEAD of this
  // q-block's causal end -> extend kv_end to cover them (bounded by sw). kv_begin
  // (sliding start) already covers span starts since spans are <= sw wide.
  if constexpr (USE_MM_PREFIX) {
    if (has_mm) {
      const int ext = (sliding_window > 0) ? sliding_window : seq_len;
      const int e2 = last_q_abs + 1 + ext;
      kv_end = (e2 < seq_len) ? e2 : seq_len;
    }
  }

  for (int kv0 = kv_begin; kv0 < kv_end; kv0 += BLOCK_N) {
    const int n_tok = min(BLOCK_N, seq_len - kv0);

    // stage K block
    for (int iv = tid; iv < BLOCK_N * hvps; iv += nthreads) {
      const int n = iv / hvps;
      const int dv = (iv - n * hvps) * VEC;
      if (n < n_tok) {
        const int tok = kv0 + n;
        const int64_t phys = block_table[tok / page_size];
        const int slot = tok % page_size;
        const cache_t* gk = k_cache + phys * kv_stride_block
                              + slot * kv_stride_slot
                              + kv_head * kv_stride_head + dv;
        *reinterpret_cast<uint4*>(sKV + n * LDH + dv) =
            *reinterpret_cast<const uint4*>(gk);
      } else {
        *reinterpret_cast<uint4*>(sKV + n * LDH + dv) = uint4{0, 0, 0, 0};
      }
    }
    __syncthreads();

    // QK: first MT*NT warps each own one S[16x16] tile, full-head contraction.
    if (warp < MT * NT) {
      const int mt = warp / NT;
      const int nt = warp % NT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.0f);
#pragma unroll
      for (int kt = 0; kt < DT; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, cache_t, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, cache_t, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, sQ + mt * 16 * LDH + kt * 16, LDH);
        wmma::load_matrix_sync(fb, sKV + nt * 16 * LDH + kt * 16, LDH);
        wmma::mma_sync(s, fa, fb, s);
      }
      wmma::store_matrix_sync(sS + mt * 16 * LDN + nt * 16, s, LDN,
                              wmma::mem_row_major);
    }
    __syncthreads();

    // mask + online softmax, parallelized: one WARP per row (32 lanes = the
    // BLOCK_N columns), warp-reduced max/sum. All warps active (vs the previous
    // 1-thread-per-row serial phase that bottlenecked between barriers).
    // Generalized to BLOCK_N == k*32: each lane owns CPL = BLOCK_N/32 columns
    // (strided by 32), reduced within-lane then across the warp. CPL==1 recovers
    // the original one-col-per-lane path.
    static_assert(BLOCK_N % 32 == 0, "warp-softmax assumes BLOCK_N % 32 == 0");
    constexpr int CPL = BLOCK_N / 32;
    for (int r = warp; r < BLOCK_M; r += NUM_WARPS) {
      const int qr = row0 + r;
      const int q_abs = context + qr;
      float sv[CPL];
      float rmax = -FLT_MAX;
#pragma unroll
      for (int cc = 0; cc < CPL; ++cc) {
        const int c = lane + cc * 32;
        const int k_abs = kv0 + c;
        // keep = (causal AND sliding) OR mm_prefix(bidirectional within an image
        // span). The mm-prefix OR overrides causal+sliding inside a span.
        bool keep = (k_abs <= q_abs);
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
    __syncthreads();

    if constexpr (!K_EQ_V) {
      for (int iv = tid; iv < BLOCK_N * hvps; iv += nthreads) {
        const int n = iv / hvps;
        const int dv = (iv - n * hvps) * VEC;
        if (n < n_tok) {
          const int tok = kv0 + n;
          const int64_t phys = block_table[tok / page_size];
          const int slot = tok % page_size;
          const cache_t* gv = v_cache + phys * kv_stride_block
                                + slot * kv_stride_slot
                                + kv_head * kv_stride_head + dv;
          *reinterpret_cast<uint4*>(sKV + n * LDH + dv) =
              *reinterpret_cast<const uint4*>(gv);
        } else {
          *reinterpret_cast<uint4*>(sKV + n * LDH + dv) = uint4{0, 0, 0, 0};
        }
      }
      __syncthreads();
    }

    // PV: rescale O frags by per-row alpha (SM80 wmma acc layout), then P@V.
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
          wmma::load_matrix_sync(fv, sKV + kt * 16 * LDH + ht * 16, LDH);
          wmma::mma_sync(Ofrag[m][j], fp, fv, Ofrag[m][j]);
        }
      }
    }
    __syncthreads();
  }

  // epilogue: store O frags to reused smem (sQ/sKV as f32, stride LDH), /L.
  float* sOt = reinterpret_cast<float*>(sQ);  // spans sQ+sKV = [BM, LDH] f32
  __syncthreads();
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int j = 0; j < HNT_W; j++) {
      const int ht = warp * HNT_W + j;
      wmma::store_matrix_sync(sOt + m * 16 * LDH + ht * 16, Ofrag[m][j], LDH,
                              wmma::mem_row_major);
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
}

}  // namespace gemma_prefill
}  // namespace vllm
