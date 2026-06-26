/*
 * Gemma4-optimized prefill attention — SM90 (Hopper) variant.
 *
 * P2: QK-load overlap — during the QK phase (warps 0..MT*NT-1), the remaining
 * warps load the NEXT KV tile into a second smem buffer. This hides KV load
 * latency behind QK compute, eliminating one serialized load+sync per tile.
 *
 * Changes vs v2:
 *   - Double-buffered sKV: sKV[2][BN, LDH] (was sKV[BN, LDH])
 *   - QK warps and load warps run concurrently within the same __syncthreads block
 *   - Extra smem: +BN*LDH*sizeof(cache_t) = +33KB at hd512 (73KB → 106KB)
 *   - 2 CTAs/SM: 106KB × 2 = 212KB ≤ 228KB ✓ (same occupancy as v2)
 *   - Register usage identical to v2 (same compute, same warp/head split)
 *
 * The SM80 v2 kernel is NOT modified — this is a new kernel for SM90 only.
 */
#pragma once

#include "gemma_prefill_attention.cuh"

namespace vllm {
namespace gemma_prefill {
namespace sm90 {

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_M,
          int BLOCK_N, int NUM_WARPS, int GQA_GROUP, bool K_EQ_V,
          bool USE_SLIDING_WINDOW, int MIN_CTA = 1, bool USE_MM_PREFIX = false>
__global__ void __launch_bounds__(NUM_WARPS * 32, MIN_CTA)
    gemma_prefill_kernel_sm90(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache, const cache_t* __restrict__ v_cache,
    const float scale, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, const int* __restrict__ cu_seqlens_q,
    const int max_num_blocks_per_seq, const int page_size, const int q_stride,
    const int64_t kv_stride_block, const int64_t kv_stride_slot,
    const int64_t kv_stride_head, const int sliding_window,
    const int* __restrict__ mm_prefix_ranges, const int max_mm_ranges,
    const bool non_causal = false, float* __restrict__ lse_out = nullptr,
    const int num_tokens = 0) {

  constexpr int MT = BLOCK_M / 16;
  constexpr int NT = BLOCK_N / 16;
  constexpr int DT = HEAD_SIZE / 16;
  constexpr int HNT_W = DT / NUM_WARPS;
  constexpr int VEC = 16 / sizeof(cache_t);
  constexpr int MM_RANGE_CAP = 32;
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
  constexpr int QK_WARPS = MT * NT;  // warps doing QK
  const bool is_qk_warp = (warp < QK_WARPS);
  const bool is_load_warp = (warp >= QK_WARPS);
  const int load_nthreads = (NUM_WARPS - QK_WARPS) * 32;
  const int load_tid = (warp - QK_WARPS) * 32 + lane;

  constexpr int SPAD = 8;
  constexpr int LDH = HEAD_SIZE + SPAD;
  constexpr int LDN = BLOCK_N + SPAD;

  // Double-buffered KV: sKV[2][BN, LDH]
  extern __shared__ char pf_sm90_smem[];
  cache_t* sQ  = reinterpret_cast<cache_t*>(pf_sm90_smem);
  cache_t* sKV = sQ + BLOCK_M * LDH;               // [2][BN, LDH]
  cache_t* sP  = sKV + 2 * BLOCK_N * LDH;           // [BM, LDN]
  float*   sS  = reinterpret_cast<float*>(sP + BLOCK_M * LDN);
  float*   sM  = sS + BLOCK_M * LDN;
  float*   sL  = sM + BLOCK_M;
  float*   sA  = sL + BLOCK_M;
  int*     sMM = reinterpret_cast<int*>(sA + BLOCK_M);

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> Ofrag[MT][HNT_W];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int j = 0; j < HNT_W; j++) wmma::fill_fragment(Ofrag[m][j], 0.0f);

  for (int r = tid; r < BLOCK_M; r += nthreads) {
    sM[r] = -FLT_MAX; sL[r] = 0.f;
  }

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

  // Helper: load KV tile at kv0 into sKV buffer `buf` using only the load warps.
#define SM90_LOAD_KV(buf, kv0_val, use_all_threads) \
  do { \
    const int _nt = (use_all_threads) ? nthreads : load_nthreads; \
    const int _tid = (use_all_threads) ? tid : load_tid; \
    if ((use_all_threads) || is_load_warp) { \
      const int _n_tok = min(BLOCK_N, seq_len - (kv0_val)); \
      for (int iv = _tid; iv < BLOCK_N * hvps; iv += _nt) { \
        const int n = iv / hvps; \
        const int dv = (iv - n * hvps) * VEC; \
        if (n < _n_tok) { \
          const int tok = (kv0_val) + n; \
          const int64_t phys = block_table[tok / page_size]; \
          const int slot = tok % page_size; \
          const cache_t* gk = k_cache + phys * kv_stride_block \
                                + slot * kv_stride_slot \
                                + kv_head * kv_stride_head + dv; \
          *reinterpret_cast<uint4*>(sKV + (buf) * BLOCK_N * LDH + n * LDH + dv) = \
              *reinterpret_cast<const uint4*>(gk); \
        } else { \
          *reinterpret_cast<uint4*>(sKV + (buf) * BLOCK_N * LDH + n * LDH + dv) = \
              uint4{0, 0, 0, 0}; \
        } \
      } \
    } \
  } while (0)

  // --- Main KV loop with QK-load overlap ---
  int buf = 0;
  int kv0 = kv_begin;

  // First tile: load with ALL warps (no QK to overlap yet).
  if (kv0 < kv_end) {
    SM90_LOAD_KV(0, kv0, true);
  }
  __syncthreads();

  for (; kv0 < kv_end; kv0 += BLOCK_N) {
    const int n_tok = min(BLOCK_N, seq_len - kv0);
    cache_t* cur_kv = sKV + buf * BLOCK_N * LDH;
    const int next_kv0 = kv0 + BLOCK_N;
    const bool has_next = (next_kv0 < kv_end);

    // --- PHASE 1: QK (warps 0..QK_WARPS-1) + next-tile load (remaining warps) ---
    if (is_qk_warp) {
      const int mt = warp / NT;
      const int nt = warp % NT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.0f);
#pragma unroll
      for (int kt = 0; kt < DT; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, cache_t, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, cache_t, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, sQ + mt * 16 * LDH + kt * 16, LDH);
        wmma::load_matrix_sync(fb, cur_kv + nt * 16 * LDH + kt * 16, LDH);
        wmma::mma_sync(s, fa, fb, s);
      }
      wmma::store_matrix_sync(sS + mt * 16 * LDN + nt * 16, s, LDN,
                              wmma::mem_row_major);
    }
    // Load warps: prefetch NEXT tile into the other buffer (concurrent with QK).
    if (has_next) {
      SM90_LOAD_KV(buf ^ 1, next_kv0, false);
    }
    __syncthreads();
    // At this point: QK result in sS, and next tile loaded into sKV[buf^1].

    // --- PHASE 2: softmax (all warps) ---
    static_assert(BLOCK_N % 32 == 0);
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
        bool keep = non_causal ? true : (k_abs <= q_abs);
        if (USE_SLIDING_WINDOW && sliding_window > 0)
          keep = keep && (k_abs > q_abs - sliding_window);
        if constexpr (USE_MM_PREFIX) {
          if (has_mm) {
#pragma unroll 1
            for (int i = 0; i < nr; i++) {
              const int s = sMM[2 * i], e = sMM[2 * i + 1];
              if (s < e && q_abs >= s && q_abs <= e && k_abs >= s && k_abs <= e) {
                keep = true; break;
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

    // --- V load for !k_eq_v ---
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
          *reinterpret_cast<uint4*>(cur_kv + n * LDH + dv) =
              *reinterpret_cast<const uint4*>(gv);
        } else {
          *reinterpret_cast<uint4*>(cur_kv + n * LDH + dv) = uint4{0, 0, 0, 0};
        }
      }
      __syncthreads();
    }

    // --- PHASE 3: PV with alpha-rescale (all warps, head-split) ---
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
          wmma::load_matrix_sync(fv, cur_kv + kt * 16 * LDH + ht * 16, LDH);
          wmma::mma_sync(Ofrag[m][j], fp, fv, Ofrag[m][j]);
        }
      }
    }
    __syncthreads();
    buf ^= 1;  // swap buffers
  }
#undef SM90_LOAD_KV

  // --- Epilogue: identical to v2 ---
  float* sOt = reinterpret_cast<float*>(sQ);
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
  if (lse_out != nullptr) {
    for (int r = warp; r < BLOCK_M; r += NUM_WARPS) {
      const int qr = row0 + r;
      if (qr < q_len && lane == 0) {
        const float l = sL[r];
        lse_out[(int64_t)q_head * num_tokens + (q_start + qr)] =
            sM[r] * scale + logf(l > 0.f ? l : 1e-30f);
      }
    }
  }
}

}  // namespace sm90
}  // namespace gemma_prefill
}  // namespace vllm
