/*
 * Launcher for the Gemma4 tensor-core (wmma) prefill kernel. hd=512-first.
 */
#include "../torch_utils.h"
#include "gemma_prefill_attention.cuh"
#include "gemma_prefill_fused.cuh"
#include "../../cuda_compat.h"

#include <cuda_runtime.h>
#include <cstdlib>

#define PF_CDIV(a, b) (((a) + (b) - 1) / (b))

static constexpr int PF_BM = 32;
static constexpr int PF_BN = 32;
static constexpr int PF_NW_V2 = 16;       // warps/CTA for hd=512
static constexpr int PF_NW_V2_256 = 8;    // warps/CTA for hd=256 (sweep knob)
// __launch_bounds__ min-CTA/SM target -> drives the register allocator to the
// occupancy the smem budget allows. Tuned per head size (profiled on A100):
//   hd512: 1->2 CTA/SM (REG 128->64, no spill) = 50% occ, ~1.33x->~1.67x.
//   hd256: 2->3 CTA/SM (REG ~120->79, no spill) = 37.5% occ, 0.9-1.15x->1.17-1.34x.
// Pushing further (hd512=3 / hd256=4) is smem-capped and only cuts ILP -> worse.
static constexpr int PF_MINCTA_512 = 2;
static constexpr int PF_MINCTA_256 = 3;

// v2: register-resident O, head-split warps. NW = warps/CTA (per head size).
// MINCTA = __launch_bounds__ min CTA/SM target (drives register allocation).
#define LAUNCH_PREFILL_V2(HEAD, NW, MINCTA, MMPF, GROUP, KEQV, USW)            \
  do {                                                                         \
    auto kern = vllm::gemma_prefill::gemma_prefill_kernel_v2<                  \
        T, CACHE_T, HEAD, PF_BM, PF_BN, NW, GROUP, KEQV, USW, MINCTA, MMPF>;  \
    size_t smem =                                                             \
        (size_t)(PF_BM * (HEAD + 8) + PF_BN * (HEAD + 8)                       \
                 + PF_BM * (PF_BN + 8)) * sizeof(CACHE_T)                       \
        + (size_t)(PF_BM * (PF_BN + 8) + 3 * PF_BM) * sizeof(float)            \
        + (size_t)(32 * 2) * sizeof(int);  /* sMM: MM_RANGE_CAP*2 */           \
    if (smem > 48 * 1024)                                                      \
      cudaFuncSetAttribute(                                                    \
          kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);           \
    cudaFuncSetAttribute(                                                      \
        kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);           \
    dim3 grid(PF_CDIV(max_q_len, PF_BM), num_q_heads, num_seqs);             \
    kern<<<grid, (NW) * 32, smem, stream>>>(                                  \
        out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, scale,            \
        block_tables_ptr, seq_lens_ptr, cu_seqlens_q_ptr,                      \
        max_num_blocks_per_seq, page_size, q_stride, kv_stride_block,          \
        kv_stride_slot, kv_stride_head, sliding_window,                        \
        mm_prefix_ranges_ptr, max_mm_ranges,                                   \
        non_causal, lse_out_ptr, num_tokens, record640, ldgsts_overlap,        \
        rescale_skip);                                                         \
  } while (0)

#define LAUNCH_PREFILL_V2_GROUP(HEAD, NW, MINCTA, MMPF, KEQV, USW)             \
  switch (gqa_group) {                                                         \
    case 1: LAUNCH_PREFILL_V2(HEAD, NW, MINCTA, MMPF, 1, KEQV, USW); break;    \
    case 2: LAUNCH_PREFILL_V2(HEAD, NW, MINCTA, MMPF, 2, KEQV, USW); break;    \
    case 4: LAUNCH_PREFILL_V2(HEAD, NW, MINCTA, MMPF, 4, KEQV, USW); break;    \
    case 8: LAUNCH_PREFILL_V2(HEAD, NW, MINCTA, MMPF, 8, KEQV, USW); break;    \
    case 16: LAUNCH_PREFILL_V2(HEAD, NW, MINCTA, MMPF, 16, KEQV, USW); break;  \
    default:                                                                   \
      STD_TORCH_CHECK(false, "Unsupported prefill GQA group: ", gqa_group);    \
  }

// Fused prefill (Gate-D dataflow port): mma.sync register-softmax +
// cp.async ring. Opt-in via GEMMA_PREFILL_FUSED=1 for A/B vs the wmma v2;
// flips to default after validation. NW: hd512 -> 8 (BN64, 1 CTA/SM);
// hd256 -> GEMMA_PREFILL_FUSED_NW (default 4 = BN32, 2 CTA/SM).
#define LAUNCH_PREFILL_FUSED(HEAD, NW, MINCTA, GROUP, KEQV, USW, MMPF)         \
  do {                                                                         \
    auto kern = vllm::gemma_prefill::gemma_prefill_fused_kernel<               \
        T, CACHE_T, HEAD, NW, GROUP, KEQV, USW, MMPF, MINCTA>;                 \
    constexpr int FLDH = (HEAD) + 8;                                           \
    constexpr int FBN = 8 * (NW);                                              \
    constexpr int FLDN = FBN + 8;                                              \
    constexpr int FSTG = FBN * FLDH * ((KEQV) ? 1 : 2);                        \
    size_t smem = (size_t)(16 * FLDH + 2 * FSTG + 16 * FLDN)                   \
                      * sizeof(CACHE_T)                                        \
                  + (size_t)(2 * (NW) * 16) * sizeof(float)                    \
                  + (size_t)(32 * 2) * sizeof(int);                            \
    {                                                                          \
      static bool _a = false;                                                  \
      if (!_a) {                                                               \
        if (smem > 48 * 1024)                                                  \
          cudaFuncSetAttribute(                                                \
              kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);        \
        cudaFuncSetAttribute(                                                  \
            kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);        \
        _a = true;                                                             \
      }                                                                        \
    }                                                                          \
    dim3 grid(PF_CDIV(max_q_len, 16), num_q_heads, num_seqs);                  \
    kern<<<grid, (NW) * 32, smem, stream>>>(                                   \
        out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, scale,             \
        block_tables_ptr, seq_lens_ptr, cu_seqlens_q_ptr,                      \
        max_num_blocks_per_seq, page_size, q_stride, kv_stride_block,          \
        kv_stride_slot, kv_stride_head, sliding_window,                        \
        mm_prefix_ranges_ptr, max_mm_ranges,                                   \
        non_causal, lse_out_ptr, num_tokens, record640);                       \
  } while (0)

#define LAUNCH_PREFILL_FUSED_GROUP(HEAD, NW, MINCTA, KEQV, USW, MMPF)          \
  switch (gqa_group) {                                                         \
    case 1: LAUNCH_PREFILL_FUSED(HEAD, NW, MINCTA, 1, KEQV, USW, MMPF); break; \
    case 2: LAUNCH_PREFILL_FUSED(HEAD, NW, MINCTA, 2, KEQV, USW, MMPF); break; \
    case 4: LAUNCH_PREFILL_FUSED(HEAD, NW, MINCTA, 4, KEQV, USW, MMPF); break; \
    case 8: LAUNCH_PREFILL_FUSED(HEAD, NW, MINCTA, 8, KEQV, USW, MMPF); break; \
    case 16:                                                                   \
      LAUNCH_PREFILL_FUSED(HEAD, NW, MINCTA, 16, KEQV, USW, MMPF);             \
      break;                                                                   \
    default:                                                                   \
      STD_TORCH_CHECK(false, "Unsupported fused prefill GQA group: ",          \
                      gqa_group);                                              \
  }

// fused2: BM=32 two-M-tile register-softmax prefill (hd512 k_eq_v only).
// GEMMA_PREFILL_FUSED=2. smem ~106KB -> 1 CTA/SM.
#define LAUNCH_PREFILL_FUSED2(GROUP)                                           \
  do {                                                                         \
    auto kern = vllm::gemma_prefill::gemma_prefill_fused2_kernel<              \
        T, CACHE_T, 512, 8, GROUP, 1>;                                         \
    constexpr int F2LDH = 512 + 8;                                             \
    constexpr int F2BN = 64;                                                   \
    constexpr int F2LDN = F2BN + 8;                                            \
    size_t smem = (size_t)(32 * F2LDH + F2BN * F2LDH + 2 * 16 * F2LDN)         \
                      * sizeof(CACHE_T)                                        \
                  + (size_t)(2 * 2 * 8 * 16) * sizeof(float);                  \
    {                                                                          \
      static bool _a = false;                                                  \
      if (!_a) {                                                               \
        if (smem > 48 * 1024)                                                  \
          cudaFuncSetAttribute(                                                \
              kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);        \
        cudaFuncSetAttribute(                                                  \
            kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);        \
        _a = true;                                                             \
      }                                                                        \
    }                                                                          \
    dim3 grid(PF_CDIV(max_q_len, 32), num_q_heads, num_seqs);                  \
    kern<<<grid, 8 * 32, smem, stream>>>(                                      \
        out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, scale,             \
        block_tables_ptr, seq_lens_ptr, cu_seqlens_q_ptr,                      \
        max_num_blocks_per_seq, page_size, q_stride, kv_stride_block,          \
        kv_stride_slot, kv_stride_head, sliding_window,                        \
        mm_prefix_ranges_ptr, max_mm_ranges,                                   \
        non_causal, lse_out_ptr, num_tokens, record640);                       \
  } while (0)

#define LAUNCH_PREFILL_FUSED2_GROUP()                                          \
  switch (gqa_group) {                                                         \
    case 1: LAUNCH_PREFILL_FUSED2(1); break;                                   \
    case 2: LAUNCH_PREFILL_FUSED2(2); break;                                   \
    case 4: LAUNCH_PREFILL_FUSED2(4); break;                                   \
    case 8: LAUNCH_PREFILL_FUSED2(8); break;                                   \
    case 16: LAUNCH_PREFILL_FUSED2(16); break;                                 \
    default:                                                                   \
      STD_TORCH_CHECK(false, "Unsupported fused2 GQA group: ", gqa_group);     \
  }

// SM90 prefill launcher — defined in gemma_prefill_attention_sm90.cu.
// Returns true if handled, false to fall through to SM80 v2 kernel.
#if defined(ENABLE_GEMMA_ATTN_SM90) && ENABLE_GEMMA_ATTN_SM90
template <typename T, typename CACHE_T>
bool gemma_prefill_sm90_launcher(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int num_kv_heads, float scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, torch::stable::Tensor& cu_seqlens_q,
    int max_q_len, int page_size, bool k_eq_v, int sliding_window,
    torch::stable::Tensor& mm_prefix_ranges, bool non_causal,
    torch::stable::Tensor& lse_out, torch::stable::Tensor& seq_lens_cpu,
    torch::stable::Tensor& cu_seqlens_q_cpu,
    torch::stable::Tensor& recon_invfreq, double recon_inv_w);
#endif

template <typename T, typename CACHE_T>
void gemma_prefill_launcher(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int num_kv_heads, float scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, torch::stable::Tensor& cu_seqlens_q,
    int max_q_len, int page_size, bool k_eq_v, int sliding_window,
    torch::stable::Tensor& mm_prefix_ranges, bool non_causal,
    torch::stable::Tensor& lse_out, torch::stable::Tensor& seq_lens_cpu,
    torch::stable::Tensor& cu_seqlens_q_cpu,
    torch::stable::Tensor& recon_invfreq, double recon_inv_w) {
  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int num_seqs = seq_lens.size(0);
  const int num_tokens = static_cast<int>(out.size(0));
  const int max_num_blocks_per_seq = block_tables.size(1);
  const int q_stride = query.stride(0);
  const int64_t kv_stride_block = key_cache.stride(0);
  const int64_t kv_stride_slot = key_cache.stride(1);
  const int64_t kv_stride_head = key_cache.stride(2);
  const int gqa_group = num_q_heads / num_kv_heads;
  const bool use_sw = (sliding_window > 0);
  // GEMMA_CACHE_V3 640-channel record pool (single plane, K never stored):
  // inferred from the channel dim, same as the decode launcher. The v2
  // kernel materializes K-hat via the record address permutation and
  // rebuilds V's 128 rotated channels before PV — SM80-native record read.
  const bool record640 = key_cache.dim() == 4 && key_cache.size(3) == 640;
  // V staging order: post-softmax (default) — at 2-3 CTA/SM the V load is
  // already hidden cross-CTA, and pre-softmax issue measured -17% TTFT at
  // mixed c32 (DRAM pressure against coexisting decode kernels) with no
  // pure-prefill gain (A/B 2026-07-17). GEMMA_PREFILL_LDGSTS_OVERLAP=1
  // re-enables the early-issue variant for experiments.
  static const bool ldgsts_overlap = []() {
    const char* e = getenv("GEMMA_PREFILL_LDGSTS_OVERLAP");
    return e != nullptr && e[0] == '1';
  }();
  // 3b identity elision (spec: docs/gemma_prefill_3b_spec.md). Default OFF
  // until the §4 protocol gates pass (bitwise parity -> skip-rate -> A/B).
  static const bool rescale_skip = []() {
    const char* e = getenv("GEMMA_PREFILL_RESCALE_SKIP");
    return e != nullptr && e[0] == '1';
  }();

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.mutable_data_ptr<int>();
  int* seq_lens_ptr = seq_lens.mutable_data_ptr<int>();
  int* cu_seqlens_q_ptr = cu_seqlens_q.mutable_data_ptr<int>();
  // mm-prefix image spans: int32 [num_seqs, max_mm_ranges, 2], empty -> none.
  const int* mm_prefix_ranges_ptr = nullptr;
  int max_mm_ranges = 0;
  if (mm_prefix_ranges.numel() > 0) {
    mm_prefix_ranges_ptr = mm_prefix_ranges.mutable_data_ptr<int>();
    max_mm_ranges = mm_prefix_ranges.size(1);
  }
  // Optional natural-log LSE output for cascade attention (empty == skip).
  float* lse_out_ptr = (lse_out.numel() > 0)
                           ? reinterpret_cast<float*>(lse_out.data_ptr())
                           : nullptr;

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  STD_TORCH_CHECK(head_size == 512 || head_size == 256,
                  "Gemma prefill kernel supports head_size 256 or 512, got ",
                  head_size);

  // --- SM90 (Hopper) dispatch ---
#if defined(ENABLE_GEMMA_ATTN_SM90) && ENABLE_GEMMA_ATTN_SM90
  {
    static const bool is_sm90 = []() {
      int dev = 0;
      cudaGetDevice(&dev);
      int major = 0;
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
      return major >= 9;
    }();
    if (is_sm90) {
      bool handled = gemma_prefill_sm90_launcher<T, CACHE_T>(
          out, query, key_cache, value_cache, num_kv_heads, scale,
          block_tables, seq_lens, cu_seqlens_q, max_q_len, page_size,
          k_eq_v, sliding_window, mm_prefix_ranges, non_causal, lse_out,
          seq_lens_cpu, cu_seqlens_q_cpu, recon_invfreq, recon_inv_w);
      if (handled) return;
      // Record pools no longer require the SM90 path: the v2 kernel below
      // reads records natively (K-hat permutation + rotor-V overwrite).
    }
  }
#endif  // ENABLE_GEMMA_ATTN_SM90

  // Correct-mode hole fix: the wmma fallback (cascade prefix passes with
  // LSE, tiny-q extends, GEMMA_SM90_PREFILL=0) has no V-reconstruction.
  // When correct mode is active, read the TRUE V plane here instead of
  // aliasing V:=K — correct function at 2x bytes on these infrequent
  // shapes. (Long-MTP acceptance collapsed to the aliased 0.803 because
  // the shared-prefix cascade pass computed the aliased function over
  // the whole context.)
  static const bool correct_mode = []() {
    const char* r = getenv("GEMMA_V_RECON");
    const char* f = getenv("GEMMA_V_FILL");
    return (r != nullptr && r[0] == '1') ||
           (f != nullptr && f[0] != '0' && f[0] != '\0');
  }();
  // Records are already the correct function (true V rebuilt in-kernel);
  // correct_mode's k_eq_v=false toggle only applies to two-plane caches.
  if (correct_mode && !record640) k_eq_v = false;

  if (record640) {
    STD_TORCH_CHECK(
        head_size == 512 && k_eq_v && !use_sw && max_mm_ranges == 0,
        "record640 prefill supports only full-attention k_eq_v hd512 layers "
        "(got head_size=", head_size, " k_eq_v=", k_eq_v,
        " sliding_window=", sliding_window,
        " mm_ranges=", max_mm_ranges, ")");
  }

  // ---- fused prefill (opt-in A/B): 1 = BM16 (measured slower, kept for
  // reference), 2 = BM32 two-M-tile fused2 (hd512 k_eq_v; rest stays v2) ----
  static const int prefill_fused_mode = []() {
    const char* e = getenv("GEMMA_PREFILL_FUSED");
    if (e == nullptr) return 0;
    return e[0] == '2' ? 2 : (e[0] == '1' ? 1 : 0);
  }();
  const bool prefill_fused = prefill_fused_mode == 1;
  if (prefill_fused_mode == 2 && head_size == 512 && k_eq_v && !use_sw &&
      max_mm_ranges == 0) {
    LAUNCH_PREFILL_FUSED2_GROUP();
    return;
  }
  static const int fused_nw256 = []() {
    const char* e = getenv("GEMMA_PREFILL_FUSED_NW");
    if (e == nullptr) return 4;
    return e[0] == '8' ? 8 : 4;
  }();
  if (prefill_fused) {
    if (head_size == 512) {
      // k_eq_v: K-only ring -> BN64/NW8 fits (153KB). !k_eq_v stages V too:
      // BN64 would need 266KB -> BN32/NW4 (152KB, 1 CTA/SM).
      if (k_eq_v) {
        LAUNCH_PREFILL_FUSED_GROUP(512, 8, 1, true, false, false);
      } else {
        LAUNCH_PREFILL_FUSED_GROUP(512, 4, 1, false, false, false);
      }
    } else if (max_mm_ranges > 0) {
      if (fused_nw256 == 8) {
        LAUNCH_PREFILL_FUSED_GROUP(256, 8, 1, false, true, true);
      } else {
        LAUNCH_PREFILL_FUSED_GROUP(256, 4, 2, false, true, true);
      }
    } else {
      if (fused_nw256 == 8) {
        LAUNCH_PREFILL_FUSED_GROUP(256, 8, 1, false, true, false);
      } else {
        LAUNCH_PREFILL_FUSED_GROUP(256, 4, 2, false, true, false);
      }
    }
    return;
  }

#define PF_DISPATCH(HEAD, NW, MINCTA, MMPF)                          \
  do {                                                               \
    if (k_eq_v && use_sw) {                                          \
      LAUNCH_PREFILL_V2_GROUP(HEAD, NW, MINCTA, MMPF, true, true);   \
    } else if (k_eq_v && !use_sw) {                                  \
      LAUNCH_PREFILL_V2_GROUP(HEAD, NW, MINCTA, MMPF, true, false);  \
    } else if (!k_eq_v && use_sw) {                                  \
      LAUNCH_PREFILL_V2_GROUP(HEAD, NW, MINCTA, MMPF, false, true);  \
    } else {                                                         \
      LAUNCH_PREFILL_V2_GROUP(HEAD, NW, MINCTA, MMPF, false, false); \
    }                                                                \
  } while (0)

  // hd512 = full-attn layers (always causal, mm cleared by the model) -> mm
  // compiled out for max perf. hd256 = sliding layers: instantiate the
  // bidirectional path only when image spans are actually present this step, so
  // text-only requests run the identical zero-overhead kernel.
  if (head_size == 512) {
    PF_DISPATCH(512, PF_NW_V2, PF_MINCTA_512, false);
  } else if (max_mm_ranges > 0) {
    PF_DISPATCH(256, PF_NW_V2_256, PF_MINCTA_256, true);
  } else {
    PF_DISPATCH(256, PF_NW_V2_256, PF_MINCTA_256, false);
  }
#undef PF_DISPATCH
}

void gemma_prefill_attention(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int64_t num_kv_heads, double scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, torch::stable::Tensor& cu_seqlens_q,
    int64_t max_q_len, int64_t block_size, bool k_eq_v,
    int64_t sliding_window, torch::stable::Tensor& mm_prefix_ranges,
    bool non_causal, torch::stable::Tensor& lse_out,
    torch::stable::Tensor& seq_lens_cpu,
    torch::stable::Tensor& cu_seqlens_q_cpu,
    torch::stable::Tensor& recon_invfreq, double recon_inv_w) {
  STD_TORCH_CHECK(
      query.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Gemma prefill kernel currently supports bfloat16 only");
  gemma_prefill_launcher<__nv_bfloat16, __nv_bfloat16>(
      out, query, key_cache, value_cache, num_kv_heads, (float)scale,
      block_tables, seq_lens, cu_seqlens_q, max_q_len, block_size, k_eq_v,
      sliding_window, mm_prefix_ranges, non_causal, lse_out, seq_lens_cpu,
      cu_seqlens_q_cpu, recon_invfreq, recon_inv_w);
}
