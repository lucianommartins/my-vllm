/*
 * Launcher for the Gemma4 tensor-core (wmma) prefill kernel. hd=512-first.
 */
#include "../torch_utils.h"
#include "gemma_prefill_attention.cuh"
#include "../../cuda_compat.h"

#include <cuda_runtime.h>
#include <cstdlib>

#define PF_CDIV(a, b) (((a) + (b) - 1) / (b))

static constexpr int PF_BM = 32;
static constexpr int PF_BN = 32;
static constexpr int PF_NW = 4;
static constexpr int PF_NW_V2 = 16;       // warps/CTA for hd=512
static constexpr int PF_NW_V2_256 = 8;    // warps/CTA for hd=256 (sweep knob)
// __launch_bounds__ min-CTA/SM target -> drives the register allocator to the
// occupancy the smem budget allows. Tuned per head size (profiled on A100):
//   hd512: 1->2 CTA/SM (REG 128->64, no spill) = 50% occ, ~1.33x->~1.67x.
//   hd256: 2->3 CTA/SM (REG ~120->79, no spill) = 37.5% occ, 0.9-1.15x->1.17-1.34x.
// Pushing further (hd512=3 / hd256=4) is smem-capped and only cuts ILP -> worse.
static constexpr int PF_MINCTA_512 = 2;
static constexpr int PF_MINCTA_256 = 3;

#define LAUNCH_PREFILL(HEAD, GROUP, KEQV, USW)                                 \
  do {                                                                         \
    auto kern = vllm::gemma_prefill::gemma_prefill_kernel<                     \
        T, CACHE_T, HEAD, PF_BM, PF_BN, PF_NW, GROUP, KEQV, USW>;             \
    size_t smem = (size_t)(PF_BM * HEAD + PF_BN * HEAD + PF_BM * PF_BN)        \
                      * sizeof(CACHE_T)                                        \
                  + (size_t)(PF_BM * PF_BN + PF_BM * HEAD + 3 * PF_BM)         \
                        * sizeof(float);                                       \
    if (smem > 48 * 1024)                                                      \
      cudaFuncSetAttribute(                                                    \
          kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);           \
    dim3 grid(PF_CDIV(max_q_len, PF_BM), num_q_heads, num_seqs);              \
    kern<<<grid, PF_NW * 32, smem, stream>>>(                                  \
        out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, scale,            \
        block_tables_ptr, seq_lens_ptr, cu_seqlens_q_ptr,                      \
        max_num_blocks_per_seq, page_size, q_stride, kv_stride_block,          \
        kv_stride_slot, kv_stride_head, sliding_window);                       \
  } while (0)

#define LAUNCH_PREFILL_GROUP(HEAD, KEQV, USW)                                  \
  switch (gqa_group) {                                                         \
    case 1: LAUNCH_PREFILL(HEAD, 1, KEQV, USW); break;                         \
    case 2: LAUNCH_PREFILL(HEAD, 2, KEQV, USW); break;                         \
    case 4: LAUNCH_PREFILL(HEAD, 4, KEQV, USW); break;                         \
    case 8: LAUNCH_PREFILL(HEAD, 8, KEQV, USW); break;                         \
    case 16: LAUNCH_PREFILL(HEAD, 16, KEQV, USW); break;                       \
    default:                                                                   \
      STD_TORCH_CHECK(false, "Unsupported prefill GQA group: ", gqa_group);    \
  }

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
        non_causal, lse_out_ptr, num_tokens);                                 \
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

template <typename T, typename CACHE_T>
void gemma_prefill_launcher(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int num_kv_heads, float scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, torch::stable::Tensor& cu_seqlens_q,
    int max_q_len, int page_size, bool k_eq_v, int sliding_window,
    torch::stable::Tensor& mm_prefix_ranges, bool non_causal,
    torch::stable::Tensor& lse_out) {
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
    bool non_causal, torch::stable::Tensor& lse_out) {
  STD_TORCH_CHECK(
      query.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Gemma prefill kernel currently supports bfloat16 only");
  gemma_prefill_launcher<__nv_bfloat16, __nv_bfloat16>(
      out, query, key_cache, value_cache, num_kv_heads, (float)scale,
      block_tables, seq_lens, cu_seqlens_q, max_q_len, block_size, k_eq_v,
      sliding_window, mm_prefix_ranges, non_causal, lse_out);
}
