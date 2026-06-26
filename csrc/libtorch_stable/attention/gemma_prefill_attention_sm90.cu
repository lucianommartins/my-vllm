/*
 * SM90 prefill launcher for Gemma4-optimized attention.
 *
 * P2: QK-load overlap — overlaps next-tile KV loading with QK compute using
 * double-buffered sKV. Same wmma compute, same register usage as v2. Extra
 * smem for the second KV buffer: +33KB at hd512 (73KB → 106KB, 2 CTAs/SM).
 */
#include "../torch_utils.h"
#include "gemma_prefill_attention_sm90.cuh"
#include "../../cuda_compat.h"

#include <cuda_runtime.h>

static constexpr int SM90_PF_NW_512 = 16;
static constexpr int SM90_PF_NW_256 = 8;
static constexpr int SM90_PF_MINCTA_512 = 2;
static constexpr int SM90_PF_MINCTA_256 = 3;
static constexpr int SM90_PF_BM = 32;
static constexpr int SM90_PF_BN = 32;

#define SM90_PF_CDIV(a, b) (((a) + (b) - 1) / (b))

#define SM90_LAUNCH_PF(HEAD, NW, MINCTA, MMPF, GROUP, KEQV, USW)              \
  do {                                                                         \
    auto kern = vllm::gemma_prefill::sm90::gemma_prefill_kernel_sm90<          \
        T, CACHE_T, HEAD, SM90_PF_BM, SM90_PF_BN, NW, GROUP, KEQV, USW,      \
        MINCTA, MMPF>;                                                         \
    constexpr int SPAD = 8;                                                   \
    constexpr int LDH = HEAD + SPAD;                                          \
    constexpr int LDN = SM90_PF_BN + SPAD;                                    \
    size_t smem =                                                             \
        (size_t)(SM90_PF_BM * LDH                                             \
                 + 2 * SM90_PF_BN * LDH                                       \
                 + SM90_PF_BM * LDN) * sizeof(CACHE_T)                         \
        + (size_t)(SM90_PF_BM * LDN + 3 * SM90_PF_BM) * sizeof(float)         \
        + (size_t)(32 * 2) * sizeof(int);                                     \
    if (smem > 48 * 1024)                                                     \
      cudaFuncSetAttribute(                                                   \
          kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);           \
    cudaFuncSetAttribute(                                                     \
        kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);          \
    dim3 grid(SM90_PF_CDIV(max_q_len, SM90_PF_BM), num_q_heads, num_seqs);   \
    kern<<<grid, (NW) * 32, smem, stream>>>(                                  \
        out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, scale,            \
        block_tables_ptr, seq_lens_ptr, cu_seqlens_q_ptr,                     \
        max_num_blocks_per_seq, page_size, q_stride, kv_stride_block,         \
        kv_stride_slot, kv_stride_head, sliding_window,                       \
        mm_prefix_ranges_ptr, max_mm_ranges,                                  \
        non_causal, lse_out_ptr, num_tokens);                                \
  } while (0)

#define SM90_PF_GROUP(HEAD, NW, MINCTA, MMPF, KEQV, USW)                     \
  switch (gqa_group) {                                                        \
    case 1: SM90_LAUNCH_PF(HEAD, NW, MINCTA, MMPF, 1, KEQV, USW); break;     \
    case 2: SM90_LAUNCH_PF(HEAD, NW, MINCTA, MMPF, 2, KEQV, USW); break;     \
    case 4: SM90_LAUNCH_PF(HEAD, NW, MINCTA, MMPF, 4, KEQV, USW); break;     \
    case 8: SM90_LAUNCH_PF(HEAD, NW, MINCTA, MMPF, 8, KEQV, USW); break;     \
    case 16: SM90_LAUNCH_PF(HEAD, NW, MINCTA, MMPF, 16, KEQV, USW); break;   \
    default:                                                                  \
      STD_TORCH_CHECK(false, "SM90 prefill bad GQA group: ", gqa_group);     \
  }

template <typename T, typename CACHE_T>
bool gemma_prefill_sm90_launcher(
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
  const int* mm_prefix_ranges_ptr = nullptr;
  int max_mm_ranges = 0;
  if (mm_prefix_ranges.numel() > 0) {
    mm_prefix_ranges_ptr = mm_prefix_ranges.mutable_data_ptr<int>();
    max_mm_ranges = mm_prefix_ranges.size(1);
  }
  float* lse_out_ptr = (lse_out.numel() > 0)
                           ? reinterpret_cast<float*>(lse_out.data_ptr())
                           : nullptr;

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  // QK-load overlap: STACK:80 (2× v2's 40) from double-buffer branching.
  // Bottleneck is FMA/softmax (not load latency — KV fits L2), so overlap
  // won't help and extra spill will hurt. Disabled.
  return false;

  if (head_size != 512 && head_size != 256) return false;

#define SM90_PF_DISPATCH(HEAD, NW, MINCTA, MMPF)                             \
  do {                                                                        \
    if (k_eq_v && use_sw) {                                                   \
      SM90_PF_GROUP(HEAD, NW, MINCTA, MMPF, true, true);                      \
    } else if (k_eq_v && !use_sw) {                                           \
      SM90_PF_GROUP(HEAD, NW, MINCTA, MMPF, true, false);                     \
    } else if (!k_eq_v && use_sw) {                                           \
      SM90_PF_GROUP(HEAD, NW, MINCTA, MMPF, false, true);                     \
    } else {                                                                  \
      SM90_PF_GROUP(HEAD, NW, MINCTA, MMPF, false, false);                    \
    }                                                                         \
  } while (0)

  if (head_size == 512) {
    SM90_PF_DISPATCH(512, SM90_PF_NW_512, SM90_PF_MINCTA_512, false);
  } else if (max_mm_ranges > 0) {
    SM90_PF_DISPATCH(256, SM90_PF_NW_256, SM90_PF_MINCTA_256, true);
  } else {
    SM90_PF_DISPATCH(256, SM90_PF_NW_256, SM90_PF_MINCTA_256, false);
  }
#undef SM90_PF_DISPATCH

  return true;
}

#undef SM90_LAUNCH_PF
#undef SM90_PF_GROUP
#undef SM90_PF_CDIV

template bool gemma_prefill_sm90_launcher<__nv_bfloat16, __nv_bfloat16>(
    torch::stable::Tensor&, torch::stable::Tensor&,
    torch::stable::Tensor&, torch::stable::Tensor&,
    int, float, torch::stable::Tensor&,
    torch::stable::Tensor&, torch::stable::Tensor&,
    int, int, bool, int,
    torch::stable::Tensor&, bool,
    torch::stable::Tensor&);
