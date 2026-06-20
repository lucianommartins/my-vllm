/*
 * Gemma4-optimized paged attention — multi-warp intra-CTA split-KV.
 * No cross-CTA partitioning — no reduce kernel needed.
 */
#include "../torch_utils.h"
#include "gemma_paged_attention.cuh"
#include "../../cuda_compat.h"

#include <cstdlib>
#include <cuda_runtime.h>

// NUM_WARPS per CTA — controls parallelism within each head.
// 4 warps = 128 threads is a good balance for A100 occupancy.
static constexpr int NUM_WARPS = 4;

#define LAUNCH_GEMMA(HEAD_SIZE, ACTUAL_HEAD_SIZE, K_EQ_V, USE_SW)              \
  vllm::gemma::gemma_flash_decode_kernel<                                      \
      T, CACHE_T, HEAD_SIZE, ACTUAL_HEAD_SIZE, BLOCK_SIZE, NUM_WARPS,          \
      KV_DTYPE, K_EQ_V, USE_SW>                                               \
      <<<grid, NUM_WARPS * WARP_SIZE, smem_size, stream>>>(                    \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,                  \
          num_kv_heads, scale, block_tables_ptr, seq_lens_ptr,                 \
          max_num_blocks_per_seq, q_stride,                                    \
          kv_stride_block, kv_stride_slot, kv_stride_head,                     \
          k_scale_ptr, v_scale_ptr, sliding_window);

// GQA-group-reuse prototype launch (one CTA per (kv_head, seq), GROUP warps).
#define LAUNCH_GEMMA_GQA(HEAD_SIZE, ACTUAL_HEAD_SIZE, GROUP, K_EQ_V, USE_SW)   \
  do {                                                                         \
    const dim3 gqa_grid(num_kv_heads, num_seqs);                              \
    size_t gqa_smem = static_cast<size_t>(BLOCK_SIZE) * (ACTUAL_HEAD_SIZE)     \
                      * sizeof(CACHE_T) * ((K_EQ_V) ? 1 : 2) * 2;             \
    auto gqa_kernel = vllm::gemma::gemma_gqa_decode_kernel<                    \
        T, CACHE_T, HEAD_SIZE, ACTUAL_HEAD_SIZE, BLOCK_SIZE, GROUP,            \
        KV_DTYPE, K_EQ_V, USE_SW>;                                            \
    if (gqa_smem > 48 * 1024) {                                                \
      cudaFuncSetAttribute(gqa_kernel,                                         \
          cudaFuncAttributeMaxDynamicSharedMemorySize, gqa_smem);             \
    }                                                                          \
    gqa_kernel<<<gqa_grid, (GROUP) * WARP_SIZE, gqa_smem, stream>>>(           \
        out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,                   \
        num_kv_heads, scale, block_tables_ptr, seq_lens_ptr,                   \
        max_num_blocks_per_seq, q_stride,                                      \
        kv_stride_block, kv_stride_slot, kv_stride_head,                       \
        k_scale_ptr, v_scale_ptr, sliding_window);                            \
  } while (0)

#define LAUNCH_GEMMA_GQA_GROUP(HEAD_SIZE, ACTUAL_HEAD_SIZE, K_EQ_V, USE_SW)    \
  switch (gqa_group) {                                                         \
    case 2:  LAUNCH_GEMMA_GQA(HEAD_SIZE, ACTUAL_HEAD_SIZE, 2, K_EQ_V, USE_SW); \
      break;                                                                   \
    case 4:  LAUNCH_GEMMA_GQA(HEAD_SIZE, ACTUAL_HEAD_SIZE, 4, K_EQ_V, USE_SW); \
      break;                                                                   \
    case 8:  LAUNCH_GEMMA_GQA(HEAD_SIZE, ACTUAL_HEAD_SIZE, 8, K_EQ_V, USE_SW); \
      break;                                                                   \
    case 16: LAUNCH_GEMMA_GQA(HEAD_SIZE, ACTUAL_HEAD_SIZE, 16, K_EQ_V,         \
                              USE_SW);                                         \
      break;                                                                   \
    default: LAUNCH_GEMMA(HEAD_SIZE, ACTUAL_HEAD_SIZE, K_EQ_V, USE_SW);        \
      break;                                                                   \
  }

// GQA-reuse + cross-CTA split-KV: phase-1 split kernel + phase-2 combine.
#define LAUNCH_GEMMA_GQA_SPLIT(HEAD_SIZE, ACTUAL_HEAD_SIZE, GROUP, K_EQ_V,     \
                               USE_SW)                                         \
  do {                                                                         \
    const dim3 split_grid(num_kv_heads, num_seqs, num_splits);                 \
    size_t split_smem = static_cast<size_t>(BLOCK_SIZE) * (ACTUAL_HEAD_SIZE)   \
                        * sizeof(CACHE_T) * ((K_EQ_V) ? 1 : 2) * 2;           \
    auto split_kernel = vllm::gemma::gemma_gqa_split_decode_kernel<            \
        T, CACHE_T, HEAD_SIZE, ACTUAL_HEAD_SIZE, BLOCK_SIZE, GROUP,            \
        KV_DTYPE, K_EQ_V, USE_SW>;                                            \
    if (split_smem > 48 * 1024) {                                              \
      cudaFuncSetAttribute(split_kernel,                                       \
          cudaFuncAttributeMaxDynamicSharedMemorySize, split_smem);           \
    }                                                                          \
    split_kernel<<<split_grid, (GROUP) * WARP_SIZE, split_smem, stream>>>(     \
        tmp_out_ptr, exp_sums_ptr, max_logits_ptr, query_ptr,                 \
        key_cache_ptr, value_cache_ptr, num_kv_heads, scale,                  \
        block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq, q_stride,     \
        kv_stride_block, kv_stride_slot, kv_stride_head,                      \
        k_scale_ptr, v_scale_ptr, sliding_window, num_splits, max_parts);     \
    const dim3 combine_grid(num_kv_heads * (GROUP), num_seqs);                 \
    vllm::gemma::gemma_split_reduce_kernel<T, HEAD_SIZE>                       \
        <<<combine_grid, WARP_SIZE, 0, stream>>>(                             \
            out_ptr, tmp_out_ptr, exp_sums_ptr, max_logits_ptr,              \
            num_splits, max_parts);                                          \
  } while (0)

#define LAUNCH_GEMMA_GQA_SPLIT_GROUP(HEAD_SIZE, ACTUAL_HEAD_SIZE, K_EQ_V,      \
                                     USE_SW)                                   \
  switch (gqa_group) {                                                         \
    case 2:  LAUNCH_GEMMA_GQA_SPLIT(HEAD_SIZE, ACTUAL_HEAD_SIZE, 2, K_EQ_V,    \
                                    USE_SW); break;                            \
    case 4:  LAUNCH_GEMMA_GQA_SPLIT(HEAD_SIZE, ACTUAL_HEAD_SIZE, 4, K_EQ_V,    \
                                    USE_SW); break;                            \
    case 8:  LAUNCH_GEMMA_GQA_SPLIT(HEAD_SIZE, ACTUAL_HEAD_SIZE, 8, K_EQ_V,    \
                                    USE_SW); break;                            \
    case 16: LAUNCH_GEMMA_GQA_SPLIT(HEAD_SIZE, ACTUAL_HEAD_SIZE, 16, K_EQ_V,   \
                                    USE_SW); break;                            \
    default: LAUNCH_GEMMA(HEAD_SIZE, ACTUAL_HEAD_SIZE, K_EQ_V, USE_SW);        \
      break;                                                                   \
  }

template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE>
void gemma_paged_attention_launcher(
    torch::stable::Tensor& out,
    torch::stable::Tensor& exp_sums,    // unused — kept for API compat
    torch::stable::Tensor& max_logits,   // unused
    torch::stable::Tensor& tmp_out,      // unused
    torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache,
    torch::stable::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens,
    int max_seq_len,
    torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale,
    int actual_head_size,
    bool k_eq_v,
    int sliding_window) {

  int num_seqs = query.size(0);
  int num_q_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);

  int64_t kv_stride_block = key_cache.stride(0);
  int64_t kv_stride_slot = key_cache.stride(1);
  int64_t kv_stride_head = key_cache.stride(2);

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.mutable_data_ptr<int>();
  int* seq_lens_ptr = seq_lens.mutable_data_ptr<int>();
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());

  // Partial buffers for split-KV (partition dim == split).
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  const int max_parts = static_cast<int>(exp_sums.size(2));

  // Grid: one CTA per (q_head, seq). All KV work done intra-CTA.
  dim3 grid(num_q_heads, num_seqs);
  // Shared memory for warp reduce: M, L scalars + acc array per warp.
  int smem_size = NUM_WARPS * (2 * sizeof(float) + actual_head_size * sizeof(float));

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  bool use_sw = (sliding_window > 0);

  const int gqa_group =
      (num_kv_heads > 0) ? (num_q_heads / num_kv_heads) : 0;

  // num_splits heuristic: split only when the batch underfills the SMs.
  // Target ~2 CTAs/SM after accounting for the (num_seqs * num_kv_heads) CTAs
  // the GQA grid already produces. Clamped to KV blocks and the partition cap.
  static int num_sms = []() {
    int dev = 0;
    cudaGetDevice(&dev);
    int n = 1;
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    return n;
  }();
  int num_splits = 1;
  {
    const int total_ctas = num_seqs * (num_kv_heads > 0 ? num_kv_heads : 1);
    // Sliding-window layers only attend to the last `sliding_window` tokens, so
    // bound the effective KV blocks by the window (avoids many empty splits).
    const int eff_seq = (sliding_window > 0 && sliding_window < max_seq_len)
                            ? sliding_window : max_seq_len;
    const int max_seq_blocks = (eff_seq + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int target = (total_ctas > 0)
                     ? (2 * num_sms + total_ctas - 1) / total_ctas
                     : 1;
    num_splits = target < 1 ? 1 : target;
    if (num_splits > max_seq_blocks) num_splits = max_seq_blocks;
    if (num_splits > max_parts) num_splits = max_parts;
    if (max_seq_blocks <= 4) num_splits = 1;
    if (num_splits < 1) num_splits = 1;
  }

  // GQA-reuse dispatch (always on): A2 split-KV when the batch underfills the
  // SMs, else the A1 single-CTA GQA kernel. For cases where actual_head_size ==
  // head_size (the combine reads HEAD_SIZE dims).
#define GEMMA_DISPATCH(HS, AHS, KEQV, USW)                  \
  do {                                                      \
    if (num_splits > 1) {                                   \
      LAUNCH_GEMMA_GQA_SPLIT_GROUP(HS, AHS, KEQV, USW);     \
    } else {                                                \
      LAUNCH_GEMMA_GQA_GROUP(HS, AHS, KEQV, USW);           \
    }                                                       \
  } while (0)

  if (head_size == 512 && actual_head_size == 256 && !k_eq_v && use_sw) {
    LAUNCH_GEMMA(512, 256, false, true);
  } else if (head_size == 512 && actual_head_size == 512 && k_eq_v && !use_sw) {
    GEMMA_DISPATCH(512, 512, true, false);
  } else if (head_size == 512 && actual_head_size == 512 && !k_eq_v && !use_sw) {
    GEMMA_DISPATCH(512, 512, false, false);
  } else if (head_size == 512 && actual_head_size == 256 && !k_eq_v && !use_sw) {
    LAUNCH_GEMMA(512, 256, false, false);
  } else if (head_size == 256 && actual_head_size == 256 && !k_eq_v && use_sw) {
    GEMMA_DISPATCH(256, 256, false, true);
  } else if (head_size == 256 && actual_head_size == 256 && !k_eq_v && !use_sw) {
    GEMMA_DISPATCH(256, 256, false, false);
  } else {
    STD_TORCH_CHECK(false,
        "Unsupported Gemma attention config: head_size=", head_size,
        " actual_head_size=", actual_head_size,
        " k_eq_v=", k_eq_v,
        " sliding_window=", sliding_window);
  }
#undef GEMMA_DISPATCH
}

#define CALL_GEMMA_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE)                 \
  gemma_paged_attention_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE>(            \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,       \
      num_kv_heads, scale, block_tables, seq_lens, max_seq_len,                \
      k_scale, v_scale, actual_head_size, k_eq_v, sliding_window);

#define CALL_GEMMA_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)             \
  switch (block_size) {                                                  \
    case 16:                                                             \
      CALL_GEMMA_LAUNCHER(T, CACHE_T, 16, KV_DTYPE);                     \
      break;                                                             \
    case 32:                                                             \
      CALL_GEMMA_LAUNCHER(T, CACHE_T, 32, KV_DTYPE);                     \
      break;                                                             \
    case 64:                                                             \
      CALL_GEMMA_LAUNCHER(T, CACHE_T, 64, KV_DTYPE);                     \
      break;                                                             \
    default:                                                             \
      STD_TORCH_CHECK(false, "Unsupported block size: ", block_size);    \
      break;                                                             \
  }

void gemma_paged_attention(
    torch::stable::Tensor& out,
    torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits,
    torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache,
    torch::stable::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens,
    int64_t block_size,
    int64_t max_seq_len,
    const std::string& kv_cache_dtype,
    torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale,
    int64_t actual_head_size,
    bool k_eq_v,
    int64_t sliding_window) {
  DISPATCH_BY_KV_CACHE_DTYPE(query.scalar_type(), kv_cache_dtype,
                             CALL_GEMMA_LAUNCHER_BLOCK_SIZE)
}
