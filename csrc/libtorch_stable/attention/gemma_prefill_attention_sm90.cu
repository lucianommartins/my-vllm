/*
 * SM90 prefill launcher — CUTLASS FMHA integration for Gemma4.
 *
 * Batched launch: one CUTLASS FMHA call per sequence with B=num_q_heads.
 * KV pages are gathered into a GQA-expanded contiguous buffer where each
 * kv-head is replicated gqa_group times, giving each q-head its own
 * contiguous [seq_len, head_size] K/V block.
 *
 * Opt-in via GEMMA_SM90_PREFILL=1.
 */
#include "../torch_utils.h"
#include "gemma_prefill_attention_sm90.cuh"
#include "../../cuda_compat.h"

#include "fmha_sm90/device_universal.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>
#include <algorithm>
#include <vector>

// Gather paged KV into GQA-expanded contiguous buffer.
// Output layout: [num_q_heads, out_seq_stride, head_size] where each q-head's
// K/V is a contiguous block, with kv-heads replicated for GQA.
// out_seq_stride >= seq_len allows padding for TMA alignment.
template <typename CACHE_T>
__global__ void gather_kv_expanded_kernel(
    CACHE_T* __restrict__ kv_out,
    const CACHE_T* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    int seq_len, int out_seq_stride, int num_q_heads, int num_kv_heads,
    int head_size, int gqa_group, int page_size,
    int64_t stride_block, int64_t stride_slot, int64_t stride_head) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_q_heads * seq_len * head_size;
  if (idx >= total) return;

  const int d = idx % head_size;
  const int t = (idx / head_size) % seq_len;
  const int h_q = idx / (head_size * seq_len);
  const int h_kv = h_q / gqa_group;

  const int logical_block = t / page_size;
  const int slot = t % page_size;
  const int physical_block = block_table[logical_block];

  kv_out[h_q * out_seq_stride * head_size + t * head_size + d] =
      kv_cache[physical_block * stride_block + slot * stride_slot +
               h_kv * stride_head + d];
}

template <int HeadDim>
struct FmhaCachedLauncher {
  using FmhaTypes = vllm::gemma_prefill::sm90::GemmaFmhaTypes<HeadDim>;
  using Kernel = typename FmhaTypes::Kernel;
  using FmhaOp = cutlass::device::Universal<Kernel>;

  FmhaOp fmha_op;
  int cached_seq_len = 0;
  int cached_num_q_heads = 0;

  bool launch(
      cutlass::bfloat16_t* q_ptr,
      cutlass::bfloat16_t* k_ptr,
      cutlass::bfloat16_t* v_ptr,
      cutlass::bfloat16_t* o_ptr,
      float* lse_ptr,
      int num_q_heads,
      int seq_q,
      int seq_k,
      int q_stride,
      float scale,
      int sliding_window,
      const int* mm_ranges_ptr,
      int max_mm_ranges,
      int device_id,
      int sm_count,
      cudaStream_t stream) {
    bool shape_changed =
        (seq_q != cached_seq_len || num_q_heads != cached_num_q_heads);

    if (shape_changed) {
      auto stride_qo = cute::make_tuple(
          q_stride, cute::_1{}, cute::make_tuple(HeadDim, HeadDim));
      const int kv_bs = seq_q * HeadDim;
      auto stride_kv = cute::make_tuple(
          HeadDim, cute::_1{}, cute::make_tuple(kv_bs, HeadDim));
      auto stride_lse = cute::make_tuple(
          cute::_1{}, cute::make_tuple(seq_q, seq_q));
      auto problem = cute::make_tuple(
          num_q_heads, 1, seq_q, seq_k, HeadDim, sliding_window, 0, 0);

      cutlass::KernelHardwareInfo hw_info;
      hw_info.device_id = device_id;
      hw_info.sm_count = sm_count;

      typename Kernel::Arguments args{
          problem,
          {q_ptr, stride_qo, k_ptr, stride_kv, v_ptr, stride_kv, scale,
           o_ptr, stride_qo, lse_ptr, mm_ranges_ptr, max_mm_ranges},
          {o_ptr, stride_qo, lse_ptr, stride_lse},
          hw_info};

      if (FmhaOp::can_implement(args) != cutlass::Status::kSuccess)
        return false;

      auto status = fmha_op.initialize(args, nullptr, stream);
      if (status != cutlass::Status::kSuccess) return false;

      cached_seq_len = seq_q;
      cached_num_q_heads = num_q_heads;
    } else {
      // Shape unchanged — update only base pointers in TMA descriptors
      auto& p = const_cast<typename Kernel::Params&>(fmha_op.params());
      cuTensorMapReplaceAddress(
          const_cast<CUtensorMap*>(p.mainloop.tma_load_q.get_tma_descriptor()),
          q_ptr);
      cuTensorMapReplaceAddress(
          const_cast<CUtensorMap*>(p.mainloop.tma_load_k.get_tma_descriptor()),
          k_ptr);
      cuTensorMapReplaceAddress(
          const_cast<CUtensorMap*>(p.mainloop.tma_load_v.get_tma_descriptor()),
          v_ptr);
      // Epilogue O TMA store descriptor
      cuTensorMapReplaceAddress(
          const_cast<CUtensorMap*>(
              p.epilogue.epilogue_TMA.tma_store_d.get_tma_descriptor()),
          o_ptr);
      // LSE pointer (not TMA, just a raw pointer)
      p.epilogue.ptr_LSE = lse_ptr;
      // Mainloop direct-store pointers (used by compute_chunked for hd=512)
      p.mainloop.ptr_O = o_ptr;
      p.mainloop.ptr_LSE = lse_ptr;
    }

    return FmhaOp::run(
               const_cast<typename Kernel::Params&>(fmha_op.params()), stream)
               == cutlass::Status::kSuccess;
  }
};

template <int HeadDim>
static bool launch_fmha_batched(
    cutlass::bfloat16_t* q_ptr,
    cutlass::bfloat16_t* k_ptr,
    cutlass::bfloat16_t* v_ptr,
    cutlass::bfloat16_t* o_ptr,
    float* lse_ptr,
    int num_q_heads,
    int seq_q,
    int seq_k,
    int q_stride,
    float scale,
    int sliding_window,
    const int* mm_ranges_ptr,
    int max_mm_ranges,
    int device_id,
    int sm_count,
    cudaStream_t stream) {
  static FmhaCachedLauncher<HeadDim> launcher;
  return launcher.launch(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, num_q_heads,
                         seq_q, seq_k, q_stride, scale, sliding_window,
                         mm_ranges_ptr, max_mm_ranges,
                         device_id, sm_count, stream);
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

  static const bool forced = []() {
    const char* e = getenv("GEMMA_SM90_PREFILL");
    return e != nullptr && e[0] == '1';
  }();
  if (!forced) return false;

  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int num_seqs = seq_lens.size(0);

  if (head_size != 256 && head_size != 512) return false;
  if (non_causal) return false;
  STD_TORCH_CHECK(lse_out.numel() == 0,
                  "SM90 prefill: cascade LSE output not supported (prefill always passes empty lse_out)");

  constexpr int kAlignment = 16 / sizeof(T);
  if (max_q_len == 0) return false;

  const int num_tokens = static_cast<int>(query.size(0));
  const bool equal_lens = (num_tokens == num_seqs * max_q_len);

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  static int s_device_id = -1;
  static int s_sm_count = 0;
  if (s_device_id < 0) {
    cudaGetDevice(&s_device_id);
    cudaDeviceGetAttribute(&s_sm_count, cudaDevAttrMultiProcessorCount,
                           s_device_id);
  }

  using Element = cutlass::bfloat16_t;
  Element* query_ptr = reinterpret_cast<Element*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  Element* out_ptr = reinterpret_cast<Element*>(out.data_ptr());
  int* block_tables_ptr = block_tables.mutable_data_ptr<int>();
  const int max_num_blocks_per_seq = block_tables.size(1);

  const int64_t kv_stride_block = key_cache.stride(0);
  const int64_t kv_stride_slot = key_cache.stride(1);
  const int64_t kv_stride_head = key_cache.stride(2);
  const int q_stride = query.stride(0);
  const int gqa_group = num_q_heads / num_kv_heads;
  const int seq_len = max_q_len;

  const int max_padded = (seq_len + kAlignment - 1) / kAlignment * kAlignment;
  const size_t kv_expanded_bytes =
      (size_t)max_padded * num_q_heads * head_size * sizeof(CACHE_T);
  const size_t lse_bytes = (size_t)max_padded * num_q_heads * sizeof(float);
  const size_t qo_scratch_bytes =
      (size_t)max_padded * num_q_heads * head_size * sizeof(T);

  // Static buffers — grow-only, never freed (avoids per-call cudaMalloc)
  static CACHE_T* k_expanded = nullptr;
  static CACHE_T* v_expanded = nullptr;
  static float* lse_scratch = nullptr;
  static T* q_scratch = nullptr;
  static T* o_scratch = nullptr;
  static size_t s_kv_cap = 0, s_v_cap = 0, s_lse_cap = 0, s_qo_cap = 0;

  if (kv_expanded_bytes > s_kv_cap) {
    if (k_expanded) cudaFree(k_expanded);
    cudaMalloc(&k_expanded, kv_expanded_bytes);
    s_kv_cap = kv_expanded_bytes;
  }
  if (!k_eq_v && kv_expanded_bytes > s_v_cap) {
    if (v_expanded) cudaFree(v_expanded);
    cudaMalloc(&v_expanded, kv_expanded_bytes);
    s_v_cap = kv_expanded_bytes;
  }
  if (lse_bytes > s_lse_cap) {
    if (lse_scratch) cudaFree(lse_scratch);
    cudaMalloc(&lse_scratch, lse_bytes);
    s_lse_cap = lse_bytes;
  }
  if (qo_scratch_bytes > s_qo_cap) {
    if (q_scratch) cudaFree(q_scratch);
    if (o_scratch) cudaFree(o_scratch);
    cudaMalloc(&q_scratch, qo_scratch_bytes);
    cudaMalloc(&o_scratch, qo_scratch_bytes);
    s_qo_cap = qo_scratch_bytes;
  }

  // For variable-length sequences: sync cu_seqlens_q to host (rare path)
  std::vector<int> h_cu_seqlens_q;
  if (!equal_lens) {
    h_cu_seqlens_q.resize(num_seqs + 1);
    cudaMemcpyAsync(h_cu_seqlens_q.data(),
                    cu_seqlens_q.mutable_data_ptr<int>(),
                    (num_seqs + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }

  bool ok = true;
  const int padded_qo_stride = num_q_heads * head_size;

  for (int s = 0; s < num_seqs && ok; s++) {
    const int token_offset = equal_lens ? s * seq_len
                                        : h_cu_seqlens_q[s];
    const int seq_len_s = equal_lens ? seq_len
                                     : (h_cu_seqlens_q[s + 1] - h_cu_seqlens_q[s]);
    if (seq_len_s == 0) continue;
    const int padded_sl_s = (seq_len_s + kAlignment - 1) / kAlignment * kAlignment;
    const bool needs_pad_s = (padded_sl_s != seq_len_s);

    const int total_elems = num_q_heads * seq_len_s * head_size;
    constexpr int kThreads = 256;
    const int gather_blocks = (total_elems + kThreads - 1) / kThreads;
    const int* seq_block_table =
        block_tables_ptr + s * max_num_blocks_per_seq;

    gather_kv_expanded_kernel<CACHE_T><<<gather_blocks, kThreads, 0, stream>>>(
        k_expanded, key_cache_ptr, seq_block_table, seq_len_s, padded_sl_s,
        num_q_heads, num_kv_heads, head_size, gqa_group, page_size,
        kv_stride_block, kv_stride_slot, kv_stride_head);

    if (!k_eq_v) {
      gather_kv_expanded_kernel<CACHE_T>
          <<<gather_blocks, kThreads, 0, stream>>>(
              v_expanded, value_cache_ptr, seq_block_table, seq_len_s,
              padded_sl_s, num_q_heads, num_kv_heads, head_size, gqa_group,
              page_size, kv_stride_block, kv_stride_slot, kv_stride_head);
    }

    Element* q_src = query_ptr + token_offset * q_stride;
    Element* o_dst = out_ptr + token_offset * q_stride;
    Element* k = reinterpret_cast<Element*>(k_expanded);
    Element* v = k_eq_v ? k : reinterpret_cast<Element*>(v_expanded);

    Element* q_fmha = q_src;
    Element* o_fmha = o_dst;
    int fmha_q_stride = q_stride;

    if (needs_pad_s) {
      const size_t row_bytes = (size_t)num_q_heads * head_size * sizeof(T);
      cudaMemsetAsync(q_scratch, 0,
                      (size_t)padded_sl_s * row_bytes / sizeof(T), stream);
      cudaMemcpyAsync(q_scratch, q_src, (size_t)seq_len_s * row_bytes,
                      cudaMemcpyDeviceToDevice, stream);
      q_fmha = reinterpret_cast<Element*>(q_scratch);
      o_fmha = reinterpret_cast<Element*>(o_scratch);
      fmha_q_stride = padded_qo_stride;
    }

    // MM prefix ranges for this sequence (bidirectional masking for image tokens)
    const int* seq_mm_ranges = nullptr;
    int seq_max_mm = 0;
    if (mm_prefix_ranges.numel() > 0) {
      seq_max_mm = mm_prefix_ranges.size(1);
      seq_mm_ranges = mm_prefix_ranges.mutable_data_ptr<int>() +
                      s * seq_max_mm * 2;
    }

    if (head_size == 256) {
      ok = launch_fmha_batched<256>(q_fmha, k, v, o_fmha, lse_scratch,
                                    num_q_heads, padded_sl_s, seq_len_s,
                                    fmha_q_stride, scale, sliding_window,
                                    seq_mm_ranges, seq_max_mm,
                                    s_device_id, s_sm_count, stream);
    } else {
      ok = launch_fmha_batched<512>(q_fmha, k, v, o_fmha, lse_scratch,
                                    num_q_heads, padded_sl_s, seq_len_s,
                                    fmha_q_stride, scale, sliding_window,
                                    seq_mm_ranges, seq_max_mm,
                                    s_device_id, s_sm_count, stream);
    }

    if (needs_pad_s && ok) {
      const size_t row_bytes = (size_t)num_q_heads * head_size * sizeof(T);
      cudaMemcpyAsync(o_dst, o_fmha, (size_t)seq_len_s * row_bytes,
                      cudaMemcpyDeviceToDevice, stream);
    }
  }

  return ok;
}

template bool gemma_prefill_sm90_launcher<__nv_bfloat16, __nv_bfloat16>(
    torch::stable::Tensor&, torch::stable::Tensor&, torch::stable::Tensor&,
    torch::stable::Tensor&, int, float, torch::stable::Tensor&,
    torch::stable::Tensor&, torch::stable::Tensor&, int, int, bool, int,
    torch::stable::Tensor&, bool, torch::stable::Tensor&);
