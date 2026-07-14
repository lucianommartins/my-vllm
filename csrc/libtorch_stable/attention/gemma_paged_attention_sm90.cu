/*
 * SM90 decode launcher — two paths:
 *   1. SIMT+paged decode (BLOCK_N=64): register-resident Q/O with paged
 *      cp.async KV loads. Graph-safe. Supports k_eq_v and non-k_eq_v.
 *   2. CUTLASS FMHA fallback: for non-64 block sizes via gather.
 *
 * Opt-in via GEMMA_SM90_DECODE=1.
 */
#include "../torch_utils.h"
#include "gemma_paged_attention_sm90.cuh"
#include "gemma_prefill_attention_sm90.cuh"
#include "gemma_decode_tma_kernel.cuh"
#include "../../attention/attention_dtypes.h"
#include "../../cuda_compat.h"

#include "fmha_sm90/device_universal.hpp"

#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <type_traits>

#include "cute/tensor.hpp"

// ---- Gather paged KV → contiguous GQA-expanded buffer ----
template <typename CACHE_T>
__global__ void decode_gather_kv_kernel(
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

// ---- FMHA cached launcher (reuses prefill CUTLASS kernel) ----
template <int HeadDim>
struct DecodeFmhaLauncher {
  using FmhaTypes = vllm::gemma_prefill::sm90::GemmaFmhaTypes<HeadDim>;
  using Kernel = typename FmhaTypes::Kernel;
  using FmhaOp = cutlass::device::Universal<Kernel>;

  FmhaOp fmha_op;
  int cached_seq_k = 0;
  int cached_num_q_heads = 0;

  // Initialize TMA descriptors with max_kv_len (fixed size, graph-safe).
  // Called once during warmup. Subsequent calls only update pointers + q_offset.
  bool init_descriptors(
      cutlass::bfloat16_t* q_ptr, cutlass::bfloat16_t* k_ptr,
      cutlass::bfloat16_t* v_ptr, cutlass::bfloat16_t* o_ptr, float* lse_ptr,
      int num_q_heads, int seq_q, int max_kv_len, int q_stride,
      float scale, int sliding_window,
      int device_id, int sm_count, cudaStream_t stream) {

    auto stride_qo = cute::make_tuple(
        q_stride, cute::_1{}, cute::make_tuple(HeadDim, HeadDim));
    const int kv_bs = max_kv_len * HeadDim;
    auto stride_kv = cute::make_tuple(
        HeadDim, cute::_1{}, cute::make_tuple(kv_bs, HeadDim));
    auto stride_lse = cute::make_tuple(
        cute::_1{}, cute::make_tuple(seq_q, seq_q));
    // Init with max_kv_len so TMA descriptors cover the full buffer
    int q_offset = max_kv_len > seq_q ? max_kv_len : 0;
    auto problem = cute::make_tuple(
        num_q_heads, 1, seq_q, max_kv_len, HeadDim, sliding_window, q_offset, 0);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count = sm_count;

    typename Kernel::Arguments args{
        problem,
        {q_ptr, stride_qo, k_ptr, stride_kv, v_ptr, stride_kv, scale,
         o_ptr, stride_qo, lse_ptr, nullptr, 0},
        {o_ptr, stride_qo, lse_ptr, stride_lse},
        hw_info};

    if (FmhaOp::can_implement(args) != cutlass::Status::kSuccess)
      return false;
    auto status = fmha_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) return false;

    cached_seq_k = max_kv_len;
    cached_num_q_heads = num_q_heads;
    return true;
  }

  // Update pointers + problem shape for each decode call (graph-safe: no cudaMalloc/cuTensorMapEncode)
  bool launch(
      cutlass::bfloat16_t* q_ptr, cutlass::bfloat16_t* k_ptr,
      cutlass::bfloat16_t* v_ptr, cutlass::bfloat16_t* o_ptr, float* lse_ptr,
      int seq_k, int q_offset,
      cudaStream_t stream) {

    auto& p = const_cast<typename Kernel::Params&>(fmha_op.params());
    // Static buffers: TMA descriptors already point to the right addresses.
    // Only update problem shape (seq_k, q_offset) — pure register writes, graph-safe.
    get<3>(p.problem_size) = seq_k;
    get<6>(p.problem_size) = q_offset;

    return FmhaOp::run(p, stream) == cutlass::Status::kSuccess;
  }
};

// ---- Paged TMA decode launcher (no gather, block_size must equal tile_n=64) ----
template <int HeadDim>
struct PagedDecodeFmhaLauncher {
  using FmhaTypes = vllm::gemma_prefill::sm90::GemmaFmhaTypes<HeadDim>;
  using Kernel = typename FmhaTypes::Kernel;
  using FmhaOp = cutlass::device::Universal<Kernel>;
  using Mainloop = typename Kernel::Mainloop;
  using SmemLayoutK = typename Mainloop::SmemLayoutK;
  using SmemLayoutV = typename Mainloop::SmemLayoutV;
  using Element = cutlass::bfloat16_t;

  FmhaOp fmha_op;
  bool initialized = false;

  int cached_q_batch_stride = 0;

  bool init(
      Element* q_ptr, Element* o_ptr, float* lse_ptr,
      Element* k_cache_ptr, Element* v_cache_ptr,
      int num_q_heads, int gqa_group, int seq_q, int max_kv_len,
      int q_stride, float scale, int sliding_window,
      int num_total_blocks, int block_size,
      int64_t stride_block, int64_t stride_slot,
      int64_t v_stride_block, int64_t v_stride_slot,
      int max_num_seqs, int max_blocks_per_seq,
      int device_id, int sm_count, cudaStream_t stream) {

    // SUPERSEDED: kPagedK/kPagedV now issue block16 page-sliced copies
    // (4/tile); this spike's whole-tile (block==tile_n=64) descriptor patch
    // no longer matches the device loop. Falsified experiment, disabled.
    const bool superseded = true;
    if (superseded) return false;

    int q_batch_stride = seq_q * q_stride;
    cached_q_batch_stride = q_batch_stride;
    auto stride_qo = cute::make_tuple(
        q_stride, cute::_1{},
        cute::make_tuple(HeadDim, q_batch_stride));
    const int kv_bs = max_kv_len * HeadDim;
    auto stride_kv = cute::make_tuple(
        HeadDim, cute::_1{}, cute::make_tuple(kv_bs, HeadDim));
    auto stride_lse = cute::make_tuple(
        cute::_1{},
        cute::make_tuple(seq_q, num_q_heads * seq_q));
    int q_offset = max_kv_len;
    auto problem = cute::make_tuple(
        num_q_heads, max_num_seqs, seq_q, max_kv_len, HeadDim,
        sliding_window, q_offset, num_total_blocks);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count = sm_count;

    typename Kernel::Arguments args{
        problem,
        {q_ptr, stride_qo, k_cache_ptr, stride_kv, v_cache_ptr, stride_kv,
         scale, o_ptr, stride_qo, lse_ptr, nullptr, 0},
        {o_ptr, stride_qo, lse_ptr, stride_lse},
        hw_info};

    if (FmhaOp::can_implement(args) != cutlass::Status::kSuccess)
      return false;
    auto status = fmha_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) return false;

    auto& p = const_cast<typename Kernel::Params&>(fmha_op.params());

    int num_kv_heads = num_q_heads / gqa_group;
    int64_t stride_head = int64_t(HeadDim);

    auto gmem_k = cute::make_tensor(
        cute::make_gmem_ptr(k_cache_ptr),
        cute::make_layout(
            cute::make_shape(block_size, HeadDim,
                             cute::make_shape(num_total_blocks, num_kv_heads)),
            cute::make_stride(int(stride_slot), cute::_1{},
                              cute::make_stride(int(stride_block),
                                                int(stride_head)))));
    auto smem_k = SmemLayoutK{}(cute::_, cute::_, cute::Int<0>{});
    auto tma_k = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_k, smem_k);

    auto gmem_v = cute::make_tensor(
        cute::make_gmem_ptr(v_cache_ptr),
        cute::make_layout(
            cute::make_shape(HeadDim, block_size,
                             cute::make_shape(num_total_blocks, num_kv_heads)),
            cute::make_stride(cute::_1{}, int(v_stride_slot),
                              cute::make_stride(int(v_stride_block),
                                                int(stride_head)))));
    auto smem_v = SmemLayoutV{}(cute::_, cute::_, cute::Int<0>{});
    auto tma_v = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_v, smem_v);

    memcpy(const_cast<CUtensorMap*>(
               p.mainloop.tma_load_k_paged.get_tma_descriptor()),
           tma_k.get_tma_descriptor(),
           sizeof(CUtensorMap));
    memcpy(const_cast<CUtensorMap*>(
               p.mainloop.tma_load_v_paged.get_tma_descriptor()),
           tma_v.get_tma_descriptor(),
           sizeof(CUtensorMap));

    p.mainloop.gqa_group = gqa_group;
    p.mainloop.max_blocks_per_seq = max_blocks_per_seq;

    initialized = true;
    return true;
  }

  // Single launch per sequence — all Q heads batched, kernel selects KV head.
  bool launch(
      int seq_k, int q_offset, const int* page_table,
      Element* q_ptr, Element* o_ptr, float* lse_ptr,
      cudaStream_t stream) {
    auto& p = const_cast<typename Kernel::Params&>(fmha_op.params());
    get<3>(p.problem_size) = seq_k;
    get<6>(p.problem_size) = q_offset;
    p.mainloop.page_table = page_table;

    cuTensorMapReplaceAddress(
        const_cast<CUtensorMap*>(
            p.mainloop.tma_load_q.get_tma_descriptor()), q_ptr);
    cuTensorMapReplaceAddress(
        const_cast<CUtensorMap*>(
            p.epilogue.epilogue_TMA.tma_store_d.get_tma_descriptor()), o_ptr);
    p.mainloop.ptr_O = o_ptr;
    p.mainloop.ptr_LSE = lse_ptr;
    p.epilogue.ptr_LSE = lse_ptr;

    return FmhaOp::run(p, stream) == cutlass::Status::kSuccess;
  }

  bool launch_batched(
      int num_seqs, int max_seq_k,
      const int* page_table_2d, int max_blocks_per_seq,
      const int* d_seq_lens,
      Element* q_buf, Element* o_buf, float* lse_buf,
      cudaStream_t stream) {
    auto& p = const_cast<typename Kernel::Params&>(fmha_op.params());
    get<3>(p.problem_size) = max_seq_k;
    get<6>(p.problem_size) = max_seq_k;
    p.mainloop.page_table = page_table_2d;
    p.mainloop.max_blocks_per_seq = max_blocks_per_seq;
    p.mainloop.d_seq_lens = d_seq_lens;
    p.mainloop.ptr_O = o_buf;
    p.mainloop.ptr_LSE = lse_buf;
    p.epilogue.ptr_LSE = lse_buf;
    p.tile_scheduler.grid.z = num_seqs;
    return FmhaOp::run(p, stream) == cutlass::Status::kSuccess;
  }
};

// Batch Q padding: zero-fill + copy each sequence's Q row into padded scratch.
template <typename Element>
__global__ void batch_pad_q_kernel(
    Element* __restrict__ dst, const Element* __restrict__ src,
    int num_seqs, int src_stride, int dst_seq_stride, int row_elems) {
  int s = blockIdx.x;
  if (s >= num_seqs) return;
  Element* d = dst + s * dst_seq_stride;
  const Element* q = src + s * src_stride;
  for (int i = threadIdx.x; i < dst_seq_stride; i += blockDim.x)
    d[i] = (i < row_elems) ? q[i] : Element(0);
}

// Batch O copy: extract each sequence's output row from padded scratch.
template <typename Element>
__global__ void batch_copy_o_kernel(
    Element* __restrict__ dst, const Element* __restrict__ src,
    int num_seqs, int dst_stride, int src_seq_stride, int row_elems) {
  int s = blockIdx.x;
  if (s >= num_seqs) return;
  Element* d = dst + s * dst_stride;
  const Element* o = src + s * src_seq_stride;
  for (int i = threadIdx.x; i < row_elems; i += blockDim.x)
    d[i] = o[i];
}

template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE>
bool gemma_paged_attention_sm90_launcher(
    torch::stable::Tensor& out,
    torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits,
    torch::stable::Tensor& tmp_out,
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
    int sliding_window,
    torch::stable::Tensor& lse_out,
    torch::stable::Tensor& selected_tiles) {

  // Opt-in via GEMMA_SM90_DECODE=1
  static const bool enabled = []() {
    const char* e = getenv("GEMMA_SM90_DECODE");
    return e != nullptr && e[0] == '1';
  }();
  if (!enabled) return false;

  constexpr bool DTYPE_OK =
      std::is_same<T, __nv_bfloat16>::value &&
      (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto);
  if constexpr (!DTYPE_OK) return false;

  if (selected_tiles.numel() > 0) return false;
  if (actual_head_size != 256 && actual_head_size != 512) return false;

  const int num_seqs = query.size(0);
  const int num_q_heads = query.size(1);
  const int head_size = actual_head_size;
  const int q_stride = query.stride(0);
  const int max_num_blocks_per_seq = block_tables.size(1);
  const int64_t kv_stride_block = key_cache.stride(0);
  const int64_t kv_stride_slot = key_cache.stride(1);
  const int64_t kv_stride_head = key_cache.stride(2);
  const int gqa_group = num_q_heads / num_kv_heads;

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  const int* seq_lens_ptr = seq_lens.mutable_data_ptr<int>();

  static int s_device_id = -1;
  static int s_sm_count = 0;
  if (s_device_id < 0) {
    cudaGetDevice(&s_device_id);
    cudaDeviceGetAttribute(&s_sm_count, cudaDevAttrMultiProcessorCount,
                           s_device_id);
  }

  using Element = cutlass::bfloat16_t;
  Element* query_ptr = reinterpret_cast<Element*>(query.data_ptr());
  CACHE_T* k_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* v_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  Element* out_ptr = reinterpret_cast<Element*>(out.data_ptr());
  int* block_tables_ptr = block_tables.mutable_data_ptr<int>();
  const int64_t v_stride_block = value_cache.stride(0);
  const int64_t v_stride_slot = value_cache.stride(1);

  if (max_seq_len <= 0) return false;

  constexpr int kAlignment = 16 / sizeof(Element);
  constexpr int padded_q = kAlignment;
  const size_t q_row_bytes = (size_t)num_q_heads * head_size * sizeof(Element);
  const int fmha_q_stride = num_q_heads * head_size;
  const int padded_kv = (max_seq_len + kAlignment - 1) / kAlignment * kAlignment;

  // ---- Static buffers: grow-only, allocated outside graph capture ----
  static CACHE_T* k_expanded = nullptr;
  static float* lse_scratch = nullptr;
  static Element* q_scratch = nullptr;
  static Element* o_scratch = nullptr;
  static size_t s_kv_cap = 0, s_lse_cap = 0, s_qo_cap = 0;
  static bool s_launchers_init = false;
  static int s_max_num_seqs = 0;

  const int q_seq_stride = padded_q * fmha_q_stride;
  const size_t kv_bytes = (size_t)padded_kv * num_q_heads * head_size * sizeof(CACHE_T);
  const size_t lse_bytes = (size_t)num_seqs * padded_q * num_q_heads * sizeof(float);
  const size_t qo_bytes = (size_t)num_seqs * q_seq_stride * sizeof(Element);
  const int row_elems = num_q_heads * head_size;

  // The SIMT+paged path (block_size=64) is fully graph-safe.
  // The gather fallback path (other block sizes) needs scratch buffers.
  if constexpr (BLOCK_SIZE != 64) {
    if (kv_bytes > s_kv_cap) {
      if (k_expanded) cudaFree(k_expanded);
      cudaMalloc(&k_expanded, kv_bytes);
      s_kv_cap = kv_bytes;
      s_launchers_init = false;
    }
    if (lse_bytes > s_lse_cap) {
      if (lse_scratch) cudaFree(lse_scratch);
      cudaMalloc(&lse_scratch, lse_bytes);
      s_lse_cap = lse_bytes;
    }
  }

  // ---- Direct paged TMA path (block_size=64 required) ----
  if constexpr (BLOCK_SIZE == 64) {
    const int num_total_blocks = key_cache.size(0);

    // SIMT+paged decode kernel — entirely graph-safe (no host APIs at launch).
    // Grid: (num_kv_heads, num_seqs). Each CTA handles one (kv_head, seq) pair.
    // BDY warps share the same K tile (GQA reuse).
    const dim3 grid(num_kv_heads, num_seqs);
    constexpr int SIMT_BN = 32;
    const size_t smem = (size_t)2 * SIMT_BN * head_size * sizeof(Element);
    float* lse_ptr = (lse_out.numel() > 0)
        ? reinterpret_cast<float*>(lse_out.data_ptr()) : nullptr;

    __nv_bfloat16* out_bf = reinterpret_cast<__nv_bfloat16*>(out_ptr);
    const __nv_bfloat16* q_bf = reinterpret_cast<const __nv_bfloat16*>(query_ptr);
    const __nv_bfloat16* k_bf = reinterpret_cast<const __nv_bfloat16*>(k_cache_ptr);
    const __nv_bfloat16* v_bf = reinterpret_cast<const __nv_bfloat16*>(v_cache_ptr);

#define LAUNCH_TMA_SIMT(HD, BDY_, BDZ_, KEV, SW)                              \
    do {                                                                       \
      auto kernel = gemma_decode_tma_simt_kernel<HD, SIMT_BN, BDY_, BDZ_, KEV, SW>; \
      { static bool _a = false; if (!_a) {                                     \
        if (smem > 48*1024)                                                    \
          cudaFuncSetAttribute(kernel,                                         \
              cudaFuncAttributeMaxDynamicSharedMemorySize, smem);               \
        _a = true;                                                             \
      }}                                                                       \
      kernel<<<grid, BDY_*BDZ_*32, smem, stream>>>(                            \
          out_bf, q_bf, k_bf, v_bf,                                \
          block_tables_ptr, seq_lens_ptr,                                       \
          num_kv_heads, scale, q_stride, max_num_blocks_per_seq,               \
          kv_stride_block, kv_stride_slot, kv_stride_head,                     \
          v_stride_block, v_stride_slot, kv_stride_head,                       \
          BLOCK_SIZE, sliding_window, lse_ptr);                                \
    } while (0)

    if (head_size == 256) {
      if (k_eq_v) {
        if (sliding_window > 0)
          LAUNCH_TMA_SIMT(256, 2, 4, true, true);
        else
          LAUNCH_TMA_SIMT(256, 2, 4, true, false);
      } else {
        if (sliding_window > 0)
          LAUNCH_TMA_SIMT(256, 2, 4, false, true);
        else
          LAUNCH_TMA_SIMT(256, 2, 4, false, false);
      }
    } else {
      if (gqa_group <= 2) {
        if (k_eq_v)
          LAUNCH_TMA_SIMT(512, 2, 4, true, false);
        else
          LAUNCH_TMA_SIMT(512, 2, 4, false, false);
      } else if (gqa_group <= 4) {
        if (k_eq_v)
          LAUNCH_TMA_SIMT(512, 4, 2, true, false);
        else
          LAUNCH_TMA_SIMT(512, 4, 2, false, false);
      } else {
        if (k_eq_v)
          LAUNCH_TMA_SIMT(512, 16, 1, true, false);
        else
          LAUNCH_TMA_SIMT(512, 16, 1, false, false);
      }
    }
#undef LAUNCH_TMA_SIMT
    return true;
  }

  // ---- Gather fallback (block_size != 64) ----
  static DecodeFmhaLauncher<256> launcher_256;
  static DecodeFmhaLauncher<512> launcher_512;

  if (!s_launchers_init) {
    Element* k = reinterpret_cast<Element*>(k_expanded);
    if (head_size == 256 || actual_head_size == 256) {
      if (!launcher_256.init_descriptors(
              q_scratch, k, k, o_scratch, lse_scratch,
              num_q_heads, padded_q, padded_kv, fmha_q_stride,
              scale, sliding_window, s_device_id, s_sm_count, stream))
        return false;
    }
    if (head_size == 512 || actual_head_size == 512) {
      if (!launcher_512.init_descriptors(
              q_scratch, k, k, o_scratch, lse_scratch,
              num_q_heads, padded_q, padded_kv, fmha_q_stride,
              scale, sliding_window, s_device_id, s_sm_count, stream))
        return false;
    }
    s_launchers_init = true;
  }

  // ---- Per-call decode loop (gather path) ----
  bool ok = true;

  for (int s = 0; s < num_seqs && ok; s++) {
    const int* seq_block_table = block_tables_ptr + s * max_num_blocks_per_seq;

    const int total_elems = num_q_heads * max_seq_len * head_size;
    constexpr int kThreads = 256;
    const int gather_blocks = (total_elems + kThreads - 1) / kThreads;

    decode_gather_kv_kernel<CACHE_T><<<gather_blocks, kThreads, 0, stream>>>(
        k_expanded, k_cache_ptr, seq_block_table, max_seq_len, padded_kv,
        num_q_heads, num_kv_heads, head_size, gqa_group, BLOCK_SIZE,
        kv_stride_block, kv_stride_slot, kv_stride_head);

    cudaMemsetAsync(q_scratch, 0, padded_q * fmha_q_stride, stream);
    cudaMemcpyAsync(q_scratch, query_ptr + s * q_stride,
                    q_row_bytes, cudaMemcpyDeviceToDevice, stream);

    Element* k = reinterpret_cast<Element*>(k_expanded);
    int q_offset = max_seq_len;

    if (head_size == 256) {
      ok = launcher_256.launch(q_scratch, k, k, o_scratch, lse_scratch,
                               max_seq_len, q_offset, stream);
    } else {
      ok = launcher_512.launch(q_scratch, k, k, o_scratch, lse_scratch,
                               max_seq_len, q_offset, stream);
    }

    if (ok) {
      cudaMemcpyAsync(out_ptr + s * q_stride, o_scratch,
                      q_row_bytes, cudaMemcpyDeviceToDevice, stream);
    }
  }

  return ok;
}

// Explicit instantiations.
#define INST_SM90_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE)              \
  template bool gemma_paged_attention_sm90_launcher<                       \
      T, CACHE_T, BLOCK_SIZE, KV_DTYPE>(                                  \
      torch::stable::Tensor&, torch::stable::Tensor&,                     \
      torch::stable::Tensor&, torch::stable::Tensor&,                     \
      torch::stable::Tensor&, torch::stable::Tensor&,                     \
      torch::stable::Tensor&, int, float,                                 \
      torch::stable::Tensor&, torch::stable::Tensor&, int,                \
      torch::stable::Tensor&, torch::stable::Tensor&,                     \
      int, bool, int,                                                     \
      torch::stable::Tensor&, torch::stable::Tensor&);

INST_SM90_LAUNCHER(__nv_bfloat16, __nv_bfloat16, 16,
                   vllm::Fp8KVCacheDataType::kAuto)
INST_SM90_LAUNCHER(__nv_bfloat16, __nv_bfloat16, 32,
                   vllm::Fp8KVCacheDataType::kAuto)
INST_SM90_LAUNCHER(__nv_bfloat16, __nv_bfloat16, 64,
                   vllm::Fp8KVCacheDataType::kAuto)

#undef INST_SM90_LAUNCHER
