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
    int64_t stride_block, int64_t stride_slot, int64_t stride_head,
    int tok_offset = 0) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_q_heads * seq_len * head_size;
  if (idx >= total) return;

  const int d = idx % head_size;
  const int t = (idx / head_size) % seq_len;
  const int h_q = idx / (head_size * seq_len);
  const int h_kv = h_q / gqa_group;

  const int tok = t + tok_offset;  // sliding: window slice, not full context
  const int logical_block = tok / page_size;
  const int slot = tok % page_size;
  const int physical_block = block_table[logical_block];

  kv_out[h_q * out_seq_stride * head_size + t * head_size + d] =
      kv_cache[physical_block * stride_block + slot * stride_slot +
               h_kv * stride_head + d];
}

// Varlen batch staging: pack ragged Q (rows cu_q[s]..cu_q[s+1]) into padded
// [seq][max_q_pad] scratch (pad rows zeroed), and scatter real O rows back.
template <typename Element>
__global__ void varlen_pad_q_kernel(
    Element* __restrict__ dst, const Element* __restrict__ src,
    const int* __restrict__ cu_q, int max_q_pad, int64_t src_stride,
    int row_elems) {
  const int s = blockIdx.y;
  const int r = blockIdx.x;
  const int q0 = cu_q[s], q1 = cu_q[s + 1];
  Element* d = dst + ((int64_t)s * max_q_pad + r) * row_elems;
  if (r < q1 - q0) {
    const Element* p = src + (int64_t)(q0 + r) * src_stride;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) d[i] = p[i];
  } else {
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) d[i] = Element(0);
  }
}

template <typename Element>
__global__ void varlen_scatter_o_kernel(
    Element* __restrict__ dst, const Element* __restrict__ src,
    const int* __restrict__ cu_q, int max_q_pad, int64_t dst_stride,
    int row_elems) {
  const int s = blockIdx.y;
  const int r = blockIdx.x;
  const int q0 = cu_q[s], q1 = cu_q[s + 1];
  if (r >= q1 - q0) return;
  const Element* p = src + ((int64_t)s * max_q_pad + r) * row_elems;
  Element* d = dst + (int64_t)(q0 + r) * dst_stride;
  for (int i = threadIdx.x; i < row_elems; i += blockDim.x) d[i] = p[i];
}

// ---- KV-split staging (skinny-q long-KV chunks; hd512 global layers) ----
// Real seq s becomes S virtual batch entries, each owning a TILE-ALIGNED kv
// slice; the final split owns the whole causal-frontier region
// [q_offset/64*64, kv) so every split has l>0 for every real q row (no
// empty-softmax rows). Combine merges via LSE weights.
__global__ void split_params_build_kernel(
    int* __restrict__ d_sl_v, int* __restrict__ d_qoff_v,
    int* __restrict__ d_kvlo_v, int* __restrict__ vtable,
    const int* __restrict__ seq_lens, const int* __restrict__ cu_q,
    const int* __restrict__ block_tables, int num_seqs, int S,
    int uniform_q_len, int max_blocks, int tile) {
  const int v = blockIdx.x;
  if (v >= num_seqs * S) return;
  const int s = v / S, j = v % S;
  const int kv = seq_lens[s];
  const int qlen = cu_q != nullptr ? cu_q[s + 1] - cu_q[s] : uniform_q_len;
  const int qoff = kv - qlen;
  if (threadIdx.x == 0) {
    const int frontier_t = qoff / tile;      // dense region = [0, frontier_t)
    // Balanced partition of the dense region over S-1 splits: slice j =
    // [j*F/(S-1), (j+1)*F/(S-1)) tiles — never empty while F >= S-1
    // (launcher guarantees min_kv; combine still skips empty-LSE entries).
    int lo_t;
    if (j == S - 1) {
      lo_t = frontier_t;
      d_sl_v[v] = kv;                        // final split: frontier .. end
    } else {
      lo_t = (int)(((int64_t)j * frontier_t) / (S - 1));
      const int hi_t = (int)(((int64_t)(j + 1) * frontier_t) / (S - 1));
      d_sl_v[v] = hi_t * tile;
    }
    d_kvlo_v[v] = lo_t * tile;
    d_qoff_v[v] = qoff;
  }
  // replicate this real seq's block-table row for the virtual entry
  for (int i = threadIdx.x; i < max_blocks; i += blockDim.x)
    vtable[(int64_t)v * max_blocks + i] = block_tables[(int64_t)s * max_blocks + i];
}

// Q replication: virtual entry v=(s,j) gets a copy of seq s's q rows.
template <typename Element>
__global__ void split_replicate_q_kernel(
    Element* __restrict__ dst, const Element* __restrict__ src,
    const int* __restrict__ cu_q, int S, int max_q_pad, int uniform_q_len,
    int64_t src_stride, int row_elems) {
  const int v = blockIdx.y;
  const int s = v / S;
  const int r = blockIdx.x;
  const int q0 = cu_q != nullptr ? cu_q[s] : s * uniform_q_len;
  const int qlen = cu_q != nullptr ? cu_q[s + 1] - q0 : uniform_q_len;
  Element* d = dst + ((int64_t)v * max_q_pad + r) * row_elems;
  if (r < qlen) {
    const Element* p = src + (int64_t)(q0 + r) * src_stride;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) d[i] = p[i];
  } else {
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) d[i] = Element(0);
  }
}

// LSE-weighted merge of S partials per real seq; writes final rows directly.
template <typename Element>
__global__ void split_combine_o_kernel(
    Element* __restrict__ out, const Element* __restrict__ o_v,
    const float* __restrict__ lse_v, const int* __restrict__ cu_q,
    int S, int max_q_pad, int uniform_q_len, int num_q_heads, int head_size,
    int64_t out_stride, int row_elems) {
  const int s = blockIdx.z;
  const int r = blockIdx.x;
  const int q0 = cu_q != nullptr ? cu_q[s] : s * uniform_q_len;
  const int qlen = cu_q != nullptr ? cu_q[s + 1] - q0 : uniform_q_len;
  if (r >= qlen) return;
  Element* dst = out + (int64_t)(q0 + r) * out_stride;
  const int64_t lse_batch = (int64_t)max_q_pad * num_q_heads;
  {
    const int h = blockIdx.y;
    // The kernel writes lse = +INF for zero-mass rows (its empty sentinel):
    // treat those splits as absent.
    float m = -INFINITY;
    for (int j = 0; j < S; j++) {
      float l = lse_v[(int64_t)(s * S + j) * lse_batch + h * max_q_pad + r];
      if (isfinite(l) && l > m) m = l;
    }
    float den = 0.f;
    for (int j = 0; j < S; j++) {
      float l = lse_v[(int64_t)(s * S + j) * lse_batch + h * max_q_pad + r];
      if (isfinite(l)) den += expf(l - m);
    }
    const float inv_den = den > 0.f ? 1.f / den : 0.f;
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
      float acc = 0.f;
      for (int j = 0; j < S; j++) {
        float l = lse_v[(int64_t)(s * S + j) * lse_batch + h * max_q_pad + r];
        if (!isfinite(l)) continue;
        float w = expf(l - m) * inv_den;
        acc += w * float(o_v[((int64_t)(s * S + j) * max_q_pad + r) *
                             row_elems + h * head_size + d]);
      }
      dst[h * head_size + d] = Element(acc);
    }
  }
}


// Paged-pool descriptor set: non-null enables gather-free KV reads straight
// from the block_size=16 cache pool (page_table = this launch's block table,
// row-major [seq][max_blocks] for batched, or a single row for per-seq).
struct PagedPool {
  const cutlass::bfloat16_t* pool_k;
  const cutlass::bfloat16_t* pool_v;  // null for k_eq_v
  int64_t stride_block, stride_slot, stride_head;
  int num_blocks;       // pool capacity (TMA extent)
  int num_kv_heads;
  const int* page_table;
  int max_blocks_per_seq;
  int page_size;        // 16 (page-sliced TMA) or 64 (whole-tile TMA)
};

template <int HeadDim, bool KEqV = false>
struct FmhaCachedLauncher {
  using FmhaTypes = vllm::gemma_prefill::sm90::GemmaFmhaTypes<HeadDim, KEqV>;
  using Kernel = typename FmhaTypes::Kernel;
  using FmhaOp = cutlass::device::Universal<Kernel>;

  FmhaOp fmha_op;
  int cached_seq_len = 0;
  int cached_seq_k = 0;
  int cached_q_offset = -1;
  int cached_num_q_heads = 0;
  int cached_num_seqs = 0;
  int cached_sliding_window = -1;
  bool cached_paged = false;
  bool cached_varlen = false;

  bool launch(
      cutlass::bfloat16_t* q_ptr,
      cutlass::bfloat16_t* k_ptr,
      cutlass::bfloat16_t* v_ptr,
      cutlass::bfloat16_t* o_ptr,
      float* lse_ptr,
      int num_q_heads,
      int gqa_group,
      int num_seqs,
      int seq_q,
      int seq_k,
      int q_offset,
      int q_stride,
      float scale,
      int sliding_window,
      const int* mm_ranges_ptr,
      int max_mm_ranges,
      int device_id,
      int sm_count,
      cudaStream_t stream,
      const PagedPool* paged = nullptr,
      const int* varlen_seq_lens = nullptr,
      const int* varlen_cu_q = nullptr,
      const int* varlen_q_offsets = nullptr,
      const int* varlen_kv_lo = nullptr) {
    // Key must cover everything baked into Params: seq_q, seq_k (drives the
    // KV extents AND q_offset = seq_k - seq_q for extends/chunked prefill),
    // heads, and the sliding window. Keying on seq_q alone served stale
    // Params to every extend that followed a full prefill of the same len.
    bool shape_changed =
        (seq_q != cached_seq_len || seq_k != cached_seq_k ||
         q_offset != cached_q_offset ||
         num_q_heads != cached_num_q_heads ||
         num_seqs != cached_num_seqs ||
         sliding_window != cached_sliding_window ||
         (paged != nullptr) != cached_paged ||
         (varlen_seq_lens != nullptr) != cached_varlen);

    if (shape_changed) {
      // Batched (uniform) launch: batch mode strides. Q/O rows are packed
      // varlen with uniform q_len -> per-seq stride seq_q*q_stride. Gathered
      // KV scratch is [seq][kv_head][seq_k][hd] dense.
      auto stride_qo = cute::make_tuple(
          q_stride, cute::_1{},
          cute::make_tuple(HeadDim, seq_q * q_stride));
      const int kv_bs = seq_k * HeadDim;  // gathered KV is [seq_k, hd]/head
      const int n_kvh = num_q_heads / (gqa_group > 0 ? gqa_group : 1);
      auto stride_kv = cute::make_tuple(
          HeadDim, cute::_1{},
          cute::make_tuple(kv_bs, n_kvh * kv_bs));
      auto stride_lse = cute::make_tuple(
          cute::_1{}, cute::make_tuple(seq_q, seq_q * num_q_heads));
      // q_offset = context length (kv_len - UNPADDED q_len): query rows map
      // to absolute positions [q_offset, q_offset + q_len). Hardcoded 0 broke
      // every extend; deriving it from the PADDED seq_q broke non-multiple-
      // of-8 lengths, so the caller passes it explicitly.
      auto problem = cute::make_tuple(
          num_q_heads, num_seqs, seq_q, seq_k, HeadDim, sliding_window,
          q_offset, 0, 0);  // trailing: num_blocks (dead spike), kv_lo

      cutlass::KernelHardwareInfo hw_info;
      hw_info.device_id = device_id;
      hw_info.sm_count = sm_count;

      typename Kernel::Arguments args{
          problem,
          {q_ptr, stride_qo, k_ptr, stride_kv, v_ptr, stride_kv, scale,
           o_ptr, stride_qo, lse_ptr, mm_ranges_ptr, max_mm_ranges},
          {o_ptr, stride_qo, lse_ptr, stride_lse},
          hw_info};
      if (paged != nullptr) {
        args.mainloop.kv_pool_k = paged->pool_k;
        args.mainloop.kv_pool_v = KEqV ? nullptr : paged->pool_v;
        args.mainloop.pool_stride_block = paged->stride_block;
        args.mainloop.pool_stride_slot = paged->stride_slot;
        args.mainloop.pool_stride_head = paged->stride_head;
        args.mainloop.pool_num_blocks = paged->num_blocks;
        args.mainloop.pool_num_kv_heads = paged->num_kv_heads;
        args.mainloop.pool_page_size = paged->page_size;
      }

      if (FmhaOp::can_implement(args) != cutlass::Status::kSuccess)
        return false;

      auto status = fmha_op.initialize(args, nullptr, stream);
      if (status != cutlass::Status::kSuccess) return false;

      cached_seq_len = seq_q;
      cached_seq_k = seq_k;
      cached_q_offset = q_offset;
      cached_num_q_heads = num_q_heads;
      cached_num_seqs = num_seqs;
      cached_sliding_window = sliding_window;
      cached_paged = (paged != nullptr);
      cached_varlen = (varlen_seq_lens != nullptr);
    } else {
      // Shape unchanged — update only base pointers in TMA descriptors
      auto& p = const_cast<typename Kernel::Params&>(fmha_op.params());
      cuTensorMapReplaceAddress(
          const_cast<CUtensorMap*>(p.mainloop.tma_load_q.get_tma_descriptor()),
          q_ptr);
      if (paged == nullptr) {
        cuTensorMapReplaceAddress(
            const_cast<CUtensorMap*>(p.mainloop.tma_load_k.get_tma_descriptor()),
            k_ptr);
        cuTensorMapReplaceAddress(
            const_cast<CUtensorMap*>(p.mainloop.tma_load_v.get_tma_descriptor()),
            v_ptr);
      } else {
        // Pool GEOMETRY is stable, but the pool BASE is PER LAYER (vLLM
        // allocates a distinct KV tensor per layer) and this launcher is
        // shared by every layer of the (HeadDim,KEqV) type: same-shape
        // calls from different layers MUST repatch the descriptor base or
        // they silently read the first layer's pool.
        CUtensorMap* kd = const_cast<CUtensorMap*>(
            paged->page_size == 64
                ? p.mainloop.tma_load_k_paged64.get_tma_descriptor()
                : p.mainloop.tma_load_k_paged.get_tma_descriptor());
        cuTensorMapReplaceAddress(
            kd, const_cast<cutlass::bfloat16_t*>(paged->pool_k));
        if (!KEqV && paged->pool_v != nullptr) {
          CUtensorMap* vd = const_cast<CUtensorMap*>(
              paged->page_size == 64
                  ? p.mainloop.tma_load_v_paged64.get_tma_descriptor()
                  : p.mainloop.tma_load_v_paged.get_tma_descriptor());
          cuTensorMapReplaceAddress(
              vd, const_cast<cutlass::bfloat16_t*>(paged->pool_v));
        }
      }
      // Per-seq multimodal ranges: pointer varies per call (pre-existing
      // staleness for equal-shaped mm sequences; fixed alongside).
      p.mainloop.mm_prefix_ranges = mm_ranges_ptr;
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

    {
      auto& p = const_cast<typename Kernel::Params&>(fmha_op.params());
      // GQA-dense KV buffers: loaders divide the q-head coord by this.
      p.mainloop.contig_gqa_group = gqa_group;
      // Paged runtime state (page_table varies per call; descriptors don't).
      if (paged != nullptr) {
        p.mainloop.page_table = paged->page_table;
        p.mainloop.gqa_group = gqa_group;
        p.mainloop.max_blocks_per_seq = paged->max_blocks_per_seq;
      } else {
        p.mainloop.page_table = nullptr;
      }
      // Varlen batch state: per-seq kv lens + q starts (null = uniform).
      p.mainloop.d_seq_lens = varlen_seq_lens;
      p.mainloop.d_cu_seqlens_q = varlen_cu_q;
      p.mainloop.d_q_offsets = varlen_q_offsets;
      p.mainloop.d_kv_lo = varlen_kv_lo;
    }
    return FmhaOp::run(
               const_cast<typename Kernel::Params&>(fmha_op.params()), stream)
               == cutlass::Status::kSuccess;
  }
};

template <int HeadDim, bool KEqV = false>
static bool launch_fmha_batched(
    cutlass::bfloat16_t* q_ptr,
    cutlass::bfloat16_t* k_ptr,
    cutlass::bfloat16_t* v_ptr,
    cutlass::bfloat16_t* o_ptr,
    float* lse_ptr,
    int num_q_heads,
    int gqa_group,
    int num_seqs,
    int seq_q,
    int seq_k,
    int q_offset,
    int q_stride,
    float scale,
    int sliding_window,
    const int* mm_ranges_ptr,
    int max_mm_ranges,
    int device_id,
    int sm_count,
    cudaStream_t stream,
    const PagedPool* paged = nullptr,
    const int* varlen_seq_lens = nullptr,
    const int* varlen_cu_q = nullptr,
    const int* varlen_q_offsets = nullptr,
    const int* varlen_kv_lo = nullptr) {
  static FmhaCachedLauncher<HeadDim, KEqV> launcher;
  return launcher.launch(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, num_q_heads,
                         gqa_group, num_seqs, seq_q, seq_k, q_offset,
                         q_stride, scale, sliding_window, mm_ranges_ptr,
                         max_mm_ranges, device_id, sm_count, stream, paged,
                         varlen_seq_lens, varlen_cu_q,
                         varlen_q_offsets, varlen_kv_lo);
}

template <typename T, typename CACHE_T>
bool gemma_prefill_sm90_launcher(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int num_kv_heads, float scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, torch::stable::Tensor& cu_seqlens_q,
    int max_q_len, int page_size, bool k_eq_v, int sliding_window,
    torch::stable::Tensor& mm_prefix_ranges, bool non_causal,
    torch::stable::Tensor& lse_out, torch::stable::Tensor& seq_lens_cpu,
    torch::stable::Tensor& cu_seqlens_q_cpu) {

  // Default ON (P7): the CUTLASS warp-spec path is the production prefill on
  // Hopper (TTFT 28.5/180.6/783ms at 512/4k/16k b=1 vs FA4's 34.4/177.4/773).
  // GEMMA_SM90_PREFILL=0 reverts to the wmma v2 kernel.
  static const bool enabled = []() {
    const char* e = getenv("GEMMA_SM90_PREFILL");
    return e == nullptr || e[0] != '0';
  }();
  if (!enabled) return false;

  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int num_seqs = seq_lens.size(0);

  if (head_size != 256 && head_size != 512) return false;
  if (non_causal) return false;   // cascade prefix pass -> wmma fallback
  if (lse_out.numel() > 0) return false;  // LSE epilogue not implemented here
  // Tiny-query steps (prefix-cache last-block recompute q<=17, small
  // extends): the per-seq gather+launch of this path costs more than the
  // attention itself; the wmma kernel reads paged KV directly (no gather)
  // and is decode-like at small q. Real prefills (chunked q>=64 or full)
  // stay here.
  const bool prefill_debug = getenv("GEMMA_PREFILL_DEBUG") != nullptr;
  if (prefill_debug) {
    fprintf(stderr, "[sm90-prefill] num_seqs=%d max_q_len=%d num_tokens=%d\n",
            num_seqs, max_q_len, static_cast<int>(query.size(0)));
  }
  // Paged KV reads straight from the block_size 16/64 cache pools: no
  // gather, descriptors stable. GEMMA_PREFILL_PAGED=0 reverts.
  static const bool paged_enabled = []() {
    const char* e = getenv("GEMMA_PREFILL_PAGED");
    return e == nullptr || e[0] != '0';
  }();
  // 16-token pools (hd256 sliding group) use page-sliced TMA; 64-token
  // pools (hd512 global group: hybrid manager unifies page BYTES across
  // groups => 64 tokens/page) use whole-tile TMA (dense boxes, no scatter
  // latency => no long-KV threshold needed).
  const bool use_paged = paged_enabled &&
                         (page_size == 16 || page_size == 64) &&
                         sizeof(CACHE_T) == sizeof(T);
  if (prefill_debug)
    fprintf(stderr, "[sm90-prefill] use_paged=%d page_size=%d\n",
            int(use_paged), page_size);
  // Tiny-q MULTI-SEQ batches (hd512 spec-verify: q=1+k, kv large) are now
  // served by the batched paged/varlen/split paths (q padded to 8, one
  // launch) — far faster than wmma at kv >> q. Single-seq tiny q keeps the
  // wmma economics.
  const bool mq_batched_ok =
      head_size == 512 && num_seqs > 1 && use_paged &&
      mm_prefix_ranges.numel() == 0 &&
      cu_seqlens_q.numel() >= num_seqs + 1 && seq_lens.numel() >= num_seqs;
  if (max_q_len <= 32 && !mq_batched_ok) return false;

  constexpr int kAlignment = 16 / sizeof(T);
  if (max_q_len == 0) return false;

  const int num_tokens = static_cast<int>(query.size(0));
  const bool equal_lens = (num_tokens == num_seqs * max_q_len);

  // This launcher performs host-side work per call (Params init, grow-only
  // cudaMalloc, cuTensorMapEncode): NONE of it is CUDA-graph-capture-safe.
  // Inside capture, bail to the wmma fallback (plain launch, persistent
  // state). Keeps captured spec-verify steps correct.
  {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(get_current_cuda_stream(), &cap);
    if (cap != cudaStreamCaptureStatusNone) return false;
  }

  // Paged KV reads straight from the block_size=16 cache pool: no gather, no
  // dense scratch, TMA descriptors stable across steps. GEMMA_PREFILL_PAGED=0
  // reverts to the gather path.


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
      (size_t)max_padded * num_kv_heads * head_size * sizeof(CACHE_T);
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

  if (!use_paged && kv_expanded_bytes > s_kv_cap) {
    if (k_expanded) cudaFree(k_expanded);
    cudaMalloc(&k_expanded, kv_expanded_bytes);
    s_kv_cap = kv_expanded_bytes;
  }
  if (!use_paged && !k_eq_v && kv_expanded_bytes > s_v_cap) {
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

  // For variable-length sequences: cu_seqlens_q on host. Prefer the CPU
  // metadata twin (zero sync); D2H fallback otherwise.
  const bool have_cpu_sl = seq_lens_cpu.numel() >= num_seqs;
  const bool have_cpu_cu = cu_seqlens_q_cpu.numel() >= num_seqs + 1;
  std::vector<int> h_cu_seqlens_q;
  if (!equal_lens) {
    h_cu_seqlens_q.resize(num_seqs + 1);
    if (have_cpu_cu) {
      const int* p_cu = cu_seqlens_q_cpu.mutable_data_ptr<int>();
      std::copy(p_cu, p_cu + num_seqs + 1, h_cu_seqlens_q.begin());
    } else {
      cudaMemcpyAsync(h_cu_seqlens_q.data(),
                      cu_seqlens_q.mutable_data_ptr<int>(),
                      (num_seqs + 1) * sizeof(int),
                      cudaMemcpyDeviceToHost, stream);
    }
  }
  // Total (context + new) length per sequence: extends (chunked prefill,
  // prefix-cache hits) have kv_len > q_len; the kernel needs the real KV
  // extent and q_offset = kv_len - q_len. Prefer the CPU metadata twins
  // (zero sync); the D2H+sync fallback drains the stream PER LAYER CALL
  // (~11ms/step measured) and only runs when CPU tensors are absent.
  std::vector<int> h_seq_lens(num_seqs);
  if (have_cpu_sl) {
    const int* p_sl = seq_lens_cpu.mutable_data_ptr<int>();
    std::copy(p_sl, p_sl + num_seqs, h_seq_lens.begin());
    if (!equal_lens && !have_cpu_cu)
      cudaStreamSynchronize(stream);  // cu D2H fallback issued above
  } else {
    cudaMemcpyAsync(h_seq_lens.data(), seq_lens.mutable_data_ptr<int>(),
                    num_seqs * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
  int max_kv_len = 0;
  for (int s2 = 0; s2 < num_seqs; s2++)
    max_kv_len = std::max(max_kv_len, h_seq_lens[s2]);
  // Long-KV chunks stay on the gather path: scattered per-page boxes are
  // latency-bound under the 2-stage KV pipeline (hd512 16k chunk 692us paged
  // vs 151us gathered; hd256 window chunks ~46us vs ~25us). Short-KV steps
  // (APC extends / short-prompt chunks) win big on paged: no gather kernels,
  // no scratch, pool-stable descriptors. GEMMA_PREFILL_PAGED_MAXKV tunes the
  // crossover; the deep fix (multi-lane issue / consecutive-page wide boxes)
  // is future work.
  static const int paged_maxkv = []() {
    const char* e = getenv("GEMMA_PREFILL_PAGED_MAXKV");
    return e != nullptr ? atoi(e) : 1024;
  }();
  // Long-KV paged only for SKINNY q: big-q prefills re-read every KV
  // tile per q-tile, and the gathered dense buffer wins on L2/TLB
  // locality (TTFT-16k regressed ~5% routing full prefills paged).
  // Session 20: hd256 sliding paged is ~2x gather at ALL KV (window-bounded
  // reads), but the multi-seq batched-paged path hangs at b>1 long-KV
  // (unresolved). Widen for single-seq only — validated -2.7% TTFT-16k;
  // b>1 stays on the proven gather path until the hang is root-caused.
  const bool paged_call =
      use_paged && (max_kv_len <= paged_maxkv ||
                    (page_size == 64 && max_q_len <= 128) ||
                    (page_size == 16 && sliding_window > 0 && num_seqs == 1));
  // Re-grow the expanded-KV scratch for the KV extent (sized for q above).
  const size_t kv_needed =
      (size_t)max_kv_len * num_kv_heads * head_size * sizeof(CACHE_T);
  if (!paged_call && kv_needed > s_kv_cap) {
    if (k_expanded) cudaFree(k_expanded);
    cudaMalloc(&k_expanded, kv_needed);
    s_kv_cap = kv_needed;
  }
  if (!paged_call && !k_eq_v && kv_needed > s_v_cap) {
    if (v_expanded) cudaFree(v_expanded);
    cudaMalloc(&v_expanded, kv_needed);
    s_v_cap = kv_needed;
  }

  // Paged pool geometry (shared by both launch paths below).
  const int pool_num_blocks = static_cast<int>(key_cache.size(0));
  PagedPool pool_proto{
      reinterpret_cast<const Element*>(key_cache_ptr),
      k_eq_v ? nullptr : reinterpret_cast<const Element*>(value_cache_ptr),
      kv_stride_block, kv_stride_slot, kv_stride_head,
      pool_num_blocks, num_kv_heads,
      /*page_table=*/nullptr, max_num_blocks_per_seq, page_size};

  bool ok = true;
  const int padded_qo_stride = num_q_heads * head_size;

  // ---- KV-split paged path (hd512 skinny-q over long KV) ----
  // FA4's remaining structural edge on re-prefill chunks: at q<=128 our
  // launch has ~num_q_heads CTAs of parallelism while the KV range is huge.
  // Split each seq into S tile-aligned kv slices as VIRTUAL batch entries
  // (per-CTA seq_k/q_offset/kv_lo overrides), one paged batched launch, then
  // an LSE-weighted combine. Splits also shrink each CTA's serial tile
  // stream, curing the paged long-KV latency wall (692us) as a side effect.
  static const bool splitkv_enabled = []() {
    const char* e = getenv("GEMMA_PREFILL_SPLITKV");
    return e == nullptr || e[0] != '0';
  }();
  int min_kv_len = max_kv_len;
  for (int s2 = 0; s2 < num_seqs; s2++)
    min_kv_len = std::min(min_kv_len, h_seq_lens[s2]);
  if (prefill_debug)
    fprintf(stderr,
            "[sm90-prefill] splitkv gate: hd=%d max_q=%d min_kv=%d paged=%d ps=%d\n",
            head_size, max_q_len, min_kv_len, int(use_paged), page_size);
  if (splitkv_enabled && use_paged && head_size == 512 && max_q_len <= 128 &&
      min_kv_len >= 2048 && mm_prefix_ranges.numel() == 0) {
    const int S = std::min(32, std::max(2, max_kv_len / 512));
    const int V = num_seqs * S;
    const int max_q_pad =
        (max_q_len + kAlignment - 1) / kAlignment * kAlignment;
    const int row_elems = num_q_heads * head_size;
    // +128 rows: Q TMA whole-tile read slack (see varlen branch note).
    const size_t sq_bytes =
        (size_t)(V * max_q_pad + 128) * row_elems * sizeof(T);
    const size_t slse_bytes =
        (size_t)V * max_q_pad * num_q_heads * sizeof(float);
    static int* d_split_meta = nullptr;   // [3][V]: sl, qoff, kvlo
    static int* d_vtable = nullptr;
    static size_t s_meta_cap = 0, s_vtable_cap = 0;
    const size_t meta_need = (size_t)3 * V * sizeof(int);
    const size_t vt_need = (size_t)V * max_num_blocks_per_seq * sizeof(int);
    // Cap env-tunable (P2): b>=8 skinny-q at long KV can exceed 512MB and
    // silently lose splits; GEMMA_SPLIT_SCRATCH_MB raises the ceiling.
    static const size_t s_scratch_cap_mb = [] {
      const char* e = getenv("GEMMA_SPLIT_SCRATCH_MB");
      return (size_t)(e ? atol(e) : 512);
    }();
    if (sq_bytes <= (s_scratch_cap_mb << 20)) {
      if (meta_need > s_meta_cap) {
        if (d_split_meta) cudaFree(d_split_meta);
        cudaMalloc(&d_split_meta, meta_need);
        s_meta_cap = meta_need;
      }
      if (vt_need > s_vtable_cap) {
        if (d_vtable) cudaFree(d_vtable);
        cudaMalloc(&d_vtable, vt_need);
        s_vtable_cap = vt_need;
      }
      if (sq_bytes > s_qo_cap) {
        if (q_scratch) cudaFree(q_scratch);
        if (o_scratch) cudaFree(o_scratch);
        cudaMalloc(&q_scratch, sq_bytes);
        cudaMalloc(&o_scratch, sq_bytes);
        s_qo_cap = sq_bytes;
      }
      if (slse_bytes > s_lse_cap) {
        if (lse_scratch) cudaFree(lse_scratch);
        cudaMalloc(&lse_scratch, slse_bytes);
        s_lse_cap = slse_bytes;
      }
      int* d_sl_v = d_split_meta;
      int* d_qoff_v = d_split_meta + V;
      int* d_kvlo_v = d_split_meta + 2 * V;
      const int* d_cu_q = (!equal_lens && cu_seqlens_q.numel() >= num_seqs + 1)
                              ? cu_seqlens_q.mutable_data_ptr<int>()
                              : nullptr;
      const int* d_sl_real = seq_lens.mutable_data_ptr<int>();
      split_params_build_kernel<<<V, 128, 0, stream>>>(
          d_sl_v, d_qoff_v, d_kvlo_v, d_vtable, d_sl_real, d_cu_q,
          block_tables_ptr, num_seqs, S, seq_len, max_num_blocks_per_seq, 64);
      dim3 sgrid(max_q_pad, V);
      split_replicate_q_kernel<Element><<<sgrid, 256, 0, stream>>>(
          reinterpret_cast<Element*>(q_scratch), query_ptr, d_cu_q, S,
          max_q_pad, seq_len, q_stride, row_elems);

      PagedPool pool = pool_proto;
      pool.page_table = d_vtable;
      Element* kc = reinterpret_cast<Element*>(key_cache_ptr);
      Element* vc = k_eq_v ? kc : reinterpret_cast<Element*>(value_cache_ptr);
      Element* qs = reinterpret_cast<Element*>(q_scratch);
      Element* os = reinterpret_cast<Element*>(o_scratch);
      const int q_off_dummy =
          max_kv_len > max_q_len ? max_kv_len - max_q_len : 0;
      bool sok;
      if (k_eq_v) {
        sok = launch_fmha_batched<512, true>(
            qs, kc, vc, os, lse_scratch, num_q_heads, gqa_group, V,
            max_q_pad, max_kv_len, q_off_dummy, row_elems, scale,
            sliding_window, nullptr, 0, s_device_id, s_sm_count, stream,
            &pool, d_sl_v, nullptr, d_qoff_v, d_kvlo_v);
      } else {
        sok = launch_fmha_batched<512>(
            qs, kc, vc, os, lse_scratch, num_q_heads, gqa_group, V,
            max_q_pad, max_kv_len, q_off_dummy, row_elems, scale,
            sliding_window, nullptr, 0, s_device_id, s_sm_count, stream,
            &pool, d_sl_v, nullptr, d_qoff_v, d_kvlo_v);
      }
      if (sok) {
        dim3 cgrid(max_q_pad, num_q_heads, num_seqs);
        split_combine_o_kernel<Element><<<cgrid, 128, 0, stream>>>(
            out_ptr, os, lse_scratch, d_cu_q, S, max_q_pad, seq_len,
            num_q_heads, head_size, q_stride, row_elems);
        return true;
      }
      // launch failure: fall through to the standard paths
    }
  }

  // ---- Uniform-batch fast path: ONE fmha launch per layer call ----
  // (equal q_len and kv_len across seqs, TMA-aligned q_len, no mm ranges).
  // The per-seq loop pays host Params re-init + serialized small kernels
  // (~4x on b=4 chunked re-prefill); the tile scheduler already supports
  // batch via grid.z and the loaders slice the batch coord.
  if (equal_lens && mm_prefix_ranges.numel() == 0 && seq_len > 0 &&
      (seq_len % kAlignment) == 0) {
    bool kv_uniform = true;
    for (int s2 = 1; s2 < num_seqs; s2++)
      kv_uniform &= (h_seq_lens[s2] == h_seq_lens[0]);
    if (kv_uniform && paged_call) {
      // Batched paged: ONE launch, block_tables is already the row-major 2D
      // page table ([seq][max_blocks]); loaders index it via the batch coord.
      // Absolute coords (no gather slice): trip_start skips out-of-window
      // tiles, so sliding layers never touch pre-window pages.
      const int kv_len_full = h_seq_lens[0];
      const int q_off_abs = kv_len_full - seq_len;
      const size_t batch_lse_bytes =
          (size_t)num_seqs * seq_len * num_q_heads * sizeof(float);
      if (batch_lse_bytes > s_lse_cap) {
        if (lse_scratch) cudaFree(lse_scratch);
        cudaMalloc(&lse_scratch, batch_lse_bytes);
        s_lse_cap = batch_lse_bytes;
      }
      PagedPool pool = pool_proto;
      pool.page_table = block_tables_ptr;
      Element* kc = reinterpret_cast<Element*>(key_cache_ptr);
      Element* vc = k_eq_v ? kc : reinterpret_cast<Element*>(value_cache_ptr);
      if (head_size == 256) {
        return launch_fmha_batched<256>(
            query_ptr, kc, vc, out_ptr, lse_scratch, num_q_heads, gqa_group,
            num_seqs, seq_len, kv_len_full, q_off_abs, q_stride, scale,
            sliding_window, nullptr, 0, s_device_id, s_sm_count, stream,
            &pool);
      } else if (k_eq_v) {
        return launch_fmha_batched<512, true>(
            query_ptr, kc, vc, out_ptr, lse_scratch, num_q_heads, gqa_group,
            num_seqs, seq_len, kv_len_full, q_off_abs, q_stride, scale,
            sliding_window, nullptr, 0, s_device_id, s_sm_count, stream,
            &pool);
      } else {
        return launch_fmha_batched<512>(
            query_ptr, kc, vc, out_ptr, lse_scratch, num_q_heads, gqa_group,
            num_seqs, seq_len, kv_len_full, q_off_abs, q_stride, scale,
            sliding_window, nullptr, 0, s_device_id, s_sm_count, stream,
            &pool);
      }
    }
    if (kv_uniform) {
      const int kv_len_full = h_seq_lens[0];
      int kv_lo = 0;
      if (sliding_window > 0) {
        kv_lo = kv_len_full - seq_len - sliding_window + 1;
        if (kv_lo < 0) kv_lo = 0;
      }
      const int kv_len_s = kv_len_full - kv_lo;
      const int q_off_s = kv_len_s - seq_len;
      const size_t seq_slot = (size_t)num_kv_heads * kv_len_s * head_size;
      const size_t batch_kv_bytes =
          (size_t)num_seqs * seq_slot * sizeof(CACHE_T);
      const size_t batch_lse_bytes =
          (size_t)num_seqs * seq_len * num_q_heads * sizeof(float);
      if (batch_kv_bytes <= ((size_t)2 << 30)) {  // scratch cap; else loop
        if (batch_kv_bytes > s_kv_cap) {
          if (k_expanded) cudaFree(k_expanded);
          cudaMalloc(&k_expanded, batch_kv_bytes);
          s_kv_cap = batch_kv_bytes;
        }
        if (!k_eq_v && batch_kv_bytes > s_v_cap) {
          if (v_expanded) cudaFree(v_expanded);
          cudaMalloc(&v_expanded, batch_kv_bytes);
          s_v_cap = batch_kv_bytes;
        }
        if (batch_lse_bytes > s_lse_cap) {
          if (lse_scratch) cudaFree(lse_scratch);
          cudaMalloc(&lse_scratch, batch_lse_bytes);
          s_lse_cap = batch_lse_bytes;
        }
        constexpr int kThreads = 256;
        const int total_elems = num_kv_heads * kv_len_s * head_size;
        const int gb = (total_elems + kThreads - 1) / kThreads;
        for (int s2 = 0; s2 < num_seqs; s2++) {
          const int* bt_s = block_tables_ptr + s2 * max_num_blocks_per_seq;
          gather_kv_expanded_kernel<CACHE_T><<<gb, kThreads, 0, stream>>>(
              k_expanded + s2 * seq_slot, key_cache_ptr, bt_s, kv_len_s,
              kv_len_s, num_kv_heads, num_kv_heads, head_size, 1, page_size,
              kv_stride_block, kv_stride_slot, kv_stride_head, kv_lo);
          if (!k_eq_v)
            gather_kv_expanded_kernel<CACHE_T><<<gb, kThreads, 0, stream>>>(
                v_expanded + s2 * seq_slot, value_cache_ptr, bt_s, kv_len_s,
                kv_len_s, num_kv_heads, num_kv_heads, head_size, 1, page_size,
                kv_stride_block, kv_stride_slot, kv_stride_head, kv_lo);
        }
        Element* kb = reinterpret_cast<Element*>(k_expanded);
        Element* vb = k_eq_v ? kb : reinterpret_cast<Element*>(v_expanded);
        if (head_size == 256) {
          return launch_fmha_batched<256>(
              query_ptr, kb, vb, out_ptr, lse_scratch, num_q_heads,
              gqa_group, num_seqs, seq_len, kv_len_s, q_off_s, q_stride,
              scale, sliding_window, nullptr, 0, s_device_id, s_sm_count,
              stream);
        } else if (k_eq_v) {
          return launch_fmha_batched<512, true>(
              query_ptr, kb, vb, out_ptr, lse_scratch, num_q_heads,
              gqa_group, num_seqs, seq_len, kv_len_s, q_off_s, q_stride,
              scale, sliding_window, nullptr, 0, s_device_id, s_sm_count,
              stream);
        } else {
          return launch_fmha_batched<512>(
              query_ptr, kb, vb, out_ptr, lse_scratch, num_q_heads,
              gqa_group, num_seqs, seq_len, kv_len_s, q_off_s, q_stride,
              scale, sliding_window, nullptr, 0, s_device_id, s_sm_count,
              stream);
        }
      }
    }
  }

  // ---- Varlen batched paged path: ONE launch per layer for STAGGERED
  // short-KV steps (ragged q and/or kv). Q is packed into padded
  // [seq][max_q_pad] scratch; the kernel derives per-seq seq_k/q_offset from
  // d_seq_lens + d_cu_seqlens_q (per-CTA problem override); padded rows
  // compute garbage into scratch rows never scattered back. Replaces the
  // per-seq loop's ~num_seqs x 48 launch storm (26.6us/launch measured).
  if (use_paged && paged_call && num_seqs > 1 &&
      mm_prefix_ranges.numel() == 0 &&
      cu_seqlens_q.numel() >= num_seqs + 1 && seq_lens.numel() >= num_seqs) {
    const int max_q_pad =
        (max_q_len + kAlignment - 1) / kAlignment * kAlignment;
    const int row_elems = num_q_heads * head_size;
    // +128 rows: the Q TMA loads whole M-tiles (up to 128 rows) per seq;
    // at max_q_pad < TileM the LAST seq's tile read overruns its region
    // (reads only; stores are seq_q-masked). Slack keeps it in-bounds.
    const size_t vq_bytes =
        (size_t)(num_seqs * max_q_pad + 128) * row_elems * sizeof(T);
    const size_t vlse_bytes =
        (size_t)num_seqs * max_q_pad * num_q_heads * sizeof(float);
    if (vq_bytes <= ((size_t)512 << 20)) {  // scratch cap; else per-seq loop
      if (vq_bytes > s_qo_cap) {
        if (q_scratch) cudaFree(q_scratch);
        if (o_scratch) cudaFree(o_scratch);
        cudaMalloc(&q_scratch, vq_bytes);
        cudaMalloc(&o_scratch, vq_bytes);
        s_qo_cap = vq_bytes;
      }
      if (vlse_bytes > s_lse_cap) {
        if (lse_scratch) cudaFree(lse_scratch);
        cudaMalloc(&lse_scratch, vlse_bytes);
        s_lse_cap = vlse_bytes;
      }
      const int* d_cu_q = cu_seqlens_q.mutable_data_ptr<int>();
      const int* d_sl = seq_lens.mutable_data_ptr<int>();
      dim3 vgrid(max_q_pad, num_seqs);
      varlen_pad_q_kernel<Element><<<vgrid, 256, 0, stream>>>(
          reinterpret_cast<Element*>(q_scratch), query_ptr, d_cu_q,
          max_q_pad, q_stride, row_elems);

      PagedPool pool = pool_proto;
      pool.page_table = block_tables_ptr;  // 2D [seq][max_blocks]
      Element* kc = reinterpret_cast<Element*>(key_cache_ptr);
      Element* vc = k_eq_v ? kc : reinterpret_cast<Element*>(value_cache_ptr);
      Element* qs = reinterpret_cast<Element*>(q_scratch);
      Element* os = reinterpret_cast<Element*>(o_scratch);
      // Baked seq_k/q_offset are per-CTA overridden; scalars only size the
      // (unused in paged mode) contiguous-KV stride math.
      const int q_off_dummy = max_kv_len > max_q_len ? max_kv_len - max_q_len : 0;
      bool vok;
      if (head_size == 256) {
        vok = launch_fmha_batched<256>(
            qs, kc, vc, os, lse_scratch, num_q_heads, gqa_group, num_seqs,
            max_q_pad, max_kv_len, q_off_dummy, row_elems, scale,
            sliding_window, nullptr, 0, s_device_id, s_sm_count, stream,
            &pool, d_sl, d_cu_q);
      } else if (k_eq_v) {
        vok = launch_fmha_batched<512, true>(
            qs, kc, vc, os, lse_scratch, num_q_heads, gqa_group, num_seqs,
            max_q_pad, max_kv_len, q_off_dummy, row_elems, scale,
            sliding_window, nullptr, 0, s_device_id, s_sm_count, stream,
            &pool, d_sl, d_cu_q);
      } else {
        vok = launch_fmha_batched<512>(
            qs, kc, vc, os, lse_scratch, num_q_heads, gqa_group, num_seqs,
            max_q_pad, max_kv_len, q_off_dummy, row_elems, scale,
            sliding_window, nullptr, 0, s_device_id, s_sm_count, stream,
            &pool, d_sl, d_cu_q);
      }
      if (vok) {
        varlen_scatter_o_kernel<Element><<<vgrid, 256, 0, stream>>>(
            out_ptr, os, d_cu_q, max_q_pad, q_stride, row_elems);
        return true;
      }
      // fall through to the per-seq loop on launch failure
    }
  }

  for (int s = 0; s < num_seqs && ok; s++) {
    const int token_offset = equal_lens ? s * seq_len
                                        : h_cu_seqlens_q[s];
    const int seq_len_s = equal_lens ? seq_len
                                     : (h_cu_seqlens_q[s + 1] - h_cu_seqlens_q[s]);
    if (seq_len_s == 0) continue;
    const int kv_len_full = h_seq_lens[s];
    // Sliding layers only attend [q_min - sw + 1, kv_end): gather just that
    // window slice. Chunked prefill previously re-gathered the WHOLE growing
    // context every chunk per layer (~186ms/step at L=16k vs FA4's paged
    // reads). kv_lo is the absolute token of gathered index 0; the kernel's
    // masks are shift-invariant (q_offset shifted consistently).
    int kv_lo = 0;
    if (!paged_call && sliding_window > 0) {
      kv_lo = kv_len_full - seq_len_s - sliding_window + 1;
      if (kv_lo < 0) kv_lo = 0;
    }
    const int kv_len_s = kv_len_full - kv_lo;  // paged: full absolute extent
    const int q_off_s = kv_len_s - seq_len_s;  // context length
    const int padded_sl_s = (seq_len_s + kAlignment - 1) / kAlignment * kAlignment;
    const bool needs_pad_s = (padded_sl_s != seq_len_s);
    const int* seq_block_table =
        block_tables_ptr + s * max_num_blocks_per_seq;

    Element* k;
    Element* v;
    PagedPool pool = pool_proto;
    if (paged_call) {
      // No gather: the kernel reads pages straight from the cache pool.
      // Sliding layers skip pre-window tiles via trip_start, so the paged
      // path never touches pages the gather-slice used to exclude.
      pool.page_table = seq_block_table;
      k = reinterpret_cast<Element*>(key_cache_ptr);
      v = k_eq_v ? k : reinterpret_cast<Element*>(value_cache_ptr);
    } else {
      // Gather the kv range DENSE PER KV HEAD ([kv_len, hd] x num_kv_heads —
      // no GQA expansion; loaders map q-head -> kv-head via contig_gqa_group).
      const int total_elems = num_kv_heads * kv_len_s * head_size;
      constexpr int kThreads = 256;
      const int gather_blocks = (total_elems + kThreads - 1) / kThreads;

      gather_kv_expanded_kernel<CACHE_T><<<gather_blocks, kThreads, 0, stream>>>(
          k_expanded, key_cache_ptr, seq_block_table, kv_len_s, kv_len_s,
          num_kv_heads, num_kv_heads, head_size, /*gqa_group=*/1, page_size,
          kv_stride_block, kv_stride_slot, kv_stride_head, kv_lo);

      if (!k_eq_v) {
        gather_kv_expanded_kernel<CACHE_T>
            <<<gather_blocks, kThreads, 0, stream>>>(
                v_expanded, value_cache_ptr, seq_block_table, kv_len_s,
                kv_len_s, num_kv_heads, num_kv_heads, head_size,
                /*gqa_group=*/1, page_size,
                kv_stride_block, kv_stride_slot, kv_stride_head, kv_lo);
      }
      k = reinterpret_cast<Element*>(k_expanded);
      v = k_eq_v ? k : reinterpret_cast<Element*>(v_expanded);
    }

    Element* q_src = query_ptr + token_offset * q_stride;
    Element* o_dst = out_ptr + token_offset * q_stride;

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

    const PagedPool* pp = paged_call ? &pool : nullptr;
    if (head_size == 256) {
      ok = launch_fmha_batched<256>(q_fmha, k, v, o_fmha, lse_scratch,
                                    num_q_heads, gqa_group, 1, padded_sl_s,
                                    kv_len_s, q_off_s, fmha_q_stride, scale,
                                    sliding_window, seq_mm_ranges, seq_max_mm,
                                    s_device_id, s_sm_count, stream, pp);
    } else if (k_eq_v) {
      // Gemma global layers: V == K -> single-slot pipeline, no V TMA loads.
      ok = launch_fmha_batched<512, true>(q_fmha, k, v, o_fmha, lse_scratch,
                                    num_q_heads, gqa_group, 1, padded_sl_s,
                                    kv_len_s, q_off_s, fmha_q_stride, scale,
                                    sliding_window, seq_mm_ranges, seq_max_mm,
                                    s_device_id, s_sm_count, stream, pp);
    } else {
      ok = launch_fmha_batched<512>(q_fmha, k, v, o_fmha, lse_scratch,
                                    num_q_heads, gqa_group, 1, padded_sl_s,
                                    kv_len_s, q_off_s, fmha_q_stride, scale,
                                    sliding_window, seq_mm_ranges, seq_max_mm,
                                    s_device_id, s_sm_count, stream, pp);
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
    torch::stable::Tensor&, bool, torch::stable::Tensor&, torch::stable::Tensor&, torch::stable::Tensor&);
