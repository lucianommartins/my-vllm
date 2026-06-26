/*
 * SM90 decode launcher for Gemma4-optimized paged attention.
 *
 * P1b: wider-tile SIMT decode (BLOCK_N=32) exploiting H100's 228KB smem.
 * Dispatches for bf16 k_eq_v hd=512 configs; non-k_eq_v falls to SM80.
 */
#include "../torch_utils.h"
#include "gemma_paged_attention_sm90.cuh"
#include "../../attention/attention_dtypes.h"
#include "../../cuda_compat.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <type_traits>

// SM90 wider tile: BLOCK_N=32 (vs 16 on A100).
// 2-stage smem = 2 × 32 × 512 × 2B = 64KB → 3 CTAs/SM in 228KB.
static constexpr int SM90_BN = 32;

// Occupancy targets for H100. With 64KB smem/CTA, 228KB allows 3 CTAs.
// The register budget at MIN_CTA=3 with 8 warps (256 threads) is
// 65536/(3*256) = 85 regs/thread. EPL=16 (hd512) → q_reg(16) + o_reg(16)
// + s[TPZ] + control ~ 50-60 regs. Should fit without spill.
static constexpr int SM90_MINCTA_G2 = 3;
static constexpr int SM90_MINCTA_G4 = 3;
static constexpr int SM90_MINCTA_G8 = 3;
static constexpr int SM90_MINCTA_G16 = 2;

// Split-reduce kernel (arch-independent, reused from SM80).
namespace vllm { namespace gemma {
template <typename scalar_t, int HEAD_SIZE>
__global__ void gemma_split_reduce_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ tmp_out,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    int num_splits, int max_parts,
    float* __restrict__ lse_out);
}}

#define LAUNCH_SM90(HEAD, BN, BDY, BDZ, USW, SPLITB, MINCTA)                 \
  do {                                                                        \
    const dim3 sgrid = (SPLITB) ? dim3(num_kv_heads, num_seqs, num_splits)    \
                                : dim3(num_kv_heads, num_seqs);                \
    size_t ktile = (size_t)2 * (BN) * (HEAD) * sizeof(CACHE_T);               \
    size_t comb = (size_t)((BDY) * (BDZ)) * ((HEAD) + 2) * sizeof(float);     \
    size_t smem = ktile > comb ? ktile : comb;                                \
    auto sk = vllm::gemma::sm90::gemma_decode_sm90_kernel<                    \
        T, CACHE_T, HEAD, BN, BDY, BDZ, USW, SPLITB, MINCTA>;                \
    if (smem > 48 * 1024)                                                     \
      cudaFuncSetAttribute(                                                   \
          sk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);            \
    cudaFuncSetAttribute(                                                     \
        sk, cudaFuncAttributePreferredSharedMemoryCarveout, 100);           \
    T* sout = (SPLITB) ? tmp_out_ptr : out_ptr;                              \
    sk<<<sgrid, (BDY) * (BDZ) * WARP_SIZE, smem, stream>>>(                  \
        sout, exp_sums_ptr, max_logits_ptr, query_ptr, k_cache_ptr,         \
        num_kv_heads, scale, block_tables_ptr, seq_lens_ptr,                 \
        max_num_blocks_per_seq, BLOCK_SIZE, q_stride, kv_stride_block,       \
        kv_stride_slot, kv_stride_head, sliding_window, num_splits,          \
        max_parts, (SPLITB) ? nullptr : lse_out_ptr,                         \
        full_sink, full_window);                                              \
    if (SPLITB) {                                                            \
      const dim3 cg(num_kv_heads * (BDY), num_seqs);                         \
      vllm::gemma::gemma_split_reduce_kernel<T, HEAD>                        \
          <<<cg, WARP_SIZE, 0, stream>>>(out_ptr, tmp_out_ptr,              \
                                          exp_sums_ptr, max_logits_ptr,     \
                                          num_splits, max_parts,            \
                                          lse_out_ptr);                     \
    }                                                                       \
  } while (0)

#define SM90_SB(HEAD, BN, BDY, BDZ, USW, MINCTA)                             \
  do {                                                                        \
    if (num_splits > 1) {                                                     \
      LAUNCH_SM90(HEAD, BN, BDY, BDZ, USW, true, MINCTA);                    \
    } else {                                                                  \
      LAUNCH_SM90(HEAD, BN, BDY, BDZ, USW, false, MINCTA);                   \
    }                                                                         \
  } while (0)

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

  // P1b wider-tile (BN=32) did not beat the SM80 SIMT (BN=16, 4 CTAs/SM).
  // The wider tile trades occupancy (3 vs 4 CTAs) for fewer iterations, net
  // flat. SM80 SIMT is already near-optimal for scalar-FMA decode.
  // Decode improvement requires TMA descriptor-based loads (future work).
  // Disabled until then — fall through to SM80.
  return false;

  constexpr bool DTYPE_OK =
      std::is_same<T, __nv_bfloat16>::value &&
      (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto);
  if constexpr (!DTYPE_OK) return false;

  if (!k_eq_v) return false;
  if (selected_tiles.numel() > 0) return false;
  if (actual_head_size != 512) return false;

  const int num_seqs = query.size(0);
  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int q_stride = query.stride(0);
  const int max_num_blocks_per_seq = block_tables.size(1);
  const int64_t kv_stride_block = key_cache.stride(0);
  const int64_t kv_stride_slot = key_cache.stride(1);
  const int64_t kv_stride_head = key_cache.stride(2);
  const int gqa_group = (num_kv_heads > 0) ? (num_q_heads / num_kv_heads) : 0;
  const int max_parts = static_cast<int>(exp_sums.size(2));
  const bool use_sw = (sliding_window > 0);

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* k_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  int* block_tables_ptr = block_tables.mutable_data_ptr<int>();
  int* seq_lens_ptr = seq_lens.mutable_data_ptr<int>();
  float* lse_out_ptr = (lse_out.numel() > 0)
                           ? reinterpret_cast<float*>(lse_out.data_ptr())
                           : nullptr;

  static const int full_sink = []() {
    const char* e = getenv("GEMMA_FULL_SINK");
    return e != nullptr ? atoi(e) : 0;
  }();
  static const int full_window = []() {
    const char* e = getenv("GEMMA_FULL_WINDOW");
    return e != nullptr ? atoi(e) : 0;
  }();

  // num_splits heuristic (H100: 132 SMs, target 3 CTA/SM).
  static constexpr int DECODE_CTA_PER_SM = 3;
  static constexpr int DECODE_WAVES = 6;
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
    const int eff_seq = (sliding_window > 0 && sliding_window < max_seq_len)
                            ? sliding_window : max_seq_len;
    const int max_seq_blocks = (eff_seq + SM90_BN - 1) / SM90_BN;
    const int desired_ctas = DECODE_WAVES * DECODE_CTA_PER_SM * num_sms;
    int target = (total_ctas > 0)
                     ? (desired_ctas + total_ctas - 1) / total_ctas : 1;
    num_splits = target < 1 ? 1 : target;
    if (num_splits > max_seq_blocks) num_splits = max_seq_blocks;
    if (num_splits > max_parts) num_splits = max_parts;
    if (max_seq_blocks <= 4) num_splits = 1;
    if (num_splits < 1) num_splits = 1;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  // BDY*BDZ ~= 8 warps. BLOCK_N=32 → TPZ=32/BDZ.
  // BDZ must divide BLOCK_N evenly.
  if (!use_sw) {
    switch (gqa_group) {
      case 2:  // BDY=2, BDZ=4 → 8 warps, TPZ=8
        SM90_SB(512, SM90_BN, 2, 4, false, SM90_MINCTA_G2); break;
      case 4:  // BDY=4, BDZ=2 → 8 warps, TPZ=16
        SM90_SB(512, SM90_BN, 4, 2, false, SM90_MINCTA_G4); break;
      case 8:  // BDY=8, BDZ=1 → 8 warps, TPZ=32
        SM90_SB(512, SM90_BN, 8, 1, false, SM90_MINCTA_G8); break;
      case 16: // BDY=16, BDZ=1 → 16 warps, TPZ=32
        SM90_SB(512, SM90_BN, 16, 1, false, SM90_MINCTA_G16); break;
      default: return false;
    }
  } else {
    switch (gqa_group) {
      case 2:
        SM90_SB(512, SM90_BN, 2, 4, true, SM90_MINCTA_G2); break;
      case 4:
        SM90_SB(512, SM90_BN, 4, 2, true, SM90_MINCTA_G4); break;
      case 8:
        SM90_SB(512, SM90_BN, 8, 1, true, SM90_MINCTA_G8); break;
      case 16:
        SM90_SB(512, SM90_BN, 16, 1, true, SM90_MINCTA_G16); break;
      default: return false;
    }
  }

  return true;
}

#undef LAUNCH_SM90
#undef SM90_SB

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
