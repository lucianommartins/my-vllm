/*
 * SM90 prefill launcher — CUTLASS FMHA integration for Gemma4.
 *
 * Phase 1: Compile and validate the CUTLASS FMHA kernel with contiguous
 * tensors. Paged KV cache adaptation is phase 2.
 *
 * The kernel is opt-in via GEMMA_SM90_PREFILL=1 for development.
 * Default falls through to SM80 v2.
 */
#include "../torch_utils.h"
#include "gemma_prefill_attention_sm90.cuh"
#include "../../cuda_compat.h"

#include <cuda_runtime.h>
#include <cstdlib>

template <typename T, typename CACHE_T>
bool gemma_prefill_sm90_launcher(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int num_kv_heads, float scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, torch::stable::Tensor& cu_seqlens_q,
    int max_q_len, int page_size, bool k_eq_v, int sliding_window,
    torch::stable::Tensor& mm_prefix_ranges, bool non_causal,
    torch::stable::Tensor& lse_out) {

  // SM90 CUTLASS FMHA kernel — opt-in for development.
  // The CUTLASS FMHA requires contiguous Q/K/V tensors (no paged KV yet).
  // Once validated, paged KV support will be added.
  static const bool forced = []() {
    const char* e = getenv("GEMMA_SM90_PREFILL");
    return e != nullptr && e[0] == '1';
  }();
  if (!forced) return false;

  // TODO: Implement CUTLASS FMHA kernel launch here.
  // For now, fall through to SM80.
  return false;
}

template bool gemma_prefill_sm90_launcher<__nv_bfloat16, __nv_bfloat16>(
    torch::stable::Tensor&, torch::stable::Tensor&,
    torch::stable::Tensor&, torch::stable::Tensor&,
    int, float, torch::stable::Tensor&,
    torch::stable::Tensor&, torch::stable::Tensor&,
    int, int, bool, int,
    torch::stable::Tensor&, bool,
    torch::stable::Tensor&);
