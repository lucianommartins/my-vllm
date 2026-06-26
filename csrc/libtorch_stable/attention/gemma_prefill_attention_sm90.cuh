/*
 * Gemma4-optimized prefill attention — SM90 (Hopper) kernel.
 *
 * Adapts CUTLASS example 88 (Hopper FMHA) for Gemma4:
 *   - wgmma (SS for QK, RS for PV) + TMA + warp specialization
 *   - Built on CUTLASS 3.x CollectiveBuilder + PipelineTmaAsync
 *   - k_eq_v: V = K (shared smem buffer)
 *   - Causal + sliding window masking via Fusion
 *   - Paged KV cache: TODO (initial version uses contiguous tensors)
 *
 * This header includes the forked CUTLASS FMHA and provides the
 * Gemma4-specific fusion and kernel instantiation types.
 */
#pragma once

#include "fmha_sm90/fmha_kernel_builder.hpp"
#include "fmha_sm90/fmha_fusion.hpp"

namespace vllm {
namespace gemma_prefill {
namespace sm90 {

// Gemma4 causal attention fusion (no dropout, scale=1.0, causal mask).
// Adapts the CUTLASS FMHA CausalFusion for Gemma4's specific properties.
using GemmaCausalFusion = cutlass::fmha::collective::CausalFusion;

// Kernel type aliases for different head dimensions.
// TileShape = (BlockQO, BlockKV, HeadDim)
// Start with hd=256 (known working in CUTLASS example).
// hd=512 requires register pressure analysis.

template <int HeadDim>
struct GemmaFmhaTypes {
  using Element = cutlass::bfloat16_t;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;

  // Tile shape selection based on head dim
  // hd=256: Shape<128, 64, 256> (from CUTLASS example)
  // hd=512: Shape<64, 64, 512> (smaller M to manage register pressure)
  using TileShape = cute::conditional_t<
      HeadDim == 256,
      cute::Shape<cute::_128, cute::_64, cute::_256>,
      cute::Shape<cute::_64, cute::_64, cute::_512>>;

  // Contiguous KV strides (paged KV adaptation is TODO)
  using StrideQ = cute::tuple<int, cute::_1, cute::tuple<int, int>>;
  using StrideK = cute::tuple<int, cute::_1, cute::tuple<int, int>>;
  using StrideV = cute::tuple<int, cute::_1, cute::tuple<int, int>>;

  using DispatchPolicy = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

  using Kernel = typename cutlass::fmha::kernel::FmhaBuilder<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShape, StrideQ, StrideK, StrideV,
      GemmaCausalFusion, DispatchPolicy>::Kernel;
};

}  // namespace sm90
}  // namespace gemma_prefill
}  // namespace vllm
