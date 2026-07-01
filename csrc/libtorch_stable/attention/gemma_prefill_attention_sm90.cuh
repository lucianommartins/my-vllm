/*
 * Gemma4-optimized prefill attention — SM90 (Hopper) kernel.
 *
 * Adapts CUTLASS example 88 (Hopper FMHA) for Gemma4:
 *   - wgmma (SS for QK, RS for PV) + TMA + warp specialization
 *   - Built on CUTLASS 3.x CollectiveBuilder + PipelineTmaAsync
 *   - k_eq_v: V = K (L2 cache serves V from K lines)
 *   - Causal masking via CausalFusion
 *   - Head-chunked PV for hd=512 (O half in smem, spill-free)
 *   - Paged KV cache: TODO (initial version gathers into contiguous buffers)
 */
#pragma once

#include "fmha_sm90/fmha_kernel_builder.hpp"
#include "fmha_sm90/fmha_fusion.hpp"
#include "fmha_sm90/fmha_options.hpp"

namespace vllm {
namespace gemma_prefill {
namespace sm90 {

using GemmaCausalFusion = cutlass::fmha::collective::SlidingWindowCausalFusion;

template <int HeadDim>
struct GemmaFmhaTypes {
  using Element = cutlass::bfloat16_t;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;

  // hd=256: Shape<128, 64, 256> — 2 cooperative WGs
  // hd=512: Shape<64, 64, 512> — 1 WG, head-chunked PV (O half in smem)
  using TileShape = cute::conditional_t<
      HeadDim == 256,
      cute::Shape<cute::_128, cute::_64, cute::_256>,
      cute::Shape<cute::_64, cute::_64, cute::_512>>;

  using StrideQ = cute::tuple<int, cute::_1, cute::tuple<int, int>>;
  using StrideK = cute::tuple<int, cute::_1, cute::tuple<int, int>>;
  using StrideV = cute::tuple<int, cute::_1, cute::tuple<int, int>>;

  using DispatchPolicy = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

  // hd=256: 2 WGs cooperative (M-split), 5 KV stages
  // hd=512: 2 WGs split-D (FA3-style), 2 KV stages
  //   WG1: QK+softmax+PV_hi, WG2: PV_lo only
  //   SS-mode PV: P written to smem, both WGs read via GMMA SS descriptors
  static constexpr int kNumMmaWGs = 2;
  static constexpr int kStagesKV = (HeadDim <= 256) ? 5 : 2;
  static constexpr bool kSplitDPV = (HeadDim > 256);

  using Kernel = typename cutlass::fmha::kernel::FmhaBuilder<
      Element, ElementAccumulatorQK, ElementAccumulatorPV,
      TileShape, StrideQ, StrideK, StrideV,
      GemmaCausalFusion, DispatchPolicy,
      cutlass::fmha::kernel::Option<
          cutlass::fmha::kernel::Tag::kNumMmaWarpGroups,
          cute::Int<kNumMmaWGs>>,
      cutlass::fmha::kernel::Option<
          cutlass::fmha::kernel::Tag::kStagesKV,
          cute::Int<kStagesKV>>,
      cutlass::fmha::kernel::Option<
          cutlass::fmha::kernel::Tag::kSplitDPV,
          cute::bool_constant<kSplitDPV>>>::Kernel;
};

}  // namespace sm90
}  // namespace gemma_prefill
}  // namespace vllm
