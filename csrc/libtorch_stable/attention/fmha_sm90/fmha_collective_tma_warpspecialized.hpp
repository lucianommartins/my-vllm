/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "fmha_common.hpp"
#include "fmha_collective_load.hpp"
#include "fmha_collective_softmax.hpp"
#include "fmha_options.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

using cutlass::fmha::kernel::Tag;
using cutlass::fmha::kernel::find_option_t;

template<
  class Element_,
  class ElementAccumulatorQK_,
  class ElementAccumulatorPV_,
  class TileShape_, // SeqQ, SeqKV, Head
  class LayoutQ_, class LayoutK_, class LayoutV_,  // SeqX, Head, (Batches)
  class Fusion,
  class... Options
>
struct FmhaMainloopTmaWarpSpecialized {

  using Element = Element_;
  using ElementAccumulatorQK = ElementAccumulatorQK_;
  using ElementAccumulatorPV = ElementAccumulatorPV_;
  using TileShape = TileShape_;

  using LayoutQ = LayoutQ_;
  using LayoutK = LayoutK_;
  using LayoutV = LayoutV_;

  // Options
  static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, false_type, Options...>::value;
  static constexpr bool kIsMainloopLocked = find_option_t<Tag::kIsMainloopLocked, false_type, Options...>::value;
  static constexpr bool kHeadChunkedPV = find_option_t<Tag::kHeadChunkedPV, false_type, Options...>::value;
  static constexpr bool kSplitDPV = find_option_t<Tag::kSplitDPV, false_type, Options...>::value;

  static constexpr int NumLoadWarpGroups = 1;
  static constexpr int NumMmaWarpGroups = find_option_t<Tag::kNumMmaWarpGroups, Int<2>, Options...>::value;
  static constexpr int NumQKWarpGroups = kSplitDPV ? 1 : NumMmaWarpGroups;
  static constexpr int StageCount = find_option_t<Tag::kStagesKV, Int<5>, Options...>::value;
  static constexpr int StageCountQ = kSplitDPV ? 1 : find_option_t<Tag::kStagesQ, Int<NumMmaWarpGroups>, Options...>::value;

  static const int kOuterLoads = 1;
  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using Stages = cutlass::gemm::collective::StageCount<StageCount>;
  using ClusterShape = Shape<_1, _1, _1>;
  static_assert(kSplitDPV || StagesQ::value >= NumMmaWarpGroups);
  static_assert(!kSplitDPV || StagesQ::value >= 1);
  static_assert(Stages::value >= 2);

  // 16B alignment lets us use TMA
  static constexpr int Alignment = 16 / sizeof(Element);

  using TileShapeQK = Shape<
    decltype(tuple_element_t<0, TileShape>{} / Int<NumQKWarpGroups>{}),
    tuple_element_t<1, TileShape>,
    tuple_element_t<2, TileShape>>;

  using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));

  // For head-chunked PV: half-D tile shape for the PV accumulation
  static constexpr int HeadDimPV = kHeadChunkedPV ? (int(get<2>(TileShape{})) / 2) : int(get<2>(TileShape{}));
  using TileShapePV_Eff = cute::conditional_t<kHeadChunkedPV,
      Shape<decltype(get<0>(TileShapeQK{})), Int<HeadDimPV>, decltype(get<1>(TileShapeQK{}))>,
      TileShapePV>;

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, LayoutQ, Alignment,
      Element, LayoutK, Alignment,
      ElementAccumulatorQK,
      TileShapeQK, ClusterShape, Stages,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  // PV collective: ALWAYS uses full-D TileShapePV for smem layout and TMA
  // (K/V union requires same stage sizes to avoid smem overlap).
  // Head-chunked PV splits the PV compute at the atom level, not the collective.
  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Element, LayoutK, Alignment,
      Element, decltype(select<1,0,2>(LayoutV{})), Alignment,
      ElementAccumulatorPV,
      TileShapePV, ClusterShape, Stages,
      cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;

  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  static constexpr int kHeadDim = int(get<2>(TileShape{}));

  // split-D: 2-WG SS mode (256 threads, P from smem, each WG handles 256 V cols)
  // non-split: RS mode (P in registers)
  using TiledMmaPV = cute::conditional_t<kSplitDPV,
    decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<
            Element, Element, ElementAccumulatorPV,
            TileShapePV,
            cute::GMMA::Major::K, cute::GMMA::Major::MN>(),
        Layout<Shape<_1, Int<kHeadDim / 256>, _1>>{})),
    decltype(convert_to_gmma_rs(typename CollectiveMmaPV::TiledMma{}))>;

  using SmemLayoutQ = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StagesQ::value>{}));
  using SmemLayoutK = typename CollectiveMmaQK::SmemLayoutB;
  using SmemLayoutV = typename CollectiveMmaPV::SmemLayoutB;

  using MainloopPipeline = cutlass::PipelineTmaAsync<Stages::value>;
  using MainloopPipelineQ = cutlass::PipelineTmaAsync<StagesQ::value>;

  using PipelineState  = typename cutlass::PipelineState<MainloopPipeline::Stages>;
  using PipelineStateQ  = typename cutlass::PipelineState<MainloopPipelineQ::Stages>;

  static constexpr int kInnerLoadBytes = size(SmemLayoutK{}(_,_,_0{})) * sizeof(Element);
  static constexpr int kOuterLoadBytes = size(SmemLayoutQ{}(_,_,_0{})) * sizeof(Element);

  static constexpr int MPerWG = int(get<0>(TileShapeQK{}));
  static constexpr int HeadDimFull = int(get<2>(TileShape{}));
  static constexpr int HeadDimHalf = HeadDimFull / 2;

  // For head-chunked PV (legacy): half-D output, direct gmem write from compute
  static constexpr int kSmemOHalfElems =
      kHeadChunkedPV ? (MPerWG * HeadDimHalf) : 0;

  // Split-D: P smem (correct Swizzle<3,4,3> from CollectiveMmaPV) + scale smem
  using SmemLayoutP = typename CollectiveMmaPV::SmemLayoutA;
  static constexpr int kSmemPElems = kSplitDPV ? cute::cosize_v<SmemLayoutP> : 0;
  static constexpr int kSmemScaleElems = kSplitDPV ? MPerWG : 0;
  static constexpr int kNumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t kBarrierPFull = 10;
  static constexpr uint32_t kBarrierPEmpty = 11;

  using TileShapeOut = TileShapePV;
  using TiledMmaOut = TiledMmaPV;
  using ElementOut = ElementAccumulatorPV;

  struct SharedStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    };
    cute::array_aligned<Element, kSmemOHalfElems> smem_o_half;
    cute::array_aligned<Element, kSmemPElems> smem_p;
    cute::array_aligned<float, kSmemScaleElems> smem_scale;
  };

  struct Arguments {
    const Element* ptr_Q;
    LayoutQ dQ;
    const Element* ptr_K;
    LayoutK dK;
    const Element* ptr_V;
    LayoutV dV;
    float scale;
    Element* ptr_O;
    LayoutQ dO;
    ElementAccumulatorPV* ptr_LSE;
    const int* mm_prefix_ranges;
    int max_mm_ranges;
  };

  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
    TMA_V tma_load_v_hi;

    float scale_softmax;
    float scale_softmax_log2;
    float rp_dropout;
    Element* ptr_O;
    LayoutQ dO;
    ElementAccumulatorPV* ptr_LSE;
    const int* mm_prefix_ranges;
    int max_mm_ranges;
    const Element* ptr_V;
    LayoutV dV;
  };

  using LoadQ = cutlass::fmha::collective::CollectiveLoadTma<
    cutlass::fmha::collective::LoadKind::kQ,
    MainloopPipelineQ,
    Element,
    SmemLayoutQ,
    TMA_Q
  >;

  using LoadK = cutlass::fmha::collective::CollectiveLoadTma<
    cutlass::fmha::collective::LoadKind::kK,
    MainloopPipeline,
    Element,
    SmemLayoutK,
    TMA_K
  >;

  using LoadV = cutlass::fmha::collective::CollectiveLoadTma<
    cutlass::fmha::collective::LoadKind::kV,
    MainloopPipeline,
    Element,
    SmemLayoutV,
    TMA_V
  >;

  // For non-chunked/non-split: QK and PV MMA sizes must match.
  static_assert(kHeadChunkedPV || kSplitDPV || size(typename CollectiveMmaQK::TiledMma{}) == size(typename CollectiveMmaPV::TiledMma{}));

  template<class ProblemShape>
  static bool can_implement(ProblemShape const& problem_size, Arguments const& args) {
    return true
      && (get<4>(problem_size) <= get<2>(TileShape{}))
      && ((get<4>(problem_size) % Alignment) == 0)
      && ((get<2>(problem_size) % Alignment) == 0)
    ;
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_size, Arguments const& args, void* workspace) {

    auto problem_shape_qk = make_shape(get<2>(problem_size), get<3>(problem_size), get<4>(problem_size), make_shape(get<0>(problem_size), get<1>(problem_size)));
    auto params_qk = CollectiveMmaQK::to_underlying_arguments(problem_shape_qk,
        typename CollectiveMmaQK::Arguments {
            args.ptr_Q, args.dQ,
            args.ptr_K, args.dK,
        }, /*workspace=*/ nullptr);

    auto problem_shape_pv = select<0,2,1,3>(problem_shape_qk);

    auto params_pv = CollectiveMmaPV::to_underlying_arguments(problem_shape_pv,
        typename CollectiveMmaPV::Arguments {
            args.ptr_K, args.dK,  // never used, dummy
            args.ptr_V, select<1,0,2>(args.dV),
        }, /*workspace=*/ nullptr);

    // V_hi TMA: only for head-chunked PV, otherwise reuse V_lo descriptor
    TMA_V tma_v_hi = params_pv.tma_load_b;
    if constexpr (kHeadChunkedPV) {
      auto params_pv_hi = CollectiveMmaPV::to_underlying_arguments(problem_shape_pv,
          typename CollectiveMmaPV::Arguments {
              args.ptr_K, args.dK,
              args.ptr_V + HeadDimPV, select<1,0,2>(args.dV),
          }, /*workspace=*/ nullptr);
      tma_v_hi = params_pv_hi.tma_load_b;
    }

    return Params{
        params_qk.tma_load_a,
        params_qk.tma_load_b,
        params_pv.tma_load_b,
        tma_v_hi,
        args.scale,
        args.scale * (float)std::log2(std::exp(1.0)),
        1.0f,
        args.ptr_O,
        args.dO,
        args.ptr_LSE,
        args.mm_prefix_ranges,
        args.max_mm_ranges,
        args.ptr_V,
        args.dV
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
    if constexpr (kHeadChunkedPV) {
      cute::prefetch_tma_descriptor(params.tma_load_v_hi.get_tma_descriptor());
    }
  }

  template<bool kLoadQ, class BlkCoord, class ProblemShape, class LoadWarpBarrier>
  CUTLASS_DEVICE void
  load_kv_maybe_q(
      int block_rank_in_cluster,
      BlkCoord const& blk_coord, Params const& params, ProblemShape const& problem_size,
      MainloopPipeline& pipeline, PipelineState& smem_pipe_write, 
      MainloopPipelineQ& pipeline_q, PipelineStateQ& smem_pipe_write_q, 
      SharedStorage& storage,
      LoadWarpBarrier& load_warp_barrier, bool do_barrier)
  {
    int fusion_tile_count = Fusion{}.get_trip_count(blk_coord, TileShape{}, problem_size);

    int lane_predicate = cute::elect_one_sync();

    uint16_t mcast_mask_b = 0;

    if (lane_predicate == 1) {
      if constexpr (cute::is_same_v<typename CollectiveMmaQK::GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,_0{},Int<0>{}));
        }
      }
    }

    auto q_tile_iter = cute::make_coord_iterator(Int<NumQKWarpGroups>{});
    [[maybe_unused]] int q_tile_count = NumQKWarpGroups;

    auto k_tile_iter = cute::make_coord_iterator(fusion_tile_count);
    int k_tile_count = 2 * fusion_tile_count;

    LoadQ load_q{params.tma_load_q, pipeline_q, storage.smem_q};
    auto load_state_q = load_q.init_state(_0{}, problem_size, TileShapeQK{}, blk_coord, NumQKWarpGroups);

    LoadK load_k{params.tma_load_k, pipeline, storage.smem_k};
    auto load_state_k = load_k.init_state(block_rank_in_cluster, problem_size, TileShapeQK{}, blk_coord, fusion_tile_count);

    LoadV load_v{params.tma_load_v, pipeline, storage.smem_v};
    // split-D: load full 512-col V (both N-tiles), not TileShapePV_Eff (256 cols)
    using TileShapePV_Load = cute::conditional_t<kSplitDPV, TileShapePV, TileShapePV_Eff>;
    auto load_state_v = load_v.init_state(block_rank_in_cluster, problem_size, TileShapePV_Load{}, blk_coord, fusion_tile_count);

    if constexpr (kLoadQ) {
      load_q.step(q_tile_iter, load_state_q, smem_pipe_write_q, lane_predicate, q_tile_count);
    }

    load_k.template step<false>(k_tile_iter, load_state_k, smem_pipe_write, lane_predicate, k_tile_count, mcast_mask_b);

    if constexpr (kLoadQ) {
      load_q.step(q_tile_iter, load_state_q, smem_pipe_write_q, lane_predicate, q_tile_count);
    }

    if constexpr (! kLoadQ) {
      if (do_barrier) {
        load_warp_barrier.arrive();
        load_warp_barrier.wait(/*phase=*/ 0);
        do_barrier = false;
      }
    }

    load_v.template step<true>(k_tile_iter, load_state_v, smem_pipe_write, lane_predicate, k_tile_count, mcast_mask_b);

    if constexpr (kLoadQ) {
      while (q_tile_count > 0) {
        load_q.step(q_tile_iter, load_state_q, smem_pipe_write_q, lane_predicate, q_tile_count);
      }
    }

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      load_k.template step<false>(k_tile_iter, load_state_k, smem_pipe_write, lane_predicate, k_tile_count, mcast_mask_b);
      load_v.template step<true>(k_tile_iter, load_state_v, smem_pipe_write, lane_predicate, k_tile_count, mcast_mask_b);
    }
  }

  template<class BlkCoord, class ProblemShape, class LoadWarpBarrier>
  CUTLASS_DEVICE void
  load_maybe_q(
      BlkCoord const& blk_coord, Params const& params, ProblemShape const& problem_size,
      MainloopPipelineQ& pipeline_q, PipelineStateQ& smem_pipe_write_q, 
      SharedStorage& storage,
      LoadWarpBarrier& load_warp_barrier, bool do_barrier)
  {
    int lane_predicate = cute::elect_one_sync();

    LoadQ load_q{params.tma_load_q, pipeline_q, storage.smem_q};
    auto load_state_q = load_q.init_state(_0{}, problem_size, TileShapeQK{}, blk_coord, NumQKWarpGroups);

    auto q_tile_iter = cute::make_coord_iterator(Int<NumQKWarpGroups>{});

    CUTLASS_PRAGMA_UNROLL
    for (int q_tile_count = 0; q_tile_count < NumQKWarpGroups; q_tile_count++) {
      int count = 1;
      load_q.step(q_tile_iter, load_state_q, smem_pipe_write_q, lane_predicate, count);
      if (q_tile_count == 0 && do_barrier) {
        load_warp_barrier.arrive();
        load_warp_barrier.wait(/*phase=*/ 0);
        do_barrier = false;
      }
    }
  }

  template<class BlkCoord, class ProblemShape, class MainloopPipelineReducer, class PipelineStateReducer>
  CUTLASS_DEVICE void
  reduce(
      BlkCoord const& blk_coord, Params const& params, ProblemShape const& problem_size,
      MainloopPipelineReducer& pipeline_reducer, PipelineStateReducer& smem_pipe_write_reducer, 
      SharedStorage& storage)
  { /* no-op */ }

  template<class BlkCoord, class ProblemShape, class MainloopPipelineReducer, class PipelineStateReducer, class MathWgOrderBarrier>
  CUTLASS_DEVICE auto
  compute(
      BlkCoord const& blk_coord, BlkCoord const& wg_coord,
      Params const& params, ProblemShape const& problem_size,
      MainloopPipeline& pipeline, PipelineState& smem_pipe_read, 
      MainloopPipelineQ& pipeline_q, PipelineStateQ& smem_pipe_read_q, 
      MainloopPipelineReducer&, PipelineStateReducer&,
      SharedStorage& storage,
      MathWgOrderBarrier& math_wg_order_barrier)
  {
    int thread_idx = int(threadIdx.x);

    PipelineState smem_pipe_release = smem_pipe_read;
    PipelineStateQ smem_pipe_release_q = smem_pipe_read_q;

    TiledMmaQK tiled_mma_qk;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
  
    // Mainloop setup QK
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
  
    Tensor tSsQ = thr_mma_qk.partition_A(sQ);                                   // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tSsK = thr_mma_qk.partition_B(sK);                                   // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tSrQ = thr_mma_qk.make_fragment_A(tSsQ);                            // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tSrK = thr_mma_qk.make_fragment_B(tSsK);                            // (MMA,MMA_M,MMA_N,PIPE)

    // Prepare: MMA PV
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
  
    // Mainloop setup PV
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tOsV = thr_mma_pv.partition_B(sV);                                   // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tOrV = thr_mma_pv.make_fragment_B(tOsV);                            // (MMA,MMA_M,MMA_N,PIPE)

    int k_tile_count = Fusion{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_size);

    pipeline_q.consumer_wait(smem_pipe_read_q);

    // mapping into QK accumulator
    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{}));
    Tensor tPcP = thr_mma_qk.partition_C(cP);
    int m_block = get<0>(wg_coord);
    tPcP.data() = tPcP.data() + E<0>{} * m_block * get<0>(TileShapeQK{});

    // Allocate PV acc
    Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePV{}));

    cutlass::fmha::collective::CollectiveSoftmax<ElementAccumulatorQK, Fusion, decltype(params)> softmax{params};
    auto softmax_state = softmax.init(acc_pv, tiled_mma_pv);

    if (true)
    {
        --k_tile_count;
        // Allocate QK acc
        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
  
        pipeline.consumer_wait(smem_pipe_read);
        math_wg_order_barrier.wait();

        // MMA QK
        warpgroup_fence_operand(acc_qk);
        warpgroup_arrive();
  
        gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,smem_pipe_read_q.index()), tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
        warpgroup_commit_batch();
        math_wg_order_barrier.arrive();

        ++smem_pipe_read;
  
        // Wait for the pipeline MMAs to drain
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_qk);

        softmax.step(acc_qk, tiled_mma_qk, tPcP, softmax_state, problem_size);
  
        Tensor acc_qk_fixed = make_acc_into_op<Element>(acc_qk, typename TiledMmaPV::LayoutA_TV{});
  
        pipeline.consumer_wait(smem_pipe_read);

        // MMA PV
        warpgroup_fence_operand(acc_pv);
        warpgroup_fence_operand(acc_qk_fixed);
        warpgroup_arrive();
  
        gemm_zero_acc(tiled_mma_pv, acc_qk_fixed, tOrV(_,_,_,smem_pipe_read.index()), acc_pv);
        warpgroup_commit_batch();

        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;

        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;

        // Advance consumer pipeline
        ++smem_pipe_read;
        tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }
  
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0)
    {
        --k_tile_count;

        // Allocate QK acc
        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
  
        pipeline.consumer_wait(smem_pipe_read);

        // MMA QK
        warpgroup_fence_operand(acc_qk);
        warpgroup_arrive();

        gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,smem_pipe_read_q.index()), tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
        warpgroup_commit_batch();

        ++smem_pipe_read;
        auto tok = pipeline.consumer_try_wait(smem_pipe_read);
  
        // Wait for the pipeline MMAs to drain
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_qk);
        warpgroup_fence_operand(acc_pv);

        if constexpr (kIsMainloopLocked) math_wg_order_barrier.wait();
        softmax.template step<false>(acc_qk, tiled_mma_qk, tPcP, softmax_state, acc_pv, tiled_mma_pv, problem_size);
        if constexpr (kIsMainloopLocked) math_wg_order_barrier.arrive();

        Tensor acc_qk_fixed = make_acc_into_op<Element>(acc_qk, typename TiledMmaPV::LayoutA_TV{});
  
        pipeline.consumer_wait(smem_pipe_read, tok);

        // MMA PV
        warpgroup_fence_operand(acc_pv);
        warpgroup_fence_operand(acc_qk_fixed);
        warpgroup_arrive();
  
        cute::gemm(tiled_mma_pv, acc_qk_fixed, tOrV(_,_,_,smem_pipe_read.index()), acc_pv);
        warpgroup_commit_batch();

        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;
  
        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;

        ++smem_pipe_read;
        tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }

    k_tile_count += Fusion{}.get_masked_trip_count(blk_coord, TileShape{}, problem_size);

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0)
    {
        --k_tile_count;

        // Allocate QK acc
        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));
  
        pipeline.consumer_wait(smem_pipe_read);

        // MMA QK
        warpgroup_fence_operand(acc_qk);
        warpgroup_arrive();

        gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,smem_pipe_read_q.index()), tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
        warpgroup_commit_batch();

        ++smem_pipe_read;
        auto tok = pipeline.consumer_try_wait(smem_pipe_read);
  
        // Wait for the pipeline MMAs to drain
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_qk);
        warpgroup_fence_operand(acc_pv);

        //if constexpr (kIsPersistent)
        //  if (k_tile_count == 0) pipeline_q.consumer_release(smem_pipe_release_q);

        if constexpr (kIsMainloopLocked) math_wg_order_barrier.wait();
        softmax.step(acc_qk, tiled_mma_qk, tPcP, softmax_state, acc_pv, tiled_mma_pv, problem_size);
        if constexpr (kIsMainloopLocked) math_wg_order_barrier.arrive();

        Tensor acc_qk_fixed = make_acc_into_op<Element>(acc_qk, typename TiledMmaPV::LayoutA_TV{});
  
        pipeline.consumer_wait(smem_pipe_read, tok);

        // MMA PV
        warpgroup_fence_operand(acc_pv);
        warpgroup_fence_operand(acc_qk_fixed);
        warpgroup_arrive();
  
        cute::gemm(tiled_mma_pv, acc_qk_fixed, tOrV(_,_,_,smem_pipe_read.index()), acc_pv);
        warpgroup_commit_batch();

        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;
  
        pipeline.consumer_release(smem_pipe_release);
        ++smem_pipe_release;

        ++smem_pipe_read;
        tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }

    if (kIsPersistent) pipeline_q.consumer_release(smem_pipe_release_q);

    // Wait for the pipeline MMAs to drain
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_pv);

    if (kIsPersistent) pipeline.consumer_release(smem_pipe_release);
    ++smem_pipe_release;

    Tensor lse = softmax.tail(softmax_state, acc_pv, tiled_mma_pv);

    return make_tuple(acc_pv, lse);
  }

  // ===== Head-chunked PV helpers =====

  // Exchange O half between registers and smem (bf16 roundtrip).
  template<class Acc>
  CUTLASS_DEVICE static void swap_o_smem(Acc&& acc, Element* smem, int count) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < count; i++) {
      float tmp = static_cast<float>(Element(smem[i]));
      smem[i] = static_cast<Element>(acc(i));
      acc(i) = tmp;
    }
  }

  // ===== Head-chunked compute =====

  template<class BlkCoord, class ProblemShape, class MainloopPipelineReducer,
           class PipelineStateReducer, class MathWgOrderBarrier>
  CUTLASS_DEVICE auto
  compute_chunked(
      BlkCoord const& blk_coord, BlkCoord const& wg_coord,
      Params const& params, ProblemShape const& problem_size,
      MainloopPipeline& pipeline, PipelineState& smem_pipe_read,
      MainloopPipelineQ& pipeline_q, PipelineStateQ& smem_pipe_read_q,
      MainloopPipelineReducer&, PipelineStateReducer&,
      SharedStorage& storage,
      MathWgOrderBarrier& math_wg_order_barrier)
  {
    static_assert(kHeadChunkedPV);

    int thread_idx = int(threadIdx.x);
    constexpr int kOPerThread = MPerWG * HeadDimHalf / cutlass::NumThreadsPerWarpGroup;

    PipelineState smem_pipe_release = smem_pipe_read;
    PipelineStateQ smem_pipe_release_q = smem_pipe_read_q;

    // QK setup
    TiledMmaQK tiled_mma_qk;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor tSsQ = thr_mma_qk.partition_A(sQ);
    Tensor tSsK = thr_mma_qk.partition_B(sK);
    Tensor tSrQ = thr_mma_qk.make_fragment_A(tSsQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(tSsK);

    // PV setup — RS mode, full-D partition for V descriptor tensor
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
    Tensor tOsV = thr_mma_pv.partition_B(sV);
    Tensor tOrV = thr_mma_pv.make_fragment_B(tOsV);

    int k_tile_count = Fusion{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_size);

    pipeline_q.consumer_wait(smem_pipe_read_q);

    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{}));
    Tensor tPcP = thr_mma_qk.partition_C(cP);
    int m_block = get<0>(wg_coord);
    tPcP.data() = tPcP.data() + E<0>{} * m_block * get<0>(TileShapeQK{});

    // Half-D accumulator: 128 regs (O_lo in regs, O_hi in smem)
    Tensor acc_lo = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePV_Eff{}));

    // O_hi smem: per-thread flat buffer
    Element* smem_o = storage.smem_o_half.data() +
                      (thread_idx % cutlass::NumThreadsPerWarpGroup) * kOPerThread;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOPerThread; i++) smem_o[i] = Element(0);

    cutlass::fmha::collective::CollectiveSoftmax<ElementAccumulatorQK, Fusion, decltype(params)> softmax{params};
    auto softmax_state = softmax.init(acc_lo, tiled_mma_pv);
    auto& s_max = get<0>(softmax_state);
    auto& a_sum = get<1>(softmax_state);

    // ===== First KV tile =====
    {
      --k_tile_count;

      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));

      pipeline.consumer_wait(smem_pipe_read);
      math_wg_order_barrier.wait();

      warpgroup_fence_operand(acc_qk);
      warpgroup_arrive();
      gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,smem_pipe_read_q.index()),
                    tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
      warpgroup_commit_batch();
      math_wg_order_barrier.arrive();
      ++smem_pipe_read;

      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_qk);

      softmax.step(acc_qk, tiled_mma_qk, tPcP, softmax_state, problem_size);
      Tensor acc_qk_f = make_acc_into_op<Element>(acc_qk, typename TiledMmaPV::LayoutA_TV{});

      // Release K stage
      pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release;

      // Two-pass PV via rank-(2,2,1) dispatch
      pipeline.consumer_wait(smem_pipe_read);

      warpgroup_fence_operand(acc_lo);
      warpgroup_fence_operand(acc_qk_f);
      warpgroup_arrive();
      gemm_zero_acc(tiled_mma_pv,
                    acc_qk_f(_, _0{}, _),
                    tOrV(_, _0{}, _, smem_pipe_read.index()),
                    acc_lo(_, _0{}, _0{}));
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_lo);

      swap_o_smem(acc_lo(_, _0{}, _0{}), smem_o, kOPerThread);

      warpgroup_fence_operand(acc_lo);
      warpgroup_fence_operand(acc_qk_f);
      warpgroup_arrive();
      gemm_zero_acc(tiled_mma_pv,
                    acc_qk_f(_, _0{}, _),
                    tOrV(_, _1{}, _, smem_pipe_read.index()),
                    acc_lo(_, _0{}, _0{}));
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_lo);

      swap_o_smem(acc_lo(_, _0{}, _0{}), smem_o, kOPerThread);

      pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release;
      ++smem_pipe_read;

      tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }

    // ===== Subsequent KV tiles =====
    k_tile_count += Fusion{}.get_masked_trip_count(blk_coord, TileShape{}, problem_size);
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      --k_tile_count;

      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));

      pipeline.consumer_wait(smem_pipe_read);
      warpgroup_fence_operand(acc_qk);
      warpgroup_arrive();
      gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,smem_pipe_read_q.index()),
                    tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
      warpgroup_commit_batch();
      pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release;
      ++smem_pipe_read;

      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_qk);
      warpgroup_fence_operand(acc_lo);

      Tensor old_max = make_fragment_like(s_max);
      cute::copy(s_max, old_max);
      softmax.step(acc_qk, tiled_mma_qk, tPcP, softmax_state,
                   acc_lo, tiled_mma_pv, problem_size);
      // Rescale O_hi in smem
      {
        auto mn_layout = layout_acc_mn(tiled_mma_pv, acc_lo.layout());
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(mn_layout); mi++) {
          float sf = ::exp2f((old_max(mi) - s_max(mi)) * params.scale_softmax_log2);
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < size<1>(mn_layout); ni++) {
            int flat = mn_layout(mi, ni);
            smem_o[flat] = static_cast<Element>(
                static_cast<float>(Element(smem_o[flat])) * sf);
          }
        }
      }

      Tensor acc_qk_f = make_acc_into_op<Element>(acc_qk, typename TiledMmaPV::LayoutA_TV{});

      // Two-pass PV with accumulation
      pipeline.consumer_wait(smem_pipe_read);

      warpgroup_fence_operand(acc_lo);
      warpgroup_fence_operand(acc_qk_f);
      warpgroup_arrive();
      tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
      gemm_reset_zero_acc(tiled_mma_pv,
                          acc_qk_f(_, _0{}, _),
                          tOrV(_, _0{}, _, smem_pipe_read.index()),
                          acc_lo(_, _0{}, _0{}));
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_lo);

      swap_o_smem(acc_lo(_, _0{}, _0{}), smem_o, kOPerThread);

      warpgroup_fence_operand(acc_lo);
      warpgroup_fence_operand(acc_qk_f);
      warpgroup_arrive();
      tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
      gemm_reset_zero_acc(tiled_mma_pv,
                          acc_qk_f(_, _0{}, _),
                          tOrV(_, _1{}, _, smem_pipe_read.index()),
                          acc_lo(_, _0{}, _0{}));
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_lo);

      swap_o_smem(acc_lo(_, _0{}, _0{}), smem_o, kOPerThread);

      pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release;
      ++smem_pipe_read;

      tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }

    if (kIsPersistent) pipeline_q.consumer_release(smem_pipe_release_q);
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_lo);
    if (kIsPersistent) pipeline.consumer_release(smem_pipe_release);
    ++smem_pipe_release;

    // Softmax tail: normalize O_lo, get LSE
    Tensor lse = softmax.tail(softmax_state, acc_lo, tiled_mma_pv);

    // Write O_lo to gmem (direct stores — before epilogue overwrites smem)
    int batch_idx = get<0>(get<2>(wg_coord));
    int head_idx = get<1>(get<2>(wg_coord));
    int seq_stride_o = get<0>(params.dO);
    int batch_stride_o = get<0>(get<2>(params.dO));
    int head_stride_o = get<1>(get<2>(params.dO));
    int o_base = batch_idx * batch_stride_o + head_idx * head_stride_o +
                 m_block * MPerWG * seq_stride_o;
    int seqlen_q = get<2>(problem_size);

    // Use identity tensor to map accumulator elements to (m, n) coordinates
    Tensor cO = make_identity_tensor(make_shape(Int<MPerWG>{}, Int<HeadDimPV>{}));
    Tensor tOcO = thr_mma_pv.partition_C(cO);
    // O_lo: first N-tile, columns [0, HeadDimHalf)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_lo(_, _0{}, _0{})); i++) {
      int m = get<0>(tOcO(i, _0{}, _0{}));
      int n = get<1>(tOcO(i, _0{}, _0{}));
      if (m + m_block * MPerWG < seqlen_q) {
        params.ptr_O[o_base + m * seq_stride_o + n] =
            static_cast<Element>(acc_lo(i, _0{}, _0{}));
      }
    }

    // Normalize O_hi in smem using MMA's (row, col) → flat index mapping
    {
      auto mn_layout = layout_acc_mn(tiled_mma_pv, acc_lo.layout());
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(mn_layout); mi++) {
        float inv_sum = (a_sum(mi) == 0.f || a_sum(mi) != a_sum(mi))
                            ? 1.f : __frcp_rn(a_sum(mi));
        float sf = params.rp_dropout * inv_sum;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(mn_layout); ni++) {
          int flat = mn_layout(mi, ni);
          smem_o[flat] = static_cast<Element>(
              static_cast<float>(Element(smem_o[flat])) * sf);
        }
      }
    }
    // O_hi: use same identity tensor mapping as O_lo, offset columns by HeadDimHalf
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_lo(_, _0{}, _0{})); i++) {
      int m = get<0>(tOcO(i, _0{}, _0{}));
      int n = get<1>(tOcO(i, _0{}, _0{})) + HeadDimHalf;
      if (m + m_block * MPerWG < seqlen_q) {
        params.ptr_O[o_base + m * seq_stride_o + n] = smem_o[i];
      }
    }

    // Write LSE
    if (params.ptr_LSE != nullptr) {
      auto acc_mn = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma_pv, tOcO.layout()));
      if (get<1>(acc_mn(_0{}, _0{})) == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(acc_mn); i++) {
          int m = get<0>(acc_mn(i, _0{}));
          int abs_m = m + m_block * MPerWG;
          if (abs_m < seqlen_q) {
            params.ptr_LSE[batch_idx * seqlen_q + abs_m] = lse(i);
          }
        }
      }
    }

    // Return half-D acc for epilogue compatibility (epilogue skipped for chunked)
    return make_tuple(acc_lo, lse);
  }

  // ===== Split-D PV: dual-QK RS mode, each WG computes QK+softmax+PV(half-V) =====
  // NTile: 0 = V columns 0..HeadDimHalf-1, 1 = V columns HeadDimHalf..HeadDimFull-1
  // ColOffset: 0 for NTile=0, HeadDimHalf for NTile=1

  template<int kVNTile, int kColOffset,
           class BlkCoord, class ProblemShape, class MainloopPipelineReducer,
           class PipelineStateReducer, class MathWgOrderBarrier>
  CUTLASS_DEVICE void
  compute_splitd(
      BlkCoord const& blk_coord, BlkCoord const& wg_coord,
      Params const& params, ProblemShape const& problem_size,
      MainloopPipeline& pipeline, PipelineState& smem_pipe_read,
      MainloopPipelineQ& pipeline_q, PipelineStateQ& smem_pipe_read_q,
      MainloopPipelineReducer&, PipelineStateReducer&,
      SharedStorage& storage,
      MathWgOrderBarrier& math_wg_order_barrier)
  {
    static_assert(kSplitDPV);
    int thread_idx = int(threadIdx.x);

    PipelineState smem_pipe_release = smem_pipe_read;
    PipelineStateQ smem_pipe_release_q = smem_pipe_read_q;

    // QK setup (single WG, 128 threads — both WGs do QK independently)
    TiledMmaQK tiled_mma_qk;
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor tSsQ = thr_mma_qk.partition_A(sQ);
    Tensor tSsK = thr_mma_qk.partition_B(sK);
    Tensor tSrQ = thr_mma_qk.make_fragment_A(tSsQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(tSsK);

    // PV setup: RS mode, V N-tile kVNTile selected from full-D V partition
    TiledMmaPV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
    Tensor tOsV = thr_mma_pv.partition_B(sV);
    Tensor tOrV = thr_mma_pv.make_fragment_B(tOsV);

    int k_tile_count = Fusion{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_size);

    pipeline_q.consumer_wait(smem_pipe_read_q);

    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{}));
    Tensor tPcP = thr_mma_qk.partition_C(cP);

    // PV accumulator — half head-dim (256 cols via N-tile selection)
    Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePV_Eff{}));

    cutlass::fmha::collective::CollectiveSoftmax<ElementAccumulatorQK, Fusion, decltype(params)> softmax{params};
    auto softmax_state = softmax.init(acc_pv, tiled_mma_pv);

    // ===== First KV tile =====
    {
      --k_tile_count;
      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));

      pipeline.consumer_wait(smem_pipe_read);
      math_wg_order_barrier.wait();

      warpgroup_fence_operand(acc_qk);
      warpgroup_arrive();
      gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,smem_pipe_read_q.index()),
                    tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
      warpgroup_commit_batch();
      math_wg_order_barrier.arrive();
      ++smem_pipe_read;

      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_qk);

      softmax.step(acc_qk, tiled_mma_qk, tPcP, softmax_state, problem_size);

      Tensor acc_qk_f = make_acc_into_op<Element>(acc_qk, typename TiledMmaPV::LayoutA_TV{});

      // Release K stage
      pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release;

      // PV GEMM: RS mode, select V N-tile kVNTile
      pipeline.consumer_wait(smem_pipe_read);
      warpgroup_fence_operand(acc_pv);
      warpgroup_fence_operand(acc_qk_f);
      warpgroup_arrive();
      gemm_zero_acc(tiled_mma_pv, acc_qk_f, tOrV(_,_,_,smem_pipe_read.index()), acc_pv);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_pv);

      pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release;
      ++smem_pipe_read;
      tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }

    // ===== Subsequent KV tiles =====
    k_tile_count += Fusion{}.get_masked_trip_count(blk_coord, TileShape{}, problem_size);
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      --k_tile_count;
      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQK{}));

      pipeline.consumer_wait(smem_pipe_read);
      warpgroup_fence_operand(acc_qk);
      warpgroup_arrive();
      gemm_zero_acc(tiled_mma_qk, tSrQ(_,_,_,smem_pipe_read_q.index()),
                    tSrK(_,_,_,smem_pipe_read.index()), acc_qk);
      warpgroup_commit_batch();
      ++smem_pipe_read;
      auto tok = pipeline.consumer_try_wait(smem_pipe_read);

      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_qk);
      warpgroup_fence_operand(acc_pv);

      softmax.step(acc_qk, tiled_mma_qk, tPcP, softmax_state,
                   acc_pv, tiled_mma_pv, problem_size);

      Tensor acc_qk_f = make_acc_into_op<Element>(acc_qk, typename TiledMmaPV::LayoutA_TV{});

      // PV GEMM: RS mode, accumulate
      pipeline.consumer_wait(smem_pipe_read, tok);
      warpgroup_fence_operand(acc_pv);
      warpgroup_fence_operand(acc_qk_f);
      warpgroup_arrive();
      tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
      cute::gemm(tiled_mma_pv, acc_qk_f, tOrV(_,_,_,smem_pipe_read.index()), acc_pv);
      warpgroup_commit_batch();

      pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release;
      pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release;
      ++smem_pipe_read;
      tPcP.data() = tPcP.data() + E<1>{} * get<1>(TileShapeQK{});
    }

    if (kIsPersistent) pipeline_q.consumer_release(smem_pipe_release_q);
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_pv);
    if (kIsPersistent) pipeline.consumer_release(smem_pipe_release);
    ++smem_pipe_release;

    Tensor lse = softmax.tail(softmax_state, acc_pv, tiled_mma_pv);

    // Direct gmem write for O (this WG's head-dim partition)
    int batch_idx = get<0>(get<2>(wg_coord));
    int head_idx = get<1>(get<2>(wg_coord));
    int seq_stride_o = get<0>(params.dO);
    int batch_stride_o = get<0>(get<2>(params.dO));
    int head_stride_o = get<1>(get<2>(params.dO));
    int o_base = batch_idx * batch_stride_o + head_idx * head_stride_o;
    int seqlen_q = get<2>(problem_size);

    Tensor cO = make_identity_tensor(take<0, 2>(TileShapePV{}));
    Tensor tOcO = thr_mma_pv.partition_C(cO);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_pv); i++) {
      int m = get<0>(tOcO(i));
      int n = get<1>(tOcO(i));
      if (m < seqlen_q && n >= kColOffset && n < kColOffset + HeadDimHalf) {
        params.ptr_O[o_base + m * seq_stride_o + n] =
            static_cast<Element>(acc_pv(i));
      }
    }

    // Write LSE (only one WG needs to do this)
    if (kVNTile == 0 && params.ptr_LSE != nullptr) {
      auto acc_mn = make_tensor(tOcO.data(), layout_acc_mn(tiled_mma_pv, tOcO.layout()));
      if (get<1>(acc_mn(_0{}, _0{})) == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(acc_mn); i++) {
          int m = get<0>(acc_mn(i, _0{}));
          if (m < seqlen_q) {
            params.ptr_LSE[batch_idx * seqlen_q + m] = lse(i);
          }
        }
      }
    }
  }
};

}  // namespace cutlass::fmha::collective
