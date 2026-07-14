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
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective {

enum class LoadKind {
  kQ, kK, kV,
  kPagedK, kPagedV,        // 16-token pages: pages_per_tile copies per tile
  kPaged64K, kPaged64V,    // 64-token pages: one whole-tile copy (page==tile)
  kBwdN, kBwdM, kBwdScalar
};

// vLLM KV cache page size (tokens). Paged loads issue kPagesPerTile
// per-page TMA copies per KV tile (tile_n / kPageSize), each landing in a
// page-slice of the stage; the pipeline's per-stage transaction-byte count
// is unchanged (the copies sum to one full stage).
static constexpr int kPagedPageSize = 16;

template<
  LoadKind kKind,
  class Pipeline,
  class Element,
  class SmemLayout,
  class TMA
>
struct CollectiveLoadTma {

  using Params = TMA;
  using SharedStorage = cute::array_aligned<Element, cute::cosize_v<SmemLayout>>;
  using PipelineState  = typename cutlass::PipelineState<Pipeline::Stages>;

  Params const& params;
  Pipeline& pipeline;
  SharedStorage& storage;
  const int* page_table;
  int gqa_group;
  int max_blocks_per_seq;
  int batch_idx;
  int num_pages;  // valid pages this seq (paged tail clamp)

  CUTLASS_DEVICE
  CollectiveLoadTma(Params const& params, Pipeline& pipeline, SharedStorage& storage,
                    const int* page_table = nullptr, int gqa_group = 1,
                    int max_blocks_per_seq = 0)
    : params(params), pipeline(pipeline), storage(storage),
      page_table(page_table), gqa_group(gqa_group),
      max_blocks_per_seq(max_blocks_per_seq), batch_idx(0), num_pages(0) {}

  template<class ProblemSize, class TileShape, class BlockCoord>
  CUTLASS_DEVICE auto init_g(ProblemSize const& problem_size, TileShape const& tile_shape,
      BlockCoord const& blk_coord, int loop_count
  ) {
    using X = Underscore;
    if constexpr (kKind == LoadKind::kK) {
      Tensor mK_full = params.get_tma_tensor(make_shape(get<3>(problem_size), get<4>(problem_size), select<0,1>(problem_size)));
      Tensor gK_full = local_tile(mK_full, tile_shape, make_coord(_, _, _), Step<X, _1, _1>{});
      // GQA-dense contiguous KV (prefill de-GQA): batch stride is per KV
      // head; map the q-head coord down. gqa_group==1 (default) keeps the
      // legacy expanded-buffer behavior.
      auto kv_coord_k = make_coord(
          int(get<0>(get<2>(blk_coord))) / (gqa_group > 0 ? gqa_group : 1),
          get<1>(get<2>(blk_coord)));
      Tensor gK = gK_full(_, _, _, _0{}, kv_coord_k);
      return gK;
    } else if constexpr (kKind == LoadKind::kQ) {
      Tensor mQ_full = params.get_tma_tensor(make_shape(get<2>(problem_size), get<4>(problem_size), select<0,1>(problem_size)));
      Tensor gQ_full = local_tile(mQ_full, tile_shape, make_coord(_, _, _), Step<_1, X, _1>{});
      Tensor gQ = gQ_full(_, _, _, _0{}, get<2>(blk_coord));
      return make_tensor(gQ.data() + loop_count * get<0>(blk_coord) * stride<2>(gQ), gQ.layout());
    } else if constexpr (kKind == LoadKind::kV) {
      Tensor mV_full = params.get_tma_tensor(make_shape(get<4>(problem_size), get<3>(problem_size), select<0,1>(problem_size)));
      Tensor gV_full = local_tile(mV_full, tile_shape, make_coord(_, _, _), Step<X, _1, _1>{});
      auto kv_coord_v = make_coord(
          int(get<0>(get<2>(blk_coord))) / (gqa_group > 0 ? gqa_group : 1),
          get<1>(get<2>(blk_coord)));
      Tensor gV = gV_full(_, _, _0{}, _, kv_coord_v);
      return gV;
    } else if constexpr (kKind == LoadKind::kPagedK) {
      // Paged K: TMA covers (page_size, head_dim, (num_blocks, num_kv_heads))
      // over the KV cache pool. One KV tile = kPagesPerTile pages; step()
      // issues one copy per page into a page-slice of the stage. kv_head is
      // derived from the q-head block coord via GQA mapping.
      int num_blocks = get<7>(problem_size);
      int num_kv_heads = get<0>(problem_size) / gqa_group;
      int kv_head = int(get<0>(get<2>(blk_coord))) / gqa_group;
      Tensor mK = params.get_tma_tensor(
          make_shape(Int<kPagedPageSize>{}, get<4>(problem_size),
                     make_shape(num_blocks, num_kv_heads)));
      return mK(_, _, make_coord(_, kv_head));
    } else if constexpr (kKind == LoadKind::kPagedV) {
      int num_blocks = get<7>(problem_size);
      int num_kv_heads = get<0>(problem_size) / gqa_group;
      int kv_head = int(get<0>(get<2>(blk_coord))) / gqa_group;
      Tensor mV = params.get_tma_tensor(
          make_shape(get<4>(problem_size), Int<kPagedPageSize>{},
                     make_shape(num_blocks, num_kv_heads)));
      return mV(_, _, make_coord(_, kv_head));
    } else if constexpr (kKind == LoadKind::kPaged64K) {
      // 64-token pages: one page == one KV tile, whole-tile copies (dense
      // 128KB boxes; none of the 16-page scatter latency).
      int num_blocks = get<7>(problem_size);
      int num_kv_heads = get<0>(problem_size) / gqa_group;
      int kv_head = int(get<0>(get<2>(blk_coord))) / gqa_group;
      Tensor mK = params.get_tma_tensor(
          make_shape(int(get<0>(tile_shape)), get<4>(problem_size),
                     make_shape(num_blocks, num_kv_heads)));
      return mK(_, _, make_coord(_, kv_head));
    } else if constexpr (kKind == LoadKind::kPaged64V) {
      int num_blocks = get<7>(problem_size);
      int num_kv_heads = get<0>(problem_size) / gqa_group;
      int kv_head = int(get<0>(get<2>(blk_coord))) / gqa_group;
      Tensor mV = params.get_tma_tensor(
          make_shape(get<4>(problem_size), int(get<0>(tile_shape)),
                     make_shape(num_blocks, num_kv_heads)));
      return mV(_, _, make_coord(_, kv_head));
    } else if constexpr (kKind == LoadKind::kBwdN) {
      Tensor m_full = params.get_tma_tensor(make_shape(get<3>(problem_size), get<4>(problem_size), select<0,1>(problem_size)));
      Tensor g_full = local_tile(m_full, tile_shape, make_coord(_, _, _), Step<_1, X, _1>{});
      Tensor g = g_full(_, _, _, _0{}, get<2>(blk_coord));
      return make_tensor(g.data() + loop_count * get<1>(blk_coord) * stride<2>(g), g.layout());
    } else if constexpr (kKind == LoadKind::kBwdM) {
      Tensor m_full = params.get_tma_tensor(make_shape(get<2>(problem_size), get<4>(problem_size), select<0,1>(problem_size)));
      Tensor g_full = local_tile(m_full, tile_shape, make_coord(_, _, _), Step<X, _1, _1>{});
      Tensor g = g_full(_, _, _, _0{}, get<2>(blk_coord));
      return g;
    } else if constexpr (kKind == LoadKind::kBwdScalar) {
      Tensor m_full = params.get_tma_tensor(select<2,0,1>(problem_size));
      Tensor g_full = local_tile(m_full, tile_shape, make_coord(_, _, _), Step<X, _1, X>{});
      Tensor g = g_full(_, _, get<2,0>(blk_coord), get<2,1>(blk_coord));
      return g;
    }
  }

  template<class ClusterRank, class ProblemSize, class TileShape, class BlockCoord>
  CUTLASS_DEVICE auto init_state(ClusterRank const& block_rank_in_cluster,
      ProblemSize const& problem_size, TileShape const& tile_shape,
      BlockCoord const& block_coord, int loop_count
  ) {
    if constexpr (kKind == LoadKind::kPagedK || kKind == LoadKind::kPagedV) {
      batch_idx = int(get<1>(get<2>(block_coord)));
      num_pages = (int(get<3>(problem_size)) + kPagedPageSize - 1) / kPagedPageSize;
    }
    if constexpr (kKind == LoadKind::kPaged64K || kKind == LoadKind::kPaged64V) {
      batch_idx = int(get<1>(get<2>(block_coord)));
      num_pages = (int(get<3>(problem_size)) + int(get<0>(tile_shape)) - 1) /
                  int(get<0>(tile_shape));
    }
    Tensor g = init_g(problem_size, tile_shape, block_coord, loop_count);
    Tensor s = make_tensor(make_smem_ptr(storage.data()), SmemLayout{});

    auto block_tma = params.get_slice(block_rank_in_cluster);
    Tensor ts = block_tma.partition_D(s);
    Tensor tg = block_tma.partition_S(g);

    return make_tuple(tg, ts);
  }

  template<bool kAdvanceIterator=true, bool kAdvancePipe=true, bool kAcquireBarrier=true, class TileIterator, class State>
  CUTLASS_DEVICE void step(TileIterator& tile_iter, State const& state,
      PipelineState& smem_pipe_write,
      int lane_predicate, int& tile_count, uint16_t mcast_mask = 0
  ) {
    if constexpr (kKind == LoadKind::kPagedK || kKind == LoadKind::kPagedV) {
      // Warp-cooperative paged issue: one copy per 16-token page, issued
      // from DIFFERENT lanes in parallel (single-thread issue of
      // pages_per_tile x hd/64 sub-boxes was the bottleneck: 32 serialized
      // TMA instructions/tile at hd512). The leader owns the pipeline slot;
      // every lane advances its local pipe/iter state to stay uniform. All
      // copies arrive on the SAME stage barrier (tx bytes = one full
      // stage). Tail pages are clamped to the last valid page (duplicate
      // data, masked by the softmax).
      if ((lane_predicate == 1) && (tile_count > 0)) {
        if constexpr (kAcquireBarrier) pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename Pipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        constexpr int kPages = decltype(size<3>(get<1>(state)))::value;
        const int* seq_pt = page_table + batch_idx * max_blocks_per_seq;
        const int base = int(*tile_iter) * kPages;
        const int stage = smem_pipe_write.index();
        CUTLASS_PRAGMA_UNROLL
        for (int pg = 0; pg < kPages; ++pg) {
          int logical = base + pg;
          if (logical >= num_pages) logical = num_pages - 1;
          int phys_block = seq_pt[logical];
          copy(params.with(*tma_barrier, mcast_mask),
               get<0>(state)(_,_,_,phys_block),
               get<1>(state)(_,_,_,pg,stage));
        }
        if constexpr (kAdvancePipe) ++smem_pipe_write;
        if constexpr (kAdvanceIterator) ++tile_iter;
      }
      --tile_count;
    } else if constexpr (kKind == LoadKind::kPaged64K ||
                         kKind == LoadKind::kPaged64V) {
      if ((lane_predicate == 1) && (tile_count > 0)) {
        if constexpr (kAcquireBarrier) pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename Pipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        int logical = int(*tile_iter);
        if (logical >= num_pages) logical = num_pages - 1;  // tail clamp
        int phys_block = page_table[batch_idx * max_blocks_per_seq + logical];
        copy(params.with(*tma_barrier, mcast_mask),
             get<0>(state)(_,_,_,phys_block),
             get<1>(state)(_,_,_,smem_pipe_write.index()));
        if constexpr (kAdvancePipe) ++smem_pipe_write;
        if constexpr (kAdvanceIterator) ++tile_iter;
      }
      --tile_count;
    } else {
      if ((lane_predicate == 1) && (tile_count > 0)) {
        if constexpr (kAcquireBarrier) pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename Pipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        if constexpr (kKind == LoadKind::kBwdScalar) {
          copy(params.with(*tma_barrier, mcast_mask), get<0>(state)(_,_,*tile_iter), get<1>(state)(_,_,smem_pipe_write.index()));
        } else {
          copy(params.with(*tma_barrier, mcast_mask), get<0>(state)(_,_,_,*tile_iter), get<1>(state)(_,_,_,smem_pipe_write.index()));
        }
        if constexpr (kAdvancePipe) ++smem_pipe_write;
        if constexpr (kAdvanceIterator) ++tile_iter;
      }
      --tile_count;
    }
  }
};

}  // namespace cutlass::fmha::collective
