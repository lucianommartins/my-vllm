# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils import random_uuid
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.worker.gpu.triton_fast_launch import FastTritonLauncher

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.attn_utils import FastPrefillBatchMetadata
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.pcp_manager import PCPBatchMetadata
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdmittedDraftCounts,
    )


class InputBuffers:
    def __init__(self, max_num_reqs: int, max_num_tokens: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device

        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # DCP: per-request local seq_lens buffer
        self.dcp_local_seq_lens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )

        # Preallocated per-step input prep buffers (avoid per-step CUDA/pinned allocs).
        self.idx_mapping = torch.empty(max_num_reqs, dtype=torch.int64, device=device)
        self.idx_mapping_i32 = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        self.cu_num_logits_arange = torch.arange(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.cu_num_logits_arange_np = np.arange(max_num_reqs + 1, dtype=np.int32)
        self.expanded_local_pos_zeros = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.logits_indices_arange = torch.arange(
            max_num_reqs, dtype=torch.int64, device=device
        )
        self.logits_indices = torch.empty(
            max_num_tokens, dtype=torch.int64, device=device
        )
        self._h2d_ring_idx = 0
        self._q_start_ring_idx = 0
        self._idx_mapping_cpu = [
            torch.empty(
                max_num_reqs, dtype=torch.int64, device="cpu", pin_memory=PIN_MEMORY
            )
            for _ in range(4)
        ]
        self._idx_mapping_cpu_np = [t.numpy() for t in self._idx_mapping_cpu]
        self._query_start_loc_cpu = [
            torch.empty(
                max_num_reqs + 1, dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
            )
            for _ in range(4)
        ]
        self._query_start_loc_cpu_np = [t.numpy() for t in self._query_start_loc_cpu]
        self._seq_lens_cpu_ub = torch.zeros(max_num_reqs, dtype=torch.int32)
        self._seq_lens_cpu_ub_np = self._seq_lens_cpu_ub.numpy()
        self._seq_lens_ub_views: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        self._uniform_q_start_cache: dict[
            tuple[int, int], tuple[torch.Tensor, np.ndarray]
        ] = {}
        self._last_uniform_q_start_key: tuple[int, int] | None = None
        self._last_is_padding_state: tuple[int, int] | None = None
        self._last_idx_mapping_bytes: bytes | None = None
        self._last_idx_mapping_np: Any = None
        self._tok_view_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._req_view_cache: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor],
        ] = {}
        self._seq_lens_view_cache: dict[int, torch.Tensor] = {}
        self._logits_arange_view_cache: dict[int, torch.Tensor] = {}

    def get_tok_views(
        self, n: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached (is_padding[:n], input_ids[:n], positions[:n]) views."""
        cached = self._tok_view_cache.get(n)
        if cached is None:
            cached = (
                self.is_padding[:n],
                self.input_ids[:n],
                self.positions[:n],
            )
            if len(self._tok_view_cache) < 256:
                self._tok_view_cache[n] = cached
        return cached

    def get_req_views(
        self, n: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        """Return cached per-request buffer slices for length n."""
        cached = self._req_view_cache.get(n)
        if cached is None:
            cached = (
                self.idx_mapping[:n],
                self.idx_mapping_i32[:n],
                self.cu_num_logits_arange[: n + 1],
                self.cu_num_logits_arange_np[: n + 1],
                self.expanded_local_pos_zeros[:n],
            )
            if len(self._req_view_cache) < 256:
                self._req_view_cache[n] = cached
        return cached

    def get_seq_lens_view(self, n: int) -> torch.Tensor:
        cached = self._seq_lens_view_cache.get(n)
        if cached is None:
            cached = self.seq_lens[:n]
            if len(self._seq_lens_view_cache) < 256:
                self._seq_lens_view_cache[n] = cached
        return cached

    def get_logits_arange_view(self, n: int) -> torch.Tensor:
        cached = self._logits_arange_view_cache.get(n)
        if cached is None:
            cached = self.logits_indices_arange[:n]
            if len(self._logits_arange_view_cache) < 256:
                self._logits_arange_view_cache[n] = cached
        return cached

    def get_uniform_decode_query_start_loc(
        self, num_reqs: int, num_reqs_padded: int
    ) -> tuple[torch.Tensor, np.ndarray, bool]:
        """Return cached CPU uniform-decode query_start_loc & whether H2D is needed."""
        key = (num_reqs, num_reqs_padded)
        cached = self._uniform_q_start_cache.get(key)
        if cached is None:
            t = torch.empty(
                self.max_num_reqs + 1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            arr = t.numpy()
            arr[: num_reqs + 1] = np.arange(num_reqs + 1, dtype=np.int32)
            arr[num_reqs + 1 :] = num_reqs
            cached = (t, arr[: num_reqs_padded + 1])
            self._uniform_q_start_cache[key] = cached
        needs_copy = self._last_uniform_q_start_key != key
        self._last_uniform_q_start_key = key
        return cached[0], cached[1], needs_copy

    def next_h2d_slot(
        self,
    ) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
        """Return next pinned H2D staging buffers from a 4-slot ring."""
        slot = self._h2d_ring_idx
        self._h2d_ring_idx = (slot + 1) & 3
        return (
            self._idx_mapping_cpu[slot],
            self._idx_mapping_cpu_np[slot],
            self._query_start_loc_cpu[slot],
            self._query_start_loc_cpu_np[slot],
        )

    def next_idx_h2d_slot(self) -> tuple[torch.Tensor, np.ndarray]:
        """Return next pinned idx_mapping H2D staging buffer from a 4-slot ring."""
        slot = self._h2d_ring_idx
        self._h2d_ring_idx = (slot + 1) & 3
        return self._idx_mapping_cpu[slot], self._idx_mapping_cpu_np[slot]

    def next_qstart_h2d_slot(self) -> tuple[torch.Tensor, np.ndarray]:
        """Return next pinned query_start_loc H2D staging buffer from a 4-slot ring."""
        slot = self._q_start_ring_idx
        self._q_start_ring_idx = (slot + 1) & 3
        return self._query_start_loc_cpu[slot], self._query_start_loc_cpu_np[slot]

    def seq_lens_ub_buffer(
        self, num_reqs_padded: int
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Return a reusable CPU int32 buffer/numpy view of length num_reqs_padded."""
        if num_reqs_padded > self._seq_lens_cpu_ub.shape[0]:
            arr = np.empty(num_reqs_padded, dtype=np.int32)
            return torch.from_numpy(arr), arr
        cached = self._seq_lens_ub_views.get(num_reqs_padded)
        if cached is None:
            cached = (
                self._seq_lens_cpu_ub[:num_reqs_padded],
                self._seq_lens_cpu_ub_np[:num_reqs_padded],
            )
            if len(self._seq_lens_ub_views) < 128:
                self._seq_lens_ub_views[num_reqs_padded] = cached
        return cached


@dataclass
class InputBatch:
    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    # Identical to idx_mapping except for spec decoding.
    expanded_idx_mapping: torch.Tensor
    # [total_num_logits] position within request for each logit
    expanded_local_pos: torch.Tensor

    # [num_reqs]
    # batch_idx -> num_scheduled_tokens, (upper bound when using adaptive verification)
    num_scheduled_tokens: np.ndarray
    # number of tokens in the batch,
    #  may be < sum(num_scheduled_tokens) when using adaptive verification
    num_tokens: int
    num_tokens_after_padding: int
    # Sum of draft tokens scheduled across requests.
    num_draft_tokens: int
    # [num_reqs] number of draft tokens scheduled for each request, if any.
    num_draft_tokens_per_req: np.ndarray | None

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    # [num_reqs] CPU upper bound on seq_lens (see CommonAttentionMetadata).
    seq_lens_cpu_upper_bound: torch.Tensor
    # [num_reqs]
    dcp_local_seq_lens: torch.Tensor | None
    # [num_reqs]
    num_computed_tokens_np: np.ndarray
    # [num_reqs]
    prefill_len_np: np.ndarray
    # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray
    # [num_reqs] CPU bool array == (num_computed_prefill_tokens_np < prefill_len_np).
    is_prefilling_np: np.ndarray
    # == np.any(is_prefilling_np)
    has_prefill: bool
    # No prefills, or only prefill rows that run as decodes.
    decode_graph_eligible: bool

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor
    # [num_tokens_after_padding]
    is_padding: torch.Tensor

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray

    # Whether any requests in batch use structured output.
    has_structured_output_reqs: bool

    # [num_reqs] per-request prompt length, only populated for R-SWA.
    prompt_lens: torch.Tensor | None

    # [num_reqs] only populated for non-dummy pipeline-parallel batches.
    max_seq_len_np: np.ndarray | None = None

    # Longest query the batch may contain. Set when a cudagraph descriptor promises
    # a query length this batch's own split does not reach, so attention metadata
    # stays valid for every replay the graph serves.
    max_query_len: int | None = None

    # Arms the KV-sharing fast prefill path for this step. Absent for dummy
    # (cudagraph capture) batches, which run the KV-sharing layers in full.
    fast_prefill: "FastPrefillBatchMetadata | None" = None

    # Set only on a PCP rank-local batch.
    pcp: "PCPBatchMetadata | None" = None

    # [num_reqs] prefilling rows that schedule one new prompt token (excluding
    # drafts) over existing context and so compute exactly like decodes.
    # None if there are no prefills.
    prefill_runs_as_decode_np: np.ndarray | None = None

    # [num_reqs] admitted draft counts in flight to the host. Set only under
    # adaptive verification with batch-sharded sampling, when the budget binds.
    num_admitted_draft_tokens: "AdmittedDraftCounts | None" = None

    # Optional pre-cast int32 expanded_idx_mapping & cached CPU metadata.
    expanded_idx_mapping_i32: torch.Tensor | None = None
    idx_mapping_bytes: bytes | None = None
    seq_lens_cpu_upper_bound_np: np.ndarray | None = None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
        max_query_len: int | None = None,
        is_padding: bool = True,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        device = input_buffers.device

        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.intp)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)
        idx_mapping_i32 = torch.arange(num_reqs, dtype=torch.int32, device=device)
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        # Distribute the remainder evenly so that no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens. Varlen graphs
        # accept any split with non-empty slots, so this shape works for them
        # too; attention metadata is built from the promised max_query_len.
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        assert max_query_len is None or base_tokens + (num_extra > 0) <= max_query_len
        num_scheduled_tokens = np.full(num_reqs, base_tokens, dtype=np.int32)
        if num_extra > 0:
            num_scheduled_tokens[-num_extra:] += 1
        assert int(num_scheduled_tokens.sum()) == num_tokens

        # seq_len equals to query_len
        input_buffers.seq_lens[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens[num_reqs - num_extra : num_reqs] = base_tokens + 1
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        if hasattr(input_buffers, "_last_uniform_q_start_key"):
            input_buffers._last_uniform_q_start_key = None
        if hasattr(input_buffers, "_last_is_padding_state"):
            input_buffers._last_is_padding_state = None
        if hasattr(input_buffers, "_last_idx_mapping_bytes"):
            input_buffers._last_idx_mapping_bytes = None
        if hasattr(input_buffers, "_last_idx_mapping_np"):
            input_buffers._last_idx_mapping_np = None
        input_buffers.query_start_loc[:1] = 0
        torch.cumsum(
            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]
        )
        # Pad for full CUDA graph mode.
        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

        input_ids = input_buffers.input_ids[:num_tokens].zero_()
        positions = input_buffers.positions[:num_tokens].zero_()

        input_buffers.is_padding[:num_tokens].fill_(is_padding)
        is_padding = input_buffers.is_padding[:num_tokens]

        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        # Copy so set_dummy_context can add context in place without touching
        # num_scheduled_tokens.
        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
            has_prefill=False,
            max_seq_len_np=None,
            decode_graph_eligible=True,
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
            prompt_lens=None,
            max_query_len=max_query_len,
            expanded_idx_mapping_i32=idx_mapping_i32,
        )


def set_dummy_context(
    input_batch: InputBatch,
    block_tables: "BlockTables",
    context_len: int,
    num_kv_blocks: int,
    max_model_len: int,
    input_block_tables: Sequence[torch.Tensor] | None = None,
) -> None:
    """Give each dummy request context_len of context, for profiling step cost
    and for DCP FULL cudagraph capture."""
    if input_block_tables is None:
        input_block_tables = block_tables.input_block_tables
    if not input_block_tables:
        # Attention-free models have no KV context to fabricate.
        return
    num_reqs = input_batch.num_reqs
    query_len = input_batch.max_query_len or int(input_batch.num_scheduled_tokens.max())
    context_len = max(min(context_len, max_model_len - query_len), 0)
    if not context_len:
        return

    # Decode-like shape: each request continues after context_len
    # already-computed tokens.
    input_batch.seq_lens += context_len
    input_batch.seq_lens_cpu_upper_bound += context_len
    input_batch.num_computed_tokens_np.fill(context_len)
    input_batch.num_computed_prefill_tokens_np.fill(context_len)
    local_pos = np.arange(input_batch.num_tokens, dtype=np.int64) - np.repeat(
        input_batch.query_start_loc_np[:-1], input_batch.num_scheduled_tokens
    )
    input_batch.positions.copy_(torch.from_numpy(local_pos + context_len))

    seq_len = context_len + query_len
    for block_table, block_size in zip(input_block_tables, block_tables.block_sizes):
        num_blocks = min(cdiv(seq_len, block_size), block_table.shape[1])
        # Spans are disjoint until the pool runs out, then they wrap and share
        # blocks: profiling only needs the reads to be realistic, not distinct.
        block_ids = (
            torch.arange(
                num_reqs * num_blocks,
                dtype=block_table.dtype,
                device=block_table.device,
            )
            % num_kv_blocks
        )
        block_table[:num_reqs, :num_blocks] = block_ids.view(num_reqs, num_blocks)


@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    next_prefill_tokens_stride,
    num_lookahead,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
    LOOKAHEAD_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    # Store the next num_lookahead prefill tokens.
    lookahead = tl.arange(0, LOOKAHEAD_BLOCK)
    pos = num_computed + query_len + lookahead
    in_lookahead = lookahead < num_lookahead
    tokens = tl.load(
        request_ptr + pos, mask=in_lookahead & (pos < prefill_len), other=0
    )
    tl.store(
        next_prefill_tokens_ptr
        + lookahead * next_prefill_tokens_stride
        + req_state_idx,
        tokens,
        mask=in_lookahead,
    )


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    num_lookahead = next_prefill_tokens.shape[0]
    _PREPARE_PREFILL_INPUTS_LAUNCHER.launch_on_device(
        num_reqs,
        1,
        1,
        input_ids.device,
        input_ids,
        next_prefill_tokens,
        next_prefill_tokens.stride(0),
        num_lookahead,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
        1024,
        triton.next_power_of_2(num_lookahead),
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    idx_mapping_i32_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    tl.store(idx_mapping_i32_ptr + req_id, req_state_idx.to(tl.int32))
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
    idx_mapping_i32: torch.Tensor | None = None,
) -> None:
    num_reqs = idx_mapping.shape[0]
    if idx_mapping_i32 is None:
        idx_mapping_i32 = torch.empty(
            num_reqs, dtype=torch.int32, device=idx_mapping.device
        )
    # NOTE(woosuk): We do +1 because the last thread block is used
    # to pad unused seq_lens as 0 for full CUDA graphs.
    _PREPARE_POS_SEQ_LENS_LAUNCHER.launch_on_device(
        num_reqs + 1,
        1,
        1,
        idx_mapping.device,
        pos,
        seq_lens,
        idx_mapping,
        idx_mapping_i32,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        1024,
    )


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Keep prompt-tail slots intact; only rewrite generated-token slots.
    first_logit_seq_pos = seq_len - num_logits
    if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
        # Write the last sampled token ID to input_ids.
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_new_sampled_tokens: int = 1,  # excl accepted draft tokens, a.k.a bonus tokens
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert num_new_sampled_tokens in (0, 1), (
        f"num_new_sampled_tokens must be 0 or 1, got {num_new_sampled_tokens}"
    )
    # use idx_mapping.shape[0] for actual request count
    num_reqs = idx_mapping.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    logits_indices = (
        torch.empty(num_logits, dtype=torch.int64, device=input_ids.device)
        if out is None
        else out[:num_logits]
    )
    _COMBINE_SAMPLED_AND_DRAFT_TOKENS_LAUNCHER.launch_on_device(
        num_reqs,
        1,
        1,
        input_ids.device,
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        triton.next_power_of_2(num_speculative_steps + num_new_sampled_tokens),
        num_new_sampled_tokens,
    )
    return logits_indices


@triton.jit(do_not_specialize=["num_reqs", "max_num_reqs"])
def _prepare_decode_inputs_kernel(
    input_ids_ptr,
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    idx_mapping_i32_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    prefill_len_ptr,
    num_reqs,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + offs, mask=mask, other=0)
    tl.store(idx_mapping_i32_ptr + offs, req_state_idx.to(tl.int32), mask=mask)
    nc = tl.load(num_computed_tokens_ptr + req_state_idx, mask=mask, other=0)
    tl.store(pos_ptr + offs, nc, mask=mask)
    seq_len = nc + 1
    if max_num_reqs <= BLOCK_SIZE:
        tl.store(
            seq_lens_ptr + offs,
            tl.where(mask, seq_len, 0),
            mask=offs < max_num_reqs,
        )
    else:
        tl.store(seq_lens_ptr + offs, seq_len, mask=mask)
        for base in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            pad_offs = base + tl.arange(0, BLOCK_SIZE)
            tl.store(seq_lens_ptr + pad_offs, 0, mask=pad_offs < max_num_reqs)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, mask=mask, other=0)
    write_mask = mask & (nc >= prefill_len)
    last_tok = tl.load(
        last_sampled_tokens_ptr + req_state_idx, mask=write_mask, other=0
    )
    tl.store(input_ids_ptr + offs, last_tok, mask=write_mask)


def prepare_decode_inputs_and_logits(
    input_ids: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
    idx_mapping: torch.Tensor,
    idx_mapping_i32: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    prefill_len: torch.Tensor,
) -> None:
    """Fused single-CTA prep for uniform single-token decode (num_reqs <= 256)."""
    num_reqs = idx_mapping.shape[0]
    assert num_reqs <= 256
    _PREPARE_DECODE_INPUTS_LAUNCHER.launch_on_device(
        1,
        1,
        1,
        input_ids.device,
        input_ids,
        pos,
        seq_lens,
        idx_mapping,
        idx_mapping_i32,
        num_computed_tokens,
        last_sampled_tokens,
        prefill_len,
        num_reqs,
        seq_lens.shape[0],
        256,
    )


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + batch_idx, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = logits_end - logits_start

    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + batch_idx, num_rejected)


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    _GET_NUM_SAMPLED_AND_REJECTED_LAUNCHER.launch_on_device(
        num_reqs,
        1,
        1,
        num_sampled.device,
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )
    return num_sampled, num_rejected


_GET_NUM_SAMPLED_AND_REJECTED_LAUNCHER = FastTritonLauncher(
    _get_num_sampled_and_rejected_kernel,
    constexpr_names=(),
)


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    draft_tokens_ptr,
    draft_tokens_out_ptr,
    num_draft_tokens,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    if req_state_idx < 0:
        # Filter rows with negative index entries.
        return

    if draft_tokens_ptr is not None:
        for i in range(num_draft_tokens):
            token_id = tl.load(draft_tokens_ptr + req_id * num_draft_tokens + i)
            tl.store(
                draft_tokens_out_ptr + req_state_idx * num_draft_tokens + i, token_id
            )

    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    if query_start_loc_ptr is None:
        query_len = 0
    else:
        query_start = tl.load(query_start_loc_ptr + req_id)
        query_end = tl.load(query_start_loc_ptr + req_id + 1)
        query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + req_id)

    computed_delta = query_len - num_rejected
    if computed_delta != 0:
        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + computed_delta)


_V1_EXT_UNSET = object()
_V1_EXT = _V1_EXT_UNSET


def _get_v1_ext():
    global _V1_EXT
    if _V1_EXT is _V1_EXT_UNSET:
        from vllm.v1.worker.gpu._v1_sm80_ops import get_ext

        _V1_EXT = get_ext()
    return _V1_EXT


def post_update(
    # [num_reqs] batch_idx -> req_state_idx; negative index means skip.
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor | None,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor | None,
    # [max_num_reqs, max_model_len]
    all_token_ids: torch.Tensor,
    # [max_num_reqs]
    total_len: torch.Tensor,
    # [num_reqs, num_speculative_steps], contiguous
    draft_tokens: torch.Tensor | None = None,
    # [max_num_reqs, num_speculative_steps], contiguous; required with draft_tokens
    draft_tokens_out: torch.Tensor | None = None,
    assume_no_rejected: bool = False,
) -> None:
    if (
        assume_no_rejected
        and draft_tokens is None
        and output_bin_counts is None
        and last_sampled_tokens.dtype == torch.int64
        and last_sampled_tokens.stride(0) == 1
        and sampled_tokens.ndim == 2
        and sampled_tokens.shape[1] == 1
        and sampled_tokens.dtype == torch.int64
        and idx_mapping.dtype == torch.int64
        and (ext := _get_v1_ext()) is not None
    ):
        ext.post_update_sm80(
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            sampled_tokens,
            num_sampled,
            query_start_loc,
            all_token_ids,
            total_len,
        )
        return
    num_reqs = idx_mapping.shape[0]
    if draft_tokens is not None:
        # The kernel uses num_speculative_steps as the row stride of both.
        assert draft_tokens_out is not None
        assert draft_tokens.shape[1] == draft_tokens_out.shape[1]
        assert draft_tokens.is_contiguous() and draft_tokens_out.is_contiguous()
    _POST_UPDATE_LAUNCHER.launch_on_device(
        num_reqs,
        1,
        1,
        idx_mapping.device,
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        draft_tokens,
        draft_tokens_out,
        draft_tokens.shape[1] if draft_tokens is not None else 0,
    )


@triton.jit
def _post_update_num_computed_tokens_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
):
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_len)


def post_update_num_computed_tokens(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _post_update_num_computed_tokens_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(
        total_num_logits, dtype=torch.int32, device=idx_mapping.device
    )
    _EXPAND_IDX_MAPPING_LAUNCHER.launch_on_device(
        num_reqs,
        1,
        1,
        idx_mapping.device,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        cu_num_logits,
        triton.next_power_of_2(max_expand_len),
    )
    return expanded_idx_mapping, expanded_local_pos


_PREPARE_PREFILL_INPUTS_LAUNCHER = FastTritonLauncher(
    _prepare_prefill_inputs_kernel,
    constexpr_names=("BLOCK_SIZE", "LOOKAHEAD_BLOCK"),
)
_EXPAND_IDX_MAPPING_LAUNCHER = FastTritonLauncher(
    _expand_idx_mapping_kernel,
    constexpr_names=("BLOCK_SIZE",),
)
_PREPARE_POS_SEQ_LENS_LAUNCHER = FastTritonLauncher(
    _prepare_pos_seq_lens_kernel,
    constexpr_names=("BLOCK_SIZE",),
)
_COMBINE_SAMPLED_AND_DRAFT_TOKENS_LAUNCHER = FastTritonLauncher(
    _combine_sampled_and_draft_tokens_kernel,
    constexpr_names=("BLOCK_SIZE", "NUM_NEW_SAMPLED_TOKENS"),
)
_PREPARE_DECODE_INPUTS_LAUNCHER = FastTritonLauncher(
    _prepare_decode_inputs_kernel,
    constexpr_names=("BLOCK_SIZE",),
    compile_kwargs={"num_warps": 2},
)
_POST_UPDATE_LAUNCHER = FastTritonLauncher(
    _post_update_kernel,
    compile_kwargs={"num_warps": 1},
)
