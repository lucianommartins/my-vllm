# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

MAX_NUM_ALLOWED_TOKEN_IDS = 1024
MAX_NUM_LOGIT_BIAS_TOKENS = 1024
MAX_NUM_STOP_TOKEN_IDS = 128


class LogitBiasState(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        self.prompt_len_np = req_states.prompt_len.np
        max_num_reqs = req_states.max_num_reqs
        device = req_states.device

        # Allowed token IDs.
        self.num_allowed_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.allowed_token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_NUM_ALLOWED_TOKEN_IDS), dtype=torch.int32, device=device
        )
        # Logit bias.
        self.num_logit_bias = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.logit_bias_token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS), dtype=torch.int32, device=device
        )
        self.logit_bias = StagedWriteTensor(
            (max_num_reqs, MAX_NUM_LOGIT_BIAS_TOKENS),
            dtype=torch.float32,
            device=device,
        )
        # Min tokens.
        self.min_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.num_stop_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.restore_when_all_masked = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.stop_token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_NUM_STOP_TOKEN_IDS), dtype=torch.int32, device=device
        )

        # Using any of the above.
        self.use_logit_bias = np.zeros(max_num_reqs, dtype=bool)
        self._uva_dirty = 0
        self._staged_dirty = 0
        self._uva_buffers = (
            self.num_allowed_token_ids,
            self.num_logit_bias,
            self.min_lens,
            self.num_stop_token_ids,
            self.restore_when_all_masked,
        )
        self._staged_buffers = (
            self.allowed_token_ids,
            self.logit_bias_token_ids,
            self.logit_bias,
            self.stop_token_ids,
        )

    def _set_uva(self, idx: int, req_idx: int, value: int) -> None:
        buf = self._uva_buffers[idx]
        if buf.np[req_idx] != value:
            buf.np[req_idx] = value
            self._uva_dirty |= 1 << idx

    def _stage(self, idx: int, req_idx: int, values: Any) -> None:
        self._staged_buffers[idx].stage_write(req_idx, 0, values)
        self._staged_dirty |= 1 << idx

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        prompt_len = int(self.prompt_len_np[req_idx])
        # Using any logit bias.
        use_logit_bias = False

        # Allowed token IDs.
        allowed_token_ids = sampling_params.allowed_token_ids
        if allowed_token_ids:
            num_allowed_token_ids = len(allowed_token_ids)
            if num_allowed_token_ids > MAX_NUM_ALLOWED_TOKEN_IDS:
                raise ValueError(
                    f"Too many allowed token IDs: {num_allowed_token_ids}. "
                    f"The max size is {MAX_NUM_ALLOWED_TOKEN_IDS}."
                )
            self._set_uva(0, req_idx, num_allowed_token_ids)
            self._stage(0, req_idx, allowed_token_ids)
            use_logit_bias = True
        else:
            self._set_uva(0, req_idx, 0)

        # Logit bias.
        logit_bias = sampling_params.logit_bias
        if logit_bias:
            num_logit_bias = len(logit_bias)
            if num_logit_bias > MAX_NUM_LOGIT_BIAS_TOKENS:
                raise ValueError(
                    f"Too many logit bias tokens: {num_logit_bias}. "
                    f"The max size is {MAX_NUM_LOGIT_BIAS_TOKENS}."
                )
            self._set_uva(1, req_idx, num_logit_bias)
            self._stage(1, req_idx, logit_bias.keys())
            self._stage(2, req_idx, logit_bias.values())
            use_logit_bias = True
        else:
            self._set_uva(1, req_idx, 0)

        # Min tokens.
        min_tokens = sampling_params.min_tokens
        stop_token_ids = sampling_params.all_stop_token_ids
        if min_tokens > 0 and stop_token_ids:
            num_stop_token_ids = len(stop_token_ids)
            if num_stop_token_ids > MAX_NUM_STOP_TOKEN_IDS:
                raise ValueError(
                    f"Too many stop tokens: {num_stop_token_ids}. "
                    f"The max size is {MAX_NUM_STOP_TOKEN_IDS}."
                )
            self._set_uva(2, req_idx, prompt_len + min_tokens)
            self._set_uva(3, req_idx, num_stop_token_ids)
            self._stage(3, req_idx, stop_token_ids)
            self._set_uva(4, req_idx, int(sampling_params.structured_outputs is not None))
            use_logit_bias = True
        else:
            self._set_uva(3, req_idx, 0)
            self._set_uva(4, req_idx, 0)

        self.use_logit_bias[req_idx] = use_logit_bias
        return use_logit_bias

    def apply_staged_writes(self) -> None:
        uva_mask = self._uva_dirty
        if uva_mask:
            for idx, buf in enumerate(self._uva_buffers):
                if uva_mask & (1 << idx):
                    buf.copy_to_uva()
            self._uva_dirty = 0
        staged_mask = self._staged_dirty
        if staged_mask:
            for idx, buf in enumerate(self._staged_buffers):
                if staged_mask & (1 << idx):
                    buf.apply_write()
            self._staged_dirty = 0

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not np.any(self.use_logit_bias[ctx.idx_mapping_np]):
            # No request uses logit bias. Skip the kernel launch.
            return logits

        enable_stop_token_restore = bool(
            np.any(self.restore_when_all_masked.np[ctx.idx_mapping_np])
        )

        apply_logit_bias(
            logits,
            ctx.expanded_idx_mapping,
            ctx.pos,
            self.num_allowed_token_ids.gpu,
            self.allowed_token_ids.gpu,
            self.num_logit_bias.gpu,
            self.logit_bias_token_ids.gpu,
            self.logit_bias.gpu,
            self.min_lens.gpu,
            self.num_stop_token_ids.gpu,
            self.restore_when_all_masked.gpu,
            self.stop_token_ids.gpu,
            enable_stop_token_restore,
        )
        return logits


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    # Allowed token IDs.
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    # Logit bias.
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    # Min tokens.
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    restore_when_all_masked_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
    CHECK_ALL_MASKED_ROWS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < num_allowed_token_ids

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask
        )

        tl.debug_barrier()  # save must read original logits before the -inf overwrite

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + token_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        tl.debug_barrier()  # -inf overwrite must finish before restoring saved logits

        # Restore logits for allowed token IDs.
        tl.store(
            logits_ptr + token_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + token_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + token_idx * logits_stride + token_ids, logits, mask=mask)

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos + 1 < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        if CHECK_ALL_MASKED_ROWS:
            should_restore_stop_logits = tl.load(
                restore_when_all_masked_ptr + req_state_idx
            )
            if should_restore_stop_logits:
                stop_logits = tl.load(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    mask=mask,
                    other=-float("inf"),
                )

                # Save must read original logits before the -inf overwrite.
                tl.debug_barrier()

                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )

                # The -inf overwrite must be visible to the row scan below.
                tl.debug_barrier()

                row_max = tl.full((), -float("inf"), tl.float32)
                for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
                    offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
                    logits = tl.load(
                        logits_ptr + token_idx * logits_stride + offset,
                        mask=offset < vocab_size,
                        other=-float("inf"),
                    )
                    row_max = tl.maximum(row_max, tl.max(logits, axis=0))

                if row_max == -float("inf"):
                    tl.store(
                        logits_ptr + token_idx * logits_stride + stop_token_ids,
                        stop_logits,
                        mask=mask
                        & (stop_logits > -float("inf"))
                        & (stop_logits < float("inf")),
                    )
            else:
                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )
        else:
            tl.store(
                logits_ptr + token_idx * logits_stride + stop_token_ids,
                -float("inf"),
                mask=mask,
            )


def apply_logit_bias(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    restore_when_all_masked: torch.Tensor,
    stop_token_ids: torch.Tensor,
    check_all_masked_rows: bool = False,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(
        max(
            allowed_token_ids.shape[-1],
            logit_bias_token_ids.shape[-1],
            stop_token_ids.shape[-1],
        )
    )
    LOGITS_BLOCK_SIZE = 8192
    _bias_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        allowed_token_ids.stride(0),
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias_token_ids.stride(0),
        logit_bias,
        logit_bias.stride(0),
        pos,
        min_lens,
        num_stop_token_ids,
        restore_when_all_masked,
        stop_token_ids,
        stop_token_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
        CHECK_ALL_MASKED_ROWS=check_all_masked_rows,
    )
