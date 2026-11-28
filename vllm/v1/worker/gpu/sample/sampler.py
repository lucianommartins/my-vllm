# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
    flashinfer_sampler_supported,
    xpu_sample,
    xpu_sampler_supported,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.logprob import (
    LogprobTokenIdsState,
    compute_topk_scores,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.sample.trace_replay import TraceReplayState
from vllm.v1.worker.gpu.states import RequestState


class Sampler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
        enable_trace_replay: bool = False,
        return_sampling_mask: bool = False,
        custom_logits_processors: Sequence[LogitsProcessor] = (),
    ):
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.
        self.use_fp64_gumbel = use_fp64_gumbel

        self.req_states = req_states
        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)

        lp_req_state = LogitsProcRequestState.from_request_state(req_states)
        self.penalties_state = PenaltiesState(vllm_config, lp_req_state)
        logit_bias_state = LogitBiasState(vllm_config, lp_req_state)
        bad_words_state = BadWordsState(vllm_config, lp_req_state)

        # List order is pipeline order: bias adds, penalties scale, so the
        # two do not commute.
        self.logits_processors: list[LogitsProcessor] = [
            logit_bias_state,
            self.penalties_state,
            bad_words_state,
            *custom_logits_processors,
        ]

        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)
        self.thinking_budget_state = ThinkingBudgetState(
            req_states, vllm_config.reasoning_config
        )
        self.trace_replay_state = (
            TraceReplayState(req_states) if enable_trace_replay else None
        )
        self.needs_logits_processing = np.zeros(max_num_reqs, dtype=bool)
        self.needs_pre_top_k_proc = np.zeros(max_num_reqs, dtype=bool)
        self._is_sm80_cuda = (
            current_platform.is_cuda()
            and (
                current_platform.is_device_capability((8, 0))
                or current_platform.is_device_capability((9, 0))
            )
        )
        self.num_speculative_tokens = num_speculative_tokens
        self.return_sampling_mask = return_sampling_mask
        self.use_flashinfer = (
            not return_sampling_mask and flashinfer_sampler_supported()
        )
        # The XPU kernel draws fp32 exponential noise, so it can't honor fp64.
        self.use_xpu_sampler = (
            not return_sampling_mask and not use_fp64_gumbel and xpu_sampler_supported()
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        needs_sampling = self.sampling_states.add_request(req_idx, sampling_params)
        needs_pre_proc = self.thinking_budget_state.add_request(
            req_idx, sampling_params
        )
        for processor in self.logits_processors:
            needs_pre_proc |= processor.add_request(req_idx, sampling_params)
        self.needs_pre_top_k_proc[req_idx] = needs_pre_proc
        self.needs_logits_processing[req_idx] = needs_sampling or needs_pre_proc

        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        if self.trace_replay_state is not None:
            self.trace_replay_state.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        for processor in self.logits_processors:
            processor.apply_staged_writes()
        self.thinking_budget_state.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()
        if self.trace_replay_state is not None:
            self.trace_replay_state.apply_staged_writes()

    def get_logprobs_dims(self, idx_mapping_np: np.ndarray) -> tuple[int, int] | None:
        """(num_logprobs, max_per_req_token_ids) for the given requests, or
        None when none of them want logprobs."""
        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        max_token_ids = self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np)
        if max_num_logprobs == NO_LOGPROBS and max_token_ids == 0:
            return None
        num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
        return num_logprobs, max_token_ids

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping = input_batch.idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        expanded_local_pos = input_batch.expanded_local_pos
        num_reqs = input_batch.num_reqs
        is_decode_single_logit = (
            not input_batch.has_prefill
            and input_batch.num_draft_tokens == 0
            and input_batch.num_tokens == num_reqs
            and cu_num_logits_np[-1] == num_reqs
        )
        if is_decode_single_logit:
            pos = input_batch.positions[:num_reqs]
            input_ids = input_batch.input_ids[:num_reqs]
        else:
            pos = input_batch.positions[input_batch.logits_indices]
            input_ids = input_batch.input_ids[input_batch.logits_indices]
        seq_lens_upper_bound_np = input_batch.seq_lens_cpu_upper_bound_np

        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None

        if (
            not self.sampling_states._has_any_logprobs
            and not self.logprob_token_ids_state._has_any_token_ids
        ):
            logprobs_dims = None
        else:
            logprobs_dims = self.get_logprobs_dims(idx_mapping_np)

        sampled, processed_logits = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            seq_lens_upper_bound_np,
            return_logprobs=logprobs_dims is not None,
            expanded_idx_mapping_i32=input_batch.expanded_idx_mapping_i32,
            idx_mapping_bytes=input_batch.idx_mapping_bytes,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
        )

        if self.trace_replay_state is not None:
            # Overwrite sampled tokens with the replay trace up-front so that
            # computed logprobs reflect the real distribution of the forced token.
            self.trace_replay_state.apply_trace(sampled, idx_mapping)

        if logprobs_dims is not None:
            num_logprobs, max_per_req_token_ids = logprobs_dims
            if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            logprobs_tensors = compute_topk_scores(
                logits,
                num_logprobs,
                sampled,
                cu_num_logits,
                logprob_token_ids_state=self.logprob_token_ids_state,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_per_req_token_ids,
                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
            )
        else:
            logprobs_tensors = None

        # 1 sampled token per request, except chunked-prefill requests
        # (seq_len < prefill_len) which aren't done prefilling and produce no
        # output token. num_rejected is always 0 here (one logit per request).
        if is_decode_single_logit:
            device = input_batch.seq_lens.device
            ones = getattr(self, "_decode_ones", None)
            if ones is None or ones.device != device or ones.shape[0] < num_reqs:
                cap = max(self.sampling_states.max_num_reqs, num_reqs)
                ones = torch.ones(cap, dtype=torch.int32, device=device)
                zeros = torch.zeros(cap, dtype=torch.int32, device=device)
                self._decode_ones = ones
                self._decode_zeros = zeros
            num_sampled = self._decode_ones[:num_reqs]
            num_rejected = self._decode_zeros[:num_reqs]
        else:
            num_sampled, num_rejected = get_num_sampled_and_rejected(
                input_batch.seq_lens.new_ones(input_batch.num_reqs),
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                input_batch.idx_mapping,
                self.req_states.prefill_len.gpu,
            )

        sampling_mask_tensors = None
        if self.return_sampling_mask:
            # Size by the validated top_k batch max; wider supports use the bitmask.
            max_num_kept = int(np.max(self.sampling_states.top_k.np[idx_mapping_np]))
            sampling_mask_tensors = SamplingMaskTensors.from_logits(
                processed_logits,
                input_batch.cu_num_logits,
                num_sampled,
                max_num_kept,
            )

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=sampling_mask_tensors,
            num_sampled_all_ones=is_decode_single_logit,
        )
        return sampler_output

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
        skip_top_k_top_p: bool = False,
    ) -> torch.Tensor:
        if not np.any(self.needs_logits_processing[idx_mapping_np]):
            return logits

        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        if np.any(self.needs_pre_top_k_proc[idx_mapping_np]):
            ctx = LogitsContext(
                expanded_idx_mapping=expanded_idx_mapping,
                idx_mapping=idx_mapping,
                idx_mapping_np=idx_mapping_np,
                expanded_local_pos=expanded_local_pos,
                input_ids=input_ids,
                pos=pos,
                seq_lens_upper_bound_np=seq_lens_upper_bound_np,
            )

            # Apply logits processors (native + any custom).
            for processor in self.logits_processors:
                logits = processor.apply(logits, ctx)

            # Forcing runs last so no stage can overwrite the forced end marker
            # or weaken it by scaling.
            self.thinking_budget_state.apply(logits, ctx)

        # Apply temperature in place.
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )

        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        if skip_top_k_top_p:
            return logits

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray | None,
        return_logprobs: bool = False,
        expanded_idx_mapping_i32: torch.Tensor | None = None,
        idx_mapping_bytes: bytes | None = None,
        seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._is_sm80_cuda
            and logits.is_cuda
            and logits.ndim == 2
            and logits.shape[-1] <= 262144
            and logits.dtype in (torch.bfloat16, torch.float32)
            and not self.return_sampling_mask
            and not self.use_fp64_gumbel
            and not envs.VLLM_BATCH_INVARIANT
            and not (return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
            and idx_mapping_np.size > 0
        ):
            cur_epoch = self.sampling_states.epoch
            cached_epoch = getattr(self, "_sm80_plan_epoch", -1)
            cached_idx_bytes = getattr(self, "_sm80_plan_idx_bytes", None)
            idx_bytes = (
                idx_mapping_bytes
                if idx_mapping_bytes is not None
                else idx_mapping_np.tobytes()
            )
            if cached_epoch == cur_epoch and cached_idx_bytes == idx_bytes:
                mode, max_k, do_top_p = self._sm80_plan_val
            else:
                mode = 0
                max_k = 0
                do_top_p = False
                if not np.any(self.needs_pre_top_k_proc[idx_mapping_np]) and np.all(
                    self.sampling_states.min_p.np[idx_mapping_np] == 0.0
                ):
                    temp_np = self.sampling_states.temperature.np[idx_mapping_np]
                    if np.all(temp_np == 0.0):
                        mode = 1
                    elif not np.any(temp_np == 0.0):
                        max_k = int(np.max(self.sampling_states.top_k.np[idx_mapping_np]))
                        if 1 <= max_k <= 256:
                            mode = 2
                            do_top_p = bool(
                                np.any(self.sampling_states.top_p.np[idx_mapping_np] < 1.0)
                            )
                self._sm80_plan_epoch = cur_epoch
                self._sm80_plan_idx_bytes = idx_bytes
                self._sm80_plan_val = (mode, max_k, do_top_p)

            if mode == 1:
                fn = getattr(self, "_sm80_fast_greedy", None)
                if fn is None:
                    from vllm._gemma4_sm80_ops import fast_greedy_argmax

                    fn = fast_greedy_argmax
                    self._sm80_fast_greedy = fn
                return fn(logits), logits
            if mode == 2:
                fn = getattr(self, "_sm80_fast_topk_topp", None)
                if fn is None:
                    from vllm._gemma4_sm80_ops import fast_topk_topp_gumbel_sample

                    fn = fast_topk_topp_gumbel_sample
                    self._sm80_fast_topk_topp = fn
                sampled = fn(
                    logits,
                    expanded_idx_mapping_i32
                    if expanded_idx_mapping_i32 is not None
                    else expanded_idx_mapping,
                    self.sampling_states.temperature.gpu,
                    self.sampling_states.top_k.gpu,
                    self.sampling_states.top_p.gpu if do_top_p else None,
                    self.sampling_states.seeds.gpu,
                    pos,
                    max_k,
                )
                return sampled, logits

        if seq_lens_upper_bound_np is None:
            assert seq_lens_cpu_upper_bound is not None
            seq_lens_upper_bound_np = seq_lens_cpu_upper_bound.numpy()

        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            seq_lens_upper_bound_np,
            skip_top_k_top_p=True,
        )
        top_k, top_p, has_topp_only_rows = self.sampling_states.get_top_k_top_p(
            expanded_idx_mapping, idx_mapping_np, logits.shape[-1]
        )
        # Don't use a fused sampler if no requests use top_k/top_p, if there are
        # any greedy requests or per-request seeds, or if post-processed
        # logprobs need to be returned for any requests.
        fused_sampler_eligible = not (
            (top_k is None and top_p is None)
            or (return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
            or self.sampling_states.any_greedy(idx_mapping_np)
            or self.sampling_states.any_explicit_seed(idx_mapping_np)
        )
        use_fused_sampler = (
            self.use_flashinfer or self.use_xpu_sampler
        ) and fused_sampler_eligible

        return self._sample_random(
            processed_logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            top_k,
            top_p,
            has_topp_only_rows,
            use_fused_sampler,
        )

    def _sample_random(
        self,
        processed_logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
        has_topp_only_rows: bool,
        use_fused_sampler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_fused_sampler:
            if self.use_flashinfer:
                sampled = flashinfer_sample(processed_logits, top_k, top_p).to(
                    torch.int64
                )
            else:  # Use XPU sampler
                sampled, _ = xpu_sample(processed_logits, top_k, top_p)
        else:
            processed_logits = apply_top_k_top_p(
                processed_logits, top_k, top_p, has_topp_only_rows=has_topp_only_rows
            )
            sampled = gumbel_sample(
                processed_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                is_drafting=False,
                use_fp64=self.use_fp64_gumbel,
            )
        return sampled, processed_logits
