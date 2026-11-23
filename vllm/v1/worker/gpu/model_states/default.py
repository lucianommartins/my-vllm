# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.attention.backends.utils import compute_mm_prefix_ranges
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.rope import get_rope_state
from vllm.v1.worker.gpu.model_states.interface import (
    ModelSpecificAttnMetadata,
    ModelState,
)
from vllm.v1.worker.gpu.model_states.mm_pruning import maybe_create_mm_pruner
from vllm.v1.worker.gpu.model_states.prompt_embeds import PromptEmbedsState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class DefaultModelState(ModelState):
    supports_prompt_embeds = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)

        self.prompt_embeds_state: PromptEmbedsState | None = None
        if self.model_config.enable_prompt_embeds:
            self.prompt_embeds_state = PromptEmbedsState(
                self.max_num_reqs, self.inputs_embeds_size, self.dtype, self.device
            )
            if not self.supports_mm_inputs:
                # Persistent buffer analogous to encoder_runner.inputs_embeds.
                embeds_buffer_size = (self.max_num_tokens, self.inputs_embeds_size)
                self.inputs_embeds = torch.zeros(
                    embeds_buffer_size, dtype=self.dtype, device=self.device
                )

        self.rope_state = get_rope_state(
            self.model_config,
            model,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

        # Pruner is used for multimodal embedding pruning (EVS).
        self.mm_pruner = maybe_create_mm_pruner(
            self.model_config, model, self.rope_state, encoder_cache
        )

        # Models that can embed text input_ids and per-layer inputs directly
        # inside their compiled forward pass (e.g. Gemma4) bypass the eager
        # inputs_embeds wrapper whenever no active MM tokens are scheduled.
        self.can_use_input_ids_without_mm: bool = (
            self.supports_mm_inputs
            and getattr(model, "supports_text_only_input_ids", False)
            and vllm_config.lora_config is None
            and vllm_config.speculative_config is None
            and not self.model_config.enable_prompt_embeds
            and self.mm_pruner is None
        )
        self._mm_req_ids: set[str] = set()

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if self.supports_mm_inputs and new_req_data.mm_features:
            if any(f.modality != "prompt_embeds" for f in new_req_data.mm_features):
                self._mm_req_ids.add(new_req_data.req_id)
        if self.rope_state is not None:
            assert new_req_data.prefill_token_ids is not None
            # `prompt_embeds` is a passthrough modality with no grid info, but
            # M-RoPE assumes per-feature grids. Filter it out.
            mm_features = [
                f for f in new_req_data.mm_features if f.modality != "prompt_embeds"
            ]
            self.rope_state.init_prefill_positions(
                req_index,
                self.model,
                new_req_data.prefill_token_ids,
                mm_features=mm_features,
            )
        if self.prompt_embeds_state is not None:
            self.prompt_embeds_state.add_request(req_index, new_req_data)

    def remove_request(self, req_id: str) -> None:
        self._mm_req_ids.discard(req_id)
        if self.prompt_embeds_state is not None:
            self.prompt_embeds_state.remove_request(req_id)

    def needs_inputs_embeds_for_step(
        self,
        scheduler_output: SchedulerOutput,
        req_states: RequestState,
    ) -> bool:
        if not self.can_use_input_ids_without_mm:
            return True
        if not self._mm_req_ids:
            return False
        assert self.encoder_cache is not None
        mm_features_map = self.encoder_cache.mm_features
        req_id_to_index = req_states.req_id_to_index
        num_computed_np = req_states.num_computed_tokens_np
        for req_id, num_sched in scheduler_output.num_scheduled_tokens.items():
            if req_id not in self._mm_req_ids:
                continue
            req_idx = req_id_to_index.get(req_id)
            if req_idx is None:
                continue
            mm_features = mm_features_map.get(req_id)
            if not mm_features:
                continue
            cur_start = int(num_computed_np[req_idx])
            lo, hi = get_mm_features_in_window(
                mm_features, start=cur_start, end=cur_start + num_sched
            )
            if lo < hi:
                return True
        return False

    def apply_staged_writes(self) -> None:
        if self.rope_state is not None:
            self.rope_state.apply_staged_writes()
        if self.prompt_embeds_state is not None:
            self.prompt_embeds_state.apply_staged_writes()

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor | None:
        """Pre-allocated inputs_embeds buffer for dummy runs (contents unused)."""
        if self.can_use_input_ids_without_mm:
            return None
        if self.supports_mm_inputs:
            return self.encoder_runner.inputs_embeds[:num_tokens]
        return self.inputs_embeds[:num_tokens]

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        # Use unpadded input_ids to match is_mm_embed size (num_tokens).
        # input_batch.input_ids may be padded for CUDA graphs.
        input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]

        if self.supports_mm_inputs:
            self.execute_mm_encoder(scheduled_encoder_inputs)

            mm_embeds, is_mm_embed = super().gather_mm_embeddings(input_batch)
            if self.mm_pruner is not None and mm_embeds:
                # EVS: recompute mrope positions for pruned media.
                mm_embeds = self.mm_pruner.recompute(mm_embeds, input_batch, req_states)
                # We must flush the staged rope updates for prepare_inputs() to pick up.
                self.apply_staged_writes()

            inputs_embeds = self.encoder_runner.get_inputs_embeds(
                input_ids_unpadded, mm_embeds, is_mm_embed
            )
        else:
            input_embeddings = self.model.embed_input_ids(input_ids_unpadded)
            self.inputs_embeds[: input_embeddings.shape[0]] = input_embeddings
            inputs_embeds = self.inputs_embeds

        if self.prompt_embeds_state is not None:
            self.prompt_embeds_state.apply(
                input_batch, req_states.num_computed_tokens.gpu, inputs_embeds
            )

        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def gather_mm_embeddings(
        self, input_batch: InputBatch, draft_lookahead: int = 0
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_embeds, is_mm_embed = super().gather_mm_embeddings(
            input_batch, draft_lookahead
        )
        if self.mm_pruner is not None:
            # EVS: strip the appended mrope-position channels.
            mm_embeds = self.mm_pruner.strip(mm_embeds)
        return mm_embeds, is_mm_embed

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        if self.rope_state is None:
            return {}  # Common case (1D positions).

        self.rope_state.prepare_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        return {"positions": positions}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = {}
        if (
            self.supports_mm_inputs and not self.can_use_input_ids_without_mm
        ) or self.prompt_embeds_state is not None:
            model_inputs["inputs_embeds"] = self.dummy_inputs_embeds(num_tokens)
        if self.rope_state is not None:
            model_inputs["positions"] = self.rope_state.get_positions(num_tokens)
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            # For piecewise cudagraphs and eager, use unpadded sizes.
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        if (
            not input_batch.has_prefill
            and input_batch.num_tokens == input_batch.num_reqs
            and input_batch.num_draft_tokens == 0
        ):
            qsl_cache = getattr(self, "_uniform_qsl_cpu_cache", None)
            if qsl_cache is None:
                qsl_cache = {}
                self._uniform_qsl_cpu_cache = qsl_cache
            qkey = (input_batch.num_reqs, num_reqs)
            query_start_loc_cpu = qsl_cache.get(qkey)
            if query_start_loc_cpu is None:
                query_start_loc_cpu = torch.from_numpy(
                    input_batch.query_start_loc_np[: num_reqs + 1]
                ).clone()
                if len(qsl_cache) < 128:
                    qsl_cache[qkey] = query_start_loc_cpu
        else:
            query_start_loc_cpu = torch.from_numpy(
                input_batch.query_start_loc_np[: num_reqs + 1]
            )
        query_start_loc_gpu = input_batch.query_start_loc[: num_reqs + 1]
        max_query_len = input_batch.max_query_len
        if max_query_len is None:
            if not input_batch.has_prefill and input_batch.num_tokens == num_reqs:
                max_query_len = 1
            else:
                max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            seq_ub_np = input_batch.seq_lens_cpu_upper_bound_np
            if seq_ub_np is None:
                seq_ub_np = seq_lens_cpu_upper_bound.numpy()
            if input_batch.num_reqs == 1:
                max_seq_len = int(seq_ub_np[0])
            else:
                max_seq_len = int(seq_ub_np[: input_batch.num_reqs].max())
        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and self.model_config.is_mm_prefix_lm
        ):
            mm_features = self.encoder_cache.mm_features
            hf_text_config = self.model_config.hf_text_config
            req_doc_ranges = compute_mm_prefix_ranges(
                [mm_features.get(req_id, ()) for req_id in input_batch.req_ids],
                sliding_window=self.model_config.get_sliding_window(),
                clamp_sliding_window=getattr(
                    self.model, "mm_prefix_clamp_sliding_window", False
                )
                or getattr(hf_text_config, "mm_prefix_clamp_sliding_window", False),
                span_pad=getattr(self.model, "mm_prefix_span_leading_pad_modulus", 0),
            )
        dcp_local_seq_lens_cpu_upper_bound = None
        if input_batch.pcp is not None:
            dcp_local_seq_lens_cpu_upper_bound = (
                input_batch.pcp.dcp_local_seq_lens_cpu_upper_bound
            )
        if not input_batch.has_prefill:
            false_cache = getattr(self, "_false_bool_cache", None)
            if false_cache is None:
                false_cache = {}
                self._false_bool_cache = false_cache
            n_real = input_batch.num_reqs
            is_prefilling_t = false_cache.get(n_real)
            if is_prefilling_t is None:
                is_prefilling_t = torch.zeros(n_real, dtype=torch.bool)
                if len(false_cache) < 128:
                    false_cache[n_real] = is_prefilling_t
        else:
            is_prefilling_t = torch.from_numpy(input_batch.is_prefilling_np)
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            dcp_local_seq_lens_cpu_upper_bound=dcp_local_seq_lens_cpu_upper_bound,
            positions=input_batch.positions,
            is_prefilling=is_prefilling_t,
            mm_req_doc_ranges=req_doc_ranges,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            req_idx=input_batch.idx_mapping_np,
            ubatch_idx=ubatch_idx,
            fast_prefill=input_batch.fast_prefill,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        return attn_metadata
