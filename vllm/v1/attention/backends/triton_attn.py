# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

from dataclasses import dataclass, replace
from typing import ClassVar

import os

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    compute_mm_prefix_range_tensor,
    get_num_attention_heads_from_layers,
    set_triton_ptr_range,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import (
    MAX_3D_QUERY_LEN,
    _get_fa2_varlen_fwd,
    _get_sm80_ext_fast,
    _run_sm80_fa2_tail,
    unified_attention,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)

logger = init_logger(__name__)


# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments
# Gemma4 checkpoints get the sliding-window work split (see builder).
_GEMMA4_MODEL_TYPES = ("gemma4", "gemma4_text", "gemma4_unified")

_FA3_FWD_CACHE = None
_FA3_FWD_INITIALIZED = False


def _get_fa3_fwd():
    global _FA3_FWD_CACHE, _FA3_FWD_INITIALIZED
    if not _FA3_FWD_INITIALIZED:
        _FA3_FWD_INITIALIZED = True
        try:
            import vllm.vllm_flash_attn  # noqa: F401
            _FA3_FWD_CACHE = torch.ops._vllm_fa3_C.fwd
        except Exception:
            _FA3_FWD_CACHE = None
    return _FA3_FWD_CACHE


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    causal: bool | torch.Tensor

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
    rswa_prefix_lens: torch.Tensor | None = None
    rswa_window: int | None = None
    # Gemma4 sliding-window groups: split the window, not the sequence,
    # across the 3D segments (unified_attention segment_sliding_window).
    segment_sliding_window: bool = False
    # Leading pure-decode request prefix count in mixed prefill+decode batches.
    num_decode_reqs: int = 0
    cu_seqlens_q_prefill: torch.Tensor | None = None


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    # Step-dependent fields reference persistent input buffers directly.
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        # Compatible with models with non-uniform per-layer head counts.
        self.num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_heads_kv = kv_cache_spec.num_kv_heads
        self.headdim = kv_cache_spec.head_size
        self.segment_sliding_window = (
            current_platform.is_cuda()
            and model_config.hf_config.model_type in _GEMMA4_MODEL_TYPES
            and getattr(kv_cache_spec, "sliding_window", None) is not None
        )
        if self.segment_sliding_window:
            logger.info_once("Triton attention: Gemma4 sliding-window split on.")

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        # On SM120, batches whose 16-segment decode grid under-fills the SMs use
        # 64. This counts requests, not Q blocks, on purpose: for 1..11 verify
        # requests (K=3/5, 8 q / 1 KV head, 2k-128k context) 64 segments were
        # faster or within 6% at 2k, and 1.1-3.6x faster from 8k up.
        self.max_seqs_64_segments = 0
        if current_platform.is_cuda() and (
            current_platform.is_device_capability((12, 0))
            # Gemma4 on A100/H100: global layers (1-2 KV heads) launch only 16-32
            # programs per sequence at 16 segments; 64 fills the SMs.
            or (
                current_platform.has_device_capability((8, 0))
                and kv_cache_spec.sliding_window is None
                and model_config.hf_config.model_type in _GEMMA4_MODEL_TYPES
            )
        ):
            self.max_seqs_64_segments = (current_platform.num_compute_units() - 1) // (
                NUM_PAR_SOFTMAX_SEGMENTS * self.num_heads_kv
            )
            if (
                current_platform.is_device_capability((8, 0))
                and self.max_seqs_64_segments > 0
            ):
                logger.info_once(
                    "Triton attention: Gemma4 64-segment decode on (max_seqs=%d).",
                    self.max_seqs_64_segments,
                )
        self.is_sm80_gemma4_sliding = (
            current_platform.is_cuda()
            and current_platform.is_device_capability((8, 0))
            and model_config.hf_config.model_type in _GEMMA4_MODEL_TYPES
            and kv_cache_spec.sliding_window is not None
            and self.num_heads_kv == 8
            and self.headdim == 256
        )
        if self.is_sm80_gemma4_sliding:
            logger.info_once(
                "Triton attention: Gemma4 wave-aware sliding 3D decode on."
            )
        headdim_padded = next_power_of_2(self.headdim)
        max_query_len_3d = 1
        speculative_config = vllm_config.speculative_config
        # The INT4 launcher runs multi-query batches in 2D only.
        if (
            speculative_config is not None
            and kv_cache_spec.kv_quant_mode != KVQuantMode.INT4_PER_TOKEN_HEAD
        ):
            num_speculative_tokens = speculative_config.num_speculative_tokens or 0
            query_len = 1 + num_speculative_tokens * (
                2 if speculative_config.parallel_drafting else 1
            )
            max_query_len_3d = min(query_len, MAX_3D_QUERY_LEN)
        # Scratch is indexed by query token, including verification tokens.
        max_num_seqs_3d = min(
            self.seq_threshold_3D, vllm_config.scheduler_config.max_num_seqs
        )
        max_num_tokens_3d = max_num_seqs_3d * max_query_len_3d
        if self.max_seqs_64_segments > 0:
            self.num_par_softmax_segments = 64
            # build() reuses this scratch at 16 segments with proportionally more rows.
            max_num_tokens_3d = max(
                min(self.max_seqs_64_segments, max_num_seqs_3d) * max_query_len_3d,
                cdiv(
                    max_num_tokens_3d,
                    self.num_par_softmax_segments // NUM_PAR_SOFTMAX_SEGMENTS,
                ),
            )
        self.softmax_segm_output = torch.empty(
            (
                max_num_tokens_3d,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (max_num_tokens_3d, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (max_num_tokens_3d, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self._segm_max_rows = self.softmax_segm_max.shape[0]
        self._segm_views: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
        if self.max_seqs_64_segments > 0 or self.is_sm80_gemma4_sliding:
            segm_dim = self.softmax_segm_output.shape[-1]
            for seg in (NUM_PAR_SOFTMAX_SEGMENTS, 8, 4):
                seg_out = self.softmax_segm_output.view(
                    -1, self.num_heads_q, seg, segm_dim
                )
                self._segm_views[seg] = (
                    seg_out,
                    self.softmax_segm_max.view(-1, self.num_heads_q, seg),
                    self.softmax_segm_expsum.view(-1, self.num_heads_q, seg),
                    seg_out.shape[0],
                )
        self.rswa_window = model_config.rswa_window
        # R-SWA restages prefix lengths in build(), which the no-op
        # update_draft_decode_metadata() does not refresh.
        self.supports_draft_decode_metadata_update = self.rswa_window is None
        self.persistent_rswa_prefix_lens: torch.Tensor | None = None
        if self.rswa_window is not None:
            self.persistent_rswa_prefix_lens = torch.empty(
                vllm_config.scheduler_config.max_num_seqs,
                dtype=torch.int32,
                device=device,
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=common_attn_metadata.causal,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
            segment_sliding_window=self.segment_sliding_window,
        )
        if self.max_seqs_64_segments > 0 and (
            self.max_seqs_64_segments < seq_lens.shape[0]
            or self._segm_max_rows < num_actual_tokens
        ):
            # The 16-segment grid fills the SMs, or the 64-segment rows cannot
            # hold every query token: same scratch, 16-segment layout.
            segments = NUM_PAR_SOFTMAX_SEGMENTS
            seg_out, seg_max, seg_expsum, _ = self._segm_views[segments]
            attn_metadata.num_par_softmax_segments = segments
            attn_metadata.softmax_segm_output = seg_out
            attn_metadata.softmax_segm_max = seg_max
            attn_metadata.softmax_segm_expsum = seg_expsum
        elif self.is_sm80_gemma4_sliding:
            n_rows = max(seq_lens.shape[0], num_actual_tokens)
            if 9 <= n_rows <= 16:
                segments = 8
                target_thresh = 16
            elif 25 <= n_rows <= 36:
                segments = 4
                target_thresh = 36
            else:
                segments = 0
                target_thresh = 16
            if segments > 0:
                seg_out, seg_max, seg_expsum, seg_rows = self._segm_views[segments]
                if seg_rows >= n_rows:
                    attn_metadata.seq_threshold_3D = target_thresh
                    attn_metadata.num_par_softmax_segments = segments
                    attn_metadata.softmax_segm_output = seg_out
                    attn_metadata.softmax_segm_max = seg_max
                    attn_metadata.softmax_segm_expsum = seg_expsum

        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if mm_ranges is not None:
            attn_metadata.mm_prefix_range = mm_ranges
            attn_metadata.mm_prefix_range_tensor = compute_mm_prefix_range_tensor(
                mm_ranges, num_reqs, seq_lens.device
            )

        rswa_prefix_lens = common_attn_metadata.rswa_prefix_lens
        if self.rswa_window is not None and rswa_prefix_lens is not None:
            assert self.persistent_rswa_prefix_lens is not None
            rswa_prefix_lens = rswa_prefix_lens.to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            persistent_prefix_lens = self.persistent_rswa_prefix_lens[:num_reqs]
            persistent_prefix_lens.copy_(rswa_prefix_lens[:num_reqs])
            attn_metadata.rswa_prefix_lens = persistent_prefix_lens
            attn_metadata.rswa_window = self.rswa_window

        if max_query_len > 1 and num_reqs > 1:
            q_loc_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
            if (
                q_loc_cpu is not None
                and q_loc_cpu.numel() >= num_reqs + 1
                and query_start_loc.shape[0] >= num_reqs + 1
                and seq_lens.shape[0] == num_reqs
            ):
                q_loc_list = q_loc_cpu[: num_reqs + 1].tolist()
                num_dec = 0
                while (
                    num_dec < num_reqs
                    and (q_loc_list[num_dec + 1] - q_loc_list[num_dec]) == 1
                ):
                    num_dec += 1
                if (
                    0 < num_dec < num_reqs
                    and q_loc_list[num_dec] == num_dec
                    and attn_metadata.softmax_segm_max.shape[0] >= num_dec
                    and all(
                        (q_loc_list[i + 1] - q_loc_list[i]) >= 1
                        for i in range(num_dec, num_reqs)
                    )
                ):
                    attn_metadata.num_decode_reqs = num_dec
                    cu_prefill = query_start_loc[num_dec : num_reqs + 1] - num_dec
                    attn_metadata.cu_seqlens_q_prefill = (
                        cu_prefill
                        if cu_prefill.dtype == torch.int32
                        else cu_prefill.to(torch.int32)
                    )

        return attn_metadata

    def update_draft_decode_metadata(self, _metadata: TritonAttentionMetadata) -> None:
        pass


class TritonAttentionBackend(AttentionBackend):
    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """Per-token-head modes pack inline fp32 scales after each head's
        data, so the content is (data + one scale) per K/V side."""
        mode = spec.kv_quant_mode
        if spec.state_content_bytes is not None or not mode.is_per_token_head:
            return spec
        hs_k, hs_v = spec.head_size, spec.head_size_v
        if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            hs_k, hs_v = hs_k // 2, hs_v // 2
        scale_bytes = get_dtype_size(torch.float32)
        content = (hs_k + hs_v) * get_dtype_size(spec.dtype) + 2 * scale_bytes
        return replace(spec, state_content_bytes=content)

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "int4_per_token_head",
        "int8_per_token_head",
        "fp8_per_token_head",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @classmethod
    def supports_rswa(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    # Per-token-head quant: scale views carved from inline head padding.
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head scale views from the padded content dimension.

        The KV cache is packed as logical shape
        ``(num_blocks, nkv, block_size, 2 * (hs + pad))`` where
        ``pad = sizeof(float32) / sizeof(cache_dtype)``.  The content dim holds
        ``[K(hs) | K_scale(pad) | V(hs) | V_scale(pad)]`` per (head, slot); the
        last ``pad`` elements of each half hold one float32 scale.  We create
        strided float32 views over those bytes.  ``kv_cache`` must be the
        packed logical tensor (call before any transpose), but may have HND or
        NHD physical strides.

        Scale shape: ``(num_blocks, block_size, num_kv_heads)``
        """
        if self._k_scale_cache is not None:
            return
        from vllm.utils.torch_utils import get_dtype_size

        num_blocks, nkv, block_size, content = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz  # e.g. 4
        padded_hs = content // 2
        hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        def to_f32_units(elements: int) -> int:
            nbytes = elements * dtype_sz
            assert nbytes % 4 == 0
            return nbytes // 4

        # Actual strides (in float32 units) from the tensor. The logical cache
        # may be physically NHD, so do not assume C-contiguous HND layout.
        strides = kv_cache.stride()
        block_f32 = to_f32_units(strides[0])
        head_f32 = to_f32_units(strides[1])
        slot_f32 = to_f32_units(strides[2])
        # Scale sits at byte offset hs within each (K, then V) content half.
        base_off_f32 = to_f32_units(kv_cache.storage_offset())
        k_scale_off_f32 = base_off_f32 + to_f32_units(hs)
        v_scale_off_f32 = base_off_f32 + to_f32_units(padded_hs + hs)

        # Not initialized: each scale is written with its token before any
        # unmasked read, and an extensible KV cache backs only a prefix of
        # num_blocks during warmup.
        # K scales (first content half)
        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=k_scale_off_f32,
        )

        # V scales (second content half)
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=v_scale_off_f32,
        )

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if current_platform.is_cuda():
            cap = current_platform.get_device_capability()
            cap_str = cap.as_version_str() if cap is not None else "unknown"
            dev = current_platform.get_device_name()
            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
                suggested = (
                    "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
                )
                raise ValueError(
                    f"FP8 KV cache is not supported by the Triton attention backend "
                    f"on {dev} (compute capability {cap_str}); native FP8 (fp8e4nv) "
                    f"requires SM89+. Re-run with --kv-cache-dtype {suggested}."
                )
            if self.kv_cache_dtype == "bfloat16" and not (
                current_platform.has_device_capability(80)
            ):
                raise ValueError(
                    f"bfloat16 KV cache is not supported on {dev} (compute capability "
                    f"{cap_str}); bfloat16 requires SM80+. Re-run with "
                    f"--kv-cache-dtype float16."
                )
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.chunk_lookback = chunk_lookback
        self.supports_quant_query_input = current_platform.is_cuda()
        if current_platform.is_rocm():
            # Enable GPUs with fp8 MMA support
            from vllm.platforms.rocm import on_gfx11, on_gfx90a

            self.supports_quant_query_input = not (on_gfx90a() or on_gfx11())

        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head

        # Enable tensor descriptors for Q/K/V load/store on platforms that
        # benefit from HW 2D block reads (Intel XPU).  The dead branch
        # is eliminated at Triton compile time, so other platforms see
        # zero cost when TD is off.
        #
        # ``VLLM_TRITON_USE_TD`` is tri-state:
        #   - unset (None): auto-select (TD on for XPU, off elsewhere),
        #   - ``1``: force TD on regardless of platform,
        #   - ``0``: force TD off regardless of platform (useful for A/B).
        td_override = envs.VLLM_TRITON_USE_TD
        if td_override is None:
            self.use_td = current_platform.is_xpu()
        else:
            self.use_td = td_override
        self._set_kv_ptr_range = current_platform.is_rocm()
        self._is_kv_quant_none = self._kv_quant_mode == KVQuantMode.NONE
        self._use_kv_views_cache = not self._set_kv_ptr_range and self._is_kv_quant_none
        self._is_encoder = self.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        )
        self._win_left, self._win_right = self.sliding_window
        self._cached_kv_obj = None
        self._cached_kv_ptr = 0
        self._cached_kv_shape = None
        self._cached_kv_key = None
        self._cached_kv_views = None
        self._cached_kv_k = None
        self._cached_kv_v = None
        is_sm80_cuda = (
            current_platform.is_cuda()
            and current_platform.has_device_capability((8, 0))
            and not self._set_kv_ptr_range
        )
        self._sm80_use_kv_update = (
            is_sm80_cuda
            and self._is_kv_quant_none
            and not self._is_encoder
        )
        self._sm80_fast_layer = (
            is_sm80_cuda
            and not envs.VLLM_BATCH_INVARIANT
            and not self.use_td
            and self._is_kv_quant_none
            and not self._is_encoder
            and (
                (self.head_size in (256, 512) and self.num_queries_per_kv == 8)
                or (self.head_size == 512 and self.num_queries_per_kv == 16 and self.num_kv_heads == 1)
            )
            and self.logits_soft_cap <= 0
            and self.alibi_slopes is None
            and not self.use_alibi_sqrt
            and self.sinks is None
            and self.chunk_lookback < 0
            and self._win_right in (-1, 0)
        )
        self._sm80_dispatch_fn = None
        self._keqv_inv_k_norm = 0.0
        self._sm80_kv_update_fn = None
        self._sm80_fa2_fwd = None
        is_sm90_cuda = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(90)
            and not self._set_kv_ptr_range
            and os.environ.get("VLLM_GEMMA4_SM90_FA3", "1") != "0"
        )
        self._sm90_fa3_fwd = _get_fa3_fwd() if is_sm90_cuda else None
        self._sm90_fa3_fast_layer = (
            self._sm90_fa3_fwd is not None
            and not envs.VLLM_BATCH_INVARIANT
            and not self.use_td
            and self._is_kv_quant_none
            and not self._is_encoder
            and self.head_size <= 256
            and self.alibi_slopes is None
            and not self.use_alibi_sqrt
            and self.sinks is None
            and self.chunk_lookback < 0
            and self._win_right in (-1, 0)
        )
        if self._sm90_fa3_fast_layer:
            logger.info_once(
                "Triton attention: SM90 Hopper FA3 direct dispatch enabled (head_size=%d, window=%s).",
                self.head_size,
                self.sliding_window,
            )
        self._sm90_sliding_decode_fn = None
        self._sm90_sliding_splitk_fn = None
        self._sm90_sliding_32w_min_bs = 12
        self._sm90_sliding_splitk_max_bs = 11
        if (
            self._sm90_fa3_fast_layer
            and self.head_size == 256
            and self.num_heads == 16
            and self.num_kv_heads == 8
            and 0 <= self._win_left <= 1024
            and self.logits_soft_cap <= 0
            and os.environ.get("VLLM_GEMMA4_SM90_SLIDING_32W", "1") != "0"
        ):
            from vllm._gemma4_sliding_decode_sm90 import (
                _get_mod,
                sliding_decode_hd256_gqa2_32w,
                sliding_decode_hd256_gqa2_splitk,
            )
            _get_mod()
            self._sm90_sliding_decode_fn = sliding_decode_hd256_gqa2_32w
            self._sm90_sliding_splitk_fn = sliding_decode_hd256_gqa2_splitk
            self._sm90_sliding_32w_min_bs = int(
                os.environ.get("VLLM_GEMMA4_SM90_SLIDING_32W_MIN_BS", "9")
            )
            self._sm90_sliding_splitk_max_bs = int(
                os.environ.get("VLLM_GEMMA4_SM90_SLIDING_SPLITK_MAX_BS", "8")
            )
            logger.info_once(
                "Triton attention: SM90a zero-spill PDL sliding decode enabled (hd=256, GQA=2, splitk<=%d, 32w>=%d).",
                self._sm90_sliding_splitk_max_bs,
                self._sm90_sliding_32w_min_bs,
            )

    def _get_kv_views(self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv_ptr = kv_cache.data_ptr()
        if (
            kv_cache is self._cached_kv_obj
            and kv_ptr == self._cached_kv_ptr
            and kv_cache.shape == self._cached_kv_shape
        ):
            return self._cached_kv_k, self._cached_kv_v
        kv_key = (
            kv_ptr,
            kv_cache.shape,
            kv_cache.stride(),
            kv_cache.dtype,
        )
        if self._cached_kv_key == kv_key:
            self._cached_kv_obj = kv_cache
            self._cached_kv_ptr = kv_ptr
            self._cached_kv_shape = kv_key[1]
            return self._cached_kv_k, self._cached_kv_v
        kv_t = kv_cache.transpose(1, 2)
        key_cache, value_cache = kv_t.split(self.head_size, dim=-1)
        self._cached_kv_obj = kv_cache
        self._cached_kv_ptr = kv_ptr
        self._cached_kv_shape = kv_key[1]
        self._cached_kv_key = kv_key
        self._cached_kv_views = (kv_key, key_cache, value_cache)
        self._cached_kv_k = key_cache
        self._cached_kv_v = value_cache
        return key_cache, value_cache

    def _try_sm80_fa2_direct(
        self,
        q_in: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        out_in: torch.Tensor,
        cu_q: torch.Tensor,
        seq_lens: torch.Tensor,
        bt: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
    ) -> bool:
        """Invoke cached _vllm_fa2_C.varlen_fwd directly when standard preconditions hold."""
        fa2_fwd = self._sm80_fa2_fwd
        num_seqs = seq_lens.shape[0]
        if (
            fa2_fwd is None
            or num_seqs <= 0
            or q_in.shape[0] <= 0
            or max_seq_len <= 0
            or not q_in.is_cuda
            or q_in.shape != out_in.shape
            or q_in.dtype is not torch.bfloat16
            or out_in.dtype is not torch.bfloat16
            or key_cache.dtype is not torch.bfloat16
            or value_cache.dtype is not torch.bfloat16
            or q_in.dim() != 3
            or q_in.stride(-1) != 1
            or out_in.stride(-1) != 1
            or key_cache.stride(-1) != 1
            or value_cache.stride(-1) != 1
            or (key_cache.shape[1] & 15) != 0
            or cu_q.dim() != 1
            or seq_lens.dim() != 1
            or bt.dim() != 2
            or cu_q.shape[0] != num_seqs + 1
            or bt.shape[0] < num_seqs
            or cu_q.dtype is not torch.int32
            or seq_lens.dtype is not torch.int32
            or bt.dtype is not torch.int32
            or not cu_q.is_contiguous()
            or not seq_lens.is_contiguous()
            or not bt.is_contiguous()
        ):
            return False
        fa2_fwd(
            q_in,
            key_cache,
            value_cache,
            out_in,
            cu_q,
            cu_q,
            seq_lens,
            None,
            bt,
            None,
            max_query_len,
            max_seq_len,
            0.0,
            self.scale,
            False,
            True,
            self.sliding_window[0],
            self.sliding_window[1],
            0.0,
            False,
            0,
            None,
        )
        return True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            layer: The attention layer, providing the q/k/v quantization scales.
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
            output: Tensor that the attention result is written into.
            output_scale: Scale for fused output quantization.
            output_block_scale: Block scale for fused output quantization;
                not supported by this backend.

        Returns:
            shape = [num_tokens, num_heads * head_size]

        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self._is_encoder:
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        q_in = query if query.shape[0] == num_actual_tokens else query[:num_actual_tokens]
        out_in = (
            output if output.shape[0] == num_actual_tokens else output[:num_actual_tokens]
        )

        # KV cache arrives in logical (B, H, N, 2*hs) order.
        # Per-token-head quantized KV cache: handled by the core unified
        # kernel, which dequantizes per-(token, head) inline via constexpr
        # branches (INT8 / FP8) and dispatches to the packed INT4 kernel.
        if self._is_per_token_head_quant:
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
            q_descale = k_descale = v_descale = None
        else:
            if self._use_kv_views_cache:
                key_cache, value_cache = self._get_kv_views(kv_cache)
            else:
                kv_t = kv_cache.transpose(1, 2)
                key_cache, value_cache = kv_t.split(self.head_size, dim=-1)
                if (
                    is_quantized_kv_cache(self.kv_cache_dtype)
                    and key_cache.dtype != self.fp8_dtype
                ):
                    key_cache = key_cache.view(self.fp8_dtype)
                    value_cache = value_cache.view(self.fp8_dtype)
                if self._set_kv_ptr_range:
                    set_triton_ptr_range(kv_t, key_cache, value_cache)

            if (
                self._sm90_fa3_fast_layer
                and output_scale is None
                and attn_metadata.mm_prefix_range_tensor is None
                and attn_metadata.rswa_prefix_lens is None
                and attn_metadata.causal is True
                and q_in.shape[0] > 0
                and attn_metadata.max_seq_len > 0
                and attn_metadata.seq_lens.shape[0] > 0
                and q_in.dtype in (torch.bfloat16, torch.float16)
                and key_cache.dtype == q_in.dtype
                and q_in.stride(-1) == 1
                and out_in.stride(-1) == 1
                and key_cache.stride(-1) == 1
                and value_cache.stride(-1) == 1
                and (key_cache.shape[1] & 15) == 0
                and (
                    attn_metadata.max_query_len > 1
                    or attn_metadata.seq_lens.shape[0] >= 4
                    or (
                        self._sm90_sliding_splitk_fn is not None
                        and attn_metadata.seq_lens.shape[0] <= self._sm90_sliding_splitk_max_bs
                    )
                )
            ):
                max_query_len = attn_metadata.max_query_len
                if (
                    max_query_len == 1
                    and self._sm90_sliding_decode_fn is not None
                    and q_in.dtype is torch.bfloat16
                    and key_cache.stride() == value_cache.stride()
                    and (key_cache.shape[1] & (key_cache.shape[1] - 1)) == 0
                    and attn_metadata.block_table.dtype is torch.int32
                    and attn_metadata.seq_lens.dtype is torch.int32
                    and attn_metadata.block_table.is_contiguous()
                    and attn_metadata.seq_lens.is_contiguous()
                ):
                    bs_cur = attn_metadata.seq_lens.shape[0]
                    if bs_cur >= self._sm90_sliding_32w_min_bs:
                        self._sm90_sliding_decode_fn(
                            q_in,
                            key_cache,
                            value_cache,
                            attn_metadata.block_table,
                            attn_metadata.seq_lens,
                            out_in,
                            self._win_left,
                            self.scale,
                        )
                        return output
                    if (
                        self._sm90_sliding_splitk_fn is not None
                        and 1 <= bs_cur <= self._sm90_sliding_splitk_max_bs
                    ):
                        self._sm90_sliding_splitk_fn(
                            q_in,
                            key_cache,
                            value_cache,
                            attn_metadata.block_table,
                            attn_metadata.seq_lens,
                            out_in,
                            self._win_left,
                            self.scale,
                        )
                        return output
                if max_query_len == 1 and 0 <= self._win_left <= 1024:
                    num_splits = 1
                    pack_gqa = True
                else:
                    num_splits = 0
                    pack_gqa = None
                self._sm90_fa3_fwd(
                    q_in,
                    key_cache,
                    value_cache,
                    None,
                    None,
                    None,
                    out_in,
                    attn_metadata.query_start_loc,
                    None,
                    None,
                    None,
                    attn_metadata.seq_lens,
                    max_query_len,
                    attn_metadata.max_seq_len,
                    attn_metadata.block_table,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self.scale,
                    True,
                    self._win_left,
                    self._win_right,
                    self.logits_soft_cap,
                    True,
                    None,
                    num_splits,
                    pack_gqa,
                    0,
                    None,
                    1,
                    0,
                    None,
                )
                return output

            if (
                self._sm80_fast_layer
                and output_scale is None
                and attn_metadata.mm_prefix_range_tensor is None
                and attn_metadata.rswa_prefix_lens is None
                and attn_metadata.causal is True
            ):
                dispatch_fn = self._sm80_dispatch_fn
                if dispatch_fn is None:
                    ext = _get_sm80_ext_fast()
                    dispatch_fn = ext.dispatch_sm80_fast_attn
                    self._sm80_dispatch_fn = dispatch_fn
                    if self._sm80_kv_update_fn is None:
                        self._sm80_kv_update_fn = ext.launch_reshape_and_cache_bf16
                    self._sm80_fa2_fwd = _get_fa2_varlen_fwd()
                num_decode_reqs = attn_metadata.num_decode_reqs
                max_query_len = attn_metadata.max_query_len
                rc = dispatch_fn(
                    out_in,
                    q_in,
                    key_cache,
                    value_cache,
                    attn_metadata.query_start_loc,
                    max_query_len,
                    attn_metadata.seq_lens,
                    self.scale,
                    self._win_left,
                    self._win_right,
                    attn_metadata.block_table,
                    attn_metadata.softmax_segm_output,
                    attn_metadata.softmax_segm_max,
                    attn_metadata.softmax_segm_expsum,
                    num_decode_reqs,
                    attn_metadata.cu_seqlens_q_prefill,
                    self._keqv_inv_k_norm,
                )
                if rc == 1:
                    return output
                if rc != 0:
                    if rc == 2:
                        num_seqs = attn_metadata.seq_lens.shape[0]
                        cu_pref = attn_metadata.cu_seqlens_q_prefill
                        if cu_pref is not None and self._try_sm80_fa2_direct(
                            q_in[num_decode_reqs:],
                            key_cache,
                            value_cache,
                            out_in[num_decode_reqs:],
                            cu_pref,
                            attn_metadata.seq_lens[num_decode_reqs:num_seqs],
                            attn_metadata.block_table[num_decode_reqs:num_seqs],
                            max_query_len,
                            attn_metadata.max_seq_len,
                        ):
                            return output
                    elif rc == 3 and self._try_sm80_fa2_direct(
                        q_in,
                        key_cache,
                        value_cache,
                        out_in,
                        attn_metadata.query_start_loc,
                        attn_metadata.seq_lens,
                        attn_metadata.block_table,
                        max_query_len,
                        attn_metadata.max_seq_len,
                    ):
                        return output
                    _run_sm80_fa2_tail(
                        rc,
                        q_in,
                        key_cache,
                        value_cache,
                        out_in,
                        attn_metadata.query_start_loc,
                        max_query_len,
                        attn_metadata.seq_lens,
                        attn_metadata.max_seq_len,
                        self.scale,
                        self.sliding_window,
                        attn_metadata.block_table,
                        num_decode_reqs,
                        attn_metadata.cu_seqlens_q_prefill,
                        fa2_fwd=self._sm80_fa2_fwd,
                    )
                    return output

            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                key_cache.shape[2],
            )
            q_descale = (
                layer._q_scale
                if (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                    and query.dtype == self.fp8_dtype
                )
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
            k_scale_cache = None
            v_scale_cache = None

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=q_in,
            k=key_cache,
            v=value_cache,
            out=out_in,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            rswa_prefix_lens=attn_metadata.rswa_prefix_lens,
            rswa_window=attn_metadata.rswa_window,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
            mm_prefix_clamp_sliding_window=getattr(
                layer, "mm_prefix_clamp_sliding_window", False
            ),
            segment_sliding_window=attn_metadata.segment_sliding_window,
            num_decode_reqs=attn_metadata.num_decode_reqs,
            cu_seqlens_q_prefill=attn_metadata.cu_seqlens_q_prefill,
        )

        return output

    def _pth_key_value_caches(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token-head K/V cache views (ensures scale caches; FP8 retyped)."""
        self._ensure_scale_caches(kv_cache)
        padded_hs = kv_cache.shape[-1] // 2
        key_cache, value_cache = kv_cache.transpose(1, 2).split(padded_hs, dim=-1)
        if self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        return key_cache, value_cache

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer

        """
        # Quantized KV cache is not supported for encoder attention.
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantized KV cache is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
            sinks=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # Reshape the input keys and values and store them in the cache.
        if self._is_per_token_head_quant:
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                value,
                key_cache,
                value_cache,
                k_scale_cache,
                v_scale_cache,
                slot_mapping,
                kv_quant_mode=self._kv_quant_mode,
            )
            return
        # For decoder and cross-attention, use KV cache as before.
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        if not self._set_kv_ptr_range and self._is_kv_quant_none:
            key_cache, value_cache = self._get_kv_views(kv_cache)
            if self._sm80_use_kv_update:
                kv_fn = self._sm80_kv_update_fn
                if kv_fn is None:
                    kv_fn = _get_sm80_ext_fast().launch_reshape_and_cache_bf16
                    self._sm80_kv_update_fn = kv_fn
                if kv_fn(key, value, key_cache, value_cache, slot_mapping):
                    return
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
            if is_quantized_kv_cache(self.kv_cache_dtype):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            if self._set_kv_ptr_range:
                set_triton_ptr_range(kv_cache, key_cache, value_cache)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        if self._is_per_token_head_quant:
            return False
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
