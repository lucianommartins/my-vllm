# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4-optimized attention backend.

Uses a custom CUDA C++ decode kernel that exploits Gemma4-specific
architectural properties (dual head_dim, k_eq_v, sliding window pruning)
and falls back to Triton prefill for non-decode requests.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

PARTITION_SIZE = 512
# Upper bound on split-KV partitions the decode kernel may use. The partition
# buffers are sized to allow ~256 tokens/split so cross-CTA split-KV can engage
# at low batch (the launcher clamps the actual num_splits to this and to the
# available KV blocks).
SPLIT_PARTITION_CAP = 128
TOKENS_PER_SPLIT = 256


@dataclass
class GemmaAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class GemmaAttentionMetadataBuilder(
    AttentionMetadataBuilder[GemmaAttentionMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.ALWAYS
    )

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
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = model_config.get_head_size()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> GemmaAttentionMetadata:
        return GemmaAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


class GemmaAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "GEMMA_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64]

    @staticmethod
    def get_impl_cls() -> type["GemmaAttentionImpl"]:
        return GemmaAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["GemmaAttentionMetadataBuilder"]:
        return GemmaAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (1, 4, 0, 2, 3, 5)
        elif cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        raise ValueError(f"Unknown cache layout: {cache_layout}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (256, 512)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


class GemmaAttentionImpl(AttentionImpl):
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
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if sliding_window is not None:
            self.sliding_window = sliding_window
            self.actual_head_size = min(head_size, 256)
            self.k_eq_v = False
        else:
            self.sliding_window = 0
            self.actual_head_size = head_size
            self.k_eq_v = False

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        hf_config = vllm_config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        config_k_eq_v = getattr(text_config, "attention_k_eq_v", False)
        global_head_dim = getattr(text_config, "global_head_dim", 0)
        local_head_dim = getattr(text_config, "head_dim", head_size)

        if sliding_window is None and config_k_eq_v and global_head_dim > 0:
            self.k_eq_v = True
            self.actual_head_size = global_head_dim
        elif sliding_window is not None and local_head_dim < head_size:
            self.actual_head_size = local_head_dim

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self._exp_sums: torch.Tensor | None = None
        self._max_logits: torch.Tensor | None = None
        self._tmp_out: torch.Tensor | None = None

    def _ensure_partition_buffers(
        self, num_seqs: int, max_seq_len: int,
        dtype: torch.dtype, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_num_partitions = max(
            1,
            min(
                SPLIT_PARTITION_CAP,
                (max_seq_len + TOKENS_PER_SPLIT - 1) // TOKENS_PER_SPLIT,
            ),
        )
        if (
            self._exp_sums is None
            or self._exp_sums.shape[0] < num_seqs
            or self._exp_sums.shape[2] < max_num_partitions
        ):
            self._exp_sums = torch.zeros(
                num_seqs, self.num_heads, max_num_partitions,
                dtype=torch.float32, device=device,
            )
            self._max_logits = torch.zeros(
                num_seqs, self.num_heads, max_num_partitions,
                dtype=torch.float32, device=device,
            )
            self._tmp_out = torch.zeros(
                num_seqs, self.num_heads, max_num_partitions, self.head_size,
                dtype=dtype, device=device,
            )
        return self._exp_sums, self._max_logits, self._tmp_out

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: GemmaAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        key_cache, value_cache = kv_cache.unbind(1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            from vllm.platforms import current_platform

            fp8_dtype = current_platform.fp8_dtype()
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)

        if attn_metadata.max_query_len > 1:
            return self._forward_prefill(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                output[:num_actual_tokens],
                attn_metadata,
            )

        return self._forward_decode(
            layer,
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            output[:num_actual_tokens],
            attn_metadata,
        )

    def _forward_prefill(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: GemmaAttentionMetadata,
    ) -> torch.Tensor:
        # Gemma4-optimized tensor-core prefill for all bf16 / non-quantized-KV
        # layers (hd=512 full and hd=256 sliding-window). fp8/fp16 fall back to
        # Triton below.
        if (
            self.head_size in (256, 512)
            and self.actual_head_size == self.head_size
            and query.dtype == torch.bfloat16
            and not is_quantized_kv_cache(self.kv_cache_dtype)
        ):
            torch.ops._C.gemma_prefill_attention(
                output,
                query,
                key_cache,
                value_cache,
                self.num_kv_heads,
                self.scale,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                attn_metadata.query_start_loc,
                attn_metadata.max_query_len,
                key_cache.shape[1],
                self.k_eq_v,
                self.sliding_window,
            )
            return output

        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )

        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=(self.sliding_window - 1, 0)
            if self.sliding_window > 0
            else (-1, -1),
            block_table=attn_metadata.block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )
        return output

    def _forward_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: GemmaAttentionMetadata,
    ) -> torch.Tensor:
        from vllm.v1.attention.ops.gemma_paged_attention import (
            gemma_paged_attention,
        )

        num_seqs = query.shape[0]
        exp_sums, max_logits, tmp_out = self._ensure_partition_buffers(
            num_seqs, attn_metadata.max_seq_len,
            query.dtype, query.device,
        )

        gemma_paged_attention(
            out=output,
            exp_sums=exp_sums,
            max_logits=max_logits,
            tmp_out=tmp_out,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            block_tables=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            block_size=key_cache.shape[1],
            max_seq_len=attn_metadata.max_seq_len,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            actual_head_size=self.actual_head_size,
            k_eq_v=self.k_eq_v,
            sliding_window=self.sliding_window,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = kv_cache.unbind(1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            from vllm.platforms import current_platform

            fp8_dtype = current_platform.fp8_dtype()
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)

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
