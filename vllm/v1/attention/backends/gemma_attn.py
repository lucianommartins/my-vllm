# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4-optimized attention backend.

Uses a custom CUDA C++ decode kernel that exploits Gemma4-specific
architectural properties (dual head_dim, k_eq_v, sliding window pruning)
and falls back to Triton prefill for non-decode requests.
"""

import os
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
from vllm.v1.attention.backends.utils import (
    compute_mm_prefix_range_tensor,
    get_kv_cache_layout,
)
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


# Split-KV scratch is pure per-step working memory: each decode op writes its
# partials and reduces them within the same call, on the same stream, before the
# next layer runs. So a single set of buffers can be shared by every attention
# layer instead of one set per layer (~60 layers on 31B -> the previously
# observed ~13GB over-allocation that OOM'd 31B on a single 80GB GPU). We key by
# the dims that actually change the shape (device, head_size, num_heads, dtype),
# so the two Gemma4 head sizes (256 sliding / 512 full) get one shared buffer
# each. Growth is monotonic so an already-captured CUDA graph never sees a freed
# buffer. This matches how Triton/FA keep a single compact workspace.
_DECODE_PARTITION_CACHE: dict[
    tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _get_decode_partition_buffers(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_partitions = max(
        1,
        min(
            SPLIT_PARTITION_CAP,
            (max_seq_len + TOKENS_PER_SPLIT - 1) // TOKENS_PER_SPLIT,
        ),
    )
    key = (device, head_size, num_heads, dtype)
    cached = _DECODE_PARTITION_CACHE.get(key)
    if (
        cached is None
        or cached[0].shape[0] < num_seqs
        or cached[0].shape[2] < max_num_partitions
    ):
        # Grow monotonically: never shrink a dimension, so buffers captured by an
        # existing CUDA graph stay valid.
        ns = num_seqs if cached is None else max(num_seqs, cached[0].shape[0])
        mp = (
            max_num_partitions
            if cached is None
            else max(max_num_partitions, cached[0].shape[2])
        )
        exp_sums = torch.zeros(
            ns, num_heads, mp, dtype=torch.float32, device=device
        )
        max_logits = torch.zeros(
            ns, num_heads, mp, dtype=torch.float32, device=device
        )
        tmp_out = torch.zeros(
            ns, num_heads, mp, head_size, dtype=dtype, device=device
        )
        cached = (exp_sums, max_logits, tmp_out)
        _DECODE_PARTITION_CACHE[key] = cached
    return cached


@dataclass
class GemmaAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    # Multimodal bidirectional ("mm-prefix") image-token spans. Field names match
    # what Gemma4ForConditionalGeneration._clear_mm_prefix_for_full_attn_layers
    # looks for (it nulls these on full-attention layers so only sliding layers
    # get bidirectional attention).
    mm_prefix_range: dict | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
    # Cascade (prefix-shared) attention: when a common prefix is shared across
    # the batch, attend to it ONCE (prefix pass) + per-request suffix + LSE
    # merge. Populated only on full-attn layers (the sliding group's builder
    # returns use_cascade_attention()->False, so common_prefix_len==0 there).
    use_cascade: bool = False
    common_prefix_len: int = 0
    num_common_kv_blocks: int = 0
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None


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

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # Decode-only cascade for now: the prefix pass treats every 1-token
        # decode query as one merged sequence vs the shared prefix. Mixed /
        # chunked-prefill cascade is a follow-up.
        query_lens = kwargs.get("query_lens")
        if query_lens is None and len(args) >= 2:
            query_lens = args[1]
        if query_lens is None or not bool((query_lens == 1).all()):
            return False
        # Cascade applies to full-attn layers only (the sliding group never
        # benefits and the prefix pass has no sliding bound). Exclude the same
        # special cases FlashAttention does.
        if (kwargs.get("use_sliding_window") or kwargs.get("use_alibi")
                or kwargs.get("use_local_attention")):
            return False
        common_prefix_len = kwargs.get("common_prefix_len")
        if common_prefix_len is None and args:
            common_prefix_len = args[0]
        # Test/debug: fire cascade for any usable shared prefix, bypassing the
        # perf heuristic (which is tuned to skip cascade at small batch).
        if os.environ.get("GEMMA_FORCE_CASCADE") == "1":
            return bool(common_prefix_len and common_prefix_len >= 256)
        # Otherwise defer the >=256-prefix / >=8-reqs / perf-model decision to
        # FlashAttention's heuristic.
        from vllm.v1.attention.backends.flash_attn import (
            use_cascade_attention as _fa_use_cascade,
        )
        return _fa_use_cascade(*args, **kwargs)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> GemmaAttentionMetadata:
        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        mm_range_tensor = None
        if mm_ranges is not None:
            mm_range_tensor = compute_mm_prefix_range_tensor(
                mm_ranges, common_attn_metadata.num_reqs,
                common_attn_metadata.seq_lens.device,
            )

        # Cascade metadata. The runner passes common_prefix_len>0 only when it
        # already decided to use cascade (use_cascade_attention above); it's 0
        # for the sliding group and non-cascade batches.
        use_cascade = common_prefix_len > 0
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        num_common_kv_blocks = 0
        if use_cascade:
            device = common_attn_metadata.seq_lens.device
            num_reqs = common_attn_metadata.num_reqs
            cu_prefix_query_lens = torch.tensor(
                [0, common_attn_metadata.num_actual_tokens],
                dtype=torch.int32, device=device,
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=device,
            )
            suffix_kv_lens = (
                common_attn_metadata.seq_lens[:num_reqs] - common_prefix_len
            ).to(torch.int32)
            num_common_kv_blocks = common_prefix_len // self.block_size

        return GemmaAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            mm_prefix_range=mm_ranges,
            mm_prefix_range_tensor=mm_range_tensor,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            num_common_kv_blocks=num_common_kv_blocks,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
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

    # The prefill kernel implements the bidirectional mm-prefix (image-token)
    # mask: within an image span attention is full, overriding causal+sliding
    # (gemma_prefill_kernel_v2). Decode queries are post-prompt text tokens never
    # inside a span, so decode stays causal. Gemma4 applies this only to sliding
    # layers; the model nulls mm_prefix_range_tensor on full-attn layers.
    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


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

    def _ensure_partition_buffers(
        self, num_seqs: int, max_seq_len: int,
        dtype: torch.dtype, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Shared across all attention layers (see _get_decode_partition_buffers).
        return _get_decode_partition_buffers(
            num_seqs, self.num_heads, self.head_size,
            max_seq_len, dtype, device,
        )

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

        if attn_metadata.use_cascade:
            return self._forward_cascade(
                layer,
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
            # mm-prefix bidirectional image-token spans (None on text-only batches
            # and on full-attn layers the model cleared) -> empty tensor = none.
            mm_ranges = attn_metadata.mm_prefix_range_tensor
            if mm_ranges is None:
                mm_ranges = torch.empty(
                    0, dtype=torch.int32, device=query.device
                )
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
                mm_ranges,
                False,  # non_causal (cascade prefix uses True)
                torch.empty(0, dtype=torch.float32, device=query.device),
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

    def _forward_cascade(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: GemmaAttentionMetadata,
    ) -> torch.Tensor:
        """Prefix-shared (cascade) decode: attend to the common prefix ONCE for
        the whole batch (non-causal prefix pass), per-request causal suffix, then
        merge via log-sum-exp. Lossless. Full (hd512) layers only — the sliding
        group never reaches here (use_cascade_attention returns False for it)."""
        from vllm.v1.attention.ops.gemma_paged_attention import (
            gemma_paged_attention,
        )
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

        num_seqs, nq, _ = query.shape
        block_size = key_cache.shape[1]
        ncb = attn_metadata.num_common_kv_blocks
        dev = query.device
        logger.info_once(
            "GEMMA_ATTN cascade active (common_prefix_len=%d, num_seqs=%d)",
            attn_metadata.common_prefix_len, num_seqs,
        )

        prefix_out = torch.empty_like(query)
        suffix_out = torch.empty_like(query)
        prefix_lse = torch.empty(nq, num_seqs, dtype=torch.float32, device=dev)
        suffix_lse = torch.empty(nq, num_seqs, dtype=torch.float32, device=dev)
        empty_mm = torch.empty(0, dtype=torch.int32, device=dev)

        # Prefix pass: all decode queries (as one merged sequence) attend to the
        # shared prefix blocks (block_table row 0), non-causal, via the prefill
        # kernel. seq_lens=[common_prefix_len], cu_q=[0, num_seqs].
        torch.ops._C.gemma_prefill_attention(
            prefix_out,
            query,
            key_cache,
            value_cache,
            self.num_kv_heads,
            self.scale,
            attn_metadata.block_table[:1],
            attn_metadata.prefix_kv_lens,
            attn_metadata.cu_prefix_query_lens,
            num_seqs,            # max_q_len: every decode token is in this seq
            block_size,
            self.k_eq_v,
            0,                   # sliding_window (full layers only here)
            empty_mm,
            True,                # non_causal
            prefix_lse,
        )

        # Suffix pass: per-request causal decode over the post-prefix KV.
        suffix_bt = attn_metadata.block_table[:, ncb:].contiguous()
        exp_sums, max_logits, tmp_out = self._ensure_partition_buffers(
            num_seqs, attn_metadata.max_seq_len, query.dtype, dev,
        )
        gemma_paged_attention(
            out=suffix_out,
            exp_sums=exp_sums,
            max_logits=max_logits,
            tmp_out=tmp_out,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            block_tables=suffix_bt,
            seq_lens=attn_metadata.suffix_kv_lens,
            block_size=block_size,
            max_seq_len=attn_metadata.max_seq_len,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            actual_head_size=self.actual_head_size,
            k_eq_v=self.k_eq_v,
            sliding_window=self.sliding_window,
            lse_out=suffix_lse,
        )

        # Lossless log-sum-exp merge of the two partial attentions.
        merge_attn_states(output, prefix_out, prefix_lse, suffix_out, suffix_lse)
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
