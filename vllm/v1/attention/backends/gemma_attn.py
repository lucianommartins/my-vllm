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

# Multi-query decode threshold: extends with max_query_len <= this run as a
# single batched paged decode over virtual sequences (MTP/spec-decode verify
# shapes). Above it, the prefill paths take over.
_MQ_DECODE_MAX = int(os.environ.get("GEMMA_MQ_DECODE_MAX", "8"))

# Contract-v3 gemma-4 cache (640-ch global records + ps64 pools). Dev gate:
# readers not yet migrated; default OFF.
_CACHE_V3 = os.environ.get("GEMMA_CACHE_V3") == "1"

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
        # FP32 split partials: bf16 partials turned split-count transitions
        # into batch-wide greedy-stream forks (killed MTP draft acceptance).
        tmp_out = torch.zeros(
            ns, num_heads, mp, head_size, dtype=torch.float32, device=device
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
    # Shortest sequence in the batch, computed CPU-side once per step (avoids a
    # per-layer GPU->CPU sync in the top-k gate). 0 when unset.
    min_seq_len: int = 0
    # Host-side twins of seq_lens / query_start_loc: the custom prefill op
    # reads per-seq lengths on the CPU; passing these avoids a per-layer
    # D2H + stream sync in the SM90 launcher.
    seq_lens_cpu: torch.Tensor | None = None
    query_start_loc_cpu: torch.Tensor | None = None
    # Multi-query extend steps (MTP/spec-decode verify, tiny extends):
    # (expanded_block_table, virtual_seq_lens, max_seq_len) — one batched
    # paged decode over per-token virtual sequences, precomputed once per
    # step. When set, forward() uses the DECODE kernel (paged, split-KV,
    # no gather, q/output rows in place) instead of any prefill path.
    tiny_extend_plan: tuple | None = None
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

        # Shortest sequence (CPU-side; no GPU sync) for the top-k gate.
        min_seq_len = 0  # only needed for topk (disabled by default)

        # Small-q extend plan (multi-query / MTP-verify shapes): every query
        # token's KV is already in the paged cache before attention runs, so
        # token t = (seq i, offset o) is a plain paged decode at
        # context_i + o + 1. Tokens become VIRTUAL SEQUENCES of one batched
        # decode call: causal + sliding masks fall out of the per-virtual
        # seq_len; q/output rows are used in place (varlen order == (i, o));
        # the expanded block table is built once here. Replaces the old
        # per-offset loop (up to 4 calls/layer + index_select/index_copy).
        tiny_extend_plan = None
        mql = common_attn_metadata.max_query_len
        if 1 < mql <= _MQ_DECODE_MAX and mm_range_tensor is None:
            dev = common_attn_metadata.query_start_loc.device
            cu = common_attn_metadata.query_start_loc_cpu.to(torch.int64)
            q_lens = cu[1:] - cu[:-1]
            slc = common_attn_metadata.seq_lens_cpu.to(torch.int64)
            # spec-decode steps may pad tokens beyond cu[-1]
            # Graph-safe: the plan is just the uniform mq (an int). The
            # forward derives everything from PERSISTENT metadata
            # (block_table/seq_lens) so captured graphs replay correctly.
            # Ragged batches (mq_uniform==0) run eager -> prefill fallback.
            mq_uniform = int(mql) if bool((q_lens == mql).all()) else 0
            tiny_extend_plan = mq_uniform if mq_uniform > 1 else None

        return GemmaAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            min_seq_len=min_seq_len,
            # seq_lens_cpu is a LAZY property that syncs GPU->CPU; touch it
            # only on prefill-shaped steps (the only consumer). Fetching it
            # unconditionally cost -12% decode tok/s at L=16k (one sync per
            # decode step).
            seq_lens_cpu=(
                common_attn_metadata.seq_lens_cpu
                if common_attn_metadata.max_query_len > 1
                else None
            ),
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            tiny_extend_plan=tiny_extend_plan,
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
        if _CACHE_V3 and head_size == 512:
            # Contract-v3 640-channel single-plane record for the global
            # k_eq_v layers: [Vperm(512) | rot64 strip(128)] per token
            # (diffkv-style shape; spec side mints head_size_v=128 so
            # page bytes and this shape derive from the same fields).
            return (num_blocks, block_size, num_kv_heads, 512 + 128)
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
            # P0 numerics experiment: GEMMA_TRUE_V=1 disables V-elision so the
            # kernels read the true (v_norm'd, un-RoPE'd) V plane.
            self.k_eq_v = os.environ.get("GEMMA_TRUE_V") != "1"
            self.actual_head_size = global_head_dim
        elif sliding_window is not None and local_head_dim < head_size:
            self.actual_head_size = local_head_dim

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Lossy query-adaptive top-k (P2, Quest-style) for FULL k_eq_v layers,
        # decode only. Experimentation knobs (env), OFF (k=0) by default ->
        # full attention (lossless). k = adaptive tiles beyond the forced sink
        # + recent window. Only valid on the SIMT decode path (group<=2).
        self.topk_k = int(os.environ.get("GEMMA_TOPK_K", "0"))
        self.topk_sink = int(os.environ.get("GEMMA_TOPK_SINK", "0"))
        self.topk_window = int(os.environ.get("GEMMA_TOPK_WINDOW", "0"))
        self.topk_enabled = (
            self.topk_k > 0 and self.k_eq_v and self.num_queries_per_kv <= 2
        )
        # Bounds scoring (read maintained min/max bounds, NOT full K) = the speed
        # path; default ON when top-k is on. GEMMA_TOPK_EXACT=1 forces the
        # read-K reference path (slower; for parity/debug). block_bounds is
        # lazily allocated to [num_blocks,2,num_kv_heads,head_size] on first KV
        # write and maintained per step (full k_eq_v layers only).
        self.topk_bounds = os.environ.get("GEMMA_TOPK_EXACT", "0") != "1"
        self.block_bounds: torch.Tensor | None = None

        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        fa_ver = get_flash_attn_version(
            head_size=self.head_size,
            head_size_v=self.actual_head_size,
        )
        if fa_ver == 3:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )
            if is_fa_version_supported(4):
                fa_ver = 4
        self._fa_version = fa_ver

        # Pre-allocated empty tensors for the decode hot path (avoids
        # per-layer tensor creation in gemma_paged_attention wrapper).
        self._empty_lse: torch.Tensor | None = None
        self._mq_expand_cache: dict = {}
        # In-kernel V reconstruction (k_eq_v layers): V = unRoPE(K) * inv_w.
        # Env-gated; tensors derived lazily at first forward (pre-capture).
        self._vrecon_on = (os.environ.get("GEMMA_V_RECON") == "1"
                           and self.k_eq_v)
        # layer-name -> Attention module registry, for resolving KV-sharing
        # targets (MTP drafter layers and the target's shared tail read a
        # cache WRITTEN by their sharing target -> must use ITS w/rope).
        self._static_fwd_ctx = getattr(
            vllm_config.compilation_config, "static_forward_context", None)
        self._vrecon_if: torch.Tensor | None = None
        self._vrecon_inv_w: float = 1.0
        self._empty_f32: torch.Tensor | None = None
        self._empty_sel: torch.Tensor | None = None
        # Cached partition buffers (avoids dict lookup per layer).
        self._cached_part: tuple[torch.Tensor, ...] | None = None
        self._cached_part_ns: int = 0
        self._cached_part_mp: int = 0

    def _ensure_partition_buffers(
        self, num_seqs: int, max_seq_len: int,
        dtype: torch.dtype, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (self._cached_part is not None
                and self._cached_part_ns >= num_seqs
                and self._cached_part_mp >= max_seq_len):
            return self._cached_part  # type: ignore[return-value]
        result = _get_decode_partition_buffers(
            num_seqs, self.num_heads, self.head_size,
            max_seq_len, dtype, device,
        )
        self._cached_part = result
        self._cached_part_ns = result[0].shape[0]
        self._cached_part_mp = max_seq_len
        return result

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
        self._layer_ref = layer
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
            if attn_metadata.tiny_extend_plan is not None:
                return self._forward_tiny_extend(
                    layer, query, key_cache, value_cache, output,
                    attn_metadata)
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

    def _forward_tiny_extend(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: GemmaAttentionMetadata,
    ) -> torch.Tensor:
        """Multi-query extend as ONE batched paged decode over virtual
        sequences (token t of seq i at offset o == decode at context+o+1
        with seq i's block table). q/output rows used in place."""
        if self._empty_lse is None:
            self._empty_lse = output.new_empty(0, dtype=torch.float32)
            self._empty_sel = output.new_empty(0, dtype=torch.int32)
        mqu = attn_metadata.tiny_extend_plan
        # Packed multi-query decode is graph-safe ONLY where it reads
        # persistent metadata: hd256 sliding, GQA_GROUP(2)*mqu <= 16. The
        # kernel maps M row r -> (position r//2, head r%2) over the REAL
        # per-seq block_table/seq_lens; q/output rows are used in place.
        # Everything else (hd512 verify, mqu>8) uses the graph-safe prefill
        # path -- the fresh-tensor virtual expansion was NOT graph-safe.
        if not (
            mqu >= 2
            and self.sliding_window > 0
            and self.actual_head_size == 256
            and 2 * mqu <= 16
        ):
            if mqu >= 2 and self.actual_head_size == 512:
                # hd512 verify: virtual-seq decode via PERSISTENT buffers
                # (graph-safe: out=/copy_ only, constants cached at warmup;
                # the decode kernel beats wmma by an order at kv >> q).
                num_real = attn_metadata.seq_lens.shape[0]
                n = num_real * mqu
                key = (n, mqu, query.device)
                cached = self._mq_expand_cache.get(key)
                if cached is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "mq expand constants missing during capture")
                    seq_idx = (
                        torch.arange(n, device=query.device) // mqu
                    ).to(torch.int64)
                    offs = (
                        torch.arange(n, device=query.device) % mqu
                    ).to(torch.int32) - (mqu - 1)
                    mb = attn_metadata.block_table.shape[1]
                    bt_buf = torch.zeros(
                        n, mb, dtype=torch.int32, device=query.device)
                    sl_buf = torch.zeros(
                        n, dtype=torch.int32, device=query.device)
                    cached = (seq_idx, offs, bt_buf, sl_buf)
                    self._mq_expand_cache[key] = cached
                seq_idx, offs, bt_buf, sl_buf = cached
                torch.index_select(
                    attn_metadata.block_table, 0, seq_idx, out=bt_buf)
                torch.index_select(
                    attn_metadata.seq_lens, 0, seq_idx, out=sl_buf)
                sl_buf.add_(offs)
                exp_sums, max_logits, tmp_out = (
                    self._ensure_partition_buffers(
                        n, attn_metadata.max_seq_len,
                        query.dtype, query.device))
                torch.ops._C.gemma_paged_attention(
                    output[:n], exp_sums, max_logits, tmp_out, query[:n],
                    key_cache, value_cache, self.num_kv_heads, self.scale,
                    bt_buf, sl_buf, key_cache.shape[1],
                    attn_metadata.max_seq_len, self.kv_cache_dtype,
                    layer._k_scale, layer._v_scale, self.actual_head_size,
                    self.k_eq_v, self.sliding_window,
                    self._empty_lse, self._empty_sel,
                    *self._vrecon_args(layer, query.device),
                )
                return output
            return self._forward_prefill(
                query[: attn_metadata.num_actual_tokens],
                key_cache,
                value_cache,
                output[: attn_metadata.num_actual_tokens],
                attn_metadata,
            )
        num_real = attn_metadata.seq_lens.shape[0]
        n = num_real * mqu  # uniform: query rows = seqs * mq
        exp_sums, max_logits, tmp_out = self._ensure_partition_buffers(
            n, attn_metadata.max_seq_len, query.dtype, query.device,
        )
        torch.ops._C.gemma_paged_attention(
            output[:n],
            exp_sums,
            max_logits,
            tmp_out,
            query[:n],
            key_cache,
            value_cache,
            self.num_kv_heads,
            self.scale,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            key_cache.shape[1],
            attn_metadata.max_seq_len,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
            self.actual_head_size,
            self.k_eq_v,
            self.sliding_window,
            self._empty_lse,
            self._empty_sel,
            *self._vrecon_args(layer, query.device),
        )
        return output

    def _forward_prefill(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: GemmaAttentionMetadata,
    ) -> torch.Tensor:
        if attn_metadata.mm_prefix_range_tensor is not None:
            return self._forward_prefill_custom(
                query, key_cache, value_cache, output, attn_metadata)

        if is_quantized_kv_cache(self.kv_cache_dtype):
            return self._forward_prefill_triton(
                query, key_cache, value_cache, output, attn_metadata)

        # P7: prefill runs on our own gemma_prefill_attention op (CUTLASS
        # SM90 kernel by default on Hopper, wmma otherwise). GEMMA_PREFILL=fa4
        # is a benchmarking escape hatch only.
        if os.environ.get("GEMMA_PREFILL", "custom") != "fa4":
            return self._forward_prefill_custom(
                query, key_cache, value_cache, output, attn_metadata)

        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
        )

        n = attn_metadata.num_actual_tokens
        flash_attn_varlen_func(
            q=query[:n],
            k=key_cache,
            v=value_cache,
            out=output[:n],
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
            fa_version=self._fa_version,
        )
        return output

    def _forward_prefill_custom(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: GemmaAttentionMetadata,
    ) -> torch.Tensor:
        mm_ranges = attn_metadata.mm_prefix_range_tensor
        if mm_ranges is None:
            mm_ranges = torch.empty(
                0, dtype=torch.int32, device=query.device
            )
        empty_i32 = torch.empty(0, dtype=torch.int32)
        seq_lens_cpu = attn_metadata.seq_lens_cpu
        q_start_cpu = attn_metadata.query_start_loc_cpu
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
            False,
            torch.empty(0, dtype=torch.float32, device=query.device),
            seq_lens_cpu if seq_lens_cpu is not None else empty_i32,
            q_start_cpu if q_start_cpu is not None else empty_i32,
            *self._vrecon_args(self._layer_ref, query.device),
        )
        return output

    def _forward_prefill_triton(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: GemmaAttentionMetadata,
    ) -> torch.Tensor:
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
        num_seqs = query.shape[0]
        exp_sums, max_logits, tmp_out = self._ensure_partition_buffers(
            num_seqs, attn_metadata.max_seq_len,
            query.dtype, query.device,
        )

        # Inline topk check (avoid method call overhead on hot path).
        # topk_enabled is False by default; when off, selected_tiles stays
        # as the pre-allocated empty tensor.
        if self._empty_lse is None:
            self._empty_lse = output.new_empty(0, dtype=torch.float32)
            self._empty_sel = output.new_empty(0, dtype=torch.int32)
        selected_tiles = self._empty_sel
        if self.topk_enabled:
            sel = self._maybe_select_tiles(query, key_cache, attn_metadata)
            if sel is not None:
                selected_tiles = sel

        torch.ops._C.gemma_paged_attention(
            output,
            exp_sums,
            max_logits,
            tmp_out,
            query,
            key_cache,
            value_cache,
            self.num_kv_heads,
            self.scale,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            key_cache.shape[1],
            attn_metadata.max_seq_len,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
            self.actual_head_size,
            self.k_eq_v,
            self.sliding_window,
            self._empty_lse,
            selected_tiles,
            *self._vrecon_args(layer, query.device),
        )
        return output

    def _vrecon_args(self, layer, dev):

        if self._empty_f32 is None:
            self._empty_f32 = torch.empty(0, dtype=torch.float32, device=dev)
        if not self._vrecon_on:
            return self._empty_f32, 1.0
        if self._vrecon_if is None:
            src = layer
            tname = getattr(layer, "kv_sharing_target_layer_name", None)
            if tname and self._static_fwd_ctx is not None:
                tgt = self._static_fwd_ctx.get(tname)
                if tgt is not None:
                    src = tgt  # cache written by the sharing target
            w = getattr(src, "_gemma_k_norm_weight", None)
            base = getattr(src, "_gemma_rope_base", None)
            if w is None or base is None:
                self._vrecon_on = False
                return self._empty_f32, 1.0
            hd = self.actual_head_size
            self._vrecon_inv_w = 1.0 / float(w[0].item())
            # Gemma4 "proportional" RoPE: only rope_angles pairs rotate;
            # the remaining pairs are identity -> ZERO frequency (the
            # kernel recurrence degenerates to scale-only for f=0).
            n_ang = int(getattr(src, "_gemma_rope_angles", hd // 2))
            iv = 1.0 / (base ** (
                torch.arange(0, hd, 2, dtype=torch.float32, device=dev)
                / hd))
            iv[n_ang:] = 0.0
            self._vrecon_if = iv.contiguous()
        if os.environ.get("GEMMA_DEBUG_RECON") == "1" and \
                not getattr(self, "_dbg_printed", False):
            self._dbg_printed = True
            import sys
            print(f"[VRECON-RET] impl={id(self)&0xffff:x} "
                  f"k_eq_v={self.k_eq_v} on={self._vrecon_on} "
                  f"numel={self._vrecon_if.numel() if self._vrecon_if is not None else -1} "
                  f"inv_w={self._vrecon_inv_w:.4f} sw={self.sliding_window}",
                  file=sys.stderr, flush=True)
        return self._vrecon_if, self._vrecon_inv_w

    def _maybe_select_tiles(
        self, query: torch.Tensor, key_cache: torch.Tensor,
        attn_metadata: GemmaAttentionMetadata,
    ) -> torch.Tensor | None:
        """Build per-(seq,kv_head) top-k selected_tiles, or None for full attn.

        Returns None unless top-k is enabled AND every sequence in the batch has
        strictly more than num_sel KV tiles (so the selection is num_sel distinct
        tiles per seq; shorter batches stay lossless). Scoring reads K (the
        quality reference); maintained-bounds scoring is the speed follow-up.
        """
        if not self.topk_enabled:
            return None
        block_size = key_cache.shape[1]
        sink_tiles = (self.topk_sink + block_size - 1) // block_size
        win_tiles = (self.topk_window + block_size - 1) // block_size
        num_sel = self.topk_k + sink_tiles + win_tiles
        seq_lens = attn_metadata.seq_lens
        num_seqs = query.shape[0]
        # min_seq_len is precomputed CPU-side in the metadata builder (once per
        # step), avoiding a per-layer GPU->CPU sync in this decode hot path.
        min_seq = attn_metadata.min_seq_len
        min_tiles = (min_seq + block_size - 1) // block_size
        if min_tiles <= num_sel:
            return None  # some seq too short -> full attention (lossless)
        # Bounds scoring (no full-K read) when bounds are available, else exact.
        if self.topk_bounds and self.block_bounds is not None:
            block_bounds = self.block_bounds
        else:
            block_bounds = query.new_empty(0, dtype=torch.float32)
        selected_tiles = torch.empty(
            num_seqs, self.num_kv_heads, num_sel,
            dtype=torch.int32, device=query.device,
        )
        torch.ops._C.gemma_topk_select(
            selected_tiles, query, key_cache, block_bounds, self.scale,
            attn_metadata.block_table, seq_lens, self.num_kv_heads,
            block_size, self.kv_cache_dtype, sink_tiles, win_tiles,
        )
        return selected_tiles

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

        suffix_out = torch.empty_like(query)
        suffix_lse = torch.empty(nq, num_seqs, dtype=torch.float32, device=dev)

        # Prefix pass: all decode queries (as one merged sequence) attend to
        # the shared prefix blocks (block_table row 0), non-causal, via the
        # own prefill op (wmma path handles non_causal + LSE; the merged
        # "sequence" is small so its speed is not TTFT-critical).
        prefix_out = torch.empty_like(query)
        prefix_lse = torch.empty(
            nq, num_seqs, dtype=torch.float32, device=dev
        )
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
            num_seqs,
            block_size,
            self.k_eq_v,
            0,
            torch.empty(0, dtype=torch.int32, device=dev),
            True,
            prefix_lse,
            torch.tensor(
                [attn_metadata.common_prefix_len], dtype=torch.int32
            ),
            torch.tensor([0, num_seqs], dtype=torch.int32),
            *self._vrecon_args(self._layer_ref, query.device),
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

        # Maintain per-block min/max key bounds for bounds-scoring top-k (speed
        # path). Recompute only the blocks touched this step from the cache:
        # correct, race-free, and paged-recycle-safe (reused blocks recompute
        # from offset 0 on their first write). Full k_eq_v layers only.
        if self.topk_enabled and self.topk_bounds:
            self._update_block_bounds(key_cache, slot_mapping)

    def _update_block_bounds(
        self, key_cache: torch.Tensor, slot_mapping: torch.Tensor,
    ) -> None:
        block_size = key_cache.shape[1]
        slot = slot_mapping
        slot = slot[slot >= 0]
        if slot.numel() == 0:
            return
        if self.block_bounds is None:
            num_blocks = key_cache.shape[0]
            # bf16 bounds (== cache dtype): lossless (min/max of bf16 keys) and
            # halves the scoring read vs fp32.
            self.block_bounds = torch.zeros(
                num_blocks, 2, self.num_kv_heads, self.actual_head_size,
                dtype=key_cache.dtype, device=key_cache.device,
            )
        blocks = (slot // block_size).to(torch.int32)
        offs = (slot % block_size)
        uniq, inv = torch.unique(blocks, return_inverse=True)
        ntok = torch.zeros(uniq.shape[0], dtype=torch.int32,
                           device=key_cache.device)
        ntok.scatter_reduce_(0, inv, (offs + 1).to(torch.int32),
                             reduce="amax", include_self=True)
        torch.ops._C.gemma_update_kv_bounds(
            self.block_bounds, key_cache, uniq.to(torch.int32), ntok,
            self.num_kv_heads, block_size, self.kv_cache_dtype,
        )
