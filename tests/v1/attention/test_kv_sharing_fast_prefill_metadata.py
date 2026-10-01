# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `make_kv_sharing_fast_prefill_common_attn_metadata`.

The metadata builders read `query_start_loc_cpu` on the host as soon as the
fast-prefill metadata is built, so it must hold final values on return even
while the GPU is still busy with earlier work (async scheduling).
"""

from typing import Any

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    KVSharingFastPrefillMetadata,
    create_fast_prefill_custom_backend,
    make_kv_sharing_fast_prefill_common_attn_metadata,
)


def _make_metadata(
    query_lens: list[int], logits_per_req: list[int], device: str
) -> tuple[CommonAttentionMetadata, list[int]]:
    num_reqs = len(query_lens)
    query_start_loc = [0]
    for q in query_lens:
        query_start_loc.append(query_start_loc[-1] + q)
    # The logits of each request are its last `k` scheduled tokens.
    logits_indices: list[int] = []
    for i, k in enumerate(logits_per_req):
        end = query_start_loc[i + 1]
        logits_indices.extend(range(end - k, end))
    expected = [0]
    for k in logits_per_req:
        expected.append(expected[-1] + k)
    # Simulate CUDA graph padding of the logits indices.
    padded = logits_indices + [logits_indices[-1]] * 3 if logits_indices else []
    qsl = torch.tensor(query_start_loc, dtype=torch.int32)
    num_tokens = query_start_loc[-1]
    metadata = CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor([q + 8 for q in query_lens], dtype=torch.int32).to(
            device
        ),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(query_lens) + 8,
        block_table_tensor=torch.zeros(num_reqs, 4, dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        logits_indices_padded=torch.tensor(padded, dtype=torch.int32, device=device),
        num_logits_indices=len(logits_indices),
        max_logits_per_req=max(logits_per_req),
    )
    return metadata, expected


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "query_lens,logits_per_req",
    [
        # One logit per request (no speculative decoding), mixed chunk sizes.
        ([15, 5, 8, 300], [1, 1, 1, 1]),
        # Several logits per request (speculative decoding).
        ([15, 5, 8], [1, 2, 1]),
        ([40, 3, 17, 9], [4, 3, 1, 2]),
        # Zero logits across the batch (num_logits_indices == 0).
        ([15, 5, 8], [0, 0, 0]),
        # A request without logits: max_logits_per_req == 1 but
        # num_logits_indices != num_reqs, so `arange` would be wrong.
        ([15, 5, 8, 9], [1, 1, 1, 0]),
        # num_logits_indices == num_reqs but max_logits_per_req > 1.
        ([15, 5, 8, 9], [2, 1, 1, 0]),
    ],
)
def test_fast_prefill_query_start_loc(
    device: str, query_lens: list[int], logits_per_req: list[int]
):
    metadata, expected = _make_metadata(query_lens, logits_per_req, device)
    out = make_kv_sharing_fast_prefill_common_attn_metadata(metadata)
    assert out.query_start_loc_cpu.device.type == "cpu"
    assert out.query_start_loc_cpu.tolist() == expected
    assert out.query_start_loc.tolist() == expected
    assert out.num_reqs == len(query_lens)
    assert out.num_actual_tokens == sum(logits_per_req)
    assert out.max_query_len == max(logits_per_req)


def test_fast_prefill_preserves_aux_fields_and_clears_caches():
    metadata, _ = _make_metadata([6, 4], [1, 1], "cpu")
    metadata.causal = False
    metadata.logits_cg_mode = CUDAGraphMode.PIECEWISE
    metadata.encoder_seq_lens = torch.tensor([3, 5], dtype=torch.int32)
    metadata.encoder_seq_lens_cpu = np.array([3, 5], dtype=np.int32)
    metadata.dcp_local_seq_lens = torch.tensor([7, 6], dtype=torch.int32)
    metadata.dcp_local_seq_lens_cpu_upper_bound = torch.tensor(
        [7, 6], dtype=torch.int32
    )
    metadata.positions = torch.arange(10, dtype=torch.int64)
    metadata.is_prefilling = torch.tensor([True, False])
    metadata.seq_lens_cpu_upper_bound = torch.tensor([14, 12], dtype=torch.int32)
    metadata.req_idx = np.array([2, 5], dtype=np.intp)
    metadata.mm_req_doc_ranges = {0: [(1, 4)]}
    metadata.rswa_prefix_lens = torch.tensor([4, 2], dtype=torch.int32)
    metadata._num_computed_tokens_cache = torch.tensor([8, 8], dtype=torch.int32)
    metadata._token_to_req_indices_cache = torch.zeros(10, dtype=torch.int32)

    out = make_kv_sharing_fast_prefill_common_attn_metadata(metadata)
    assert out.causal is False
    assert out.logits_cg_mode == CUDAGraphMode.PIECEWISE
    assert out.encoder_seq_lens is metadata.encoder_seq_lens
    assert out.encoder_seq_lens_cpu is metadata.encoder_seq_lens_cpu
    assert out.dcp_local_seq_lens is metadata.dcp_local_seq_lens
    assert (
        out.dcp_local_seq_lens_cpu_upper_bound
        is metadata.dcp_local_seq_lens_cpu_upper_bound
    )
    assert out.positions is not None
    assert out.positions.tolist() == [5, 9]
    assert out.is_prefilling is metadata.is_prefilling
    assert out.seq_lens_cpu_upper_bound is metadata.seq_lens_cpu_upper_bound
    assert out.req_idx is metadata.req_idx
    assert out.mm_req_doc_ranges is None
    assert out.rswa_prefix_lens is metadata.rswa_prefix_lens
    assert out._num_computed_tokens_cache is None
    assert out._token_to_req_indices_cache is None


def test_create_fast_prefill_custom_backend_stamps_underlying_and_cg_mode():
    from dataclasses import dataclass

    @dataclass
    class _DummyMetadata(AttentionMetadata):
        num_tokens: int

    class _DummyBuilder(AttentionMetadataBuilder[_DummyMetadata]):
        def __init__(self, *args, **kwargs):
            pass

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> _DummyMetadata:
            return _DummyMetadata(num_tokens=common_attn_metadata.num_actual_tokens)

    class _DummyBackend(AttentionBackend):
        @staticmethod
        def get_name() -> str:
            return "DUMMY"

        @staticmethod
        def get_impl_cls() -> type[AttentionImpl]:
            raise NotImplementedError

        @staticmethod
        def get_builder_cls():
            return _DummyBuilder

    wrapped_backend = create_fast_prefill_custom_backend("FastPrefill", _DummyBackend)
    assert getattr(wrapped_backend, "_underlying_attn_backend", None) is _DummyBackend

    common, _ = _make_metadata([6, 4], [1, 1], "cpu")
    common.logits_cg_mode = CUDAGraphMode.PIECEWISE
    builder = wrapped_backend.get_builder_cls()(None, [], None, torch.device("cpu"))
    built = builder.build(0, common)
    assert isinstance(built, KVSharingFastPrefillMetadata)
    assert isinstance(built, _DummyMetadata)
    assert built.logits_cg_mode == CUDAGraphMode.PIECEWISE
    assert built.num_tokens == 2


def test_create_fast_prefill_custom_backend_composite_routing():
    from dataclasses import dataclass
    from types import SimpleNamespace

    from vllm.v1.attention.backends.composite import (
        MMPrefixAttentionRouting,
        create_composite_attention_backend,
    )

    @dataclass
    class _CompositeDummyMeta(AttentionMetadata):
        num_tokens: int
        variant_name: str
        _attention_backend_variant: int = 0

    class _GeneralBuilder(AttentionMetadataBuilder[_CompositeDummyMeta]):
        reorder_batch_threshold = None

        def __init__(self, *args, **kwargs):
            pass

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> _CompositeDummyMeta:
            return _CompositeDummyMeta(
                num_tokens=common_attn_metadata.num_actual_tokens,
                variant_name="general",
            )

    class _CausalBuilder(AttentionMetadataBuilder[_CompositeDummyMeta]):
        reorder_batch_threshold = None

        def __init__(self, *args, **kwargs):
            pass

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> _CompositeDummyMeta:
            return _CompositeDummyMeta(
                num_tokens=common_attn_metadata.num_actual_tokens,
                variant_name="causal",
            )

    class _GeneralImpl(AttentionImpl):
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    class _CausalImpl(AttentionImpl):
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    class _GeneralBackend(AttentionBackend):
        forward_includes_kv_cache_update = False

        @staticmethod
        def get_name() -> str:
            return "GENERAL_DUMMY"

        @staticmethod
        def get_impl_cls() -> type[AttentionImpl]:
            return _GeneralImpl

        @staticmethod
        def get_builder_cls():
            return _GeneralBuilder

    class _CausalBackend(AttentionBackend):
        forward_includes_kv_cache_update = False

        @staticmethod
        def get_name() -> str:
            return "CAUSAL_DUMMY"

        @staticmethod
        def get_impl_cls() -> type[AttentionImpl]:
            return _CausalImpl

        @staticmethod
        def get_builder_cls():
            return _CausalBuilder

    composite_backend = create_composite_attention_backend(
        _GeneralBackend,
        _CausalBackend,
        name="TestMMCompositeBackend",
        backend_name="TEST_MM_COMPOSITE",
        module=__name__,
        routing_policy=MMPrefixAttentionRouting,
    )
    wrapped_backend = create_fast_prefill_custom_backend(
        "FastPrefillComposite", composite_backend
    )
    assert getattr(wrapped_backend, "_underlying_attn_backend", None) is (
        composite_backend
    )

    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"layer_0": SimpleNamespace(sliding_window=None)}
        )
    )
    builder = wrapped_backend.get_builder_cls()(
        None, ["layer_0"], vllm_config, torch.device("cpu")
    )

    # Batch with bidirectional image tokens inside the active query chunk [8, 14)
    # routes to variant 0 (general) when fast prefill is not armed.
    non_fast, _ = _make_metadata([6, 4], [1, 1], "cpu")
    non_fast.logits_indices_padded = None
    non_fast.num_logits_indices = None
    non_fast.mm_req_doc_ranges = {0: [(8, 12)]}
    non_fast.seq_lens_cpu_upper_bound = torch.tensor([14, 12], dtype=torch.int32)
    built_non_fast = builder.build(0, non_fast)
    assert built_non_fast._attention_backend_variant == 0
    assert built_non_fast.variant_name == "general"

    # Even when the full step had mm_req_doc_ranges inside the active chunk,
    # fast-prefill clears mm_req_doc_ranges and routes to variant 1 (causal)
    # while preserving KVSharingFastPrefillMetadata fields.
    fast, _ = _make_metadata([6, 4], [1, 1], "cpu")
    fast.mm_req_doc_ranges = {0: [(8, 12)]}
    fast.seq_lens_cpu_upper_bound = torch.tensor([14, 12], dtype=torch.int32)
    fast.logits_cg_mode = CUDAGraphMode.PIECEWISE
    built_fast = builder.build(0, fast)
    impl: Any = wrapped_backend.get_impl_cls()(num_heads=1, head_size=16, scale=1.0)
    assert impl.get_impl_for_metadata(built_non_fast) is impl.general_impl
    assert impl.get_impl_for_metadata(built_fast) is impl.causal_impl

    assert isinstance(built_fast, KVSharingFastPrefillMetadata)
    assert isinstance(built_fast, _CompositeDummyMeta)
    assert built_fast._attention_backend_variant == 1
    assert built_fast.variant_name == "causal"
    assert built_fast.logits_indices_padded is fast.logits_indices_padded
    assert built_fast.num_logits_indices == 2
    assert built_fast.logits_cg_mode == CUDAGraphMode.PIECEWISE
    assert built_fast.num_tokens == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("logits_per_req", [[0, 1], [1, 0], [0, 2, 0]])
def test_fast_prefill_builder_compacts_zero_logit_reqs(
    device: str, logits_per_req: list[int]
):
    from dataclasses import dataclass

    @dataclass
    class _CapturedMetadata(AttentionMetadata):
        num_reqs: int
        query_start_loc_cpu: list[int]
        seq_lens: torch.Tensor
        block_table_tensor: torch.Tensor
        rswa_prefix_lens: torch.Tensor | None
        req_idx: list[int]

    class _CapturingBuilder(AttentionMetadataBuilder[_CapturedMetadata]):
        def __init__(self, *args, **kwargs):
            pass

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> _CapturedMetadata:
            return _CapturedMetadata(
                num_reqs=common_attn_metadata.num_reqs,
                query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu.tolist(),
                seq_lens=common_attn_metadata.seq_lens,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                rswa_prefix_lens=common_attn_metadata.rswa_prefix_lens,
                req_idx=common_attn_metadata.req_idx.tolist()
                if common_attn_metadata.req_idx is not None
                else [],
            )

    class _CapturingBackend(AttentionBackend):
        @staticmethod
        def get_name() -> str:
            return "CAPTURING"

        @staticmethod
        def get_impl_cls() -> type[AttentionImpl]:
            raise NotImplementedError

        @staticmethod
        def get_builder_cls():
            return _CapturingBuilder

    wrapped = create_fast_prefill_custom_backend("FastPrefill", _CapturingBackend)
    builder = wrapped.get_builder_cls()(None, [], None, torch.device(device))

    query_lens = [5 + i for i in range(len(logits_per_req))]
    common, expected = _make_metadata(query_lens, logits_per_req, device)
    common.cu_num_logits_np = np.array(expected, dtype=np.int32)
    # Include extra trailing padded rows in block_table_tensor (> num_reqs) to
    # verify request-padded buffers do not trigger boolean-mask IndexError.
    common.block_table_tensor = torch.tensor(
        [[10 + i, 20 + i, 0, 0] for i in range(len(logits_per_req) + 2)],
        dtype=torch.int32,
        device=device,
    )
    common.rswa_prefix_lens = torch.tensor(
        [100 + i for i in range(len(logits_per_req))],
        dtype=torch.int32,
        device=device,
    )
    common.req_idx = np.array([200 + i for i in range(len(logits_per_req))])

    if device == "cuda":
        prev_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            built = builder.build(0, common)
        finally:
            torch.cuda.set_sync_debug_mode(prev_mode)
    else:
        built = builder.build(0, common)

    assert isinstance(built, _CapturedMetadata)
    active = [i for i, k in enumerate(logits_per_req) if k > 0]
    assert built.num_reqs == len(active)
    assert built.seq_lens.tolist() == [query_lens[i] + 8 for i in active]
    assert built.block_table_tensor.tolist() == [[10 + i, 20 + i, 0, 0] for i in active]
    assert built.rswa_prefix_lens is not None
    assert built.rswa_prefix_lens.tolist() == [100 + i for i in active]
    assert built.req_idx == [200 + i for i in active]
    expected_qsl = [0]
    for i in active:
        expected_qsl.append(expected_qsl[-1] + logits_per_req[i])
    assert built.query_start_loc_cpu == expected_qsl


def test_common_attn_metadata_unpadded_preserves_fast_prefill_fields():
    metadata, _ = _make_metadata([6, 4, 3], [1, 2, 1], "cpu")
    metadata.logits_cg_mode = CUDAGraphMode.PIECEWISE
    metadata.cu_num_logits_np = np.array([0, 1, 3, 4], dtype=np.int32)

    unpadded = metadata.unpadded(num_actual_tokens=10, num_actual_reqs=2)
    assert unpadded.logits_cg_mode == CUDAGraphMode.PIECEWISE
    assert unpadded.cu_num_logits_np is not None
    assert unpadded.cu_num_logits_np.tolist() == [0, 1, 3]


def test_fast_prefill_builder_zero_logits_skips_underlying_build():
    from dataclasses import dataclass

    calls = 0

    @dataclass
    class _NeverMetadata(AttentionMetadata):
        pass

    class _NeverBuilder(AttentionMetadataBuilder[_NeverMetadata]):
        def __init__(self, *args, **kwargs):
            pass

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> _NeverMetadata:
            nonlocal calls
            calls += 1
            raise AssertionError("underlying builder must not run for 0 logits")

    class _NeverBackend(AttentionBackend):
        @staticmethod
        def get_name() -> str:
            return "NEVER"

        @staticmethod
        def get_impl_cls() -> type[AttentionImpl]:
            raise NotImplementedError

        @staticmethod
        def get_builder_cls():
            return _NeverBuilder

    wrapped = create_fast_prefill_custom_backend("FastPrefill", _NeverBackend)
    builder = wrapped.get_builder_cls()(None, [], None, torch.device("cpu"))
    common, _ = _make_metadata([6, 4], [0, 0], "cpu")
    common.logits_cg_mode = CUDAGraphMode.NONE

    built = builder.build(0, common)
    assert calls == 0
    assert isinstance(built, KVSharingFastPrefillMetadata)
    assert built.num_logits_indices == 0
    assert built.logits_cg_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize("device", DEVICES)
def test_fast_prefill_cu_num_logits_np_zero_sync(device: str, monkeypatch):
    metadata, expected = _make_metadata([15, 5, 8], [1, 2, 1], device)
    metadata.cu_num_logits_np = np.array([0, 1, 3, 4], dtype=np.int32)
    monkeypatch.setattr(
        torch,
        "bucketize",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not bucketize")),
    )
    if device == "cuda":
        prev_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = make_kv_sharing_fast_prefill_common_attn_metadata(metadata)
            assert out.query_start_loc_cpu.tolist() == expected
            torch.cuda.set_sync_debug_mode("default")
            assert out.query_start_loc.tolist() == expected
        finally:
            torch.cuda.set_sync_debug_mode(prev_mode)
    else:
        out = make_kv_sharing_fast_prefill_common_attn_metadata(metadata)
        assert out.query_start_loc_cpu.tolist() == expected
        assert out.query_start_loc.tolist() == expected


def test_gemma4_fast_prefill_helper_and_meta_device(monkeypatch):
    from types import SimpleNamespace

    import torch.nn as nn

    import vllm.model_executor.models.gemma4 as gemma4_mod
    from vllm.config import VllmConfig, set_current_vllm_config

    assert hasattr(gemma4_mod, "_is_gemma4_fast_prefill_enabled")

    def _cfg(enabled=True, pp=1, shared=2):
        return SimpleNamespace(
            cache_config=SimpleNamespace(kv_sharing_fast_prefill=enabled),
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_kv_shared_layers=shared)
            ),
        )

    assert gemma4_mod._is_gemma4_fast_prefill_enabled(_cfg(True, 1, 2)) is True
    assert gemma4_mod._is_gemma4_fast_prefill_enabled(_cfg(False, 1, 2)) is False
    assert gemma4_mod._is_gemma4_fast_prefill_enabled(_cfg(True, 2, 2)) is False
    assert gemma4_mod._is_gemma4_fast_prefill_enabled(_cfg(True, 1, 0)) is False

    # Instantiate Gemma4Model inside meta device even when device_config says cuda.
    class _MetaEmbed(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    class _MetaLinear(nn.Module):
        def __init__(self, in_f, out_f, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_f, in_f))

    class _MetaNorm(nn.Module):
        def __init__(self, dim, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(dim))

    class _MetaLayer(nn.Module):
        def __init__(self, layer_name: str):
            super().__init__()
            self.self_attn = SimpleNamespace(
                attn=SimpleNamespace(layer_name=layer_name)
            )
            self.p = nn.Parameter(torch.empty(1))

    monkeypatch.setattr(gemma4_mod, "VocabParallelEmbedding", _MetaEmbed)
    monkeypatch.setattr(gemma4_mod, "ColumnParallelLinear", _MetaLinear)
    monkeypatch.setattr(gemma4_mod, "RMSNorm", _MetaNorm)
    monkeypatch.setattr(
        gemma4_mod,
        "make_layers",
        lambda n, fn, prefix="": (
            0,
            n,
            nn.ModuleList([_MetaLayer(f"{prefix}.{i}") for i in range(n)]),
        ),
    )

    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    hf_cfg = Gemma4TextConfig(  # type: ignore[call-arg]
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        hidden_size_per_layer_input=8,
        vocab_size=64,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    vllm_cfg = VllmConfig()
    vllm_cfg.model_config = SimpleNamespace(
        hf_config=hf_cfg,
        dtype=torch.float32,
        enforce_eager=True,
    )
    vllm_cfg.cache_config.kv_sharing_fast_prefill = True
    vllm_cfg.device_config = SimpleNamespace(device=torch.device("cuda"))
    vllm_cfg.scheduler_config.max_num_batched_tokens = 16

    with set_current_vllm_config(vllm_cfg), torch.device("meta"):
        model = gemma4_mod.Gemma4Model(vllm_config=vllm_cfg, prefix="model")

    assert model.fast_prefill_enabled is True
    assert model.positions.device.type == "meta"
    assert model.hidden_states.device.type == "meta"
    assert model.per_layer_inputs is not None
    assert model.per_layer_inputs.device.type == "meta"
    assert not hasattr(model, "_cudagraph_capture_sizes")
    assert not hasattr(model, "_use_piecewise_cudagraph")


def test_gemma4_fast_prefill_forward_dispatch_and_aliasing(monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace

    import torch.nn as nn

    import vllm.model_executor.models.gemma4 as gemma4_mod
    from vllm.config import VllmConfig
    from vllm.forward_context import (
        BatchDescriptor,
        get_forward_context,
        set_forward_context,
    )

    @dataclass
    class _DummyMeta(AttentionMetadata):
        pass

    class _DummyWrapper(_DummyMeta, KVSharingFastPrefillMetadata):
        def __init__(self, logits_indices_padded, num_logits_indices, logits_cg_mode):
            self.logits_indices_padded = logits_indices_padded
            self.num_logits_indices = num_logits_indices
            self.logits_cg_mode = logits_cg_mode

    class _FakeLayer(nn.Module):
        def __init__(self, layer_name: str):
            super().__init__()
            self.self_attn = SimpleNamespace(
                attn=SimpleNamespace(layer_name=layer_name)
            )

    model = gemma4_mod.Gemma4Model.__new__(gemma4_mod.Gemma4Model)
    nn.Module.__init__(model)
    model.fast_prefill_enabled = True
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([_FakeLayer("l0"), _FakeLayer("l1")])
    model.positions = torch.zeros(16, dtype=torch.int64)
    model.hidden_states = torch.zeros((16, 4), dtype=torch.float32)
    model.per_layer_inputs = torch.zeros((16, 2, 2), dtype=torch.float32)

    recorded: list[dict] = []
    self_out_holder: list[torch.Tensor] = []

    def _self_decoder(
        input_ids=None,
        positions=None,
        inputs_embeds=None,
        per_layer_inputs=None,
        **kwargs,
    ):
        out = torch.arange(positions.size(0) * 4, dtype=torch.float32).reshape(
            positions.size(0), 4
        )
        self_out_holder.clear()
        self_out_holder.append(out)
        ple = torch.ones((positions.size(0), 2, 2), dtype=torch.float32)
        return out, ple

    def _cross_decoder(positions, hidden_states, per_layer_inputs=None, **kwargs):
        ctx = get_forward_context()
        recorded.append(
            {
                "mode": ctx.cudagraph_runtime_mode,
                "batch_desc": ctx.batch_descriptor,
                "positions_ptr": positions.data_ptr(),
                "hidden_ptr": hidden_states.data_ptr(),
                "num_tokens": positions.size(0),
            }
        )
        return hidden_states + 100.0

    model.self_decoder = _self_decoder
    model.cross_decoder = _cross_decoder
    vllm_config = VllmConfig()

    # 1. aux_hidden_state_layers bypasses fast_prefill_forward.
    model.aux_hidden_state_layers = (1,)
    called_fast = False

    def _fail_fast(*a, **k):
        nonlocal called_fast
        called_fast = True
        raise AssertionError("fast_prefill_forward should be bypassed")

    monkeypatch.setattr(model, "fast_prefill_forward", _fail_fast)
    model.layers = nn.ModuleList([])
    monkeypatch.setattr(
        model, "norm", lambda x, *a: (x, None) if a else x, raising=False
    )
    monkeypatch.setattr(
        model, "embed_input_ids", lambda ids: torch.zeros((ids.size(0), 4))
    )
    monkeypatch.setattr(model, "get_per_layer_inputs", lambda ids: None)
    monkeypatch.setattr(model, "project_per_layer_inputs", lambda h, p: None)
    monkeypatch.setattr(
        gemma4_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    out_aux = model.forward(
        torch.tensor([1, 2]), torch.tensor([0, 1]), intermediate_tensors=None
    )
    assert not called_fast
    assert isinstance(out_aux, torch.Tensor)
    monkeypatch.undo()
    model.aux_hidden_state_layers = ()
    model.layers = nn.ModuleList([_FakeLayer("l0"), _FakeLayer("l1")])

    positions = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
    input_ids = torch.zeros(6, dtype=torch.int64)
    orig_desc = BatchDescriptor(num_tokens=6, num_active_loras=0)

    # 2. logits_indices_padded is None under NONE skips static buffer copy.
    model.positions.zero_()
    recorded.clear()
    with set_forward_context(
        {"l1": object()},
        vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=orig_desc,
    ):
        model.fast_prefill_forward(input_ids, positions)
    assert recorded[0]["positions_ptr"] == positions.data_ptr()
    assert torch.all(model.positions == 0)

    # 3. num_logits_indices == 0 skips cross_decoder and returns [batch_size, H].
    recorded.clear()
    meta_zero = _DummyWrapper(torch.empty(0, dtype=torch.int32), 0, CUDAGraphMode.NONE)
    with set_forward_context(
        {"l1": meta_zero},
        vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=orig_desc,
    ):
        out_zero = model.fast_prefill_forward(input_ids, positions)
    assert len(recorded) == 0
    assert out_zero.shape == (6, 4)

    # 4. (PIECEWISE, PIECEWISE) clones self_decoder output (no aliasing),
    # switches context to PIECEWISE with padded BatchDescriptor, and restores context.
    recorded.clear()
    meta_pw = _DummyWrapper(
        torch.tensor([2, 5, 5, 5], dtype=torch.int32),
        2,
        CUDAGraphMode.PIECEWISE,
    )
    with set_forward_context(
        {"l1": meta_pw},
        vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=orig_desc,
    ):
        out_pw = model.fast_prefill_forward(input_ids, positions)
        ctx_after = get_forward_context()
        assert ctx_after.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        assert ctx_after.batch_descriptor == orig_desc
    assert out_pw.data_ptr() != self_out_holder[0].data_ptr()
    assert recorded[0]["mode"] == CUDAGraphMode.PIECEWISE
    assert recorded[0]["batch_desc"] == BatchDescriptor(
        num_tokens=4, num_active_loras=0
    )
    assert recorded[0]["positions_ptr"] == model.positions.data_ptr()
    torch.testing.assert_close(out_pw[2], self_out_holder[0][2] + 100.0)
    torch.testing.assert_close(out_pw[5], self_out_holder[0][5] + 100.0)

    # 5. (NONE, NONE) stays zero-clone and skips static buffer copy;
    # skip_compiled forces NONE.
    recorded.clear()
    with set_forward_context(
        {"l1": meta_pw},
        vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=orig_desc,
        skip_compiled=True,
    ):
        out_none = model.fast_prefill_forward(input_ids, positions)
        ctx_after = get_forward_context()
        assert ctx_after.cudagraph_runtime_mode == CUDAGraphMode.NONE
        assert ctx_after.batch_descriptor == orig_desc
    assert out_none.data_ptr() == self_out_holder[0].data_ptr()
    assert recorded[0]["mode"] == CUDAGraphMode.NONE
    assert recorded[0]["positions_ptr"] != model.positions.data_ptr()

    # 6. (PIECEWISE, NONE): PIECEWISE self_decoder output is cloned before NONE
    # cross_decoder runs so global_graph_pool is never mutated in-place.
    recorded.clear()
    meta_none = _DummyWrapper(
        torch.tensor([2, 5], dtype=torch.int32),
        2,
        CUDAGraphMode.NONE,
    )
    with set_forward_context(
        {"l1": meta_none},
        vllm_config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=orig_desc,
    ):
        out_pw_none = model.fast_prefill_forward(input_ids, positions)
        ctx_after = get_forward_context()
        assert ctx_after.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        assert ctx_after.batch_descriptor == orig_desc
    assert out_pw_none.data_ptr() != self_out_holder[0].data_ptr()
    assert recorded[0]["mode"] == CUDAGraphMode.NONE
    assert recorded[0]["positions_ptr"] != model.positions.data_ptr()
    torch.testing.assert_close(out_pw_none[2], self_out_holder[0][2] + 100.0)
    torch.testing.assert_close(out_pw_none[5], self_out_holder[0][5] + 100.0)

    # 7. _set_aux_hidden_state_layers rejects non-empty layers when fast prefill is on.
    with pytest.raises(ValueError, match="aux_hidden_state_layers"):
        model._set_aux_hidden_state_layers((1,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("logits_per_req", [[1, 1, 1, 1], [2, 1, 3, 1]])
def test_fast_prefill_query_start_loc_cpu_ready_while_gpu_busy(
    logits_per_req: list[int],
):
    """With async scheduling the CPU builds metadata while the GPU still runs
    the previous step. Keep the stream busy and read the host offsets right
    away, as the attention metadata builders do."""
    metadata, expected = _make_metadata([64, 7, 33, 12], logits_per_req, "cuda")
    torch.accelerator.synchronize()
    torch.cuda._sleep(2_000_000_000)  # keep the current stream busy
    out = make_kv_sharing_fast_prefill_common_attn_metadata(metadata)
    observed = out.query_start_loc_cpu.tolist()  # no synchronization before
    torch.accelerator.synchronize()
    assert observed == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fast_prefill_query_start_loc_cpu_under_cuda_default_device():
    """The host offsets must stay on the CPU even when the default device is
    CUDA."""
    metadata, expected = _make_metadata([15, 5, 8, 300], [1, 1, 1, 1], "cuda")
    with torch.device("cuda"):
        out = make_kv_sharing_fast_prefill_common_attn_metadata(metadata)
    assert out.query_start_loc_cpu.device.type == "cpu"
    assert out.query_start_loc_cpu.tolist() == expected
