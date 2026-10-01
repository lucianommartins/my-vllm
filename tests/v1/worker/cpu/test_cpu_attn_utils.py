# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu import attn_utils
from vllm.v1.worker.cpu.attn_utils import FastPrefillHelper

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


@pytest.mark.parametrize(
    ("layer_names", "shared_layers", "enabled", "expected"),
    [
        (["t0", "t1"], {}, True, set()),
        (
            ["t0", "t1", "t2", "t3"],
            {"t2": "t1", "t3": "t1"},
            True,
            {"t2", "t3"},
        ),
        (
            ["t0", "t1", "t2", "t3"],
            {"t1": "t0", "t3": "t0"},
            True,
            {"t3"},
        ),
        (["t0"], {"t0": "t0"}, False, set()),
    ],
)
def test_fast_prefill_selects_only_trailing_shared_layers(
    monkeypatch: pytest.MonkeyPatch,
    layer_names: list[str],
    shared_layers: dict[str, str],
    enabled: bool,
    expected: set[str],
) -> None:
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: dict.fromkeys(layer_names),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_shared_kv_cache_layers",
        lambda config: shared_layers,
    )
    config = VllmConfig()
    config.cache_config.kv_sharing_fast_prefill = enabled

    assert attn_utils.get_kv_sharing_fast_prefill_eligible_layers(config) == expected


def test_fast_prefill_uses_unpadded_cpu_logits_indices() -> None:
    helper = FastPrefillHelper()
    logits_indices = torch.tensor([3, 9], dtype=torch.int32)
    cu_num_logits = np.array([0, 1, 2], dtype=np.int32)

    assert helper.prepare(logits_indices, cu_num_logits, has_prefill=False) is None

    metadata = helper.prepare(logits_indices, cu_num_logits, has_prefill=True)

    assert metadata is not None
    assert metadata.logits_indices_padded is logits_indices
    assert metadata.num_logits_indices == 2
    assert metadata.max_logits_per_req == 1
    assert metadata.logits_cg_mode == CUDAGraphMode.NONE

    # Fallback on prompt logprobs, active LoRA, or num_logits >= num_tokens.
    assert (
        helper.prepare(
            logits_indices, cu_num_logits, has_prefill=True, needs_prompt_logprobs=True
        )
        is None
    )
    assert (
        helper.prepare(
            logits_indices, cu_num_logits, has_prefill=True, num_active_loras=1
        )
        is None
    )
    assert (
        helper.prepare(logits_indices, cu_num_logits, has_prefill=True, num_tokens=2)
        is None
    )

    empty_meta = helper.prepare(
        torch.empty(0, dtype=torch.int32),
        np.array([0, 0, 0], dtype=np.int32),
        has_prefill=True,
    )
    assert empty_meta is not None
    assert empty_meta.num_logits_indices == 0
    assert empty_meta.logits_cg_mode == CUDAGraphMode.NONE


def test_cpu_build_attn_metadata_same_group_decode_cache() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from tests.v1.core.utils import new_kv_cache_group, new_kv_cache_spec
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.attention.backends.utils import create_fast_prefill_custom_backend
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.utils import AttentionGroup

    class _BaseBackend(AttentionBackend):
        @staticmethod
        def get_name() -> str:
            return "BASE"

        @staticmethod
        def get_impl_cls():
            raise NotImplementedError

        @staticmethod
        def get_builder_cls():
            return MagicMock

    fast_backend = create_fast_prefill_custom_backend("FastPrefill", _BaseBackend)
    spec = new_kv_cache_spec(block_size=16)
    kv_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[new_kv_cache_group(["l0", "l1"], spec)],
    )

    def _make_group(backend, layers):
        builder = MagicMock()
        builder.supports_update_block_table = False
        builder.build.side_effect = lambda *a, **k: SimpleNamespace(layer=layers[0])
        group = AttentionGroup(
            backend=backend,
            layer_names=layers,
            kv_cache_spec=spec,
            kv_cache_group_id=0,
            num_heads_q=8,
        )
        group.metadata_builders = [builder]
        return group, builder

    g0, b0 = _make_group(_BaseBackend, ["l0"])
    g1, b1 = _make_group(fast_backend, ["l1"])
    qsl = torch.tensor([0, 1], dtype=torch.int32)
    meta = attn_utils.build_attn_metadata(
        attn_groups=[[g0, g1]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc=qsl,
        max_query_len=1,
        seq_lens=torch.tensor([4], dtype=torch.int32),
        max_seq_len=4,
        block_tables=[torch.zeros((1, 2), dtype=torch.int32)],
        slot_mappings=torch.zeros((1, 1), dtype=torch.int64),
        kv_cache_config=kv_config,
        fast_prefill=None,
    )
    assert meta["l0"] is meta["l1"]
    assert b0.build.call_count == 1
    assert b1.build.call_count == 0
