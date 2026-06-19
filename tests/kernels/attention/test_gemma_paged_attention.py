# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Gemma4-optimized paged attention CUDA kernel.

Compares kernel output against a PyTorch reference implementation
across all Gemma4 layer configurations.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("CUDA required", allow_module_level=True)


DTYPES = [torch.bfloat16]
BLOCK_SIZES = [16, 32]
GEMMA4_CONFIGS = [
    # (num_q_heads, num_kv_heads, head_size, actual_head_size, k_eq_v, sliding_window)
    (16, 8, 512, 256, False, 1024),   # Sliding attention, GQA 2:1
    (16, 2, 512, 512, True, 0),       # Full attention with k_eq_v, GQA 8:1
    (8, 1, 512, 256, False, 512),     # E2B sliding, GQA 8:1
    (8, 1, 512, 512, False, 0),       # Full attention without k_eq_v
    (32, 4, 512, 512, True, 0),       # 31B full, GQA 8:1
]
PARTITION_SIZE = 512


def ref_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    actual_head_size: int,
    k_eq_v: bool,
    sliding_window: int,
) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]
    # NHD layout: (num_blocks, block_size, num_kv_heads, head_size)
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    num_queries_per_kv = num_heads // num_kv_heads

    outputs = []
    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx].item()
        q = query[seq_idx]  # (num_heads, head_size)

        num_blocks = (seq_len + block_size - 1) // block_size
        block_indices = block_tables[seq_idx, :num_blocks].long()
        # Gather from NHD: (num_blocks, block_size, num_kv_heads, head_size)
        k = key_cache[block_indices].reshape(-1, num_kv_heads, head_size)
        k = k[:seq_len, :, :actual_head_size]

        if k_eq_v:
            v = k.clone()
        else:
            v = value_cache[block_indices].reshape(-1, num_kv_heads, head_size)
            v = v[:seq_len, :, :actual_head_size]

        if num_queries_per_kv > 1:
            k = k.repeat_interleave(num_queries_per_kv, dim=1)
            v = v.repeat_interleave(num_queries_per_kv, dim=1)

        q_actual = q[:, :actual_head_size]
        attn_scores = torch.einsum("hd,shd->hs", q_actual.float(), k.float())
        attn_scores *= scale

        if sliding_window > 0:
            positions = torch.arange(seq_len, device=q.device)
            query_pos = seq_len - 1
            mask = (query_pos - positions) < sliding_window
            attn_scores = attn_scores.masked_fill(
                ~mask.unsqueeze(0), float("-inf")
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum("hs,shd->hd", attn_weights.to(v.dtype), v)

        if actual_head_size < head_size:
            padding = torch.zeros(
                num_heads, head_size - actual_head_size,
                dtype=out.dtype, device=out.device,
            )
            out = torch.cat([out, padding], dim=-1)

        outputs.append(out)

    return torch.stack(outputs)


def run_cuda_kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    actual_head_size: int,
    k_eq_v: bool,
    sliding_window: int,
) -> torch.Tensor:
    """Call the actual CUDA kernel."""
    import vllm._custom_ops as ops

    num_seqs, num_heads, head_size = query.shape
    max_seq_len = seq_lens.max().item()
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]

    max_num_partitions = (max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    out = torch.zeros(
        num_seqs, num_heads, head_size,
        dtype=query.dtype, device=query.device,
    )
    exp_sums = torch.zeros(
        num_seqs, num_heads, max_num_partitions,
        dtype=torch.float32, device=query.device,
    )
    max_logits = torch.zeros(
        num_seqs, num_heads, max_num_partitions,
        dtype=torch.float32, device=query.device,
    )
    tmp_out = torch.zeros(
        num_seqs, num_heads, max_num_partitions, head_size,
        dtype=query.dtype, device=query.device,
    )
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)

    ops.gemma_paged_attention(
        out, exp_sums, max_logits, tmp_out,
        query, key_cache, value_cache,
        num_kv_heads, scale,
        block_tables, seq_lens,
        block_size, max_seq_len,
        "auto", k_scale, v_scale,
        actual_head_size, k_eq_v, sliding_window,
    )
    return out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_size,actual_head_size,k_eq_v,sliding_window",
    GEMMA4_CONFIGS,
)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [256, 1024])
def test_gemma_kernel_vs_reference(
    dtype: torch.dtype,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    actual_head_size: int,
    k_eq_v: bool,
    sliding_window: int,
    batch_size: int,
    seq_len: int,
):
    """Test CUDA kernel output matches PyTorch reference."""
    torch.manual_seed(42)
    device = "cuda"
    scale = 1.0

    num_blocks = (seq_len * batch_size) // block_size + batch_size * 2

    # Scale down random values to avoid fp16 overflow in QK dot product
    # (Gemma4 uses scale=1.0; Q/K norms keep real values small)
    init_scale = 0.01 if dtype == torch.float16 else 1.0
    query = torch.randn(
        batch_size, num_q_heads, head_size, dtype=dtype, device=device,
    ) * init_scale
    # NHD layout: (num_blocks, block_size, num_kv_heads, head_size)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size,
        dtype=dtype, device=device,
    ) * init_scale
    value_cache = key_cache.clone() if k_eq_v else torch.randn_like(key_cache) * init_scale

    max_num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    block_tables = torch.zeros(
        batch_size, max_num_blocks_per_seq, dtype=torch.int32, device=device,
    )
    block_idx = 0
    for i in range(batch_size):
        for j in range(max_num_blocks_per_seq):
            block_tables[i, j] = block_idx
            block_idx += 1

    seq_lens_tensor = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device,
    )

    ref_out = ref_paged_attention(
        query, key_cache, value_cache, block_tables, seq_lens_tensor,
        scale, actual_head_size, k_eq_v, sliding_window,
    )

    kernel_out = run_cuda_kernel(
        query, key_cache, value_cache, block_tables, seq_lens_tensor,
        scale, actual_head_size, k_eq_v, sliding_window,
    )

    # Compare only the ACTUAL_HEAD_SIZE dims (rest should be zero)
    torch.testing.assert_close(
        kernel_out[:, :, :actual_head_size],
        ref_out[:, :, :actual_head_size],
        atol=1e-2, rtol=1e-2,
    )


@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,actual_head_size,k_eq_v,sliding_window",
    [
        (16, 8, 256, False, 1024),
        (16, 2, 512, True, 0),
    ],
)
def test_sliding_window_skips_old_tokens(
    num_q_heads: int,
    num_kv_heads: int,
    actual_head_size: int,
    k_eq_v: bool,
    sliding_window: int,
):
    """Verify sliding window correctly ignores tokens outside the window."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    head_size = 512
    block_size = 16
    seq_len = 4096
    scale = 1.0

    num_blocks = seq_len // block_size + 2
    query = torch.randn(1, num_q_heads, head_size, dtype=dtype, device=device)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size,
        dtype=dtype, device=device,
    )
    value_cache = key_cache.clone() if k_eq_v else torch.randn_like(key_cache)

    max_num_blocks_per_seq = seq_len // block_size
    block_tables = torch.arange(
        max_num_blocks_per_seq, dtype=torch.int32, device=device,
    ).unsqueeze(0)
    seq_lens_tensor = torch.tensor([seq_len], dtype=torch.int32, device=device)

    ref_out = ref_paged_attention(
        query, key_cache, value_cache, block_tables, seq_lens_tensor,
        scale, actual_head_size, k_eq_v, sliding_window,
    )

    kernel_out = run_cuda_kernel(
        query, key_cache, value_cache, block_tables, seq_lens_tensor,
        scale, actual_head_size, k_eq_v, sliding_window,
    )

    torch.testing.assert_close(
        kernel_out[:, :, :actual_head_size],
        ref_out[:, :, :actual_head_size],
        atol=1e-2, rtol=1e-2,
    )

    if sliding_window > 0:
        old_blocks = (seq_len - sliding_window) // block_size
        key_cache_corrupted = key_cache.clone()
        key_cache_corrupted[:old_blocks] = 999.0
        value_cache_corrupted = (
            key_cache_corrupted.clone() if k_eq_v
            else value_cache.clone()
        )
        if not k_eq_v:
            value_cache_corrupted[:old_blocks] = 999.0

        kernel_out_corrupted = run_cuda_kernel(
            query, key_cache_corrupted, value_cache_corrupted,
            block_tables, seq_lens_tensor,
            scale, actual_head_size, k_eq_v, sliding_window,
        )
        torch.testing.assert_close(
            kernel_out[:, :, :actual_head_size],
            kernel_out_corrupted[:, :, :actual_head_size],
            atol=1e-3, rtol=1e-3,
        )
