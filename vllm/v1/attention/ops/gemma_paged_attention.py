# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python wrapper for the Gemma4-optimized paged attention CUDA kernel."""

import torch

from vllm import _custom_ops as ops


def gemma_paged_attention(
    out: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    actual_head_size: int,
    k_eq_v: bool,
    sliding_window: int,
) -> None:
    """Gemma4-optimized paged attention decode kernel.

    This kernel exploits Gemma4-specific architectural properties:
    - Dual head_dim: only loads ACTUAL_HEAD_SIZE dims from KV cache
    - k_eq_v: skips V cache reads when K==V
    - Sliding window: prunes KV blocks outside the window
    - No softcapping: no tanh-clamp overhead

    Args:
        out: Output tensor [num_seqs, num_heads, head_size]
        exp_sums: Partition exp sums [num_seqs, num_heads, max_partitions]
        max_logits: Partition max logits [num_seqs, num_heads, max_partitions]
        tmp_out: Partition outputs [num_seqs, num_heads, max_partitions, head_size]
        query: Query tensor [num_seqs, num_heads, head_size]
        key_cache: Paged K cache
        value_cache: Paged V cache
        num_kv_heads: Number of KV heads
        scale: Softmax scale (1.0 for Gemma4)
        block_tables: Block table [num_seqs, max_blocks_per_seq]
        seq_lens: Sequence lengths [num_seqs]
        block_size: KV cache page size
        max_seq_len: Maximum sequence length in batch
        kv_cache_dtype: KV cache dtype string
        k_scale: FP8 K scale
        v_scale: FP8 V scale
        actual_head_size: Real head dim (256 for sliding, 512 for full)
        k_eq_v: Whether K==V (full attention layers with k_eq_v config)
        sliding_window: Sliding window size (0 = disabled)
    """
    ops.gemma_paged_attention(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        kv_cache_dtype,
        k_scale,
        v_scale,
        actual_head_size,
        k_eq_v,
        sliding_window,
    )
