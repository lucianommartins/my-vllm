# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma-4 640-channel global-layer cache record writer (contract v3).

Record per (token, kv_head):  [ Vperm(512) | strip(128) ]  bf16, 1.25 KB
  Vperm  = [ V{64..256} | V{320..512} | V{0..64} | V{256..320} ]
  strip  = rope-rotated values of V's 64 pairs at the token's position:
             lo[j] = c_j*V[j]   - s_j*V[j+256]
             hi[j] = s_j*V[j]   + c_j*V[j+256]     (theta 1e6, j < 64)

The writer consumes ONLY (v, positions, slot_mapping, cos_sin_cache) —
K never exists as an input (K == w * rope64(V) with w folded into the
attention scale). cos/sin come from the SAME precomputed fp32 table the
engine uses to rope Q (RotaryEmbedding.cos_sin_cache), so strip and
query rotations are bit-consistent by construction; recomputing angles
in-kernel at fp32 loses ~1e-2 at positions ~2e5 (angle ulp), which the
G1 harness caught.
Executable spec / proofs: tests-evals/tests/gemma_attn/record640_unit.py
"""

import torch
import triton
import triton.language as tl

_NROT = 64
_HD = 512
_REC = 640
_THETA = 1_000_000.0


@triton.jit
def _write_record640_kernel(
    v_ptr,            # [num_tokens, kvh, 512] bf16 (natural channel order)
    pos_ptr,          # [num_tokens] int32/int64 positions
    slot_ptr,         # [num_tokens] int64 flat slot (block*page_size + off)
    pool_ptr,         # [blocks, page, kvh, 640] bf16 (contiguous record)
    cs_ptr,           # [max_pos, 512] fp32: [cos(256) | sin(256)] per pos
    v_stride_tok, v_stride_head,
    pool_stride_slot, pool_stride_head,
    cs_stride_pos,
    kvh: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_ptr + tok).to(tl.int64)
    # slot_mapping convention: -1 (padding) writes nowhere
    if slot < 0:
        return
    pos = tl.load(pos_ptr + tok).to(tl.int64)

    vbase = v_ptr + tok * v_stride_tok + head * v_stride_head
    obase = pool_ptr + slot * pool_stride_slot + head * pool_stride_head

    # ---- rotated-pair originals (needed for both Vperm tail and strip)
    j = tl.arange(0, 64)
    v_lo = tl.load(vbase + j).to(tl.float32)            # V[0..64)
    v_hi = tl.load(vbase + 256 + j).to(tl.float32)      # V[256..320)

    # ---- Vperm: four contiguous segment copies (192 = masked 256 range;
    # triton aranges must be powers of two)
    a = tl.arange(0, 256)
    am = a < 192
    seg0 = tl.load(vbase + 64 + a, mask=am, other=0.0)   # V[64..256)
    tl.store(obase + a, seg0, mask=am)
    seg1 = tl.load(vbase + 320 + a, mask=am, other=0.0)  # V[320..512)
    tl.store(obase + 192 + a, seg1, mask=am)
    tl.store(obase + 384 + j, v_lo.to(v_ptr.dtype.element_ty))
    tl.store(obase + 448 + j, v_hi.to(v_ptr.dtype.element_ty))

    # ---- strip: rope rotation of the 64 pairs, cos/sin from the ENGINE
    # table (bit-consistent with Q's rope; in-kernel fp32 angles lose
    # ~1e-2 at pos ~2e5 — G1-caught)
    csbase = cs_ptr + pos * cs_stride_pos
    c = tl.load(csbase + j)
    s = tl.load(csbase + 256 + j)
    lo = c * v_lo - s * v_hi
    hi = s * v_lo + c * v_hi
    tl.store(obase + 512 + j, lo.to(v_ptr.dtype.element_ty))
    tl.store(obase + 576 + j, hi.to(v_ptr.dtype.element_ty))


def write_record640(
    v: torch.Tensor,             # [num_tokens, kvh, 512] bf16
    positions: torch.Tensor,     # [num_tokens] int
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    pool: torch.Tensor,          # [blocks, page, kvh, 640] bf16
    cos_sin_cache: torch.Tensor, # [max_pos, 512] fp32 engine rope table
) -> None:
    num_tokens, kvh, hd = v.shape
    assert hd == _HD and pool.shape[-1] == _REC
    assert cos_sin_cache.dtype == torch.float32
    blocks, page = pool.shape[0], pool.shape[1]
    pool_flat = pool.view(blocks * page, kvh, _REC)
    grid = (num_tokens, kvh)
    _write_record640_kernel[grid](
        v, positions, slot_mapping, pool_flat, cos_sin_cache,
        v.stride(0), v.stride(1),
        pool_flat.stride(0), pool_flat.stride(1),
        cos_sin_cache.stride(0),
        kvh=kvh,
    )
