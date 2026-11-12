# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 (NVIDIA A100) specialized CUDA C++ kernels for Gemma 4 26B-A4B:
  - Rank #1a: 3-buffer cp.async-pipelined mma.sync Paged FlashAttention-2 for hd=512 global prefill
  - Rank #3a: Cross-layer 2-warpgroup Multi-RMSNorms (0-ULP triple-BF16 router rounding)
  - Rank #3b: Deterministic (0 atomicAdd) uint4 skinny GEMVs for M <= 4 (gate_up+gelu_tanh, down, qkv, o_proj)
  - Rank #4a: Fused in-place BF16/FP32 logit softcapping (0/65,282 BF16 bit mismatches)
  - Rank #4b: Zero-sync V2 vectorized greedy argmax & 56-bit tie-free radix top-k/top-p Murmur3 Gumbel sampler
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from types import ModuleType

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>

namespace {

struct bf16x8 {
    __nv_bfloat16 v[8];
};
static_assert(sizeof(bf16x8) == 16, "bf16x8 must be 16 bytes");

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* glob_src, bool pred) {
    uint32_t s = smem_u32(smem_dst);
    int p_int = pred ? 1 : 0;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
        "}\n" :: "r"(s), "l"(glob_src), "r"(p_int));
}

__device__ __forceinline__ void cp_async_16B_unchk(void* smem_dst, const void* glob_src) {
    uint32_t s = smem_u32(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(glob_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N_WAIT>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* ptr) {
    uint32_t s = smem_u32(ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* ptr) {
    uint32_t s = smem_u32(ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float x, float y) {
    __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
    uint32_t out;
    memcpy(&out, &v, sizeof(uint32_t));
    return out;
}

__device__ __forceinline__ int swz512(int row, int chunk64) {
    return (row << 6) + (chunk64 ^ (row & 7));
}

__device__ __forceinline__ int swz64(int row, int chunk8) {
    return (row << 3) + (chunk8 ^ (row & 7));
}

__device__ __forceinline__ bool resolve_q_tile(
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ seqused_k,
    int num_seqs,
    int block_q,
    int pid0_raw,
    int grid_x,
    int& seq_idx,
    int& q_block_local,
    int& q_start,
    int& q_len,
    int& seq_len)
{
    int pid0 = (grid_x - 1) - pid0_raw;
    int acc_blocks = 0;
    for (int b = 0; b < num_seqs; ++b) {
        int qs = cu_seqlens_q[b];
        int qe = cu_seqlens_q[b + 1];
        int ql = qe - qs;
        int nb = (ql + block_q - 1) / block_q;
        if (pid0 < acc_blocks + nb) {
            seq_idx = b;
            q_block_local = pid0 - acc_blocks;
            q_start = qs;
            q_len = ql;
            seq_len = seqused_k[b];
            return true;
        }
        acc_blocks += nb;
    }
    return false;
}

// ============================================================================
// Rank #1a: SM80 3-Buffer Pipelined mma.sync Paged FA2 for hd=512 (0 spills)
// ============================================================================
template <int NUM_Q_PER_KV>
__global__ __launch_bounds__(256, 1)
void sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    int num_seqs,
    int block_size,
    int64_t bt_stride0,
    int64_t q_stride0,
    int64_t q_stride1,
    int64_t o_stride0,
    int64_t o_stride1,
    int64_t k_stride0,
    int64_t k_stride1,
    int64_t k_stride2,
    int64_t v_stride0,
    int64_t v_stride1,
    int64_t v_stride2)
{
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BLOCK_Q = BM / NUM_Q_PER_KV;
    constexpr float LOG2E = 1.4426950408889634f;

    int seq_idx, q_blk, q_start, q_len, seq_len;
    if (!resolve_q_tile(cu_seqlens_q, seqused_k, num_seqs, BLOCK_Q,
                        blockIdx.x, gridDim.x, seq_idx, q_blk, q_start, q_len, seq_len)) {
        return;
    }

    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id >> 1;
    const int warp_half = warp_id & 1;
    const int group_id = lane >> 2;
    const int quad_tid = lane & 3;

    extern __shared__ char smem_raw[];
    uint4* sQ    = reinterpret_cast<uint4*>(smem_raw);        // 65536 B
    uint4* sBuf0 = sQ + 64 * 64;                              // 32768 B
    uint4* sBuf1 = sBuf0 + 32 * 64;                           // 32768 B
    uint4* sBuf2 = sBuf1 + 32 * 64;                           // 32768 B
    float* s_red = reinterpret_cast<float*>(sBuf2 + 32 * 64); // 1024 B
    float* s_red_sum = s_red + 128;

    #pragma unroll
    for (int idx = tid; idx < 64 * 64; idx += 256) {
        int r = idx >> 6;
        int c = idx & 63;
        int q_pos = q_blk * BLOCK_Q + (r / NUM_Q_PER_KV);
        int q_h = kv_head * NUM_Q_PER_KV + (r % NUM_Q_PER_KV);
        bool ok = q_pos < q_len;
        const __nv_bfloat16* gptr = q + (int64_t)(q_start + (ok ? q_pos : 0)) * q_stride0 + (int64_t)q_h * q_stride1 + (c << 3);
        cp_async_16B(&sQ[swz512(r, c)], gptr, ok);
    }

    const int context_len = seq_len - q_len;
    const int q_max_local = min((q_blk + 1) * BLOCK_Q, q_len) - 1;
    const int max_seq_prefix = min(context_len + q_max_local + 1, seq_len);
    const int full_tiles = (context_len + q_blk * BLOCK_Q) / BN;
    const int total_tiles = (max_seq_prefix + BN - 1) / BN;
    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;
    const float scale_log2 = scale * LOG2E;

    auto issue_load_32 = [&](uint4* dst_smem, const __nv_bfloat16* cache_base,
                             int64_t s0, int64_t s1, int64_t s2, int start_tok, bool masked) {
        if (masked && start_tok >= max_seq_prefix) {
            #pragma unroll
            for (int idx = tid; idx < 32 * 64; idx += 256) {
                int r = idx >> 6;
                int c = idx & 63;
                dst_smem[swz512(r, c)] = make_uint4(0, 0, 0, 0);
            }
            return;
        }
        int page = start_tok / block_size;
        int off  = start_tok - page * block_size;
        int64_t pblk = (int64_t)bt_row[page];
        const __nv_bfloat16* ptr = cache_base + pblk * s0 + (int64_t)kv_head * s2;
        if (!masked) {
            #pragma unroll
            for (int idx = tid; idx < 32 * 64; idx += 256) {
                int r = idx >> 6;
                int c = idx & 63;
                cp_async_16B_unchk(&dst_smem[swz512(r, c)], ptr + (int64_t)(off + r) * s1 + (c << 3));
            }
        } else {
            #pragma unroll
            for (int idx = tid; idx < 32 * 64; idx += 256) {
                int r = idx >> 6;
                int c = idx & 63;
                bool ok = (start_tok + r) < max_seq_prefix;
                cp_async_16B(&dst_smem[swz512(r, c)], ptr + (int64_t)(off + (ok ? r : 0)) * s1 + (c << 3), ok);
            }
        }
    };

    uint4* bK_lo = sBuf0;
    uint4* bK_hi = sBuf1;
    uint4* bStg  = sBuf2;

    if (total_tiles > 0) {
        bool m0 = (0 >= full_tiles);
        issue_load_32(bK_lo, k_cache, k_stride0, k_stride1, k_stride2, 0, m0);
        issue_load_32(bK_hi, k_cache, k_stride0, k_stride1, k_stride2, 32, m0);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    float acc_o[32][4];
    #pragma unroll
    for (int dt = 0; dt < 32; ++dt) {
        acc_o[dt][0] = 0.f; acc_o[dt][1] = 0.f; acc_o[dt][2] = 0.f; acc_o[dt][3] = 0.f;
    }

    float row_m0 = -INFINITY, row_m1 = -INFINITY;
    float row_l0 = 0.0f, row_l1 = 0.0f;

    const int m_row0 = warp_m * 16 + group_id;
    const int m_row1 = m_row0 + 8;
    const int q_pos0 = q_blk * BLOCK_Q + (m_row0 / NUM_Q_PER_KV);
    const int q_pos1 = q_blk * BLOCK_Q + (m_row1 / NUM_Q_PER_KV);
    const bool q_valid0 = q_pos0 < q_len;
    const bool q_valid1 = q_pos1 < q_len;
    const int q_abs0 = context_len + q_pos0;
    const int q_abs1 = context_len + q_pos1;

    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col_chunk = (lane >> 4);
    const int ldm_bk_row = (lane >> 4) * 8 + (lane & 7);
    const int ldm_bk_col_chunk = ((lane >> 3) & 1);

    for (int j = 0; j < total_tiles; ++j) {
        const int start_n = j * BN;
        const bool masked_j = (j >= full_tiles);

        issue_load_32(bStg, v_cache, v_stride0, v_stride1, v_stride2, start_n, masked_j);
        cp_async_commit();

        uint4* my_sK = (warp_half == 0) ? bK_lo : bK_hi;
        float s_frag[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s_frag[nt][0] = 0.f; s_frag[nt][1] = 0.f; s_frag[nt][2] = 0.f; s_frag[nt][3] = 0.f;
        }
        #pragma unroll 4
        for (int ks = 0; ks < 32; ++ks) {
            uint32_t a_reg[4], b_reg0[4], b_reg1[4];
            ldmatrix_x4(a_reg, &sQ[swz512(warp_m * 16 + ldm_a_row, ks * 2 + ldm_a_col_chunk)]);
            ldmatrix_x4(b_reg0, &my_sK[swz512(ldm_bk_row, ks * 2 + ldm_bk_col_chunk)]);
            ldmatrix_x4(b_reg1, &my_sK[swz512(16 + ldm_bk_row, ks * 2 + ldm_bk_col_chunk)]);
            uint32_t b00[2] = {b_reg0[0], b_reg0[1]}, b01[2] = {b_reg0[2], b_reg0[3]};
            uint32_t b10[2] = {b_reg1[0], b_reg1[1]}, b11[2] = {b_reg1[2], b_reg1[3]};
            mma_m16n8k16_bf16(s_frag[0], a_reg, b00);
            mma_m16n8k16_bf16(s_frag[1], a_reg, b01);
            mma_m16n8k16_bf16(s_frag[2], a_reg, b10);
            mma_m16n8k16_bf16(s_frag[3], a_reg, b11);
        }

        float m0_loc = -INFINITY, m1_loc = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            int col0 = start_n + warp_half * 32 + nt * 8 + quad_tid * 2;
            int col1 = col0 + 1;
            if (!masked_j) {
                s_frag[nt][0] = q_valid0 ? (s_frag[nt][0] * scale_log2) : -INFINITY;
                s_frag[nt][1] = q_valid0 ? (s_frag[nt][1] * scale_log2) : -INFINITY;
                s_frag[nt][2] = q_valid1 ? (s_frag[nt][2] * scale_log2) : -INFINITY;
                s_frag[nt][3] = q_valid1 ? (s_frag[nt][3] * scale_log2) : -INFINITY;
            } else {
                s_frag[nt][0] = (q_valid0 && col0 < seq_len && col0 <= q_abs0) ? (s_frag[nt][0] * scale_log2) : -INFINITY;
                s_frag[nt][1] = (q_valid0 && col1 < seq_len && col1 <= q_abs0) ? (s_frag[nt][1] * scale_log2) : -INFINITY;
                s_frag[nt][2] = (q_valid1 && col0 < seq_len && col0 <= q_abs1) ? (s_frag[nt][2] * scale_log2) : -INFINITY;
                s_frag[nt][3] = (q_valid1 && col1 < seq_len && col1 <= q_abs1) ? (s_frag[nt][3] * scale_log2) : -INFINITY;
            }
            m0_loc = fmaxf(m0_loc, fmaxf(s_frag[nt][0], s_frag[nt][1]));
            m1_loc = fmaxf(m1_loc, fmaxf(s_frag[nt][2], s_frag[nt][3]));
        }
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 1));
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 2));
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 1));
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 2));

        if (quad_tid == 0) {
            s_red[(warp_m * 2 + warp_half) * 16 + group_id] = m0_loc;
            s_red[(warp_m * 2 + warp_half) * 16 + group_id + 8] = m1_loc;
        }
        cp_async_wait<0>();
        __syncthreads();

        uint4* bV_lo = bStg;
        uint4* bV_hi = bK_hi;
        uint4* sP    = bK_lo;

        issue_load_32(bV_hi, v_cache, v_stride0, v_stride1, v_stride2, start_n + 32, masked_j);
        cp_async_commit();

        float tile_m0 = fmaxf(m0_loc, s_red[(warp_m * 2 + (1 - warp_half)) * 16 + group_id]);
        float tile_m1 = fmaxf(m1_loc, s_red[(warp_m * 2 + (1 - warp_half)) * 16 + group_id + 8]);
        float new_m0 = fmaxf(row_m0, tile_m0);
        float new_m1 = fmaxf(row_m1, tile_m1);
        float alpha0 = (new_m0 == -INFINITY) ? 0.f : exp2f(row_m0 - new_m0);
        float alpha1 = (new_m1 == -INFINITY) ? 0.f : exp2f(row_m1 - new_m1);
        float sub_m0 = (new_m0 == -INFINITY) ? 0.f : new_m0;
        float sub_m1 = (new_m1 == -INFINITY) ? 0.f : new_m1;

        float sum0_loc = 0.f, sum1_loc = 0.f;
        uint32_t* sP_u32 = reinterpret_cast<uint32_t*>(sP);
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            float pf0 = (s_frag[nt][0] == -INFINITY) ? 0.f : exp2f(s_frag[nt][0] - sub_m0);
            float pf1 = (s_frag[nt][1] == -INFINITY) ? 0.f : exp2f(s_frag[nt][1] - sub_m0);
            float pf2 = (s_frag[nt][2] == -INFINITY) ? 0.f : exp2f(s_frag[nt][2] - sub_m1);
            float pf3 = (s_frag[nt][3] == -INFINITY) ? 0.f : exp2f(s_frag[nt][3] - sub_m1);
            sum0_loc += pf0 + pf1;
            sum1_loc += pf2 + pf3;
            int c_chunk = warp_half * 4 + nt;
            sP_u32[swz64(m_row0, c_chunk) * 4 + quad_tid] = pack_bf16x2(pf0, pf1);
            sP_u32[swz64(m_row1, c_chunk) * 4 + quad_tid] = pack_bf16x2(pf2, pf3);
        }
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 1);
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 2);
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 1);
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 2);

        if (j > 0) {
            #pragma unroll
            for (int dt = 0; dt < 32; ++dt) {
                acc_o[dt][0] *= alpha0; acc_o[dt][1] *= alpha0;
                acc_o[dt][2] *= alpha1; acc_o[dt][3] *= alpha1;
            }
        }
        if (quad_tid == 0) {
            s_red_sum[(warp_m * 2 + warp_half) * 16 + group_id] = sum0_loc;
            s_red_sum[(warp_m * 2 + warp_half) * 16 + group_id + 8] = sum1_loc;
        }
        __syncthreads();

        row_l0 = row_l0 * alpha0 + sum0_loc + s_red_sum[(warp_m * 2 + (1 - warp_half)) * 16 + group_id];
        row_l1 = row_l1 * alpha1 + sum1_loc + s_red_sum[(warp_m * 2 + (1 - warp_half)) * 16 + group_id + 8];
        row_m0 = new_m0;
        row_m1 = new_m1;

        uint32_t p_reg[4][4];
        #pragma unroll
        for (int kp = 0; kp < 4; ++kp) {
            ldmatrix_x4(p_reg[kp], &sP[swz64(warp_m * 16 + ldm_a_row, kp * 2 + ldm_a_col_chunk)]);
        }

        const int d_base_chunk = warp_half * 32;
        #pragma unroll
        for (int kp = 0; kp < 2; ++kp) {
            int vr = kp * 16 + ldm_a_row;
            #pragma unroll
            for (int dg = 0; dg < 16; ++dg) {
                uint32_t v_reg[4];
                ldmatrix_x4_trans(v_reg, &bV_lo[swz512(vr, d_base_chunk + dg * 2 + ldm_a_col_chunk)]);
                uint32_t vb0[2] = {v_reg[0], v_reg[1]};
                uint32_t vb1[2] = {v_reg[2], v_reg[3]};
                mma_m16n8k16_bf16(acc_o[dg * 2 + 0], p_reg[kp], vb0);
                mma_m16n8k16_bf16(acc_o[dg * 2 + 1], p_reg[kp], vb1);
            }
        }

        cp_async_wait<0>();
        __syncthreads();

        uint4* nxt_K_lo = sP;
        uint4* nxt_K_hi = bV_lo;
        uint4* nxt_Stg  = bV_hi;

        if (j + 1 < total_tiles) {
            bool m_nxt = ((j + 1) >= full_tiles);
            int nxt_n = (j + 1) * BN;
            issue_load_32(nxt_K_lo, k_cache, k_stride0, k_stride1, k_stride2, nxt_n, m_nxt);
            issue_load_32(nxt_K_hi, k_cache, k_stride0, k_stride1, k_stride2, nxt_n + 32, m_nxt);
            cp_async_commit();
        }

        #pragma unroll
        for (int kp = 2; kp < 4; ++kp) {
            int vr = (kp - 2) * 16 + ldm_a_row;
            #pragma unroll
            for (int dg = 0; dg < 16; ++dg) {
                uint32_t v_reg[4];
                ldmatrix_x4_trans(v_reg, &bV_hi[swz512(vr, d_base_chunk + dg * 2 + ldm_a_col_chunk)]);
                uint32_t vb0[2] = {v_reg[0], v_reg[1]};
                uint32_t vb1[2] = {v_reg[2], v_reg[3]};
                mma_m16n8k16_bf16(acc_o[dg * 2 + 0], p_reg[kp], vb0);
                mma_m16n8k16_bf16(acc_o[dg * 2 + 1], p_reg[kp], vb1);
            }
        }

        if (j + 1 < total_tiles) {
            cp_async_wait<0>();
        }
        __syncthreads();

        bK_lo = nxt_K_lo;
        bK_hi = nxt_K_hi;
        bStg  = nxt_Stg;
    }

    float inv_l0 = (row_l0 > 0.f) ? (1.0f / row_l0) : 0.f;
    float inv_l1 = (row_l1 > 0.f) ? (1.0f / row_l1) : 0.f;
    int q_h0 = kv_head * NUM_Q_PER_KV + (m_row0 % NUM_Q_PER_KV);
    int q_h1 = kv_head * NUM_Q_PER_KV + (m_row1 % NUM_Q_PER_KV);
    __nv_bfloat16* out_row0 = out + (int64_t)(q_start + q_pos0) * o_stride0 + (int64_t)q_h0 * o_stride1;
    __nv_bfloat16* out_row1 = out + (int64_t)(q_start + q_pos1) * o_stride0 + (int64_t)q_h1 * o_stride1;

    #pragma unroll
    for (int dt = 0; dt < 32; ++dt) {
        int col_pair = warp_half * 256 + dt * 8 + quad_tid * 2;
        if (q_valid0) {
            *reinterpret_cast<uint32_t*>(out_row0 + col_pair) =
                pack_bf16x2(acc_o[dt][0] * inv_l0, acc_o[dt][1] * inv_l0);
        }
        if (q_valid1) {
            *reinterpret_cast<uint32_t*>(out_row1 + col_pair) =
                pack_bf16x2(acc_o[dt][2] * inv_l1, acc_o[dt][3] * inv_l1);
        }
    }
}

// ============================================================================
// Rank #3a: Cross-Layer Multi-RMSNorms (D = 2816 = 352 * 8)
// ============================================================================
__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float block_sum_256(float val, float* s_warp) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warp_sum(val);
    if (lane == 0) s_warp[wid] = val;
    __syncthreads();
    float tot = (lane < 8) ? s_warp[lane] : 0.f;
    tot = warp_sum(tot);
    return tot;
}

__global__ __launch_bounds__(256)
void gemma4_post_attn_pre_ffn_norms_kernel(
    __nv_bfloat16* __restrict__ h_res_out,
    __nv_bfloat16* __restrict__ h_ff1_out,
    __nv_bfloat16* __restrict__ h_ff2_out,
    __nv_bfloat16* __restrict__ h_router_out,
    const __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ residual_in,
    const __nv_bfloat16* __restrict__ w_post_attn,
    const __nv_bfloat16* __restrict__ w_pre_ff1,
    const __nv_bfloat16* __restrict__ w_pre_ff2,
    const __nv_bfloat16* __restrict__ router_scale,
    const float* __restrict__ router_scale_fp32,
    float root_size_bf16_f,
    float eps)
{
    constexpr int D = 2816;
    constexpr int N_CHUNKS = D / 8;
    constexpr float INV_D = 1.0f / 2816.0f;

    __shared__ float s_warp[8];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;

    const bf16x8* a_row = reinterpret_cast<const bf16x8*>(attn_out + row * D);
    const bf16x8* r_row = reinterpret_cast<const bf16x8*>(residual_in + row * D);
    const bf16x8* w_pa  = reinterpret_cast<const bf16x8*>(w_post_attn);
    const bf16x8* w_f1  = reinterpret_cast<const bf16x8*>(w_pre_ff1);
    const bf16x8* w_f2  = reinterpret_cast<const bf16x8*>(w_pre_ff2);
    const bf16x8* r_sc  = router_scale ? reinterpret_cast<const bf16x8*>(router_scale) : nullptr;

    bf16x8 a0 = a_row[tid];
    bf16x8 a1;
    float sq_a = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float x = __bfloat162float(a0.v[j]);
        sq_a = __fmaf_rn(x, x, sq_a);
    }
    if (tid + 256 < N_CHUNKS) {
        a1 = a_row[tid + 256];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x = __bfloat162float(a1.v[j]);
            sq_a = __fmaf_rn(x, x, sq_a);
        }
    }
    float inv_rms_a = rsqrtf(__fmaf_rn(block_sum_256(sq_a, s_warp), INV_D, eps));

    bf16x8 h0, h1_vec;
    bf16x8 r0_in = r_row[tid];
    bf16x8 wpa0  = w_pa[tid];
    float sq_h = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float x = __bfloat162float(a0.v[j]);
        float w = __bfloat162float(wpa0.v[j]);
        __nv_bfloat16 normed = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x, inv_rms_a), w));
        __nv_bfloat16 hr = __float2bfloat16_rn(__fadd_rn(__bfloat162float(normed), __bfloat162float(r0_in.v[j])));
        h0.v[j] = hr;
        float hf = __bfloat162float(hr);
        sq_h = __fmaf_rn(hf, hf, sq_h);
    }
    if (tid + 256 < N_CHUNKS) {
        bf16x8 r1_in = r_row[tid + 256];
        bf16x8 wpa1  = w_pa[tid + 256];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x = __bfloat162float(a1.v[j]);
            float w = __bfloat162float(wpa1.v[j]);
            __nv_bfloat16 normed = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x, inv_rms_a), w));
            __nv_bfloat16 hr = __float2bfloat16_rn(__fadd_rn(__bfloat162float(normed), __bfloat162float(r1_in.v[j])));
            h1_vec.v[j] = hr;
            float hf = __bfloat162float(hr);
            sq_h = __fmaf_rn(hf, hf, sq_h);
        }
    }

    __syncthreads();
    float inv_rms_h = rsqrtf(__fmaf_rn(block_sum_256(sq_h, s_warp), INV_D, eps));

    bf16x8* out_res = reinterpret_cast<bf16x8*>(h_res_out + row * D);
    bf16x8* out_ff1 = reinterpret_cast<bf16x8*>(h_ff1_out + row * D);
    bf16x8* out_ff2 = reinterpret_cast<bf16x8*>(h_ff2_out + row * D);
    bf16x8* out_rtr = reinterpret_cast<bf16x8*>(h_router_out + row * D);

    out_res[tid] = h0;
    {
        bf16x8 wf1 = w_f1[tid], wf2 = w_f2[tid], o1, o2, ort, rsc_b;
        if (r_sc) rsc_b = r_sc[tid];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float scaled = __fmul_rn(__bfloat162float(h0.v[j]), inv_rms_h);
            o1.v[j] = __float2bfloat16_rn(__fmul_rn(scaled, __bfloat162float(wf1.v[j])));
            o2.v[j] = __float2bfloat16_rn(__fmul_rn(scaled, __bfloat162float(wf2.v[j])));
            __nv_bfloat16 s0 = __float2bfloat16_rn(scaled);
            __nv_bfloat16 s1 = __float2bfloat16_rn(__fmul_rn(__bfloat162float(s0), root_size_bf16_f));
            float sc_bf16 = r_sc ? __bfloat162float(rsc_b.v[j]) : __bfloat162float(__float2bfloat16_rn(router_scale_fp32[tid * 8 + j]));
            ort.v[j] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(s1), sc_bf16));
        }
        out_ff1[tid] = o1; out_ff2[tid] = o2; out_rtr[tid] = ort;
    }
    if (tid + 256 < N_CHUNKS) {
        int c = tid + 256;
        out_res[c] = h1_vec;
        bf16x8 wf1 = w_f1[c], wf2 = w_f2[c], o1, o2, ort, rsc_b;
        if (r_sc) rsc_b = r_sc[c];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float scaled = __fmul_rn(__bfloat162float(h1_vec.v[j]), inv_rms_h);
            o1.v[j] = __float2bfloat16_rn(__fmul_rn(scaled, __bfloat162float(wf1.v[j])));
            o2.v[j] = __float2bfloat16_rn(__fmul_rn(scaled, __bfloat162float(wf2.v[j])));
            __nv_bfloat16 s0 = __float2bfloat16_rn(scaled);
            __nv_bfloat16 s1 = __float2bfloat16_rn(__fmul_rn(__bfloat162float(s0), root_size_bf16_f));
            float sc_bf16 = r_sc ? __bfloat162float(rsc_b.v[j]) : __bfloat162float(__float2bfloat16_rn(router_scale_fp32[c * 8 + j]));
            ort.v[j] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(s1), sc_bf16));
        }
        out_ff1[c] = o1; out_ff2[c] = o2; out_rtr[c] = ort;
    }
}

__global__ __launch_bounds__(256)
void gemma4_post_ffn_cross_layer_norms_kernel(
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ next_normed_out,
    const __nv_bfloat16* __restrict__ mlp_out,
    const __nv_bfloat16* __restrict__ moe_out,
    const __nv_bfloat16* __restrict__ residual_in,
    const __nv_bfloat16* __restrict__ w_post_ff1,
    const __nv_bfloat16* __restrict__ w_post_ff2,
    const __nv_bfloat16* __restrict__ w_post_ff,
    const void* __restrict__ layer_scalar_ptr,
    bool scalar_is_bf16,
    const __nv_bfloat16* __restrict__ w_next_in,
    float eps)
{
    constexpr int D = 2816;
    constexpr int N_CHUNKS = D / 8;
    constexpr float INV_D = 1.0f / 2816.0f;

    __shared__ bf16x8 s_mlp[N_CHUNKS];
    __shared__ bf16x8 s_moe[N_CHUNKS];
    __shared__ float s_warp[8];
    __shared__ float s_inv_half[2];

    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int wg = tid >> 7;
    const int sub_tid = tid & 127;

    const bf16x8* src_wg = reinterpret_cast<const bf16x8*>((wg == 0 ? mlp_out : moe_out) + row * D);
    const bf16x8* w_wg   = reinterpret_cast<const bf16x8*>(wg == 0 ? w_post_ff1 : w_post_ff2);
    bf16x8* dst_smem     = (wg == 0) ? s_mlp : s_moe;

    float sq_wg = 0.f;
    bf16x8 v0 = src_wg[sub_tid];
    bf16x8 v1 = src_wg[sub_tid + 128];
    bf16x8 v2;
    const bool has_v2 = (sub_tid + 256 < N_CHUNKS);
    if (has_v2) v2 = src_wg[sub_tid + 256];

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float x0 = __bfloat162float(v0.v[j]), x1 = __bfloat162float(v1.v[j]);
        sq_wg = __fmaf_rn(x0, x0, sq_wg);
        sq_wg = __fmaf_rn(x1, x1, sq_wg);
    }
    if (has_v2) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x2 = __bfloat162float(v2.v[j]);
            sq_wg = __fmaf_rn(x2, x2, sq_wg);
        }
    }
    sq_wg = warp_sum(sq_wg);
    if (lane == 0) s_warp[wid] = sq_wg;
    __syncthreads();

    if (sub_tid == 0) {
        float tot = s_warp[wg * 4 + 0] + s_warp[wg * 4 + 1] + s_warp[wg * 4 + 2] + s_warp[wg * 4 + 3];
        s_inv_half[wg] = rsqrtf(__fmaf_rn(tot, INV_D, eps));
    }
    __syncthreads();

    float inv_wg = s_inv_half[wg];
    {
        bf16x8 w0 = w_wg[sub_tid], w1 = w_wg[sub_tid + 128], o0, o1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            o0.v[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(v0.v[j]), inv_wg), __bfloat162float(w0.v[j])));
            o1.v[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(v1.v[j]), inv_wg), __bfloat162float(w1.v[j])));
        }
        dst_smem[sub_tid] = o0;
        dst_smem[sub_tid + 128] = o1;
        if (has_v2) {
            bf16x8 w2 = w_wg[sub_tid + 256], o2;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                o2.v[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(v2.v[j]), inv_wg), __bfloat162float(w2.v[j])));
            }
            dst_smem[sub_tid + 256] = o2;
        }
    }
    __syncthreads();

    bf16x8 c0, c1;
    float sq_c = 0.f;
    {
        bf16x8 m0 = s_mlp[tid], e0 = s_moe[tid];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat16 cb = __float2bfloat16_rn(__fadd_rn(__bfloat162float(m0.v[j]), __bfloat162float(e0.v[j])));
            c0.v[j] = cb;
            float cf = __bfloat162float(cb);
            sq_c = __fmaf_rn(cf, cf, sq_c);
        }
    }
    if (tid + 256 < N_CHUNKS) {
        int c = tid + 256;
        bf16x8 m1 = s_mlp[c], e1 = s_moe[c];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat16 cb = __float2bfloat16_rn(__fadd_rn(__bfloat162float(m1.v[j]), __bfloat162float(e1.v[j])));
            c1.v[j] = cb;
            float cf = __bfloat162float(cb);
            sq_c = __fmaf_rn(cf, cf, sq_c);
        }
    }
    float inv_rms_c = rsqrtf(__fmaf_rn(block_sum_256(sq_c, s_warp), INV_D, eps));

    const float l_scalar = scalar_is_bf16
        ? __bfloat162float(*static_cast<const __nv_bfloat16*>(layer_scalar_ptr))
        : (*static_cast<const float*>(layer_scalar_ptr));

    const bf16x8* res_row = reinterpret_cast<const bf16x8*>(residual_in + row * D);
    const bf16x8* wpff    = reinterpret_cast<const bf16x8*>(w_post_ff);
    bf16x8* out_h         = reinterpret_cast<bf16x8*>(h_out + row * D);

    bf16x8 ho0, ho1;
    float sq_next = 0.f;
    {
        bf16x8 r0 = res_row[tid], wp0 = wpff[tid];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat16 pnorm = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(c0.v[j]), inv_rms_c), __bfloat162float(wp0.v[j])));
            __nv_bfloat16 hadd  = __float2bfloat16_rn(__fadd_rn(__bfloat162float(pnorm), __bfloat162float(r0.v[j])));
            __nv_bfloat16 hfin  = __float2bfloat16_rn(__fmul_rn(__bfloat162float(hadd), l_scalar));
            ho0.v[j] = hfin;
            float hf = __bfloat162float(hfin);
            sq_next = __fmaf_rn(hf, hf, sq_next);
        }
        out_h[tid] = ho0;
    }
    if (tid + 256 < N_CHUNKS) {
        int c = tid + 256;
        bf16x8 r1 = res_row[c], wp1 = wpff[c];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat16 pnorm = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(c1.v[j]), inv_rms_c), __bfloat162float(wp1.v[j])));
            __nv_bfloat16 hadd  = __float2bfloat16_rn(__fadd_rn(__bfloat162float(pnorm), __bfloat162float(r1.v[j])));
            __nv_bfloat16 hfin  = __float2bfloat16_rn(__fmul_rn(__bfloat162float(hadd), l_scalar));
            ho1.v[j] = hfin;
            float hf = __bfloat162float(hfin);
            sq_next = __fmaf_rn(hf, hf, sq_next);
        }
        out_h[c] = ho1;
    }

    if (w_next_in != nullptr && next_normed_out != nullptr) {
        __syncthreads();
        float inv_rms_nxt = rsqrtf(__fmaf_rn(block_sum_256(sq_next, s_warp), INV_D, eps));
        const bf16x8* wnxt = reinterpret_cast<const bf16x8*>(w_next_in);
        bf16x8* out_nxt    = reinterpret_cast<bf16x8*>(next_normed_out + row * D);
        {
            bf16x8 wn0 = wnxt[tid], no0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                no0.v[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(ho0.v[j]), inv_rms_nxt), __bfloat162float(wn0.v[j])));
            }
            out_nxt[tid] = no0;
        }
        if (tid + 256 < N_CHUNKS) {
            int c = tid + 256;
            bf16x8 wn1 = wnxt[c], no1;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                no1.v[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(ho1.v[j]), inv_rms_nxt), __bfloat162float(wn1.v[j])));
            }
            out_nxt[c] = no1;
        }
    }
}

// ============================================================================
// Rank #3b: Deterministic SMEM-Cached uint4 Skinny GEMVs (M in 1..4, 0 atomicAdd)
// ============================================================================
__device__ __forceinline__ float dot8_bf16(const uint4& w_u4, const uint4& x_u4) {
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w_u4);
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&x_u4);
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 wf = __bfloat1622float2(wp[i]);
        float2 xf = __bfloat1622float2(xp[i]);
        sum = __fmaf_rn(wf.x, xf.x, sum);
        sum = __fmaf_rn(wf.y, xf.y, sum);
    }
    return sum;
}

__device__ __forceinline__ __nv_bfloat16 exact_vllm_gelu_tanh_mul(float gate_f, float up_f) {
    float g = __bfloat162float(__float2bfloat16_rn(gate_f));
    float u = __bfloat162float(__float2bfloat16_rn(up_f));
    constexpr float BETA = 0.7978845608028654f;
    constexpr float KAPPA = 0.044715f;
    float inner = BETA * (g + KAPPA * g * g * g);
    float act_bf16 = __bfloat162float(__float2bfloat16_rn(0.5f * g * (1.0f + tanhf(inner))));
    return __float2bfloat16_rn(act_bf16 * u);
}

template <int M, int WARPS_PER_CTA>
__global__ __launch_bounds__(WARPS_PER_CTA * 32)
void gemma4_skinny_gate_up_gelu_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w)
{
    constexpr int K_CHUNKS = 2816 / 8; // 352
    constexpr int N_HALF = 2112;
    constexpr int THREADS = WARPS_PER_CTA * 32;

    __shared__ uint4 sX[M * K_CHUNKS];
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
    for (int idx = tid; idx < M * K_CHUNKS; idx += THREADS) {
        sX[idx] = x_u4[idx];
    }
    __syncthreads();

    const int n = blockIdx.x * WARPS_PER_CTA + wid;
    if (n >= N_HALF) return;

    const uint4* w_gate = reinterpret_cast<const uint4*>(w) + (int64_t)n * K_CHUNKS;
    const uint4* w_up   = reinterpret_cast<const uint4*>(w) + (int64_t)(n + N_HALF) * K_CHUNKS;

    float acc_g[M], acc_u[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) { acc_g[m] = 0.f; acc_u[m] = 0.f; }

    #pragma unroll
    for (int it = 0; it < 11; ++it) {
        int c = it * 32 + lane;
        uint4 wg = __ldg(&w_gate[c]);
        uint4 wu = __ldg(&w_up[c]);
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            uint4 xr = sX[m * K_CHUNKS + c];
            acc_g[m] += dot8_bf16(wg, xr);
            acc_u[m] += dot8_bf16(wu, xr);
        }
    }

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float g = warp_sum(acc_g[m]);
        float u = warp_sum(acc_u[m]);
        if (lane == 0) {
            out[(int64_t)m * N_HALF + n] = exact_vllm_gelu_tanh_mul(g, u);
        }
    }
}

template <int M, int K_CHUNKS, int WARPS_PER_CTA, int ROWS_PER_WARP>
__global__ __launch_bounds__(WARPS_PER_CTA * 32)
void gemma4_skinny_gemv_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N)
{
    extern __shared__ uint4 sX[];
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    constexpr int THREADS = WARPS_PER_CTA * 32;

    const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
    for (int idx = tid; idx < M * K_CHUNKS; idx += THREADS) {
        sX[idx] = x_u4[idx];
    }
    __syncthreads();

    const int n_base = (blockIdx.x * WARPS_PER_CTA + wid) * ROWS_PER_WARP;
    if (n_base >= N) return;

    float acc[ROWS_PER_WARP][M];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

    const uint4* w_base = reinterpret_cast<const uint4*>(w) + (int64_t)n_base * K_CHUNKS;
    constexpr int FULL_ITERS = K_CHUNKS / 32;
    constexpr int REM_CHUNKS = K_CHUNKS % 32;

    #pragma unroll 4
    for (int it = 0; it < FULL_ITERS; ++it) {
        int c = it * 32 + lane;
        uint4 x_reg[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) x_reg[m] = sX[m * K_CHUNKS + c];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            if (ROWS_PER_WARP == 1 || n_base + r < N) {
                uint4 w_reg = __ldg(&w_base[r * K_CHUNKS + c]);
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[r][m] += dot8_bf16(w_reg, x_reg[m]);
            }
        }
    }
    if constexpr (REM_CHUNKS > 0) {
        int c = FULL_ITERS * 32 + lane;
        if (lane < REM_CHUNKS) {
            uint4 x_reg[M];
            #pragma unroll
            for (int m = 0; m < M; ++m) x_reg[m] = sX[m * K_CHUNKS + c];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if (ROWS_PER_WARP == 1 || n_base + r < N) {
                    uint4 w_reg = __ldg(&w_base[r * K_CHUNKS + c]);
                    #pragma unroll
                    for (int m = 0; m < M; ++m) acc[r][m] += dot8_bf16(w_reg, x_reg[m]);
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        if (ROWS_PER_WARP == 1 || n_base + r < N) {
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                float s = warp_sum(acc[r][m]);
                if (lane == 0) out[(int64_t)m * N + (n_base + r)] = __float2bfloat16_rn(s);
            }
        }
    }
}

// ============================================================================
// Rank #4a & #4b: Fused Softcap + Zero-Sync V2 Greedy & Top-K/Top-P Gumbel Sampler
// ============================================================================
template <bool HAS_SCALE>
__global__ __launch_bounds__(256)
void gemma4_softcap_bf16_vec8_kernel(bf16x8* __restrict__ data, int64_t n_vec8, float cap, float scale) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n_vec8; i += stride) {
        bf16x8 x = data[i];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float xf = __bfloat162float(x.v[j]);
            __nv_bfloat16 d = __float2bfloat16_rn(__fdiv_rn(xf, cap));
            __nv_bfloat16 t = __float2bfloat16_rn(tanhf(__bfloat162float(d)));
            __nv_bfloat16 y = __float2bfloat16_rn(__fmul_rn(__bfloat162float(t), cap));
            if constexpr (HAS_SCALE) y = __float2bfloat16_rn(__fmul_rn(__bfloat162float(y), scale));
            x.v[j] = y;
        }
        data[i] = x;
    }
}

template <bool HAS_SCALE>
__global__ __launch_bounds__(256)
void gemma4_softcap_bf16_scalar_kernel(__nv_bfloat16* __restrict__ data, int64_t n_el, float cap, float scale) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n_el; i += stride) {
        float xf = __bfloat162float(data[i]);
        __nv_bfloat16 d = __float2bfloat16_rn(__fdiv_rn(xf, cap));
        __nv_bfloat16 t = __float2bfloat16_rn(tanhf(__bfloat162float(d)));
        __nv_bfloat16 y = __float2bfloat16_rn(__fmul_rn(__bfloat162float(t), cap));
        if constexpr (HAS_SCALE) y = __float2bfloat16_rn(__fmul_rn(__bfloat162float(y), scale));
        data[i] = y;
    }
}

template <bool HAS_SCALE>
__global__ __launch_bounds__(256)
void gemma4_softcap_fp32_kernel(float* __restrict__ data, int64_t n_el, float cap, float scale) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n_el; i += stride) {
        float y = tanhf(data[i] / cap) * cap;
        if constexpr (HAS_SCALE) y *= scale;
        data[i] = y;
    }
}

__device__ __forceinline__ uint32_t float_to_ord_u32(float x) {
    if (isnan(x)) return 0xFFFFFFFFu;
    if (x == 0.0f) x = 0.0f;
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? (~bits) : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float ord_u32_to_float(uint32_t k) {
    uint32_t bits = (k & 0x80000000u) ? (k ^ 0x80000000u) : (~k);
    return __uint_as_float(bits);
}

__device__ __forceinline__ uint64_t pack_comp_key(float val, int32_t tok_id, bool valid) {
    if (!valid) return 0ull;
    uint32_t uk = float_to_ord_u32(val);
    uint32_t inv_tok = 0x00FFFFFFu ^ static_cast<uint32_t>(tok_id & 0x00FFFFFF);
    return (static_cast<uint64_t>(uk) << 24) | static_cast<uint64_t>(inv_tok);
}

__device__ __forceinline__ void unpack_comp_key(uint64_t ck, float& val, int32_t& tok_id) {
    uint32_t uk = static_cast<uint32_t>(ck >> 24);
    uint32_t inv_tok = static_cast<uint32_t>(ck & 0x00FFFFFFull);
    val = ord_u32_to_float(uk);
    tok_id = static_cast<int32_t>(0x00FFFFFFu ^ inv_tok);
}

__device__ __forceinline__ uint32_t rotl32(uint32_t v, int s) { return (v << s) | (v >> (32 - s)); }

__device__ __forceinline__ uint32_t murmur3_mix(uint32_t h, uint32_t key) {
    key *= 0xCC9E2D51u; key = rotl32(key, 15); key *= 0x1B873593u;
    h ^= key; h = rotl32(h, 13); return h * 5u + 0xE6546B64u;
}

__device__ __forceinline__ uint32_t murmur3_fmix32(uint32_t h) {
    h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; return h ^ (h >> 16);
}

__device__ __forceinline__ float vllm_gumbel_noise_fp32(int64_t seed, int64_t pos, uint32_t tok_id) {
    uint32_t h = 0u;
    h = murmur3_mix(h, static_cast<uint32_t>(seed & 0xFFFFFFFFll));
    h = murmur3_mix(h, static_cast<uint32_t>((static_cast<uint64_t>(seed) >> 32) & 0xFFFFFFFFull));
    h = murmur3_mix(h, static_cast<uint32_t>(pos & 0xFFFFFFFFll));
    h = murmur3_mix(h, tok_id);
    uint32_t r32 = murmur3_fmix32(h ^ 16u);
    int32_t hi16 = static_cast<int32_t>(r32 >> 16);
    int32_t lo16 = static_cast<int32_t>(r32 & 0xFFFFu);
    float u = static_cast<float>(hi16) * 1.52587890625e-05f
            + (static_cast<float>(lo16) + 0.5f) * 2.3283064365386963e-10f;
    float poly = 1.0f / 8.0f;
    poly = 1.0f / 7.0f + u * poly; poly = 1.0f / 6.0f + u * poly;
    poly = 1.0f / 5.0f + u * poly; poly = 1.0f / 4.0f + u * poly;
    poly = 1.0f / 3.0f + u * poly; poly = 1.0f / 2.0f + u * poly;
    poly = 1.0f + u * poly;
    float series = -u * poly;
    float direct = logf(fmaxf(1.0f - u, 5.960464477539063e-08f));
    return -logf(-((u < 0.25f) ? series : direct));
}

template <bool IS_BF16>
__global__ __launch_bounds__(256)
void gemma4_greedy_stage1_kernel(
    float* __restrict__ part_val, int32_t* __restrict__ part_idx,
    const void* __restrict__ logits, int64_t row_stride, int vocab_size)
{
    __shared__ float s_val[8];
    __shared__ int32_t s_idx[8];
    const int b = blockIdx.x, blk = blockIdx.y, tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    float best_v = -INFINITY;
    int32_t best_i = 0x7fffffff;
    const int base_col = blk * 8192;
    if constexpr (IS_BF16) {
        const __nv_bfloat16* row_ptr = static_cast<const __nv_bfloat16*>(logits) + (int64_t)b * row_stride;
        if (((row_stride & 7) == 0) && base_col + 8192 <= vocab_size) {
            const bf16x8* vec_ptr = reinterpret_cast<const bf16x8*>(row_ptr + base_col);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int vec_idx = u * 256 + tid;
                bf16x8 pack = vec_ptr[vec_idx];
                int tok0 = base_col + vec_idx * 8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float v = __bfloat162float(pack.v[j]);
                    int tok = tok0 + j;
                    if (v > best_v || (v == best_v && tok < best_i)) { best_v = v; best_i = tok; }
                }
            }
        } else {
            for (int col = base_col + tid; col < min(base_col + 8192, vocab_size); col += 256) {
                float v = __bfloat162float(row_ptr[col]);
                if (v > best_v || (v == best_v && col < best_i)) { best_v = v; best_i = col; }
            }
        }
    } else {
        const float* row_ptr = static_cast<const float*>(logits) + (int64_t)b * row_stride;
        for (int col = base_col + tid; col < min(base_col + 8192, vocab_size); col += 256) {
            float v = row_ptr[col];
            if (v > best_v || (v == best_v && col < best_i)) { best_v = v; best_i = col; }
        }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best_v, mask);
        int32_t oi = __shfl_xor_sync(0xffffffff, best_i, mask);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_val[wid] = best_v; s_idx[wid] = best_i; }
    __syncthreads();
    if (wid == 0) {
        best_v = (lane < 8) ? s_val[lane] : -INFINITY;
        best_i = (lane < 8) ? s_idx[lane] : 0x7fffffff;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best_v, mask);
            int32_t oi = __shfl_xor_sync(0xffffffff, best_i, mask);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) { part_val[b * gridDim.y + blk] = best_v; part_idx[b * gridDim.y + blk] = best_i; }
    }
}

__global__ __launch_bounds__(32)
void gemma4_greedy_stage2_kernel(
    int64_t* __restrict__ sampled_out, const float* __restrict__ part_val,
    const int32_t* __restrict__ part_idx, int num_parts)
{
    const int b = blockIdx.x, lane = threadIdx.x;
    float best_v = -INFINITY; int32_t best_i = 0x7fffffff;
    for (int p = lane; p < num_parts; p += 32) {
        float v = part_val[b * num_parts + p]; int32_t i = part_idx[b * num_parts + p];
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best_v, mask);
        int32_t oi = __shfl_xor_sync(0xffffffff, best_i, mask);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) sampled_out[b] = static_cast<int64_t>(best_i);
}

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void tie_free_radix_select_cta(
    const uint64_t (&ckey)[ITEMS_PER_THREAD], int K,
    uint64_t* __restrict__ out_ckeys, uint32_t* __restrict__ s_hist, uint64_t* __restrict__ s_state)
{
    const int tid = threadIdx.x;
    uint64_t prefix = 0ull, mask_prefix = 0ull;
    int rem_k = K;
    for (int pass = 6; pass >= 0; --pass) {
        s_hist[tid] = 0u;
        __syncthreads();
        int shift = pass * 8;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            if ((ckey[i] & mask_prefix) == prefix) {
                uint32_t bucket = static_cast<uint32_t>((ckey[i] >> shift) & 0xFFull);
                atomicAdd(&s_hist[bucket], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0;
            for (int b = 255; b >= 0; --b) {
                int c = static_cast<int>(s_hist[b]);
                if (acc + c >= rem_k) {
                    s_state[0] = static_cast<uint64_t>(b);
                    s_state[1] = static_cast<uint64_t>(rem_k - acc);
                    s_state[2] = static_cast<uint64_t>(c);
                    break;
                }
                acc += c;
            }
        }
        __syncthreads();
        uint64_t chosen = s_state[0];
        rem_k = static_cast<int>(s_state[1]);
        int bucket_cnt = static_cast<int>(s_state[2]);
        prefix |= (chosen << shift);
        mask_prefix |= (0xFFull << shift);
        if (bucket_cnt == rem_k) break;
        __syncthreads();
    }
    if (tid == 0) s_hist[0] = 0u;
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (ckey[i] >= prefix && ckey[i] != 0ull) {
            uint32_t slot = atomicAdd(&s_hist[0], 1u);
            if (slot < static_cast<uint32_t>(K)) out_ckeys[slot] = ckey[i];
        }
    }
}

template <bool IS_BF16>
__global__ __launch_bounds__(256)
void gemma4_topk_stage1_kernel(
    uint64_t* __restrict__ stage1_ckeys, const void* __restrict__ logits, int64_t row_stride,
    const int32_t* __restrict__ expanded_idx_mapping, const float* __restrict__ temperatures,
    const int32_t* __restrict__ top_k_gpu, int vocab_size, int max_k)
{
    __shared__ uint32_t s_hist[256];
    __shared__ uint64_t s_state[4];
    const int b = blockIdx.x, blk = blockIdx.y, tid = threadIdx.x;
    const int req_idx = expanded_idx_mapping[b];
    const float temp = (req_idx >= 0) ? temperatures[req_idx] : 1.0f;
    const bool scale_temp = (temp > 0.f && temp != 1.0f);
    int k_req = (req_idx >= 0) ? top_k_gpu[req_idx] : max_k;
    int K = min(max(k_req, 1), max_k);

    uint64_t ckey[32];
    const int base_col = blk * 8192;
    if constexpr (IS_BF16) {
        const __nv_bfloat16* row_ptr = static_cast<const __nv_bfloat16*>(logits) + (int64_t)b * row_stride;
        if (((row_stride & 7) == 0) && base_col + 8192 <= vocab_size) {
            const bf16x8* vec_ptr = reinterpret_cast<const bf16x8*>(row_ptr + base_col);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int vec_idx = u * 256 + tid;
                bf16x8 pack = vec_ptr[vec_idx];
                int tok0 = base_col + vec_idx * 8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float v = __bfloat162float(pack.v[j]);
                    if (scale_temp) v = v / temp;
                    ckey[u * 8 + j] = pack_comp_key(v, tok0 + j, true);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                int col = base_col + i * 256 + tid;
                if (col < vocab_size) {
                    float v = __bfloat162float(row_ptr[col]);
                    if (scale_temp) v = v / temp;
                    ckey[i] = pack_comp_key(v, col, true);
                } else ckey[i] = 0ull;
            }
        }
    } else {
        const float* row_ptr = static_cast<const float*>(logits) + (int64_t)b * row_stride;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int col = base_col + i * 256 + tid;
            if (col < vocab_size) {
                float v = row_ptr[col];
                if (scale_temp) v = v / temp;
                ckey[i] = pack_comp_key(v, col, true);
            } else ckey[i] = 0ull;
        }
    }
    uint64_t* dst = stage1_ckeys + ((int64_t)b * gridDim.y + blk) * max_k;
    tie_free_radix_select_cta<32>(ckey, K, dst, s_hist, s_state);
}

__global__ __launch_bounds__(256)
void gemma4_topk_topp_gumbel_stage2_kernel(
    int64_t* __restrict__ sampled_out, const uint64_t* __restrict__ stage1_ckeys,
    const int32_t* __restrict__ expanded_idx_mapping, const int32_t* __restrict__ top_k_gpu,
    const float* __restrict__ top_p_gpu, const int64_t* __restrict__ seeds_gpu,
    const int64_t* __restrict__ pos_gpu, int num_blocks, int max_k)
{
    __shared__ uint32_t s_hist[256];
    __shared__ uint64_t s_state[4];
    __shared__ uint64_t s_top[256];
    __shared__ float s_prob[256];
    const int b = blockIdx.x, tid = threadIdx.x;
    const int req_idx = expanded_idx_mapping[b];
    int k_req = (req_idx >= 0) ? top_k_gpu[req_idx] : max_k;
    const int K = min(max(k_req, 1), max_k);
    const float p_thresh = (top_p_gpu != nullptr && req_idx >= 0) ? top_p_gpu[req_idx] : 1.0f;
    const int64_t seed = (req_idx >= 0) ? seeds_gpu[req_idx] : 0ll;
    const int64_t pos = pos_gpu[b];

    const int total_cand = num_blocks * K;
    const uint64_t* row_cands = stage1_ckeys + (int64_t)b * num_blocks * max_k;
    uint64_t ckey[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int idx = i * 256 + tid;
        if (idx < total_cand) {
            int blk = idx / K;
            ckey[i] = row_cands[blk * max_k + (idx - blk * K)];
        } else ckey[i] = 0ull;
    }

    s_top[tid] = 0ull;
    __syncthreads();
    tie_free_radix_select_cta<32>(ckey, K, s_top, s_hist, s_state);
    __syncthreads();

    for (int size = 2; size <= 256; size <<= 1) {
        bool dir_desc = ((tid & size) == 0);
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int ixj = tid ^ stride;
            if (ixj > tid) {
                uint64_t a = s_top[tid], c = s_top[ixj];
                if ((a < c) == dir_desc) { s_top[tid] = c; s_top[ixj] = a; }
            }
            __syncthreads();
        }
    }

    float val_i = -INFINITY; int32_t tok_i = 0x7fffffff;
    if (tid < K && s_top[tid] != 0ull) unpack_comp_key(s_top[tid], val_i, tok_i);
    __shared__ float s_max_v;
    if (tid == 0) s_max_v = val_i;
    __syncthreads();

    s_prob[tid] = (tid < K && val_i > -INFINITY) ? expf(val_i - s_max_v) : 0.f;
    __syncthreads();
    for (int offset = 1; offset < 256; offset <<= 1) {
        float add = (tid >= offset) ? s_prob[tid - offset] : 0.f;
        __syncthreads();
        s_prob[tid] += add;
        __syncthreads();
    }

    float z_topk = s_prob[K - 1];
    float excl_cum = (tid == 0) ? 0.f : s_prob[tid - 1];
    bool keep = (tid == 0) || (tid < K && ((p_thresh >= 1.0f) || (excl_cum < p_thresh * z_topk)));

    float noisy_score = (keep && val_i > -INFINITY)
        ? (val_i + vllm_gumbel_noise_fp32(seed, pos, static_cast<uint32_t>(tok_i)))
        : -INFINITY;

    const int wid = tid >> 5, lane = tid & 31;
    __shared__ float s_best_v[8];
    __shared__ int32_t s_best_t[8];
    float bv = noisy_score; int32_t bt = tok_i;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, mask);
        int32_t ot = __shfl_xor_sync(0xffffffff, bt, mask);
        if (ov > bv || (ov == bv && ot < bt)) { bv = ov; bt = ot; }
    }
    if (lane == 0) { s_best_v[wid] = bv; s_best_t[wid] = bt; }
    __syncthreads();
    if (wid == 0) {
        bv = (lane < 8) ? s_best_v[lane] : -INFINITY;
        bt = (lane < 8) ? s_best_t[lane] : 0x7fffffff;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, mask);
            int32_t ot = __shfl_xor_sync(0xffffffff, bt, mask);
            if (ov > bv || (ov == bv && ot < bt)) { bv = ov; bt = ot; }
        }
        if (lane == 0) sampled_out[b] = static_cast<int64_t>(bt);
    }
}

} // namespace

void launch_paged_fa2_hd512(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q, double scale)
{
    int total_q = q.size(0);
    int num_seqs = seqused_k.size(0);
    if (total_q == 0 || num_seqs == 0) return;
    static bool init_fa2 = false;
    constexpr int SMEM = (64 * 64 + 32 * 64 * 3) * 16 + 256 * 4; // 164864 B <= 166912 B
    if (!init_fa2) {
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init_fa2 = true;
    }
    int nkv = k_cache.size(2);
    int block_size = k_cache.size(1);
    int grid_x = total_q / 8 + num_seqs;
    sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8><<<dim3(grid_x, nkv), 256, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        block_table.data_ptr<int32_t>(),
        seqused_k.data_ptr<int32_t>(),
        cu_seqlens_q.data_ptr<int32_t>(),
        static_cast<float>(scale),
        num_seqs, block_size,
        block_table.stride(0),
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2));
}

void post_attn_pre_ffn_norms(
    torch::Tensor h_res_out, torch::Tensor h_ff1_out, torch::Tensor h_ff2_out, torch::Tensor h_router_out,
    torch::Tensor attn_out, torch::Tensor residual_in,
    torch::Tensor w_post_attn, torch::Tensor w_pre_ff1, torch::Tensor w_pre_ff2,
    torch::Tensor router_scale, double root_size_bf16_f, double eps)
{
    int T = attn_out.size(0);
    if (T == 0) return;
    const __nv_bfloat16* rsc_bf16 = (router_scale.scalar_type() == torch::kBFloat16)
        ? reinterpret_cast<const __nv_bfloat16*>(router_scale.data_ptr()) : nullptr;
    const float* rsc_fp32 = (router_scale.scalar_type() == torch::kFloat)
        ? router_scale.data_ptr<float>() : nullptr;
    gemma4_post_attn_pre_ffn_norms_kernel<<<T, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__nv_bfloat16*>(h_res_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_ff1_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_ff2_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(h_router_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(attn_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(residual_in.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_post_attn.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_pre_ff1.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_pre_ff2.data_ptr()),
        rsc_bf16, rsc_fp32, static_cast<float>(root_size_bf16_f), static_cast<float>(eps));
}

void post_ffn_cross_layer_norms(
    torch::Tensor h_out, torch::Tensor next_normed_out,
    torch::Tensor mlp_out, torch::Tensor moe_out, torch::Tensor residual_in,
    torch::Tensor w_post_ff1, torch::Tensor w_post_ff2, torch::Tensor w_post_ff,
    torch::Tensor layer_scalar, torch::Tensor w_next_in, double eps)
{
    int T = mlp_out.size(0);
    if (T == 0) return;
    bool sc_bf16 = (layer_scalar.scalar_type() == torch::kBFloat16);
    gemma4_post_ffn_cross_layer_norms_kernel<<<T, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__nv_bfloat16*>(h_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(next_normed_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(mlp_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(moe_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(residual_in.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_post_ff1.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_post_ff2.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_post_ff.data_ptr()),
        layer_scalar.data_ptr(), sc_bf16,
        reinterpret_cast<const __nv_bfloat16*>(w_next_in.data_ptr()),
        static_cast<float>(eps));
}

#define DISPATCH_M4(M_VAL, CALL) \
    switch (M_VAL) { \
        case 1: { constexpr int M_C = 1; CALL; break; } \
        case 2: { constexpr int M_C = 2; CALL; break; } \
        case 3: { constexpr int M_C = 3; CALL; break; } \
        case 4: { constexpr int M_C = 4; CALL; break; } \
        default: TORCH_CHECK(false, "Unsupported M=", M_VAL); \
    }

void skinny_gate_up_gelu(torch::Tensor out, torch::Tensor x, torch::Tensor w) {
    int M = x.size(0);
    constexpr int W = 4;
    int grid = (2112 + W - 1) / W;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_M4(M, (gemma4_skinny_gate_up_gelu_kernel<M_C, W><<<grid, W * 32, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()))));
}

void skinny_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor w, int rows_per_warp) {
    int M = x.size(0), K = x.size(1), N = w.size(0);
    constexpr int W = 4;
    int smem = M * (K / 8) * 16;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    __nv_bfloat16* o_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const __nv_bfloat16* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const __nv_bfloat16* w_ptr = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());

    if (K == 2112) {
        if (rows_per_warp == 2) {
            int grid = (N + W * 2 - 1) / (W * 2);
            DISPATCH_M4(M, (gemma4_skinny_gemv_kernel<M_C, 264, W, 2><<<grid, W * 32, smem, stream>>>(o_ptr, x_ptr, w_ptr, N)));
        } else {
            int grid = (N + W - 1) / W;
            DISPATCH_M4(M, (gemma4_skinny_gemv_kernel<M_C, 264, W, 1><<<grid, W * 32, smem, stream>>>(o_ptr, x_ptr, w_ptr, N)));
        }
    } else if (K == 2816) {
        int grid = (N + W - 1) / W;
        DISPATCH_M4(M, (gemma4_skinny_gemv_kernel<M_C, 352, W, 1><<<grid, W * 32, smem, stream>>>(o_ptr, x_ptr, w_ptr, N)));
    } else if (K == 4096) {
        int grid = (N + W - 1) / W;
        DISPATCH_M4(M, (gemma4_skinny_gemv_kernel<M_C, 512, W, 1><<<grid, W * 32, smem, stream>>>(o_ptr, x_ptr, w_ptr, N)));
    } else if (K == 8192) {
        static bool init8k = false;
        if (!init8k && smem > 49152) {
            cudaFuncSetAttribute(gemma4_skinny_gemv_kernel<4, 1024, W, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            init8k = true;
        }
        int grid = (N + W - 1) / W;
        DISPATCH_M4(M, (gemma4_skinny_gemv_kernel<M_C, 1024, W, 1><<<grid, W * 32, smem, stream>>>(o_ptr, x_ptr, w_ptr, N)));
    } else {
        TORCH_CHECK(false, "Unsupported K=", K);
    }
}

void fused_softcap_inplace(torch::Tensor logits, double soft_cap, double scale) {
    int64_t n = logits.numel();
    if (n == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool has_scale = (scale != 1.0);
    if (logits.scalar_type() == torch::kBFloat16) {
        if ((n & 7) == 0 && ((reinterpret_cast<uintptr_t>(logits.data_ptr()) & 15) == 0)) {
            int64_t nv = n / 8;
            int grid = static_cast<int>(std::min<int64_t>((nv + 255) / 256, 2160));
            if (has_scale) gemma4_softcap_bf16_vec8_kernel<true><<<grid, 256, 0, stream>>>(reinterpret_cast<bf16x8*>(logits.data_ptr()), nv, (float)soft_cap, (float)scale);
            else gemma4_softcap_bf16_vec8_kernel<false><<<grid, 256, 0, stream>>>(reinterpret_cast<bf16x8*>(logits.data_ptr()), nv, (float)soft_cap, 1.0f);
        } else {
            int grid = static_cast<int>(std::min<int64_t>((n + 255) / 256, 4320));
            if (has_scale) gemma4_softcap_bf16_scalar_kernel<true><<<grid, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()), n, (float)soft_cap, (float)scale);
            else gemma4_softcap_bf16_scalar_kernel<false><<<grid, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()), n, (float)soft_cap, 1.0f);
        }
    } else if (logits.scalar_type() == torch::kFloat) {
        int grid = static_cast<int>(std::min<int64_t>((n + 255) / 256, 4320));
        if (has_scale) gemma4_softcap_fp32_kernel<true><<<grid, 256, 0, stream>>>(logits.data_ptr<float>(), n, (float)soft_cap, (float)scale);
        else gemma4_softcap_fp32_kernel<false><<<grid, 256, 0, stream>>>(logits.data_ptr<float>(), n, (float)soft_cap, 1.0f);
    }
}

torch::Tensor fast_greedy_argmax(torch::Tensor logits) {
    int B = logits.size(0), V = logits.size(1);
    auto out = torch::empty({B}, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
    if (B == 0) return out;
    int num_parts = (V + 8191) / 8192;
    auto part_val = torch::empty({B, num_parts}, torch::TensorOptions().dtype(torch::kFloat).device(logits.device()));
    auto part_idx = torch::empty({B, num_parts}, torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid1(B, num_parts);
    if (logits.scalar_type() == torch::kBFloat16)
        gemma4_greedy_stage1_kernel<true><<<grid1, 256, 0, stream>>>(part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(), logits.data_ptr(), logits.stride(0), V);
    else
        gemma4_greedy_stage1_kernel<false><<<grid1, 256, 0, stream>>>(part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(), logits.data_ptr(), logits.stride(0), V);
    gemma4_greedy_stage2_kernel<<<B, 32, 0, stream>>>(out.data_ptr<int64_t>(), part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(), num_parts);
    return out;
}

torch::Tensor fast_topk_topp_gumbel_sample(
    torch::Tensor logits, torch::Tensor expanded_idx_mapping, torch::Tensor temperatures,
    torch::Tensor top_k_gpu, c10::optional<torch::Tensor> top_p_gpu,
    torch::Tensor seeds_gpu, torch::Tensor pos_gpu, int max_k)
{
    int B = logits.size(0), V = logits.size(1);
    auto out = torch::empty({B}, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
    if (B == 0) return out;
    int num_blocks = (V + 8191) / 8192;
    auto stage1 = torch::empty({B, num_blocks, max_k}, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid1(B, num_blocks);
    if (logits.scalar_type() == torch::kBFloat16) {
        gemma4_topk_stage1_kernel<true><<<grid1, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(stage1.data_ptr()), logits.data_ptr(), logits.stride(0),
            expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
            top_k_gpu.data_ptr<int32_t>(), V, max_k);
    } else {
        gemma4_topk_stage1_kernel<false><<<grid1, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(stage1.data_ptr()), logits.data_ptr(), logits.stride(0),
            expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
            top_k_gpu.data_ptr<int32_t>(), V, max_k);
    }
    gemma4_topk_topp_gumbel_stage2_kernel<<<B, 256, 0, stream>>>(
        out.data_ptr<int64_t>(), reinterpret_cast<const uint64_t*>(stage1.data_ptr()),
        expanded_idx_mapping.data_ptr<int32_t>(), top_k_gpu.data_ptr<int32_t>(),
        top_p_gpu.has_value() ? top_p_gpu->data_ptr<float>() : nullptr,
        seeds_gpu.data_ptr<int64_t>(), pos_gpu.data_ptr<int64_t>(), num_blocks, max_k);
    return out;
}
"""

_CPP_SRC = r"""
void launch_paged_fa2_hd512(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q, double scale);
void post_attn_pre_ffn_norms(
    torch::Tensor h_res_out, torch::Tensor h_ff1_out, torch::Tensor h_ff2_out, torch::Tensor h_router_out,
    torch::Tensor attn_out, torch::Tensor residual_in,
    torch::Tensor w_post_attn, torch::Tensor w_pre_ff1, torch::Tensor w_pre_ff2,
    torch::Tensor router_scale, double root_size_bf16_f, double eps);
void post_ffn_cross_layer_norms(
    torch::Tensor h_out, torch::Tensor next_normed_out,
    torch::Tensor mlp_out, torch::Tensor moe_out, torch::Tensor residual_in,
    torch::Tensor w_post_ff1, torch::Tensor w_post_ff2, torch::Tensor w_post_ff,
    torch::Tensor layer_scalar, torch::Tensor w_next_in, double eps);
void skinny_gate_up_gelu(torch::Tensor out, torch::Tensor x, torch::Tensor w);
void skinny_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor w, int rows_per_warp);
void fused_softcap_inplace(torch::Tensor logits, double soft_cap, double scale);
torch::Tensor fast_greedy_argmax(torch::Tensor logits);
torch::Tensor fast_topk_topp_gumbel_sample(
    torch::Tensor logits, torch::Tensor expanded_idx_mapping, torch::Tensor temperatures,
    torch::Tensor top_k_gpu, c10::optional<torch::Tensor> top_p_gpu,
    torch::Tensor seeds_gpu, torch::Tensor pos_gpu, int max_k);
"""

_EXT_MOD: ModuleType | None = None
_EXT_HASH = hashlib.sha256((_CUDA_SRC + _CPP_SRC).encode("utf-8")).hexdigest()[:12]
_MOD_NAME = f"gemma4_sm80_ops_{_EXT_HASH}"


def _resolve_cache_dir() -> str:
    env_dir = os.environ.get("VLLM_GEMMA4_SM80_EXT_DIR", "").strip()
    if env_dir:
        return os.path.join(env_dir, _MOD_NAME)
    proj_dir = "/projects/gemma4-vllm/gemma-binedit/logs/p3_cuda_ext"
    if os.path.isdir("/projects/gemma4-vllm/gemma-binedit"):
        return os.path.join(proj_dir, _MOD_NAME)
    return os.path.expanduser(f"~/.cache/vllm/gemma4_sm80_ext/{_MOD_NAME}")


def get_gemma4_sm80_ext() -> ModuleType:
    global _EXT_MOD
    if _EXT_MOD is not None:
        return _EXT_MOD

    build_dir = _resolve_cache_dir()
    so_path = os.path.join(build_dir, f"{_MOD_NAME}.so")
    if os.path.isfile(so_path):
        spec = importlib.util.spec_from_file_location(_MOD_NAME, so_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT_MOD = mod
        return mod

    os.makedirs(os.path.join(build_dir, "tmp"), exist_ok=True)
    py_bin = os.path.dirname(sys.executable)
    extra_paths = [p for p in (py_bin, "/usr/local/cuda/bin") if p and os.path.isdir(p)]
    cur_path = os.environ.get("PATH", "")
    for p in reversed(extra_paths):
        if p not in cur_path.split(":"):
            cur_path = f"{p}:{cur_path}" if cur_path else p
    os.environ["PATH"] = cur_path
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    prev_tmp = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = os.path.join(build_dir, "tmp")
    try:
        from torch.utils.cpp_extension import load_inline

        mod = load_inline(
            name=_MOD_NAME,
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=[
                "launch_paged_fa2_hd512",
                "post_attn_pre_ffn_norms",
                "post_ffn_cross_layer_norms",
                "skinny_gate_up_gelu",
                "skinny_gemv",
                "fused_softcap_inplace",
                "fast_greedy_argmax",
                "fast_topk_topp_gumbel_sample",
            ],
            extra_cuda_cflags=["-O3", "-arch=sm_80"],
            build_directory=build_dir,
            verbose=False,
        )
    finally:
        if prev_tmp is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = prev_tmp

    _EXT_MOD = mod
    return mod


def paged_fa2_hd512(
    out: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    scale: float,
) -> None:
    assert (
        q.stride(2) == 1
        and out.stride(2) == 1
        and q.stride(1) == 512
        and out.stride(1) == 512
        and k_cache.stride(3) == 1
        and v_cache.stride(3) == 1
        and k_cache.size(1) % 32 == 0
        and block_table.stride(1) == 1
    )
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if seqused_k.dtype != torch.int32 or not seqused_k.is_contiguous():
        seqused_k = seqused_k.to(torch.int32).contiguous()
    if cu_seqlens_q.dtype != torch.int32 or not cu_seqlens_q.is_contiguous():
        cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    get_gemma4_sm80_ext().launch_paged_fa2_hd512(
        out, q, k_cache, v_cache, block_table, seqused_k, cu_seqlens_q, float(scale)
    )


def gemma4_post_attn_pre_ffn_norms_op(
    attn_out: torch.Tensor,
    residual_in: torch.Tensor,
    w_post_attn: torch.Tensor,
    w_pre_ff1: torch.Tensor,
    w_pre_ff2: torch.Tensor,
    router_scale: torch.Tensor,
    root_size_bf16_f: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h_res = torch.empty_like(attn_out)
    h_ff1 = torch.empty_like(attn_out)
    h_ff2 = torch.empty_like(attn_out)
    h_rtr = torch.empty_like(attn_out)
    if attn_out.shape[0] > 0:
        get_gemma4_sm80_ext().post_attn_pre_ffn_norms(
            h_res,
            h_ff1,
            h_ff2,
            h_rtr,
            attn_out,
            residual_in,
            w_post_attn,
            w_pre_ff1,
            w_pre_ff2,
            router_scale,
            root_size_bf16_f,
            eps,
        )
    return h_res, h_ff1, h_ff2, h_rtr


def gemma4_post_attn_pre_ffn_norms_fake(
    attn_out: torch.Tensor,
    residual_in: torch.Tensor,
    w_post_attn: torch.Tensor,
    w_pre_ff1: torch.Tensor,
    w_pre_ff2: torch.Tensor,
    router_scale: torch.Tensor,
    root_size_bf16_f: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(attn_out),
        torch.empty_like(attn_out),
        torch.empty_like(attn_out),
        torch.empty_like(attn_out),
    )


direct_register_custom_op(
    op_name="gemma4_post_attn_pre_ffn_norms",
    op_func=gemma4_post_attn_pre_ffn_norms_op,
    fake_impl=gemma4_post_attn_pre_ffn_norms_fake,
)


def gemma4_post_ffn_cross_layer_norms_op(
    mlp_out: torch.Tensor,
    moe_out: torch.Tensor,
    residual_in: torch.Tensor,
    w_post_ff1: torch.Tensor,
    w_post_ff2: torch.Tensor,
    w_post_ff: torch.Tensor,
    layer_scalar: torch.Tensor,
    w_next_in: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_out = torch.empty_like(mlp_out)
    nxt_out = torch.empty_like(mlp_out)
    if mlp_out.shape[0] > 0:
        get_gemma4_sm80_ext().post_ffn_cross_layer_norms(
            h_out,
            nxt_out,
            mlp_out,
            moe_out,
            residual_in,
            w_post_ff1,
            w_post_ff2,
            w_post_ff,
            layer_scalar,
            w_next_in,
            eps,
        )
    return h_out, nxt_out


def gemma4_post_ffn_cross_layer_norms_fake(
    mlp_out: torch.Tensor,
    moe_out: torch.Tensor,
    residual_in: torch.Tensor,
    w_post_ff1: torch.Tensor,
    w_post_ff2: torch.Tensor,
    w_post_ff: torch.Tensor,
    layer_scalar: torch.Tensor,
    w_next_in: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(mlp_out), torch.empty_like(mlp_out)


direct_register_custom_op(
    op_name="gemma4_post_ffn_cross_layer_norms",
    op_func=gemma4_post_ffn_cross_layer_norms_op,
    fake_impl=gemma4_post_ffn_cross_layer_norms_fake,
)


def gemma4_mlp_gate_up_gelu_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m = x.shape[0]
    n_half = weight.shape[0] // 2
    out = x.new_empty((m, n_half))
    if m == 0:
        return out
    if m <= 4 and x.is_contiguous() and weight.is_contiguous() and n_half == 2112 and x.shape[1] == 2816:
        get_gemma4_sm80_ext().skinny_gate_up_gelu(out, x, weight)
        return out
    gu = F.linear(x, weight)
    torch.ops._C.gelu_tanh_and_mul(out, gu)
    return out


def gemma4_mlp_gate_up_gelu_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


direct_register_custom_op(
    op_name="gemma4_mlp_gate_up_gelu",
    op_func=gemma4_mlp_gate_up_gelu_op,
    fake_impl=gemma4_mlp_gate_up_gelu_fake,
)


def gemma4_skinny_linear_op(
    x: torch.Tensor, weight: torch.Tensor, max_m: int
) -> torch.Tensor:
    m = x.shape[0]
    if 1 <= m <= max_m and x.is_contiguous() and weight.is_contiguous():
        out = x.new_empty((m, weight.shape[0]))
        rpw = 2 if (m == 1 and x.shape[1] == 2112) else 1
        get_gemma4_sm80_ext().skinny_gemv(out, x, weight, rpw)
        return out
    return F.linear(x, weight)


def gemma4_skinny_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, max_m: int
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="gemma4_skinny_linear",
    op_func=gemma4_skinny_linear_op,
    fake_impl=gemma4_skinny_linear_fake,
)


def fused_softcap_inplace(logits: torch.Tensor, soft_cap: float, scale: float = 1.0) -> None:
    get_gemma4_sm80_ext().fused_softcap_inplace(logits, float(soft_cap), float(scale))


def fast_greedy_argmax(logits: torch.Tensor) -> torch.Tensor:
    return get_gemma4_sm80_ext().fast_greedy_argmax(logits)


def fast_topk_topp_gumbel_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperatures: torch.Tensor,
    top_k_gpu: torch.Tensor,
    top_p_gpu: torch.Tensor | None,
    seeds_gpu: torch.Tensor,
    pos_gpu: torch.Tensor,
    max_k: int,
) -> torch.Tensor:
    if (
        expanded_idx_mapping.dtype != torch.int32
        or not expanded_idx_mapping.is_contiguous()
    ):
        expanded_idx_mapping = expanded_idx_mapping.to(torch.int32).contiguous()
    if temperatures.dtype != torch.float32 or not temperatures.is_contiguous():
        temperatures = temperatures.to(torch.float32).contiguous()
    if top_k_gpu.dtype != torch.int32 or not top_k_gpu.is_contiguous():
        top_k_gpu = top_k_gpu.to(torch.int32).contiguous()
    if top_p_gpu is not None and (
        top_p_gpu.dtype != torch.float32 or not top_p_gpu.is_contiguous()
    ):
        top_p_gpu = top_p_gpu.to(torch.float32).contiguous()
    if seeds_gpu.dtype != torch.int64 or not seeds_gpu.is_contiguous():
        seeds_gpu = seeds_gpu.to(torch.int64).contiguous()
    if pos_gpu.dtype != torch.int64 or not pos_gpu.is_contiguous():
        pos_gpu = pos_gpu.to(torch.int64).contiguous()
    return get_gemma4_sm80_ext().fast_topk_topp_gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperatures,
        top_k_gpu,
        top_p_gpu,
        seeds_gpu,
        pos_gpu,
        int(max_k),
    )
