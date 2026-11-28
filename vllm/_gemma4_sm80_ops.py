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
#include <mutex>

namespace {

struct bf16x8 {
    __nv_bfloat16 v[8];
};
static_assert(sizeof(bf16x8) == 16, "bf16x8 must be 16 bytes");

struct alignas(16) bf16x8_a16 {
    __nv_bfloat16 v[8];
};

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void sm90_pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

template <typename P0>
__device__ __forceinline__ void sm90_pdl_wait_1ptr(P0& p0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n" : "+l"(p0) :: "memory");
#endif
}

template <typename P0, typename P1>
__device__ __forceinline__ void sm90_pdl_wait_2ptrs(P0& p0, P1& p1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n" : "+l"(p0), "+l"(p1) :: "memory");
#endif
}

template <typename P0, typename P1, typename P2>
__device__ __forceinline__ void sm90_pdl_wait_3ptrs(P0& p0, P1& p1, P2& p2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n" : "+l"(p0), "+l"(p1), "+l"(p2) :: "memory");
#endif
}

template <typename KernelT, typename... Args>
static inline void sm90_launch_pdl(
    KernelT kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args...);
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

__device__ __forceinline__ void cp_async_16B_u32(uint32_t smem_dst, const void* glob_src, bool pred) {
    int p_int = pred ? 1 : 0;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
        "}\n" :: "r"(smem_dst), "l"(glob_src), "r"(p_int));
}

__device__ __forceinline__ void cp_async_16B_zfill_u32(uint32_t smem_dst, const void* glob_src, bool pred) {
    int sz = pred ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_dst), "l"(glob_src), "r"(sz));
}

__device__ __forceinline__ void cp_async_16B_unchk_u32(uint32_t smem_dst, const void* glob_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_dst), "l"(glob_src));
}

__device__ __forceinline__ void st_shared_zero_u32(uint32_t smem_dst) {
    asm volatile("st.shared.v4.u32 [%0], {0, 0, 0, 0};\n" :: "r"(smem_dst));
}

__device__ __forceinline__ void st_shared_u32(uint32_t smem_dst, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(smem_dst), "r"(val));
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

__device__ __forceinline__ void ldmatrix_u32_x4(uint32_t (&r)[4], uint32_t s) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void ldmatrix_u32_x4_trans(uint32_t (&r)[4], uint32_t s) {
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

__device__ __forceinline__ void mma_m16n8k16_bf16_nv(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm(
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

__device__ __forceinline__ float fa2_ex2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void bar_sync_64(int id) {
    asm volatile("bar.sync %0, 64;" :: "r"(id) : "memory");
}

__device__ __forceinline__ int swz256(int row, int chunk32) {
    return (row << 5) + (chunk32 ^ (row & 7));
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
// Rank #1a: SM80 3-Buffer Pipelined mma.sync Paged FA2 for hd=512 (BS32 & BS16)
// ============================================================================
template <int NUM_Q_PER_KV = 8, bool BS_EQ_32 = false, bool BS32 = false, bool BS_EQ_16 = false>
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
    const uint32_t tid = threadIdx.x;
    const int warp_id = static_cast<int>(tid >> 5);
    const int lane = static_cast<int>(tid & 31u);
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

    const uint32_t sQ_u32_base = smem_u32(sQ);
    const uint32_t ld_r0 = tid >> 6;
    const uint32_t ld_c  = tid & 63u;
    const uint32_t ld_dst_ev = static_cast<uint32_t>(swz512(static_cast<int>(ld_r0), static_cast<int>(ld_c))) * 16u;
    const uint32_t ld_dst_od = static_cast<uint32_t>(swz512(static_cast<int>(ld_r0 + 4u), static_cast<int>(ld_c))) * 16u;

    #pragma unroll
    for (int idx = static_cast<int>(tid); idx < 64 * 64; idx += 256) {
        int r = idx >> 6;
        int c = idx & 63;
        int q_pos = q_blk * BLOCK_Q + (r / NUM_Q_PER_KV);
        int q_h = kv_head * NUM_Q_PER_KV + (r % NUM_Q_PER_KV);
        bool ok = q_pos < q_len;
        const __nv_bfloat16* gptr = q + (int64_t)(q_start + (ok ? q_pos : 0)) * q_stride0 + (int64_t)q_h * q_stride1 + (c << 3);
        cp_async_16B_u32(sQ_u32_base + static_cast<uint32_t>(swz512(r, c)) * 16u, gptr, ok);
    }

    const int context_len = seq_len - q_len;
    const int q_max_local = min((q_blk + 1) * BLOCK_Q, q_len) - 1;
    const int max_seq_prefix = min(context_len + q_max_local + 1, seq_len);
    const int full_tiles = (context_len + q_blk * BLOCK_Q) / BN;
    const int total_tiles = (max_seq_prefix + BN - 1) / BN;
    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;
    const float scale_log2 = scale * LOG2E;
    const bool blk_pow2 = (BS_EQ_32 || BS_EQ_16) ? true : ((block_size & (block_size - 1)) == 0);
    const int blk_shift = BS_EQ_32 ? 5 : (BS_EQ_16 ? 4 : (blk_pow2 ? (__ffs(block_size) - 1) : 0));
    const int blk_mask = BS_EQ_32 ? 31 : (BS_EQ_16 ? 15 : (block_size - 1));

    const int64_t k_s1_bytes = k_stride1 * sizeof(__nv_bfloat16);
    const int64_t k_s1_4_bytes = k_s1_bytes * 4;
    const int64_t v_s1_bytes = v_stride1 * sizeof(__nv_bfloat16);
    const int64_t v_s1_4_bytes = v_s1_bytes * 4;
    const int64_t col_bytes = static_cast<int64_t>(ld_c) * (8 * sizeof(__nv_bfloat16));
    const __nv_bfloat16* k_head_base = k_cache + (int64_t)kv_head * k_stride2;
    const __nv_bfloat16* v_head_base = v_cache + (int64_t)kv_head * v_stride2;

    auto issue_load_32 = [&](uint32_t dst_u32, const __nv_bfloat16* cache_head_base,
                             int64_t s0, int64_t s1, int64_t s1_bytes, int64_t s1_4_bytes,
                             int start_tok, bool masked) {
        if (masked && start_tok >= max_seq_prefix) {
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                uint32_t g_off = dst_u32 + static_cast<uint32_t>(g * 8192);
                st_shared_zero_u32(g_off + ld_dst_ev);
                st_shared_zero_u32(g_off + ld_dst_od);
            }
            return;
        }
        if constexpr (BS_EQ_32 || BS32) {
            int page = BS_EQ_32 ? (start_tok >> 5) : (blk_pow2 ? (start_tok >> blk_shift) : (start_tok / block_size));
            int off  = BS_EQ_32 ? 0 : (blk_pow2 ? (start_tok & blk_mask) : (start_tok - page * block_size));
            int64_t pblk = (int64_t)bt_row[page];
            const char* ptr0 = reinterpret_cast<const char*>(
                cache_head_base + pblk * s0 + (int64_t)off * s1) + col_bytes;
            const char* cur = ptr0 + static_cast<int64_t>(ld_r0) * s1_bytes;
            if (!masked) {
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    uint32_t g_off = dst_u32 + static_cast<uint32_t>(g * 8192);
                    cp_async_16B_unchk_u32(g_off + ld_dst_ev, cur);
                    cur += s1_4_bytes;
                    cp_async_16B_unchk_u32(g_off + ld_dst_od, cur);
                    cur += s1_4_bytes;
                }
            } else {
                const int rem = max_seq_prefix - start_tok;
                int re = static_cast<int>(ld_r0);
                int ro = re + 4;
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    uint32_t g_off = dst_u32 + static_cast<uint32_t>(g * 8192);
                    bool ok_ev = re < rem;
                    cp_async_16B_u32(g_off + ld_dst_ev, ok_ev ? cur : ptr0, ok_ev);
                    cur += s1_4_bytes;
                    bool ok_od = ro < rem;
                    cp_async_16B_u32(g_off + ld_dst_od, ok_od ? cur : ptr0, ok_od);
                    cur += s1_4_bytes;
                    re += 8;
                    ro += 8;
                }
            }
        } else {
            int page0 = BS_EQ_16 ? (start_tok >> 4) : (blk_pow2 ? (start_tok >> blk_shift) : (start_tok / block_size));
            int off0  = BS_EQ_16 ? 0 : (blk_pow2 ? (start_tok & blk_mask) : (start_tok - page0 * block_size));
            int64_t pblk0 = (int64_t)bt_row[page0];
            const char* ptr0 = BS_EQ_16
                ? (reinterpret_cast<const char*>(cache_head_base + pblk0 * s0) + col_bytes)
                : (reinterpret_cast<const char*>(cache_head_base + pblk0 * s0 + (int64_t)off0 * s1) + col_bytes);
            int tok1 = start_tok + 16;
            if (!masked) {
                int page1 = BS_EQ_16 ? (page0 + 1) : (blk_pow2 ? (tok1 >> blk_shift) : (tok1 / block_size));
                int off1  = BS_EQ_16 ? 0 : (blk_pow2 ? (tok1 & blk_mask) : (tok1 - page1 * block_size));
                int64_t pblk1 = (int64_t)bt_row[page1];
                const char* ptr1 = BS_EQ_16
                    ? (reinterpret_cast<const char*>(cache_head_base + pblk1 * s0) + col_bytes)
                    : (reinterpret_cast<const char*>(cache_head_base + pblk1 * s0 + (int64_t)off1 * s1) + col_bytes);
                const char* cur0 = ptr0 + static_cast<int64_t>(ld_r0) * s1_bytes;
                #pragma unroll
                for (int g = 0; g < 2; ++g) {
                    uint32_t g_off = dst_u32 + static_cast<uint32_t>(g * 8192);
                    cp_async_16B_unchk_u32(g_off + ld_dst_ev, cur0);
                    cur0 += s1_4_bytes;
                    cp_async_16B_unchk_u32(g_off + ld_dst_od, cur0);
                    cur0 += s1_4_bytes;
                }
                const char* cur1 = ptr1 + static_cast<int64_t>(ld_r0) * s1_bytes;
                #pragma unroll
                for (int g = 2; g < 4; ++g) {
                    uint32_t g_off = dst_u32 + static_cast<uint32_t>(g * 8192);
                    cp_async_16B_unchk_u32(g_off + ld_dst_ev, cur1);
                    cur1 += s1_4_bytes;
                    cp_async_16B_unchk_u32(g_off + ld_dst_od, cur1);
                    cur1 += s1_4_bytes;
                }
            } else {
                bool hi_valid = tok1 < max_seq_prefix;
                const char* ptr1 = ptr0;
                if (hi_valid) {
                    int page1 = BS_EQ_16 ? (page0 + 1) : (blk_pow2 ? (tok1 >> blk_shift) : (tok1 / block_size));
                    int off1  = BS_EQ_16 ? 0 : (blk_pow2 ? (tok1 & blk_mask) : (tok1 - page1 * block_size));
                    int64_t pblk1 = (int64_t)bt_row[page1];
                    ptr1 = BS_EQ_16
                        ? (reinterpret_cast<const char*>(cache_head_base + pblk1 * s0) + col_bytes)
                        : (reinterpret_cast<const char*>(cache_head_base + pblk1 * s0 + (int64_t)off1 * s1) + col_bytes);
                }
                const int rem = max_seq_prefix - start_tok;
                int re = static_cast<int>(ld_r0);
                int ro = re + 4;
                const char* cur0 = ptr0 + static_cast<int64_t>(ld_r0) * s1_bytes;
                #pragma unroll
                for (int g = 0; g < 2; ++g) {
                    uint32_t g_off = dst_u32 + static_cast<uint32_t>(g * 8192);
                    bool ok_ev = re < rem;
                    cp_async_16B_u32(g_off + ld_dst_ev, ok_ev ? cur0 : ptr0, ok_ev);
                    cur0 += s1_4_bytes;
                    bool ok_od = ro < rem;
                    cp_async_16B_u32(g_off + ld_dst_od, ok_od ? cur0 : ptr0, ok_od);
                    cur0 += s1_4_bytes;
                    re += 8;
                    ro += 8;
                }
                const char* cur1 = ptr1 + static_cast<int64_t>(ld_r0) * s1_bytes;
                #pragma unroll
                for (int g = 2; g < 4; ++g) {
                    uint32_t g_off = dst_u32 + static_cast<uint32_t>(g * 8192);
                    bool ok_ev = re < rem;
                    cp_async_16B_u32(g_off + ld_dst_ev, ok_ev ? cur1 : ptr1, ok_ev);
                    cur1 += s1_4_bytes;
                    bool ok_od = ro < rem;
                    cp_async_16B_u32(g_off + ld_dst_od, ok_od ? cur1 : ptr1, ok_od);
                    cur1 += s1_4_bytes;
                    re += 8;
                    ro += 8;
                }
            }
        }
    };

    uint32_t bK_lo_u32 = smem_u32(sBuf0);
    uint32_t bK_hi_u32 = smem_u32(sBuf1);
    uint32_t bStg_u32  = smem_u32(sBuf2);

    if (total_tiles > 0) {
        bool m0 = (0 >= full_tiles);
        issue_load_32(bK_lo_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_4_bytes, 0, m0);
        issue_load_32(bK_hi_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_4_bytes, 32, m0);
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
    const int d_base_chunk = warp_half * 32;

    const int lane_6 = lane & 6;
    const int sw4_0 = 0 ^ lane_6;
    const int sw4_1 = 2 ^ lane_6;
    const int sw4_2 = 4 ^ lane_6;
    const int sw4_3 = 6 ^ lane_6;
    const int q_base = ((warp_m * 16 + ldm_a_row) << 6) + ((lane & 1) ^ ldm_a_col_chunk);
    const int k_base = (ldm_bk_row << 6) + ((lane & 1) ^ ldm_bk_col_chunk);
    const int v_base = (ldm_a_row << 6) + d_base_chunk + (ldm_a_col_chunk ^ (lane & 1));
    const int sp_w_base = (m_row0 << 5) + (((warp_half * 4) ^ (m_row0 & 7)) << 2) + quad_tid;
    const int sp_rd_base = ((warp_m * 16 + ldm_a_row) << 3) + (ldm_a_col_chunk ^ (lane & 1));

    const uint32_t q_off_u32[4] = {
        sQ_u32_base + static_cast<uint32_t>(q_base + sw4_0) * 16u,
        sQ_u32_base + static_cast<uint32_t>(q_base + sw4_1) * 16u,
        sQ_u32_base + static_cast<uint32_t>(q_base + sw4_2) * 16u,
        sQ_u32_base + static_cast<uint32_t>(q_base + sw4_3) * 16u,
    };
    const uint32_t k_off_u32[4] = {
        static_cast<uint32_t>(k_base + sw4_0) * 16u,
        static_cast<uint32_t>(k_base + sw4_1) * 16u,
        static_cast<uint32_t>(k_base + sw4_2) * 16u,
        static_cast<uint32_t>(k_base + sw4_3) * 16u,
    };
    const uint32_t v_off_u32[4] = {
        static_cast<uint32_t>(v_base + sw4_0) * 16u,
        static_cast<uint32_t>(v_base + sw4_1) * 16u,
        static_cast<uint32_t>(v_base + sw4_2) * 16u,
        static_cast<uint32_t>(v_base + sw4_3) * 16u,
    };
    const uint32_t sp_w_off_u32[4] = {
        static_cast<uint32_t>(sp_w_base ^ 0) * 4u,
        static_cast<uint32_t>(sp_w_base ^ 4) * 4u,
        static_cast<uint32_t>(sp_w_base ^ 8) * 4u,
        static_cast<uint32_t>(sp_w_base ^ 12) * 4u,
    };
    const uint32_t sp_rd_off_u32[4] = {
        static_cast<uint32_t>(sp_rd_base + sw4_0) * 16u,
        static_cast<uint32_t>(sp_rd_base + sw4_1) * 16u,
        static_cast<uint32_t>(sp_rd_base + sw4_2) * 16u,
        static_cast<uint32_t>(sp_rd_base + sw4_3) * 16u,
    };

#define GEMMA4_HD512_RUN_TILE(IS_FAST_VAL)                                                   \
    do {                                                                                     \
        constexpr bool IS_FAST = (IS_FAST_VAL);                                              \
        const int start_n = j * BN;                                                          \
        const bool masked_j = IS_FAST ? false : (j >= full_tiles);                           \
                                                                                             \
        issue_load_32(bStg_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_4_bytes, start_n, masked_j); \
        cp_async_commit();                                                                   \
                                                                                             \
        uint32_t my_sK_u32 = (warp_half == 0) ? bK_lo_u32 : bK_hi_u32;                       \
        float s_frag[4][4] = {                                                               \
            {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f},                                      \
            {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}                                       \
        };                                                                                   \
        uint32_t bk_col_u32[4] = {                                                           \
            my_sK_u32 + k_off_u32[0], my_sK_u32 + k_off_u32[1],                              \
            my_sK_u32 + k_off_u32[2], my_sK_u32 + k_off_u32[3]                               \
        };                                                                                   \
        _Pragma("unroll 2")                                                                  \
        for (int ko = 0; ko < 8; ++ko) {                                                     \
            uint32_t k_step_b = static_cast<uint32_t>(ko * 128);                             \
            _Pragma("unroll")                                                                \
            for (int kin = 0; kin < 4; ++kin) {                                              \
                uint32_t a_reg[4], b_reg0[4], b_reg1[4];                                     \
                ldmatrix_u32_x4(b_reg0, bk_col_u32[kin] + k_step_b);                         \
                ldmatrix_u32_x4(a_reg, q_off_u32[kin] + k_step_b);                           \
                ldmatrix_u32_x4(b_reg1, bk_col_u32[kin] + (k_step_b + 16384u));             \
                uint32_t b00[2] = {b_reg0[0], b_reg0[1]}, b01[2] = {b_reg0[2], b_reg0[3]};   \
                uint32_t b10[2] = {b_reg1[0], b_reg1[1]}, b11[2] = {b_reg1[2], b_reg1[3]};   \
                mma_m16n8k16_bf16_nv(s_frag[0], a_reg, b00);                                    \
                mma_m16n8k16_bf16_nv(s_frag[1], a_reg, b01);                                    \
                mma_m16n8k16_bf16_nv(s_frag[2], a_reg, b10);                                    \
                mma_m16n8k16_bf16_nv(s_frag[3], a_reg, b11);                                    \
            }                                                                                \
        }                                                                                    \
                                                                                             \
        float m0_raw = -INFINITY, m1_raw = -INFINITY;                                        \
        if constexpr (IS_FAST) {                                                             \
            _Pragma("unroll")                                                                \
            for (int nt = 0; nt < 4; ++nt) {                                                 \
                m0_raw = fmaxf(m0_raw, fmaxf(s_frag[nt][0], s_frag[nt][1]));                 \
                m1_raw = fmaxf(m1_raw, fmaxf(s_frag[nt][2], s_frag[nt][3]));                 \
            }                                                                                \
        } else {                                                                             \
            _Pragma("unroll")                                                                \
            for (int nt = 0; nt < 4; ++nt) {                                                 \
                if (!masked_j) {                                                             \
                    if (!q_valid0) { s_frag[nt][0] = -INFINITY; s_frag[nt][1] = -INFINITY; } \
                    if (!q_valid1) { s_frag[nt][2] = -INFINITY; s_frag[nt][3] = -INFINITY; } \
                } else {                                                                     \
                    int col0 = start_n + warp_half * 32 + nt * 8 + quad_tid * 2;             \
                    int col1 = col0 + 1;                                                     \
                    if (!(q_valid0 && col0 < seq_len && col0 <= q_abs0)) s_frag[nt][0] = -INFINITY; \
                    if (!(q_valid0 && col1 < seq_len && col1 <= q_abs0)) s_frag[nt][1] = -INFINITY; \
                    if (!(q_valid1 && col0 < seq_len && col0 <= q_abs1)) s_frag[nt][2] = -INFINITY; \
                    if (!(q_valid1 && col1 < seq_len && col1 <= q_abs1)) s_frag[nt][3] = -INFINITY; \
                }                                                                            \
                m0_raw = fmaxf(m0_raw, fmaxf(s_frag[nt][0], s_frag[nt][1]));                 \
                m1_raw = fmaxf(m1_raw, fmaxf(s_frag[nt][2], s_frag[nt][3]));                 \
            }                                                                                \
        }                                                                                    \
        float m0_loc = m0_raw * scale_log2;                                                  \
        float m1_loc = m1_raw * scale_log2;                                                  \
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 1));                      \
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 2));                      \
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 1));                      \
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 2));                      \
                                                                                             \
        if (quad_tid == 0) {                                                                 \
            s_red[(warp_m * 2 + warp_half) * 16 + group_id] = m0_loc;                        \
            s_red[(warp_m * 2 + warp_half) * 16 + group_id + 8] = m1_loc;                    \
        }                                                                                    \
        __syncthreads();                                                                     \
                                                                                             \
        uint32_t bV_lo_u32 = bStg_u32;                                                       \
        uint32_t bV_hi_u32 = bK_hi_u32;                                                      \
        uint32_t sP_u32_base = bK_lo_u32;                                                    \
                                                                                             \
        issue_load_32(bV_hi_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_4_bytes, start_n + 32, masked_j); \
        cp_async_commit();                                                                   \
                                                                                             \
        float tile_m0 = fmaxf(m0_loc, s_red[(warp_m * 2 + (1 - warp_half)) * 16 + group_id]); \
        float tile_m1 = fmaxf(m1_loc, s_red[(warp_m * 2 + (1 - warp_half)) * 16 + group_id + 8]); \
        float new_m0 = fmaxf(row_m0, tile_m0);                                               \
        float new_m1 = fmaxf(row_m1, tile_m1);                                               \
        float alpha0 = IS_FAST ? ((j == 0) ? 0.f : fa2_ex2_approx(row_m0 - new_m0))          \
                               : ((row_m0 == -INFINITY || new_m0 == -INFINITY) ? 0.f : fa2_ex2_approx(row_m0 - new_m0)); \
        float alpha1 = IS_FAST ? ((j == 0) ? 0.f : fa2_ex2_approx(row_m1 - new_m1))          \
                               : ((row_m1 == -INFINITY || new_m1 == -INFINITY) ? 0.f : fa2_ex2_approx(row_m1 - new_m1)); \
        float sub_m0 = IS_FAST ? new_m0 : ((new_m0 == -INFINITY) ? 0.f : new_m0);            \
        float sub_m1 = IS_FAST ? new_m1 : ((new_m1 == -INFINITY) ? 0.f : new_m1);            \
        float neg_sub_m0 = -sub_m0;                                                          \
        float neg_sub_m1 = -sub_m1;                                                          \
                                                                                             \
        float sum0_loc = 0.f, sum1_loc = 0.f;                                                \
        _Pragma("unroll")                                                                    \
        for (int nt = 0; nt < 4; ++nt) {                                                     \
            float pf0, pf1, pf2, pf3;                                                        \
            if constexpr (IS_FAST) {                                                         \
                pf0 = fa2_ex2_approx(__fmaf_rn(s_frag[nt][0], scale_log2, neg_sub_m0));      \
                pf1 = fa2_ex2_approx(__fmaf_rn(s_frag[nt][1], scale_log2, neg_sub_m0));      \
                pf2 = fa2_ex2_approx(__fmaf_rn(s_frag[nt][2], scale_log2, neg_sub_m1));      \
                pf3 = fa2_ex2_approx(__fmaf_rn(s_frag[nt][3], scale_log2, neg_sub_m1));      \
            } else {                                                                         \
                pf0 = (s_frag[nt][0] == -INFINITY) ? 0.f : fa2_ex2_approx(__fmaf_rn(s_frag[nt][0], scale_log2, neg_sub_m0)); \
                pf1 = (s_frag[nt][1] == -INFINITY) ? 0.f : fa2_ex2_approx(__fmaf_rn(s_frag[nt][1], scale_log2, neg_sub_m0)); \
                pf2 = (s_frag[nt][2] == -INFINITY) ? 0.f : fa2_ex2_approx(__fmaf_rn(s_frag[nt][2], scale_log2, neg_sub_m1)); \
                pf3 = (s_frag[nt][3] == -INFINITY) ? 0.f : fa2_ex2_approx(__fmaf_rn(s_frag[nt][3], scale_log2, neg_sub_m1)); \
            }                                                                                \
            sum0_loc += pf0 + pf1;                                                           \
            sum1_loc += pf2 + pf3;                                                           \
            uint32_t sp_addr = sP_u32_base + sp_w_off_u32[nt];                               \
            st_shared_u32(sp_addr, pack_bf16x2(pf0, pf1));                                   \
            st_shared_u32(sp_addr + 1024u, pack_bf16x2(pf2, pf3));                           \
        }                                                                                    \
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 1);                                \
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 2);                                \
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 1);                                \
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 2);                                \
                                                                                             \
        if (j > 0) {                                                                         \
            _Pragma("unroll")                                                                \
            for (int dt = 0; dt < 32; ++dt) {                                                \
                acc_o[dt][0] *= alpha0; acc_o[dt][1] *= alpha0;                              \
                acc_o[dt][2] *= alpha1; acc_o[dt][3] *= alpha1;                              \
            }                                                                                \
        }                                                                                    \
        if (quad_tid == 0) {                                                                 \
            s_red_sum[(warp_m * 2 + warp_half) * 16 + group_id] = sum0_loc;                  \
            s_red_sum[(warp_m * 2 + warp_half) * 16 + group_id + 8] = sum1_loc;              \
        }                                                                                    \
        bar_sync_64(warp_m + 1);                                                                     \
                                                                                             \
        row_l0 = row_l0 * alpha0 + sum0_loc + s_red_sum[(warp_m * 2 + (1 - warp_half)) * 16 + group_id]; \
        row_l1 = row_l1 * alpha1 + sum1_loc + s_red_sum[(warp_m * 2 + (1 - warp_half)) * 16 + group_id + 8]; \
        row_m0 = new_m0;                                                                     \
        row_m1 = new_m1;                                                                     \
                                                                                             \
        uint32_t p_reg[4][4];                                                                \
        ldmatrix_u32_x4(p_reg[0], sP_u32_base + sp_rd_off_u32[0]);                           \
        ldmatrix_u32_x4(p_reg[1], sP_u32_base + sp_rd_off_u32[1]);                           \
        ldmatrix_u32_x4(p_reg[2], sP_u32_base + sp_rd_off_u32[2]);                           \
        ldmatrix_u32_x4(p_reg[3], sP_u32_base + sp_rd_off_u32[3]);                           \
                                                                                             \
        const bool has_next = IS_FAST ? true : (j + 1 < total_tiles);                        \
        uint32_t nxt_K_lo_u32 = sP_u32_base;                                                 \
        uint32_t nxt_K_hi_u32 = bV_lo_u32;                                                   \
        uint32_t nxt_Stg_u32  = bV_hi_u32;                                                   \
                                                                                             \
        cp_async_wait<1>();                                                                  \
        __syncthreads();                                                                     \
        if (has_next) {                                                                      \
            bool m_nxt = IS_FAST ? false : ((j + 1) >= full_tiles);                          \
            int nxt_n = (j + 1) * BN;                                                        \
            issue_load_32(nxt_K_lo_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_4_bytes, nxt_n, m_nxt); \
            cp_async_commit();                                                               \
        }                                                                                    \
                                                                                             \
        {                                                                                    \
            uint32_t bv_lo_col[4] = {                                                        \
                bV_lo_u32 + v_off_u32[0], bV_lo_u32 + v_off_u32[1],                          \
                bV_lo_u32 + v_off_u32[2], bV_lo_u32 + v_off_u32[3]                           \
            };                                                                               \
            _Pragma("unroll")                                                                \
            for (int kp = 0; kp < 2; ++kp) {                                                 \
                uint32_t kp_b = static_cast<uint32_t>(kp * 16384);                           \
                _Pragma("unroll")                                                            \
                for (int d_hi = 0; d_hi < 32; d_hi += 8) {                                   \
                    uint32_t off_b = kp_b + static_cast<uint32_t>(d_hi * 16);                \
                    _Pragma("unroll")                                                        \
                    for (int d_lo = 0; d_lo < 4; ++d_lo) {                                   \
                        uint32_t v_reg[4];                                                   \
                        ldmatrix_u32_x4_trans(v_reg, bv_lo_col[d_lo] + off_b);               \
                        uint32_t vb0[2] = {v_reg[0], v_reg[1]};                              \
                        uint32_t vb1[2] = {v_reg[2], v_reg[3]};                              \
                        int dg = (d_hi >> 1) + d_lo;                                         \
                        mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 0], p_reg[kp], vb0);                \
                        mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 1], p_reg[kp], vb1);                \
                    }                                                                        \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
                                                                                             \
        if (has_next) {                                                                      \
            cp_async_wait<1>();                                                              \
        } else {                                                                             \
            cp_async_wait<0>();                                                              \
        }                                                                                    \
        __syncthreads();                                                                     \
        if (has_next) {                                                                      \
            bool m_nxt = IS_FAST ? false : ((j + 1) >= full_tiles);                          \
            int nxt_n = (j + 1) * BN;                                                        \
            issue_load_32(nxt_K_hi_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_4_bytes, nxt_n + 32, m_nxt); \
            cp_async_commit();                                                               \
        }                                                                                    \
                                                                                             \
        {                                                                                    \
            uint32_t bv_hi_col[4] = {                                                        \
                bV_hi_u32 + v_off_u32[0], bV_hi_u32 + v_off_u32[1],                          \
                bV_hi_u32 + v_off_u32[2], bV_hi_u32 + v_off_u32[3]                           \
            };                                                                               \
            _Pragma("unroll")                                                                \
            for (int kp = 2; kp < 4; ++kp) {                                                 \
                uint32_t kp_b = static_cast<uint32_t>((kp - 2) * 16384);                     \
                _Pragma("unroll")                                                            \
                for (int d_hi = 0; d_hi < 32; d_hi += 8) {                                   \
                    uint32_t off_b = kp_b + static_cast<uint32_t>(d_hi * 16);                \
                    _Pragma("unroll")                                                        \
                    for (int d_lo = 0; d_lo < 4; ++d_lo) {                                   \
                        uint32_t v_reg[4];                                                   \
                        ldmatrix_u32_x4_trans(v_reg, bv_hi_col[d_lo] + off_b);               \
                        uint32_t vb0[2] = {v_reg[0], v_reg[1]};                              \
                        uint32_t vb1[2] = {v_reg[2], v_reg[3]};                              \
                        int dg = (d_hi >> 1) + d_lo;                                         \
                        mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 0], p_reg[kp], vb0);                \
                        mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 1], p_reg[kp], vb1);                \
                    }                                                                        \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
                                                                                             \
        if (has_next) {                                                                      \
            cp_async_wait<0>();                                                              \
        }                                                                                    \
        __syncthreads();                                                                     \
                                                                                             \
        bK_lo_u32 = nxt_K_lo_u32;                                                            \
        bK_hi_u32 = nxt_K_hi_u32;                                                            \
        bStg_u32  = nxt_Stg_u32;                                                             \
    } while (0)

    const int fast_end = (q_valid0 && q_valid1 && full_tiles > 1) ? (full_tiles - 1) : 0;
    for (int j = 0; j < fast_end; ++j) {
        GEMMA4_HD512_RUN_TILE(true);
    }
    for (int j = fast_end; j < total_tiles; ++j) {
        GEMMA4_HD512_RUN_TILE(false);
    }
#undef GEMMA4_HD512_RUN_TILE

    float inv_l0 = (row_l0 > 0.f) ? (1.0f / row_l0) : 0.f;
    float inv_l1 = (row_l1 > 0.f) ? (1.0f / row_l1) : 0.f;
    int q_h0 = kv_head * NUM_Q_PER_KV + (m_row0 % NUM_Q_PER_KV);
    int q_h1 = kv_head * NUM_Q_PER_KV + (m_row1 % NUM_Q_PER_KV);
    __nv_bfloat16* out_row0 = out + (int64_t)(q_start + q_pos0) * o_stride0 + (int64_t)q_h0 * o_stride1 + warp_half * 256 + quad_tid * 2;
    __nv_bfloat16* out_row1 = out + (int64_t)(q_start + q_pos1) * o_stride0 + (int64_t)q_h1 * o_stride1 + warp_half * 256 + quad_tid * 2;

    if (q_valid0) {
        #pragma unroll
        for (int dt = 0; dt < 32; ++dt) {
            *reinterpret_cast<uint32_t*>(out_row0 + dt * 8) =
                pack_bf16x2(acc_o[dt][0] * inv_l0, acc_o[dt][1] * inv_l0);
        }
    }
    if (q_valid1) {
        #pragma unroll
        for (int dt = 0; dt < 32; ++dt) {
            *reinterpret_cast<uint32_t*>(out_row1 + dt * 8) =
                pack_bf16x2(acc_o[dt][2] * inv_l1, acc_o[dt][3] * inv_l1);
        }
    }
}

// ============================================================================
// SM80 mma.sync Split-K Paged Decode + Warp-Parallel Reduce (nq_per_kv=8)
// ============================================================================
__device__ __forceinline__ int swz_pow2(int row, int chunk, int chunks_per_row) {
    return row * chunks_per_row + (chunk ^ (row & 7));
}

template <int HEAD_SIZE, int NUM_Q_PER_KV = 8, bool BS32 = false, bool FAST1 = false>
__global__ __launch_bounds__(128, 4)
void sm80_paged_decode_mma_splitk_kernel(
    float* __restrict__ segm_out,
    float* __restrict__ segm_max,
    float* __restrict__ segm_expsum,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    int sliding_window_val,
    int num_seqs,
    int num_kv_heads,
    int num_splits,
    int block_size,
    int64_t bt_stride0,
    int64_t q_stride0,
    int64_t q_stride1,
    int64_t k_stride0,
    int64_t k_stride1,
    int64_t k_stride2,
    int64_t v_stride0,
    int64_t v_stride1,
    int64_t v_stride2)
{
    static_assert(NUM_Q_PER_KV == 8 && HEAD_SIZE == 256, "Specialized for hd=256 nq=8");
    constexpr int D_PER_WARP = HEAD_SIZE / 4;
    constexpr int NUM_SLICES = D_PER_WARP / 16;
    constexpr int CHUNKS = HEAD_SIZE / 8;
    constexpr uint32_t KV_PLANE_BYTES = 16 * CHUNKS * 16;
    constexpr float LOG2E = 1.4426950408889634f;

    const int seq_idx = blockIdx.x, kv_head = blockIdx.y, split_idx = blockIdx.z;
    const int total_q_heads = num_kv_heads * NUM_Q_PER_KV;
    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int group_id = lane >> 2, quad_tid = lane & 3;

    int q_tok = seq_idx;
    if (cu_seqlens_q != nullptr) {
        q_tok = cu_seqlens_q[seq_idx];
        if (cu_seqlens_q[seq_idx + 1] <= q_tok) return;
    }

    const int seq_len = seqused_k[seq_idx];
    int k_min = 0;
    if (sliding_window_val > 0 && seq_len > sliding_window_val) k_min = seq_len - sliding_window_val;
    const int tile_lo_g = k_min >> 4;
    const int tile_hi_g = (seq_len + 15) >> 4;
    const int n_active_tiles = (seq_len > 0 && tile_hi_g > tile_lo_g) ? (tile_hi_g - tile_lo_g) : 0;
    const int t_per = n_active_tiles / num_splits;
    const int t_rem = n_active_tiles - t_per * num_splits;
    const int my_tiles = t_per + (split_idx < t_rem ? 1 : 0);
    const int my_t0 = tile_lo_g + split_idx * t_per + min(split_idx, t_rem);
    if (my_tiles <= 0) return;

    __shared__ struct {
        uint4 bK[16 * CHUNKS];
        uint4 bV[16 * CHUNKS];
        float logits[4][16][9];
    } smem;

    const uint32_t bK_u32 = smem_u32(smem.bK);
    const uint32_t bV_u32 = bK_u32 + KV_PLANE_BYTES;

    uint32_t sK_load_u32[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        sK_load_u32[u] = bK_u32 + static_cast<uint32_t>(swz256(warp_id + (u << 2), lane) << 4);
    }

    const int d_warp = warp_id * D_PER_WARP;
    const int d_warp_chunks = d_warp >> 3;
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col_chunk = (lane >> 4);
    const int ldm_bk_row = (lane >> 4) * 8 + (lane & 7);
    const int ldm_bk_col_chunk = ((lane >> 3) & 1);
    uint32_t ldm_k_u32[NUM_SLICES], ldm_v_u32[NUM_SLICES];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) {
        ldm_k_u32[s] = bK_u32 + static_cast<uint32_t>(swz256(ldm_a_row, d_warp_chunks + s * 2 + ldm_a_col_chunk) << 4);
        ldm_v_u32[s] = bV_u32 + static_cast<uint32_t>(swz256(ldm_bk_row, d_warp_chunks + s * 2 + ldm_bk_col_chunk) << 4);
    }

    const uint32_t* q_u32 = reinterpret_cast<const uint32_t*>(
        q + (int64_t)q_tok * q_stride0 + (int64_t)(kv_head * NUM_Q_PER_KV + group_id) * q_stride1 + d_warp);
    uint32_t q_reg[NUM_SLICES][2];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) {
        q_reg[s][0] = q_u32[s * 8 + quad_tid];
        q_reg[s][1] = q_u32[s * 8 + 4 + quad_tid];
    }

    float acc_o[NUM_SLICES][4];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) { acc_o[s][0] = 0.f; acc_o[s][1] = 0.f; acc_o[s][2] = 0.f; acc_o[s][3] = 0.f; }
    float row_m_h = -INFINITY, row_l_h = 0.0f;

    const bool blk_pow2 = BS32 ? true : ((block_size & (block_size - 1)) == 0);
    const int blk_shift = BS32 ? 5 : (blk_pow2 ? (__ffs(block_size) - 1) : 0);
    const int blk_mask = BS32 ? 31 : (block_size - 1);
    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;
    const __nv_bfloat16* k_head_base = k_cache + (int64_t)kv_head * k_stride2 + (lane << 3);
    const __nv_bfloat16* v_head_base = v_cache + (int64_t)kv_head * v_stride2 + (lane << 3);
    const int64_t k_step4 = k_stride1 << 2;
    const int64_t v_step4 = v_stride1 << 2;

    auto resolve_tile = [&](int tile_idx, int& off, int64_t& pblk) {
        if constexpr (BS32) {
            off = (tile_idx & 1) << 4;
            pblk = (int64_t)bt_row[tile_idx >> 1];
        } else {
            int tok0 = tile_idx << 4;
            int page = blk_pow2 ? (tok0 >> blk_shift) : (tok0 / block_size);
            off = blk_pow2 ? (tok0 & blk_mask) : (tok0 - page * block_size);
            pblk = (int64_t)bt_row[page];
        }
    };

    auto issue_k = [&](int64_t pblk, int off, int tile_idx) {
        int tok0 = tile_idx << 4;
        const __nv_bfloat16* p0 = k_head_base + pblk * k_stride0 + (int64_t)off * k_stride1;
        if ((tok0 + 16) <= seq_len) {
            const __nv_bfloat16* pr = p0 + (int64_t)warp_id * k_stride1;
            cp_async_16B_unchk_u32(sK_load_u32[0], pr);
            cp_async_16B_unchk_u32(sK_load_u32[1], pr + k_step4);
            cp_async_16B_unchk_u32(sK_load_u32[2], pr + (k_step4 << 1));
            cp_async_16B_unchk_u32(sK_load_u32[3], pr + k_step4 * 3);
        } else {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int r = warp_id + (u << 2);
                bool ok = (tok0 + r) < seq_len;
                cp_async_16B_zfill_u32(sK_load_u32[u], p0 + (ok ? (int64_t)r * k_stride1 : 0), ok);
            }
        }
    };

    auto issue_v = [&](int64_t pblk, int off, int tile_idx) {
        int tok0 = tile_idx << 4;
        const __nv_bfloat16* p0 = v_head_base + pblk * v_stride0 + (int64_t)off * v_stride1;
        if ((tok0 + 16) <= seq_len) {
            const __nv_bfloat16* pr = p0 + (int64_t)warp_id * v_stride1;
            cp_async_16B_unchk_u32(sK_load_u32[0] + KV_PLANE_BYTES, pr);
            cp_async_16B_unchk_u32(sK_load_u32[1] + KV_PLANE_BYTES, pr + v_step4);
            cp_async_16B_unchk_u32(sK_load_u32[2] + KV_PLANE_BYTES, pr + (v_step4 << 1));
            cp_async_16B_unchk_u32(sK_load_u32[3] + KV_PLANE_BYTES, pr + v_step4 * 3);
        } else {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int r = warp_id + (u << 2);
                bool ok = (tok0 + r) < seq_len;
                cp_async_16B_zfill_u32(sK_load_u32[u] + KV_PLANE_BYTES, p0 + (ok ? (int64_t)r * v_stride1 : 0), ok);
            }
        }
    };

    int cur_tile = my_t0, cur_off = 0;
    int64_t cur_pblk = 0;
    resolve_tile(cur_tile, cur_off, cur_pblk);
    issue_k(cur_pblk, cur_off, cur_tile);
    cp_async_commit();
    issue_v(cur_pblk, cur_off, cur_tile);
    cp_async_commit();
    cp_async_wait<1>(); __syncthreads();

    const float scale_log2 = scale * LOG2E;

    if constexpr (FAST1) {
        if (my_tiles == 1) {
            int tok_base = cur_tile << 4;
            float s_frag[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int s = 0; s < NUM_SLICES; ++s) {
                uint32_t k_reg[4];
                ldmatrix_u32_x4(k_reg, ldm_k_u32[s]);
                mma_m16n8k16_bf16(s_frag, k_reg, q_reg[s]);
            }
            int h0 = 2 * quad_tid, h1 = h0 + 1;
            smem.logits[warp_id][group_id][h0] = s_frag[0]; smem.logits[warp_id][group_id][h1] = s_frag[1];
            smem.logits[warp_id][group_id + 8][h0] = s_frag[2]; smem.logits[warp_id][group_id + 8][h1] = s_frag[3];
            cp_async_wait<0>(); __syncthreads();
            int h_me = group_id, k0 = 2 * quad_tid, k1 = k0 + 1, k2 = k0 + 8, k3 = k0 + 9;
            float l0 = smem.logits[0][k0][h_me] + smem.logits[1][k0][h_me] + smem.logits[2][k0][h_me] + smem.logits[3][k0][h_me];
            float l1 = smem.logits[0][k1][h_me] + smem.logits[1][k1][h_me] + smem.logits[2][k1][h_me] + smem.logits[3][k1][h_me];
            float l2 = smem.logits[0][k2][h_me] + smem.logits[1][k2][h_me] + smem.logits[2][k2][h_me] + smem.logits[3][k2][h_me];
            float l3 = smem.logits[0][k3][h_me] + smem.logits[1][k3][h_me] + smem.logits[2][k3][h_me] + smem.logits[3][k3][h_me];
            int g0 = tok_base + k0, g1 = tok_base + k1, g2 = tok_base + k2, g3 = tok_base + k3;
            l0 = (g0 >= k_min && g0 < seq_len) ? (l0 * scale_log2) : -INFINITY;
            l1 = (g1 >= k_min && g1 < seq_len) ? (l1 * scale_log2) : -INFINITY;
            l2 = (g2 >= k_min && g2 < seq_len) ? (l2 * scale_log2) : -INFINITY;
            l3 = (g3 >= k_min && g3 < seq_len) ? (l3 * scale_log2) : -INFINITY;
            float m_tile = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 1));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 2));
            float sub_m_h = (m_tile == -INFINITY) ? 0.f : m_tile;
            float p0 = (l0 == -INFINITY) ? 0.f : exp2f(l0 - sub_m_h);
            float p1 = (l1 == -INFINITY) ? 0.f : exp2f(l1 - sub_m_h);
            float p2 = (l2 == -INFINITY) ? 0.f : exp2f(l2 - sub_m_h);
            float p3 = (l3 == -INFINITY) ? 0.f : exp2f(l3 - sub_m_h);
            float l_tile = (p0 + p1) + (p2 + p3);
            l_tile += __shfl_xor_sync(0xffffffff, l_tile, 1);
            l_tile += __shfl_xor_sync(0xffffffff, l_tile, 2);
            row_l_h = l_tile; row_m_h = m_tile;
            uint32_t p_reg[2] = {pack_bf16x2(p0, p1), pack_bf16x2(p2, p3)};
            #pragma unroll
            for (int s = 0; s < NUM_SLICES; ++s) {
                uint32_t v_reg[4];
                ldmatrix_u32_x4_trans(v_reg, ldm_v_u32[s]);
                mma_m16n8k16_bf16(acc_o[s], v_reg, p_reg);
            }
            goto WRITE_OUT_SPLITK;
        }
    }

    for (int j = 0; j < my_tiles; ++j) {
        int tok_base = cur_tile << 4;
        if (j > 0) {
            issue_v(cur_pblk, cur_off, cur_tile);
            cp_async_commit();
        }

        float s_frag[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int s = 0; s < NUM_SLICES; ++s) {
            uint32_t k_reg[4];
            ldmatrix_u32_x4(k_reg, ldm_k_u32[s]);
            mma_m16n8k16_bf16(s_frag, k_reg, q_reg[s]);
        }

        int h0 = 2 * quad_tid, h1 = h0 + 1;
        smem.logits[warp_id][group_id][h0] = s_frag[0]; smem.logits[warp_id][group_id][h1] = s_frag[1];
        smem.logits[warp_id][group_id + 8][h0] = s_frag[2]; smem.logits[warp_id][group_id + 8][h1] = s_frag[3];

        cp_async_wait<0>(); __syncthreads();

        int nxt_tile = cur_tile + 1, nxt_off = 0;
        int64_t nxt_pblk = 0;
        if (j + 1 < my_tiles) {
            if constexpr (BS32) {
                nxt_off = (nxt_tile & 1) << 4;
                nxt_pblk = (nxt_off != 0) ? cur_pblk : (int64_t)bt_row[nxt_tile >> 1];
            } else {
                resolve_tile(nxt_tile, nxt_off, nxt_pblk);
            }
            issue_k(nxt_pblk, nxt_off, nxt_tile);
            cp_async_commit();
        }

        int h_me = group_id, k0 = 2 * quad_tid, k1 = k0 + 1, k2 = k0 + 8, k3 = k0 + 9;
        float l0 = smem.logits[0][k0][h_me] + smem.logits[1][k0][h_me] + smem.logits[2][k0][h_me] + smem.logits[3][k0][h_me];
        float l1 = smem.logits[0][k1][h_me] + smem.logits[1][k1][h_me] + smem.logits[2][k1][h_me] + smem.logits[3][k1][h_me];
        float l2 = smem.logits[0][k2][h_me] + smem.logits[1][k2][h_me] + smem.logits[2][k2][h_me] + smem.logits[3][k2][h_me];
        float l3 = smem.logits[0][k3][h_me] + smem.logits[1][k3][h_me] + smem.logits[2][k3][h_me] + smem.logits[3][k3][h_me];

        int g0 = tok_base + k0, g1 = tok_base + k1, g2 = tok_base + k2, g3 = tok_base + k3;
        l0 = (g0 >= k_min && g0 < seq_len) ? (l0 * scale_log2) : -INFINITY;
        l1 = (g1 >= k_min && g1 < seq_len) ? (l1 * scale_log2) : -INFINITY;
        l2 = (g2 >= k_min && g2 < seq_len) ? (l2 * scale_log2) : -INFINITY;
        l3 = (g3 >= k_min && g3 < seq_len) ? (l3 * scale_log2) : -INFINITY;

        float m_tile = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 1));
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 2));

        float new_m_h = fmaxf(row_m_h, m_tile);
        float alpha_h = (row_m_h == -INFINITY || new_m_h == -INFINITY) ? 0.f : exp2f(row_m_h - new_m_h);
        float sub_m_h = (new_m_h == -INFINITY) ? 0.f : new_m_h;

        float p0 = (l0 == -INFINITY) ? 0.f : exp2f(l0 - sub_m_h);
        float p1 = (l1 == -INFINITY) ? 0.f : exp2f(l1 - sub_m_h);
        float p2 = (l2 == -INFINITY) ? 0.f : exp2f(l2 - sub_m_h);
        float p3 = (l3 == -INFINITY) ? 0.f : exp2f(l3 - sub_m_h);

        float l_tile = (p0 + p1) + (p2 + p3);
        l_tile += __shfl_xor_sync(0xffffffff, l_tile, 1);
        l_tile += __shfl_xor_sync(0xffffffff, l_tile, 2);
        row_l_h = row_l_h * alpha_h + l_tile; row_m_h = new_m_h;

        uint32_t p_reg[2] = {pack_bf16x2(p0, p1), pack_bf16x2(p2, p3)};
        float alpha0 = __shfl_sync(0xffffffff, alpha_h, h0 << 2), alpha1 = __shfl_sync(0xffffffff, alpha_h, h1 << 2);
        if (j > 0) {
            #pragma unroll
            for (int s = 0; s < NUM_SLICES; ++s) { acc_o[s][0] *= alpha0; acc_o[s][1] *= alpha1; acc_o[s][2] *= alpha0; acc_o[s][3] *= alpha1; }
        }
        #pragma unroll
        for (int s = 0; s < NUM_SLICES; ++s) {
            uint32_t v_reg[4];
            ldmatrix_u32_x4_trans(v_reg, ldm_v_u32[s]);
            mma_m16n8k16_bf16(acc_o[s], v_reg, p_reg);
        }
        if (j + 1 < my_tiles) {
            cur_tile = nxt_tile; cur_off = nxt_off; cur_pblk = nxt_pblk;
            cp_async_wait<0>(); __syncthreads();
        }
    }

WRITE_OUT_SPLITK:;
    const int qh_base = kv_head * NUM_Q_PER_KV;
    if (warp_id == 0 && quad_tid == 0) {
        int64_t ml_idx = ((int64_t)seq_idx * total_q_heads + (qh_base + group_id)) * num_splits + split_idx;
        segm_max[ml_idx] = row_m_h; segm_expsum[ml_idx] = row_l_h;
    }
    const int h_e0 = qh_base + 2 * quad_tid, h_e1 = h_e0 + 1;
    const int64_t out_off0 = (((int64_t)seq_idx * total_q_heads + h_e0) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    const int64_t out_off1 = (((int64_t)seq_idx * total_q_heads + h_e1) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) {
        int d0 = s * 16 + group_id, d1 = d0 + 8;
        segm_out[out_off0 + d0] = acc_o[s][0]; segm_out[out_off1 + d0] = acc_o[s][1];
        segm_out[out_off0 + d1] = acc_o[s][2]; segm_out[out_off1 + d1] = acc_o[s][3];
    }
}

template <int NUM_Q_PER_KV = 8, bool BS32 = false>
__global__ __launch_bounds__(128)
void sm80_paged_decode_kv4w_hd256_kernel(
    float* __restrict__ segm_out,
    float* __restrict__ segm_max,
    float* __restrict__ segm_expsum,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    int sliding_window_val,
    int num_seqs,
    int num_kv_heads,
    int num_splits,
    int block_size,
    int64_t bt_stride0,
    int64_t q_stride0,
    int64_t q_stride1,
    int64_t k_stride0,
    int64_t k_stride1,
    int64_t k_stride2,
    int64_t v_stride0,
    int64_t v_stride1,
    int64_t v_stride2)
{
    static_assert(NUM_Q_PER_KV == 8, "Specialized for nq_per_kv == 8");
    constexpr int CHUNKS = 32;
    constexpr int NUM_SLICES = 16;
    constexpr float LOG2E = 1.4426950408889634f;

    const int seq_idx = blockIdx.x, kv_head = blockIdx.y, split_idx = blockIdx.z;
    const int total_q_heads = num_kv_heads * NUM_Q_PER_KV;
    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31, group_id = lane >> 2, quad_tid = lane & 3;
    int q_tok = seq_idx;
    if (cu_seqlens_q != nullptr) { q_tok = cu_seqlens_q[seq_idx]; if (cu_seqlens_q[seq_idx + 1] <= q_tok) return; }
    const int seq_len = seqused_k[seq_idx];
    int k_min = (sliding_window_val > 0 && seq_len > sliding_window_val) ? (seq_len - sliding_window_val) : 0;
    const int tile_lo_g = k_min >> 4, tile_hi_g = (seq_len + 15) >> 4;
    const int n_active_tiles = (seq_len > 0 && tile_hi_g > tile_lo_g) ? (tile_hi_g - tile_lo_g) : 0;
    const int t_per = n_active_tiles / num_splits, t_rem = n_active_tiles - t_per * num_splits;
    const int my_tiles = t_per + (split_idx < t_rem ? 1 : 0);
    const int my_t0 = tile_lo_g + split_idx * t_per + min(split_idx, t_rem);
    if (my_tiles <= 0) return;

    extern __shared__ char smem_raw[];
    uint4* smem_u4 = reinterpret_cast<uint4*>(smem_raw);
    uint4* bK_w = smem_u4 + warp_id * (2 * 16 * CHUNKS);
    uint4* bV_w = bK_w + (16 * CHUNKS);
    const uint32_t bK_u32 = smem_u32(bK_w), bV_u32 = smem_u32(bV_w);

    const bool blk_pow2 = BS32 ? true : ((block_size & (block_size - 1)) == 0);
    const int blk_shift = BS32 ? 5 : (blk_pow2 ? (__ffs(block_size) - 1) : 0);
    const int blk_mask = BS32 ? 31 : (block_size - 1);
    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;

    uint32_t ld_off[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        ld_off[i] = static_cast<uint32_t>(swz256(i, lane)) * 16u;
    }
    const int64_t k_s1_bytes = k_stride1 * sizeof(__nv_bfloat16), v_s1_bytes = v_stride1 * sizeof(__nv_bfloat16);
    const int64_t col_bytes = static_cast<int64_t>(lane) * (8 * sizeof(__nv_bfloat16));
    const __nv_bfloat16* k_head_base = k_cache + (int64_t)kv_head * k_stride2;
    const __nv_bfloat16* v_head_base = v_cache + (int64_t)kv_head * v_stride2;

    auto issue_warp_load_16 = [&](uint32_t dst_u32, const __nv_bfloat16* cache_head_base,
                                  int64_t s0, int64_t s1, int64_t s1_bytes,
                                  int tile_idx, bool masked) {
        int tok0 = tile_idx << 4;
        int page = BS32 ? (tok0 >> 5) : (blk_pow2 ? (tok0 >> blk_shift) : (tok0 / block_size));
        int off  = BS32 ? ((tile_idx & 1) << 4) : (blk_pow2 ? (tok0 & blk_mask) : (tok0 - page * block_size));
        int64_t pblk = (int64_t)bt_row[page];
        const char* ptr_b = reinterpret_cast<const char*>(cache_head_base + pblk * s0 + (int64_t)off * s1) + col_bytes;
        const char* cur = ptr_b;
        int rem = seq_len - tok0, r_cur = 0;
        #pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            uint32_t h_off = dst_u32 + static_cast<uint32_t>(hi * 4096);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                bool ok = r_cur < rem;
                cp_async_16B_zfill_u32(h_off + ld_off[i], ok ? cur : ptr_b, ok);
                cur += s1_bytes;
                ++r_cur;
            }
        }
    };

    const uint32_t* q_u32 = reinterpret_cast<const uint32_t*>(
        q + (int64_t)q_tok * q_stride0 + (int64_t)(kv_head * NUM_Q_PER_KV + group_id) * q_stride1);
    uint32_t q_reg[NUM_SLICES][2];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) { q_reg[s][0] = q_u32[s * 8 + quad_tid]; q_reg[s][1] = q_u32[s * 8 + 4 + quad_tid]; }
    float acc_o[NUM_SLICES][4];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) { acc_o[s][0] = acc_o[s][1] = acc_o[s][2] = acc_o[s][3] = 0.f; }
    float row_m_h = -INFINITY, row_l_h = 0.0f;
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col_chunk = (lane >> 4);
    const int ldm_bk_row = (lane >> 4) * 8 + (lane & 7), ldm_bk_col_chunk = ((lane >> 3) & 1);
    const float scale_log2 = scale * LOG2E;
    uint32_t k_base[4], v_base[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        k_base[u] = bK_u32 + static_cast<uint32_t>(swz256(ldm_a_row, u * 2 + ldm_a_col_chunk)) * 16u;
        v_base[u] = bV_u32 + static_cast<uint32_t>(swz256(ldm_bk_row, u * 2 + ldm_bk_col_chunk)) * 16u;
    }
    if (warp_id < my_tiles) {
        int tile0 = my_t0 + warp_id;
        issue_warp_load_16(bK_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, tile0, ((tile0 << 4) + 16) > seq_len);
        cp_async_commit();
    }
    for (int j = warp_id; j < my_tiles; j += 4) {
        int tile_idx = my_t0 + j, tok_base = tile_idx << 4;
        issue_warp_load_16(bV_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, tile_idx, (tok_base + 16) > seq_len);
        cp_async_commit();
        cp_async_wait<1>();
        __syncwarp();
        float s_frag[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            uint32_t t_bytes = static_cast<uint32_t>(t * 128);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                uint32_t k_reg[4];
                ldmatrix_u32_x4(k_reg, k_base[u] + t_bytes);
                mma_m16n8k16_bf16(s_frag, k_reg, q_reg[t * 4 + u]);
            }
        }
        __syncwarp();
        const bool has_next = (j + 4 < my_tiles);
        if (has_next) {
            int nxt_tile = tile_idx + 4;
            issue_warp_load_16(bK_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, nxt_tile, ((nxt_tile << 4) + 16) > seq_len);
            cp_async_commit();
        }
        const int src_q = group_id >> 1, h_odd = group_id & 1;
        const int src_lane0 = ((2 * quad_tid) << 2) + src_q, src_lane1 = ((2 * quad_tid + 1) << 2) + src_q;
        float s00 = __shfl_sync(0xffffffff, s_frag[0], src_lane0), s01 = __shfl_sync(0xffffffff, s_frag[1], src_lane0);
        float s10 = __shfl_sync(0xffffffff, s_frag[0], src_lane1), s11 = __shfl_sync(0xffffffff, s_frag[1], src_lane1);
        float s20 = __shfl_sync(0xffffffff, s_frag[2], src_lane0), s21 = __shfl_sync(0xffffffff, s_frag[3], src_lane0);
        float s30 = __shfl_sync(0xffffffff, s_frag[2], src_lane1), s31 = __shfl_sync(0xffffffff, s_frag[3], src_lane1);
        float l0 = h_odd ? s01 : s00, l1 = h_odd ? s11 : s10, l2 = h_odd ? s21 : s20, l3 = h_odd ? s31 : s30;
        int k0 = 2 * quad_tid, k1 = k0 + 1, k2 = k0 + 8, k3 = k0 + 9;
        int g0 = tok_base + k0, g1 = tok_base + k1, g2 = tok_base + k2, g3 = tok_base + k3;
        l0 = (g0 >= k_min && g0 < seq_len) ? (l0 * scale_log2) : -INFINITY;
        l1 = (g1 >= k_min && g1 < seq_len) ? (l1 * scale_log2) : -INFINITY;
        l2 = (g2 >= k_min && g2 < seq_len) ? (l2 * scale_log2) : -INFINITY;
        l3 = (g3 >= k_min && g3 < seq_len) ? (l3 * scale_log2) : -INFINITY;
        float m_tile = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 1));
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 2));
        float new_m_h = fmaxf(row_m_h, m_tile);
        float alpha_h = (row_m_h == -INFINITY || new_m_h == -INFINITY) ? 0.f : fa2_ex2_approx(row_m_h - new_m_h);
        float sub_m_h = (new_m_h == -INFINITY) ? 0.f : new_m_h;
        float p0 = (l0 == -INFINITY) ? 0.f : fa2_ex2_approx(l0 - sub_m_h), p1 = (l1 == -INFINITY) ? 0.f : fa2_ex2_approx(l1 - sub_m_h);
        float p2 = (l2 == -INFINITY) ? 0.f : fa2_ex2_approx(l2 - sub_m_h), p3 = (l3 == -INFINITY) ? 0.f : fa2_ex2_approx(l3 - sub_m_h);
        float l_tile = (p0 + p1) + (p2 + p3);
        l_tile += __shfl_xor_sync(0xffffffff, l_tile, 1); l_tile += __shfl_xor_sync(0xffffffff, l_tile, 2);
        row_l_h = row_l_h * alpha_h + l_tile; row_m_h = new_m_h;
        uint32_t p_reg[2] = {pack_bf16x2(p0, p1), pack_bf16x2(p2, p3)};
        int h0 = 2 * quad_tid, h1 = h0 + 1;
        float alpha0 = __shfl_sync(0xffffffff, alpha_h, h0 << 2), alpha1 = __shfl_sync(0xffffffff, alpha_h, h1 << 2);
        if (j > warp_id) {
            #pragma unroll
            for (int s = 0; s < NUM_SLICES; ++s) { acc_o[s][0] *= alpha0; acc_o[s][1] *= alpha1; acc_o[s][2] *= alpha0; acc_o[s][3] *= alpha1; }
        }
        if (has_next) cp_async_wait<1>(); else cp_async_wait<0>();
        __syncwarp();
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            uint32_t t_bytes = static_cast<uint32_t>(t * 128);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                uint32_t v_reg[4];
                ldmatrix_u32_x4_trans(v_reg, v_base[u] + t_bytes);
                mma_m16n8k16_bf16(acc_o[t * 4 + u], v_reg, p_reg);
            }
        }
        __syncwarp();
    }
    if (my_tiles == 1) {
        if (warp_id == 0) {
            const int qh_base = kv_head * NUM_Q_PER_KV;
            if (quad_tid == 0) {
                int64_t ml_idx = ((int64_t)seq_idx * total_q_heads + (qh_base + group_id)) * num_splits + split_idx;
                segm_max[ml_idx] = row_m_h; segm_expsum[ml_idx] = row_l_h;
            }
            const int h_e0 = qh_base + 2 * quad_tid, h_e1 = h_e0 + 1;
            const int64_t out_off0 = (((int64_t)seq_idx * total_q_heads + h_e0) * num_splits + split_idx) * 256;
            const int64_t out_off1 = (((int64_t)seq_idx * total_q_heads + h_e1) * num_splits + split_idx) * 256;
            #pragma unroll
            for (int s = 0; s < NUM_SLICES; ++s) {
                int d0 = s * 16 + group_id, d1 = d0 + 8;
                segm_out[out_off0 + d0] = acc_o[s][0]; segm_out[out_off1 + d0] = acc_o[s][1];
                segm_out[out_off0 + d1] = acc_o[s][2]; segm_out[out_off1 + d1] = acc_o[s][3];
            }
        }
        return;
    }
    constexpr int W_STRIDE = 4096;
    constexpr int R_STRIDE = 260;
    float* smem_f = reinterpret_cast<float*>(smem_raw);
    float* w_base = smem_f + warp_id * W_STRIDE;
    float* s_warp_m = w_base + 8 * R_STRIDE;
    float* s_warp_l = s_warp_m + 8;
    if (quad_tid == 0) { s_warp_m[group_id] = row_m_h; s_warp_l[group_id] = row_l_h; }
    const int h_e0 = 2 * quad_tid, h_e1 = h_e0 + 1;
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) {
        int d0 = s * 16 + group_id, d1 = d0 + 8;
        w_base[h_e0 * R_STRIDE + d0] = acc_o[s][0]; w_base[h_e1 * R_STRIDE + d0] = acc_o[s][1];
        w_base[h_e0 * R_STRIDE + d1] = acc_o[s][2]; w_base[h_e1 * R_STRIDE + d1] = acc_o[s][3];
    }
    __syncthreads();
    const int qh_base = kv_head * NUM_Q_PER_KV;
    if (tid < 8) {
        int h = tid;
        float m0 = smem_f[0 * W_STRIDE + 8 * R_STRIDE + h], m1 = smem_f[1 * W_STRIDE + 8 * R_STRIDE + h];
        float m2 = smem_f[2 * W_STRIDE + 8 * R_STRIDE + h], m3 = smem_f[3 * W_STRIDE + 8 * R_STRIDE + h];
        float gm = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
        float e0 = (m0 > -INFINITY) ? fa2_ex2_approx(m0 - gm) : 0.f;
        float e1 = (m1 > -INFINITY) ? fa2_ex2_approx(m1 - gm) : 0.f;
        float e2 = (m2 > -INFINITY) ? fa2_ex2_approx(m2 - gm) : 0.f;
        float e3 = (m3 > -INFINITY) ? fa2_ex2_approx(m3 - gm) : 0.f;
        float gl = ((m0 > -INFINITY) ? smem_f[0 * W_STRIDE + 8 * R_STRIDE + 8 + h] * e0 : 0.f)
                 + ((m1 > -INFINITY) ? smem_f[1 * W_STRIDE + 8 * R_STRIDE + 8 + h] * e1 : 0.f)
                 + ((m2 > -INFINITY) ? smem_f[2 * W_STRIDE + 8 * R_STRIDE + 8 + h] * e2 : 0.f)
                 + ((m3 > -INFINITY) ? smem_f[3 * W_STRIDE + 8 * R_STRIDE + 8 + h] * e3 : 0.f);
        smem_f[0 * W_STRIDE + 8 * R_STRIDE + h] = e0; smem_f[1 * W_STRIDE + 8 * R_STRIDE + h] = e1;
        smem_f[2 * W_STRIDE + 8 * R_STRIDE + h] = e2; smem_f[3 * W_STRIDE + 8 * R_STRIDE + h] = e3;
        int64_t ml_idx = ((int64_t)seq_idx * total_q_heads + (qh_base + h)) * num_splits + split_idx;
        segm_max[ml_idx] = gm; segm_expsum[ml_idx] = gl;
    }
    __syncthreads();
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        int idx4 = it * 128 + tid, h = idx4 >> 6, d4 = (idx4 & 63) << 2;
        float e0 = smem_f[0 * W_STRIDE + 8 * R_STRIDE + h], e1 = smem_f[1 * W_STRIDE + 8 * R_STRIDE + h];
        float e2 = smem_f[2 * W_STRIDE + 8 * R_STRIDE + h], e3 = smem_f[3 * W_STRIDE + 8 * R_STRIDE + h];
        float4 v0 = *reinterpret_cast<const float4*>(&smem_f[0 * W_STRIDE + h * R_STRIDE + d4]);
        float4 v1 = *reinterpret_cast<const float4*>(&smem_f[1 * W_STRIDE + h * R_STRIDE + d4]);
        float4 v2 = *reinterpret_cast<const float4*>(&smem_f[2 * W_STRIDE + h * R_STRIDE + d4]);
        float4 v3 = *reinterpret_cast<const float4*>(&smem_f[3 * W_STRIDE + h * R_STRIDE + d4]);
        float4 res; res.x = v0.x*e0 + v1.x*e1 + v2.x*e2 + v3.x*e3; res.y = v0.y*e0 + v1.y*e1 + v2.y*e2 + v3.y*e3;
        res.z = v0.z*e0 + v1.z*e1 + v2.z*e2 + v3.z*e3; res.w = v0.w*e0 + v1.w*e1 + v2.w*e2 + v3.w*e3;
        int64_t dst_off = (((int64_t)seq_idx * total_q_heads + (qh_base + h)) * num_splits + split_idx) * 256 + d4;
        *reinterpret_cast<float4*>(&segm_out[dst_off]) = res;
    }
}

template <int HEAD_SIZE, int NUM_Q_PER_KV = 8, bool BS32 = false, bool BS16 = false, bool FAST1 = false>
__global__ __launch_bounds__(128)
void sm80_paged_decode_mma_bn32_kernel(
    float* __restrict__ segm_out,
    float* __restrict__ segm_max,
    float* __restrict__ segm_expsum,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    int sliding_window_val,
    int num_seqs,
    int num_kv_heads,
    int num_splits,
    int block_size,
    int64_t bt_stride0,
    int64_t q_stride0,
    int64_t q_stride1,
    int64_t k_stride0,
    int64_t k_stride1,
    int64_t k_stride2,
    int64_t v_stride0,
    int64_t v_stride1,
    int64_t v_stride2)
{
    static_assert(HEAD_SIZE == 512, "Specialized for hd=512");
    constexpr int D_PER_WARP = HEAD_SIZE / 4, NUM_SLICES = D_PER_WARP / 16, CHUNKS = 64;
    constexpr float LOG2E = 1.4426950408889634f;
    const int seq_idx = blockIdx.x, kv_head = blockIdx.y, split_idx = blockIdx.z;
    const int total_q_heads = num_kv_heads * NUM_Q_PER_KV;
    const uint32_t tid = threadIdx.x;
    const int warp_id = static_cast<int>(tid >> 5), lane = static_cast<int>(tid & 31u);
    const int group_id = lane >> 2, quad_tid = lane & 3;
    int q_tok = seq_idx;
    if (cu_seqlens_q != nullptr) { q_tok = cu_seqlens_q[seq_idx]; if (cu_seqlens_q[seq_idx + 1] <= q_tok) return; }
    const int seq_len = seqused_k[seq_idx];
    int k_min = (sliding_window_val > 0 && seq_len > sliding_window_val) ? (seq_len - sliding_window_val) : 0;
    const int tile_lo_g = k_min >> 4, tile_hi_g = (seq_len + 15) >> 4;
    const int n_active_tiles = (seq_len > 0 && tile_hi_g > tile_lo_g) ? (tile_hi_g - tile_lo_g) : 0;
    const int eff_splits = (gridDim.x == 1 && num_splits == 64 && sliding_window_val == 0 && seq_len <= 2560) ? 32 : num_splits;
    if (split_idx >= eff_splits) return;
    const int t_per = n_active_tiles / eff_splits, t_rem = n_active_tiles - t_per * eff_splits;
    const int my_tiles = t_per + (split_idx < t_rem ? 1 : 0);
    const int my_t0 = tile_lo_g + split_idx * t_per + min(split_idx, t_rem);
    if (my_tiles <= 0) return;

    extern __shared__ char smem_raw[];
    uint4* bK0 = reinterpret_cast<uint4*>(smem_raw); uint4* bK1 = bK0 + 16 * CHUNKS;
    uint4* bV0 = bK1 + 16 * CHUNKS; uint4* bV1 = bV0 + 16 * CHUNKS;
    float* s_logits_flat = reinterpret_cast<float*>(bV1 + 16 * CHUNKS);
    auto s_log = [&](int t2, int w, int k, int h) -> float& { return s_logits_flat[((t2 * 4 + w) * 16 + k) * 9 + h]; };
    const uint32_t bK0_u32 = smem_u32(bK0), bK1_u32 = smem_u32(bK1), bV0_u32 = smem_u32(bV0), bV1_u32 = smem_u32(bV1);
    const bool blk_pow2 = (BS16 || BS32) ? true : ((block_size & (block_size - 1)) == 0);
    const int blk_shift = BS32 ? 5 : (blk_pow2 ? (__ffs(block_size) - 1) : 0), blk_mask = BS32 ? 31 : (block_size - 1);
    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;
    const uint32_t r0 = tid >> 6, c0 = tid & 63u;
    uint32_t ld_dst[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) ld_dst[i] = static_cast<uint32_t>(swz512(static_cast<int>(r0) + i * 2, static_cast<int>(c0))) * 16u;
    const int64_t k_s1_bytes = k_stride1 * sizeof(__nv_bfloat16), k_s1_2_bytes = k_s1_bytes * 2;
    const int64_t v_s1_bytes = v_stride1 * sizeof(__nv_bfloat16), v_s1_2_bytes = v_s1_bytes * 2;
    const int64_t col_bytes = static_cast<int64_t>(c0) * (8 * sizeof(__nv_bfloat16));
    const __nv_bfloat16* k_head_base = k_cache + (int64_t)kv_head * k_stride2;
    const __nv_bfloat16* v_head_base = v_cache + (int64_t)kv_head * v_stride2;

    auto issue_load_16 = [&](uint32_t dst_u32, const __nv_bfloat16* cache_head_base,
                             int64_t s0, int64_t s1, int64_t s1_bytes, int64_t s1_2_bytes, int tile_idx, bool masked) {
        int tok0 = tile_idx << 4;
        int page = BS16 ? tile_idx : (BS32 ? (tok0 >> 5) : (blk_pow2 ? (tok0 >> blk_shift) : (tok0 / block_size)));
        int off  = (BS16 || BS32) ? (BS16 ? 0 : ((tile_idx & 1) << 4)) : (blk_pow2 ? (tok0 & blk_mask) : (tok0 - page * block_size));
        int64_t pblk = (int64_t)bt_row[page];
        const char* ptr_b = BS16
            ? (reinterpret_cast<const char*>(cache_head_base + pblk * s0) + col_bytes)
            : (reinterpret_cast<const char*>(cache_head_base + pblk * s0 + (int64_t)off * s1) + col_bytes);
        const char* cur = ptr_b + static_cast<int64_t>(r0) * s1_bytes;
        int rem = seq_len - tok0, r_cur = static_cast<int>(r0);
        #pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            uint32_t h_off = dst_u32 + static_cast<uint32_t>(hi * 8192);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                bool ok = r_cur < rem;
                cp_async_16B_zfill_u32(h_off + ld_dst[i], (BS16 || BS32) ? cur : (ok ? cur : ptr_b), ok);
                cur += s1_2_bytes; r_cur += 2;
            }
        }
    };

    const int d_warp = warp_id * D_PER_WARP, d_warp_chunks = d_warp >> 3;
    const uint32_t* q_u32 = reinterpret_cast<const uint32_t*>(
        q + (int64_t)q_tok * q_stride0 + (int64_t)(kv_head * NUM_Q_PER_KV + group_id) * q_stride1 + d_warp);
    uint32_t q_reg[NUM_SLICES][2];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) { q_reg[s][0] = q_u32[s * 8 + quad_tid]; q_reg[s][1] = q_u32[s * 8 + 4 + quad_tid]; }
    float acc_o[NUM_SLICES][4];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) { acc_o[s][0] = acc_o[s][1] = acc_o[s][2] = acc_o[s][3] = 0.f; }
    float row_m_h = -INFINITY, row_l_h = 0.0f;
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col_chunk = (lane >> 4);
    const int ldm_bk_row = (lane >> 4) * 8 + (lane & 7), ldm_bk_col_chunk = ((lane >> 3) & 1);
    const float scale_log2 = scale * LOG2E;
    uint32_t sK_off[4], sV_off[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        sK_off[u] = static_cast<uint32_t>(swz512(ldm_a_row, d_warp_chunks + u * 2 + ldm_a_col_chunk)) * 16u;
        sV_off[u] = static_cast<uint32_t>(swz512(ldm_bk_row, d_warp_chunks + u * 2 + ldm_bk_col_chunk)) * 16u;
    }
    issue_load_16(bK0_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, my_t0, ((my_t0 << 4) + 16) > seq_len);
    if constexpr (FAST1) {
        if (my_tiles == 1) {
            cp_async_commit();
            issue_load_16(bV0_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_2_bytes, my_t0, ((my_t0 << 4) + 16) > seq_len);
            cp_async_commit();
            cp_async_wait<1>(); __syncthreads();
            int h0 = 2 * quad_tid, h1 = h0 + 1;
            float s_frag0[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t t_bytes = static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t k_reg[4]; ldmatrix_u32_x4(k_reg, bK0_u32 + sK_off[u] + t_bytes);
                    mma_m16n8k16_bf16(s_frag0, k_reg, q_reg[t * 4 + u]);
                }
            }
            s_log(0, warp_id, group_id, h0) = s_frag0[0]; s_log(0, warp_id, group_id, h1) = s_frag0[1];
            s_log(0, warp_id, group_id + 8, h0) = s_frag0[2]; s_log(0, warp_id, group_id + 8, h1) = s_frag0[3];
            cp_async_wait<0>(); __syncthreads();
            int h_me = group_id, k0 = 2 * quad_tid, k1 = k0 + 1, k2 = k0 + 8, k3 = k0 + 9;
            float l0 = s_log(0, 0, k0, h_me) + s_log(0, 1, k0, h_me) + s_log(0, 2, k0, h_me) + s_log(0, 3, k0, h_me);
            float l1 = s_log(0, 0, k1, h_me) + s_log(0, 1, k1, h_me) + s_log(0, 2, k1, h_me) + s_log(0, 3, k1, h_me);
            float l2 = s_log(0, 0, k2, h_me) + s_log(0, 1, k2, h_me) + s_log(0, 2, k2, h_me) + s_log(0, 3, k2, h_me);
            float l3 = s_log(0, 0, k3, h_me) + s_log(0, 1, k3, h_me) + s_log(0, 2, k3, h_me) + s_log(0, 3, k3, h_me);
            int tok_base = my_t0 << 4; int g0 = tok_base + k0, g1 = tok_base + k1, g2 = tok_base + k2, g3 = tok_base + k3;
            l0 = (g0 >= k_min && g0 < seq_len) ? (l0 * scale_log2) : -INFINITY;
            l1 = (g1 >= k_min && g1 < seq_len) ? (l1 * scale_log2) : -INFINITY;
            l2 = (g2 >= k_min && g2 < seq_len) ? (l2 * scale_log2) : -INFINITY;
            l3 = (g3 >= k_min && g3 < seq_len) ? (l3 * scale_log2) : -INFINITY;
            float m_tile = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 1)); m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 2));
            float sub_m_h = (m_tile == -INFINITY) ? 0.f : m_tile;
            float p0 = (l0 == -INFINITY) ? 0.f : exp2f(l0 - sub_m_h), p1 = (l1 == -INFINITY) ? 0.f : exp2f(l1 - sub_m_h);
            float p2 = (l2 == -INFINITY) ? 0.f : exp2f(l2 - sub_m_h), p3 = (l3 == -INFINITY) ? 0.f : exp2f(l3 - sub_m_h);
            float l_tile = (p0 + p1) + (p2 + p3);
            l_tile += __shfl_xor_sync(0xffffffff, l_tile, 1); l_tile += __shfl_xor_sync(0xffffffff, l_tile, 2);
            row_l_h = l_tile; row_m_h = m_tile;
            uint32_t p_reg[2] = {pack_bf16x2(p0, p1), pack_bf16x2(p2, p3)};
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t t_bytes = static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t v_reg[4]; ldmatrix_u32_x4_trans(v_reg, bV0_u32 + sV_off[u] + t_bytes);
                    mma_m16n8k16_bf16(acc_o[t * 4 + u], v_reg, p_reg);
                }
            }
            goto WRITE_OUT_BN32;
        }
    }
    if (1 < my_tiles) { int t1 = my_t0 + 1; issue_load_16(bK1_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, t1, ((t1 << 4) + 16) > seq_len); }
    cp_async_commit(); cp_async_wait<0>(); __syncthreads();
    for (int j = 0; j < my_tiles; j += 2) {
        int t0 = my_t0 + j; bool has1 = (j + 1 < my_tiles); int t1 = t0 + 1;
        issue_load_16(bV0_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_2_bytes, t0, ((t0 << 4) + 16) > seq_len);
        if (has1) issue_load_16(bV1_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_2_bytes, t1, ((t1 << 4) + 16) > seq_len);
        cp_async_commit();
        int h0 = 2 * quad_tid, h1 = h0 + 1;
        float s_frag0[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            uint32_t t_bytes = static_cast<uint32_t>(t * 128);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                uint32_t k_reg[4]; ldmatrix_u32_x4(k_reg, bK0_u32 + sK_off[u] + t_bytes);
                mma_m16n8k16_bf16(s_frag0, k_reg, q_reg[t * 4 + u]);
            }
        }
        s_log(0, warp_id, group_id, h0) = s_frag0[0]; s_log(0, warp_id, group_id, h1) = s_frag0[1];
        s_log(0, warp_id, group_id + 8, h0) = s_frag0[2]; s_log(0, warp_id, group_id + 8, h1) = s_frag0[3];
        if (has1) {
            float s_frag1[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t t_bytes = static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t k_reg[4]; ldmatrix_u32_x4(k_reg, bK1_u32 + sK_off[u] + t_bytes);
                    mma_m16n8k16_bf16(s_frag1, k_reg, q_reg[t * 4 + u]);
                }
            }
            s_log(1, warp_id, group_id, h0) = s_frag1[0]; s_log(1, warp_id, group_id, h1) = s_frag1[1];
            s_log(1, warp_id, group_id + 8, h0) = s_frag1[2]; s_log(1, warp_id, group_id + 8, h1) = s_frag1[3];
        }
        cp_async_wait<0>(); __syncthreads();
        bool has_next = (j + 2 < my_tiles);
        if (has_next) {
            int nxt0 = t0 + 2; issue_load_16(bK0_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, nxt0, ((nxt0 << 4) + 16) > seq_len);
            if (j + 3 < my_tiles) { int nxt1 = t0 + 3; issue_load_16(bK1_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, nxt1, ((nxt1 << 4) + 16) > seq_len); }
            cp_async_commit();
        }
        int h_me = group_id, k0 = 2 * quad_tid, k1 = k0 + 1, k2 = k0 + 8, k3 = k0 + 9;
        {
            float l0 = s_log(0, 0, k0, h_me) + s_log(0, 1, k0, h_me) + s_log(0, 2, k0, h_me) + s_log(0, 3, k0, h_me);
            float l1 = s_log(0, 0, k1, h_me) + s_log(0, 1, k1, h_me) + s_log(0, 2, k1, h_me) + s_log(0, 3, k1, h_me);
            float l2 = s_log(0, 0, k2, h_me) + s_log(0, 1, k2, h_me) + s_log(0, 2, k2, h_me) + s_log(0, 3, k2, h_me);
            float l3 = s_log(0, 0, k3, h_me) + s_log(0, 1, k3, h_me) + s_log(0, 2, k3, h_me) + s_log(0, 3, k3, h_me);
            int tok_base = t0 << 4; int g0 = tok_base + k0, g1 = tok_base + k1, g2 = tok_base + k2, g3 = tok_base + k3;
            l0 = (g0 >= k_min && g0 < seq_len) ? (l0 * scale_log2) : -INFINITY;
            l1 = (g1 >= k_min && g1 < seq_len) ? (l1 * scale_log2) : -INFINITY;
            l2 = (g2 >= k_min && g2 < seq_len) ? (l2 * scale_log2) : -INFINITY;
            l3 = (g3 >= k_min && g3 < seq_len) ? (l3 * scale_log2) : -INFINITY;
            float m_tile = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 1)); m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 2));
            float new_m_h = fmaxf(row_m_h, m_tile);
            float alpha_h = (row_m_h == -INFINITY || new_m_h == -INFINITY) ? 0.f : exp2f(row_m_h - new_m_h);
            float sub_m_h = (new_m_h == -INFINITY) ? 0.f : new_m_h;
            float p0 = (l0 == -INFINITY) ? 0.f : exp2f(l0 - sub_m_h), p1 = (l1 == -INFINITY) ? 0.f : exp2f(l1 - sub_m_h);
            float p2 = (l2 == -INFINITY) ? 0.f : exp2f(l2 - sub_m_h), p3 = (l3 == -INFINITY) ? 0.f : exp2f(l3 - sub_m_h);
            float l_tile = (p0 + p1) + (p2 + p3);
            l_tile += __shfl_xor_sync(0xffffffff, l_tile, 1); l_tile += __shfl_xor_sync(0xffffffff, l_tile, 2);
            row_l_h = row_l_h * alpha_h + l_tile; row_m_h = new_m_h;
            uint32_t p_reg[2] = {pack_bf16x2(p0, p1), pack_bf16x2(p2, p3)};
            float alpha0 = __shfl_sync(0xffffffff, alpha_h, h0 << 2), alpha1 = __shfl_sync(0xffffffff, alpha_h, h1 << 2);
            if (j > 0) {
                #pragma unroll
                for (int s = 0; s < NUM_SLICES; ++s) { acc_o[s][0] *= alpha0; acc_o[s][1] *= alpha1; acc_o[s][2] *= alpha0; acc_o[s][3] *= alpha1; }
            }
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t t_bytes = static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t v_reg[4]; ldmatrix_u32_x4_trans(v_reg, bV0_u32 + sV_off[u] + t_bytes);
                    mma_m16n8k16_bf16(acc_o[t * 4 + u], v_reg, p_reg);
                }
            }
        }
        if (has1) {
            float l0 = s_log(1, 0, k0, h_me) + s_log(1, 1, k0, h_me) + s_log(1, 2, k0, h_me) + s_log(1, 3, k0, h_me);
            float l1 = s_log(1, 0, k1, h_me) + s_log(1, 1, k1, h_me) + s_log(1, 2, k1, h_me) + s_log(1, 3, k1, h_me);
            float l2 = s_log(1, 0, k2, h_me) + s_log(1, 1, k2, h_me) + s_log(1, 2, k2, h_me) + s_log(1, 3, k2, h_me);
            float l3 = s_log(1, 0, k3, h_me) + s_log(1, 1, k3, h_me) + s_log(1, 2, k3, h_me) + s_log(1, 3, k3, h_me);
            int tok_base = t1 << 4; int g0 = tok_base + k0, g1 = tok_base + k1, g2 = tok_base + k2, g3 = tok_base + k3;
            l0 = (g0 >= k_min && g0 < seq_len) ? (l0 * scale_log2) : -INFINITY;
            l1 = (g1 >= k_min && g1 < seq_len) ? (l1 * scale_log2) : -INFINITY;
            l2 = (g2 >= k_min && g2 < seq_len) ? (l2 * scale_log2) : -INFINITY;
            l3 = (g3 >= k_min && g3 < seq_len) ? (l3 * scale_log2) : -INFINITY;
            float m_tile = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 1)); m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 2));
            float new_m_h = fmaxf(row_m_h, m_tile);
            float alpha_h = (row_m_h == -INFINITY || new_m_h == -INFINITY) ? 0.f : exp2f(row_m_h - new_m_h);
            float sub_m_h = (new_m_h == -INFINITY) ? 0.f : new_m_h;
            float p0 = (l0 == -INFINITY) ? 0.f : exp2f(l0 - sub_m_h), p1 = (l1 == -INFINITY) ? 0.f : exp2f(l1 - sub_m_h);
            float p2 = (l2 == -INFINITY) ? 0.f : exp2f(l2 - sub_m_h), p3 = (l3 == -INFINITY) ? 0.f : exp2f(l3 - sub_m_h);
            float l_tile = (p0 + p1) + (p2 + p3);
            l_tile += __shfl_xor_sync(0xffffffff, l_tile, 1); l_tile += __shfl_xor_sync(0xffffffff, l_tile, 2);
            row_l_h = row_l_h * alpha_h + l_tile; row_m_h = new_m_h;
            uint32_t p_reg[2] = {pack_bf16x2(p0, p1), pack_bf16x2(p2, p3)};
            float alpha0 = __shfl_sync(0xffffffff, alpha_h, h0 << 2), alpha1 = __shfl_sync(0xffffffff, alpha_h, h1 << 2);
            #pragma unroll
            for (int s = 0; s < NUM_SLICES; ++s) { acc_o[s][0] *= alpha0; acc_o[s][1] *= alpha1; acc_o[s][2] *= alpha0; acc_o[s][3] *= alpha1; }
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t t_bytes = static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t v_reg[4]; ldmatrix_u32_x4_trans(v_reg, bV1_u32 + sV_off[u] + t_bytes);
                    mma_m16n8k16_bf16(acc_o[t * 4 + u], v_reg, p_reg);
                }
            }
        }
        if (has_next) { cp_async_wait<0>(); __syncthreads(); }
    }
WRITE_OUT_BN32:;
    const int qh_base = kv_head * NUM_Q_PER_KV;
    if (warp_id == 0 && quad_tid == 0) {
        int64_t ml_idx = ((int64_t)seq_idx * total_q_heads + (qh_base + group_id)) * num_splits + split_idx;
        segm_max[ml_idx] = row_m_h; segm_expsum[ml_idx] = row_l_h;
    }
    const int h_e0 = qh_base + 2 * quad_tid, h_e1 = h_e0 + 1;
    const int64_t out_off0 = (((int64_t)seq_idx * total_q_heads + h_e0) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    const int64_t out_off1 = (((int64_t)seq_idx * total_q_heads + h_e1) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) {
        int d0 = s * 16 + group_id, d1 = d0 + 8;
        segm_out[out_off0 + d0] = acc_o[s][0]; segm_out[out_off1 + d0] = acc_o[s][1];
        segm_out[out_off0 + d1] = acc_o[s][2]; segm_out[out_off1 + d1] = acc_o[s][3];
    }
}


template <bool BS32 = false, bool BS16 = false>
__global__ __launch_bounds__(128, 3)
void sm90_paged_decode_mma_bn32_nq16_kernel(
    float* __restrict__ segm_out,
    float* __restrict__ segm_max,
    float* __restrict__ segm_expsum,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    int sliding_window_val,
    int num_seqs,
    int num_kv_heads,
    int num_splits,
    int block_size,
    int64_t bt_stride0,
    int64_t q_stride0,
    int64_t q_stride1,
    int64_t k_stride0,
    int64_t k_stride1,
    int64_t k_stride2,
    int64_t v_stride0,
    int64_t v_stride1,
    int64_t v_stride2)
{
    constexpr int HEAD_SIZE = 512;
    constexpr int NUM_Q_PER_KV = 16;
    constexpr int D_PER_WARP = HEAD_SIZE / 4;
    constexpr int NUM_SLICES = D_PER_WARP / 16;
    constexpr int CHUNKS = 64;
    constexpr float LOG2E = 1.4426950408889634f;

    const int seq_idx = blockIdx.x, kv_head = blockIdx.y, split_idx = blockIdx.z;
    const int total_q_heads = num_kv_heads * NUM_Q_PER_KV;
    const uint32_t tid = threadIdx.x;
    const int warp_id = static_cast<int>(tid >> 5), lane = static_cast<int>(tid & 31u);
    const int group_id = lane >> 2, quad_tid = lane & 3;
    int q_tok = seq_idx;
    if (cu_seqlens_q != nullptr) { q_tok = cu_seqlens_q[seq_idx]; if (cu_seqlens_q[seq_idx + 1] <= q_tok) return; }
    const int seq_len = seqused_k[seq_idx];
    int k_min = (sliding_window_val > 0 && seq_len > sliding_window_val) ? (seq_len - sliding_window_val) : 0;
    const int tile_lo_g = k_min >> 4, tile_hi_g = (seq_len + 15) >> 4;
    const int n_active_tiles = (seq_len > 0 && tile_hi_g > tile_lo_g) ? (tile_hi_g - tile_lo_g) : 0;
    const int t_per = n_active_tiles / num_splits, t_rem = n_active_tiles - t_per * num_splits;
    const int my_tiles = t_per + (split_idx < t_rem ? 1 : 0);
    const int my_t0 = tile_lo_g + split_idx * t_per + min(split_idx, t_rem);
    if (my_tiles <= 0) return;

    extern __shared__ char smem_raw[];
    uint4* bK0 = reinterpret_cast<uint4*>(smem_raw);
    uint4* bK1 = bK0 + 16 * CHUNKS;
    uint4* bV0 = bK1 + 16 * CHUNKS;
    uint4* bV1 = bV0 + 16 * CHUNKS;
    float* s_logits_flat = reinterpret_cast<float*>(bV1 + 16 * CHUNKS);
    auto s_log = [&](int t2, int w, int k, int h) -> float& { return s_logits_flat[((t2 * 4 + w) * 16 + k) * 17 + h]; };

    const uint32_t bK0_u32 = smem_u32(bK0), bK1_u32 = smem_u32(bK1), bV0_u32 = smem_u32(bV0), bV1_u32 = smem_u32(bV1);
    const bool blk_pow2 = (BS16 || BS32) ? true : ((block_size & (block_size - 1)) == 0);
    const int blk_shift = BS32 ? 5 : (blk_pow2 ? (__ffs(block_size) - 1) : 0), blk_mask = BS32 ? 31 : (block_size - 1);
    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;
    const uint32_t r0 = tid >> 6, c0 = tid & 63u;
    uint32_t ld_dst[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) ld_dst[i] = static_cast<uint32_t>(swz512(static_cast<int>(r0) + i * 2, static_cast<int>(c0))) * 16u;
    const int64_t k_s1_bytes = k_stride1 * sizeof(__nv_bfloat16), k_s1_2_bytes = k_s1_bytes * 2;
    const int64_t v_s1_bytes = v_stride1 * sizeof(__nv_bfloat16), v_s1_2_bytes = v_s1_bytes * 2;
    const int64_t col_bytes = static_cast<int64_t>(c0) * (8 * sizeof(__nv_bfloat16));
    const __nv_bfloat16* k_head_base = k_cache + (int64_t)kv_head * k_stride2;
    const __nv_bfloat16* v_head_base = v_cache + (int64_t)kv_head * v_stride2;

    auto issue_load_16 = [&](uint32_t dst_u32, const __nv_bfloat16* cache_head_base,
                             int64_t s0, int64_t s1, int64_t s1_bytes, int64_t s1_2_bytes, int tile_idx) {
        int tok0 = tile_idx << 4;
        int page = BS16 ? tile_idx : (BS32 ? (tok0 >> 5) : (blk_pow2 ? (tok0 >> blk_shift) : (tok0 / block_size)));
        int off  = (BS16 || BS32) ? (BS16 ? 0 : ((tile_idx & 1) << 4)) : (blk_pow2 ? (tok0 & blk_mask) : (tok0 - page * block_size));
        int64_t pblk = (int64_t)bt_row[page];
        const char* ptr_b = BS16
            ? (reinterpret_cast<const char*>(cache_head_base + pblk * s0) + col_bytes)
            : (reinterpret_cast<const char*>(cache_head_base + pblk * s0 + (int64_t)off * s1) + col_bytes);
        const char* cur = ptr_b + static_cast<int64_t>(r0) * s1_bytes;
        int rem = seq_len - tok0, r_cur = static_cast<int>(r0);
        #pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            uint32_t h_off = dst_u32 + static_cast<uint32_t>(hi * 8192);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                bool ok = r_cur < rem;
                cp_async_16B_zfill_u32(h_off + ld_dst[i], (BS16 || BS32) ? cur : (ok ? cur : ptr_b), ok);
                cur += s1_2_bytes; r_cur += 2;
            }
        }
    };

    const int d_warp = warp_id * D_PER_WARP, d_warp_chunks = d_warp >> 3;
    const uint32_t* q0_u32 = reinterpret_cast<const uint32_t*>(
        q + (int64_t)q_tok * q_stride0 + (int64_t)(kv_head * NUM_Q_PER_KV + group_id) * q_stride1 + d_warp);
    const uint32_t* q1_u32 = reinterpret_cast<const uint32_t*>(
        q + (int64_t)q_tok * q_stride0 + (int64_t)(kv_head * NUM_Q_PER_KV + 8 + group_id) * q_stride1 + d_warp);
    uint32_t q_reg0[NUM_SLICES][2], q_reg1[NUM_SLICES][2];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) {
        q_reg0[s][0] = q0_u32[s * 8 + quad_tid]; q_reg0[s][1] = q0_u32[s * 8 + 4 + quad_tid];
        q_reg1[s][0] = q1_u32[s * 8 + quad_tid]; q_reg1[s][1] = q1_u32[s * 8 + 4 + quad_tid];
    }
    float acc_o0[NUM_SLICES][4], acc_o1[NUM_SLICES][4];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) {
        acc_o0[s][0] = acc_o0[s][1] = acc_o0[s][2] = acc_o0[s][3] = 0.f;
        acc_o1[s][0] = acc_o1[s][1] = acc_o1[s][2] = acc_o1[s][3] = 0.f;
    }
    float row_m0 = -INFINITY, row_l0 = 0.0f;
    float row_m1 = -INFINITY, row_l1 = 0.0f;

    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col_chunk = (lane >> 4);
    const int ldm_bk_row = (lane >> 4) * 8 + (lane & 7), ldm_bk_col_chunk = ((lane >> 3) & 1);
    const float scale_log2 = scale * LOG2E;
    uint32_t sK_off[4], sV_off[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        sK_off[u] = static_cast<uint32_t>(swz512(ldm_a_row, d_warp_chunks + u * 2 + ldm_a_col_chunk)) * 16u;
        sV_off[u] = static_cast<uint32_t>(swz512(ldm_bk_row, d_warp_chunks + u * 2 + ldm_bk_col_chunk)) * 16u;
    }

    issue_load_16(bK0_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, my_t0);
    if (1 < my_tiles) { issue_load_16(bK1_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, my_t0 + 1); }
    cp_async_commit(); cp_async_wait<0>(); __syncthreads();

    const int h0 = 2 * quad_tid, h1 = h0 + 1;
    const int k0 = 2 * quad_tid, k1 = k0 + 1, k2 = k0 + 8, k3 = k0 + 9;

    for (int j = 0; j < my_tiles; j += 2) {
        int t0 = my_t0 + j; bool has1 = (j + 1 < my_tiles); int t1 = t0 + 1;
        issue_load_16(bV0_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_2_bytes, t0);
        if (has1) issue_load_16(bV1_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_2_bytes, t1);
        cp_async_commit();

        float s_lo0[4] = {0.f, 0.f, 0.f, 0.f}, s_hi0[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            uint32_t t_bytes = static_cast<uint32_t>(t * 128);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                uint32_t k_reg[4]; ldmatrix_u32_x4(k_reg, bK0_u32 + sK_off[u] + t_bytes);
                mma_m16n8k16_bf16(s_lo0, k_reg, q_reg0[t * 4 + u]);
                mma_m16n8k16_bf16(s_hi0, k_reg, q_reg1[t * 4 + u]);
            }
        }
        s_log(0, warp_id, group_id, h0) = s_lo0[0];     s_log(0, warp_id, group_id, h1) = s_lo0[1];
        s_log(0, warp_id, group_id + 8, h0) = s_lo0[2]; s_log(0, warp_id, group_id + 8, h1) = s_lo0[3];
        s_log(0, warp_id, group_id, 8 + h0) = s_hi0[0]; s_log(0, warp_id, group_id, 8 + h1) = s_hi0[1];
        s_log(0, warp_id, group_id + 8, 8 + h0) = s_hi0[2]; s_log(0, warp_id, group_id + 8, 8 + h1) = s_hi0[3];

        if (has1) {
            float s_lo1[4] = {0.f, 0.f, 0.f, 0.f}, s_hi1[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t t_bytes = static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t k_reg[4]; ldmatrix_u32_x4(k_reg, bK1_u32 + sK_off[u] + t_bytes);
                    mma_m16n8k16_bf16(s_lo1, k_reg, q_reg0[t * 4 + u]);
                    mma_m16n8k16_bf16(s_hi1, k_reg, q_reg1[t * 4 + u]);
                }
            }
            s_log(1, warp_id, group_id, h0) = s_lo1[0];     s_log(1, warp_id, group_id, h1) = s_lo1[1];
            s_log(1, warp_id, group_id + 8, h0) = s_lo1[2]; s_log(1, warp_id, group_id + 8, h1) = s_lo1[3];
            s_log(1, warp_id, group_id, 8 + h0) = s_hi1[0]; s_log(1, warp_id, group_id, 8 + h1) = s_hi1[1];
            s_log(1, warp_id, group_id + 8, 8 + h0) = s_hi1[2]; s_log(1, warp_id, group_id + 8, 8 + h1) = s_hi1[3];
        }
        cp_async_wait<0>(); __syncthreads();

        bool has_next = (j + 2 < my_tiles);
        if (has_next) {
            issue_load_16(bK0_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, t0 + 2);
            if (j + 3 < my_tiles) { issue_load_16(bK1_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, t0 + 3); }
            cp_async_commit();
        }

        auto do_pv = [&](int slot, int tile_id, uint32_t bV_u32, bool rescale) {
            int tok_base = tile_id << 4;
            int g0 = tok_base + k0, g1 = tok_base + k1, g2 = tok_base + k2, g3 = tok_base + k3;
            bool ok0 = (g0 >= k_min && g0 < seq_len), ok1 = (g1 >= k_min && g1 < seq_len);
            bool ok2 = (g2 >= k_min && g2 < seq_len), ok3 = (g3 >= k_min && g3 < seq_len);

            // Low 8 heads (0..7)
            int h_lo = group_id;
            float l0 = ok0 ? ((s_log(slot,0,k0,h_lo)+s_log(slot,1,k0,h_lo)+s_log(slot,2,k0,h_lo)+s_log(slot,3,k0,h_lo)) * scale_log2) : -INFINITY;
            float l1 = ok1 ? ((s_log(slot,0,k1,h_lo)+s_log(slot,1,k1,h_lo)+s_log(slot,2,k1,h_lo)+s_log(slot,3,k1,h_lo)) * scale_log2) : -INFINITY;
            float l2 = ok2 ? ((s_log(slot,0,k2,h_lo)+s_log(slot,1,k2,h_lo)+s_log(slot,2,k2,h_lo)+s_log(slot,3,k2,h_lo)) * scale_log2) : -INFINITY;
            float l3 = ok3 ? ((s_log(slot,0,k3,h_lo)+s_log(slot,1,k3,h_lo)+s_log(slot,2,k3,h_lo)+s_log(slot,3,k3,h_lo)) * scale_log2) : -INFINITY;
            float mt0 = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
            mt0 = fmaxf(mt0, __shfl_xor_sync(0xffffffff, mt0, 1)); mt0 = fmaxf(mt0, __shfl_xor_sync(0xffffffff, mt0, 2));
            float nm0 = fmaxf(row_m0, mt0);
            float al0 = (row_m0 == -INFINITY || nm0 == -INFINITY) ? 0.f : exp2f(row_m0 - nm0);
            float sm0 = (nm0 == -INFINITY) ? 0.f : nm0;
            float p00 = (l0 == -INFINITY) ? 0.f : exp2f(l0 - sm0), p01 = (l1 == -INFINITY) ? 0.f : exp2f(l1 - sm0);
            float p02 = (l2 == -INFINITY) ? 0.f : exp2f(l2 - sm0), p03 = (l3 == -INFINITY) ? 0.f : exp2f(l3 - sm0);
            float lt0 = (p00 + p01) + (p02 + p03);
            lt0 += __shfl_xor_sync(0xffffffff, lt0, 1); lt0 += __shfl_xor_sync(0xffffffff, lt0, 2);
            row_l0 = row_l0 * al0 + lt0; row_m0 = nm0;
            uint32_t p_reg0[2] = {pack_bf16x2(p00, p01), pack_bf16x2(p02, p03)};
            if (rescale) {
                float a0 = __shfl_sync(0xffffffff, al0, h0 << 2), a1 = __shfl_sync(0xffffffff, al0, h1 << 2);
                #pragma unroll
                for (int s = 0; s < NUM_SLICES; ++s) { acc_o0[s][0] *= a0; acc_o0[s][1] *= a1; acc_o0[s][2] *= a0; acc_o0[s][3] *= a1; }
            }

            // High 8 heads (8..15)
            int h_hi = 8 + group_id;
            float r0_ = ok0 ? ((s_log(slot,0,k0,h_hi)+s_log(slot,1,k0,h_hi)+s_log(slot,2,k0,h_hi)+s_log(slot,3,k0,h_hi)) * scale_log2) : -INFINITY;
            float r1_ = ok1 ? ((s_log(slot,0,k1,h_hi)+s_log(slot,1,k1,h_hi)+s_log(slot,2,k1,h_hi)+s_log(slot,3,k1,h_hi)) * scale_log2) : -INFINITY;
            float r2_ = ok2 ? ((s_log(slot,0,k2,h_hi)+s_log(slot,1,k2,h_hi)+s_log(slot,2,k2,h_hi)+s_log(slot,3,k2,h_hi)) * scale_log2) : -INFINITY;
            float r3_ = ok3 ? ((s_log(slot,0,k3,h_hi)+s_log(slot,1,k3,h_hi)+s_log(slot,2,k3,h_hi)+s_log(slot,3,k3,h_hi)) * scale_log2) : -INFINITY;
            float mt1 = fmaxf(fmaxf(r0_, r1_), fmaxf(r2_, r3_));
            mt1 = fmaxf(mt1, __shfl_xor_sync(0xffffffff, mt1, 1)); mt1 = fmaxf(mt1, __shfl_xor_sync(0xffffffff, mt1, 2));
            float nm1 = fmaxf(row_m1, mt1);
            float al1 = (row_m1 == -INFINITY || nm1 == -INFINITY) ? 0.f : exp2f(row_m1 - nm1);
            float sm1 = (nm1 == -INFINITY) ? 0.f : nm1;
            float p10 = (r0_ == -INFINITY) ? 0.f : exp2f(r0_ - sm1), p11 = (r1_ == -INFINITY) ? 0.f : exp2f(r1_ - sm1);
            float p12 = (r2_ == -INFINITY) ? 0.f : exp2f(r2_ - sm1), p13 = (r3_ == -INFINITY) ? 0.f : exp2f(r3_ - sm1);
            float lt1 = (p10 + p11) + (p12 + p13);
            lt1 += __shfl_xor_sync(0xffffffff, lt1, 1); lt1 += __shfl_xor_sync(0xffffffff, lt1, 2);
            row_l1 = row_l1 * al1 + lt1; row_m1 = nm1;
            uint32_t p_reg1[2] = {pack_bf16x2(p10, p11), pack_bf16x2(p12, p13)};
            if (rescale) {
                float a0 = __shfl_sync(0xffffffff, al1, h0 << 2), a1 = __shfl_sync(0xffffffff, al1, h1 << 2);
                #pragma unroll
                for (int s = 0; s < NUM_SLICES; ++s) { acc_o1[s][0] *= a0; acc_o1[s][1] *= a1; acc_o1[s][2] *= a0; acc_o1[s][3] *= a1; }
            }

            // Shared V load across BOTH 8-head halves!
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t t_bytes = static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t v_reg[4]; ldmatrix_u32_x4_trans(v_reg, bV_u32 + sV_off[u] + t_bytes);
                    mma_m16n8k16_bf16(acc_o0[t * 4 + u], v_reg, p_reg0);
                    mma_m16n8k16_bf16(acc_o1[t * 4 + u], v_reg, p_reg1);
                }
            }
        };

        do_pv(0, t0, bV0_u32, j > 0);
        if (has1) do_pv(1, t1, bV1_u32, true);
        if (has_next) { cp_async_wait<0>(); __syncthreads(); }
    }

    const int qh_base = kv_head * NUM_Q_PER_KV;
    if (warp_id == 0 && quad_tid == 0) {
        int64_t ml0 = ((int64_t)seq_idx * total_q_heads + (qh_base + group_id)) * num_splits + split_idx;
        int64_t ml1 = ((int64_t)seq_idx * total_q_heads + (qh_base + 8 + group_id)) * num_splits + split_idx;
        segm_max[ml0] = row_m0; segm_expsum[ml0] = row_l0;
        segm_max[ml1] = row_m1; segm_expsum[ml1] = row_l1;
    }
    const int h_e0 = qh_base + 2 * quad_tid, h_e1 = h_e0 + 1;
    const int64_t o0_0 = (((int64_t)seq_idx * total_q_heads + h_e0) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    const int64_t o0_1 = (((int64_t)seq_idx * total_q_heads + h_e1) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    const int64_t o1_0 = (((int64_t)seq_idx * total_q_heads + (h_e0 + 8)) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    const int64_t o1_1 = (((int64_t)seq_idx * total_q_heads + (h_e1 + 8)) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) {
        int d0 = s * 16 + group_id, d1 = d0 + 8;
        segm_out[o0_0 + d0] = acc_o0[s][0]; segm_out[o0_1 + d0] = acc_o0[s][1];
        segm_out[o0_0 + d1] = acc_o0[s][2]; segm_out[o0_1 + d1] = acc_o0[s][3];
        segm_out[o1_0 + d0] = acc_o1[s][0]; segm_out[o1_1 + d0] = acc_o1[s][1];
        segm_out[o1_0 + d1] = acc_o1[s][2]; segm_out[o1_1 + d1] = acc_o1[s][3];
    }
}

template <int HEAD_SIZE, int D_CHUNK, int VEC>
__global__ __launch_bounds__(32)
void sm80_paged_decode_reduce_0bar_1w_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ segm_out,
    const float* __restrict__ segm_max,
    const float* __restrict__ segm_expsum,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    int sliding_window_val,
    int num_splits,
    int64_t o_stride0,
    int64_t o_stride1)
{
    static_assert(D_CHUNK == 32 * VEC, "D_CHUNK == 32*VEC");
    const int seq_idx = blockIdx.x, qh = blockIdx.y, total_q_heads = gridDim.y;
    const int lane = threadIdx.x;

    int out_tok = seq_idx;
    if (cu_seqlens_q != nullptr) {
        out_tok = cu_seqlens_q[seq_idx];
        if (cu_seqlens_q[seq_idx + 1] <= out_tok) return;
    }

    const int seq_len = seqused_k[seq_idx];
    int k_min = 0;
    if (sliding_window_val > 0 && seq_len > sliding_window_val) {
        k_min = seq_len - sliding_window_val;
    }
    const int tile_lo_g = k_min >> 4;
    const int tile_hi_g = (seq_len + 15) >> 4;
    const int n_active_tiles = (seq_len > 0 && tile_hi_g > tile_lo_g) ? (tile_hi_g - tile_lo_g) : 0;
    int eff_splits = num_splits;
    if constexpr (HEAD_SIZE == 512) {
        if (gridDim.x == 1 && num_splits == 64 && sliding_window_val == 0 && seq_len <= 2560) {
            eff_splits = 32;
        }
    }
    const int active_splits = min(eff_splits, n_active_tiles);

    const int d_out = blockIdx.z * D_CHUNK + lane * VEC;
    const int64_t final_idx = (int64_t)out_tok * o_stride0 + (int64_t)qh * o_stride1 + d_out;

    auto write_zero = [&]() {
        if constexpr (VEC == 4) {
            *reinterpret_cast<uint2*>(&out[final_idx]) = make_uint2(0u, 0u);
        } else if constexpr (VEC == 2) {
            *reinterpret_cast<uint32_t*>(&out[final_idx]) = 0u;
        } else {
            out[final_idx] = __float2bfloat16(0.0f);
        }
    };

    if (active_splits <= 0) { sm90_pdl_launch_dependents(); write_zero(); return; }
    const int64_t ml_base = ((int64_t)seq_idx * total_q_heads + qh) * num_splits;
    sm90_pdl_wait_3ptrs(segm_out, segm_max, segm_expsum);
    sm90_pdl_launch_dependents();

    float m0 = (lane < active_splits) ? segm_max[ml_base + lane] : -INFINITY;
    float m1 = ((lane + 32) < active_splits) ? segm_max[ml_base + lane + 32] : -INFINITY;
    float gm = fmaxf(m0, m1);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) gm = fmaxf(gm, __shfl_xor_sync(0xffffffff, gm, mask));
    if (gm == -INFINITY) { write_zero(); return; }

    float w0 = (lane < active_splits && m0 > -INFINITY) ? exp2f(m0 - gm) : 0.0f;
    float w1 = ((lane + 32) < active_splits && m1 > -INFINITY) ? exp2f(m1 - gm) : 0.0f;
    float l_loc = ((w0 > 0.f) ? (segm_expsum[ml_base + lane] * w0) : 0.0f)
                + ((w1 > 0.f) ? (segm_expsum[ml_base + lane + 32] * w1) : 0.0f);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) l_loc += __shfl_xor_sync(0xffffffff, l_loc, mask);
    if (l_loc <= 0.0f) { write_zero(); return; }
    const float inv_l = 1.0f / l_loc;

    const float* src_base = &segm_out[ml_base * HEAD_SIZE + d_out];
    float acc[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.0f;

    for (int s = 0; s < active_splits; ++s) {
        float w = (s < 32) ? __shfl_sync(0xffffffff, w0, s) : __shfl_sync(0xffffffff, w1, s - 32);
        if (w > 0.0f) {
            const float* src = src_base + (int64_t)s * HEAD_SIZE;
            if constexpr (VEC == 4) {
                float4 val = *reinterpret_cast<const float4*>(src);
                acc[0] += val.x * w; acc[1] += val.y * w; acc[2] += val.z * w; acc[3] += val.w * w;
            } else if constexpr (VEC == 2) {
                float2 val = *reinterpret_cast<const float2*>(src);
                acc[0] += val.x * w; acc[1] += val.y * w;
            } else {
                acc[0] += (*src) * w;
            }
        }
    }

    if constexpr (VEC == 4) {
        __nv_bfloat162 b01 = __floats2bfloat162_rn(acc[0] * inv_l, acc[1] * inv_l);
        __nv_bfloat162 b23 = __floats2bfloat162_rn(acc[2] * inv_l, acc[3] * inv_l);
        *reinterpret_cast<uint2*>(&out[final_idx]) = make_uint2(*reinterpret_cast<uint32_t*>(&b01), *reinterpret_cast<uint32_t*>(&b23));
    } else if constexpr (VEC == 2) {
        __nv_bfloat162 b01 = __floats2bfloat162_rn(acc[0] * inv_l, acc[1] * inv_l);
        *reinterpret_cast<uint32_t*>(&out[final_idx]) = *reinterpret_cast<uint32_t*>(&b01);
    } else {
        out[final_idx] = __float2bfloat16(acc[0] * inv_l);
    }
}

template <int HEAD_SIZE, int D_CHUNK, int VEC, int WARPS>
__global__ __launch_bounds__(WARPS * 32)
void sm80_paged_decode_reduce_1bar_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ segm_out,
    const float* __restrict__ segm_max,
    const float* __restrict__ segm_expsum,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    int sliding_window_val,
    int num_splits,
    int64_t o_stride0,
    int64_t o_stride1)
{
    static_assert(D_CHUNK == 32 * VEC, "D_CHUNK == 32*VEC");
    const int seq_idx = blockIdx.x, qh = blockIdx.y, total_q_heads = gridDim.y;
    const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;

    int out_tok = seq_idx;
    if (cu_seqlens_q != nullptr) {
        out_tok = cu_seqlens_q[seq_idx];
        if (cu_seqlens_q[seq_idx + 1] <= out_tok) {
            return;
        }
    }

    const int seq_len = seqused_k[seq_idx];
    int k_min = 0;
    if (sliding_window_val > 0 && seq_len > sliding_window_val) {
        k_min = seq_len - sliding_window_val;
    }
    const int tile_lo_g = k_min >> 4;
    const int tile_hi_g = (seq_len + 15) >> 4;
    const int n_active_tiles = (seq_len > 0 && tile_hi_g > tile_lo_g) ? (tile_hi_g - tile_lo_g) : 0;
    int eff_splits = num_splits;
    if constexpr (HEAD_SIZE == 512) {
        if (gridDim.x == 1 && num_splits == 64 && sliding_window_val == 0 && seq_len <= 2560) {
            eff_splits = 32;
        }
    }
    const int active_splits = min(eff_splits, n_active_tiles);

    const int d_out = blockIdx.z * D_CHUNK + lane * VEC;
    const int64_t final_idx = (int64_t)out_tok * o_stride0 + (int64_t)qh * o_stride1 + d_out;

    auto write_zero = [&]() {
        if (warp_id == 0) {
            if constexpr (VEC == 4) {
                *reinterpret_cast<uint2*>(&out[final_idx]) = make_uint2(0u, 0u);
            } else if constexpr (VEC == 2) {
                *reinterpret_cast<uint32_t*>(&out[final_idx]) = 0u;
            } else {
                #pragma unroll
                for (int v = 0; v < VEC; ++v) out[final_idx + v] = __float2bfloat16(0.0f);
            }
        }
    };

    if (active_splits <= 0) { sm90_pdl_launch_dependents(); write_zero(); return; }
    const int64_t ml_base = ((int64_t)seq_idx * total_q_heads + qh) * num_splits;
    sm90_pdl_wait_3ptrs(segm_out, segm_max, segm_expsum);
    sm90_pdl_launch_dependents();

    float m0 = (lane < active_splits) ? segm_max[ml_base + lane] : -INFINITY;
    float m1 = ((lane + 32) < active_splits) ? segm_max[ml_base + lane + 32] : -INFINITY;
    float gm = fmaxf(m0, m1);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) gm = fmaxf(gm, __shfl_xor_sync(0xffffffff, gm, mask));
    if (gm == -INFINITY) { write_zero(); return; }

    float w0 = (lane < active_splits && m0 > -INFINITY) ? exp2f(m0 - gm) : 0.0f;
    float w1 = ((lane + 32) < active_splits && m1 > -INFINITY) ? exp2f(m1 - gm) : 0.0f;
    float l_loc = ((w0 > 0.f) ? (segm_expsum[ml_base + lane] * w0) : 0.0f)
                + ((w1 > 0.f) ? (segm_expsum[ml_base + lane + 32] * w1) : 0.0f);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) l_loc += __shfl_xor_sync(0xffffffff, l_loc, mask);
    if (l_loc <= 0.0f) { write_zero(); return; }
    const float inv_l = 1.0f / l_loc;

    const int64_t out_seg_base = ml_base * HEAD_SIZE;
    float acc[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.0f;

    if (active_splits <= 32) {
        for (int s = warp_id; s < active_splits; s += WARPS) {
            float w = __shfl_sync(0xffffffff, w0, s);
            if (w > 0.0f) {
                const float* src = &segm_out[(out_seg_base + (int64_t)s * HEAD_SIZE) + d_out];
                if constexpr (VEC == 4) {
                    float4 val = *reinterpret_cast<const float4*>(src);
                    acc[0] += val.x * w; acc[1] += val.y * w; acc[2] += val.z * w; acc[3] += val.w * w;
                } else if constexpr (VEC == 2) {
                    float2 val = *reinterpret_cast<const float2*>(src);
                    acc[0] += val.x * w; acc[1] += val.y * w;
                } else {
                    #pragma unroll
                    for (int v = 0; v < VEC; ++v) acc[v] += src[v] * w;
                }
            }
        }
    } else {
        for (int s = warp_id; s < active_splits; s += WARPS) {
            float w = (s < 32) ? __shfl_sync(0xffffffff, w0, s) : __shfl_sync(0xffffffff, w1, s - 32);
            if (w > 0.0f) {
                const float* src = &segm_out[(out_seg_base + (int64_t)s * HEAD_SIZE) + d_out];
                if constexpr (VEC == 4) {
                    float4 val = *reinterpret_cast<const float4*>(src);
                    acc[0] += val.x * w; acc[1] += val.y * w; acc[2] += val.z * w; acc[3] += val.w * w;
                } else if constexpr (VEC == 2) {
                    float2 val = *reinterpret_cast<const float2*>(src);
                    acc[0] += val.x * w; acc[1] += val.y * w;
                } else {
                    #pragma unroll
                    for (int v = 0; v < VEC; ++v) acc[v] += src[v] * w;
                }
            }
        }
    }

    __shared__ float s_red[WARPS][32 * VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) s_red[warp_id][lane * VEC + v] = acc[v];
    __syncthreads();

    if (warp_id == 0) {
        if constexpr (VEC == 4) {
            float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                r0 += s_red[w][lane * 4 + 0]; r1 += s_red[w][lane * 4 + 1];
                r2 += s_red[w][lane * 4 + 2]; r3 += s_red[w][lane * 4 + 3];
            }
            __nv_bfloat162 b01 = __floats2bfloat162_rn(r0 * inv_l, r1 * inv_l);
            __nv_bfloat162 b23 = __floats2bfloat162_rn(r2 * inv_l, r3 * inv_l);
            *reinterpret_cast<uint2*>(&out[final_idx]) = make_uint2(*reinterpret_cast<uint32_t*>(&b01), *reinterpret_cast<uint32_t*>(&b23));
        } else if constexpr (VEC == 2) {
            float r0 = 0.f, r1 = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                r0 += s_red[w][lane * 2 + 0]; r1 += s_red[w][lane * 2 + 1];
            }
            __nv_bfloat162 b01 = __floats2bfloat162_rn(r0 * inv_l, r1 * inv_l);
            *reinterpret_cast<uint32_t*>(&out[final_idx]) = *reinterpret_cast<uint32_t*>(&b01);
        } else {
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                float rv = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) rv += s_red[w][lane * VEC + v];
                out[final_idx + v] = __float2bfloat16(rv * inv_l);
            }
        }
    }
}

template <int HEAD_SIZE, int D_CHUNK, int VEC, int WARPS = 4>
__global__ __launch_bounds__(WARPS * 32)
void sm80_paged_decode_reduce_vec_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ segm_out,
    const float* __restrict__ segm_max,
    const float* __restrict__ segm_expsum,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    int sliding_window_val,
    int num_splits,
    int64_t o_stride0,
    int64_t o_stride1)
{
    static_assert(D_CHUNK == 32 * VEC, "D_CHUNK must equal 32 * VEC");
    const int seq_idx = blockIdx.x;
    const int qh = blockIdx.y;
    const int total_q_heads = gridDim.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    int out_tok = seq_idx;
    if (cu_seqlens_q != nullptr) {
        out_tok = cu_seqlens_q[seq_idx];
        if (cu_seqlens_q[seq_idx + 1] <= out_tok) {
            return;
        }
    }

    const int seq_len = seqused_k[seq_idx];
    int k_min = 0;
    if (sliding_window_val > 0 && seq_len > sliding_window_val) {
        k_min = seq_len - sliding_window_val;
    }
    const int tile_lo_g = k_min >> 4;
    const int tile_hi_g = (seq_len + 15) >> 4;
    const int n_active_tiles = (seq_len > 0 && tile_hi_g > tile_lo_g) ? (tile_hi_g - tile_lo_g) : 0;
    int eff_splits = num_splits;
    if constexpr (HEAD_SIZE == 512) {
        if (gridDim.x == 1 && num_splits == 64 && sliding_window_val == 0 && seq_len <= 2560) {
            eff_splits = 32;
        }
    }
    const int active_splits = min(eff_splits, n_active_tiles);

    const int d_out = blockIdx.z * D_CHUNK + lane * VEC;
    const int64_t final_idx = (int64_t)out_tok * o_stride0 + (int64_t)qh * o_stride1 + d_out;

    auto write_zero = [&]() {
        if (warp_id == 0) {
            #pragma unroll
            for (int v = 0; v < VEC; ++v) out[final_idx + v] = __float2bfloat16(0.0f);
        }
    };

    if (active_splits <= 0) {
        sm90_pdl_launch_dependents();
        write_zero();
        return;
    }

    const int64_t ml_base = ((int64_t)seq_idx * total_q_heads + qh) * num_splits;
    sm90_pdl_wait_3ptrs(segm_out, segm_max, segm_expsum);
    sm90_pdl_launch_dependents();
    __shared__ float s_w[128];
    __shared__ float s_inv_l;
    __shared__ float s_global_m;

    if (warp_id == 0) {
        float m_loc = -INFINITY;
        float ms_reg = -INFINITY;
        if (active_splits <= 32) {
            if (lane < active_splits) {
                ms_reg = segm_max[ml_base + lane];
                m_loc = ms_reg;
            }
        } else {
            for (int s = lane; s < active_splits; s += 32) {
                m_loc = fmaxf(m_loc, segm_max[ml_base + s]);
            }
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            m_loc = fmaxf(m_loc, __shfl_xor_sync(0xffffffff, m_loc, mask));
        }
        const float global_m = m_loc;
        float l_loc = 0.0f;
        if (global_m != -INFINITY) {
            if (active_splits <= 32) {
                float w = (ms_reg > -INFINITY) ? exp2f(ms_reg - global_m) : 0.0f;
                if (lane < active_splits) s_w[lane] = w;
                if (w > 0.f) l_loc = segm_expsum[ml_base + lane] * w;
            } else {
                for (int s = lane; s < active_splits; s += 32) {
                    float ms = segm_max[ml_base + s];
                    float w = (ms > -INFINITY) ? exp2f(ms - global_m) : 0.0f;
                    if (s < 128) s_w[s] = w;
                    if (w > 0.f) l_loc += segm_expsum[ml_base + s] * w;
                }
            }
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                l_loc += __shfl_xor_sync(0xffffffff, l_loc, mask);
            }
        }
        if (lane == 0) {
            s_inv_l = (global_m != -INFINITY && l_loc > 0.0f) ? (1.0f / l_loc) : 0.0f;
            s_global_m = global_m;
        }
    }
    __syncthreads();

    const float inv_l = s_inv_l;
    if (inv_l <= 0.0f) {
        write_zero();
        return;
    }

    const float global_m = s_global_m;
    const int64_t out_seg_base = ml_base * HEAD_SIZE;
    float acc[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.0f;

    for (int s = warp_id; s < active_splits; s += WARPS) {
        float w = (s < 128) ? s_w[s] : exp2f(segm_max[ml_base + s] - global_m);
        if (w > 0.0f) {
            const float* src = &segm_out[(out_seg_base + (int64_t)s * HEAD_SIZE) + d_out];
            if constexpr (VEC == 4) {
                float4 val = *reinterpret_cast<const float4*>(src);
                acc[0] += val.x * w; acc[1] += val.y * w;
                acc[2] += val.z * w; acc[3] += val.w * w;
            } else if constexpr (VEC == 2) {
                float2 val = *reinterpret_cast<const float2*>(src);
                acc[0] += val.x * w; acc[1] += val.y * w;
            } else {
                acc[0] += src[0] * w;
            }
        }
    }

    __shared__ float s_red[WARPS][32 * VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) s_red[warp_id][lane * VEC + v] = acc[v];
    __syncthreads();

    if (warp_id == 0) {
        if constexpr (VEC == 4) {
            float r0 = 0.f, r1 = 0.f, r2 = 0.f, r3 = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                r0 += s_red[w][lane * 4 + 0]; r1 += s_red[w][lane * 4 + 1];
                r2 += s_red[w][lane * 4 + 2]; r3 += s_red[w][lane * 4 + 3];
            }
            __nv_bfloat162 b01 = __floats2bfloat162_rn(r0 * inv_l, r1 * inv_l);
            __nv_bfloat162 b23 = __floats2bfloat162_rn(r2 * inv_l, r3 * inv_l);
            uint2 pack = make_uint2(*reinterpret_cast<uint32_t*>(&b01), *reinterpret_cast<uint32_t*>(&b23));
            *reinterpret_cast<uint2*>(&out[final_idx]) = pack;
        } else if constexpr (VEC == 2) {
            float r0 = 0.f, r1 = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                r0 += s_red[w][lane * 2 + 0]; r1 += s_red[w][lane * 2 + 1];
            }
            __nv_bfloat162 b01 = __floats2bfloat162_rn(r0 * inv_l, r1 * inv_l);
            *reinterpret_cast<uint32_t*>(&out[final_idx]) = *reinterpret_cast<uint32_t*>(&b01);
        } else {
            float tot = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) tot += s_red[w][lane];
            out[final_idx] = __float2bfloat16(tot * inv_l);
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
struct alignas(8) bf16x4_a8 {
    __nv_bfloat16 v[4];
};
static_assert(sizeof(bf16x4_a8) == 8, "bf16x4_a8 must be 8 bytes");

template <bool HAS_SCALE>
__global__ __launch_bounds__(128)
void gemma4_softcap_bf16_vec8_128_kernel(bf16x8_a16* __restrict__ data, int64_t n_vec8, float cap, float scale) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n_vec8; i += stride) {
        bf16x8_a16 x = data[i];
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
__global__ __launch_bounds__(128)
void gemma4_softcap_bf16_vec4_128_kernel(bf16x4_a8* __restrict__ data, int64_t n_vec4, float cap, float scale) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n_vec4; i += stride) {
        bf16x4_a8 x = data[i];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
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

__device__ __forceinline__ uint16_t bf16_to_ord_u16(__nv_bfloat16 x) {
    uint16_t raw = *reinterpret_cast<const uint16_t*>(&x);
    if ((raw & 0x7FFFu) > 0x7F80u) return 0xFFFFu;
    if (raw == 0x8000u) raw = 0x0000u;
    return (raw & 0x8000u) ? static_cast<uint16_t>(~raw) : static_cast<uint16_t>(raw ^ 0x8000u);
}

__device__ __forceinline__ uint32_t ord_u16_to_ord_u32(uint16_t u16) {
    if (u16 == 0xFFFFu) return 0xFFFFFFFFu;
    return (static_cast<uint32_t>(u16) << 16) | ((u16 < 0x8000u) ? 0xFFFFu : 0u);
}

__device__ __forceinline__ uint64_t pack_comp_key_ord32(uint32_t uk, int32_t tok_id) {
    uint32_t inv_tok = 0x00FFFFFFu ^ static_cast<uint32_t>(tok_id & 0x00FFFFFF);
    return (static_cast<uint64_t>(uk) << 24) | static_cast<uint64_t>(inv_tok);
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

template <bool IS_BF16, int VECS_PER_THREAD = 4>
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
    constexpr int COLS_PER_BLK = VECS_PER_THREAD * 256 * 8;
    const int base_col = blk * COLS_PER_BLK;
    sm90_pdl_wait_1ptr(logits);
    if constexpr (IS_BF16) {
        const __nv_bfloat16* row_ptr = static_cast<const __nv_bfloat16*>(logits) + (int64_t)b * row_stride;
        if (((row_stride & 7) == 0) && base_col + COLS_PER_BLK <= vocab_size) {
            const bf16x8* vec_ptr = reinterpret_cast<const bf16x8*>(row_ptr + base_col);
            #pragma unroll
            for (int u = 0; u < VECS_PER_THREAD; ++u) {
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
            for (int col = base_col + tid; col < min(base_col + COLS_PER_BLK, vocab_size); col += 256) {
                float v = __bfloat162float(row_ptr[col]);
                if (v > best_v || (v == best_v && col < best_i)) { best_v = v; best_i = col; }
            }
        }
    } else {
        const float* row_ptr = static_cast<const float*>(logits) + (int64_t)b * row_stride;
        for (int col = base_col + tid; col < min(base_col + COLS_PER_BLK, vocab_size); col += 256) {
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
        if (lane == 0) { sm90_pdl_launch_dependents(); part_val[b * gridDim.y + blk] = best_v; part_idx[b * gridDim.y + blk] = best_i; }
    }
}

__global__ __launch_bounds__(32)
void gemma4_greedy_stage2_kernel(
    int64_t* __restrict__ sampled_out, const float* __restrict__ part_val,
    const int32_t* __restrict__ part_idx, int num_parts)
{
    const int b = blockIdx.x, lane = threadIdx.x;
    float best_v = -INFINITY; int32_t best_i = 0x7fffffff;
    sm90_pdl_wait_2ptrs(part_val, part_idx);
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

// Fast warp-shuffle radix picker (4 barriers vs 19); volatile s_res prevents cicc register forwarding
__device__ __forceinline__ void gemma4_hist_pick(
    int rem_k, uint32_t (&s_hist)[4][256], uint32_t* __restrict__ s_scan, volatile uint32_t (&s_res)[4],
    uint32_t& out_bidx, uint32_t& out_nrem, uint32_t& out_done)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t wid = tid >> 5u;
    const uint32_t lane = tid & 31u;
    uint32_t b_idx = 255u - tid;
    uint32_t c = s_hist[0][b_idx] + s_hist[1][b_idx] + s_hist[2][b_idx] + s_hist[3][b_idx];

    uint32_t x = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        uint32_t y = __shfl_up_sync(0xFFFFFFFFu, x, off);
        if (lane >= static_cast<uint32_t>(off)) x += y;
    }
    if (lane == 31u) s_scan[wid] = x;
    __syncthreads();

    if (wid == 0u) {
        uint32_t wsum = (lane < 8u) ? s_scan[lane] : 0u;
        uint32_t winc = wsum;
        #pragma unroll
        for (int off = 1; off < 8; off <<= 1) {
            uint32_t wy = __shfl_up_sync(0xFFFFFFFFu, winc, off);
            if (lane >= static_cast<uint32_t>(off)) winc += wy;
        }
        if (lane < 8u) s_scan[lane] = winc - wsum;
        if (lane == 7u) {
            uint32_t total = winc;
            s_res[0] = 0u;
            s_res[1] = static_cast<uint32_t>(rem_k);
            s_res[2] = total;
            s_res[3] = (total <= static_cast<uint32_t>(rem_k)) ? 2u : 0u;
        }
    }
    __syncthreads();

    uint32_t inc = x + s_scan[wid];
    uint32_t exc = inc - c;
    uint32_t init_done = s_res[3];
    if (init_done == 0u && exc < static_cast<uint32_t>(rem_k) && inc >= static_cast<uint32_t>(rem_k)) {
        uint32_t nrem = static_cast<uint32_t>(rem_k) - exc;
        s_res[0] = b_idx;
        s_res[1] = nrem;
        s_res[2] = c;
        s_res[3] = (c == nrem) ? 1u : 0u;
    }
    __syncthreads();
    out_bidx = s_res[0];
    out_nrem = s_res[1];
    out_done = s_res[3];
    __syncthreads();
}

template <int ITEMS>
__device__ __forceinline__ void gemma4_bf16_stage1_select(
    const uint16_t (&ord16)[ITEMS], int base_col, bool vec_layout, int vocab_size, int K,
    uint64_t* __restrict__ out_ckeys, uint32_t (&s_hist)[4][256], uint32_t* __restrict__ s_scan,
    volatile uint32_t (&s_res)[4], uint32_t& s_out_cnt)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t hbank = (tid >> 5u) & 3u;
    if (tid < static_cast<uint32_t>(K)) out_ckeys[tid] = 0ull;

    auto col_at = [&](int i) -> int {
        if (vec_layout) {
            int u = i >> 3;
            int j = i & 7;
            return base_col + (u * 256 + static_cast<int>(tid)) * 8 + j;
        }
        return base_col + i * 256 + static_cast<int>(tid);
    };

    #pragma unroll
    for (int w = 0; w < 4; ++w) s_hist[w][tid] = 0u;
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        if (col_at(i) < vocab_size) atomicAdd(&s_hist[hbank][ord16[i] >> 8], 1u);
    }
    __syncthreads();
    uint32_t hi8 = 0u, nrem_u = 0u, done = 0u;
    gemma4_hist_pick(K, s_hist, s_scan, s_res, hi8, nrem_u, done);
    int rem_k = static_cast<int>(nrem_u);
    uint16_t thresh16 = (done == 2u) ? 0u : static_cast<uint16_t>(hi8 << 8);

    if (done == 0u) {
        #pragma unroll
        for (int w = 0; w < 4; ++w) s_hist[w][tid] = 0u;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            if (col_at(i) < vocab_size && (static_cast<uint32_t>(ord16[i]) >> 8) == hi8) {
                atomicAdd(&s_hist[hbank][ord16[i] & 0xFFu], 1u);
            }
        }
        __syncthreads();
        uint32_t lo8 = 0u;
        gemma4_hist_pick(rem_k, s_hist, s_scan, s_res, lo8, nrem_u, done);
        rem_k = static_cast<int>(nrem_u);
        thresh16 = static_cast<uint16_t>((hi8 << 8) | lo8);
    }

    if (done != 0u) {
        if (tid == 0u) s_out_cnt = 0u;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            int c = col_at(i);
            if (c < vocab_size && ord16[i] >= thresh16) {
                uint32_t slot = atomicAdd(&s_out_cnt, 1u);
                if (slot < static_cast<uint32_t>(K)) {
                    out_ckeys[slot] = pack_comp_key_ord32(ord_u16_to_ord_u32(ord16[i]), c);
                }
            }
        }
    } else {
        if (tid == 0u) s_out_cnt = 0u;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            int c = col_at(i);
            if (c < vocab_size && ord16[i] > thresh16) {
                uint32_t slot = atomicAdd(&s_out_cnt, 1u);
                if (slot < static_cast<uint32_t>(K)) {
                    out_ckeys[slot] = pack_comp_key_ord32(ord_u16_to_ord_u32(ord16[i]), c);
                }
            }
        }
        __syncthreads();

        uint32_t t_prefix = 0u, t_mask = 0u;
        int t_rem = rem_k;
        bool t_active = true;
        for (int p = 2; p >= 0; --p) {
            if (t_active) {
                #pragma unroll
                for (int w = 0; w < 4; ++w) s_hist[w][tid] = 0u;
                __syncthreads();
                int sh = p * 8;
                #pragma unroll
                for (int i = 0; i < ITEMS; ++i) {
                    int c = col_at(i);
                    if (c < vocab_size && ord16[i] == thresh16) {
                        uint32_t itok = 0x00FFFFFFu ^ static_cast<uint32_t>(c & 0x00FFFFFF);
                        if ((itok & t_mask) == t_prefix) atomicAdd(&s_hist[hbank][(itok >> sh) & 0xFFu], 1u);
                    }
                }
                __syncthreads();
                uint32_t ch = 0u, tnrem = 0u, td = 0u;
                gemma4_hist_pick(t_rem, s_hist, s_scan, s_res, ch, tnrem, td);
                t_rem = static_cast<int>(tnrem);
                if (td != 2u) {
                    t_prefix |= (ch << sh);
                    t_mask |= (0xFFu << sh);
                }
                if (td != 0u) t_active = false;
            }
        }
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            int c = col_at(i);
            if (c < vocab_size && ord16[i] == thresh16) {
                uint32_t itok = 0x00FFFFFFu ^ static_cast<uint32_t>(c & 0x00FFFFFF);
                if (itok >= t_prefix) {
                    uint32_t slot = atomicAdd(&s_out_cnt, 1u);
                    if (slot < static_cast<uint32_t>(K)) {
                        out_ckeys[slot] = pack_comp_key_ord32(ord_u16_to_ord_u32(ord16[i]), c);
                    }
                }
            }
        }
    }
}

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void gemma4_u64_radix_select(
    const uint64_t (&ckey)[ITEMS_PER_THREAD], int K,
    uint64_t* __restrict__ out_ckeys, uint32_t (&s_hist)[4][256], uint32_t* __restrict__ s_scan,
    volatile uint32_t (&s_res)[4], uint32_t& s_out_cnt)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t hbank = (tid >> 5u) & 3u;
    if (tid < static_cast<uint32_t>(K)) out_ckeys[tid] = 0ull;

    uint64_t prefix = 0ull, mask_prefix = 0ull;
    int rem_k = K;
    bool active = true;

    for (int pass = 6; pass >= 0; --pass) {
        if (active) {
            #pragma unroll
            for (int w = 0; w < 4; ++w) s_hist[w][tid] = 0u;
            __syncthreads();
            const int shift = pass * 8;
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if (ckey[i] != 0ull && ((ckey[i] & mask_prefix) == prefix)) {
                    atomicAdd(&s_hist[hbank][static_cast<uint32_t>((ckey[i] >> shift) & 0xFFull)], 1u);
                }
            }
            __syncthreads();
            uint32_t ch32 = 0u, nrem_u = 0u, td = 0u;
            gemma4_hist_pick(rem_k, s_hist, s_scan, s_res, ch32, nrem_u, td);
            uint64_t ch = static_cast<uint64_t>(ch32);
            rem_k = static_cast<int>(nrem_u);
            if (td != 2u) {
                prefix |= (ch << shift);
                mask_prefix |= (0xFFull << shift);
            }
            if (td != 0u) active = false;
        }
    }

    if (tid == 0u) s_out_cnt = 0u;
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (ckey[i] >= prefix && ckey[i] != 0ull) {
            uint32_t slot = atomicAdd(&s_out_cnt, 1u);
            if (slot < static_cast<uint32_t>(K)) out_ckeys[slot] = ckey[i];
        }
    }
}

template <int BLOCK_COLS>
__global__ __launch_bounds__(256)
void gemma4_topk_stage1_bf16_auto_kernel(
    uint64_t* __restrict__ stage1_ckeys, const __nv_bfloat16* __restrict__ logits, int64_t row_stride,
    const int32_t* __restrict__ expanded_idx_mapping, const float* __restrict__ temperatures,
    const int32_t* __restrict__ top_k_gpu, int vocab_size, int max_k)
{
    constexpr int ITEMS = BLOCK_COLS / 256;
    __shared__ uint32_t s_hist[4][256];
    __shared__ uint32_t s_scan[256];
    __shared__ volatile uint32_t s_res[4];
    __shared__ uint32_t s_out_cnt;
    const int b = blockIdx.x, blk = blockIdx.y, tid = threadIdx.x;
    const int req_idx = expanded_idx_mapping[b];
    const float temp = (req_idx >= 0) ? temperatures[req_idx] : 1.0f;
    const bool scale_temp = (temp > 0.f && temp != 1.0f);
    int k_req = (req_idx >= 0) ? top_k_gpu[req_idx] : max_k;
    int K = min(max(k_req, 1), max_k);
    uint64_t* dst = stage1_ckeys + ((int64_t)b * gridDim.y + blk) * max_k;
    const int base_col = blk * BLOCK_COLS;
    const __nv_bfloat16* row_ptr = logits + (int64_t)b * row_stride;
    bool vec_layout = ((reinterpret_cast<uintptr_t>(row_ptr + base_col) & 15u) == 0u) && (base_col + BLOCK_COLS <= vocab_size);

    if (!scale_temp) {
        uint16_t ord16[ITEMS];
        if (vec_layout) {
            const bf16x8_a16* vec_ptr = reinterpret_cast<const bf16x8_a16*>(row_ptr + base_col);
            constexpr int VEC_ITERS = ITEMS / 8;
            #pragma unroll
            for (int u = 0; u < VEC_ITERS; ++u) {
                bf16x8_a16 pack = vec_ptr[u * 256 + tid];
                #pragma unroll
                for (int j = 0; j < 8; ++j) ord16[u * 8 + j] = bf16_to_ord_u16(pack.v[j]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ITEMS; ++i) {
                int col = base_col + i * 256 + tid;
                ord16[i] = (col < vocab_size) ? bf16_to_ord_u16(row_ptr[col]) : 0u;
            }
        }
        gemma4_bf16_stage1_select<ITEMS>(ord16, base_col, vec_layout, vocab_size, K, dst, s_hist, s_scan, s_res, s_out_cnt);
    } else {
        uint64_t ckey[ITEMS];
        if (vec_layout) {
            const bf16x8_a16* vec_ptr = reinterpret_cast<const bf16x8_a16*>(row_ptr + base_col);
            constexpr int VEC_ITERS = ITEMS / 8;
            #pragma unroll
            for (int u = 0; u < VEC_ITERS; ++u) {
                int vec_idx = u * 256 + tid;
                bf16x8_a16 pack = vec_ptr[vec_idx];
                int tok0 = base_col + vec_idx * 8;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    ckey[u * 8 + j] = pack_comp_key(__bfloat162float(pack.v[j]) / temp, tok0 + j, true);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ITEMS; ++i) {
                int col = base_col + i * 256 + tid;
                ckey[i] = (col < vocab_size) ? pack_comp_key(__bfloat162float(row_ptr[col]) / temp, col, true) : 0ull;
            }
        }
        gemma4_u64_radix_select<ITEMS>(ckey, K, dst, s_hist, s_scan, s_res, s_out_cnt);
    }
}

template <bool IS_BF16, int BLOCK_COLS>
__global__ __launch_bounds__(256, 2)
void gemma4_topk_stage1_kernel(
    uint64_t* __restrict__ stage1_ckeys, const void* __restrict__ logits, int64_t row_stride,
    const int32_t* __restrict__ expanded_idx_mapping, const float* __restrict__ temperatures,
    const int32_t* __restrict__ top_k_gpu, int vocab_size, int max_k)
{
    constexpr int ITEMS = BLOCK_COLS / 256;
    __shared__ uint32_t s_hist[4][256];
    __shared__ uint32_t s_scan[256];
    __shared__ volatile uint32_t s_res[4];
    __shared__ uint32_t s_out_cnt;
    const int b = blockIdx.x, blk = blockIdx.y, tid = threadIdx.x;
    const int req_idx = expanded_idx_mapping[b];
    const float temp = (req_idx >= 0) ? temperatures[req_idx] : 1.0f;
    const bool scale_temp = (temp > 0.f && temp != 1.0f);
    int k_req = (req_idx >= 0) ? top_k_gpu[req_idx] : max_k;
    int K = min(max(k_req, 1), max_k);
    uint64_t* dst = stage1_ckeys + ((int64_t)b * gridDim.y + blk) * max_k;
    const int base_col = blk * BLOCK_COLS;

    uint64_t ckey[ITEMS];
    if constexpr (IS_BF16) {
        const __nv_bfloat16* row_ptr = static_cast<const __nv_bfloat16*>(logits) + (int64_t)b * row_stride;
        if (((reinterpret_cast<uintptr_t>(row_ptr + base_col) & 15u) == 0u) && base_col + BLOCK_COLS <= vocab_size) {
            const bf16x8_a16* vec_ptr = reinterpret_cast<const bf16x8_a16*>(row_ptr + base_col);
            constexpr int VEC_ITERS = ITEMS / 8;
            #pragma unroll
            for (int u = 0; u < VEC_ITERS; ++u) {
                int vec_idx = u * 256 + tid;
                bf16x8_a16 pack = vec_ptr[vec_idx];
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
            for (int i = 0; i < ITEMS; ++i) {
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
        for (int i = 0; i < ITEMS; ++i) {
            int col = base_col + i * 256 + tid;
            if (col < vocab_size) {
                float v = row_ptr[col];
                if (scale_temp) v = v / temp;
                ckey[i] = pack_comp_key(v, col, true);
            } else ckey[i] = 0ull;
        }
    }
    gemma4_u64_radix_select<ITEMS>(ckey, K, dst, s_hist, s_scan, s_res, s_out_cnt);
}

__device__ __forceinline__ uint32_t gemma4_shfl_xor_key(uint32_t val, int mask) {
    return __shfl_xor_sync(0xffffffffu, val, mask);
}
__device__ __forceinline__ uint64_t gemma4_shfl_xor_key(uint64_t val, int mask) {
    uint32_t lo = static_cast<uint32_t>(val);
    uint32_t hi = static_cast<uint32_t>(val >> 32);
    lo = __shfl_xor_sync(0xffffffffu, lo, mask);
    hi = __shfl_xor_sync(0xffffffffu, hi, mask);
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

template <typename T>
__device__ __forceinline__ T gemma4_bitonic_sort_le256(T val, int sort_n, T (*s_buf)[256]) {
    const int tid = threadIdx.x;
    if (tid < sort_n) {
        #pragma unroll
        for (int size = 2; size <= 32; size <<= 1) {
            bool dir_desc = ((tid & size) == 0);
            #pragma unroll
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                T o = gemma4_shfl_xor_key(val, stride);
                bool want_min = (((tid & stride) != 0) == dir_desc);
                val = want_min ? ((o < val) ? o : val) : ((o > val) ? o : val);
            }
        }
    }
    int sbuf = 0;
    for (int size = 64; size <= sort_n; size <<= 1) {
        bool dir_desc = ((tid & size) == 0);
        for (int stride = size >> 1; stride >= 32; stride >>= 1) {
            if (tid < sort_n) s_buf[sbuf][tid] = val;
            __syncthreads();
            if (tid < sort_n) {
                T o = s_buf[sbuf][tid ^ stride];
                bool want_min = (((tid & stride) != 0) == dir_desc);
                val = want_min ? ((o < val) ? o : val) : ((o > val) ? o : val);
            }
            sbuf ^= 1;
        }
        if (tid < sort_n) {
            #pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                T o = gemma4_shfl_xor_key(val, stride);
                bool want_min = (((tid & stride) != 0) == dir_desc);
                val = want_min ? ((o < val) ? o : val) : ((o > val) ? o : val);
            }
        }
    }
    return val;
}

template <typename T, int N_ITEMS>
__device__ __forceinline__ void gemma4_warp_compact_ge(
    const T (&vals)[N_ITEMS], T thresh, uint32_t& s_cnt, T* __restrict__ s_out, int max_cands, int lane)
{
    int my_cnt = 0;
    #pragma unroll
    for (int i = 0; i < N_ITEMS; ++i) {
        if (vals[i] >= thresh) ++my_cnt;
    }
    int exc_warp = my_cnt;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, exc_warp, off);
        if (lane >= off) exc_warp += y;
    }
    int wtot = __shfl_sync(0xffffffffu, exc_warp, 31);
    if (wtot > 0) {
        int wbase = 0;
        if (lane == 31) wbase = static_cast<int>(atomicAdd(&s_cnt, static_cast<uint32_t>(wtot)));
        wbase = __shfl_sync(0xffffffffu, wbase, 31);
        int wpos = wbase + (exc_warp - my_cnt);
        #pragma unroll
        for (int i = 0; i < N_ITEMS; ++i) {
            T v = vals[i];
            if (v >= thresh && wpos < max_cands) s_out[wpos++] = v;
        }
    }
}

template <typename T>
__device__ __forceinline__ void gemma4_bitonic_sort_inplace_large(T* s_buf, int sort_n) {
    const int tid = threadIdx.x;
    if ((sort_n & 255) != 0) {
        for (int size = 2; size <= sort_n; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int p = tid; p < sort_n; p += 256) {
                    int q = p ^ stride;
                    if (p < q) {
                        T a = s_buf[p];
                        T b = s_buf[q];
                        bool dir_desc = ((p & size) == 0);
                        if (dir_desc ? (a < b) : (a > b)) {
                            s_buf[p] = b;
                            s_buf[q] = a;
                        }
                    }
                }
                __syncthreads();
            }
        }
        return;
    }
    for (int base = 0; base < sort_n; base += 256) {
        int p = base + tid;
        T v = s_buf[p];
        #pragma unroll
        for (int size = 2; size <= 32; size <<= 1) {
            bool dir_desc = ((p & size) == 0);
            #pragma unroll
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                T o = gemma4_shfl_xor_key(v, stride);
                bool want_min = (((p & stride) != 0) == dir_desc);
                v = want_min ? ((o < v) ? o : v) : ((o > v) ? o : v);
            }
        }
        s_buf[p] = v;
    }
    __syncthreads();
    for (int size = 64; size <= sort_n; size <<= 1) {
        for (int stride = size >> 1; stride >= 32; stride >>= 1) {
            for (int p = tid; p < sort_n; p += 256) {
                int q = p ^ stride;
                if (p < q) {
                    T a = s_buf[p];
                    T b = s_buf[q];
                    bool dir_desc = ((p & size) == 0);
                    if (dir_desc ? (a < b) : (a > b)) {
                        s_buf[p] = b;
                        s_buf[q] = a;
                    }
                }
            }
            __syncthreads();
        }
        for (int base = 0; base < sort_n; base += 256) {
            int p = base + tid;
            T v = s_buf[p];
            bool dir_desc = ((p & size) == 0);
            #pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                T o = gemma4_shfl_xor_key(v, stride);
                bool want_min = (((p & stride) != 0) == dir_desc);
                v = want_min ? ((o < v) ? o : v) : ((o > v) ? o : v);
            }
            s_buf[p] = v;
        }
        __syncthreads();
    }
}

template <int BLOCK_COLS, int COL_BITS>
__global__ __launch_bounds__(256)
void gemma4_topk_stage1_bf16_prune_kernel(
    uint64_t* __restrict__ stage1_ckeys, const __nv_bfloat16* __restrict__ logits, int64_t row_stride,
    const int32_t* __restrict__ top_k_gpu, const int32_t* __restrict__ expanded_idx_mapping,
    int vocab_size, int max_k)
{
    constexpr int ITEMS = BLOCK_COLS / 256;
    constexpr int MAX_CANDS = (ITEMS <= 8) ? 512 : ((ITEMS <= 16) ? 1024 : 2048);
    constexpr int R2_ITEMS = MAX_CANDS / 256;
    constexpr uint32_t COL_MASK = (1u << COL_BITS) - 1u;

    __shared__ uint32_t s_cands[MAX_CANDS];
    __shared__ uint32_t s_cnt;
    __shared__ uint32_t s_thresh;

    const int b = blockIdx.x, blk = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
    const int req_idx = expanded_idx_mapping[b];
    int k_req = (req_idx >= 0) ? top_k_gpu[req_idx] : max_k;
    const int K = min(max(k_req, 1), max_k);
    uint64_t* dst = stage1_ckeys + ((int64_t)b * gridDim.y + blk) * max_k;
    const int base_col = blk * BLOCK_COLS;
    sm90_pdl_wait_1ptr(logits);
    const __nv_bfloat16* row_ptr = logits + (int64_t)b * row_stride;
    const bool vec_layout = ((reinterpret_cast<uintptr_t>(row_ptr + base_col) & 15u) == 0u) && (base_col + BLOCK_COLS <= vocab_size);

    uint32_t key_reg[ITEMS];
    uint32_t tmax = 0u;
    if (vec_layout) {
        const bf16x8_a16* vec_ptr = reinterpret_cast<const bf16x8_a16*>(row_ptr + base_col);
        constexpr int VEC_ITERS = ITEMS / 8;
        #pragma unroll
        for (int u = 0; u < VEC_ITERS; ++u) {
            int vec_idx = u * 256 + tid;
            bf16x8_a16 pack = vec_ptr[vec_idx];
            uint32_t rel0 = static_cast<uint32_t>(vec_idx * 8);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t o16 = static_cast<uint32_t>(bf16_to_ord_u16(pack.v[j]));
                uint32_t k = (o16 << COL_BITS) | (COL_MASK ^ (rel0 + static_cast<uint32_t>(j)));
                key_reg[u * 8 + j] = k;
                if (k > tmax) tmax = k;
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            int rel = i * 256 + tid;
            int col = base_col + rel;
            uint32_t k = 0u;
            if (col < vocab_size) {
                uint32_t o16 = static_cast<uint32_t>(bf16_to_ord_u16(row_ptr[col]));
                k = (o16 << COL_BITS) | (COL_MASK ^ static_cast<uint32_t>(rel));
            }
            key_reg[i] = k;
            if (k > tmax) tmax = k;
        }
    }

    uint32_t (*s_sort_alias)[256] = reinterpret_cast<uint32_t (*)[256]>(s_cands);
    uint32_t sorted_tmax = gemma4_bitonic_sort_le256<uint32_t>(tmax, 256, s_sort_alias);
    if (tid == K - 1) s_thresh = (sorted_tmax > 0u) ? sorted_tmax : 1u;
    if (tid == 0) s_cnt = 0u;
    __syncthreads();
    const uint32_t thresh = s_thresh;

    gemma4_warp_compact_ge<uint32_t, ITEMS>(key_reg, thresh, s_cnt, s_cands, MAX_CANDS, lane);
    __syncthreads();

    int G = static_cast<int>(s_cnt);
    if (G > MAX_CANDS) G = MAX_CANDS;

    auto emit_u32 = [&](uint32_t k32) -> uint64_t {
        if (k32 == 0u) return 0ull;
        uint16_t ord16 = static_cast<uint16_t>(k32 >> COL_BITS);
        int32_t rel_col = static_cast<int32_t>((k32 & COL_MASK) ^ COL_MASK);
        return pack_comp_key_ord32(ord_u16_to_ord_u32(ord16), base_col + rel_col);
    };

    if (G <= K) {
        if (tid == 0) sm90_pdl_launch_dependents();
        if (tid < K) dst[tid] = (tid < G) ? emit_u32(s_cands[tid]) : 0ull;
        return;
    }

    if (G > 256) {
        uint32_t r2[R2_ITEMS];
        uint32_t tmax2 = 0u;
        #pragma unroll
        for (int i = 0; i < R2_ITEMS; ++i) {
            int p = i * 256 + tid;
            uint32_t x = (p < G) ? s_cands[p] : 0u;
            r2[i] = x;
            if (x > tmax2) tmax2 = x;
        }
        __syncthreads();
        uint32_t st2 = gemma4_bitonic_sort_le256<uint32_t>(tmax2, 256, s_sort_alias);
        if (tid == K - 1) s_thresh = (st2 > 0u) ? st2 : 1u;
        if (tid == 0) s_cnt = 0u;
        __syncthreads();
        uint32_t th2 = s_thresh;
        gemma4_warp_compact_ge<uint32_t, R2_ITEMS>(r2, th2, s_cnt, s_cands, MAX_CANDS, lane);
        __syncthreads();
        G = static_cast<int>(s_cnt);
        if (G > MAX_CANDS) G = MAX_CANDS;
        if (G <= K) {
            if (tid == 0) sm90_pdl_launch_dependents();
        if (tid < K) dst[tid] = (tid < G) ? emit_u32(s_cands[tid]) : 0ull;
            return;
        }
    }

    if (G <= 256) {
        int sort_n = (G <= 128) ? 128 : 256;
        uint32_t v = (tid < G) ? s_cands[tid] : 0u;
        __syncthreads();
        v = gemma4_bitonic_sort_le256<uint32_t>(v, sort_n, s_sort_alias);
        if (tid == 0) sm90_pdl_launch_dependents();
        if (tid < K) dst[tid] = emit_u32(v);
    } else {
        int sort_n = (G <= 512) ? 512 : 1024;
        for (int p = G + tid; p < sort_n; p += 256) s_cands[p] = 0u;
        __syncthreads();
        gemma4_bitonic_sort_inplace_large<uint32_t>(s_cands, sort_n);
        if (tid == 0) sm90_pdl_launch_dependents();
        if (tid < K) dst[tid] = emit_u32(s_cands[tid]);
    }
}

template <int STAGE2_ITEMS>
__global__ __launch_bounds__(256)
void gemma4_topk_topp_gumbel_stage2_prune_kernel(
    int64_t* __restrict__ sampled_out, const uint64_t* __restrict__ stage1_ckeys,
    const int32_t* __restrict__ expanded_idx_mapping, const float* __restrict__ temperatures,
    const int32_t* __restrict__ top_k_gpu, const float* __restrict__ top_p_gpu,
    const int64_t* __restrict__ seeds_gpu, const int64_t* __restrict__ pos_gpu,
    int num_blocks, int max_k)
{
    constexpr int MAX_CANDS = (STAGE2_ITEMS <= 8) ? 512 : 1024;
    constexpr int R2_ITEMS = (STAGE2_ITEMS <= 8) ? 2 : 4;
    __shared__ uint64_t s_top[MAX_CANDS];
    __shared__ uint32_t s_cnt;
    __shared__ uint64_t s_thresh;

    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int req_idx = expanded_idx_mapping[b];
    int k_req = (req_idx >= 0) ? top_k_gpu[req_idx] : max_k;
    const int K = min(max(k_req, 1), max_k);
    const float p_thresh = (top_p_gpu != nullptr && req_idx >= 0) ? top_p_gpu[req_idx] : 1.0f;
    const float temp = (req_idx >= 0) ? temperatures[req_idx] : 1.0f;
    const bool do_scale = (temp > 0.f && temp != 1.0f);
    const int64_t seed = (req_idx >= 0) ? seeds_gpu[req_idx] : 0ll;
    const int64_t pos = pos_gpu[b];

    const int total_cand = num_blocks * K;
    sm90_pdl_wait_1ptr(stage1_ckeys);
    const uint64_t* row_cands = stage1_ckeys + (int64_t)b * num_blocks * max_k;
    uint64_t ckey[STAGE2_ITEMS];
    uint64_t tmax = 0ull;
    if (K == max_k) {
        #pragma unroll
        for (int i = 0; i < STAGE2_ITEMS; ++i) {
            int idx = i * 256 + tid;
            uint64_t v = (idx < total_cand) ? row_cands[idx] : 0ull;
            ckey[i] = v;
            if (v > tmax) tmax = v;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < STAGE2_ITEMS; ++i) {
            int idx = i * 256 + tid;
            uint64_t v = 0ull;
            if (idx < total_cand) {
                int blk = idx / K;
                v = row_cands[blk * max_k + (idx - blk * K)];
            }
            ckey[i] = v;
            if (v > tmax) tmax = v;
        }
    }

    uint64_t (*s_sort_alias)[256] = reinterpret_cast<uint64_t (*)[256]>(s_top);
    uint64_t sorted_tmax = gemma4_bitonic_sort_le256<uint64_t>(tmax, 256, s_sort_alias);
    if (tid == K - 1) s_thresh = (sorted_tmax > 0ull) ? sorted_tmax : 1ull;
    if (tid == 0) s_cnt = 0u;
    __syncthreads();
    const uint64_t thresh = s_thresh;

    gemma4_warp_compact_ge<uint64_t, STAGE2_ITEMS>(ckey, thresh, s_cnt, s_top, MAX_CANDS, lane);
    __syncthreads();

    int G = static_cast<int>(s_cnt);
    if (G > MAX_CANDS) G = MAX_CANDS;

    if (G > 256) {
        uint64_t r2[R2_ITEMS];
        uint64_t tmax2 = 0ull;
        #pragma unroll
        for (int i = 0; i < R2_ITEMS; ++i) {
            int p = i * 256 + tid;
            uint64_t x = (p < G) ? s_top[p] : 0ull;
            r2[i] = x;
            if (x > tmax2) tmax2 = x;
        }
        __syncthreads();
        uint64_t st2 = gemma4_bitonic_sort_le256<uint64_t>(tmax2, 256, s_sort_alias);
        if (tid == K - 1) s_thresh = (st2 > 0ull) ? st2 : 1ull;
        if (tid == 0) s_cnt = 0u;
        __syncthreads();
        uint64_t th2 = s_thresh;
        gemma4_warp_compact_ge<uint64_t, R2_ITEMS>(r2, th2, s_cnt, s_top, 256, lane);
        __syncthreads();
        G = static_cast<int>(s_cnt);
        if (G > 256) G = 256;
    }

    int sort_n = (G <= 64) ? 64 : ((G <= 128) ? 128 : 256);
    uint64_t val = (tid < G) ? s_top[tid] : 0ull;
    __syncthreads();
    val = gemma4_bitonic_sort_le256<uint64_t>(val, sort_n, s_sort_alias);
    if (tid < 64) s_top[tid] = val;
    __syncthreads();

    float val_i = -INFINITY; int32_t tok_i = 0x7fffffff;
    if (tid < K && val != 0ull) {
        unpack_comp_key(val, val_i, tok_i);
        if (do_scale) val_i = val_i / temp;
    }

    bool keep = (tid == 0) || (tid < K);
    if (p_thresh < 1.0f && K > 1) {
        float max_v = -INFINITY;
        if (s_top[0] != 0ull) {
            int32_t dt; unpack_comp_key(s_top[0], max_v, dt);
            if (do_scale) max_v = max_v / temp;
        }
        float p_val = (tid < K && val_i > -INFINITY) ? expf(val_i - max_v) : 0.f;
        __shared__ float s_wsum[2];
        __shared__ float s_excl[64];
        float p_warp = p_val;
        if (tid < 64) {
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                float up = __shfl_up_sync(0xffffffffu, p_warp, off);
                if (lane >= off) p_warp += up;
            }
            if (lane == 31 && wid == 0) s_wsum[0] = p_warp;
        }
        __syncthreads();
        float tot = p_warp + ((wid == 1) ? s_wsum[0] : 0.f);
        if (tid < 64) s_excl[tid] = tot;
        __syncthreads();
        float z_topk = s_excl[K - 1];
        float excl_cum = (tid > 0 && tid < K) ? s_excl[tid - 1] : 0.f;
        keep = (tid == 0) || (tid < K && (excl_cum < p_thresh * z_topk));
    }

    float noisy_score = (keep && val_i > -INFINITY)
        ? (val_i + vllm_gumbel_noise_fp32(seed, pos, static_cast<uint32_t>(tok_i)))
        : -INFINITY;

    __shared__ float s_best_v[2];
    __shared__ int32_t s_best_t[2];
    float bv = noisy_score; int32_t bt = tok_i;
    const int active_warps = (K <= 32) ? 1 : 2;
    if (wid < active_warps) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, mask);
            int32_t ot = __shfl_xor_sync(0xffffffffu, bt, mask);
            if (ov > bv || (ov == bv && ot < bt)) { bv = ov; bt = ot; }
        }
        if (lane == 0) { s_best_v[wid] = bv; s_best_t[wid] = bt; }
    }
    if (active_warps > 1) {
        __syncthreads();
        if (wid == 0) {
            bv = (lane < 2) ? s_best_v[lane] : -INFINITY;
            bt = (lane < 2) ? s_best_t[lane] : 0x7fffffff;
            float ov = __shfl_xor_sync(0xffffffffu, bv, 1);
            int32_t ot = __shfl_xor_sync(0xffffffffu, bt, 1);
            if (ov > bv || (ov == bv && ot < bt)) { bv = ov; bt = ot; }
            if (lane == 0) sampled_out[b] = static_cast<int64_t>(bt);
        }
    } else if (tid == 0) {
        sampled_out[b] = static_cast<int64_t>(bt);
    }
}

template <int STAGE2_ITEMS>
__global__ __launch_bounds__(256, 2)
void gemma4_topk_topp_gumbel_stage2_kernel(
    int64_t* __restrict__ sampled_out, const uint64_t* __restrict__ stage1_ckeys,
    const int32_t* __restrict__ expanded_idx_mapping, const int32_t* __restrict__ top_k_gpu,
    const float* __restrict__ top_p_gpu, const int64_t* __restrict__ seeds_gpu,
    const int64_t* __restrict__ pos_gpu, int num_blocks, int max_k)
{
    __shared__ uint32_t s_hist[4][256];
    __shared__ uint32_t s_scan[256];
    __shared__ volatile uint32_t s_res[4];
    __shared__ uint32_t s_out_cnt;
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
    uint64_t ckey[STAGE2_ITEMS];
    #pragma unroll
    for (int i = 0; i < STAGE2_ITEMS; ++i) {
        int idx = i * 256 + tid;
        if (idx < total_cand) {
            int blk = idx / K;
            ckey[i] = row_cands[blk * max_k + (idx - blk * K)];
        } else ckey[i] = 0ull;
    }

    s_top[tid] = 0ull;
    __syncthreads();
    gemma4_u64_radix_select<STAGE2_ITEMS>(ckey, K, s_top, s_hist, s_scan, s_res, s_out_cnt);
    __syncthreads();

    int sort_n = (K <= 64) ? 64 : 256;
    for (int size = 2; size <= sort_n; size <<= 1) {
        bool dir_desc = ((tid & size) == 0);
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int ixj = tid ^ stride;
            if (tid < sort_n && ixj > tid) {
                uint64_t a = s_top[tid], c = s_top[ixj];
                if ((a < c) == dir_desc) { s_top[tid] = c; s_top[ixj] = a; }
            }
            __syncthreads();
        }
    }

    float val_i = -INFINITY; int32_t tok_i = 0x7fffffff;
    if (tid < K && s_top[tid] != 0ull) unpack_comp_key(s_top[tid], val_i, tok_i);
    __shared__ volatile float s_max_v;
    if (tid == 0) s_max_v = val_i;
    __syncthreads();
    float max_v = s_max_v;

    s_prob[tid] = (tid < K && val_i > -INFINITY) ? expf(val_i - max_v) : 0.f;
    __syncthreads();
    for (int offset = 1; offset < sort_n; offset <<= 1) {
        float add = (tid < sort_n && tid >= offset) ? s_prob[tid - offset] : 0.f;
        __syncthreads();
        if (tid < sort_n) s_prob[tid] += add;
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
        float ov = __shfl_xor_sync(0xffffffffu, bv, mask);
        int32_t ot = __shfl_xor_sync(0xffffffffu, bt, mask);
        if (ov > bv || (ov == bv && ot < bt)) { bv = ov; bt = ot; }
    }
    if (lane == 0) { s_best_v[wid] = bv; s_best_t[wid] = bt; }
    __syncthreads();
    if (wid == 0) {
        bv = (lane < 8) ? s_best_v[lane] : -INFINITY;
        bt = (lane < 8) ? s_best_t[lane] : 0x7fffffff;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, mask);
            int32_t ot = __shfl_xor_sync(0xffffffffu, bt, mask);
            if (ov > bv || (ov == bv && ot < bt)) { bv = ov; bt = ot; }
        }
        if (lane == 0) sampled_out[b] = static_cast<int64_t>(bt);
    }
}

template <int NUM_Q_PER_KV = 8, bool BS32 = false, bool BS16 = false>
__global__ __launch_bounds__(128, 3)
void sm80_paged_fa2_hd512_bm16_bn16_4w_kernel(
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
    constexpr int BLOCK_Q = 2;
    constexpr int BN = 16;
    constexpr int HEAD_SIZE = 512;
    constexpr float LOG2E = 1.4426950408889634f;

    int seq_idx = 0, q_blk = 0, q_start = 0, q_len = 0, seq_len = 0;
    if (!resolve_q_tile(cu_seqlens_q, seqused_k, num_seqs, BLOCK_Q,
                        blockIdx.x, gridDim.x, seq_idx, q_blk, q_start, q_len, seq_len)) {
        return;
    }

    const int kv_head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const int warp_id = static_cast<int>(tid >> 5);
    const int lane = static_cast<int>(tid & 31u);
    const int group_id = lane >> 2;
    const int quad_tid = lane & 3;

    const int context_len = seq_len - q_len;
    const int q_max_local = min((q_blk + 1) * BLOCK_Q, q_len) - 1;
    const int max_seq_prefix = min(context_len + q_max_local + 1, seq_len);
    const int n_tiles = (max_seq_prefix + BN - 1) >> 4;
    if (n_tiles <= 0) {
        if (tid < 16u) {
            int qp = q_blk * BLOCK_Q + (static_cast<int>(tid) / NUM_Q_PER_KV);
            if (qp < q_len) {
                int qh = kv_head * NUM_Q_PER_KV + (static_cast<int>(tid) % NUM_Q_PER_KV);
                __nv_bfloat16* row = out + (int64_t)(q_start + qp) * o_stride0 + (int64_t)qh * o_stride1;
                for (int d = 0; d < HEAD_SIZE; ++d) row[d] = __float2bfloat16(0.f);
            }
        }
        return;
    }

    extern __shared__ char smem_raw[];
    uint4* sBuf0 = reinterpret_cast<uint4*>(smem_raw);
    uint4* sBuf1 = sBuf0 + 16 * 64;
    uint4* sBuf2 = sBuf1 + 16 * 64;
    float* s_qk  = reinterpret_cast<float*>(sBuf2 + 16 * 64);

    const uint32_t sBuf0_u32 = smem_u32(sBuf0);
    const uint32_t r0 = tid >> 6;  // 0..1
    const uint32_t c0 = tid & 63u; // 0..63
    uint32_t ld_dst[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        ld_dst[i] = static_cast<uint32_t>(swz512(static_cast<int>(r0) + i * 2, static_cast<int>(c0))) * 16u;
    }

    #pragma unroll
    for (int idx = static_cast<int>(tid); idx < 16 * 64; idx += 128) {
        int r = idx >> 6;
        int c = idx & 63;
        int q_pos = q_blk * BLOCK_Q + (r / NUM_Q_PER_KV);
        int q_h = kv_head * NUM_Q_PER_KV + (r % NUM_Q_PER_KV);
        bool ok = q_pos < q_len;
        const __nv_bfloat16* gptr = q + (int64_t)(q_start + (ok ? q_pos : 0)) * q_stride0 + (int64_t)q_h * q_stride1 + (c << 3);
        cp_async_16B_u32(sBuf0_u32 + static_cast<uint32_t>(swz512(r, c)) * 16u, gptr, ok);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col_chunk = (lane >> 4);
    const int ldm_bk_row = (lane >> 4) * 8 + (lane & 7);
    const int ldm_bk_col_chunk = ((lane >> 3) & 1);
    const int d_warp_chunks = warp_id * 16;

    uint32_t q_reg[8][4];
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        ldmatrix_u32_x4(q_reg[ks], sBuf0_u32 + static_cast<uint32_t>(swz512(ldm_a_row, d_warp_chunks + ks * 2 + ldm_a_col_chunk)) * 16u);
    }
    __syncthreads();

    const bool blk_pow2 = (BS32 || BS16) ? true : ((block_size & (block_size - 1)) == 0);
    const int blk_shift = BS32 ? 5 : (BS16 ? 4 : (blk_pow2 ? (__ffs(block_size) - 1) : 0));
    const int blk_mask = BS32 ? 31 : (BS16 ? 15 : (block_size - 1));
    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;
    const float scale_log2 = scale * LOG2E;

    const int64_t k_s1_bytes = k_stride1 * sizeof(__nv_bfloat16);
    const int64_t k_s1_2_bytes = k_s1_bytes * 2;
    const int64_t v_s1_bytes = v_stride1 * sizeof(__nv_bfloat16);
    const int64_t v_s1_2_bytes = v_s1_bytes * 2;
    const int64_t col_bytes = static_cast<int64_t>(c0) * (8 * sizeof(__nv_bfloat16));
    const __nv_bfloat16* k_head_base = k_cache + (int64_t)kv_head * k_stride2;
    const __nv_bfloat16* v_head_base = v_cache + (int64_t)kv_head * v_stride2;

    auto issue_load_16 = [&](uint32_t dst_u32, const __nv_bfloat16* cache_head_base,
                             int64_t s0, int64_t s1, int64_t s1_bytes, int64_t s1_2_bytes,
                             int tile_idx, bool masked) {
        int tok0 = tile_idx << 4;
        int page = BS16 ? tile_idx : (blk_pow2 ? (tok0 >> blk_shift) : (tok0 / block_size));
        int off  = BS16 ? 0 : (blk_pow2 ? (tok0 & blk_mask) : (tok0 - page * block_size));
        int64_t pblk = (int64_t)bt_row[page];
        const char* ptr_b = BS16
            ? (reinterpret_cast<const char*>(cache_head_base + pblk * s0) + col_bytes)
            : (reinterpret_cast<const char*>(cache_head_base + pblk * s0 + (int64_t)off * s1) + col_bytes);
        const char* cur = ptr_b + static_cast<int64_t>(r0) * s1_bytes;
        if (!masked) {
            #pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                uint32_t h_off = dst_u32 + static_cast<uint32_t>(hi * 8192);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    cp_async_16B_unchk_u32(h_off + ld_dst[i], cur);
                    cur += s1_2_bytes;
                }
            }
        } else {
            int rem = max_seq_prefix - tok0;
            int r_cur = static_cast<int>(r0);
            #pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                uint32_t h_off = dst_u32 + static_cast<uint32_t>(hi * 8192);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    bool ok = r_cur < rem;
                    cp_async_16B_u32(h_off + ld_dst[i], ok ? cur : ptr_b, ok);
                    cur += s1_2_bytes;
                    r_cur += 2;
                }
            }
        }
    };

    const int full_tiles = (context_len + q_blk * BLOCK_Q) >> 4;
    uint32_t bK_cur_u32 = sBuf0_u32;
    uint32_t bV_cur_u32 = smem_u32(sBuf1);
    uint32_t bK_nxt_u32 = smem_u32(sBuf2);
    issue_load_16(bK_cur_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, 0, 0 >= full_tiles);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    uint32_t sK_off[4];
    uint32_t sV_off[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        sK_off[u] = static_cast<uint32_t>(swz512(ldm_bk_row, d_warp_chunks + u * 2 + ldm_bk_col_chunk)) * 16u;
        sV_off[u] = static_cast<uint32_t>(swz512(ldm_a_row, d_warp_chunks + u * 2 + ldm_a_col_chunk)) * 16u;
    }

    float acc_o[16][4];
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        acc_o[dt][0] = 0.f; acc_o[dt][1] = 0.f; acc_o[dt][2] = 0.f; acc_o[dt][3] = 0.f;
    }
    float row_m0 = -INFINITY, row_m1 = -INFINITY;
    float row_l0 = 0.f, row_l1 = 0.f;

    const int m_row0 = group_id;
    const int m_row1 = group_id + 8;
    const int q_pos0 = q_blk * BLOCK_Q + (m_row0 / NUM_Q_PER_KV);
    const int q_pos1 = q_blk * BLOCK_Q + (m_row1 / NUM_Q_PER_KV);
    const bool q_valid0 = q_pos0 < q_len;
    const bool q_valid1 = q_pos1 < q_len;
    const int q_abs0 = context_len + q_pos0;
    const int q_abs1 = context_len + q_pos1;

    for (int j = 0; j < n_tiles; ++j) {
        const bool masked_j = (j >= full_tiles);
        issue_load_16(bV_cur_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_2_bytes, j, masked_j);
        cp_async_commit();
        if (j + 1 < n_tiles) {
            issue_load_16(bK_nxt_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_2_bytes, j + 1, (j + 1) >= full_tiles);
            cp_async_commit();
        }

        float s_frag[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            uint32_t t_bytes = static_cast<uint32_t>(t * 128);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int ks = t * 4 + u;
                uint32_t k_reg[4];
                ldmatrix_u32_x4(k_reg, bK_cur_u32 + sK_off[u] + t_bytes);
                uint32_t b0[2] = {k_reg[0], k_reg[1]};
                uint32_t b1[2] = {k_reg[2], k_reg[3]};
                mma_m16n8k16_bf16_nv(s_frag[0], q_reg[ks], b0);
                mma_m16n8k16_bf16_nv(s_frag[1], q_reg[ks], b1);
            }
        }

        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            int c0_col = nt * 8 + quad_tid * 2;
            s_qk[(warp_id * 16 + m_row0) * 17 + c0_col]     = s_frag[nt][0];
            s_qk[(warp_id * 16 + m_row0) * 17 + c0_col + 1] = s_frag[nt][1];
            s_qk[(warp_id * 16 + m_row1) * 17 + c0_col]     = s_frag[nt][2];
            s_qk[(warp_id * 16 + m_row1) * 17 + c0_col + 1] = s_frag[nt][3];
        }

        if (j + 1 < n_tiles) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        float l[2][4];
        float m0_loc = -INFINITY, m1_loc = -INFINITY;
        int start_n = j * BN;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            int c0_col = nt * 8 + quad_tid * 2;
            int col0 = start_n + c0_col, col1 = col0 + 1;
            float r00 = s_qk[m_row0 * 17 + c0_col] + s_qk[(16 + m_row0) * 17 + c0_col] + s_qk[(32 + m_row0) * 17 + c0_col] + s_qk[(48 + m_row0) * 17 + c0_col];
            float r01 = s_qk[m_row0 * 17 + c0_col + 1] + s_qk[(16 + m_row0) * 17 + c0_col + 1] + s_qk[(32 + m_row0) * 17 + c0_col + 1] + s_qk[(48 + m_row0) * 17 + c0_col + 1];
            float r10 = s_qk[m_row1 * 17 + c0_col] + s_qk[(16 + m_row1) * 17 + c0_col] + s_qk[(32 + m_row1) * 17 + c0_col] + s_qk[(48 + m_row1) * 17 + c0_col];
            float r11 = s_qk[m_row1 * 17 + c0_col + 1] + s_qk[(16 + m_row1) * 17 + c0_col + 1] + s_qk[(32 + m_row1) * 17 + c0_col + 1] + s_qk[(48 + m_row1) * 17 + c0_col + 1];
            if (!masked_j) {
                l[nt][0] = q_valid0 ? (r00 * scale_log2) : -INFINITY;
                l[nt][1] = q_valid0 ? (r01 * scale_log2) : -INFINITY;
                l[nt][2] = q_valid1 ? (r10 * scale_log2) : -INFINITY;
                l[nt][3] = q_valid1 ? (r11 * scale_log2) : -INFINITY;
            } else {
                l[nt][0] = (q_valid0 && col0 < seq_len && col0 <= q_abs0) ? (r00 * scale_log2) : -INFINITY;
                l[nt][1] = (q_valid0 && col1 < seq_len && col1 <= q_abs0) ? (r01 * scale_log2) : -INFINITY;
                l[nt][2] = (q_valid1 && col0 < seq_len && col0 <= q_abs1) ? (r10 * scale_log2) : -INFINITY;
                l[nt][3] = (q_valid1 && col1 < seq_len && col1 <= q_abs1) ? (r11 * scale_log2) : -INFINITY;
            }
            m0_loc = fmaxf(m0_loc, fmaxf(l[nt][0], l[nt][1]));
            m1_loc = fmaxf(m1_loc, fmaxf(l[nt][2], l[nt][3]));
        }
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 1));
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 2));
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 1));
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 2));

        float new_m0 = fmaxf(row_m0, m0_loc);
        float new_m1 = fmaxf(row_m1, m1_loc);
        float alpha0 = (row_m0 == -INFINITY || new_m0 == -INFINITY) ? 0.f : fa2_ex2_approx(row_m0 - new_m0);
        float alpha1 = (row_m1 == -INFINITY || new_m1 == -INFINITY) ? 0.f : fa2_ex2_approx(row_m1 - new_m1);
        float sub_m0 = (new_m0 == -INFINITY) ? 0.f : new_m0;
        float sub_m1 = (new_m1 == -INFINITY) ? 0.f : new_m1;

        float pf[2][4];
        float sum0_loc = 0.f, sum1_loc = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pf[nt][0] = (l[nt][0] == -INFINITY) ? 0.f : fa2_ex2_approx(l[nt][0] - sub_m0);
            pf[nt][1] = (l[nt][1] == -INFINITY) ? 0.f : fa2_ex2_approx(l[nt][1] - sub_m0);
            pf[nt][2] = (l[nt][2] == -INFINITY) ? 0.f : fa2_ex2_approx(l[nt][2] - sub_m1);
            pf[nt][3] = (l[nt][3] == -INFINITY) ? 0.f : fa2_ex2_approx(l[nt][3] - sub_m1);
            sum0_loc += pf[nt][0] + pf[nt][1];
            sum1_loc += pf[nt][2] + pf[nt][3];
        }
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 1);
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 2);
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 1);
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 2);

        if (j > 0) {
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                acc_o[dt][0] *= alpha0; acc_o[dt][1] *= alpha0;
                acc_o[dt][2] *= alpha1; acc_o[dt][3] *= alpha1;
            }
        }
        row_l0 = row_l0 * alpha0 + sum0_loc;
        row_l1 = row_l1 * alpha1 + sum1_loc;
        row_m0 = new_m0;
        row_m1 = new_m1;

        uint32_t p_reg[4] = {
            pack_bf16x2(pf[0][0], pf[0][1]),
            pack_bf16x2(pf[0][2], pf[0][3]),
            pack_bf16x2(pf[1][0], pf[1][1]),
            pack_bf16x2(pf[1][2], pf[1][3])
        };

        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            uint32_t t_bytes = static_cast<uint32_t>(t * 128);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int dg = t * 4 + u;
                uint32_t v_reg[4];
                ldmatrix_u32_x4_trans(v_reg, bV_cur_u32 + sV_off[u] + t_bytes);
                uint32_t vb0[2] = {v_reg[0], v_reg[1]};
                uint32_t vb1[2] = {v_reg[2], v_reg[3]};
                mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 0], p_reg, vb0);
                mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 1], p_reg, vb1);
            }
        }

        if (j + 1 < n_tiles) {
            cp_async_wait<0>();
            __syncthreads();
            uint32_t tmp = bK_cur_u32;
            bK_cur_u32 = bK_nxt_u32;
            bK_nxt_u32 = bV_cur_u32;
            bV_cur_u32 = tmp;
        }
    }

    int q_h0 = kv_head * NUM_Q_PER_KV + (m_row0 % NUM_Q_PER_KV);
    int q_h1 = kv_head * NUM_Q_PER_KV + (m_row1 % NUM_Q_PER_KV);
    float inv_l0 = (row_l0 > 0.f) ? (1.0f / row_l0) : 0.f;
    float inv_l1 = (row_l1 > 0.f) ? (1.0f / row_l1) : 0.f;
    __nv_bfloat16* out_row0 = out + (int64_t)(q_start + q_pos0) * o_stride0 + (int64_t)q_h0 * o_stride1 + warp_id * 128;
    __nv_bfloat16* out_row1 = out + (int64_t)(q_start + q_pos1) * o_stride0 + (int64_t)q_h1 * o_stride1 + warp_id * 128;
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        int col_pair = dt * 8 + quad_tid * 2;
        if (q_valid0) *reinterpret_cast<uint32_t*>(out_row0 + col_pair) = pack_bf16x2(acc_o[dt][0] * inv_l0, acc_o[dt][1] * inv_l0);
        if (q_valid1) *reinterpret_cast<uint32_t*>(out_row1 + col_pair) = pack_bf16x2(acc_o[dt][2] * inv_l1, acc_o[dt][3] * inv_l1);
    }
}

template <int NUM_Q_PER_KV = 8, bool BS_EQ_32 = false, bool BS32 = false>
__global__ __launch_bounds__(128, 2)
void sm80_paged_fa2_hd256_bm32_bn32_4w_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    int sliding_window_val,
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
    constexpr int BLOCK_Q = 4;
    constexpr int BN = 32;
    constexpr int HEAD_SIZE = 256;
    constexpr int CHUNKS = 32;
    constexpr float LOG2E = 1.4426950408889634f;

    int seq_idx = 0, q_blk = 0, q_start = 0, q_len = 0, seq_len = 0;
    if (!resolve_q_tile(cu_seqlens_q, seqused_k, num_seqs, BLOCK_Q,
                        blockIdx.x, gridDim.x, seq_idx, q_blk, q_start, q_len, seq_len)) {
        return;
    }

    const int kv_head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const int warp_id = static_cast<int>(tid >> 5);
    const int warp_m = warp_id >> 1;
    const int warp_d = warp_id & 1;
    const int lane = static_cast<int>(tid & 31u);
    const int group_id = lane >> 2;
    const int quad_tid = lane & 3;

    const int context_len = seq_len - q_len;
    const int q_min_local = q_blk * BLOCK_Q;
    const int q_max_local = min((q_blk + 1) * BLOCK_Q, q_len) - 1;
    const int max_seq_prefix = min(context_len + q_max_local + 1, seq_len);
    int k_min_abs = 0;
    if (sliding_window_val > 0) {
        k_min_abs = max(0, context_len + q_min_local + 1 - sliding_window_val);
    }
    const int tile_lo = k_min_abs >> 5;
    const int tile_hi = (max_seq_prefix + BN - 1) >> 5;
    if (tile_hi <= tile_lo) {
        if (tid < 32u) {
            int qp = q_blk * BLOCK_Q + (static_cast<int>(tid) / NUM_Q_PER_KV);
            if (qp < q_len) {
                int qh = kv_head * NUM_Q_PER_KV + (static_cast<int>(tid) % NUM_Q_PER_KV);
                __nv_bfloat16* row = out + (int64_t)(q_start + qp) * o_stride0 + (int64_t)qh * o_stride1;
                for (int d = 0; d < HEAD_SIZE; ++d) row[d] = __float2bfloat16(0.f);
            }
        }
        return;
    }

    extern __shared__ char smem_raw[];
    uint4* sQ    = reinterpret_cast<uint4*>(smem_raw);
    uint4* sBuf0 = sQ + 32 * CHUNKS;
    uint4* sBuf1 = sBuf0 + 32 * CHUNKS;
    uint4* sBuf2 = sBuf1 + 32 * CHUNKS;
    float* s_qk  = reinterpret_cast<float*>(sBuf2 + 32 * CHUNKS);

    const uint32_t sQ_u32_base = smem_u32(sQ);
    const uint32_t r0 = tid >> 5;
    const uint32_t c0 = tid & 31u;
    const uint32_t ld_dst_even = static_cast<uint32_t>(swz256(static_cast<int>(r0), static_cast<int>(c0))) * 16u;
    const uint32_t ld_dst_odd  = static_cast<uint32_t>(swz256(static_cast<int>(r0 + 4u), static_cast<int>(c0))) * 16u;

    #pragma unroll
    for (int idx = static_cast<int>(tid); idx < 32 * CHUNKS; idx += 128) {
        int r = idx >> 5;
        int c = idx & 31;
        int q_pos = q_blk * BLOCK_Q + (r / NUM_Q_PER_KV);
        int q_h = kv_head * NUM_Q_PER_KV + (r % NUM_Q_PER_KV);
        bool ok = q_pos < q_len;
        const __nv_bfloat16* gptr = q + (int64_t)(q_start + (ok ? q_pos : 0)) * q_stride0 + (int64_t)q_h * q_stride1 + (c << 3);
        cp_async_16B_u32(sQ_u32_base + static_cast<uint32_t>(swz256(r, c)) * 16u, gptr, ok);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col_chunk = (lane >> 4);
    const int ldm_bk_row = (lane >> 4) * 8 + (lane & 7);
    const int ldm_bk_col_chunk = ((lane >> 3) & 1);
    const int d_warp_chunks = warp_d * 16;

    uint32_t q_reg[8][4];
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        ldmatrix_u32_x4(q_reg[ks], sQ_u32_base + static_cast<uint32_t>(swz256(warp_m * 16 + ldm_a_row, d_warp_chunks + ks * 2 + ldm_a_col_chunk)) * 16u);
    }
    __syncthreads();

    const bool blk_pow2 = BS_EQ_32 ? true : ((block_size & (block_size - 1)) == 0);
    const int blk_shift = BS_EQ_32 ? 5 : (blk_pow2 ? (__ffs(block_size) - 1) : 0);
    const int blk_mask = BS_EQ_32 ? 31 : (block_size - 1);
    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;
    const float scale_log2 = scale * LOG2E;

    const int64_t k_s1_bytes = k_stride1 * sizeof(__nv_bfloat16);
    const int64_t k_s1_4_bytes = k_s1_bytes * 4;
    const int64_t v_s1_bytes = v_stride1 * sizeof(__nv_bfloat16);
    const int64_t v_s1_4_bytes = v_s1_bytes * 4;
    const int64_t col_bytes = static_cast<int64_t>(c0) * (8 * sizeof(__nv_bfloat16));
    const __nv_bfloat16* k_head_base = k_cache + (int64_t)kv_head * k_stride2;
    const __nv_bfloat16* v_head_base = v_cache + (int64_t)kv_head * v_stride2;

    auto issue_load_32 = [&](uint32_t dst_u32, const __nv_bfloat16* cache_head_base,
                             int64_t s0, int64_t s1, int64_t s1_bytes, int64_t s1_4_bytes,
                             int start_tok, bool masked) {
        if (masked && start_tok >= max_seq_prefix) {
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                st_shared_zero_u32(p_off + ld_dst_even);
                st_shared_zero_u32(p_off + ld_dst_odd);
            }
            return;
        }
        if constexpr (BS32) {
            int page = BS_EQ_32 ? (start_tok >> 5) : (blk_pow2 ? (start_tok >> blk_shift) : (start_tok / block_size));
            int off  = BS_EQ_32 ? 0 : (blk_pow2 ? (start_tok & blk_mask) : (start_tok - page * block_size));
            int64_t pblk = (int64_t)bt_row[page];
            const char* ptr_b = reinterpret_cast<const char*>(
                cache_head_base + pblk * s0 + (int64_t)off * s1) + col_bytes;
            if (!masked) {
                const char* cur = ptr_b + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    cp_async_16B_unchk_u32(p_off + ld_dst_even, cur);
                    cur += s1_4_bytes;
                    cp_async_16B_unchk_u32(p_off + ld_dst_odd, cur);
                    cur += s1_4_bytes;
                }
            } else {
                int rem = max_seq_prefix - start_tok;
                int re = static_cast<int>(r0);
                int ro = re + 4;
                const char* cur = ptr_b + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    bool ok_e = re < rem;
                    cp_async_16B_u32(p_off + ld_dst_even, ok_e ? cur : ptr_b, ok_e);
                    cur += s1_4_bytes;
                    bool ok_o = ro < rem;
                    cp_async_16B_u32(p_off + ld_dst_odd, ok_o ? cur : ptr_b, ok_o);
                    cur += s1_4_bytes;
                    re += 8;
                    ro += 8;
                }
            }
        } else {
            int page0 = blk_pow2 ? (start_tok >> blk_shift) : (start_tok / block_size);
            int off0  = blk_pow2 ? (start_tok & blk_mask) : (start_tok - page0 * block_size);
            int64_t pblk0 = (int64_t)bt_row[page0];
            const char* ptr0 = reinterpret_cast<const char*>(
                cache_head_base + pblk0 * s0 + (int64_t)off0 * s1) + col_bytes;
            int tok1 = start_tok + 16;
            if (!masked) {
                int page1 = blk_pow2 ? (tok1 >> blk_shift) : (tok1 / block_size);
                int off1  = blk_pow2 ? (tok1 & blk_mask) : (tok1 - page1 * block_size);
                int64_t pblk1 = (int64_t)bt_row[page1];
                const char* ptr1 = reinterpret_cast<const char*>(
                    cache_head_base + pblk1 * s0 + (int64_t)off1 * s1) + col_bytes;
                const char* cur0 = ptr0 + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    cp_async_16B_unchk_u32(p_off + ld_dst_even, cur0);
                    cur0 += s1_4_bytes;
                    cp_async_16B_unchk_u32(p_off + ld_dst_odd, cur0);
                    cur0 += s1_4_bytes;
                }
                const char* cur1 = ptr1 + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 2; p < 4; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    cp_async_16B_unchk_u32(p_off + ld_dst_even, cur1);
                    cur1 += s1_4_bytes;
                    cp_async_16B_unchk_u32(p_off + ld_dst_odd, cur1);
                    cur1 += s1_4_bytes;
                }
            } else {
                bool hi_valid = tok1 < max_seq_prefix;
                const char* ptr1 = ptr0;
                if (hi_valid) {
                    int page1 = blk_pow2 ? (tok1 >> blk_shift) : (tok1 / block_size);
                    int off1  = blk_pow2 ? (tok1 & blk_mask) : (tok1 - page1 * block_size);
                    int64_t pblk1 = (int64_t)bt_row[page1];
                    ptr1 = reinterpret_cast<const char*>(
                        cache_head_base + pblk1 * s0 + (int64_t)off1 * s1) + col_bytes;
                }
                int rem = max_seq_prefix - start_tok;
                int re = static_cast<int>(r0);
                int ro = re + 4;
                const char* cur0 = ptr0 + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    bool ok_e = re < rem;
                    cp_async_16B_u32(p_off + ld_dst_even, ok_e ? cur0 : ptr0, ok_e);
                    cur0 += s1_4_bytes;
                    bool ok_o = ro < rem;
                    cp_async_16B_u32(p_off + ld_dst_odd, ok_o ? cur0 : ptr0, ok_o);
                    cur0 += s1_4_bytes;
                    re += 8;
                    ro += 8;
                }
                const char* cur1 = ptr1 + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 2; p < 4; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    bool ok_e = re < rem;
                    cp_async_16B_u32(p_off + ld_dst_even, ok_e ? cur1 : ptr1, ok_e);
                    cur1 += s1_4_bytes;
                    bool ok_o = ro < rem;
                    cp_async_16B_u32(p_off + ld_dst_odd, ok_o ? cur1 : ptr1, ok_o);
                    cur1 += s1_4_bytes;
                    re += 8;
                    ro += 8;
                }
            }
        }
    };

    const int full_tiles_hi = (context_len + q_min_local) >> 5;
    const int sw_full_lo = (sliding_window_val > 0)
        ? ((max(0, context_len + q_max_local + 1 - sliding_window_val) + 31) >> 5)
        : 0;

    uint32_t bK_cur_u32 = smem_u32(sBuf0);
    uint32_t bV_cur_u32 = smem_u32(sBuf1);
    uint32_t bK_nxt_u32 = smem_u32(sBuf2);
    issue_load_32(bK_cur_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_4_bytes, tile_lo << 5, tile_lo >= full_tiles_hi);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    uint32_t sK_off[4];
    uint32_t sV_off[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        sK_off[u] = static_cast<uint32_t>(swz256(ldm_bk_row, d_warp_chunks + u * 2 + ldm_bk_col_chunk)) * 16u;
        sV_off[u] = static_cast<uint32_t>(swz256(ldm_a_row, d_warp_chunks + u * 2 + ldm_a_col_chunk)) * 16u;
    }

    float acc_o[16][4];
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        acc_o[dt][0] = 0.f; acc_o[dt][1] = 0.f; acc_o[dt][2] = 0.f; acc_o[dt][3] = 0.f;
    }
    float row_m0 = -INFINITY, row_m1 = -INFINITY;
    float row_l0 = 0.f, row_l1 = 0.f;

    const int m_row0 = warp_m * 16 + group_id;
    const int m_row1 = m_row0 + 8;
    const int q_pos0 = q_blk * BLOCK_Q + (m_row0 / NUM_Q_PER_KV);
    const int q_pos1 = q_blk * BLOCK_Q + (m_row1 / NUM_Q_PER_KV);
    const bool q_valid0 = q_pos0 < q_len;
    const bool q_valid1 = q_pos1 < q_len;
    const int q_abs0 = context_len + q_pos0;
    const int q_abs1 = context_len + q_pos1;
    const int q_lo0 = (sliding_window_val > 0) ? max(0, q_abs0 - sliding_window_val + 1) : 0;
    const int q_lo1 = (sliding_window_val > 0) ? max(0, q_abs1 - sliding_window_val + 1) : 0;

    auto run_j = [&](int j, bool is_fast) __attribute__((always_inline)) {
        const int start_n = j << 5;
        const bool masked_j = is_fast ? false : ((j >= full_tiles_hi) || (j < sw_full_lo));
        issue_load_32(bV_cur_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_4_bytes, start_n, is_fast ? false : (j >= full_tiles_hi));
        cp_async_commit();
        const bool has_next = (j + 1 < tile_hi);
        if (has_next) {
            issue_load_32(bK_nxt_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_4_bytes, (j + 1) << 5, (j + 1) >= full_tiles_hi);
            cp_async_commit();
        }

        uint32_t k_u32[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            k_u32[u] = bK_cur_u32 + sK_off[u];
        }

        float s_frag[4][4] = {
            {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f},
            {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}
        };
        #pragma unroll
        for (int np = 0; np < 2; ++np) {
            uint32_t np_bytes = static_cast<uint32_t>(np * 8192);
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t nt_bytes = np_bytes + static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    int ks = t * 4 + u;
                    uint32_t k_reg[4];
                    ldmatrix_u32_x4(k_reg, k_u32[u] + nt_bytes);
                    uint32_t b0[2] = {k_reg[0], k_reg[1]};
                    uint32_t b1[2] = {k_reg[2], k_reg[3]};
                    mma_m16n8k16_bf16_nv(s_frag[np * 2 + 0], q_reg[ks], b0);
                    mma_m16n8k16_bf16_nv(s_frag[np * 2 + 1], q_reg[ks], b1);
                }
            }
        }

        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            int c0_col = nt * 8 + quad_tid * 2;
            s_qk[(warp_id * 16 + group_id) * 33 + c0_col]         = s_frag[nt][0];
            s_qk[(warp_id * 16 + group_id) * 33 + c0_col + 1]     = s_frag[nt][1];
            s_qk[(warp_id * 16 + group_id + 8) * 33 + c0_col]     = s_frag[nt][2];
            s_qk[(warp_id * 16 + group_id + 8) * 33 + c0_col + 1] = s_frag[nt][3];
        }

        if (has_next) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        int s_base0 = (warp_m * 2) * 16 + group_id;
        int s_base1 = (warp_m * 2 + 1) * 16 + group_id;

        float l[4][4];
        float m0_loc = -INFINITY, m1_loc = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            int c0_col = nt * 8 + quad_tid * 2;
            int col0 = start_n + c0_col, col1 = col0 + 1;
            float r00 = s_qk[s_base0 * 33 + c0_col] + s_qk[s_base1 * 33 + c0_col];
            float r01 = s_qk[s_base0 * 33 + c0_col + 1] + s_qk[s_base1 * 33 + c0_col + 1];
            float r10 = s_qk[(s_base0 + 8) * 33 + c0_col] + s_qk[(s_base1 + 8) * 33 + c0_col];
            float r11 = s_qk[(s_base0 + 8) * 33 + c0_col + 1] + s_qk[(s_base1 + 8) * 33 + c0_col + 1];
            if (is_fast) {
                l[nt][0] = r00 * scale_log2;
                l[nt][1] = r01 * scale_log2;
                l[nt][2] = r10 * scale_log2;
                l[nt][3] = r11 * scale_log2;
            } else if (!masked_j) {
                l[nt][0] = q_valid0 ? (r00 * scale_log2) : -INFINITY;
                l[nt][1] = q_valid0 ? (r01 * scale_log2) : -INFINITY;
                l[nt][2] = q_valid1 ? (r10 * scale_log2) : -INFINITY;
                l[nt][3] = q_valid1 ? (r11 * scale_log2) : -INFINITY;
            } else {
                l[nt][0] = (q_valid0 && col0 < seq_len && col0 <= q_abs0 && col0 >= q_lo0) ? (r00 * scale_log2) : -INFINITY;
                l[nt][1] = (q_valid0 && col1 < seq_len && col1 <= q_abs0 && col1 >= q_lo0) ? (r01 * scale_log2) : -INFINITY;
                l[nt][2] = (q_valid1 && col0 < seq_len && col0 <= q_abs1 && col0 >= q_lo1) ? (r10 * scale_log2) : -INFINITY;
                l[nt][3] = (q_valid1 && col1 < seq_len && col1 <= q_abs1 && col1 >= q_lo1) ? (r11 * scale_log2) : -INFINITY;
            }
            m0_loc = fmaxf(m0_loc, fmaxf(l[nt][0], l[nt][1]));
            m1_loc = fmaxf(m1_loc, fmaxf(l[nt][2], l[nt][3]));
        }
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 1));
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 2));
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 1));
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 2));

        float new_m0 = fmaxf(row_m0, m0_loc);
        float new_m1 = fmaxf(row_m1, m1_loc);
        float alpha0 = is_fast ? fa2_ex2_approx(row_m0 - new_m0)
                               : ((row_m0 == -INFINITY || new_m0 == -INFINITY) ? 0.f : fa2_ex2_approx(row_m0 - new_m0));
        float alpha1 = is_fast ? fa2_ex2_approx(row_m1 - new_m1)
                               : ((row_m1 == -INFINITY || new_m1 == -INFINITY) ? 0.f : fa2_ex2_approx(row_m1 - new_m1));
        float sub_m0 = is_fast ? new_m0 : ((new_m0 == -INFINITY) ? 0.f : new_m0);
        float sub_m1 = is_fast ? new_m1 : ((new_m1 == -INFINITY) ? 0.f : new_m1);

        float pf[4][4];
        float sum0_loc = 0.f, sum1_loc = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            if (is_fast) {
                pf[nt][0] = fa2_ex2_approx(l[nt][0] - sub_m0);
                pf[nt][1] = fa2_ex2_approx(l[nt][1] - sub_m0);
                pf[nt][2] = fa2_ex2_approx(l[nt][2] - sub_m1);
                pf[nt][3] = fa2_ex2_approx(l[nt][3] - sub_m1);
            } else {
                pf[nt][0] = (l[nt][0] == -INFINITY) ? 0.f : fa2_ex2_approx(l[nt][0] - sub_m0);
                pf[nt][1] = (l[nt][1] == -INFINITY) ? 0.f : fa2_ex2_approx(l[nt][1] - sub_m0);
                pf[nt][2] = (l[nt][2] == -INFINITY) ? 0.f : fa2_ex2_approx(l[nt][2] - sub_m1);
                pf[nt][3] = (l[nt][3] == -INFINITY) ? 0.f : fa2_ex2_approx(l[nt][3] - sub_m1);
            }
            sum0_loc += pf[nt][0] + pf[nt][1];
            sum1_loc += pf[nt][2] + pf[nt][3];
        }
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 1);
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 2);
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 1);
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 2);

        if (is_fast || (j > tile_lo)) {
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                acc_o[dt][0] *= alpha0; acc_o[dt][1] *= alpha0;
                acc_o[dt][2] *= alpha1; acc_o[dt][3] *= alpha1;
            }
        }
        row_l0 = row_l0 * alpha0 + sum0_loc;
        row_l1 = row_l1 * alpha1 + sum1_loc;
        row_m0 = new_m0;
        row_m1 = new_m1;

        uint32_t v_u32[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            v_u32[u] = bV_cur_u32 + sV_off[u];
        }

        #pragma unroll
        for (int vp = 0; vp < 2; ++vp) {
            int n0 = vp * 2;
            uint32_t p_reg[4] = {
                pack_bf16x2(pf[n0 + 0][0], pf[n0 + 0][1]),
                pack_bf16x2(pf[n0 + 0][2], pf[n0 + 0][3]),
                pack_bf16x2(pf[n0 + 1][0], pf[n0 + 1][1]),
                pack_bf16x2(pf[n0 + 1][2], pf[n0 + 1][3])
            };
            uint32_t vp_bytes = static_cast<uint32_t>(vp * 8192);
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t vt_bytes = vp_bytes + static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    int dg = t * 4 + u;
                    uint32_t v_reg[4];
                    ldmatrix_u32_x4_trans(v_reg, v_u32[u] + vt_bytes);
                    uint32_t vb0[2] = {v_reg[0], v_reg[1]};
                    uint32_t vb1[2] = {v_reg[2], v_reg[3]};
                    mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 0], p_reg, vb0);
                    mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 1], p_reg, vb1);
                }
            }
        }

        if (has_next) {
            cp_async_wait<0>();
            __syncthreads();
            uint32_t tmp = bK_cur_u32;
            bK_cur_u32 = bK_nxt_u32;
            bK_nxt_u32 = bV_cur_u32;
            bV_cur_u32 = tmp;
        }
    };

    const int f_lo = max(tile_lo + 1, sw_full_lo);
    const int f_hi = min(full_tiles_hi, tile_hi);
    if (q_valid0 && q_valid1 && (sliding_window_val == 0 || sliding_window_val >= BLOCK_Q) && f_lo < f_hi) {
        for (int j = tile_lo; j < f_lo; ++j) run_j(j, false);
        for (int j = f_lo; j < f_hi; ++j) run_j(j, true);
        for (int j = f_hi; j < tile_hi; ++j) run_j(j, false);
    } else {
        for (int j = tile_lo; j < tile_hi; ++j) run_j(j, false);
    }

    int q_h0 = kv_head * NUM_Q_PER_KV + (m_row0 % NUM_Q_PER_KV);
    int q_h1 = kv_head * NUM_Q_PER_KV + (m_row1 % NUM_Q_PER_KV);
    float inv_l0 = (row_l0 > 0.f) ? (1.0f / row_l0) : 0.f;
    float inv_l1 = (row_l1 > 0.f) ? (1.0f / row_l1) : 0.f;
    __nv_bfloat16* out_row0 = out + (int64_t)(q_start + q_pos0) * o_stride0 + (int64_t)q_h0 * o_stride1 + warp_d * 128;
    __nv_bfloat16* out_row1 = out + (int64_t)(q_start + q_pos1) * o_stride0 + (int64_t)q_h1 * o_stride1 + warp_d * 128;
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        int col_pair = dt * 8 + quad_tid * 2;
        if (q_valid0) *reinterpret_cast<uint32_t*>(out_row0 + col_pair) = pack_bf16x2(acc_o[dt][0] * inv_l0, acc_o[dt][1] * inv_l0);
        if (q_valid1) *reinterpret_cast<uint32_t*>(out_row1 + col_pair) = pack_bf16x2(acc_o[dt][2] * inv_l1, acc_o[dt][3] * inv_l1);
    }
}

template <int NUM_Q_PER_KV = 8, bool BS_EQ_32 = false, bool BS32 = false>
__global__ __launch_bounds__(128)
void sm80_paged_fa2_hd256_bm64_bn32_4w_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    int sliding_window_val,
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
    constexpr int BLOCK_Q = 8;
    constexpr int BN = 32;
    constexpr int HEAD_SIZE = 256;
    constexpr int CHUNKS = 32;
    constexpr float LOG2E = 1.4426950408889634f;

    int seq_idx = 0, q_blk = 0, q_start = 0, q_len = 0, seq_len = 0;
    if (!resolve_q_tile(cu_seqlens_q, seqused_k, num_seqs, BLOCK_Q,
                        blockIdx.x, gridDim.x, seq_idx, q_blk, q_start, q_len, seq_len)) {
        return;
    }

    const int kv_head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const int warp_id = static_cast<int>(tid >> 5);
    const int lane = static_cast<int>(tid & 31u);
    const int group_id = lane >> 2;
    const int quad_tid = lane & 3;

    const int context_len = seq_len - q_len;
    const int q_min_local = q_blk * BLOCK_Q;
    const int q_max_local = min((q_blk + 1) * BLOCK_Q, q_len) - 1;
    const int max_seq_prefix = min(context_len + q_max_local + 1, seq_len);
    int k_min_abs = 0;
    if (sliding_window_val > 0) {
        k_min_abs = max(0, context_len + q_min_local + 1 - sliding_window_val);
    }
    const int tile_lo = k_min_abs >> 5;
    const int tile_hi = (max_seq_prefix + BN - 1) >> 5;
    if (tile_hi <= tile_lo) {
        if (tid < 64u) {
            int qp = q_blk * BLOCK_Q + (static_cast<int>(tid) / NUM_Q_PER_KV);
            if (qp < q_len) {
                int qh = kv_head * NUM_Q_PER_KV + (static_cast<int>(tid) % NUM_Q_PER_KV);
                __nv_bfloat16* row = out + (int64_t)(q_start + qp) * o_stride0 + (int64_t)qh * o_stride1;
                for (int d = 0; d < HEAD_SIZE; ++d) row[d] = __float2bfloat16(0.f);
            }
        }
        return;
    }

    extern __shared__ char smem_raw[];
    uint4* sQ    = reinterpret_cast<uint4*>(smem_raw);
    uint4* sBuf0 = sQ + 64 * CHUNKS;
    uint4* sBuf1 = sBuf0 + 32 * CHUNKS;
    uint4* sBuf2 = sBuf1 + 32 * CHUNKS;

    const uint32_t sQ_u32_base = smem_u32(sQ);
    const uint32_t r0 = tid >> 5;
    const uint32_t c0 = tid & 31u;
    const uint32_t ld_dst_even = static_cast<uint32_t>(swz256(static_cast<int>(r0), static_cast<int>(c0))) * 16u;
    const uint32_t ld_dst_odd  = static_cast<uint32_t>(swz256(static_cast<int>(r0 + 4u), static_cast<int>(c0))) * 16u;

    #pragma unroll
    for (int idx = static_cast<int>(tid); idx < 64 * CHUNKS; idx += 128) {
        int r = idx >> 5;
        int c = idx & 31;
        int q_pos = q_blk * BLOCK_Q + (r / NUM_Q_PER_KV);
        int q_h = kv_head * NUM_Q_PER_KV + (r % NUM_Q_PER_KV);
        bool ok = q_pos < q_len;
        const __nv_bfloat16* gptr = q + (int64_t)(q_start + (ok ? q_pos : 0)) * q_stride0 + (int64_t)q_h * q_stride1 + (c << 3);
        cp_async_16B_zfill_u32(sQ_u32_base + static_cast<uint32_t>(swz256(r, c)) * 16u, gptr, ok);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const bool blk_pow2 = BS_EQ_32 ? true : ((block_size & (block_size - 1)) == 0);
    const int blk_shift = BS_EQ_32 ? 5 : (blk_pow2 ? (__ffs(block_size) - 1) : 0);
    const int blk_mask = BS_EQ_32 ? 31 : (block_size - 1);
    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;
    const float scale_log2 = scale * LOG2E;

    const int64_t k_s1_bytes = k_stride1 * sizeof(__nv_bfloat16);
    const int64_t k_s1_4_bytes = k_s1_bytes * 4;
    const int64_t v_s1_bytes = v_stride1 * sizeof(__nv_bfloat16);
    const int64_t v_s1_4_bytes = v_s1_bytes * 4;
    const int64_t col_bytes = static_cast<int64_t>(c0) * (8 * sizeof(__nv_bfloat16));
    const __nv_bfloat16* k_head_base = k_cache + (int64_t)kv_head * k_stride2;
    const __nv_bfloat16* v_head_base = v_cache + (int64_t)kv_head * v_stride2;

    auto issue_load_32 = [&](uint32_t dst_u32, const __nv_bfloat16* cache_head_base,
                             int64_t s0, int64_t s1, int64_t s1_bytes, int64_t s1_4_bytes,
                             int start_tok, bool masked) {
        if constexpr (BS_EQ_32) {
            int64_t pblk = (int64_t)bt_row[start_tok >> 5];
            const char* ptr_b = reinterpret_cast<const char*>(cache_head_base + pblk * s0) + col_bytes;
            int rem = max_seq_prefix - start_tok;
            int re = static_cast<int>(r0);
            int ro = re + 4;
            const char* cur = ptr_b + static_cast<int64_t>(r0) * s1_bytes;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                bool ok_e = !masked || (re < rem);
                bool ok_o = !masked || (ro < rem);
                cp_async_16B_zfill_u32(p_off + ld_dst_even, cur, ok_e);
                cur += s1_4_bytes;
                cp_async_16B_zfill_u32(p_off + ld_dst_odd, cur, ok_o);
                cur += s1_4_bytes;
                re += 8;
                ro += 8;
            }
            return;
        }
        if (masked && start_tok >= max_seq_prefix) {
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                st_shared_zero_u32(p_off + ld_dst_even);
                st_shared_zero_u32(p_off + ld_dst_odd);
            }
            return;
        }
        if constexpr (BS32) {
            int page = BS_EQ_32 ? (start_tok >> 5) : (blk_pow2 ? (start_tok >> blk_shift) : (start_tok / block_size));
            int off  = BS_EQ_32 ? 0 : (blk_pow2 ? (start_tok & blk_mask) : (start_tok - page * block_size));
            int64_t pblk = (int64_t)bt_row[page];
            const char* ptr_b = reinterpret_cast<const char*>(
                cache_head_base + pblk * s0 + (int64_t)off * s1) + col_bytes;
            if (!masked) {
                const char* cur = ptr_b + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    cp_async_16B_unchk_u32(p_off + ld_dst_even, cur);
                    cur += s1_4_bytes;
                    cp_async_16B_unchk_u32(p_off + ld_dst_odd, cur);
                    cur += s1_4_bytes;
                }
            } else {
                int rem = max_seq_prefix - start_tok;
                int re = static_cast<int>(r0);
                int ro = re + 4;
                const char* cur = ptr_b + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    bool ok_e = re < rem;
                    cp_async_16B_u32(p_off + ld_dst_even, ok_e ? cur : ptr_b, ok_e);
                    cur += s1_4_bytes;
                    bool ok_o = ro < rem;
                    cp_async_16B_u32(p_off + ld_dst_odd, ok_o ? cur : ptr_b, ok_o);
                    cur += s1_4_bytes;
                    re += 8;
                    ro += 8;
                }
            }
        } else {
            int page0 = blk_pow2 ? (start_tok >> blk_shift) : (start_tok / block_size);
            int off0  = blk_pow2 ? (start_tok & blk_mask) : (start_tok - page0 * block_size);
            int64_t pblk0 = (int64_t)bt_row[page0];
            const char* ptr0 = reinterpret_cast<const char*>(
                cache_head_base + pblk0 * s0 + (int64_t)off0 * s1) + col_bytes;
            int tok1 = start_tok + 16;
            if (!masked) {
                int page1 = blk_pow2 ? (tok1 >> blk_shift) : (tok1 / block_size);
                int off1  = blk_pow2 ? (tok1 & blk_mask) : (tok1 - page1 * block_size);
                int64_t pblk1 = (int64_t)bt_row[page1];
                const char* ptr1 = reinterpret_cast<const char*>(
                    cache_head_base + pblk1 * s0 + (int64_t)off1 * s1) + col_bytes;
                const char* cur0 = ptr0 + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    cp_async_16B_unchk_u32(p_off + ld_dst_even, cur0);
                    cur0 += s1_4_bytes;
                    cp_async_16B_unchk_u32(p_off + ld_dst_odd, cur0);
                    cur0 += s1_4_bytes;
                }
                const char* cur1 = ptr1 + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 2; p < 4; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    cp_async_16B_unchk_u32(p_off + ld_dst_even, cur1);
                    cur1 += s1_4_bytes;
                    cp_async_16B_unchk_u32(p_off + ld_dst_odd, cur1);
                    cur1 += s1_4_bytes;
                }
            } else {
                bool hi_valid = tok1 < max_seq_prefix;
                const char* ptr1 = ptr0;
                if (hi_valid) {
                    int page1 = blk_pow2 ? (tok1 >> blk_shift) : (tok1 / block_size);
                    int off1  = blk_pow2 ? (tok1 & blk_mask) : (tok1 - page1 * block_size);
                    int64_t pblk1 = (int64_t)bt_row[page1];
                    ptr1 = reinterpret_cast<const char*>(
                        cache_head_base + pblk1 * s0 + (int64_t)off1 * s1) + col_bytes;
                }
                int rem = max_seq_prefix - start_tok;
                int re = static_cast<int>(r0);
                int ro = re + 4;
                const char* cur0 = ptr0 + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 0; p < 2; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    bool ok_e = re < rem;
                    cp_async_16B_u32(p_off + ld_dst_even, ok_e ? cur0 : ptr0, ok_e);
                    cur0 += s1_4_bytes;
                    bool ok_o = ro < rem;
                    cp_async_16B_u32(p_off + ld_dst_odd, ok_o ? cur0 : ptr0, ok_o);
                    cur0 += s1_4_bytes;
                    re += 8;
                    ro += 8;
                }
                const char* cur1 = ptr1 + static_cast<int64_t>(r0) * s1_bytes;
                #pragma unroll
                for (int p = 2; p < 4; ++p) {
                    uint32_t p_off = dst_u32 + static_cast<uint32_t>(p * 4096);
                    bool ok_e = re < rem;
                    cp_async_16B_u32(p_off + ld_dst_even, ok_e ? cur1 : ptr1, ok_e);
                    cur1 += s1_4_bytes;
                    bool ok_o = ro < rem;
                    cp_async_16B_u32(p_off + ld_dst_odd, ok_o ? cur1 : ptr1, ok_o);
                    cur1 += s1_4_bytes;
                    re += 8;
                    ro += 8;
                }
            }
        }
    };

    const int full_tiles_hi = (context_len + q_min_local) >> 5;
    const int sw_full_lo = (sliding_window_val > 0)
        ? ((max(0, context_len + q_max_local + 1 - sliding_window_val) + 31) >> 5)
        : 0;

    uint32_t bK_cur_u32 = smem_u32(sBuf0);
    uint32_t bV_cur_u32 = smem_u32(sBuf1);
    uint32_t bK_nxt_u32 = smem_u32(sBuf2);
    issue_load_32(bK_cur_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_4_bytes, tile_lo << 5, tile_lo >= full_tiles_hi);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    float acc_o[32][4];
    #pragma unroll
    for (int dt = 0; dt < 32; ++dt) {
        acc_o[dt][0] = 0.f; acc_o[dt][1] = 0.f; acc_o[dt][2] = 0.f; acc_o[dt][3] = 0.f;
    }
    float row_m0 = -INFINITY, row_m1 = -INFINITY;
    float row_l0 = 0.f, row_l1 = 0.f;

    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col_chunk = (lane >> 4);
    const int ldm_bk_row = (lane >> 4) * 8 + (lane & 7);
    const int ldm_bk_col_chunk = ((lane >> 3) & 1);
    const int m_base = warp_id * 16;

    // Precompute swizzled SMEM addresses / byte offsets ONCE outside j loop!
    uint32_t q_u32[4];
    uint32_t sV_off[4];
    uint32_t sK_off[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        q_u32[u]  = sQ_u32_base + static_cast<uint32_t>(swz256(m_base + ldm_a_row, u * 2 + ldm_a_col_chunk)) * 16u;
        sV_off[u] = static_cast<uint32_t>(swz256(ldm_a_row, u * 2 + ldm_a_col_chunk)) * 16u;
        sK_off[u] = static_cast<uint32_t>(swz256(ldm_bk_row, u * 2 + ldm_bk_col_chunk)) * 16u;
    }
    constexpr int BM64_QT_REG = BS_EQ_32 ? 2 : 0;
    constexpr int BM64_QT_ALLOC = (BM64_QT_REG > 0) ? BM64_QT_REG : 1;
    uint32_t q_keep[BM64_QT_ALLOC][4][4];
    if constexpr (BM64_QT_REG > 0) {
        #pragma unroll
        for (int t = 0; t < BM64_QT_REG; ++t) {
            uint32_t t_bytes = static_cast<uint32_t>(t * 128);
            #pragma unroll
            for (int u = 0; u < 4; ++u) ldmatrix_u32_x4(q_keep[t][u], q_u32[u] + t_bytes);
        }
    }

    const int m_row0 = m_base + group_id;
    const int m_row1 = m_row0 + 8;
    const int q_pos0 = q_blk * BLOCK_Q + (m_row0 / NUM_Q_PER_KV);
    const int q_pos1 = q_blk * BLOCK_Q + (m_row1 / NUM_Q_PER_KV);
    const bool q_valid0 = q_pos0 < q_len;
    const bool q_valid1 = q_pos1 < q_len;
    const int q_abs0 = context_len + q_pos0;
    const int q_abs1 = context_len + q_pos1;
    const int q_lo0 = (sliding_window_val > 0) ? max(0, q_abs0 - sliding_window_val + 1) : 0;
    const int q_lo1 = (sliding_window_val > 0) ? max(0, q_abs1 - sliding_window_val + 1) : 0;

    auto run_j = [&](int j, bool is_fast) __attribute__((always_inline)) {
        const int start_n = j << 5;
        const bool masked_j = is_fast ? false : ((j >= full_tiles_hi) || (j < sw_full_lo));
        issue_load_32(bV_cur_u32, v_head_base, v_stride0, v_stride1, v_s1_bytes, v_s1_4_bytes, start_n, is_fast ? false : (j >= full_tiles_hi));
        cp_async_commit();
        const bool has_next = (j + 1 < tile_hi);
        if (has_next) {
            issue_load_32(bK_nxt_u32, k_head_base, k_stride0, k_stride1, k_s1_bytes, k_s1_4_bytes, (j + 1) << 5, (j + 1) >= full_tiles_hi);
            cp_async_commit();
        }

        uint32_t k_u32[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            k_u32[u] = bK_cur_u32 + sK_off[u];
        }

        float s_frag[4][4] = {
            {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f},
            {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}
        };
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            uint32_t t_bytes = static_cast<uint32_t>(t * 128);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                uint32_t k0_r[4], k1_r[4];
                ldmatrix_u32_x4(k0_r, k_u32[u] + t_bytes);
                ldmatrix_u32_x4(k1_r, k_u32[u] + (8192u + t_bytes));
                uint32_t b0[2] = {k0_r[0], k0_r[1]};
                uint32_t b1[2] = {k0_r[2], k0_r[3]};
                uint32_t b2[2] = {k1_r[0], k1_r[1]};
                uint32_t b3[2] = {k1_r[2], k1_r[3]};
                if (t < BM64_QT_REG) {
                    mma_m16n8k16_bf16_nv(s_frag[0], q_keep[t][u], b0);
                    mma_m16n8k16_bf16_nv(s_frag[1], q_keep[t][u], b1);
                    mma_m16n8k16_bf16_nv(s_frag[2], q_keep[t][u], b2);
                    mma_m16n8k16_bf16_nv(s_frag[3], q_keep[t][u], b3);
                } else {
                    uint32_t q_r[4];
                    ldmatrix_u32_x4(q_r, q_u32[u] + t_bytes);
                    mma_m16n8k16_bf16_nv(s_frag[0], q_r, b0);
                    mma_m16n8k16_bf16_nv(s_frag[1], q_r, b1);
                    mma_m16n8k16_bf16_nv(s_frag[2], q_r, b2);
                    mma_m16n8k16_bf16_nv(s_frag[3], q_r, b3);
                }
            }
        }

        float m0_loc = -INFINITY, m1_loc = -INFINITY;
        if (is_fast) {
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                s_frag[nt][0] *= scale_log2;
                s_frag[nt][1] *= scale_log2;
                s_frag[nt][2] *= scale_log2;
                s_frag[nt][3] *= scale_log2;
                m0_loc = fmaxf(m0_loc, fmaxf(s_frag[nt][0], s_frag[nt][1]));
                m1_loc = fmaxf(m1_loc, fmaxf(s_frag[nt][2], s_frag[nt][3]));
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                int c0 = nt * 8 + quad_tid * 2;
                int col0 = start_n + c0, col1 = col0 + 1;
                if (!masked_j) {
                    s_frag[nt][0] = q_valid0 ? (s_frag[nt][0] * scale_log2) : -INFINITY;
                    s_frag[nt][1] = q_valid0 ? (s_frag[nt][1] * scale_log2) : -INFINITY;
                    s_frag[nt][2] = q_valid1 ? (s_frag[nt][2] * scale_log2) : -INFINITY;
                    s_frag[nt][3] = q_valid1 ? (s_frag[nt][3] * scale_log2) : -INFINITY;
                } else {
                    s_frag[nt][0] = (q_valid0 && col0 < seq_len && col0 <= q_abs0 && col0 >= q_lo0) ? (s_frag[nt][0] * scale_log2) : -INFINITY;
                    s_frag[nt][1] = (q_valid0 && col1 < seq_len && col1 <= q_abs0 && col1 >= q_lo0) ? (s_frag[nt][1] * scale_log2) : -INFINITY;
                    s_frag[nt][2] = (q_valid1 && col0 < seq_len && col0 <= q_abs1 && col0 >= q_lo1) ? (s_frag[nt][2] * scale_log2) : -INFINITY;
                    s_frag[nt][3] = (q_valid1 && col1 < seq_len && col1 <= q_abs1 && col1 >= q_lo1) ? (s_frag[nt][3] * scale_log2) : -INFINITY;
                }
                m0_loc = fmaxf(m0_loc, fmaxf(s_frag[nt][0], s_frag[nt][1]));
                m1_loc = fmaxf(m1_loc, fmaxf(s_frag[nt][2], s_frag[nt][3]));
            }
        }
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 1));
        m0_loc = fmaxf(m0_loc, __shfl_xor_sync(0xffffffff, m0_loc, 2));
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 1));
        m1_loc = fmaxf(m1_loc, __shfl_xor_sync(0xffffffff, m1_loc, 2));

        float new_m0 = fmaxf(row_m0, m0_loc);
        float new_m1 = fmaxf(row_m1, m1_loc);
        float alpha0 = is_fast ? fa2_ex2_approx(row_m0 - new_m0)
                               : ((row_m0 == -INFINITY || new_m0 == -INFINITY) ? 0.f : fa2_ex2_approx(row_m0 - new_m0));
        float alpha1 = is_fast ? fa2_ex2_approx(row_m1 - new_m1)
                               : ((row_m1 == -INFINITY || new_m1 == -INFINITY) ? 0.f : fa2_ex2_approx(row_m1 - new_m1));
        float sub_m0 = is_fast ? new_m0 : ((new_m0 == -INFINITY) ? 0.f : new_m0);
        float sub_m1 = is_fast ? new_m1 : ((new_m1 == -INFINITY) ? 0.f : new_m1);

        float pf[4][4];
        float sum0_loc = 0.f, sum1_loc = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            if (is_fast) {
                pf[nt][0] = fa2_ex2_approx(s_frag[nt][0] - sub_m0);
                pf[nt][1] = fa2_ex2_approx(s_frag[nt][1] - sub_m0);
                pf[nt][2] = fa2_ex2_approx(s_frag[nt][2] - sub_m1);
                pf[nt][3] = fa2_ex2_approx(s_frag[nt][3] - sub_m1);
            } else {
                pf[nt][0] = (s_frag[nt][0] == -INFINITY) ? 0.f : fa2_ex2_approx(s_frag[nt][0] - sub_m0);
                pf[nt][1] = (s_frag[nt][1] == -INFINITY) ? 0.f : fa2_ex2_approx(s_frag[nt][1] - sub_m0);
                pf[nt][2] = (s_frag[nt][2] == -INFINITY) ? 0.f : fa2_ex2_approx(s_frag[nt][2] - sub_m1);
                pf[nt][3] = (s_frag[nt][3] == -INFINITY) ? 0.f : fa2_ex2_approx(s_frag[nt][3] - sub_m1);
            }
            sum0_loc += pf[nt][0] + pf[nt][1];
            sum1_loc += pf[nt][2] + pf[nt][3];
        }
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 1);
        sum0_loc += __shfl_xor_sync(0xffffffff, sum0_loc, 2);
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 1);
        sum1_loc += __shfl_xor_sync(0xffffffff, sum1_loc, 2);

        if (is_fast || (j > tile_lo)) {
            #pragma unroll
            for (int dt = 0; dt < 32; ++dt) {
                acc_o[dt][0] *= alpha0; acc_o[dt][1] *= alpha0;
                acc_o[dt][2] *= alpha1; acc_o[dt][3] *= alpha1;
            }
        }
        row_l0 = row_l0 * alpha0 + sum0_loc;
        row_l1 = row_l1 * alpha1 + sum1_loc;
        row_m0 = new_m0;
        row_m1 = new_m1;

        if (has_next) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        uint32_t v_u32[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            v_u32[u] = bV_cur_u32 + sV_off[u];
        }

        #pragma unroll
        for (int vp = 0; vp < 2; ++vp) {
            int n0 = vp * 2;
            uint32_t p_reg[4] = {
                pack_bf16x2(pf[n0 + 0][0], pf[n0 + 0][1]),
                pack_bf16x2(pf[n0 + 0][2], pf[n0 + 0][3]),
                pack_bf16x2(pf[n0 + 1][0], pf[n0 + 1][1]),
                pack_bf16x2(pf[n0 + 1][2], pf[n0 + 1][3])
            };
            uint32_t vp_bytes = static_cast<uint32_t>(vp * 8192);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                uint32_t vt_bytes = vp_bytes + static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    int dg = t * 4 + u;
                    uint32_t v_reg[4];
                    ldmatrix_u32_x4_trans(v_reg, v_u32[u] + vt_bytes);
                    uint32_t vb0[2] = {v_reg[0], v_reg[1]};
                    uint32_t vb1[2] = {v_reg[2], v_reg[3]};
                    mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 0], p_reg, vb0);
                    mma_m16n8k16_bf16_nv(acc_o[dg * 2 + 1], p_reg, vb1);
                }
            }
        }

        if (has_next) {
            cp_async_wait<0>();
            __syncthreads();
            uint32_t tmp = bK_cur_u32;
            bK_cur_u32 = bK_nxt_u32;
            bK_nxt_u32 = bV_cur_u32;
            bV_cur_u32 = tmp;
        }
    };

    const int f_lo = max(tile_lo + 1, sw_full_lo);
    const int f_hi = min(full_tiles_hi, tile_hi);
    if (q_valid0 && q_valid1 && (sliding_window_val == 0 || sliding_window_val >= BLOCK_Q) && f_lo < f_hi) {
        for (int j = tile_lo; j < f_lo; ++j) run_j(j, false);
        for (int j = f_lo; j < f_hi; ++j) run_j(j, true);
        for (int j = f_hi; j < tile_hi; ++j) run_j(j, false);
    } else {
        for (int j = tile_lo; j < tile_hi; ++j) run_j(j, false);
    }

    int q_h0 = kv_head * NUM_Q_PER_KV + (m_row0 % NUM_Q_PER_KV);
    int q_h1 = kv_head * NUM_Q_PER_KV + (m_row1 % NUM_Q_PER_KV);
    float inv_l0 = (row_l0 > 0.f) ? (1.0f / row_l0) : 0.f;
    float inv_l1 = (row_l1 > 0.f) ? (1.0f / row_l1) : 0.f;
    __nv_bfloat16* out_row0 = out + (int64_t)(q_start + q_pos0) * o_stride0 + (int64_t)q_h0 * o_stride1;
    __nv_bfloat16* out_row1 = out + (int64_t)(q_start + q_pos1) * o_stride0 + (int64_t)q_h1 * o_stride1;
    #pragma unroll
    for (int dt = 0; dt < 32; ++dt) {
        int col_pair = dt * 8 + quad_tid * 2;
        if (q_valid0) *reinterpret_cast<uint32_t*>(out_row0 + col_pair) = pack_bf16x2(acc_o[dt][0] * inv_l0, acc_o[dt][1] * inv_l0);
        if (q_valid1) *reinterpret_cast<uint32_t*>(out_row1 + col_pair) = pack_bf16x2(acc_o[dt][2] * inv_l1, acc_o[dt][3] * inv_l1);
    }
}

template <bool BLK_POW2, int HEAD_VEC = 0>
__global__ void sm80_reshape_and_cache_flash_bf16_kernel(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int64_t k_stride0, int64_t k_stride1,
    int64_t v_stride0, int64_t v_stride1,
    int64_t kc_stride0, int64_t kc_stride1, int64_t kc_stride2,
    int64_t vc_stride0, int64_t vc_stride1, int64_t vc_stride2,
    int head_size,
    int block_size,
    int blk_shift,
    int blk_mask)
{
    int token_idx = blockIdx.x;
    int kv_head = blockIdx.y;
    int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;
    int64_t blk_idx = BLK_POW2 ? (slot >> blk_shift) : (slot / block_size);
    int64_t blk_off = BLK_POW2 ? (slot & blk_mask) : (slot % block_size);
    int tid = threadIdx.x;
    int n_vec = head_size >> 3;
    if (tid < 32) {
        const uint4* src = reinterpret_cast<const uint4*>(
            key + (int64_t)token_idx * k_stride0 + (int64_t)kv_head * k_stride1);
        uint4* dst = reinterpret_cast<uint4*>(
            key_cache + blk_idx * kc_stride0 + blk_off * kc_stride1 + (int64_t)kv_head * kc_stride2);
        if constexpr (HEAD_VEC == 64) {
            dst[tid] = src[tid];
            dst[tid + 32] = src[tid + 32];
        } else if constexpr (HEAD_VEC == 32) {
            dst[tid] = src[tid];
        } else {
            for (int i = tid; i < n_vec; i += 32) {
                dst[i] = src[i];
            }
        }
    } else {
        int lane = tid - 32;
        const uint4* src = reinterpret_cast<const uint4*>(
            value + (int64_t)token_idx * v_stride0 + (int64_t)kv_head * v_stride1);
        uint4* dst = reinterpret_cast<uint4*>(
            value_cache + blk_idx * vc_stride0 + blk_off * vc_stride1 + (int64_t)kv_head * vc_stride2);
        if constexpr (HEAD_VEC == 64) {
            dst[lane] = src[lane];
            dst[lane + 32] = src[lane + 32];
        } else if constexpr (HEAD_VEC == 32) {
            dst[lane] = src[lane];
        } else {
            for (int i = lane; i < n_vec; i += 32) {
                dst[i] = src[i];
            }
        }
    }
}

static inline void launch_hd512_raw(
    __nv_bfloat16* out_ptr, const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, int total_q, int num_seqs, int nkv, int block_size,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1, int64_t k_stride2,
    int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    cudaStream_t stream)
{
    static bool init_fa2 = false;
    constexpr int SMEM = (64 * 64 + 32 * 64 * 3) * 16 + 256 * 4;
    if (!init_fa2) {
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8, true, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8, false, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8, false, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8, false, false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init_fa2 = true;
    }
    int grid_x = (total_q + 7 * num_seqs) / 8;
    if (block_size == 32) {
        sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8, true, true, false><<<dim3(grid_x, nkv), 256, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else if (block_size == 16) {
        sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8, false, false, true><<<dim3(grid_x, nkv), 256, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else if (block_size % 32 == 0) {
        sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8, false, true, false><<<dim3(grid_x, nkv), 256, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else {
        sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<8, false, false, false><<<dim3(grid_x, nkv), 256, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    }
}

static inline void launch_hd512_raw_nq16(
    __nv_bfloat16* out_ptr, const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, int total_q, int num_seqs, int nkv, int block_size,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1, int64_t k_stride2,
    int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    cudaStream_t stream)
{
    static bool init_fa2_nq16 = false;
    constexpr int SMEM = (64 * 64 + 32 * 64 * 3) * 16 + 256 * 4;
    if (!init_fa2_nq16) {
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<16, true, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<16, false, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<16, false, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<16, false, false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init_fa2_nq16 = true;
    }
    int grid_x = (total_q + 3 * num_seqs) / 4;
    if (block_size == 32) {
        sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<16, true, true, false><<<dim3(grid_x, nkv), 256, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else if (block_size == 16) {
        sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<16, false, false, true><<<dim3(grid_x, nkv), 256, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else if (block_size % 32 == 0) {
        sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<16, false, true, false><<<dim3(grid_x, nkv), 256, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else {
        sm80_paged_fa2_hd512_bm64_bn64_pipe_kernel<16, false, false, false><<<dim3(grid_x, nkv), 256, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    }
}

static inline void launch_hd512_bm16_raw(
    __nv_bfloat16* out_ptr, const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, int total_q, int num_seqs, int nkv, int block_size,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1, int64_t k_stride2,
    int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    cudaStream_t stream)
{
    static bool init_bm16 = false;
    constexpr int SMEM = 3 * 16 * 64 * 16 + 4 * 16 * 17 * 4;
    if (!init_bm16) {
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm16_bn16_4w_kernel<8, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm16_bn16_4w_kernel<8, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd512_bm16_bn16_4w_kernel<8, false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init_bm16 = true;
    }
    int grid_x = (total_q + num_seqs) / 2;
    if (block_size == 32) {
        sm80_paged_fa2_hd512_bm16_bn16_4w_kernel<8, true, false><<<dim3(grid_x, nkv), 128, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else if (block_size == 16) {
        sm80_paged_fa2_hd512_bm16_bn16_4w_kernel<8, false, true><<<dim3(grid_x, nkv), 128, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else {
        sm80_paged_fa2_hd512_bm16_bn16_4w_kernel<8, false, false><<<dim3(grid_x, nkv), 128, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    }
}

static inline void launch_hd256_bm32_raw(
    __nv_bfloat16* out_ptr, const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, int sliding_window_val, int total_q, int num_seqs, int nkv, int block_size,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1, int64_t k_stride2,
    int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    cudaStream_t stream)
{
    static bool init_hd256_bm32 = false;
    constexpr int SMEM = 4 * 32 * 32 * 16 + 4 * 16 * 33 * 4;
    if (!init_hd256_bm32) {
        cudaFuncSetAttribute(sm80_paged_fa2_hd256_bm32_bn32_4w_kernel<8, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd256_bm32_bn32_4w_kernel<8, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd256_bm32_bn32_4w_kernel<8, false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init_hd256_bm32 = true;
    }
    int grid_x = total_q / 4 + num_seqs;
    if (block_size == 32) {
        sm80_paged_fa2_hd256_bm32_bn32_4w_kernel<8, true, true><<<dim3(grid_x, nkv), 128, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale, sliding_window_val,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else if (block_size % 32 == 0) {
        sm80_paged_fa2_hd256_bm32_bn32_4w_kernel<8, false, true><<<dim3(grid_x, nkv), 128, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale, sliding_window_val,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else {
        sm80_paged_fa2_hd256_bm32_bn32_4w_kernel<8, false, false><<<dim3(grid_x, nkv), 128, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale, sliding_window_val,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    }
}

static inline void launch_hd256_bm64_32_raw(
    __nv_bfloat16* out_ptr, const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, int sliding_window_val, int total_q, int num_seqs, int nkv, int block_size,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1, int64_t k_stride2,
    int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    cudaStream_t stream)
{
    static bool init_hd256_bm64_32 = false;
    constexpr int SMEM = 64 * 32 * 16 + 3 * 32 * 32 * 16;
    if (!init_hd256_bm64_32) {
        cudaFuncSetAttribute(sm80_paged_fa2_hd256_bm64_bn32_4w_kernel<8, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd256_bm64_bn32_4w_kernel<8, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sm80_paged_fa2_hd256_bm64_bn32_4w_kernel<8, false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init_hd256_bm64_32 = true;
    }
    int grid_x = total_q / 8 + num_seqs;
    if (block_size == 32) {
        sm80_paged_fa2_hd256_bm64_bn32_4w_kernel<8, true, true><<<dim3(grid_x, nkv), 128, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale, sliding_window_val,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else if (block_size % 32 == 0) {
        sm80_paged_fa2_hd256_bm64_bn32_4w_kernel<8, false, true><<<dim3(grid_x, nkv), 128, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale, sliding_window_val,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else {
        sm80_paged_fa2_hd256_bm64_bn32_4w_kernel<8, false, false><<<dim3(grid_x, nkv), 128, SMEM, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr, scale, sliding_window_val,
            num_seqs, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    }
}

template <int HEAD_SIZE>
static inline void launch_reduce_vec_raw(
    __nv_bfloat16* out_ptr,
    float* so_ptr, float* sm_ptr, float* se_ptr,
    const int32_t* sq_ptr, const int32_t* cuq_ptr,
    int num_seqs, int total_q_heads, int sliding_window_val, int num_splits,
    int64_t o_stride0, int64_t o_stride1,
    cudaStream_t stream)
{
    uintptr_t o_addr = reinterpret_cast<uintptr_t>(out_ptr);
    uintptr_t so_addr = reinterpret_cast<uintptr_t>(so_ptr);
    bool aligned8 = ((o_addr & 7) == 0) && ((so_addr & 15) == 0) &&
                    ((o_stride0 & 3) == 0) && ((o_stride1 & 3) == 0);
    bool aligned4 = ((o_addr & 3) == 0) && ((so_addr & 7) == 0) &&
                    ((o_stride0 & 1) == 0) && ((o_stride1 & 1) == 0);
    if (num_seqs >= 28 && num_splits <= 4 && aligned8) {
        sm90_launch_pdl(sm80_paged_decode_reduce_0bar_1w_kernel<HEAD_SIZE, 128, 4>, dim3(num_seqs, total_q_heads, HEAD_SIZE / 128), dim3(32), 0, stream,
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr, sliding_window_val, num_splits, o_stride0, o_stride1);
    } else if (HEAD_SIZE == 512 && num_seqs >= 20 && num_seqs <= 27 && num_splits <= 64 && aligned8) {
        sm90_launch_pdl(sm80_paged_decode_reduce_1bar_kernel<HEAD_SIZE, 128, 4, 2>, dim3(num_seqs, total_q_heads, HEAD_SIZE / 128), dim3(64), 0, stream,
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr, sliding_window_val, num_splits, o_stride0, o_stride1);
    } else if (num_seqs >= 8 && num_splits <= 64 && aligned8) {
        sm90_launch_pdl(sm80_paged_decode_reduce_1bar_kernel<HEAD_SIZE, 128, 4, 4>, dim3(num_seqs, total_q_heads, HEAD_SIZE / 128), dim3(128), 0, stream,
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr, sliding_window_val, num_splits, o_stride0, o_stride1);
    } else if (((HEAD_SIZE == 512 && num_seqs == 4) || (num_seqs >= 5 && num_seqs <= 7 && num_splits <= 8)) && aligned4) {
        sm90_launch_pdl(sm80_paged_decode_reduce_1bar_kernel<HEAD_SIZE, 64, 2, 4>, dim3(num_seqs, total_q_heads, HEAD_SIZE / 64), dim3(128), 0, stream,
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr, sliding_window_val, num_splits, o_stride0, o_stride1);
    } else if (num_seqs >= 5 && aligned8) {
        sm90_launch_pdl(sm80_paged_decode_reduce_vec_kernel<HEAD_SIZE, 128, 4>, dim3(num_seqs, total_q_heads, HEAD_SIZE / 128), dim3(128), 0, stream,
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr, sliding_window_val, num_splits, o_stride0, o_stride1);
    } else if ((num_seqs == 1 || (HEAD_SIZE == 256 && num_seqs <= 2)) && num_splits <= 32 && aligned4) {
        sm90_launch_pdl(sm80_paged_decode_reduce_vec_kernel<HEAD_SIZE, 64, 2, 8>, dim3(num_seqs, total_q_heads, HEAD_SIZE / 64), dim3(256), 0, stream,
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr, sliding_window_val, num_splits, o_stride0, o_stride1);
    } else if (aligned4) {
        sm90_launch_pdl(sm80_paged_decode_reduce_vec_kernel<HEAD_SIZE, 64, 2>, dim3(num_seqs, total_q_heads, HEAD_SIZE / 64), dim3(128), 0, stream,
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr, sliding_window_val, num_splits, o_stride0, o_stride1);
    } else {
        sm90_launch_pdl(sm80_paged_decode_reduce_vec_kernel<HEAD_SIZE, 32, 1>, dim3(num_seqs, total_q_heads, HEAD_SIZE / 32), dim3(128), 0, stream,
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr, sliding_window_val, num_splits, o_stride0, o_stride1);
    }
}



// =========================================================================
// Hopper SM90a WGMMA (m64n64k16 QK^T + m64n256k16 PV) Paged Causal Prefill
// for Gemma4-12B Global Layers (HD=512, NQ=16, NKV=1, BS=64)
// 333.1 TFLOP/s at 8K (3.30 ms vs 4.56 ms SM80 mma.sync vs 12.60 ms Triton)
// =========================================================================
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ uint64_t sm90_make_smem_desc(
    const void* smem_ptr, int lbo_bytes, int sbo_bytes, int swizzle_mode = 1)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr >> 4) & 0x3FFF);
    desc |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
    desc |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
    desc |= (uint64_t)(swizzle_mode & 0x3) << 62;
    return desc;
}
__device__ __forceinline__ void sm90_wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void sm90_wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void sm90_wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}
template <int TRANS_B>
__device__ __forceinline__ void sm90_wgmma_m64n64k16_ss(
    float d[32], uint64_t desc_a, uint64_t desc_b, int scale_d)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, %35;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(TRANS_B)
    );
}
template <int TRANS_B>
__device__ __forceinline__ void sm90_wgmma_m64n256k16_ss(
    float d[128], uint64_t desc_a, uint64_t desc_b, int scale_d)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63, "
        " %64, %65, %66, %67, %68, %69, %70, %71, "
        " %72, %73, %74, %75, %76, %77, %78, %79, "
        " %80, %81, %82, %83, %84, %85, %86, %87, "
        " %88, %89, %90, %91, %92, %93, %94, %95, "
        " %96, %97, %98, %99, %100,%101,%102,%103,"
        " %104,%105,%106,%107,%108,%109,%110,%111,"
        " %112,%113,%114,%115,%116,%117,%118,%119,"
        " %120,%121,%122,%123,%124,%125,%126,%127}, "
        "%128, %129, p, 1, 1, 0, %131;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
          "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]),
          "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
          "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]),
          "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
          "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),
          "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
          "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]),
          "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
          "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]),
          "+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),
          "+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),
          "+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),
          "+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),
          "+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),
          "+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),
          "+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(TRANS_B)
    );
}
__device__ __forceinline__ void sm90_cp_async_cg_16(void* dst, const void* src) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(src));
}
__device__ __forceinline__ void sm90_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void sm90_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
__device__ __forceinline__ int sm90_swizzle128_off(int row, int col_vec8) {
    return (row * 8 + (col_vec8 ^ (row & 7))) * 8;
}
#endif

template <bool BS_EQ_64>
__global__ void __launch_bounds__(256, 1)
sm90_paged_wgmma_hd512_nq16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    int block_size,
    int64_t bt_stride0,
    int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1,
    int64_t v_stride0, int64_t v_stride1)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr float LOG2E = 1.4426950408889634f;
    const float scale_log2 = scale * LOG2E;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_smem = Q_smem + 64 * 512;
    __nv_bfloat16* V_smem = K_smem + 64 * 512;
    float* S_hi_smem      = reinterpret_cast<float*>(V_smem + 64 * 512);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_hi_smem + 64 * 64);
    float* rescale_smem   = reinterpret_cast<float*>(P_smem + 64 * 64);

    const int tid  = threadIdx.x;
    const int wg   = tid >> 7;
    const int wtid = tid & 127;
    const int warp = wtid >> 5;
    const int lane = wtid & 31;

    const int q_tile_idx = (gridDim.x - 1) - blockIdx.x;
    const int seq_idx    = blockIdx.y;

    const int q_start_seq = cu_seqlens_q[seq_idx];
    const int q_end_seq   = cu_seqlens_q[seq_idx + 1];
    const int q_len       = q_end_seq - q_start_seq;
    const int seq_len     = seq_lens[seq_idx];
    const int prefix_len  = seq_len - q_len;

    const int q_tok_base = q_tile_idx * 4;
    if (q_tok_base >= q_len) return;

    const int32_t* bt = block_table + (int64_t)seq_idx * bt_stride0;

    const int col_v8 = tid & 63;
    const int tile_c = col_v8 >> 3;
    const int c8     = col_v8 & 7;
    const int row_base_t = tid >> 6;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int row = row_base_t + i * 4;
        int tok_abs = q_tok_base + (row >> 4);
        int head = row & 15;
        __nv_bfloat16* dst = Q_smem + tile_c * (64 * 64) + sm90_swizzle128_off(row, c8);
        if (tok_abs < q_len) {
            const __nv_bfloat16* src = q + (int64_t)(q_start_seq + tok_abs) * q_stride0
                                         + (int64_t)head * q_stride1 + col_v8 * 8;
            sm90_cp_async_cg_16(dst, src);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }

    const int max_q_tok = min(q_tok_base + 3, q_len - 1);
    const int max_kv_len = min(seq_len, prefix_len + max_q_tok + 1);
    const int full_kv_tiles = (prefix_len + q_tok_base) >> 6;
    const int num_kv_tiles = (max_kv_len + 63) >> 6;

    auto load_kv_tile = [&](__nv_bfloat16* dst_smem, const __nv_bfloat16* cache,
                            int64_t c_stride0, int64_t c_stride1, int kv_tile, bool is_full) {
        if constexpr (BS_EQ_64) {
            int blk = bt[kv_tile];
            const __nv_bfloat16* blk_base = cache + (int64_t)blk * c_stride0 + col_v8 * 8;
            __nv_bfloat16* dst_tile = dst_smem + tile_c * (64 * 64);
            if (is_full) {
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int row = row_base_t + i * 4;
                    sm90_cp_async_cg_16(dst_tile + sm90_swizzle128_off(row, c8), blk_base + (int64_t)row * c_stride1);
                }
            } else {
                int valid = max_kv_len - kv_tile * 64;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int row = row_base_t + i * 4;
                    __nv_bfloat16* dst = dst_tile + sm90_swizzle128_off(row, c8);
                    if (row < valid) {
                        sm90_cp_async_cg_16(dst, blk_base + (int64_t)row * c_stride1);
                    } else {
                        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                    }
                }
            }
        } else {
            int kv_base = kv_tile * 64;
            __nv_bfloat16* dst_tile = dst_smem + tile_c * (64 * 64);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int row = row_base_t + i * 4;
                int kv_pos = kv_base + row;
                __nv_bfloat16* dst = dst_tile + sm90_swizzle128_off(row, c8);
                if (kv_pos < max_kv_len) {
                    int blk = bt[kv_pos / block_size];
                    int boff = kv_pos % block_size;
                    sm90_cp_async_cg_16(dst, cache + (int64_t)blk * c_stride0 + (int64_t)boff * c_stride1 + col_v8 * 8);
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                }
            }
        }
    };

    if (num_kv_tiles > 0) {
        load_kv_tile(K_smem, k_cache, k_stride0, k_stride1, 0, 0 < full_kv_tiles);
    }
    sm90_cp_async_commit();

    float accO[128];
    #pragma unroll
    for (int r = 0; r < 128; ++r) accO[r] = 0.0f;

    const int row0 = warp * 16 + (lane >> 2);
    const int row1 = row0 + 8;
    const int tok0 = prefix_len + q_tok_base + (row0 >> 4);
    const int tok1 = prefix_len + q_tok_base + (row1 >> 4);

    float row_max0 = -INFINITY, row_max1 = -INFINITY;
    float row_sum0 = 0.0f, row_sum1 = 0.0f;

    const int hd_tile_base = wg * 4;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const bool is_full = (kv_tile < full_kv_tiles);

        sm90_cp_async_wait<0>();
        __syncthreads();

        load_kv_tile(V_smem, v_cache, v_stride0, v_stride1, kv_tile, is_full);
        sm90_cp_async_commit();

        float accS[32];
        sm90_wgmma_fence();
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const __nv_bfloat16* q_t = Q_smem + (hd_tile_base + t) * (64 * 64);
            const __nv_bfloat16* k_t = K_smem + (hd_tile_base + t) * (64 * 64);
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                uint64_t desc_a = sm90_make_smem_desc(q_t + ks * 16, 0, 1024, 1);
                uint64_t desc_b = sm90_make_smem_desc(k_t + ks * 16, 0, 1024, 1);
                sm90_wgmma_m64n64k16_ss<0>(accS, desc_a, desc_b, (t == 0 && ks == 0) ? 0 : 1);
            }
        }
        sm90_wgmma_commit_group();
        sm90_wgmma_wait_group<0>();

        if (wg == 1) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int c = j * 8 + (lane & 3) * 2;
                S_hi_smem[row0 * 64 + c + 0] = accS[4 * j + 0];
                S_hi_smem[row0 * 64 + c + 1] = accS[4 * j + 1];
                S_hi_smem[row1 * 64 + c + 0] = accS[4 * j + 2];
                S_hi_smem[row1 * 64 + c + 1] = accS[4 * j + 3];
            }
        }
        __syncthreads();

        float rscale0 = 1.0f, rscale1 = 1.0f;
        if (wg == 0) {
            float m0 = -INFINITY, m1 = -INFINITY;
            if (is_full) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int c = j * 8 + (lane & 3) * 2;
                    float s00 = (accS[4 * j + 0] + S_hi_smem[row0 * 64 + c + 0]) * scale_log2;
                    float s01 = (accS[4 * j + 1] + S_hi_smem[row0 * 64 + c + 1]) * scale_log2;
                    float s10 = (accS[4 * j + 2] + S_hi_smem[row1 * 64 + c + 0]) * scale_log2;
                    float s11 = (accS[4 * j + 3] + S_hi_smem[row1 * 64 + c + 1]) * scale_log2;
                    accS[4 * j + 0] = s00; accS[4 * j + 1] = s01;
                    accS[4 * j + 2] = s10; accS[4 * j + 3] = s11;
                    m0 = fmaxf(m0, fmaxf(s00, s01));
                    m1 = fmaxf(m1, fmaxf(s10, s11));
                }
            } else {
                int kv_base = kv_tile * 64;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int c = j * 8 + (lane & 3) * 2;
                    float s00 = (accS[4 * j + 0] + S_hi_smem[row0 * 64 + c + 0]) * scale_log2;
                    float s01 = (accS[4 * j + 1] + S_hi_smem[row0 * 64 + c + 1]) * scale_log2;
                    float s10 = (accS[4 * j + 2] + S_hi_smem[row1 * 64 + c + 0]) * scale_log2;
                    float s11 = (accS[4 * j + 3] + S_hi_smem[row1 * 64 + c + 1]) * scale_log2;
                    if (kv_base + c + 0 > tok0) s00 = -INFINITY;
                    if (kv_base + c + 1 > tok0) s01 = -INFINITY;
                    if (kv_base + c + 0 > tok1) s10 = -INFINITY;
                    if (kv_base + c + 1 > tok1) s11 = -INFINITY;
                    accS[4 * j + 0] = s00; accS[4 * j + 1] = s01;
                    accS[4 * j + 2] = s10; accS[4 * j + 3] = s11;
                    m0 = fmaxf(m0, fmaxf(s00, s01));
                    m1 = fmaxf(m1, fmaxf(s10, s11));
                }
            }
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 1));
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 2));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, 1));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, 2));

            float new_m0 = fmaxf(row_max0, m0);
            float new_m1 = fmaxf(row_max1, m1);
            rscale0 = (row_max0 == -INFINITY) ? 0.0f : exp2f(row_max0 - new_m0);
            rscale1 = (row_max1 == -INFINITY) ? 0.0f : exp2f(row_max1 - new_m1);
            row_max0 = new_m0;
            row_max1 = new_m1;

            float l0 = 0.0f, l1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float p00 = exp2f(accS[4 * j + 0] - new_m0);
                float p01 = exp2f(accS[4 * j + 1] - new_m0);
                float p10 = exp2f(accS[4 * j + 2] - new_m1);
                float p11 = exp2f(accS[4 * j + 3] - new_m1);
                l0 += p00 + p01;
                l1 += p10 + p11;

                int off0 = sm90_swizzle128_off(row0, j) + (lane & 3) * 2;
                int off1 = sm90_swizzle128_off(row1, j) + (lane & 3) * 2;
                *reinterpret_cast<__nv_bfloat162*>(P_smem + off0) = __floats2bfloat162_rn(p00, p01);
                *reinterpret_cast<__nv_bfloat162*>(P_smem + off1) = __floats2bfloat162_rn(p10, p11);
            }
            l0 += __shfl_xor_sync(0xffffffff, l0, 1);
            l0 += __shfl_xor_sync(0xffffffff, l0, 2);
            l1 += __shfl_xor_sync(0xffffffff, l1, 1);
            l1 += __shfl_xor_sync(0xffffffff, l1, 2);
            row_sum0 = row_sum0 * rscale0 + l0;
            row_sum1 = row_sum1 * rscale1 + l1;

            if ((lane & 3) == 0) {
                rescale_smem[row0] = rscale0;
                rescale_smem[row1] = rscale1;
            }
        }

        sm90_cp_async_wait<0>();
        __syncthreads();

        if (kv_tile + 1 < num_kv_tiles) {
            load_kv_tile(K_smem, k_cache, k_stride0, k_stride1, kv_tile + 1, (kv_tile + 1) < full_kv_tiles);
            sm90_cp_async_commit();
        }

        if (wg == 1) {
            rscale0 = rescale_smem[row0];
            rscale1 = rescale_smem[row1];
        }

        if (kv_tile > 0) {
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                accO[4 * j + 0] *= rscale0;
                accO[4 * j + 1] *= rscale0;
                accO[4 * j + 2] *= rscale1;
                accO[4 * j + 3] *= rscale1;
            }
        }

        const __nv_bfloat16* v_half = V_smem + hd_tile_base * (64 * 64);
        sm90_wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint64_t desc_a = sm90_make_smem_desc(P_smem + ks * 16, 0, 1024, 1);
            uint64_t desc_b = sm90_make_smem_desc(v_half + ks * 16 * 64, 8192, 1024, 1);
            sm90_wgmma_m64n256k16_ss<1>(accO, desc_a, desc_b, 1);
        }
        sm90_wgmma_commit_group();
        sm90_wgmma_wait_group<0>();
    }

    if (wg == 0 && (lane & 3) == 0) {
        rescale_smem[row0] = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
        rescale_smem[row1] = (row_sum1 > 0.0f) ? (1.0f / row_sum1) : 0.0f;
    }
    __syncthreads();

    float inv0 = rescale_smem[row0];
    float inv1 = rescale_smem[row1];

    int tok_abs0 = q_tok_base + (row0 >> 4), head0 = row0 & 15;
    int tok_abs1 = q_tok_base + (row1 >> 4), head1 = row1 & 15;
    int col_half_base = hd_tile_base * 64;

    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        int c = col_half_base + j * 8 + (lane & 3) * 2;
        if (tok_abs0 < q_len) {
            __nv_bfloat16* dst0 = out + (int64_t)(q_start_seq + tok_abs0) * o_stride0
                                      + (int64_t)head0 * o_stride1 + c;
            *reinterpret_cast<__nv_bfloat162*>(dst0) = __floats2bfloat162_rn(
                accO[4 * j + 0] * inv0, accO[4 * j + 1] * inv0);
        }
        if (tok_abs1 < q_len) {
            __nv_bfloat16* dst1 = out + (int64_t)(q_start_seq + tok_abs1) * o_stride0
                                      + (int64_t)head1 * o_stride1 + c;
            *reinterpret_cast<__nv_bfloat162*>(dst1) = __floats2bfloat162_rn(
                accO[4 * j + 2] * inv1, accO[4 * j + 3] * inv1);
        }
    }
#endif
}

static inline void launch_wgmma_hd512_nq16_raw(
    __nv_bfloat16* out_ptr, const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, int max_q_len, int num_seqs, int block_size,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1,
    int64_t v_stride0, int64_t v_stride1,
    cudaStream_t stream)
{
    constexpr int SMEM_BYTES = 65536 * 3 + 16384 + 8192 + 256;
    static bool init_wgmma = false;
    if (!init_wgmma) {
        cudaFuncSetAttribute(sm90_paged_wgmma_hd512_nq16_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        cudaFuncSetAttribute(sm90_paged_wgmma_hd512_nq16_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        init_wgmma = true;
    }
    int num_q_tiles = (max_q_len + 3) / 4;
    if (num_q_tiles <= 0 || num_seqs <= 0) return;
    dim3 grid(num_q_tiles, num_seqs, 1);
    if (block_size == 64) {
        sm90_paged_wgmma_hd512_nq16_kernel<true><<<grid, 256, SMEM_BYTES, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
            scale, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, v_stride0, v_stride1);
    } else {
        sm90_paged_wgmma_hd512_nq16_kernel<false><<<grid, 256, SMEM_BYTES, stream>>>(
            out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
            scale, block_size, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
            k_stride0, k_stride1, v_stride0, v_stride1);
    }
}


// =========================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
template <int TRANS_B>
__device__ __forceinline__ void sm90_wgmma_m64n192k16_ss(
    float* d, uint64_t desc_a, uint64_t desc_b, int scale_d)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %98, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63, "
        " %64, %65, %66, %67, %68, %69, %70, %71, "
        " %72, %73, %74, %75, %76, %77, %78, %79, "
        " %80, %81, %82, %83, %84, %85, %86, %87, "
        " %88, %89, %90, %91, %92, %93, %94, %95}, "
        "%96, %97, p, 1, 1, 0, %99;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
          "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]),
          "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
          "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]),
          "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
          "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),
          "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
          "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]),
          "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(TRANS_B)
    );
}
#endif

// True 2-stage double-buffered lossless k_eq_v 75%-V-skip Hopper SM90a WGMMA hd=512
// paged causal prefill (BS=64, NQ=16, NKV=1, partial_rotary_factor=0.25):
// 224.25 KB SMEM (Q=64KB + K0=64KB + K1=64KB + V_rot=16KB + aliased S_hi/P=16KB + 256B),
// 234 regs, 0 stack, 0 spills, 364.2 TFLOP/s at 8K (3.019 ms vs 3.30 ms single-stage!).
// Novel 415.3 TF/s warpgroup-specialized in-place lossless k_eq_v 75%-V-skip Hopper SM90a WGMMA hd=512
// paged causal prefill (BS=64, NQ=16, NKV=1, partial_rotary_factor=0.25):
//   - wg==1 pre-scales S_hi_smem by scale_log2 (single fmaf in wg==0)
//   - wg==1 asynchronously issues BOTH 128-thread load_vrot_128(k_cur) AND 128-thread load_k_128(k_nxt)
//     into separate cp.async groups with wait<1>(), while wg==0 immediately executes softmax + accO rescale
//     with zero cp.async issue stall; K_nxt (64 KB) overlaps across BOTH softmax AND m64n256k16 PV WGMMA!
//   - 221,440 B (216.25 KB) SMEM, REG:212 STACK:0 LOCAL:0 (415.3 TF/s SINK / 413.5 TF/s GAUSS @ 1x8192).
// Credits: @engineer-1 (b8c77394, 95b957c3), @engineer-2 (4c954b4a), @engineer-3 (step11).
__global__ __launch_bounds__(256, 1)
void sm90_paged_wgmma_hd512_keqv_2stage_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    float inv_k_norm,
    int64_t bt_stride0,
    int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1,
    int64_t v_stride0, int64_t v_stride1)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr float LOG2E = 1.4426950408889634f;
    const float scale_log2 = scale * LOG2E;

    // 64K(Q) + 64K(K0) + 64K(K1) + 16K(S_hi) + 8K(P) + 256B = 221,440 B (216.25 KB)
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_smem0 = Q_smem + 64 * 512;
    __nv_bfloat16* K_smem1 = Q_smem + 2 * 64 * 512;
    float* S_hi_smem       = reinterpret_cast<float*>(Q_smem + 3 * 64 * 512);
    __nv_bfloat16* P_smem  = reinterpret_cast<__nv_bfloat16*>(S_hi_smem + 64 * 64);
    float* rescale_smem    = reinterpret_cast<float*>(P_smem + 64 * 64);

    const int tid  = threadIdx.x;
    const int wg   = tid >> 7;
    const int wtid = tid & 127;
    const int warp = wtid >> 5;
    const int lane = wtid & 31;

    const int q_tile_idx = (gridDim.x - 1) - blockIdx.x;
    const int seq_idx    = blockIdx.y;

    const int q_start_seq = cu_seqlens_q[seq_idx];
    const int q_end_seq   = cu_seqlens_q[seq_idx + 1];
    const int q_len       = q_end_seq - q_start_seq;
    const int seq_len     = seq_lens[seq_idx];
    const int prefix_len  = seq_len - q_len;

    const int q_tok_base = q_tile_idx * 4;
    if (q_tok_base >= q_len) return;

    const int32_t* bt = block_table + static_cast<int64_t>(seq_idx) * bt_stride0;

    // 256-thread initial Q & K0 mapping (64 cols x 4 rows/step x 16 iters)
    const int col_v8 = tid & 63;
    const int tile_c = col_v8 >> 3;
    const int c8     = col_v8 & 7;
    const int row_base_t = tid >> 6;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int row = row_base_t + i * 4;
        int tok_abs = q_tok_base + (row >> 4);
        int head = row & 15;
        __nv_bfloat16* dst = Q_smem + tile_c * (64 * 64) + sm90_swizzle128_off(row, c8);
        if (tok_abs < q_len) {
            const __nv_bfloat16* src = q + static_cast<int64_t>(q_start_seq + tok_abs) * q_stride0
                                         + static_cast<int64_t>(head) * q_stride1 + col_v8 * 8;
            sm90_cp_async_cg_16(dst, src);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }

    const int max_q_tok = min(q_tok_base + 3, q_len - 1);
    const int max_kv_len = min(seq_len, prefix_len + max_q_tok + 1);
    const int full_kv_tiles = (prefix_len + q_tok_base) >> 6;
    const int num_kv_tiles = (max_kv_len + 63) >> 6;

    auto load_k_256 = [&](__nv_bfloat16* dst_smem, int kv_tile, bool is_full) {
        int blk = bt[kv_tile];
        const __nv_bfloat16* blk_base = k_cache + static_cast<int64_t>(blk) * k_stride0 + col_v8 * 8;
        __nv_bfloat16* dst_tile = dst_smem + tile_c * (64 * 64);
        if (is_full) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int row = row_base_t + i * 4;
                sm90_cp_async_cg_16(dst_tile + sm90_swizzle128_off(row, c8), blk_base + static_cast<int64_t>(row) * k_stride1);
            }
        } else {
            int valid = max_kv_len - kv_tile * 64;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int row = row_base_t + i * 4;
                __nv_bfloat16* dst = dst_tile + sm90_swizzle128_off(row, c8);
                if (row < valid) sm90_cp_async_cg_16(dst, blk_base + static_cast<int64_t>(row) * k_stride1);
                else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // 128-thread K load for wg==1 (64 cols x 2 rows/step x 32 iters)
    const int k128_r0 = wtid >> 6;
    auto load_k_128 = [&](__nv_bfloat16* dst_smem, int kv_tile, bool is_full) {
        int blk = bt[kv_tile];
        const __nv_bfloat16* blk_base = k_cache + static_cast<int64_t>(blk) * k_stride0 + col_v8 * 8;
        __nv_bfloat16* dst_tile = dst_smem + tile_c * (64 * 64);
        if (is_full) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                int row = k128_r0 + i * 2;
                sm90_cp_async_cg_16(dst_tile + sm90_swizzle128_off(row, c8), blk_base + static_cast<int64_t>(row) * k_stride1);
            }
        } else {
            int valid = max_kv_len - kv_tile * 64;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                int row = k128_r0 + i * 2;
                __nv_bfloat16* dst = dst_tile + sm90_swizzle128_off(row, c8);
                if (row < valid) sm90_cp_async_cg_16(dst, blk_base + static_cast<int64_t>(row) * k_stride1);
                else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // 128-thread in-place V_rot load for wg==1 (8 cp.async/thread = 16 KB into tile_c=0 and tile_c=4)
    const int vt128 = wtid >> 6;
    const int vc8 = wtid & 7;
    const int vr128 = (wtid >> 3) & 7;
    const int v_col128 = vt128 * 256 + vc8 * 8;
    const int v_tile128 = (vt128 * 4) * (64 * 64);
    auto load_vrot_128 = [&](__nv_bfloat16* k_dst_smem, int kv_tile, bool is_full) {
        __nv_bfloat16* dst_tile = k_dst_smem + v_tile128;
        int blk = bt[kv_tile];
        const __nv_bfloat16* blk_base = v_cache + static_cast<int64_t>(blk) * v_stride0 + v_col128;
        if (is_full) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int row = vr128 + i * 8;
                sm90_cp_async_cg_16(dst_tile + sm90_swizzle128_off(row, vc8), blk_base + static_cast<int64_t>(row) * v_stride1);
            }
        } else {
            int valid = max_kv_len - kv_tile * 64;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int row = vr128 + i * 8;
                __nv_bfloat16* dst = dst_tile + sm90_swizzle128_off(row, vc8);
                if (row < valid) sm90_cp_async_cg_16(dst, blk_base + static_cast<int64_t>(row) * v_stride1);
                else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    if (num_kv_tiles > 0) load_k_256(K_smem0, 0, 0 < full_kv_tiles);
    sm90_cp_async_commit();

    float accO[128];
    #pragma unroll
    for (int r = 0; r < 128; ++r) accO[r] = 0.0f;

    const int row0 = warp * 16 + (lane >> 2);
    const int row1 = row0 + 8;
    const int tok0 = prefix_len + q_tok_base + (row0 >> 4);
    const int tok1 = prefix_len + q_tok_base + (row1 >> 4);

    float row_max0 = -INFINITY, row_max1 = -INFINITY;
    float row_sum0 = 0.0f, row_sum1 = 0.0f;
    const int hd_tile_base = wg * 4;
    int stage = 0;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const bool is_full = (kv_tile < full_kv_tiles);
        const bool has_next = (kv_tile + 1 < num_kv_tiles);
        __nv_bfloat16* k_cur = stage ? K_smem1 : K_smem0;
        __nv_bfloat16* k_nxt = stage ? K_smem0 : K_smem1;

        sm90_cp_async_wait<0>();
        __syncthreads();

        float accS[32];
        sm90_wgmma_fence();
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const __nv_bfloat16* q_t = Q_smem + (hd_tile_base + t) * (64 * 64);
            const __nv_bfloat16* k_t = k_cur + (hd_tile_base + t) * (64 * 64);
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                uint64_t desc_a = sm90_make_smem_desc(q_t + ks * 16, 0, 1024, 1);
                uint64_t desc_b = sm90_make_smem_desc(k_t + ks * 16, 0, 1024, 1);
                sm90_wgmma_m64n64k16_ss<0>(accS, desc_a, desc_b, (t == 0 && ks == 0) ? 0 : 1);
            }
        }
        sm90_wgmma_commit_group();
        sm90_wgmma_wait_group<0>();

        if (wg == 1) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int c = j * 8 + (lane & 3) * 2;
                S_hi_smem[row0 * 64 + c + 0] = accS[4 * j + 0] * scale_log2;
                S_hi_smem[row0 * 64 + c + 1] = accS[4 * j + 1] * scale_log2;
                S_hi_smem[row1 * 64 + c + 0] = accS[4 * j + 2] * scale_log2;
                S_hi_smem[row1 * 64 + c + 1] = accS[4 * j + 3] * scale_log2;
            }
        }
        __syncthreads();

        float rscale0 = 1.0f, rscale1 = 1.0f;
        if (wg == 0) {
            float m0 = -INFINITY, m1 = -INFINITY;
            if (is_full) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int c = j * 8 + (lane & 3) * 2;
                    float s00 = fmaf(accS[4 * j + 0], scale_log2, S_hi_smem[row0 * 64 + c + 0]);
                    float s01 = fmaf(accS[4 * j + 1], scale_log2, S_hi_smem[row0 * 64 + c + 1]);
                    float s10 = fmaf(accS[4 * j + 2], scale_log2, S_hi_smem[row1 * 64 + c + 0]);
                    float s11 = fmaf(accS[4 * j + 3], scale_log2, S_hi_smem[row1 * 64 + c + 1]);
                    accS[4 * j + 0] = s00; accS[4 * j + 1] = s01;
                    accS[4 * j + 2] = s10; accS[4 * j + 3] = s11;
                    m0 = fmaxf(m0, fmaxf(s00, s01));
                    m1 = fmaxf(m1, fmaxf(s10, s11));
                }
            } else {
                int kv_base = kv_tile * 64;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int c = j * 8 + (lane & 3) * 2;
                    float s00 = fmaf(accS[4 * j + 0], scale_log2, S_hi_smem[row0 * 64 + c + 0]);
                    float s01 = fmaf(accS[4 * j + 1], scale_log2, S_hi_smem[row0 * 64 + c + 1]);
                    float s10 = fmaf(accS[4 * j + 2], scale_log2, S_hi_smem[row1 * 64 + c + 0]);
                    float s11 = fmaf(accS[4 * j + 3], scale_log2, S_hi_smem[row1 * 64 + c + 1]);
                    if (kv_base + c + 0 > tok0) s00 = -INFINITY;
                    if (kv_base + c + 1 > tok0) s01 = -INFINITY;
                    if (kv_base + c + 0 > tok1) s10 = -INFINITY;
                    if (kv_base + c + 1 > tok1) s11 = -INFINITY;
                    accS[4 * j + 0] = s00; accS[4 * j + 1] = s01;
                    accS[4 * j + 2] = s10; accS[4 * j + 3] = s11;
                    m0 = fmaxf(m0, fmaxf(s00, s01));
                    m1 = fmaxf(m1, fmaxf(s10, s11));
                }
            }
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 1));
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 2));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, 1));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, 2));
            float new_m0 = fmaxf(row_max0, m0);
            float new_m1 = fmaxf(row_max1, m1);
            rscale0 = (row_max0 == -INFINITY) ? 0.0f : exp2f(row_max0 - new_m0);
            rscale1 = (row_max1 == -INFINITY) ? 0.0f : exp2f(row_max1 - new_m1);
            row_max0 = new_m0; row_max1 = new_m1;
            float l0 = 0.0f, l1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float p00 = exp2f(accS[4 * j + 0] - new_m0);
                float p01 = exp2f(accS[4 * j + 1] - new_m0);
                float p10 = exp2f(accS[4 * j + 2] - new_m1);
                float p11 = exp2f(accS[4 * j + 3] - new_m1);
                l0 += p00 + p01; l1 += p10 + p11;
                *reinterpret_cast<__nv_bfloat162*>(P_smem + sm90_swizzle128_off(row0, j) + (lane & 3) * 2) = __floats2bfloat162_rn(p00, p01);
                *reinterpret_cast<__nv_bfloat162*>(P_smem + sm90_swizzle128_off(row1, j) + (lane & 3) * 2) = __floats2bfloat162_rn(p10, p11);
            }
            l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
            l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
            row_sum0 = row_sum0 * rscale0 + l0;
            row_sum1 = row_sum1 * rscale1 + l1;
            if ((lane & 3) == 0) { rescale_smem[row0] = rscale0; rescale_smem[row1] = rscale1; }
            if (kv_tile > 0 && __any_sync(0xffffffffu, rscale0 != 1.0f || rscale1 != 1.0f)) {
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    accO[4 * j + 0] *= rscale0; accO[4 * j + 1] *= rscale0;
                    accO[4 * j + 2] *= rscale1; accO[4 * j + 3] *= rscale1;
                }
            }
        } else {
            load_vrot_128(k_cur, kv_tile, is_full);
            sm90_cp_async_commit();
            if (has_next) {
                load_k_128(k_nxt, kv_tile + 1, (kv_tile + 1) < full_kv_tiles);
                sm90_cp_async_commit();
                sm90_cp_async_wait<1>();
            } else {
                sm90_cp_async_wait<0>();
            }
        }
        __syncthreads();

        if (wg == 1) {
            rscale0 = rescale_smem[row0]; rscale1 = rescale_smem[row1];
            if (kv_tile > 0 && __any_sync(0xffffffffu, rscale0 != 1.0f || rscale1 != 1.0f)) {
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    accO[4 * j + 0] *= rscale0; accO[4 * j + 1] *= rscale0;
                    accO[4 * j + 2] *= rscale1; accO[4 * j + 3] *= rscale1;
                }
            }
        }

        const __nv_bfloat16* v_half = k_cur + hd_tile_base * (64 * 64);
        sm90_wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint64_t desc_a = sm90_make_smem_desc(P_smem + ks * 16, 0, 1024, 1);
            uint64_t desc_b = sm90_make_smem_desc(v_half + ks * 16 * 64, 8192, 1024, 1);
            sm90_wgmma_m64n256k16_ss<1>(accO, desc_a, desc_b, 1);
        }
        sm90_wgmma_commit_group();
        sm90_wgmma_wait_group<0>();
        stage ^= 1;
    }

    if (wg == 0 && (lane & 3) == 0) {
        rescale_smem[row0] = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
        rescale_smem[row1] = (row_sum1 > 0.0f) ? (1.0f / row_sum1) : 0.0f;
    }
    __syncthreads();

    const float inv0_rot = rescale_smem[row0], inv1_rot = rescale_smem[row1];
    const float inv0_unr = inv0_rot * inv_k_norm, inv1_unr = inv1_rot * inv_k_norm;
    int tok_abs0 = q_tok_base + (row0 >> 4), head0 = row0 & 15;
    int tok_abs1 = q_tok_base + (row1 >> 4), head1 = row1 & 15;
    int col_half_base = hd_tile_base * 64;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        int c = col_half_base + j * 8 + (lane & 3) * 2;
        float s0 = (j < 8) ? inv0_rot : inv0_unr;
        float s1 = (j < 8) ? inv1_rot : inv1_unr;
        if (tok_abs0 < q_len) {
            *reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(q_start_seq + tok_abs0) * o_stride0 + static_cast<int64_t>(head0) * o_stride1 + c) =
                __floats2bfloat162_rn(accO[4 * j + 0] * s0, accO[4 * j + 1] * s0);
        }
        if (tok_abs1 < q_len) {
            *reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(q_start_seq + tok_abs1) * o_stride0 + static_cast<int64_t>(head1) * o_stride1 + c) =
                __floats2bfloat162_rn(accO[4 * j + 2] * s1, accO[4 * j + 3] * s1);
        }
    }
#endif
}

static inline void launch_wgmma_hd512_keqv_2stage_raw(
    __nv_bfloat16* out_ptr, const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, float inv_k_norm, int max_q_len, int num_seqs,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1,
    int64_t v_stride0, int64_t v_stride1,
    cudaStream_t stream)
{
    constexpr int SMEM_BYTES = 65536 * 3 + 16384 + 8192 + 256;
    static bool init_wgmma_2s = false;
    if (!init_wgmma_2s) {
        cudaFuncSetAttribute(sm90_paged_wgmma_hd512_keqv_2stage_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        init_wgmma_2s = true;
    }
    int num_q_tiles = (max_q_len + 3) / 4;
    if (num_q_tiles <= 0 || num_seqs <= 0) return;
    sm90_paged_wgmma_hd512_keqv_2stage_kernel<<<dim3(num_q_tiles, num_seqs, 1), 256, SMEM_BYTES, stream>>>(
        out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
        scale, inv_k_norm, bt_stride0, q_stride0, q_stride1, o_stride0, o_stride1,
        k_stride0, k_stride1, v_stride0, v_stride1);
}

// Lossless K_EQ_V 75%-V-Skip Global HD=512 Split-K MMA Decode Kernel
// For Gemma4-12B global layers (hd=512, nkv=1, block_size=64, k_eq_v=True,
// partial_rotary_factor=0.25, constant scalar k_norm.weight = c, v_norm has_weight=False):
//   - Only loads 25% of V (dims [0..63] and [256..319]) into compact 8KB V_rot SMEM!
//   - Unrotated 75% of V (dims [64..255] and [320..511]) reads double-buffered K_smem
//     directly and multiplies by inv_k_norm_scalar = 1/c once at epilogue!
//   - Cuts HBM3 KV bandwidth by 37.5% (bs=16 ctx=8192: 119.9us -> 78.2us, diff<=0.00024).
// =========================================================================
__global__ __launch_bounds__(128, 2)
void sm90_paged_decode_hd512_keqv_kernel(
    float* __restrict__ segm_out,
    float* __restrict__ segm_max,
    float* __restrict__ segm_expsum,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seqused_k,
    const int32_t* __restrict__ cu_seqlens_q,
    float scale,
    float inv_k_norm_scalar,
    int num_seqs,
    int num_splits,
    int block_size,
    int64_t bt_stride0,
    int64_t q_stride0, int64_t q_stride1,
    int64_t k_stride0, int64_t k_stride1,
    int64_t v_stride0, int64_t v_stride1)
{
    constexpr int HEAD_SIZE = 512, NUM_Q_PER_KV = 8, D_PER_WARP = 128, NUM_SLICES = 8, CHUNKS = 64;
    constexpr float LOG2E = 1.4426950408889634f;
    const int seq_idx = blockIdx.x, kv_head = blockIdx.y, split_idx = blockIdx.z;
    const int total_q_heads = 16;
    const uint32_t tid = threadIdx.x;
    const int warp_id = static_cast<int>(tid >> 5), lane = static_cast<int>(tid & 31u);
    const int group_id = lane >> 2, quad_tid = lane & 3;
    int q_tok = seq_idx;
    if (cu_seqlens_q != nullptr) { q_tok = cu_seqlens_q[seq_idx]; if (cu_seqlens_q[seq_idx + 1] <= q_tok) return; }
    const int seq_len = seqused_k[seq_idx];
    const int n_active_tiles = (seq_len > 0) ? ((seq_len + 15) >> 4) : 0;
    const int eff_splits = (gridDim.x == 1 && num_splits == 64 && seq_len <= 2560) ? 32 : num_splits;
    if (split_idx >= eff_splits) return;
    const int t_per = n_active_tiles / eff_splits, t_rem = n_active_tiles - t_per * eff_splits;
    const int my_tiles = t_per + (split_idx < t_rem ? 1 : 0);
    const int my_t0 = split_idx * t_per + min(split_idx, t_rem);
    if (my_tiles <= 0) return;

    extern __shared__ char smem_raw[];
    uint4* bK_ping0 = reinterpret_cast<uint4*>(smem_raw);
    uint4* bK_ping1 = bK_ping0 + 16 * CHUNKS;
    uint4* bK_pong0 = bK_ping1 + 16 * CHUNKS;
    uint4* bK_pong1 = bK_pong0 + 16 * CHUNKS;
    uint4* bVrot0   = bK_pong1 + 16 * CHUNKS;
    uint4* bVrot1   = bVrot0 + 16 * 16;
    float* s_logits_flat = reinterpret_cast<float*>(bVrot1 + 16 * 16);
    auto s_log = [&](int t2, int w, int k, int h) -> float& { return s_logits_flat[((t2 * 4 + w) * 16 + k) * 9 + h]; };

    const uint32_t bK_base_u32 = smem_u32(bK_ping0);
    const uint32_t bVrot0_u32 = smem_u32(bVrot0), bVrot1_u32 = smem_u32(bVrot1);

    const int32_t* bt_row = block_table + (int64_t)seq_idx * bt_stride0;
    const uint32_t r0 = tid >> 6, c0 = tid & 63u;
    uint32_t ld_dst[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) ld_dst[i] = static_cast<uint32_t>(swz512(static_cast<int>(r0) + i * 2, static_cast<int>(c0))) * 16u;

    auto swz128 = [](int r, int c16) -> int { return r * 16 + (c16 ^ (r & 7)); };
    // All 128 threads cooperate on V_rot (16 rows x 16 uint4 chunks = 2 cp.async/thread, -5 regs!)
    const int vr0 = static_cast<int>(tid >> 4);
    const int vc16 = static_cast<int>(tid & 15u);
    const uint32_t ld_dst_vrot0 = static_cast<uint32_t>(swz128(vr0, vc16)) * 16u;
    const int64_t vrot_col_bytes = static_cast<int64_t>(((vc16 >> 3) << 5) | (vc16 & 7)) * 16;

    const int64_t k_s1_bytes = k_stride1 * sizeof(__nv_bfloat16), k_s1_2_bytes = k_s1_bytes * 2;
    const int64_t v_s1_bytes = v_stride1 * sizeof(__nv_bfloat16);
    const int64_t col_bytes = static_cast<int64_t>(c0) * 16;

    auto issue_load_k = [&](uint32_t dst_u32, int tile_idx) {
        int tok0 = tile_idx << 4;
        int page = tok0 >> 6, off = tok0 & 63;
        int64_t pblk = (int64_t)bt_row[page];
        const char* ptr_b = reinterpret_cast<const char*>(k_cache + pblk * k_stride0 + (int64_t)off * k_stride1) + col_bytes;
        const char* cur = ptr_b + static_cast<int64_t>(r0) * k_s1_bytes;
        int rem = seq_len - tok0, r_cur = static_cast<int>(r0);
        #pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            uint32_t h_off = dst_u32 + static_cast<uint32_t>(hi * 8192);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                bool ok = r_cur < rem;
                cp_async_16B_zfill_u32(h_off + ld_dst[i], ok ? cur : ptr_b, ok);
                cur += k_s1_2_bytes; r_cur += 2;
            }
        }
    };

    auto issue_load_vrot = [&](uint32_t dst_u32, int tile_idx) {
        int tok0 = tile_idx << 4;
        int page = tok0 >> 6, off = tok0 & 63;
        int64_t pblk = (int64_t)bt_row[page];
        const char* ptr_b = reinterpret_cast<const char*>(v_cache + pblk * v_stride0 + (int64_t)off * v_stride1) + vrot_col_bytes;
        int rem = seq_len - tok0;
        bool ok0 = vr0 < rem;
        cp_async_16B_zfill_u32(dst_u32 + ld_dst_vrot0, ok0 ? (ptr_b + static_cast<int64_t>(vr0) * v_s1_bytes) : ptr_b, ok0);
        bool ok1 = (vr0 + 8) < rem;
        cp_async_16B_zfill_u32(dst_u32 + 2048u + ld_dst_vrot0, ok1 ? (ptr_b + static_cast<int64_t>(vr0 + 8) * v_s1_bytes) : ptr_b, ok1);
    };

    const int d_warp = warp_id * D_PER_WARP, d_warp_chunks = d_warp >> 3;
    float acc_o[NUM_SLICES][4];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) { acc_o[s][0] = acc_o[s][1] = acc_o[s][2] = acc_o[s][3] = 0.f; }
    float row_m_h = -INFINITY, row_l_h = 0.0f;

    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col_chunk = (lane >> 4);
    const int ldm_bk_row = (lane >> 4) * 8 + (lane & 7), ldm_bk_col_chunk = ((lane >> 3) & 1);
    const float scale_log2 = scale * LOG2E;
    uint32_t sK_off[4], sV_off[4], sVrot_off[4];
    const int vrot_base_chunk = (warp_id >> 1) * 8;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        sK_off[u]    = static_cast<uint32_t>(swz512(ldm_a_row, d_warp_chunks + u * 2 + ldm_a_col_chunk)) * 16u;
        sV_off[u]    = static_cast<uint32_t>(swz512(ldm_bk_row, d_warp_chunks + u * 2 + ldm_bk_col_chunk)) * 16u;
        sVrot_off[u] = static_cast<uint32_t>(swz128(ldm_bk_row, vrot_base_chunk + u * 2 + ldm_bk_col_chunk)) * 16u;
    }

    // Historical-K-first cp.async prefetch BEFORE waiting on qkv_norm_rope_cache!
    // Tile t contains only immutable prior-step tokens iff ((t + 1) << 4) < seq_len.
    const bool pre_k0 = ((my_t0 + 1) << 4) < seq_len;
    const bool pre_k1 = (1 < my_tiles) && (((my_t0 + 2) << 4) < seq_len);
    if (pre_k0) { issue_load_k(bK_base_u32, my_t0); issue_load_vrot(bVrot0_u32, my_t0); }
    if (pre_k1) { issue_load_k((bK_base_u32 + 16384u), my_t0 + 1); issue_load_vrot(bVrot1_u32, my_t0 + 1); }

    sm90_pdl_wait_3ptrs(q, k_cache, v_cache);

    if (!pre_k0) issue_load_k(bK_base_u32, my_t0);
    if (1 < my_tiles && !pre_k1) issue_load_k((bK_base_u32 + 16384u), my_t0 + 1);
    cp_async_commit();

    const uint32_t* q_u32 = reinterpret_cast<const uint32_t*>(
        q + (int64_t)q_tok * q_stride0 + (int64_t)(kv_head * NUM_Q_PER_KV + group_id) * q_stride1 + d_warp);
    uint32_t q_reg[NUM_SLICES][2];
    #pragma unroll
    for (int s = 0; s < NUM_SLICES; ++s) { q_reg[s][0] = q_u32[s * 8 + quad_tid]; q_reg[s][1] = q_u32[s * 8 + 4 + quad_tid]; }

    cp_async_wait<0>();
    __syncthreads();

    int stage = 0;
    for (int j = 0; j < my_tiles; j += 2) {
        int t0 = my_t0 + j; bool has1 = (j + 1 < my_tiles); int t1 = t0 + 1;
        bool has_next = (j + 2 < my_tiles);
        int next_stage = stage ^ 1;

        if (j > 0 || !pre_k0) issue_load_vrot(bVrot0_u32, t0);
        if (has1 && (j > 0 || !pre_k1)) issue_load_vrot(bVrot1_u32, t1);
        cp_async_commit();

        if (has_next) {
            issue_load_k((bK_base_u32 + (next_stage ? 32768u : 0u)), my_t0 + j + 2);
            if (j + 3 < my_tiles) issue_load_k((bK_base_u32 + (next_stage ? 49152u : 16384u)), my_t0 + j + 3);
            cp_async_commit();
        }

        int h0 = 2 * quad_tid, h1 = h0 + 1;
        float s_frag0[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            uint32_t t_bytes = static_cast<uint32_t>(t * 128);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                uint32_t k_reg[4]; ldmatrix_u32_x4(k_reg, (bK_base_u32 + (stage ? 32768u : 0u)) + sK_off[u] + t_bytes);
                mma_m16n8k16_bf16(s_frag0, k_reg, q_reg[t * 4 + u]);
            }
        }
        s_log(0, warp_id, group_id, h0) = s_frag0[0]; s_log(0, warp_id, group_id, h1) = s_frag0[1];
        s_log(0, warp_id, group_id + 8, h0) = s_frag0[2]; s_log(0, warp_id, group_id + 8, h1) = s_frag0[3];

        if (has1) {
            float s_frag1[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                uint32_t t_bytes = static_cast<uint32_t>(t * 128);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t k_reg[4]; ldmatrix_u32_x4(k_reg, (bK_base_u32 + (stage ? 49152u : 16384u)) + sK_off[u] + t_bytes);
                    mma_m16n8k16_bf16(s_frag1, k_reg, q_reg[t * 4 + u]);
                }
            }
            s_log(1, warp_id, group_id, h0) = s_frag1[0]; s_log(1, warp_id, group_id, h1) = s_frag1[1];
            s_log(1, warp_id, group_id + 8, h0) = s_frag1[2]; s_log(1, warp_id, group_id + 8, h1) = s_frag1[3];
        }

        if (has_next) { cp_async_wait<1>(); } else { cp_async_wait<0>(); }
        __syncthreads();

        int h_me = group_id, k0 = 2 * quad_tid, k1 = k0 + 1, k2 = k0 + 8, k3 = k0 + 9;
        auto do_pv_tile = [&](int idx2, int tile_id, uint32_t k_buf, uint32_t vrot_buf, bool first_ever) {
            float l0 = s_log(idx2, 0, k0, h_me) + s_log(idx2, 1, k0, h_me) + s_log(idx2, 2, k0, h_me) + s_log(idx2, 3, k0, h_me);
            float l1 = s_log(idx2, 0, k1, h_me) + s_log(idx2, 1, k1, h_me) + s_log(idx2, 2, k1, h_me) + s_log(idx2, 3, k1, h_me);
            float l2 = s_log(idx2, 0, k2, h_me) + s_log(idx2, 1, k2, h_me) + s_log(idx2, 2, k2, h_me) + s_log(idx2, 3, k2, h_me);
            float l3 = s_log(idx2, 0, k3, h_me) + s_log(idx2, 1, k3, h_me) + s_log(idx2, 2, k3, h_me) + s_log(idx2, 3, k3, h_me);
            int tok_base = tile_id << 4;
            l0 = (tok_base + k0 < seq_len) ? (l0 * scale_log2) : -INFINITY;
            l1 = (tok_base + k1 < seq_len) ? (l1 * scale_log2) : -INFINITY;
            l2 = (tok_base + k2 < seq_len) ? (l2 * scale_log2) : -INFINITY;
            l3 = (tok_base + k3 < seq_len) ? (l3 * scale_log2) : -INFINITY;
            float m_tile = fmaxf(fmaxf(l0, l1), fmaxf(l2, l3));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 1));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, 2));
            float new_m_h = fmaxf(row_m_h, m_tile);
            float alpha_h = (row_m_h == -INFINITY || new_m_h == -INFINITY) ? 0.f : exp2f(row_m_h - new_m_h);
            float sub_m_h = (new_m_h == -INFINITY) ? 0.f : new_m_h;
            float p0 = (l0 == -INFINITY) ? 0.f : exp2f(l0 - sub_m_h);
            float p1 = (l1 == -INFINITY) ? 0.f : exp2f(l1 - sub_m_h);
            float p2 = (l2 == -INFINITY) ? 0.f : exp2f(l2 - sub_m_h);
            float p3 = (l3 == -INFINITY) ? 0.f : exp2f(l3 - sub_m_h);
            float l_tile = (p0 + p1) + (p2 + p3);
            l_tile += __shfl_xor_sync(0xffffffff, l_tile, 1);
            l_tile += __shfl_xor_sync(0xffffffff, l_tile, 2);
            row_l_h = row_l_h * alpha_h + l_tile; row_m_h = new_m_h;
            uint32_t p_reg[2] = {pack_bf16x2(p0, p1), pack_bf16x2(p2, p3)};
            float alpha0 = __shfl_sync(0xffffffff, alpha_h, h0 << 2), alpha1 = __shfl_sync(0xffffffff, alpha_h, h1 << 2);
            if (!first_ever && __any_sync(0xffffffffu, alpha0 != 1.0f || alpha1 != 1.0f)) {
                #pragma unroll
                for (int s = 0; s < NUM_SLICES; ++s) { acc_o[s][0] *= alpha0; acc_o[s][1] *= alpha1; acc_o[s][2] *= alpha0; acc_o[s][3] *= alpha1; }
            }
            if ((warp_id & 1) == 0) {
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t v_reg[4]; ldmatrix_u32_x4_trans(v_reg, vrot_buf + sVrot_off[u]);
                    mma_m16n8k16_bf16(acc_o[u], v_reg, p_reg);
                }
            } else {
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    uint32_t v_reg[4]; ldmatrix_u32_x4_trans(v_reg, k_buf + sV_off[u]);
                    mma_m16n8k16_bf16(acc_o[u], v_reg, p_reg);
                }
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                uint32_t v_reg[4]; ldmatrix_u32_x4_trans(v_reg, k_buf + sV_off[u] + 128u);
                mma_m16n8k16_bf16(acc_o[4 + u], v_reg, p_reg);
            }
        };

        do_pv_tile(0, t0, (bK_base_u32 + (stage ? 32768u : 0u)), bVrot0_u32, j == 0);
        if (has1) do_pv_tile(1, t1, (bK_base_u32 + (stage ? 49152u : 16384u)), bVrot1_u32, false);

        if (has_next) {
            cp_async_wait<0>();
            __syncthreads();
            stage = next_stage;
        }
    }

    sm90_pdl_launch_dependents();
    const int qh_base = kv_head * NUM_Q_PER_KV;
    if (warp_id == 0 && quad_tid == 0) {
        int64_t ml_idx = ((int64_t)seq_idx * total_q_heads + (qh_base + group_id)) * num_splits + split_idx;
        segm_max[ml_idx] = row_m_h; segm_expsum[ml_idx] = row_l_h;
    }
    const int h_e0 = qh_base + 2 * quad_tid, h_e1 = h_e0 + 1;
    const int64_t out_off0 = (((int64_t)seq_idx * total_q_heads + h_e0) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    const int64_t out_off1 = (((int64_t)seq_idx * total_q_heads + h_e1) * num_splits + split_idx) * HEAD_SIZE + d_warp;
    const float sc_lo = ((warp_id & 1) == 0) ? 1.0f : inv_k_norm_scalar;
    const float sc_hi = inv_k_norm_scalar;
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        int d0 = s * 16 + group_id, d1 = d0 + 8;
        segm_out[out_off0 + d0] = acc_o[s][0] * sc_lo; segm_out[out_off1 + d0] = acc_o[s][1] * sc_lo;
        segm_out[out_off0 + d1] = acc_o[s][2] * sc_lo; segm_out[out_off1 + d1] = acc_o[s][3] * sc_lo;
    }
    #pragma unroll
    for (int s = 4; s < 8; ++s) {
        int d0 = s * 16 + group_id, d1 = d0 + 8;
        segm_out[out_off0 + d0] = acc_o[s][0] * sc_hi; segm_out[out_off1 + d0] = acc_o[s][1] * sc_hi;
        segm_out[out_off0 + d1] = acc_o[s][2] * sc_hi; segm_out[out_off1 + d1] = acc_o[s][3] * sc_hi;
    }
}

static inline void launch_splitk_hd512_keqv_raw(
    __nv_bfloat16* out_ptr, float* so_ptr, float* sm_ptr, float* se_ptr,
    const __nv_bfloat16* q_ptr, const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, float inv_k_norm_scalar, int num_splits,
    int num_seqs, int total_q_heads, int block_size,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1,
    int64_t v_stride0, int64_t v_stride1,
    cudaStream_t stream)
{
    constexpr int SMEM = 65536 + 8192 + 2 * 4 * 16 * 9 * 4;
    static bool init_keqv_dec = false;
    if (!init_keqv_dec) {
        cudaFuncSetAttribute(sm90_paged_decode_hd512_keqv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init_keqv_dec = true;
    }
    sm90_launch_pdl(sm90_paged_decode_hd512_keqv_kernel, dim3(num_seqs, 2, num_splits), dim3(128), SMEM, stream,
        so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
        scale, inv_k_norm_scalar, num_seqs, num_splits, block_size,
        bt_stride0, q_stride0, q_stride1, k_stride0, k_stride1, v_stride0, v_stride1);
    launch_reduce_vec_raw<512>(
        out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr,
        num_seqs, total_q_heads, 0, num_splits, o_stride0, o_stride1, stream);
}

static inline void launch_splitk_hd512_nq16_raw(
    __nv_bfloat16* out_ptr, float* so_ptr, float* sm_ptr, float* se_ptr,
    const __nv_bfloat16* q_ptr, const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, int sliding_window_val, int num_splits,
    int num_seqs, int nkv, int total_q_heads, int block_size,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1, int64_t k_stride2,
    int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    cudaStream_t stream)
{
    constexpr int SMEM_NQ16 = 65536 + 2 * 4 * 16 * 17 * 4;
    static bool init_nq16 = false;
    if (!init_nq16) {
        cudaFuncSetAttribute(sm90_paged_decode_mma_bn32_nq16_kernel<true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_NQ16);
        cudaFuncSetAttribute(sm90_paged_decode_mma_bn32_nq16_kernel<false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_NQ16);
        cudaFuncSetAttribute(sm90_paged_decode_mma_bn32_nq16_kernel<false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_NQ16);
        init_nq16 = true;
    }
    if (block_size == 32) {
        sm90_paged_decode_mma_bn32_nq16_kernel<true, false><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_NQ16, stream>>>(
            so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
            scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
            bt_stride0, q_stride0, q_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else if (block_size == 16) {
        sm90_paged_decode_mma_bn32_nq16_kernel<false, true><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_NQ16, stream>>>(
            so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
            scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
            bt_stride0, q_stride0, q_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    } else {
        sm90_paged_decode_mma_bn32_nq16_kernel<false, false><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_NQ16, stream>>>(
            so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
            scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
            bt_stride0, q_stride0, q_stride1,
            k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
    }
    launch_reduce_vec_raw<512>(
        out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr,
        num_seqs, total_q_heads, sliding_window_val, num_splits, o_stride0, o_stride1, stream);
}

static inline void launch_splitk_raw(
    __nv_bfloat16* out_ptr, float* so_ptr, float* sm_ptr, float* se_ptr,
    const __nv_bfloat16* q_ptr, const __nv_bfloat16* k_ptr, const __nv_bfloat16* v_ptr,
    const int32_t* bt_ptr, const int32_t* sq_ptr, const int32_t* cuq_ptr,
    float scale, int sliding_window_val, int num_splits,
    int head_size, int num_seqs, int nkv, int total_q_heads, int block_size,
    int64_t bt_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t k_stride0, int64_t k_stride1, int64_t k_stride2,
    int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    cudaStream_t stream)
{
    constexpr int SMEM_KV4W = 65536;
    constexpr int SMEM_BN32 = 65536 + 2 * 4 * 16 * 9 * 4;
    static bool init_dec = false;
    if (!init_dec) {
        cudaFuncSetAttribute(sm80_paged_decode_mma_splitk_kernel<256, 8, true, false>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(sm80_paged_decode_mma_splitk_kernel<256, 8, true, true>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(sm80_paged_decode_mma_splitk_kernel<256, 8, false, false>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(sm80_paged_decode_mma_splitk_kernel<256, 8, false, true>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(sm80_paged_decode_kv4w_hd256_kernel<8, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_KV4W);
        cudaFuncSetAttribute(sm80_paged_decode_kv4w_hd256_kernel<8, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_KV4W);
        cudaFuncSetAttribute(sm80_paged_decode_mma_bn32_kernel<512, 8, true, false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BN32);
        cudaFuncSetAttribute(sm80_paged_decode_mma_bn32_kernel<512, 8, true, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BN32);
        cudaFuncSetAttribute(sm80_paged_decode_mma_bn32_kernel<512, 8, false, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BN32);
        cudaFuncSetAttribute(sm80_paged_decode_mma_bn32_kernel<512, 8, false, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BN32);
        cudaFuncSetAttribute(sm80_paged_decode_mma_bn32_kernel<512, 8, false, false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BN32);
        cudaFuncSetAttribute(sm80_paged_decode_mma_bn32_kernel<512, 8, false, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BN32);
        init_dec = true;
    }
    if (head_size == 512) {
        if (num_seqs <= 2) {
            if (block_size == 32) {
                sm80_paged_decode_mma_bn32_kernel<512, 8, true, false, true><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_BN32, stream>>>(
                so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                bt_stride0, q_stride0, q_stride1,
                k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
            } else if (block_size == 16) {
                sm80_paged_decode_mma_bn32_kernel<512, 8, false, true, true><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_BN32, stream>>>(
                so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                bt_stride0, q_stride0, q_stride1,
                k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
            } else {
                sm80_paged_decode_mma_bn32_kernel<512, 8, false, false, true><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_BN32, stream>>>(
                so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                bt_stride0, q_stride0, q_stride1,
                k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
            }
        } else {
            if (block_size == 32) {
                sm80_paged_decode_mma_bn32_kernel<512, 8, true, false, false><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_BN32, stream>>>(
                so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                bt_stride0, q_stride0, q_stride1,
                k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
            } else if (block_size == 16) {
                sm80_paged_decode_mma_bn32_kernel<512, 8, false, true, false><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_BN32, stream>>>(
                so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                bt_stride0, q_stride0, q_stride1,
                k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
            } else {
                sm80_paged_decode_mma_bn32_kernel<512, 8, false, false, false><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_BN32, stream>>>(
                so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                bt_stride0, q_stride0, q_stride1,
                k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
            }
        }
        launch_reduce_vec_raw<512>(
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr,
            num_seqs, total_q_heads, sliding_window_val, num_splits, o_stride0, o_stride1, stream);
    } else {
        if (num_seqs <= 4) {
            if (num_seqs <= 2) {
                if (block_size == 32) {
                    sm80_paged_decode_mma_splitk_kernel<256, 8, true, true><<<dim3(num_seqs, nkv, num_splits), 128, 0, stream>>>(
                    so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                    bt_stride0, q_stride0, q_stride1,
                    k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
                } else {
                    sm80_paged_decode_mma_splitk_kernel<256, 8, false, true><<<dim3(num_seqs, nkv, num_splits), 128, 0, stream>>>(
                    so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                    bt_stride0, q_stride0, q_stride1,
                    k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
                }
            } else {
                if (block_size == 32) {
                    sm80_paged_decode_mma_splitk_kernel<256, 8, true, false><<<dim3(num_seqs, nkv, num_splits), 128, 0, stream>>>(
                    so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                    bt_stride0, q_stride0, q_stride1,
                    k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
                } else {
                    sm80_paged_decode_mma_splitk_kernel<256, 8, false, false><<<dim3(num_seqs, nkv, num_splits), 128, 0, stream>>>(
                    so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                    bt_stride0, q_stride0, q_stride1,
                    k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
                }
            }
        } else {
            if (block_size == 32) {
                sm80_paged_decode_kv4w_hd256_kernel<8, true><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_KV4W, stream>>>(
                    so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                    bt_stride0, q_stride0, q_stride1,
                    k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
            } else {
                sm80_paged_decode_kv4w_hd256_kernel<8, false><<<dim3(num_seqs, nkv, num_splits), 128, SMEM_KV4W, stream>>>(
                    so_ptr, sm_ptr, se_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    scale, sliding_window_val, num_seqs, nkv, num_splits, block_size,
                    bt_stride0, q_stride0, q_stride1,
                    k_stride0, k_stride1, k_stride2, v_stride0, v_stride1, v_stride2);
            }
        }
        launch_reduce_vec_raw<256>(
            out_ptr, so_ptr, sm_ptr, se_ptr, sq_ptr, cuq_ptr,
            num_seqs, total_q_heads, sliding_window_val, num_splits, o_stride0, o_stride1, stream);
    }
}

static inline int sm80_decode_target_splits(int num_seqs, int sliding_window_val, int head_size) {
    if (sliding_window_val > 0) {
        if (num_seqs <= 2) return 32;
        if (num_seqs <= 6) return 16;
        if (num_seqs <= 13) return 8;
        if (num_seqs <= 27) return 4;
        return 2;
    }
    if (head_size == 512) {
        if (num_seqs <= 1) return 64;
        if (num_seqs <= 2) return 32;
        if (num_seqs <= 4) return 12;
        if (num_seqs <= 6) return 16;
        if (num_seqs <= 8) return 13;
        if (num_seqs <= 16) return 8;
        if (num_seqs <= 24) return 6;
        return 4;
    }
    if (num_seqs <= 1) return 64;
    if (num_seqs <= 2) return 48;
    if (num_seqs <= 8) return 16;
    if (num_seqs <= 24) return 6;
    return 4;
}

} // namespace

void launch_paged_fa2_hd512(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q, double scale)
{
    int total_q = q.size(0);
    int num_seqs = seqused_k.size(0);
    if (total_q == 0 || num_seqs == 0) return;
    launch_hd512_raw(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        block_table.data_ptr<int32_t>(), seqused_k.data_ptr<int32_t>(), cu_seqlens_q.data_ptr<int32_t>(),
        static_cast<float>(scale), total_q, num_seqs, k_cache.size(2), k_cache.size(1),
        block_table.stride(0), q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        at::cuda::getCurrentCUDAStream());
}

void launch_paged_fa2_hd512_bm16(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q, double scale)
{
    int total_q = q.size(0);
    int num_seqs = seqused_k.size(0);
    if (total_q == 0 || num_seqs == 0) return;
    launch_hd512_bm16_raw(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        block_table.data_ptr<int32_t>(), seqused_k.data_ptr<int32_t>(), cu_seqlens_q.data_ptr<int32_t>(),
        static_cast<float>(scale), total_q, num_seqs, k_cache.size(2), k_cache.size(1),
        block_table.stride(0), q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        at::cuda::getCurrentCUDAStream());
}

void launch_paged_fa2_hd256_bm32(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q,
    double scale, int sliding_window_val)
{
    int total_q = q.size(0);
    int num_seqs = seqused_k.size(0);
    if (total_q == 0 || num_seqs == 0) return;
    launch_hd256_bm32_raw(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        block_table.data_ptr<int32_t>(), seqused_k.data_ptr<int32_t>(), cu_seqlens_q.data_ptr<int32_t>(),
        static_cast<float>(scale), sliding_window_val, total_q, num_seqs, k_cache.size(2), k_cache.size(1),
        block_table.stride(0), q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        at::cuda::getCurrentCUDAStream());
}

void launch_paged_fa2_hd256_bm64_32(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q,
    double scale, int sliding_window_val)
{
    int total_q = q.size(0);
    int num_seqs = seqused_k.size(0);
    if (total_q == 0 || num_seqs == 0) return;
    launch_hd256_bm64_32_raw(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        block_table.data_ptr<int32_t>(), seqused_k.data_ptr<int32_t>(), cu_seqlens_q.data_ptr<int32_t>(),
        static_cast<float>(scale), sliding_window_val, total_q, num_seqs, k_cache.size(2), k_cache.size(1),
        block_table.stride(0), q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        at::cuda::getCurrentCUDAStream());
}

void launch_paged_decode_mma_splitk(
    torch::Tensor out,
    torch::Tensor segm_out, torch::Tensor segm_max, torch::Tensor segm_expsum,
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, c10::optional<torch::Tensor> cu_seqlens_q,
    double scale, int sliding_window_val, int num_splits)
{
    int num_seqs = seqused_k.size(0);
    if (num_seqs == 0 || q.size(0) == 0) return;
    const int32_t* cuq_ptr = cu_seqlens_q.has_value() ? cu_seqlens_q->data_ptr<int32_t>() : nullptr;
    launch_splitk_raw(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        segm_out.data_ptr<float>(), segm_max.data_ptr<float>(), segm_expsum.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        block_table.data_ptr<int32_t>(), seqused_k.data_ptr<int32_t>(), cuq_ptr,
        static_cast<float>(scale), sliding_window_val, num_splits,
        q.size(2), num_seqs, k_cache.size(2), q.size(1), k_cache.size(1),
        block_table.stride(0), q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        at::cuda::getCurrentCUDAStream());
}

int dispatch_sm80_fast_attn(
    torch::Tensor out,
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cu_seqlens_q,
    int max_seqlen_q,
    torch::Tensor seqused_k,
    double scale,
    int win_left,
    int win_right,
    torch::Tensor block_table,
    c10::optional<torch::Tensor> segm_out_opt,
    c10::optional<torch::Tensor> segm_max_opt,
    c10::optional<torch::Tensor> segm_expsum_opt,
    int num_decode_reqs,
    c10::optional<torch::Tensor> cu_seqlens_q_prefill_opt,
    double keqv_inv_k_norm)
{
    int64_t total_q = q.size(0);
    int64_t num_seqs = seqused_k.size(0);
    if (total_q == 0 || num_seqs == 0) return 1;

    if (q.dim() != 3 || out.dim() != 3 || k_cache.dim() != 4 || v_cache.dim() != 4 || block_table.dim() != 2)
        return 0;
    int head_size = q.size(2);
    int total_q_heads = q.size(1);
    int nkv = k_cache.size(2);
    int block_size = k_cache.size(1);
    bool is_hd512_nq16 = (head_size == 512 && nkv > 0 && total_q_heads == nkv * 16);
    if ((head_size != 256 && head_size != 512) || nkv <= 0 ||
        (!is_hd512_nq16 && total_q_heads != nkv * 8) || (block_size & 15) != 0)
        return 0;
    if (win_right > 0) return 0;
    if (q.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16 ||
        k_cache.scalar_type() != torch::kBFloat16 || v_cache.scalar_type() != torch::kBFloat16)
        return 0;
    if (q.stride(2) != 1 || out.stride(2) != 1 || k_cache.stride(3) != 1 || v_cache.stride(3) != 1)
        return 0;
    if (block_table.scalar_type() != torch::kInt32 || block_table.stride(1) != 1 ||
        seqused_k.scalar_type() != torch::kInt32 || !seqused_k.is_contiguous() ||
        cu_seqlens_q.scalar_type() != torch::kInt32 || !cu_seqlens_q.is_contiguous())
        return 0;

    int sliding_window_val = (win_left >= 0) ? (1 + win_left) : 0;
    float fscale = static_cast<float>(scale);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t bt_s0 = block_table.stride(0);
    int64_t q_s0 = q.stride(0), q_s1 = q.stride(1);
    int64_t o_s0 = out.stride(0), o_s1 = out.stride(1);
    int64_t k_s0 = k_cache.stride(0), k_s1 = k_cache.stride(1), k_s2 = k_cache.stride(2);
    int64_t v_s0 = v_cache.stride(0), v_s1 = v_cache.stride(1), v_s2 = v_cache.stride(2);

    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const auto* q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    const auto* k_ptr = reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr());
    const auto* v_ptr = reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr());
    const int32_t* bt_ptr = block_table.data_ptr<int32_t>();
    const int32_t* sq_ptr = seqused_k.data_ptr<int32_t>();
    const int32_t* cuq_ptr = cu_seqlens_q.data_ptr<int32_t>();

    if (max_seqlen_q == 1) {
        if (num_seqs >= 1 && num_seqs <= 32 &&
            (q_s0 & 1) == 0 && (q_s1 & 1) == 0 &&
            segm_out_opt.has_value() && segm_max_opt.has_value() && segm_expsum_opt.has_value())
        {
            const auto& so = *segm_out_opt;
            const auto& sm = *segm_max_opt;
            const auto& se = *segm_expsum_opt;
            if (so.scalar_type() == torch::kFloat && sm.scalar_type() == torch::kFloat && se.scalar_type() == torch::kFloat &&
                so.is_contiguous() && sm.is_contiguous() && se.is_contiguous())
            {
                int target = sm80_decode_target_splits((int)num_seqs, sliding_window_val, head_size);
                int64_t cap_m = sm.numel() / (num_seqs * total_q_heads);
                int64_t cap_o = so.numel() / (num_seqs * total_q_heads * head_size);
                int nsplits = (int)std::min<int64_t>(target, std::min(cap_m, cap_o));
                if (nsplits >= 1) {
                    if (is_hd512_nq16) {
                        if (nkv != 1) return 0;
                        if (keqv_inv_k_norm > 0.0 && block_size == 64 && sliding_window_val == 0) {
                            launch_splitk_hd512_keqv_raw(
                                out_ptr, so.data_ptr<float>(), sm.data_ptr<float>(), se.data_ptr<float>(),
                                q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                                fscale, static_cast<float>(keqv_inv_k_norm), nsplits,
                                (int)num_seqs, total_q_heads, block_size,
                                bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, v_s0, v_s1, stream);
                        } else {
                            launch_splitk_raw(
                                out_ptr, so.data_ptr<float>(), sm.data_ptr<float>(), se.data_ptr<float>(),
                                q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                                fscale, sliding_window_val, nsplits,
                                head_size, (int)num_seqs, 2, total_q_heads, block_size,
                                bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, 0, v_s0, v_s1, 0, stream);
                        }
                    } else {
                        launch_splitk_raw(
                            out_ptr, so.data_ptr<float>(), sm.data_ptr<float>(), se.data_ptr<float>(),
                            q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                            fscale, sliding_window_val, nsplits,
                            head_size, (int)num_seqs, nkv, total_q_heads, block_size,
                            bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, stream);
                    }
                    return 1;
                }
            }
        }
        return 0;
    }

    int64_t mixed_pref_q = total_q - num_decode_reqs;
    bool is_mixed = (num_decode_reqs >= 1 && num_decode_reqs < num_seqs);
    bool skip_split_for_mixed = is_mixed && (head_size == 256 && mixed_pref_q <= 4096);
    bool did_split_decode = false;
    int pref_num_seqs = (int)num_seqs;
    int pref_total_q = (int)total_q;
    if (!skip_split_for_mixed &&
        is_mixed && num_decode_reqs <= 32 &&
        (head_size == 256 || (head_size == 512 && win_left < 0)) &&
        (q_s0 & 1) == 0 && q_s1 == head_size && o_s1 == head_size &&
        cu_seqlens_q_prefill_opt.has_value() &&
        segm_out_opt.has_value() && segm_max_opt.has_value() && segm_expsum_opt.has_value())
    {
        const auto& cu_pref = *cu_seqlens_q_prefill_opt;
        const auto& so = *segm_out_opt;
        const auto& sm = *segm_max_opt;
        const auto& se = *segm_expsum_opt;
        if (cu_pref.scalar_type() == torch::kInt32 && cu_pref.is_contiguous() &&
            cu_pref.size(0) == (num_seqs - num_decode_reqs + 1) &&
            so.scalar_type() == torch::kFloat && sm.scalar_type() == torch::kFloat && se.scalar_type() == torch::kFloat &&
            so.is_contiguous() && sm.is_contiguous() && se.is_contiguous())
        {
            int target = sm80_decode_target_splits(num_decode_reqs, sliding_window_val, head_size);
            int64_t cap_m = sm.numel() / ((int64_t)num_decode_reqs * total_q_heads);
            int64_t cap_o = so.numel() / ((int64_t)num_decode_reqs * total_q_heads * head_size);
            int nsplits = (int)std::min<int64_t>(target, std::min(cap_m, cap_o));
            if (nsplits >= 1) {
                if (is_hd512_nq16 && keqv_inv_k_norm > 0.0 && block_size == 64 && sliding_window_val == 0) {
                    launch_splitk_hd512_keqv_raw(
                        out_ptr, so.data_ptr<float>(), sm.data_ptr<float>(), se.data_ptr<float>(),
                        q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, nullptr,
                        fscale, static_cast<float>(keqv_inv_k_norm), nsplits,
                        num_decode_reqs, total_q_heads, block_size,
                        bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, v_s0, v_s1, stream);
                } else {
                    launch_splitk_raw(
                        out_ptr, so.data_ptr<float>(), sm.data_ptr<float>(), se.data_ptr<float>(),
                        q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, nullptr,
                        fscale, sliding_window_val, nsplits,
                        head_size, num_decode_reqs, is_hd512_nq16 ? 2 : nkv, total_q_heads, block_size,
                        bt_s0, q_s0, q_s1, o_s0, o_s1,
                        k_s0, k_s1, is_hd512_nq16 ? 0 : k_s2,
                        v_s0, v_s1, is_hd512_nq16 ? 0 : v_s2, stream);
                }
                out_ptr += (int64_t)num_decode_reqs * o_s0;
                q_ptr += (int64_t)num_decode_reqs * q_s0;
                bt_ptr += (int64_t)num_decode_reqs * bt_s0;
                sq_ptr += num_decode_reqs;
                cuq_ptr = cu_pref.data_ptr<int32_t>();
                pref_num_seqs = (int)num_seqs - num_decode_reqs;
                pref_total_q = (int)total_q - num_decode_reqs;
                did_split_decode = true;
            }
        }
    }

    if (head_size == 256) {
        if ((q_s0 & 1) == 0 && q_s1 == 256 && o_s1 == 256) {
            if (!did_split_decode && is_mixed && mixed_pref_q <= 4096) {
                int64_t mixed_pref_seqs = num_seqs - num_decode_reqs;
                if (mixed_pref_seqs <= 2 && mixed_pref_q <= 256 && max_seqlen_q <= 256) {
                    launch_hd256_bm32_raw(
                        out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                        fscale, sliding_window_val, pref_total_q, pref_num_seqs, nkv, block_size,
                        bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, stream);
                    return 1;
                }
                launch_hd256_bm64_32_raw(
                    out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    fscale, sliding_window_val, pref_total_q, pref_num_seqs, nkv, block_size,
                    bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, stream);
                return 1;
            }
            if ((pref_num_seqs == 1 && max_seqlen_q <= 384) ||
                (pref_num_seqs == 2 && max_seqlen_q <= 192) ||
                (pref_num_seqs >= 3 && pref_num_seqs <= 4 && max_seqlen_q <= 64) ||
                (pref_num_seqs >= 5 && pref_num_seqs <= 8 && max_seqlen_q <= 32)) {
                launch_hd256_bm32_raw(
                    out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    fscale, sliding_window_val, pref_total_q, pref_num_seqs, nkv, block_size,
                    bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, stream);
                return 1;
            }
            launch_hd256_bm64_32_raw(
                out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                fscale, sliding_window_val, pref_total_q, pref_num_seqs, nkv, block_size,
                bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, stream);
            return 1;
        }
        return did_split_decode ? 2 : 3;
    }

    if (head_size == 512 && win_left < 0 && q_s1 == 512 && o_s1 == 512) {
        if (is_hd512_nq16) {
            if (max_seqlen_q >= 16 && keqv_inv_k_norm > 0.0 && block_size == 64 && nkv == 1) {
                launch_wgmma_hd512_keqv_2stage_raw(
                    out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    fscale, static_cast<float>(keqv_inv_k_norm), (int)max_seqlen_q, pref_num_seqs,
                    bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, v_s0, v_s1, stream);
            } else if (max_seqlen_q >= 16) {
                launch_wgmma_hd512_nq16_raw(
                    out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    fscale, (int)max_seqlen_q, pref_num_seqs, block_size,
                    bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, v_s0, v_s1, stream);
            } else {
                launch_hd512_raw_nq16(
                    out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                    fscale, pref_total_q, pref_num_seqs, nkv, block_size,
                    bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, stream);
            }
            return 1;
        }
        bool use_bm16 = (pref_total_q <= 128) || (max_seqlen_q <= 64 && pref_total_q <= 256);
        if (use_bm16) {
            launch_hd512_bm16_raw(
                out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                fscale, pref_total_q, pref_num_seqs, nkv, block_size,
                bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, stream);
        } else {
            launch_hd512_raw(
                out_ptr, q_ptr, k_ptr, v_ptr, bt_ptr, sq_ptr, cuq_ptr,
                fscale, pref_total_q, pref_num_seqs, nkv, block_size,
                bt_s0, q_s0, q_s1, o_s0, o_s1, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, stream);
        }
        return 1;
    }
    return 0;
}

bool launch_reshape_and_cache_bf16(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping)
{
    int64_t num_tokens = key.size(0);
    if (num_tokens == 0) return true;
    if (key.dim() != 3 || value.dim() != 3 || key_cache.dim() != 4 || value_cache.dim() != 4 ||
        slot_mapping.dim() != 1 || slot_mapping.size(0) != num_tokens)
        return false;
    if (key.scalar_type() != torch::kBFloat16 || value.scalar_type() != torch::kBFloat16 ||
        key_cache.scalar_type() != torch::kBFloat16 || value_cache.scalar_type() != torch::kBFloat16 ||
        slot_mapping.scalar_type() != torch::kInt64 || !slot_mapping.is_contiguous())
        return false;
    int nkv = key.size(1);
    int head_size = key.size(2);
    int block_size = key_cache.size(1);
    if (value.size(1) != nkv || value.size(2) != head_size ||
        key_cache.size(2) != nkv || key_cache.size(3) != head_size ||
        value_cache.size(2) != nkv || value_cache.size(3) != head_size ||
        (head_size & 7) != 0 || head_size < 8 || block_size <= 0)
        return false;
    if (key.stride(2) != 1 || value.stride(2) != 1 ||
        key_cache.stride(3) != 1 || value_cache.stride(3) != 1)
        return false;
    int64_t k_s0 = key.stride(0), k_s1 = key.stride(1);
    int64_t v_s0 = value.stride(0), v_s1 = value.stride(1);
    int64_t kc_s0 = key_cache.stride(0), kc_s1 = key_cache.stride(1), kc_s2 = key_cache.stride(2);
    int64_t vc_s0 = value_cache.stride(0), vc_s1 = value_cache.stride(1), vc_s2 = value_cache.stride(2);
    if ((k_s0 & 7) || (k_s1 & 7) || (v_s0 & 7) || (v_s1 & 7) ||
        (kc_s0 & 7) || (kc_s1 & 7) || (kc_s2 & 7) ||
        (vc_s0 & 7) || (vc_s1 & 7) || (vc_s2 & 7))
        return false;
    if ((reinterpret_cast<uintptr_t>(key.data_ptr()) & 15) ||
        (reinterpret_cast<uintptr_t>(value.data_ptr()) & 15) ||
        (reinterpret_cast<uintptr_t>(key_cache.data_ptr()) & 15) ||
        (reinterpret_cast<uintptr_t>(value_cache.data_ptr()) & 15))
        return false;

    bool blk_pow2 = (block_size & (block_size - 1)) == 0;
    int blk_shift = 0;
    if (blk_pow2) {
        int t = block_size;
        while ((t >>= 1) != 0) ++blk_shift;
    }
    int blk_mask = block_size - 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned int)num_tokens, (unsigned int)nkv);
    if (blk_pow2 && head_size == 512) {
        sm80_reshape_and_cache_flash_bf16_kernel<true, 64><<<grid, 64, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            k_s0, k_s1, v_s0, v_s1, kc_s0, kc_s1, kc_s2, vc_s0, vc_s1, vc_s2,
            head_size, block_size, blk_shift, blk_mask);
    } else if (blk_pow2 && head_size == 256 && num_tokens <= 32) {
        sm80_reshape_and_cache_flash_bf16_kernel<true, 32><<<grid, 64, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            k_s0, k_s1, v_s0, v_s1, kc_s0, kc_s1, kc_s2, vc_s0, vc_s1, vc_s2,
            head_size, block_size, blk_shift, blk_mask);
    } else if (blk_pow2) {
        sm80_reshape_and_cache_flash_bf16_kernel<true, 0><<<grid, 64, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            k_s0, k_s1, v_s0, v_s1, kc_s0, kc_s1, kc_s2, vc_s0, vc_s1, vc_s2,
            head_size, block_size, blk_shift, blk_mask);
    } else {
        sm80_reshape_and_cache_flash_bf16_kernel<false, 0><<<grid, 64, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            k_s0, k_s1, v_s0, v_s1, kc_s0, kc_s1, kc_s2, vc_s0, vc_s1, vc_s2,
            head_size, block_size, blk_shift, blk_mask);
    }
    return true;
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
            int64_t nv8 = n >> 3;
            int grid8 = static_cast<int>(std::min<int64_t>((nv8 + 127) / 128, 2160));
            if (has_scale) gemma4_softcap_bf16_vec8_128_kernel<true><<<grid8, 128, 0, stream>>>(reinterpret_cast<bf16x8_a16*>(logits.data_ptr()), nv8, (float)soft_cap, (float)scale);
            else gemma4_softcap_bf16_vec8_128_kernel<false><<<grid8, 128, 0, stream>>>(reinterpret_cast<bf16x8_a16*>(logits.data_ptr()), nv8, (float)soft_cap, 1.0f);
        } else if ((n & 3) == 0 && ((reinterpret_cast<uintptr_t>(logits.data_ptr()) & 7) == 0)) {
            int64_t nv = n / 4;
            int grid = static_cast<int>(std::min<int64_t>((nv + 127) / 128, 2160));
            if (has_scale) gemma4_softcap_bf16_vec4_128_kernel<true><<<grid, 128, 0, stream>>>(reinterpret_cast<bf16x4_a8*>(logits.data_ptr()), nv, (float)soft_cap, (float)scale);
            else gemma4_softcap_bf16_vec4_128_kernel<false><<<grid, 128, 0, stream>>>(reinterpret_cast<bf16x4_a8*>(logits.data_ptr()), nv, (float)soft_cap, 1.0f);
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

static torch::Tensor s_greedy_val_ws[8];
static torch::Tensor s_greedy_idx_ws[8];
static std::mutex s_greedy_ws_mu;

torch::Tensor fast_greedy_argmax(torch::Tensor logits) {
    int B = logits.size(0), V = logits.size(1);
    auto out = torch::empty({B}, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
    if (B == 0) return out;
    bool use_fine = (logits.scalar_type() == torch::kBFloat16) && (B <= 4);
    int cols_per_blk = use_fine ? 2048 : 8192;
    int num_parts = (V + cols_per_blk - 1) / cols_per_blk;
    int64_t needed = static_cast<int64_t>(B) * num_parts;
    int dev_idx = logits.device().index();
    if (dev_idx < 0 || dev_idx >= 8) dev_idx = 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap_st);
    torch::Tensor part_val, part_idx;
    if (cap_st == cudaStreamCaptureStatusNone) {
        std::lock_guard<std::mutex> lk(s_greedy_ws_mu);
        if (!s_greedy_val_ws[dev_idx].defined() || s_greedy_val_ws[dev_idx].numel() < needed) {
            int64_t alloc_n = std::max<int64_t>(needed, 512LL * 128LL);
            s_greedy_val_ws[dev_idx] = torch::empty({alloc_n}, torch::TensorOptions().dtype(torch::kFloat).device(logits.device()));
            s_greedy_idx_ws[dev_idx] = torch::empty({alloc_n}, torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
        }
        part_val = s_greedy_val_ws[dev_idx];
        part_idx = s_greedy_idx_ws[dev_idx];
    } else {
        part_val = torch::empty({needed}, torch::TensorOptions().dtype(torch::kFloat).device(logits.device()));
        part_idx = torch::empty({needed}, torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
    }
    dim3 grid1(B, num_parts);
    if (logits.scalar_type() == torch::kBFloat16) {
        if (use_fine)
            sm90_launch_pdl(gemma4_greedy_stage1_kernel<true, 1>, grid1, dim3(256), 0, stream, part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(), logits.data_ptr(), logits.stride(0), V);
        else
            sm90_launch_pdl(gemma4_greedy_stage1_kernel<true, 4>, grid1, dim3(256), 0, stream, part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(), logits.data_ptr(), logits.stride(0), V);
    } else {
        sm90_launch_pdl(gemma4_greedy_stage1_kernel<false, 4>, grid1, dim3(256), 0, stream, part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(), logits.data_ptr(), logits.stride(0), V);
    }
    sm90_launch_pdl(gemma4_greedy_stage2_kernel, dim3(B), dim3(32), 0, stream, out.data_ptr<int64_t>(), part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(), num_parts);
    return out;
}

static torch::Tensor s_topk_ws[8];
static std::mutex s_topk_ws_mu;

torch::Tensor fast_topk_topp_gumbel_sample(
    torch::Tensor logits, torch::Tensor expanded_idx_mapping, torch::Tensor temperatures,
    torch::Tensor top_k_gpu, c10::optional<torch::Tensor> top_p_gpu,
    torch::Tensor seeds_gpu, torch::Tensor pos_gpu, int max_k)
{
    int B = logits.size(0), V = logits.size(1);
    auto out = torch::empty({B}, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
    if (B == 0) return out;
    TORCH_CHECK(max_k >= 1 && max_k <= 256, "max_k must be in [1, 256]");
    bool is_bf16 = (logits.scalar_type() == torch::kBFloat16);
    bool use_prune = is_bf16 && max_k <= 64 && V <= 524288;
    int bc = use_prune ? 8192 : 4096;
    if (!use_prune) {
        while (bc < 8192 && ((V + bc - 1) / bc) * max_k > 8192) {
            bc *= 2;
        }
    }
    int num_blocks = (V + bc - 1) / bc;
    int64_t needed = (int64_t)B * num_blocks * max_k;
    int dev_idx = logits.device().index();
    if (dev_idx < 0 || dev_idx >= 8) dev_idx = 0;
    torch::Tensor stage1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap_st);
    if (cap_st == cudaStreamCaptureStatusNone) {
        std::lock_guard<std::mutex> lk(s_topk_ws_mu);
        if (!s_topk_ws[dev_idx].defined() || s_topk_ws[dev_idx].numel() < needed) {
            int64_t alloc_n = std::max<int64_t>(needed, 64LL * 128LL * 64LL);
            s_topk_ws[dev_idx] = torch::empty({alloc_n}, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
        }
        stage1 = s_topk_ws[dev_idx];
    } else {
        stage1 = torch::empty({needed}, torch::TensorOptions().dtype(torch::kInt64).device(logits.device()));
    }
    dim3 grid1(B, num_blocks);
    int total_cands = num_blocks * max_k;
    if (use_prune) {
        sm90_launch_pdl(gemma4_topk_stage1_bf16_prune_kernel<8192, 13>, grid1, dim3(256), 0, stream,
            reinterpret_cast<uint64_t*>(stage1.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), logits.stride(0),
            top_k_gpu.data_ptr<int32_t>(), expanded_idx_mapping.data_ptr<int32_t>(), V, max_k);
        if (total_cands <= 1024) {
            sm90_launch_pdl(gemma4_topk_topp_gumbel_stage2_prune_kernel<4>, dim3(B), dim3(256), 0, stream,
                out.data_ptr<int64_t>(), reinterpret_cast<const uint64_t*>(stage1.data_ptr()),
                expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
                top_k_gpu.data_ptr<int32_t>(), top_p_gpu.has_value() ? top_p_gpu->data_ptr<float>() : nullptr,
                seeds_gpu.data_ptr<int64_t>(), pos_gpu.data_ptr<int64_t>(), num_blocks, max_k);
        } else if (total_cands <= 2048) {
            sm90_launch_pdl(gemma4_topk_topp_gumbel_stage2_prune_kernel<8>, dim3(B), dim3(256), 0, stream,
                out.data_ptr<int64_t>(), reinterpret_cast<const uint64_t*>(stage1.data_ptr()),
                expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
                top_k_gpu.data_ptr<int32_t>(), top_p_gpu.has_value() ? top_p_gpu->data_ptr<float>() : nullptr,
                seeds_gpu.data_ptr<int64_t>(), pos_gpu.data_ptr<int64_t>(), num_blocks, max_k);
        } else {
            sm90_launch_pdl(gemma4_topk_topp_gumbel_stage2_prune_kernel<16>, dim3(B), dim3(256), 0, stream,
                out.data_ptr<int64_t>(), reinterpret_cast<const uint64_t*>(stage1.data_ptr()),
                expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
                top_k_gpu.data_ptr<int32_t>(), top_p_gpu.has_value() ? top_p_gpu->data_ptr<float>() : nullptr,
                seeds_gpu.data_ptr<int64_t>(), pos_gpu.data_ptr<int64_t>(), num_blocks, max_k);
        }
        return out;
    }
    if (is_bf16 && B == 1) {
        const __nv_bfloat16* lptr = reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr());
        if (bc == 4096) gemma4_topk_stage1_bf16_auto_kernel<4096><<<grid1, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(stage1.data_ptr()), lptr, logits.stride(0),
            expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
            top_k_gpu.data_ptr<int32_t>(), V, max_k);
        else gemma4_topk_stage1_bf16_auto_kernel<8192><<<grid1, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(stage1.data_ptr()), lptr, logits.stride(0),
            expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
            top_k_gpu.data_ptr<int32_t>(), V, max_k);
    } else if (is_bf16) {
        if (bc == 4096) gemma4_topk_stage1_kernel<true, 4096><<<grid1, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(stage1.data_ptr()), logits.data_ptr(), logits.stride(0),
            expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
            top_k_gpu.data_ptr<int32_t>(), V, max_k);
        else gemma4_topk_stage1_kernel<true, 8192><<<grid1, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(stage1.data_ptr()), logits.data_ptr(), logits.stride(0),
            expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
            top_k_gpu.data_ptr<int32_t>(), V, max_k);
    } else {
        if (bc == 4096) gemma4_topk_stage1_kernel<false, 4096><<<grid1, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(stage1.data_ptr()), logits.data_ptr(), logits.stride(0),
            expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
            top_k_gpu.data_ptr<int32_t>(), V, max_k);
        else gemma4_topk_stage1_kernel<false, 8192><<<grid1, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(stage1.data_ptr()), logits.data_ptr(), logits.stride(0),
            expanded_idx_mapping.data_ptr<int32_t>(), temperatures.data_ptr<float>(),
            top_k_gpu.data_ptr<int32_t>(), V, max_k);
    }
    if (total_cands <= 2048) {
        gemma4_topk_topp_gumbel_stage2_kernel<8><<<B, 256, 0, stream>>>(
            out.data_ptr<int64_t>(), reinterpret_cast<const uint64_t*>(stage1.data_ptr()),
            expanded_idx_mapping.data_ptr<int32_t>(), top_k_gpu.data_ptr<int32_t>(),
            top_p_gpu.has_value() ? top_p_gpu->data_ptr<float>() : nullptr,
            seeds_gpu.data_ptr<int64_t>(), pos_gpu.data_ptr<int64_t>(), num_blocks, max_k);
    } else if (total_cands <= 4096) {
        gemma4_topk_topp_gumbel_stage2_kernel<16><<<B, 256, 0, stream>>>(
            out.data_ptr<int64_t>(), reinterpret_cast<const uint64_t*>(stage1.data_ptr()),
            expanded_idx_mapping.data_ptr<int32_t>(), top_k_gpu.data_ptr<int32_t>(),
            top_p_gpu.has_value() ? top_p_gpu->data_ptr<float>() : nullptr,
            seeds_gpu.data_ptr<int64_t>(), pos_gpu.data_ptr<int64_t>(), num_blocks, max_k);
    } else {
        gemma4_topk_topp_gumbel_stage2_kernel<32><<<B, 256, 0, stream>>>(
            out.data_ptr<int64_t>(), reinterpret_cast<const uint64_t*>(stage1.data_ptr()),
            expanded_idx_mapping.data_ptr<int32_t>(), top_k_gpu.data_ptr<int32_t>(),
            top_p_gpu.has_value() ? top_p_gpu->data_ptr<float>() : nullptr,
            seeds_gpu.data_ptr<int64_t>(), pos_gpu.data_ptr<int64_t>(), num_blocks, max_k);
    }
    return out;
}
"""

_CPP_SRC = r"""
void launch_paged_fa2_hd512(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q, double scale);
void launch_paged_fa2_hd512_bm16(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q, double scale);
void launch_paged_fa2_hd256_bm32(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q,
    double scale, int sliding_window_val);
void launch_paged_fa2_hd256_bm64_32(
    torch::Tensor out, torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, torch::Tensor cu_seqlens_q,
    double scale, int sliding_window_val);
void launch_paged_decode_mma_splitk(
    torch::Tensor out,
    torch::Tensor segm_out, torch::Tensor segm_max, torch::Tensor segm_expsum,
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor seqused_k, c10::optional<torch::Tensor> cu_seqlens_q,
    double scale, int sliding_window_val, int num_splits);
int dispatch_sm80_fast_attn(
    torch::Tensor out,
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cu_seqlens_q,
    int max_seqlen_q,
    torch::Tensor seqused_k,
    double scale,
    int win_left,
    int win_right,
    torch::Tensor block_table,
    c10::optional<torch::Tensor> segm_out_opt,
    c10::optional<torch::Tensor> segm_max_opt,
    c10::optional<torch::Tensor> segm_expsum_opt,
    int num_decode_reqs,
    c10::optional<torch::Tensor> cu_seqlens_q_prefill_opt,
    double keqv_inv_k_norm);
bool launch_reshape_and_cache_bf16(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping);
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
                "launch_paged_fa2_hd512_bm16",
                "launch_paged_fa2_hd256_bm32",
                "launch_paged_fa2_hd256_bm64_32",
                "launch_paged_decode_mma_splitk",
                "dispatch_sm80_fast_attn",
                "launch_reshape_and_cache_bf16",
                "post_attn_pre_ffn_norms",
                "post_ffn_cross_layer_norms",
                "skinny_gate_up_gelu",
                "skinny_gemv",
                "fused_softcap_inplace",
                "fast_greedy_argmax",
                "fast_topk_topp_gumbel_sample",
            ],
            extra_cuda_cflags=["-O3", "-arch=sm_90a" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9) else "-arch=sm_80"],
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
        and k_cache.size(1) % 16 == 0
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


def paged_fa2_hd512_bm16(
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
        and k_cache.size(1) % 16 == 0
        and block_table.stride(1) == 1
    )
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if seqused_k.dtype != torch.int32 or not seqused_k.is_contiguous():
        seqused_k = seqused_k.to(torch.int32).contiguous()
    if cu_seqlens_q.dtype != torch.int32 or not cu_seqlens_q.is_contiguous():
        cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    get_gemma4_sm80_ext().launch_paged_fa2_hd512_bm16(
        out, q, k_cache, v_cache, block_table, seqused_k, cu_seqlens_q, float(scale)
    )


def paged_fa2_hd256_bm32(
    out: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    scale: float,
    sliding_window_val: int,
) -> None:
    assert (
        q.stride(2) == 1
        and out.stride(2) == 1
        and q.stride(1) == 256
        and out.stride(1) == 256
        and k_cache.stride(3) == 1
        and v_cache.stride(3) == 1
        and k_cache.size(1) % 16 == 0
        and block_table.stride(1) == 1
    )
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if seqused_k.dtype != torch.int32 or not seqused_k.is_contiguous():
        seqused_k = seqused_k.to(torch.int32).contiguous()
    if cu_seqlens_q.dtype != torch.int32 or not cu_seqlens_q.is_contiguous():
        cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    get_gemma4_sm80_ext().launch_paged_fa2_hd256_bm32(
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seqused_k,
        cu_seqlens_q,
        float(scale),
        int(sliding_window_val),
    )


def paged_fa2_hd256_bm64_32(
    out: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    scale: float,
    sliding_window_val: int,
) -> None:
    assert (
        q.stride(2) == 1
        and out.stride(2) == 1
        and q.stride(1) == 256
        and out.stride(1) == 256
        and k_cache.stride(3) == 1
        and v_cache.stride(3) == 1
        and k_cache.size(1) % 16 == 0
        and block_table.stride(1) == 1
    )
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if seqused_k.dtype != torch.int32 or not seqused_k.is_contiguous():
        seqused_k = seqused_k.to(torch.int32).contiguous()
    if cu_seqlens_q.dtype != torch.int32 or not cu_seqlens_q.is_contiguous():
        cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    get_gemma4_sm80_ext().launch_paged_fa2_hd256_bm64_32(
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seqused_k,
        cu_seqlens_q,
        float(scale),
        int(sliding_window_val),
    )


def paged_decode_mma_splitk(
    out: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    segm_out: torch.Tensor,
    segm_max: torch.Tensor,
    segm_expsum: torch.Tensor,
    scale: float,
    sliding_window_val: int,
    num_splits: int,
    cu_seqlens_q: torch.Tensor | None = None,
) -> None:
    num_seqs = seqused_k.size(0)
    total_q_heads = q.size(1)
    head_size = q.size(2)
    nkv = k_cache.size(2)
    if cu_seqlens_q is None:
        assert num_seqs == q.size(0)
    assert (
        head_size in (256, 512)
        and total_q_heads == nkv * 8
        and q.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
        and v_cache.dtype == torch.bfloat16
        and q.stride(2) == 1
        and out.stride(2) == 1
        and k_cache.stride(3) == 1
        and v_cache.stride(3) == 1
        and k_cache.size(1) % 16 == 0
        and block_table.stride(1) == 1
        and segm_out.dtype == torch.float32
        and segm_max.dtype == torch.float32
        and segm_expsum.dtype == torch.float32
        and segm_out.is_contiguous()
        and segm_max.is_contiguous()
        and segm_expsum.is_contiguous()
        and segm_out.numel() >= num_seqs * total_q_heads * num_splits * head_size
        and segm_max.numel() >= num_seqs * total_q_heads * num_splits
        and segm_expsum.numel() >= num_seqs * total_q_heads * num_splits
    )
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if seqused_k.dtype != torch.int32 or not seqused_k.is_contiguous():
        seqused_k = seqused_k.to(torch.int32).contiguous()
    if cu_seqlens_q is not None and (
        cu_seqlens_q.dtype != torch.int32 or not cu_seqlens_q.is_contiguous()
    ):
        cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    get_gemma4_sm80_ext().launch_paged_decode_mma_splitk(
        out,
        segm_out,
        segm_max,
        segm_expsum,
        q,
        k_cache,
        v_cache,
        block_table,
        seqused_k,
        cu_seqlens_q,
        float(scale),
        int(sliding_window_val),
        int(num_splits),
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
