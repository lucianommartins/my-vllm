# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90a (NVIDIA H100) zero-spill Hopper PDL paged sliding-window decode attention
for Gemma 4 12B (40x sliding layers: hd=256, nq=16, nkv=8, GQA=2, window=1024).

Covers BOTH:
  - bs >= 12: Zero-spill 32-warp/CTA single-pass kernel (128 CTAs @ bs=16, 0 split-K,
    early griddepcontrol.launch_dependents before final reduction -> overlaps with o_proj).
  - bs = 1..11: Zero-spill PDL split-K kernel with historical-KV-first HBM3 prefetch
    BEFORE griddepcontrol.wait (all tokens pos < seq_len - 1 are immutable!) +
    zero-barrier 1-warp PDL reduce kernel (eliminating all 80 non-PDL Triton/FA3
    boundaries per bs=1 decode step).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace gemma4_sliding_sm90 {

static constexpr int MAX_SPLITK_BS = 11;
static constexpr int MAX_SPLITK_SPLITS = 32;
static constexpr float LOG2E_F = 1.4426950408889634074f;
static constexpr float NEG_INF_SENTINEL = -1e30f;

static torch::Tensor g_segm_o;
static torch::Tensor g_segm_m;
static torch::Tensor g_segm_l;
static bool g_scratch_init = false;

static inline void ensure_scratch() {
    if (g_scratch_init) return;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    g_segm_o = torch::empty({MAX_SPLITK_BS, 16, MAX_SPLITK_SPLITS, 256}, opts);
    g_segm_m = torch::empty({MAX_SPLITK_BS, 16, MAX_SPLITK_SPLITS}, opts);
    g_segm_l = torch::empty({MAX_SPLITK_BS, 16, MAX_SPLITK_SPLITS}, opts);
    g_scratch_init = true;
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

template <typename P0, typename P1, typename P2>
__device__ __forceinline__ void pdl_wait_3ptrs(P0& p0, P1& p1, P2& p2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n" : "+l"(p0), "+l"(p1), "+l"(p2) :: "memory");
#endif
}

template <typename KernelT, typename... Args>
static inline void launch_pdl(
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

__device__ __forceinline__ uint4 ld_cs_u4(const void* ptr) {
    uint4 ret;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint4 ld_ca_u4(const void* ptr) {
    uint4 ret;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void prefetch_l1(const void* ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

__device__ __forceinline__ void st_u4(void* ptr, uint4 val) {
    *reinterpret_cast<uint4*>(ptr) = val;
}

__device__ __forceinline__ float dot_bf16x8(uint4 a, uint4 b) {
    float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.x));
    float2 b0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.x));
    float s = fmaf(a0.x, b0.x, a0.y * b0.y);
    float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.y));
    float2 b1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.y));
    s = fmaf(a1.x, b1.x, fmaf(a1.y, b1.y, s));
    float2 a2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.z));
    float2 b2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.z));
    s = fmaf(a2.x, b2.x, fmaf(a2.y, b2.y, s));
    float2 a3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.w));
    float2 b3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.w));
    s = fmaf(a3.x, b3.x, fmaf(a3.y, b3.y, s));
    return s;
}

__device__ __forceinline__ void acc_bf16x8(float o[8], float alpha, float p, uint4 v) {
    float2 v0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
    o[0] = fmaf(o[0], alpha, p * v0.x);
    o[1] = fmaf(o[1], alpha, p * v0.y);
    float2 v1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
    o[2] = fmaf(o[2], alpha, p * v1.x);
    o[3] = fmaf(o[3], alpha, p * v1.y);
    float2 v2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.z));
    o[4] = fmaf(o[4], alpha, p * v2.x);
    o[5] = fmaf(o[5], alpha, p * v2.y);
    float2 v3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.w));
    o[6] = fmaf(o[6], alpha, p * v3.x);
    o[7] = fmaf(o[7], alpha, p * v3.y);
}

__device__ __forceinline__ uint4 pack_bf16x8(const float in[8]) {
    uint4 raw;
    __nv_bfloat162 v0 = __floats2bfloat162_rn(in[0], in[1]);
    __nv_bfloat162 v1 = __floats2bfloat162_rn(in[2], in[3]);
    __nv_bfloat162 v2 = __floats2bfloat162_rn(in[4], in[5]);
    __nv_bfloat162 v3 = __floats2bfloat162_rn(in[6], in[7]);
    raw.x = *reinterpret_cast<const uint32_t*>(&v0);
    raw.y = *reinterpret_cast<const uint32_t*>(&v1);
    raw.z = *reinterpret_cast<const uint32_t*>(&v2);
    raw.w = *reinterpret_cast<const uint32_t*>(&v3);
    return raw;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

// =========================================================================
// 1. Single-pass 32-warp/CTA zero-spill PDL kernel (bs >= 12)
// =========================================================================
template <int NWARPS, bool EARLY_V = true, bool AFTER_LOOP_PDL = false>
__global__ void __launch_bounds__(NWARPS * 32, 1)
sliding_single_32w_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ Out,
    int64_t q_stride0, int64_t q_stride1,
    int64_t o_stride0, int64_t o_stride1,
    int64_t kv_stride0, int64_t kv_stride1, int64_t kv_stride2,
    int64_t bt_stride0,
    int block_size,
    int window_left,
    float scale_log2e)
{
    extern __shared__ float smem[];
    float* sm_o = smem;
    float* sm_m = sm_o + 2 * NWARPS * 256;
    float* sm_l = sm_m + 2 * NWARPS;

    const int seq_idx = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;

    const int seq_len = seq_lens[seq_idx];
    const int q_head0 = kv_head * 2;

    if (seq_len <= 0) {
        pdl_launch_dependents();
        if (warp_id == 0) {
            uint4 z = {0, 0, 0, 0};
            st_u4(Out + seq_idx * o_stride0 + q_head0 * o_stride1 + lane * 8, z);
            st_u4(Out + seq_idx * o_stride0 + (q_head0 + 1) * o_stride1 + lane * 8, z);
        }
        return;
    }

    int start_pos = 0;
    if (window_left >= 0) {
        int s = seq_len - 1 - window_left;
        if (s > 0) start_pos = s;
    }

    const int32_t* bt_row = block_table + seq_idx * bt_stride0;
    const int64_t head_lane_off = kv_head * kv_stride2 + lane * 8;
    const int blk_mask = block_size - 1;
    const int blk_shift = __ffs(block_size) - 1;

    // Historical-K L1 prefetch BEFORE waiting on qkv_norm_rope_cache!
    {
        const int p0 = start_pos + warp_id;
        const int p1 = p0 + NWARPS;
        if (p0 < seq_len - 1) {
            const int64_t o0 = static_cast<int64_t>(bt_row[p0 >> blk_shift]) * kv_stride0 + (p0 & blk_mask) * kv_stride1;
            prefetch_l1(K_cache + head_lane_off + o0);
        }
        if (p1 < seq_len - 1) {
            const int64_t o1 = static_cast<int64_t>(bt_row[p1 >> blk_shift]) * kv_stride0 + (p1 & blk_mask) * kv_stride1;
            prefetch_l1(K_cache + head_lane_off + o1);
        }
    }

    __shared__ uint4 sm_q[2][32];
    pdl_wait_3ptrs(Q, K_cache, V_cache);
    if (warp_id == 0) {
        sm_q[0][lane] = ld_ca_u4(Q + seq_idx * q_stride0 + q_head0 * q_stride1 + lane * 8);
        sm_q[1][lane] = ld_ca_u4(Q + seq_idx * q_stride0 + (q_head0 + 1) * q_stride1 + lane * 8);
    }
    __syncthreads();

    const __nv_bfloat16* k_base = K_cache + head_lane_off;
    const __nv_bfloat16* v_base = V_cache + head_lane_off;

    float m0 = NEG_INF_SENTINEL, l0 = 0.0f;
    float m1 = NEG_INF_SENTINEL, l1 = 0.0f;
    float o0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float o1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    int pos = start_pos + warp_id;
    for (; pos + NWARPS < seq_len; pos += 2 * NWARPS) {
        const int pos2 = pos + NWARPS;
        const int64_t off1 = static_cast<int64_t>(bt_row[pos >> blk_shift]) * kv_stride0 + (pos & blk_mask) * kv_stride1;
        const int64_t off2 = static_cast<int64_t>(bt_row[pos2 >> blk_shift]) * kv_stride0 + (pos2 & blk_mask) * kv_stride1;

        uint4 k1 = ld_cs_u4(k_base + off1);
        uint4 k2 = ld_cs_u4(k_base + off2);
        uint4 v1, v2;
        if constexpr (EARLY_V) {
            v1 = ld_cs_u4(v_base + off1);
            v2 = ld_cs_u4(v_base + off2);
        }

        float s0_a = warp_reduce_sum(dot_bf16x8(sm_q[0][lane], k1)) * scale_log2e;
        float s1_a = warp_reduce_sum(dot_bf16x8(sm_q[1][lane], k1)) * scale_log2e;
        float s0_b = warp_reduce_sum(dot_bf16x8(sm_q[0][lane], k2)) * scale_log2e;
        float s1_b = warp_reduce_sum(dot_bf16x8(sm_q[1][lane], k2)) * scale_log2e;

        if constexpr (!EARLY_V) {
            v1 = ld_cs_u4(v_base + off1);
            v2 = ld_cs_u4(v_base + off2);
        }

        float m0_new = fmaxf(m0, fmaxf(s0_a, s0_b));
        float a0 = (m0 > NEG_INF_SENTINEL * 0.5f) ? exp2f(m0 - m0_new) : 0.0f;
        float p0_a = exp2f(s0_a - m0_new);
        float p0_b = exp2f(s0_b - m0_new);
        l0 = fmaf(l0, a0, p0_a + p0_b);
        m0 = m0_new;

        float m1_new = fmaxf(m1, fmaxf(s1_a, s1_b));
        float a1 = (m1 > NEG_INF_SENTINEL * 0.5f) ? exp2f(m1 - m1_new) : 0.0f;
        float p1_a = exp2f(s1_a - m1_new);
        float p1_b = exp2f(s1_b - m1_new);
        l1 = fmaf(l1, a1, p1_a + p1_b);
        m1 = m1_new;

        acc_bf16x8(o0, a0, p0_a, v1);
        acc_bf16x8(o0, 1.0f, p0_b, v2);
        acc_bf16x8(o1, a1, p1_a, v1);
        acc_bf16x8(o1, 1.0f, p1_b, v2);
    }

    if (pos < seq_len) {
        const int64_t off1 = static_cast<int64_t>(bt_row[pos >> blk_shift]) * kv_stride0 + (pos & blk_mask) * kv_stride1;
        uint4 k1 = ld_cs_u4(k_base + off1);
        uint4 v1 = ld_cs_u4(v_base + off1);
        float s0 = warp_reduce_sum(dot_bf16x8(sm_q[0][lane], k1)) * scale_log2e;
        float s1 = warp_reduce_sum(dot_bf16x8(sm_q[1][lane], k1)) * scale_log2e;

        float m0_new = fmaxf(m0, s0);
        float a0 = (m0 > NEG_INF_SENTINEL * 0.5f) ? exp2f(m0 - m0_new) : 0.0f;
        float p0 = exp2f(s0 - m0_new);
        l0 = fmaf(l0, a0, p0);
        m0 = m0_new;

        float m1_new = fmaxf(m1, s1);
        float a1 = (m1 > NEG_INF_SENTINEL * 0.5f) ? exp2f(m1 - m1_new) : 0.0f;
        float p1 = exp2f(s1 - m1_new);
        l1 = fmaf(l1, a1, p1);
        m1 = m1_new;

        acc_bf16x8(o0, a0, p0, v1);
        acc_bf16x8(o1, a1, p1, v1);
    }

    if constexpr (AFTER_LOOP_PDL) pdl_launch_dependents();
    float* dst0 = sm_o + (0 * NWARPS + warp_id) * 256 + lane * 8;
    float* dst1 = sm_o + (1 * NWARPS + warp_id) * 256 + lane * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst0[i] = o0[i];
        dst1[i] = o1[i];
    }
    if (lane == 0) {
        sm_m[0 * NWARPS + warp_id] = m0;
        sm_l[0 * NWARPS + warp_id] = l0;
        sm_m[1 * NWARPS + warp_id] = m1;
        sm_l[1 * NWARPS + warp_id] = l1;
    }
    __syncthreads();

    if (warp_id < 16) {
        const int g = warp_id >> 3;
        const int w0 = (warp_id & 7) * 4;
        float m_max = NEG_INF_SENTINEL;
        #pragma unroll
        for (int k = 0; k < 4; ++k) m_max = fmaxf(m_max, sm_m[g * NWARPS + w0 + k]);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float l_sum = 0.0f;
        if (m_max > NEG_INF_SENTINEL * 0.5f) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float mw = sm_m[g * NWARPS + w0 + k];
                if (mw > NEG_INF_SENTINEL * 0.5f) {
                    float sc = exp2f(mw - m_max);
                    l_sum += sm_l[g * NWARPS + w0 + k] * sc;
                    const float* src = sm_o + (g * NWARPS + w0 + k) * 256 + lane * 8;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[i] = fmaf(src[i], sc, acc[i]);
                }
            }
        }
        float* d = sm_o + (g * NWARPS + w0) * 256 + lane * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) d[i] = acc[i];
        if (lane == 0) {
            sm_m[g * NWARPS + w0] = m_max;
            sm_l[g * NWARPS + w0] = l_sum;
        }
    }
    __syncthreads();

    // Wake o_proj early so its CTAs launch & prefetch stage-0 weights during final reduction!
    if constexpr (!AFTER_LOOP_PDL) pdl_launch_dependents();

    if (warp_id < 2) {
        const int g = warp_id;
        float m_max = NEG_INF_SENTINEL;
        #pragma unroll
        for (int grp = 0; grp < 8; ++grp) m_max = fmaxf(m_max, sm_m[g * NWARPS + grp * 4]);
        float out_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (m_max > NEG_INF_SENTINEL * 0.5f) {
            float l_sum = 0.0f;
            #pragma unroll
            for (int grp = 0; grp < 8; ++grp) {
                float mw = sm_m[g * NWARPS + grp * 4];
                if (mw > NEG_INF_SENTINEL * 0.5f) {
                    float sc = exp2f(mw - m_max);
                    l_sum += sm_l[g * NWARPS + grp * 4] * sc;
                    const float* src = sm_o + (g * NWARPS + grp * 4) * 256 + lane * 8;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) out_f[i] = fmaf(src[i], sc, out_f[i]);
                }
            }
            float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) out_f[i] *= inv_l;
        }
        st_u4(Out + seq_idx * o_stride0 + (q_head0 + g) * o_stride1 + lane * 8, pack_bf16x8(out_f));
    }
}

// =========================================================================
// 2. Zero-spill PDL Split-K kernel with historical-KV-first prefetch (bs=1..11)
//    All tokens pos < seq_len - 1 are immutable in KV cache from prior steps!
// =========================================================================
template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32)
sliding_splitk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ segm_o,
    float* __restrict__ segm_m,
    float* __restrict__ segm_l,
    int64_t q_stride0, int64_t q_stride1,
    int64_t kv_stride0, int64_t kv_stride1, int64_t kv_stride2,
    int64_t bt_stride0,
    int block_size,
    int window_left,
    int num_splits,
    float scale_log2e)
{
    const int seq_idx = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    int start_pos = 0;
    if (window_left >= 0) {
        int s = seq_len - 1 - window_left;
        if (s > 0) start_pos = s;
    }
    const int total_toks = seq_len - start_pos;
    if (total_toks <= 0) return;

    const int tok_per = total_toks / num_splits;
    const int tok_rem = total_toks - tok_per * num_splits;
    const int my_toks = tok_per + (split_idx < tok_rem ? 1 : 0);
    if (my_toks <= 0) return;
    const int my_start = start_pos + split_idx * tok_per + min(split_idx, tok_rem);
    const int my_end = my_start + my_toks;

    const int q_head0 = kv_head * 2;
    const int32_t* bt_row = block_table + seq_idx * bt_stride0;
    const int64_t head_lane_off = kv_head * kv_stride2 + lane * 8;
    const int blk_mask = block_size - 1;
    const int blk_shift = __ffs(block_size) - 1;

    // Historical-KV-first HBM3 prefetch BEFORE waiting on qkv_norm_rope_cache!
    // Tokens strictly before (seq_len - 1) were written in previous steps and never change.
    const int pos0 = my_start + warp_id;
    const int pos1 = pos0 + NWARPS;
    const bool has0 = (pos0 < my_end);
    const bool has1 = (pos1 < my_end);
    const bool pre0 = has0 && (pos0 < seq_len - 1);
    const bool pre1 = has1 && (pos1 < seq_len - 1);
    int64_t off0 = 0, off1 = 0;
    uint4 k0_pre = {0, 0, 0, 0}, k1_pre = {0, 0, 0, 0};
    if (has0) {
        off0 = static_cast<int64_t>(bt_row[pos0 >> blk_shift]) * kv_stride0 + (pos0 & blk_mask) * kv_stride1;
        if (pre0) {
            k0_pre = ld_cs_u4(K_cache + head_lane_off + off0);
            prefetch_l1(V_cache + head_lane_off + off0);
        }
    }
    if (has1) {
        off1 = static_cast<int64_t>(bt_row[pos1 >> blk_shift]) * kv_stride0 + (pos1 & blk_mask) * kv_stride1;
        if (pre1) {
            k1_pre = ld_cs_u4(K_cache + head_lane_off + off1);
            prefetch_l1(V_cache + head_lane_off + off1);
        }
    }

    __shared__ uint4 sm_q[2][32];
    __shared__ float sm_o[2 * NWARPS * 256];
    __shared__ float sm_m[2 * NWARPS];
    __shared__ float sm_l[2 * NWARPS];

    pdl_wait_3ptrs(Q, K_cache, V_cache);
    if (warp_id == 0) {
        sm_q[0][lane] = ld_ca_u4(Q + seq_idx * q_stride0 + q_head0 * q_stride1 + lane * 8);
        sm_q[1][lane] = ld_ca_u4(Q + seq_idx * q_stride0 + (q_head0 + 1) * q_stride1 + lane * 8);
    }
    __syncthreads();

    const uint4 q0_v = sm_q[0][lane];
    const uint4 q1_v = sm_q[1][lane];
    const __nv_bfloat16* k_base = K_cache + head_lane_off;
    const __nv_bfloat16* v_base = V_cache + head_lane_off;

    float m0 = NEG_INF_SENTINEL, l0 = 0.0f;
    float m1 = NEG_INF_SENTINEL, l1 = 0.0f;
    float o0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float o1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    if (has1) {
        uint4 k1 = pre0 ? k0_pre : ld_cs_u4(k_base + off0);
        uint4 k2 = pre1 ? k1_pre : ld_cs_u4(k_base + off1);
        uint4 v1 = ld_cs_u4(v_base + off0);
        uint4 v2 = ld_cs_u4(v_base + off1);
        float s0_a = warp_reduce_sum(dot_bf16x8(q0_v, k1)) * scale_log2e;
        float s1_a = warp_reduce_sum(dot_bf16x8(q1_v, k1)) * scale_log2e;
        float s0_b = warp_reduce_sum(dot_bf16x8(q0_v, k2)) * scale_log2e;
        float s1_b = warp_reduce_sum(dot_bf16x8(q1_v, k2)) * scale_log2e;

        m0 = fmaxf(s0_a, s0_b);
        float p0_a = exp2f(s0_a - m0);
        float p0_b = exp2f(s0_b - m0);
        l0 = p0_a + p0_b;

        m1 = fmaxf(s1_a, s1_b);
        float p1_a = exp2f(s1_a - m1);
        float p1_b = exp2f(s1_b - m1);
        l1 = p1_a + p1_b;

        acc_bf16x8(o0, 0.0f, p0_a, v1);
        acc_bf16x8(o0, 1.0f, p0_b, v2);
        acc_bf16x8(o1, 0.0f, p1_a, v1);
        acc_bf16x8(o1, 1.0f, p1_b, v2);

        int pos = pos0 + 2 * NWARPS;
        for (; pos + NWARPS < my_end; pos += 2 * NWARPS) {
            const int pos2 = pos + NWARPS;
            const int64_t oa = static_cast<int64_t>(bt_row[pos >> blk_shift]) * kv_stride0 + (pos & blk_mask) * kv_stride1;
            const int64_t ob = static_cast<int64_t>(bt_row[pos2 >> blk_shift]) * kv_stride0 + (pos2 & blk_mask) * kv_stride1;

            uint4 ka = ld_cs_u4(k_base + oa);
            uint4 kb = ld_cs_u4(k_base + ob);
            uint4 va = ld_cs_u4(v_base + oa);
            uint4 vb = ld_cs_u4(v_base + ob);

            float r0_a = warp_reduce_sum(dot_bf16x8(q0_v, ka)) * scale_log2e;
            float r1_a = warp_reduce_sum(dot_bf16x8(q1_v, ka)) * scale_log2e;
            float r0_b = warp_reduce_sum(dot_bf16x8(q0_v, kb)) * scale_log2e;
            float r1_b = warp_reduce_sum(dot_bf16x8(q1_v, kb)) * scale_log2e;

            float m0_new = fmaxf(m0, fmaxf(r0_a, r0_b));
            float a0 = exp2f(m0 - m0_new);
            float q0_a = exp2f(r0_a - m0_new);
            float q0_b = exp2f(r0_b - m0_new);
            l0 = fmaf(l0, a0, q0_a + q0_b);
            m0 = m0_new;

            float m1_new = fmaxf(m1, fmaxf(r1_a, r1_b));
            float a1 = exp2f(m1 - m1_new);
            float q1_a = exp2f(r1_a - m1_new);
            float q1_b = exp2f(r1_b - m1_new);
            l1 = fmaf(l1, a1, q1_a + q1_b);
            m1 = m1_new;

            acc_bf16x8(o0, a0, q0_a, va);
            acc_bf16x8(o0, 1.0f, q0_b, vb);
            acc_bf16x8(o1, a1, q1_a, va);
            acc_bf16x8(o1, 1.0f, q1_b, vb);
        }

        if (pos < my_end) {
            const int64_t oa = static_cast<int64_t>(bt_row[pos >> blk_shift]) * kv_stride0 + (pos & blk_mask) * kv_stride1;
            uint4 ka = ld_cs_u4(k_base + oa);
            uint4 va = ld_cs_u4(v_base + oa);
            float s0 = warp_reduce_sum(dot_bf16x8(q0_v, ka)) * scale_log2e;
            float s1 = warp_reduce_sum(dot_bf16x8(q1_v, ka)) * scale_log2e;

            float m0_new = fmaxf(m0, s0);
            float a0 = exp2f(m0 - m0_new);
            float p0 = exp2f(s0 - m0_new);
            l0 = fmaf(l0, a0, p0);
            m0 = m0_new;

            float m1_new = fmaxf(m1, s1);
            float a1 = exp2f(m1 - m1_new);
            float p1 = exp2f(s1 - m1_new);
            l1 = fmaf(l1, a1, p1);
            m1 = m1_new;

            acc_bf16x8(o0, a0, p0, va);
            acc_bf16x8(o1, a1, p1, va);
        }
    } else if (has0) {
        uint4 k1 = pre0 ? k0_pre : ld_cs_u4(k_base + off0);
        uint4 v1 = ld_cs_u4(v_base + off0);
        float s0 = warp_reduce_sum(dot_bf16x8(q0_v, k1)) * scale_log2e;
        float s1 = warp_reduce_sum(dot_bf16x8(q1_v, k1)) * scale_log2e;
        m0 = s0; l0 = 1.0f;
        m1 = s1; l1 = 1.0f;
        acc_bf16x8(o0, 0.0f, 1.0f, v1);
        acc_bf16x8(o1, 0.0f, 1.0f, v1);
    }

    float* dst0 = sm_o + (0 * NWARPS + warp_id) * 256 + lane * 8;
    float* dst1 = sm_o + (1 * NWARPS + warp_id) * 256 + lane * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst0[i] = o0[i];
        dst1[i] = o1[i];
    }
    if (lane == 0) {
        sm_m[0 * NWARPS + warp_id] = m0;
        sm_l[0 * NWARPS + warp_id] = l0;
        sm_m[1 * NWARPS + warp_id] = m1;
        sm_l[1 * NWARPS + warp_id] = l1;
    }
    __syncthreads();

    // Wake 1-warp reduce kernel early!
    pdl_launch_dependents();

    if (warp_id < 2) {
        const int g = warp_id;
        float m_max = NEG_INF_SENTINEL;
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) m_max = fmaxf(m_max, sm_m[g * NWARPS + w]);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float l_sum = 0.0f;
        if (m_max > NEG_INF_SENTINEL * 0.5f) {
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                float mw = sm_m[g * NWARPS + w];
                if (mw > NEG_INF_SENTINEL * 0.5f) {
                    float sc = exp2f(mw - m_max);
                    l_sum += sm_l[g * NWARPS + w] * sc;
                    const float* src = sm_o + (g * NWARPS + w) * 256 + lane * 8;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[i] = fmaf(src[i], sc, acc[i]);
                }
            }
        }
        const int64_t ml_idx = ((static_cast<int64_t>(seq_idx) * 16 + (q_head0 + g)) * MAX_SPLITK_SPLITS) + split_idx;
        if (lane == 0) {
            segm_m[ml_idx] = m_max;
            segm_l[ml_idx] = l_sum;
        }
        float* so_dst = segm_o + ml_idx * 256 + lane * 8;
        *reinterpret_cast<float4*>(so_dst)     = make_float4(acc[0], acc[1], acc[2], acc[3]);
        *reinterpret_cast<float4*>(so_dst + 4) = make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
}

// =========================================================================
// 3. Zero-barrier 1-warp PDL reduce kernel (0 SMEM, 0 __syncthreads)
// =========================================================================
template <int D_CHUNK, int VEC>
__global__ void __launch_bounds__(32)
sliding_reduce_hd256_1w_kernel(
    __nv_bfloat16* __restrict__ Out,
    const float* __restrict__ segm_o,
    const float* __restrict__ segm_m,
    const float* __restrict__ segm_l,
    const int32_t* __restrict__ seq_lens,
    int64_t o_stride0, int64_t o_stride1,
    int window_left,
    int num_splits)
{
    static_assert(D_CHUNK == 32 * VEC, "D_CHUNK == 32 * VEC");
    const int seq_idx = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    int start_pos = 0;
    if (window_left >= 0 && seq_len > 0) {
        int s = seq_len - 1 - window_left;
        if (s > 0) start_pos = s;
    }
    const int total_toks = (seq_len > start_pos) ? (seq_len - start_pos) : 0;
    const int active_splits = min(num_splits, total_toks);

    const int d_out = blockIdx.z * D_CHUNK + lane * VEC;
    __nv_bfloat16* out_dst = Out + static_cast<int64_t>(seq_idx) * o_stride0 + static_cast<int64_t>(qh) * o_stride1 + d_out;

    auto write_zero = [&]() {
        if constexpr (VEC == 4) {
            *reinterpret_cast<uint2*>(out_dst) = make_uint2(0u, 0u);
        } else if constexpr (VEC == 2) {
            *reinterpret_cast<uint32_t*>(out_dst) = 0u;
        } else {
            *out_dst = __float2bfloat16_rn(0.0f);
        }
    };

    if (active_splits <= 0) {
        pdl_launch_dependents();
        write_zero();
        return;
    }

    const int64_t ml_base = (static_cast<int64_t>(seq_idx) * 16 + qh) * MAX_SPLITK_SPLITS;
    pdl_wait_3ptrs(segm_o, segm_m, segm_l);
    // Wake o_proj immediately after waiting so its weights prefetch during the split-K reduce!
    pdl_launch_dependents();

    float ms = (lane < active_splits) ? segm_m[ml_base + lane] : NEG_INF_SENTINEL;
    float gm = ms;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        gm = fmaxf(gm, __shfl_xor_sync(0xffffffff, gm, mask));
    }
    if (gm <= NEG_INF_SENTINEL * 0.5f) {
        write_zero();
        return;
    }

    float ws = (lane < active_splits && ms > NEG_INF_SENTINEL * 0.5f) ? exp2f(ms - gm) : 0.0f;
    float ls = (ws > 0.0f) ? (segm_l[ml_base + lane] * ws) : 0.0f;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        ls += __shfl_xor_sync(0xffffffff, ls, mask);
    }
    if (ls <= 0.0f) {
        write_zero();
        return;
    }
    const float inv_l = 1.0f / ls;

    const float* src_base = segm_o + ml_base * 256 + d_out;
    float acc[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.0f;

    for (int s = 0; s < active_splits; ++s) {
        float w = __shfl_sync(0xffffffff, ws, s);
        if (w > 0.0f) {
            const float* src = src_base + static_cast<int64_t>(s) * 256;
            if constexpr (VEC == 4) {
                float4 val = *reinterpret_cast<const float4*>(src);
                acc[0] = fmaf(val.x, w, acc[0]);
                acc[1] = fmaf(val.y, w, acc[1]);
                acc[2] = fmaf(val.z, w, acc[2]);
                acc[3] = fmaf(val.w, w, acc[3]);
            } else if constexpr (VEC == 2) {
                float2 val = *reinterpret_cast<const float2*>(src);
                acc[0] = fmaf(val.x, w, acc[0]);
                acc[1] = fmaf(val.y, w, acc[1]);
            } else {
                acc[0] = fmaf(*src, w, acc[0]);
            }
        }
    }


    if constexpr (VEC == 4) {
        __nv_bfloat162 b01 = __floats2bfloat162_rn(acc[0] * inv_l, acc[1] * inv_l);
        __nv_bfloat162 b23 = __floats2bfloat162_rn(acc[2] * inv_l, acc[3] * inv_l);
        *reinterpret_cast<uint2*>(out_dst) = make_uint2(
            *reinterpret_cast<const uint32_t*>(&b01),
            *reinterpret_cast<const uint32_t*>(&b23));
    } else if constexpr (VEC == 2) {
        __nv_bfloat162 b01 = __floats2bfloat162_rn(acc[0] * inv_l, acc[1] * inv_l);
        *reinterpret_cast<uint32_t*>(out_dst) = *reinterpret_cast<const uint32_t*>(&b01);
    } else {
        *out_dst = __float2bfloat16_rn(acc[0] * inv_l);
    }
}

void init_sliding_32w() {
    const size_t smem_bytes = (2 * 32 * 256 + 4 * 32) * sizeof(float);
    cudaFuncSetAttribute(sliding_single_32w_kernel<32, true, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaFuncSetAttribute(sliding_single_32w_kernel<32, true, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaFuncSetAttribute(sliding_single_32w_kernel<32, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    ensure_scratch();
}

void sliding_decode_hd256_gqa2_32w(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor bt, torch::Tensor seq_lens, torch::Tensor Out,
    int64_t win_left, double scale)
{
    const size_t smem_bytes = (2 * 32 * 256 + 4 * 32) * sizeof(float);
    const int64_t bs = seq_lens.size(0);
    // AFTER_LOOP_PDL gated to bs>=12 (-37.1us/40L @ bs=16, -7.3us @ bs=12; avoids +105.9us regression @ bs=9!)
    auto kfn = (bs <= 11) ? sliding_single_32w_kernel<32, true, false>
             : (bs <= 14) ? sliding_single_32w_kernel<32, true, true>
                          : sliding_single_32w_kernel<32, false, true>;
    launch_pdl(
        kfn,
        dim3(seq_lens.size(0), 8),
        dim3(32, 32),
        smem_bytes,
        at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        bt.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(Out.data_ptr()),
        Q.stride(0), Q.stride(1), Out.stride(0), Out.stride(1),
        K.stride(0), K.stride(1), K.stride(2),
        bt.stride(0), static_cast<int>(K.size(1)), static_cast<int>(win_left),
        static_cast<float>(scale) * LOG2E_F);
}

void sliding_decode_hd256_gqa2_splitk_cfg(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor bt, torch::Tensor seq_lens, torch::Tensor Out,
    int64_t win_left, double scale,
    int64_t num_splits_in, int64_t nwarps_in, int64_t d_chunk_in)
{
    ensure_scratch();
    const int bs = static_cast<int>(seq_lens.size(0));
    const int num_splits = static_cast<int>(num_splits_in);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* qp = reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr());
    auto* kp = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr());
    auto* vp = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr());
    auto* btp = bt.data_ptr<int32_t>();
    auto* sqp = seq_lens.data_ptr<int32_t>();
    auto* op = reinterpret_cast<__nv_bfloat16*>(Out.data_ptr());
    float* sop = g_segm_o.data_ptr<float>();
    float* smp = g_segm_m.data_ptr<float>();
    float* slp = g_segm_l.data_ptr<float>();
    const float sc_log2 = static_cast<float>(scale) * LOG2E_F;
    const int blk_sz = static_cast<int>(K.size(1));
    const int wl = static_cast<int>(win_left);

    if (nwarps_in == 4) {
        launch_pdl(
            sliding_splitk_kernel<4>,
            dim3(bs, 8, num_splits), dim3(32, 4), 0, stream,
            qp, kp, vp, btp, sqp, sop, smp, slp,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            bt.stride(0), blk_sz, wl, num_splits, sc_log2);
    } else if (nwarps_in == 16) {
        launch_pdl(
            sliding_splitk_kernel<16>,
            dim3(bs, 8, num_splits), dim3(32, 16), 0, stream,
            qp, kp, vp, btp, sqp, sop, smp, slp,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            bt.stride(0), blk_sz, wl, num_splits, sc_log2);
    } else {
        launch_pdl(
            sliding_splitk_kernel<8>,
            dim3(bs, 8, num_splits), dim3(32, 8), 0, stream,
            qp, kp, vp, btp, sqp, sop, smp, slp,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            bt.stride(0), blk_sz, wl, num_splits, sc_log2);
    }

    if (d_chunk_in == 32) {
        launch_pdl(
            sliding_reduce_hd256_1w_kernel<32, 1>,
            dim3(bs, 16, 8), dim3(32), 0, stream,
            op, sop, smp, slp, sqp, Out.stride(0), Out.stride(1), wl, num_splits);
    } else if (d_chunk_in == 128) {
        launch_pdl(
            sliding_reduce_hd256_1w_kernel<128, 4>,
            dim3(bs, 16, 2), dim3(32), 0, stream,
            op, sop, smp, slp, sqp, Out.stride(0), Out.stride(1), wl, num_splits);
    } else {
        launch_pdl(
            sliding_reduce_hd256_1w_kernel<64, 2>,
            dim3(bs, 16, 4), dim3(32), 0, stream,
            op, sop, smp, slp, sqp, Out.stride(0), Out.stride(1), wl, num_splits);
    }
}

void sliding_decode_hd256_gqa2_splitk(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor bt, torch::Tensor seq_lens, torch::Tensor Out,
    int64_t win_left, double scale)
{
    const int bs = static_cast<int>(seq_lens.size(0));
    int num_splits = 2;
    int nwarps = 8;
    int d_chunk = 64;
    if (bs == 1) {
        num_splits = 16; nwarps = 16; d_chunk = 32;
    } else if (bs == 2) {
        num_splits = 8; nwarps = 16; d_chunk = 64;
    } else if (bs == 3) {
        num_splits = 5; nwarps = 16; d_chunk = 64;
    } else if (bs == 4) {
        num_splits = 8; nwarps = 8; d_chunk = 64;
    } else if (bs == 5) {
        num_splits = 6; nwarps = 8; d_chunk = 64;
    } else if (bs == 6) {
        num_splits = 4; nwarps = 8; d_chunk = 128;
    } else {
        num_splits = 2; nwarps = 16; d_chunk = 128;
    }
    sliding_decode_hd256_gqa2_splitk_cfg(
        Q, K, V, bt, seq_lens, Out, win_left, scale, num_splits, nwarps, d_chunk);
}

} // namespace gemma4_sliding_sm90

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_sliding_32w", &gemma4_sliding_sm90::init_sliding_32w);
    m.def("sliding_decode_hd256_gqa2_32w", &gemma4_sliding_sm90::sliding_decode_hd256_gqa2_32w);
    m.def("sliding_decode_hd256_gqa2_splitk", &gemma4_sliding_sm90::sliding_decode_hd256_gqa2_splitk);
    m.def("sliding_decode_hd256_gqa2_splitk_cfg", &gemma4_sliding_sm90::sliding_decode_hd256_gqa2_splitk_cfg);
}
"""

_MOD = None


def _get_mod():
    global _MOD
    if _MOD is not None:
        return _MOD
    h = hashlib.sha256(_CUDA_SRC.encode("utf-8")).hexdigest()[:12]
    name = f"gemma4_sliding_decode_sm90_pdl_{h}"
    cache_root = os.environ.get(
        "VLLM_CACHE_ROOT",
        os.path.join(os.path.expanduser("~"), ".cache", "vllm"),
    )
    build_dir = os.path.join(cache_root, "torch_extensions", name)
    os.makedirs(build_dir, exist_ok=True)
    so_path = os.path.join(build_dir, f"{name}.so")
    if os.path.exists(so_path):
        spec = importlib.util.spec_from_file_location(name, so_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mod.init_sliding_32w()
        _MOD = mod
        return _MOD
    _MOD = load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=["-O3", "-arch=sm_90a", "--expt-relaxed-constexpr"],
        build_directory=build_dir,
        with_cuda=True,
        verbose=False,
    )
    _MOD.init_sliding_32w()
    return _MOD


def sliding_decode_hd256_gqa2_32w(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    window_left: int,
    scale: float,
) -> None:
    _get_mod().sliding_decode_hd256_gqa2_32w(
        q, key_cache, value_cache, block_table, seq_lens, out, window_left, float(scale)
    )


def sliding_decode_hd256_gqa2_splitk(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    window_left: int,
    scale: float,
) -> None:
    _get_mod().sliding_decode_hd256_gqa2_splitk(
        q, key_cache, value_cache, block_table, seq_lens, out, window_left, float(scale)
    )
