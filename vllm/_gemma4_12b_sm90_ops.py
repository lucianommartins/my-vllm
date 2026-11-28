# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90a (NVIDIA H100) specialized CUDA C++ kernels for Gemma 4 12B (dense) — v5:
  - Hopper SM90a Programmatic Dependent Launch (PDL / griddepcontrol) across
    the entire 48-layer dense chain (qkv_proj -> qkv_norm_rope_cache ->
    o_proj -> post_attn_pre_ffn_norm -> gate_up+gelu -> down_proj ->
    post_ffn_cross_norm -> lm_head+softcap), with weight/residual prefetch
    before griddepcontrol.wait! (Building on @engineer-3 779aaf79 PDL + 4-stage
    M=3..16 MLP/down_proj + tuned M=1 lm_head.)
  - Weight-first stage-0 cp.async prefetch before griddepcontrol.wait in
    gu_ksplit_mma (KS=5, TN=1, STAGES=4) and lin_ksplit_mma (down KS=8, TN=1, STAGES=4).
  - Single-pass native SM90a CUDA qkv_norm_rope_cache (HD=256/512, k_eq_v=True/False)
    with Hopper PDL + streaming ld.cs/st.cs prefill path (2.03x faster than Triton).
  - Global k_eq_v=True qkv_proj (N=8704) + 3.35 TB/s streaming gelu_tanh_and_mul,
    softcap_inplace, and streaming prefill cross-layer RMSNorms (D=3840).
All cp.async instructions omit L2::cache_hint to avoid the CUDA 12.8 sm_90a
odd-UR ptxas bug. Compiled without --use_fast_math.
"""

from __future__ import annotations

import hashlib
import os

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
#include <type_traits>

namespace gemma4_12b_sm90 {

struct alignas(16) bf16x8 {
    __nv_bfloat16 v[8];
};
static_assert(sizeof(bf16x8) == 16, "bf16x8 must be 16 bytes");

struct alignas(8) bf16x4 {
    __nv_bfloat16 v[4];
};
static_assert(sizeof(bf16x4) == 8, "bf16x4 must be 8 bytes");

__device__ __forceinline__ int4 ld_cs_int4(const int4* ptr) {
    int4 ret;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int4 ld_cs_256b_int4(const int4* ptr) {
    int4 ret;
    asm volatile("ld.global.cs.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int4 ld_ca_int4(const int4* ptr) {
    int4 ret;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_cs_int4(int4* ptr, const int4& v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ uint2 ld_cs_u64(const uint2* ptr) {
    uint2 ret;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];"
                 : "=r"(ret.x), "=r"(ret.y)
                 : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_cs_u64(uint2* ptr, const uint2& v) {
    asm volatile("st.global.cs.v2.u32 [%0], {%1,%2};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

template <typename T>
__device__ __forceinline__ const T* pdl_wait_ptr(const T* ptr) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" : "+l"(ptr) :: "memory");
#endif
    return ptr;
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

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Plain cp.async (NO L2::cache_hint — avoids CUDA 12.8 sm_90a odd-UR ptxas crash!)
__device__ __forceinline__ void cp_async_ca_zero(void* smem_dst, const void* glob_src, int valid) {
    uint32_t s = smem_u32(smem_dst);
    int sz = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(glob_src), "r"(sz));
}

__device__ __forceinline__ void cp_async_cg_uncond(void* smem_dst, const void* glob_src) {
    uint32_t s = smem_u32(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(s), "l"(glob_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ int swz64_col(int row, int col) {
    return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_ptr) {
    uint32_t s = smem_u32(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ __nv_bfloat16 exact_vllm_gelu_tanh_mul(float g_in, float u_in) {
    float g = __bfloat162float(__float2bfloat16_rn(g_in));
    float u = __bfloat162float(__float2bfloat16_rn(u_in));
    constexpr float kBeta = 0.7978845608028654f;
    constexpr float kKappa = 0.044715f;
    float inner = kBeta * (g + kKappa * g * g * g);
    float act = 0.5f * g * (1.0f + tanhf(inner));
    return __float2bfloat16_rn(act * u);
}

__device__ __forceinline__ __nv_bfloat16 exact_softcap_bf16(float v_in, float cap, float inv_cap) {
    float v = __bfloat162float(__float2bfloat16_rn(v_in));
    float t = tanhf(v * inv_cap);
    return __float2bfloat16_rn(t * cap);
}

// ============================================================
// 1. M=1..2 Fused gate_up + gelu_tanh_mul streaming GEMV (K=3840) + PDL
// ============================================================
template <int M_VAL, int ROWS, int TPR, int UNROLL, int USE_CS>
__global__ void gu_gemv_k3840_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ w_up,
    int nh) {
    constexpr int K = 3840;
    constexpr int K8 = K / 8;  // 480
    constexpr int STEP = TPR * UNROLL;
    constexpr int N_IT = (K8 + STEP - 1) / STEP;
    constexpr bool EXACT_DIV = (K8 % STEP == 0);
    constexpr int WPR = TPR / 32;

    const int tid = threadIdx.x;
    const int row_in_cta = tid / TPR;
    const int lane_in_row = tid % TPR;
    const int row = blockIdx.x * ROWS + row_in_cta;
    if (row >= nh) return;

    const int4* wg8 = reinterpret_cast<const int4*>(w_gate + static_cast<int64_t>(row) * K);
    const int4* wu8 = reinterpret_cast<const int4*>(w_up + static_cast<int64_t>(row) * K);
    int4 vg0[UNROLL], vu0[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        int idx = lane_in_row + u * TPR;
        if (EXACT_DIV || idx < K8) {
            vg0[u] = (USE_CS == 2) ? ld_cs_256b_int4(wg8 + idx) : (USE_CS ? ld_cs_int4(wg8 + idx) : __ldg(wg8 + idx));
            vu0[u] = (USE_CS == 2) ? ld_cs_256b_int4(wu8 + idx) : (USE_CS ? ld_cs_int4(wu8 + idx) : __ldg(wu8 + idx));
        }
    }
    const __nv_bfloat16* x_ready = pdl_wait_ptr(x);
    const int4* x8_0 = reinterpret_cast<const int4*>(x_ready);
    const int4* x8_1 = (M_VAL > 1) ? reinterpret_cast<const int4*>(x_ready + K) : nullptr;

    float g0 = 0.f, u0 = 0.f, g1 = 0.f, u1 = 0.f;
    #pragma unroll
    for (int it = 0; it < N_IT; ++it) {
        int base = lane_in_row + it * STEP;
        int4 vg[UNROLL], vu[UNROLL], vx0[UNROLL], vx1[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = base + u * TPR;
            if (EXACT_DIV || idx < K8) {
                if (it == 0) {
                    vg[u] = vg0[u];
                    vu[u] = vu0[u];
                } else {
                    vg[u] = (USE_CS == 2) ? ld_cs_256b_int4(wg8 + idx) : (USE_CS ? ld_cs_int4(wg8 + idx) : __ldg(wg8 + idx));
                    vu[u] = (USE_CS == 2) ? ld_cs_256b_int4(wu8 + idx) : (USE_CS ? ld_cs_int4(wu8 + idx) : __ldg(wu8 + idx));
                }
                vx0[u] = ld_ca_int4(x8_0 + idx);
                if constexpr (M_VAL > 1) vx1[u] = ld_ca_int4(x8_1 + idx);
            }
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = base + u * TPR;
            if (EXACT_DIV || idx < K8) {
                const __nv_bfloat162* wg2 = reinterpret_cast<const __nv_bfloat162*>(&vg[u]);
                const __nv_bfloat162* wu2 = reinterpret_cast<const __nv_bfloat162*>(&vu[u]);
                const __nv_bfloat162* a0  = reinterpret_cast<const __nv_bfloat162*>(&vx0[u]);
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float2 fg = __bfloat1622float2(wg2[c]);
                    float2 fu = __bfloat1622float2(wu2[c]);
                    float2 f0 = __bfloat1622float2(a0[c]);
                    g0 = fmaf(f0.x, fg.x, fmaf(f0.y, fg.y, g0));
                    u0 = fmaf(f0.x, fu.x, fmaf(f0.y, fu.y, u0));
                }
                if constexpr (M_VAL > 1) {
                    const __nv_bfloat162* a1 = reinterpret_cast<const __nv_bfloat162*>(&vx1[u]);
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        float2 fg = __bfloat1622float2(wg2[c]);
                        float2 fu = __bfloat1622float2(wu2[c]);
                        float2 f1 = __bfloat1622float2(a1[c]);
                        g1 = fmaf(f1.x, fg.x, fmaf(f1.y, fg.y, g1));
                        u1 = fmaf(f1.x, fu.x, fmaf(f1.y, fu.y, u1));
                    }
                }
            }
        }
    }

    pdl_launch_dependents();

    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        g0 += __shfl_xor_sync(0xffffffffu, g0, off, 32);
        u0 += __shfl_xor_sync(0xffffffffu, u0, off, 32);
        if constexpr (M_VAL > 1) {
            g1 += __shfl_xor_sync(0xffffffffu, g1, off, 32);
            u1 += __shfl_xor_sync(0xffffffffu, u1, off, 32);
        }
    }

    if constexpr (WPR == 1) {
        if (lane_in_row == 0) {
            out[row] = exact_vllm_gelu_tanh_mul(g0, u0);
            if constexpr (M_VAL > 1) out[nh + row] = exact_vllm_gelu_tanh_mul(g1, u1);
        }
    } else {
        __shared__ float sm_g0[ROWS][WPR];
        __shared__ float sm_u0[ROWS][WPR];
        __shared__ float sm_g1[ROWS][WPR];
        __shared__ float sm_u1[ROWS][WPR];
        const int w = lane_in_row >> 5;
        const int l = lane_in_row & 31;
        if (l == 0) {
            sm_g0[row_in_cta][w] = g0;
            sm_u0[row_in_cta][w] = u0;
            if constexpr (M_VAL > 1) {
                sm_g1[row_in_cta][w] = g1;
                sm_u1[row_in_cta][w] = u1;
            }
        }
        __syncthreads();
        if (lane_in_row == 0) {
            float tg0 = sm_g0[row_in_cta][0], tu0 = sm_u0[row_in_cta][0];
            float tg1 = 0.f, tu1 = 0.f;
            if constexpr (M_VAL > 1) {
                tg1 = sm_g1[row_in_cta][0];
                tu1 = sm_u1[row_in_cta][0];
            }
            #pragma unroll
            for (int k = 1; k < WPR; ++k) {
                tg0 += sm_g0[row_in_cta][k];
                tu0 += sm_u0[row_in_cta][k];
                if constexpr (M_VAL > 1) {
                    tg1 += sm_g1[row_in_cta][k];
                    tu1 += sm_u1[row_in_cta][k];
                }
            }
            out[row] = exact_vllm_gelu_tanh_mul(tg0, tu0);
            if constexpr (M_VAL > 1) out[nh + row] = exact_vllm_gelu_tanh_mul(tg1, tu1);
        }
    }
}

// ============================================================
// 2. M=3..16 Fused gate_up + gelu_tanh_mul 4-stage K-split MMA + PDL
//    Weight-first stage-0 cp.async prefetch before pdl_wait_ptr(x_in)!
// ============================================================
// Zero-loop 1-shot 512-thread M=1 gate_up GEMV (480 active lanes = 1 int4 load/thread, credit @engineer-1 v10)
template <int USE_CS>
__global__ __launch_bounds__(512)
void gu_gemv_k3840_1shot512_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ w_up,
    int nh) {
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= nh) return;
    int4 vg, vu;
    if (lane < 480) {
        const int4* wg8 = reinterpret_cast<const int4*>(w_gate + static_cast<int64_t>(row) * 3840);
        const int4* wu8 = reinterpret_cast<const int4*>(w_up + static_cast<int64_t>(row) * 3840);
        vg = (USE_CS == 2) ? ld_cs_256b_int4(wg8 + lane) : ld_cs_int4(wg8 + lane);
        vu = (USE_CS == 2) ? ld_cs_256b_int4(wu8 + lane) : ld_cs_int4(wu8 + lane);
    }
    const int4* x8 = reinterpret_cast<const int4*>(pdl_wait_ptr(x));
    float g = 0.f, uu = 0.f;
    if (lane < 480) {
        int4 vx = ld_ca_int4(x8 + lane);
        const __nv_bfloat162* wg2 = reinterpret_cast<const __nv_bfloat162*>(&vg);
        const __nv_bfloat162* wu2 = reinterpret_cast<const __nv_bfloat162*>(&vu);
        const __nv_bfloat162* x2  = reinterpret_cast<const __nv_bfloat162*>(&vx);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float2 fx  = __bfloat1622float2(x2[c]);
            float2 fwg = __bfloat1622float2(wg2[c]);
            float2 fwu = __bfloat1622float2(wu2[c]);
            g  = fmaf(fx.x, fwg.x, fmaf(fx.y, fwg.y, g));
            uu = fmaf(fx.x, fwu.x, fmaf(fx.y, fwu.y, uu));
        }
    }
    pdl_launch_dependents();
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        g  += __shfl_xor_sync(0xffffffffu, g, off, 32);
        uu += __shfl_xor_sync(0xffffffffu, uu, off, 32);
    }
    __shared__ float smg[16], smu[16];
    if ((lane & 31) == 0) {
        smg[lane >> 5] = g;
        smu[lane >> 5] = uu;
    }
    __syncthreads();
    if (lane == 0) {
        float sg = 0.f, su = 0.f;
        #pragma unroll
        for (int i = 0; i < 15; ++i) {
            sg += smg[i];
            su += smu[i];
        }
        out[row] = exact_vllm_gelu_tanh_mul(sg, su);
    }
}

template <int K_DIM, int K_SPLITS, int TILES_N, int STAGES = 4, bool PACK_V2 = false>
__global__ void gu_ksplit_mma_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x_in,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ w_up,
    int M,
    int nh) {
    constexpr int BK = 64, A_ELEMS = 16 * BK, B_ROWS = 2 * TILES_N * 16, B_ELEMS = B_ROWS * BK;
    constexpr int HALF_B = TILES_N * 16, HALF_CH = (HALF_B * 8) / 32, NT = TILES_N * 2;
    const int tid = threadIdx.x, ks = tid >> 5, lane = tid & 31, n_base = blockIdx.x * HALF_B;
    if (n_base >= nh) return;

    extern __shared__ char dyn_smem[];
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(
        dyn_smem + static_cast<size_t>(K_SPLITS) * STAGES * A_ELEMS * sizeof(__nv_bfloat16));
    __nv_bfloat16* wa = smem_a + (ks * STAGES) * A_ELEMS;
    __nv_bfloat16* wb = smem_b + (ks * STAGES) * B_ELEMS;

    constexpr int chunks_total = K_DIM / BK;
    constexpr int chunks_per = (chunks_total + K_SPLITS - 1) / K_SPLITS;
    const int ch_start = ks * chunks_per;
    const int ch_end = min(chunks_total, ch_start + chunks_per);
    const int n_ch = max(0, ch_end - ch_start);

    const int r0_ld = lane >> 3, c8_ld = (lane & 7) << 3;
    const __nv_bfloat16* wg_cur = w_gate + static_cast<int64_t>(n_base + r0_ld) * K_DIM + ch_start * BK + c8_ld;
    const __nv_bfloat16* wu_cur = w_up + static_cast<int64_t>(n_base + r0_ld) * K_DIM + ch_start * BK + c8_ld;

    uint32_t x_mask = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if ((r0_ld + i * 4) < M) x_mask |= (1u << i);
    }

    float acc_g[NT][4], acc_u[NT][4];
    #pragma unroll
    for (int t = 0; t < NT; ++t) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) { acc_g[t][i] = 0.0f; acc_u[t][i] = 0.0f; }
    }

    // Prefetch 2 stages of weight tiles into SMEM BEFORE waiting on x_in (-47.7us/step @ M=16)!
    const int w_pref = min(2, min(STAGES - 1, n_ch));
    for (int wp = 0; wp < w_pref; ++wp) {
        __nv_bfloat16* sb0 = wb + wp * B_ELEMS;
        #pragma unroll
        for (int i = 0; i < HALF_CH; ++i) {
            int r = r0_ld + i * 4, ru = HALF_B + r;
            cp_async_cg_uncond(&sb0[r * BK + swz64_col(r, c8_ld)],
                               wg_cur + static_cast<int64_t>(i * 4) * K_DIM);
            cp_async_cg_uncond(&sb0[ru * BK + swz64_col(ru, c8_ld)],
                               wu_cur + static_cast<int64_t>(i * 4) * K_DIM);
        }
        wg_cur += BK;
        wu_cur += BK;
    }

    const __nv_bfloat16* x = pdl_wait_ptr(x_in);
    const __nv_bfloat16* x_cur = x + static_cast<int64_t>(r0_ld) * K_DIM + ch_start * BK + c8_ld;

    for (int wp = 0; wp < w_pref; ++wp) {
        __nv_bfloat16* sa0 = wa + wp * A_ELEMS;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = r0_ld + i * 4, valid = ((x_mask >> i) & 1u) ? 1 : 0;
            cp_async_ca_zero(&sa0[r * BK + swz64_col(r, c8_ld)],
                             valid ? (x_cur + static_cast<int64_t>(i * 4) * K_DIM) : x, valid);
        }
        x_cur += BK;
        cp_async_commit();
    }

    auto issue_stage = [&](int stage) {
        __nv_bfloat16* sa = wa + stage * A_ELEMS;
        __nv_bfloat16* sb = wb + stage * B_ELEMS;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = r0_ld + i * 4, valid = ((x_mask >> i) & 1u) ? 1 : 0;
            cp_async_ca_zero(&sa[r * BK + swz64_col(r, c8_ld)],
                             valid ? (x_cur + static_cast<int64_t>(i * 4) * K_DIM) : x, valid);
        }
        x_cur += BK;
        #pragma unroll
        for (int i = 0; i < HALF_CH; ++i) {
            int r = r0_ld + i * 4, ru = HALF_B + r;
            cp_async_cg_uncond(&sb[r * BK + swz64_col(r, c8_ld)],
                               wg_cur + static_cast<int64_t>(i * 4) * K_DIM);
            cp_async_cg_uncond(&sb[ru * BK + swz64_col(ru, c8_ld)],
                               wu_cur + static_cast<int64_t>(i * 4) * K_DIM);
        }
        wg_cur += BK;
        wu_cur += BK;
        cp_async_commit();
    };

    int preload = min(STAGES - 1, n_ch);
    for (int p = w_pref; p < preload; ++p) issue_stage(p);
    const int a_r = lane & 15, a_c = ((lane >> 4) & 1) << 3;
    const int b_off_row = (((lane >> 4) & 1) << 3) + (lane & 7);
    const int b_off_col = (((lane >> 3) & 1) << 3);

    for (int c = 0; c < n_ch; ++c) {
        int st = c % STAGES;
        int next_c = c + STAGES - 1;
        if (next_c < n_ch) {
            issue_stage(next_c % STAGES);
            cp_async_wait<STAGES - 1>();
        } else {
            int rem = n_ch - 1 - c;
            if (rem >= 3) cp_async_wait<3>();
            else if (rem == 2) cp_async_wait<2>();
            else if (rem == 1) cp_async_wait<1>();
            else cp_async_wait<0>();
        }
        __syncwarp();
        __nv_bfloat16* sa = wa + st * A_ELEMS;
        __nv_bfloat16* sb = wb + st * B_ELEMS;

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t ra[4];
            ldmatrix_x4(ra, &sa[a_r * BK + swz64_col(a_r, kk + a_c)]);
            #pragma unroll
            for (int tn = 0; tn < TILES_N; ++tn) {
                uint32_t rbg[4], rbu[4];
                int rg = tn * 16 + b_off_row, ru = HALF_B + rg;
                ldmatrix_x4(rbg, &sb[rg * BK + swz64_col(rg, kk + b_off_col)]);
                ldmatrix_x4(rbu, &sb[ru * BK + swz64_col(ru, kk + b_off_col)]);
                uint32_t bg0[2] = {rbg[0], rbg[1]}, bg1[2] = {rbg[2], rbg[3]};
                uint32_t bu0[2] = {rbu[0], rbu[1]}, bu1[2] = {rbu[2], rbu[3]};
                mma_m16n8k16_bf16(acc_g[tn * 2 + 0], ra, bg0);
                mma_m16n8k16_bf16(acc_g[tn * 2 + 1], ra, bg1);
                mma_m16n8k16_bf16(acc_u[tn * 2 + 0], ra, bu0);
                mma_m16n8k16_bf16(acc_u[tn * 2 + 1], ra, bu1);
            }
        }
    }

    pdl_launch_dependents();
    __syncthreads();
    float* red_g = reinterpret_cast<float*>(dyn_smem);
    constexpr int PER_KS = NT * 128;
    float* red_u = red_g + K_SPLITS * PER_KS;
    #pragma unroll
    for (int t = 0; t < NT; ++t) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            red_g[ks * PER_KS + t * 128 + lane * 4 + i] = acc_g[t][i];
            red_u[ks * PER_KS + t * 128 + lane * 4 + i] = acc_u[t][i];
        }
    }
    __syncthreads();

    // All-warp parallel reduction + 0-ULP exact epilogue (scalar or packed bf16x2)
    constexpr int NTHR = K_SPLITS * 32;
    if constexpr (PACK_V2) {
        constexpr int TOT_PAIRS = NT * 64;
        #pragma unroll
        for (int p = threadIdx.x; p < TOT_PAIRS; p += NTHR) {
            int idx = p * 2;
            int t = idx >> 7;
            int ln = (idx >> 2) & 31;
            int hi = p & 1;
            int r = (ln >> 2) + (hi ? 8 : 0);
            int c = n_base + t * 8 + (ln & 3) * 2;
            if (r < M && c + 1 < nh) {
                float sg0 = 0.f, sg1 = 0.f, su0 = 0.f, su1 = 0.f;
                #pragma unroll
                for (int k = 0; k < K_SPLITS; ++k) {
                    float2 vg = *reinterpret_cast<const float2*>(&red_g[k * PER_KS + idx]);
                    float2 vu = *reinterpret_cast<const float2*>(&red_u[k * PER_KS + idx]);
                    sg0 += vg.x; sg1 += vg.y;
                    su0 += vu.x; su1 += vu.y;
                }
                *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r) * nh + c]) =
                    __halves2bfloat162(exact_vllm_gelu_tanh_mul(sg0, su0), exact_vllm_gelu_tanh_mul(sg1, su1));
            }
        }
    } else {
        constexpr int TOT_ELEMS = NT * 128;
        #pragma unroll
        for (int idx = threadIdx.x; idx < TOT_ELEMS; idx += NTHR) {
            int t = idx >> 7;
            int ln = (idx >> 2) & 31;
            int ii = idx & 3;
            int r = (ln >> 2) + ((ii >= 2) ? 8 : 0);
            int c = n_base + t * 8 + (ln & 3) * 2 + (ii & 1);
            if (r < M && c < nh) {
                float sg = 0.f, su = 0.f;
                #pragma unroll
                for (int k = 0; k < K_SPLITS; ++k) {
                    sg += red_g[k * PER_KS + idx];
                    su += red_u[k * PER_KS + idx];
                }
                out[static_cast<int64_t>(r) * nh + c] = exact_vllm_gelu_tanh_mul(sg, su);
            }
        }
    }
}

// 3.35 TB/s streaming gelu_tanh_and_mul for M > 16 prefill
__global__ void gelu_tanh_and_mul_bf16x8_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    int64_t M,
    int64_t d) {
    int64_t d8 = d >> 3;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = M * d8;
    if (idx >= total) return;
    int64_t row = idx / d8;
    int64_t col8 = idx - row * d8;
    const int4* in8 = reinterpret_cast<const int4*>(input + row * (2 * d));
    int4* out8 = reinterpret_cast<int4*>(out + row * d);
    int4 g_raw = ld_cs_int4(in8 + col8);
    int4 u_raw = ld_cs_int4(in8 + d8 + col8);
    const bf16x8* gv = reinterpret_cast<const bf16x8*>(&g_raw);
    const bf16x8* uv = reinterpret_cast<const bf16x8*>(&u_raw);
    bf16x8 ov;
    constexpr float kBeta = 0.7978845608028654f;
    constexpr float kKappa = 0.044715f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float g = __bfloat162float(gv->v[j]);
        float u = __bfloat162float(uv->v[j]);
        float inner = kBeta * (g + kKappa * g * g * g);
        float act = 0.5f * g * (1.0f + tanhf(inner));
        ov.v[j] = __float2bfloat16_rn(act * u);
    }
    st_cs_int4(out8 + col8, *reinterpret_cast<const int4*>(&ov));
}

template <bool STREAM_IN, bool STREAM_OUT>
__global__ void gelu_tanh_and_mul_d15360_2d_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    int M) {
    constexpr int D = 15360, D8 = 1920;
    int col8 = blockIdx.x * 256 + threadIdx.x;
    int row = blockIdx.y;
    if (col8 >= D8 || row >= M) return;
    const int4* in8 = reinterpret_cast<const int4*>(input + static_cast<int64_t>(row) * (2 * D));
    int4* out8 = reinterpret_cast<int4*>(out + static_cast<int64_t>(row) * D);
    int4 g_raw = STREAM_IN ? ld_cs_int4(in8 + col8) : in8[col8];
    int4 u_raw = STREAM_IN ? ld_cs_int4(in8 + D8 + col8) : in8[D8 + col8];
    const bf16x8* gv = reinterpret_cast<const bf16x8*>(&g_raw);
    const bf16x8* uv = reinterpret_cast<const bf16x8*>(&u_raw);
    bf16x8 ov;
    constexpr float kBeta = 0.7978845608028654f;
    constexpr float kKappa = 0.044715f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float g = __bfloat162float(gv->v[j]);
        float u = __bfloat162float(uv->v[j]);
        float inner = kBeta * (g + kKappa * g * g * g);
        float act = 0.5f * g * (1.0f + tanhf(inner));
        ov.v[j] = __float2bfloat16_rn(act * u);
    }
    int4 raw = *reinterpret_cast<const int4*>(&ov);
    if constexpr (STREAM_OUT) st_cs_int4(out8 + col8, raw);
    else out8[col8] = raw;
}

// ============================================================
// 3. M=1..2 Linear streaming GEMV (and optional softcap) + PDL
// ============================================================
template <int K_CONST, int ROWS, int TPR, int UNROLL, int USE_CS, bool SOFTCAP, bool PLAIN_X = false>
__global__ void lin_m1_kconst_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N,
    float cap,
    float inv_cap) {
    constexpr int K8 = K_CONST / 8;
    constexpr int STEP = TPR * UNROLL;
    constexpr int N_IT = (K8 + STEP - 1) / STEP;
    constexpr bool EXACT = (K8 % STEP == 0);
    constexpr int WPR = TPR / 32;

    const int tid = threadIdx.x;
    const int row_in_cta = tid / TPR;
    const int lane_in_row = tid % TPR;
    const int row = blockIdx.x * ROWS + row_in_cta;
    if (row >= N) return;

    const int4* w8 = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * K_CONST);
    int4 vw0[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        int idx = lane_in_row + u * TPR;
        if (EXACT || idx < K8) {
            vw0[u] = (USE_CS == 2) ? ld_cs_256b_int4(w8 + idx) : (USE_CS ? ld_cs_int4(w8 + idx) : __ldg(w8 + idx));
        }
    }
    const int4* x8 = reinterpret_cast<const int4*>(pdl_wait_ptr(x));

    float acc = 0.f;
    #pragma unroll
    for (int it = 0; it < N_IT; ++it) {
        int base = it * STEP + lane_in_row;
        int4 vw[UNROLL], vx[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = base + u * TPR;
            if (EXACT || idx < K8) {
                vw[u] = (it == 0) ? vw0[u] : ((USE_CS == 2) ? ld_cs_256b_int4(w8 + idx) : (USE_CS ? ld_cs_int4(w8 + idx) : __ldg(w8 + idx)));
                vx[u] = PLAIN_X ? x8[idx] : ld_ca_int4(x8 + idx);
            }
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = base + u * TPR;
            if (EXACT || idx < K8) {
                const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw[u]);
                const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&vx[u]);
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float2 fw = __bfloat1622float2(w2[c]);
                    float2 fx = __bfloat1622float2(x2[c]);
                    acc = fmaf(fx.x, fw.x, fmaf(fx.y, fw.y, acc));
                }
            }
        }
    }

    pdl_launch_dependents();

    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off, 32);
    }
    auto emit_val = [&](float v) {
        if constexpr (SOFTCAP) v = cap * tanhf(v * inv_cap);
        out[row] = __float2bfloat16(v);
    };
    if constexpr (WPR == 1) {
        if (lane_in_row == 0) emit_val(acc);
    } else {
        __shared__ float sm[ROWS][WPR];
        if ((lane_in_row & 31) == 0) sm[row_in_cta][lane_in_row >> 5] = acc;
        __syncthreads();
        if (lane_in_row == 0) {
            float t = sm[row_in_cta][0];
            #pragma unroll
            for (int k = 1; k < WPR; ++k) t += sm[row_in_cta][k];
            emit_val(t);
        }
    }
}

template <int K_CONST, int ROWS, int TPR, int UNROLL, int USE_CS, bool SOFTCAP>
__global__ void lin_m2_kconst_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N,
    float cap,
    float inv_cap) {
    constexpr int K8 = K_CONST / 8;
    constexpr int STEP = TPR * UNROLL;
    constexpr int N_IT = (K8 + STEP - 1) / STEP;
    constexpr bool EXACT = (K8 % STEP == 0);
    constexpr int WPR = TPR / 32;

    const int tid = threadIdx.x;
    const int row_in_cta = tid / TPR;
    const int lane_in_row = tid % TPR;
    const int row = blockIdx.x * ROWS + row_in_cta;
    if (row >= N) return;

    const int4* w8 = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * K_CONST);
    int4 vw0[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        int idx = lane_in_row + u * TPR;
        if (EXACT || idx < K8) {
            vw0[u] = (USE_CS == 2) ? ld_cs_256b_int4(w8 + idx) : (USE_CS ? ld_cs_int4(w8 + idx) : __ldg(w8 + idx));
        }
    }
    const __nv_bfloat16* xr = pdl_wait_ptr(x);
    const int4* x8_0 = reinterpret_cast<const int4*>(xr);
    const int4* x8_1 = reinterpret_cast<const int4*>(xr + K_CONST);

    float a0 = 0.f, a1 = 0.f;
    #pragma unroll
    for (int it = 0; it < N_IT; ++it) {
        int base = it * STEP + lane_in_row;
        int4 vw[UNROLL], vx0[UNROLL], vx1[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = base + u * TPR;
            if (EXACT || idx < K8) {
                vw[u] = (it == 0) ? vw0[u] : ((USE_CS == 2) ? ld_cs_256b_int4(w8 + idx) : (USE_CS ? ld_cs_int4(w8 + idx) : __ldg(w8 + idx)));
                vx0[u] = ld_ca_int4(x8_0 + idx);
                vx1[u] = ld_ca_int4(x8_1 + idx);
            }
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = base + u * TPR;
            if (EXACT || idx < K8) {
                const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw[u]);
                const __nv_bfloat162* x2_0 = reinterpret_cast<const __nv_bfloat162*>(&vx0[u]);
                const __nv_bfloat162* x2_1 = reinterpret_cast<const __nv_bfloat162*>(&vx1[u]);
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float2 fw = __bfloat1622float2(w2[c]);
                    float2 f0 = __bfloat1622float2(x2_0[c]);
                    float2 f1 = __bfloat1622float2(x2_1[c]);
                    a0 = fmaf(f0.x, fw.x, fmaf(f0.y, fw.y, a0));
                    a1 = fmaf(f1.x, fw.x, fmaf(f1.y, fw.y, a1));
                }
            }
        }
    }

    pdl_launch_dependents();

    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, off, 32);
        a1 += __shfl_xor_sync(0xffffffffu, a1, off, 32);
    }
    if constexpr (WPR == 1) {
        if (lane_in_row == 0) {
            out[row] = SOFTCAP ? exact_softcap_bf16(a0, cap, inv_cap) : __float2bfloat16_rn(a0);
            out[static_cast<int64_t>(N) + row] = SOFTCAP ? exact_softcap_bf16(a1, cap, inv_cap) : __float2bfloat16_rn(a1);
        }
    } else {
        __shared__ float sm[ROWS][WPR][2];
        if ((lane_in_row & 31) == 0) {
            sm[row_in_cta][lane_in_row >> 5][0] = a0;
            sm[row_in_cta][lane_in_row >> 5][1] = a1;
        }
        __syncthreads();
        if (lane_in_row == 0) {
            float t0 = sm[row_in_cta][0][0], t1 = sm[row_in_cta][0][1];
            #pragma unroll
            for (int k = 1; k < WPR; ++k) {
                t0 += sm[row_in_cta][k][0];
                t1 += sm[row_in_cta][k][1];
            }
            out[row] = SOFTCAP ? exact_softcap_bf16(t0, cap, inv_cap) : __float2bfloat16_rn(t0);
            out[static_cast<int64_t>(N) + row] = SOFTCAP ? exact_softcap_bf16(t1, cap, inv_cap) : __float2bfloat16_rn(t1);
        }
    }
}

template <int M_VAL, int ROWS, int TPR, int UNROLL, int USE_CS, bool SOFTCAP = false>
__global__ void lin_gemv_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N,
    int K,
    float cap,
    float inv_cap) {
    const int K8 = K >> 3;
    const int tid = threadIdx.x;
    const int row_in_cta = tid / TPR;
    const int lane_in_row = tid % TPR;
    const int row = blockIdx.x * ROWS + row_in_cta;
    if (row >= N) return;

    const int4* w8 = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * K);
    const int step = TPR * UNROLL;
    int4 vw0[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        int idx = lane_in_row + u * TPR;
        vw0[u] = (USE_CS == 2) ? ld_cs_256b_int4(w8 + idx) : (USE_CS ? ld_cs_int4(w8 + idx) : __ldg(w8 + idx));
    }
    const __nv_bfloat16* x_ready = pdl_wait_ptr(x);
    const int4* x8_0 = reinterpret_cast<const int4*>(x_ready);
    const int4* x8_1 = (M_VAL > 1) ? reinterpret_cast<const int4*>(x_ready + K) : nullptr;

    float a0 = 0.f, a1 = 0.f;
    {
        int4 vx0[UNROLL], vx1[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = lane_in_row + u * TPR;
            vx0[u] = ld_ca_int4(x8_0 + idx);
            if constexpr (M_VAL > 1) vx1[u] = ld_ca_int4(x8_1 + idx);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw0[u]);
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&vx0[u]);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                float2 fw = __bfloat1622float2(w2[c]);
                float2 fx = __bfloat1622float2(x2[c]);
                a0 = fmaf(fx.x, fw.x, fmaf(fx.y, fw.y, a0));
            }
            if constexpr (M_VAL > 1) {
                const __nv_bfloat162* x2_1 = reinterpret_cast<const __nv_bfloat162*>(&vx1[u]);
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float2 fw = __bfloat1622float2(w2[c]);
                    float2 fx = __bfloat1622float2(x2_1[c]);
                    a1 = fmaf(fx.x, fw.x, fmaf(fx.y, fw.y, a1));
                }
            }
        }
    }
    for (int base = lane_in_row + step; base < K8; base += step) {
        int4 vw[UNROLL], vx0[UNROLL], vx1[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = base + u * TPR;
            if (idx < K8) {
                vw[u] = (USE_CS == 2) ? ld_cs_256b_int4(w8 + idx) : (USE_CS ? ld_cs_int4(w8 + idx) : __ldg(w8 + idx));
                vx0[u] = ld_ca_int4(x8_0 + idx);
                if constexpr (M_VAL > 1) vx1[u] = ld_ca_int4(x8_1 + idx);
            }
        }
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int idx = base + u * TPR;
            if (idx < K8) {
                const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw[u]);
                const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&vx0[u]);
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float2 fw = __bfloat1622float2(w2[c]);
                    float2 fx = __bfloat1622float2(x2[c]);
                    a0 = fmaf(fx.x, fw.x, fmaf(fx.y, fw.y, a0));
                }
                if constexpr (M_VAL > 1) {
                    const __nv_bfloat162* x2_1 = reinterpret_cast<const __nv_bfloat162*>(&vx1[u]);
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        float2 fw = __bfloat1622float2(w2[c]);
                        float2 fx = __bfloat1622float2(x2_1[c]);
                        a1 = fmaf(fx.x, fw.x, fmaf(fx.y, fw.y, a1));
                    }
                }
            }
        }
    }

    pdl_launch_dependents();

    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, off, 32);
        if constexpr (M_VAL > 1) a1 += __shfl_xor_sync(0xffffffffu, a1, off, 32);
    }

    if constexpr (TPR == 32) {
        if (lane_in_row == 0) {
            out[row] = SOFTCAP ? exact_softcap_bf16(a0, cap, inv_cap) : __float2bfloat16_rn(a0);
            if constexpr (M_VAL > 1)
                out[static_cast<int64_t>(N) + row] = SOFTCAP ? exact_softcap_bf16(a1, cap, inv_cap) : __float2bfloat16_rn(a1);
        }
    } else {
        constexpr int NW = TPR >> 5;
        __shared__ float smem[ROWS][NW][M_VAL];
        int ww = lane_in_row >> 5;
        if ((lane_in_row & 31) == 0) {
            smem[row_in_cta][ww][0] = a0;
            if constexpr (M_VAL > 1) smem[row_in_cta][ww][1] = a1;
        }
        __syncthreads();
        if (lane_in_row == 0) {
            float s0 = 0.f, s1 = 0.f;
            #pragma unroll
            for (int w = 0; w < NW; ++w) {
                s0 += smem[row_in_cta][w][0];
                if constexpr (M_VAL > 1) s1 += smem[row_in_cta][w][1];
            }
            out[row] = SOFTCAP ? exact_softcap_bf16(s0, cap, inv_cap) : __float2bfloat16_rn(s0);
            if constexpr (M_VAL > 1)
                out[static_cast<int64_t>(N) + row] = SOFTCAP ? exact_softcap_bf16(s1, cap, inv_cap) : __float2bfloat16_rn(s1);
        }
    }
}

// ============================================================
// 4. M=3..16 Linear K-split MMA kernel (and optional softcap) + PDL
//    Weight-first stage-0 cp.async prefetch before pdl_wait_ptr(x_in)!
// ============================================================
template <int K_SPLITS, int TILES_N, bool SOFTCAP = false, int STAGES = 2, bool PAR_EPI = false>
__global__ void lin_ksplit_mma_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x_in,
    const __nv_bfloat16* __restrict__ w,
    int M, int N, int K,
    float cap, float inv_cap) {
    constexpr int BK = 64, B_ROWS = TILES_N * 16;
    const int tid = threadIdx.x, ks = tid >> 5, lane = tid & 31;
    const int n_base = blockIdx.x * B_ROWS;
    if (n_base >= N) return;

    extern __shared__ char dyn_smem[];
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
    const size_t a_bytes = static_cast<size_t>(K_SPLITS) * STAGES * 16 * BK * sizeof(__nv_bfloat16);
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(dyn_smem + a_bytes);

    __nv_bfloat16* wa = smem_a + static_cast<size_t>(ks) * STAGES * 16 * BK;
    __nv_bfloat16* wb = smem_b + static_cast<size_t>(ks) * STAGES * B_ROWS * BK;

    const int chunks_total = K / BK;
    const int chunks_per = (chunks_total + K_SPLITS - 1) / K_SPLITS;
    const int ch_start = ks * chunks_per;
    const int ch_end = min(chunks_total, ch_start + chunks_per);
    const int n_ch = max(0, ch_end - ch_start);

    float acc[TILES_N * 2][4];
    #pragma unroll
    for (int t = 0; t < TILES_N * 2; ++t)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[t][i] = 0.0f;

    // Prefetch 2 stages of weights into SMEM BEFORE waiting on x_in (-47.7us/step @ M=16)!
    const int w_pref = min(2, min(STAGES - 1, n_ch));
    for (int wp = 0; wp < w_pref; ++wp) {
        int k0 = (ch_start + wp) * BK;
        __nv_bfloat16* sb0 = wb + wp * (B_ROWS * BK);
        #pragma unroll
        for (int e = lane; e < B_ROWS * 8; e += 32) {
            int r = e >> 3, c8 = (e & 7) << 3;
            cp_async_cg_uncond(&sb0[r * BK + swz64_col(r, c8)],
                               &w[static_cast<int64_t>(n_base + r) * K + k0 + c8]);
        }
    }

    const __nv_bfloat16* x = pdl_wait_ptr(x_in);

    for (int wp = 0; wp < w_pref; ++wp) {
        int k0 = (ch_start + wp) * BK;
        __nv_bfloat16* sa0 = wa + wp * (16 * BK);
        #pragma unroll
        for (int e = lane; e < 128; e += 32) {
            int r = e >> 3, c8 = (e & 7) << 3;
            int valid = (r < M) ? 1 : 0;
            cp_async_ca_zero(&sa0[r * BK + swz64_col(r, c8)],
                             valid ? &x[static_cast<int64_t>(r) * K + k0 + c8] : x, valid);
        }
        cp_async_commit();
    }

    auto issue_stage = [&](int stage, int k0) {
        __nv_bfloat16* sa = wa + stage * (16 * BK);
        __nv_bfloat16* sb = wb + stage * (B_ROWS * BK);
        #pragma unroll
        for (int e = lane; e < 128; e += 32) {
            int r = e >> 3, c8 = (e & 7) << 3;
            int valid = (r < M) ? 1 : 0;
            cp_async_ca_zero(&sa[r * BK + swz64_col(r, c8)],
                             valid ? &x[static_cast<int64_t>(r) * K + k0 + c8] : x, valid);
        }
        #pragma unroll
        for (int e = lane; e < B_ROWS * 8; e += 32) {
            int r = e >> 3, c8 = (e & 7) << 3;
            cp_async_cg_uncond(&sb[r * BK + swz64_col(r, c8)],
                               &w[static_cast<int64_t>(n_base + r) * K + k0 + c8]);
        }
        cp_async_commit();
    };

    int preload = min(STAGES - 1, n_ch);
    for (int p = w_pref; p < preload; ++p) issue_stage(p, (ch_start + p) * BK);
    const int a_r = lane & 15, a_c = ((lane >> 4) & 1) << 3;
    const int b_off_row = (((lane >> 4) & 1) << 3) + (lane & 7);
    const int b_off_col = (((lane >> 3) & 1) << 3);

    for (int c = 0; c < n_ch; ++c) {
        int st = c % STAGES;
        int next_c = c + STAGES - 1;
        if (next_c < n_ch) {
            issue_stage(next_c % STAGES, (ch_start + next_c) * BK);
            cp_async_wait<STAGES - 1>();
        } else {
            int rem = n_ch - 1 - c;
            if (rem >= 3) cp_async_wait<3>();
            else if (rem == 2) cp_async_wait<2>();
            else if (rem == 1) cp_async_wait<1>();
            else cp_async_wait<0>();
        }
        __syncwarp();
        __nv_bfloat16* sa = wa + st * (16 * BK);
        __nv_bfloat16* sb = wb + st * (B_ROWS * BK);
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t ra[4];
            ldmatrix_x4(ra, &sa[a_r * BK + swz64_col(a_r, kk + a_c)]);
            #pragma unroll
            for (int tn = 0; tn < TILES_N; ++tn) {
                uint32_t rb[4];
                int r_s = tn * 16 + b_off_row;
                ldmatrix_x4(rb, &sb[r_s * BK + swz64_col(r_s, kk + b_off_col)]);
                uint32_t b0[2] = {rb[0], rb[1]}, b1[2] = {rb[2], rb[3]};
                mma_m16n8k16_bf16(acc[tn * 2 + 0], ra, b0);
                mma_m16n8k16_bf16(acc[tn * 2 + 1], ra, b1);
            }
        }
    }

    pdl_launch_dependents();
    __syncthreads();
    float* red = reinterpret_cast<float*>(dyn_smem);
    constexpr int PER_KS = TILES_N * 2 * 128;
    #pragma unroll
    for (int t = 0; t < TILES_N * 2; ++t)
        #pragma unroll
        for (int i = 0; i < 4; ++i) red[ks * PER_KS + t * 128 + lane * 4 + i] = acc[t][i];
    __syncthreads();

    if constexpr (PAR_EPI) {
        constexpr int TOT_PAIRS = TILES_N * 128;
        constexpr int NTHR = K_SPLITS * 32;
        #pragma unroll
        for (int p = threadIdx.x; p < TOT_PAIRS; p += NTHR) {
            int idx = p * 2;
            int t = idx >> 7;
            int ln = (idx >> 2) & 31;
            int hi = p & 1;
            int r = (ln >> 2) + (hi ? 8 : 0);
            int c = n_base + t * 8 + (ln & 3) * 2;
            if (r < M && c + 1 < N) {
                float s0 = 0.f, s1 = 0.f;
                #pragma unroll
                for (int k = 0; k < K_SPLITS; ++k) {
                    float2 v = *reinterpret_cast<const float2*>(&red[k * PER_KS + idx]);
                    s0 += v.x;
                    s1 += v.y;
                }
                __nv_bfloat162 packed = SOFTCAP
                    ? __halves2bfloat162(exact_softcap_bf16(s0, cap, inv_cap), exact_softcap_bf16(s1, cap, inv_cap))
                    : __floats2bfloat162_rn(s0, s1);
                *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r) * N + c]) = packed;
            }
        }
    } else {
        int r0 = lane >> 2, r1 = r0 + 8, c0 = (lane & 3) * 2;
        #pragma unroll
        for (int t = 0; t < TILES_N * 2; ++t) {
            if ((t % K_SPLITS) == ks) {
                int n_col = n_base + t * 8 + c0;
                float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
                #pragma unroll
                for (int k = 0; k < K_SPLITS; ++k) {
                    s0 += red[k * PER_KS + t * 128 + lane * 4 + 0];
                    s1 += red[k * PER_KS + t * 128 + lane * 4 + 1];
                    s2 += red[k * PER_KS + t * 128 + lane * 4 + 2];
                    s3 += red[k * PER_KS + t * 128 + lane * 4 + 3];
                }
                if constexpr (SOFTCAP) {
                    if (r0 < M) {
                        out[static_cast<int64_t>(r0) * N + n_col + 0] = exact_softcap_bf16(s0, cap, inv_cap);
                        out[static_cast<int64_t>(r0) * N + n_col + 1] = exact_softcap_bf16(s1, cap, inv_cap);
                    }
                    if (r1 < M) {
                        out[static_cast<int64_t>(r1) * N + n_col + 0] = exact_softcap_bf16(s2, cap, inv_cap);
                        out[static_cast<int64_t>(r1) * N + n_col + 1] = exact_softcap_bf16(s3, cap, inv_cap);
                    }
                } else {
                    if (r0 < M) *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r0) * N + n_col]) = __floats2bfloat162_rn(s0, s1);
                    if (r1 < M) *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r1) * N + n_col]) = __floats2bfloat162_rn(s2, s3);
                }
            }
        }
    }
}

__global__ void softcap_inplace_bf16x8_kernel(__nv_bfloat16* __restrict__ ptr, int64_t n8, float cap, float inv_cap) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n8) return;
    int4* p8 = reinterpret_cast<int4*>(ptr);
    int4 raw = ld_cs_int4(p8 + idx);
    bf16x8* v = reinterpret_cast<bf16x8*>(&raw);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float x = __bfloat162float(v->v[j]);
        v->v[j] = __float2bfloat16_rn(tanhf(x * inv_cap) * cap);
    }
    st_cs_int4(p8 + idx, raw);
}

// ============================================================
// 5. Cross-Layer Dense RMSNorms for D=3840 (0-ULP exact)
//    STREAM=false: L2-cached + PDL (decode M < 256)
//    STREAM=true:  streaming ld.cs/st.cs (prefill M >= 256)
// ============================================================
__device__ __forceinline__ float block_sum_256(float val, float* s_warp) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, off, 32);
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) s_warp[wid] = val;
    __syncthreads();
    float sum = (lane < 8) ? s_warp[lane] : 0.f;
    if (wid == 0) {
        #pragma unroll
        for (int off = 4; off >= 1; off >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, off, 32);
        if (lane == 0) s_warp[0] = sum;
    }
    __syncthreads();
    return s_warp[0];
}

template <bool STREAM>
__device__ __forceinline__ bf16x8 load_bf16x8(const bf16x8* p) {
    if constexpr (STREAM) {
        int4 raw = ld_cs_int4(reinterpret_cast<const int4*>(p));
        bf16x8 r;
        *reinterpret_cast<int4*>(&r) = raw;
        return r;
    } else {
        return *p;
    }
}

template <bool STREAM>
__device__ __forceinline__ void store_bf16x8(bf16x8* p, const bf16x8& v) {
    if constexpr (STREAM) {
        st_cs_int4(reinterpret_cast<int4*>(p), *reinterpret_cast<const int4*>(&v));
    } else {
        *p = v;
    }
}

template <bool STREAM, bool EARLY_PDL = false>
__global__ __launch_bounds__(256)
void post_attn_pre_ffn_norm_3840_kernel(
    __nv_bfloat16* __restrict__ h_res_out,
    __nv_bfloat16* __restrict__ h_ffn_out,
    const __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ residual_in,
    const __nv_bfloat16* __restrict__ w_post_attn,
    const __nv_bfloat16* __restrict__ w_pre_ffn,
    float eps) {
    constexpr int D = 3840;
    constexpr int N_CHUNKS = D / 8;  // 480
    constexpr float INV_D = 1.0f / 3840.0f;

    __shared__ float s_warp[8], s_warp2[8];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;

    // Prefetch full residual & BOTH norm weights before waiting on attn_out!
    const bf16x8* r_row = reinterpret_cast<const bf16x8*>(residual_in + row * D);
    const bf16x8* w_pa  = reinterpret_cast<const bf16x8*>(w_post_attn);
    const bf16x8* w_pf  = reinterpret_cast<const bf16x8*>(w_pre_ffn);
    const bool has_hi = (tid + 256 < N_CHUNKS);
    bf16x8 r0 = load_bf16x8<STREAM>(r_row + tid);
    bf16x8 wpa0 = w_pa[tid];
    bf16x8 wpf0 = w_pf[tid];
    bf16x8 r1, wpa1, wpf1;
    if (has_hi) {
        r1 = load_bf16x8<STREAM>(r_row + tid + 256);
        wpa1 = w_pa[tid + 256];
        wpf1 = w_pf[tid + 256];
    }

    const __nv_bfloat16* a_ready = STREAM ? attn_out : pdl_wait_ptr(attn_out);
    if constexpr (!STREAM && EARLY_PDL) pdl_launch_dependents();
    const bf16x8* a_row = reinterpret_cast<const bf16x8*>(a_ready + row * D);

    bf16x8 a0 = load_bf16x8<STREAM>(a_row + tid);
    bf16x8 a1;
    float sq_a = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float x = __bfloat162float(a0.v[j]);
        sq_a = __fmaf_rn(x, x, sq_a);
    }
    if (has_hi) {
        a1 = load_bf16x8<STREAM>(a_row + tid + 256);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x = __bfloat162float(a1.v[j]);
            sq_a = __fmaf_rn(x, x, sq_a);
        }
    }
    float inv_rms_a = rsqrtf(__fmaf_rn(block_sum_256(sq_a, s_warp), INV_D, eps));

    bf16x8 h0, h1;
    float sq_h = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float x = __bfloat162float(a0.v[j]);
        float w = __bfloat162float(wpa0.v[j]);
        __nv_bfloat16 normed = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x, inv_rms_a), w));
        __nv_bfloat16 hr = __float2bfloat16_rn(__fadd_rn(__bfloat162float(normed), __bfloat162float(r0.v[j])));
        h0.v[j] = hr;
        float hf = __bfloat162float(hr);
        sq_h = __fmaf_rn(hf, hf, sq_h);
    }
    if (has_hi) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x = __bfloat162float(a1.v[j]);
            float w = __bfloat162float(wpa1.v[j]);
            __nv_bfloat16 normed = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x, inv_rms_a), w));
            __nv_bfloat16 hr = __float2bfloat16_rn(__fadd_rn(__bfloat162float(normed), __bfloat162float(r1.v[j])));
            h1.v[j] = hr;
            float hf = __bfloat162float(hr);
            sq_h = __fmaf_rn(hf, hf, sq_h);
        }
    }

    float inv_rms_h = rsqrtf(__fmaf_rn(block_sum_256(sq_h, s_warp2), INV_D, eps));

    if constexpr (!STREAM && !EARLY_PDL) pdl_launch_dependents();
    bf16x8* out_res = reinterpret_cast<bf16x8*>(h_res_out + row * D);
    bf16x8* out_ffn = reinterpret_cast<bf16x8*>(h_ffn_out + row * D);

    store_bf16x8<STREAM>(out_res + tid, h0);
    {
        bf16x8 o0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            o0.v[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(h0.v[j]), inv_rms_h), __bfloat162float(wpf0.v[j])));
        }
        store_bf16x8<STREAM>(out_ffn + tid, o0);
    }
    if (has_hi) {
        int c = tid + 256;
        store_bf16x8<STREAM>(out_res + c, h1);
        bf16x8 o1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            o1.v[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(h1.v[j]), inv_rms_h), __bfloat162float(wpf1.v[j])));
        }
        store_bf16x8<STREAM>(out_ffn + c, o1);
    }
}

template <bool STREAM, bool EARLY_PDL = false>
__global__ __launch_bounds__(256)
void post_ffn_cross_norm_3840_kernel(
    __nv_bfloat16* __restrict__ h_res_out,
    __nv_bfloat16* __restrict__ h_next_out,
    const __nv_bfloat16* __restrict__ mlp_out,
    const __nv_bfloat16* __restrict__ residual_in,
    const __nv_bfloat16* __restrict__ w_post_ffn,
    const __nv_bfloat16* __restrict__ layer_scalar,
    const __nv_bfloat16* __restrict__ w_next_in,
    float eps) {
    constexpr int D = 3840;
    constexpr int N_CHUNKS = D / 8;
    constexpr float INV_D = 1.0f / 3840.0f;

    __shared__ float s_warp[8], s_warp2[8];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const float sc = __bfloat162float(layer_scalar[0]);

    // Prefetch full residual & BOTH norm weights before waiting on mlp_out!
    const bf16x8* r_row = reinterpret_cast<const bf16x8*>(residual_in + row * D);
    const bf16x8* w_pf  = reinterpret_cast<const bf16x8*>(w_post_ffn);
    const bf16x8* w_ni  = reinterpret_cast<const bf16x8*>(w_next_in);
    const bool has_hi = (tid + 256 < N_CHUNKS);
    bf16x8 r0 = load_bf16x8<STREAM>(r_row + tid);
    bf16x8 wpf0 = w_pf[tid];
    bf16x8 wni0 = w_ni[tid];
    bf16x8 r1, wpf1, wni1;
    if (has_hi) {
        r1 = load_bf16x8<STREAM>(r_row + tid + 256);
        wpf1 = w_pf[tid + 256];
        wni1 = w_ni[tid + 256];
    }

    const __nv_bfloat16* m_ready = STREAM ? mlp_out : pdl_wait_ptr(mlp_out);
    if constexpr (!STREAM && EARLY_PDL) pdl_launch_dependents();
    const bf16x8* m_row = reinterpret_cast<const bf16x8*>(m_ready + row * D);

    bf16x8 m0 = load_bf16x8<STREAM>(m_row + tid);
    bf16x8 m1;
    float sq_m = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float x = __bfloat162float(m0.v[j]);
        sq_m = __fmaf_rn(x, x, sq_m);
    }
    if (has_hi) {
        m1 = load_bf16x8<STREAM>(m_row + tid + 256);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x = __bfloat162float(m1.v[j]);
            sq_m = __fmaf_rn(x, x, sq_m);
        }
    }
    float inv_rms_m = rsqrtf(__fmaf_rn(block_sum_256(sq_m, s_warp), INV_D, eps));

    bf16x8 h0, h1;
    float sq_h = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float x = __bfloat162float(m0.v[j]);
        float w = __bfloat162float(wpf0.v[j]);
        __nv_bfloat16 normed = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x, inv_rms_m), w));
        __nv_bfloat16 added  = __float2bfloat16_rn(__fadd_rn(__bfloat162float(normed), __bfloat162float(r0.v[j])));
        __nv_bfloat16 scaled = __float2bfloat16_rn(__fmul_rn(__bfloat162float(added), sc));
        h0.v[j] = scaled;
        float hf = __bfloat162float(scaled);
        sq_h = __fmaf_rn(hf, hf, sq_h);
    }
    if (has_hi) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x = __bfloat162float(m1.v[j]);
            float w = __bfloat162float(wpf1.v[j]);
            __nv_bfloat16 normed = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x, inv_rms_m), w));
            __nv_bfloat16 added  = __float2bfloat16_rn(__fadd_rn(__bfloat162float(normed), __bfloat162float(r1.v[j])));
            __nv_bfloat16 scaled = __float2bfloat16_rn(__fmul_rn(__bfloat162float(added), sc));
            h1.v[j] = scaled;
            float hf = __bfloat162float(scaled);
            sq_h = __fmaf_rn(hf, hf, sq_h);
        }
    }

    float inv_rms_h = rsqrtf(__fmaf_rn(block_sum_256(sq_h, s_warp2), INV_D, eps));

    if constexpr (!STREAM && !EARLY_PDL) pdl_launch_dependents();
    bf16x8* out_res = reinterpret_cast<bf16x8*>(h_res_out + row * D);
    bf16x8* out_nxt = reinterpret_cast<bf16x8*>(h_next_out + row * D);

    store_bf16x8<STREAM>(out_res + tid, h0);
    {
        bf16x8 o0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            o0.v[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(h0.v[j]), inv_rms_h), __bfloat162float(wni0.v[j])));
        }
        store_bf16x8<STREAM>(out_nxt + tid, o0);
    }
    if (has_hi) {
        int c = tid + 256;
        store_bf16x8<STREAM>(out_res + c, h1);
        bf16x8 o1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            o1.v[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(h1.v[j]), inv_rms_h), __bfloat162float(wni1.v[j])));
        }
        store_bf16x8<STREAM>(out_nxt + c, o1);
    }
}

// ============================================================
// 6. Native SM90a single-pass QKV-Norm + RoPE + KV-Cache kernel + PDL
//    1 warp/head, 0 SMEM, 0 syncthreads, supports HD=256/512 & k_eq_v!
//    Prefetches slot_mapping, positions, norm weights & cos/sin BEFORE pdl_wait_ptr(qkv)!
// ============================================================
template <int HD, bool STREAM>
__global__ void qkv_norm_rope_cache_sm90_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos_sin,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kv_cache,
    int64_t qkv_stride_t,
    int64_t pos_stride_t,
    int64_t slot_stride_t,
    int64_t q_out_stride_t,
    int64_t cos_sin_stride_t,
    int64_t kv_stride_blk,
    int64_t kv_stride_h,
    int64_t kv_stride_slot,
    int num_tokens,
    int num_slots,
    int num_q,
    int num_kv,
    int block_size,
    int k_eq_v,
    float eps) {
    constexpr int HALF = HD / 2;
    constexpr int VEC = HALF / 32;  // 4 for HD=256, 8 for HD=512
    constexpr float INV_HD = 1.0f / static_cast<float>(HD);

    const int warps_per_blk = blockDim.x >> 5;
    const int global_warp = blockIdx.x * warps_per_blk + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int total_heads = num_q + 2 * num_kv;
    const int total_work = num_tokens * total_heads;
    if (global_warp >= total_work) return;

    const int tok_idx = global_warp / total_heads;
    const int head_id = global_warp - tok_idx * total_heads;

    const bool is_q = (head_id < num_q);
    const bool is_k = (!is_q && head_id < num_q + num_kv);
    int64_t slot = -1;
    if (!is_q) {
        if (tok_idx >= num_slots) return;
        slot = slot_mapping[tok_idx * slot_stride_t];
        if (slot < 0) return;
    }

    // Prefetch norm weights + cos/sin + compute dst pointers BEFORE waiting on qkv!
    __nv_bfloat16* dst1 = nullptr;
    __nv_bfloat16* dst2 = nullptr;
    uint2 uw1_256, uw2_256, ucs1_256, ucs2_256;
    int4 uw1_512, uw2_512, ucs1_512, ucs2_512;
    if (is_q || is_k) {
        const __nv_bfloat16* w_ptr = is_q ? q_weight : k_weight;
        const int64_t pos = positions[tok_idx * pos_stride_t];
        const __nv_bfloat16* cs_base = cos_sin + pos * cos_sin_stride_t;
        if (is_q) {
            dst1 = q_out + static_cast<int64_t>(tok_idx) * q_out_stride_t + head_id * HD;
        } else {
            const int kh = head_id - num_q;
            const int64_t blk = slot / block_size;
            const int64_t boff = slot - blk * block_size;
            dst1 = kv_cache + blk * kv_stride_blk + kh * kv_stride_h + boff * kv_stride_slot;
        }
        dst2 = dst1 + HALF;
        if constexpr (HD == 256) {
            uw1_256 = __ldg(reinterpret_cast<const uint2*>(w_ptr) + lane);
            uw2_256 = __ldg(reinterpret_cast<const uint2*>(w_ptr + HALF) + lane);
            ucs1_256 = __ldg(reinterpret_cast<const uint2*>(cs_base) + lane);
            ucs2_256 = __ldg(reinterpret_cast<const uint2*>(cs_base + HALF) + lane);
        } else {
            uw1_512 = __ldg(reinterpret_cast<const int4*>(w_ptr) + lane);
            uw2_512 = __ldg(reinterpret_cast<const int4*>(w_ptr + HALF) + lane);
            ucs1_512 = __ldg(reinterpret_cast<const int4*>(cs_base) + lane);
            ucs2_512 = __ldg(reinterpret_cast<const int4*>(cs_base + HALF) + lane);
        }
    } else {
        const int vh = head_id - (num_q + num_kv);
        const int64_t blk = slot / block_size;
        const int64_t boff = slot - blk * block_size;
        dst1 = kv_cache + blk * kv_stride_blk + vh * kv_stride_h + boff * kv_stride_slot + HD;
        dst2 = dst1 + HALF;
    }

    const __nv_bfloat16* qkv_ready = STREAM ? qkv : pdl_wait_ptr(qkv);
    if constexpr (!STREAM) pdl_launch_dependents();
    const int src_head = (!is_q && !is_k && k_eq_v) ? (head_id - num_kv) : head_id;
    const __nv_bfloat16* base_in = qkv_ready + static_cast<int64_t>(tok_idx) * qkv_stride_t + src_head * HD;

    float x1f[VEC], x2f[VEC];
    float sq = 0.f;
    if constexpr (HD == 256) {
        const uint2* p1 = reinterpret_cast<const uint2*>(base_in) + lane;
        const uint2* p2 = reinterpret_cast<const uint2*>(base_in + HALF) + lane;
        uint2 u1 = STREAM ? ld_cs_u64(p1) : *p1;
        uint2 u2 = STREAM ? ld_cs_u64(p2) : *p2;
        const __nv_bfloat16* b1 = reinterpret_cast<const __nv_bfloat16*>(&u1);
        const __nv_bfloat16* b2 = reinterpret_cast<const __nv_bfloat16*>(&u2);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            x1f[j] = __bfloat162float(b1[j]);
            x2f[j] = __bfloat162float(b2[j]);
            sq = fmaf(x1f[j], x1f[j], sq);
            sq = fmaf(x2f[j], x2f[j], sq);
        }
    } else {
        const int4* p1 = reinterpret_cast<const int4*>(base_in) + lane;
        const int4* p2 = reinterpret_cast<const int4*>(base_in + HALF) + lane;
        int4 u1 = STREAM ? ld_cs_int4(p1) : *p1;
        int4 u2 = STREAM ? ld_cs_int4(p2) : *p2;
        const __nv_bfloat16* b1 = reinterpret_cast<const __nv_bfloat16*>(&u1);
        const __nv_bfloat16* b2 = reinterpret_cast<const __nv_bfloat16*>(&u2);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            x1f[j] = __bfloat162float(b1[j]);
            x2f[j] = __bfloat162float(b2[j]);
            sq = fmaf(x1f[j], x1f[j], sq);
            sq = fmaf(x2f[j], x2f[j], sq);
        }
    }

    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, off, 32);
    }
    const float inv_rms = rsqrtf(fmaf(sq, INV_HD, eps));

    if (is_q || is_k) {
        if constexpr (HD == 256) {
            const __nv_bfloat16* bw1 = reinterpret_cast<const __nv_bfloat16*>(&uw1_256);
            const __nv_bfloat16* bw2 = reinterpret_cast<const __nv_bfloat16*>(&uw2_256);
            const __nv_bfloat16* bc  = reinterpret_cast<const __nv_bfloat16*>(&ucs1_256);
            const __nv_bfloat16* bs  = reinterpret_cast<const __nv_bfloat16*>(&ucs2_256);
            bf16x4 o1, o2;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float n1 = x1f[j] * inv_rms * __bfloat162float(bw1[j]);
                float n2 = x2f[j] * inv_rms * __bfloat162float(bw2[j]);
                float c  = __bfloat162float(bc[j]);
                float s  = __bfloat162float(bs[j]);
                o1.v[j] = __float2bfloat16_rn(n1 * c - n2 * s);
                o2.v[j] = __float2bfloat16_rn(n2 * c + n1 * s);
            }
            if constexpr (STREAM) {
                st_cs_u64(reinterpret_cast<uint2*>(dst1) + lane, *reinterpret_cast<const uint2*>(&o1));
                st_cs_u64(reinterpret_cast<uint2*>(dst2) + lane, *reinterpret_cast<const uint2*>(&o2));
            } else {
                reinterpret_cast<uint2*>(dst1)[lane] = *reinterpret_cast<const uint2*>(&o1);
                reinterpret_cast<uint2*>(dst2)[lane] = *reinterpret_cast<const uint2*>(&o2);
            }
        } else {
            const __nv_bfloat16* bw1 = reinterpret_cast<const __nv_bfloat16*>(&uw1_512);
            const __nv_bfloat16* bw2 = reinterpret_cast<const __nv_bfloat16*>(&uw2_512);
            const __nv_bfloat16* bc  = reinterpret_cast<const __nv_bfloat16*>(&ucs1_512);
            const __nv_bfloat16* bs  = reinterpret_cast<const __nv_bfloat16*>(&ucs2_512);
            bf16x8 o1, o2;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float n1 = x1f[j] * inv_rms * __bfloat162float(bw1[j]);
                float n2 = x2f[j] * inv_rms * __bfloat162float(bw2[j]);
                float c  = __bfloat162float(bc[j]);
                float s  = __bfloat162float(bs[j]);
                o1.v[j] = __float2bfloat16_rn(n1 * c - n2 * s);
                o2.v[j] = __float2bfloat16_rn(n2 * c + n1 * s);
            }
            if constexpr (STREAM) {
                st_cs_int4(reinterpret_cast<int4*>(dst1) + lane, *reinterpret_cast<const int4*>(&o1));
                st_cs_int4(reinterpret_cast<int4*>(dst2) + lane, *reinterpret_cast<const int4*>(&o2));
            } else {
                reinterpret_cast<int4*>(dst1)[lane] = *reinterpret_cast<const int4*>(&o1);
                reinterpret_cast<int4*>(dst2)[lane] = *reinterpret_cast<const int4*>(&o2);
            }
        }
    } else {
        if constexpr (HD == 256) {
            bf16x4 o1, o2;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                o1.v[j] = __float2bfloat16_rn(x1f[j] * inv_rms);
                o2.v[j] = __float2bfloat16_rn(x2f[j] * inv_rms);
            }
            if constexpr (STREAM) {
                st_cs_u64(reinterpret_cast<uint2*>(dst1) + lane, *reinterpret_cast<const uint2*>(&o1));
                st_cs_u64(reinterpret_cast<uint2*>(dst2) + lane, *reinterpret_cast<const uint2*>(&o2));
            } else {
                reinterpret_cast<uint2*>(dst1)[lane] = *reinterpret_cast<const uint2*>(&o1);
                reinterpret_cast<uint2*>(dst2)[lane] = *reinterpret_cast<const uint2*>(&o2);
            }
        } else {
            bf16x8 o1, o2;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                o1.v[j] = __float2bfloat16_rn(x1f[j] * inv_rms);
                o2.v[j] = __float2bfloat16_rn(x2f[j] * inv_rms);
            }
            if constexpr (STREAM) {
                st_cs_int4(reinterpret_cast<int4*>(dst1) + lane, *reinterpret_cast<const int4*>(&o1));
                st_cs_int4(reinterpret_cast<int4*>(dst2) + lane, *reinterpret_cast<const int4*>(&o2));
            } else {
                reinterpret_cast<int4*>(dst1)[lane] = *reinterpret_cast<const int4*>(&o1);
                reinterpret_cast<int4*>(dst2)[lane] = *reinterpret_cast<const int4*>(&o2);
            }
        }
    }
}

template <int K_SPLITS, int TILES_N, bool SOFTCAP = false, int STAGES = 2, bool PAR_EPI = false>
static inline void launch_lin_ksplit(
    __nv_bfloat16* op, const __nv_bfloat16* xp, const __nv_bfloat16* wp,
    int M, int N, int K, float cap, float inv_cap, cudaStream_t stream) {
    constexpr int B_ROWS = TILES_N * 16;
    size_t smem_ab = (static_cast<size_t>(K_SPLITS) * STAGES * (16 * 64 + B_ROWS * 64)) * sizeof(__nv_bfloat16);
    size_t smem_red = static_cast<size_t>(K_SPLITS) * (TILES_N * 2) * 128 * sizeof(float);
    size_t smem = smem_ab > smem_red ? smem_ab : smem_red;
    auto k = lin_ksplit_mma_kernel<K_SPLITS, TILES_N, SOFTCAP, STAGES, PAR_EPI>;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        attr_set = true;
    }
    launch_pdl(k, dim3((N + B_ROWS - 1) / B_ROWS), dim3(K_SPLITS * 32), smem, stream, op, xp, wp, M, N, K, cap, inv_cap);
}

} // namespace gemma4_12b_sm90

void g12b_launch_gate_up_gelu(torch::Tensor out, torch::Tensor x, torch::Tensor w) {
    int M = static_cast<int>(x.size(0));
    int K = static_cast<int>(x.size(1));
    int nh = static_cast<int>(w.size(0) / 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto* wgp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>());
    auto* wup = wgp + static_cast<int64_t>(nh) * K;
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    if (M == 1) {
        gemma4_12b_sm90::launch_pdl(
            gemma4_12b_sm90::gu_gemv_k3840_1shot512_kernel<1>,
            dim3(nh), dim3(512), 0, stream, op, xp, wgp, wup, nh);
    } else if (M == 2) {
        gemma4_12b_sm90::launch_pdl(
            gemma4_12b_sm90::gu_gemv_k3840_kernel<2, 1, 256, 2, 1>,
            dim3(nh), dim3(256), 0, stream, op, xp, wgp, wup, nh);
    } else if (M >= 13) {
        constexpr int KS = 15, TN = 1, STAGES = 2, HALF_B = TN * 16;
        size_t smem_ab = (static_cast<size_t>(KS) * STAGES * (16 * 64 + 2 * HALF_B * 64)) * sizeof(__nv_bfloat16);
        size_t smem_red = static_cast<size_t>(KS) * (TN * 2) * 128 * 2 * sizeof(float);
        size_t smem = smem_ab > smem_red ? smem_ab : smem_red;
        auto k = gemma4_12b_sm90::gu_ksplit_mma_kernel<3840, KS, TN, STAGES, true>;
        static bool attr_set_15 = false;
        if (!attr_set_15) {
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
            attr_set_15 = true;
        }
        gemma4_12b_sm90::launch_pdl(k, dim3((nh + HALF_B - 1) / HALF_B), dim3(KS * 32), smem, stream, op, xp, wgp, wup, M, nh);
    } else {
        constexpr int KS = 12, TN = 1, STAGES = 3, HALF_B = TN * 16;
        size_t smem_ab = (static_cast<size_t>(KS) * STAGES * (16 * 64 + 2 * HALF_B * 64)) * sizeof(__nv_bfloat16);
        size_t smem_red = static_cast<size_t>(KS) * (TN * 2) * 128 * 2 * sizeof(float);
        size_t smem = smem_ab > smem_red ? smem_ab : smem_red;
        auto k = gemma4_12b_sm90::gu_ksplit_mma_kernel<3840, KS, TN, STAGES>;
        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
            attr_set = true;
        }
        gemma4_12b_sm90::launch_pdl(k, dim3((nh + HALF_B - 1) / HALF_B), dim3(KS * 32), smem, stream, op, xp, wgp, wup, M, nh);
    }
}

void g12b_launch_gelu_tanh_and_mul(torch::Tensor out, torch::Tensor gu) {
    int64_t M = gu.size(0);
    int64_t d = gu.size(1) / 2;
    if (M == 0) return;
    auto st = at::cuda::getCurrentCUDAStream();
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    auto* ip = reinterpret_cast<const __nv_bfloat16*>(gu.data_ptr<at::BFloat16>());
    if (d == 15360 && M <= 65535) {
        dim3 grid(8, static_cast<unsigned int>(M));
        if (M <= 512) {
            gemma4_12b_sm90::gelu_tanh_and_mul_d15360_2d_kernel<false, false><<<grid, 256, 0, st>>>(op, ip, static_cast<int>(M));
        } else if (M <= 768) {
            gemma4_12b_sm90::gelu_tanh_and_mul_d15360_2d_kernel<true, false><<<grid, 256, 0, st>>>(op, ip, static_cast<int>(M));
        } else {
            gemma4_12b_sm90::gelu_tanh_and_mul_d15360_2d_kernel<true, true><<<grid, 256, 0, st>>>(op, ip, static_cast<int>(M));
        }
        return;
    }
    int64_t total = M * (d >> 3);
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    gemma4_12b_sm90::gelu_tanh_and_mul_bf16x8_kernel<<<blocks, threads, 0, st>>>(op, ip, M, d);
}

void g12b_launch_skinny_linear(torch::Tensor out, torch::Tensor x, torch::Tensor w) {
    int M = static_cast<int>(x.size(0));
    int K = static_cast<int>(x.size(1));
    int N = static_cast<int>(w.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>());
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    if (K == 15360 && N == 3840) {
        if (M == 1) {
            gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m1_kconst_kernel<15360, 1, 512, 2, 1, false, false>, dim3(N), dim3(512), 0, stream, op, xp, wp, N, 0.f, 0.f);
        } else if (M == 2) {
            gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m2_kconst_kernel<15360, 1, 512, 2, 1, false>, dim3(N), dim3(512), 0, stream, op, xp, wp, N, 0.f, 0.f);
        } else if (M <= 12) {
            gemma4_12b_sm90::launch_lin_ksplit<8, 1, false, 4, true>(op, xp, wp, M, N, K, 0.f, 0.f, stream);
        } else {
            gemma4_12b_sm90::launch_lin_ksplit<15, 2, false, 2, true>(op, xp, wp, M, N, K, 0.f, 0.f, stream);
        }
    } else if (K == 4096 && N == 3840) {
        if (M == 1) {
            gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m1_kconst_kernel<4096, 2, 128, 2, 1, false>, dim3((N + 1) / 2), dim3(256), 0, stream, op, xp, wp, N, 0.f, 0.f);
        } else if (M == 2) {
            gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m2_kconst_kernel<4096, 1, 256, 2, 1, false>, dim3(N), dim3(256), 0, stream, op, xp, wp, N, 0.f, 0.f);
        } else {
            gemma4_12b_sm90::launch_lin_ksplit<12, 2, false, 2, false>(op, xp, wp, M, N, K, 0.f, 0.f, stream);
        }
    } else if (K == 8192 && N == 3840) {
        if (M == 1) {
            gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m1_kconst_kernel<8192, 1, 256, 2, 1, false>, dim3(N), dim3(256), 0, stream, op, xp, wp, N, 0.f, 0.f);
        } else if (M == 2) {
            gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m2_kconst_kernel<8192, 2, 128, 4, 1, false>, dim3((N + 1) / 2), dim3(256), 0, stream, op, xp, wp, N, 0.f, 0.f);
        } else {
            gemma4_12b_sm90::launch_lin_ksplit<8, 1, false, 3, false>(op, xp, wp, M, N, K, 0.f, 0.f, stream);
        }
    } else if (K == 3840 && (N == 8192 || N == 8704 || N == 9216)) {
        if (M == 1) {
            gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m1_kconst_kernel<3840, 2, 128, 2, 1, false>, dim3((N + 1) / 2), dim3(256), 0, stream, op, xp, wp, N, 0.f, 0.f);
        } else if (M == 2) {
            gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m2_kconst_kernel<3840, 2, 128, 4, 1, false>, dim3((N + 1) / 2), dim3(256), 0, stream, op, xp, wp, N, 0.f, 0.f);
        } else if (N == 8192) {
            gemma4_12b_sm90::launch_lin_ksplit<8, 1, false, 3, false>(op, xp, wp, M, N, K, 0.f, 0.f, stream);
        } else {
            gemma4_12b_sm90::launch_lin_ksplit<6, 2, false, 2, false>(op, xp, wp, M, N, K, 0.f, 0.f, stream);
        }
    }
}

void g12b_launch_lm_head_softcap(torch::Tensor out, torch::Tensor x, torch::Tensor w, double cap_d) {
    int M = static_cast<int>(x.size(0));
    int K = static_cast<int>(x.size(1));
    int N = static_cast<int>(w.size(0));
    float cap = static_cast<float>(cap_d);
    float inv_cap = 1.0f / cap;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>());
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    if (M == 1) {
        gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m1_kconst_kernel<3840, 1, 256, 2, 2, true>, dim3(N), dim3(256), 0, stream, op, xp, wp, N, cap, inv_cap);
    } else if (M == 2) {
        gemma4_12b_sm90::launch_pdl(gemma4_12b_sm90::lin_m2_kconst_kernel<3840, 2, 128, 4, 1, true>, dim3((N + 1) / 2), dim3(256), 0, stream, op, xp, wp, N, cap, inv_cap);
    } else if (M <= 12) {
        gemma4_12b_sm90::launch_lin_ksplit<12, 1, true, 3, true>(op, xp, wp, M, N, K, cap, inv_cap, stream);
    } else {
        gemma4_12b_sm90::launch_lin_ksplit<12, 1, true, 4, true>(op, xp, wp, M, N, K, cap, inv_cap, stream);
    }
}

void g12b_launch_softcap_inplace(torch::Tensor logits, double cap_d) {
    int64_t n8 = logits.numel() >> 3;
    float cap = static_cast<float>(cap_d);
    float inv_cap = 1.0f / cap;
    int threads = 256;
    int blocks = static_cast<int>((n8 + threads - 1) / threads);
    gemma4_12b_sm90::softcap_inplace_bf16x8_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__nv_bfloat16*>(logits.data_ptr<at::BFloat16>()), n8, cap, inv_cap);
}

void g12b_launch_post_attn_pre_ffn_norm(
    torch::Tensor h_res, torch::Tensor h_ffn,
    torch::Tensor a, torch::Tensor r, torch::Tensor wpa, torch::Tensor wpf, double eps) {
    int M = static_cast<int>(a.size(0));
    if (M == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* hr = reinterpret_cast<__nv_bfloat16*>(h_res.data_ptr<at::BFloat16>());
    auto* hf = reinterpret_cast<__nv_bfloat16*>(h_ffn.data_ptr<at::BFloat16>());
    auto* ap = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>());
    auto* rp = reinterpret_cast<const __nv_bfloat16*>(r.data_ptr<at::BFloat16>());
    auto* wpap = reinterpret_cast<const __nv_bfloat16*>(wpa.data_ptr<at::BFloat16>());
    auto* wpfp = reinterpret_cast<const __nv_bfloat16*>(wpf.data_ptr<at::BFloat16>());
    if (M >= 256) {
        gemma4_12b_sm90::post_attn_pre_ffn_norm_3840_kernel<true, false><<<M, 256, 0, stream>>>(
            hr, hf, ap, rp, wpap, wpfp, static_cast<float>(eps));
    } else {
        gemma4_12b_sm90::launch_pdl(
            gemma4_12b_sm90::post_attn_pre_ffn_norm_3840_kernel<false, true>,
            dim3(M), dim3(256), 0, stream,
            hr, hf, ap, rp, wpap, wpfp, static_cast<float>(eps));
    }
}

void g12b_launch_post_ffn_cross_norm(
    torch::Tensor h_res, torch::Tensor h_next,
    torch::Tensor m, torch::Tensor r, torch::Tensor wpf, torch::Tensor sc, torch::Tensor wni, double eps) {
    int M = static_cast<int>(m.size(0));
    if (M == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* hr = reinterpret_cast<__nv_bfloat16*>(h_res.data_ptr<at::BFloat16>());
    auto* hn = reinterpret_cast<__nv_bfloat16*>(h_next.data_ptr<at::BFloat16>());
    auto* mp = reinterpret_cast<const __nv_bfloat16*>(m.data_ptr<at::BFloat16>());
    auto* rp = reinterpret_cast<const __nv_bfloat16*>(r.data_ptr<at::BFloat16>());
    auto* wpfp = reinterpret_cast<const __nv_bfloat16*>(wpf.data_ptr<at::BFloat16>());
    auto* scp = reinterpret_cast<const __nv_bfloat16*>(sc.data_ptr<at::BFloat16>());
    auto* wnip = reinterpret_cast<const __nv_bfloat16*>(wni.data_ptr<at::BFloat16>());
    if (M >= 256) {
        gemma4_12b_sm90::post_ffn_cross_norm_3840_kernel<true, false><<<M, 256, 0, stream>>>(
            hr, hn, mp, rp, wpfp, scp, wnip, static_cast<float>(eps));
    } else {
        gemma4_12b_sm90::launch_pdl(
            gemma4_12b_sm90::post_ffn_cross_norm_3840_kernel<false, true>,
            dim3(M), dim3(256), 0, stream,
            hr, hn, mp, rp, wpfp, scp, wnip, static_cast<float>(eps));
    }
}

void g12b_launch_qkv_norm_rope_cache(
    torch::Tensor qkv,
    torch::Tensor positions,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos_sin,
    torch::Tensor slot_mapping,
    torch::Tensor q_out,
    torch::Tensor kv_cache,
    double eps) {
    int num_tokens = static_cast<int>(qkv.size(0));
    if (num_tokens == 0) return;
    int64_t num_q = q_out.size(1);
    int64_t head_dim = q_out.size(2);
    int64_t num_kv = kv_cache.size(1);
    int64_t block_size = kv_cache.size(2);
    int64_t k_eq_v = (qkv.size(1) == (num_q + num_kv) * head_dim) ? 1 : 0;
    int num_slots = static_cast<int>(slot_mapping.size(0));
    int total_work = num_tokens * static_cast<int>(num_q + 2 * num_kv);
    int warps_per_blk = (num_tokens <= 4) ? 2 : 4;
    int threads = warps_per_blk * 32;
    int blocks = (total_work + warps_per_blk - 1) / warps_per_blk;
    auto stream = at::cuda::getCurrentCUDAStream();

    auto* qkv_p = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>());
    auto* pos_p = positions.data_ptr<int64_t>();
    auto* qw_p = reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>());
    auto* kw_p = reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>());
    auto* cs_p = reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>());
    auto* sm_p = slot_mapping.data_ptr<int64_t>();
    auto* qo_p = reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr<at::BFloat16>());
    auto* kv_p = reinterpret_cast<__nv_bfloat16*>(kv_cache.data_ptr<at::BFloat16>());

    int64_t qkv_st = qkv.stride(0);
    int64_t pos_st = positions.stride(0);
    int64_t slot_st = slot_mapping.stride(0);
    int64_t qo_st = q_out.stride(0);
    int64_t cs_st = cos_sin.stride(0);
    int64_t kv_sblk = kv_cache.stride(0);
    int64_t kv_sh = kv_cache.stride(1);
    int64_t kv_ss = kv_cache.stride(2);
    float eps_f = static_cast<float>(eps);

    if (head_dim == 256) {
        if (num_tokens >= 256) {
            gemma4_12b_sm90::qkv_norm_rope_cache_sm90_kernel<256, true><<<blocks, threads, 0, stream>>>(
                qkv_p, pos_p, qw_p, kw_p, cs_p, sm_p, qo_p, kv_p,
                qkv_st, pos_st, slot_st, qo_st, cs_st, kv_sblk, kv_sh, kv_ss,
                num_tokens, num_slots, static_cast<int>(num_q), static_cast<int>(num_kv),
                static_cast<int>(block_size), static_cast<int>(k_eq_v), eps_f);
        } else {
            gemma4_12b_sm90::launch_pdl(
                gemma4_12b_sm90::qkv_norm_rope_cache_sm90_kernel<256, false>,
                dim3(blocks), dim3(threads), 0, stream,
                qkv_p, pos_p, qw_p, kw_p, cs_p, sm_p, qo_p, kv_p,
                qkv_st, pos_st, slot_st, qo_st, cs_st, kv_sblk, kv_sh, kv_ss,
                num_tokens, num_slots, static_cast<int>(num_q), static_cast<int>(num_kv),
                static_cast<int>(block_size), static_cast<int>(k_eq_v), eps_f);
        }
    } else if (head_dim == 512) {
        if (num_tokens >= 256) {
            gemma4_12b_sm90::qkv_norm_rope_cache_sm90_kernel<512, true><<<blocks, threads, 0, stream>>>(
                qkv_p, pos_p, qw_p, kw_p, cs_p, sm_p, qo_p, kv_p,
                qkv_st, pos_st, slot_st, qo_st, cs_st, kv_sblk, kv_sh, kv_ss,
                num_tokens, num_slots, static_cast<int>(num_q), static_cast<int>(num_kv),
                static_cast<int>(block_size), static_cast<int>(k_eq_v), eps_f);
        } else {
            gemma4_12b_sm90::launch_pdl(
                gemma4_12b_sm90::qkv_norm_rope_cache_sm90_kernel<512, false>,
                dim3(blocks), dim3(threads), 0, stream,
                qkv_p, pos_p, qw_p, kw_p, cs_p, sm_p, qo_p, kv_p,
                qkv_st, pos_st, slot_st, qo_st, cs_st, kv_sblk, kv_sh, kv_ss,
                num_tokens, num_slots, static_cast<int>(num_q), static_cast<int>(num_kv),
                static_cast<int>(block_size), static_cast<int>(k_eq_v), eps_f);
        }
    }
}
"""

_CPP_SRC = r"""
#include <torch/library.h>
#include <ATen/ATen.h>
#include <cstdint>
#include <tuple>

void g12b_launch_gate_up_gelu(torch::Tensor out, torch::Tensor x, torch::Tensor w);
void g12b_launch_gelu_tanh_and_mul(torch::Tensor out, torch::Tensor gu);
void g12b_launch_skinny_linear(torch::Tensor out, torch::Tensor x, torch::Tensor w);
void g12b_launch_lm_head_softcap(torch::Tensor out, torch::Tensor x, torch::Tensor w, double cap);
void g12b_launch_softcap_inplace(torch::Tensor logits, double cap);
void g12b_launch_post_attn_pre_ffn_norm(
    torch::Tensor h_res, torch::Tensor h_ffn,
    torch::Tensor a, torch::Tensor r, torch::Tensor wpa, torch::Tensor wpf, double eps);
void g12b_launch_post_ffn_cross_norm(
    torch::Tensor h_res, torch::Tensor h_next,
    torch::Tensor m, torch::Tensor r, torch::Tensor wpf, torch::Tensor sc, torch::Tensor wni, double eps);
void g12b_launch_qkv_norm_rope_cache(
    torch::Tensor qkv, torch::Tensor positions, torch::Tensor q_weight,
    torch::Tensor k_weight, torch::Tensor cos_sin, torch::Tensor slot_mapping,
    torch::Tensor q_out, torch::Tensor kv_cache, double eps);

torch::Tensor gemma4_12b_mlp_gate_up_gelu_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight) {
    int64_t m = x.size(0);
    int64_t k = x.size(1);
    int64_t nh = weight.size(0) / 2;
    if (m == 0) return x.new_empty({0, nh});
    if (
        k == 3840 && nh == 15360 &&
        x.is_cuda() && weight.is_cuda() &&
        x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
        x.is_contiguous() && weight.is_contiguous() &&
        (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0)
    ) {
        auto out = x.new_empty({m, nh});
        if (m <= 16) {
            g12b_launch_gate_up_gelu(out, x, weight);
            return out;
        }
        auto gu = at::linear(x, weight);
        g12b_launch_gelu_tanh_and_mul(out, gu);
        return out;
    }
    auto gu = at::linear(x, weight);
    auto out = x.new_empty({m, nh});
    g12b_launch_gelu_tanh_and_mul(out, gu);
    return out;
}

torch::Tensor gemma4_12b_skinny_linear_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight) {
    int64_t m = x.size(0);
    int64_t k = x.size(1);
    int64_t n = weight.size(0);
    if (m == 0) return x.new_empty({0, n});
    if (
        x.is_cuda() && weight.is_cuda() &&
        x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
        x.is_contiguous() && weight.is_contiguous() &&
        (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0)
    ) {
        if ((k == 15360 || k == 4096 || k == 8192) && n == 3840 && m <= 16) {
            auto out = x.new_empty({m, n});
            g12b_launch_skinny_linear(out, x, weight);
            return out;
        }
        if (k == 3840 && (n == 8192 || n == 8704 || n == 9216) && m <= 16) {
            auto out = x.new_empty({m, n});
            g12b_launch_skinny_linear(out, x, weight);
            return out;
        }
    }
    return at::linear(x, weight);
}

torch::Tensor gemma4_12b_lm_head_softcap_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double soft_cap) {
    int64_t m = x.size(0);
    int64_t k = x.size(1);
    int64_t n = weight.size(0);
    if (m == 0) return x.new_empty({0, n});
    if (
        k == 3840 && (n & 31) == 0 && soft_cap > 0.0 &&
        x.dim() == 2 && weight.dim() == 2 &&
        x.is_cuda() && weight.is_cuda() &&
        x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
        x.is_contiguous() && weight.is_contiguous() &&
        (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0)
    ) {
        if (m <= 16) {
            auto out = x.new_empty({m, n});
            g12b_launch_lm_head_softcap(out, x, weight, soft_cap);
            return out;
        }
        auto lg = at::linear(x, weight);
        g12b_launch_softcap_inplace(lg, soft_cap);
        return lg;
    }
    auto lg = at::linear(x, weight);
    lg.div_(soft_cap);
    lg.tanh_();
    lg.mul_(soft_cap);
    return lg;
}

std::tuple<torch::Tensor, torch::Tensor> gemma4_12b_post_attn_pre_ffn_norm_cuda(
    const torch::Tensor& attn_out,
    const torch::Tensor& residual_in,
    const torch::Tensor& w_post_attn,
    const torch::Tensor& w_pre_ffn,
    double eps) {
    auto h_res = torch::empty_like(attn_out);
    auto h_ffn = torch::empty_like(attn_out);
    g12b_launch_post_attn_pre_ffn_norm(
        h_res, h_ffn, attn_out, residual_in, w_post_attn, w_pre_ffn, eps);
    return {h_res, h_ffn};
}

std::tuple<torch::Tensor, torch::Tensor> gemma4_12b_post_ffn_cross_norm_cuda(
    const torch::Tensor& mlp_out,
    const torch::Tensor& residual_in,
    const torch::Tensor& w_post_ffn,
    const torch::Tensor& layer_scalar,
    const torch::Tensor& w_next_in,
    double eps) {
    auto h_res = torch::empty_like(mlp_out);
    auto h_next = torch::empty_like(mlp_out);
    g12b_launch_post_ffn_cross_norm(
        h_res, h_next, mlp_out, residual_in, w_post_ffn, layer_scalar, w_next_in, eps);
    return {h_res, h_next};
}

void gemma4_12b_qkv_norm_rope_cache_cuda(
    const torch::Tensor& qkv,
    const torch::Tensor& positions,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos_sin,
    const torch::Tensor& slot_mapping,
    torch::Tensor& q_out,
    torch::Tensor& kv_cache,
    double eps) {
    g12b_launch_qkv_norm_rope_cache(
        qkv, positions, q_weight, k_weight, cos_sin, slot_mapping,
        q_out, kv_cache, eps);
}

TORCH_LIBRARY_FRAGMENT(vllm, m) {
    m.def("gemma4_12b_mlp_gate_up_gelu(Tensor x, Tensor weight) -> Tensor");
    m.def("gemma4_12b_skinny_linear(Tensor x, Tensor weight) -> Tensor");
    m.def("gemma4_12b_lm_head_softcap(Tensor x, Tensor weight, float soft_cap) -> Tensor");
    m.def("gemma4_12b_post_attn_pre_ffn_norm(Tensor attn_out, Tensor residual_in, Tensor w_post_attn, Tensor w_pre_ffn, float eps) -> (Tensor, Tensor)");
    m.def("gemma4_12b_post_ffn_cross_norm(Tensor mlp_out, Tensor residual_in, Tensor w_post_ffn, Tensor layer_scalar, Tensor w_next_in, float eps) -> (Tensor, Tensor)");
    m.def("gemma4_12b_qkv_norm_rope_cache(Tensor qkv, Tensor positions, Tensor q_weight, Tensor k_weight, Tensor cos_sin, Tensor slot_mapping, Tensor(a!) q_out, Tensor(b!) kv_cache, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(vllm, CUDA, m) {
    m.impl("gemma4_12b_mlp_gate_up_gelu", &gemma4_12b_mlp_gate_up_gelu_cuda);
    m.impl("gemma4_12b_skinny_linear", &gemma4_12b_skinny_linear_cuda);
    m.impl("gemma4_12b_lm_head_softcap", &gemma4_12b_lm_head_softcap_cuda);
    m.impl("gemma4_12b_post_attn_pre_ffn_norm", &gemma4_12b_post_attn_pre_ffn_norm_cuda);
    m.impl("gemma4_12b_post_ffn_cross_norm", &gemma4_12b_post_ffn_cross_norm_cuda);
    m.impl("gemma4_12b_qkv_norm_rope_cache", &gemma4_12b_qkv_norm_rope_cache_cuda);
}
"""

_EXT_MOD = None
_FAKE_REGISTERED = False


def _resolve_cache_dir() -> str:
    base = (
        os.environ.get("VLLM_CACHE_ROOT")
        or os.environ.get("TORCH_EXTENSIONS_DIR")
        or os.path.expanduser("~/.cache/vllm")
    )
    return os.path.join(base, "gemma4_12b_sm90_ext")


def get_gemma4_12b_sm90_ext():
    global _EXT_MOD
    if _EXT_MOD is not None:
        return _EXT_MOD
    h = hashlib.sha256((_CUDA_SRC + _CPP_SRC).encode("utf-8")).hexdigest()[:12]
    mod_name = f"_gemma4_12b_sm90_ext_{h}"
    build_dir = os.path.join(_resolve_cache_dir(), mod_name)
    os.makedirs(build_dir, exist_ok=True)
    so_path = os.path.join(build_dir, f"{mod_name}.so")
    if os.path.exists(so_path):
        torch.ops.load_library(so_path)
        _EXT_MOD = True
        return _EXT_MOD
    load_inline(
        name=mod_name,
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=[
            "-O3",
            "-gencode",
            "arch=compute_90a,code=sm_90a",
        ],
        build_directory=build_dir,
        is_python_module=False,
        verbose=False,
    )
    torch.ops.load_library(so_path)
    _EXT_MOD = True
    return _EXT_MOD


def _register_ops():
    global _FAKE_REGISTERED
    get_gemma4_12b_sm90_ext()
    if _FAKE_REGISTERED:
        return
    _FAKE_REGISTERED = True

    @torch.library.register_fake("vllm::gemma4_12b_mlp_gate_up_gelu")
    def _gu_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x.new_empty((x.shape[0], weight.shape[0] // 2))

    @torch.library.register_fake("vllm::gemma4_12b_skinny_linear")
    def _lin_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x.new_empty((x.shape[0], weight.shape[0]))

    @torch.library.register_fake("vllm::gemma4_12b_lm_head_softcap")
    def _lm_fake(
        x: torch.Tensor, weight: torch.Tensor, soft_cap: float
    ) -> torch.Tensor:
        return x.new_empty((x.shape[0], weight.shape[0]))

    @torch.library.register_fake("vllm::gemma4_12b_post_attn_pre_ffn_norm")
    def _pa_fake(
        attn_out: torch.Tensor,
        residual_in: torch.Tensor,
        w_post_attn: torch.Tensor,
        w_pre_ffn: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(attn_out), torch.empty_like(attn_out)

    @torch.library.register_fake("vllm::gemma4_12b_post_ffn_cross_norm")
    def _pf_fake(
        mlp_out: torch.Tensor,
        residual_in: torch.Tensor,
        w_post_ffn: torch.Tensor,
        layer_scalar: torch.Tensor,
        w_next_in: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(mlp_out), torch.empty_like(mlp_out)

    @torch.library.register_fake("vllm::gemma4_12b_qkv_norm_rope_cache")
    def _qkv_fake(
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        q_out: torch.Tensor,
        kv_cache: torch.Tensor,
        eps: float,
    ) -> None:
        return None


def gemma4_12b_lm_head_softcap(
    x: torch.Tensor, weight: torch.Tensor, soft_cap: float
) -> torch.Tensor:
    return torch.ops.vllm.gemma4_12b_lm_head_softcap(x, weight, float(soft_cap))


def gemma4_12b_qkv_norm_rope_cache(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_out: torch.Tensor,
    kv_cache: torch.Tensor,
    eps: float,
) -> None:
    torch.ops.vllm.gemma4_12b_qkv_norm_rope_cache(
        qkv,
        positions,
        q_weight,
        k_weight,
        cos_sin,
        slot_mapping,
        q_out,
        kv_cache,
        float(eps),
    )


_register_ops()