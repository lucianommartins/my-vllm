# SPDX-License-Identifier: Apache-2.0
"""E2B-specific SM80 CUDA kernels for KV-shared q_norm+RoPE and MLP/attn projections."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

from vllm.utils.torch_utils import direct_register_custom_op

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef GEMMA4_HAS_CUTLASS_DUAL
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "device/dual_gemm.h"

namespace cutlass {
namespace epilogue {
namespace thread {

template <
  typename ElementOutput_,
  int Count,
  typename ElementAccumulator_ = ElementOutput_,
  typename ElementCompute_ = float,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LeftGeluTanhAndMul {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;
  struct Params {};

  CUTLASS_HOST_DEVICE LeftGeluTanhAndMul(Params const&) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_DEVICE
  static ElementOutput eval_one(ElementAccumulator g_in, ElementAccumulator u_in) {
    float g = static_cast<float>(g_in);
    float u = static_cast<float>(u_in);
    constexpr float kBeta = 0.7978845608028654f;
    constexpr float kKappa = 0.044715f;
    float inner = kBeta * (g + kKappa * g * g * g);
    float tanh_v;
    asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_v) : "f"(inner));
    float gelu_g = 0.5f * g * (1.0f + tanh_v);
    __nv_bfloat16 gelu_bf16 = __float2bfloat16_rn(gelu_g);
    __nv_bfloat16 u_bf16 = __float2bfloat16_rn(u);
    __nv_bfloat16 prod = __hmul(gelu_bf16, u_bf16);
    ElementOutput res;
    memcpy(&res, &prod, sizeof(res));
    return res;
  }

  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& lhs, FragmentAccumulator const& rhs) const {
    FragmentOutput out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      out[i] = eval_one(lhs[i], rhs[i]);
    }
    return out;
  }

  CUTLASS_DEVICE
  ElementOutput operator()(ElementAccumulator const& lhs, ElementAccumulator const& rhs) const {
    return eval_one(lhs, rhs);
  }
};

} // namespace thread
} // namespace epilogue
} // namespace cutlass
#endif

namespace {

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

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N_WAIT>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* ptr) {
    uint32_t s = smem_u32(ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(s));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* ptr) {
    uint32_t s = smem_u32(ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
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

__device__ __forceinline__ int swz64(int row, int chunk8) {
    return (row << 3) + (chunk8 ^ (row & 7));
}

__device__ __forceinline__ float warp_sum_16(float val) {
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float warp_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    return warp_sum_16(val);
}

template <int GROUP_SZ, int HEAD_DIM>
__device__ __forceinline__ void qnr_paired_u4_norm_step(
    uint4 qv_lo,
    uint4 qv_hi,
    uint4 wv_lo,
    uint4 wv_hi,
    float eps,
    float (&nlo)[8],
    float (&nhi)[8]
) {
    const __nv_bfloat162* q2_lo = reinterpret_cast<const __nv_bfloat162*>(&qv_lo);
    const __nv_bfloat162* q2_hi = reinterpret_cast<const __nv_bfloat162*>(&qv_hi);
    const __nv_bfloat162* w2_lo = reinterpret_cast<const __nv_bfloat162*>(&wv_lo);
    const __nv_bfloat162* w2_hi = reinterpret_cast<const __nv_bfloat162*>(&wv_hi);
    float lo_v[8], hi_v[8], lo_w[8], hi_w[8];
    float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 fq0 = __bfloat1622float2(q2_lo[j]);
        float2 fq1 = __bfloat1622float2(q2_hi[j]);
        float2 fw0 = __bfloat1622float2(w2_lo[j]);
        float2 fw1 = __bfloat1622float2(w2_hi[j]);
        lo_v[2 * j + 0] = fq0.x;
        lo_v[2 * j + 1] = fq0.y;
        hi_v[2 * j + 0] = fq1.x;
        hi_v[2 * j + 1] = fq1.y;
        lo_w[2 * j + 0] = fw0.x;
        lo_w[2 * j + 1] = fw0.y;
        hi_w[2 * j + 0] = fw1.x;
        hi_w[2 * j + 1] = fw1.y;
        ss = __fmaf_rn(fq0.x, fq0.x, ss);
        ss = __fmaf_rn(fq0.y, fq0.y, ss);
        ss = __fmaf_rn(fq1.x, fq1.x, ss);
        ss = __fmaf_rn(fq1.y, fq1.y, ss);
    }
    ss = (GROUP_SZ == 16) ? warp_sum_16(ss) : warp_sum(ss);
    float inv_rms = rsqrtf(ss * (1.0f / static_cast<float>(HEAD_DIM)) + eps);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float t0 = __bfloat162float(__float2bfloat16_rn(lo_v[i] * inv_rms));
        float t1 = __bfloat162float(__float2bfloat16_rn(hi_v[i] * inv_rms));
        nlo[i] = __bfloat162float(__float2bfloat16_rn(t0 * lo_w[i]));
        nhi[i] = __bfloat162float(__float2bfloat16_rn(t1 * hi_w[i]));
    }
}

__device__ __forceinline__ void qnr_paired_u4_store(
    uint4* o_u4,
    int hi_u4_off,
    int lane_vec,
    const float (&nlo)[8],
    const float (&nhi)[8],
    bool apply_rope,
    uint4 cos_v,
    uint4 sin_v
) {
    uint4 out_lo, out_hi;
    __nv_bfloat162* o2_lo = reinterpret_cast<__nv_bfloat162*>(&out_lo);
    __nv_bfloat162* o2_hi = reinterpret_cast<__nv_bfloat162*>(&out_hi);
    if (apply_rope) {
        const __nv_bfloat162* cos2 = reinterpret_cast<const __nv_bfloat162*>(&cos_v);
        const __nv_bfloat162* sin2 = reinterpret_cast<const __nv_bfloat162*>(&sin_v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 fc = __bfloat1622float2(cos2[j]);
            float2 fs = __bfloat1622float2(sin2[j]);
            o2_lo[j] = __floats2bfloat162_rn(
                nlo[2 * j + 0] * fc.x - nhi[2 * j + 0] * fs.x,
                nlo[2 * j + 1] * fc.y - nhi[2 * j + 1] * fs.y);
            o2_hi[j] = __floats2bfloat162_rn(
                nhi[2 * j + 0] * fc.x + nlo[2 * j + 0] * fs.x,
                nhi[2 * j + 1] * fc.y + nlo[2 * j + 1] * fs.y);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            o2_lo[j] = __floats2bfloat162_rn(nlo[2 * j + 0], nlo[2 * j + 1]);
            o2_hi[j] = __floats2bfloat162_rn(nhi[2 * j + 0], nhi[2 * j + 1]);
        }
    }
    o_u4[lane_vec] = out_lo;
    o_u4[hi_u4_off + lane_vec] = out_hi;
}

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
    float tanh_v;
    asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_v) : "f"(inner));
    float act_bf16 = __bfloat162float(__float2bfloat16_rn(0.5f * g * (1.0f + tanh_v)));
    return __float2bfloat16_rn(act_bf16 * u);
}

// ---------------------------------------------------------------------------
// 1. Fused q_norm + partial neox RoPE for KV-shared layers (hd in {256, 512})
// Matches vllm.ir.ops.rms_norm rounding: bf16(x * inv_rms) * w_bf16 -> bf16 before RoPE
// ---------------------------------------------------------------------------
template <int HD>
__global__ void __launch_bounds__(32)
e2b_q_norm_rope_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const int64_t* __restrict__ pos,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ cos_sin_cache,
    int q_row_stride,
    int cache_stride,
    int n_heads,
    int rope_angles,
    float eps
) {
    const int h_idx = blockIdx.x;
    const int t = h_idx / n_heads;
    const int h = h_idx - t * n_heads;
    const int tid = threadIdx.x;
    constexpr int VEC = HD / 32; // 8 (HD=256) or 16 (HD=512)

    const int64_t p = pos[t];
    const __nv_bfloat16* q_ptr = q + (size_t)t * q_row_stride + (size_t)h * HD;
    __nv_bfloat16* o_ptr = out + (size_t)h_idx * HD;
    const int cache_half = cache_stride >> 1;
    const __nv_bfloat16* cs_ptr = cos_sin_cache + (size_t)p * cache_stride;

    float vals[VEC];
    float ss = 0.f;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int c = tid + i * 32;
        float v = __bfloat162float(q_ptr[c]);
        vals[i] = v;
        ss = __fmaf_rn(v, v, ss);
    }
    ss = warp_sum(ss);
    float inv_rms = rsqrtf(ss / (float)HD + eps);

    float qn[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int c = tid + i * 32;
        float norm_only = __bfloat162float(__float2bfloat16_rn(vals[i] * inv_rms));
        qn[i] = __bfloat162float(__float2bfloat16_rn(norm_only * __bfloat162float(q_weight[c])));
    }

    constexpr int HALF = HD >> 1;
    constexpr int HALF_VEC = VEC >> 1;
    #pragma unroll
    for (int i = 0; i < HALF_VEC; ++i) {
        int c = tid + i * 32;
        if (c < rope_angles) {
            float q1 = qn[i];
            float q2 = qn[i + HALF_VEC];
            float cos_v = __bfloat162float(cs_ptr[c]);
            float sin_v = __bfloat162float(cs_ptr[cache_half + c]);
            float y1 = q1 * cos_v - q2 * sin_v;
            float y2 = q2 * cos_v + q1 * sin_v;
            o_ptr[c] = __float2bfloat16_rn(y1);
            o_ptr[c + HALF] = __float2bfloat16_rn(y2);
        } else {
            o_ptr[c] = __float2bfloat16_rn(qn[i]);
            o_ptr[c + HALF] = __float2bfloat16_rn(qn[i + HALF_VEC]);
        }
    }
}

__global__ void __launch_bounds__(64)
e2b_q_norm_rope_512_2w_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const int64_t* __restrict__ pos,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ cos_sin_cache,
    int q_row_stride,
    int cache_stride,
    int n_heads,
    int rope_angles,
    float eps
) {
    const int h_idx = blockIdx.x;
    const int t = h_idx / n_heads;
    const int h = h_idx - t * n_heads;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const __nv_bfloat16* q_ptr = q + (size_t)t * q_row_stride + (size_t)h * 512;
    __nv_bfloat16* o_ptr = out + (size_t)h_idx * 512;
    const int64_t p = pos[t];
    const __nv_bfloat16* cs_ptr = cos_sin_cache + (size_t)p * cache_stride;
    const int cache_half = cache_stride >> 1;

    float2 flo[2], fhi[2];
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int d_lo = (tid + k * 64) * 2;
        int d_hi = d_lo + 256;
        flo[k] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q_ptr + d_lo));
        fhi[k] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q_ptr + d_hi));
        ss = __fmaf_rn(flo[k].x, flo[k].x, ss);
        ss = __fmaf_rn(flo[k].y, flo[k].y, ss);
        ss = __fmaf_rn(fhi[k].x, fhi[k].x, ss);
        ss = __fmaf_rn(fhi[k].y, fhi[k].y, ss);
    }
    ss = warp_sum(ss);
    __shared__ float sm_ss[2];
    if (lane == 0) sm_ss[wid] = ss;
    __syncthreads();
    float inv_rms = rsqrtf((sm_ss[0] + sm_ss[1]) * (1.0f / 512.0f) + eps);

    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int d_lo = (tid + k * 64) * 2;
        int d_hi = d_lo + 256;
        float2 wlo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q_weight + d_lo));
        float2 whi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q_weight + d_hi));
        float nlo_x = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(flo[k].x * inv_rms)) * wlo.x));
        float nlo_y = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(flo[k].y * inv_rms)) * wlo.y));
        float nhi_x = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(fhi[k].x * inv_rms)) * whi.x));
        float nhi_y = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(fhi[k].y * inv_rms)) * whi.y));
        if (d_lo < rope_angles) {
            float2 cv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(cs_ptr + d_lo));
            float2 sv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(cs_ptr + cache_half + d_lo));
            *reinterpret_cast<__nv_bfloat162*>(o_ptr + d_lo) = __floats2bfloat162_rn(nlo_x * cv.x - nhi_x * sv.x, nlo_y * cv.y - nhi_y * sv.y);
            *reinterpret_cast<__nv_bfloat162*>(o_ptr + d_hi) = __floats2bfloat162_rn(nhi_x * cv.x + nlo_x * sv.x, nhi_y * cv.y + nlo_y * sv.y);
        } else {
            *reinterpret_cast<__nv_bfloat162*>(o_ptr + d_lo) = __floats2bfloat162_rn(nlo_x, nlo_y);
            *reinterpret_cast<__nv_bfloat162*>(o_ptr + d_hi) = __floats2bfloat162_rn(nhi_x, nhi_y);
        }
    }
}

__global__ void __launch_bounds__(32)
e2b_q_norm_rope_vec_256_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const int64_t* __restrict__ pos,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ cos_sin_cache,
    int q_row_stride,
    int cache_stride,
    int n_heads,
    int rope_angles,
    float eps
) {
    const int h_idx = blockIdx.x;
    const int t = h_idx / n_heads;
    const int h = h_idx - t * n_heads;
    const int lane = threadIdx.x;
    const int half_id = lane >> 4;
    const int vec_idx = lane & 15;
    const int c8 = half_id * 16 + vec_idx;
    const int c0 = vec_idx * 8;

    const int64_t p = pos[t];
    const uint4* q_u4 = reinterpret_cast<const uint4*>(q + (size_t)t * q_row_stride + (size_t)h * 256);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(q_weight);
    uint4* o_u4 = reinterpret_cast<uint4*>(out + (size_t)h_idx * 256);
    const int cache_half8 = (cache_stride >> 1) >> 3;
    const uint4* cs_u4 = reinterpret_cast<const uint4*>(cos_sin_cache + (size_t)p * cache_stride);

    uint4 qv = __ldg(q_u4 + c8);
    uint4 wv = __ldg(w_u4 + c8);
    const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
    float vals[8], wts[8];
    float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 fq = __bfloat1622float2(q2[j]);
        float2 fw = __bfloat1622float2(w2[j]);
        vals[2 * j + 0] = fq.x;
        vals[2 * j + 1] = fq.y;
        wts[2 * j + 0] = fw.x;
        wts[2 * j + 1] = fw.y;
        ss = __fmaf_rn(fq.x, fq.x, ss);
        ss = __fmaf_rn(fq.y, fq.y, ss);
    }
    ss = warp_sum(ss);
    float inv_rms = rsqrtf(ss * (1.0f / 256.0f) + eps);

    float qn[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float norm_only = __bfloat162float(__float2bfloat16_rn(vals[i] * inv_rms));
        qn[i] = __bfloat162float(__float2bfloat16_rn(norm_only * wts[i]));
    }

    uint4 out_u4;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out_u4);
    if (c0 + 7 < rope_angles) {
        uint4 cos_v = __ldg(cs_u4 + vec_idx);
        uint4 sin_v = __ldg(cs_u4 + cache_half8 + vec_idx);
        const __nv_bfloat162* cos2 = reinterpret_cast<const __nv_bfloat162*>(&cos_v);
        const __nv_bfloat162* sin2 = reinterpret_cast<const __nv_bfloat162*>(&sin_v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p0 = __shfl_xor_sync(0xffffffffu, qn[2 * j + 0], 16);
            float p1 = __shfl_xor_sync(0xffffffffu, qn[2 * j + 1], 16);
            float2 fc = __bfloat1622float2(cos2[j]);
            float2 fs = __bfloat1622float2(sin2[j]);
            float y0 = half_id ? (qn[2 * j + 0] * fc.x + p0 * fs.x) : (qn[2 * j + 0] * fc.x - p0 * fs.x);
            float y1 = half_id ? (qn[2 * j + 1] * fc.y + p1 * fs.y) : (qn[2 * j + 1] * fc.y - p1 * fs.y);
            o2[j] = __floats2bfloat162_rn(y0, y1);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            (void)__shfl_xor_sync(0xffffffffu, qn[2 * j + 0], 16);
            (void)__shfl_xor_sync(0xffffffffu, qn[2 * j + 1], 16);
            o2[j] = __floats2bfloat162_rn(qn[2 * j + 0], qn[2 * j + 1]);
        }
    }
    o_u4[c8] = out_u4;
}

__global__ void __launch_bounds__(256)
e2b_q_norm_rope_256_8h_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const int64_t* __restrict__ pos,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ cos_sin_cache,
    int q_row_stride,
    int cache_stride,
    int rope_angles,
    float eps
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int h = tid >> 5;
    const int lane = tid & 31;
    const int half_id = lane >> 4;
    const int vec_idx = lane & 15;
    const int c8 = half_id * 16 + vec_idx;
    const int c0 = vec_idx * 8;
    const int cache_half8 = (cache_stride >> 1) >> 3;

    __shared__ uint4 s_cos[16];
    __shared__ uint4 s_sin[16];
    __shared__ uint4 s_w[32];

    if (tid < 32) {
        s_w[tid] = __ldg(reinterpret_cast<const uint4*>(q_weight) + tid);
        if (tid < 16 && (tid * 8 + 7) < rope_angles) {
            int64_t p = pos[t];
            const uint4* cs_u4 = reinterpret_cast<const uint4*>(cos_sin_cache + (size_t)p * cache_stride);
            s_cos[tid] = __ldg(cs_u4 + tid);
            s_sin[tid] = __ldg(cs_u4 + cache_half8 + tid);
        }
    }
    __syncthreads();

    const uint4* q_u4 = reinterpret_cast<const uint4*>(q + (size_t)t * q_row_stride + (size_t)h * 256);
    uint4* o_u4 = reinterpret_cast<uint4*>(out + ((size_t)t * 8 + h) * 256);

    uint4 qv = __ldg(q_u4 + c8);
    uint4 wv = s_w[c8];
    const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
    float vals[8], wts[8], ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 fq = __bfloat1622float2(q2[j]);
        float2 fw = __bfloat1622float2(w2[j]);
        vals[2 * j + 0] = fq.x;
        vals[2 * j + 1] = fq.y;
        wts[2 * j + 0] = fw.x;
        wts[2 * j + 1] = fw.y;
        ss = __fmaf_rn(fq.x, fq.x, ss);
        ss = __fmaf_rn(fq.y, fq.y, ss);
    }
    ss = warp_sum(ss);
    float inv_rms = rsqrtf(ss * (1.0f / 256.0f) + eps);

    float qn[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float norm_only = __bfloat162float(__float2bfloat16_rn(vals[i] * inv_rms));
        qn[i] = __bfloat162float(__float2bfloat16_rn(norm_only * wts[i]));
    }

    uint4 out_u4;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out_u4);
    if (c0 + 7 < rope_angles) {
        uint4 cos_v = s_cos[vec_idx];
        uint4 sin_v = s_sin[vec_idx];
        const __nv_bfloat162* cos2 = reinterpret_cast<const __nv_bfloat162*>(&cos_v);
        const __nv_bfloat162* sin2 = reinterpret_cast<const __nv_bfloat162*>(&sin_v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p0 = __shfl_xor_sync(0xffffffffu, qn[2 * j + 0], 16);
            float p1 = __shfl_xor_sync(0xffffffffu, qn[2 * j + 1], 16);
            float2 fc = __bfloat1622float2(cos2[j]);
            float2 fs = __bfloat1622float2(sin2[j]);
            float y0 = half_id ? (qn[2 * j + 0] * fc.x + p0 * fs.x) : (qn[2 * j + 0] * fc.x - p0 * fs.x);
            float y1 = half_id ? (qn[2 * j + 1] * fc.y + p1 * fs.y) : (qn[2 * j + 1] * fc.y - p1 * fs.y);
            o2[j] = __floats2bfloat162_rn(y0, y1);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            (void)__shfl_xor_sync(0xffffffffu, qn[2 * j + 0], 16);
            (void)__shfl_xor_sync(0xffffffffu, qn[2 * j + 1], 16);
            o2[j] = __floats2bfloat162_rn(qn[2 * j + 0], qn[2 * j + 1]);
        }
    }
    o_u4[c8] = out_u4;
}

// hd=512, 1 warp/head, paired uint4 (128 <= T <= 2048, 0 SMEM, 0 CALL)
__global__ void __launch_bounds__(32)
e2b_q_norm_rope_512_1w_u4_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const int64_t* __restrict__ pos,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ cos_sin_cache,
    int q_row_stride,
    int cache_stride,
    int n_heads,
    int rope_angles,
    float eps
) {
    const int h_idx = blockIdx.x;
    const int t = h_idx / n_heads;
    const int h = h_idx - t * n_heads;
    const int lane = threadIdx.x;
    const int c0 = lane * 8;
    const int cache_half8 = (cache_stride >> 1) >> 3;
    const int64_t p = pos[t];

    const uint4* q_u4 = reinterpret_cast<const uint4*>(q + (size_t)t * q_row_stride + (size_t)h * 512);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(q_weight);
    uint4* o_u4 = reinterpret_cast<uint4*>(out + (size_t)h_idx * 512);
    const uint4* cs_u4 = reinterpret_cast<const uint4*>(cos_sin_cache + (size_t)p * cache_stride);

    uint4 qv_lo = __ldg(q_u4 + lane), qv_hi = __ldg(q_u4 + 32 + lane);
    uint4 wv_lo = __ldg(w_u4 + lane), wv_hi = __ldg(w_u4 + 32 + lane);
    float nlo[8], nhi[8];
    qnr_paired_u4_norm_step<32, 512>(qv_lo, qv_hi, wv_lo, wv_hi, eps, nlo, nhi);

    bool do_rope = (c0 + 7 < rope_angles);
    uint4 cos_v{}, sin_v{};
    if (do_rope) {
        cos_v = __ldg(cs_u4 + lane);
        sin_v = __ldg(cs_u4 + cache_half8 + lane);
    }
    qnr_paired_u4_store(o_u4, 32, lane, nlo, nhi, do_rope, cos_v, sin_v);
}

// hd=256, 2 heads/warp (T > 512, 0 SMEM, 0 CALL)
__global__ void __launch_bounds__(32)
e2b_q_norm_rope_256_vec_16t_u4_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const int64_t* __restrict__ pos,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ cos_sin_cache,
    int q_row_stride,
    int cache_stride,
    int total_heads,
    int n_heads,
    int rope_angles,
    float eps
) {
    const int h_idx = blockIdx.x * 2 + (threadIdx.x >> 4);
    if (h_idx >= total_heads) return;
    const int t = h_idx / n_heads;
    const int h = h_idx - t * n_heads;
    const int lane16 = threadIdx.x & 15;
    const int c0 = lane16 * 8;
    const int cache_half8 = (cache_stride >> 1) >> 3;
    const int64_t p = pos[t];

    const uint4* q_u4 = reinterpret_cast<const uint4*>(q + (size_t)t * q_row_stride + (size_t)h * 256);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(q_weight);
    uint4* o_u4 = reinterpret_cast<uint4*>(out + (size_t)h_idx * 256);
    const uint4* cs_u4 = reinterpret_cast<const uint4*>(cos_sin_cache + (size_t)p * cache_stride);

    uint4 qv_lo = __ldg(q_u4 + lane16), qv_hi = __ldg(q_u4 + 16 + lane16);
    uint4 wv_lo = __ldg(w_u4 + lane16), wv_hi = __ldg(w_u4 + 16 + lane16);
    float nlo[8], nhi[8];
    qnr_paired_u4_norm_step<16, 256>(qv_lo, qv_hi, wv_lo, wv_hi, eps, nlo, nhi);

    bool do_rope = (c0 + 7 < rope_angles);
    uint4 cos_v{}, sin_v{};
    if (do_rope) {
        cos_v = __ldg(cs_u4 + lane16);
        sin_v = __ldg(cs_u4 + cache_half8 + lane16);
    }
    qnr_paired_u4_store(o_u4, 16, lane16, nlo, nhi, do_rope, cos_v, sin_v);
}

// ---------------------------------------------------------------------------
// 5. Prefill bf16x8 vectorized gelu_tanh_and_mul (exact vLLM BF16 rounding)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
e2b_gelu_tanh_and_mul_bf16x8_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    int M,
    int N_half
) {
    const int64_t n8 = N_half >> 3;
    const int64_t total8 = (int64_t)M * n8;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const uint4* __restrict__ in_u4 = reinterpret_cast<const uint4*>(input);
    uint4* __restrict__ out_u4 = reinterpret_cast<uint4*>(out);

    for (int64_t idx = tid; idx < total8; idx += stride) {
        int64_t row = idx / n8;
        int64_t col8 = idx - row * n8;
        int64_t row_in_base = row * (n8 << 1);
        uint4 g_u4 = __ldg(&in_u4[row_in_base + col8]);
        uint4 u_u4 = __ldg(&in_u4[row_in_base + n8 + col8]);

        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&g_u4);
        const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&u_u4);
        uint4 res_u4;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&res_u4);
        constexpr float BETA = 0.7978845608028654f;
        constexpr float KAPPA = 0.044715f;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 gf = __bfloat1622float2(g2[i]);
            float2 uf = __bfloat1622float2(u2[i]);
            float ix = BETA * (gf.x + KAPPA * gf.x * gf.x * gf.x);
            float iy = BETA * (gf.y + KAPPA * gf.y * gf.y * gf.y);
            float tx, ty;
            asm("tanh.approx.f32 %0, %1;" : "=f"(tx) : "f"(ix));
            asm("tanh.approx.f32 %0, %1;" : "=f"(ty) : "f"(iy));
            float ax = __bfloat162float(__float2bfloat16_rn(0.5f * gf.x * (1.0f + tx)));
            float ay = __bfloat162float(__float2bfloat16_rn(0.5f * gf.y * (1.0f + ty)));
            o2[i] = __floats2bfloat162_rn(ax * uf.x, ay * uf.y);
        }
        out_u4[idx] = res_u4;
    }
}

// ---------------------------------------------------------------------------

} // namespace

// --- Host launchers ---

void e2b_launch_q_norm_rope(
    torch::Tensor out,
    torch::Tensor q,
    torch::Tensor pos,
    torch::Tensor q_weight,
    torch::Tensor cos_sin_cache,
    int head_dim,
    int rope_angles,
    double eps
) {
    int T = q.size(0);
    if (T == 0) return;
    int n_heads = q.size(1) / head_dim;
    int q_row_stride = q.stride(0);
    int cache_stride = cos_sin_cache.size(1);
    int blocks = T * n_heads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (head_dim == 256) {
        bool vec_ok = (
            ((reinterpret_cast<uintptr_t>(q_weight.data_ptr()) & 15) == 0)
            && ((reinterpret_cast<uintptr_t>(cos_sin_cache.data_ptr()) & 15) == 0)
            && (cos_sin_cache.stride(0) == cache_stride)
            && ((rope_angles & 7) == 0)
        );
        bool vec16t_ok = (
            vec_ok && rope_angles <= 128
            && ((reinterpret_cast<uintptr_t>(q.data_ptr()) & 15) == 0)
            && ((reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0)
            && ((q_row_stride & 7) == 0)
            && ((cache_stride & 15) == 0)
        );
        bool vec_8h_ok = (
            vec_ok && T >= 17 && n_heads == 8 && rope_angles <= 128
            && ((reinterpret_cast<uintptr_t>(q.data_ptr()) & 15) == 0)
            && ((reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0)
            && ((q_row_stride & 7) == 0)
        );
        if (vec16t_ok && T > 512) {
            e2b_q_norm_rope_256_vec_16t_u4_kernel<<<(blocks + 1) / 2, 32, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                pos.data_ptr<int64_t>(),
                reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
                q_row_stride, cache_stride, blocks, n_heads, rope_angles, static_cast<float>(eps));
        } else if (vec_8h_ok) {
            e2b_q_norm_rope_256_8h_kernel<<<T, 256, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                pos.data_ptr<int64_t>(),
                reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
                q_row_stride, cache_stride, rope_angles, static_cast<float>(eps));
        } else if (vec_ok) {
            e2b_q_norm_rope_vec_256_kernel<<<blocks, 32, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                pos.data_ptr<int64_t>(),
                reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
                q_row_stride, cache_stride, n_heads, rope_angles, static_cast<float>(eps));
        } else {
            e2b_q_norm_rope_kernel<256><<<blocks, 32, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                pos.data_ptr<int64_t>(),
                reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
                q_row_stride, cache_stride, n_heads, rope_angles, static_cast<float>(eps));
        }
    } else if (head_dim == 512) {
        bool vec512_ok = (
            ((reinterpret_cast<uintptr_t>(q.data_ptr()) & 15) == 0)
            && ((reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0)
            && ((reinterpret_cast<uintptr_t>(q_weight.data_ptr()) & 15) == 0)
            && ((reinterpret_cast<uintptr_t>(cos_sin_cache.data_ptr()) & 15) == 0)
            && (cos_sin_cache.stride(0) == cache_stride)
            && ((q_row_stride & 7) == 0)
            && ((cache_stride & 1) == 0)
            && ((rope_angles & 1) == 0)
            && rope_angles <= 256
        );
        if (vec512_ok && (cache_stride & 15) == 0 && (rope_angles & 7) == 0 && T >= 128 && T <= 2048) {
            e2b_q_norm_rope_512_1w_u4_kernel<<<blocks, 32, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                pos.data_ptr<int64_t>(),
                reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
                q_row_stride, cache_stride, n_heads, rope_angles, static_cast<float>(eps));
        } else if (vec512_ok && T >= 128) {
            e2b_q_norm_rope_512_2w_kernel<<<blocks, 64, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                pos.data_ptr<int64_t>(),
                reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
                q_row_stride, cache_stride, n_heads, rope_angles, static_cast<float>(eps));
        } else {
            e2b_q_norm_rope_kernel<512><<<blocks, 32, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
                pos.data_ptr<int64_t>(),
                reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
                q_row_stride, cache_stride, n_heads, rope_angles, static_cast<float>(eps));
        }
    } else {
        TORCH_CHECK(false, "Unsupported head_dim for e2b_launch_q_norm_rope: ", head_dim);
    }
}

void e2b_launch_gelu_tanh_and_mul_bf16x8(torch::Tensor out, torch::Tensor input) {
    int M = input.size(0);
    int N_half = input.size(1) / 2;
    if (M == 0) return;
    int64_t total8 = (int64_t)M * (N_half >> 3);
    int threads = 256;
    int blocks = static_cast<int>((total8 + threads - 1) / threads);
    if (blocks > 864) blocks = 864;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    e2b_gelu_tanh_and_mul_bf16x8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        M, N_half);
}

#ifdef GEMMA4_HAS_CUTLASS_DUAL
template <int BM, int BN, int BK, int WM, int WN, int WK, int STAGES>
bool launch_cutlass_dual_tpl(
    __nv_bfloat16* out,
    const __nv_bfloat16* x,
    const __nv_bfloat16* w_gate,
    const __nv_bfloat16* w_up,
    int M, int N_half, int K,
    cudaStream_t stream)
{
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementAcc = float;
    using ElementCompute = float;

    using EpilogueOp01 = cutlass::epilogue::thread::LinearCombination<
        ElementC, 8, ElementAcc, ElementCompute, cutlass::epilogue::thread::ScaleType::Nothing>;
    using EpilogueOp2 = cutlass::epilogue::thread::LeftGeluTanhAndMul<
        ElementC, 8, ElementC, ElementCompute>;

    using DualGemm = cutlass::gemm::device::DualGemm<
        ElementA, cutlass::layout::RowMajor,
        ElementB, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,
        ElementC, cutlass::layout::RowMajor,
        ElementAcc,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<BM, BN, BK>,
        cutlass::gemm::GemmShape<WM, WN, WK>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        EpilogueOp01,
        EpilogueOp01,
        EpilogueOp2,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        STAGES,
        false,
        false,
        false,
        8,
        8>;

    cutlass::gemm::GemmCoord prob(M, N_half, K);
    typename DualGemm::Arguments args(
        cutlass::gemm::DualGemmMode::kGemm,
        prob,
        {reinterpret_cast<const ElementA*>(x), K},
        {reinterpret_cast<const ElementB*>(w_gate), K},
        {nullptr, N_half},
        {nullptr, N_half},
        {reinterpret_cast<const ElementB*>(w_up), K},
        {nullptr, N_half},
        {nullptr, N_half},
        {reinterpret_cast<ElementC*>(out), N_half},
        {1.0f, 0.0f},
        {1.0f, 0.0f},
        {},
        1, 1, 0, 0, 0, 0, 0);

    DualGemm op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) return false;
    if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
    return op.run(stream) == cutlass::Status::kSuccess;
}
#endif

bool e2b_launch_dual_gemm_gelu(torch::Tensor out, torch::Tensor x, torch::Tensor w, int variant) {
#ifdef GEMMA4_HAS_CUTLASS_DUAL
    int M = x.size(0), K = x.size(1), N_half = w.size(0) / 2;
    if (M == 0) return true;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto* wg = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    auto* wu = wg + (int64_t)N_half * K;
    if (variant == 0) return launch_cutlass_dual_tpl<128, 64, 32, 64, 32, 32, 3>(o, xp, wg, wu, M, N_half, K, stream);
    if (variant == 1) return launch_cutlass_dual_tpl<128, 64, 32, 64, 32, 32, 5>(o, xp, wg, wu, M, N_half, K, stream);
    if (variant == 2) return launch_cutlass_dual_tpl<64, 64, 32, 32, 32, 32, 5>(o, xp, wg, wu, M, N_half, K, stream);
    return false;
#else
    return false;
#endif
}

namespace e2b_s3 {

__device__ __forceinline__ uint32_t s3_smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void s3_cp_async_16B(void* smem_dst, const void* glob_src) {
  uint32_t s = s3_smem_u32(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(glob_src) : "memory");
}

__device__ __forceinline__ void s3_cp_async_cg_16B(void* smem_dst, const void* glob_src) {
  uint32_t s = s3_smem_u32(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(glob_src) : "memory");
}

__device__ __forceinline__ void s3_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void s3_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void s3_ldmatrix_x4(uint32_t (&r)[4], const void* smem_src) {
  uint32_t s = s3_smem_u32(smem_src);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void s3_ldmatrix_x2(uint32_t (&r)[2], const void* smem_src) {
  uint32_t s = s3_smem_u32(smem_src);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

__device__ __forceinline__ void s3_mma_m16n8k16_bf16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ __nv_bfloat16 s3_gelu_mul_bf16(float g, float u) {
  constexpr float kBeta = 0.7978845608028654f;
  constexpr float kKappa = 0.044715f;
  float inner = kBeta * (g + kKappa * g * g * g);
  float tanh_v;
  asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_v) : "f"(inner));
  float gelu_g = 0.5f * g * (1.0f + tanh_v);
  return __hmul(__float2bfloat16_rn(gelu_g), __float2bfloat16_rn(u));
}

__device__ __forceinline__ int s3_swz64_col(int row, int col) {
  return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ void s3_cp_async_ca_zero(void* smem_dst, const void* glob_src, int pred) {
  uint32_t s = s3_smem_u32(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(s), "l"(glob_src), "r"(pred ? 16 : 0) : "memory");
}

__device__ __forceinline__ void s3_cp_async_cg_zero(void* smem_dst, const void* glob_src, int pred) {
  uint32_t s = s3_smem_u32(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(s), "l"(glob_src), "r"(pred ? 16 : 0) : "memory");
}

__device__ __forceinline__ uint64_t s3_l2_evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}

__device__ __forceinline__ void s3_cp_async_cg_evict_zero(
    void* smem_dst, const void* glob_src, int pred, uint64_t pol) {
  uint32_t s = s3_smem_u32(smem_dst);
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
               :: "r"(s), "l"(glob_src), "r"(pred ? 16 : 0), "l"(pol) : "memory");
}

template <int EVICT_W>
__device__ __forceinline__ void s3_cp_async_cg_w(
    void* smem_dst, const void* glob_src, int pred, uint64_t pol) {
  if constexpr (EVICT_W) {
    s3_cp_async_cg_evict_zero(smem_dst, glob_src, pred, pol);
  } else {
    s3_cp_async_cg_zero(smem_dst, glob_src, pred);
  }
}

template <typename Fn>
static inline void s3_set_max_smem(Fn fn, size_t smem_bytes) {
  if (smem_bytes > 49152) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  }
}

__device__ __forceinline__ int4 s3_ld_cs_int4(const int4* ptr) {
  int4 r;
  asm volatile("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(ptr));
  return r;
}

template <int ROWS, int T_PER_ROW, int M_VAL, int USE_CS>
__global__ void s3_preload_gu_gemv_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ w_up,
    int nh) {
  constexpr int K = 1536;
  constexpr int K8 = 192;
  constexpr int ITERS = K8 / T_PER_ROW;
  const int tid = threadIdx.x;
  const int row_in_cta = tid / T_PER_ROW;
  const int lane_in_row = tid % T_PER_ROW;
  const int row = blockIdx.x * ROWS + row_in_cta;
  if (row >= nh) return;

  const int4* wg8 = reinterpret_cast<const int4*>(w_gate + static_cast<int64_t>(row) * K);
  const int4* wu8 = reinterpret_cast<const int4*>(w_up + static_cast<int64_t>(row) * K);
  const int4* x8_0 = reinterpret_cast<const int4*>(x);
  const int4* x8_1 = (M_VAL > 1) ? reinterpret_cast<const int4*>(x + K) : nullptr;

  int4 vg[ITERS], vu[ITERS], vx0[ITERS], vx1[ITERS];
  #pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    int idx = lane_in_row + it * T_PER_ROW;
    vg[it] = USE_CS ? s3_ld_cs_int4(wg8 + idx) : __ldg(wg8 + idx);
    vu[it] = USE_CS ? s3_ld_cs_int4(wu8 + idx) : __ldg(wu8 + idx);
    vx0[it] = __ldg(x8_0 + idx);
    if constexpr (M_VAL > 1) vx1[it] = __ldg(x8_1 + idx);
  }

  float g0 = 0.f, u0 = 0.f, g1 = 0.f, u1 = 0.f;
  #pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const __nv_bfloat162* wg2 = reinterpret_cast<const __nv_bfloat162*>(&vg[it]);
    const __nv_bfloat162* wu2 = reinterpret_cast<const __nv_bfloat162*>(&vu[it]);
    const __nv_bfloat162* a0 = reinterpret_cast<const __nv_bfloat162*>(&vx0[it]);
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      float2 fg = __bfloat1622float2(wg2[c]);
      float2 fu = __bfloat1622float2(wu2[c]);
      float2 f0 = __bfloat1622float2(a0[c]);
      g0 = fmaf(f0.x, fg.x, fmaf(f0.y, fg.y, g0));
      u0 = fmaf(f0.x, fu.x, fmaf(f0.y, fu.y, u0));
    }
    if constexpr (M_VAL > 1) {
      const __nv_bfloat162* a1 = reinterpret_cast<const __nv_bfloat162*>(&vx1[it]);
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

  #pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    g0 += __shfl_xor_sync(0xffffffffu, g0, off, 32);
    u0 += __shfl_xor_sync(0xffffffffu, u0, off, 32);
    if constexpr (M_VAL > 1) {
      g1 += __shfl_xor_sync(0xffffffffu, g1, off, 32);
      u1 += __shfl_xor_sync(0xffffffffu, u1, off, 32);
    }
  }

  if constexpr (T_PER_ROW >= 64) {
    constexpr int NW = T_PER_ROW >> 5;
    __shared__ float smem[ROWS][NW][2 * M_VAL];
    const int ww = lane_in_row >> 5;
    if ((lane_in_row & 31) == 0) {
      smem[row_in_cta][ww][0] = g0;
      smem[row_in_cta][ww][1] = u0;
      if constexpr (M_VAL > 1) {
        smem[row_in_cta][ww][2] = g1;
        smem[row_in_cta][ww][3] = u1;
      }
    }
    __syncthreads();
    if (lane_in_row == 0) {
      float sg0 = 0.f, su0 = 0.f, sg1 = 0.f, su1 = 0.f;
      #pragma unroll
      for (int w = 0; w < NW; ++w) {
        sg0 += smem[row_in_cta][w][0];
        su0 += smem[row_in_cta][w][1];
        if constexpr (M_VAL > 1) {
          sg1 += smem[row_in_cta][w][2];
          su1 += smem[row_in_cta][w][3];
        }
      }
      out[row] = s3_gelu_mul_bf16(sg0, su0);
      if constexpr (M_VAL > 1) out[nh + row] = s3_gelu_mul_bf16(sg1, su1);
    }
  } else {
    if (lane_in_row == 0) {
      out[row] = s3_gelu_mul_bf16(g0, u0);
      if constexpr (M_VAL > 1) out[nh + row] = s3_gelu_mul_bf16(g1, u1);
    }
  }
}

__device__ __forceinline__ void s18_cp_async_ca_zero(void* smem_dst, const void* glob_src, int pred) {
  uint32_t s = s3_smem_u32(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(s), "l"(glob_src), "r"(pred ? 16 : 0) : "memory");
}

__device__ __forceinline__ void s18_cp_async_ca_zero_old(void* smem_dst, const void* glob_src, int pred) {
  uint32_t s = s3_smem_u32(smem_dst);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %2, 0;\n"
      "  @p cp.async.ca.shared.global [%0], [%1], 16;\n"
      "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
      "}\n" :: "r"(s), "l"(glob_src), "r"(pred));
}

template <int EVICT_W>
__device__ __forceinline__ void s18_cp_async_cg_w(
    void* smem_dst, const void* glob_src, int pred, uint64_t pol) {
  uint32_t s = s3_smem_u32(smem_dst);
  if constexpr (EVICT_W) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
                 :: "r"(s), "l"(glob_src), "r"(pred ? 16 : 0), "l"(pol) : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(glob_src), "r"(pred ? 16 : 0) : "memory");
  }
}

template <int EVICT_W>
__device__ __forceinline__ void s18_cp_async_cg_w_old(
    void* smem_dst, const void* glob_src, int pred, uint64_t pol) {
  uint32_t s = s3_smem_u32(smem_dst);
  if constexpr (EVICT_W) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %3;\n"
        "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
        "}\n" :: "r"(s), "l"(glob_src), "r"(pred), "l"(pol) : "memory");
  } else {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
        "}\n" :: "r"(s), "l"(glob_src), "r"(pred) : "memory");
  }
}

template <int EVICT_W>
__device__ __forceinline__ void s18_cp_async_cg_uncond(
    void* smem_dst, const void* glob_src, uint64_t pol) {
  uint32_t s = s3_smem_u32(smem_dst);
  if constexpr (EVICT_W) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(glob_src), "l"(pol) : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(s), "l"(glob_src) : "memory");
  }
}

__device__ __forceinline__ void s18_mma_acc(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int K_DIM, int K_SPLITS, int TILES_N, int EVICT_W, bool EXACT_N>
__global__ void s18_gu_ksplit_mma_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ w_up,
    int M,
    int nh) {
  constexpr int BK = 64, A_ELEMS = 16 * BK, B_ROWS = 2 * TILES_N * 16, B_ELEMS = B_ROWS * BK;
  constexpr int HALF_B = TILES_N * 16, HALF_CH = (HALF_B * 8) / 32, NT = TILES_N * 2;
  const int tid = threadIdx.x, ks = tid >> 5, lane = tid & 31, n_base = blockIdx.x * HALF_B;
  if (n_base >= nh) return;
  const uint64_t pol = EVICT_W ? s3_l2_evict_first_policy() : 0ull;

  extern __shared__ char dyn_smem[];
  __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(
      dyn_smem + static_cast<size_t>(K_SPLITS) * 2 * A_ELEMS * sizeof(__nv_bfloat16));
  __nv_bfloat16* wa = smem_a + (ks * 2) * A_ELEMS;
  __nv_bfloat16* wb = smem_b + (ks * 2) * B_ELEMS;

  constexpr int chunks_total = K_DIM / BK;
  constexpr int chunks_per = (chunks_total + K_SPLITS - 1) / K_SPLITS;
  const int ch_start = ks * chunks_per;
  const int ch_end = min(chunks_total, ch_start + chunks_per);
  const int n_ch = max(0, ch_end - ch_start);

  const int r0_ld = lane >> 3, c8_ld = (lane & 7) << 3;
  const __nv_bfloat16* x_cur = x + static_cast<int64_t>(r0_ld) * K_DIM + ch_start * BK + c8_ld;
  const __nv_bfloat16* wg_cur = w_gate + static_cast<int64_t>(n_base + r0_ld) * K_DIM + ch_start * BK + c8_ld;
  const __nv_bfloat16* wu_cur = w_up + static_cast<int64_t>(n_base + r0_ld) * K_DIM + ch_start * BK + c8_ld;

  uint32_t x_mask = 0;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    if ((r0_ld + i * 4) < M) x_mask |= (1u << i);
  }
  uint32_t w_mask = 0;
  if constexpr (!EXACT_N) {
    #pragma unroll
    for (int i = 0; i < HALF_CH; ++i) {
      if ((n_base + r0_ld + i * 4) < nh) w_mask |= (1u << i);
    }
  }

  float acc_g[NT][4], acc_u[NT][4];
  #pragma unroll
  for (int t = 0; t < NT; ++t) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) { acc_g[t][i] = 0.0f; acc_u[t][i] = 0.0f; }
  }

  auto issue_stage = [&](int stage) {
    __nv_bfloat16* sa = wa + stage * A_ELEMS;
    __nv_bfloat16* sb = wb + stage * B_ELEMS;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int r = r0_ld + i * 4, valid = ((x_mask >> i) & 1u) ? 1 : 0;
      s18_cp_async_ca_zero(&sa[r * BK + s3_swz64_col(r, c8_ld)],
                           valid ? (x_cur + static_cast<int64_t>(i * 4) * K_DIM) : x, valid);
    }
    x_cur += BK;
    #pragma unroll
    for (int i = 0; i < HALF_CH; ++i) {
      int r = r0_ld + i * 4, ru = HALF_B + r;
      if constexpr (EXACT_N) {
        s18_cp_async_cg_uncond<EVICT_W>(&sb[r * BK + s3_swz64_col(r, c8_ld)],
                                        wg_cur + static_cast<int64_t>(i * 4) * K_DIM, pol);
        s18_cp_async_cg_uncond<EVICT_W>(&sb[ru * BK + s3_swz64_col(ru, c8_ld)],
                                        wu_cur + static_cast<int64_t>(i * 4) * K_DIM, pol);
      } else {
        int valid = ((w_mask >> i) & 1u) ? 1 : 0;
        s18_cp_async_cg_w<EVICT_W>(&sb[r * BK + s3_swz64_col(r, c8_ld)],
                                   valid ? (wg_cur + static_cast<int64_t>(i * 4) * K_DIM) : w_gate, valid, pol);
        s18_cp_async_cg_w<EVICT_W>(&sb[ru * BK + s3_swz64_col(ru, c8_ld)],
                                   valid ? (wu_cur + static_cast<int64_t>(i * 4) * K_DIM) : w_up, valid, pol);
      }
    }
    wg_cur += BK;
    wu_cur += BK;
    s3_cp_async_commit();
  };

  if (n_ch > 0) issue_stage(0);
  const int a_r = lane & 15, a_c = ((lane >> 4) & 1) << 3;
  const int b_off_row = (((lane >> 4) & 1) << 3) + (lane & 7);
  const int b_off_col = (((lane >> 3) & 1) << 3);

  for (int c = 0; c < n_ch; ++c) {
    int st = c & 1;
    if (c + 1 < n_ch) {
      __syncwarp();
      issue_stage(st ^ 1);
      s3_cp_async_wait<1>();
    } else {
      s3_cp_async_wait<0>();
    }
    __syncwarp();
    __nv_bfloat16* sa = wa + st * A_ELEMS;
    __nv_bfloat16* sb = wb + st * B_ELEMS;

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t ra[4];
      s3_ldmatrix_x4(ra, &sa[a_r * BK + s3_swz64_col(a_r, kk + a_c)]);
      #pragma unroll
      for (int tn = 0; tn < TILES_N; ++tn) {
        uint32_t rbg[4], rbu[4];
        int rg = tn * 16 + b_off_row, ru = HALF_B + rg;
        s3_ldmatrix_x4(rbg, &sb[rg * BK + s3_swz64_col(rg, kk + b_off_col)]);
        s3_ldmatrix_x4(rbu, &sb[ru * BK + s3_swz64_col(ru, kk + b_off_col)]);
        uint32_t bg0[2] = {rbg[0], rbg[1]}, bg1[2] = {rbg[2], rbg[3]};
        uint32_t bu0[2] = {rbu[0], rbu[1]}, bu1[2] = {rbu[2], rbu[3]};
        s18_mma_acc(acc_g[tn * 2 + 0], ra, bg0);
        s18_mma_acc(acc_g[tn * 2 + 1], ra, bg1);
        s18_mma_acc(acc_u[tn * 2 + 0], ra, bu0);
        s18_mma_acc(acc_u[tn * 2 + 1], ra, bu1);
      }
    }
  }

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

  int r0 = lane >> 2, r1 = r0 + 8, c0 = (lane & 3) * 2;
  #pragma unroll
  for (int t = 0; t < NT; ++t) {
    if ((t % K_SPLITS) == ks) {
      int n_col = n_base + t * 8 + c0;
      if (EXACT_N || n_col + 1 < nh) {
        float sg0 = 0.f, sg1 = 0.f, sg2 = 0.f, sg3 = 0.f;
        float su0 = 0.f, su1 = 0.f, su2 = 0.f, su3 = 0.f;
        #pragma unroll
        for (int k = 0; k < K_SPLITS; ++k) {
          sg0 += red_g[k * PER_KS + t * 128 + lane * 4 + 0]; su0 += red_u[k * PER_KS + t * 128 + lane * 4 + 0];
          sg1 += red_g[k * PER_KS + t * 128 + lane * 4 + 1]; su1 += red_u[k * PER_KS + t * 128 + lane * 4 + 1];
          sg2 += red_g[k * PER_KS + t * 128 + lane * 4 + 2]; su2 += red_u[k * PER_KS + t * 128 + lane * 4 + 2];
          sg3 += red_g[k * PER_KS + t * 128 + lane * 4 + 3]; su3 += red_u[k * PER_KS + t * 128 + lane * 4 + 3];
        }
        if (r0 < M) {
          out[static_cast<int64_t>(r0) * nh + n_col + 0] = exact_vllm_gelu_tanh_mul(sg0, su0);
          out[static_cast<int64_t>(r0) * nh + n_col + 1] = exact_vllm_gelu_tanh_mul(sg1, su1);
        }
        if (r1 < M) {
          out[static_cast<int64_t>(r1) * nh + n_col + 0] = exact_vllm_gelu_tanh_mul(sg2, su2);
          out[static_cast<int64_t>(r1) * nh + n_col + 1] = exact_vllm_gelu_tanh_mul(sg3, su3);
        }
      }
    }
  }
}

template <int K_SPLITS, int TILES_N, int EVICT_W = 0>
__global__ void s3_lin_ksplit_bk64_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int M,
    int N,
    int K) {
  constexpr int BK = 64, A_STRIDE = 64, B_STRIDE = 64;
  constexpr int B_ROWS = TILES_N * 16;

  const int tid = threadIdx.x;
  const int ks = tid >> 5;
  const int lane = tid & 31;
  const int n_base = blockIdx.x * B_ROWS;
  if (n_base >= N) return;
  const uint64_t pol = EVICT_W ? s3_l2_evict_first_policy() : 0ull;

  extern __shared__ char dyn_smem[];
  __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  const size_t a_bytes = static_cast<size_t>(K_SPLITS) * 2 * 16 * A_STRIDE * sizeof(__nv_bfloat16);
  __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(dyn_smem + a_bytes);

  __nv_bfloat16* wa = smem_a + static_cast<size_t>(ks) * 2 * 16 * A_STRIDE;
  __nv_bfloat16* wb = smem_b + static_cast<size_t>(ks) * 2 * B_ROWS * B_STRIDE;

  const int chunks_total = K / BK;
  const int chunks_per = (chunks_total + K_SPLITS - 1) / K_SPLITS;
  const int ch_start = ks * chunks_per;
  const int ch_end = min(chunks_total, ch_start + chunks_per);
  const int n_ch = max(0, ch_end - ch_start);

  float acc[TILES_N * 2][4];
  #pragma unroll
  for (int t = 0; t < TILES_N * 2; ++t) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc[t][i] = 0.0f;
  }

  auto issue_stage = [&](int stage, int k0) {
    __nv_bfloat16* sa = wa + stage * (16 * A_STRIDE);
    __nv_bfloat16* sb = wb + stage * (B_ROWS * B_STRIDE);
    #pragma unroll
    for (int e = lane; e < 128; e += 32) {
      int r = e >> 3, c8 = (e & 7) << 3;
      int valid = (r < M) ? 1 : 0;
      s3_cp_async_ca_zero(&sa[r * A_STRIDE + s3_swz64_col(r, c8)],
                          valid ? &x[static_cast<int64_t>(r) * K + k0 + c8] : x, valid);
    }
    #pragma unroll
    for (int e = lane; e < B_ROWS * 8; e += 32) {
      int r = e >> 3, c8 = (e & 7) << 3, gn = n_base + r;
      int valid = (gn < N) ? 1 : 0;
      s3_cp_async_cg_w<EVICT_W>(&sb[r * B_STRIDE + s3_swz64_col(r, c8)],
                                valid ? &w[static_cast<int64_t>(gn) * K + k0 + c8] : w, valid, pol);
    }
    s3_cp_async_commit();
  };

  if (n_ch > 0) issue_stage(0, ch_start * BK);
  const int a_r = lane & 15, a_c = ((lane >> 4) & 1) << 3;
  const int b_off_row = (((lane >> 4) & 1) << 3) + (lane & 7);
  const int b_off_col = (((lane >> 3) & 1) << 3);

  for (int c = 0; c < n_ch; ++c) {
    int st = c & 1;
    if (c + 1 < n_ch) {
      __syncwarp();
      issue_stage(st ^ 1, (ch_start + c + 1) * BK);
      s3_cp_async_wait<1>();
    } else {
      s3_cp_async_wait<0>();
    }
    __syncwarp();
    __nv_bfloat16* sa = wa + st * (16 * A_STRIDE);
    __nv_bfloat16* sb = wb + st * (B_ROWS * B_STRIDE);
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t ra[4];
      s3_ldmatrix_x4(ra, &sa[a_r * A_STRIDE + s3_swz64_col(a_r, kk + a_c)]);
      #pragma unroll
      for (int tn = 0; tn < TILES_N; ++tn) {
        uint32_t rb[4];
        int r_s = tn * 16 + b_off_row;
        s3_ldmatrix_x4(rb, &sb[r_s * B_STRIDE + s3_swz64_col(r_s, kk + b_off_col)]);
        uint32_t b0[2] = {rb[0], rb[1]}, b1[2] = {rb[2], rb[3]};
        s3_mma_m16n8k16_bf16(acc[tn * 2 + 0], ra, b0, acc[tn * 2 + 0]);
        s3_mma_m16n8k16_bf16(acc[tn * 2 + 1], ra, b1, acc[tn * 2 + 1]);
      }
    }
  }

  __syncthreads();
  float* red = reinterpret_cast<float*>(dyn_smem);
  constexpr int PER_KS = TILES_N * 2 * 128;
  #pragma unroll
  for (int t = 0; t < TILES_N * 2; ++t) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) red[ks * PER_KS + t * 128 + lane * 4 + i] = acc[t][i];
  }
  __syncthreads();

  int r0 = lane >> 2, r1 = r0 + 8, c0 = (lane & 3) * 2;
  #pragma unroll
  for (int t = 0; t < TILES_N * 2; ++t) {
    if ((t % K_SPLITS) == ks) {
      int n_col = n_base + t * 8 + c0;
      if (n_col + 1 < N) {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        #pragma unroll
        for (int k = 0; k < K_SPLITS; ++k) {
          s0 += red[k * PER_KS + t * 128 + lane * 4 + 0];
          s1 += red[k * PER_KS + t * 128 + lane * 4 + 1];
          s2 += red[k * PER_KS + t * 128 + lane * 4 + 2];
          s3 += red[k * PER_KS + t * 128 + lane * 4 + 3];
        }
        if (r0 < M) *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r0) * N + n_col]) = __floats2bfloat162_rn(s0, s1);
        if (r1 < M) *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r1) * N + n_col]) = __floats2bfloat162_rn(s2, s3);
      }
    }
  }
}

template <int K_SPLITS, int TILES_N, int EVICT_W = 0>
__global__ void s3_lin_ksplit_m32_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int M,
    int N,
    int K) {
  constexpr int BK = 64, A_STRIDE = 64, B_STRIDE = 64, MT = 2, A_ROWS = 32, B_ROWS = TILES_N * 16;
  const int tid = threadIdx.x, ks = tid >> 5, lane = tid & 31;
  const int n_base = blockIdx.x * B_ROWS;
  if (n_base >= N) return;
  const uint64_t pol = EVICT_W ? s3_l2_evict_first_policy() : 0ull;

  extern __shared__ char dyn_smem[];
  __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  const size_t a_bytes = static_cast<size_t>(K_SPLITS) * 2 * A_ROWS * A_STRIDE * sizeof(__nv_bfloat16);
  __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(dyn_smem + a_bytes);
  __nv_bfloat16* wa = smem_a + static_cast<size_t>(ks) * 2 * A_ROWS * A_STRIDE;
  __nv_bfloat16* wb = smem_b + static_cast<size_t>(ks) * 2 * B_ROWS * B_STRIDE;

  const int chunks_total = K / BK;
  const int chunks_per = (chunks_total + K_SPLITS - 1) / K_SPLITS;
  const int ch_start = ks * chunks_per;
  const int ch_end = min(chunks_total, ch_start + chunks_per);
  const int n_ch = max(0, ch_end - ch_start);

  float acc[MT][TILES_N * 2][4];
  #pragma unroll
  for (int mt = 0; mt < MT; ++mt)
    #pragma unroll
    for (int t = 0; t < TILES_N * 2; ++t)
      for (int i = 0; i < 4; ++i) acc[mt][t][i] = 0.f;

  auto issue_stage = [&](int stage, int k0) {
    __nv_bfloat16* sa = wa + stage * (A_ROWS * A_STRIDE);
    __nv_bfloat16* sb = wb + stage * (B_ROWS * B_STRIDE);
    #pragma unroll
    for (int e = lane; e < A_ROWS * 8; e += 32) {
      int r = e >> 3, c8 = (e & 7) << 3, valid = (r < M) ? 1 : 0;
      s3_cp_async_ca_zero(&sa[r * A_STRIDE + s3_swz64_col(r, c8)],
                          valid ? &x[static_cast<int64_t>(r) * K + k0 + c8] : x, valid);
    }
    #pragma unroll
    for (int e = lane; e < B_ROWS * 8; e += 32) {
      int r = e >> 3, c8 = (e & 7) << 3, gn = n_base + r, valid = (gn < N) ? 1 : 0;
      s3_cp_async_cg_w<EVICT_W>(&sb[r * B_STRIDE + s3_swz64_col(r, c8)],
                                valid ? &w[static_cast<int64_t>(gn) * K + k0 + c8] : w, valid, pol);
    }
    s3_cp_async_commit();
  };
  if (n_ch > 0) issue_stage(0, ch_start * BK);
  const int a_r = lane & 15, a_c = ((lane >> 4) & 1) << 3;
  const int b_off_row = (((lane >> 4) & 1) << 3) + (lane & 7), b_off_col = (((lane >> 3) & 1) << 3);

  for (int c = 0; c < n_ch; ++c) {
    int st = c & 1;
    if (c + 1 < n_ch) {
      __syncwarp();
      issue_stage(st ^ 1, (ch_start + c + 1) * BK);
      s3_cp_async_wait<1>();
    } else {
      s3_cp_async_wait<0>();
    }
    __syncwarp();
    __nv_bfloat16* sa = wa + st * (A_ROWS * A_STRIDE);
    __nv_bfloat16* sb = wb + st * (B_ROWS * B_STRIDE);
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t ra[MT][4];
      #pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        int rs = mt * 16 + a_r;
        s3_ldmatrix_x4(ra[mt], &sa[rs * A_STRIDE + s3_swz64_col(rs, kk + a_c)]);
      }
      #pragma unroll
      for (int tn = 0; tn < TILES_N; ++tn) {
        uint32_t rb[4];
        int r_s = tn * 16 + b_off_row;
        s3_ldmatrix_x4(rb, &sb[r_s * B_STRIDE + s3_swz64_col(r_s, kk + b_off_col)]);
        uint32_t b0[2] = {rb[0], rb[1]}, b1[2] = {rb[2], rb[3]};
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          s3_mma_m16n8k16_bf16(acc[mt][tn * 2 + 0], ra[mt], b0, acc[mt][tn * 2 + 0]);
          s3_mma_m16n8k16_bf16(acc[mt][tn * 2 + 1], ra[mt], b1, acc[mt][tn * 2 + 1]);
        }
      }
    }
  }
  __syncthreads();
  float* red = reinterpret_cast<float*>(dyn_smem);
  constexpr int NT = TILES_N * 2;
  constexpr int PER_KS = MT * NT * 128;
  #pragma unroll
  for (int mt = 0; mt < MT; ++mt)
    #pragma unroll
    for (int t = 0; t < NT; ++t)
      #pragma unroll
      for (int i = 0; i < 4; ++i)
        red[ks * PER_KS + (mt * NT + t) * 128 + lane * 4 + i] = acc[mt][t][i];
  __syncthreads();

  int r0_base = lane >> 2, c0 = (lane & 3) * 2;
  #pragma unroll
  for (int idx = 0; idx < MT * NT; ++idx) {
    if ((idx % K_SPLITS) == ks) {
      int mt = idx / NT, t = idx - mt * NT;
      int r0 = mt * 16 + r0_base, r1 = r0 + 8;
      int n_col = n_base + t * 8 + c0;
      if (n_col + 1 < N) {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        #pragma unroll
        for (int k = 0; k < K_SPLITS; ++k) {
          size_t off = static_cast<size_t>(k) * PER_KS + idx * 128 + lane * 4;
          s0 += red[off + 0]; s1 += red[off + 1]; s2 += red[off + 2]; s3 += red[off + 3];
        }
        if (r0 < M) *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r0) * N + n_col]) = __floats2bfloat162_rn(s0, s1);
        if (r1 < M) *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r1) * N + n_col]) = __floats2bfloat162_rn(s2, s3);
      }
    }
  }
}

template <int ROWS, int K_DIM, int USE_CS, int UNROLL>
__global__ void s18_wide_gemv_m1_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N) {
  constexpr int K8 = K_DIM >> 3;
  constexpr int STEP = 32 * UNROLL;
  static_assert((K8 % STEP) == 0, "K8 must divide 32 * UNROLL");
  constexpr int N_IT = K8 / STEP;

  const int tid = threadIdx.x;
  const int row_in_cta = tid >> 5;
  const int lane = tid & 31;
  const int row = blockIdx.x * ROWS + row_in_cta;
  if (row >= N) return;

  const int4* w8 = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * K_DIM);
  const int4* x8 = reinterpret_cast<const int4*>(x);
  float acc = 0.f;

  #pragma unroll
  for (int it = 0; it < N_IT; ++it) {
    int base = lane + it * STEP;
    int4 vw[UNROLL], vx[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int idx = base + u * 32;
      vw[u] = USE_CS ? s3_ld_cs_int4(w8 + idx) : __ldg(w8 + idx);
      vx[u] = __ldg(x8 + idx);
    }
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw[u]);
      const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&vx[u]);
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        float2 fw = __bfloat1622float2(w2[c]);
        float2 f = __bfloat1622float2(a2[c]);
        acc = fmaf(f.x, fw.x, fmaf(f.y, fw.y, acc));
      }
    }
  }
  #pragma unroll
  for (int off = 16; off >= 1; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off, 32);
  if (lane == 0) out[row] = __float2bfloat16_rn(acc);
}

static inline void launch_zgate_up_gemv(
    torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight, int nh) {
  int m = static_cast<int>(x.size(0));
  int k = static_cast<int>(x.size(1));
  const auto* x_p = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* wg_p = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
  const auto* wu_p = wg_p + static_cast<int64_t>(nh) * k;
  auto* o_p = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (m == 1) {
    s3_preload_gu_gemv_kernel<2, 192, 1, 1><<<(nh + 1) / 2, 384, 0, stream>>>(o_p, x_p, wg_p, wu_p, nh);
  } else {
    if (nh == 6144) s3_preload_gu_gemv_kernel<2, 64, 2, 1><<<(nh + 1) / 2, 128, 0, stream>>>(o_p, x_p, wg_p, wu_p, nh);
    else s3_preload_gu_gemv_kernel<1, 64, 2, 1><<<nh, 64, 0, stream>>>(o_p, x_p, wg_p, wu_p, nh);
  }
}

template <int M_VAL, int ROWS, int TPR, int UNROLL, int USE_CS>
__global__ void s3_lin_k1536_u_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N) {
  constexpr int K = 1536;
  constexpr int K8 = 192;
  const int tid = threadIdx.x;
  const int row_in_cta = tid / TPR;
  const int lane_in_row = tid % TPR;
  const int row = blockIdx.x * ROWS + row_in_cta;
  if (row >= N) return;
  const int4* w8 = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * K);
  const int4* x8_0 = reinterpret_cast<const int4*>(x);
  const int4* x8_1 = (M_VAL > 1) ? reinterpret_cast<const int4*>(x + K) : nullptr;
  float a0 = 0.f, a1 = 0.f;
  #pragma unroll
  for (int i = lane_in_row; i < K8; i += TPR * UNROLL) {
    int4 vw[UNROLL], vx0[UNROLL], vx1[UNROLL];
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int idx = i + u * TPR;
      vw[u] = USE_CS ? s3_ld_cs_int4(w8 + idx) : __ldg(w8 + idx);
      vx0[u] = __ldg(x8_0 + idx);
      if constexpr (M_VAL > 1) vx1[u] = __ldg(x8_1 + idx);
    }
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw[u]);
      const __nv_bfloat162* px0 = reinterpret_cast<const __nv_bfloat162*>(&vx0[u]);
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        float2 fw = __bfloat1622float2(w2[c]);
        float2 f0 = __bfloat1622float2(px0[c]);
        a0 = fmaf(f0.x, fw.x, fmaf(f0.y, fw.y, a0));
      }
      if constexpr (M_VAL > 1) {
        const __nv_bfloat162* px1 = reinterpret_cast<const __nv_bfloat162*>(&vx1[u]);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          float2 fw = __bfloat1622float2(w2[c]);
          float2 f1 = __bfloat1622float2(px1[c]);
          a1 = fmaf(f1.x, fw.x, fmaf(f1.y, fw.y, a1));
        }
      }
    }
  }
  #pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    a0 += __shfl_xor_sync(0xffffffffu, a0, off, 32);
    if constexpr (M_VAL > 1) a1 += __shfl_xor_sync(0xffffffffu, a1, off, 32);
  }
  if constexpr (TPR >= 64) {
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
      for (int w2 = 0; w2 < NW; ++w2) {
        s0 += smem[row_in_cta][w2][0];
        if constexpr (M_VAL > 1) s1 += smem[row_in_cta][w2][1];
      }
      out[row] = __float2bfloat16_rn(s0);
      if constexpr (M_VAL > 1) out[N + row] = __float2bfloat16_rn(s1);
    }
  } else {
    if (lane_in_row == 0) {
      out[row] = __float2bfloat16_rn(a0);
      if constexpr (M_VAL > 1) out[N + row] = __float2bfloat16_rn(a1);
    }
  }
}

static inline void launch_lin_k1536_u(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight, int n) {
  auto* o_p = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const auto* x_p = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* w_p = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  s3_lin_k1536_u_kernel<2, 2, 32, 3, 1><<<(n + 1) / 2, 64, 0, stream>>>(o_p, x_p, w_p, n);
}

static inline void launch_gu_ksplit_mma(
    torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight, int nh) {
  int m = static_cast<int>(x.size(0));
  int k = static_cast<int>(x.size(1));
  const auto* x_p = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* wg_p = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
  const auto* wu_p = wg_p + static_cast<int64_t>(nh) * k;
  auto* o_p = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int KS = 3, TN = 1;
  constexpr size_t smem_bytes = static_cast<size_t>(KS) * 2 * (16 + 2 * TN * 16) * 64 * sizeof(__nv_bfloat16);
  const bool evict = (m >= 8 || (nh == 6144 && m >= 4));
  const bool exact_n = ((nh % (TN * 16)) == 0);
  if (evict) {
    if (exact_n) s18_gu_ksplit_mma_kernel<1536, KS, TN, 1, true><<<nh / 16, KS * 32, smem_bytes, stream>>>(o_p, x_p, wg_p, wu_p, m, nh);
    else s18_gu_ksplit_mma_kernel<1536, KS, TN, 1, false><<<(nh + 15) / 16, KS * 32, smem_bytes, stream>>>(o_p, x_p, wg_p, wu_p, m, nh);
  } else {
    if (exact_n) s18_gu_ksplit_mma_kernel<1536, KS, TN, 0, true><<<nh / 16, KS * 32, smem_bytes, stream>>>(o_p, x_p, wg_p, wu_p, m, nh);
    else s18_gu_ksplit_mma_kernel<1536, KS, TN, 0, false><<<(nh + 15) / 16, KS * 32, smem_bytes, stream>>>(o_p, x_p, wg_p, wu_p, m, nh);
  }
}

static inline void launch_wide_gemv(
    torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight, int m, int n, int k, int mode) {
  const auto* x_p = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* w_p = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
  auto* o_p = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  (void)mode;
  if (m == 1) {
    if (k == 6144) s18_wide_gemv_m1_kernel<2, 6144, 1, 6><<<(n + 1) / 2, 64, 0, stream>>>(o_p, x_p, w_p, n);
    else if (k == 12288) s18_wide_gemv_m1_kernel<2, 12288, 1, 6><<<(n + 1) / 2, 64, 0, stream>>>(o_p, x_p, w_p, n);
  }
}

template <int KS, int TN, int EVICT_W = 0>
static inline void launch_lin_ksplit_tpl(
    torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight, int m, int n, int k) {
  const auto* x_p = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* w_p = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
  auto* o_p = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int B_ROWS_D = TN * 16;
  constexpr size_t smem_bytes = static_cast<size_t>(KS) * 2 * (16 + B_ROWS_D) * 64 * sizeof(__nv_bfloat16);
  auto kfn = s3_lin_ksplit_bk64_kernel<KS, TN, EVICT_W>;
  [[maybe_unused]] static const cudaError_t _attr =
      (smem_bytes > 49152)
          ? cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes))
          : cudaSuccess;
  int grid = (n + B_ROWS_D - 1) / B_ROWS_D;
  kfn<<<grid, KS * 32, smem_bytes, stream>>>(o_p, x_p, w_p, m, n, k);
}

template <int KS, int TN, int EVICT_W = 0>
static inline void launch_lin_ksplit_m32_tpl(
    torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight, int m, int n, int k) {
  const auto* x_p = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* w_p = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
  auto* o_p = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int B_ROWS_D = TN * 16;
  constexpr size_t a_s = static_cast<size_t>(KS) * 2 * 32 * 64 * sizeof(__nv_bfloat16);
  constexpr size_t b_s = static_cast<size_t>(KS) * 2 * B_ROWS_D * 64 * sizeof(__nv_bfloat16);
  constexpr size_t r_s = static_cast<size_t>(KS) * 2 * (TN * 2) * 128 * sizeof(float);
  constexpr size_t smem_bytes = (a_s + b_s > r_s) ? (a_s + b_s) : r_s;
  auto kfn = s3_lin_ksplit_m32_kernel<KS, TN, EVICT_W>;
  [[maybe_unused]] static const cudaError_t _attr =
      (smem_bytes > 49152)
          ? cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes))
          : cudaSuccess;
  int grid = (n + B_ROWS_D - 1) / B_ROWS_D;
  kfn<<<grid, KS * 32, smem_bytes, stream>>>(o_p, x_p, w_p, m, n, k);
}

template <int ROWS, int TPR, int M_VAL>
__global__ void s9_lin_gemv_mr_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N, int K) {
  const int tid = threadIdx.x, r_c = tid / TPR, l_r = tid % TPR;
  const int row = blockIdx.x * ROWS + r_c;
  const bool valid = (row < N);
  const int K8 = K >> 3;
  const int4* w8 = valid ? reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * K) : nullptr;
  const int4* x8 = reinterpret_cast<const int4*>(x);
  float acc[M_VAL];
  #pragma unroll
  for (int m = 0; m < M_VAL; ++m) acc[m] = 0.f;
  if (valid) {
    for (int i = l_r; i < K8; i += TPR) {
      const int4 vw = s3_ld_cs_int4(w8 + i);
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw);
      #pragma unroll
      for (int m = 0; m < M_VAL; ++m) {
        const int4 vx = __ldg(x8 + static_cast<int64_t>(m) * K8 + i);
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&vx);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          float2 fw = __bfloat1622float2(w2[c]);
          float2 fa = __bfloat1622float2(a2[c]);
          acc[m] = fmaf(fa.x, fw.x, fmaf(fa.y, fw.y, acc[m]));
        }
      }
    }
    #pragma unroll
    for (int m = 0; m < M_VAL; ++m) {
      #pragma unroll
      for (int off = 16; off >= 1; off >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], off);
    }
  }
  if constexpr (TPR >= 64) {
    constexpr int NW = TPR >> 5;
    __shared__ float sm[ROWS][NW][M_VAL];
    const int ww = l_r >> 5;
    if (valid && (l_r & 31) == 0) {
      #pragma unroll
      for (int m = 0; m < M_VAL; ++m) sm[r_c][ww][m] = acc[m];
    }
    __syncthreads();
    if (valid && l_r == 0) {
      #pragma unroll
      for (int m = 0; m < M_VAL; ++m) {
        float s = 0.f;
        #pragma unroll
        for (int ww2 = 0; ww2 < NW; ++ww2) s += sm[r_c][ww2][m];
        out[static_cast<int64_t>(m) * N + row] = __float2bfloat16_rn(s);
      }
    }
  } else {
    if (valid && l_r == 0) {
      #pragma unroll
      for (int m = 0; m < M_VAL; ++m) out[static_cast<int64_t>(m) * N + row] = __float2bfloat16_rn(acc[m]);
    }
  }
}

template <int ROWS, int TPR, int K_DIM, int USE_CS>
__global__ void s16_lin_kconst_m1_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N) {
  constexpr int K8 = K_DIM >> 3;
  constexpr int ITERS = K8 / TPR;
  const int tid = threadIdx.x;
  const int row_in_cta = tid / TPR;
  const int lane_in_row = tid % TPR;
  const int row = blockIdx.x * ROWS + row_in_cta;
  if (row >= N) return;
  const int4* w8 = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * K_DIM);
  const int4* x8 = reinterpret_cast<const int4*>(x);
  int4 vw[ITERS], vx[ITERS];
  #pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    int idx = lane_in_row + it * TPR;
    vw[it] = USE_CS ? s3_ld_cs_int4(w8 + idx) : __ldg(w8 + idx);
    vx[it] = __ldg(x8 + idx);
  }
  float acc = 0.f;
  #pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw[it]);
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&vx[it]);
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      float2 fw = __bfloat1622float2(w2[c]);
      float2 fa = __bfloat1622float2(a2[c]);
      acc = fmaf(fa.x, fw.x, fmaf(fa.y, fw.y, acc));
    }
  }
  #pragma unroll
  for (int off = 16; off >= 1; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  constexpr int NW = TPR >> 5;
  __shared__ float smem[ROWS][NW];
  const int ww = lane_in_row >> 5;
  if ((lane_in_row & 31) == 0) smem[row_in_cta][ww] = acc;
  __syncthreads();
  if (lane_in_row == 0) {
    float s = 0.f;
    #pragma unroll
    for (int w2 = 0; w2 < NW; ++w2) s += smem[row_in_cta][w2];
    out[row] = __float2bfloat16_rn(s);
  }
}

static inline void launch_s9_lin_gemv(
    torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight,
    int m, int n, int k, int rows, int tpr) {
  auto* o_p = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const auto* x_p = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* w_p = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (m == 1) {
    if (k == 1536) s16_lin_kconst_m1_kernel<1, 192, 1536, 1><<<n, 192, 0, stream>>>(o_p, x_p, w_p, n);
    else if (k == 2048) s16_lin_kconst_m1_kernel<1, 256, 2048, 1><<<n, 256, 0, stream>>>(o_p, x_p, w_p, n);
    else if (k == 4096) s16_lin_kconst_m1_kernel<1, 512, 4096, 1><<<n, 512, 0, stream>>>(o_p, x_p, w_p, n);
    else {
      if (rows == 2 && tpr == 64) s9_lin_gemv_mr_kernel<2, 64, 1><<<(n + 1) / 2, 128, 0, stream>>>(o_p, x_p, w_p, n, k);
      else if (rows == 3 && tpr == 64) s9_lin_gemv_mr_kernel<3, 64, 1><<<(n + 2) / 3, 192, 0, stream>>>(o_p, x_p, w_p, n, k);
      else if (rows == 2 && tpr == 96) s9_lin_gemv_mr_kernel<2, 96, 1><<<(n + 1) / 2, 192, 0, stream>>>(o_p, x_p, w_p, n, k);
    }
  } else if (m == 2) {
    if (rows == 4 && tpr == 32) s9_lin_gemv_mr_kernel<4, 32, 2><<<(n + 3) / 4, 128, 0, stream>>>(o_p, x_p, w_p, n, k);
    else if (rows == 2 && tpr == 96) s9_lin_gemv_mr_kernel<2, 96, 2><<<(n + 1) / 2, 192, 0, stream>>>(o_p, x_p, w_p, n, k);
  } else if (m == 4 && rows == 2 && tpr == 96) {
    s9_lin_gemv_mr_kernel<2, 96, 4><<<(n + 1) / 2, 192, 0, stream>>>(o_p, x_p, w_p, n, k);
  }
}

__device__ __forceinline__ int4 s3g_ld_cs_int4(const int4* ptr) {
  int4 r;
  asm volatile("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(ptr));
  return r;
}
__device__ __forceinline__ __nv_bfloat16 s3g_softcap_bf16(float v, float cap, float inv_cap) {
  return __float2bfloat16_rn(cap * tanhf(v * inv_cap));
}
__device__ __forceinline__ __nv_bfloat162 s3g_softcap_bf162(float v0, float v1, float cap, float inv_cap) {
  return __floats2bfloat162_rn(cap * tanhf(v0 * inv_cap), cap * tanhf(v1 * inv_cap));
}
__device__ __forceinline__ void s3g_cp_async_cg_zero(void* smem_dst, const void* glob_src, int pred) {
  uint32_t s = smem_u32(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(s), "l"(glob_src), "r"(pred ? 16 : 0) : "memory");
}
__device__ __forceinline__ void s3g_cp_async_ca_zero(void* smem_dst, const void* glob_src, int pred) {
  uint32_t s = smem_u32(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(s), "l"(glob_src), "r"(pred ? 16 : 0) : "memory");
}

template <int ROWS, int TPR, int M_VAL, int USE_CS>
__global__ void s3g_lm_gemv_k1536_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N, float cap, float inv_cap) {
  const int tid = threadIdx.x;
  const int row_in_cta = tid / TPR;
  const int lane_in_row = tid % TPR;
  const int row = blockIdx.x * ROWS + row_in_cta;
  if (row >= N) return;

  const int4* w8 = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * 1536);
  const int4* x8_0 = reinterpret_cast<const int4*>(x);
  const int4* x8_1 = (M_VAL > 1) ? reinterpret_cast<const int4*>(x + 1536) : nullptr;
  float acc0 = 0.f, acc1 = 0.f;

  if constexpr (TPR == 64) {
    #pragma unroll
    for (int s = 0; s < 3; ++s) {
      int idx = lane_in_row + s * 64;
      int4 vw = USE_CS ? s3g_ld_cs_int4(w8 + idx) : __ldg(w8 + idx);
      int4 vx0 = __ldg(x8_0 + idx);
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw);
      const __nv_bfloat162* a0 = reinterpret_cast<const __nv_bfloat162*>(&vx0);
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        float2 fw = __bfloat1622float2(w2[c]);
        float2 f0 = __bfloat1622float2(a0[c]);
        acc0 = fmaf(f0.x, fw.x, fmaf(f0.y, fw.y, acc0));
      }
      if constexpr (M_VAL > 1) {
        int4 vx1 = __ldg(x8_1 + idx);
        const __nv_bfloat162* a1 = reinterpret_cast<const __nv_bfloat162*>(&vx1);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          float2 fw = __bfloat1622float2(w2[c]);
          float2 f1 = __bfloat1622float2(a1[c]);
          acc1 = fmaf(f1.x, fw.x, fmaf(f1.y, fw.y, acc1));
        }
      }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      acc0 += __shfl_xor_sync(0xffffffffu, acc0, off);
      if constexpr (M_VAL > 1) acc1 += __shfl_xor_sync(0xffffffffu, acc1, off);
    }
    __shared__ float smem[ROWS][2][M_VAL];
    const int ww = lane_in_row >> 5;
    if ((lane_in_row & 31) == 0) {
      smem[row_in_cta][ww][0] = acc0;
      if constexpr (M_VAL > 1) smem[row_in_cta][ww][1] = acc1;
    }
    __syncthreads();
    if (lane_in_row == 0) {
      out[row] = s3g_softcap_bf16(smem[row_in_cta][0][0] + smem[row_in_cta][1][0], cap, inv_cap);
      if constexpr (M_VAL > 1)
        out[N + row] = s3g_softcap_bf16(smem[row_in_cta][0][1] + smem[row_in_cta][1][1], cap, inv_cap);
    }
  } else {
    #pragma unroll
    for (int s = 0; s < 6; ++s) {
      int idx = lane_in_row + s * 32;
      int4 vw = USE_CS ? s3g_ld_cs_int4(w8 + idx) : __ldg(w8 + idx);
      int4 vx0 = __ldg(x8_0 + idx);
      const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw);
      const __nv_bfloat162* a0 = reinterpret_cast<const __nv_bfloat162*>(&vx0);
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        float2 fw = __bfloat1622float2(w2[c]);
        float2 f0 = __bfloat1622float2(a0[c]);
        acc0 = fmaf(f0.x, fw.x, fmaf(f0.y, fw.y, acc0));
      }
      if constexpr (M_VAL > 1) {
        int4 vx1 = __ldg(x8_1 + idx);
        const __nv_bfloat162* a1 = reinterpret_cast<const __nv_bfloat162*>(&vx1);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          float2 fw = __bfloat1622float2(w2[c]);
          float2 f1 = __bfloat1622float2(a1[c]);
          acc1 = fmaf(f1.x, fw.x, fmaf(f1.y, fw.y, acc1));
        }
      }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
      acc0 += __shfl_xor_sync(0xffffffffu, acc0, off);
      if constexpr (M_VAL > 1) acc1 += __shfl_xor_sync(0xffffffffu, acc1, off);
    }
    if (lane_in_row == 0) {
      out[row] = s3g_softcap_bf16(acc0, cap, inv_cap);
      if constexpr (M_VAL > 1) out[N + row] = s3g_softcap_bf16(acc1, cap, inv_cap);
    }
  }
}

template <int WARPS, int RPT, int M_VAL, int USE_CS>
__global__ void s3g_lm_gemv_2r_k1536_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N, float cap, float inv_cap) {
  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int rb = blockIdx.x * (WARPS * RPT) + wid * RPT;
  if (rb >= N) return;
  const int4* x8_0 = reinterpret_cast<const int4*>(x);
  const int4* x8_1 = (M_VAL > 1) ? reinterpret_cast<const int4*>(x + 1536) : nullptr;
  int4 vx0[6], vx1[6];
  #pragma unroll
  for (int s = 0; s < 6; ++s) {
    int idx = lane + s * 32;
    vx0[s] = __ldg(x8_0 + idx);
    if constexpr (M_VAL > 1) vx1[s] = __ldg(x8_1 + idx);
  }
  int4 vw[RPT][6];
  float acc0[RPT], acc1[RPT];
  #pragma unroll
  for (int r = 0; r < RPT; ++r) {
    int row = rb + r;
    if (row < N) {
      const int4* w8 = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * 1536);
      #pragma unroll
      for (int s = 0; s < 6; ++s) {
        vw[r][s] = USE_CS ? s3g_ld_cs_int4(w8 + lane + s * 32) : __ldg(w8 + lane + s * 32);
      }
    }
    acc0[r] = 0.f;
    acc1[r] = 0.f;
  }
  #pragma unroll
  for (int r = 0; r < RPT; ++r) {
    int row = rb + r;
    if (row < N) {
      #pragma unroll
      for (int s = 0; s < 6; ++s) {
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw[r][s]);
        const __nv_bfloat162* a0 = reinterpret_cast<const __nv_bfloat162*>(&vx0[s]);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          float2 fw = __bfloat1622float2(w2[c]);
          float2 f0 = __bfloat1622float2(a0[c]);
          acc0[r] = fmaf(f0.x, fw.x, fmaf(f0.y, fw.y, acc0[r]));
        }
        if constexpr (M_VAL > 1) {
          const __nv_bfloat162* a1 = reinterpret_cast<const __nv_bfloat162*>(&vx1[s]);
          #pragma unroll
          for (int c = 0; c < 4; ++c) {
            float2 fw = __bfloat1622float2(w2[c]);
            float2 f1 = __bfloat1622float2(a1[c]);
            acc1[r] = fmaf(f1.x, fw.x, fmaf(f1.y, fw.y, acc1[r]));
          }
        }
      }
      #pragma unroll
      for (int off = 16; off >= 1; off >>= 1) {
        acc0[r] += __shfl_xor_sync(0xffffffffu, acc0[r], off);
        if constexpr (M_VAL > 1) acc1[r] += __shfl_xor_sync(0xffffffffu, acc1[r], off);
      }
      if (lane == 0) {
        out[row] = s3g_softcap_bf16(acc0[r], cap, inv_cap);
        if constexpr (M_VAL > 1) out[N + row] = s3g_softcap_bf16(acc1[r], cap, inv_cap);
      }
    }
  }
}

template <int M_TILES, int KS, int TN, int EVICT_W, bool ZFILL, bool UNCOND_W>
__global__ void s18_lm_mma_softcap_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int M, int N, float cap, float inv_cap) {
  constexpr int K = 1536, BK = 64;
  constexpr int M_ROWS = M_TILES * 16;
  constexpr int B_ROWS = TN * 8;
  constexpr int A_ELEMS = M_ROWS * BK;
  constexpr int B_ELEMS = B_ROWS * BK;
  constexpr int A_CH = (M_ROWS * 8) / 32;
  constexpr int B_CH = (B_ROWS * 8) / 32;
  constexpr int NUM_THREADS = KS * 32;
  constexpr int total_chunks = K / BK;
  constexpr int chunks_per_warp = total_chunks / KS;
  static_assert((total_chunks % KS) == 0 && (chunks_per_warp % 2) == 0, "requires even chunks/warp");

  const int tid = threadIdx.x, ks = tid >> 5, lane = tid & 31;
  const int n_base = blockIdx.x * B_ROWS;
  if (n_base >= N) return;

  const uint64_t pol = EVICT_W ? s3_l2_evict_first_policy() : 0ull;
  extern __shared__ char dyn_smem[];
  __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(
      dyn_smem + static_cast<size_t>(KS) * 2 * A_ELEMS * sizeof(__nv_bfloat16));
  __nv_bfloat16* my_sa0 = smem_a + (ks * 2) * A_ELEMS;
  __nv_bfloat16* my_sb0 = smem_b + (ks * 2) * B_ELEMS;

  const int c_start = ks * chunks_per_warp;
  const int r0 = lane >> 3, c8 = (lane & 7) << 3;
  const __nv_bfloat16* x_cur = x + static_cast<int64_t>(r0) * K + c_start * BK + c8;
  const __nv_bfloat16* w_cur = w + static_cast<int64_t>(n_base + r0) * K + c_start * BK + c8;

  uint32_t x_mask = 0;
  #pragma unroll
  for (int i = 0; i < A_CH; ++i) if ((r0 + i * 4) < M) x_mask |= (1u << i);
  uint32_t w_mask = 0;
  if constexpr (!UNCOND_W) {
    #pragma unroll
    for (int i = 0; i < B_CH; ++i) if ((n_base + r0 + i * 4) < N) w_mask |= (1u << i);
  }

  float acc[M_TILES][TN][4];
  #pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt)
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn)
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc[mt][tn][i] = 0.f;

  auto issue_stage = [&](int st) {
    __nv_bfloat16* sa = my_sa0 + st * A_ELEMS;
    __nv_bfloat16* sb = my_sb0 + st * B_ELEMS;
    #pragma unroll
    for (int i = 0; i < A_CH; ++i) {
      int r = r0 + i * 4, valid = ((x_mask >> i) & 1u) ? 1 : 0;
      const __nv_bfloat16* src = valid ? (x_cur + static_cast<int64_t>(i * 4) * K) : x;
      if constexpr (ZFILL) s18_cp_async_ca_zero(&sa[r * BK + s3_swz64_col(r, c8)], src, valid);
      else s18_cp_async_ca_zero_old(&sa[r * BK + s3_swz64_col(r, c8)], src, valid);
    }
    x_cur += BK;
    #pragma unroll
    for (int i = 0; i < B_CH; ++i) {
      int r = r0 + i * 4;
      if constexpr (UNCOND_W) {
        s18_cp_async_cg_uncond<EVICT_W>(&sb[r * BK + s3_swz64_col(r, c8)], w_cur + static_cast<int64_t>(i * 4) * K, pol);
      } else {
        int valid = ((w_mask >> i) & 1u) ? 1 : 0;
        const __nv_bfloat16* src = valid ? (w_cur + static_cast<int64_t>(i * 4) * K) : w;
        if constexpr (ZFILL) s18_cp_async_cg_w<EVICT_W>(&sb[r * BK + s3_swz64_col(r, c8)], src, valid, pol);
        else s18_cp_async_cg_w_old<EVICT_W>(&sb[r * BK + s3_swz64_col(r, c8)], src, valid, pol);
      }
    }
    w_cur += BK;
    cp_async_commit();
  };

  const int a_r = lane & 15, a_c = ((lane >> 4) & 1) << 3;
  const int b_r = lane & 7, b_c = ((lane >> 3) & 1) << 3;
  auto consume_stage = [&](int st) {
    __nv_bfloat16* sa = my_sa0 + st * A_ELEMS;
    __nv_bfloat16* sb = my_sb0 + st * B_ELEMS;
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t ra[M_TILES][4];
      #pragma unroll
      for (int mt = 0; mt < M_TILES; ++mt) {
        int rs = mt * 16 + a_r;
        ldmatrix_x4(ra[mt], &sa[rs * BK + s3_swz64_col(rs, kk + a_c)]);
      }
      #pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        uint32_t rb[2];
        int rs_b = tn * 8 + b_r;
        ldmatrix_x2(rb, &sb[rs_b * BK + s3_swz64_col(rs_b, kk + b_c)]);
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) mma_m16n8k16_bf16(acc[mt][tn], ra[mt], rb);
      }
    }
  };

  issue_stage(0);
  #pragma unroll 1
  for (int pair = 0; pair < chunks_per_warp / 2; ++pair) {
    issue_stage(1);
    cp_async_wait<1>();
    __syncwarp();
    consume_stage(0);
    __syncwarp();
    if (pair + 1 < chunks_per_warp / 2) {
      issue_stage(0);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncwarp();
    consume_stage(1);
    __syncwarp();
  }

  __syncthreads();
  float* smem_red = reinterpret_cast<float*>(dyn_smem);
  #pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt)
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      int base = (((ks * M_TILES + mt) * TN + tn) * 32 + lane) * 4;
      #pragma unroll
      for (int i = 0; i < 4; ++i) smem_red[base + i] = acc[mt][tn][i];
    }
  __syncthreads();

  for (int e = tid; e < M_TILES * TN * 32; e += NUM_THREADS) {
    int t = e >> 5, l = e & 31;
    int mt = t / TN, tn = t - mt * TN;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    #pragma unroll
    for (int k = 0; k < KS; ++k) {
      int base = (((k * M_TILES + mt) * TN + tn) * 32 + l) * 4;
      s0 += smem_red[base + 0]; s1 += smem_red[base + 1];
      s2 += smem_red[base + 2]; s3 += smem_red[base + 3];
    }
    int r0_out = mt * 16 + (l >> 2), r1_out = r0_out + 8;
    int col = n_base + tn * 8 + (l & 3) * 2;
    if (UNCOND_W || col + 1 < N) {
      if (r0_out < M) {
        *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r0_out) * N + col]) =
            s3g_softcap_bf162(s0, s1, cap, inv_cap);
      }
      if (r1_out < M) {
        *reinterpret_cast<__nv_bfloat162*>(&out[static_cast<int64_t>(r1_out) * N + col]) =
            s3g_softcap_bf162(s2, s3, cap, inv_cap);
      }
    }
  }
}

template <int WARPS_N, int TN_PER_WARP, int STAGES, int M_TILES, int EVICT_W, bool EXACT_N>
__global__ void s18_gu_shareA_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wg,
    const __nv_bfloat16* __restrict__ wu,
    int M, int NH) {
  constexpr int K = 1536, BK = 64, M_ROWS = M_TILES * 16;
  constexpr int C_ROWS = WARPS_N * TN_PER_WARP * 8;
  constexpr int B_ROWS = WARPS_N * TN_PER_WARP * 16;
  constexpr int NUM_THREADS = WARPS_N * 32;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int c_base = blockIdx.x * C_ROWS;
  if (c_base >= NH) return;
  const uint64_t pol = EVICT_W ? s3_l2_evict_first_policy() : 0ull;

  extern __shared__ char dyn_smem[];
  __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(dyn_smem);
  __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(
      dyn_smem + static_cast<size_t>(STAGES) * M_ROWS * BK * sizeof(__nv_bfloat16));

  constexpr int chunks = K / BK;
  float ag[M_TILES][TN_PER_WARP][4], au[M_TILES][TN_PER_WARP][4];
  #pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt)
    #pragma unroll
    for (int t = 0; t < TN_PER_WARP; ++t)
      #pragma unroll
      for (int i = 0; i < 4; ++i) ag[mt][t][i] = au[mt][t][i] = 0.f;

  auto issue_stage = [&](int stage, int k0) {
    __nv_bfloat16* sa = smem_a + stage * (M_ROWS * BK);
    __nv_bfloat16* sb = smem_b + stage * (B_ROWS * BK);
    #pragma unroll
    for (int e = tid; e < M_ROWS * 8; e += NUM_THREADS) {
      int r = e >> 3, c8 = (e & 7) << 3, valid = (r < M) ? 1 : 0;
      s18_cp_async_ca_zero(&sa[r * BK + s3_swz64_col(r, c8)],
                           valid ? &x[static_cast<int64_t>(r) * K + k0 + c8] : x, valid);
    }
    #pragma unroll
    for (int e = tid; e < C_ROWS * 8; e += NUM_THREADS) {
      int rc = e >> 3, c8 = (e & 7) << 3, slot = rc >> 3, sub = rc & 7, gn = c_base + rc;
      int bg_r = slot * 16 + sub, bu_r = bg_r + 8;
      if constexpr (EXACT_N) {
        s18_cp_async_cg_uncond<EVICT_W>(&sb[bg_r * BK + s3_swz64_col(bg_r, c8)], &wg[static_cast<int64_t>(gn) * K + k0 + c8], pol);
        s18_cp_async_cg_uncond<EVICT_W>(&sb[bu_r * BK + s3_swz64_col(bu_r, c8)], &wu[static_cast<int64_t>(gn) * K + k0 + c8], pol);
      } else {
        int valid = (gn < NH) ? 1 : 0;
        s18_cp_async_cg_w<EVICT_W>(&sb[bg_r * BK + s3_swz64_col(bg_r, c8)],
                                   valid ? &wg[static_cast<int64_t>(gn) * K + k0 + c8] : wg, valid, pol);
        s18_cp_async_cg_w<EVICT_W>(&sb[bu_r * BK + s3_swz64_col(bu_r, c8)],
                                   valid ? &wu[static_cast<int64_t>(gn) * K + k0 + c8] : wu, valid, pol);
      }
    }
    cp_async_commit();
  };

  for (int s = 0; s < STAGES - 1 && s < chunks; ++s) issue_stage(s, s * BK);
  const int my_slot_base = wid * TN_PER_WARP;
  const int a_row_lane = lane & 15, a_col_lane = ((lane >> 4) & 1) << 3;
  const int b_off_row = (((lane >> 4) & 1) << 3) + (lane & 7);
  const int b_off_col = ((lane >> 3) & 1) << 3;

  for (int c = 0; c < chunks; ++c) {
    int st = c % STAGES, next_c = c + (STAGES - 1);
    if (next_c < chunks) {
      issue_stage(next_c % STAGES, next_c * BK);
      cp_async_wait<STAGES - 1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    __nv_bfloat16* sa = smem_a + st * (M_ROWS * BK);
    __nv_bfloat16* sb = smem_b + st * (B_ROWS * BK);
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t ra[M_TILES][4];
      #pragma unroll
      for (int mt = 0; mt < M_TILES; ++mt) {
        int rs = mt * 16 + a_row_lane;
        ldmatrix_x4(ra[mt], &sa[rs * BK + s3_swz64_col(rs, kk + a_col_lane)]);
      }
      #pragma unroll
      for (int tn = 0; tn < TN_PER_WARP; ++tn) {
        uint32_t rb[4];
        int r_s = (my_slot_base + tn) * 16 + b_off_row;
        ldmatrix_x4(rb, &sb[r_s * BK + s3_swz64_col(r_s, kk + b_off_col)]);
        uint32_t bg[2] = {rb[0], rb[1]}, bu[2] = {rb[2], rb[3]};
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
          mma_m16n8k16_bf16(ag[mt][tn], ra[mt], bg);
          mma_m16n8k16_bf16(au[mt][tn], ra[mt], bu);
        }
      }
    }
    __syncthreads();
  }

  const int r0_base = lane >> 2, c0 = (lane & 3) * 2;
  #pragma unroll
  for (int mt = 0; mt < M_TILES; ++mt) {
    int r0 = mt * 16 + r0_base, r1 = r0 + 8;
    #pragma unroll
    for (int tn = 0; tn < TN_PER_WARP; ++tn) {
      int n_col = c_base + (my_slot_base + tn) * 8 + c0;
      if (EXACT_N || n_col + 1 < NH) {
        if (r0 < M) {
          out[static_cast<int64_t>(r0) * NH + n_col] = exact_vllm_gelu_tanh_mul(ag[mt][tn][0], au[mt][tn][0]);
          out[static_cast<int64_t>(r0) * NH + n_col + 1] = exact_vllm_gelu_tanh_mul(ag[mt][tn][1], au[mt][tn][1]);
        }
        if (r1 < M) {
          out[static_cast<int64_t>(r1) * NH + n_col] = exact_vllm_gelu_tanh_mul(ag[mt][tn][2], au[mt][tn][2]);
          out[static_cast<int64_t>(r1) * NH + n_col + 1] = exact_vllm_gelu_tanh_mul(ag[mt][tn][3], au[mt][tn][3]);
        }
      }
    }
  }
}

static inline void launch_gu_shareA(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w, int nh, int mode) {
  int M = static_cast<int>(x.size(0));
  int K = static_cast<int>(x.size(1));
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* wg = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>());
  const auto* wu = wg + static_cast<int64_t>(nh) * K;
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  (void)mode;
  constexpr int WN = 4, TN = 2, CR = WN * TN * 8, STG = 3, MT = 2;
  constexpr size_t smem = static_cast<size_t>(STG) * (MT * 16 + 2 * CR) * 64 * sizeof(__nv_bfloat16);
  if ((nh % CR) == 0) {
    auto kfn = s18_gu_shareA_kernel<WN, TN, STG, MT, 1, true>;
    s3_set_max_smem(kfn, smem);
    kfn<<<nh / CR, WN * 32, smem, st>>>(op, xp, wg, wu, M, nh);
  } else {
    auto kfn = s18_gu_shareA_kernel<WN, TN, STG, MT, 1, false>;
    s3_set_max_smem(kfn, smem);
    kfn<<<(nh + CR - 1) / CR, WN * 32, smem, st>>>(op, xp, wg, wu, M, nh);
  }
}

template <int WARPS, int RPT, int M_VAL>
__global__ void s9_lm_gemv_mr_reg_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int N, float cap, float inv_cap) {
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rb = blockIdx.x * (WARPS * RPT) + wid * RPT;
  if (rb >= N) return;
  float acc[RPT][M_VAL];
  #pragma unroll
  for (int r = 0; r < RPT; ++r) {
    #pragma unroll
    for (int m = 0; m < M_VAL; ++m) acc[r][m] = 0.f;
  }
  #pragma unroll
  for (int s = 0; s < 6; ++s) {
    int4 vx[M_VAL];
    #pragma unroll
    for (int m = 0; m < M_VAL; ++m) {
      const int4* xm = reinterpret_cast<const int4*>(x + static_cast<int64_t>(m) * 1536);
      vx[m] = __ldg(xm + lane + s * 32);
    }
    #pragma unroll
    for (int r = 0; r < RPT; ++r) {
      int row = rb + r;
      if (row < N) {
        const int4* w8 = reinterpret_cast<const int4*>(w + static_cast<int64_t>(row) * 1536);
        int4 vw = s3g_ld_cs_int4(w8 + lane + s * 32);
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&vw);
        #pragma unroll
        for (int m = 0; m < M_VAL; ++m) {
          const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&vx[m]);
          #pragma unroll
          for (int c = 0; c < 4; ++c) {
            float2 fw = __bfloat1622float2(w2[c]);
            float2 fa = __bfloat1622float2(a2[c]);
            acc[r][m] = fmaf(fa.x, fw.x, fmaf(fa.y, fw.y, acc[r][m]));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < RPT; ++r) {
    int row = rb + r;
    if (row < N) {
      #pragma unroll
      for (int m = 0; m < M_VAL; ++m) {
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) acc[r][m] += __shfl_xor_sync(0xffffffffu, acc[r][m], off);
        if (lane == 0) out[static_cast<int64_t>(m) * N + row] = s3g_softcap_bf16(acc[r][m], cap, inv_cap);
      }
    }
  }
}

static inline void launch_lm_head_softcap(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w, double cap) {
  int M = static_cast<int>(x.size(0));
  int N = static_cast<int>(w.size(0));
  int K = static_cast<int>(w.size(1));
  float fcap = static_cast<float>(cap), finv = 1.0f / fcap;
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    s3g_lm_gemv_k1536_kernel<2, 64, 1, 0><<<(N + 1) / 2, 128, 0, st>>>(op, xp, wp, N, fcap, finv);
  } else if (M == 2) {
    s3g_lm_gemv_2r_k1536_kernel<4, 2, 2, 1><<<(N + 7) / 8, 128, 0, st>>>(op, xp, wp, N, fcap, finv);
  } else if (M == 3) {
    s9_lm_gemv_mr_reg_kernel<4, 2, 3><<<(N + 7) / 8, 128, 0, st>>>(op, xp, wp, N, fcap, finv);
  } else if (M == 4) {
    s9_lm_gemv_mr_reg_kernel<4, 1, 4><<<(N + 3) / 4, 128, 0, st>>>(op, xp, wp, N, fcap, finv);
  } else if (M <= 16) {
    constexpr int MT = 1, KS = 3, TN = 5, BR = TN * 8;
    constexpr size_t ab_s = static_cast<size_t>(KS) * 2 * (MT * 16 + BR) * 64 * sizeof(__nv_bfloat16);
    constexpr size_t red_s = static_cast<size_t>(KS) * MT * TN * 128 * sizeof(float);
    constexpr size_t smem = (ab_s > red_s) ? ab_s : red_s;
    auto kfn = s18_lm_mma_softcap_kernel<MT, KS, TN, 1, false, false>;
    s3_set_max_smem(kfn, smem);
    kfn<<<(N + BR - 1) / BR, KS * 32, smem, st>>>(op, xp, wp, M, N, fcap, finv);
  } else if (M <= 24) {
    constexpr int MT = 2, KS = 3, TN = 4, BR = TN * 8;
    constexpr size_t ab_s = static_cast<size_t>(KS) * 2 * (MT * 16 + BR) * 64 * sizeof(__nv_bfloat16);
    constexpr size_t red_s = static_cast<size_t>(KS) * MT * TN * 128 * sizeof(float);
    constexpr size_t smem = (ab_s > red_s) ? ab_s : red_s;
    if ((N % BR) == 0) {
      auto kfn = s18_lm_mma_softcap_kernel<MT, KS, TN, 1, false, true>;
      s3_set_max_smem(kfn, smem);
      kfn<<<N / BR, KS * 32, smem, st>>>(op, xp, wp, M, N, fcap, finv);
    } else {
      auto kfn = s18_lm_mma_softcap_kernel<MT, KS, TN, 1, false, false>;
      s3_set_max_smem(kfn, smem);
      kfn<<<(N + BR - 1) / BR, KS * 32, smem, st>>>(op, xp, wp, M, N, fcap, finv);
    }
  } else {
    constexpr int MT = 2, KS = 3, TN = 6, BR = TN * 8;
    constexpr size_t ab_s = static_cast<size_t>(KS) * 2 * (MT * 16 + BR) * 64 * sizeof(__nv_bfloat16);
    constexpr size_t red_s = static_cast<size_t>(KS) * MT * TN * 128 * sizeof(float);
    constexpr size_t smem = (ab_s > red_s) ? ab_s : red_s;
    auto kfn = s18_lm_mma_softcap_kernel<MT, KS, TN, 1, true, false>;
    s3_set_max_smem(kfn, smem);
    kfn<<<(N + BR - 1) / BR, KS * 32, smem, st>>>(op, xp, wp, M, N, fcap, finv);
  }
}

}  // namespace e2b_s3

void e2b_s3_launch_zgate_up_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int nh) {
  e2b_s3::launch_zgate_up_gemv(out, x, weight, nh);
}

void e2b_s3_launch_gu_ksplit_mma(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int nh) {
  e2b_s3::launch_gu_ksplit_mma(out, x, weight, nh);
}

void e2b_s3_launch_lin_ksplit(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int m, int n, int k, int ks, int tn, int evict) {
  if (evict) {
    if (ks == 12 && tn == 1) e2b_s3::launch_lin_ksplit_tpl<12, 1, 1>(out, x, weight, m, n, k);
    else if (ks == 16 && tn == 1) e2b_s3::launch_lin_ksplit_tpl<16, 1, 1>(out, x, weight, m, n, k);
    else e2b_s3_launch_lin_ksplit(out, x, weight, m, n, k, ks, tn, 0);
    return;
  }
  if (ks == 4 && tn == 1) e2b_s3::launch_lin_ksplit_tpl<4, 1, 0>(out, x, weight, m, n, k);
  else if (ks == 6 && tn == 1) e2b_s3::launch_lin_ksplit_tpl<6, 1, 0>(out, x, weight, m, n, k);
  else if (ks == 6 && tn == 2) e2b_s3::launch_lin_ksplit_tpl<6, 2, 0>(out, x, weight, m, n, k);
  else if (ks == 8 && tn == 1) e2b_s3::launch_lin_ksplit_tpl<8, 1, 0>(out, x, weight, m, n, k);
  else if (ks == 12 && tn == 1) e2b_s3::launch_lin_ksplit_tpl<12, 1, 0>(out, x, weight, m, n, k);
  else if (ks == 16 && tn == 1) e2b_s3::launch_lin_ksplit_tpl<16, 1, 0>(out, x, weight, m, n, k);
}

void e2b_s3_launch_lin_ksplit_m32(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int m, int n, int k, int ks, int tn, int evict) {
  if (evict) {
    if (ks == 8 && tn == 1) e2b_s3::launch_lin_ksplit_m32_tpl<8, 1, 1>(out, x, weight, m, n, k);
    else if (ks == 12 && tn == 1) e2b_s3::launch_lin_ksplit_m32_tpl<12, 1, 1>(out, x, weight, m, n, k);
    else e2b_s3_launch_lin_ksplit_m32(out, x, weight, m, n, k, ks, tn, 0);
    return;
  }
  if (ks == 3 && tn == 1) e2b_s3::launch_lin_ksplit_m32_tpl<3, 1, 0>(out, x, weight, m, n, k);
  else if (ks == 4 && tn == 1) e2b_s3::launch_lin_ksplit_m32_tpl<4, 1, 0>(out, x, weight, m, n, k);
  else if (ks == 6 && tn == 2) e2b_s3::launch_lin_ksplit_m32_tpl<6, 2, 0>(out, x, weight, m, n, k);
  else if (ks == 8 && tn == 1) e2b_s3::launch_lin_ksplit_m32_tpl<8, 1, 0>(out, x, weight, m, n, k);
  else if (ks == 12 && tn == 1) e2b_s3::launch_lin_ksplit_m32_tpl<12, 1, 0>(out, x, weight, m, n, k);
}

void e2b_s3_launch_lin_k1536_u(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int n) {
  e2b_s3::launch_lin_k1536_u(out, x, weight, n);
}

void e2b_s3_launch_gu_shareA(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int nh, int mode) {
  e2b_s3::launch_gu_shareA(out, x, weight, nh, mode);
}

void e2b_s3_launch_lm_head_softcap(torch::Tensor out, torch::Tensor x, torch::Tensor weight, double cap) {
  e2b_s3::launch_lm_head_softcap(out, x, weight, cap);
}

void e2b_s3_launch_wide_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int m, int n, int k, int mode) {
  e2b_s3::launch_wide_gemv(out, x, weight, m, n, k, mode);
}

void e2b_s9_launch_lin_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int m, int n, int k, int rows, int tpr) {
  e2b_s3::launch_s9_lin_gemv(out, x, weight, m, n, k, rows, tpr);
}
"""

_CPP_SRC = r"""
#include <torch/library.h>
#include <ATen/ATen.h>
#include <cstdint>

void e2b_launch_q_norm_rope(
    torch::Tensor out, torch::Tensor q, torch::Tensor pos, torch::Tensor q_weight,
    torch::Tensor cos_sin_cache, int head_dim, int rope_angles, double eps);
void e2b_launch_gelu_tanh_and_mul_bf16x8(torch::Tensor out, torch::Tensor input);
bool e2b_launch_dual_gemm_gelu(torch::Tensor out, torch::Tensor x, torch::Tensor w, int variant);
void e2b_s3_launch_zgate_up_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int nh);
void e2b_s3_launch_gu_ksplit_mma(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int nh);
void e2b_s3_launch_lin_ksplit(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int m, int n, int k, int ks, int tn, int evict = 0);
void e2b_s3_launch_lin_ksplit_m32(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int m, int n, int k, int ks, int tn, int evict = 0);
void e2b_s3_launch_lin_k1536_u(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int n);
void e2b_s3_launch_wide_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int m, int n, int k, int mode);
void e2b_s9_launch_lin_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int m, int n, int k, int rows, int tpr);
void e2b_s3_launch_gu_shareA(torch::Tensor out, torch::Tensor x, torch::Tensor weight, int nh, int mode);
void e2b_s3_launch_lm_head_softcap(torch::Tensor out, torch::Tensor x, torch::Tensor weight, double cap);

torch::Tensor gemma4_q_norm_rope_cuda(
    const torch::Tensor& q,
    const torch::Tensor& positions,
    const torch::Tensor& q_weight,
    const torch::Tensor& cos_sin_cache,
    double eps,
    int64_t head_dim,
    int64_t rope_angles) {
  auto out = torch::empty(q.sizes(), q.options());
  if (q.numel() == 0) return out;
  auto pos_flat = positions.reshape({-1});
  if (pos_flat.scalar_type() != at::kLong) {
    pos_flat = pos_flat.to(at::kLong);
  }
  TORCH_CHECK(
      q.dim() == 2 && q.is_cuda() && q.scalar_type() == at::kBFloat16 &&
      q.stride(1) == 1 && (q.stride(0) % 8 == 0) &&
      (reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0) &&
      head_dim > 0 && (q.size(1) % head_dim == 0) &&
      pos_flat.size(0) == q.size(0) &&
      q_weight.is_cuda() && q_weight.scalar_type() == at::kBFloat16 &&
      q_weight.is_contiguous() && q_weight.numel() == head_dim &&
      cos_sin_cache.is_cuda() && cos_sin_cache.scalar_type() == at::kBFloat16 &&
      cos_sin_cache.stride(1) == 1 && (cos_sin_cache.stride(0) % 8 == 0) &&
      (cos_sin_cache.size(1) % 16 == 0) &&
      0 <= rope_angles && rope_angles <= cos_sin_cache.size(1) / 2 &&
      rope_angles <= head_dim / 2 && (rope_angles % 32 == 0),
      "Invalid arguments for gemma4_q_norm_rope");
  e2b_launch_q_norm_rope(
      out, q, pos_flat, q_weight, cos_sin_cache,
      static_cast<int>(head_dim), static_cast<int>(rope_angles), eps);
  return out;
}

torch::Tensor e2b_mlp_gate_up_gelu_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight) {
  int64_t m = x.size(0);
  int64_t k = x.size(1);
  int64_t n_half = weight.size(0) / 2;
  if (m == 0) {
    return x.new_empty({0, n_half});
  }
  if (
      k == 1536 && x.is_cuda() && weight.is_cuda() &&
      x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
      x.is_contiguous() && weight.is_contiguous() &&
      (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0)
  ) {
    auto out = x.new_empty({m, n_half});
    if ((n_half == 6144 || n_half == 12288)) {
      if (m <= 2) {
        e2b_s3_launch_zgate_up_gemv(out, x, weight, static_cast<int>(n_half));
        return out;
      }
      if (m <= 16) {
        e2b_s3_launch_gu_ksplit_mma(out, x, weight, static_cast<int>(n_half));
        return out;
      }
      if (m <= ((n_half == 12288) ? 32 : 30)) {
        e2b_s3_launch_gu_shareA(out, x, weight, static_cast<int>(n_half), 0);
        return out;
      }
    }
    if (n_half == 6144 && m >= 31) {
      int vid = (m <= 64) ? 2 : ((m <= 512) ? 1 : 0);
      if (e2b_launch_dual_gemm_gelu(out, x, weight, vid)) return out;
      if (m <= 32) {
        e2b_s3_launch_gu_shareA(out, x, weight, static_cast<int>(n_half), 0);
        return out;
      }
    }
    if (n_half == 12288 && m >= 33) {
      int vid = (m <= 64 || (m >= 129 && m <= 192)) ? 2 : ((m > 192 && m <= 768) ? 1 : 0);
      if (e2b_launch_dual_gemm_gelu(out, x, weight, vid)) return out;
    }
    auto gu = at::linear(x, weight);
    e2b_launch_gelu_tanh_and_mul_bf16x8(out, gu);
    return out;
  }
  auto gu = at::linear(x, weight);
  auto chunks = gu.chunk(2, -1);
  return at::gelu(chunks[0], "tanh") * chunks[1];
}

torch::Tensor e2b_skinny_linear_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight) {
  int64_t m = x.size(0);
  int64_t n = weight.size(0);
  if (m == 0) {
    return x.new_empty({0, n});
  }
  if (m > 32) {
    return at::linear(x, weight);
  }
  int64_t k = x.size(1);
  if (
      x.is_cuda() && weight.is_cuda() &&
      x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
      x.is_contiguous() && weight.is_contiguous() &&
      (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0)
  ) {
    if (k == 1536) {
      if (n == 2048 || n == 2560 || n == 4096 || n == 5120) {
        if (m == 2) {
          auto out = x.new_empty({2, n});
          if (n == 4096) {
            e2b_s9_launch_lin_gemv(out, x, weight, 2, 4096, 1536, 4, 32);
          } else {
            e2b_s3_launch_lin_k1536_u(out, x, weight, static_cast<int>(n));
          }
          return out;
        }
        auto out = x.new_empty({m, n});
        if (m > 16) {
          if (n == 2048 || n == 2560) {
            e2b_s3_launch_lin_ksplit_m32(out, x, weight, static_cast<int>(m), static_cast<int>(n), 1536, 6, 2);
          } else {
            int ks = (m > 24) ? 3 : 4;
            e2b_s3_launch_lin_ksplit_m32(out, x, weight, static_cast<int>(m), static_cast<int>(n), 1536, ks, 1);
          }
          return out;
        }
        if (m == 1) {
          int rows = (n == 2560 || n == 5120) ? 3 : 2;
          e2b_s9_launch_lin_gemv(out, x, weight, 1, static_cast<int>(n), 1536, rows, 64);
        } else if (n == 2048 || n == 2560) {
          if (m <= 12) {
            e2b_s3_launch_lin_ksplit(out, x, weight, static_cast<int>(m), static_cast<int>(n), 1536, 6, 1);
          } else {
            e2b_s3_launch_lin_ksplit(out, x, weight, static_cast<int>(m), static_cast<int>(n), 1536, 6, 2);
          }
        } else {
          e2b_s3_launch_lin_ksplit(out, x, weight, static_cast<int>(m), static_cast<int>(n), 1536, 4, 1);
        }
        return out;
      }
    } else if (k == 2048 && n == 1536) {
      auto out = x.new_empty({m, n});
      if (m <= 2) {
        e2b_s9_launch_lin_gemv(out, x, weight, static_cast<int>(m), 1536, 2048, 2, 96);
      } else if (m <= 16) {
        e2b_s3_launch_lin_ksplit(out, x, weight, static_cast<int>(m), 1536, 2048, 8, 1);
      } else {
        e2b_s3_launch_lin_ksplit_m32(out, x, weight, static_cast<int>(m), 1536, 2048, 8, 1);
      }
      return out;
    } else if (k == 4096 && n == 1536) {
      if (m > 32) {
        return at::linear(x, weight);
      }
      auto out = x.new_empty({m, n});
      if (m == 1) {
        e2b_s9_launch_lin_gemv(out, x, weight, 1, 1536, 4096, 2, 64);
      } else if (m == 2) {
        e2b_s9_launch_lin_gemv(out, x, weight, 2, 1536, 4096, 2, 96);
      } else if (m <= 8) {
        e2b_s3_launch_lin_ksplit(out, x, weight, static_cast<int>(m), 1536, 4096, 8, 1, (m == 8) ? 1 : 0);
      } else if (m <= 16) {
        e2b_s3_launch_lin_ksplit(out, x, weight, static_cast<int>(m), 1536, 4096, 12, 1, 1);
      } else if (m == 17) {
        e2b_s3_launch_lin_ksplit_m32(out, x, weight, 17, 1536, 4096, 12, 1);
      } else {
        e2b_s3_launch_lin_ksplit_m32(out, x, weight, static_cast<int>(m), 1536, 4096, 8, 1);
      }
      return out;
    } else if (k == 6144 && n == 1536) {
      auto out = x.new_empty({m, n});
      if (m == 1) {
        e2b_s3_launch_wide_gemv(out, x, weight, 1, 1536, 6144, 1);
      } else if (m == 2) {
        e2b_s9_launch_lin_gemv(out, x, weight, 2, 1536, 6144, 2, 96);
      } else if (m <= 16) {
        e2b_s3_launch_lin_ksplit(out, x, weight, static_cast<int>(m), 1536, 6144, 12, 1, (m >= 6) ? 1 : 0);
      } else if (m <= 18) {
        e2b_s3_launch_lin_ksplit_m32(out, x, weight, static_cast<int>(m), 1536, 6144, 12, 1, 1);
      } else {
        e2b_s3_launch_lin_ksplit_m32(out, x, weight, static_cast<int>(m), 1536, 6144, 8, 1, 1);
      }
      return out;
    } else if (k == 12288 && n == 1536) {
      auto out = x.new_empty({m, n});
      if (m == 1) {
        e2b_s3_launch_wide_gemv(out, x, weight, 1, 1536, 12288, 1);
      } else if (m == 2 || m == 4) {
        e2b_s9_launch_lin_gemv(out, x, weight, static_cast<int>(m), 1536, 12288, 2, 96);
      } else if (m >= 21) {
        e2b_s3_launch_lin_ksplit_m32(out, x, weight, static_cast<int>(m), 1536, 12288, 8, 1, 1);
      } else if (m >= 17) {
        e2b_s3_launch_lin_ksplit_m32(out, x, weight, static_cast<int>(m), 1536, 12288, 12, 1, 1);
      } else if (m >= 8) {
        e2b_s3_launch_lin_ksplit(out, x, weight, static_cast<int>(m), 1536, 12288, 16, 1, 1);
      } else {
        e2b_s3_launch_lin_ksplit(out, x, weight, static_cast<int>(m), 1536, 12288, 12, 1, (m >= 6) ? 1 : 0);
      }
      return out;
    }
  }
  return at::linear(x, weight);
}

torch::Tensor e2b_lm_head_softcap_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double soft_cap) {
  int64_t m = x.size(0);
  int64_t n = weight.size(0);
  if (m == 0) {
    return x.new_empty({0, n});
  }
  int64_t k = x.size(1);
  if (
      m >= 1 && m <= 32 && k == 1536 && (n & 7) == 0 &&
      soft_cap > 0.0 &&
      x.dim() == 2 && weight.dim() == 2 && weight.size(1) == 1536 &&
      x.is_cuda() && weight.is_cuda() &&
      x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
      x.is_contiguous() && weight.is_contiguous() &&
      (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0)
  ) {
    auto out = x.new_empty({m, n});
    e2b_s3_launch_lm_head_softcap(out, x, weight, soft_cap);
    return out;
  }
  auto lg = at::linear(x, weight);
  lg.div_(soft_cap);
  lg.tanh_();
  lg.mul_(soft_cap);
  return lg;
}

TORCH_LIBRARY_FRAGMENT(vllm, m) {
  m.def(
      "gemma4_q_norm_rope(Tensor q, Tensor positions, Tensor q_weight, "
      "Tensor cos_sin_cache, float eps, int head_dim, int rope_angles) -> Tensor");
  m.def("e2b_mlp_gate_up_gelu(Tensor x, Tensor weight) -> Tensor");
  m.def("e2b_skinny_linear(Tensor x, Tensor weight) -> Tensor");
  m.def("e2b_lm_head_softcap(Tensor x, Tensor weight, float soft_cap) -> Tensor");
}

TORCH_LIBRARY_IMPL(vllm, CUDA, m) {
  m.impl("gemma4_q_norm_rope", &gemma4_q_norm_rope_cuda);
  m.impl("e2b_mlp_gate_up_gelu", &e2b_mlp_gate_up_gelu_cuda);
  m.impl("e2b_skinny_linear", &e2b_skinny_linear_cuda);
  m.impl("e2b_lm_head_softcap", &e2b_lm_head_softcap_cuda);
}
"""

_EXT_MOD = None
_HAS_CUTLASS_DUAL = False


def _prepare_cutlass_dual_includes(build_dir: str) -> list[str]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, ".deps", "cutlass-src"),
        "/projects/gemma4-vllm/my-vllm/.deps/cutlass-src",
    ]
    for root in candidates:
        cutlass_inc = os.path.join(root, "include")
        dual_src = os.path.join(root, "examples", "45_dual_gemm")
        if os.path.isdir(cutlass_inc) and os.path.isdir(dual_src):
            staged = os.path.join(build_dir, "dual_gemm_inc")
            marker = os.path.join(staged, ".patched_v1")
            if not os.path.isfile(marker):
                if os.path.isdir(staged):
                    shutil.rmtree(staged, ignore_errors=True)
                os.makedirs(staged, exist_ok=True)
                shutil.copy2(
                    os.path.join(dual_src, "dual_gemm_common.h"),
                    os.path.join(staged, "dual_gemm_common.h"),
                )
                for sub in ("device", "kernel", "threadblock"):
                    shutil.copytree(
                        os.path.join(dual_src, sub),
                        os.path.join(staged, sub),
                    )
                epi_path = os.path.join(staged, "threadblock", "dual_epilogue.h")
                with open(epi_path) as f:
                    epi_src = f.read()
                old_blk = (
                    "      CUTLASS_PRAGMA_UNROLL\n"
                    "      for (int i = 0; i < 2; ++i) {\n"
                    "        source_iterator[i].load(source_fragment[i]);\n"
                    "        ++source_iterator[i];\n"
                    "      }"
                )
                new_blk = (
                    "      if (output_op0.is_source_needed()) {\n"
                    "        source_iterator[0].load(source_fragment[0]);\n"
                    "        ++source_iterator[0];\n"
                    "      }\n"
                    "      if (output_op1.is_source_needed()) {\n"
                    "        source_iterator[1].load(source_fragment[1]);\n"
                    "        ++source_iterator[1];\n"
                    "      }"
                )
                if old_blk in epi_src:
                    with open(epi_path, "w") as f:
                        f.write(epi_src.replace(old_blk, new_blk, 1))
                with open(marker, "w") as f:
                    f.write("ok\n")
            return [cutlass_inc, staged]
    return []


def get_gemma4_e2b_proj_ext():
    global _EXT_MOD, _HAS_CUTLASS_DUAL
    if _EXT_MOD is None:
        ext_root = (
            os.environ.get("VLLM_GEMMA4_SM80_EXT_DIR")
            or "/var/tmp/vllm_gemma4_e2b_proj_ext"
        )
        src_hash = hashlib.sha256((_CUDA_SRC + _CPP_SRC).encode()).hexdigest()[:12]
        ext_name = f"gemma4_e2b_proj_ops_{src_hash}"
        build_dir = os.path.join(ext_root, ext_name)
        os.makedirs(build_dir, exist_ok=True)
        inc_paths = _prepare_cutlass_dual_includes(build_dir)
        cuda_cflags = ["-O3", "-arch=sm_80", "--expt-relaxed-constexpr"]
        if inc_paths:
            cuda_cflags.append("-DGEMMA4_HAS_CUTLASS_DUAL")
            _HAS_CUTLASS_DUAL = True
        else:
            _HAS_CUTLASS_DUAL = False
        so_path = os.path.join(build_dir, f"{ext_name}.so")
        if os.path.isfile(so_path):
            try:
                spec = importlib.util.spec_from_file_location(ext_name, so_path)
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    _EXT_MOD = mod
                    return _EXT_MOD
            except Exception:
                pass
        _EXT_MOD = load_inline(
            name=ext_name,
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=[
                "e2b_launch_q_norm_rope",
                "e2b_launch_gelu_tanh_and_mul_bf16x8",
                "e2b_launch_dual_gemm_gelu",
            ],
            extra_include_paths=inc_paths,
            extra_cuda_cflags=cuda_cflags,
            build_directory=build_dir,
            verbose=False,
        )
    return _EXT_MOD


_FAKE_REGISTERED = False


def _register_e2b_proj_ops():
    global _FAKE_REGISTERED
    get_gemma4_e2b_proj_ext()
    if _FAKE_REGISTERED:
        return
    _FAKE_REGISTERED = True

    @torch.library.register_fake("vllm::gemma4_q_norm_rope")
    def gemma4_q_norm_rope_fake(
        q: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        eps: float,
        head_dim: int,
        rope_angles: int,
    ) -> torch.Tensor:
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)

    @torch.library.register_fake("vllm::e2b_mlp_gate_up_gelu")
    def e2b_mlp_gate_up_gelu_fake(
        x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return x.new_empty((x.shape[0], weight.shape[0] // 2))

    @torch.library.register_fake("vllm::e2b_skinny_linear")
    def e2b_skinny_linear_fake(
        x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return x.new_empty((x.shape[0], weight.shape[0]))

    @torch.library.register_fake("vllm::e2b_lm_head_softcap")
    def e2b_lm_head_softcap_fake(
        x: torch.Tensor, weight: torch.Tensor, soft_cap: float
    ) -> torch.Tensor:
        return x.new_empty((x.shape[0], weight.shape[0]))


def e2b_lm_head_softcap(
    x: torch.Tensor, weight: torch.Tensor, soft_cap: float
) -> torch.Tensor:
    return torch.ops.vllm.e2b_lm_head_softcap(x, weight, float(soft_cap))


_register_e2b_proj_ops()
