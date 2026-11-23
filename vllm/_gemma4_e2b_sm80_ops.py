import hashlib
import importlib.util
import os
from types import ModuleType

import torch
from torch.utils.cpp_extension import load_inline

_CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor gemma4_e2b_project_ple_combine(
    torch::Tensor proj,
    c10::optional<torch::Tensor> ple_in,
    torch::Tensor norm_w,
    torch::Tensor proj_scale,
    torch::Tensor in_scale,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> gemma4_e2b_post_attn_pre_ffn_norms(
    torch::Tensor attn_out,
    torch::Tensor residual,
    torch::Tensor post_attn_w,
    torch::Tensor pre_ffn_w,
    double eps);

torch::Tensor gemma4_e2b_post_ffn_res_norm(
    torch::Tensor mlp_out,
    torch::Tensor residual,
    torch::Tensor post_ffn_w,
    double eps);

torch::Tensor gemma4_e2b_ple_gate_gelu_mul(
    torch::Tensor hidden,
    torch::Tensor w_gate,
    torch::Tensor ple_in);

torch::Tensor gemma4_e2b_ple_proj_gemv(
    torch::Tensor gated_ple,
    torch::Tensor w_proj);

std::tuple<torch::Tensor, torch::Tensor> gemma4_e2b_ple_post_norm_scalar_next_norm(
    torch::Tensor ple_proj,
    torch::Tensor residual,
    torch::Tensor post_ple_w,
    torch::Tensor layer_scalar,
    torch::Tensor next_norm_w,
    double eps);
torch::Tensor gemma4_e2b_ple_post_norm_scalar(
    torch::Tensor ple_proj,
    torch::Tensor residual,
    torch::Tensor post_ple_w,
    torch::Tensor layer_scalar,
    double eps);
torch::Tensor gemma4_e2b_model_proj_gemv(
    torch::Tensor x,
    torch::Tensor w_proj);

torch::Tensor gemma4_e2b_embed_scale(
    torch::Tensor input_ids,
    torch::Tensor weight,
    torch::Tensor scale);

torch::Tensor gemma4_e2b_project_ple_combine_from_ids(
    torch::Tensor proj,
    torch::Tensor input_ids,
    torch::Tensor emb_ple_w,
    torch::Tensor norm_w,
    torch::Tensor proj_scale,
    torch::Tensor emb_scale,
    torch::Tensor in_scale,
    int64_t vocab_ple,
    int64_t layer_offset,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> gemma4_e2b_post_ffn_ple_gate(
    torch::Tensor mlp_out,
    torch::Tensor residual,
    torch::Tensor post_ffn_w,
    torch::Tensor w_gate,
    torch::Tensor ple_in,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> gemma4_e2b_embed_scale_and_norm(
    torch::Tensor input_ids,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor norm_w,
    double eps);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <vector>

namespace {

struct alignas(16) bf16x8 {
    __nv_bfloat16 v[8];
};

struct alignas(16) bf162x4 {
    __nv_bfloat162 v[4];
};

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float block_sum_192_single_barrier(float val, float* s_warp) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    val = warp_sum(val);
    if (lane == 0) s_warp[wid] = val;
    __syncthreads();
    // Each call passes a disjoint s_warp buffer so back-to-back calls need only 1 barrier.
    const float v = (lane < 6) ? s_warp[lane] : 0.f;
    return warp_sum(v);
}

__device__ __forceinline__ float read_scalar_f32(const void* ptr, bool is_bf16) {
    if (is_bf16) {
        return __bfloat162float(__ldg(reinterpret_cast<const __nv_bfloat16*>(ptr)));
    }
    return __ldg(reinterpret_cast<const float*>(ptr));
}

__device__ __forceinline__ float dot8_bf16(uint4 w, uint4 x) {
    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w);
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&x);
    float s0 = 0.f, s1 = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 wf = __bfloat1622float2(wp[i]);
        float2 xf = __bfloat1622float2(xp[i]);
        s0 = fmaf(wf.x, xf.x, s0);
        s1 = fmaf(wf.y, xf.y, s1);
    }
    return s0 + s1;
}

__device__ __forceinline__ uint4 ld_cs_u4(const uint4* ptr) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ __nv_bfloat16 gelu_tanh_mul_bf16_val(float g, __nv_bfloat16 ple) {
    constexpr float BETA = 0.7978845608028654f;
    constexpr float KAPPA = 0.044715f;
    float inner = BETA * (g + KAPPA * g * g * g);
    float gelu_bf16 = __bfloat162float(__float2bfloat16_rn(0.5f * g * (1.0f + tanhf(inner))));
    return __float2bfloat16_rn(gelu_bf16 * __bfloat162float(ple));
}

__device__ __forceinline__ __nv_bfloat16 gelu_tanh_mul_bf16(float gate_acc, __nv_bfloat16 ple) {
    float g = __bfloat162float(__float2bfloat16_rn(gate_acc));
    return gelu_tanh_mul_bf16_val(g, ple);
}

__device__ __forceinline__ uint32_t pack_bf16x2_raw(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void p6_cp_async_ca_16B(void* smem_ptr, const void* gmem_ptr, bool pred) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.ca.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
        "}\n" :: "r"(s), "l"(gmem_ptr), "r"(pred ? 1 : 0) : "memory");
}

__device__ __forceinline__ void p6_cp_async_cg_16B(void* smem_ptr, const void* gmem_ptr, bool pred) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
        "}\n" :: "r"(s), "l"(gmem_ptr), "r"(pred ? 1 : 0) : "memory");
}

__device__ __forceinline__ void p6_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N_WAIT>
__device__ __forceinline__ void p6_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT) : "memory");
}

__device__ __forceinline__ int p6_swz64(int r, int c8) {
    return c8 ^ (r & 7);
}

__device__ __forceinline__ void p6_ldmatrix_x2(uint32_t dst[2], const void* smem_ptr) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(dst[0]), "=r"(dst[1]) : "r"(s));
}

__device__ __forceinline__ void p6_ldmatrix_x4(uint32_t dst[4], const void* smem_ptr) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(s));
}

__device__ __forceinline__ void p6_mma_m16n8k16_bf16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ int p10_swz64(int r, int c8) {
    return c8 ^ ((r & 7) ^ ((r >> 3) & 1));
}

__device__ __forceinline__ void p10_cp_async_ca_16B(void* smem_ptr, const void* gmem_ptr, bool pred) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem_ptr), "r"(pred ? 16 : 0) : "memory");
}

__device__ __forceinline__ void p10_cp_async_cg_16B(void* smem_ptr, const void* gmem_ptr, bool pred) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem_ptr), "r"(pred ? 16 : 0) : "memory");
}

__device__ __forceinline__ void p10_cp_async_cg_evict_16B(void* smem_ptr, const void* gmem_ptr, bool pred) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n .reg .pred p;\n .reg .b64 desc;\n"
        " createpolicy.fractional.L2::evict_first.b64 desc, 1.0;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, desc;\n"
        " @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n}\n"
        :: "r"(s), "l"(gmem_ptr), "r"(pred ? 1 : 0) : "memory");
}

// ----------------------------------------------------------------------------
// 1. project_ple_combine: [M, 8960] (+ optional [M, 35, 256]) -> contiguous [35, M, 256]
// ----------------------------------------------------------------------------
template <bool HAS_PLE_IN>
__global__ __launch_bounds__(256)
void gemma4_e2b_project_ple_combine_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ proj,
    const __nv_bfloat16* __restrict__ ple_in,
    const __nv_bfloat16* __restrict__ norm_w,
    const void* __restrict__ proj_scale_ptr,
    bool proj_scale_bf16,
    const void* __restrict__ in_scale_ptr,
    bool in_scale_bf16,
    int M,
    int64_t proj_m_stride,
    int64_t ple_m_stride,
    float eps)
{
    constexpr int L = 35;
    constexpr int D_CHUNKS = 32;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * 8 + wid;
    const int64_t total_rows = static_cast<int64_t>(M) * L;
    if (row_idx >= total_rows) return;

    const int64_t m = row_idx / L;
    const int64_t l = row_idx - m * L;

    const float p_scale = read_scalar_f32(proj_scale_ptr, proj_scale_bf16);
    const uint4 p_u4 = reinterpret_cast<const uint4*>(proj + m * proj_m_stride + l * 256)[lane];
    const uint4 w_u4 = __ldg(&reinterpret_cast<const uint4*>(norm_w)[lane]);
    const bf16x8* p_vec = reinterpret_cast<const bf16x8*>(&p_u4);
    const bf16x8* w_vec = reinterpret_cast<const bf16x8*>(&w_u4);

    float p0[8];
    float sq = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v = __bfloat162float(__float2bfloat16_rn(__bfloat162float(p_vec->v[j]) * p_scale));
        p0[j] = v;
        sq = fmaf(v, v, sq);
    }
    sq = warp_sum(sq);
    const float inv_rms = rsqrtf(sq * (1.0f / 256.0f) + eps);

    bf16x8 out_vec;
    if constexpr (HAS_PLE_IN) {
        const float i_scale = read_scalar_f32(in_scale_ptr, in_scale_bf16);
        const uint4 ple_u4 = reinterpret_cast<const uint4*>(ple_in + m * ple_m_stride + l * 256)[lane];
        const bf16x8* ple_vec = reinterpret_cast<const bf16x8*>(&ple_u4);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float norm_b = __bfloat162float(__float2bfloat16_rn(p0[j] * inv_rms * __bfloat162float(w_vec->v[j])));
            float sum_b = __bfloat162float(__float2bfloat16_rn(norm_b + __bfloat162float(ple_vec->v[j])));
            out_vec.v[j] = __float2bfloat16_rn(sum_b * i_scale);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_vec.v[j] = __float2bfloat16_rn(p0[j] * inv_rms * __bfloat162float(w_vec->v[j]));
        }
    }

    const int64_t out_row = l * static_cast<int64_t>(M) + m;
    reinterpret_cast<uint4*>(out)[out_row * D_CHUNKS + lane] = *reinterpret_cast<const uint4*>(&out_vec);
}

// ----------------------------------------------------------------------------
// 1b. Fused embed_scale: input_ids [M] -> embed[tok] * scale -> [M, 1536]
// ----------------------------------------------------------------------------
template <typename idx_t>
__global__ __launch_bounds__(192)
void gemma4_e2b_embed_scale_kernel(
    __nv_bfloat16* __restrict__ out,
    const idx_t* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ weight,
    const void* __restrict__ scale_ptr,
    bool scale_bf16,
    int64_t num_embeddings,
    int M)
{
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int64_t tok = static_cast<int64_t>(input_ids[m]);
    const float sc = read_scalar_f32(scale_ptr, scale_bf16);
    uint4 out_u4 = make_uint4(0, 0, 0, 0);
    if (tok >= 0 && tok < num_embeddings) {
        const uint4 w_u4 = __ldg(&reinterpret_cast<const uint4*>(weight + tok * 1536)[tid]);
        const bf16x8* wv = reinterpret_cast<const bf16x8*>(&w_u4);
        bf16x8* ov = reinterpret_cast<bf16x8*>(&out_u4);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            ov->v[j] = __float2bfloat16_rn(__bfloat162float(wv->v[j]) * sc);
        }
    }
    reinterpret_cast<uint4*>(out + static_cast<int64_t>(m) * 1536)[tid] = out_u4;
}

template <typename idx_t>
__global__ __launch_bounds__(192)
void gemma4_e2b_embed_scale_and_norm_192t_kernel(
    __nv_bfloat16* __restrict__ hidden_out,
    __nv_bfloat16* __restrict__ norm_out,
    const idx_t* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ embed_w,
    const void* __restrict__ scale_ptr,
    bool scale_bf16,
    const __nv_bfloat16* __restrict__ norm_w,
    int64_t num_embeddings,
    int64_t embed_stride0,
    float eps)
{
    __shared__ float s_warp[6];
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t tok = static_cast<int64_t>(__ldg(&input_ids[m]));
    const float scale_f = read_scalar_f32(scale_ptr, scale_bf16);
    const uint4 e_u4 = (tok >= 0 && tok < num_embeddings)
        ? __ldg(&reinterpret_cast<const uint4*>(embed_w + tok * embed_stride0)[tid])
        : make_uint4(0, 0, 0, 0);
    const uint4 w_u4 = __ldg(&reinterpret_cast<const uint4*>(norm_w)[tid]);
    const bf162x4* ev = reinterpret_cast<const bf162x4*>(&e_u4);
    const bf162x4* wv = reinterpret_cast<const bf162x4*>(&w_u4);

    uint4 h_u4;
    bf162x4* hv = reinterpret_cast<bf162x4*>(&h_u4);
    float2 hf[4];
    float sq = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 ef = __bfloat1622float2(ev->v[j]);
        __nv_bfloat162 hb = __floats2bfloat162_rn(ef.x * scale_f, ef.y * scale_f);
        hv->v[j] = hb;
        float2 h = __bfloat1622float2(hb);
        hf[j] = h;
        sq = fmaf(h.x, h.x, sq);
        sq = fmaf(h.y, h.y, sq);
    }
    reinterpret_cast<uint4*>(hidden_out + static_cast<int64_t>(m) * 1536)[tid] = h_u4;
    const float inv = rsqrtf(block_sum_192_single_barrier(sq, s_warp) * (1.0f / 1536.0f) + eps);
    uint4 n_u4;
    bf162x4* nv = reinterpret_cast<bf162x4*>(&n_u4);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 wf = __bfloat1622float2(wv->v[j]);
        nv->v[j] = __floats2bfloat162_rn(hf[j].x * inv * wf.x, hf[j].y * inv * wf.y);
    }
    reinterpret_cast<uint4*>(norm_out + static_cast<int64_t>(m) * 1536)[tid] = n_u4;
}

template <typename idx_t>
__global__ __launch_bounds__(256)
void gemma4_e2b_embed_scale_and_norm_warp_kernel(
    __nv_bfloat16* __restrict__ hidden_out,
    __nv_bfloat16* __restrict__ norm_out,
    const idx_t* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ embed_w,
    const void* __restrict__ scale_ptr,
    bool scale_bf16,
    const __nv_bfloat16* __restrict__ norm_w,
    int M,
    int64_t num_embeddings,
    int64_t embed_stride0,
    float eps)
{
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.x * (blockDim.x >> 5) + wid;
    if (m >= M) return;

    const int64_t tok = static_cast<int64_t>(__ldg(&input_ids[m]));
    const float scale_f = read_scalar_f32(scale_ptr, scale_bf16);
    const bool valid_tok = (tok >= 0 && tok < num_embeddings);
    const uint4* ep = valid_tok
        ? reinterpret_cast<const uint4*>(embed_w + tok * embed_stride0)
        : nullptr;
    const uint4* wp = reinterpret_cast<const uint4*>(norm_w);

    float2 hf[6][4];
    float sq = 0.f;
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int idx = it * 32 + lane;
        uint4 e_u4 = valid_tok ? __ldg(&ep[idx]) : make_uint4(0, 0, 0, 0);
        const bf162x4* ev = reinterpret_cast<const bf162x4*>(&e_u4);
        uint4 h_u4;
        bf162x4* hv = reinterpret_cast<bf162x4*>(&h_u4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 ef = __bfloat1622float2(ev->v[j]);
            __nv_bfloat162 hb = __floats2bfloat162_rn(ef.x * scale_f, ef.y * scale_f);
            hv->v[j] = hb;
            float2 h = __bfloat1622float2(hb);
            hf[it][j] = h;
            sq = fmaf(h.x, h.x, sq);
            sq = fmaf(h.y, h.y, sq);
        }
        reinterpret_cast<uint4*>(hidden_out + static_cast<int64_t>(m) * 1536)[idx] = h_u4;
    }
    const float inv = rsqrtf(warp_sum(sq) * (1.0f / 1536.0f) + eps);
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int idx = it * 32 + lane;
        uint4 w_u4 = __ldg(&wp[idx]);
        const bf162x4* wv = reinterpret_cast<const bf162x4*>(&w_u4);
        uint4 n_u4;
        bf162x4* nv = reinterpret_cast<bf162x4*>(&n_u4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wf = __bfloat1622float2(wv->v[j]);
            nv->v[j] = __floats2bfloat162_rn(hf[it][j].x * inv * wf.x, hf[it][j].y * inv * wf.y);
        }
        reinterpret_cast<uint4*>(norm_out + static_cast<int64_t>(m) * 1536)[idx] = n_u4;
    }
}

// ----------------------------------------------------------------------------
// 1c. Fused project_ple_combine_from_ids: proj [M, L_out*256] + input_ids -> [L_out, M, 256]
// ----------------------------------------------------------------------------
template <typename idx_t, int WPC = 8, bool EMB_CS = false>
__global__ __launch_bounds__(WPC * 32)
void gemma4_e2b_project_ple_combine_from_ids_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ proj,
    const idx_t* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ emb_ple_w,
    const __nv_bfloat16* __restrict__ norm_w,
    const void* __restrict__ proj_scale_ptr,
    bool proj_scale_bf16,
    const void* __restrict__ emb_scale_ptr,
    bool emb_scale_bf16,
    const void* __restrict__ in_scale_ptr,
    bool in_scale_bf16,
    int M,
    int num_out_layers,
    int layer_offset,
    int64_t proj_m_stride,
    int64_t emb_stride0,
    int64_t vocab_limit,
    float eps)
{
    constexpr int D_CHUNKS = 32;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * WPC + wid;
    const int64_t total_rows = static_cast<int64_t>(M) * num_out_layers;
    if (row_idx >= total_rows) return;

    const int64_t m = row_idx / num_out_layers;
    const int64_t l_out = row_idx - m * num_out_layers;
    const int64_t l_global = layer_offset + l_out;

    const float p_scale = read_scalar_f32(proj_scale_ptr, proj_scale_bf16);
    const float e_scale = read_scalar_f32(emb_scale_ptr, emb_scale_bf16);
    const float i_scale = read_scalar_f32(in_scale_ptr, in_scale_bf16);

    const uint4 p_u4 = __ldg(
        &reinterpret_cast<const uint4*>(proj + m * proj_m_stride + l_out * 256)[lane]);
    const uint4 w_u4 = __ldg(&reinterpret_cast<const uint4*>(norm_w)[lane]);

    const int64_t tok = static_cast<int64_t>(input_ids[m]);
    const int64_t lookup_tok = (tok >= 0 && tok < vocab_limit) ? tok : 0;
    const uint4* emb_ptr = &reinterpret_cast<const uint4*>(
        emb_ple_w + lookup_tok * emb_stride0 + l_global * 256)[lane];
    const uint4 emb_u4 = EMB_CS ? ld_cs_u4(emb_ptr) : __ldg(emb_ptr);

    const bf16x8* p_vec = reinterpret_cast<const bf16x8*>(&p_u4);
    const bf16x8* w_vec = reinterpret_cast<const bf16x8*>(&w_u4);
    const bf16x8* e_vec = reinterpret_cast<const bf16x8*>(&emb_u4);

    float p0[8];
    float sq = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v = __bfloat162float(__float2bfloat16_rn(__bfloat162float(p_vec->v[j]) * p_scale));
        p0[j] = v;
        sq = fmaf(v, v, sq);
    }
    sq = warp_sum(sq);
    const float inv_rms = rsqrtf(sq * (1.0f / 256.0f) + eps);

    bf16x8 out_vec;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float norm_b = __bfloat162float(__float2bfloat16_rn(p0[j] * inv_rms * __bfloat162float(w_vec->v[j])));
        float raw_b = __bfloat162float(__float2bfloat16_rn(__bfloat162float(e_vec->v[j]) * e_scale));
        float sum_b = __bfloat162float(__float2bfloat16_rn(norm_b + raw_b));
        out_vec.v[j] = __float2bfloat16_rn(sum_b * i_scale);
    }

    const int64_t out_row = l_out * static_cast<int64_t>(M) + m;
    reinterpret_cast<uint4*>(out)[out_row * D_CHUNKS + lane] = *reinterpret_cast<const uint4*>(&out_vec);
}

template <typename idx_t>
__global__ __launch_bounds__(128)
void gemma4_e2b_project_ple_combine_from_ids_coal_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ proj,
    const idx_t* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ emb_ple_w,
    const __nv_bfloat16* __restrict__ norm_w,
    const void* __restrict__ proj_scale_ptr,
    bool proj_scale_bf16,
    const void* __restrict__ emb_scale_ptr,
    bool emb_scale_bf16,
    const void* __restrict__ in_scale_ptr,
    bool in_scale_bf16,
    int M,
    int64_t layer_offset,
    int64_t proj_m_stride,
    int64_t emb_stride0,
    int64_t vocab_limit,
    float eps)
{
    constexpr int D_CHUNKS = 32;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t m = static_cast<int64_t>(blockIdx.x) * 4 + wid;
    if (m >= M) return;

    const int64_t l_out = static_cast<int64_t>(blockIdx.y);
    const int64_t l_global = layer_offset + l_out;

    const float p_scale = read_scalar_f32(proj_scale_ptr, proj_scale_bf16);
    const float e_scale = read_scalar_f32(emb_scale_ptr, emb_scale_bf16);
    const float i_scale = read_scalar_f32(in_scale_ptr, in_scale_bf16);

    const uint4 p_u4 = __ldg(&reinterpret_cast<const uint4*>(proj + m * proj_m_stride + l_out * 256)[lane]);
    const uint4 w_u4 = __ldg(&reinterpret_cast<const uint4*>(norm_w)[lane]);

    const int64_t tok = static_cast<int64_t>(input_ids[m]);
    const int64_t lookup_tok = (tok >= 0 && tok < vocab_limit) ? tok : 0;
    const uint4 emb_u4 = __ldg(
        &reinterpret_cast<const uint4*>(emb_ple_w + lookup_tok * emb_stride0 + l_global * 256)[lane]);

    const bf16x8* p_vec = reinterpret_cast<const bf16x8*>(&p_u4);
    const bf16x8* w_vec = reinterpret_cast<const bf16x8*>(&w_u4);
    const bf16x8* e_vec = reinterpret_cast<const bf16x8*>(&emb_u4);

    float p0[8];
    float sq = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v = __bfloat162float(__float2bfloat16_rn(__bfloat162float(p_vec->v[j]) * p_scale));
        p0[j] = v;
        sq = fmaf(v, v, sq);
    }
    sq = warp_sum(sq);
    const float inv_rms = rsqrtf(sq * (1.0f / 256.0f) + eps);

    bf16x8 out_vec;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float norm_b = __bfloat162float(__float2bfloat16_rn(p0[j] * inv_rms * __bfloat162float(w_vec->v[j])));
        float raw_b = __bfloat162float(__float2bfloat16_rn(__bfloat162float(e_vec->v[j]) * e_scale));
        float sum_b = __bfloat162float(__float2bfloat16_rn(norm_b + raw_b));
        out_vec.v[j] = __float2bfloat16_rn(sum_b * i_scale);
    }

    const int64_t out_row = l_out * static_cast<int64_t>(M) + m;
    reinterpret_cast<uint4*>(out)[out_row * D_CHUNKS + lane] = *reinterpret_cast<const uint4*>(&out_vec);
}

// ----------------------------------------------------------------------------
// 2. post_attn_pre_ffn_norms: D=1536 (192 threads x 8 BF16), 3 -> 1 kernel
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(192)
void gemma4_e2b_post_attn_pre_ffn_norms_kernel(
    __nv_bfloat16* __restrict__ res_out,
    __nv_bfloat16* __restrict__ norm_out,
    const __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ post_attn_w,
    const __nv_bfloat16* __restrict__ pre_ffn_w,
    int64_t attn_stride0,
    int64_t res_stride0,
    float eps)
{
    __shared__ float s_warp0[6];
    __shared__ float s_warp1[6];
    const int m = blockIdx.x;
    const int tid = threadIdx.x;

    const uint4 a_u4 = reinterpret_cast<const uint4*>(attn_out + static_cast<int64_t>(m) * attn_stride0)[tid];
    const uint4 r_u4 = reinterpret_cast<const uint4*>(residual + static_cast<int64_t>(m) * res_stride0)[tid];
    const uint4 w1_u4 = __ldg(&reinterpret_cast<const uint4*>(post_attn_w)[tid]);
    const uint4 w2_u4 = __ldg(&reinterpret_cast<const uint4*>(pre_ffn_w)[tid]);

    const bf162x4* av = reinterpret_cast<const bf162x4*>(&a_u4);
    const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_u4);
    const bf162x4* w1 = reinterpret_cast<const bf162x4*>(&w1_u4);
    const bf162x4* w2 = reinterpret_cast<const bf162x4*>(&w2_u4);

    float2 af[4];
    float sq1 = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 x = __bfloat1622float2(av->v[j]);
        af[j] = x;
        sq1 = fmaf(x.x, x.x, sq1);
        sq1 = fmaf(x.y, x.y, sq1);
    }
    const float tot1 = block_sum_192_single_barrier(sq1, s_warp0);
    const float inv1 = rsqrtf(tot1 * (1.0f / 1536.0f) + eps);

    uint4 res_u4;
    bf162x4* res_v = reinterpret_cast<bf162x4*>(&res_u4);
    float2 rf[4];
    float sq2 = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 w1f = __bfloat1622float2(w1->v[j]);
        float2 rv_f = __bfloat1622float2(rv->v[j]);
        float2 nw1 = __bfloat1622float2(__floats2bfloat162_rn(af[j].x * inv1 * w1f.x, af[j].y * inv1 * w1f.y));
        __nv_bfloat162 r1_b = __floats2bfloat162_rn(nw1.x + rv_f.x, nw1.y + rv_f.y);
        res_v->v[j] = r1_b;
        float2 r1_f = __bfloat1622float2(r1_b);
        rf[j] = r1_f;
        sq2 = fmaf(r1_f.x, r1_f.x, sq2);
        sq2 = fmaf(r1_f.y, r1_f.y, sq2);
    }
    reinterpret_cast<uint4*>(res_out + static_cast<int64_t>(m) * 1536)[tid] = res_u4;

    const float tot2 = block_sum_192_single_barrier(sq2, s_warp1);
    const float inv2 = rsqrtf(tot2 * (1.0f / 1536.0f) + eps);

    uint4 norm_u4;
    bf162x4* norm_v = reinterpret_cast<bf162x4*>(&norm_u4);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 w2f = __bfloat1622float2(w2->v[j]);
        norm_v->v[j] = __floats2bfloat162_rn(rf[j].x * inv2 * w2f.x, rf[j].y * inv2 * w2f.y);
    }
    reinterpret_cast<uint4*>(norm_out + static_cast<int64_t>(m) * 1536)[tid] = norm_u4;
}

__global__ __launch_bounds__(128, 6)
void gemma4_e2b_post_attn_pre_ffn_norms_warp_kernel(
    __nv_bfloat16* __restrict__ res_out,
    __nv_bfloat16* __restrict__ norm_out,
    const __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ post_attn_w,
    const __nv_bfloat16* __restrict__ pre_ffn_w,
    int M,
    int64_t attn_stride0,
    int64_t res_stride0,
    float eps)
{
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.x * (blockDim.x >> 5) + wid;
    if (m >= M) return;

    const uint4* ap = reinterpret_cast<const uint4*>(attn_out + static_cast<int64_t>(m) * attn_stride0);
    const uint4* rp = reinterpret_cast<const uint4*>(residual + static_cast<int64_t>(m) * res_stride0);
    const uint4* w1p = reinterpret_cast<const uint4*>(post_attn_w);
    const uint4* w2p = reinterpret_cast<const uint4*>(pre_ffn_w);

    uint4 a_u4[6];
    uint4 r_u4[6];
    float sq1 = 0.f;
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int idx = it * 32 + lane;
        uint4 av_u4 = __ldg(&ap[idx]);
        a_u4[it] = av_u4;
        r_u4[it] = __ldg(&rp[idx]);
        const bf162x4* av = reinterpret_cast<const bf162x4*>(&av_u4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 x = __bfloat1622float2(av->v[j]);
            sq1 = fmaf(x.x, x.x, sq1);
            sq1 = fmaf(x.y, x.y, sq1);
        }
    }
    const float inv1 = rsqrtf(warp_sum(sq1) * (1.0f / 1536.0f) + eps);

    float sq2 = 0.f;
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int idx = it * 32 + lane;
        uint4 w1_u4 = __ldg(&w1p[idx]);
        const bf162x4* av = reinterpret_cast<const bf162x4*>(&a_u4[it]);
        const bf162x4* w1v = reinterpret_cast<const bf162x4*>(&w1_u4);
        const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_u4[it]);
        uint4 ro_u4;
        bf162x4* rov = reinterpret_cast<bf162x4*>(&ro_u4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 af_j = __bfloat1622float2(av->v[j]);
            float2 w1f = __bfloat1622float2(w1v->v[j]);
            float2 rvf = __bfloat1622float2(rv->v[j]);
            float2 nw1 = __bfloat1622float2(__floats2bfloat162_rn(af_j.x * inv1 * w1f.x, af_j.y * inv1 * w1f.y));
            __nv_bfloat162 r1b = __floats2bfloat162_rn(nw1.x + rvf.x, nw1.y + rvf.y);
            rov->v[j] = r1b;
            float2 r1f = __bfloat1622float2(r1b);
            sq2 = fmaf(r1f.x, r1f.x, sq2);
            sq2 = fmaf(r1f.y, r1f.y, sq2);
        }
        r_u4[it] = ro_u4;
        reinterpret_cast<uint4*>(res_out + static_cast<int64_t>(m) * 1536)[idx] = ro_u4;
    }
    const float inv2 = rsqrtf(warp_sum(sq2) * (1.0f / 1536.0f) + eps);

    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int idx = it * 32 + lane;
        uint4 w2_u4 = __ldg(&w2p[idx]);
        const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_u4[it]);
        const bf162x4* w2v = reinterpret_cast<const bf162x4*>(&w2_u4);
        uint4 no_u4;
        bf162x4* nov = reinterpret_cast<bf162x4*>(&no_u4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 rf_j = __bfloat1622float2(rv->v[j]);
            float2 w2f = __bfloat1622float2(w2v->v[j]);
            nov->v[j] = __floats2bfloat162_rn(rf_j.x * inv2 * w2f.x, rf_j.y * inv2 * w2f.y);
        }
        reinterpret_cast<uint4*>(norm_out + static_cast<int64_t>(m) * 1536)[idx] = no_u4;
    }
}

// ----------------------------------------------------------------------------
// 3. post_ffn_res_norm: D=1536 (192 threads x 8 BF16), 2 -> 1 kernel
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(192)
void gemma4_e2b_post_ffn_res_norm_kernel(
    __nv_bfloat16* __restrict__ res_out,
    const __nv_bfloat16* __restrict__ mlp_out,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ post_ffn_w,
    int64_t mlp_stride0,
    int64_t res_stride0,
    float eps)
{
    __shared__ float s_warp[6];
    const int m = blockIdx.x;
    const int tid = threadIdx.x;

    const uint4 m_u4 = reinterpret_cast<const uint4*>(mlp_out + static_cast<int64_t>(m) * mlp_stride0)[tid];
    const uint4 r_u4 = reinterpret_cast<const uint4*>(residual + static_cast<int64_t>(m) * res_stride0)[tid];
    const uint4 w_u4 = __ldg(&reinterpret_cast<const uint4*>(post_ffn_w)[tid]);

    const bf162x4* mv = reinterpret_cast<const bf162x4*>(&m_u4);
    const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_u4);
    const bf162x4* wv = reinterpret_cast<const bf162x4*>(&w_u4);

    float2 mf[4];
    float sq = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 x = __bfloat1622float2(mv->v[j]);
        mf[j] = x;
        sq = fmaf(x.x, x.x, sq);
        sq = fmaf(x.y, x.y, sq);
    }
    const float tot = block_sum_192_single_barrier(sq, s_warp);
    const float inv = rsqrtf(tot * (1.0f / 1536.0f) + eps);

    uint4 out_u4;
    bf162x4* out_v = reinterpret_cast<bf162x4*>(&out_u4);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 wf = __bfloat1622float2(wv->v[j]);
        float2 rvf = __bfloat1622float2(rv->v[j]);
        float2 nw = __bfloat1622float2(__floats2bfloat162_rn(mf[j].x * inv * wf.x, mf[j].y * inv * wf.y));
        out_v->v[j] = __floats2bfloat162_rn(nw.x + rvf.x, nw.y + rvf.y);
    }
    reinterpret_cast<uint4*>(res_out + static_cast<int64_t>(m) * 1536)[tid] = out_u4;
}

__global__ __launch_bounds__(128, 5)
void gemma4_e2b_post_ffn_res_norm_warp_kernel(
    __nv_bfloat16* __restrict__ res_out,
    const __nv_bfloat16* __restrict__ mlp_out,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ post_ffn_w,
    int M,
    int64_t mlp_stride0,
    int64_t res_stride0,
    float eps)
{
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.x * (blockDim.x >> 5) + wid;
    if (m >= M) return;

    const uint4* mp = reinterpret_cast<const uint4*>(mlp_out + static_cast<int64_t>(m) * mlp_stride0);
    const uint4* rp = reinterpret_cast<const uint4*>(residual + static_cast<int64_t>(m) * res_stride0);
    const uint4* wp = reinterpret_cast<const uint4*>(post_ffn_w);

    uint4 m_u4[6];
    float sq = 0.f;
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        uint4 mv_u4 = __ldg(&mp[it * 32 + lane]);
        m_u4[it] = mv_u4;
        const bf162x4* mv = reinterpret_cast<const bf162x4*>(&mv_u4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 x = __bfloat1622float2(mv->v[j]);
            sq = fmaf(x.x, x.x, sq);
            sq = fmaf(x.y, x.y, sq);
        }
    }
    const float inv = rsqrtf(warp_sum(sq) * (1.0f / 1536.0f) + eps);
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int idx = it * 32 + lane;
        uint4 r_u4 = __ldg(&rp[idx]);
        uint4 w_u4 = __ldg(&wp[idx]);
        const bf162x4* mv = reinterpret_cast<const bf162x4*>(&m_u4[it]);
        const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_u4);
        const bf162x4* wv = reinterpret_cast<const bf162x4*>(&w_u4);
        uint4 o_u4;
        bf162x4* ov = reinterpret_cast<bf162x4*>(&o_u4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 mf_j = __bfloat1622float2(mv->v[j]);
            float2 wf = __bfloat1622float2(wv->v[j]);
            float2 rvf = __bfloat1622float2(rv->v[j]);
            float2 nw = __bfloat1622float2(__floats2bfloat162_rn(mf_j.x * inv * wf.x, mf_j.y * inv * wf.y));
            ov->v[j] = __floats2bfloat162_rn(nw.x + rvf.x, nw.y + rvf.y);
        }
        reinterpret_cast<uint4*>(res_out + static_cast<int64_t>(m) * 1536)[idx] = o_u4;
    }
}

// ----------------------------------------------------------------------------
// 3b. Zero-barrier fused post_ffn_res_norm + ple_gate_gelu_mul (M <= 3)
// ----------------------------------------------------------------------------
template <int MAX_M, int COLS_PER_CTA, bool USE_CS = false>
__global__ __launch_bounds__(192)
void gemma4_e2b_post_ffn_ple_gate_small_m_kernel(
    __nv_bfloat16* __restrict__ res2_out,
    __nv_bfloat16* __restrict__ gated_out,
    const __nv_bfloat16* __restrict__ mlp_out,
    const __nv_bfloat16* __restrict__ res1,
    const __nv_bfloat16* __restrict__ post_ffn_w,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t mlp_s0,
    int64_t res1_s0,
    int64_t ple_s0,
    float eps)
{
    __shared__ float s_norm[6];
    __shared__ float s_warp[COLS_PER_CTA][MAX_M][6];
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int bid = blockIdx.x;
    const int n0 = bid * COLS_PER_CTA;

    uint4 wg_reg[COLS_PER_CTA];
    if constexpr (USE_CS) {
        #pragma unroll
        for (int c = 0; c < COLS_PER_CTA; ++c) {
            wg_reg[c] = ld_cs_u4(reinterpret_cast<const uint4*>(w_gate) + static_cast<int64_t>(n0 + c) * 192 + tid);
        }
    }

    const uint4 pfw_u4 = __ldg(&reinterpret_cast<const uint4*>(post_ffn_w)[tid]);
    const bf162x4* wv = reinterpret_cast<const bf162x4*>(&pfw_u4);

    uint4 res2_reg[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const uint4 m_u4 = __ldg(&reinterpret_cast<const uint4*>(mlp_out + static_cast<int64_t>(m) * mlp_s0)[tid]);
            const uint4 r_u4 = __ldg(&reinterpret_cast<const uint4*>(res1 + static_cast<int64_t>(m) * res1_s0)[tid]);
            const bf162x4* mv = reinterpret_cast<const bf162x4*>(&m_u4);
            const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_u4);
            float2 mf[4];
            float sq = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 x = __bfloat1622float2(mv->v[j]);
                mf[j] = x;
                sq = fmaf(x.x, x.x, sq);
                sq = fmaf(x.y, x.y, sq);
            }
            const float tot = block_sum_192_single_barrier(sq, s_norm);
            const float inv = rsqrtf(tot * (1.0f / 1536.0f) + eps);
            bf162x4* out_v = reinterpret_cast<bf162x4*>(&res2_reg[m]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 wf = __bfloat1622float2(wv->v[j]);
                float2 rvf = __bfloat1622float2(rv->v[j]);
                float2 nw = __bfloat1622float2(__floats2bfloat162_rn(mf[j].x * inv * wf.x, mf[j].y * inv * wf.y));
                out_v->v[j] = __floats2bfloat162_rn(nw.x + rvf.x, nw.y + rvf.y);
            }
            if (bid == m) {
                reinterpret_cast<uint4*>(res2_out + static_cast<int64_t>(m) * 1536)[tid] = res2_reg[m];
            }
        }
    }

    if constexpr (!USE_CS) {
        #pragma unroll
        for (int c = 0; c < COLS_PER_CTA; ++c) {
            wg_reg[c] = __ldg(&reinterpret_cast<const uint4*>(w_gate)[static_cast<int64_t>(n0 + c) * 192 + tid]);
        }
    }
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            #pragma unroll
            for (int c = 0; c < COLS_PER_CTA; ++c) {
                float ws = warp_sum(dot8_bf16(wg_reg[c], res2_reg[m]));
                if (lane == 0) s_warp[c][m][wid] = ws;
            }
        }
    }
    __syncthreads();
    if (tid < COLS_PER_CTA * M) {
        const int c = tid % COLS_PER_CTA;
        const int m = tid / COLS_PER_CTA;
        const int n = n0 + c;
        float sum = 0.f;
        #pragma unroll
        for (int w = 0; w < 6; ++w) sum += s_warp[c][m][w];
        __nv_bfloat16 pv = __ldg(&ple_in[static_cast<int64_t>(m) * ple_s0 + n]);
        gated_out[static_cast<int64_t>(m) * 256 + n] = gelu_tanh_mul_bf16(sum, pv);
    }
}

template <int MAX_M, int COLS_PER_CTA, bool USE_CS = false>
__global__ __launch_bounds__(192)
void gemma4_e2b_post_ffn_ple_gate_batched_kernel(
    __nv_bfloat16* __restrict__ res2_out,
    __nv_bfloat16* __restrict__ gated_out,
    const __nv_bfloat16* __restrict__ mlp_out,
    const __nv_bfloat16* __restrict__ res1,
    const __nv_bfloat16* __restrict__ post_ffn_w,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t mlp_s0,
    int64_t res1_s0,
    int64_t ple_s0,
    float eps)
{
    __shared__ float s_norm[MAX_M][6];
    __shared__ float s_warp[COLS_PER_CTA][MAX_M][6];
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int bid = blockIdx.x;
    const int n0 = bid * COLS_PER_CTA;

    const uint4 pfw_u4 = __ldg(&reinterpret_cast<const uint4*>(post_ffn_w)[tid]);
    uint4 wg_reg[COLS_PER_CTA];
    #pragma unroll
    for (int c = 0; c < COLS_PER_CTA; ++c) {
        const uint4* wp = &reinterpret_cast<const uint4*>(w_gate)[static_cast<int64_t>(n0 + c) * 192 + tid];
        wg_reg[c] = USE_CS ? ld_cs_u4(wp) : __ldg(wp);
    }

    uint4 m_reg[MAX_M], r_reg[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            m_reg[m] = __ldg(&reinterpret_cast<const uint4*>(mlp_out + static_cast<int64_t>(m) * mlp_s0)[tid]);
            r_reg[m] = __ldg(&reinterpret_cast<const uint4*>(res1 + static_cast<int64_t>(m) * res1_s0)[tid]);
            const bf162x4* mv = reinterpret_cast<const bf162x4*>(&m_reg[m]);
            float sq = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 x = __bfloat1622float2(mv->v[j]);
                sq = fmaf(x.x, x.x, sq);
                sq = fmaf(x.y, x.y, sq);
            }
            float ws = warp_sum(sq);
            if (lane == 0) s_norm[m][wid] = ws;
        }
    }
    __syncthreads();

    const bf162x4* wv = reinterpret_cast<const bf162x4*>(&pfw_u4);
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            const float tot = warp_sum(lane < 6 ? s_norm[m][lane] : 0.f);
            const float inv = rsqrtf(tot * (1.0f / 1536.0f) + eps);
            const bf162x4* mv = reinterpret_cast<const bf162x4*>(&m_reg[m]);
            const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_reg[m]);
            uint4 res2_reg;
            bf162x4* out_v = reinterpret_cast<bf162x4*>(&res2_reg);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 mf = __bfloat1622float2(mv->v[j]);
                float2 wf = __bfloat1622float2(wv->v[j]);
                float2 rvf = __bfloat1622float2(rv->v[j]);
                float2 nw = __bfloat1622float2(__floats2bfloat162_rn(mf.x * inv * wf.x, mf.y * inv * wf.y));
                out_v->v[j] = __floats2bfloat162_rn(nw.x + rvf.x, nw.y + rvf.y);
            }
            if (bid == m) {
                reinterpret_cast<uint4*>(res2_out + static_cast<int64_t>(m) * 1536)[tid] = res2_reg;
            }
            #pragma unroll
            for (int c = 0; c < COLS_PER_CTA; ++c) {
                float ws = warp_sum(dot8_bf16(wg_reg[c], res2_reg));
                if (lane == 0) s_warp[c][m][wid] = ws;
            }
        }
    }
    __syncthreads();
    if (tid < COLS_PER_CTA * M) {
        const int c = tid % COLS_PER_CTA;
        const int m = tid / COLS_PER_CTA;
        const int n = n0 + c;
        float sum = 0.f;
        #pragma unroll
        for (int w = 0; w < 6; ++w) sum += s_warp[c][m][w];
        __nv_bfloat16 pv = __ldg(&ple_in[static_cast<int64_t>(m) * ple_s0 + n]);
        gated_out[static_cast<int64_t>(m) * 256 + n] = gelu_tanh_mul_bf16(sum, pv);
    }
}

// Zero-atomic fused post_ffn_res_norm + ple_gate_gelu_mul for 4 <= M <= 32
template <int STEP1_WARPS, int STEP2_WARPS>
__global__ __launch_bounds__(STEP1_WARPS * 32)
void p6_post_ffn_ple_gate_mma16_kernel(
    __nv_bfloat16* __restrict__ res2_out,
    __nv_bfloat16* __restrict__ gated_out,
    const __nv_bfloat16* __restrict__ mlp_out,
    const __nv_bfloat16* __restrict__ res1,
    const __nv_bfloat16* __restrict__ post_ffn_w,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t mlp_s0,
    int64_t res1_s0,
    int64_t ple_s0,
    float eps)
{
    constexpr int K = 1536;
    constexpr int K_TILES_TOTAL = 24;  // K / 64
    constexpr int K_TILES_PER_WARP = K_TILES_TOTAL / STEP2_WARPS;
    static_assert(K_TILES_TOTAL % STEP2_WARPS == 0, "24 must divide STEP2_WARPS");

    extern __shared__ __align__(16) char dyn_smem[];
    uint4* sRes2_u4 = reinterpret_cast<uint4*>(dyn_smem);  // 24 * 16 * 8 = 49152 B
    uint4* sB_u4 = sRes2_u4 + 24 * 16 * 8;
    float* sRed = reinterpret_cast<float*>(sB_u4 + STEP2_WARPS * 2 * 64);

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int bid = blockIdx.x;
    const int m_base = blockIdx.y * 16;
    const int m_rows = (M - m_base < 16) ? (M - m_base) : 16;

    for (int ml = wid; ml < m_rows; ml += STEP1_WARPS) {
        const int gm = m_base + ml;
        const uint4* mp = reinterpret_cast<const uint4*>(mlp_out + static_cast<int64_t>(gm) * mlp_s0);
        const uint4* rp = reinterpret_cast<const uint4*>(res1 + static_cast<int64_t>(gm) * res1_s0);
        const uint4* wp = reinterpret_cast<const uint4*>(post_ffn_w);
        uint4 m_u4[6];
        float sq = 0.f;
        #pragma unroll
        for (int it = 0; it < 6; ++it) {
            uint4 mv_u4 = __ldg(&mp[it * 32 + lane]);
            m_u4[it] = mv_u4;
            const bf162x4* mv = reinterpret_cast<const bf162x4*>(&mv_u4);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 x = __bfloat1622float2(mv->v[j]);
                sq = fmaf(x.x, x.x, sq);
                sq = fmaf(x.y, x.y, sq);
            }
        }
        const float inv = rsqrtf(warp_sum(sq) * (1.0f / 1536.0f) + eps);
        const bool do_write = (bid == ml);
        uint4* gop = reinterpret_cast<uint4*>(res2_out + static_cast<int64_t>(gm) * 1536);
        #pragma unroll
        for (int it = 0; it < 6; ++it) {
            const int idx = it * 32 + lane;
            uint4 r_u4 = __ldg(&rp[idx]);
            uint4 w_u4 = __ldg(&wp[idx]);
            const bf162x4* mv = reinterpret_cast<const bf162x4*>(&m_u4[it]);
            const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_u4);
            const bf162x4* wv = reinterpret_cast<const bf162x4*>(&w_u4);
            uint4 o_u4;
            bf162x4* ov = reinterpret_cast<bf162x4*>(&o_u4);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 mf = __bfloat1622float2(mv->v[j]);
                float2 wf = __bfloat1622float2(wv->v[j]);
                float2 rvf = __bfloat1622float2(rv->v[j]);
                float2 nw = __bfloat1622float2(__floats2bfloat162_rn(mf.x * inv * wf.x, mf.y * inv * wf.y));
                ov->v[j] = __floats2bfloat162_rn(nw.x + rvf.x, nw.y + rvf.y);
            }
            const int chunk_id = idx >> 3;
            const int c8 = idx & 7;
            sRes2_u4[(chunk_id * 16 + ml) * 8 + p6_swz64(ml, c8)] = o_u4;
            if (do_write) {
                gop[idx] = o_u4;
            }
        }
    }
    __syncthreads();

    const int n_base = bid * 8;
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col = (lane >> 4);
    const int ldm_b_row = lane & 7;
    const int ldm_b_col = ((lane >> 3) & 1);
    const int gid = lane >> 2;
    const int t4 = lane & 3;

    if (wid < STEP2_WARPS) {
        const int k_tile_start = wid * K_TILES_PER_WARP;
        uint4* my_sB = sB_u4 + wid * 2 * 64;
        auto issue_b = [&](int stg, int kt_global) {
            const int k_base = kt_global * 64;
            uint4* dst = my_sB + stg * 64;
            #pragma unroll
            for (int idx = lane; idx < 64; idx += 32) {
                int r = idx >> 3;
                int c = idx & 7;
                p6_cp_async_cg_16B(
                    &dst[r * 8 + p6_swz64(r, c)],
                    w_gate + static_cast<int64_t>(n_base + r) * K + k_base + (c << 3),
                    true);
            }
        };
        issue_b(0, k_tile_start);
        p6_cp_async_commit();

        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int kt = 0; kt < K_TILES_PER_WARP; ++kt) {
            int cur = kt & 1;
            int kg = k_tile_start + kt;
            if (kt + 1 < K_TILES_PER_WARP) {
                issue_b(cur ^ 1, kg + 1);
                p6_cp_async_commit();
                p6_cp_async_wait<1>();
            } else {
                p6_cp_async_wait<0>();
            }
            __syncwarp();
            uint4* cur_b = my_sB + cur * 64;
            uint4* cur_a = sRes2_u4 + kg * 16 * 8;
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                uint32_t a_reg[4], b_reg[2];
                p6_ldmatrix_x4(a_reg, &cur_a[ldm_a_row * 8 + p6_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
                p6_ldmatrix_x2(b_reg, &cur_b[ldm_b_row * 8 + p6_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
                p6_mma_m16n8k16_bf16(acc, a_reg, b_reg);
            }
        }
        sRed[wid * 128 + gid * 8 + t4 * 2] = acc[0];
        sRed[wid * 128 + gid * 8 + t4 * 2 + 1] = acc[1];
        sRed[wid * 128 + (gid + 8) * 8 + t4 * 2] = acc[2];
        sRed[wid * 128 + (gid + 8) * 8 + t4 * 2 + 1] = acc[3];
    }
    __syncthreads();

    if constexpr (STEP2_WARPS == 8) {
        if (wid == 0 && gid < m_rows) {
            const int gm = m_base + gid;
            const int col0 = n_base + t4 * 2;
            float d0 = 0.f, d1 = 0.f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                const float* rp = &sRed[w * 128];
                d0 += rp[gid * 8 + t4 * 2];
                d1 += rp[gid * 8 + t4 * 2 + 1];
            }
            __nv_bfloat162 pv = __ldg(reinterpret_cast<const __nv_bfloat162*>(
                ple_in + static_cast<int64_t>(gm) * ple_s0 + col0));
            __nv_bfloat162 ov = make_bfloat162(
                gelu_tanh_mul_bf16(d0, pv.x),
                gelu_tanh_mul_bf16(d1, pv.y));
            *reinterpret_cast<__nv_bfloat162*>(gated_out + static_cast<int64_t>(gm) * 256 + col0) = ov;
        }
    } else {
        if (tid < m_rows * 8) {
            const int ml = tid >> 3;
            const int nl = tid & 7;
            const int gm = m_base + ml;
            float sum = 0.f;
            #pragma unroll
            for (int w = 0; w < STEP2_WARPS; ++w) {
                sum += sRed[w * 128 + tid];
            }
            __nv_bfloat16 pv = __ldg(&ple_in[static_cast<int64_t>(gm) * ple_s0 + n_base + nl]);
            gated_out[static_cast<int64_t>(gm) * 256 + n_base + nl] = gelu_tanh_mul_bf16(sum, pv);
        }
    }
}

// ----------------------------------------------------------------------------
// 4. ple_post_norm_scalar[_next_norm]: 3/4 -> 1 kernel
// ----------------------------------------------------------------------------
template <bool HAS_NEXT_NORM>
__global__ __launch_bounds__(192)
void gemma4_e2b_ple_post_norm_scalar_next_norm_kernel(
    __nv_bfloat16* __restrict__ next_res_out,
    __nv_bfloat16* __restrict__ next_norm_out,
    const __nv_bfloat16* __restrict__ ple_proj,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ post_ple_w,
    const void* __restrict__ layer_scalar_ptr,
    bool layer_scalar_bf16,
    const __nv_bfloat16* __restrict__ next_norm_w,
    int64_t ple_stride0,
    int64_t res_stride0,
    float eps)
{
    __shared__ float s_warp0[6];
    __shared__ float s_warp1[6];
    const int m = blockIdx.x;
    const int tid = threadIdx.x;

    const float scalar_f = read_scalar_f32(layer_scalar_ptr, layer_scalar_bf16);
    const uint4 p_u4 = reinterpret_cast<const uint4*>(ple_proj + static_cast<int64_t>(m) * ple_stride0)[tid];
    const uint4 r_u4 = reinterpret_cast<const uint4*>(residual + static_cast<int64_t>(m) * res_stride0)[tid];
    const uint4 w1_u4 = __ldg(&reinterpret_cast<const uint4*>(post_ple_w)[tid]);

    const bf162x4* pv = reinterpret_cast<const bf162x4*>(&p_u4);
    const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_u4);
    const bf162x4* w1v = reinterpret_cast<const bf162x4*>(&w1_u4);

    float2 pf[4];
    float sq1 = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(pv->v[j]);
        pf[j] = f;
        sq1 = fmaf(f.x, f.x, sq1);
        sq1 = fmaf(f.y, f.y, sq1);
    }
    const float tot1 = block_sum_192_single_barrier(sq1, s_warp0);
    const float inv1 = rsqrtf(tot1 * (1.0f / 1536.0f) + eps);

    uint4 nr_u4;
    bf162x4* nrv = reinterpret_cast<bf162x4*>(&nr_u4);
    float2 nrf[4];
    float sq2 = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 w1f = __bfloat1622float2(w1v->v[j]);
        float2 rf = __bfloat1622float2(rv->v[j]);
        float2 nw1 = __bfloat1622float2(__floats2bfloat162_rn(pf[j].x * inv1 * w1f.x, pf[j].y * inv1 * w1f.y));
        float2 h = __bfloat1622float2(__floats2bfloat162_rn(rf.x + nw1.x, rf.y + nw1.y));
        __nv_bfloat162 hsc_b = __floats2bfloat162_rn(h.x * scalar_f, h.y * scalar_f);
        nrv->v[j] = hsc_b;
        if constexpr (HAS_NEXT_NORM) {
            float2 hsc_f = __bfloat1622float2(hsc_b);
            nrf[j] = hsc_f;
            sq2 = fmaf(hsc_f.x, hsc_f.x, sq2);
            sq2 = fmaf(hsc_f.y, hsc_f.y, sq2);
        }
    }
    reinterpret_cast<uint4*>(next_res_out + static_cast<int64_t>(m) * 1536)[tid] = nr_u4;

    if constexpr (!HAS_NEXT_NORM) {
        return;
    }

    const uint4 w2_u4 = __ldg(&reinterpret_cast<const uint4*>(next_norm_w)[tid]);
    const bf162x4* w2v = reinterpret_cast<const bf162x4*>(&w2_u4);
    const float tot2 = block_sum_192_single_barrier(sq2, s_warp1);
    const float inv2 = rsqrtf(tot2 * (1.0f / 1536.0f) + eps);

    uint4 nn_u4;
    bf162x4* nnv = reinterpret_cast<bf162x4*>(&nn_u4);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 w2f = __bfloat1622float2(w2v->v[j]);
        nnv->v[j] = __floats2bfloat162_rn(nrf[j].x * inv2 * w2f.x, nrf[j].y * inv2 * w2f.y);
    }
    reinterpret_cast<uint4*>(next_norm_out + static_cast<int64_t>(m) * 1536)[tid] = nn_u4;
}

template <bool HAS_NEXT_NORM>
__global__ __launch_bounds__(128, 6)
void gemma4_e2b_ple_post_norm_scalar_next_norm_warp_kernel(
    __nv_bfloat16* __restrict__ next_res_out,
    __nv_bfloat16* __restrict__ next_norm_out,
    const __nv_bfloat16* __restrict__ ple_proj,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ post_ple_w,
    const void* __restrict__ layer_scalar_ptr,
    bool layer_scalar_bf16,
    const __nv_bfloat16* __restrict__ next_norm_w,
    int M,
    int64_t ple_stride0,
    int64_t res_stride0,
    float eps)
{
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.x * (blockDim.x >> 5) + wid;
    if (m >= M) return;

    const float scalar_f = read_scalar_f32(layer_scalar_ptr, layer_scalar_bf16);
    const uint4* pp = reinterpret_cast<const uint4*>(ple_proj + static_cast<int64_t>(m) * ple_stride0);
    const uint4* rp = reinterpret_cast<const uint4*>(residual + static_cast<int64_t>(m) * res_stride0);
    const uint4* w1p = reinterpret_cast<const uint4*>(post_ple_w);

    uint4 p_u4[6];
    float sq1 = 0.f;
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        uint4 pv_u4 = __ldg(&pp[it * 32 + lane]);
        p_u4[it] = pv_u4;
        const bf162x4* pv = reinterpret_cast<const bf162x4*>(&pv_u4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 x = __bfloat1622float2(pv->v[j]);
            sq1 = fmaf(x.x, x.x, sq1);
            sq1 = fmaf(x.y, x.y, sq1);
        }
    }
    const float inv1 = rsqrtf(warp_sum(sq1) * (1.0f / 1536.0f) + eps);

    float sq2 = 0.f;
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int idx = it * 32 + lane;
        uint4 r_u4 = __ldg(&rp[idx]);
        uint4 w1_u4 = __ldg(&w1p[idx]);
        const bf162x4* pv = reinterpret_cast<const bf162x4*>(&p_u4[it]);
        const bf162x4* rv = reinterpret_cast<const bf162x4*>(&r_u4);
        const bf162x4* w1v = reinterpret_cast<const bf162x4*>(&w1_u4);
        uint4 nro;
        bf162x4* nrov = reinterpret_cast<bf162x4*>(&nro);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 pf_j = __bfloat1622float2(pv->v[j]);
            float2 w1f = __bfloat1622float2(w1v->v[j]);
            float2 rvf = __bfloat1622float2(rv->v[j]);
            float2 nw1 = __bfloat1622float2(__floats2bfloat162_rn(pf_j.x * inv1 * w1f.x, pf_j.y * inv1 * w1f.y));
            float2 h = __bfloat1622float2(__floats2bfloat162_rn(rvf.x + nw1.x, rvf.y + nw1.y));
            __nv_bfloat162 hsc = __floats2bfloat162_rn(h.x * scalar_f, h.y * scalar_f);
            nrov->v[j] = hsc;
            if constexpr (HAS_NEXT_NORM) {
                float2 hscf = __bfloat1622float2(hsc);
                sq2 = fmaf(hscf.x, hscf.x, sq2);
                sq2 = fmaf(hscf.y, hscf.y, sq2);
            }
        }
        if constexpr (HAS_NEXT_NORM) {
            p_u4[it] = nro;
        }
        reinterpret_cast<uint4*>(next_res_out + static_cast<int64_t>(m) * 1536)[idx] = nro;
    }

    if constexpr (!HAS_NEXT_NORM) {
        return;
    }

    const uint4* w2p = reinterpret_cast<const uint4*>(next_norm_w);
    const float inv2 = rsqrtf(warp_sum(sq2) * (1.0f / 1536.0f) + eps);
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int idx = it * 32 + lane;
        uint4 w2_u4 = __ldg(&w2p[idx]);
        const bf162x4* nrv = reinterpret_cast<const bf162x4*>(&p_u4[it]);
        const bf162x4* w2v = reinterpret_cast<const bf162x4*>(&w2_u4);
        uint4 nno;
        bf162x4* nnov = reinterpret_cast<bf162x4*>(&nno);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 nrf_j = __bfloat1622float2(nrv->v[j]);
            float2 w2f = __bfloat1622float2(w2v->v[j]);
            nnov->v[j] = __floats2bfloat162_rn(nrf_j.x * inv2 * w2f.x, nrf_j.y * inv2 * w2f.y);
        }
        reinterpret_cast<uint4*>(next_norm_out + static_cast<int64_t>(m) * 1536)[idx] = nno;
    }
}

// ----------------------------------------------------------------------------
// 5a. gelu_tanh_mul for M >= 401 CG & M > 1536 EG after cuBLAS linear
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint4 gelu_tanh_mul_bf16x4_u4(uint4 g_u4, uint4 p_u4) {
    constexpr float BETA = 0.7978845608028654f;
    constexpr float KAPPA = 0.044715f;
    const bf162x4* gv = reinterpret_cast<const bf162x4*>(&g_u4);
    const bf162x4* pv = reinterpret_cast<const bf162x4*>(&p_u4);
    uint4 o_u4;
    bf162x4* ov = reinterpret_cast<bf162x4*>(&o_u4);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 gf = __bfloat1622float2(gv->v[j]);
        const float2 pf = __bfloat1622float2(pv->v[j]);
        const float ix = BETA * fmaf(KAPPA * gf.x * gf.x, gf.x, gf.x);
        const float iy = BETA * fmaf(KAPPA * gf.y * gf.y, gf.y, gf.y);
        const __nv_bfloat162 gb2 = __floats2bfloat162_rn(
            0.5f * gf.x * (1.0f + tanhf(ix)),
            0.5f * gf.y * (1.0f + tanhf(iy)));
        const float2 gf2 = __bfloat1622float2(gb2);
        ov->v[j] = __floats2bfloat162_rn(gf2.x * pf.x, gf2.y * pf.y);
    }
    return o_u4;
}

__global__ __launch_bounds__(256)
void gemma4_e2b_gelu_tanh_mul_inplace_vec_kernel(
    __nv_bfloat16* __restrict__ inout,
    const __nv_bfloat16* __restrict__ ple_in,
    int64_t total_u4)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (idx >= total_u4) return;
    const uint4 g_u4 = reinterpret_cast<const uint4*>(inout)[idx];
    const uint4 p_u4 = __ldg(&reinterpret_cast<const uint4*>(ple_in)[idx]);
    reinterpret_cast<uint4*>(inout)[idx] = gelu_tanh_mul_bf16x4_u4(g_u4, p_u4);
}

__global__ __launch_bounds__(256)
void gemma4_e2b_gelu_tanh_mul_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t gate_stride0,
    int64_t ple_stride0)
{
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = blockIdx.x * 8 + wid;
    if (m >= M) return;

    const uint4 g_u4 = __ldg(&reinterpret_cast<const uint4*>(gate + static_cast<int64_t>(m) * gate_stride0)[lane]);
    const uint4 p_u4 = __ldg(&reinterpret_cast<const uint4*>(ple_in + static_cast<int64_t>(m) * ple_stride0)[lane]);
    reinterpret_cast<uint4*>(out + static_cast<int64_t>(m) * 256)[lane] = gelu_tanh_mul_bf16x4_u4(g_u4, p_u4);
}

template <bool HAS_PLE_IN>
__global__ __launch_bounds__(32)
void gemma4_e2b_project_ple_1warp_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ proj,
    const __nv_bfloat16* __restrict__ ple_in,
    const __nv_bfloat16* __restrict__ norm_w,
    const void* __restrict__ proj_scale_ptr,
    bool proj_scale_bf16,
    const void* __restrict__ in_scale_ptr,
    bool in_scale_bf16,
    int M,
    int64_t proj_m_stride,
    int64_t ple_m_stride,
    float eps)
{
    constexpr int L = 35;
    constexpr int D_CHUNKS = 32;
    const int lane = threadIdx.x;
    const int64_t row_idx = blockIdx.x;
    const int64_t m = row_idx / L;
    const int64_t l = row_idx - m * L;

    const float p_scale = read_scalar_f32(proj_scale_ptr, proj_scale_bf16);
    const uint4 p_u4 = reinterpret_cast<const uint4*>(proj + m * proj_m_stride + l * 256)[lane];
    const uint4 w_u4 = __ldg(&reinterpret_cast<const uint4*>(norm_w)[lane]);
    const bf16x8* p_vec = reinterpret_cast<const bf16x8*>(&p_u4);
    const bf16x8* w_vec = reinterpret_cast<const bf16x8*>(&w_u4);

    float p0[8];
    float sq = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v = __bfloat162float(__float2bfloat16_rn(__bfloat162float(p_vec->v[j]) * p_scale));
        p0[j] = v;
        sq = fmaf(v, v, sq);
    }
    sq = warp_sum(sq);
    const float inv_rms = rsqrtf(sq * (1.0f / 256.0f) + eps);

    bf16x8 out_vec;
    if constexpr (HAS_PLE_IN) {
        const float i_scale = read_scalar_f32(in_scale_ptr, in_scale_bf16);
        const uint4 ple_u4 = reinterpret_cast<const uint4*>(ple_in + m * ple_m_stride + l * 256)[lane];
        const bf16x8* ple_vec = reinterpret_cast<const bf16x8*>(&ple_u4);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float norm_b = __bfloat162float(__float2bfloat16_rn(p0[j] * inv_rms * __bfloat162float(w_vec->v[j])));
            float sum_b = __bfloat162float(__float2bfloat16_rn(norm_b + __bfloat162float(ple_vec->v[j])));
            out_vec.v[j] = __float2bfloat16_rn(sum_b * i_scale);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_vec.v[j] = __float2bfloat16_rn(p0[j] * inv_rms * __bfloat162float(w_vec->v[j]));
        }
    }
    const int64_t out_row = l * static_cast<int64_t>(M) + m;
    reinterpret_cast<uint4*>(out)[out_row * D_CHUNKS + lane] = *reinterpret_cast<const uint4*>(&out_vec);
}

// 5a. ple_gate_gelu_mul high-occupancy Split-K / 1col / 2col / MMA32 kernels (K=1536, N=256)
template <int MAX_M>
__global__ __launch_bounds__(192, 4)
void gemma4_e2b_ple_gate_1col_192_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t x_stride0,
    int64_t ple_stride0)
{
    constexpr int K_CHUNKS = 192;
    constexpr int N = 256;
    __shared__ float s_warp[MAX_M][6];

    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    const uint4 w_reg = __ldg(&reinterpret_cast<const uint4*>(w_gate)[static_cast<int64_t>(n) * K_CHUNKS + tid]);

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        float acc = 0.f;
        if (m < M) {
            const uint4 x_reg = __ldg(&reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * x_stride0)[tid]);
            acc = dot8_bf16(w_reg, x_reg);
        }
        float ws = warp_sum(acc);
        if (lane == 0) s_warp[m][wid] = ws;
    }
    __syncthreads();

    if (tid < M) {
        const int m = tid;
        float sum = 0.f;
        #pragma unroll
        for (int w = 0; w < 6; ++w) sum += s_warp[m][w];
        __nv_bfloat16 ple_v = ple_in[static_cast<int64_t>(m) * ple_stride0 + n];
        out[static_cast<int64_t>(m) * N + n] = gelu_tanh_mul_bf16(sum, ple_v);
    }
}

template <int MAX_M>
__global__ __launch_bounds__(192, 4)
void gemma4_e2b_ple_gate_2col_192_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t x_stride0,
    int64_t ple_stride0)
{
    constexpr int K_CHUNKS = 192;
    constexpr int N = 256;
    __shared__ float s_warp[2][MAX_M][6];

    const int n0 = blockIdx.x * 2;
    const int n1 = n0 + 1;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    const uint4 w0 = __ldg(&reinterpret_cast<const uint4*>(w_gate)[static_cast<int64_t>(n0) * K_CHUNKS + tid]);
    const uint4 w1 = __ldg(&reinterpret_cast<const uint4*>(w_gate)[static_cast<int64_t>(n1) * K_CHUNKS + tid]);

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        float a0 = 0.f, a1 = 0.f;
        if (m < M) {
            const uint4 x_reg = __ldg(&reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * x_stride0)[tid]);
            a0 = dot8_bf16(w0, x_reg);
            a1 = dot8_bf16(w1, x_reg);
        }
        float ws0 = warp_sum(a0);
        float ws1 = warp_sum(a1);
        if (lane == 0) {
            s_warp[0][m][wid] = ws0;
            s_warp[1][m][wid] = ws1;
        }
    }
    __syncthreads();

    if (tid < 2 * M) {
        const int c = tid & 1;
        const int m = tid >> 1;
        const int n = n0 + c;
        float sum = 0.f;
        #pragma unroll
        for (int w = 0; w < 6; ++w) sum += s_warp[c][m][w];
        __nv_bfloat16 ple_v = ple_in[static_cast<int64_t>(m) * ple_stride0 + n];
        out[static_cast<int64_t>(m) * N + n] = gelu_tanh_mul_bf16(sum, ple_v);
    }
}

__global__ __launch_bounds__(64)
void gemma4_e2b_ple_gate_mma16_splitk4_kernel(
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ workspace,
    int* __restrict__ tile_done,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t x_stride0,
    int64_t ple_stride0)
{
    constexpr int K = 1536;
    constexpr int SPLIT_K = 384;
    constexpr int SMEM_STRIDE = 392;
    __shared__ __nv_bfloat16 sX[16 * SMEM_STRIDE];
    __shared__ float sWarp[16 * 8];
    __shared__ bool is_last;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int t4 = lane & 3;
    const int n_tile = blockIdx.x;
    const int split_id = blockIdx.y;
    const int n_base = n_tile * 8;
    const int k_base = split_id * SPLIT_K;

    #pragma unroll
    for (int it = 0; it < 12; ++it) {
        int idx = it * 64 + tid;
        int m = idx / 48;
        int c48 = idx - m * 48;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (m < M) {
            v = __ldg(&reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * x_stride0)[(k_base >> 3) + c48]);
        }
        *reinterpret_cast<uint4*>(sX + m * SMEM_STRIDE + c48 * 8) = v;
    }
    __syncthreads();

    const int wk_off = wid * 192;
    const __nv_bfloat16* w_row = w_gate + static_cast<int64_t>(n_base + gid) * K + k_base + wk_off;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    #pragma unroll
    for (int k = 0; k < 192; k += 16) {
        uint32_t b0 = __ldg(reinterpret_cast<const uint32_t*>(w_row + k + t4 * 2));
        uint32_t b1 = __ldg(reinterpret_cast<const uint32_t*>(w_row + k + 8 + t4 * 2));
        uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX + gid * SMEM_STRIDE + wk_off + k + t4 * 2);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(sX + (gid + 8) * SMEM_STRIDE + wk_off + k + t4 * 2);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX + gid * SMEM_STRIDE + wk_off + k + 8 + t4 * 2);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(sX + (gid + 8) * SMEM_STRIDE + wk_off + k + 8 + t4 * 2);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    }

    if (wid == 1) {
        sWarp[gid * 8 + t4 * 2] = c0;
        sWarp[gid * 8 + t4 * 2 + 1] = c1;
        sWarp[(gid + 8) * 8 + t4 * 2] = c2;
        sWarp[(gid + 8) * 8 + t4 * 2 + 1] = c3;
    }
    __syncthreads();

    float* ws_dst = workspace + (static_cast<int64_t>(n_tile) * 4 + split_id) * 128;
    if (wid == 0) {
        ws_dst[gid * 8 + t4 * 2] = c0 + sWarp[gid * 8 + t4 * 2];
        ws_dst[gid * 8 + t4 * 2 + 1] = c1 + sWarp[gid * 8 + t4 * 2 + 1];
        ws_dst[(gid + 8) * 8 + t4 * 2] = c2 + sWarp[(gid + 8) * 8 + t4 * 2];
        ws_dst[(gid + 8) * 8 + t4 * 2 + 1] = c3 + sWarp[(gid + 8) * 8 + t4 * 2 + 1];
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        int prev = atomicAdd(&tile_done[n_tile], 1);
        is_last = (prev == 3);
        if (is_last) tile_done[n_tile] = 0;
    }
    __syncthreads();
    if (!is_last) return;

    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int e = it * 64 + tid;
        int m = e >> 3;
        int nl = e & 7;
        if (m < M) {
            const float* tbase = workspace + static_cast<int64_t>(n_tile) * (4 * 128) + e;
            float sum = tbase[0] + tbase[128] + tbase[256] + tbase[384];
            __nv_bfloat16 ple_v = ple_in[static_cast<int64_t>(m) * ple_stride0 + n_base + nl];
            out[static_cast<int64_t>(m) * 256 + n_base + nl] = gelu_tanh_mul_bf16(sum, ple_v);
        }
    }
}

// High-throughput cp.async + ldmatrix ple_gate_gelu_mul for M=17..400 & M=513..1536
template <int BN, int BM, int K_SPLITS>
__global__ __launch_bounds__(K_SPLITS * 32)
void p7_ple_gate_ca_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t x_s0,
    int64_t ple_s0)
{
    constexpr int K = 1536;
    constexpr int K_TILES_PER_WARP = 24 / K_SPLITS;
    constexpr int TN = BN / 8;
    constexpr int MT = BM / 16;
    extern __shared__ __align__(16) char dyn_smem[];
    auto sA = reinterpret_cast<uint4 (*)[2][BM * 8]>(dyn_smem);
    auto sB = reinterpret_cast<uint4 (*)[2][BN * 8]>(dyn_smem + K_SPLITS * 2 * BM * 8 * 16);
    auto sRed = reinterpret_cast<float (*)[MT][32 * TN * 4]>(
        dyn_smem + K_SPLITS * 2 * (BM + BN) * 8 * 16);

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = blockIdx.x * BN;
    const int m_base = blockIdx.y * BM;
    const int k_tile_start = wid * K_TILES_PER_WARP;

    auto issue_tile = [&](int stg, int kt_global) {
        const int k_base = kt_global * 64;
        #pragma unroll
        for (int idx = lane; idx < BM * 8; idx += 32) {
            int r = idx >> 3;
            int c = idx & 7;
            int gm = m_base + r;
            p6_cp_async_ca_16B(
                &sA[wid][stg][r * 8 + p6_swz64(r, c)],
                x + static_cast<int64_t>(gm < M ? gm : 0) * x_s0 + k_base + (c << 3),
                gm < M);
        }
        #pragma unroll
        for (int idx = lane; idx < BN * 8; idx += 32) {
            int r = idx >> 3;
            int c = idx & 7;
            p6_cp_async_cg_16B(
                &sB[wid][stg][r * 8 + p6_swz64(r, c)],
                w_gate + static_cast<int64_t>(cta_n + r) * K + k_base + (c << 3),
                true);
        }
    };
    issue_tile(0, k_tile_start);
    p6_cp_async_commit();

    float acc[MT][TN][4] = {};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col = (lane >> 4);
    const int ldm_b_row = ((lane >> 4) * 8 + (lane & 7));
    const int ldm_b_col = ((lane >> 3) & 1);
    for (int kt = 0; kt < K_TILES_PER_WARP; ++kt) {
        int cur = kt & 1;
        if (kt + 1 < K_TILES_PER_WARP) {
            issue_tile(cur ^ 1, k_tile_start + kt + 1);
            p6_cp_async_commit();
            p6_cp_async_wait<1>();
        } else {
            p6_cp_async_wait<0>();
        }
        __syncwarp();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn += 2) {
                uint32_t b_reg[4];
                p6_ldmatrix_x4(b_reg, &sB[wid][cur][(tn * 8 + ldm_b_row) * 8 + p6_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
                uint32_t b0[2] = {b_reg[0], b_reg[1]}, b1[2] = {b_reg[2], b_reg[3]};
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    uint32_t a_reg[4];
                    p6_ldmatrix_x4(a_reg, &sA[wid][cur][(mt * 16 + ldm_a_row) * 8 + p6_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
                    p6_mma_m16n8k16_bf16(acc[mt][tn], a_reg, b0);
                    p6_mma_m16n8k16_bf16(acc[mt][tn + 1], a_reg, b1);
                }
            }
        }
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int t = 0; t < TN; ++t) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sRed[wid][mt][lane * (TN * 4) + t * 4 + i] = acc[mt][t][i];
            }
        }
    }
    __syncthreads();

    for (int ot = wid; ot < MT * TN; ot += K_SPLITS) {
        const int mt = ot / TN;
        const int t = ot - mt * TN;
        const int gid = lane >> 2;
        const int t4 = lane & 3;
        const int off = t * 4;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        #pragma unroll
        for (int w = 0; w < K_SPLITS; ++w) {
            const float* rp = &sRed[w][mt][lane * (TN * 4) + off];
            d0 += rp[0]; d1 += rp[1]; d2 += rp[2]; d3 += rp[3];
        }
        int col0 = cta_n + t * 8 + t4 * 2;
        int gm0 = m_base + mt * 16 + gid;
        int gm1 = gm0 + 8;
        if (gm0 < M) {
            __nv_bfloat162 pv = __ldg(reinterpret_cast<const __nv_bfloat162*>(ple_in + static_cast<int64_t>(gm0) * ple_s0 + col0));
            *reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(gm0) * 256 + col0) = make_bfloat162(
                gelu_tanh_mul_bf16(d0, pv.x), gelu_tanh_mul_bf16(d1, pv.y));
        }
        if (gm1 < M) {
            __nv_bfloat162 pv = __ldg(reinterpret_cast<const __nv_bfloat162*>(ple_in + static_cast<int64_t>(gm1) * ple_s0 + col0));
            *reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(gm1) * 256 + col0) = make_bfloat162(
                gelu_tanh_mul_bf16(d2, pv.x), gelu_tanh_mul_bf16(d3, pv.y));
        }
    }
}

// ----------------------------------------------------------------------------
// 5b. ple_gate_gelu_mul for decode M <= 4 (K=1536, N=256)
// ----------------------------------------------------------------------------
template <int M>
__global__ __launch_bounds__(128, 4)
void gemma4_e2b_ple_gate_gelu_mul_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int64_t x_stride0,
    int64_t ple_stride0)
{
    constexpr int K_CHUNKS = 192;
    constexpr int N = 256;
    __shared__ uint4 sX[M * K_CHUNKS];
    __shared__ float sPart[2][M];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    for (int idx = tid; idx < M * K_CHUNKS; idx += 128) {
        int m = idx / K_CHUNKS;
        int kc = idx - m * K_CHUNKS;
        sX[idx] = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * x_stride0)[kc];
    }
    __syncthreads();

    const int row_local = wid >> 1;
    const int split_id = wid & 1;
    const int n = blockIdx.x * 2 + row_local;
    if (n >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(w_gate) + static_cast<int64_t>(n) * K_CHUNKS;
    const int k_base = split_id * 96;

    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;

    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int c = k_base + it * 32 + lane;
        uint4 w_reg = __ldg(&w_row[c]);
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[m] += dot8_bf16(w_reg, sX[m * K_CHUNKS + c]);
        }
    }

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = warp_sum(acc[m]);
        if (lane == 0 && split_id == 1) {
            sPart[row_local][m] = v;
        }
        acc[m] = v;
    }
    __syncthreads();

    if (split_id == 0 && lane == 0) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float g = acc[m] + sPart[row_local][m];
            __nv_bfloat16 ple_val = ple_in[static_cast<int64_t>(m) * ple_stride0 + n];
            out[static_cast<int64_t>(m) * N + n] = gelu_tanh_mul_bf16(g, ple_val);
        }
    }
}

// ----------------------------------------------------------------------------
// 6. ple_proj_gemv for decode M <= 4 (K=256, N=1536)
// ----------------------------------------------------------------------------
template <int M>
__global__ __launch_bounds__(128)
void gemma4_e2b_ple_proj_gemv_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int64_t x_stride0)
{
    constexpr int K_CHUNKS = 32;
    constexpr int N = 1536;
    __shared__ uint4 sX[M * K_CHUNKS];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    for (int idx = tid; idx < M * K_CHUNKS; idx += 128) {
        int m = idx / K_CHUNKS;
        int kc = idx - m * K_CHUNKS;
        sX[idx] = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * x_stride0)[kc];
    }
    __syncthreads();

    const int n = blockIdx.x * 4 + wid;
    if (n >= N) return;

    const uint4 w_reg = __ldg(&reinterpret_cast<const uint4*>(w_proj)[static_cast<int64_t>(n) * K_CHUNKS + lane]);
    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        acc[m] = dot8_bf16(w_reg, sX[m * K_CHUNKS + lane]);
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float s = warp_sum(acc[m]);
        if (lane == 0) out[static_cast<int64_t>(m) * N + n] = __float2bfloat16_rn(s);
    }
}

template <int M>
__global__ __launch_bounds__(128)
void gemma4_e2b_ple_proj_nosmem_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int64_t x_stride0)
{
    constexpr int K_CHUNKS = 32;
    constexpr int N = 1536;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + wid;
    if (n >= N) return;

    const uint4 w_reg = __ldg(&reinterpret_cast<const uint4*>(w_proj)[static_cast<int64_t>(n) * K_CHUNKS + lane]);
    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        uint4 x_reg = __ldg(&reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * x_stride0)[lane]);
        acc[m] = dot8_bf16(w_reg, x_reg);
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float s = warp_sum(acc[m]);
        if (lane == 0) out[static_cast<int64_t>(m) * N + n] = __float2bfloat16_rn(s);
    }
}

// 4-col no-SMEM ple_proj for M=1 (96 CTAs x 128t = 384 warps * 4 cols = 1536)
template <int M, int COLS>
__global__ __launch_bounds__(128)
void p7_ple_proj_mcol_nosmem_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int64_t x_s0)
{
    constexpr int K_CHUNKS = 32;
    constexpr int N = 1536;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n_base = (blockIdx.x * 4 + wid) * COLS;
    if (n_base >= N) return;

    uint4 w_reg[COLS];
    #pragma unroll
    for (int c = 0; c < COLS; ++c) {
        w_reg[c] = __ldg(&reinterpret_cast<const uint4*>(w_proj)[static_cast<int64_t>(n_base + c) * K_CHUNKS + lane]);
    }
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        uint4 x_reg = __ldg(&reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * x_s0)[lane]);
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            float s = warp_sum(dot8_bf16(w_reg[c], x_reg));
            if (lane == 0) out[static_cast<int64_t>(m) * N + n_base + c] = __float2bfloat16_rn(s);
        }
    }
}

// Multi-block m16_ca ple_proj for M = 33..72 (grid(96, ceil(M/16)) x 128t)
__global__ __launch_bounds__(128)
void p7_ple_proj_m16_tiled_ca_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int M,
    int64_t x_s0)
{
    constexpr int K = 256;
    constexpr int N = 1536;
    constexpr int TILES_N = 2;
    constexpr int BN = 16;
    __shared__ uint4 sA[4][16 * 8];
    __shared__ uint4 sB[4][BN * 8];
    __shared__ float sRed[4][32 * TILES_N * 4];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = blockIdx.x * BN;
    const int m_base = blockIdx.y * 16;

    for (int idx = tid; idx < 16 * 32; idx += 128) {
        int ml = idx >> 5;
        int c32 = idx & 31;
        int chunk = c32 >> 3;
        int c8 = c32 & 7;
        int gm = m_base + ml;
        p6_cp_async_ca_16B(
            &sA[chunk][ml * 8 + p6_swz64(ml, c8)],
            x + static_cast<int64_t>(gm < M ? gm : 0) * x_s0 + c32 * 8,
            gm < M);
    }
    {
        const int k_base = wid * 64;
        #pragma unroll
        for (int idx = lane; idx < BN * 8; idx += 32) {
            int r = idx >> 3;
            int c = idx & 7;
            p6_cp_async_cg_16B(
                &sB[wid][r * 8 + p6_swz64(r, c)],
                w_proj + static_cast<int64_t>(cta_n + r) * K + k_base + (c << 3),
                true);
        }
    }
    p6_cp_async_commit();
    p6_cp_async_wait<0>();
    __syncthreads();

    float acc0[4] = {0.f, 0.f, 0.f, 0.f};
    float acc1[4] = {0.f, 0.f, 0.f, 0.f};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col = (lane >> 4);
    const int ldm_b_row = ((lane >> 4) * 8 + (lane & 7));
    const int ldm_b_col = ((lane >> 3) & 1);

    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        uint32_t a_reg[4], b_reg[4];
        p6_ldmatrix_x4(a_reg, &sA[wid][ldm_a_row * 8 + p6_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
        p6_ldmatrix_x4(b_reg, &sB[wid][ldm_b_row * 8 + p6_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
        uint32_t b0[2] = {b_reg[0], b_reg[1]};
        uint32_t b1[2] = {b_reg[2], b_reg[3]};
        p6_mma_m16n8k16_bf16(acc0, a_reg, b0);
        p6_mma_m16n8k16_bf16(acc1, a_reg, b1);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sRed[wid][lane * 8 + i] = acc0[i];
        sRed[wid][lane * 8 + 4 + i] = acc1[i];
    }
    __syncthreads();

    if (wid < 2) {
        const int gid = lane >> 2;
        const int t4 = lane & 3;
        const int off = wid * 4;
        float d0 = sRed[0][lane * 8 + off + 0] + sRed[1][lane * 8 + off + 0] + sRed[2][lane * 8 + off + 0] + sRed[3][lane * 8 + off + 0];
        float d1 = sRed[0][lane * 8 + off + 1] + sRed[1][lane * 8 + off + 1] + sRed[2][lane * 8 + off + 1] + sRed[3][lane * 8 + off + 1];
        float d2 = sRed[0][lane * 8 + off + 2] + sRed[1][lane * 8 + off + 2] + sRed[2][lane * 8 + off + 2] + sRed[3][lane * 8 + off + 2];
        float d3 = sRed[0][lane * 8 + off + 3] + sRed[1][lane * 8 + off + 3] + sRed[2][lane * 8 + off + 3] + sRed[3][lane * 8 + off + 3];
        int col0 = cta_n + wid * 8 + t4 * 2;
        int gm0 = m_base + gid;
        int gm1 = gm0 + 8;
        if (gm0 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gm0) * N + col0) = pack_bf16x2_raw(d0, d1);
        if (gm1 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gm1) * N + col0) = pack_bf16x2_raw(d2, d3);
    }
}

// High-throughput cp.async + ldmatrix ple_proj for M = 5..16 (K=256, N=1536)
__global__ __launch_bounds__(128)
void p6_ple_proj_m16_ca_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int M,
    int64_t x_s0)
{
    constexpr int K = 256;
    constexpr int N = 1536;
    constexpr int TILES_N = 2;
    constexpr int BN = 16;
    __shared__ uint4 sA[4][16 * 8];
    __shared__ uint4 sB[4][BN * 8];
    __shared__ float sRed[4][32 * TILES_N * 4];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = blockIdx.x * BN;

    for (int idx = tid; idx < 16 * 32; idx += 128) {
        int m = idx >> 5;
        int c32 = idx & 31;
        int chunk = c32 >> 3;
        int c8 = c32 & 7;
        p6_cp_async_ca_16B(
            &sA[chunk][m * 8 + p6_swz64(m, c8)],
            x + static_cast<int64_t>(m < M ? m : 0) * x_s0 + c32 * 8,
            m < M);
    }
    {
        const int k_base = wid * 64;
        #pragma unroll
        for (int idx = lane; idx < BN * 8; idx += 32) {
            int r = idx >> 3;
            int c = idx & 7;
            p6_cp_async_cg_16B(
                &sB[wid][r * 8 + p6_swz64(r, c)],
                w_proj + static_cast<int64_t>(cta_n + r) * K + k_base + (c << 3),
                true);
        }
    }
    p6_cp_async_commit();
    p6_cp_async_wait<0>();
    __syncthreads();

    float acc0[4] = {0.f, 0.f, 0.f, 0.f};
    float acc1[4] = {0.f, 0.f, 0.f, 0.f};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col = (lane >> 4);
    const int ldm_b_row = ((lane >> 4) * 8 + (lane & 7));
    const int ldm_b_col = ((lane >> 3) & 1);

    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        uint32_t a_reg[4], b_reg[4];
        p6_ldmatrix_x4(a_reg, &sA[wid][ldm_a_row * 8 + p6_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
        p6_ldmatrix_x4(b_reg, &sB[wid][ldm_b_row * 8 + p6_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
        uint32_t b0[2] = {b_reg[0], b_reg[1]};
        uint32_t b1[2] = {b_reg[2], b_reg[3]};
        p6_mma_m16n8k16_bf16(acc0, a_reg, b0);
        p6_mma_m16n8k16_bf16(acc1, a_reg, b1);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sRed[wid][lane * 8 + i] = acc0[i];
        sRed[wid][lane * 8 + 4 + i] = acc1[i];
    }
    __syncthreads();

    if (wid < 2) {
        const int gid = lane >> 2;
        const int t4 = lane & 3;
        const int off = wid * 4;
        float d0 = sRed[0][lane * 8 + off + 0] + sRed[1][lane * 8 + off + 0] + sRed[2][lane * 8 + off + 0] + sRed[3][lane * 8 + off + 0];
        float d1 = sRed[0][lane * 8 + off + 1] + sRed[1][lane * 8 + off + 1] + sRed[2][lane * 8 + off + 1] + sRed[3][lane * 8 + off + 1];
        float d2 = sRed[0][lane * 8 + off + 2] + sRed[1][lane * 8 + off + 2] + sRed[2][lane * 8 + off + 2] + sRed[3][lane * 8 + off + 2];
        float d3 = sRed[0][lane * 8 + off + 3] + sRed[1][lane * 8 + off + 3] + sRed[2][lane * 8 + off + 3] + sRed[3][lane * 8 + off + 3];
        int col0 = cta_n + wid * 8 + t4 * 2;
        if (gid < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid) * N + col0) = pack_bf16x2_raw(d0, d1);
        if (gid + 8 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid + 8) * N + col0) = pack_bf16x2_raw(d2, d3);
    }
}

// High-throughput cp.async + ldmatrix ple_proj for M = 17..64 (K=256, N=1536)
__global__ __launch_bounds__(128)
void p6_ple_proj_m32_ca_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int M,
    int64_t x_s0)
{
    constexpr int K = 256;
    constexpr int N = 1536;
    constexpr int TILES_N = 2;
    constexpr int BN = 16;
    __shared__ uint4 sA[4][32 * 8];
    __shared__ uint4 sB[4][BN * 8];
    __shared__ float sRed[4][2][32 * TILES_N * 4];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = blockIdx.x * BN;
    const int m_base = blockIdx.y * 32;

    for (int idx = tid; idx < 32 * 32; idx += 128) {
        int ml = idx >> 5;
        int c32 = idx & 31;
        int chunk = c32 >> 3;
        int c8 = c32 & 7;
        int gm = m_base + ml;
        p6_cp_async_ca_16B(
            &sA[chunk][ml * 8 + p6_swz64(ml, c8)],
            x + static_cast<int64_t>(gm < M ? gm : 0) * x_s0 + c32 * 8,
            gm < M);
    }
    {
        const int k_base = wid * 64;
        #pragma unroll
        for (int idx = lane; idx < BN * 8; idx += 32) {
            int r = idx >> 3;
            int c = idx & 7;
            p6_cp_async_cg_16B(
                &sB[wid][r * 8 + p6_swz64(r, c)],
                w_proj + static_cast<int64_t>(cta_n + r) * K + k_base + (c << 3),
                true);
        }
    }
    p6_cp_async_commit();
    p6_cp_async_wait<0>();
    __syncthreads();

    float acc[2][2][4] = {};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col = (lane >> 4);
    const int ldm_b_row = ((lane >> 4) * 8 + (lane & 7));
    const int ldm_b_col = ((lane >> 3) & 1);

    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        uint32_t a0[4], a1[4], b_reg[4];
        p6_ldmatrix_x4(a0, &sA[wid][ldm_a_row * 8 + p6_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
        p6_ldmatrix_x4(a1, &sA[wid][(16 + ldm_a_row) * 8 + p6_swz64(16 + ldm_a_row, ks * 2 + ldm_a_col)]);
        p6_ldmatrix_x4(b_reg, &sB[wid][ldm_b_row * 8 + p6_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
        uint32_t b0[2] = {b_reg[0], b_reg[1]}, b1[2] = {b_reg[2], b_reg[3]};
        p6_mma_m16n8k16_bf16(acc[0][0], a0, b0);
        p6_mma_m16n8k16_bf16(acc[0][1], a0, b1);
        p6_mma_m16n8k16_bf16(acc[1][0], a1, b0);
        p6_mma_m16n8k16_bf16(acc[1][1], a1, b1);
    }

    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sRed[wid][mt][lane * 8 + i] = acc[mt][0][i];
            sRed[wid][mt][lane * 8 + 4 + i] = acc[mt][1][i];
        }
    }
    __syncthreads();

    const int mt = wid >> 1;
    const int t = wid & 1;
    const int gid = lane >> 2;
    const int t4 = lane & 3;
    const int off = t * 4;
    float d0 = sRed[0][mt][lane * 8 + off + 0] + sRed[1][mt][lane * 8 + off + 0] + sRed[2][mt][lane * 8 + off + 0] + sRed[3][mt][lane * 8 + off + 0];
    float d1 = sRed[0][mt][lane * 8 + off + 1] + sRed[1][mt][lane * 8 + off + 1] + sRed[2][mt][lane * 8 + off + 1] + sRed[3][mt][lane * 8 + off + 1];
    float d2 = sRed[0][mt][lane * 8 + off + 2] + sRed[1][mt][lane * 8 + off + 2] + sRed[2][mt][lane * 8 + off + 2] + sRed[3][mt][lane * 8 + off + 2];
    float d3 = sRed[0][mt][lane * 8 + off + 3] + sRed[1][mt][lane * 8 + off + 3] + sRed[2][mt][lane * 8 + off + 3] + sRed[3][mt][lane * 8 + off + 3];
    int col0 = cta_n + t * 8 + t4 * 2;
    int gm0 = m_base + mt * 16 + gid;
    int gm1 = gm0 + 8;
    if (gm0 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gm0) * N + col0) = pack_bf16x2_raw(d0, d1);
    if (gm1 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gm1) * N + col0) = pack_bf16x2_raw(d2, d3);
}

struct SplitKWorkspaceEntry {
    int dev;
    uintptr_t stream;
    torch::Tensor ws;
    torch::Tensor tile_done;
};

static inline std::pair<float*, int*> get_splitk_workspace(
    const torch::Tensor& ref,
    cudaStream_t stream)
{
    static std::mutex mu;
    static std::vector<SplitKWorkspaceEntry> entries;
    const int dev = ref.get_device();
    const uintptr_t s_id = reinterpret_cast<uintptr_t>(stream);
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& e : entries) {
        if (e.dev == dev && e.stream == s_id) {
            return {e.ws.data_ptr<float>(), e.tile_done.data_ptr<int>()};
        }
    }
    auto opts_f32 = torch::TensorOptions().device(ref.device()).dtype(torch::kFloat32);
    auto opts_i32 = torch::TensorOptions().device(ref.device()).dtype(torch::kInt32);
    SplitKWorkspaceEntry created{
        dev,
        s_id,
        torch::empty({32, 4, 16, 8}, opts_f32),
        torch::zeros({32}, opts_i32),
    };
    float* ws_p = created.ws.data_ptr<float>();
    int* td_p = created.tile_done.data_ptr<int>();
    entries.push_back(std::move(created));
    return {ws_p, td_p};
}

template <int M, bool USE_CS = false>
__global__ __launch_bounds__(256)
void gemma4_e2b_model_proj_gemv_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int64_t x_stride0)
{
    constexpr int K_CHUNKS = 192;
    constexpr int N = 8960;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + wid;
    if (n >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(w_proj) + static_cast<int64_t>(n) * K_CHUNKS;
    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;

    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        int c = it * 32 + lane;
        uint4 w_reg = USE_CS ? ld_cs_u4(&w_row[c]) : __ldg(&w_row[c]);
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            uint4 x_reg = __ldg(&reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * x_stride0)[c]);
            acc[m] += dot8_bf16(w_reg, x_reg);
        }
    }

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float s = warp_sum(acc[m]);
        if (lane == 0) out[static_cast<int64_t>(m) * N + n] = __float2bfloat16_rn(s);
    }
}

template <int M, int COLS, bool USE_CS = false>
__global__ __launch_bounds__(128)
void p9_model_proj_mcol_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int64_t x_stride0)
{
    constexpr int K_CHUNKS = 192;
    constexpr int N = 8960;
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n_base = (blockIdx.x * 4 + wid) * COLS;
    if (n_base >= N) return;

    float acc[COLS][M] = {};
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        int c = it * 32 + lane;
        uint4 x_reg[M];
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            x_reg[m] = __ldg(&reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * x_stride0)[c]);
        }
        #pragma unroll
        for (int col = 0; col < COLS; ++col) {
            const uint4* wp = reinterpret_cast<const uint4*>(w_proj) + static_cast<int64_t>(n_base + col) * K_CHUNKS + c;
            uint4 w_reg = USE_CS ? ld_cs_u4(wp) : __ldg(wp);
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[col][m] += dot8_bf16(w_reg, x_reg[m]);
            }
        }
    }
    #pragma unroll
    for (int col = 0; col < COLS; ++col) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float s = warp_sum(acc[col][m]);
            if (lane == 0) out[static_cast<int64_t>(m) * N + n_base + col] = __float2bfloat16_rn(s);
        }
    }
}

template <int K_SPLITS, int TILES_N>
__global__ __launch_bounds__(K_SPLITS * 32)
void p6_model_proj_ca_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int M,
    int64_t x_s0)
{
    constexpr int K = 1536;
    constexpr int N = 8960;
    constexpr int BN = TILES_N * 8;
    constexpr int K_TILES_TOTAL = K / 64;  // 24
    constexpr int K_TILES_PER_WARP = K_TILES_TOTAL / K_SPLITS;
    static_assert(K_TILES_TOTAL % K_SPLITS == 0, "24 must divide K_SPLITS");

    __shared__ uint4 sA[K_SPLITS][2][16 * 8];
    __shared__ uint4 sB[K_SPLITS][2][BN * 8];
    __shared__ float sRed[K_SPLITS][32 * TILES_N * 4];

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = blockIdx.x * BN;
    const int k_tile_start = wid * K_TILES_PER_WARP;

    auto issue_tile = [&](int stg, int kt_global) {
        const int k_base = kt_global * 64;
        #pragma unroll
        for (int idx = lane; idx < 16 * 8; idx += 32) {
            int r = idx >> 3;
            int c = idx & 7;
            p6_cp_async_ca_16B(
                &sA[wid][stg][r * 8 + p6_swz64(r, c)],
                x + static_cast<int64_t>(r < M ? r : 0) * x_s0 + k_base + (c << 3),
                r < M);
        }
        #pragma unroll
        for (int idx = lane; idx < BN * 8; idx += 32) {
            int r = idx >> 3;
            int c = idx & 7;
            int gn = cta_n + r;
            p6_cp_async_cg_16B(
                &sB[wid][stg][r * 8 + p6_swz64(r, c)],
                w + static_cast<int64_t>(gn < N ? gn : 0) * K + k_base + (c << 3),
                gn < N);
        }
    };
    issue_tile(0, k_tile_start);
    p6_cp_async_commit();

    float acc[TILES_N][4] = {};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ldm_a_col = (lane >> 4);
    const int ldm_b_row = (TILES_N == 2) ? ((lane >> 4) * 8 + (lane & 7)) : (lane & 7);
    const int ldm_b_col = ((lane >> 3) & 1);

    for (int kt = 0; kt < K_TILES_PER_WARP; ++kt) {
        int cur = kt & 1;
        if (kt + 1 < K_TILES_PER_WARP) {
            issue_tile(cur ^ 1, k_tile_start + kt + 1);
            p6_cp_async_commit();
            p6_cp_async_wait<1>();
        } else {
            p6_cp_async_wait<0>();
        }
        __syncwarp();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t a_reg[4];
            p6_ldmatrix_x4(a_reg, &sA[wid][cur][ldm_a_row * 8 + p6_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
            if constexpr (TILES_N == 2) {
                uint32_t b_reg[4];
                p6_ldmatrix_x4(b_reg, &sB[wid][cur][ldm_b_row * 8 + p6_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
                uint32_t b0[2] = {b_reg[0], b_reg[1]}, b1[2] = {b_reg[2], b_reg[3]};
                p6_mma_m16n8k16_bf16(acc[0], a_reg, b0);
                p6_mma_m16n8k16_bf16(acc[1], a_reg, b1);
            } else {
                uint32_t b_reg[2];
                p6_ldmatrix_x2(b_reg, &sB[wid][cur][ldm_b_row * 8 + p6_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
                p6_mma_m16n8k16_bf16(acc[0], a_reg, b_reg);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < TILES_N; ++t) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sRed[wid][lane * (TILES_N * 4) + t * 4 + i] = acc[t][i];
        }
    }
    __syncthreads();

    if (wid < TILES_N) {
        const int t = wid;
        const int gid = lane >> 2;
        const int t4 = lane & 3;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        #pragma unroll
        for (int w = 0; w < K_SPLITS; ++w) {
            const float* rp = &sRed[w][lane * (TILES_N * 4) + t * 4];
            d0 += rp[0]; d1 += rp[1]; d2 += rp[2]; d3 += rp[3];
        }
        int col0 = cta_n + t * 8 + t4 * 2;
        if (col0 + 1 < N) {
            if (gid < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid) * N + col0) = pack_bf16x2_raw(d0, d1);
            if (gid + 8 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid + 8) * N + col0) = pack_bf16x2_raw(d2, d3);
        }
    }
}

constexpr int P6_PG_SMEM_8W = 49152 + 8 * 2 * 64 * 16 + 8 * 128 * 4;
constexpr int P6_PG_SMEM_16W = 49152 + 12 * 2 * 64 * 16 + 12 * 128 * 4;
constexpr int P10_PG_BN8_12W_SMEM = 12 * 2 * (16 + 8) * 8 * 16 + 12 * 128 * 4;
constexpr int p7_gate_smem_bytes(int bn, int bm, int ks) {
    return ks * 2 * (bm + bn) * 8 * 16 + ks * (bm / 16) * (32 * (bn / 8) * 4) * 4;
}
constexpr int P7_GATE_SMEM_16_16_6 = p7_gate_smem_bytes(16, 16, 6);
constexpr int P7_GATE_SMEM_32_16_4 = p7_gate_smem_bytes(32, 16, 4);
constexpr int P7_GATE_SMEM_32_32_4 = p7_gate_smem_bytes(32, 32, 4);

__global__ __launch_bounds__(384)
void p10_ple_gate_bn8_12w_kernel(
    __nv_bfloat16* __restrict__ gated_out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t x_s0,
    int64_t ple_s0)
{
    constexpr int K = 1536;
    constexpr int WARPS = 12;
    constexpr int TILES = 2;
    extern __shared__ __align__(16) char dyn_smem[];
    auto sA = reinterpret_cast<uint4 (*)[2][16 * 8]>(dyn_smem);
    auto sB = reinterpret_cast<uint4 (*)[2][8 * 8]>(dyn_smem + WARPS * 2 * 16 * 8 * 16);
    auto sRed = reinterpret_cast<float (*)[128]>(dyn_smem + WARPS * 2 * 24 * 8 * 16);

    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, cta_n = blockIdx.x * 8;
    const int k_tile_start = wid * TILES;
    auto issue = [&](int stg, int kt_global) {
        int k_base = kt_global * 64;
        #pragma unroll
        for (int idx = lane; idx < 128; idx += 32) {
            int r = idx >> 3, c = idx & 7;
            p10_cp_async_ca_16B(&sA[wid][stg][r * 8 + p10_swz64(r, c)],
                                x + static_cast<int64_t>(r < M ? r : 0) * x_s0 + k_base + (c << 3), r < M);
        }
        #pragma unroll
        for (int idx = lane; idx < 64; idx += 32) {
            int r = idx >> 3, c = idx & 7;
            p10_cp_async_cg_16B(&sB[wid][stg][r * 8 + p10_swz64(r, c)],
                                w_gate + static_cast<int64_t>(cta_n + r) * K + k_base + (c << 3), true);
        }
    };
    issue(0, k_tile_start);
    p6_cp_async_commit();
    float acc[4] = {};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col = (lane >> 4);
    const int ldm_b_row = lane & 7, ldm_b_col = ((lane >> 3) & 1);
    const int gid = lane >> 2, t4 = lane & 3;
    #pragma unroll
    for (int kt = 0; kt < TILES; ++kt) {
        int cur = kt & 1;
        if (kt + 1 < TILES) { issue(cur ^ 1, k_tile_start + kt + 1); p6_cp_async_commit(); p6_cp_async_wait<1>(); }
        else p6_cp_async_wait<0>();
        __syncwarp();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t a_reg[4], b_reg[2];
            p6_ldmatrix_x4(a_reg, &sA[wid][cur][ldm_a_row * 8 + p10_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
            p6_ldmatrix_x2(b_reg, &sB[wid][cur][ldm_b_row * 8 + p10_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
            p6_mma_m16n8k16_bf16(acc, a_reg, b_reg);
        }
    }
    sRed[wid][gid * 8 + t4 * 2] = acc[0];
    sRed[wid][gid * 8 + t4 * 2 + 1] = acc[1];
    sRed[wid][(gid + 8) * 8 + t4 * 2] = acc[2];
    sRed[wid][(gid + 8) * 8 + t4 * 2 + 1] = acc[3];
    __syncthreads();
    if (tid < M * 4) {
        int ml = tid >> 2, c2 = tid & 3, off = ml * 8 + c2 * 2;
        float d0 = 0.f, d1 = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) { d0 += sRed[w][off]; d1 += sRed[w][off + 1]; }
        int col0 = cta_n + c2 * 2;
        __nv_bfloat162 pv = __ldg(reinterpret_cast<const __nv_bfloat162*>(ple_in + static_cast<int64_t>(ml) * ple_s0 + col0));
        *reinterpret_cast<__nv_bfloat162*>(gated_out + static_cast<int64_t>(ml) * 256 + col0) = make_bfloat162(
            gelu_tanh_mul_bf16(d0, pv.x), gelu_tanh_mul_bf16(d1, pv.y));
    }
}

__global__ __launch_bounds__(128)
void p10_ple_proj_bn16_4w_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int M,
    int64_t x_s0)
{
    constexpr int K = 256, N = 1536, TILES_N = 2, BN = 16;
    __shared__ uint4 sA[4][16 * 8];
    __shared__ uint4 sB[4][BN * 8];
    __shared__ float sRed[4][TILES_N * 128];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, cta_n = blockIdx.x * BN;
    for (int idx = tid; idx < 16 * 32; idx += 128) {
        int m = idx >> 5, c32 = idx & 31;
        p10_cp_async_ca_16B(&sA[c32 >> 3][m * 8 + p10_swz64(m, c32 & 7)],
                            x + static_cast<int64_t>(m < M ? m : 0) * x_s0 + c32 * 8, m < M);
    }
    for (int idx = tid; idx < BN * 32; idx += 128) {
        int r = idx >> 5, c32 = idx & 31;
        p10_cp_async_cg_16B(&sB[c32 >> 3][r * 8 + p10_swz64(r, c32 & 7)],
                            w_proj + static_cast<int64_t>(cta_n + r) * K + c32 * 8, true);
    }
    p6_cp_async_commit();
    p6_cp_async_wait<0>();
    __syncthreads();
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col = (lane >> 4);
    const int ldm_b_row = ((lane >> 4) * 8 + (lane & 7)), ldm_b_col = ((lane >> 3) & 1);
    float acc0[4] = {}, acc1[4] = {};
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        uint32_t a_reg[4], b_reg[4];
        p6_ldmatrix_x4(a_reg, &sA[wid][ldm_a_row * 8 + p10_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
        p6_ldmatrix_x4(b_reg, &sB[wid][ldm_b_row * 8 + p10_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
        uint32_t b0[2] = {b_reg[0], b_reg[1]}, b1[2] = {b_reg[2], b_reg[3]};
        p6_mma_m16n8k16_bf16(acc0, a_reg, b0);
        p6_mma_m16n8k16_bf16(acc1, a_reg, b1);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sRed[wid][lane * 8 + i] = acc0[i];
        sRed[wid][lane * 8 + 4 + i] = acc1[i];
    }
    __syncthreads();
    if (wid < TILES_N) {
        const int t = wid, gid = lane >> 2, t4 = lane & 3;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float* rp = &sRed[w][lane * 8 + t * 4];
            d0 += rp[0]; d1 += rp[1]; d2 += rp[2]; d3 += rp[3];
        }
        int col0 = cta_n + t * 8 + t4 * 2;
        if (gid < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid) * N + col0) = pack_bf16x2_raw(d0, d1);
        if (gid + 8 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid + 8) * N + col0) = pack_bf16x2_raw(d2, d3);
    }
}

template <int BM>
__global__ __launch_bounds__(128)
void p11_ple_proj_fullk_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int M,
    int64_t x_s0)
{
    constexpr int K = 256, N = 1536, BN = 32, WARPS = 4, MT = BM / 16;
    __shared__ uint4 sA[4][BM * 8];
    __shared__ uint4 sB[4][BN * 8];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int cta_n = blockIdx.x * BN, m_base = blockIdx.y * BM;
    for (int idx = tid; idx < BM * 32; idx += WARPS * 32) {
        int m = idx >> 5, c32 = idx & 31, gm = m_base + m;
        p10_cp_async_ca_16B(&sA[c32 >> 3][m * 8 + p10_swz64(m, c32 & 7)],
                            x + static_cast<int64_t>(gm < M ? gm : 0) * x_s0 + c32 * 8, gm < M);
    }
    for (int idx = tid; idx < BN * 32; idx += WARPS * 32) {
        int r = idx >> 5, c32 = idx & 31;
        p10_cp_async_cg_16B(&sB[c32 >> 3][r * 8 + p10_swz64(r, c32 & 7)],
                            w_proj + static_cast<int64_t>(cta_n + r) * K + c32 * 8, true);
    }
    p6_cp_async_commit();
    p6_cp_async_wait<0>();
    __syncthreads();
    float acc[MT][4] = {};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col = lane >> 4;
    const int ldm_b_row = lane & 7, ldm_b_col = (lane >> 3) & 1, b_row_off = wid * 8 + ldm_b_row;
    #pragma unroll
    for (int chk = 0; chk < 4; ++chk) {
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t b_reg[2];
            p6_ldmatrix_x2(b_reg, &sB[chk][b_row_off * 8 + p10_swz64(b_row_off, ks * 2 + ldm_b_col)]);
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                uint32_t a_reg[4];
                int ra = mt * 16 + ldm_a_row;
                p6_ldmatrix_x4(a_reg, &sA[chk][ra * 8 + p10_swz64(ra, ks * 2 + ldm_a_col)]);
                p6_mma_m16n8k16_bf16(acc[mt], a_reg, b_reg);
            }
        }
    }
    const int gid = lane >> 2, t4 = lane & 3, col0 = cta_n + wid * 8 + t4 * 2;
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        int gm0 = m_base + mt * 16 + gid, gm1 = gm0 + 8;
        if (gm0 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gm0) * N + col0) = pack_bf16x2_raw(acc[mt][0], acc[mt][1]);
        if (gm1 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gm1) * N + col0) = pack_bf16x2_raw(acc[mt][2], acc[mt][3]);
    }
}

template <bool LE8>
__global__ __launch_bounds__(128)
void p13_ple_proj_4w_bn16_sk2_soa_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int M,
    int64_t x_s0)
{
    constexpr int K = 256, N = 1536, BN = 16, A_ROWS = LE8 ? 8 : 16;
    __shared__ uint4 sA[4][A_ROWS * 8];
    __shared__ uint4 sB[4][BN * 8];
    __shared__ float sRed[2][2][4][32];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, cta_n = blockIdx.x * BN;
    #pragma unroll
    for (int idx = tid; idx < A_ROWS * 32; idx += 128) {
        int m = idx >> 5, c32 = idx & 31;
        p10_cp_async_ca_16B(&sA[c32 >> 3][m * 8 + p10_swz64(m, c32 & 7)],
                            x + static_cast<int64_t>(m < M ? m : 0) * x_s0 + c32 * 8, m < M);
    }
    #pragma unroll
    for (int idx = tid; idx < BN * 32; idx += 128) {
        int r = idx >> 5, c32 = idx & 31;
        p10_cp_async_cg_16B(&sB[c32 >> 3][r * 8 + p10_swz64(r, c32 & 7)],
                            w_proj + static_cast<int64_t>(cta_n + r) * K + c32 * 8, true);
    }
    p6_cp_async_commit();
    p6_cp_async_wait<0>();
    __syncthreads();

    const int kh = wid >> 1, nt = wid & 1;
    const int ldm_a_row = (LE8 ? 0 : (((lane >> 3) & 1) * 8)) + (lane & 7);
    const int ldm_a_col = (lane >> 4);
    const int ldm_b_row = nt * 8 + (lane & 7), ldm_b_col = ((lane >> 3) & 1);
    float acc[4] = {};
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        int stg = kh * 2 + sub;
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t a_reg[4], b_reg[2];
            p6_ldmatrix_x4(a_reg, &sA[stg][ldm_a_row * 8 + p10_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
            p6_ldmatrix_x2(b_reg, &sB[stg][ldm_b_row * 8 + p10_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
            p6_mma_m16n8k16_bf16(acc, a_reg, b_reg);
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) sRed[kh][nt][i][lane] = acc[i];
    __syncthreads();

    if (kh == 0) {
        const int gid = lane >> 2, t4 = lane & 3, col0 = cta_n + nt * 8 + t4 * 2;
        if (gid < M) {
            float d0 = sRed[0][nt][0][lane] + sRed[1][nt][0][lane];
            float d1 = sRed[0][nt][1][lane] + sRed[1][nt][1][lane];
            *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid) * N + col0) = pack_bf16x2_raw(d0, d1);
        }
        if constexpr (!LE8) {
            if (gid + 8 < M) {
                float d2 = sRed[0][nt][2][lane] + sRed[1][nt][2][lane];
                float d3 = sRed[0][nt][3][lane] + sRed[1][nt][3][lane];
                *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid + 8) * N + col0) = pack_bf16x2_raw(d2, d3);
            }
        }
    }
}

template <bool LE8, bool EVICT_W = true>
__global__ __launch_bounds__(256)
void p13_ple_proj_8w_bn16_sk4_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_proj,
    int M,
    int64_t x_s0)
{
    constexpr int K = 256, N = 1536, BN = 16, A_ROWS = LE8 ? 8 : 16;
    __shared__ uint4 sA[4][A_ROWS * 8];
    __shared__ uint4 sB[4][BN * 8];
    __shared__ float sRed[4][2][4][32];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, cta_n = blockIdx.x * BN;
    #pragma unroll
    for (int idx = tid; idx < A_ROWS * 32; idx += 256) {
        int m = idx >> 5, c32 = idx & 31;
        p10_cp_async_ca_16B(&sA[c32 >> 3][m * 8 + p10_swz64(m, c32 & 7)],
                            x + static_cast<int64_t>(m < M ? m : 0) * x_s0 + c32 * 8, m < M);
    }
    #pragma unroll
    for (int idx = tid; idx < BN * 32; idx += 256) {
        int r = idx >> 5, c32 = idx & 31;
        void* sp = &sB[c32 >> 3][r * 8 + p10_swz64(r, c32 & 7)];
        const void* gp = w_proj + static_cast<int64_t>(cta_n + r) * K + c32 * 8;
        if constexpr (EVICT_W) p10_cp_async_cg_evict_16B(sp, gp, true);
        else p10_cp_async_cg_16B(sp, gp, true);
    }
    p6_cp_async_commit();
    p6_cp_async_wait<0>();
    __syncthreads();

    const int kh = wid >> 1, nt = wid & 1;
    const int ldm_a_row = (LE8 ? 0 : (((lane >> 3) & 1) * 8)) + (lane & 7);
    const int ldm_a_col = (lane >> 4);
    const int ldm_b_row = nt * 8 + (lane & 7), ldm_b_col = ((lane >> 3) & 1);
    float acc[4] = {};
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        uint32_t a_reg[4], b_reg[2];
        p6_ldmatrix_x4(a_reg, &sA[kh][ldm_a_row * 8 + p10_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
        p6_ldmatrix_x2(b_reg, &sB[kh][ldm_b_row * 8 + p10_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
        p6_mma_m16n8k16_bf16(acc, a_reg, b_reg);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) sRed[kh][nt][i][lane] = acc[i];
    __syncthreads();

    if (kh == 0) {
        const int gid = lane >> 2, t4 = lane & 3, col0 = cta_n + nt * 8 + t4 * 2;
        if (gid < M) {
            float d0 = sRed[0][nt][0][lane] + sRed[1][nt][0][lane] + sRed[2][nt][0][lane] + sRed[3][nt][0][lane];
            float d1 = sRed[0][nt][1][lane] + sRed[1][nt][1][lane] + sRed[2][nt][1][lane] + sRed[3][nt][1][lane];
            *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid) * N + col0) = pack_bf16x2_raw(d0, d1);
        }
        if constexpr (!LE8) {
            if (gid + 8 < M) {
                float d2 = sRed[0][nt][2][lane] + sRed[1][nt][2][lane] + sRed[2][nt][2][lane] + sRed[3][nt][2][lane];
                float d3 = sRed[0][nt][3][lane] + sRed[1][nt][3][lane] + sRed[2][nt][3][lane] + sRed[3][nt][3][lane];
                *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gid + 8) * N + col0) = pack_bf16x2_raw(d2, d3);
            }
        }
    }
}

template <int WARPS>
__global__ __launch_bounds__(WARPS * 32)
void p13_ple_gate_bn8_stat_kernel(
    __nv_bfloat16* __restrict__ gated_out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t x_s0,
    int64_t ple_s0)
{
    constexpr int K = 1536;
    constexpr int TILES = 24 / WARPS;
    __shared__ __align__(16) uint4 sA[WARPS][2][16 * 8];
    __shared__ __align__(16) uint4 sB[WARPS][2][8 * 8];
    __shared__ __align__(16) float sRed[WARPS][128];

    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int cta_n = blockIdx.x * 8, m_base = blockIdx.y * 16, k_tile_start = wid * TILES;

    auto issue = [&](int stg, int kt_global) {
        int k_base = kt_global * 64;
        #pragma unroll
        for (int idx = lane; idx < 128; idx += 32) {
            int r = idx >> 3, c = idx & 7, gm = m_base + r;
            p10_cp_async_ca_16B(&sA[wid][stg][r * 8 + p10_swz64(r, c)],
                                x + static_cast<int64_t>(gm < M ? gm : 0) * x_s0 + k_base + (c << 3),
                                gm < M);
        }
        #pragma unroll
        for (int idx = lane; idx < 64; idx += 32) {
            int r = idx >> 3, c = idx & 7;
            p10_cp_async_cg_16B(&sB[wid][stg][r * 8 + p10_swz64(r, c)],
                                w_gate + static_cast<int64_t>(cta_n + r) * K + k_base + (c << 3),
                                true);
        }
    };
    issue(0, k_tile_start);
    p6_cp_async_commit();

    float acc[4] = {};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col = (lane >> 4);
    const int ldm_b_row = lane & 7, ldm_b_col = ((lane >> 3) & 1);
    const int gid = lane >> 2, t4 = lane & 3;
    #pragma unroll
    for (int kt = 0; kt < TILES; ++kt) {
        int cur = kt & 1;
        if (kt + 1 < TILES) {
            issue(cur ^ 1, k_tile_start + kt + 1);
            p6_cp_async_commit();
            p6_cp_async_wait<1>();
        } else {
            p6_cp_async_wait<0>();
        }
        __syncwarp();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t a_reg[4], b_reg[2];
            p6_ldmatrix_x4(a_reg, &sA[wid][cur][ldm_a_row * 8 + p10_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
            p6_ldmatrix_x2(b_reg, &sB[wid][cur][ldm_b_row * 8 + p10_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
            p6_mma_m16n8k16_bf16(acc, a_reg, b_reg);
        }
    }
    sRed[wid][gid * 8 + t4 * 2] = acc[0];
    sRed[wid][gid * 8 + t4 * 2 + 1] = acc[1];
    sRed[wid][(gid + 8) * 8 + t4 * 2] = acc[2];
    sRed[wid][(gid + 8) * 8 + t4 * 2 + 1] = acc[3];
    __syncthreads();

    const int rem = M - m_base;
    const int valid_rows = rem < 16 ? rem : 16;
    if (tid < valid_rows * 4) {
        int ml = tid >> 2, c2 = tid & 3, off = ml * 8 + c2 * 2;
        float d0 = 0.f, d1 = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            d0 += sRed[w][off];
            d1 += sRed[w][off + 1];
        }
        int gm = m_base + ml;
        int col0 = cta_n + c2 * 2;
        __nv_bfloat162 pv = __ldg(reinterpret_cast<const __nv_bfloat162*>(ple_in + static_cast<int64_t>(gm) * ple_s0 + col0));
        *reinterpret_cast<__nv_bfloat162*>(gated_out + static_cast<int64_t>(gm) * 256 + col0) = make_bfloat162(
            gelu_tanh_mul_bf16(d0, pv.x), gelu_tanh_mul_bf16(d1, pv.y));
    }
}

template <int BN, int BM, int K_SPLITS>
__global__ __launch_bounds__(K_SPLITS * 32)
void p13_ple_gate_ca_stat_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t x_s0,
    int64_t ple_s0)
{
    constexpr int K = 1536;
    constexpr int K_TILES_PER_WARP = 24 / K_SPLITS;
    constexpr int TN = BN / 8;
    constexpr int MT = BM / 16;
    __shared__ __align__(16) uint4 sA[K_SPLITS][2][BM * 8];
    __shared__ __align__(16) uint4 sB[K_SPLITS][2][BN * 8];
    __shared__ __align__(16) float sRed[K_SPLITS][MT][32 * TN * 4];

    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int cta_n = blockIdx.x * BN, m_base = blockIdx.y * BM, k_tile_start = wid * K_TILES_PER_WARP;

    auto issue_tile = [&](int stg, int kt_global) {
        const int k_base = kt_global * 64;
        #pragma unroll
        for (int idx = lane; idx < BM * 8; idx += 32) {
            int r = idx >> 3, c = idx & 7, gm = m_base + r;
            p10_cp_async_ca_16B(&sA[wid][stg][r * 8 + p10_swz64(r, c)],
                                x + static_cast<int64_t>(gm < M ? gm : 0) * x_s0 + k_base + (c << 3),
                                gm < M);
        }
        #pragma unroll
        for (int idx = lane; idx < BN * 8; idx += 32) {
            int r = idx >> 3, c = idx & 7;
            p10_cp_async_cg_16B(&sB[wid][stg][r * 8 + p10_swz64(r, c)],
                                w_gate + static_cast<int64_t>(cta_n + r) * K + k_base + (c << 3),
                                true);
        }
    };
    issue_tile(0, k_tile_start);
    p6_cp_async_commit();

    float acc[MT][TN][4] = {};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col = (lane >> 4);
    const int ldm_b_row = ((lane >> 4) * 8 + (lane & 7)), ldm_b_col = ((lane >> 3) & 1);
    for (int kt = 0; kt < K_TILES_PER_WARP; ++kt) {
        int cur = kt & 1;
        if (kt + 1 < K_TILES_PER_WARP) {
            issue_tile(cur ^ 1, k_tile_start + kt + 1);
            p6_cp_async_commit();
            p6_cp_async_wait<1>();
        } else {
            p6_cp_async_wait<0>();
        }
        __syncwarp();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn += 2) {
                uint32_t b_reg[4];
                p6_ldmatrix_x4(b_reg, &sB[wid][cur][(tn * 8 + ldm_b_row) * 8 + p10_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
                uint32_t b0[2] = {b_reg[0], b_reg[1]}, b1[2] = {b_reg[2], b_reg[3]};
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    uint32_t a_reg[4];
                    p6_ldmatrix_x4(a_reg, &sA[wid][cur][(mt * 16 + ldm_a_row) * 8 + p10_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
                    p6_mma_m16n8k16_bf16(acc[mt][tn], a_reg, b0);
                    p6_mma_m16n8k16_bf16(acc[mt][tn + 1], a_reg, b1);
                }
            }
        }
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int t = 0; t < TN; ++t) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sRed[wid][mt][lane * (TN * 4) + t * 4 + i] = acc[mt][t][i];
            }
        }
    }
    __syncthreads();

    for (int ot = wid; ot < MT * TN; ot += K_SPLITS) {
        const int mt = ot / TN, t = ot - mt * TN;
        const int gid = lane >> 2, t4 = lane & 3, off = t * 4;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        #pragma unroll
        for (int w = 0; w < K_SPLITS; ++w) {
            const float* rp = &sRed[w][mt][lane * (TN * 4) + off];
            d0 += rp[0]; d1 += rp[1]; d2 += rp[2]; d3 += rp[3];
        }
        int col0 = cta_n + t * 8 + t4 * 2;
        int gm0 = m_base + mt * 16 + gid;
        int gm1 = gm0 + 8;
        if (gm0 < M) {
            __nv_bfloat162 pv = __ldg(reinterpret_cast<const __nv_bfloat162*>(ple_in + static_cast<int64_t>(gm0) * ple_s0 + col0));
            *reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(gm0) * 256 + col0) = make_bfloat162(
                gelu_tanh_mul_bf16(d0, pv.x), gelu_tanh_mul_bf16(d1, pv.y));
        }
        if (gm1 < M) {
            __nv_bfloat162 pv = __ldg(reinterpret_cast<const __nv_bfloat162*>(ple_in + static_cast<int64_t>(gm1) * ple_s0 + col0));
            *reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(gm1) * 256 + col0) = make_bfloat162(
                gelu_tanh_mul_bf16(d2, pv.x), gelu_tanh_mul_bf16(d3, pv.y));
        }
    }
}

template <int BN, int BM, int K_SPLITS>
__global__ __launch_bounds__(K_SPLITS * 32)
void p13_ple_gate_ca_dyn_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w_gate,
    const __nv_bfloat16* __restrict__ ple_in,
    int M,
    int64_t x_s0,
    int64_t ple_s0)
{
    constexpr int K = 1536;
    constexpr int K_TILES_PER_WARP = 24 / K_SPLITS;
    constexpr int TN = BN / 8;
    constexpr int MT = BM / 16;
    extern __shared__ __align__(16) char dyn_smem[];
    auto sA = reinterpret_cast<uint4 (*)[2][BM * 8]>(dyn_smem);
    auto sB = reinterpret_cast<uint4 (*)[2][BN * 8]>(dyn_smem + K_SPLITS * 2 * BM * 8 * 16);
    auto sRed = reinterpret_cast<float (*)[MT][32 * TN * 4]>(
        dyn_smem + K_SPLITS * 2 * (BM + BN) * 8 * 16);

    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int cta_n = blockIdx.x * BN, m_base = blockIdx.y * BM, k_tile_start = wid * K_TILES_PER_WARP;

    auto issue_tile = [&](int stg, int kt_global) {
        const int k_base = kt_global * 64;
        #pragma unroll
        for (int idx = lane; idx < BM * 8; idx += 32) {
            int r = idx >> 3, c = idx & 7, gm = m_base + r;
            p10_cp_async_ca_16B(&sA[wid][stg][r * 8 + p10_swz64(r, c)],
                                x + static_cast<int64_t>(gm < M ? gm : 0) * x_s0 + k_base + (c << 3),
                                gm < M);
        }
        #pragma unroll
        for (int idx = lane; idx < BN * 8; idx += 32) {
            int r = idx >> 3, c = idx & 7;
            p10_cp_async_cg_16B(&sB[wid][stg][r * 8 + p10_swz64(r, c)],
                                w_gate + static_cast<int64_t>(cta_n + r) * K + k_base + (c << 3),
                                true);
        }
    };
    issue_tile(0, k_tile_start);
    p6_cp_async_commit();

    float acc[MT][TN][4] = {};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col = (lane >> 4);
    const int ldm_b_row = ((lane >> 4) * 8 + (lane & 7)), ldm_b_col = ((lane >> 3) & 1);
    for (int kt = 0; kt < K_TILES_PER_WARP; ++kt) {
        int cur = kt & 1;
        if (kt + 1 < K_TILES_PER_WARP) {
            issue_tile(cur ^ 1, k_tile_start + kt + 1);
            p6_cp_async_commit();
            p6_cp_async_wait<1>();
        } else {
            p6_cp_async_wait<0>();
        }
        __syncwarp();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn += 2) {
                uint32_t b_reg[4];
                p6_ldmatrix_x4(b_reg, &sB[wid][cur][(tn * 8 + ldm_b_row) * 8 + p10_swz64(ldm_b_row, ks * 2 + ldm_b_col)]);
                uint32_t b0[2] = {b_reg[0], b_reg[1]}, b1[2] = {b_reg[2], b_reg[3]};
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    uint32_t a_reg[4];
                    p6_ldmatrix_x4(a_reg, &sA[wid][cur][(mt * 16 + ldm_a_row) * 8 + p10_swz64(ldm_a_row, ks * 2 + ldm_a_col)]);
                    p6_mma_m16n8k16_bf16(acc[mt][tn], a_reg, b0);
                    p6_mma_m16n8k16_bf16(acc[mt][tn + 1], a_reg, b1);
                }
            }
        }
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int t = 0; t < TN; ++t) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sRed[wid][mt][lane * (TN * 4) + t * 4 + i] = acc[mt][t][i];
            }
        }
    }
    __syncthreads();

    for (int ot = wid; ot < MT * TN; ot += K_SPLITS) {
        const int mt = ot / TN, t = ot - mt * TN;
        const int gid = lane >> 2, t4 = lane & 3, off = t * 4;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        #pragma unroll
        for (int w = 0; w < K_SPLITS; ++w) {
            const float* rp = &sRed[w][mt][lane * (TN * 4) + off];
            d0 += rp[0]; d1 += rp[1]; d2 += rp[2]; d3 += rp[3];
        }
        int col0 = cta_n + t * 8 + t4 * 2;
        int gm0 = m_base + mt * 16 + gid;
        int gm1 = gm0 + 8;
        if (gm0 < M) {
            __nv_bfloat162 pv = __ldg(reinterpret_cast<const __nv_bfloat162*>(ple_in + static_cast<int64_t>(gm0) * ple_s0 + col0));
            *reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(gm0) * 256 + col0) = make_bfloat162(
                gelu_tanh_mul_bf16(d0, pv.x), gelu_tanh_mul_bf16(d1, pv.y));
        }
        if (gm1 < M) {
            __nv_bfloat162 pv = __ldg(reinterpret_cast<const __nv_bfloat162*>(ple_in + static_cast<int64_t>(gm1) * ple_s0 + col0));
            *reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(gm1) * 256 + col0) = make_bfloat162(
                gelu_tanh_mul_bf16(d2, pv.x), gelu_tanh_mul_bf16(d3, pv.y));
        }
    }
}

template <int M_TILES, int TILES_N, int MIN_CTAS>
__global__ __launch_bounds__(128, MIN_CTAS)
void p15_model_proj_ca_union_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    int M,
    int64_t x_s0)
{
    constexpr int K = 1536, N = 8960, K_SPLITS = 4, K_TILES_PER_WARP = 6;
    constexpr int BM = M_TILES * 16, BN = TILES_N * 8;
    struct alignas(16) WarpSlot {
        union {
            struct { uint4 sA[2][BM * 8]; uint4 sB[2][BN * 8]; } ab;
            float sRed[M_TILES][32 * TILES_N * 4];
        };
    };
    __shared__ WarpSlot slots[K_SPLITS];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int cta_n = blockIdx.x * BN, m_base = blockIdx.y * BM, k_tile_start = wid * K_TILES_PER_WARP;

    auto issue = [&](int stg, int kt_global) {
        int k_base = kt_global * 64;
        #pragma unroll
        for (int idx = lane; idx < BM * 8; idx += 32) {
            int r = idx >> 3, c = idx & 7, gm = m_base + r;
            p10_cp_async_ca_16B(&slots[wid].ab.sA[stg][r * 8 + p10_swz64(r, c)],
                                x + static_cast<int64_t>(gm < M ? gm : 0) * x_s0 + k_base + (c << 3), gm < M);
        }
        #pragma unroll
        for (int idx = lane; idx < BN * 8; idx += 32) {
            int r = idx >> 3, c = idx & 7, gn = cta_n + r;
            p10_cp_async_cg_16B(&slots[wid].ab.sB[stg][r * 8 + p10_swz64(r, c)],
                                w + static_cast<int64_t>(gn < N ? gn : 0) * K + k_base + (c << 3), gn < N);
        }
    };
    issue(0, k_tile_start);
    p6_cp_async_commit();

    float acc[M_TILES][TILES_N][4] = {};
    const int ldm_a_row = ((lane >> 3) & 1) * 8 + (lane & 7), ldm_a_col = (lane >> 4);
    const int ldm_b_row = (lane >> 4) * 8 + (lane & 7), ldm_b_col = ((lane >> 3) & 1);
    const int ldm_b1_row = lane & 7;
    #pragma unroll
    for (int kt = 0; kt < K_TILES_PER_WARP; ++kt) {
        int cur = kt & 1;
        if (kt + 1 < K_TILES_PER_WARP) { issue(cur ^ 1, k_tile_start + kt + 1); p6_cp_async_commit(); p6_cp_async_wait<1>(); }
        else p6_cp_async_wait<0>();
        __syncwarp();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                uint32_t a_reg[4];
                int ra = mt * 16 + ldm_a_row;
                p6_ldmatrix_x4(a_reg, &slots[wid].ab.sA[cur][ra * 8 + p10_swz64(ra, ks * 2 + ldm_a_col)]);
                if constexpr (TILES_N == 2) {
                    uint32_t b_reg[2], b_reg2[2];
                    p6_ldmatrix_x2(b_reg, &slots[wid].ab.sB[cur][ldm_b1_row * 8 + p10_swz64(ldm_b1_row, ks * 2 + ldm_b_col)]);
                    p6_ldmatrix_x2(b_reg2, &slots[wid].ab.sB[cur][(8 + ldm_b1_row) * 8 + p10_swz64(8 + ldm_b1_row, ks * 2 + ldm_b_col)]);
                    p6_mma_m16n8k16_bf16(acc[mt][0], a_reg, b_reg);
                    p6_mma_m16n8k16_bf16(acc[mt][1], a_reg, b_reg2);
                } else {
                    #pragma unroll
                    for (int tg = 0; tg < TILES_N / 2; ++tg) {
                        uint32_t b_reg[4];
                        int rb = tg * 16 + ldm_b_row;
                        p6_ldmatrix_x4(b_reg, &slots[wid].ab.sB[cur][rb * 8 + p10_swz64(rb, ks * 2 + ldm_b_col)]);
                        uint32_t b0[2] = {b_reg[0], b_reg[1]}, b1[2] = {b_reg[2], b_reg[3]};
                        p6_mma_m16n8k16_bf16(acc[mt][tg * 2], a_reg, b0);
                        p6_mma_m16n8k16_bf16(acc[mt][tg * 2 + 1], a_reg, b1);
                    }
                }
            }
        }
    }
    __syncwarp();
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt)
        #pragma unroll
        for (int t = 0; t < TILES_N; ++t)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                slots[wid].sRed[mt][lane * (TILES_N * 4) + t * 4 + i] = acc[mt][t][i];
    __syncthreads();
    for (int ot = wid; ot < M_TILES * TILES_N; ot += K_SPLITS) {
        const int mt = ot / TILES_N, t = ot - mt * TILES_N;
        const int gid = lane >> 2, t4 = lane & 3;
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        #pragma unroll
        for (int w = 0; w < K_SPLITS; ++w) {
            const float* rp = &slots[w].sRed[mt][lane * (TILES_N * 4) + t * 4];
            d0 += rp[0]; d1 += rp[1]; d2 += rp[2]; d3 += rp[3];
        }
        int col0 = cta_n + t * 8 + t4 * 2;
        int gm0 = m_base + mt * 16 + gid, gm1 = gm0 + 8;
        if (col0 + 1 < N) {
            if (gm0 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gm0) * N + col0) = pack_bf16x2_raw(d0, d1);
            if (gm1 < M) *reinterpret_cast<uint32_t*>(out + static_cast<int64_t>(gm1) * N + col0) = pack_bf16x2_raw(d2, d3);
        }
    }
}

inline void ensure_p6_pg_m16_smem(int dev) {
    static std::once_flag flags[16];
    const int idx = (dev >= 0 && dev < 16) ? dev : 0;
    std::call_once(flags[idx], []() {
        TORCH_CHECK(
            cudaFuncSetAttribute(
                p6_post_ffn_ple_gate_mma16_kernel<8, 8>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                P6_PG_SMEM_8W) == cudaSuccess,
            "cudaFuncSetAttribute failed for p6_post_ffn_ple_gate_mma16_kernel<8,8>");
        TORCH_CHECK(
            cudaFuncSetAttribute(
                p6_post_ffn_ple_gate_mma16_kernel<16, 12>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                P6_PG_SMEM_16W) == cudaSuccess,
            "cudaFuncSetAttribute failed for p6_post_ffn_ple_gate_mma16_kernel<16,12>");
        TORCH_CHECK(
            cudaFuncSetAttribute(
                p7_ple_gate_ca_kernel<16, 16, 6>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                P7_GATE_SMEM_16_16_6) == cudaSuccess,
            "cudaFuncSetAttribute failed for p7_ple_gate_ca_kernel<16,16,6>");
        TORCH_CHECK(
            cudaFuncSetAttribute(
                p13_ple_gate_ca_dyn_kernel<32, 32, 4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                P7_GATE_SMEM_32_32_4) == cudaSuccess,
            "cudaFuncSetAttribute failed for p13_ple_gate_ca_dyn_kernel<32,32,4>");
        TORCH_CHECK(
            cudaFuncSetAttribute(
                p13_ple_gate_ca_dyn_kernel<32, 16, 4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                P7_GATE_SMEM_32_16_4) == cudaSuccess,
            "cudaFuncSetAttribute failed for p13_ple_gate_ca_dyn_kernel<32,16,4>");
        TORCH_CHECK(
            cudaFuncSetAttribute(
                p10_ple_gate_bn8_12w_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                P10_PG_BN8_12W_SMEM) == cudaSuccess,
            "cudaFuncSetAttribute failed for p10_ple_gate_bn8_12w_kernel");
    });
}

} // namespace

torch::Tensor gemma4_e2b_model_proj_gemv(
    torch::Tensor x,
    torch::Tensor w_proj)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
    TORCH_CHECK(w_proj.is_cuda() && w_proj.scalar_type() == torch::kBFloat16, "w_proj must be CUDA bf16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 1536, "x must have shape [M, 1536]");
    TORCH_CHECK(w_proj.dim() == 2 && w_proj.size(0) == 8960 && w_proj.size(1) == 1536,
                "w_proj must have shape [8960, 1536]");
    c10::cuda::CUDAGuard device_guard(x.device());
    const int64_t M = x.size(0);
    if (M == 0) return torch::empty({0, 8960}, x.options());
    if (M > 16) {
        return at::linear(x, w_proj);
    }
    auto x_c = (x.stride(1) == 1 && (x.stride(0) & 7) == 0 &&
                (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0)
        ? x : x.contiguous();
    auto w_c = w_proj.is_contiguous() ? w_proj : w_proj.contiguous();
    auto out = torch::empty({M, 8960}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t xs0 = x_c.stride(0);
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr());
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr());
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    if (M == 1) {
        p9_model_proj_mcol_kernel<1, 4, true><<<560, 128, 0, stream>>>(op, xp, wp, xs0);
    } else if (M == 2) {
        gemma4_e2b_model_proj_gemv_kernel<2, true><<<1120, 256, 0, stream>>>(op, xp, wp, xs0);
    } else if (M == 3) {
        p9_model_proj_mcol_kernel<3, 4, true><<<560, 128, 0, stream>>>(op, xp, wp, xs0);
    } else {
        p15_model_proj_ca_union_kernel<1, 2, 4><<<560, 128, 0, stream>>>(op, xp, wp, static_cast<int>(M), xs0);
    }
    return out;
}

torch::Tensor gemma4_e2b_project_ple_combine(
    torch::Tensor proj,
    c10::optional<torch::Tensor> ple_in,
    torch::Tensor norm_w,
    torch::Tensor proj_scale,
    torch::Tensor in_scale,
    double eps)
{
    c10::cuda::CUDAGuard device_guard(proj.device());
    TORCH_CHECK(proj.size(1) == 8960, "proj must have width 8960");
    const int64_t M = proj.size(0);
    auto out = torch::empty({35, M, 256}, proj.options());
    if (M == 0) return out;
    auto proj_c = (proj.stride(1) == 1) ? proj : proj.contiguous();
    auto norm_w_c = norm_w.is_contiguous() ? norm_w : norm_w.contiguous();
    const bool ps_bf16 = (proj_scale.scalar_type() == torch::kBFloat16);
    const bool is_bf16 = (in_scale.scalar_type() == torch::kBFloat16);
    const int blocks = static_cast<int>((M * 35 + 7) / 8);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (M <= 4) {
        if (ple_in.has_value()) {
            auto ple_c = (ple_in->stride(2) == 1 && ple_in->stride(1) == 256)
                ? *ple_in : ple_in->contiguous();
            gemma4_e2b_project_ple_1warp_kernel<true><<<static_cast<int>(M * 35), 32, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(proj_c.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(norm_w_c.data_ptr()),
                proj_scale.data_ptr(), ps_bf16,
                in_scale.data_ptr(), is_bf16,
                static_cast<int>(M), proj_c.stride(0), ple_c.stride(0),
                static_cast<float>(eps));
        } else {
            gemma4_e2b_project_ple_1warp_kernel<false><<<static_cast<int>(M * 35), 32, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(proj_c.data_ptr()),
                nullptr,
                reinterpret_cast<const __nv_bfloat16*>(norm_w_c.data_ptr()),
                proj_scale.data_ptr(), ps_bf16,
                in_scale.data_ptr(), is_bf16,
                static_cast<int>(M), proj_c.stride(0), 0,
                static_cast<float>(eps));
        }
        return out;
    }
    if (ple_in.has_value()) {
        auto ple_c = (ple_in->stride(2) == 1 && ple_in->stride(1) == 256)
            ? *ple_in : ple_in->contiguous();
        gemma4_e2b_project_ple_combine_kernel<true><<<blocks, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(proj_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(norm_w_c.data_ptr()),
            proj_scale.data_ptr(), ps_bf16,
            in_scale.data_ptr(), is_bf16,
            static_cast<int>(M), proj_c.stride(0), ple_c.stride(0),
            static_cast<float>(eps));
    } else {
        gemma4_e2b_project_ple_combine_kernel<false><<<blocks, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(proj_c.data_ptr()),
            nullptr,
            reinterpret_cast<const __nv_bfloat16*>(norm_w_c.data_ptr()),
            proj_scale.data_ptr(), ps_bf16,
            in_scale.data_ptr(), is_bf16,
            static_cast<int>(M), proj_c.stride(0), 0,
            static_cast<float>(eps));
    }
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> gemma4_e2b_post_attn_pre_ffn_norms(
    torch::Tensor attn_out,
    torch::Tensor residual,
    torch::Tensor post_attn_w,
    torch::Tensor pre_ffn_w,
    double eps)
{
    c10::cuda::CUDAGuard device_guard(attn_out.device());
    TORCH_CHECK(attn_out.size(1) == 1536, "E2B norms require hidden_size=1536");
    const int64_t M = attn_out.size(0);
    auto res_out = torch::empty({M, 1536}, attn_out.options());
    auto norm_out = torch::empty({M, 1536}, attn_out.options());
    if (M == 0) return {res_out, norm_out};
    auto attn_c = (attn_out.stride(1) == 1 && (attn_out.stride(0) & 7) == 0 &&
                   (reinterpret_cast<uintptr_t>(attn_out.data_ptr()) & 15) == 0)
        ? attn_out : attn_out.contiguous();
    auto res_c = (residual.stride(1) == 1 && (residual.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(residual.data_ptr()) & 15) == 0)
        ? residual : residual.contiguous();
    auto w1_c = post_attn_w.is_contiguous() ? post_attn_w : post_attn_w.contiguous();
    auto w2_c = pre_ffn_w.is_contiguous() ? pre_ffn_w : pre_ffn_w.contiguous();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if ((M >= 1024 && M <= 1600) || (M >= 1920 && M <= 3072)) {
        gemma4_e2b_post_attn_pre_ffn_norms_warp_kernel<<<static_cast<int>((M + 3) / 4), 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(attn_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(res_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w1_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w2_c.data_ptr()),
            static_cast<int>(M), attn_c.stride(0), res_c.stride(0), static_cast<float>(eps));
    } else {
        gemma4_e2b_post_attn_pre_ffn_norms_kernel<<<static_cast<int>(M), 192, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(attn_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(res_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w1_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w2_c.data_ptr()),
            attn_c.stride(0), res_c.stride(0), static_cast<float>(eps));
    }
    return {res_out, norm_out};
}

torch::Tensor gemma4_e2b_post_ffn_res_norm(
    torch::Tensor mlp_out,
    torch::Tensor residual,
    torch::Tensor post_ffn_w,
    double eps)
{
    c10::cuda::CUDAGuard device_guard(mlp_out.device());
    TORCH_CHECK(mlp_out.size(1) == 1536, "E2B norms require hidden_size=1536");
    const int64_t M = mlp_out.size(0);
    auto res_out = torch::empty({M, 1536}, mlp_out.options());
    if (M == 0) return res_out;
    auto mlp_c = (mlp_out.stride(1) == 1 && (mlp_out.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(mlp_out.data_ptr()) & 15) == 0)
        ? mlp_out : mlp_out.contiguous();
    auto res_c = (residual.stride(1) == 1 && (residual.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(residual.data_ptr()) & 15) == 0)
        ? residual : residual.contiguous();
    auto w_c = post_ffn_w.is_contiguous() ? post_ffn_w : post_ffn_w.contiguous();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (M >= 1280 && M <= 1536) {
        gemma4_e2b_post_ffn_res_norm_warp_kernel<<<static_cast<int>((M + 3) / 4), 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(mlp_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(res_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr()),
            static_cast<int>(M), mlp_c.stride(0), res_c.stride(0), static_cast<float>(eps));
        return res_out;
    }
    gemma4_e2b_post_ffn_res_norm_kernel<<<static_cast<int>(M), 192, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(mlp_c.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(res_c.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr()),
        mlp_c.stride(0), res_c.stride(0), static_cast<float>(eps));
    return res_out;
}

torch::Tensor gemma4_e2b_ple_gate_gelu_mul(
    torch::Tensor hidden,
    torch::Tensor w_gate,
    torch::Tensor ple_in)
{
    c10::cuda::CUDAGuard device_guard(hidden.device());
    const int64_t M = hidden.size(0);
    if (M == 0) return torch::empty({0, 256}, hidden.options());
    if (M > 432) {
        auto g = at::linear(hidden, w_gate);
        auto ple_c = (ple_in.stride(1) == 1 && (ple_in.stride(0) & 7) == 0 &&
                      (reinterpret_cast<uintptr_t>(ple_in.data_ptr()) & 15) == 0)
            ? ple_in : ple_in.contiguous();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        if (g.is_contiguous() && ple_c.stride(0) == 256 &&
            (reinterpret_cast<uintptr_t>(g.data_ptr()) & 15) == 0 &&
            (g.numel() & 7) == 0) {
            const int64_t total_u4 = g.numel() >> 3;
            gemma4_e2b_gelu_tanh_mul_inplace_vec_kernel<<<
                static_cast<int>((total_u4 + 255) / 256), 256, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr()),
                total_u4);
            return g;
        }
        auto g_c = (g.stride(1) == 1 && (g.stride(0) & 7) == 0 &&
                    (reinterpret_cast<uintptr_t>(g.data_ptr()) & 15) == 0)
            ? g : g.contiguous();
        auto out = torch::empty({M, 256}, hidden.options());
        gemma4_e2b_gelu_tanh_mul_kernel<<<static_cast<int>((M + 7) / 8), 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(g_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr()),
            static_cast<int>(M), g_c.stride(0), ple_c.stride(0));
        return out;
    }
    auto out = torch::empty({M, 256}, hidden.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto hidden_c = (hidden.stride(1) == 1 && (hidden.stride(0) & 7) == 0 &&
                     (reinterpret_cast<uintptr_t>(hidden.data_ptr()) & 15) == 0)
        ? hidden : hidden.contiguous();
    auto ple_c = (ple_in.stride(1) == 1 && (ple_in.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(ple_in.data_ptr()) & 15) == 0)
        ? ple_in : ple_in.contiguous();
    auto w_c = w_gate.is_contiguous() ? w_gate : w_gate.contiguous();
    const int64_t hs0 = hidden_c.stride(0);
    const int64_t ps0 = ple_c.stride(0);
    const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(hidden_c.data_ptr());
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr());
    const __nv_bfloat16* pp = reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr());
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

    if (M == 1) {
        gemma4_e2b_ple_gate_2col_192_kernel<1><<<128, 192, 0, stream>>>(
            op, hp, wp, pp, 1, hs0, ps0);
    } else if (M == 2) {
        gemma4_e2b_ple_gate_2col_192_kernel<2><<<128, 192, 0, stream>>>(
            op, hp, wp, pp, 2, hs0, ps0);
    } else if (M <= 4) {
        gemma4_e2b_ple_gate_1col_192_kernel<4><<<256, 192, 0, stream>>>(
            op, hp, wp, pp, static_cast<int>(M), hs0, ps0);
    } else if (M <= 16) {
        auto [ws_ptr, td_ptr] = get_splitk_workspace(hidden, stream);
        dim3 grid(32, 4);
        gemma4_e2b_ple_gate_mma16_splitk4_kernel<<<grid, 64, 0, stream>>>(
            op, ws_ptr, td_ptr, hp, wp, pp, static_cast<int>(M), hs0, ps0);
    } else if (M <= 59) {
        dim3 grid(32, static_cast<unsigned int>((M + 15) / 16));
        p13_ple_gate_bn8_stat_kernel<6><<<grid, 192, 0, stream>>>(
            op, hp, wp, pp, static_cast<int>(M), hs0, ps0);
    } else if (M <= 192) {
        ensure_p6_pg_m16_smem(hidden.get_device());
        dim3 grid(16, static_cast<unsigned int>((M + 15) / 16));
        p7_ple_gate_ca_kernel<16, 16, 6><<<grid, 192, P7_GATE_SMEM_16_16_6, stream>>>(
            op, hp, wp, pp, static_cast<int>(M), hs0, ps0);
    } else if (M <= 208) {
        ensure_p6_pg_m16_smem(hidden.get_device());
        dim3 grid(8, static_cast<unsigned int>((M + 15) / 16));
        p13_ple_gate_ca_dyn_kernel<32, 16, 4><<<grid, 128, P7_GATE_SMEM_32_16_4, stream>>>(
            op, hp, wp, pp, static_cast<int>(M), hs0, ps0);
    } else if (M <= 416) {
        ensure_p6_pg_m16_smem(hidden.get_device());
        dim3 grid(8, static_cast<unsigned int>((M + 31) / 32));
        p13_ple_gate_ca_dyn_kernel<32, 32, 4><<<grid, 128, P7_GATE_SMEM_32_32_4, stream>>>(
            op, hp, wp, pp, static_cast<int>(M), hs0, ps0);
    } else {
        ensure_p6_pg_m16_smem(hidden.get_device());
        dim3 grid(8, static_cast<unsigned int>((M + 15) / 16));
        p13_ple_gate_ca_dyn_kernel<32, 16, 4><<<grid, 128, P7_GATE_SMEM_32_16_4, stream>>>(
            op, hp, wp, pp, static_cast<int>(M), hs0, ps0);
    }
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> gemma4_e2b_post_ffn_ple_gate(
    torch::Tensor mlp_out,
    torch::Tensor residual,
    torch::Tensor post_ffn_w,
    torch::Tensor w_gate,
    torch::Tensor ple_in,
    double eps)
{
    c10::cuda::CUDAGuard device_guard(mlp_out.device());
    TORCH_CHECK(mlp_out.size(1) == 1536, "E2B norms require hidden_size=1536");
    const int64_t M = mlp_out.size(0);
    if (M == 0) {
        return {
            torch::empty({0, 1536}, mlp_out.options()),
            torch::empty({0, 256}, mlp_out.options()),
        };
    }
    if (M > 17) {
        auto res2 = gemma4_e2b_post_ffn_res_norm(mlp_out, residual, post_ffn_w, eps);
        auto gated = gemma4_e2b_ple_gate_gelu_mul(res2, w_gate, ple_in);
        return {res2, gated};
    }
    ensure_p6_pg_m16_smem(mlp_out.get_device());
    auto res2_out = torch::empty({M, 1536}, mlp_out.options());
    auto gated_out = torch::empty({M, 256}, mlp_out.options());
    auto mlp_c = (mlp_out.stride(1) == 1 && (mlp_out.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(mlp_out.data_ptr()) & 15) == 0)
        ? mlp_out : mlp_out.contiguous();
    auto res_c = (residual.stride(1) == 1 && (residual.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(residual.data_ptr()) & 15) == 0)
        ? residual : residual.contiguous();
    auto w_c = post_ffn_w.is_contiguous() ? post_ffn_w : post_ffn_w.contiguous();
    auto wg_c = w_gate.is_contiguous() ? w_gate : w_gate.contiguous();
    auto ple_c = (ple_in.stride(1) == 1 && (ple_in.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(ple_in.data_ptr()) & 15) == 0)
        ? ple_in : ple_in.contiguous();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    __nv_bfloat16* r2p = reinterpret_cast<__nv_bfloat16*>(res2_out.data_ptr());
    __nv_bfloat16* gp = reinterpret_cast<__nv_bfloat16*>(gated_out.data_ptr());
    const __nv_bfloat16* mp = reinterpret_cast<const __nv_bfloat16*>(mlp_c.data_ptr());
    const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(res_c.data_ptr());
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr());
    const __nv_bfloat16* wgp = reinterpret_cast<const __nv_bfloat16*>(wg_c.data_ptr());
    const __nv_bfloat16* pp = reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr());
    if (M == 1) {
        gemma4_e2b_post_ffn_ple_gate_small_m_kernel<1, 4, true><<<64, 192, 0, stream>>>(
            r2p, gp, mp, rp, wp, wgp, pp,
            1, mlp_c.stride(0), res_c.stride(0), ple_c.stride(0), static_cast<float>(eps));
    } else if (M == 2) {
        gemma4_e2b_post_ffn_ple_gate_batched_kernel<2, 4, true><<<64, 192, 0, stream>>>(
            r2p, gp, mp, rp, wp, wgp, pp,
            2, mlp_c.stride(0), res_c.stride(0), ple_c.stride(0), static_cast<float>(eps));
    } else if (M == 3 || (M >= 6 && M <= 16)) {
        gemma4_e2b_post_ffn_res_norm_kernel<<<static_cast<int>(M), 192, 0, stream>>>(
            r2p, mp, rp, wp, mlp_c.stride(0), res_c.stride(0), static_cast<float>(eps));
        p10_ple_gate_bn8_12w_kernel<<<32, 384, P10_PG_BN8_12W_SMEM, stream>>>(
            gp, r2p, wgp, pp, static_cast<int>(M), 1536, ple_c.stride(0));
    } else if (M <= 5) {
        p6_post_ffn_ple_gate_mma16_kernel<8, 8><<<32, 256, P6_PG_SMEM_8W, stream>>>(
            r2p, gp, mp, rp, wp, wgp, pp,
            static_cast<int>(M), mlp_c.stride(0), res_c.stride(0), ple_c.stride(0), static_cast<float>(eps));
    } else {
        dim3 grid(32, static_cast<unsigned int>((M + 15) / 16));
        p6_post_ffn_ple_gate_mma16_kernel<16, 12><<<grid, 512, P6_PG_SMEM_16W, stream>>>(
            r2p, gp, mp, rp, wp, wgp, pp,
            static_cast<int>(M), mlp_c.stride(0), res_c.stride(0), ple_c.stride(0), static_cast<float>(eps));
    }
    return {res2_out, gated_out};
}

torch::Tensor gemma4_e2b_ple_proj_gemv(
    torch::Tensor gated_ple,
    torch::Tensor w_proj)
{
    const int64_t M = gated_ple.size(0);
    if (M == 0) return torch::empty({0, 1536}, gated_ple.options());
    if (M > 80) {
        return at::linear(gated_ple, w_proj);
    }
    c10::cuda::CUDAGuard device_guard(gated_ple.device());
    auto gated_c = (gated_ple.stride(1) == 1 && (gated_ple.stride(0) & 7) == 0 &&
                    (reinterpret_cast<uintptr_t>(gated_ple.data_ptr()) & 15) == 0)
        ? gated_ple : gated_ple.contiguous();
    auto w_c = w_proj.is_contiguous() ? w_proj : w_proj.contiguous();
    auto out = torch::empty({M, 1536}, gated_ple.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t xs0 = gated_c.stride(0);
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(gated_c.data_ptr());
    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr());
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

    if (M <= 4) {
        switch (M) {
            case 1: p7_ple_proj_mcol_nosmem_kernel<1, 4><<<96, 128, 0, stream>>>(op, xp, wp, xs0); break;
            case 2: p7_ple_proj_mcol_nosmem_kernel<2, 2><<<192, 128, 0, stream>>>(op, xp, wp, xs0); break;
            case 3: gemma4_e2b_ple_proj_nosmem_kernel<3><<<384, 128, 0, stream>>>(op, xp, wp, xs0); break;
            case 4: p7_ple_proj_mcol_nosmem_kernel<4, 2><<<192, 128, 0, stream>>>(op, xp, wp, xs0); break;
        }
    } else if (M <= 8) {
        p13_ple_proj_8w_bn16_sk4_kernel<true><<<96, 256, 0, stream>>>(
            op, xp, wp, static_cast<int>(M), xs0);
    } else if (M <= 15) {
        p13_ple_proj_8w_bn16_sk4_kernel<false><<<96, 256, 0, stream>>>(
            op, xp, wp, static_cast<int>(M), xs0);
    } else if (M <= 32) {
        dim3 grid(48, static_cast<unsigned int>((M + 15) / 16));
        p11_ple_proj_fullk_kernel<16><<<grid, 128, 0, stream>>>(
            op, xp, wp, static_cast<int>(M), xs0);
    } else {
        dim3 grid(48, static_cast<unsigned int>((M + 31) / 32));
        p11_ple_proj_fullk_kernel<32><<<grid, 128, 0, stream>>>(
            op, xp, wp, static_cast<int>(M), xs0);
    }
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> gemma4_e2b_ple_post_norm_scalar_next_norm(
    torch::Tensor ple_proj,
    torch::Tensor residual,
    torch::Tensor post_ple_w,
    torch::Tensor layer_scalar,
    torch::Tensor next_norm_w,
    double eps)
{
    c10::cuda::CUDAGuard device_guard(ple_proj.device());
    TORCH_CHECK(ple_proj.size(1) == 1536, "E2B norms require hidden_size=1536");
    const int64_t M = ple_proj.size(0);
    auto next_res = torch::empty({M, 1536}, ple_proj.options());
    auto next_norm = torch::empty({M, 1536}, ple_proj.options());
    if (M == 0) return {next_res, next_norm};
    auto ple_c = (ple_proj.stride(1) == 1 && (ple_proj.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(ple_proj.data_ptr()) & 15) == 0)
        ? ple_proj : ple_proj.contiguous();
    auto res_c = (residual.stride(1) == 1 && (residual.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(residual.data_ptr()) & 15) == 0)
        ? residual : residual.contiguous();
    auto w1_c = post_ple_w.is_contiguous() ? post_ple_w : post_ple_w.contiguous();
    auto w2_c = next_norm_w.is_contiguous() ? next_norm_w : next_norm_w.contiguous();
    const bool sc_bf16 = (layer_scalar.scalar_type() == torch::kBFloat16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if ((M >= 1152 && M <= 1600) || (M >= 1920 && M <= 3072)) {
        gemma4_e2b_ple_post_norm_scalar_next_norm_warp_kernel<true><<<static_cast<int>((M + 3) / 4), 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(next_res.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(next_norm.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(res_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w1_c.data_ptr()),
            layer_scalar.data_ptr(), sc_bf16,
            reinterpret_cast<const __nv_bfloat16*>(w2_c.data_ptr()),
            static_cast<int>(M), ple_c.stride(0), res_c.stride(0), static_cast<float>(eps));
    } else {
        gemma4_e2b_ple_post_norm_scalar_next_norm_kernel<true><<<static_cast<int>(M), 192, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(next_res.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(next_norm.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(res_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w1_c.data_ptr()),
            layer_scalar.data_ptr(), sc_bf16,
            reinterpret_cast<const __nv_bfloat16*>(w2_c.data_ptr()),
            ple_c.stride(0), res_c.stride(0), static_cast<float>(eps));
    }
    return {next_res, next_norm};
}

torch::Tensor gemma4_e2b_ple_post_norm_scalar(
    torch::Tensor ple_proj,
    torch::Tensor residual,
    torch::Tensor post_ple_w,
    torch::Tensor layer_scalar,
    double eps)
{
    c10::cuda::CUDAGuard device_guard(ple_proj.device());
    TORCH_CHECK(ple_proj.size(1) == 1536, "E2B norms require hidden_size=1536");
    const int64_t M = ple_proj.size(0);
    auto next_res = torch::empty({M, 1536}, ple_proj.options());
    if (M == 0) return next_res;
    auto ple_c = (ple_proj.stride(1) == 1 && (ple_proj.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(ple_proj.data_ptr()) & 15) == 0)
        ? ple_proj : ple_proj.contiguous();
    auto res_c = (residual.stride(1) == 1 && (residual.stride(0) & 7) == 0 &&
                  (reinterpret_cast<uintptr_t>(residual.data_ptr()) & 15) == 0)
        ? residual : residual.contiguous();
    auto w1_c = post_ple_w.is_contiguous() ? post_ple_w : post_ple_w.contiguous();
    const bool sc_bf16 = (layer_scalar.scalar_type() == torch::kBFloat16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (M > 512 && M <= 2048) {
        gemma4_e2b_ple_post_norm_scalar_next_norm_warp_kernel<false><<<
            static_cast<int>((M + 3) / 4), 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(next_res.data_ptr()),
            nullptr,
            reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(res_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w1_c.data_ptr()),
            layer_scalar.data_ptr(), sc_bf16,
            nullptr,
            static_cast<int>(M), ple_c.stride(0), res_c.stride(0), static_cast<float>(eps));
    } else {
        gemma4_e2b_ple_post_norm_scalar_next_norm_kernel<false><<<static_cast<int>(M), 192, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(next_res.data_ptr()),
            nullptr,
            reinterpret_cast<const __nv_bfloat16*>(ple_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(res_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w1_c.data_ptr()),
            layer_scalar.data_ptr(), sc_bf16,
            nullptr,
            ple_c.stride(0), res_c.stride(0), static_cast<float>(eps));
    }
    return next_res;
}

torch::Tensor gemma4_e2b_embed_scale(
    torch::Tensor input_ids,
    torch::Tensor weight,
    torch::Tensor scale)
{
    c10::cuda::CUDAGuard device_guard(weight.device());
    TORCH_CHECK(input_ids.dim() == 1, "input_ids must be 1D");
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 1536, "weight must be [V, 1536]");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be CUDA bf16");
    const int64_t M = input_ids.size(0);
    auto out = torch::empty({M, 1536}, weight.options());
    if (M == 0) return out;
    auto ids_c = input_ids.is_contiguous() ? input_ids : input_ids.contiguous();
    auto w_c = weight.is_contiguous() ? weight : weight.contiguous();
    const bool sc_bf16 = (scale.scalar_type() == torch::kBFloat16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (ids_c.scalar_type() == torch::kLong) {
        gemma4_e2b_embed_scale_kernel<int64_t><<<static_cast<int>(M), 192, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            ids_c.data_ptr<int64_t>(),
            reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr()),
            scale.data_ptr(), sc_bf16,
            w_c.size(0), static_cast<int>(M));
    } else if (ids_c.scalar_type() == torch::kInt) {
        gemma4_e2b_embed_scale_kernel<int32_t><<<static_cast<int>(M), 192, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            ids_c.data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr()),
            scale.data_ptr(), sc_bf16,
            w_c.size(0), static_cast<int>(M));
    } else {
        TORCH_CHECK(false, "input_ids must be int32 or int64");
    }
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> gemma4_e2b_embed_scale_and_norm(
    torch::Tensor input_ids,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor norm_w,
    double eps)
{
    c10::cuda::CUDAGuard device_guard(weight.device());
    TORCH_CHECK(input_ids.dim() == 1, "input_ids must be 1D");
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 1536, "weight must be [V, 1536]");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be CUDA bf16");
    TORCH_CHECK(norm_w.dim() == 1 && norm_w.size(0) == 1536, "norm_w must be [1536]");
    TORCH_CHECK(norm_w.scalar_type() == torch::kBFloat16, "norm_w must be CUDA bf16");
    const int64_t M = input_ids.size(0);
    auto hidden_out = torch::empty({M, 1536}, weight.options());
    auto norm_out = torch::empty({M, 1536}, weight.options());
    if (M == 0) return {hidden_out, norm_out};
    auto ids_c = input_ids.is_contiguous() ? input_ids : input_ids.contiguous();
    auto w_c = weight.is_contiguous() ? weight : weight.contiguous();
    auto nw_c = norm_w.is_contiguous() ? norm_w : norm_w.contiguous();
    const bool sc_bf16 = (scale.scalar_type() == torch::kBFloat16);
    const int64_t num_embeddings = w_c.size(0);
    const int64_t embed_stride0 = w_c.stride(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto launch_esn = [&](auto* dummy_tag) {
        using idx_t = std::remove_pointer_t<decltype(dummy_tag)>;
        __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(hidden_out.data_ptr());
        __nv_bfloat16* np = reinterpret_cast<__nv_bfloat16*>(norm_out.data_ptr());
        const idx_t* idp = ids_c.data_ptr<idx_t>();
        const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr());
        const __nv_bfloat16* nwp = reinterpret_cast<const __nv_bfloat16*>(nw_c.data_ptr());
        if (M <= 4 || (M > 16 && M <= 128)) {
            gemma4_e2b_embed_scale_and_norm_192t_kernel<idx_t><<<static_cast<int>(M), 192, 0, stream>>>(
                hp, np, idp, wp, scale.data_ptr(), sc_bf16, nwp,
                num_embeddings, embed_stride0, static_cast<float>(eps));
        } else {
            const int rpc = (M <= 16) ? 1 : 4;
            gemma4_e2b_embed_scale_and_norm_warp_kernel<idx_t><<<
                static_cast<int>((M + rpc - 1) / rpc), rpc * 32, 0, stream>>>(
                hp, np, idp, wp, scale.data_ptr(), sc_bf16, nwp,
                static_cast<int>(M), num_embeddings, embed_stride0, static_cast<float>(eps));
        }
    };
    if (ids_c.scalar_type() == torch::kLong) {
        launch_esn(static_cast<int64_t*>(nullptr));
    } else if (ids_c.scalar_type() == torch::kInt) {
        launch_esn(static_cast<int32_t*>(nullptr));
    } else {
        TORCH_CHECK(false, "input_ids must be int32 or int64");
    }
    return {hidden_out, norm_out};
}

torch::Tensor gemma4_e2b_project_ple_combine_from_ids(
    torch::Tensor proj,
    torch::Tensor input_ids,
    torch::Tensor emb_ple_w,
    torch::Tensor norm_w,
    torch::Tensor proj_scale,
    torch::Tensor emb_scale,
    torch::Tensor in_scale,
    int64_t vocab_ple,
    int64_t layer_offset,
    double eps)
{
    c10::cuda::CUDAGuard device_guard(proj.device());
    TORCH_CHECK(proj.dim() == 2 && (proj.size(1) % 256) == 0 && proj.size(1) > 0,
                "proj must be [M, L_out*256]");
    TORCH_CHECK(input_ids.dim() == 1 && input_ids.size(0) == proj.size(0),
                "input_ids must match proj.size(0)");
    const int64_t num_out_layers = proj.size(1) / 256;
    TORCH_CHECK(layer_offset >= 0 && layer_offset + num_out_layers <= 35,
                "layer_offset + num_out_layers exceeds 35");
    const int64_t M = proj.size(0);
    auto out = torch::empty({num_out_layers, M, 256}, proj.options());
    if (M == 0) return out;
    auto proj_c = (proj.stride(1) == 1 && (proj.stride(0) & 7) == 0 &&
                   (reinterpret_cast<uintptr_t>(proj.data_ptr()) & 15) == 0)
        ? proj : proj.contiguous();
    auto ids_c = input_ids.is_contiguous() ? input_ids : input_ids.contiguous();
    auto emb_c = emb_ple_w.is_contiguous() ? emb_ple_w : emb_ple_w.contiguous();
    auto nw_c = norm_w.is_contiguous() ? norm_w : norm_w.contiguous();
    const bool ps_bf16 = (proj_scale.scalar_type() == torch::kBFloat16);
    const bool es_bf16 = (emb_scale.scalar_type() == torch::kBFloat16);
    const bool is_bf16 = (in_scale.scalar_type() == torch::kBFloat16);
    const int64_t vocab_limit = std::min<int64_t>(vocab_ple, emb_c.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto launch_pfi = [&](auto* dummy_tag) {
        using idx_t = std::remove_pointer_t<decltype(dummy_tag)>;
        __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
        const __nv_bfloat16* pp = reinterpret_cast<const __nv_bfloat16*>(proj_c.data_ptr());
        const idx_t* idp = ids_c.data_ptr<idx_t>();
        const __nv_bfloat16* ep = reinterpret_cast<const __nv_bfloat16*>(emb_c.data_ptr());
        const __nv_bfloat16* nwp = reinterpret_cast<const __nv_bfloat16*>(nw_c.data_ptr());
        if (M >= 48) {
            dim3 grid(static_cast<int>((M + 3) / 4), static_cast<int>(num_out_layers));
            gemma4_e2b_project_ple_combine_from_ids_coal_kernel<idx_t><<<grid, 128, 0, stream>>>(
                op, pp, idp, ep, nwp,
                proj_scale.data_ptr(), ps_bf16,
                emb_scale.data_ptr(), es_bf16,
                in_scale.data_ptr(), is_bf16,
                static_cast<int>(M), static_cast<int64_t>(layer_offset),
                proj_c.stride(0), emb_c.stride(0), vocab_limit,
                static_cast<float>(eps));
        } else if (M <= 4) {
            const int blocks = static_cast<int>((M * num_out_layers + 3) / 4);
            gemma4_e2b_project_ple_combine_from_ids_kernel<idx_t, 4, true><<<blocks, 128, 0, stream>>>(
                op, pp, idp, ep, nwp,
                proj_scale.data_ptr(), ps_bf16,
                emb_scale.data_ptr(), es_bf16,
                in_scale.data_ptr(), is_bf16,
                static_cast<int>(M), static_cast<int>(num_out_layers),
                static_cast<int>(layer_offset),
                proj_c.stride(0), emb_c.stride(0), vocab_limit,
                static_cast<float>(eps));
        } else {
            const int blocks = static_cast<int>((M * num_out_layers + 7) / 8);
            gemma4_e2b_project_ple_combine_from_ids_kernel<idx_t, 8, false><<<blocks, 256, 0, stream>>>(
                op, pp, idp, ep, nwp,
                proj_scale.data_ptr(), ps_bf16,
                emb_scale.data_ptr(), es_bf16,
                in_scale.data_ptr(), is_bf16,
                static_cast<int>(M), static_cast<int>(num_out_layers),
                static_cast<int>(layer_offset),
                proj_c.stride(0), emb_c.stride(0), vocab_limit,
                static_cast<float>(eps));
        }
    };
    if (ids_c.scalar_type() == torch::kLong) {
        launch_pfi(static_cast<int64_t*>(nullptr));
    } else if (ids_c.scalar_type() == torch::kInt) {
        launch_pfi(static_cast<int32_t*>(nullptr));
    } else {
        TORCH_CHECK(false, "input_ids must be int32 or int64");
    }
    return out;
}

TORCH_LIBRARY_FRAGMENT(vllm, m) {
    m.def(
        "gemma4_e2b_project_ple_combine(Tensor proj, Tensor? ple_in, Tensor norm_w, "
        "Tensor proj_scale, Tensor in_scale, float eps) -> Tensor");
    m.def(
        "gemma4_e2b_post_attn_pre_ffn_norms(Tensor attn_out, Tensor residual, "
        "Tensor post_attn_w, Tensor pre_ffn_w, float eps) -> (Tensor, Tensor)");
    m.def(
        "gemma4_e2b_post_ffn_res_norm(Tensor mlp_out, Tensor residual, "
        "Tensor post_ffn_w, float eps) -> Tensor");
    m.def(
        "gemma4_e2b_ple_gate_gelu_mul(Tensor hidden, Tensor w_gate, "
        "Tensor ple_in) -> Tensor");
    m.def(
        "gemma4_e2b_ple_proj_gemv(Tensor gated_ple, Tensor w_proj) -> Tensor");
    m.def(
        "gemma4_e2b_ple_post_norm_scalar_next_norm(Tensor ple_proj, Tensor residual, "
        "Tensor post_ple_w, Tensor layer_scalar, Tensor next_norm_w, float eps) "
        "-> (Tensor, Tensor)");
    m.def(
        "gemma4_e2b_ple_post_norm_scalar(Tensor ple_proj, Tensor residual, "
        "Tensor post_ple_w, Tensor layer_scalar, float eps) -> Tensor");
    m.def(
        "gemma4_e2b_model_proj_gemv(Tensor x, Tensor w_proj) -> Tensor");
    m.def(
        "gemma4_e2b_embed_scale(Tensor input_ids, Tensor weight, Tensor scale) -> Tensor");
    m.def(
        "gemma4_e2b_project_ple_combine_from_ids(Tensor proj, Tensor input_ids, "
        "Tensor emb_ple_w, Tensor norm_w, Tensor proj_scale, Tensor emb_scale, "
        "Tensor in_scale, int vocab_ple, int layer_offset, float eps) -> Tensor");
    m.def(
        "gemma4_e2b_post_ffn_ple_gate(Tensor mlp_out, Tensor residual, "
        "Tensor post_ffn_w, Tensor w_gate, Tensor ple_in, float eps) -> (Tensor, Tensor)");
    m.def(
        "gemma4_e2b_embed_scale_and_norm(Tensor input_ids, Tensor weight, "
        "Tensor scale, Tensor norm_w, float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(vllm, CUDA, m) {
    m.impl("gemma4_e2b_project_ple_combine", &gemma4_e2b_project_ple_combine);
    m.impl("gemma4_e2b_post_attn_pre_ffn_norms", &gemma4_e2b_post_attn_pre_ffn_norms);
    m.impl("gemma4_e2b_post_ffn_res_norm", &gemma4_e2b_post_ffn_res_norm);
    m.impl("gemma4_e2b_ple_gate_gelu_mul", &gemma4_e2b_ple_gate_gelu_mul);
    m.impl("gemma4_e2b_ple_proj_gemv", &gemma4_e2b_ple_proj_gemv);
    m.impl(
        "gemma4_e2b_ple_post_norm_scalar_next_norm",
        &gemma4_e2b_ple_post_norm_scalar_next_norm);
    m.impl("gemma4_e2b_ple_post_norm_scalar", &gemma4_e2b_ple_post_norm_scalar);
    m.impl("gemma4_e2b_model_proj_gemv", &gemma4_e2b_model_proj_gemv);
    m.impl("gemma4_e2b_embed_scale", &gemma4_e2b_embed_scale);
    m.impl(
        "gemma4_e2b_project_ple_combine_from_ids",
        &gemma4_e2b_project_ple_combine_from_ids);
    m.impl("gemma4_e2b_post_ffn_ple_gate", &gemma4_e2b_post_ffn_ple_gate);
    m.impl("gemma4_e2b_embed_scale_and_norm", &gemma4_e2b_embed_scale_and_norm);
}
"""

_EXT_HASH = hashlib.sha256((_CUDA_SRC + _CPP_SRC).encode("utf-8")).hexdigest()[:12]
_EXT_MOD: ModuleType | None = None
_FAKE_REGISTERED = False


def _resolve_cache_dir() -> str:
    ext_dir = os.environ.get("VLLM_GEMMA4_SM80_EXT_DIR")
    if ext_dir:
        return ext_dir
    return os.path.join(os.path.expanduser("~"), ".cache", "vllm", "gemma4_sm80_ext")


def _register_e2b_sm80_fakes() -> None:
    global _FAKE_REGISTERED
    if _FAKE_REGISTERED:
        return
    _FAKE_REGISTERED = True

    @torch.library.register_fake("vllm::gemma4_e2b_project_ple_combine")
    def gemma4_e2b_project_ple_combine_fake(
        proj: torch.Tensor,
        ple_in: torch.Tensor | None,
        norm_w: torch.Tensor,
        proj_scale: torch.Tensor,
        in_scale: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return proj.new_empty((35, proj.size(0), 256))

    @torch.library.register_fake("vllm::gemma4_e2b_post_attn_pre_ffn_norms")
    def gemma4_e2b_post_attn_pre_ffn_norms_fake(
        attn_out: torch.Tensor,
        residual: torch.Tensor,
        post_attn_w: torch.Tensor,
        pre_ffn_w: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = attn_out.size(0)
        return attn_out.new_empty((m, 1536)), attn_out.new_empty((m, 1536))

    @torch.library.register_fake("vllm::gemma4_e2b_post_ffn_res_norm")
    def gemma4_e2b_post_ffn_res_norm_fake(
        mlp_out: torch.Tensor,
        residual: torch.Tensor,
        post_ffn_w: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return mlp_out.new_empty((mlp_out.size(0), 1536))

    @torch.library.register_fake("vllm::gemma4_e2b_ple_gate_gelu_mul")
    def gemma4_e2b_ple_gate_gelu_mul_fake(
        hidden: torch.Tensor,
        w_gate: torch.Tensor,
        ple_in: torch.Tensor,
    ) -> torch.Tensor:
        return hidden.new_empty((hidden.size(0), 256))

    @torch.library.register_fake("vllm::gemma4_e2b_ple_proj_gemv")
    def gemma4_e2b_ple_proj_gemv_fake(
        gated_ple: torch.Tensor,
        w_proj: torch.Tensor,
    ) -> torch.Tensor:
        return gated_ple.new_empty((gated_ple.size(0), 1536))

    @torch.library.register_fake("vllm::gemma4_e2b_ple_post_norm_scalar_next_norm")
    def gemma4_e2b_ple_post_norm_scalar_next_norm_fake(
        ple_proj: torch.Tensor,
        residual: torch.Tensor,
        post_ple_w: torch.Tensor,
        layer_scalar: torch.Tensor,
        next_norm_w: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = ple_proj.size(0)
        return ple_proj.new_empty((m, 1536)), ple_proj.new_empty((m, 1536))

    @torch.library.register_fake("vllm::gemma4_e2b_ple_post_norm_scalar")
    def gemma4_e2b_ple_post_norm_scalar_fake(
        ple_proj: torch.Tensor,
        residual: torch.Tensor,
        post_ple_w: torch.Tensor,
        layer_scalar: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return ple_proj.new_empty((ple_proj.size(0), 1536))

    @torch.library.register_fake("vllm::gemma4_e2b_model_proj_gemv")
    def gemma4_e2b_model_proj_gemv_fake(
        x: torch.Tensor,
        w_proj: torch.Tensor,
    ) -> torch.Tensor:
        return x.new_empty((x.size(0), 8960))

    @torch.library.register_fake("vllm::gemma4_e2b_embed_scale")
    def gemma4_e2b_embed_scale_fake(
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return weight.new_empty((input_ids.size(0), 1536))

    @torch.library.register_fake("vllm::gemma4_e2b_project_ple_combine_from_ids")
    def gemma4_e2b_project_ple_combine_from_ids_fake(
        proj: torch.Tensor,
        input_ids: torch.Tensor,
        emb_ple_w: torch.Tensor,
        norm_w: torch.Tensor,
        proj_scale: torch.Tensor,
        emb_scale: torch.Tensor,
        in_scale: torch.Tensor,
        vocab_ple: int,
        layer_offset: int,
        eps: float,
    ) -> torch.Tensor:
        return proj.new_empty((proj.size(1) // 256, proj.size(0), 256))

    @torch.library.register_fake("vllm::gemma4_e2b_post_ffn_ple_gate")
    def gemma4_e2b_post_ffn_ple_gate_fake(
        mlp_out: torch.Tensor,
        residual: torch.Tensor,
        post_ffn_w: torch.Tensor,
        w_gate: torch.Tensor,
        ple_in: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = mlp_out.size(0)
        return mlp_out.new_empty((m, 1536)), mlp_out.new_empty((m, 256))

    @torch.library.register_fake("vllm::gemma4_e2b_embed_scale_and_norm")
    def gemma4_e2b_embed_scale_and_norm_fake(
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = input_ids.size(0)
        return weight.new_empty((m, 1536)), weight.new_empty((m, 1536))


def get_gemma4_e2b_sm80_ext() -> ModuleType:
    global _EXT_MOD
    if _EXT_MOD is not None:
        _register_e2b_sm80_fakes()
        return _EXT_MOD

    name = f"gemma4_e2b_sm80_ops_{_EXT_HASH}"
    build_dir = os.path.join(_resolve_cache_dir(), name)
    so_path = os.path.join(build_dir, f"{name}.so")
    if os.path.isfile(so_path):
        spec = importlib.util.spec_from_file_location(name, so_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EXT_MOD = mod
        _register_e2b_sm80_fakes()
        return _EXT_MOD

    if os.environ.get("VLLM_GEMMA4_SM80_NO_COMPILE", "0") == "1":
        raise RuntimeError(
            f"VLLM_GEMMA4_SM80_NO_COMPILE=1 but prebuilt extension missing: {so_path}"
        )
    os.makedirs(build_dir, exist_ok=True)
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    _EXT_MOD = load_inline(
        name=name,
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=[
            "gemma4_e2b_project_ple_combine",
            "gemma4_e2b_post_attn_pre_ffn_norms",
            "gemma4_e2b_post_ffn_res_norm",
            "gemma4_e2b_ple_gate_gelu_mul",
            "gemma4_e2b_ple_proj_gemv",
            "gemma4_e2b_ple_post_norm_scalar_next_norm",
            "gemma4_e2b_ple_post_norm_scalar",
            "gemma4_e2b_model_proj_gemv",
            "gemma4_e2b_embed_scale",
            "gemma4_e2b_project_ple_combine_from_ids",
            "gemma4_e2b_post_ffn_ple_gate",
            "gemma4_e2b_embed_scale_and_norm",
        ],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "-arch=sm_80", "-std=c++20"],
        with_cuda=True,
    )
    _register_e2b_sm80_fakes()
    return _EXT_MOD


get_gemma4_e2b_sm80_ext()
