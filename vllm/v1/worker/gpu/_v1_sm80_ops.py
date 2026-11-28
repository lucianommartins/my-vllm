# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 AOT C++/CUDA fused V1 decode input-prep, post-update, and async copy ops."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from typing import Any

import torch

_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern void launch_prepare_decode_all(
    cudaStream_t stream,
    int32_t* input_ids,
    int64_t* positions,
    int32_t* seq_lens,
    int32_t seq_lens_size,
    int32_t* idx_mapping_i32,
    const int64_t* idx_mapping,
    const int32_t* num_computed,
    const int64_t* last_sampled,
    const int32_t* prefill_len,
    const uint64_t* src_ptrs,
    const uint64_t* dst_ptrs,
    const int64_t* src_strides,
    const int64_t* dst_strides,
    const int32_t* num_blocks,
    int64_t num_blocks_stride,
    const int32_t* sw_reach,
    const int32_t* block_sizes,
    const uint8_t* slot_enabled,
    int64_t* slot_mappings,
    int64_t slot_stride,
    int32_t num_reqs,
    int32_t num_reqs_padded,
    int32_t bt_rows,
    int32_t clear_end,
    int32_t num_groups,
    int32_t dcp_size,
    int32_t dcp_rank,
    int32_t cp_interleave);

extern void launch_post_update_sm80(
    cudaStream_t stream,
    const int64_t* idx_mapping,
    int32_t* num_computed,
    int64_t* last_sampled,
    const int64_t* sampled,
    int64_t sampled_stride,
    const int32_t* num_sampled,
    const int32_t* query_start_loc,
    int32_t* all_token_ids,
    int64_t all_token_stride,
    int32_t* total_len,
    int32_t num_reqs);

void prepare_decode_all(
    torch::Tensor input_ids,
    torch::Tensor positions,
    torch::Tensor seq_lens,
    torch::Tensor idx_mapping_i32,
    torch::Tensor idx_mapping,
    torch::Tensor num_computed,
    torch::Tensor last_sampled,
    torch::Tensor prefill_len,
    torch::Tensor src_ptrs,
    torch::Tensor dst_ptrs,
    torch::Tensor src_strides,
    torch::Tensor dst_strides,
    torch::Tensor num_blocks,
    torch::Tensor sw_reach,
    torch::Tensor block_sizes,
    torch::Tensor slot_enabled,
    torch::Tensor slot_mappings,
    int64_t num_reqs,
    int64_t num_reqs_padded,
    int64_t bt_rows,
    int64_t clear_end,
    int64_t dcp_size,
    int64_t dcp_rank,
    int64_t cp_interleave)
{
    TORCH_CHECK(last_sampled.stride(0) == 1, "last_sampled stride(0) must be 1");
    TORCH_CHECK(positions.dim() == 1 && positions.scalar_type() == torch::kInt64, "positions must be 1D int64");
    TORCH_CHECK(input_ids.scalar_type() == torch::kInt32, "input_ids must be int32");
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(idx_mapping.scalar_type() == torch::kInt64, "idx_mapping must be int64");
    TORCH_CHECK(last_sampled.scalar_type() == torch::kInt64, "last_sampled must be int64");
    TORCH_CHECK(slot_enabled.scalar_type() == torch::kBool && slot_enabled.is_contiguous(),
                "slot_enabled must be contiguous bool");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_prepare_decode_all(
        stream,
        input_ids.data_ptr<int32_t>(),
        positions.data_ptr<int64_t>(),
        seq_lens.data_ptr<int32_t>(),
        static_cast<int32_t>(seq_lens.size(0)),
        idx_mapping_i32.data_ptr<int32_t>(),
        idx_mapping.data_ptr<int64_t>(),
        num_computed.data_ptr<int32_t>(),
        last_sampled.data_ptr<int64_t>(),
        prefill_len.data_ptr<int32_t>(),
        reinterpret_cast<const uint64_t*>(src_ptrs.data_ptr()),
        reinterpret_cast<const uint64_t*>(dst_ptrs.data_ptr()),
        src_strides.data_ptr<int64_t>(),
        dst_strides.data_ptr<int64_t>(),
        num_blocks.data_ptr<int32_t>(),
        num_blocks.stride(0),
        sw_reach.data_ptr<int32_t>(),
        block_sizes.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t*>(slot_enabled.data_ptr()),
        slot_mappings.data_ptr<int64_t>(),
        slot_mappings.stride(0),
        static_cast<int32_t>(num_reqs),
        static_cast<int32_t>(num_reqs_padded),
        static_cast<int32_t>(bt_rows),
        static_cast<int32_t>(clear_end),
        static_cast<int32_t>(src_ptrs.size(0)),
        static_cast<int32_t>(dcp_size),
        static_cast<int32_t>(dcp_rank),
        static_cast<int32_t>(cp_interleave));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "prepare_decode_all CUDA error: ", cudaGetErrorString(err));
}

void post_update_sm80(
    torch::Tensor idx_mapping,
    torch::Tensor num_computed,
    torch::Tensor last_sampled,
    torch::Tensor sampled,
    c10::optional<torch::Tensor> num_sampled,
    c10::optional<torch::Tensor> query_start_loc,
    torch::Tensor all_token_ids,
    torch::Tensor total_len)
{
    TORCH_CHECK(last_sampled.stride(0) == 1 && last_sampled.scalar_type() == torch::kInt64,
                "last_sampled must be contiguous int64");
    TORCH_CHECK(sampled.scalar_type() == torch::kInt64, "sampled must be int64");
    TORCH_CHECK(idx_mapping.scalar_type() == torch::kInt64, "idx_mapping must be int64");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int32_t* ns_ptr = num_sampled.has_value() ? num_sampled->data_ptr<int32_t>() : nullptr;
    const int32_t* qsl_ptr = query_start_loc.has_value() ? query_start_loc->data_ptr<int32_t>() : nullptr;
    launch_post_update_sm80(
        stream,
        idx_mapping.data_ptr<int64_t>(),
        num_computed.data_ptr<int32_t>(),
        last_sampled.data_ptr<int64_t>(),
        sampled.data_ptr<int64_t>(),
        sampled.stride(0),
        ns_ptr,
        qsl_ptr,
        all_token_ids.data_ptr<int32_t>(),
        all_token_ids.stride(0),
        total_len.data_ptr<int32_t>(),
        static_cast<int32_t>(idx_mapping.size(0)));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "post_update_sm80 CUDA error: ", cudaGetErrorString(err));
}

void async_copy_sampled_and_record(
    torch::Tensor sampled_gpu,
    torch::Tensor dst_cpu,
    int64_t n,
    int64_t main_stream_ptr,
    int64_t copy_stream_ptr,
    int64_t wait_event_ptr,
    int64_t copy_event_ptr)
{
    TORCH_CHECK(sampled_gpu.scalar_type() == torch::kInt64 && sampled_gpu.stride(0) == 1,
                "sampled_gpu must be int64 with stride(0)==1");
    TORCH_CHECK(dst_cpu.scalar_type() == torch::kInt64 && dst_cpu.is_contiguous(),
                "dst_cpu must be contiguous int64");
    TORCH_CHECK(n >= 0 && n <= sampled_gpu.size(0) && n <= dst_cpu.size(0),
                "n out of bounds for sampled_gpu/dst_cpu");
    cudaStream_t ms = reinterpret_cast<cudaStream_t>(main_stream_ptr);
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(copy_stream_ptr);
    cudaEvent_t wev = reinterpret_cast<cudaEvent_t>(wait_event_ptr);
    cudaEvent_t cev = reinterpret_cast<cudaEvent_t>(copy_event_ptr);
    TORCH_CHECK(wev != nullptr && cev != nullptr, "CUDA event handle not initialized");
    cudaError_t err = cudaEventRecord(wev, ms);
    TORCH_CHECK(err == cudaSuccess, "cudaEventRecord(wait) failed: ", cudaGetErrorString(err));
    err = cudaStreamWaitEvent(cs, wev, 0);
    TORCH_CHECK(err == cudaSuccess, "cudaStreamWaitEvent failed: ", cudaGetErrorString(err));
    if (n > 0) {
        err = cudaMemcpyAsync(
            dst_cpu.data_ptr(),
            sampled_gpu.data_ptr(),
            static_cast<size_t>(n) * sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            cs);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync D2H failed: ", cudaGetErrorString(err));
    }
    err = cudaEventRecord(cev, cs);
    TORCH_CHECK(err == cudaSuccess, "cudaEventRecord(copy) failed: ", cudaGetErrorString(err));
}
"""

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void _prepare_decode_all_kernel(
    int32_t* __restrict__ input_ids,
    int64_t* __restrict__ positions,
    int32_t* __restrict__ seq_lens,
    int32_t seq_lens_size,
    int32_t* __restrict__ idx_mapping_i32,
    const int64_t* __restrict__ idx_mapping,
    const int32_t* __restrict__ num_computed,
    const int64_t* __restrict__ last_sampled,
    const int32_t* __restrict__ prefill_len,
    const uint64_t* __restrict__ src_ptrs,
    const uint64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ src_strides,
    const int64_t* __restrict__ dst_strides,
    const int32_t* __restrict__ num_blocks,
    int64_t num_blocks_stride,
    const int32_t* __restrict__ sw_reach,
    const int32_t* __restrict__ block_sizes,
    const uint8_t* __restrict__ slot_enabled,
    int64_t* __restrict__ slot_mappings,
    int64_t slot_stride,
    int32_t num_reqs,
    int32_t num_reqs_padded,
    int32_t bt_rows,
    int32_t clear_end,
    int32_t dcp_size,
    int32_t dcp_rank,
    int32_t cp_interleave)
{
    const int r = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;

    // Zero tail of seq_lens[num_reqs..seq_lens_size) once on block (0, 0).
    if (g == 0 && r == 0 && seq_lens_size > num_reqs) {
        for (int idx = num_reqs + tid; idx < seq_lens_size; idx += blockDim.x) {
            seq_lens[idx] = 0;
        }
    }

    // Clear trailing slot_mappings if clear_end > num_reqs_padded.
    if (r == 0 && clear_end > num_reqs_padded) {
        int64_t* sm_g = slot_mappings + g * slot_stride;
        for (int idx = num_reqs_padded + tid; idx < clear_end; idx += blockDim.x) {
            sm_g[idx] = -1;
        }
    }

    if (r >= num_reqs) {
        if (tid == 0 && r < num_reqs_padded) {
            slot_mappings[g * slot_stride + r] = -1;
        }
        if (r < bt_rows) {
            const int32_t dst_stride = static_cast<int32_t>(__ldg(dst_strides + g));
            int32_t* __restrict__ dst =
                reinterpret_cast<int32_t*>(__ldg(dst_ptrs + g)) + r * dst_stride;
            for (int i = tid; i < dst_stride; i += blockDim.x) {
                dst[i] = 0;
            }
        }
        return;
    }

    const int64_t req_idx = __ldg(idx_mapping + r);
    if (req_idx < 0) {
        if (g == 0 && tid == 0) {
            idx_mapping_i32[r] = static_cast<int32_t>(req_idx);
            seq_lens[r] = 0;
            positions[r] = 0;
        }
        if (tid == 0) {
            slot_mappings[g * slot_stride + r] = -1;
        }
        return;
    }

    const int32_t nc = __ldg(num_computed + req_idx);

    if (g == 0 && tid == 0) {
        idx_mapping_i32[r] = static_cast<int32_t>(req_idx);
        seq_lens[r] = nc + 1;
        positions[r] = static_cast<int64_t>(nc);
        if (nc >= __ldg(prefill_len + req_idx)) {
            input_ids[r] = static_cast<int32_t>(__ldg(last_sampled + req_idx));
        }
    }

    const int32_t dst_stride = static_cast<int32_t>(__ldg(dst_strides + g));
    const int32_t n_blks_raw = __ldg(num_blocks + g * num_blocks_stride + req_idx);
    const int32_t n_blks = n_blks_raw < dst_stride ? n_blks_raw : dst_stride;
    const int32_t bs = __ldg(block_sizes + g);
    const int32_t* __restrict__ src =
        reinterpret_cast<const int32_t*>(__ldg(src_ptrs + g)) +
        req_idx * __ldg(src_strides + g);
    int32_t* __restrict__ dst =
        reinterpret_cast<int32_t*>(__ldg(dst_ptrs + g)) + r * dst_stride;

    if (tid == 0) {
        int64_t slot = -1;
        if (__ldg(slot_enabled + g)) {
            const int32_t pos = nc;
            int32_t b_idx = pos / bs;
            int32_t b_off = pos % bs;
            if (dcp_size > 1) {
                const int32_t dcp_bs = bs * dcp_size;
                const int32_t sub = pos % dcp_bs;
                if ((sub / cp_interleave) % dcp_size == dcp_rank) {
                    const int32_t local_sub =
                        (sub / (cp_interleave * dcp_size)) * cp_interleave +
                        (sub % cp_interleave);
                    b_idx = pos / dcp_bs;
                    b_off = local_sub;
                } else {
                    b_idx = -1;
                }
            }
            if (b_idx >= 0 && b_idx < n_blks_raw) {
                const int32_t blk_num = __ldg(src + b_idx);
                slot = static_cast<int64_t>(blk_num) * static_cast<int64_t>(bs) +
                       static_cast<int64_t>(b_off);
            }
        }
        slot_mappings[g * slot_stride + r] = slot;
    }

    const int32_t reach = __ldg(sw_reach + g);
    int32_t null_limit = 0;
    if (reach > 0 && nc > reach) {
        null_limit = (nc - reach) / bs;
    }

    const bool aligned =
        ((reinterpret_cast<uintptr_t>(src) & 15) == 0) &&
        ((reinterpret_cast<uintptr_t>(dst) & 15) == 0);
    if (aligned && null_limit == 0) {
        const int vec_n = n_blks >> 2;
        const int4* src4 = reinterpret_cast<const int4*>(src);
        int4* dst4 = reinterpret_cast<int4*>(dst);
        for (int i = tid; i < vec_n; i += blockDim.x) {
            dst4[i] = __ldg(src4 + i);
        }
        for (int i = (vec_n << 2) + tid; i < n_blks; i += blockDim.x) {
            dst[i] = __ldg(src + i);
        }
    } else {
        for (int i = tid; i < n_blks; i += blockDim.x) {
            dst[i] = (i < null_limit) ? 0 : __ldg(src + i);
        }
    }
}

__global__ void _post_update_sm80_kernel(
    const int64_t* __restrict__ idx_mapping,
    int32_t* __restrict__ num_computed,
    int64_t* __restrict__ last_sampled,
    const int64_t* __restrict__ sampled,
    int64_t sampled_stride,
    const int32_t* __restrict__ num_sampled,
    const int32_t* __restrict__ query_start_loc,
    int32_t* __restrict__ all_token_ids,
    int64_t all_token_stride,
    int32_t* __restrict__ total_len,
    int32_t num_reqs)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_reqs) return;
    const int64_t req_idx = idx_mapping[r];
    if (req_idx < 0) return;

    const int32_t ns = num_sampled ? num_sampled[r] : 1;
    const int32_t qlen =
        query_start_loc ? (query_start_loc[r + 1] - query_start_loc[r]) : 0;
    num_computed[req_idx] += qlen;

    if (ns > 0) {
        const int64_t tok = sampled[r * sampled_stride];
        const int32_t tl = total_len[req_idx];
        last_sampled[req_idx] = tok;
        all_token_ids[req_idx * all_token_stride + tl] = static_cast<int32_t>(tok);
        total_len[req_idx] = tl + 1;
    }
}

void launch_prepare_decode_all(
    cudaStream_t stream,
    int32_t* input_ids, int64_t* positions, int32_t* seq_lens, int32_t seq_lens_size,
    int32_t* idx_mapping_i32, const int64_t* idx_mapping,
    const int32_t* num_computed, const int64_t* last_sampled, const int32_t* prefill_len,
    const uint64_t* src_ptrs, const uint64_t* dst_ptrs,
    const int64_t* src_strides, const int64_t* dst_strides,
    const int32_t* num_blocks, int64_t num_blocks_stride,
    const int32_t* sw_reach, const int32_t* block_sizes, const uint8_t* slot_enabled,
    int64_t* slot_mappings, int64_t slot_stride,
    int32_t num_reqs, int32_t num_reqs_padded, int32_t bt_rows, int32_t clear_end, int32_t num_groups,
    int32_t dcp_size, int32_t dcp_rank, int32_t cp_interleave)
{
    int grid_x = num_reqs_padded > bt_rows ? num_reqs_padded : bt_rows;
    if (grid_x < 1) grid_x = 1;
    dim3 grid(grid_x, num_groups);
    _prepare_decode_all_kernel<<<grid, 128, 0, stream>>>(
        input_ids, positions, seq_lens, seq_lens_size, idx_mapping_i32,
        idx_mapping, num_computed, last_sampled, prefill_len,
        src_ptrs, dst_ptrs, src_strides, dst_strides,
        num_blocks, num_blocks_stride,
        sw_reach, block_sizes, slot_enabled,
        slot_mappings, slot_stride,
        num_reqs, num_reqs_padded, bt_rows, clear_end,
        dcp_size, dcp_rank, cp_interleave);
}

void launch_post_update_sm80(
    cudaStream_t stream,
    const int64_t* idx_mapping,
    int32_t* num_computed,
    int64_t* last_sampled,
    const int64_t* sampled,
    int64_t sampled_stride,
    const int32_t* num_sampled,
    const int32_t* query_start_loc,
    int32_t* all_token_ids,
    int64_t all_token_stride,
    int32_t* total_len,
    int32_t num_reqs)
{
    if (num_reqs <= 0) return;
    const int threads = 64;
    const int blocks = (num_reqs + threads - 1) / threads;
    _post_update_sm80_kernel<<<blocks, threads, 0, stream>>>(
        idx_mapping, num_computed, last_sampled, sampled, sampled_stride,
        num_sampled, query_start_loc, all_token_ids, all_token_stride,
        total_len, num_reqs);
}
"""

_SRC_HASH = hashlib.sha256((_CPP_SRC + _CUDA_SRC).encode("utf-8")).hexdigest()[:12]
_EXT_NAME = f"v1_sm80_ops_{_SRC_HASH}"
_CACHED_EXT: Any = None


def _candidate_ext_dirs() -> list[str]:
    dirs: list[str] = []
    if d := os.environ.get("VLLM_GEMMA4_SM80_EXT_DIR"):
        dirs.append(d)
    if c := os.environ.get("VLLM_CACHE_ROOT"):
        dirs.append(os.path.join(c, "v1_sm80_ext"))
    dirs.append(
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "vllm_v1_sm80",
        )
    )
    return dirs


def _find_prebuilt_so() -> str | None:
    so_name = f"{_EXT_NAME}.so"
    for d in _candidate_ext_dirs():
        candidate = os.path.join(d, so_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_so_direct(so_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(_EXT_NAME, so_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_ext(allow_compile: bool = True) -> Any | None:
    """Return compiled SM80 V1 extension module (zero-JIT when prebuilt `.so` exists)."""
    global _CACHED_EXT
    if _CACHED_EXT is not None:
        return _CACHED_EXT

    so_path = _find_prebuilt_so()
    if so_path is not None:
        _CACHED_EXT = _load_so_direct(so_path)
        if _CACHED_EXT is not None:
            return _CACHED_EXT

    if not allow_compile:
        return None

    py_bin = os.path.dirname(sys.executable)
    cur_path = os.environ.get("PATH", "")
    parts = cur_path.split(":") if cur_path else []
    for p in ("/usr/local/cuda/bin", py_bin):
        if p and os.path.isdir(p) and p not in parts:
            parts.insert(0, p)
    os.environ["PATH"] = ":".join(parts)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

    build_dir = _candidate_ext_dirs()[0]
    os.makedirs(build_dir, exist_ok=True)

    from torch.utils.cpp_extension import load_inline

    _ext_name = _EXT_NAME + ("_sm90a" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9) else "")
    _CACHED_EXT = load_inline(
        name=_ext_name,
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=[
            "prepare_decode_all",
            "post_update_sm80",
            "async_copy_sampled_and_record",
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90a" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9) else "-arch=sm_80"],
        build_directory=build_dir,
        verbose=False,
    )
    return _CACHED_EXT
