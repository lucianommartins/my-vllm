# GEMMA_ATTN H100 Kernel Development — Session Checkpoint

**Date**: 2026-06-26
**Branch**: `lucianommartins/gemma-attn`
**Codebase**: `/projects/gemma4-vllm/my-vllm-gemma-attn`
**Venv**: `/projects/gemma4-vllm/gemma-attn-env`
**Hardware**: 4× NVIDIA H100-SXM-80GB (sm_90), nvcc 12.8, torch 2.11+cu128

---

## 1. What Was Built

### SM90 Build Scaffolding (DONE, committed)
- `CMakeLists.txt`: conditional `sm_90a` compilation block for gemma attention SM90 files
- `gemma_paged_attention.cu`: `#if ENABLE_GEMMA_ATTN_SM90` dispatch (forward declaration + runtime arch check)
- `gemma_prefill_attention.cu`: same pattern
- `gemma_paged_attention_sm90.cu/.cuh`: SM90 decode stubs (return false → SM80 fallthrough)
- `gemma_prefill_attention_sm90.cu/.cuh`: SM90 prefill (currently opt-in via `GEMMA_SM90_PREFILL=1`)

### CUTLASS FMHA Fork (DONE, compiles, not yet wired to launch)
- `csrc/libtorch_stable/attention/fmha_sm90/` — 13 files forked from CUTLASS example 88
- All include paths fixed (flat directory, no `../collective/` references)
- Compiles cleanly within the vLLM build system (verified: step 45/366 in build log)
- Include paths: `CUTLASS_INCLUDE_DIR` and `CUTLASS_TOOLS_UTIL_INCLUDE_DIR` already on `_C_stable_libtorch` target

### Benchmark Harnesses (DONE, in tests-evals repo)
- `/tmp/measure_e2e.py` — e2e tok/s comparison (GEMMA_ATTN vs FLASH_ATTN)
- `/tmp/decode_bw.py` — decode bandwidth microbench
- `/tmp/prefill_microbench.py` — prefill throughput isolation
- Also saved to `tests-evals/tests/benchmarks/performance/gemma_attn/`

---

## 2. Baseline Performance (12B, H100, CUDA graphs)

**GEMMA_ATTN (SM80 wmma kernel running on H100) vs FLASH_ATTN (FA4):**

| Regime | GEMMA_ATTN | FA4 | Delta |
|---|---|---|---|
| decode-short (512/128/b8) | 533.0 | 566.7 | -5.9% |
| decode-vlong (8K/512/b24) | 569.3 | 716.7 | -20.6% |
| prefill-short (2K/32/b4) | 152.4 | 176.6 | -13.7% |
| prefill-med (4K/32/b8) | 117.7 | 150.4 | -21.7% |
| prefill-long (8K/32/b8) | 58.8 | 83.1 | -29.2% |
| context-12K | 194.6 | 261.8 | -25.7% |

**Root cause**: SM80 wmma kernels leave H100 features unused (wgmma, TMA, warp specialization). FA4 uses all of them.

---

## 3. What Was Tried and Why It Failed

### Decode Kernel Attempts
| Approach | Result | Root Cause |
|---|---|---|
| SIMT 3-stage pipeline (BN=16) | Regressed | Pipeline overhead > latency benefit |
| SIMT wider tile (BN=32) | Flat | Occupancy drop (80 regs, 3 CTAs vs 4) cancelled tile benefit |
| wgmma for decode | Not viable | wgmma M=64 minimum vs decode BDY=2 → 97% wasted compute |

**Conclusion**: SM80 SIMT decode is near-optimal for scalar vector-matrix attention. Real decode improvement requires TMA descriptor-based loads (hardware address generation replacing per-thread `bt[g/page_size]`). Parked.

### Prefill Kernel Attempts
| Approach | Result | Root Cause |
|---|---|---|
| 8 warps / MIN_CTA=3 (SM90 native) | Regressed | REG:80 STACK:200 — spill killed occupancy gain |
| 16 warps / MIN_CTA=2 (SM90 native codegen) | Flat | Identical codegen to SM80 binary |
| QK-load overlap (double-buffered sKV) | Not tested | STACK:80 (2× v2), bottleneck is FMA/softmax not load latency |
| Head-chunked wmma (BM_PAD=64, 2-pass) | -26% vs FA4 | 50% padding waste + 2× PV work |
| O-in-smem v1 (atomicAdd) | -59% vs FA4 | atomicAdd smem contention |
| O-in-smem v2 (store_matrix_sync + scatter) | -41% vs FA4 | Cooperative scatter + 16 sync barriers per KV tile |

**Conclusion**: wmma-based approaches cannot beat FA4. The gap is from FA4's fundamentally different compute model (wgmma + TMA + warp specialization). Only a CUTLASS-native FMHA kernel can close it.

---

## 4. Current State: CUTLASS FMHA Integration (IN PROGRESS)

### What's Done
1. Forked CUTLASS example 88 (Hopper FMHA) into `csrc/libtorch_stable/attention/fmha_sm90/`
2. Fixed all include paths for flat directory layout
3. Verified it compiles within the vLLM build system
4. Wrote type aliases in `gemma_prefill_attention_sm90.cuh` for Gemma4 instantiation

### What's Next: Wire the CUTLASS FMHA Kernel Launch

The next step is writing the actual kernel launch code in `gemma_prefill_attention_sm90.cu`:

**Step 1: Contiguous tensor launch (validate correctness)**
- Construct `Arguments` struct with contiguous Q/K/V pointers and strides
- Call `cutlass::device::Universal<Kernel>::run(args, workspace, stream)`
- ProblemShape = `(batch=num_seqs, heads=num_q_heads, seqQ=max_q_len, seqK=seq_len, headDim=512)`
- Start with hd=256 (known working in CUTLASS example), then push to hd=512

**Step 2: Paged KV adaptation**
- The CUTLASS FMHA uses TMA for KV loads, which requires contiguous memory
- For paged KV: either (a) gather KV into a contiguous staging buffer before calling FMHA, or (b) modify the FMHA mainloop to use cp.async with page-table indirection instead of TMA for K/V loads
- Option (a) is simpler but adds a copy; option (b) is harder but zero-copy

**Step 3: k_eq_v optimization**
- Skip V load entirely, reuse K smem buffer as V
- Requires modifying `fmha_collective_tma_warpspecialized.hpp` producer loop

**Step 4: Gemma4 masking**
- Replace `CausalFusion` with Gemma4-specific fusion that handles sliding window + causal

### Key CUTLASS FMHA API Reference

**Kernel instantiation** (from `fmha_kernel_builder.hpp`):
```cpp
using Kernel = cutlass::fmha::kernel::FmhaBuilder<
    Element,                    // cutlass::bfloat16_t
    ElementAccumulatorQK,       // float
    ElementAccumulatorPV,       // float
    TileShape,                  // Shape<BlockQO, BlockKV, HeadDim>
    StrideQ, StrideK, StrideV,  // tuple<int, _1, tuple<int, int>>
    Fusion,                     // CausalFusion
    DispatchPolicy              // KernelTmaWarpSpecializedCooperative
>::Kernel;
```

**Arguments construction**:
```cpp
typename Kernel::Arguments args{
    {batch, heads, seq_q, seq_k, head_dim},  // ProblemShape (B, H, Q, K, D)
    {ptr_Q, stride_Q, ptr_K, stride_K, ptr_V, stride_V},  // mainloop
    {ptr_O, stride_O, ptr_LSE, stride_LSE},                // epilogue
    {sm_count}                                              // hw_info
};
```

**Stride format**: `tuple<int, _1, tuple<int, int>>` where:
- First int: batch stride (B dimension)
- `_1`: element stride (contiguous along head dim)
- `tuple<int, int>`: (head stride, sequence stride)

**Launch**:
```cpp
cutlass::device::Universal<Kernel> fmha_op;
auto status = fmha_op.run(args, nullptr, stream);
```

**Tile shapes that work** (from example):
- hd=128: `Shape<_128, _128, _128>` (2 MMA WGs cooperative)
- hd=256: `Shape<_128, _64, _256>` (2 MMA WGs cooperative)
- hd=512: `Shape<_64, _64, _512>` (UNTESTED — register pressure unknown)

### Architecture of the CUTLASS FMHA

**Thread organization**: 384 threads = 3 warpgroups
- WG0 (128 threads): Producer — TMA loads for Q, K, V
- WG1 (128 threads): Consumer 0 — wgmma QK + softmax + wgmma PV
- WG2 (128 threads): Consumer 1 — same (cooperative, splits M dimension)

**Compute flow per KV tile** (from `fmha_collective_tma_warpspecialized.hpp`):
1. Producer: TMA load K into pipeline stage
2. Consumer: wait K → wgmma QK (SS variant, Q and K from smem) → S accumulator
3. Softmax: online softmax on S, rescale O accumulator, convert S→P in registers
4. Producer: TMA load V into pipeline stage
5. Consumer: wait V → wgmma PV (RS variant, P from registers, V from smem) → O accumulator
6. Release pipeline stages

**Key design choices**:
- QK uses SS wgmma (both operands from smem descriptors)
- PV uses RS wgmma (P from registers as A operand, V from smem as B)
- P never goes to smem — converted in-place via `make_acc_into_op`
- O lives in registers throughout the KV loop
- K and V share a smem union (K consumed before V loaded)
- 2 consumer WGs split the M dimension cooperatively

---

## 5. File Map

### Modified files (committed)
- `CMakeLists.txt` — SM90 conditional compilation block
- `csrc/libtorch_stable/attention/gemma_paged_attention.cu` — SM90 decode dispatch
- `csrc/libtorch_stable/attention/gemma_prefill_attention.cu` — SM90 prefill dispatch

### New files (SM90 scaffolding, committed)
- `csrc/libtorch_stable/attention/gemma_paged_attention_sm90.cu` — decode stub (returns false)
- `csrc/libtorch_stable/attention/gemma_paged_attention_sm90.cuh` — decode header
- `csrc/libtorch_stable/attention/gemma_prefill_attention_sm90.cu` — prefill launcher (opt-in)
- `csrc/libtorch_stable/attention/gemma_prefill_attention_sm90.cuh` — prefill types + FMHA integration

### New files (CUTLASS FMHA fork, NOT yet committed)
- `csrc/libtorch_stable/attention/fmha_sm90/device_universal.hpp`
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_collective_tma_warpspecialized.hpp` — core mainloop
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_collective_tma.hpp` — non-warp-spec variant
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_collective_softmax.hpp` — online softmax
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_collective_load.hpp` — TMA load wrapper
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_common.hpp` — helpers
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_epilogue.hpp` — O + LSE writeback
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_fusion.hpp` — causal/identity fusion
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_kernel_builder.hpp` — type dispatch
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_kernel_tma_warpspecialized.hpp` — kernel entry
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_kernel_tma.hpp` — non-warp-spec kernel
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_options.hpp` — tag-based options
- `csrc/libtorch_stable/attention/fmha_sm90/fmha_tile_scheduler.hpp` — tile scheduling

### Benchmark scripts
- `/tmp/measure_e2e.py` — e2e benchmark harness
- `/tmp/decode_bw.py` — decode bandwidth microbench
- `/tmp/prefill_microbench.py` — prefill throughput isolation

### Results files
- `/tmp/results_12b.json` — baseline 12B full suite
- `/tmp/results_12b_context.json` — baseline 12B context sweep
- `/tmp/sm90_osmem_v2_12b.json` — latest SM90 attempt

---

## 6. Build & Test Commands

```bash
# Activate
source /projects/gemma4-vllm/gemma-attn-env/bin/activate
cd /projects/gemma4-vllm/my-vllm-gemma-attn

# Build (from source, compiles SM90 kernels)
pip install -e . --torch-backend=auto

# Verify SM90 compiled
grep "Building gemma_attn_sm90" /tmp/nohup.out

# Check register usage
/usr/local/cuda/bin/cuobjdump --dump-resource-usage vllm/_C_stable_libtorch.abi3.so 2>/dev/null \
    | grep -B1 -A3 "prefill.*sm90\|decode.*sm90"

# Quick correctness test (SM80 fallthrough, default)
CUDA_VISIBLE_DEVICES=0 python /tmp/measure_e2e.py \
    --model google/gemma-4-12b-it --suite quick

# SM90 prefill (opt-in, for development)
CUDA_VISIBLE_DEVICES=0 GEMMA_SM90_PREFILL=1 python /tmp/measure_e2e.py \
    --model google/gemma-4-12b-it --suite quick

# Full suite
CUDA_VISIBLE_DEVICES=0 python /tmp/measure_e2e.py \
    --model google/gemma-4-12b-it --suite full --output-json /tmp/results.json

# Context sweep
CUDA_VISIBLE_DEVICES=0 python /tmp/measure_e2e.py \
    --model google/gemma-4-12b-it --suite context --output-json /tmp/context.json
```

---

## 7. Key Learnings

1. **wmma has the same throughput on SM80 and SM90** — recompiling for sm_90a gives zero benefit
2. **Decode is vector-matrix** — wgmma (M=64 minimum) doesn't fit. Improvement requires TMA bulk loads
3. **Prefill register wall at hd=512** — O[64,512] = 256 f32/thread. Can't fit in registers with wgmma M=64. Head chunking or O-in-smem needed
4. **FA4 at hd=512 is also constrained** — uses 64×64 tiles, single pipeline stage, PV through smem
5. **The k_eq_v advantage (half KV bytes) only materializes when DRAM-bound** — on H100 with SM80 kernels, the kernel is FMA/softmax-bound, not memory-bound
6. **CUTLASS FMHA example 88 is the right starting point** — complete wgmma+TMA+warp-spec implementation, 2600 lines, handles up to hd=256 natively
7. **atomicAdd into smem is catastrophically slow** — use store_matrix_sync + cooperative add instead
8. **The SM80 v2 wmma kernel at -21% vs FA4 is the best achievable without Hopper-native compute**

---

## 8. Gemma4 Model Configurations (for reference)

| Model | k_eq_v | GQA group | Layers (sliding/full) | hd sliding | hd full |
|---|---|---|---|---|---|
| E2B | False | 8 | 28/7 | 256 | 512 |
| 12B | **True** | 2 | 40/8 | 256 | 512 |
| 31B | **True** | 2 | 50/10 | 256 | 512 |

The SM90 prefill kernel targets **hd=512 full-attention layers** (k_eq_v=True on 12B/31B). These are ~17% of layers but the most compute-intensive for prefill (27.5% of prefill GPU time on 12B).

---

## 9. FA4 Weaknesses to Exploit (the moats)

1. **k_eq_v**: FA4 loads K and V separately (2× bandwidth). GEMMA_ATTN loads K once
2. **Sliding window block-skipping**: FA4 processes all KV blocks. GEMMA_ATTN skips out-of-window
3. **scale=1.0**: Gemma4 has no softmax scale — `exp2f(qk * LOG2E)` simplifies to `expf(qk)`
4. **No softcap**: FA4 supports softcap (overhead); Gemma4 doesn't use it
