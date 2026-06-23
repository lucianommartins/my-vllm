# GEMMA_ATTN — Technical Report

**A Gemma4-overfit CUDA attention backend for vLLM**

Status date: 2026-06-23 · Branch: `lucianommartins/gemma-attn` · HEAD: `6b12d1de4`
Hardware of record: 8× NVIDIA A100-SXM4-80GB · nvcc 12.4 · torch 2.11+cu128 · vLLM 0.23.1rc1
Repo: `/projects/gemma4-vllm/my-vllm`

---

## 0. Abstract

GEMMA_ATTN is a custom vLLM v1 attention backend **deliberately overfit to the Gemma4 model
family** — it exploits structural invariants that no generic kernel (FlashAttention, Triton
`unified_attention`, CUTLASS-FMHA) is allowed to assume. The thesis: by hard-coding Gemma4's
`k_eq_v` full layers (K≡V), its dual head dimensions (512 full / 256 sliding), its sliding+full
layer mix, and its lack of attention softcap (scale = 1.0), we can beat the generic A100 default
(Triton) end-to-end.

**Bottom-line result (production-representative, CUDA graphs, vs A100-default `TRITON_ATTN`):**
GEMMA_ATTN wins **+5–9 % e2e** on the regimes where attention is a real share of runtime —
prefill-heavy (+9.2 % on 12B, +5.7 % on 31B) and long-context decode (win grows with context:
~0 % at 4.5 K → +1.3 % at 6.7 K → +8.3 % at 10 K). It also holds a **capability moat**: it is the
only backend that runs Gemma4's `head_dim = 512` full layers on A100 at all (FA2 caps at 256).

The headline scientific findings are as much **negative results** as wins, and they are the
report's most durable contribution: (1) Gemma4 is ~83 % sliding-window layers, so lossy-KV
reduction (top-k / eviction) is **e2e-inert by architecture**; (2) low-batch decode is
weight-bandwidth-bound and high-batch decode is dense-MLP-compute-bound, so **attention is rarely
the e2e bottleneck**; (3) the decisive 2× decode and the FA2-class prefill rewrite are both
**blocked on A100 by a register/occupancy wall** and are H100/Blackwell work. The overfit
attention/KV project has reached its **A100 frontier**.

---

## 1. Target, scope, and constraints

### 1.1 Gemma4 architecture (the substrate)

| Model | Class | Layers (sliding / full) | Full-layer GQA | Heads (q/kv) | Notes |
|---|---|---|---|---|---|
| E2B-it | gemma4 | mostly sliding | — | small | `is_mm_prefix_lm=False` |
| E4B-it | gemma4 | mostly sliding | — | small | `is_mm_prefix_lm=False` |
| 12B-it | gemma4_**unified** | 40 / 8 | group 2 | — | separate integration bug (see §9.4) |
| 26B-A4B-it | gemma4 (MoE) | 25 / 5 | group 8 | — | A4B = MoE, **not dense** |
| 31B-it | gemma4 (**dense**) | 50 / 10 | group 2 | 32 / 16 | primary big-model target |

Invariants exploited (the **moats**):
- **`k_eq_v`**: full (non-sliding) layers have **K ≡ V**. The kernel reads K **once** and reuses
  it as V → **half the KV bytes read** on full layers. No generic kernel knows K==V.
- **Dual head dim**: full layers `hd=512`, sliding layers `hd=256`. Load only the dims needed.
- **Layer mix**: ~83 % sliding (window-bounded, cheap, lossless long-context) + ~17 % full
  (unbounded KV, the e2e-critical path for long context).
- **No attention-score softcap** (only a final-logit softcap of 30); **scale = 1.0**.

### 1.2 The competitive landscape (exploited flaws)

- **FlashAttention 2** caps `head_dim` at 256 → **cannot run Gemma4 full layers on A100**. FA3/FA4
  are Hopper-only. ⇒ on the 5–10 full layers per model the kernel competes with **nothing** from FA.
- **Triton `unified_attention`** is the **A100 default and runs Gemma4 fine** (incl. hd512). It is
  a strong tensor-core baseline (tl.dot, GQA packing, `num_stages` pipeline, `.cg` streaming
  cache-modifiers) that hits ~81–82 % HBM roofline on decode — but it is **generic**: it reads K and
  V separately (ignores `k_eq_v`) and cannot specialize the head dims. This is the gap we attack.

### 1.3 Standing constraints (engineering discipline)

- **Never `enforce_eager`** in any benchmark — it disables CUDA graphs and inflates numbers;
  every shippable conclusion is measured under graphs.
- **Cosine similarity / max-abs** for parity, never element-wise relative error (it blows up on the
  near-zero outputs that signed V produces via cancellation).
- Generate git commit commands; do not auto-commit. No `grep`/`tail` hiding of benchmark errors.

---

## 2. System architecture

```
my-vllm/
├─ csrc/libtorch_stable/attention/
│   ├─ gemma_paged_attention.cuh   (79 KB) — DECODE kernels + top-k/bounds + split-reduce
│   ├─ gemma_paged_attention.cu    (40 KB) — decode launcher, dispatch, env toggles
│   ├─ gemma_prefill_attention.cuh (16 KB) — PREFILL kernel (v2)
│   └─ gemma_prefill_attention.cu  (8 KB)  — prefill launcher
├─ csrc/libtorch_stable/torch_bindings.cpp — op schemas (gemma_paged_attention,
│       gemma_prefill_attention, gemma_topk_select, gemma_update_kv_bounds)
├─ vllm/v1/attention/backends/gemma_attn.py (780 L) — backend, metadata, dispatch,
│       cascade, top-k gating, KV-bounds maintenance
└─ vllm/v1/attention/ops/gemma_paged_attention.py (94 L) — python op wrappers
```

**Kernels currently in the tree** (`__global__` entry points):

*Decode* (`gemma_paged_attention.cuh`):
- `gemma_decode_simt_kernel` — **bandwidth-first SIMT**, the shipped default for `k_eq_v`, group ≤ 2.
- `gemma_decode_stream_kernel` — wmma + cp.async pipeline, default for the other bf16 non-quant cases.
- `gemma_decode_mma_kernel` — mma.sync flash-decode, **gated off** (`GEMMA_DECODE_MMA=1`), banked IP.
- `gemma_flash_decode_kernel`, `gemma_gqa_decode_kernel`, `gemma_gqa_split_decode_kernel` — earlier
  scalar/wmma decode variants + split paths.
- `gemma_split_reduce_kernel` — cross-CTA split-KV online-softmax/LSE combine.
- `gemma_topk_select_kernel`, `gemma_topk_select_bounds_kernel`, `gemma_update_kv_bounds_kernel` —
  Quest-style lossy-KV scoring/maintenance (off by default).

*Prefill* (`gemma_prefill_attention.cuh`):
- `gemma_prefill_kernel_v2` — **the sole prefill kernel**: register-resident O (head-split warps),
  warp-per-row parallel softmax, smem bank-conflict padding, `__launch_bounds__` occupancy control,
  cascade (`non_causal` + LSE), mm-prefix bidirectional mask. Handles both hd512 and hd256.

**Dispatch policy** (auto, env-overridable):
- Decode: `k_eq_v && group ≤ 2` → SIMT (the 31B/12B full layers); else stream; fp8/fp16 → scalar.
- Prefill: hd ∈ {256, 512} + bf16 + non-quant KV → `gemma_prefill_kernel_v2`; fp8 → Triton fallback.
- Opt-in via `--attention-backend GEMMA_ATTN` (config.py does **not** auto-select it).

---

## 3. Prefill kernel — development trajectory

Prefill attention is the **genuine lossless e2e lever**: profiling (12B, ~12 K ctx, N=2, CUDA
graphs) shows prefill GPU time is **~54 % GEMM** (QKV/O-proj + MLP, cutlass, near-peak, not a lever)
and **~41 % attention** — `O(L²)`, growing with context. The full-layer kernel alone is **27.5 %** of
all prefill GPU time (the single biggest attention cost); sliding-layer kernel ~13 %.

### 3.1 The hd512 problem and the v1→v2 redesign

`head_dim = 512` is the enemy: an `O[BM×512]` fp32 accumulator is too large to hold in registers for
a whole warp, which forces O into shared memory (v1) → 1 CTA/SM → occupancy collapse.

Measured prefill trajectory **(× vs Triton `unified_attention`)**, each step a distinct optimization:

| Step | Technique | × Triton |
|---|---|---|
| v1 | wmma, O-accumulator in smem | 0.20× |
| v2 | register-resident O via **head-split warps** (each warp owns a head slice) | 0.49× |
| v2 + pad | smem bank-conflict padding (`LDH=HEAD+8`, `LDN=BN+8`) | 0.67× |
| + NWARPS 8→16 | ncu showed occupancy/barrier-bound, not TC-bound (12.5 %→25 % occ) | 0.78× |
| + **parallel softmax** | replace serial 1-warp softmax with **warp-per-row** (32 lanes = cols, warp-reduce) | **1.18×** (1.49× short) |
| + `__launch_bounds__` | `MIN_CTA` per head size drives register allocator to target occupancy | **1.29–1.71×** (hd512), **1.18–1.34×** (hd256) |

The `__launch_bounds__(NUM_WARPS*32, MIN_CTA)` knob was the single highest-ROI prefill lever:
- hd512: MIN_CTA 1→2 (REG 128→64, **no spill**) = 50 % occ → ~1.33× became **1.29–1.71×** (avg ~1.67×).
- hd256: MIN_CTA 2→3 (REG ~120→79, **no spill**) = 37.5 % occ → 0.87–1.15× became **1.18–1.34×**.

Discipline note: always `cuobjdump --dump-resource-usage` to confirm **no register spill** before
trusting a config; pushing MIN_CTA further is smem-capped and only cuts ILP → measured worse, reverted.

### 3.2 ncu root cause (the remaining gap)

Deep ncu of the live hd512 v2 at L=8192: **latency/softmax-bound, not memory or tensor-core bound** —
DRAM 0.7 % (KV fits L2, 98.8 % hit); 48.6 % no-eligible-warp; top pipe **FMA 29.1 %** (fp32 softmax),
tensor pipe lower; occupancy 50 % capped by **both** registers and dynamic smem (74.9 KB/block). Root
cause: the "naive flash" structure materializes S in smem, does fp32 online softmax, and **round-trips
P through smem** (wmma cannot pass the QK accumulator fragment to the PV A-operand) — the serial chain
+ smem round-trip + `__syncthreads` barriers are the stalls; the 75 KB smem caps occupancy at 50 %.

### 3.3 Feature completeness

`gemma_prefill_kernel_v2` is feature-complete for Gemma4 prefill:
- **Varlen / paged**: `cu_seqlens_q`, paged block-table K/V staging, GQA group templating.
- **Masks**: causal + context offset; sliding-window (`kv_begin` pruning); **mm-prefix bidirectional
  image-token mask** (`USE_MM_PREFIX` template — compiled out for text-only/full layers → zero
  overhead; instantiated only when image spans present this step).
- **Cascade**: `non_causal` flag (whole-key-range prefix pass) + natural-log LSE output for
  cross-pass merge.
- Cleanup (2026-06-23): the dead v1 kernel + its launch macros were removed; v2 is the sole kernel.

---

## 4. Decode kernel — development trajectory and the bandwidth finding

### 4.1 The pivotal measurement: decode is not compute-bound

A tensor-core (wmma) GQA-packed decode kernel was built and parity-clean, but the bench **refuted the
compute-bound hypothesis**: wmma ≈ scalar everywhere. Both win at batch=1 (the split-KV latency
regime) and both **lose hard at batch ≥ 8** vs Triton. Tensor cores don't move the needle because
**decode is HBM-bandwidth-bound**.

The root-cause measurement (b=64, L=8192, hd512 `k_eq_v`):

| Kernel | Bytes read | Time | Achieved BW | % A100 peak (2039 GB/s) |
|---|---|---|---|---|
| Ours (early) | 1.07 GB (`k_eq_v` half-bytes) | 3206 µs | 335 GB/s | **16 %** |
| Triton | 2.15 GB (reads K+V separately) | 1277 µs | 1682 GB/s | **82 %** |

Triton moves 2× the bytes 2.5× faster — it **saturates HBM, we didn't**. We were latency-bound (too
few memory requests in flight: 200 regs → 1 CTA/SM, no cp.async pipelining). **Because `k_eq_v` reads
half the bytes, we only need ~41 % HBM to match Triton, ~82 % to 2× it.** This defined the goal.

### 4.2 The bandwidth campaign (what shipped)

1. **num_splits heuristic fix** (committed): old `2×SMs` split target under-filled the device at
   medium batch (0.8 waves, 14 % HBM). New target `DECODE_WAVES(6)×CTA_PER_SM(3)×num_sms`. Lifted
   b=8 from 0.40× → 0.82× vs Triton.
2. **Stream kernel** `gemma_decode_stream_kernel`: GQA-packed wmma (light compute, no per-token
   shuffle) + **double-buffered cp.async** (overlap tile t+1 load with tile t compute). Trajectory
   via occupancy tuning: 21 % → 37 % → 44 % → **48 % roofline**:
   - BLOCK_N 32→16 (halve the two 33 KB hd512 K-stages → 51 KB) + NW 16→8 + MIN_CTA=3 (REG 78, no
     spill) = **3 CTA/SM = 37.5 % occupancy** → beats or ties Triton at **every** tested config
     (b8 1.65×, b64/L1024 1.40×, b128 1.19×, b256 1.00×).
   - Phase 2b **V-from-global for hd512 !k_eq_v** (E2B full layers): since BLOCK_N=16 ≤ page_size,
     V is wmma-loadable straight from global with no V-smem → 51 KB → 3 CTA/SM (was 1). Made
     conditional (`!K_EQ_V && HEAD≥512`) because it regressed hd256 sliding. Turned the worst case
     (0.59×) into 0.75–0.90×.
   - Stream **strictly dominates** scalar at every config → became the **default** (the
     `GEMMA_DECODE_STREAM` toggle was removed).
3. **SIMT bandwidth-first kernel** `gemma_decode_simt_kernel` (the breakthrough, committed default
   for `k_eq_v` group ≤ 2): register-resident Q/scores/O; smem holds **only** the K tile (GQA reuse,
   `k_eq_v` reuses K as V); `bdz` within-block KV-split (8 warps even for group 2) + cross-warp LSE
   combine; cp.async 2-stage; and **critically vectorized `uint4` smem reads of K**. The uint4 read
   was the unlock — element-wise smem reads saturated L1/TEX at 87 % (DRAM only 35 %); uint4 dropped
   L1 to 68 % and lifted the 31B config (group2/nkv16/hd512, b128/L4096) to **53.8 % HBM = 1.32×
   faster than Triton** (Triton 81 % but reads 2× bytes). Dispatch = AUTO: SIMT for group ≤ 2, stream
   for larger.

### 4.3 The 2× wall (the decisive negative result)

The 2× decode (≈80 % roofline × `k_eq_v` half-bytes) was attacked three ways and **resolved as
A100-blocked**:
- **wmma**: smem-operand → occupancy-bound; reverted.
- **stream**: plateaus at 48 % roofline; ncu shows DRAM only 40 % (2× physically available) but
  blocked by a QK warp-imbalance barrier (33 %) at BLOCK_N=16. Phase-2a split-K QK to fix it
  **failed** (the +8 KB reduction scratch dropped 3→2 CTA/SM; **occupancy trumps the barrier**).
- **mma.sync** `gemma_decode_mma_kernel`: tensor cores killed the compute wall (77 %→35 %) but the
  coordination (ldmatrix-K-twice for `k_eq_v` + S/P smem round-trips + warp0 softmax) **saturated
  shared memory instead** (84 % L1/TEX, 22 % DRAM). Reached 31–50 %, ≤ SIMT, **not 2×**.

**Conclusion**: every hand-written kernel plateaus ~50 % DRAM. *The `k_eq_v` 2× only materializes if
the kernel is DRAM-bound, but on A100 the hd512 head is so large that occupancy/coordination binds
first.* Triton's 81 % comes from compiler-class register-resident-S/P, conflict-free scheduling that
is impractical by hand. **Clean 2× decode on hd512/group2 is blocked on A100 → it is H100 work**
(TMA + fp8 tensor cores + 2× smem + warp specialization). Best shipped decode = SIMT 1.32×.

---

## 5. Memory & cache management

### 5.1 KV workspace over-allocation fix — DONE, live (commit `e0a6054ad`)

Root cause was **per-layer** allocation: each `GemmaAttentionImpl` held its own split-KV scratch
(`_exp_sums`/`_max_logits`/`_tmp_out`) → ~60 copies on 31B = **~13 GB**, OOM'ing 31B on a single
80 GB GPU. Fix (Python-only): a module-level `_DECODE_PARTITION_CACHE` keyed by
`(device, head_size, num_heads, dtype)` shared by **all** layers (split-KV scratch is written +
reduced within one op call on one stream before the next layer runs → safe; monotonic growth → CUDA
graphs stay valid). 60 buffers → 2 (one per head size). Kept `SPLIT_PARTITION_CAP=128` so num_splits
(and the long-ctx win) is unchanged. **Validated**: byte-identical output vs Triton; 31B now runs
**single-GPU** (was OOM) with KV 0.31 → 11.07 GB available.

### 5.2 Cascade / prefix-shared attention — DONE, validated

RadixAttention/cascade-class lossless throughput lever: read a shared prefix's KV **once** across
requests. Implemented in 3 phases (decode LSE output + prefill `non_causal` + LSE; backend
`use_cascade_attention` override delegating to FA's heuristic; `_forward_cascade` = prefix pass
(prefill op, non_causal) + suffix pass (decode op) + `merge_attn_states`). Auto-targets full (hd512)
layers only.
- **E2B** (attention-heavy small model), 192 reqs × 1760-tok shared prefix: **1.50×** (heuristic) /
  **1.67×** (forced) out tok/s; byte-identical to Triton.
- **31B** (dense, batch 192): **~neutral** (1.01–1.04×) — at high batch the model is compute-bound on
  MLP/proj GEMMs; cascade only saves full-layer KV-read (~1.7 ms vs ~19 ms MLP → ~9 % ceiling).
  *Same lesson recurring: attention wins only surface where attention dominates runtime.*

### 5.3 mm-prefix bidirectional mask — implemented properly

The big VLMs (12B/26B/31B) were originally **gated out** of GEMMA_ATTN: vLLM rejects backends that
don't `supports_mm_prefix()` when `model_config.is_mm_prefix_lm=True` (bidirectional image-token
attention). Implemented the real mask (prefill-only — decode queries are post-prompt text, always
causal): `USE_MM_PREFIX` template, mask = `(causal & sliding) | (q,k in same image span)`, kv_end
extended by the window when spans present; full layers compile it out (zero overhead). `mm_prefix`
metadata is cleared on full layers by the model's `_clear_mm_prefix_for_full_attn_layers`, so only
sliding layers get bidirectional. `supports_mm_prefix()→True` is now **legitimate**. Validated e2e on
**31B (TP=2): text AND image coherent** (accurate image descriptions). ⇒ the kernel now runs
out-of-the-box on plain-gemma4 incl. 31B. (Exception: 12B `gemma4_unified` garbles even text-only — a
**separate pre-existing integration bug** in that model class, not the kernel.)

---

## 6. Lossy-KV experiments — correct kernels, e2e-inert by architecture

Goal: surpass 2× e2e in long context by lossy reduction of the **unbounded full-layer** KV. All
opt-in, quality-gated, off by default.

- **Phase 1 sink+window** (`GEMMA_FULL_SINK/WINDOW`): full layers attend `[0,S) ∪ [L−W, L)`. Kernel
  speedup huge (6.1× at b128/L4096; 11.3× at b64/L8192, super-linear via occupancy). **Lossy** (drops
  mid-context) → fails recall (needle cos 0.328) — motivates top-k.
- **Phase 2 Quest top-k** (`GEMMA_TOPK_K`): per-step block selection from maintained per-block
  channel **min/max key bounds** (upper-bound score `Σ q_d>0 ? q_d·max_d : q_d·min_d`), force-include
  sink+recent. EXACT top-k recovers needle cos=1.0 **at k=1**; bounds path picks the same needle.
  **bf16 bounds** (lossless: min/max of bf16 keys are bf16) halved the scoring read → decode-attn
  component **1.64–2.95×** (grows with L and batch). Kernels: `gemma_update_kv_bounds_kernel`,
  `gemma_topk_select_bounds_kernel`.
- **Phase 3 eviction** (3C plumbing): `EvictableFullAttentionSpec/Manager` free the contiguous middle
  of full-layer KV each step; freed ~57 % of full-layer KV on 31B, output identical to P1.

**The architectural verdict (definitive).** Under CUDA graphs the levers are **e2e-inert**:
- top-k: **0 %** e2e (decode-heavy N4 89.5/89.5; prefill-heavy N32 53.6/53.6).
- eviction 3C: **+3.8 %** under graphs (the eager +22 % was a graph-eliminable overhead artifact).
- 12B context sweep (8 K/19 K/30 K): **dead flat** at all three — no crossover trend.

Root cause: **Gemma4 is ~83 % sliding layers**, which already solve long-context KV losslessly
(window-bounded). The lossy levers' addressable surface is the ~17 % full layers, and within those
attention is a fraction of the layer (MLP dominates) and is **never the decode bottleneck at
reachable context (≤30 K)** — low-batch decode is weight-bandwidth-bound, high-batch is MLP-bound.
**These are correct, fast kernels (2–3× attention-component) that are e2e-inert on Gemma4 by design.**
Kept off by default; do not invest further unless an all-full-attention model is the target.

---

## 7. Attempts that hit walls (2026-06-23 session)

### 7.1 `k_eq_v` single-slot KV — BLOCKED by vLLM core
Idea: full layers K≡V → store K **once** (1 slot not 2) → ~half full-layer KV → 1.5–2× capacity.
Implemented 4 coordinated edits (env-gated): `AttentionSpec.single_kv_slot` (1× page bytes), merge
propagation, spec emission for hd512, 1-slot shape, `key=value=kv_cache[:,0]`. **Failed at startup**
in the cudagraph-memory-profiling path (`_reshape_kv_cache_tensors`): `shape [512,1,32,4,512] invalid
for input of size 67108864` — the env-based shape (1 slot) **diverged** from the spec-based page_size
(2×) in the profiling minimal-config. The K+V=2 assumption is woven through allocation **and**
profiling; the env-vs-spec consistency is fragile. Fixing it needs deeper `gpu_model_runner` surgery →
not robust for the modest payoff. **Reverted clean.** Per-layer single-slot KV is a vLLM-core change,
not an overfit-kernel change.

### 7.2 FA2-class register-resident-softmax prefill (hd512) — register-walled
The validated way to ~50 % MFU is FlashAttention-2's register-resident softmax over the mma
D-accumulator. But `O[16,512] = 256 fp32/lane > 255` registers ⇒ register-O is **infeasible** for a
warp at hd512. hd512 forces head-split + smem-S (exactly what v2 does). Dataflow validated standalone
at hd≤256 (cos 1.0) but **cannot be applied where it's needed**.

### 7.3 M-split register-softmax prefill `v3` (hd256) — correct but loses; reverted
Built a full M-split register-resident-softmax kernel (each warp owns 16 query rows + full head →
independent warps, no cross-warp softmax; S+O in registers; sVt-transposed V for PV). Online
multi-block register-O rescale validated standalone (`/tmp/mma_online_proto.cu`, cos 1.0). Integrated
kernel **parity-clean (cos 0.999998 == v2)** but **~1.3–1.7× slower** than the tuned v2 wmma on hd256
(best 8-warp BM=128 = 27 vs 35 TFLOP/s @ L16K, SW1024).

**The deep finding**: register-softmax wins the **softmax-bound** regime = the hd512 full layers
(18 % MFU, FMA/stall-bound) — but hd512 is **register-walled** so it can't use register-O. The regime
where register-O **fits** (hd256 sliding) is **occupancy/overhead-bound** (low FLOP/token from the
window) + pays a per-block transpose tax, so v2's higher occupancy wins. **The optimization and the
opportunity sit on opposite sides of the register wall.** Reverted (the env gate would have been
dead-by-default cruft, and the sm_80 mma body would be rewritten with wgmma on Hopper anyway).

---

## 8. End-to-end results (the bottom line)

**GEMMA_ATTN vs A100-default `TRITON_ATTN`, both under CUDA graphs** (`/tmp/measure_e2e.py`):

| Model | Regime | GEMMA | Triton | Δ |
|---|---|---|---|---|
| 12B | decode-heavy long (~10 K, N24) | 747.2 | 689.7 | **+8.3 %** |
| 12B | prefill-heavy (~6.8 K, N…) | 6131 | 5615 | **+9.2 %** |
| 12B | decode-short (~4.5 K) | 782.5 | 785.2 | ~flat |
| 31B | prefill-heavy (≤8 K) | 1471 | 1392 | **+5.7 %** |
| 31B | decode N8 (~6.7 K) | 166.4 | 164.2 | **+1.3 %** |

Within GEMMA_ATTN, SIMT-vs-wmma decode is **flat e2e** (745.5 vs 746.1) — confirming the banked
"SIMT 1.32×" is a kernel-component win that is e2e-inert here, exactly as the architecture predicts.
**The e2e value lives in the lossless shipped backend** (the prefill kernel + `k_eq_v` decode +
cascade + the workspace fix), **not** the lossy decode add-ons. Net: **+5–9 % over the default,
bigger at long context / prefill-heavy** — a real, production-representative win on a strong baseline.

Ancillary datapoint: Gemma4 12B **W4A16** gives 1.85× (49.76 → 91.94 tok/s) — orthogonal weight-quant
lever, not part of the attention kernel.

---

## 9. Methodology and lessons

1. **Always bench the hypothesis before trusting it.** The wmma decode kernel was correct but solved
   the wrong bottleneck (decode isn't compute-bound). Profiling, not intuition, located every real win.
2. **Measure e2e under CUDA graphs.** Eager numbers were systematically inflated (eviction +22 % eager
   → +3.8 % graphs; top-k +7 % eager → 0 % graphs). Graphs make the GPU compute-bound at modest
   concurrency, evaporating "fit more requests" benefits.
3. **Microbench parity ≠ e2e correctness.** A garbage-output bug was a stale mid-rebase build, not the
   kernel; cos>0.99999 didn't catch it. Always run an e2e smoke test on a clean build.
4. **Attention speedups only translate where attention is a real share of runtime.** Repeatedly: E2B
   (small, sliding-heavy) e2e-tied despite 1.3–2× attention wins; 31B high-batch MLP-bound; lossy-KV
   e2e-inert. The win is on large models, long context, prefill-heavy.
5. **`__launch_bounds__` MIN_CTA + `cuobjdump` no-spill verification** is the highest-ROI occupancy
   knob on A100; occupancy repeatedly **trumps barrier removal** (split-K QK failed twice for this).
6. **The negative results are the asset.** Knowing *why* lossy-KV, single-slot KV, FA2-hd512, and
   register-softmax are walled prevents re-attempting them and points precisely at the H100 frontier.

---

## 10. Current shipped status

- **Prefill**: `gemma_prefill_kernel_v2`, default for bf16 hd∈{256,512}; **1.18–1.71× vs Triton**;
  causal/sliding/mm-prefix/cascade complete.
- **Decode**: SIMT (group ≤ 2, `k_eq_v`) + stream (default) + scalar (fp8/fp16); `k_eq_v` half-bytes;
  num_splits heuristic; **1.32× component vs Triton**, e2e win grows with context.
- **Memory**: shared split-KV scratch (13 GB fix, 31B single-GPU); cascade prefix-sharing.
- **Multimodal**: mm-prefix mask → runs on 31B text+image.
- **Off by default** (validated, parity-clean): lossy-KV (sink+window/top-k/eviction), mma decode.
- **Reverted/parked**: single-slot KV (vLLM-core blocked), register-softmax v3 (loses on A100).
- **Net e2e**: **+5–9 %** over A100-default Triton on attention-relevant regimes; capability moat on
  hd512. **The A100 overfit-attention frontier is reached.**

---

## 11. Hopper (H100) and Blackwell (B100/B200/GB200) opportunities

This is where the A100 walls dissolve. The recurring A100 blocker is the **hd512 register/occupancy
wall** + the inability to reach DRAM-bound execution by hand; Hopper/Blackwell change exactly those
constraints. Banked, validated IP that pays off here: the mma.sync micro-tests (`/tmp/mma_*.cu`), the
register-softmax dataflow (`/tmp/mma_proto2_cos.cu`, `/tmp/mma_online_proto.cu`, cos 1.0), and
`gemma_decode_mma_kernel` (gated, correct).

### 11.1 The decisive 2× decode (the headline H100 win)
- **TMA (Tensor Memory Accelerator)** for KV loads: bulk async copies bypass the per-thread address
  generation and the cp.async occupancy tax that capped the stream/SIMT kernels at ~48 % roofline.
  Removes the "too few requests in flight" latency bound directly.
- **Larger smem (228 KB/SM on H100 vs 164 KB A100)**: the hd512 K-stages (the thing that forced
  1 CTA/SM and BLOCK_N=16) now fit at 2–4 CTA/SM → the occupancy that A100 could never afford.
- **`wgmma` (warpgroup async MMA)** + **register-resident S/P**: replicates Triton's compiler-class
  conflict-free scheduling that was "impractical by hand" on sm_80. Kills the mma-decode's
  smem-coordination wall (the 84 % L1/TEX saturation) → DRAM-bound execution → the `k_eq_v` half-byte
  advantage finally materializes as **~2× over Triton at high batch**.
- **fp8 tensor cores**: combine with `k_eq_v` for fp8-KV decode at full TC throughput (the bandwidth
  lever shelved on A100; see §11.4).

### 11.2 FA2/FA3-class prefill at hd512 (lift 18 % → 50–70 % MFU)
- **Register-resident softmax is feasible**: H100/Blackwell expose a larger register file and
  `wgmma`'s warpgroup accumulator layout, so the `O[16,512] = 256 fp32/lane > 255` wall that blocked
  the FA2 prefill rewrite on A100 (§7.2) is relaxed — O can stay in registers/accumulators across the
  KV loop. This removes v2's S/P smem round-trip + the FMA-bound serial softmax stalls (the 41 %-of-
  prefill cost), the single biggest lossless prefill lever.
- **Warp specialization + TMA pipelining** (FA3 design): producer warps issue TMA KV loads, consumer
  warpgroups run `wgmma` — hides memory behind compute, which the A100 v2 (4 `__syncthreads`/block)
  could not.
- The **M-split register-softmax v3** dataflow (validated, reverted on A100 because hd256 is
  occupancy-bound) is the **correct skeleton to rewrite in `wgmma`** for both head sizes — on Hopper
  the bigger regfile/smem flips its occupancy math.

### 11.3 Blackwell-specific (B100/B200/GB200)
- **5th-gen tensor cores + native microscaling fp4/fp6 (MXFP)**: KV-cache and/or QK·PV in fp4 with
  hardware block-scaling — the largest possible bandwidth multiplier on the `k_eq_v` full layers,
  preserving the overfit half-byte ratio on top of a 4× narrower datatype.
- **TMEM (tensor memory) + `tcgen05` MMA**: a dedicated accumulator space removes accumulator register
  pressure entirely — the hd512 wall becomes irrelevant; hold full-head O accumulators for both head
  sizes.
- **2-SM/CTA-pair cooperative kernels + distributed shared memory (DSMEM)**: split the hd512 head /
  KV range across an SM pair sharing smem — attacks the single-CTA hd512 occupancy ceiling structurally.
- **Larger DSMEM clusters** for cross-CTA split-KV reduction without the global round-trip the A100
  `gemma_split_reduce_kernel` pays.

### 11.4 Cross-arch levers worth re-evaluating on Hopper/Blackwell
- **fp8 (Hopper) / fp4-MXFP (Blackwell) KV cache**: halves/quarters KV bytes at **all** batch sizes —
  the one un-tried bandwidth lever (deprioritized on A100). On TC-native fp8/fp4 hardware it is far
  more attractive and compounds with `k_eq_v`. *Caveat: it helps DRAM-bound Triton too, so it narrows
  rather than widens the lead unless combined with the `k_eq_v` K-only compression Triton can't do.*
- **`k_eq_v` single-slot KV storage**: the A100 attempt was blocked by vLLM-core allocation/profiling
  assumptions, not hardware — revisit as a proper upstream hybrid-allocator change; the capacity win
  (1.5–2× full-layer KV) is hardware-independent and compounds with everything above.
- **Persistent / megakernel decode** (NanoFlow-style compute/memory op-overlap): more tractable with
  Hopper's async TMA + warp specialization than on A100.

### 11.5 Expected payoff ranking on H100/Blackwell
1. **2× decode** (TMA + wgmma + larger smem; §11.1) — the headline, A100-blocked, hardware-unlocked.
2. **hd512 register-softmax prefill** (§11.2) — 18 %→50–70 % MFU on the 27.5 %-of-prefill full-layer
   kernel; the biggest lossless prefill lever, A100-register-walled.
3. **fp4/fp8 KV** (§11.4, §11.3) — multiplicative bandwidth lever on TC-native datatypes.
4. **TMEM/2-SM cooperative hd512** (Blackwell; §11.3) — dissolves the head-size wall structurally.

> Strategic note: on Hopper/Blackwell, attention also becomes a larger share of e2e relative to MLP
> (faster TC GEMMs shrink the MLP slice), so these attention wins should translate to e2e more
> directly than they did on A100 — where the recurring lesson was that attention rarely bound e2e.

---

*Harnesses (recreate from memory if `/tmp` is wiped): `measure_e2e.py`, `roofline.py`, `decode_bw.py`,
`triton_bw.py`, `prefill_microbench.py`, `prefill_v3_parity.py`, `mma_proto2_cos.cu`,
`mma_online_proto.cu`, `topk_bounds_parity.py`, `cascade_*_parity.py`, `sink_window_parity.py`.*

*Design references: `Gemma4_Attention_Kernel_Design.md`, `Gemma4_Attention_Optimization_Set.md`,
`FA4_for_Gemma4.md` (in `/projects/gemma4-vllm/`).*
