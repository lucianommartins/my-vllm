# GEMMA_ATTN — Hopper (H100) & Blackwell (B100/B200/GB200) speedup opportunities

> Forward-looking roadmap for the Gemma4-overfit attention backend. The A100
> overfit-attention frontier is reached; every remaining big lever is blocked on
> A100 by the **hd512 register/occupancy wall** plus the inability to reach
> DRAM-bound execution by hand on `sm_80`. Hopper/Blackwell change exactly those
> constraints.
>
> Full build write-up: `GEMMA_ATTN_TECHNICAL_REPORT.md` (§11).
> Banked, validated IP for the port: the mma.sync micro-tests (`/tmp/mma_*.cu`),
> the register-softmax dataflow prototypes (`mma_proto2_cos.cu`,
> `mma_online_proto.cu`, both cos = 1.0), and `gemma_decode_mma_kernel`
> (gated off, correct).

---

## Why A100 is walled (the constraints these architectures lift)

- **hd512 register wall.** `O[16,512] = 256 fp32/lane > 255` registers ⇒
  register-resident softmax/O is infeasible for a warp at hd512 → forced
  smem-S round-trip + FMA-serial softmax stalls (≈41 % of prefill GPU time).
- **Decode never reaches DRAM-bound by hand.** Every A100 decode kernel plateaus
  ~50 % DRAM: SIMT is compute-bound on warp-shuffles, mma is smem-coordination-
  bound (84 % L1/TEX), stream is occupancy-bound. The `k_eq_v` half-byte 2× only
  materializes if the kernel is DRAM-bound. Triton's 81 % comes from compiler-
  class register-resident-S/P scheduling that is impractical to hand-write on sm_80.

---

## 1. The decisive 2× decode (headline H100 win)

A100 decode plateaus at ~48–54 % roofline. Unlocks:

- **TMA (Tensor Memory Accelerator)** for bulk async KV loads → removes the
  cp.async occupancy tax and the "too few memory requests in flight" latency bound
  that capped the stream/SIMT kernels.
- **228 KB smem/SM (vs 164 KB on A100)** → the hd512 K-stages that forced
  1 CTA/SM and BLOCK_N=16 on A100 now fit at 2–4 CTA/SM.
- **`wgmma` (warpgroup async MMA) + register-resident S/P** → replicates Triton's
  conflict-free scheduling; kills the mma-decode smem-coordination wall →
  DRAM-bound execution → the `k_eq_v` half-byte advantage finally yields ~2× over
  Triton at high batch.
- **fp8 tensor cores** combined with `k_eq_v`.

## 2. FA2/FA3-class hd512 prefill (lift 18 % → 50–70 % MFU)

The single biggest lossless prefill lever (the full-layer kernel alone is 27.5 %
of prefill GPU time). A100 blocker: the register wall forces v2's smem-S round-trip
+ serial softmax stalls. Unlocks:

- **Larger register file + `wgmma` warpgroup accumulator layout** make
  register-resident O feasible at hd512 → removes the S/P round-trip.
- **Warp specialization + TMA pipelining** (FA3 design): producer warps issue TMA
  KV loads, consumer warpgroups run `wgmma` — hides memory behind compute (the
  A100 v2 has 4 `__syncthreads`/block and cannot).
- **Rewrite the validated M-split register-softmax (`v3`) dataflow in `wgmma`** for
  *both* head sizes. On A100 it lost to v2 on hd256 by 1.3–1.7× (occupancy-bound);
  on Hopper the bigger regfile/smem flips that occupancy math.

## 3. Blackwell-specific (B100/B200/GB200)

- **5th-gen tensor cores + native fp4/fp6 microscaling (MXFP)** → KV-cache and/or
  QK·PV in fp4 with hardware block-scaling = the largest bandwidth multiplier on
  the `k_eq_v` full layers, on top of a 4× narrower datatype.
- **TMEM (tensor memory) + `tcgen05` MMA** → a dedicated accumulator space removes
  accumulator register pressure entirely → the hd512 wall becomes irrelevant; hold
  full-head O accumulators for both head sizes.
- **2-SM / CTA-pair cooperative kernels + distributed shared memory (DSMEM)** →
  split the hd512 head / KV range across an SM pair sharing smem → structural fix
  for the single-CTA hd512 occupancy ceiling; cross-CTA split-KV reduction via
  DSMEM (no global round-trip like the A100 `gemma_split_reduce_kernel`).

## 4. Cross-architecture levers to re-evaluate

- **fp8 (Hopper) / fp4-MXFP (Blackwell) KV cache** — halves/quarters KV bytes at
  *all* batch sizes; the one un-tried A100 bandwidth lever (deprioritized there).
  Far more attractive on TC-native fp8/fp4 and compounds with `k_eq_v`.
  *Caveat:* it also helps DRAM-bound Triton, so combine it with the `k_eq_v`
  K-only compression Triton cannot do — to **widen**, not narrow, the lead.
- **`k_eq_v` single-slot KV storage** — the A100 attempt was blocked by vLLM-core
  allocation/profiling assumptions (not hardware). Revisit as a proper upstream
  hybrid-allocator change; the 1.5–2× full-layer KV capacity win is
  hardware-independent and compounds with everything above.
- **Persistent / megakernel decode** (NanoFlow-style compute/memory op-overlap) —
  tractable with Hopper async TMA + warp specialization.

---

## H100 results (2026-06-25): FA4 k_eq_v prefill dispatch

### What shipped (GEMMA_FA4_KEQV=1)

On H100, `k_eq_v` full-attention layers (hd=512) now dispatch to FA4 with
`key_cache` passed as both K and V. FA4's wgmma + TMA + warp specialization
gives significantly higher prefill throughput than the custom wmma kernel.
The redundant V load is absorbed by L2 cache (98.8 % hit rate at ≤12 K context).

**12B e2e results (GEMMA_ATTN vs FLASH_ATTN baseline, CUDA graphs):**

| Regime | Baseline gap | **With FA4 k_eq_v** | Improvement |
|---|---|---|---|
| decode-vlong (8K) | -20.6 % | **-14.0 %** | +6.6 pp |
| prefill-long (8K) | -29.2 % | **-18.7 %** | +10.5 pp |
| prefill-med (4K) | -21.7 % | **-17.3 %** | +4.4 pp |
| batch-128 | -19.2 % | **-17.1 %** | +2.1 pp |
| context-12K | -25.7 % | **-15.3 %** | +10.4 pp |

Benefit **grows with context**: +0.8 pp at 2 K → +10.4 pp at 12 K. Enabled by
`GEMMA_FA4_KEQV=1` env var. A100 path is unchanged (env var is SM90-gated).

### What was tried and ruled out (H100 kernel development)

- **SIMT 3-stage pipeline (BN=16):** pipeline overhead > latency benefit. Regressed.
- **SIMT wider tile (BN=32):** occupancy drop (80 regs, 3 CTAs) cancelled tile benefit. Flat.
- **wgmma for decode:** wgmma M=64 minimum vs decode BDY=2 rows → 97 % wasted compute.
- **Prefill 8w/MIN_CTA=3:** spills 200 B to stack. Occupancy gain cancelled by spill.
- **Prefill native sm_90a codegen (16w/MIN_CTA=2):** identical codegen to sm_80. Flat.
- **Prefill QK-load overlap:** double-buffered sKV, idle warps load during QK. STACK:80
  (2× v2). Bottleneck is FMA/softmax (not load latency — KV fits L2). No benefit.

Root cause: SM80 wmma kernels are already well-tuned for their compute pattern.
The gap vs FA4 is from FA4's fundamentally different Hopper primitives (wgmma +
TMA + CUTLASS pipeline + warp specialization), not from tuning parameters.

### Follow-up: FA4 CuteDSL V-pipeline elimination (Option B)

**Status: designed, not implemented. Parked due to deadlock risk.**

Modifying FA4's CuteDSL kernel (`flash_fwd_sm90.py`) to skip the V TMA pipeline
when `k_eq_v` would eliminate the redundant V load entirely. This matters when KV
exceeds L2 cache (>16 K context on H100's 50 MB L2). Estimated +5–15 % additional
prefill throughput at very long context.

**Modification plan (14 change points, ~200–400 lines):**

1. **SharedStorage (lines 145-163):** remove `mbar_ptr_V` and `sV` fields when k_eq_v
2. **TMA descriptors (lines 308-323):** skip `tma_atom_V`, alias to `tma_atom_K`
3. **Pipeline creation (lines 507-542):** set `pipeline_v = pipeline_k` (shared barriers)
4. **Smem buffers (lines 550-559):** set `sV = sK`, `sVt = transpose_view(sK)`
5. **Producer loop — non-overlap (lines 854-881):** remove V acquire/load/commit
6. **Producer loop — overlap (lines 882-920):** remove K[n]/V[n-1] interleaving,
   simplify to K-only sequential load
7. **Producer tail (line 963):** `pipeline_v.producer_tail` → `pipeline_k.producer_tail`
8. **Consumer `mma_one_n_block` (lines 1465-1505):** skip `pipeline_k.consumer_release`
   after QK (K data still needed for PV), skip `pipeline_v.consumer_wait` before PV,
   single `pipeline_k.consumer_release` after PV
9. **Consumer `mma_one_n_block_intrawg_overlap` (lines 1526-1551):** skip
   `pipeline_v.consumer_wait`, adjust release to `pipeline_k.consumer_release`
10. **`first_half_block_overlap` (line 1387):** defer K release to PV phase
11. **`last_half_block_overlap` (lines 1439-1442):** skip V wait, use K release

**Risk:** mbarrier acquire/wait/release sequences — a single missed or extra
barrier operation deadlocks the GPU silently. CuteDSL is JIT-compiled so errors
only surface at runtime. Recommend standalone parity test before integration.

## Payoff ranking (H100 / Blackwell, updated)

0. **FA4 k_eq_v prefill dispatch** — SHIPPED, +3–10 pp e2e (Python-only, zero kernel risk).
1. **FA4 V-pipeline elimination** — designed, +5–15 % additional at very long context.
2. **TMA decode** — TMA descriptor-based KV loads for DRAM-bound decode (infra effort).
3. **fp4/fp8 KV** — multiplicative bandwidth lever on TC-native datatypes.
4. **TMEM / 2-SM cooperative hd512** (Blackwell) — dissolves the head-size wall.

## Strategic note

On Hopper/Blackwell, faster tensor-core GEMMs shrink the MLP slice of e2e runtime,
so **attention becomes a larger share of e2e** → these attention wins should
translate to e2e *more* directly than they did on A100, where the recurring lesson
was that attention rarely bound e2e.
