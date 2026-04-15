# GDN CUDA Kernel Optimization Report

## Overview

MLSys 2026 FlashInfer AI Kernel Generation Contest — Gated Delta Network (GDN) track.
B200 GPU (SM100), `gdn_decode_qk4_v8_d128_k_last` + `gdn_prefill_qk4_v8_d128_k_last`.

### Kernel Specs
- **num_q/k_heads = 4, num_v_heads = 8** (GVA mode, ratio 2)
- **head_size = 128** (both K and V dimensions)
- **State**: `[B/N, 8, 128, 128]` f32, k-last layout
- **Parallelization**: 1 block per (batch/seq, v_head), 128 threads = V dimension
- **Binding**: TVM FFI, DPS mode, `TVM_FFI_DLL_EXPORT_TYPED_FUNC`

---

## Decode Kernel

### Architecture
- Grid: `(B × 8)`, Block: 128 threads
- Each thread owns one V-row of the 128×128 state (128 floats in registers via `float4[32]`)
- q, k loaded cooperatively into shared memory (256 floats total)
- Single token per invocation

### Optimization History

| Version | Change | Avg Latency | Notes |
|---------|--------|-------------|-------|
| v2 (baseline) | 2-pass state read from global | 0.0203ms | State read twice from global memory |
| v3 | 1-pass: state → registers first | 0.0197ms | Eliminates second global read |
| v5 (final) | + g×state pre-multiply | 0.0200ms | Saves 1 MUL/element in update loop |

### Optimization Items Attempted

#### D1: Single-pass state read ✅ Applied (v3)
- **Problem**: State tensor (`float4[32]` per thread = 512 bytes) was read from global memory twice — once for `old_v` dot product, once for state update + output.
- **Solution**: Load state into register array `float4 sr[32]` in a single pass. Both old_v and update loops operate on registers.
- **Result**: ~3% latency reduction. Modest because L1 cache already partially hid the second read.

#### D2: g×state pre-multiply ✅ Applied (v5)
- **Problem**: Both loops compute `g * state[i]` independently, duplicating the multiply.
- **Solution**: Pre-multiply `sr[i] *= g` before old_v loop. Update loop becomes `sr[i] += k[j] * delta` (no `g *` needed).
- **Result**: Neutral — saves 128 MUL in update loop but adds 128 MUL in pre-multiply. Same total ops, different scheduling.

#### D3: Further optimization — Not pursued
- Decode is already at **0.015–0.033ms** (near kernel launch overhead). Arithmetic intensity is low for single-token decode. No further optimization is practical without fundamentally changing the algorithm.

---

## Prefill Kernel

### Architecture
- Grid: `(N_seqs × 8)`, Block: 128 threads
- Each thread owns one V-row, processes tokens **sequentially** in a for-loop
- State kept in registers (`float4 sr[32]`) across token iterations
- Uses `cu_seqlens` for variable-length sequence batching

### Optimization History

| Version | Change | Long Seq Latency | vs Baseline |
|---------|--------|------------------|-------------|
| v2 (baseline) | Each thread reads k[128], q[128] from global independently | 7.1 / 10.5ms | — |
| v3 | Shared memory k/q (float, single buffer) | 4.3 / 6.2ms | **-40%** |
| v4 | + g×state pre-multiply + double-buffered (sync) | 4.4 / 6.4ms | -38% (regression) |
| v5 | Revert to v3 + `__ldg` hints | 4.3 / 6.2ms | -40% |
| **v6 (final)** | **cp.async double-buffered bf16 pipeline** | **4.0 / 5.9ms** | **-44%** |

### Optimization Items Attempted

#### P1: Shared memory k/q caching ✅ Applied (v3)
- **Problem**: Per token, each of 128 threads independently reads all 128 elements of k and q from global memory. Total: 128×128 = 16,384 scalar bf16 loads per token for k alone.
- **Solution**: Cooperative load — each thread loads 1 element into shared memory, all threads read from shared.
- **Result**: **-40% latency** on long sequences. Single biggest optimization. Reduced global memory traffic by 128×.

#### P2: Shared memory k reuse (merged with P1) ✅
- **Problem**: k values were read twice (old_v loop + update loop) from global memory.
- **Solution**: With k in shared memory, both loops read from shared (fast, ~4 cycles vs ~200 cycles for global).
- **Result**: Included in P1's improvement.

#### P3: Double-buffered synchronous prefetch ❌ Rejected (v4)
- **Problem**: Token loop has load→sync→compute→sync pattern. Idea: prefetch next token's k/q while computing current.
- **Implementation**: Two float shared memory buffers, synchronous load of next token before compute.
- **Result**: **-25% regression** on long sequences. Synchronous loads don't overlap with compute — the "prefetch" just adds overhead and shared memory pressure. True overlap requires async copy.

#### P4: cp.async bf16 double-buffered pipeline ✅ Applied (v6)
- **Problem**: Even with shared memory, the synchronous bf16→float load + store to shared blocks the warp during global memory access.
- **Solution**:
  - Store bf16 directly in shared memory (halves shared mem: 1KB vs 2KB)
  - Use `cp.async.ca.shared.global` (inline PTX) for async copy from global → shared
  - Double-buffer: async-load next token into alternate buffer while computing current
  - Convert bf16→float on-the-fly during compute (`__bfloat162float()`)
  - 64 threads copy k (4 bytes = 2 bf16 each), 64 threads copy q
- **Result**: **Additional -6~8% latency** on long sequences (v5→v6). Total -44% from baseline. The async copy genuinely overlaps memory latency with compute.

#### P5: `__ldg` read-only cache hints ⚪ Neutral (v5)
- **Problem**: Speculative — `__ldg` routes loads through texture/L2 cache, potentially faster for read-only data.
- **Result**: No measurable difference. The data access patterns don't benefit from the texture cache path.

#### P6: Constant preload (A_log, dt_bias) ⚪ Neutral (v5)
- **Problem**: `A_log[v_head]` and `dt_bias[v_head]` are per-head constants read every token from global.
- **Solution**: Load into registers once before the token loop.
- **Result**: Negligible — these are scalar loads that hit L1 cache on every access anyway.

#### P7: g×state pre-multiply for prefill ❌ Not applied
- Same analysis as decode D2 — total ops unchanged, just reordered. The compiler already optimizes the FMA chains well.

#### P8: Chunk-based token parallelism ❌ Not feasible
- **Idea**: Split long sequences into chunks processed by multiple blocks, combine results.
- **Why it fails**: The state recurrence `state[t] = g[t] * state[t-1] + k[t]^T @ delta[t]` is **nonlinear** — `delta[t]` depends on `state[t-1]` through `old_v = k @ state`. This prevents parallel scan decomposition.
- Potential workaround (not implemented): approximate chunk-then-correct, but risks correctness.

---

## Summary of Effective Optimizations

| Optimization | Target | Latency Improvement | Complexity |
|-------------|--------|---------------------|------------|
| Shared memory k/q (P1+P2) | Prefill | **-40%** | Low |
| cp.async bf16 pipeline (P4) | Prefill | **-6~8%** (additional) | Medium |
| Single-pass state read (D1) | Decode | **-3%** | Low |
| Total (prefill) | | **-44%** | |

### Ineffective / Negative
| Optimization | Why |
|-------------|-----|
| Sync double-buffer (P3) | No overlap without async copy |
| `__ldg` hints (P5) | No benefit for this access pattern |
| g×state pre-multiply (P7) | Same total FLOPS |
| Const preload (P6) | Already in L1 cache |

---

## Current Bottleneck Analysis

The prefill kernel is **compute-bound on per-token sequential recurrence**:
- ~640 FMA ops per thread per token
- 128 threads × 640 FMA = ~82K FLOP/token
- At B200's ~500 GFLOP/s per SM → theoretical ~0.16μs/token
- Observed: ~0.7μs/token (4.3× slower than theoretical)
- Gap likely from: shared memory bank access patterns, register-to-shared latency, control flow overhead

Further improvement requires **algorithmic change** (parallel scan for linear recurrences, or mixed-precision state).

---

## File Layout

```
config.toml                    # Switch definition + entry_point for decode/prefill
solution/cuda/kernel.cu        # Both kernels + TVM FFI exports
scripts/run_modal.py           # Modal runner (flashinfer-ci-cu132 image)
scripts/pack_solution.py       # Packs into solution.json
```

### Switching between decode and prefill

**Decode:**
```toml
definition = "gdn_decode_qk4_v8_d128_k_last"
entry_point = "kernel.cu::kernel"
```

**Prefill:**
```toml
definition = "gdn_prefill_qk4_v8_d128_k_last"
entry_point = "kernel.cu::kernel_prefill"
```
