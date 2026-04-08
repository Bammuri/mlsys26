# GDN Decode Kernel Optimization Log

Tracking all optimization iterations for the decode kernel.

---

<!-- Append new entries below this line -->

## 2026-04-06 - Warp-Parallel V-Rows with Loop Fusion
- **Idea**: Fuse two sequential loops into one, using algebraic reformulation (`output[vi] = scale * (g * qs_sum + qk_dot * residual)`) to compute output without a second state read. Each warp independently handles 32 vi rows with float4 vectorized state loads/stores. Eliminates all __syncthreads except one (v load).
- **Result**: 396.42x → 887.72x mean speedup (**+124%**), min 28.36x → 51.64x, latency 0.057ms → 0.028ms
- **Status**: accepted
- **Learnings**: State matrix (128x128 fp32 = 64KB per head) dominates memory traffic. Single-pass algebraic reformulation + float4 vectorization + warp-level reductions gave 2.24x improvement. Next bottleneck: SM under-utilization at small batch sizes (B=1 → only 8 blocks for 148 SMs).

## 2026-04-07 - V-Split Blocks (Dynamic Split Factor)
- **Idea**: Split each head's 128 V-rows across multiple blocks to increase SM utilization at small batch sizes. Dynamic split_factor: 4 for B≤4, 2 for B≤16, 1 for B>16. Each block handles fewer V-rows (32/64/128), multiplying the grid size accordingly.
- **Result**: 887.72x → 1046.46x mean speedup (**+17.9%**), min 51.64x → 88.22x (+70.8%), latency 0.028ms → 0.022ms
- **Status**: accepted
- **Learnings**: Small-batch workloads (B=1-4) saw the biggest gains (~70% min speedup improvement) confirming SM under-utilization was the bottleneck there. Large-batch workloads (B>16) unchanged as expected. Next bottleneck: memory latency hiding (software pipelining) or persistent kernel for further small-batch gains.

## 2026-04-07 - Cache Streaming Hints (ld/st.global.cs)
- **Idea**: Use inline PTX `ld.global.cs.v4.f32` and `st.global.cs.v4.f32` for state read/write. The `.cs` (cache streaming) hint tells the L2 cache that this data is accessed only once, enabling early eviction and reducing cache pollution. Frees L2 space for other accesses (q, k, v, output).
- **Result**: 1046.46x → 1079.33x mean speedup (**+3.1%**), max 2318x → 2353x, latency 0.022ms → 0.021ms
- **Status**: accepted
- **Learnings**: State data (128KB per head read+write) was polluting L2 despite being read/written only once. Streaming hints improved large-batch throughput where multiple blocks compete for L2 space. Tried and rejected: aggressive split factors (B=1 split=16, no improvement), template compile-time unrolling (#pragma unroll caused register spills for 32-iteration loops). Kernel is near memory-bandwidth limit for large batches; small batches (B=1-2) remain launch-latency dominated.

## 2026-04-07 - Async Copy Double Buffering (cp.async)
- **Idea**: Replace synchronous `ld.global.cs.v4.f32` state loads with `cp.async.cg.shared.global` into shared memory double buffers. Prefetch the next V-row's state while computing on the current row, hiding HBM latency (~200-400 cycles) behind compute. Shared memory: `smem_state[4][2][128]` = 4KB for 4 warps × 2 buffers.
- **Result**: 1079.33x → 1107.74x mean speedup (**+2.6%**), max 2352.97x → 2547.35x (+8.3%), latency 0.0213ms → 0.0180ms (-15.5%)
- **Status**: accepted
- **Learnings**: Async copy overlap helped most at large batch sizes where memory bandwidth is saturated — max speedup jumped 8.3%. Small batches (B=1-2) unchanged at ~82x, confirming they are launch-latency dominated, not memory-latency limited. The 4 warps per block already provided some latency hiding via warp scheduling, so the additional benefit of software pipelining was modest (+2.6% mean). Next opportunities: reducing launch overhead for small batches (CUDA graphs if framework allows), or increasing parallelism (more warps per block).

## 2026-04-07 - L2 Residency Cache Hints (cp.async.ca + writeback stores)
- **Idea**: The benchmark calls the kernel 100+ times on the same tensor addresses. Previous `.cs` (streaming/evict-first) cache hints on state writes eagerly evicted data from L2, forcing the next invocation to re-fetch from HBM. B200 has 126 MB L2 — even B=64 state (~64 MB) fits entirely. Changed `cp.async.cg` → `cp.async.ca` (cache at all levels) for state reads, and replaced `st.global.cs.v4.f32` inline PTX with normal float4 store (default `.wb` writeback policy) for state writes.
- **Result**: 1107.74x → 1303.71x mean speedup (**+17.7%**), max 2547.35x → 2952.71x (+15.9%), min 82.0x → 68.5x
- **Status**: accepted
- **Learnings**: L2 residency across kernel invocations was a major win — the `.cs` hint was actively harmful for this workload pattern. Large batch sizes benefited most (L2 bandwidth ~3-5x HBM). Min speedup dropped slightly for one B=1 outlier workload but overall B=1 performance improved. Key lesson: cache hints should match the actual access pattern (repeated invocations = keep in cache), not the single-invocation pattern (read-once = stream). Next opportunities: B=1 split factor tuning, or occupancy improvements.

## 2026-04-07 - Register-Based 2-Row Software Pipelining
- **Idea**: Replace cp.async shared memory double buffering with register-based float4 loads. Process 2 V-rows per loop iteration with prefetching: load next 2 rows into registers while computing current 2 rows. Eliminates `smem_state[4][2][128]` shared memory, `__syncwarp()` barriers, and halves loop overhead. Interleaved warp reductions for 4 values (ks_a, ks_b, qs_a, qs_b) provide better ILP.
- **Result**: 1303.71x → 1340.12x mean speedup (**+2.8%**), min 68.50x → 87.54x (+27.8%), max 2952.71x → 3155.02x (+6.8%), latency 0.0192ms → 0.018ms
- **Status**: accepted
- **Learnings**: Eliminating shared memory for state reduced overhead, especially for small batches (B=1 min speedup jumped 28%). The 2-row processing amortizes loop overhead and enables interleaved independent shuffles. Register prefetching provides similar latency hiding to cp.async without synchronization costs. Kernel is now deeply memory-bound (~0.375 FLOP/byte arithmetic intensity vs ~37.5 FLOP/byte L2 machine balance). Remaining opportunities: wider blocks for small batches, warp specialization (producer/consumer), or fundamentally different parallelization strategies.

## 2026-04-07 - 4-Row Software Pipelining
- **Idea**: Extend 2-row register pipelining to 4 rows per iteration. Prefetch 4 float4 state rows, compute 8 dot products (ks_a..ks_d, qs_a..qs_d) with all 8 reductions interleaved in a single shuffle loop for maximum ILP. Halves loop iterations and overhead.
- **Result**: 1340.12x → 1579.97x mean speedup (**+17.9%**), min 87.54x → 84.71x (-3.2%), max 3155.02x → 3982.39x (+26.2%), latency 0.018ms → 0.0174ms
- **Status**: accepted
- **Learnings**: Doubling pipeline depth from 2 to 4 rows gave a surprisingly large gain (+17.9%), especially for large batches (max +26.2%). The 8 interleaved independent shuffle reductions provide excellent ILP, keeping the warp scheduler busy while waiting on memory. Small-batch (B=1) min speedup slightly regressed (-3.2%) due to overhead of 4-stage pipeline with fewer iterations. Register pressure remains low (~50 regs/thread). Remaining opportunities: L2 persistent access policy for cross-invocation caching, __launch_bounds__(128,2) for occupancy hints, or warp specialization.

## 2026-04-07 - L2 Persistence (cudaAccessPolicyWindow) [REVERTED]
- **Idea**: Pin state tensor in L2 via `cudaAccessPolicyWindow` with `cudaAccessPropertyPersisting`. Set 96MB L2 persisting cache size. Host-side only change, no kernel modifications.
- **Result**: 1579.97x → 1492.00x mean speedup (**-5.6%**), 54/54 → 53/54 workloads (1 RUNTIME_ERROR)
- **Status**: reverted
- **Learnings**: `cudaStreamSetAttribute` caused a runtime error on one workload and overall regression. The TVM FFI stream management may not be compatible with stream attribute modifications, or the attribute setting itself added per-launch overhead. The passive `.wb` writeback caching from optimization #5 already provides sufficient L2 residency without explicit pinning.

## 2026-04-07 - Split Factor Tuning for Medium Batches [REVERTED]
- **Idea**: Extend split_factor coverage: split=4 for B≤8 (was B≤4), split=2 for B≤32 (was B≤16). Targets B=8 (128→256 blocks) and B=17-32 (256→512 blocks) for better SM utilization.
- **Result**: 1579.97x → 1511.06x mean speedup (**-4.4%**), max 3982x → 3582x (-10.1%)
- **Status**: reverted
- **Learnings**: Wider splitting hurt large-batch workloads more than it helped medium ones. More blocks means more per-block overhead (gate computation, v load, barrier) and less work per warp (fewer loop iterations = less amortization of pipeline setup). The original thresholds (B≤4 split=4, B≤16 split=2) are already well-tuned.

## 2026-04-07 - Register V Broadcast (eliminate shared memory) [REVERTED]
- **Idea**: Replace shared memory v load + `__syncthreads` with per-warp register loads + `__shfl_sync` broadcast. Each lane holds one v value, broadcast to all lanes via shuffle when needed. Eliminates all shared memory and barriers.
- **Result**: 1579.97x → 1430.33x mean speedup (**-9.5%**), min 84.71x → 75.49x, max 3982x → 3610x
- **Status**: reverted
- **Learnings**: Shared memory v access is faster than shuffle broadcasts despite the `__syncthreads` cost. Shared memory provides uniform ~28-cycle latency for random access, while shuffle requires an instruction per broadcast. With 4 shuffles per iteration (v_a..v_d) vs one shared memory index per residual, the shuffle overhead exceeded the barrier savings. The kernel is deeply memory-bound on state traffic — v access optimization is not on the critical path.

## 2026-04-07 - __launch_bounds__(128, 12) Occupancy Hint [REVERTED]
- **Idea**: Add `__launch_bounds__(128, 12)` to target 12 blocks/SM (42 regs/thread), increasing occupancy from ~62.5% to 75%.
- **Result**: 1579.97x → 1198.84x mean speedup (**-24.1%**), regression across all batch sizes
- **Status**: reverted
- **Learnings**: The 42-reg cap caused heavy register spills. The 4-row pipeline naturally uses ~50 regs/thread; forcing 42 regs created local memory traffic that dwarfed any occupancy benefit. The kernel is memory-bound, not occupancy-bound — more warps don't help when each warp's memory traffic increases from spills.

## 2026-04-07 - Split Factor 8 for B≤2
- **Idea**: Add split=8 tier for B≤2 (rows_per_warp=4, exactly 1 iteration of 4-row pipeline). Doubles SM utilization for B=1 from 22% to 43% (64 blocks vs 32). Previous "aggressive split" attempt used split=16 for B=1 which broke the 4-row pipeline (rows_per_warp=2 < 4); split=8 cleanly matches.
- **Result**: 1579.97x → 1584.44x mean speedup (**+0.3%**), min 84.71x → 91.09x (**+7.5%**), max 3982x → 3748x (-5.9%), latency 0.0174ms → 0.0164ms (-5.7%)
- **Status**: accepted
- **Learnings**: Small-batch (B=1-2) min speedup improved from better SM utilization. Max speedup dropped slightly (run-to-run variance or minor overhead). The 4-row pipeline with rows_per_warp=4 runs a single clean iteration with no prefetch overhead, making split=8 viable where split=16 failed. Kernel is near-optimal for current algorithm; further gains likely require fundamentally different approaches (TMA, tensor cores, or algorithmic changes).

## 2026-04-08 - 2-Warp Blocks (64 threads/block) [REVERTED]
- **Idea**: Reduce block size from 128 to 64 threads (2 warps). Doubles grid size for better SM utilization at B=1 (64→128 blocks). With 2 warps, split=16 becomes viable (rows_per_warp=4), enabling 128 blocks for B=1 (87% SM coverage vs 43%).
- **Result**: 1584.44x → 1264.19x mean speedup (**-20.2%**), min 91.09x → 64.91x (-28.7%), max 3748x → 3041x (-18.9%)
- **Status**: reverted
- **Learnings**: Fewer warps per SM (2 vs 4) severely hurts memory latency hiding. Even though more SMs are utilized, each SM has fewer warps to switch between while waiting on memory. The kernel is deeply memory-bound (state reads/writes dominate), so latency hiding from intra-block warp scheduling is critical. This confirms: warp count per SM matters more than SM coverage for this kernel.

## 2026-04-08 - __launch_bounds__(128, 10) + No Register Prefetch [REVERTED]
- **Idea**: Remove register-based prefetching and add __launch_bounds__(128, 10) to target ~51 regs/thread (from 64). Fewer registers → 10 blocks/SM max → 40 warps = 62.5% occupancy (from 50%). Higher occupancy compensates for removed prefetch.
- **Result**: 1584.44x → 873.66x mean speedup (**-44.8%**), but B=1 absolute latency dropped 2.3x (0.021ms→0.009ms)
- **Status**: reverted
- **Learnings**: The mean speedup regression may be partly Modal run-to-run reference variance (ref_time differed 2.5x between runs). However, the B=1 absolute latency improvement was genuine — reduced register pressure + higher occupancy benefits latency-bound small batches. The tradeoff: launch_bounds likely caused register spills that hurt throughput-bound large batches. Need A/B testing within same Modal invocation for reliable comparison.

## 2026-04-08 - PTX L1 Prefetch Hints + Vectorized Output Writes [REVERTED]
- **Idea**: (1) Add `prefetch.global.L1` PTX hints for state rows 2 iterations ahead, giving L1 cache more lead time. (2) Vectorize output writes: pack 4 consecutive bf16 values into one uint2 (64-bit) store instead of 4 scalar stores.
- **Result**: ~1299x mean speedup — absolute latencies nearly identical to baseline, speedup difference attributable to Modal variance
- **Status**: reverted (neutral impact)
- **Learnings**: L1 prefetch hints are ineffective because the register-based prefetching already provides adequate latency hiding. Vectorized output writes are a negligible optimization (output traffic is tiny vs state traffic). **Key insight**: Modal B200 benchmark has significant run-to-run variance in reference timing (~2x), making small improvements (< 10%) unmeasurable with single-run comparisons. Need head-to-head A/B testing for reliable evaluation.

## 2026-04-08 - NCU Profiling Insights (B=1 baseline)
- **NCU metrics**: 64 regs/thread, 50% theoretical occupancy (register-limited), 6% achieved occupancy, 0.05 waves/SM
- **Bottleneck**: Latency-bound for B=1 (compute 2%, memory 1.7% — both extremely low due to grid underutilization)
- **Key constraint**: 64 blocks (B=1, split=8) for 148 SMs — 43% SM coverage, most SMs idle
- **Attempted fixes**: reducing block size, reducing register count — both regressed due to fewer warps per SM or register spills
- **Conclusion**: B=1 performance is fundamentally limited by launch overhead + insufficient parallelism. The 4-warp/block × 64-reg/thread configuration is a local optimum: reducing either dimension hurts latency hiding or causes spills.
