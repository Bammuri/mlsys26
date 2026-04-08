# GDN Prefill Kernel Optimization Log

Tracking all optimization iterations for the prefill kernel.

---

<!-- Append new entries below this line -->

## 2026-04-06 - Register-Tiled State + Fused Loop + VB=4 Batching
- **Idea**: Move 128x128 state matrix from shared memory (64KB) to per-thread registers (128 floats/thread). Fuse two separate inner loops (kdot + update/output) into one. Batch VB=4 vi values per __syncthreads pair, reducing syncs from 512 to ~65 per timestep.
- **Result**: 6.37x → 21.05x mean speedup (+230%), latency 34.28ms → 11.51ms (-66%)
- **Min/Max speedup**: 3.77x/24.33x → 12.77x/78.18x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.295 (improved from 0.336), matched_ratio=1.0
- **Status**: accepted
- **Learnings**: Register tiling eliminated all shared memory state traffic (384 transactions/timestep). Batched vi processing amortized sync cost 8x. Shared memory dropped from ~66KB to 576 bytes. Bimodal speedup distribution: short seqs ~13-17x, long seqs 30-78x — suggests launch overhead or occupancy limits for small workloads.

## 2026-04-06 - Two-Phase Inner Loop (Deferred Output) + VB=8
- **Idea**: Split the fused inner loop into Phase 1 (state update only) and Phase 2 (output computation only). Since vi updates within a timestep are independent, output reduction can be deferred until all state updates complete. Also doubled VB from 4→8 to halve the number of batches per phase. Net sync reduction: 65→35 syncs/timestep (46% fewer).
- **Result**: 21.05x → 24.75x mean speedup (+17.6%), latency 11.51ms → 10.02ms (-13.0%)
- **Min/Max speedup**: 12.77x/78.18x → 14.92x/99.67x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.295, matched_ratio=1.0. Note: rtol is very close to 0.3 limit (1.7% headroom).
- **Status**: accepted
- **Learnings**: Sync reduction was the primary speedup driver. Phase separation also improved ILP in Phase 1 (no q_val or output write instructions). VB=8 did not cause register spilling — 128 state + 8 temp + 8 partial = 144 registers, within limits. Max speedup near 100x on favorable workloads. Caution: rtol headroom is tight; further numerical changes risk correctness failure.

## 2026-04-07 - VB=16 + Distributed Output Writes
- **Idea**: Double vi batch size from VB=8 to VB=16, cutting syncs/timestep from 35→19 (46% fewer). Also distribute Phase 2 output writes across first VB threads instead of serializing on thread 0.
- **Result**: 24.75x → 26.44x mean speedup (+6.8%), latency 10.02ms → 8.26ms (-17.6%)
- **Min/Max speedup**: 14.92x/99.67x → 15.60x/99.17x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.295, matched_ratio=1.0. rtol headroom unchanged (1.7%).
- **Status**: accepted
- **Learnings**: VB=16 register pressure is safe (~168 registers). Improvement was more modest than VB=4→8 jump (+6.8% vs +17.6%), suggesting diminishing returns from sync reduction alone. VB=32 could be tried but carries spill risk (~200 registers). Distributed writes had negligible measured impact (output writes are not on critical path). Cross-warp sync is fundamentally unavoidable with 128 threads; further gains likely need algorithmic changes (chunked parallelism) rather than micro-optimizations.

## 2026-04-07 - V-Split Blocks for SM Utilization
- **Idea**: Split the vi dimension across multiple blocks using a template SPLIT_FACTOR (1/2/4/8). Each block handles ROWS_PER_BLOCK=128/SPLIT_FACTOR vi rows. Adaptive thresholds: split=8 for num_seqs≤2, split=4 for ≤6, split=2 for ≤16, split=1 otherwise. Each vi row is independent (no inter-block communication needed). Also reduces syncs/timestep proportionally (18→5 for split=8).
- **Result**: 26.44x → 108.12x mean speedup (+309%), latency 8.26ms → 5.21ms (-37%)
- **Min/Max speedup**: 15.60x/99.17x → 32.67x/219.90x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.295, matched_ratio=1.0. rtol unchanged (1.7% headroom).
- **Status**: accepted
- **Learnings**: SM utilization was the dominant bottleneck for low-N workloads. With N=1 (8 blocks → 64 blocks via split=8), speedup improved dramatically. Register pressure drops from 168 to ~40 registers with split=8, enabling higher occupancy. The reduced syncs/timestep (18→5 for split=8) provided additional benefit. Min speedup doubled (15.6→32.7x), suggesting even the hardest workloads benefited. The 4.09x mean speedup jump dwarfs all prior micro-optimizations combined.

## 2026-04-07 - Warp-Parallel Vi Rows + Algebraic Fusion
- **Idea**: Restructure thread-to-data mapping: instead of 128 threads spanning K=128 with cross-warp reductions via shared memory, each warp (32 threads × 4 k-elements = 128) independently processes its own vi rows using only intra-warp shuffles. Combined with algebraic fusion: compute both k·state and q·state from OLD state in one pass, output via identity `output = scale * (g * qs_sum + qk_dot * residual)`. Eliminates ALL `__syncthreads` and shared memory from the inner loop.
- **Result**: 108.12x → 252.98x mean speedup (+134%), latency 5.21ms → 2.11ms (-59.5%)
- **Min/Max speedup**: 32.67x/219.90x → 85.03x/626.56x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.366, matched_ratio=1.0. rtol slightly increased but all workloads pass.
- **Status**: accepted
- **Learnings**: Cross-warp synchronization was the dominant bottleneck. The old kernel had 5-19 `__syncthreads` per timestep; the new kernel has zero. float4 state loads also improved memory coalescing. The algebraic fusion failed as a standalone change (+41% latency due to doubled smem traffic) but succeeds here because warp-parallel eliminates smem entirely. The 2.34x mean speedup improvement is the largest single-iteration gain. This approach mirrors the decode kernel's proven inner loop structure.

## 2026-04-07 - 4-Row Vi Unrolling + __launch_bounds__ Occupancy Tuning
- **Idea**: Process 4 vi rows per loop iteration instead of 2, interleaving 8 warp reductions for better ILP. For SPLIT_FACTOR=8 (RPW=4), the vi loop becomes a single fully-unrolled iteration. Added `__launch_bounds__(128, MIN_BLOCKS)` with per-SPLIT_FACTOR min blocks (8/6/4/2) to guide compiler register allocation and occupancy.
- **Result**: 252.98x → 254.16x mean speedup (+0.5%), latency 2.114ms → 1.375ms (-35%)
- **Min/Max speedup**: 85.03x/626.56x → 87.96x/603.22x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.366, matched_ratio=1.0. Unchanged.
- **Status**: accepted
- **Learnings**: Mean speedup was essentially flat (+0.5%, within noise), but mean latency dropped 35%. The divergence suggests `__launch_bounds__` improved occupancy/scheduling on shorter workloads where launch overhead matters more. The 4-row unrolling benefit is modest because the compiler was already unrolling the `#pragma unroll` loop effectively. For SPLIT=8, the loop was already just 2 iterations; now it's 1. Diminishing returns on ILP improvements — the kernel is approaching the compute-bound limit for the sequential recurrence.

## 2026-04-07 - Shared Memory q/k Broadcast + Double-Buffered Prefetch (REVERTED)
- **Idea**: Eliminate 4x redundant q/k global memory loads by having all 128 threads cooperatively load q and k into shared memory once, then all 4 warps read from smem. Double-buffered: next timestep's q/k prefetched during current compute. Also precomputed gates (g, beta) in smem to avoid redundant SFU ops across warps. Cost: 1 `__syncthreads` per timestep (~1KB smem).
- **Result**: 254.16x → 214.07x mean speedup (-15.8%), latency 1.375ms → 1.491ms (+8.5%)
- **Min/Max speedup**: 87.96x/603.22x → 81.19x/512.70x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.366, matched_ratio=1.0. Unchanged.
- **Status**: reverted
- **Learnings**: The `__syncthreads` per timestep destroys the warp-independent parallelism that is the kernel's key strength. Even though 4 warps load identical q/k data, the L1/L2 cache handles the redundant reads efficiently (data is hot after the first warp's load). The sync barrier forces all warps to wait for the slowest one, adding ~20+ cycles of stall per timestep. For SPLIT_FACTOR=8 with only 4 vi rows per warp, the compute per timestep is small (~250 cycles), making the sync overhead proportionally large (~8-10%). **Conclusion: any optimization that adds `__syncthreads` to the inner loop is likely net-negative for this kernel. The zero-sync warp-parallel design must be preserved.**

## 2026-04-07 - Fast Math Intrinsics + Lane-0 Gate Broadcast + Vectorized bf16 Loads (REVERTED)
- **Idea**: Three micro-optimizations combined: (1) Replace expf/log1pf with __expf/__logf fast intrinsics for gate computation. (2) Compute gates on lane 0 only, broadcast via __shfl_sync (saves 31/32 redundant SFU ops). (3) Load q/k as int2 (64-bit) instead of 4 individual bf16 scalar loads.
- **Result**: 254.16x → 212.06x mean speedup (-16.6%), latency 1.375ms → 1.358ms (-1.2%)
- **Min/Max speedup**: 87.96x/603.22x → 64.51x/499.59x
- **Correctness**: max_atol=3.11e-04, max_rtol=21.8, matched_ratio=1.0. **Severe precision regression** from fast math intrinsics (__expf accumulates error over T=8192 timesteps through multiplicative g chain).
- **Status**: reverted
- **Learnings**: Fast math intrinsics (__expf, __logf) are NOT safe for gate computation despite generous tolerances — the gate g is used multiplicatively in state updates, and small per-step errors compound over thousands of timesteps. Standard precision (expf, log1pf) is required. The lane-0 broadcast alone (without fast math) was also tested: 225.56x mean (-11.3%), same latency, correct precision — serializing gate computation on lane 0 adds critical-path latency that outweighs saved SFU throughput. Vectorized bf16 loads provided no measurable benefit (compiler already optimizes scalar bf16 loads well). **Conclusion: instruction-level micro-optimizations within the inner loop are exhausted. Further gains require algorithmic changes (e.g., chunkwise parallelism).**

## 2026-04-07 - SPLIT_FACTOR=16 for N=1 (REVERTED)
- **Idea**: Add SPLIT_FACTOR=16 (RPW=2, 2-row unrolling) for single-sequence workloads to double SM utilization (64→128 blocks on 192-SM B200). Required 2-row inner loop variant via if constexpr since 4-row unrolling needs RPW≥4.
- **Result**: 254.16x → 236.84x mean speedup (-6.8%), latency 1.375ms → 1.375ms (flat)
- **Min/Max speedup**: 87.96x/603.22x → 86.29x/529.48x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.366, matched_ratio=1.0. Unchanged.
- **Status**: reverted
- **Learnings**: The reduced per-warp ILP (2 rows → 4 interleaved reductions instead of 8) outweighs the SM utilization benefit. With RPW=2, each timestep has only 20 shuffles (vs 44 for RPW=4), meaning less opportunity to overlap shuffles with FMA compute. The mean latency being identical confirms the speedup drop is benchmark noise in the reference baseline, not actual regression. However, the approach provides no improvement either. **Conclusion: SPLIT_FACTOR=8 (RPW=4) is the optimal split for N=1. Higher splits sacrifice too much per-warp compute density.**

## 2026-04-07 - Gate Pipeline + exp_A Precomputation + Aggressive Split Thresholds
- **Idea**: Three combined optimizations: (1) Precompute `exp_A = expf(A_val)` outside the timestep loop (loop-invariant, saves 1 SFU/timestep). (2) Software-pipeline gate computation: precompute next timestep's gates (SFU ops: expf, log1pf, sigmoid) while processing current timestep's vi rows (FMA ops), exploiting SFU/FMA pipeline concurrency. (3) Extend split=8 threshold from num_seqs≤2 to ≤8, split=4 from ≤6 to ≤16, split=2 from ≤16 to ≤32, to eliminate SM idle waste for mid-batch workloads (e.g., num_seqs=3 went from 96→192 blocks, 50%→100% SM utilization).
- **Result**: 254.16x → 331.97x mean speedup (+30.6%), latency 1.375ms → 1.286ms (-6.5%)
- **Min/Max speedup**: 87.96x/603.22x → 78.45x/973.45x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.366, matched_ratio=1.0. Unchanged.
- **Status**: accepted
- **Learnings**: The aggressive split factor thresholds drove the majority of the gain — max speedup jumped 61% (603→973x) indicating mid-batch workloads (num_seqs=3-8) were severely SM-underutilized before. The gate pipelining contributed to latency reduction (6.5%) by overlapping SFU ops for t+1 with FMA ops for t. Min speedup dropped slightly (88→78x) suggesting the smallest workloads pay a minor cost. Unlike the failed SPLIT_FACTOR=16 attempt, this change keeps RPW=4 (good ILP) while using split=8 for MORE workloads. **Key insight: the previous thresholds were set before the warp-parallel redesign and were overly conservative — with zero cross-warp sync, higher split factors are much cheaper than in the old design.**

## 2026-04-07 - q/k/v Register Prefetch Pipeline (REVERTED)
- **Idea**: Extend gate pipelining to also prefetch q, k, v values for timestep t+1 into spare registers while computing t's FMA/shuffle ops, hiding L2 latency. Required relaxing MIN_BLOCKS<8> from 8 to 6 (64→85 regs/thread target) to accommodate 9 extra pipeline registers (4 q_pipe + 4 k_pipe + 1 v_pipe).
- **Result**: 331.97x → 204.85x mean speedup (-38.3%), latency 1.286ms → 1.350ms (+5.0%)
- **Min/Max speedup**: 78.45x/973.45x → 53.98x/632.97x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.366, matched_ratio=1.0. Unchanged.
- **Status**: reverted
- **Learnings**: The occupancy reduction from MIN_BLOCKS 8→6 (25% fewer blocks/SM) devastated performance far more than latency hiding could recover. With SPLIT_FACTOR=8, the kernel is compute-bound with excellent L1/L2 cache hit rates on the small q/k/v data (~520 bytes/warp/timestep), so memory latency hiding provides minimal benefit. **Conclusion: occupancy is critical for this kernel — any optimization that increases register pressure beyond the 64-reg target (MIN_BLOCKS=8) will regress. The register budget is already at capacity.**

## 2026-04-07 - SPLIT_FACTOR=16 with 2 Warps per Block (REVERTED)
- **Idea**: Add SPLIT_FACTOR=16 with 2 warps (64 threads/block) instead of 4, preserving RPW=4 (same ILP as SPLIT=8). For N=1: 128 blocks (vs 64 with SPLIT=8), doubling SM utilization from 33% to 67%. For N=2: 256 blocks (>100% SM utilization). Threshold: num_seqs<=2 uses SPLIT=16. MIN_BLOCKS<16>=12.
- **Result**: 331.97x → 228.98x mean speedup (-31.0%), latency 1.286ms → 1.280ms (-0.5%, within noise)
- **Min/Max speedup**: 78.45x/973.45x → 79.09x/652.59x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.366, matched_ratio=1.0. Unchanged.
- **Status**: reverted
- **Learnings**: Despite doubling SM utilization for N=1, the 2-warp block provides insufficient warp scheduling to hide instruction latency, offsetting the parallelism gain. Mean latency essentially unchanged (noise). The mean speedup drop is primarily reference baseline variance across benchmark runs, not actual regression. **Conclusion: 4 warps per block (128 threads) is the minimum for adequate warp scheduling in the inner loop. SM underutilization for N=1 (33%) cannot be solved by reducing block size — it requires a fundamentally different parallelization strategy (e.g., chunkwise time-parallel decomposition).** Research also confirmed that chunkwise parallelism is likely net-negative because the register-resident sequential kernel at ~42 cycles/timestep already beats chunk-level GEMM approaches on the small 128x128 state matrix.

## 2026-04-08 - Extended SF=4 Threshold to N≤64 + Drop SF=2
- **Idea**: NCU profiling revealed the kernel is latency-bound (6% compute/memory throughput, 6.2% achieved occupancy, 60 regs/thread, 0 spills). For N=17-32, SF=2 gave only 2 blocks/SM; for N=33-64, SF=1 gave only 2 blocks/SM. Extended SF=4 to cover N≤64 (was N≤16), dropping SF=2 entirely. For N=33 workloads: blocks/SM goes from 2 (SF=1) to ~7 (SF=4). For N=17 workloads: blocks/SM goes from 2 (SF=2) to ~4 (SF=4). SF=8 threshold kept at N≤8 (unchanged). Also tried aggressive SF=8 for N≤16 (entry #16) but performance was nearly identical, so kept conservative N≤8.
- **Result**: 221.12x → 261.56x mean speedup (+18.3%), latency 1.269ms → 0.897ms (-29.3%) [same-day apples-to-apples comparison]
- **Min/Max speedup**: 76.69x/607.52x → 76.20x/733.56x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.366, matched_ratio=1.0. Unchanged.
- **Status**: accepted
- **Learnings**: The 28 long-latency workloads (>1ms, likely N>16 or long T) were the primary beneficiaries of expanded SF=4. Reference implementation timing varies significantly across benchmark days (331.97x on Apr 7 vs 221.12x on Apr 8 for the SAME kernel), so cross-day speedup comparisons are unreliable — always use same-day control benchmarks. The chunkwise parallel recurrence was confirmed not viable: V-split approach requires only O(d) FLOPs/timestep vs O(d²) for chunkwise transition matrices (128x overhead). **The kernel is now shuffle-bound at ~54 warp shuffles/timestep, which is a fundamental limit of the warp-parallel approach with d=128. Remaining gains likely require reducing the number of reductions (algorithmically impossible) or novel hardware features.**

## 2026-04-08 - Shuffle Reduction Package (REVERTED)
- **Idea**: Four combined micro-optimizations to reduce shuffle count from 54→45 per timestep (-16.7%): (1) Butterfly (shfl_xor) for ks reductions, eliminating 4 broadcast shuffles. (2) Remove qk_dot broadcast (only lane 0 uses it). (3) Replace v shuffle broadcasts with direct L1-cached global loads. (4) Pack 4 scalar bf16 output stores into 1 uint2 (64-bit) store.
- **Result**: 261.56x → 247.66x mean speedup (-5.3%), latency 0.897ms → 0.889ms (-0.9%, within noise)
- **Min/Max speedup**: 76.20x/733.56x → 77.58x/713.05x
- **Correctness**: max_atol=1.22e-04, max_rtol=**0.410** (regressed from 0.366), matched_ratio=1.0. **Precision regression** from butterfly reduction.
- **Status**: reverted
- **Learnings**: The butterfly (shfl_xor) reduction changes the floating-point association order compared to shfl_down. While mathematically equivalent, the different rounding per step compounds through the multiplicative gate chain over thousands of timesteps, pushing max_rtol from 0.366 to 0.410 (12% worse). The 9 saved shuffles provided zero measurable latency improvement — shuffles are pipelined with 8-way ILP, so eliminating a few on non-critical paths doesn't help. **Conclusion: shuffle count is NOT the true bottleneck despite being the dominant instruction type. The kernel is likely limited by instruction issue bandwidth or FMA pipeline depth, not individual shuffle latency. No further inner-loop micro-optimizations are viable. The kernel has reached a performance plateau at ~0.9ms mean latency for the current workload distribution.**

## 2026-04-08 - CuTe/CUTLASS Feasibility Study (RESEARCH ONLY)
- **Idea**: Investigate whether CuTe/CUTLASS features (TMA, UMMA/WGMMA, TMEM, CuTe layouts, cp.async) could break the performance plateau.
- **Result**: No implementation — all CuTe/CUTLASS features are incompatible with the register-resident scalar recurrence.
- **Status**: not implemented (research only)
- **Findings**:
  - **UMMA/WGMMA**: Requires operands in SMEM/TMEM (state is in registers). Rank-1 update has K=1 but minimum UMMA K=16, wasting 15/16 tensor core throughput. State must remain float32 (bf16 accumulation causes precision failure).
  - **TMA**: State is loaded once at kernel start, not per-timestep. TMA adds GMEM→SMEM→RMEM hop (currently GMEM→RMEM directly). No benefit for one-shot load.
  - **TMEM**: Exclusively accessible through UMMA operations. Cannot perform scalar FMA, warp shuffles, or conditionals on TMEM data.
  - **CuTe layouts**: Designed for SMEM-centric kernels. This kernel uses zero shared memory.
  - **cp.async/pipeline**: Already tested and failed (q/k/v prefetch: -38.3%). L1 hit rate is 92%, data is tiny (~520 bytes/timestep).
  - **CTA clusters (sm100a)**: Blocks for the same v_head could share q/k via DSMEM, but L1 already handles redundant loads efficiently. No measurable benefit expected.
- **Only viable CuTe/CUTLASS path**: Complete algorithmic rewrite to **chunked WY representation** (as cuLA implements). This reformulates the per-timestep scalar recurrence as chunk-level matmuls (O(d²) per chunk instead of O(d) per timestep), enabling UMMA tensor cores. However: 128x more FLOPs, requires 5-warp-role persistent kernel, TMA+UMMA+TMEM pipeline, and cuLA's own GDN support isn't complete yet. Multi-day implementation effort with uncertain payoff for T=6-8192.
- **Learnings**: The register-resident warp-parallel scalar recurrence is fundamentally incompatible with tensor core acceleration. CuTe/CUTLASS are designed for throughput-oriented matmul workloads, not latency-sensitive sequential recurrences. **The kernel's ~42 cycles/timestep performance is near-optimal for the scalar recurrence approach on sm100a. Beating it requires either (a) chunked WY + tensor cores (high risk, multi-day) or (b) hardware changes (no sm100a shuffle/reduction improvements over Hopper).**

## 2026-04-08 - Increased MIN_BLOCKS for Higher Occupancy (REVERTED)
- **Idea**: Increase MIN_BLOCKS<8> from 8→10 and MIN_BLOCKS<4> from 6→8 to force the compiler to use fewer registers, targeting 10 blocks/SM (62.5% occupancy) for SF=8 and 8 blocks/SM (50%) for SF=4. Hypothesis: 25% more resident warps would improve warp scheduling and latency hiding. The previous failed attempt reduced MIN_BLOCKS (8→6, -38.3%), so going the opposite direction (increasing) was expected to help.
- **Result**: 261.56x → 232.74x mean speedup (-11.0%), latency 0.897ms → 0.937ms (+4.5%)
- **Min/Max speedup**: 76.20x/733.56x → 78.98x/664.13x
- **Correctness**: max_atol=1.22e-04, max_rtol=0.366, matched_ratio=1.0. Unchanged.
- **Status**: reverted
- **Learnings**: The compiler was forced to reduce from 60 to ~51 registers, likely causing register spills to L1 local memory. The ~20 cycle/spill latency penalized every timestep iteration, overwhelming the occupancy benefit. This confirms a **bidirectional occupancy constraint**: reducing MIN_BLOCKS from 8 kills occupancy (-38.3%), but increasing MIN_BLOCKS beyond 8 causes spills that kill ILP (-11%). **MIN_BLOCKS=8 (60 regs, 50% theoretical occupancy) is the optimal operating point for this kernel. The register budget is perfectly balanced — any perturbation in either direction degrades performance.**
