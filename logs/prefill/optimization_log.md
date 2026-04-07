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
