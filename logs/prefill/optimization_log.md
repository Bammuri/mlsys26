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
