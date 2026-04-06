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
