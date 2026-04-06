# GDN Decode Kernel Optimization Log

Tracking all optimization iterations for the decode kernel.

---

<!-- Append new entries below this line -->

## 2026-04-06 - Warp-Parallel V-Rows with Loop Fusion
- **Idea**: Fuse two sequential loops into one, using algebraic reformulation (`output[vi] = scale * (g * qs_sum + qk_dot * residual)`) to compute output without a second state read. Each warp independently handles 32 vi rows with float4 vectorized state loads/stores. Eliminates all __syncthreads except one (v load).
- **Result**: 396.42x → 887.72x mean speedup (**+124%**), min 28.36x → 51.64x, latency 0.057ms → 0.028ms
- **Status**: accepted
- **Learnings**: State matrix (128x128 fp32 = 64KB per head) dominates memory traffic. Single-pass algebraic reformulation + float4 vectorization + warp-level reductions gave 2.24x improvement. Next bottleneck: SM under-utilization at small batch sizes (B=1 → only 8 blocks for 148 SMs).
