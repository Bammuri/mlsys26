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
