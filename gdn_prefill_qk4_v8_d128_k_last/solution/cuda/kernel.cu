/*
 * GDN Prefill Kernel: gdn_prefill_qk4_v8_d128_k_last
 *
 * Variable-length prefill with recurrent state update.
 * State layout: [N, H=8, V=128, K=128] float32 (k-last)
 * GVA: q_heads=4, k_heads=4, v_heads=8 (2 v_heads per q/k head)
 *
 * Warp-parallel formula (per timestep, per vi row):
 *   g = exp(-exp(A_log) * softplus(a + dt_bias))
 *   beta = sigmoid(b)
 *   qk_dot = sum_k(q[k] * k[k])               (warp reduce, once per timestep)
 *   ks_sum = sum_k(k[k] * state[vi,k])         (warp reduce, from OLD state)
 *   qs_sum = sum_k(q[k] * state[vi,k])         (warp reduce, from OLD state)
 *   residual = beta * (v[vi] - g * ks_sum)
 *   state_new[vi,k] = g * state[vi,k] + k[k] * residual
 *   output[vi] = scale * (g * qs_sum + qk_dot * residual)
 *
 * Optimizations:
 *   - Warp-parallel: each warp independently processes its own vi rows (no cross-warp sync)
 *   - float4 state tiling: each thread holds 4 contiguous k-elements per vi row
 *   - Algebraic fusion: output computed from OLD state via identity (eliminates Phase 2)
 *   - Zero shared memory in inner loop: all reductions via warp shuffle
 *   - V values broadcast via __shfl_sync (no smem needed)
 *   - 4 vi rows per iteration for ILP (8 interleaved reductions)
 *   - V-split: SPLIT_FACTOR splits vi dimension across blocks for better SM utilization
 *   - __launch_bounds__ occupancy tuning per SPLIT_FACTOR
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/c_env_api.h>

using bf16 = __nv_bfloat16;

constexpr int NUM_Q_HEADS = 4;
constexpr int NUM_K_HEADS = 4;
constexpr int NUM_V_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int V_PER_Q = NUM_V_HEADS / NUM_Q_HEADS;  // 2
constexpr int NUM_WARPS = HEAD_DIM / 32;  // 4

__device__ __forceinline__ float softplus(float x) {
    return log1pf(expf(x));
}

/*
 * One block per (seq_idx, v_head, split_id) triple.
 * Grid: (num_seqs * NUM_V_HEADS * SPLIT_FACTOR,)
 * Block: (128,) — 4 warps, each warp independently handles ROWS_PER_WARP vi rows
 *
 * Each warp: 32 threads × 4 k-elements/thread = 128 k-elements (full K dimension)
 * State tiling: float4 per vi row per thread — reg_state[r] = state[vi, lane*4 : lane*4+4]
 * No shared memory or __syncthreads in the timestep loop.
 */
template <int SF> inline constexpr int MIN_BLOCKS = 2;
template <> inline constexpr int MIN_BLOCKS<8> = 8;
template <> inline constexpr int MIN_BLOCKS<4> = 6;
template <> inline constexpr int MIN_BLOCKS<2> = 4;

template <int SPLIT_FACTOR>
__global__ __launch_bounds__(128, MIN_BLOCKS<SPLIT_FACTOR>) void gdn_prefill_kernel(
    const bf16* __restrict__ q,         // [T, 4, 128]
    const bf16* __restrict__ k,         // [T, 4, 128]
    const bf16* __restrict__ v,         // [T, 8, 128]
    const float* __restrict__ state,    // [N, 8, 128, 128] k-last
    const float* __restrict__ A_log,    // [8]
    const bf16* __restrict__ a,         // [T, 8]
    const float* __restrict__ dt_bias,  // [8]
    const bf16* __restrict__ b_gate,    // [T, 8]
    const int64_t* __restrict__ cu_seqlens, // [N+1]
    const float scale,
    bf16* __restrict__ output,          // [T, 8, 128]
    float* __restrict__ new_state,      // [N, 8, 128, 128]
    int num_seqs
) {
    constexpr int ROWS_PER_BLOCK = HEAD_DIM / SPLIT_FACTOR;
    constexpr int ROWS_PER_WARP = ROWS_PER_BLOCK / NUM_WARPS;
    constexpr int HEADS_X_SPLIT = NUM_V_HEADS * SPLIT_FACTOR;

    const int idx = blockIdx.x;
    const int seq = idx / HEADS_X_SPLIT;
    const int remainder = idx % HEADS_X_SPLIT;
    const int vh = remainder / SPLIT_FACTOR;
    const int split_id = remainder % SPLIT_FACTOR;
    const int qkh = vh / V_PER_Q;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    if (seq >= num_seqs) return;

    const int64_t seq_start = cu_seqlens[seq];
    const int64_t seq_end = cu_seqlens[seq + 1];
    const int seq_len = (int)(seq_end - seq_start);
    if (seq_len <= 0) return;

    const float* state_base = state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float* new_state_base = new_state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;

    // Vi range for this warp
    const int vi_start = split_id * ROWS_PER_BLOCK + warp_id * ROWS_PER_WARP;

    // Load state into registers: float4 per vi row (4 contiguous k-elements per thread)
    float4 reg_state[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        reg_state[r] = *reinterpret_cast<const float4*>(
            state_base + (vi_start + r) * HEAD_DIM + lane * 4);
    }

    const float A_val = A_log[vh];
    const float dt_val = dt_bias[vh];

    for (int t_offset = 0; t_offset < seq_len; t_offset++) {
        const int t = (int)seq_start + t_offset;

        // Gates (all threads compute same scalars)
        float a_val = __bfloat162float(a[t * NUM_V_HEADS + vh]);
        float g = expf(-expf(A_val) * softplus(a_val + dt_val));
        float beta = 1.0f / (1.0f + expf(-__bfloat162float(b_gate[t * NUM_V_HEADS + vh])));

        // Each thread loads 4 contiguous k and q values
        const int kq_offset = t * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + lane * 4;
        float k_vals[4], q_vals[4];
        k_vals[0] = __bfloat162float(k[kq_offset + 0]);
        k_vals[1] = __bfloat162float(k[kq_offset + 1]);
        k_vals[2] = __bfloat162float(k[kq_offset + 2]);
        k_vals[3] = __bfloat162float(k[kq_offset + 3]);
        const int q_offset = t * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + lane * 4;
        q_vals[0] = __bfloat162float(q[q_offset + 0]);
        q_vals[1] = __bfloat162float(q[q_offset + 1]);
        q_vals[2] = __bfloat162float(q[q_offset + 2]);
        q_vals[3] = __bfloat162float(q[q_offset + 3]);

        // qk_dot = sum_k(q[k]*k[k]) — warp reduce + broadcast
        float qk_local = q_vals[0]*k_vals[0] + q_vals[1]*k_vals[1]
                        + q_vals[2]*k_vals[2] + q_vals[3]*k_vals[3];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            qk_local += __shfl_down_sync(0xffffffff, qk_local, offset);
        float qk_dot = __shfl_sync(0xffffffff, qk_local, 0);

        // Load v values for this warp's vi rows via shuffle broadcast
        float v_local = 0.0f;
        if (lane < ROWS_PER_WARP)
            v_local = __bfloat162float(v[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi_start + lane]);

        // Process 4 vi rows at a time for ILP (8 interleaved reductions)
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r += 4) {
            float4 st4_a = reg_state[r];
            float4 st4_b = reg_state[r + 1];
            float4 st4_c = reg_state[r + 2];
            float4 st4_d = reg_state[r + 3];

            // Broadcast v values from holding lanes
            float v_a = __shfl_sync(0xffffffff, v_local, r);
            float v_b = __shfl_sync(0xffffffff, v_local, r + 1);
            float v_c = __shfl_sync(0xffffffff, v_local, r + 2);
            float v_d = __shfl_sync(0xffffffff, v_local, r + 3);

            // Compute partial dot products (k*state and q*state from OLD state)
            float ks_a = k_vals[0]*st4_a.x + k_vals[1]*st4_a.y + k_vals[2]*st4_a.z + k_vals[3]*st4_a.w;
            float qs_a = q_vals[0]*st4_a.x + q_vals[1]*st4_a.y + q_vals[2]*st4_a.z + q_vals[3]*st4_a.w;
            float ks_b = k_vals[0]*st4_b.x + k_vals[1]*st4_b.y + k_vals[2]*st4_b.z + k_vals[3]*st4_b.w;
            float qs_b = q_vals[0]*st4_b.x + q_vals[1]*st4_b.y + q_vals[2]*st4_b.z + q_vals[3]*st4_b.w;
            float ks_c = k_vals[0]*st4_c.x + k_vals[1]*st4_c.y + k_vals[2]*st4_c.z + k_vals[3]*st4_c.w;
            float qs_c = q_vals[0]*st4_c.x + q_vals[1]*st4_c.y + q_vals[2]*st4_c.z + q_vals[3]*st4_c.w;
            float ks_d = k_vals[0]*st4_d.x + k_vals[1]*st4_d.y + k_vals[2]*st4_d.z + k_vals[3]*st4_d.w;
            float qs_d = q_vals[0]*st4_d.x + q_vals[1]*st4_d.y + q_vals[2]*st4_d.z + q_vals[3]*st4_d.w;

            // Interleaved warp reductions (better ILP)
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                ks_a += __shfl_down_sync(0xffffffff, ks_a, offset);
                ks_b += __shfl_down_sync(0xffffffff, ks_b, offset);
                ks_c += __shfl_down_sync(0xffffffff, ks_c, offset);
                ks_d += __shfl_down_sync(0xffffffff, ks_d, offset);
                qs_a += __shfl_down_sync(0xffffffff, qs_a, offset);
                qs_b += __shfl_down_sync(0xffffffff, qs_b, offset);
                qs_c += __shfl_down_sync(0xffffffff, qs_c, offset);
                qs_d += __shfl_down_sync(0xffffffff, qs_d, offset);
            }

            // Broadcast ks sums to all lanes for state update
            ks_a = __shfl_sync(0xffffffff, ks_a, 0);
            ks_b = __shfl_sync(0xffffffff, ks_b, 0);
            ks_c = __shfl_sync(0xffffffff, ks_c, 0);
            ks_d = __shfl_sync(0xffffffff, ks_d, 0);

            // Compute residuals
            float res_a = beta * (v_a - g * ks_a);
            float res_b = beta * (v_b - g * ks_b);
            float res_c = beta * (v_c - g * ks_c);
            float res_d = beta * (v_d - g * ks_d);

            // Update state registers
            reg_state[r] = make_float4(
                g*st4_a.x + k_vals[0]*res_a, g*st4_a.y + k_vals[1]*res_a,
                g*st4_a.z + k_vals[2]*res_a, g*st4_a.w + k_vals[3]*res_a);
            reg_state[r + 1] = make_float4(
                g*st4_b.x + k_vals[0]*res_b, g*st4_b.y + k_vals[1]*res_b,
                g*st4_b.z + k_vals[2]*res_b, g*st4_b.w + k_vals[3]*res_b);
            reg_state[r + 2] = make_float4(
                g*st4_c.x + k_vals[0]*res_c, g*st4_c.y + k_vals[1]*res_c,
                g*st4_c.z + k_vals[2]*res_c, g*st4_c.w + k_vals[3]*res_c);
            reg_state[r + 3] = make_float4(
                g*st4_d.x + k_vals[0]*res_d, g*st4_d.y + k_vals[1]*res_d,
                g*st4_d.z + k_vals[2]*res_d, g*st4_d.w + k_vals[3]*res_d);

            // Lane 0 writes outputs using algebraic identity
            if (lane == 0) {
                const int out_offset = t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi_start;
                output[out_offset + r]     = __float2bfloat16(scale * (g * qs_a + qk_dot * res_a));
                output[out_offset + r + 1] = __float2bfloat16(scale * (g * qs_b + qk_dot * res_b));
                output[out_offset + r + 2] = __float2bfloat16(scale * (g * qs_c + qk_dot * res_c));
                output[out_offset + r + 3] = __float2bfloat16(scale * (g * qs_d + qk_dot * res_d));
            }
        }
    }

    // Write final state back via float4
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        *reinterpret_cast<float4*>(
            new_state_base + (vi_start + r) * HEAD_DIM + lane * 4) = reg_state[r];
    }
}

// TVM FFI entry point (DPS style)
void gdn_prefill(
    tvm::ffi::TensorView q,         // [T, 4, 128] bf16
    tvm::ffi::TensorView k,         // [T, 4, 128] bf16
    tvm::ffi::TensorView v,         // [T, 8, 128] bf16
    tvm::ffi::TensorView state,     // [N, 8, 128, 128] f32
    tvm::ffi::TensorView A_log,     // [8] f32
    tvm::ffi::TensorView a,         // [T, 8] bf16
    tvm::ffi::TensorView dt_bias,   // [8] f32
    tvm::ffi::TensorView b_gate,    // [T, 8] bf16
    tvm::ffi::TensorView cu_seqlens,// [N+1] i64
    double scale,
    tvm::ffi::TensorView output,    // [T, 8, 128] bf16
    tvm::ffi::TensorView new_state  // [N, 8, 128, 128] f32
) {
    int num_seqs = cu_seqlens.size(0) - 1;

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    // Choose split factor based on batch size for SM utilization
    int split_factor;
    if (num_seqs <= 2)       split_factor = 8;
    else if (num_seqs <= 6)  split_factor = 4;
    else if (num_seqs <= 16) split_factor = 2;
    else                     split_factor = 1;

    dim3 grid(num_seqs * NUM_V_HEADS * split_factor);
    dim3 block(HEAD_DIM);

    // No shared memory needed — all reductions via warp shuffle
    const int smem_bytes = 0;

    const bf16* q_ptr = static_cast<const bf16*>(q.data_ptr());
    const bf16* k_ptr = static_cast<const bf16*>(k.data_ptr());
    const bf16* v_ptr = static_cast<const bf16*>(v.data_ptr());
    const float* state_ptr = static_cast<const float*>(state.data_ptr());
    const float* A_log_ptr = static_cast<const float*>(A_log.data_ptr());
    const bf16* a_ptr = static_cast<const bf16*>(a.data_ptr());
    const float* dt_bias_ptr = static_cast<const float*>(dt_bias.data_ptr());
    const bf16* b_gate_ptr = static_cast<const bf16*>(b_gate.data_ptr());
    const int64_t* cu_seqlens_ptr = static_cast<const int64_t*>(cu_seqlens.data_ptr());
    float scale_f = static_cast<float>(scale);
    bf16* output_ptr = static_cast<bf16*>(output.data_ptr());
    float* new_state_ptr = static_cast<float*>(new_state.data_ptr());

    auto launch = [&](auto kernel_fn) {
        kernel_fn<<<grid, block, smem_bytes, stream>>>(
            q_ptr, k_ptr, v_ptr, state_ptr, A_log_ptr, a_ptr,
            dt_bias_ptr, b_gate_ptr, cu_seqlens_ptr, scale_f,
            output_ptr, new_state_ptr, num_seqs
        );
    };

    switch (split_factor) {
        case 8:  launch(gdn_prefill_kernel<8>); break;
        case 4:  launch(gdn_prefill_kernel<4>); break;
        case 2:  launch(gdn_prefill_kernel<2>); break;
        default: launch(gdn_prefill_kernel<1>); break;
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, gdn_prefill);
