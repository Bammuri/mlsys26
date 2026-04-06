/*
 * GDN Prefill Kernel: gdn_prefill_qk4_v8_d128_k_last
 *
 * Variable-length prefill with recurrent state update.
 * State layout: [N, H=8, V=128, K=128] float32 (k-last)
 * GVA: q_heads=4, k_heads=4, v_heads=8 (2 v_heads per q/k head)
 *
 * Simplified formula (per timestep, per head):
 *   g = exp(-exp(A_log) * softplus(a + dt_bias))
 *   beta = sigmoid(b)
 *   temp = g * state
 *   residual = beta * (v - k @ temp)
 *   state_new = temp + k^T @ residual
 *   output = scale * (q @ state_new)
 *
 * Optimizations:
 *   - Register-tiled state: 128x128 state in registers (each thread holds 128 floats)
 *   - Two-phase inner loop: Phase 1 does all state updates, Phase 2 computes all outputs
 *   - Batched VB=8 vi values per syncthreads (1 sync per batch per phase)
 *   - Deferred output reduction: 34 syncs/timestep (1 v-load + 16 phase1 + 1 barrier + 16 phase2)
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
constexpr int VB = 8;  // vi batch size
constexpr int NUM_WARPS = HEAD_DIM / 32;  // 4

__device__ __forceinline__ float softplus(float x) {
    return log1pf(expf(x));
}

/*
 * One block per (seq_idx, v_head) pair.
 * Grid: (num_seqs * NUM_V_HEADS,)
 * Block: (128,) — one thread per K dimension
 *
 * State is register-tiled: each thread holds reg_state[128] where
 * reg_state[vi] = state[vi][tid]. No shared memory needed for state.
 */
__global__ void gdn_prefill_kernel(
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
    const int idx = blockIdx.x;
    const int seq = idx / NUM_V_HEADS;
    const int vh = idx % NUM_V_HEADS;
    const int qkh = vh / V_PER_Q;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (seq >= num_seqs) return;

    const int64_t seq_start = cu_seqlens[seq];
    const int64_t seq_end = cu_seqlens[seq + 1];
    const int seq_len = (int)(seq_end - seq_start);
    if (seq_len <= 0) return;

    const float* state_base = state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float* new_state_base = new_state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;

    // Register-tiled state: each thread holds one column (k-index = tid)
    float reg_state[HEAD_DIM];
    for (int vi = 0; vi < HEAD_DIM; vi++) {
        reg_state[vi] = state_base[vi * HEAD_DIM + tid];
    }

    // Shared memory layout (tiny):
    // s_v[128] — v values for current timestep (512 bytes)
    // s_reduce[NUM_WARPS * VB] = [32] — cross-warp reduction buffer (128 bytes)
    extern __shared__ float smem[];
    float* s_v = smem;
    float* s_reduce = smem + HEAD_DIM;

    float A_val = A_log[vh];
    float dt_val = dt_bias[vh];

    for (int t_offset = 0; t_offset < seq_len; t_offset++) {
        int t = (int)seq_start + t_offset;

        float a_val = __bfloat162float(a[t * NUM_V_HEADS + vh]);
        float g = expf(-expf(A_val) * softplus(a_val + dt_val));
        float beta = 1.0f / (1.0f + expf(-__bfloat162float(b_gate[t * NUM_V_HEADS + vh])));

        float k_val = __bfloat162float(k[t * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);
        float q_val = __bfloat162float(q[t * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);
        float v_val = __bfloat162float(v[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + tid]);
        s_v[tid] = v_val;
        __syncthreads();

        // Phase 1: State update only — loop over vi in batches of VB=8
        for (int vi_base = 0; vi_base < HEAD_DIM; vi_base += VB) {
            float temp[VB];
            float partial_k[VB];

            // Compute temp and partial kdot products
            #pragma unroll
            for (int vb = 0; vb < VB; vb++) {
                temp[vb] = g * reg_state[vi_base + vb];
                partial_k[vb] = k_val * temp[vb];
            }

            // Warp-level reduction for all VB partial_k values
            #pragma unroll
            for (int vb = 0; vb < VB; vb++) {
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    partial_k[vb] += __shfl_down_sync(0xffffffff, partial_k[vb], offset);
                if (lane_id == 0)
                    s_reduce[warp_id * VB + vb] = partial_k[vb];
            }
            __syncthreads();  // One sync for VB kdot reductions

            // All threads read VB kdot sums and update state
            #pragma unroll
            for (int vb = 0; vb < VB; vb++) {
                float kdot = s_reduce[0 * VB + vb] + s_reduce[1 * VB + vb]
                           + s_reduce[2 * VB + vb] + s_reduce[3 * VB + vb];
                float residual = beta * (s_v[vi_base + vb] - kdot);
                reg_state[vi_base + vb] = temp[vb] + k_val * residual;
            }
        }

        __syncthreads();  // Barrier between phases: ensure all Phase 1 s_reduce reads complete

        // Phase 2: Output only — loop over vi in batches of VB=8
        for (int vi_base = 0; vi_base < HEAD_DIM; vi_base += VB) {
            float partial_q[VB];

            // Compute partial qdot products
            #pragma unroll
            for (int vb = 0; vb < VB; vb++) {
                partial_q[vb] = q_val * reg_state[vi_base + vb];
            }

            // Warp-level reduction for all VB partial_q values
            #pragma unroll
            for (int vb = 0; vb < VB; vb++) {
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    partial_q[vb] += __shfl_down_sync(0xffffffff, partial_q[vb], offset);
                if (lane_id == 0)
                    s_reduce[warp_id * VB + vb] = partial_q[vb];
            }
            __syncthreads();  // One sync for VB output reductions

            // Thread 0 writes VB output values
            if (tid == 0) {
                #pragma unroll
                for (int vb = 0; vb < VB; vb++) {
                    float sum = s_reduce[0 * VB + vb] + s_reduce[1 * VB + vb]
                              + s_reduce[2 * VB + vb] + s_reduce[3 * VB + vb];
                    output[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi_base + vb] =
                        __float2bfloat16(scale * sum);
                }
            }
        }
        __syncthreads();  // Ensure s_v/s_reduce safe before next timestep
    }

    // Write final state back
    for (int vi = 0; vi < HEAD_DIM; vi++) {
        new_state_base[vi * HEAD_DIM + tid] = reg_state[vi];
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

    dim3 grid(num_seqs * NUM_V_HEADS);
    dim3 block(HEAD_DIM);

    // Dynamic shared memory: s_v[128] + s_reduce[NUM_WARPS * VB = 32]
    const int smem_bytes = (HEAD_DIM + NUM_WARPS * VB) * sizeof(float);
    cudaFuncSetAttribute(gdn_prefill_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    gdn_prefill_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const bf16*>(q.data_ptr()),
        static_cast<const bf16*>(k.data_ptr()),
        static_cast<const bf16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const bf16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const bf16*>(b_gate.data_ptr()),
        static_cast<const int64_t*>(cu_seqlens.data_ptr()),
        static_cast<float>(scale),
        static_cast<bf16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()),
        num_seqs
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, gdn_prefill);
