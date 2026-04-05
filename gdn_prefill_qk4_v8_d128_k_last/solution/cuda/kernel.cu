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
 * Uses sequential per-token update (baseline, not yet chunked).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

using bf16 = __nv_bfloat16;

constexpr int NUM_Q_HEADS = 4;
constexpr int NUM_K_HEADS = 4;
constexpr int NUM_V_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int V_PER_Q = NUM_V_HEADS / NUM_Q_HEADS;  // 2

__device__ __forceinline__ float softplus(float x) {
    return log1pf(expf(x));
}

/*
 * One block per (seq_idx, v_head) pair.
 * Grid: (num_seqs * NUM_V_HEADS,)
 * Block: (128,) — one thread per K dimension
 *
 * Each block processes all tokens in one sequence for one v_head.
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
    const long* __restrict__ cu_seqlens, // [N+1]
    const float scale,
    bf16* __restrict__ output,          // [T, 8, 128]
    float* __restrict__ new_state,      // [N, 8, 128, 128]
    int num_seqs
) {
    const int idx = blockIdx.x;
    const int seq = idx / NUM_V_HEADS;
    const int vh = idx % NUM_V_HEADS;
    const int qkh = vh / V_PER_Q;
    const int tid = threadIdx.x;  // k index

    if (seq >= num_seqs) return;

    const long seq_start = cu_seqlens[seq];
    const long seq_end = cu_seqlens[seq + 1];
    const int seq_len = (int)(seq_end - seq_start);
    if (seq_len <= 0) return;

    // Load initial state into registers (one column: state[vi][tid] for all vi)
    // State is [V, K], we process column tid (one K index)
    const float* state_base = state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float* new_state_base = new_state + (seq * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;

    // We can't hold all V=128 floats in registers efficiently, so use shared mem
    __shared__ float s_state[HEAD_DIM * HEAD_DIM];  // [V][K] = 64KB — fits in B200's 228KB SMEM

    // Load state to shared memory
    for (int vi = 0; vi < HEAD_DIM; vi++) {
        s_state[vi * HEAD_DIM + tid] = state_base[vi * HEAD_DIM + tid];
    }
    __syncthreads();

    __shared__ float s_reduce[4];   // warp reduction scratch (128/32 = 4 warps)
    __shared__ float s_kdot[HEAD_DIM];
    __shared__ float s_v[HEAD_DIM];

    float A_val = A_log[vh];
    float dt_val = dt_bias[vh];

    // Process each token sequentially
    for (int t_offset = 0; t_offset < seq_len; t_offset++) {
        int t = (int)seq_start + t_offset;

        // Compute gates
        float a_val = __bfloat162float(a[t * NUM_V_HEADS + vh]);
        float g = expf(-expf(A_val) * softplus(a_val + dt_val));
        float beta = 1.0f / (1.0f + expf(-__bfloat162float(b_gate[t * NUM_V_HEADS + vh])));

        // Load k, q, v for this timestep
        float k_val = __bfloat162float(k[t * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);
        float q_val = __bfloat162float(q[t * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);
        float v_val = __bfloat162float(v[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + tid]);
        s_v[tid] = v_val;
        __syncthreads();

        // Step 1: k @ temp for each vi (temp = g * state)
        for (int vi = 0; vi < HEAD_DIM; vi++) {
            float temp_val = g * s_state[vi * HEAD_DIM + tid];
            float partial = k_val * temp_val;

            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            if (tid % 32 == 0)
                s_reduce[tid / 32] = partial;
            __syncthreads();

            if (tid == 0) {
                float sum = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
                s_kdot[vi] = sum;
            }
            __syncthreads();
        }

        // Steps 2-4: update state and compute output
        for (int vi = 0; vi < HEAD_DIM; vi++) {
            float residual = beta * (s_v[vi] - s_kdot[vi]);
            float new_s = g * s_state[vi * HEAD_DIM + tid] + k_val * residual;
            s_state[vi * HEAD_DIM + tid] = new_s;

            float partial = q_val * new_s;
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            if (tid % 32 == 0)
                s_reduce[tid / 32] = partial;
            __syncthreads();

            if (tid == 0) {
                float sum = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
                output[t * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi] =
                    __float2bfloat16(scale * sum);
            }
            __syncthreads();
        }
    }

    // Write final state back
    for (int vi = 0; vi < HEAD_DIM; vi++) {
        new_state_base[vi * HEAD_DIM + tid] = s_state[vi * HEAD_DIM + tid];
    }
}

extern "C" void launch_gdn_prefill(
    const bf16* q, const bf16* k, const bf16* v,
    const float* state, const float* A_log, const bf16* a,
    const float* dt_bias, const bf16* b_gate,
    const long* cu_seqlens, float scale,
    bf16* output, float* new_state, int num_seqs
) {
    dim3 grid(num_seqs * NUM_V_HEADS);
    dim3 block(HEAD_DIM);  // 128 threads

    // Request 64KB shared memory for state
    cudaFuncSetAttribute(gdn_prefill_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 0);

    gdn_prefill_kernel<<<grid, block>>>(
        q, k, v, state, A_log, a, dt_bias, b_gate,
        cu_seqlens, scale, output, new_state, num_seqs
    );
}
