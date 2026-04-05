/*
 * GDN Decode Kernel: gdn_decode_qk4_v8_d128_k_last
 *
 * Single-token decode with recurrent state update.
 * State layout: [B, H=8, V=128, K=128] float32 (k-last)
 * GVA: q_heads=4, k_heads=4, v_heads=8 (2 v_heads per q/k head)
 *
 * Simplified formula (per head):
 *   g = exp(-exp(A_log) * softplus(a + dt_bias))
 *   beta = sigmoid(b)
 *   temp = g * state                     (elementwise, state in [K,V] internally)
 *   residual = beta * (v - k @ temp)     (GEMV + elementwise)
 *   state_new = temp + k^T @ residual    (rank-1 update)
 *   output = scale * (q @ state_new)     (GEMV)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

using bf16 = __nv_bfloat16;

// Constants
constexpr int NUM_Q_HEADS = 4;
constexpr int NUM_K_HEADS = 4;
constexpr int NUM_V_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int V_PER_Q = NUM_V_HEADS / NUM_Q_HEADS;  // 2

__device__ __forceinline__ float softplus(float x) {
    return log1pf(expf(x));
}

/*
 * One block per (batch, v_head) pair.
 * Grid: (B * NUM_V_HEADS,)
 * Block: (128,) — one thread per K dimension
 *
 * Each thread owns one row of state[k, :] (V=128 values).
 * But 128 threads * 128 floats = 64KB registers — too much.
 * Instead: iterate over V in tiles, using shared memory for state.
 */
__global__ void gdn_decode_kernel(
    const bf16* __restrict__ q,         // [B, 1, 4, 128]
    const bf16* __restrict__ k,         // [B, 1, 4, 128]
    const bf16* __restrict__ v,         // [B, 1, 8, 128]
    const float* __restrict__ state,    // [B, 8, 128, 128] k-last [H,V,K]
    const float* __restrict__ A_log,    // [8]
    const bf16* __restrict__ a,         // [B, 1, 8]
    const float* __restrict__ dt_bias,  // [8]
    const bf16* __restrict__ b_gate,    // [B, 1, 8]
    const float scale,
    bf16* __restrict__ output,          // [B, 1, 8, 128]
    float* __restrict__ new_state       // [B, 8, 128, 128]
) {
    const int idx = blockIdx.x;
    const int batch = idx / NUM_V_HEADS;
    const int vh = idx % NUM_V_HEADS;
    const int qkh = vh / V_PER_Q;  // which q/k head this v_head maps to
    const int tid = threadIdx.x;    // thread id = k index

    // Compute gates
    float a_val = __bfloat162float(a[batch * NUM_V_HEADS + vh]);
    float dt_val = dt_bias[vh];
    float A_val = A_log[vh];
    float g = expf(-expf(A_val) * softplus(a_val + dt_val));
    float beta = 1.0f / (1.0f + expf(-__bfloat162float(b_gate[batch * NUM_V_HEADS + vh])));

    // Load k and q vectors for this head (128 elements, one per thread)
    float k_val = __bfloat162float(k[batch * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);
    float q_val = __bfloat162float(q[batch * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);

    // Load v vector into shared memory
    __shared__ float s_v[HEAD_DIM];
    float v_val = __bfloat162float(v[batch * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + tid]);
    s_v[tid] = v_val;
    __syncthreads();

    // State base pointer for this (batch, v_head): [V, K] layout
    // state[batch][vh][vi][ki] = state[(batch * 8 + vh) * 128 * 128 + vi * 128 + ki]
    const float* state_base = state + (batch * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float* new_state_base = new_state + (batch * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;

    // Step 1: Compute k @ temp (GEMV over K dimension)
    // temp[vi][ki] = g * state[vi][ki]
    // k @ temp = sum_ki(k[ki] * temp[vi][ki]) for each vi
    // Each thread handles one ki, so we need reduction across threads for each vi

    // We iterate over V dimension. For each vi:
    //   partial = k_val * g * state[vi][tid]
    //   reduce across threads -> dot product for vi

    // Use shared memory for reduction
    __shared__ float s_reduce[HEAD_DIM];  // for warp reduction results
    __shared__ float s_kdot[HEAD_DIM];    // k @ temp results (one per vi)

    // Compute k @ temp for each vi (128 iterations, each needs 128-wide reduction)
    for (int vi = 0; vi < HEAD_DIM; vi++) {
        float state_val = state_base[vi * HEAD_DIM + tid];  // state[vi][tid=ki]
        float temp_val = g * state_val;
        float partial = k_val * temp_val;

        // Warp reduction
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);

        // Write warp results to shared memory
        if (tid % 32 == 0)
            s_reduce[tid / 32] = partial;
        __syncthreads();

        // Thread 0 sums across warps (4 warps)
        if (tid == 0) {
            float sum = 0.0f;
            for (int w = 0; w < HEAD_DIM / 32; w++)
                sum += s_reduce[w];
            s_kdot[vi] = sum;  // k @ temp[vi]
        }
        __syncthreads();
    }

    // Step 2: residual[vi] = beta * (v[vi] - kdot[vi])
    // Step 3: state_new[vi][ki] = g * state[vi][ki] + k[ki] * residual[vi]
    // Step 4: output[vi] = scale * sum_ki(q[ki] * state_new[vi][ki])

    // Process each vi: compute new state and output simultaneously
    __shared__ float s_output_partial[HEAD_DIM];  // partial output per vi (need reduction later)

    float output_accum = 0.0f;  // each thread accumulates q_val * state_new contribution

    for (int vi = 0; vi < HEAD_DIM; vi++) {
        float residual = beta * (s_v[vi] - s_kdot[vi]);
        float state_val = state_base[vi * HEAD_DIM + tid];
        float new_s = g * state_val + k_val * residual;

        // Write new state
        new_state_base[vi * HEAD_DIM + tid] = new_s;

        // Accumulate output: q @ state_new
        // output[vi] = scale * sum_ki(q[ki] * state_new[vi][ki])
        // Each thread has q[tid] * state_new[vi][tid]
        float partial = q_val * new_s;

        // Warp reduction
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);

        if (tid % 32 == 0)
            s_reduce[tid / 32] = partial;
        __syncthreads();

        if (tid == 0) {
            float sum = 0.0f;
            for (int w = 0; w < HEAD_DIM / 32; w++)
                sum += s_reduce[w];
            // Write output for this vi
            output[batch * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi] =
                __float2bfloat16(scale * sum);
        }
        __syncthreads();
    }
}

extern "C" void launch_gdn_decode(
    const bf16* q, const bf16* k, const bf16* v,
    const float* state, const float* A_log, const bf16* a,
    const float* dt_bias, const bf16* b_gate, float scale,
    bf16* output, float* new_state, int batch_size
) {
    dim3 grid(batch_size * NUM_V_HEADS);
    dim3 block(HEAD_DIM);  // 128 threads
    gdn_decode_kernel<<<grid, block>>>(
        q, k, v, state, A_log, a, dt_bias, b_gate, scale,
        output, new_state
    );
}
