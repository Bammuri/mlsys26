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
 *   temp = g * state
 *   residual = beta * (v - k @ temp)
 *   state_new = temp + k^T @ residual
 *   output = scale * (q @ state_new)
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

__device__ __forceinline__ float softplus(float x) {
    return log1pf(expf(x));
}

/*
 * One block per (batch, v_head) pair.
 * Grid: (B * NUM_V_HEADS,)
 * Block: (128,) — one thread per K dimension
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
    float* __restrict__ new_state,      // [B, 8, 128, 128]
    int batch_size
) {
    const int idx = blockIdx.x;
    const int batch = idx / NUM_V_HEADS;
    const int vh = idx % NUM_V_HEADS;
    const int qkh = vh / V_PER_Q;
    const int tid = threadIdx.x;    // k index

    // Compute gates
    float a_val = __bfloat162float(a[batch * NUM_V_HEADS + vh]);
    float dt_val = dt_bias[vh];
    float A_val = A_log[vh];
    float g = expf(-expf(A_val) * softplus(a_val + dt_val));
    float beta = 1.0f / (1.0f + expf(-__bfloat162float(b_gate[batch * NUM_V_HEADS + vh])));

    // Load k and q vectors
    float k_val = __bfloat162float(k[batch * NUM_K_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);
    float q_val = __bfloat162float(q[batch * NUM_Q_HEADS * HEAD_DIM + qkh * HEAD_DIM + tid]);

    // Load v vector into shared memory
    __shared__ float s_v[HEAD_DIM];
    float v_val = __bfloat162float(v[batch * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + tid]);
    s_v[tid] = v_val;
    __syncthreads();

    // State base pointers
    const float* state_base = state + (batch * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;
    float* new_state_base = new_state + (batch * NUM_V_HEADS + vh) * HEAD_DIM * HEAD_DIM;

    __shared__ float s_reduce[4];  // 128/32 = 4 warps
    __shared__ float s_kdot[HEAD_DIM];

    // Step 1: k @ temp for each vi
    for (int vi = 0; vi < HEAD_DIM; vi++) {
        float state_val = state_base[vi * HEAD_DIM + tid];
        float temp_val = g * state_val;
        float partial = k_val * temp_val;

        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);

        if (tid % 32 == 0)
            s_reduce[tid / 32] = partial;
        __syncthreads();

        if (tid == 0) {
            s_kdot[vi] = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
        }
        __syncthreads();
    }

    // Steps 2-4: update state and compute output
    for (int vi = 0; vi < HEAD_DIM; vi++) {
        float residual = beta * (s_v[vi] - s_kdot[vi]);
        float state_val = state_base[vi * HEAD_DIM + tid];
        float new_s = g * state_val + k_val * residual;

        new_state_base[vi * HEAD_DIM + tid] = new_s;

        float partial = q_val * new_s;
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);

        if (tid % 32 == 0)
            s_reduce[tid / 32] = partial;
        __syncthreads();

        if (tid == 0) {
            float sum = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
            output[batch * NUM_V_HEADS * HEAD_DIM + vh * HEAD_DIM + vi] =
                __float2bfloat16(scale * sum);
        }
        __syncthreads();
    }
}

// TVM FFI entry point (DPS style)
void gdn_decode(
    tvm::ffi::TensorView q,         // [B, 1, 4, 128] bf16
    tvm::ffi::TensorView k,         // [B, 1, 4, 128] bf16
    tvm::ffi::TensorView v,         // [B, 1, 8, 128] bf16
    tvm::ffi::TensorView state,     // [B, 8, 128, 128] f32
    tvm::ffi::TensorView A_log,     // [8] f32
    tvm::ffi::TensorView a,         // [B, 1, 8] bf16
    tvm::ffi::TensorView dt_bias,   // [8] f32
    tvm::ffi::TensorView b_gate,    // [B, 1, 8] bf16
    double scale,
    tvm::ffi::TensorView output,    // [B, 1, 8, 128] bf16
    tvm::ffi::TensorView new_state  // [B, 8, 128, 128] f32
) {
    int batch_size = q.size(0);

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    dim3 grid(batch_size * NUM_V_HEADS);
    dim3 block(HEAD_DIM);

    gdn_decode_kernel<<<grid, block, 0, stream>>>(
        static_cast<const bf16*>(q.data_ptr()),
        static_cast<const bf16*>(k.data_ptr()),
        static_cast<const bf16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const bf16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const bf16*>(b_gate.data_ptr()),
        static_cast<float>(scale),
        static_cast<bf16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()),
        batch_size
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, gdn_decode);
