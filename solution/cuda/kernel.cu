/*
 * GDN decode + prefill CUDA kernels with TVM FFI binding.
 *
 * Delta rule update (per head, working in [K,V] space):
 *   g = exp(-exp(A_log) * softplus(a + dt_bias))
 *   beta = sigmoid(b)
 *   old_v = k @ (g * state)
 *   delta = beta * (v - old_v)
 *   new_state = g * state + k^T @ delta   (rank-1 update)
 *   output = scale * q @ new_state
 *
 * State layout: k-last [B/N, H_V=8, V=128, K=128]
 * Parallelization: one block per (batch/seq, v_head), 128 threads = V dim.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

namespace {

constexpr int kNumQHeads = 4;
constexpr int kNumKHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kHeadSize = 128;
constexpr int kThreads = 128;

__device__ inline float softplusf_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ inline float sigmoidf_stable(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(x);
    return z / (1.0f + z);
}

// ---------------------------------------------------------------------------
// Decode kernel
// ---------------------------------------------------------------------------
__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    int64_t batch_size)
{
    const int64_t batch_head = static_cast<int64_t>(blockIdx.x);
    const int64_t batch_idx = batch_head / kNumVHeads;
    const int64_t v_head_idx = batch_head % kNumVHeads;
    const int64_t row_idx = static_cast<int64_t>(threadIdx.x);

    if (batch_idx >= batch_size || row_idx >= kHeadSize) return;

    const int64_t q_head_idx = v_head_idx / (kNumVHeads / kNumQHeads);
    const int64_t k_head_idx = v_head_idx / (kNumVHeads / kNumKHeads);

    const int64_t q_base = (batch_idx * kNumQHeads + q_head_idx) * kHeadSize;
    const int64_t k_base = (batch_idx * kNumKHeads + k_head_idx) * kHeadSize;
    const int64_t v_base = (batch_idx * kNumVHeads + v_head_idx) * kHeadSize;
    const int64_t gate_base = batch_idx * kNumVHeads + v_head_idx;
    const int64_t state_row_base =
        ((batch_idx * kNumVHeads + v_head_idx) * kHeadSize + row_idx) * kHeadSize;
    const int64_t out_base =
        (batch_idx * kNumVHeads + v_head_idx) * kHeadSize + row_idx;

    __shared__ __align__(16) float s_q[kHeadSize];
    __shared__ __align__(16) float s_k[kHeadSize];
    __shared__ float s_g;
    __shared__ float s_beta;

    s_q[row_idx] = __bfloat162float(q[q_base + row_idx]);
    s_k[row_idx] = __bfloat162float(k[k_base + row_idx]);
    if (row_idx == 0) {
        const float x = __bfloat162float(a[gate_base]) + dt_bias[v_head_idx];
        s_g = expf(-expf(A_log[v_head_idx]) * softplusf_stable(x));
        s_beta = sigmoidf_stable(__bfloat162float(b[gate_base]));
    }
    __syncthreads();

    const auto* state_vec = reinterpret_cast<const float4*>(state + state_row_base);
    auto* new_state_vec = reinterpret_cast<float4*>(new_state + state_row_base);

    float old_v = 0.0f;
    #pragma unroll
    for (int vi = 0; vi < kHeadSize / 4; ++vi) {
        const int base = vi * 4;
        const float4 prev = state_vec[vi];
        old_v += s_k[base + 0] * (s_g * prev.x);
        old_v += s_k[base + 1] * (s_g * prev.y);
        old_v += s_k[base + 2] * (s_g * prev.z);
        old_v += s_k[base + 3] * (s_g * prev.w);
    }

    const float value_val = __bfloat162float(v[v_base + row_idx]);
    const float delta = s_beta * (value_val - old_v);

    float out_acc = 0.0f;
    #pragma unroll
    for (int vi = 0; vi < kHeadSize / 4; ++vi) {
        const int base = vi * 4;
        const float4 prev = state_vec[vi];
        float4 updated;
        updated.x = s_g * prev.x + s_k[base + 0] * delta;
        updated.y = s_g * prev.y + s_k[base + 1] * delta;
        updated.z = s_g * prev.z + s_k[base + 2] * delta;
        updated.w = s_g * prev.w + s_k[base + 3] * delta;
        new_state_vec[vi] = updated;
        out_acc += s_q[base + 0] * updated.x;
        out_acc += s_q[base + 1] * updated.y;
        out_acc += s_q[base + 2] * updated.z;
        out_acc += s_q[base + 3] * updated.w;
    }

    output[out_base] = __float2bfloat16(scale * out_acc);
}

// ---------------------------------------------------------------------------
// Prefill kernel
// ---------------------------------------------------------------------------
__global__ void gdn_prefill_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    const int64_t* __restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    int64_t num_seqs)
{
    const int64_t seq_head = static_cast<int64_t>(blockIdx.x);
    const int64_t seq_idx = seq_head / kNumVHeads;
    const int64_t v_head_idx = seq_head % kNumVHeads;
    const int64_t row_idx = static_cast<int64_t>(threadIdx.x);

    if (seq_idx >= num_seqs || row_idx >= kHeadSize) return;

    const int64_t seq_start = cu_seqlens[seq_idx];
    const int64_t seq_end = cu_seqlens[seq_idx + 1];
    if (seq_end <= seq_start) return;

    const int64_t state_row_base =
        ((seq_idx * kNumVHeads + v_head_idx) * kHeadSize + row_idx) * kHeadSize;

    float4 state_frag[kHeadSize / 4];
    {
        const auto* state_vec = reinterpret_cast<const float4*>(state + state_row_base);
        #pragma unroll
        for (int vi = 0; vi < kHeadSize / 4; ++vi)
            state_frag[vi] = state_vec[vi];
    }

    const int64_t q_head_idx = v_head_idx / (kNumVHeads / kNumQHeads);
    const int64_t k_head_idx = v_head_idx / (kNumVHeads / kNumKHeads);

    for (int64_t t = seq_start; t < seq_end; ++t) {
        const int64_t q_base = (t * kNumQHeads + q_head_idx) * kHeadSize;
        const int64_t k_base = (t * kNumKHeads + k_head_idx) * kHeadSize;
        const int64_t v_base = (t * kNumVHeads + v_head_idx) * kHeadSize;
        const int64_t gate_base = t * kNumVHeads + v_head_idx;
        const int64_t out_base = (t * kNumVHeads + v_head_idx) * kHeadSize + row_idx;

        const float x = __bfloat162float(a[gate_base]) + dt_bias[v_head_idx];
        const float g = expf(-expf(A_log[v_head_idx]) * softplusf_stable(x));
        const float beta = sigmoidf_stable(__bfloat162float(b[gate_base]));

        float old_v = 0.0f;
        #pragma unroll
        for (int vi = 0; vi < kHeadSize / 4; ++vi) {
            const int base = vi * 4;
            const float4 prev = state_frag[vi];
            old_v += __bfloat162float(k[k_base + base + 0]) * (g * prev.x);
            old_v += __bfloat162float(k[k_base + base + 1]) * (g * prev.y);
            old_v += __bfloat162float(k[k_base + base + 2]) * (g * prev.z);
            old_v += __bfloat162float(k[k_base + base + 3]) * (g * prev.w);
        }

        const float value_val = __bfloat162float(v[v_base + row_idx]);
        const float delta_val = beta * (value_val - old_v);

        float out_acc = 0.0f;
        #pragma unroll
        for (int vi = 0; vi < kHeadSize / 4; ++vi) {
            const int base = vi * 4;
            const float4 prev = state_frag[vi];
            float4 updated;
            updated.x = g * prev.x + __bfloat162float(k[k_base + base + 0]) * delta_val;
            updated.y = g * prev.y + __bfloat162float(k[k_base + base + 1]) * delta_val;
            updated.z = g * prev.z + __bfloat162float(k[k_base + base + 2]) * delta_val;
            updated.w = g * prev.w + __bfloat162float(k[k_base + base + 3]) * delta_val;
            state_frag[vi] = updated;
            out_acc += __bfloat162float(q[q_base + base + 0]) * updated.x;
            out_acc += __bfloat162float(q[q_base + base + 1]) * updated.y;
            out_acc += __bfloat162float(q[q_base + base + 2]) * updated.z;
            out_acc += __bfloat162float(q[q_base + base + 3]) * updated.w;
        }

        output[out_base] = __float2bfloat16(scale * out_acc);
    }

    auto* new_state_vec = reinterpret_cast<float4*>(new_state + state_row_base);
    #pragma unroll
    for (int vi = 0; vi < kHeadSize / 4; ++vi)
        new_state_vec[vi] = state_frag[vi];
}

// ---------------------------------------------------------------------------
// TVM FFI host functions (DPS: inputs + outputs as params)
// ---------------------------------------------------------------------------

// Decode: q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state
void GDNDecode(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView b,
    double scale,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView new_state)
{
    const int64_t batch_size = q.size(0);
    float scale_f = (scale == 0.0) ? (1.0f / sqrtf(128.0f)) : static_cast<float>(scale);

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    const dim3 grid(batch_size * kNumVHeads);
    const dim3 block(kThreads);

    gdn_decode_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b.data_ptr()),
        scale_f,
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()),
        batch_size);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, GDNDecode);

// Prefill: q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state
void GDNPrefill(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView cu_seqlens,
    double scale,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView new_state)
{
    const int64_t num_seqs = cu_seqlens.size(0) - 1;
    float scale_f = (scale == 0.0) ? (1.0f / sqrtf(128.0f)) : static_cast<float>(scale);

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    const dim3 grid(num_seqs * kNumVHeads);
    const dim3 block(kThreads);

    gdn_prefill_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b.data_ptr()),
        static_cast<const int64_t*>(cu_seqlens.data_ptr()),
        scale_f,
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()),
        num_seqs);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_prefill, GDNPrefill);

}  // namespace
