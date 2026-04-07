/*
 * Naive CUDA implementation for FlashInfer Gated Delta Net workloads.
 *
 * This file exposes two torch extension entry points:
 * - gdn_decode
 * - gdn_prefill
 *
 * Both mirror the reference definition exactly and prioritize correctness.
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <tuple>

namespace {

constexpr int kNumQHeads = 4;
constexpr int kNumKHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kHeadSize = 128;
constexpr int kDecodeThreads = 64;
constexpr int kPrefillThreads = 128;

inline void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

inline float resolve_scale(double scale) {
    if (scale == 0.0) {
        return 1.0f / std::sqrt(static_cast<float>(kHeadSize));
    }
    return static_cast<float>(scale);
}

__device__ inline float bf16_to_float(const at::BFloat16* ptr, int64_t index) {
    const auto* raw = reinterpret_cast<const __nv_bfloat16*>(ptr);
    return __bfloat162float(raw[index]);
}

__device__ inline void float_to_bf16(at::BFloat16* ptr, int64_t index, float value) {
    auto* raw = reinterpret_cast<__nv_bfloat16*>(ptr);
    raw[index] = __float2bfloat16(value);
}

__device__ inline float softplusf_stable(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
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

__global__ void gdn_decode_kernel(
    const at::BFloat16* __restrict__ q,
    const at::BFloat16* __restrict__ k,
    const at::BFloat16* __restrict__ v,
    const float* __restrict__ state,
    bool has_state,
    const float* __restrict__ A_log,
    const at::BFloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const at::BFloat16* __restrict__ b,
    float scale,
    at::BFloat16* __restrict__ output,
    float* __restrict__ new_state,
    int64_t batch_size) {
    const int64_t batch_head = static_cast<int64_t>(blockIdx.x);
    const int64_t batch_idx = batch_head / kNumVHeads;
    const int64_t v_head_idx = batch_head % kNumVHeads;
    const int64_t thread_idx = static_cast<int64_t>(threadIdx.x);

    if (batch_idx >= batch_size || thread_idx >= kDecodeThreads) {
        return;
    }

    const int64_t q_head_idx = v_head_idx / (kNumVHeads / kNumQHeads);
    const int64_t k_head_idx = v_head_idx / (kNumVHeads / kNumKHeads);

    const int64_t q_base = (((batch_idx * 1 + 0) * kNumQHeads + q_head_idx) * kHeadSize);
    const int64_t k_base = (((batch_idx * 1 + 0) * kNumKHeads + k_head_idx) * kHeadSize);
    const int64_t v_base = (((batch_idx * 1 + 0) * kNumVHeads + v_head_idx) * kHeadSize);
    const int64_t gate_base = ((batch_idx * 1 + 0) * kNumVHeads + v_head_idx);
    const int64_t row_idx0 = thread_idx;
    const int64_t row_idx1 = thread_idx + kDecodeThreads;
    const int64_t state_row_base0 =
        (((batch_idx * kNumVHeads + v_head_idx) * kHeadSize + row_idx0) * kHeadSize);
    const int64_t state_row_base1 =
        (((batch_idx * kNumVHeads + v_head_idx) * kHeadSize + row_idx1) * kHeadSize);
    const int64_t out_base0 =
        (((batch_idx * 1 + 0) * kNumVHeads + v_head_idx) * kHeadSize + row_idx0);
    const int64_t out_base1 =
        (((batch_idx * 1 + 0) * kNumVHeads + v_head_idx) * kHeadSize + row_idx1);

    __shared__ __align__(16) float s_q[kHeadSize];
    __shared__ __align__(16) float s_k[kHeadSize];
    __shared__ float s_g;
    __shared__ float s_beta;

    s_q[row_idx0] = bf16_to_float(q, q_base + row_idx0);
    s_q[row_idx1] = bf16_to_float(q, q_base + row_idx1);
    s_k[row_idx0] = bf16_to_float(k, k_base + row_idx0);
    s_k[row_idx1] = bf16_to_float(k, k_base + row_idx1);
    if (thread_idx == 0) {
        const float x = bf16_to_float(a, gate_base) + dt_bias[v_head_idx];
        s_g = expf(-expf(A_log[v_head_idx]) * softplusf_stable(x));
        s_beta = sigmoidf_stable(bf16_to_float(b, gate_base));
    }
    __syncthreads();

    const auto* state_vec0 = reinterpret_cast<const float4*>(state + state_row_base0);
    const auto* state_vec1 = reinterpret_cast<const float4*>(state + state_row_base1);
    auto* new_state_vec0 = reinterpret_cast<float4*>(new_state + state_row_base0);
    auto* new_state_vec1 = reinterpret_cast<float4*>(new_state + state_row_base1);

    float old_v0 = 0.0f;
    float old_v1 = 0.0f;
    #pragma unroll
    for (int vec_idx = 0; vec_idx < kHeadSize / 4; ++vec_idx) {
        const int base = vec_idx * 4;
        const float4 prev0 = has_state ? state_vec0[vec_idx] : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 prev1 = has_state ? state_vec1[vec_idx] : make_float4(0.f, 0.f, 0.f, 0.f);
        old_v0 += s_k[base + 0] * (s_g * prev0.x);
        old_v0 += s_k[base + 1] * (s_g * prev0.y);
        old_v0 += s_k[base + 2] * (s_g * prev0.z);
        old_v0 += s_k[base + 3] * (s_g * prev0.w);
        old_v1 += s_k[base + 0] * (s_g * prev1.x);
        old_v1 += s_k[base + 1] * (s_g * prev1.y);
        old_v1 += s_k[base + 2] * (s_g * prev1.z);
        old_v1 += s_k[base + 3] * (s_g * prev1.w);
    }

    const float value_val0 = bf16_to_float(v, v_base + row_idx0);
    const float value_val1 = bf16_to_float(v, v_base + row_idx1);
    const float delta0 = s_beta * (value_val0 - old_v0);
    const float delta1 = s_beta * (value_val1 - old_v1);

    float out_acc0 = 0.0f;
    float out_acc1 = 0.0f;
    #pragma unroll
    for (int vec_idx = 0; vec_idx < kHeadSize / 4; ++vec_idx) {
        const int base = vec_idx * 4;
        const float4 prev0 = has_state ? state_vec0[vec_idx] : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 prev1 = has_state ? state_vec1[vec_idx] : make_float4(0.f, 0.f, 0.f, 0.f);
        float4 updated0;
        float4 updated1;
        updated0.x = s_g * prev0.x + s_k[base + 0] * delta0;
        updated0.y = s_g * prev0.y + s_k[base + 1] * delta0;
        updated0.z = s_g * prev0.z + s_k[base + 2] * delta0;
        updated0.w = s_g * prev0.w + s_k[base + 3] * delta0;
        updated1.x = s_g * prev1.x + s_k[base + 0] * delta1;
        updated1.y = s_g * prev1.y + s_k[base + 1] * delta1;
        updated1.z = s_g * prev1.z + s_k[base + 2] * delta1;
        updated1.w = s_g * prev1.w + s_k[base + 3] * delta1;
        new_state_vec0[vec_idx] = updated0;
        new_state_vec1[vec_idx] = updated1;
        out_acc0 += s_q[base + 0] * updated0.x;
        out_acc0 += s_q[base + 1] * updated0.y;
        out_acc0 += s_q[base + 2] * updated0.z;
        out_acc0 += s_q[base + 3] * updated0.w;
        out_acc1 += s_q[base + 0] * updated1.x;
        out_acc1 += s_q[base + 1] * updated1.y;
        out_acc1 += s_q[base + 2] * updated1.z;
        out_acc1 += s_q[base + 3] * updated1.w;
    }

    float_to_bf16(output, out_base0, scale * out_acc0);
    float_to_bf16(output, out_base1, scale * out_acc1);
}

inline void configure_decode_kernel_launch() {
    static std::once_flag once;
    std::call_once(once, []() {
        const auto err = cudaFuncSetCacheConfig(gdn_decode_kernel, cudaFuncCachePreferL1);
        TORCH_CHECK(
            err == cudaSuccess,
            "gdn_decode cache config setup failed: ",
            cudaGetErrorString(err));
    });
}

__global__ void gdn_prefill_kernel(
    const at::BFloat16* q,
    const at::BFloat16* k,
    const at::BFloat16* v,
    const float* state,
    bool has_state,
    const float* A_log,
    const at::BFloat16* a,
    const float* dt_bias,
    const at::BFloat16* b,
    const int64_t* cu_seqlens,
    float scale,
    at::BFloat16* output,
    float* new_state,
    int64_t num_seqs) {
    const int64_t seq_head = static_cast<int64_t>(blockIdx.x);
    const int64_t seq_idx = seq_head / kNumVHeads;
    const int64_t v_head_idx = seq_head % kNumVHeads;
    const int64_t row_idx = static_cast<int64_t>(threadIdx.x);

    if (seq_idx >= num_seqs || row_idx >= kHeadSize) {
        return;
    }

    const int64_t seq_start = cu_seqlens[seq_idx];
    const int64_t seq_end = cu_seqlens[seq_idx + 1];
    const int64_t seq_len = seq_end - seq_start;

    const int64_t state_row_base =
        (((seq_idx * kNumVHeads + v_head_idx) * kHeadSize + row_idx) * kHeadSize);

    if (seq_len <= 0) {
        return;
    }

    if (has_state) {
        for (int64_t col_idx = 0; col_idx < kHeadSize; ++col_idx) {
            new_state[state_row_base + col_idx] = state[state_row_base + col_idx];
        }
    }

    const int64_t q_head_idx = v_head_idx / (kNumVHeads / kNumQHeads);
    const int64_t k_head_idx = v_head_idx / (kNumVHeads / kNumKHeads);

    for (int64_t token_idx = seq_start; token_idx < seq_end; ++token_idx) {
        const int64_t q_base = ((token_idx * kNumQHeads + q_head_idx) * kHeadSize);
        const int64_t k_base = ((token_idx * kNumKHeads + k_head_idx) * kHeadSize);
        const int64_t v_base = ((token_idx * kNumVHeads + v_head_idx) * kHeadSize);
        const int64_t gate_base = (token_idx * kNumVHeads + v_head_idx);
        const int64_t out_base = ((token_idx * kNumVHeads + v_head_idx) * kHeadSize + row_idx);

        const float x = bf16_to_float(a, gate_base) + dt_bias[v_head_idx];
        const float g = expf(-expf(A_log[v_head_idx]) * softplusf_stable(x));
        const float beta = sigmoidf_stable(bf16_to_float(b, gate_base));

        float old_v = 0.0f;
        for (int64_t col_idx = 0; col_idx < kHeadSize; ++col_idx) {
            old_v += bf16_to_float(k, k_base + col_idx) * (g * new_state[state_row_base + col_idx]);
        }

        const float value_val = bf16_to_float(v, v_base + row_idx);
        const float delta = beta * (value_val - old_v);

        float out_acc = 0.0f;
        for (int64_t col_idx = 0; col_idx < kHeadSize; ++col_idx) {
            const float updated =
                g * new_state[state_row_base + col_idx] + bf16_to_float(k, k_base + col_idx) * delta;
            new_state[state_row_base + col_idx] = updated;
            out_acc += bf16_to_float(q, q_base + col_idx) * updated;
        }

        float_to_bf16(output, out_base, scale * out_acc);
    }
}

std::tuple<torch::Tensor, torch::Tensor> gdn_decode(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    double scale) {
    check_cuda_tensor(q, "q");
    check_cuda_tensor(k, "k");
    check_cuda_tensor(v, "v");
    check_cuda_tensor(A_log, "A_log");
    check_cuda_tensor(a, "a");
    check_cuda_tensor(dt_bias, "dt_bias");
    check_cuda_tensor(b, "b");
    if (state.has_value()) {
        check_cuda_tensor(*state, "state");
    }

    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "b must be bfloat16");
    TORCH_CHECK(A_log.scalar_type() == torch::kFloat32, "A_log must be float32");
    TORCH_CHECK(dt_bias.scalar_type() == torch::kFloat32, "dt_bias must be float32");
    if (state.has_value()) {
        TORCH_CHECK(state->scalar_type() == torch::kFloat32, "state must be float32");
    }

    TORCH_CHECK(q.dim() == 4, "q must be rank 4");
    TORCH_CHECK(k.dim() == 4, "k must be rank 4");
    TORCH_CHECK(v.dim() == 4, "v must be rank 4");
    TORCH_CHECK(a.dim() == 3, "a must be rank 3");
    TORCH_CHECK(b.dim() == 3, "b must be rank 3");
    TORCH_CHECK(A_log.dim() == 1, "A_log must be rank 1");
    TORCH_CHECK(dt_bias.dim() == 1, "dt_bias must be rank 1");

    const auto batch_size = q.size(0);
    TORCH_CHECK(q.size(1) == 1, "decode q seq_len must be 1");
    TORCH_CHECK(k.size(1) == 1, "decode k seq_len must be 1");
    TORCH_CHECK(v.size(1) == 1, "decode v seq_len must be 1");
    TORCH_CHECK(q.size(2) == kNumQHeads && q.size(3) == kHeadSize, "unexpected q shape");
    TORCH_CHECK(k.size(2) == kNumKHeads && k.size(3) == kHeadSize, "unexpected k shape");
    TORCH_CHECK(v.size(2) == kNumVHeads && v.size(3) == kHeadSize, "unexpected v shape");
    TORCH_CHECK(a.size(0) == batch_size && a.size(1) == 1 && a.size(2) == kNumVHeads, "unexpected a shape");
    TORCH_CHECK(b.size(0) == batch_size && b.size(1) == 1 && b.size(2) == kNumVHeads, "unexpected b shape");
    TORCH_CHECK(A_log.size(0) == kNumVHeads, "unexpected A_log shape");
    TORCH_CHECK(dt_bias.size(0) == kNumVHeads, "unexpected dt_bias shape");
    if (state.has_value()) {
        TORCH_CHECK(
            state->sizes() == at::IntArrayRef({batch_size, kNumVHeads, kHeadSize, kHeadSize}),
            "unexpected state shape");
    }

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto A_log_c = A_log.contiguous();
    auto a_c = a.contiguous();
    auto dt_bias_c = dt_bias.contiguous();
    auto b_c = b.contiguous();
    auto state_c = state.has_value() ? state->contiguous() : torch::Tensor();

    auto output = torch::empty({batch_size, 1, kNumVHeads, kHeadSize}, q_c.options());
    auto new_state = torch::empty(
        {batch_size, kNumVHeads, kHeadSize, kHeadSize},
        torch::TensorOptions().device(q_c.device()).dtype(torch::kFloat32));

    configure_decode_kernel_launch();

    const dim3 grid(batch_size * kNumVHeads);
    const dim3 block(kDecodeThreads);

    gdn_decode_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_c.data_ptr<at::BFloat16>(),
        k_c.data_ptr<at::BFloat16>(),
        v_c.data_ptr<at::BFloat16>(),
        state.has_value() ? state_c.data_ptr<float>() : nullptr,
        state.has_value(),
        A_log_c.data_ptr<float>(),
        a_c.data_ptr<at::BFloat16>(),
        dt_bias_c.data_ptr<float>(),
        b_c.data_ptr<at::BFloat16>(),
        resolve_scale(scale),
        output.data_ptr<at::BFloat16>(),
        new_state.data_ptr<float>(),
        batch_size);

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gdn_decode kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(output, new_state);
}

std::tuple<torch::Tensor, torch::Tensor> gdn_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    double scale) {
    check_cuda_tensor(q, "q");
    check_cuda_tensor(k, "k");
    check_cuda_tensor(v, "v");
    check_cuda_tensor(A_log, "A_log");
    check_cuda_tensor(a, "a");
    check_cuda_tensor(dt_bias, "dt_bias");
    check_cuda_tensor(b, "b");
    check_cuda_tensor(cu_seqlens, "cu_seqlens");
    if (state.has_value()) {
        check_cuda_tensor(*state, "state");
    }

    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "b must be bfloat16");
    TORCH_CHECK(A_log.scalar_type() == torch::kFloat32, "A_log must be float32");
    TORCH_CHECK(dt_bias.scalar_type() == torch::kFloat32, "dt_bias must be float32");
    TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt64, "cu_seqlens must be int64");
    if (state.has_value()) {
        TORCH_CHECK(state->scalar_type() == torch::kFloat32, "state must be float32");
    }

    TORCH_CHECK(q.dim() == 3, "q must be rank 3");
    TORCH_CHECK(k.dim() == 3, "k must be rank 3");
    TORCH_CHECK(v.dim() == 3, "v must be rank 3");
    TORCH_CHECK(a.dim() == 2, "a must be rank 2");
    TORCH_CHECK(b.dim() == 2, "b must be rank 2");
    TORCH_CHECK(A_log.dim() == 1, "A_log must be rank 1");
    TORCH_CHECK(dt_bias.dim() == 1, "dt_bias must be rank 1");
    TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be rank 1");

    const auto total_seq_len = q.size(0);
    const auto num_seqs = cu_seqlens.size(0) - 1;
    TORCH_CHECK(q.size(1) == kNumQHeads && q.size(2) == kHeadSize, "unexpected q shape");
    TORCH_CHECK(k.size(1) == kNumKHeads && k.size(2) == kHeadSize, "unexpected k shape");
    TORCH_CHECK(v.size(1) == kNumVHeads && v.size(2) == kHeadSize, "unexpected v shape");
    TORCH_CHECK(a.size(0) == total_seq_len && a.size(1) == kNumVHeads, "unexpected a shape");
    TORCH_CHECK(b.size(0) == total_seq_len && b.size(1) == kNumVHeads, "unexpected b shape");
    TORCH_CHECK(A_log.size(0) == kNumVHeads, "unexpected A_log shape");
    TORCH_CHECK(dt_bias.size(0) == kNumVHeads, "unexpected dt_bias shape");
    TORCH_CHECK(num_seqs >= 0, "cu_seqlens length must be at least 1");
    if (state.has_value()) {
        TORCH_CHECK(
            state->sizes() == at::IntArrayRef({num_seqs, kNumVHeads, kHeadSize, kHeadSize}),
            "unexpected state shape");
    }

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto A_log_c = A_log.contiguous();
    auto a_c = a.contiguous();
    auto dt_bias_c = dt_bias.contiguous();
    auto b_c = b.contiguous();
    auto cu_seqlens_c = cu_seqlens.contiguous();
    auto state_c = state.has_value() ? state->contiguous() : torch::Tensor();

    auto output = torch::empty({total_seq_len, kNumVHeads, kHeadSize}, q_c.options());
    auto new_state = torch::zeros(
        {num_seqs, kNumVHeads, kHeadSize, kHeadSize},
        torch::TensorOptions().device(q_c.device()).dtype(torch::kFloat32));

    const dim3 grid(num_seqs * kNumVHeads);
    const dim3 block(kPrefillThreads);

    gdn_prefill_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_c.data_ptr<at::BFloat16>(),
        k_c.data_ptr<at::BFloat16>(),
        v_c.data_ptr<at::BFloat16>(),
        state.has_value() ? state_c.data_ptr<float>() : nullptr,
        state.has_value(),
        A_log_c.data_ptr<float>(),
        a_c.data_ptr<at::BFloat16>(),
        dt_bias_c.data_ptr<float>(),
        b_c.data_ptr<at::BFloat16>(),
        cu_seqlens_c.data_ptr<int64_t>(),
        resolve_scale(scale),
        output.data_ptr<at::BFloat16>(),
        new_state.data_ptr<float>(),
        num_seqs);

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gdn_prefill kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(output, new_state);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_decode", &gdn_decode, "Naive GDN decode");
    m.def("gdn_prefill", &gdn_prefill, "Naive GDN prefill");
}
