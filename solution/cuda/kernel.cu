#include <cmath>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace py = pybind11;
namespace {

constexpr int64_t kNumQHeads = 4;
constexpr int64_t kNumKHeads = 4;
constexpr int64_t kNumVHeads = 8;
constexpr int64_t kHeadSize = 128;

double resolve_scale(double scale) {
    return scale == 0.0 ? 1.0 / std::sqrt(static_cast<double>(kHeadSize)) : scale;
}

void check_common_inputs(
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias) {
    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == kNumVHeads, "A_log must have shape [8]");
    TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == kNumVHeads, "dt_bias must have shape [8]");
}

torch::Tensor expand_heads(const torch::Tensor& tensor, int64_t target_heads) {
    const auto source_heads = tensor.size(1);
    TORCH_CHECK(target_heads % source_heads == 0, "target heads must be divisible by source heads");
    const auto factor = target_heads / source_heads;
    auto indices = torch::floor_divide(
        torch::arange(target_heads, tensor.options().dtype(torch::kLong)),
        factor
    );
    return tensor.index_select(1, indices);
}

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* ptr) {
    return __bfloat162float(*ptr);
}

__device__ __forceinline__ float4 load_global_v4(const float* ptr) {
    float4 value;
    asm volatile(
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
        : "l"(ptr)
    );
    return value;
}

__device__ __forceinline__ void store_global_v4(float* ptr, const float4& value) {
    asm volatile(
        "st.global.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
    );
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    int batch_size,
    float scale) {
    const int batch_idx = blockIdx.x;
    const int v_head_idx = blockIdx.y;
    const int lane = threadIdx.x;

    __shared__ float sh_q[kHeadSize];
    __shared__ float sh_k[kHeadSize];
    __shared__ float sh_qk;
    __shared__ float sh_g;
    __shared__ float sh_beta;
    __shared__ float warp_partials[4];

    const int q_head_idx = v_head_idx >> 1;
    const int q_base = (((batch_idx * 1 + 0) * kNumQHeads + q_head_idx) * kHeadSize);
    const int k_base = (((batch_idx * 1 + 0) * kNumKHeads + q_head_idx) * kHeadSize);
    if (lane < kHeadSize) {
        sh_q[lane] = load_bf16(q + q_base + lane);
        sh_k[lane] = load_bf16(k + k_base + lane);
    }

    float qk_partial = sh_q[lane] * sh_k[lane];
    qk_partial = warp_sum(qk_partial);
    if ((lane & 31) == 0) {
        warp_partials[lane >> 5] = qk_partial;
    }

    if (lane == 0) {
        float qk = 0.0f;
        #pragma unroll
        for (int warp_idx = 0; warp_idx < 4; ++warp_idx) {
            qk += warp_partials[warp_idx];
        }
        sh_qk = qk;

        const int gate_base = (batch_idx * 1 + 0) * kNumVHeads + v_head_idx;
        const float a_val = load_bf16(a + gate_base);
        const float b_val = load_bf16(b + gate_base);
        const float x = a_val + dt_bias[v_head_idx];
        sh_g = expf(-expf(A_log[v_head_idx]) * log1pf(expf(x)));
        sh_beta = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    const int row_idx = lane;
    const int state_row_base =
        (((batch_idx * kNumVHeads + v_head_idx) * kHeadSize + row_idx) * kHeadSize);
    const int output_base =
        (((batch_idx * 1 + 0) * kNumVHeads + v_head_idx) * kHeadSize + row_idx);

    const float* state_row_ptr = state + state_row_base;
    float* new_state_row_ptr = new_state + state_row_base;

    float old_v = 0.0f;
    float q_old = 0.0f;
    #pragma unroll 8
    for (int k_idx = 0; k_idx < kHeadSize; k_idx += 4) {
        const float4 state_vec = load_global_v4(state_row_ptr + k_idx);
        const float4 q_vec = *reinterpret_cast<const float4*>(&sh_q[k_idx]);
        const float4 k_vec = *reinterpret_cast<const float4*>(&sh_k[k_idx]);
        old_v += k_vec.x * state_vec.x + k_vec.y * state_vec.y + k_vec.z * state_vec.z + k_vec.w * state_vec.w;
        q_old += q_vec.x * state_vec.x + q_vec.y * state_vec.y + q_vec.z * state_vec.z + q_vec.w * state_vec.w;
    }
    old_v *= sh_g;
    q_old *= sh_g;

    const float v_val = load_bf16(v + output_base);
    const float delta = sh_beta * (v_val - old_v);

    #pragma unroll 8
    for (int k_idx = 0; k_idx < kHeadSize; k_idx += 4) {
        const float4 state_vec = load_global_v4(state_row_ptr + k_idx);
        const float4 k_vec = *reinterpret_cast<const float4*>(&sh_k[k_idx]);
        float4 updated;
        updated.x = sh_g * state_vec.x + k_vec.x * delta;
        updated.y = sh_g * state_vec.y + k_vec.y * delta;
        updated.z = sh_g * state_vec.z + k_vec.z * delta;
        updated.w = sh_g * state_vec.w + k_vec.w * delta;
        store_global_v4(new_state_row_ptr + k_idx, updated);
    }

    const float out_val = scale * (q_old + sh_qk * delta);
    output[output_base] = __float2bfloat16(out_val);
}

std::tuple<torch::Tensor, torch::Tensor> gdn_decode_reference(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& state,
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& b,
    double scale) {
    auto q_f32 = q.squeeze(1).to(torch::kFloat);
    auto k_f32 = k.squeeze(1).to(torch::kFloat);
    auto v_f32 = v.squeeze(1).to(torch::kFloat);
    auto a_f32 = a.to(torch::kFloat);
    auto b_f32 = b.to(torch::kFloat);
    auto A_log_f32 = A_log.to(q.device(), torch::kFloat);
    auto dt_bias_f32 = dt_bias.to(q.device(), torch::kFloat);

    auto g = torch::exp(-torch::exp(A_log_f32) * torch::softplus(a_f32 + dt_bias_f32, 1.0, 20.0));
    auto beta = torch::sigmoid(b_f32);

    auto q_exp = expand_heads(q_f32, kNumVHeads);
    auto k_exp = expand_heads(k_f32, kNumVHeads);

    torch::Tensor state_f32;
    if (state.has_value() && state->defined()) {
        state_f32 = state->to(torch::kFloat);
    } else {
        state_f32 = torch::zeros({q.size(0), kNumVHeads, kHeadSize, kHeadSize},
                                 torch::TensorOptions().device(q.device()).dtype(torch::kFloat));
    }

    auto new_state = torch::zeros_like(state_f32);
    auto output = torch::zeros({q.size(0), kNumVHeads, kHeadSize},
                               torch::TensorOptions().device(q.device()).dtype(torch::kFloat));

    for (int64_t batch_idx = 0; batch_idx < q.size(0); ++batch_idx) {
        for (int64_t head_idx = 0; head_idx < kNumVHeads; ++head_idx) {
            auto q_h = q_exp.index({batch_idx, head_idx});
            auto k_h = k_exp.index({batch_idx, head_idx});
            auto v_h = v_f32.index({batch_idx, head_idx});
            auto h_state = state_f32.index({batch_idx, head_idx}).clone().transpose(-1, -2);
            auto g_val = g.index({batch_idx, 0, head_idx});
            auto beta_val = beta.index({batch_idx, 0, head_idx});

            auto old_state = g_val * h_state;
            auto old_v = torch::matmul(k_h, old_state);
            auto new_v = beta_val * v_h + (1.0 - beta_val) * old_v;
            auto state_remove = torch::matmul(k_h.unsqueeze(1), old_v.unsqueeze(0));
            auto state_update = torch::matmul(k_h.unsqueeze(1), new_v.unsqueeze(0));
            h_state = old_state - state_remove + state_update;

            output.index_put_({batch_idx, head_idx}, scale * torch::matmul(q_h, h_state));
            new_state.index_put_({batch_idx, head_idx}, h_state.transpose(-1, -2));
        }
    }

    return std::make_tuple(output.unsqueeze(1).to(torch::kBFloat16), new_state);
}

std::tuple<torch::Tensor, torch::Tensor> gdn_decode(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& state,
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& b,
    double scale = 0.0) {
    TORCH_CHECK(q.dim() == 4 && q.size(1) == 1 && q.size(2) == kNumQHeads && q.size(3) == kHeadSize,
                "q must have shape [B, 1, 4, 128]");
    TORCH_CHECK(k.dim() == 4 && k.size(1) == 1 && k.size(2) == kNumKHeads && k.size(3) == kHeadSize,
                "k must have shape [B, 1, 4, 128]");
    TORCH_CHECK(v.dim() == 4 && v.size(1) == 1 && v.size(2) == kNumVHeads && v.size(3) == kHeadSize,
                "v must have shape [B, 1, 8, 128]");
    TORCH_CHECK(a.dim() == 3 && a.size(0) == q.size(0) && a.size(1) == 1 && a.size(2) == kNumVHeads,
                "a must have shape [B, 1, 8]");
    TORCH_CHECK(b.dim() == 3 && b.size(0) == q.size(0) && b.size(1) == 1 && b.size(2) == kNumVHeads,
                "b must have shape [B, 1, 8]");
    check_common_inputs(A_log, dt_bias);

    const auto batch_size = q.size(0);
    const auto scale_value = resolve_scale(scale);
    if (!q.is_cuda()) {
        return gdn_decode_reference(q, k, v, state, A_log, a, dt_bias, b, scale_value);
    }

    const auto device = q.device();
    auto q_cuda = q.contiguous();
    auto k_cuda = k.contiguous();
    auto v_cuda = v.contiguous();
    auto a_cuda = a.contiguous();
    auto b_cuda = b.contiguous();
    auto A_log_cuda = A_log.to(device, torch::kFloat).contiguous();
    auto dt_bias_cuda = dt_bias.to(device, torch::kFloat).contiguous();

    torch::Tensor state_cuda;
    if (state.has_value() && state->defined()) {
        TORCH_CHECK(state->dim() == 4 && state->size(0) == batch_size && state->size(1) == kNumVHeads &&
                        state->size(2) == kHeadSize && state->size(3) == kHeadSize,
                    "state must have shape [B, 8, 128, 128]");
        state_cuda = state->to(device, torch::kFloat).contiguous();
    } else {
        state_cuda = torch::zeros({batch_size, kNumVHeads, kHeadSize, kHeadSize},
                                  torch::TensorOptions().device(device).dtype(torch::kFloat));
    }

    auto output = torch::empty({batch_size, 1, kNumVHeads, kHeadSize},
                               torch::TensorOptions().device(device).dtype(torch::kBFloat16));
    auto new_state = torch::empty_like(state_cuda);

    const dim3 grid(batch_size, kNumVHeads);
    const dim3 block(kHeadSize);
    gdn_decode_kernel<<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_cuda.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_cuda.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v_cuda.data_ptr<at::BFloat16>()),
        state_cuda.data_ptr<float>(),
        A_log_cuda.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a_cuda.data_ptr<at::BFloat16>()),
        dt_bias_cuda.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(b_cuda.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        new_state.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<float>(scale_value));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(output, new_state);
}

std::tuple<torch::Tensor, torch::Tensor> gdn_prefill(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& state,
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& dt_bias,
    const torch::Tensor& b,
    const torch::Tensor& cu_seqlens,
    double scale = 0.0) {
    TORCH_CHECK(q.dim() == 3 && q.size(1) == kNumQHeads && q.size(2) == kHeadSize,
                "q must have shape [T, 4, 128]");
    TORCH_CHECK(k.dim() == 3 && k.size(1) == kNumKHeads && k.size(2) == kHeadSize,
                "k must have shape [T, 4, 128]");
    TORCH_CHECK(v.dim() == 3 && v.size(1) == kNumVHeads && v.size(2) == kHeadSize,
                "v must have shape [T, 8, 128]");
    TORCH_CHECK(a.dim() == 2 && a.size(0) == q.size(0) && a.size(1) == kNumVHeads,
                "a must have shape [T, 8]");
    TORCH_CHECK(b.dim() == 2 && b.size(0) == q.size(0) && b.size(1) == kNumVHeads,
                "b must have shape [T, 8]");
    TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be 1D");
    check_common_inputs(A_log, dt_bias);

    const auto total_seq_len = q.size(0);
    const auto num_seqs = cu_seqlens.size(0) - 1;
    const auto scale_value = resolve_scale(scale);
    const auto device = q.device();

    auto q_f32 = q.to(torch::kFloat);
    auto k_f32 = k.to(torch::kFloat);
    auto v_f32 = v.to(torch::kFloat);
    auto a_f32 = a.to(torch::kFloat);
    auto b_f32 = b.to(torch::kFloat);
    auto A_log_f32 = A_log.to(device, torch::kFloat);
    auto dt_bias_f32 = dt_bias.to(device, torch::kFloat);

    auto g = torch::exp(-torch::exp(A_log_f32) * torch::softplus(a_f32 + dt_bias_f32, 1.0, 20.0));
    auto beta = torch::sigmoid(b_f32);

    auto q_exp = expand_heads(q_f32, kNumVHeads);
    auto k_exp = expand_heads(k_f32, kNumVHeads);

    torch::Tensor state_f32;
    if (state.has_value() && state->defined()) {
        TORCH_CHECK(state->dim() == 4 && state->size(0) == num_seqs && state->size(1) == kNumVHeads &&
                        state->size(2) == kHeadSize && state->size(3) == kHeadSize,
                    "state must have shape [N, 8, 128, 128]");
        state_f32 = state->to(torch::kFloat);
    } else {
        state_f32 = torch::zeros({num_seqs, kNumVHeads, kHeadSize, kHeadSize},
                                 torch::TensorOptions().device(device).dtype(torch::kFloat));
    }

    auto output = torch::zeros({total_seq_len, kNumVHeads, kHeadSize},
                               torch::TensorOptions().device(device).dtype(torch::kBFloat16));
    auto new_state = torch::zeros_like(state_f32);
    auto cu_seqlens_cpu = cu_seqlens.to(torch::kCPU, torch::kLong);
    auto cu_ptr = cu_seqlens_cpu.data_ptr<int64_t>();

    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
        const auto seq_start = cu_ptr[seq_idx];
        const auto seq_end = cu_ptr[seq_idx + 1];
        if (seq_end <= seq_start) {
            continue;
        }

        auto state_hkv = state_f32.index({seq_idx}).clone().transpose(-1, -2);
        for (int64_t token_idx = seq_start; token_idx < seq_end; ++token_idx) {
            auto q_h1k = q_exp.index({token_idx}).unsqueeze(1);
            auto k_h1k = k_exp.index({token_idx}).unsqueeze(1);
            auto v_h1v = v_f32.index({token_idx}).unsqueeze(1);
            auto g_h11 = g.index({token_idx}).unsqueeze(1).unsqueeze(2);
            auto beta_h11 = beta.index({token_idx}).unsqueeze(1).unsqueeze(2);

            auto old_state = g_h11 * state_hkv;
            auto old_v = torch::bmm(k_h1k, old_state);
            auto new_v = beta_h11 * v_h1v + (1.0 - beta_h11) * old_v;
            auto state_remove = torch::bmm(k_h1k.transpose(-1, -2), old_v);
            auto state_update = torch::bmm(k_h1k.transpose(-1, -2), new_v);
            state_hkv = old_state - state_remove + state_update;

            auto o_h1v = scale_value * torch::bmm(q_h1k, state_hkv);
            output.index_put_({token_idx}, o_h1v.squeeze(1).to(torch::kBFloat16));
        }

        new_state.index_put_({seq_idx}, state_hkv.transpose(-1, -2));
    }

    return std::make_tuple(output, new_state);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gdn_decode", &gdn_decode, "Gated Delta Net decode");
    m.def("gdn_prefill", &gdn_prefill, "Gated Delta Net prefill");
}
