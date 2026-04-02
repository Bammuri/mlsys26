#include <cmath>
#include <tuple>

#include <torch/extension.h>

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
    const auto device = q.device();

    auto q_f32 = q.squeeze(1).to(torch::kFloat);
    auto k_f32 = k.squeeze(1).to(torch::kFloat);
    auto v_f32 = v.squeeze(1).to(torch::kFloat);
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
        TORCH_CHECK(state->dim() == 4 && state->size(0) == batch_size && state->size(1) == kNumVHeads &&
                        state->size(2) == kHeadSize && state->size(3) == kHeadSize,
                    "state must have shape [B, 8, 128, 128]");
        state_f32 = state->to(torch::kFloat);
    } else {
        state_f32 = torch::zeros({batch_size, kNumVHeads, kHeadSize, kHeadSize},
                                 torch::TensorOptions().device(device).dtype(torch::kFloat));
    }

    auto new_state = torch::zeros_like(state_f32);
    auto output = torch::zeros({batch_size, kNumVHeads, kHeadSize},
                               torch::TensorOptions().device(device).dtype(torch::kFloat));

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
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

            output.index_put_({batch_idx, head_idx}, scale_value * torch::matmul(q_h, h_state));
            new_state.index_put_({batch_idx, head_idx}, h_state.transpose(-1, -2));
        }
    }

    return std::make_tuple(output.unsqueeze(1).to(torch::kBFloat16), new_state);
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
