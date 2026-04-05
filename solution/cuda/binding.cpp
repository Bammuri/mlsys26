#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> gdn_prefill_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    c10::optional<torch::Tensor> state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    torch::Tensor cu_seqlens,
    double scale);

std::tuple<torch::Tensor, torch::Tensor> run(
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
  return gdn_prefill_cuda(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "Native CUDA GDN prefill");
}
