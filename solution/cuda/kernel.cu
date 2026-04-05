#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kHeadSize = 128;
constexpr int kNumQHeads = 4;
constexpr int kNumKHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kThreads = 128;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_F32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_I64(x) TORCH_CHECK((x).scalar_type() == torch::kInt64, #x " must be int64")

__device__ inline float bf16_to_float(const c10::BFloat16* ptr) {
  const __nv_bfloat16* raw = reinterpret_cast<const __nv_bfloat16*>(ptr);
  return __bfloat162float(*raw);
}

__device__ inline void float_to_bf16(float x, c10::BFloat16* ptr) {
  __nv_bfloat16* raw = reinterpret_cast<__nv_bfloat16*>(ptr);
  *raw = __float2bfloat16(x);
}

__device__ inline float softplusf_stable(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

__device__ inline float dot_float4(const float4& a, const float4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__global__ void gdn_prefill_kernel(
    const c10::BFloat16* __restrict__ q,
    const c10::BFloat16* __restrict__ k,
    const c10::BFloat16* __restrict__ v,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    const float* __restrict__ A_log,
    const c10::BFloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const c10::BFloat16* __restrict__ b,
    const int64_t* __restrict__ cu_seqlens,
    c10::BFloat16* __restrict__ output,
    int64_t num_seqs,
    double scale,
    bool has_state) {
  int seq_idx = blockIdx.y;
  int head_idx = blockIdx.x;
  int v_idx = threadIdx.x;

  if (seq_idx >= num_seqs || head_idx >= kNumVHeads || v_idx >= kHeadSize) {
    return;
  }

  extern __shared__ float shared_mem[];
  float* state_sh = shared_mem;  // [128, 128]
  float* q_sh = state_sh + kHeadSize * kHeadSize;
  float* k_sh = q_sh + kHeadSize;
  __shared__ float gate_sh;
  __shared__ float beta_sh;

  float* row = state_sh + v_idx * kHeadSize;
  float4* row4 = reinterpret_cast<float4*>(row);
  int64_t state_base = ((static_cast<int64_t>(seq_idx) * kNumVHeads + head_idx) * kHeadSize + v_idx) * kHeadSize;

  if (has_state) {
    const float4* state_in4 = reinterpret_cast<const float4*>(state_in + state_base);
#pragma unroll
    for (int i = 0; i < kHeadSize / 4; ++i) {
      row4[i] = state_in4[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < kHeadSize / 4; ++i) {
      row4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  __syncthreads();

  int64_t seq_start = cu_seqlens[seq_idx];
  int64_t seq_end = cu_seqlens[seq_idx + 1];
  int q_head_idx = head_idx / (kNumVHeads / kNumQHeads);
  int k_head_idx = head_idx / (kNumVHeads / kNumKHeads);

  for (int64_t t = seq_start; t < seq_end; ++t) {
    int64_t k_base = (t * kNumKHeads + k_head_idx) * kHeadSize;
    int64_t q_base = (t * kNumQHeads + q_head_idx) * kHeadSize;
    int64_t v_base = (t * kNumVHeads + head_idx) * kHeadSize;

    q_sh[v_idx] = bf16_to_float(q + q_base + v_idx);
    k_sh[v_idx] = bf16_to_float(k + k_base + v_idx);
    __syncthreads();

    if (v_idx == 0) {
      float a_val = bf16_to_float(a + t * kNumVHeads + head_idx);
      float b_val = bf16_to_float(b + t * kNumVHeads + head_idx);
      float x = a_val + dt_bias[head_idx];
      gate_sh = expf(-expf(A_log[head_idx]) * softplusf_stable(x));
      beta_sh = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    float old_v = 0.0f;
    const float4* k4 = reinterpret_cast<const float4*>(k_sh);
#pragma unroll
    for (int i = 0; i < kHeadSize / 4; ++i) {
      old_v += dot_float4(k4[i], row4[i]);
    }
    old_v *= gate_sh;

    float v_val = bf16_to_float(v + v_base + v_idx);
    float diff = beta_sh * (v_val - old_v);

#pragma unroll
    for (int i = 0; i < kHeadSize / 4; ++i) {
      float4 k_vec = k4[i];
      float4 r_vec = row4[i];
      r_vec.x = gate_sh * r_vec.x + k_vec.x * diff;
      r_vec.y = gate_sh * r_vec.y + k_vec.y * diff;
      r_vec.z = gate_sh * r_vec.z + k_vec.z * diff;
      r_vec.w = gate_sh * r_vec.w + k_vec.w * diff;
      row4[i] = r_vec;
    }

    float out = 0.0f;
    const float4* q4 = reinterpret_cast<const float4*>(q_sh);
#pragma unroll
    for (int i = 0; i < kHeadSize / 4; ++i) {
      out += dot_float4(q4[i], row4[i]);
    }
    float_to_bf16(static_cast<float>(scale) * out, output + v_base + v_idx);
    __syncthreads();
  }

  float4* state_out4 = reinterpret_cast<float4*>(state_out + state_base);
#pragma unroll
  for (int i = 0; i < kHeadSize / 4; ++i) {
    state_out4[i] = row4[i];
  }
}

}  // namespace

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
    double scale) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_CUDA(A_log);
  CHECK_CUDA(a);
  CHECK_CUDA(dt_bias);
  CHECK_CUDA(b);
  CHECK_CUDA(cu_seqlens);

  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);
  CHECK_CONTIGUOUS(A_log);
  CHECK_CONTIGUOUS(a);
  CHECK_CONTIGUOUS(dt_bias);
  CHECK_CONTIGUOUS(b);
  CHECK_CONTIGUOUS(cu_seqlens);

  CHECK_BF16(q);
  CHECK_BF16(k);
  CHECK_BF16(v);
  CHECK_BF16(a);
  CHECK_BF16(b);
  CHECK_F32(A_log);
  CHECK_F32(dt_bias);
  CHECK_I64(cu_seqlens);

  TORCH_CHECK(q.dim() == 3, "q must have shape [total_seq_len, 4, 128]");
  TORCH_CHECK(k.dim() == 3, "k must have shape [total_seq_len, 4, 128]");
  TORCH_CHECK(v.dim() == 3, "v must have shape [total_seq_len, 8, 128]");
  TORCH_CHECK(q.size(1) == kNumQHeads && k.size(1) == kNumKHeads && v.size(1) == kNumVHeads,
              "unexpected head counts");
  TORCH_CHECK(q.size(2) == kHeadSize && k.size(2) == kHeadSize && v.size(2) == kHeadSize,
              "head size must be 128");
  TORCH_CHECK(A_log.numel() == kNumVHeads, "A_log must have 8 elements");
  TORCH_CHECK(dt_bias.numel() == kNumVHeads, "dt_bias must have 8 elements");
  TORCH_CHECK(a.size(0) == q.size(0) && a.size(1) == kNumVHeads, "a must have shape [T, 8]");
  TORCH_CHECK(b.size(0) == q.size(0) && b.size(1) == kNumVHeads, "b must have shape [T, 8]");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2, "cu_seqlens must be [N+1]");

  c10::cuda::CUDAGuard device_guard(q.device());

  if (scale == 0.0) {
    scale = 1.0 / std::sqrt(static_cast<double>(kHeadSize));
  }

  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  A_log = A_log.contiguous();
  a = a.contiguous();
  dt_bias = dt_bias.contiguous();
  b = b.contiguous();
  cu_seqlens = cu_seqlens.contiguous();

  bool has_state = state.has_value() && state.value().defined();
  torch::Tensor state_in;
  if (has_state) {
    state_in = state.value().contiguous();
    CHECK_CUDA(state_in);
    CHECK_CONTIGUOUS(state_in);
    CHECK_F32(state_in);
  }

  int64_t num_seqs = cu_seqlens.numel() - 1;
  auto output = torch::empty({q.size(0), kNumVHeads, kHeadSize}, q.options());
  auto new_state = torch::empty(
      {num_seqs, kNumVHeads, kHeadSize, kHeadSize},
      torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

  if (has_state) {
    TORCH_CHECK(state_in.sizes() == new_state.sizes(), "state must have shape [num_seqs, 8, 128, 128]");
  }

  dim3 grid(kNumVHeads, static_cast<unsigned int>(num_seqs), 1);
  dim3 block(kThreads, 1, 1);
  size_t shared_bytes =
      static_cast<size_t>(kHeadSize * kHeadSize + kHeadSize + kHeadSize) * sizeof(float);

  cudaFuncSetAttribute(
      gdn_prefill_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes));

  auto stream = c10::cuda::getDefaultCUDAStream();
  gdn_prefill_kernel<<<grid, block, shared_bytes, stream.stream()>>>(
      q.data_ptr<c10::BFloat16>(),
      k.data_ptr<c10::BFloat16>(),
      v.data_ptr<c10::BFloat16>(),
      has_state ? state_in.data_ptr<float>() : nullptr,
      new_state.data_ptr<float>(),
      A_log.data_ptr<float>(),
      a.data_ptr<c10::BFloat16>(),
      dt_bias.data_ptr<float>(),
      b.data_ptr<c10::BFloat16>(),
      cu_seqlens.data_ptr<int64_t>(),
      output.data_ptr<c10::BFloat16>(),
      num_seqs,
      scale,
      has_state);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(output, new_state);
}
