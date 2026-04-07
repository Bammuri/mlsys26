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
constexpr int kWarpSize = 32;
constexpr int kVecSize = 4;
constexpr int kWarpsPerBlock = 2;
constexpr int kRowsPerBlock = kWarpsPerBlock;
constexpr int kThreads = kWarpsPerBlock * kWarpSize;
constexpr int kRowTilesPerHead = kHeadSize / kRowsPerBlock;

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
  float acc = 0.0f;
  acc = fmaf(a.x, b.x, acc);
  acc = fmaf(a.y, b.y, acc);
  acc = fmaf(a.z, b.z, acc);
  acc = fmaf(a.w, b.w, acc);
  return acc;
}

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ inline float4 load_bf16x4(const c10::BFloat16* ptr) {
  const uint2 raw = *reinterpret_cast<const uint2*>(ptr);
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
  return make_float4(
      __bfloat162float(lo.x),
      __bfloat162float(lo.y),
      __bfloat162float(hi.x),
      __bfloat162float(hi.y));
}

__global__ __launch_bounds__(256, 2) void compute_gate_beta_kernel(
    const float* __restrict__ A_log,
    const c10::BFloat16* __restrict__ a,
    const float* __restrict__ dt_bias,
    const c10::BFloat16* __restrict__ b,
    float2* __restrict__ gate_beta,
    int total_seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = total_seq_len * kNumVHeads;
  if (idx >= total) {
    return;
  }

  int head_idx = idx % kNumVHeads;
  float a_val = bf16_to_float(a + idx);
  float b_val = bf16_to_float(b + idx);
  float x = a_val + dt_bias[head_idx];
  float a_log_exp = expf(A_log[head_idx]);
  gate_beta[idx] = make_float2(
      expf(-a_log_exp * softplusf_stable(x)),
      1.0f / (1.0f + expf(-b_val)));
}

__global__ __launch_bounds__(kThreads) void gdn_prefill_kernel(
    const c10::BFloat16* __restrict__ q,
    const c10::BFloat16* __restrict__ k,
    const c10::BFloat16* __restrict__ v,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    const float2* __restrict__ gate_beta,
    const int64_t* __restrict__ cu_seqlens,
    c10::BFloat16* __restrict__ output,
    int64_t num_seqs,
    double scale,
    bool has_state) {
  int seq_idx = blockIdx.y;
  int head_idx = blockIdx.x / kRowTilesPerHead;
  int tile_idx = blockIdx.x % kRowTilesPerHead;
  int lane = threadIdx.x % kWarpSize;
  int warp_id = threadIdx.x / kWarpSize;
  int row_idx = tile_idx * kRowsPerBlock + warp_id;

  if (seq_idx >= num_seqs || head_idx >= kNumVHeads || warp_id >= kWarpsPerBlock || row_idx >= kHeadSize) {
    return;
  }

  int64_t seq_start = cu_seqlens[seq_idx];
  int64_t seq_end = cu_seqlens[seq_idx + 1];
  int q_head_idx = head_idx / (kNumVHeads / kNumQHeads);
  int k_head_idx = head_idx / (kNumVHeads / kNumKHeads);
  float scale_f = static_cast<float>(scale);
  int64_t state_base =
      ((static_cast<int64_t>(seq_idx) * kNumVHeads + head_idx) * kHeadSize + row_idx) * kHeadSize;

  float4 row_frag;
  if (has_state) {
    const float4* state_in4 = reinterpret_cast<const float4*>(state_in + state_base);
    row_frag = state_in4[lane];
  } else {
    row_frag = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  for (int64_t t = seq_start; t < seq_end; ++t) {
    int64_t k_base = (t * kNumKHeads + k_head_idx) * kHeadSize;
    int64_t q_base = (t * kNumQHeads + q_head_idx) * kHeadSize;
    int64_t v_base = (t * kNumVHeads + head_idx) * kHeadSize;

    float4 q_frag = load_bf16x4(q + q_base + lane * kVecSize);
    float4 k_frag = load_bf16x4(k + k_base + lane * kVecSize);

    float gate_val = 0.0f;
    float beta_val = 0.0f;
    if (lane == 0) {
      int64_t gate_idx = t * kNumVHeads + head_idx;
      float2 gate_beta_vec = gate_beta[gate_idx];
      gate_val = gate_beta_vec.x;
      beta_val = gate_beta_vec.y;
    }
    gate_val = __shfl_sync(0xffffffff, gate_val, 0);
    beta_val = __shfl_sync(0xffffffff, beta_val, 0);

    float old_v = warp_sum(dot_float4(row_frag, k_frag));
    old_v *= gate_val;

    float v_val = 0.0f;
    if (lane == 0) {
      v_val = bf16_to_float(v + v_base + row_idx);
    }
    v_val = __shfl_sync(0xffffffff, v_val, 0);

    float diff = beta_val * (v_val - old_v);
    row_frag.x = fmaf(k_frag.x, diff, gate_val * row_frag.x);
    row_frag.y = fmaf(k_frag.y, diff, gate_val * row_frag.y);
    row_frag.z = fmaf(k_frag.z, diff, gate_val * row_frag.z);
    row_frag.w = fmaf(k_frag.w, diff, gate_val * row_frag.w);

    float out = warp_sum(dot_float4(q_frag, row_frag));
    if (lane == 0) {
      float_to_bf16(scale_f * out, output + v_base + row_idx);
    }
  }

  float4* state_out4 = reinterpret_cast<float4*>(state_out + state_base);
  state_out4[lane] = row_frag;
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

  bool has_state = state.has_value() && state.value().defined();
  torch::Tensor state_in;
  if (has_state) {
    state_in = state.value();
    CHECK_CUDA(state_in);
    CHECK_CONTIGUOUS(state_in);
    CHECK_F32(state_in);
  }

  int64_t num_seqs = cu_seqlens.numel() - 1;
  auto output = torch::empty({q.size(0), kNumVHeads, kHeadSize}, q.options());
  auto new_state = has_state
      ? state_in
      : torch::empty(
            {num_seqs, kNumVHeads, kHeadSize, kHeadSize},
            torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
  auto gate_beta = torch::empty(
      {q.size(0), kNumVHeads, 2},
      torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

  if (has_state) {
    TORCH_CHECK(state_in.sizes() == new_state.sizes(), "state must have shape [num_seqs, 8, 128, 128]");
  }

  dim3 grid(kNumVHeads * kRowTilesPerHead, static_cast<unsigned int>(num_seqs), 1);
  dim3 block(kThreads, 1, 1);

  auto stream = c10::cuda::getDefaultCUDAStream();
  int total_gate_elems = static_cast<int>(q.size(0) * kNumVHeads);
  int pre_threads = 256;
  int pre_blocks = (total_gate_elems + pre_threads - 1) / pre_threads;
  compute_gate_beta_kernel<<<pre_blocks, pre_threads, 0, stream.stream()>>>(
      A_log.data_ptr<float>(),
      a.data_ptr<c10::BFloat16>(),
      dt_bias.data_ptr<float>(),
      b.data_ptr<c10::BFloat16>(),
      reinterpret_cast<float2*>(gate_beta.data_ptr<float>()),
      static_cast<int>(q.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gdn_prefill_kernel<<<grid, block, 0, stream.stream()>>>(
      q.data_ptr<c10::BFloat16>(),
      k.data_ptr<c10::BFloat16>(),
      v.data_ptr<c10::BFloat16>(),
      has_state ? state_in.data_ptr<float>() : nullptr,
      new_state.data_ptr<float>(),
      reinterpret_cast<const float2*>(gate_beta.data_ptr<float>()),
      cu_seqlens.data_ptr<int64_t>(),
      output.data_ptr<c10::BFloat16>(),
      num_seqs,
      scale,
      has_state);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(output, new_state);
}
