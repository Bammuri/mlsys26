#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>

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

__device__ __forceinline__ float bf16_to_float(const uint16_t* ptr) {
  const __nv_bfloat16* raw = reinterpret_cast<const __nv_bfloat16*>(ptr);
  return __bfloat162float(*raw);
}

__device__ __forceinline__ void float_to_bf16(float x, uint16_t* ptr) {
  __nv_bfloat16* raw = reinterpret_cast<__nv_bfloat16*>(ptr);
  *raw = __float2bfloat16(x);
}

__device__ __forceinline__ float softplusf_stable(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

__device__ __forceinline__ float4 load_bf16x4(const uint16_t* ptr) {
  const __nv_bfloat162* raw = reinterpret_cast<const __nv_bfloat162*>(ptr);
  const float2 xy = __bfloat1622float2(raw[0]);
  const float2 zw = __bfloat1622float2(raw[1]);
  return make_float4(xy.x, xy.y, zw.x, zw.y);
}

__device__ __forceinline__ float dot_float4(const float4& a, const float4& b) {
  float acc = 0.0f;
  acc = fmaf(a.x, b.x, acc);
  acc = fmaf(a.y, b.y, acc);
  acc = fmaf(a.z, b.z, acc);
  acc = fmaf(a.w, b.w, acc);
  return acc;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_broadcast_0(float value) {
  return __shfl_sync(0xffffffffu, value, 0);
}

__global__ __launch_bounds__(256, 2) void compute_gate_beta_kernel(
    const float* __restrict__ A_log,
    const uint16_t* __restrict__ a,
    const float* __restrict__ dt_bias,
    const uint16_t* __restrict__ b,
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

__global__ __launch_bounds__(kThreads, 4) void gdn_prefill_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    const float2* __restrict__ gate_beta,
    const int64_t* __restrict__ cu_seqlens,
    uint16_t* __restrict__ output,
    int64_t num_seqs,
    float scale) {
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x / kRowTilesPerHead;
  const int row_tile_idx = blockIdx.x % kRowTilesPerHead;
  const int warp_idx = threadIdx.x / kWarpSize;
  const int lane_idx = threadIdx.x % kWarpSize;
  const int row_idx = row_tile_idx * kRowsPerBlock + warp_idx;

  if (seq_idx >= num_seqs || head_idx >= kNumVHeads || row_idx >= kHeadSize) {
    return;
  }

  const int col_base = lane_idx * kVecSize;
  const int q_head_idx = head_idx / (kNumVHeads / kNumQHeads);
  const int k_head_idx = head_idx / (kNumVHeads / kNumKHeads);
  const int64_t seq_start = cu_seqlens[seq_idx];
  const int64_t seq_end = cu_seqlens[seq_idx + 1];
  const int64_t state_offset =
      (((static_cast<int64_t>(seq_idx) * kNumVHeads + head_idx) * kHeadSize + row_idx) * kHeadSize +
       col_base);

  float4 state_vec = reinterpret_cast<const float4*>(state_in + state_offset)[0];

  for (int64_t t = seq_start; t < seq_end; ++t) {
    const int64_t q_offset = ((t * kNumQHeads + q_head_idx) * kHeadSize) + col_base;
    const int64_t k_offset = ((t * kNumKHeads + k_head_idx) * kHeadSize) + col_base;
    const int64_t v_offset = ((t * kNumVHeads + head_idx) * kHeadSize) + row_idx;

    const float4 q_vec = load_bf16x4(q + q_offset);
    const float4 k_vec = load_bf16x4(k + k_offset);

    float gate = 0.0f;
    float beta = 0.0f;
    if (lane_idx == 0) {
      const float2 gate_beta_vec = gate_beta[t * kNumVHeads + head_idx];
      gate = gate_beta_vec.x;
      beta = gate_beta_vec.y;
    }
    gate = warp_broadcast_0(gate);
    beta = warp_broadcast_0(beta);

    float old_v = warp_sum(dot_float4(k_vec, state_vec));
    old_v = gate * warp_broadcast_0(old_v);

    float v_val = 0.0f;
    if (lane_idx == 0) {
      v_val = bf16_to_float(v + v_offset);
    }
    v_val = warp_broadcast_0(v_val);

    const float diff = beta * (v_val - old_v);
    state_vec.x = fmaf(k_vec.x, diff, gate * state_vec.x);
    state_vec.y = fmaf(k_vec.y, diff, gate * state_vec.y);
    state_vec.z = fmaf(k_vec.z, diff, gate * state_vec.z);
    state_vec.w = fmaf(k_vec.w, diff, gate * state_vec.w);

    float out = warp_sum(dot_float4(q_vec, state_vec));
    if (lane_idx == 0) {
      float_to_bf16(scale * out, output + v_offset);
    }
  }

  reinterpret_cast<float4*>(state_out + state_offset)[0] = state_vec;
}

inline bool is_dtype(const tvm::ffi::TensorView& t, uint8_t code, uint8_t bits) {
  DLDataType dt = t.dtype();
  return dt.code == code && dt.bits == bits && dt.lanes == 1;
}

inline void check_cuda_tensor(const tvm::ffi::TensorView& t, const char* name) {
  if (t.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(ValueError) << name << " must be a CUDA tensor";
  }
  if (!t.IsContiguous()) {
    TVM_FFI_THROW(ValueError) << name << " must be contiguous";
  }
}

void msinfer_gdn_prefill(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView cu_seqlens,
    float scale,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView new_state) {
  check_cuda_tensor(q, "q");
  check_cuda_tensor(k, "k");
  check_cuda_tensor(v, "v");
  check_cuda_tensor(state, "state");
  check_cuda_tensor(A_log, "A_log");
  check_cuda_tensor(a, "a");
  check_cuda_tensor(dt_bias, "dt_bias");
  check_cuda_tensor(b, "b");
  check_cuda_tensor(cu_seqlens, "cu_seqlens");
  check_cuda_tensor(output, "output");
  check_cuda_tensor(new_state, "new_state");

  if (!is_dtype(q, kDLBfloat, 16) || !is_dtype(k, kDLBfloat, 16) || !is_dtype(v, kDLBfloat, 16) ||
      !is_dtype(a, kDLBfloat, 16) || !is_dtype(b, kDLBfloat, 16) || !is_dtype(output, kDLBfloat, 16)) {
    TVM_FFI_THROW(TypeError) << "q/k/v/a/b/output must be bfloat16";
  }
  if (!is_dtype(A_log, kDLFloat, 32) || !is_dtype(dt_bias, kDLFloat, 32) || !is_dtype(state, kDLFloat, 32) ||
      !is_dtype(new_state, kDLFloat, 32)) {
    TVM_FFI_THROW(TypeError) << "A_log/dt_bias/state/new_state must be float32";
  }
  if (!is_dtype(cu_seqlens, kDLInt, 64)) {
    TVM_FFI_THROW(TypeError) << "cu_seqlens must be int64";
  }

  if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3) {
    TVM_FFI_THROW(ValueError) << "q/k/v must be rank-3";
  }
  if (state.ndim() != 4 || new_state.ndim() != 4 || output.ndim() != 3) {
    TVM_FFI_THROW(ValueError) << "state/new_state/output ranks are invalid";
  }
  if (q.size(1) != kNumQHeads || k.size(1) != kNumKHeads || v.size(1) != kNumVHeads) {
    TVM_FFI_THROW(ValueError) << "unexpected head counts";
  }
  if (q.size(2) != kHeadSize || k.size(2) != kHeadSize || v.size(2) != kHeadSize) {
    TVM_FFI_THROW(ValueError) << "head size must be 128";
  }
  if (A_log.numel() != kNumVHeads || dt_bias.numel() != kNumVHeads) {
    TVM_FFI_THROW(ValueError) << "A_log and dt_bias must have 8 elements";
  }
  if (a.size(0) != q.size(0) || a.size(1) != kNumVHeads || b.size(0) != q.size(0) || b.size(1) != kNumVHeads) {
    TVM_FFI_THROW(ValueError) << "a/b shape mismatch";
  }
  if (cu_seqlens.ndim() != 1 || cu_seqlens.numel() < 2) {
    TVM_FFI_THROW(ValueError) << "cu_seqlens must be [N+1]";
  }

  const int64_t num_seqs = cu_seqlens.numel() - 1;
  if (state.size(0) != num_seqs || new_state.size(0) != num_seqs) {
    TVM_FFI_THROW(ValueError) << "state/new_state num_seqs mismatch";
  }
  if (state.size(1) != kNumVHeads || state.size(2) != kHeadSize || state.size(3) != kHeadSize ||
      new_state.size(1) != kNumVHeads || new_state.size(2) != kHeadSize || new_state.size(3) != kHeadSize) {
    TVM_FFI_THROW(ValueError) << "state/new_state shape mismatch";
  }
  if (output.size(0) != q.size(0) || output.size(1) != kNumVHeads || output.size(2) != kHeadSize) {
    TVM_FFI_THROW(ValueError) << "output shape mismatch";
  }

  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(kHeadSize));
  }

  DLDevice dev = q.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  float2* gate_beta = nullptr;
  cudaError_t err = cudaMallocAsync(&gate_beta, static_cast<size_t>(q.size(0) * kNumVHeads) * sizeof(float2), stream);
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError) << "cudaMallocAsync(gate_beta) failed: " << cudaGetErrorString(err);
  }

  const int total_gate_elems = static_cast<int>(q.size(0) * kNumVHeads);
  const int pre_threads = 256;
  const int pre_blocks = (total_gate_elems + pre_threads - 1) / pre_threads;
  compute_gate_beta_kernel<<<pre_blocks, pre_threads, 0, stream>>>(
      static_cast<float*>(A_log.data_ptr()),
      static_cast<uint16_t*>(a.data_ptr()),
      static_cast<float*>(dt_bias.data_ptr()),
      static_cast<uint16_t*>(b.data_ptr()),
      gate_beta,
      static_cast<int>(q.size(0)));
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFreeAsync(gate_beta, stream);
    TVM_FFI_THROW(RuntimeError) << "compute_gate_beta_kernel launch failed: " << cudaGetErrorString(err);
  }

  const dim3 grid(kNumVHeads * kRowTilesPerHead, static_cast<unsigned int>(num_seqs), 1);
  const dim3 block(kThreads, 1, 1);
  gdn_prefill_kernel<<<grid, block, 0, stream>>>(
      static_cast<uint16_t*>(q.data_ptr()),
      static_cast<uint16_t*>(k.data_ptr()),
      static_cast<uint16_t*>(v.data_ptr()),
      static_cast<float*>(state.data_ptr()),
      static_cast<float*>(new_state.data_ptr()),
      gate_beta,
      static_cast<int64_t*>(cu_seqlens.data_ptr()),
      static_cast<uint16_t*>(output.data_ptr()),
      num_seqs,
      scale);
  err = cudaGetLastError();
  cudaFreeAsync(gate_beta, stream);
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError) << "gdn_prefill_kernel launch failed: " << cudaGetErrorString(err);
  }
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(msinfer_gdn_prefill, msinfer_gdn_prefill);
