/*
 * Gated Delta Net decode kernel for MLSys 2026.
 *
 * This is a correctness-first implementation that mirrors the reference
 * definition in definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json.
 */

#include <cmath>
#include <cstdint>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace {

using tvm::ffi::TensorView;
namespace cg = cooperative_groups;

constexpr int kSeqLen = 1;
constexpr int kNumQHeads = 4;
constexpr int kNumKHeads = 4;
constexpr int kNumVHeads = 8;
constexpr int kHeadSize = 128;
constexpr int kVTileRows = 32;
constexpr int kMaxTrackedCudaDevices = 16;

__device__ inline float load_bf16(const __nv_bfloat16* ptr, int64_t idx) {
  return __bfloat162float(ptr[idx]);
}

__device__ inline void store_bf16(__nv_bfloat16* ptr, int64_t idx, float value) {
  ptr[idx] = __float2bfloat16(value);
}

__global__ void gdn_decode_optimized_kernel(const __nv_bfloat16* __restrict__ q,
                                            const __nv_bfloat16* __restrict__ k,
                                            const __nv_bfloat16* __restrict__ v,
                                            const float* __restrict__ state,
                                            const float* __restrict__ A_log,
                                            const __nv_bfloat16* __restrict__ a,
                                            const float* __restrict__ dt_bias,
                                            const __nv_bfloat16* __restrict__ b, float scale,
                                            __nv_bfloat16* __restrict__ output,
                                            float* __restrict__ new_state) {
  int64_t batch_idx = blockIdx.x;
  int64_t hv_idx = blockIdx.y;
  int64_t v_tile = blockIdx.z;
  int local_v_idx = threadIdx.x;
  int64_t v_idx = v_tile * kVTileRows + local_v_idx;
  if (v_idx >= kHeadSize) {
    return;
  }

  constexpr int64_t kGroupQ = kNumVHeads / kNumQHeads;
  constexpr int64_t kGroupK = kNumVHeads / kNumKHeads;
  int64_t q_head = hv_idx / kGroupQ;
  int64_t k_head = hv_idx / kGroupK;

  int64_t q_base = (((batch_idx * kSeqLen) * kNumQHeads + q_head) * kHeadSize);
  int64_t k_base = (((batch_idx * kSeqLen) * kNumKHeads + k_head) * kHeadSize);
  int64_t v_base = (((batch_idx * kSeqLen) * kNumVHeads + hv_idx) * kHeadSize);
  int64_t state_base = (((batch_idx * kNumVHeads + hv_idx) * kHeadSize + v_idx) * kHeadSize);
  int64_t output_base = (((batch_idx * kSeqLen) * kNumVHeads + hv_idx) * kHeadSize + v_idx);

  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ float shared_storage[];
  float* state_sh = shared_storage;
  float* q_sh = state_sh + (kVTileRows * kHeadSize);
  float* k_sh = q_sh + kHeadSize;
  __shared__ float g_gate_sh;
  __shared__ float beta_sh;

  const float* state_tile =
      state + ((((batch_idx * kNumVHeads + hv_idx) * kHeadSize) + v_tile * kVTileRows) * kHeadSize);
  cg::memcpy_async(cta, state_sh, state_tile, sizeof(float) * kVTileRows * kHeadSize);

  const __nv_bfloat162* q_vec2 = reinterpret_cast<const __nv_bfloat162*>(q + q_base);
  const __nv_bfloat162* k_vec2 = reinterpret_cast<const __nv_bfloat162*>(k + k_base);
  for (int idx = local_v_idx; idx < kHeadSize / 2; idx += blockDim.x) {
    float2 q_vals = __bfloat1622float2(q_vec2[idx]);
    float2 k_vals = __bfloat1622float2(k_vec2[idx]);
    int dst = idx * 2;
    q_sh[dst] = q_vals.x;
    q_sh[dst + 1] = q_vals.y;
    k_sh[dst] = k_vals.x;
    k_sh[dst + 1] = k_vals.y;
  }

  if (local_v_idx == 0) {
    int64_t gate_base = ((batch_idx * kSeqLen) * kNumVHeads + hv_idx);
    g_gate_sh =
        expf(-expf(A_log[hv_idx]) * log1pf(expf(load_bf16(a, gate_base) + dt_bias[hv_idx])));
    beta_sh = 1.0f / (1.0f + expf(-load_bf16(b, gate_base)));
  }
  cg::wait(cta);
  __syncthreads();

  float v_value = load_bf16(v, v_base + v_idx);
  float* state_row = state_sh + static_cast<int64_t>(local_v_idx) * kHeadSize;
  const float4* state_row_vec4 = reinterpret_cast<const float4*>(state_row);
  const float4* q_vec4 = reinterpret_cast<const float4*>(q_sh);
  const float4* k_vec4 = reinterpret_cast<const float4*>(k_sh);

  float old_v = 0.0f;
  #pragma unroll
  for (int k_idx = 0; k_idx < kHeadSize / 4; ++k_idx) {
    float4 state_vals = state_row_vec4[k_idx];
    float4 k_vals = k_vec4[k_idx];
    old_v += k_vals.x * state_vals.x + k_vals.y * state_vals.y + k_vals.z * state_vals.z +
             k_vals.w * state_vals.w;
  }
  old_v *= g_gate_sh;

  float new_v = beta_sh * v_value + (1.0f - beta_sh) * old_v;
  float delta = new_v - old_v;
  float out = 0.0f;
  float4* new_state_vec4 = reinterpret_cast<float4*>(new_state + state_base);
  #pragma unroll
  for (int k_idx = 0; k_idx < kHeadSize / 4; ++k_idx) {
    float4 state_vals = state_row_vec4[k_idx];
    float4 q_vals = q_vec4[k_idx];
    float4 k_vals = k_vec4[k_idx];
    float4 new_state_vals = make_float4(g_gate_sh * state_vals.x + k_vals.x * delta,
                                        g_gate_sh * state_vals.y + k_vals.y * delta,
                                        g_gate_sh * state_vals.z + k_vals.z * delta,
                                        g_gate_sh * state_vals.w + k_vals.w * delta);
    new_state_vec4[k_idx] = new_state_vals;
    out += q_vals.x * new_state_vals.x + q_vals.y * new_state_vals.y +
           q_vals.z * new_state_vals.z + q_vals.w * new_state_vals.w;
  }

  store_bf16(output, output_base, scale * out);
}

inline void check_cuda_tensor(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA)
      << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

inline void check_bfloat16_tensor(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.dtype().code, kDLBfloat) << name << " must use bfloat16";
  TVM_FFI_ICHECK_EQ(tensor.dtype().bits, 16) << name << " must use bfloat16";
}

inline void check_float32_tensor(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.dtype().code, kDLFloat) << name << " must use float32";
  TVM_FFI_ICHECK_EQ(tensor.dtype().bits, 32) << name << " must use float32";
}

inline void check_int64_tensor(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.dtype().code, kDLInt) << name << " must use int64";
  TVM_FFI_ICHECK_EQ(tensor.dtype().bits, 64) << name << " must use int64";
}

__global__ void gdn_prefill_baseline_kernel(const __nv_bfloat16* __restrict__ q,
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
                                            float* __restrict__ new_state) {
  const int64_t seq_idx = blockIdx.x;
  const int64_t hv_idx = blockIdx.y;
  const int64_t v_tile = blockIdx.z;
  const int local_v_idx = threadIdx.x;
  const int64_t v_idx = v_tile * kVTileRows + local_v_idx;
  if (v_idx >= kHeadSize) {
    return;
  }

  const int64_t seq_start = cu_seqlens[seq_idx];
  const int64_t seq_end = cu_seqlens[seq_idx + 1];
  if (seq_end <= seq_start) {
    return;
  }

  constexpr int64_t kGroupQ = kNumVHeads / kNumQHeads;
  constexpr int64_t kGroupK = kNumVHeads / kNumKHeads;
  const int64_t q_head = hv_idx / kGroupQ;
  const int64_t k_head = hv_idx / kGroupK;
  const int64_t state_tile_base =
      ((((seq_idx * kNumVHeads + hv_idx) * kHeadSize) + v_tile * kVTileRows) * kHeadSize);

  extern __shared__ float shared_storage[];
  float* state_sh = shared_storage;
  float* q_sh = state_sh + (kVTileRows * kHeadSize);
  float* k_sh = q_sh + kHeadSize;
  __shared__ float g_gate_sh;
  __shared__ float beta_sh;

  for (int idx = local_v_idx; idx < kVTileRows * kHeadSize; idx += blockDim.x) {
    state_sh[idx] = state[state_tile_base + idx];
  }
  __syncthreads();

  float* state_row = state_sh + static_cast<int64_t>(local_v_idx) * kHeadSize;

  for (int64_t token_idx = seq_start; token_idx < seq_end; ++token_idx) {
    const int64_t q_base = ((token_idx * kNumQHeads + q_head) * kHeadSize);
    const int64_t k_base = ((token_idx * kNumKHeads + k_head) * kHeadSize);
    const int64_t v_base = ((token_idx * kNumVHeads + hv_idx) * kHeadSize);
    const int64_t output_base = ((token_idx * kNumVHeads + hv_idx) * kHeadSize);

    for (int idx = local_v_idx; idx < kHeadSize; idx += blockDim.x) {
      q_sh[idx] = load_bf16(q, q_base + idx);
      k_sh[idx] = load_bf16(k, k_base + idx);
    }
    if (local_v_idx == 0) {
      const int64_t gate_base = token_idx * kNumVHeads + hv_idx;
      const float x = load_bf16(a, gate_base) + dt_bias[hv_idx];
      g_gate_sh = expf(-expf(A_log[hv_idx]) * log1pf(expf(x)));
      beta_sh = 1.0f / (1.0f + expf(-load_bf16(b, gate_base)));
    }
    __syncthreads();

    float old_v = 0.0f;
    float out = 0.0f;
    for (int k_idx = 0; k_idx < kHeadSize; ++k_idx) {
      const float state_val = state_row[k_idx];
      old_v += k_sh[k_idx] * state_val;
      out += q_sh[k_idx] * state_val;
    }
    old_v *= g_gate_sh;
    out *= g_gate_sh;

    const float v_value = load_bf16(v, v_base + v_idx);
    const float new_v = beta_sh * v_value + (1.0f - beta_sh) * old_v;
    const float delta = new_v - old_v;

    for (int k_idx = 0; k_idx < kHeadSize; ++k_idx) {
      state_row[k_idx] = g_gate_sh * state_row[k_idx] + k_sh[k_idx] * delta;
      out += q_sh[k_idx] * k_sh[k_idx] * delta;
    }
    store_bf16(output, output_base + v_idx, scale * out);
    __syncthreads();
  }

  for (int idx = local_v_idx; idx < kVTileRows * kHeadSize; idx += blockDim.x) {
    new_state[state_tile_base + idx] = state_sh[idx];
  }
}

}  // namespace

void gdn_decode_qk4_v8_d128_k_last(TensorView q, TensorView k, TensorView v, TensorView state,
                                   TensorView A_log, TensorView a, TensorView dt_bias,
                                   TensorView b, double scale, TensorView output,
                                   TensorView new_state) {
  check_cuda_tensor(q, "q");
  check_cuda_tensor(k, "k");
  check_cuda_tensor(v, "v");
  check_cuda_tensor(state, "state");
  check_cuda_tensor(A_log, "A_log");
  check_cuda_tensor(a, "a");
  check_cuda_tensor(dt_bias, "dt_bias");
  check_cuda_tensor(b, "b");
  check_cuda_tensor(output, "output");
  check_cuda_tensor(new_state, "new_state");

  check_bfloat16_tensor(q, "q");
  check_bfloat16_tensor(k, "k");
  check_bfloat16_tensor(v, "v");
  check_bfloat16_tensor(a, "a");
  check_bfloat16_tensor(b, "b");
  check_bfloat16_tensor(output, "output");
  check_float32_tensor(state, "state");
  check_float32_tensor(A_log, "A_log");
  check_float32_tensor(dt_bias, "dt_bias");
  check_float32_tensor(new_state, "new_state");

  TVM_FFI_ICHECK_EQ(q.ndim(), 4);
  TVM_FFI_ICHECK_EQ(k.ndim(), 4);
  TVM_FFI_ICHECK_EQ(v.ndim(), 4);
  TVM_FFI_ICHECK_EQ(state.ndim(), 4);
  TVM_FFI_ICHECK_EQ(A_log.ndim(), 1);
  TVM_FFI_ICHECK_EQ(a.ndim(), 3);
  TVM_FFI_ICHECK_EQ(dt_bias.ndim(), 1);
  TVM_FFI_ICHECK_EQ(b.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.ndim(), 4);
  TVM_FFI_ICHECK_EQ(new_state.ndim(), 4);

  int64_t batch_size = q.size(0);
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_EQ(q.size(1), kSeqLen);
  TVM_FFI_ICHECK_EQ(k.size(1), kSeqLen);
  TVM_FFI_ICHECK_EQ(v.size(1), kSeqLen);
  TVM_FFI_ICHECK_EQ(a.size(1), kSeqLen);
  TVM_FFI_ICHECK_EQ(b.size(1), kSeqLen);
  TVM_FFI_ICHECK_EQ(q.size(2), kNumQHeads);
  TVM_FFI_ICHECK_EQ(k.size(2), kNumKHeads);
  TVM_FFI_ICHECK_EQ(v.size(2), kNumVHeads);
  TVM_FFI_ICHECK_EQ(a.size(2), kNumVHeads);
  TVM_FFI_ICHECK_EQ(b.size(2), kNumVHeads);
  TVM_FFI_ICHECK_EQ(q.size(3), kHeadSize);
  TVM_FFI_ICHECK_EQ(k.size(3), kHeadSize);
  TVM_FFI_ICHECK_EQ(v.size(3), kHeadSize);
  TVM_FFI_ICHECK_EQ(state.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(state.size(1), kNumVHeads);
  TVM_FFI_ICHECK_EQ(state.size(2), kHeadSize);
  TVM_FFI_ICHECK_EQ(state.size(3), kHeadSize);
  TVM_FFI_ICHECK_EQ(A_log.size(0), kNumVHeads);
  TVM_FFI_ICHECK_EQ(dt_bias.size(0), kNumVHeads);
  TVM_FFI_ICHECK_EQ(output.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(output.size(1), kSeqLen);
  TVM_FFI_ICHECK_EQ(output.size(2), kNumVHeads);
  TVM_FFI_ICHECK_EQ(output.size(3), kHeadSize);
  TVM_FFI_ICHECK_EQ(new_state.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(new_state.size(1), kNumVHeads);
  TVM_FFI_ICHECK_EQ(new_state.size(2), kHeadSize);
  TVM_FFI_ICHECK_EQ(new_state.size(3), kHeadSize);
  TVM_FFI_ICHECK_EQ(q.device().device_id, k.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, v.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, state.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, new_state.device().device_id);

  float scale_f = static_cast<float>(scale);
  if (scale_f == 0.0f) {
    scale_f = 1.0f / std::sqrt(static_cast<float>(kHeadSize));
  }

  constexpr int threads = kVTileRows;
  constexpr int v_tiles = kHeadSize / kVTileRows;
  dim3 blocks(static_cast<unsigned int>(batch_size), static_cast<unsigned int>(kNumVHeads),
              static_cast<unsigned int>(v_tiles));
  size_t shared_mem_bytes =
      static_cast<size_t>(kVTileRows * kHeadSize + 2 * kHeadSize) * sizeof(float);
  int device_id = q.device().device_id;
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(q.device().device_type,
                                                             device_id));

  static bool kernel_attrs_configured[kMaxTrackedCudaDevices] = {};
  TVM_FFI_ICHECK_GE(device_id, 0);
  TVM_FFI_ICHECK_LT(device_id, kMaxTrackedCudaDevices);
  cudaError_t err = cudaSuccess;
  if (!kernel_attrs_configured[device_id]) {
    err = cudaFuncSetAttribute(gdn_decode_optimized_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shared_mem_bytes));
    TVM_FFI_ICHECK_EQ(err, cudaSuccess)
        << "Failed to raise dynamic shared memory limit: " << cudaGetErrorString(err);
    err = cudaFuncSetAttribute(gdn_decode_optimized_kernel,
                               cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    TVM_FFI_ICHECK_EQ(err, cudaSuccess)
        << "Failed to set shared memory carveout: " << cudaGetErrorString(err);
    kernel_attrs_configured[device_id] = true;
  }

  gdn_decode_optimized_kernel<<<blocks, threads, shared_mem_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()), static_cast<const __nv_bfloat16*>(k.data_ptr()),
      static_cast<const __nv_bfloat16*>(v.data_ptr()), static_cast<const float*>(state.data_ptr()),
      static_cast<const float*>(A_log.data_ptr()), static_cast<const __nv_bfloat16*>(a.data_ptr()),
      static_cast<const float*>(dt_bias.data_ptr()), static_cast<const __nv_bfloat16*>(b.data_ptr()),
      scale_f, static_cast<__nv_bfloat16*>(output.data_ptr()), static_cast<float*>(new_state.data_ptr()));

  err = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "CUDA kernel launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_decode_qk4_v8_d128_k_last, gdn_decode_qk4_v8_d128_k_last);

void gdn_prefill_qk4_v8_d128_k_last(TensorView q, TensorView k, TensorView v, TensorView state,
                                    TensorView A_log, TensorView a, TensorView dt_bias,
                                    TensorView b, TensorView cu_seqlens, double scale,
                                    TensorView output, TensorView new_state) {
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

  check_bfloat16_tensor(q, "q");
  check_bfloat16_tensor(k, "k");
  check_bfloat16_tensor(v, "v");
  check_bfloat16_tensor(a, "a");
  check_bfloat16_tensor(b, "b");
  check_bfloat16_tensor(output, "output");
  check_float32_tensor(state, "state");
  check_float32_tensor(A_log, "A_log");
  check_float32_tensor(dt_bias, "dt_bias");
  check_float32_tensor(new_state, "new_state");
  check_int64_tensor(cu_seqlens, "cu_seqlens");

  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(k.ndim(), 3);
  TVM_FFI_ICHECK_EQ(v.ndim(), 3);
  TVM_FFI_ICHECK_EQ(state.ndim(), 4);
  TVM_FFI_ICHECK_EQ(A_log.ndim(), 1);
  TVM_FFI_ICHECK_EQ(a.ndim(), 2);
  TVM_FFI_ICHECK_EQ(dt_bias.ndim(), 1);
  TVM_FFI_ICHECK_EQ(b.ndim(), 2);
  TVM_FFI_ICHECK_EQ(cu_seqlens.ndim(), 1);
  TVM_FFI_ICHECK_EQ(output.ndim(), 3);
  TVM_FFI_ICHECK_EQ(new_state.ndim(), 4);

  const int64_t total_seq_len = q.size(0);
  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  TVM_FFI_ICHECK_GT(total_seq_len, 0);
  TVM_FFI_ICHECK_GT(num_seqs, 0);
  TVM_FFI_ICHECK_EQ(q.size(1), kNumQHeads);
  TVM_FFI_ICHECK_EQ(k.size(1), kNumKHeads);
  TVM_FFI_ICHECK_EQ(v.size(1), kNumVHeads);
  TVM_FFI_ICHECK_EQ(a.size(1), kNumVHeads);
  TVM_FFI_ICHECK_EQ(b.size(1), kNumVHeads);
  TVM_FFI_ICHECK_EQ(q.size(2), kHeadSize);
  TVM_FFI_ICHECK_EQ(k.size(2), kHeadSize);
  TVM_FFI_ICHECK_EQ(v.size(2), kHeadSize);
  TVM_FFI_ICHECK_EQ(state.size(0), num_seqs);
  TVM_FFI_ICHECK_EQ(state.size(1), kNumVHeads);
  TVM_FFI_ICHECK_EQ(state.size(2), kHeadSize);
  TVM_FFI_ICHECK_EQ(state.size(3), kHeadSize);
  TVM_FFI_ICHECK_EQ(A_log.size(0), kNumVHeads);
  TVM_FFI_ICHECK_EQ(dt_bias.size(0), kNumVHeads);
  TVM_FFI_ICHECK_EQ(output.size(0), total_seq_len);
  TVM_FFI_ICHECK_EQ(output.size(1), kNumVHeads);
  TVM_FFI_ICHECK_EQ(output.size(2), kHeadSize);
  TVM_FFI_ICHECK_EQ(new_state.size(0), num_seqs);
  TVM_FFI_ICHECK_EQ(new_state.size(1), kNumVHeads);
  TVM_FFI_ICHECK_EQ(new_state.size(2), kHeadSize);
  TVM_FFI_ICHECK_EQ(new_state.size(3), kHeadSize);
  TVM_FFI_ICHECK_EQ(q.device().device_id, state.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK_EQ(q.device().device_id, new_state.device().device_id);

  float scale_f = static_cast<float>(scale);
  if (scale_f == 0.0f) {
    scale_f = 1.0f / std::sqrt(static_cast<float>(kHeadSize));
  }

  constexpr int threads = kVTileRows;
  constexpr int v_tiles = kHeadSize / kVTileRows;
  dim3 blocks(static_cast<unsigned int>(num_seqs), static_cast<unsigned int>(kNumVHeads),
              static_cast<unsigned int>(v_tiles));
  size_t shared_mem_bytes =
      static_cast<size_t>(kVTileRows * kHeadSize + 2 * kHeadSize) * sizeof(float);
  const int device_id = q.device().device_id;
  auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(q.device().device_type, device_id));

  gdn_prefill_baseline_kernel<<<blocks, threads, shared_mem_bytes, stream>>>(
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
      static_cast<float*>(new_state.data_ptr()));

  cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "CUDA kernel launch failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_prefill_qk4_v8_d128_k_last, gdn_prefill_qk4_v8_d128_k_last);
