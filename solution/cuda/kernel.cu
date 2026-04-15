/*
 * GDN decode + prefill CUDA kernels with TVM FFI binding.
 * v6: cp.async pipeline for prefill (double-buffered bf16 shared mem).
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
constexpr int kVecsPerRow = kHeadSize / 4;

__device__ __forceinline__ float softplusf_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float sigmoidf_stable(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(x);
    return z / (1.0f + z);
}

// cp.async helpers via inline PTX
__device__ __forceinline__ void cp_async_4b(void* smem, const void* gmem) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

// ---------------------------------------------------------------------------
// Decode kernel (unchanged)
// ---------------------------------------------------------------------------
__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_in,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_in,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    int64_t batch_size)
{
    const int bid = blockIdx.x;
    const int batch_idx = bid / kNumVHeads;
    const int v_head = bid % kNumVHeads;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const int qk_head = v_head / (kNumVHeads / kNumQHeads);
    const int64_t q_off  = (batch_idx * kNumQHeads + qk_head) * kHeadSize;
    const int64_t k_off  = (batch_idx * kNumKHeads + qk_head) * kHeadSize;
    const int64_t v_off  = (batch_idx * kNumVHeads + v_head) * kHeadSize;
    const int64_t g_off  = batch_idx * kNumVHeads + v_head;
    const int64_t st_off = ((int64_t)(batch_idx * kNumVHeads + v_head) * kHeadSize + tid) * kHeadSize;

    __shared__ __align__(16) float s_q[kHeadSize];
    __shared__ __align__(16) float s_k[kHeadSize];
    __shared__ float s_g, s_beta;

    s_q[tid] = __bfloat162float(q[q_off + tid]);
    s_k[tid] = __bfloat162float(k[k_off + tid]);
    if (tid == 0) {
        const float x = __bfloat162float(a_in[g_off]) + dt_bias[v_head];
        s_g = expf(-expf(A_log[v_head]) * softplusf_stable(x));
        s_beta = sigmoidf_stable(__bfloat162float(b_in[g_off]));
    }
    __syncthreads();

    const float g = s_g;
    const float beta = s_beta;
    const auto* st_vec = reinterpret_cast<const float4*>(state + st_off);
    auto* ns_vec = reinterpret_cast<float4*>(new_state + st_off);

    float4 sr[kVecsPerRow];
    float old_v = 0.0f;

    #pragma unroll
    for (int i = 0; i < kVecsPerRow; ++i) {
        float4 tmp = st_vec[i];
        const int b4 = i * 4;
        tmp.x *= g; tmp.y *= g; tmp.z *= g; tmp.w *= g;
        sr[i] = tmp;
        old_v += s_k[b4+0]*tmp.x + s_k[b4+1]*tmp.y + s_k[b4+2]*tmp.z + s_k[b4+3]*tmp.w;
    }

    const float delta = beta * (__bfloat162float(v[v_off + tid]) - old_v);

    float out_acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < kVecsPerRow; ++i) {
        const int b4 = i * 4;
        sr[i].x += s_k[b4+0]*delta; sr[i].y += s_k[b4+1]*delta;
        sr[i].z += s_k[b4+2]*delta; sr[i].w += s_k[b4+3]*delta;
        ns_vec[i] = sr[i];
        out_acc += s_q[b4+0]*sr[i].x + s_q[b4+1]*sr[i].y + s_q[b4+2]*sr[i].z + s_q[b4+3]*sr[i].w;
    }

    output[(batch_idx * kNumVHeads + v_head) * kHeadSize + tid] = __float2bfloat16(scale * out_acc);
}

// ---------------------------------------------------------------------------
// Prefill kernel — cp.async double-buffered bf16 pipeline
// ---------------------------------------------------------------------------
__global__ void gdn_prefill_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ state,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a_in,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b_in,
    const int64_t* __restrict__ cu_seqlens,
    float scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ new_state,
    int64_t num_seqs)
{
    const int bid = blockIdx.x;
    const int seq_idx = bid / kNumVHeads;
    const int v_head = bid % kNumVHeads;
    const int tid = threadIdx.x;

    if (seq_idx >= num_seqs) return;

    const int64_t seq_start = cu_seqlens[seq_idx];
    const int64_t seq_end   = cu_seqlens[seq_idx + 1];
    if (seq_end <= seq_start) return;

    const int qk_head = v_head / (kNumVHeads / kNumQHeads);
    const int64_t st_off = ((int64_t)(seq_idx * kNumVHeads + v_head) * kHeadSize + tid) * kHeadSize;

    const float A_log_val = A_log[v_head];
    const float dt_bias_val = dt_bias[v_head];

    // Load initial state
    float4 sr[kVecsPerRow];
    {
        const auto* sv = reinterpret_cast<const float4*>(state + st_off);
        #pragma unroll
        for (int i = 0; i < kVecsPerRow; ++i)
            sr[i] = sv[i];
    }

    // Double-buffered bf16 shared memory (1KB total)
    __shared__ __align__(16) __nv_bfloat16 s_k_bf16[2][kHeadSize];
    __shared__ __align__(16) __nv_bfloat16 s_q_bf16[2][kHeadSize];

    int buf = 0;

    // Async preload first token: tid<64 copies k (4B each), tid>=64 copies q
    {
        const int64_t k0 = (seq_start * kNumKHeads + qk_head) * kHeadSize;
        const int64_t q0 = (seq_start * kNumQHeads + qk_head) * kHeadSize;
        if (tid < 64) {
            cp_async_4b(&s_k_bf16[0][tid * 2], &k[k0 + tid * 2]);
        } else {
            const int qtid = tid - 64;
            cp_async_4b(&s_q_bf16[0][qtid * 2], &q[q0 + qtid * 2]);
        }
        cp_async_commit();
        cp_async_wait_all();
    }
    __syncthreads();

    for (int64_t t = seq_start; t < seq_end; ++t) {
        // Async prefetch next token into alternate buffer
        if (t + 1 < seq_end) {
            const int nb = 1 - buf;
            const int64_t nk = ((t + 1) * kNumKHeads + qk_head) * kHeadSize;
            const int64_t nq = ((t + 1) * kNumQHeads + qk_head) * kHeadSize;
            if (tid < 64) {
                cp_async_4b(&s_k_bf16[nb][tid * 2], &k[nk + tid * 2]);
            } else {
                const int qtid = tid - 64;
                cp_async_4b(&s_q_bf16[nb][qtid * 2], &q[nq + qtid * 2]);
            }
            cp_async_commit();
        }

        // Compute current token using buf
        const int64_t g_off = t * kNumVHeads + v_head;
        const float x = __bfloat162float(a_in[g_off]) + dt_bias_val;
        const float g = expf(-expf(A_log_val) * softplusf_stable(x));
        const float beta = sigmoidf_stable(__bfloat162float(b_in[g_off]));

        const __nv_bfloat16* ck = s_k_bf16[buf];
        const __nv_bfloat16* cq = s_q_bf16[buf];

        float old_v = 0.0f;
        #pragma unroll
        for (int i = 0; i < kVecsPerRow; ++i) {
            const int b4 = i * 4;
            const float4 prev = sr[i];
            old_v += __bfloat162float(ck[b4+0]) * (g * prev.x);
            old_v += __bfloat162float(ck[b4+1]) * (g * prev.y);
            old_v += __bfloat162float(ck[b4+2]) * (g * prev.z);
            old_v += __bfloat162float(ck[b4+3]) * (g * prev.w);
        }

        const float delta = beta * (__bfloat162float(v[(t * kNumVHeads + v_head) * kHeadSize + tid]) - old_v);

        float out_acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < kVecsPerRow; ++i) {
            const int b4 = i * 4;
            float4 upd;
            upd.x = g * sr[i].x + __bfloat162float(ck[b4+0]) * delta;
            upd.y = g * sr[i].y + __bfloat162float(ck[b4+1]) * delta;
            upd.z = g * sr[i].z + __bfloat162float(ck[b4+2]) * delta;
            upd.w = g * sr[i].w + __bfloat162float(ck[b4+3]) * delta;
            sr[i] = upd;
            out_acc += __bfloat162float(cq[b4+0]) * upd.x;
            out_acc += __bfloat162float(cq[b4+1]) * upd.y;
            out_acc += __bfloat162float(cq[b4+2]) * upd.z;
            out_acc += __bfloat162float(cq[b4+3]) * upd.w;
        }

        output[(t * kNumVHeads + v_head) * kHeadSize + tid] = __float2bfloat16(scale * out_acc);

        // Wait for async copy, sync block, swap buffers
        cp_async_wait_all();
        __syncthreads();
        buf = 1 - buf;
    }

    // Write final state
    auto* ns_vec = reinterpret_cast<float4*>(new_state + st_off);
    #pragma unroll
    for (int i = 0; i < kVecsPerRow; ++i)
        ns_vec[i] = sr[i];
}

// ---------------------------------------------------------------------------
// TVM FFI host functions
// ---------------------------------------------------------------------------

void GDNDecode(
    tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v,
    tvm::ffi::TensorView state, tvm::ffi::TensorView A_log, tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias, tvm::ffi::TensorView b,
    double scale, tvm::ffi::TensorView output, tvm::ffi::TensorView new_state)
{
    const int64_t bs = q.size(0);
    float sf = (scale == 0.0) ? (1.0f / sqrtf(128.0f)) : static_cast<float>(scale);
    DLDevice dev = q.device();
    cudaStream_t st = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
    gdn_decode_kernel<<<bs * kNumVHeads, kThreads, 0, st>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b.data_ptr()),
        sf, static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()), bs);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, GDNDecode);

void GDNPrefill(
    tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v,
    tvm::ffi::TensorView state, tvm::ffi::TensorView A_log, tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias, tvm::ffi::TensorView b,
    tvm::ffi::TensorView cu_seqlens, double scale,
    tvm::ffi::TensorView output, tvm::ffi::TensorView new_state)
{
    const int64_t ns = cu_seqlens.size(0) - 1;
    float sf = (scale == 0.0) ? (1.0f / sqrtf(128.0f)) : static_cast<float>(scale);
    DLDevice dev = q.device();
    cudaStream_t st = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
    gdn_prefill_kernel<<<ns * kNumVHeads, kThreads, 0, st>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b.data_ptr()),
        static_cast<const int64_t*>(cu_seqlens.data_ptr()),
        sf, static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()), ns);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_prefill, GDNPrefill);

}  // namespace
