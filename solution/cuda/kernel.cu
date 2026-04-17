/*
 * GDN decode + prefill CUDA kernels with TVM FFI binding.
 * v16: v15 minus streaming store — restore default cache write policy.
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

__device__ __forceinline__ void cp_async_4b(void* smem, const void* gmem) {
    unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

// Streaming store (write-once, no L1 alloc) for state and output
__device__ __forceinline__ void st_global_cs_v4(float4* addr, float4 v) {
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
}

// ---------------------------------------------------------------------------
// Decode kernel — v13: V-dim split (2 blocks per work item, 64 threads each)
// Block layout: bid encodes (batch, v_head, split). split=0 → rows 0-63, split=1 → 64-127.
// ---------------------------------------------------------------------------
constexpr int kSplits = 4;
constexpr int kThreadsV13 = 32;
constexpr int kRowsPerBlock = 32;

__global__
void gdn_decode_kernel(
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
    const int batch_v_split = bid;
    const int split = batch_v_split % kSplits;
    const int v_idx = (batch_v_split / kSplits) % kNumVHeads;
    const int batch_idx = batch_v_split / (kSplits * kNumVHeads);
    const int v_head = v_idx;
    const int tid = threadIdx.x;
    const int row = split * kRowsPerBlock + tid;  // V-dim row this thread owns

    if (batch_idx >= batch_size) return;

    const int qk_head = v_head / (kNumVHeads / kNumQHeads);
    const int64_t q_off  = (batch_idx * kNumQHeads + qk_head) * kHeadSize;
    const int64_t k_off  = (batch_idx * kNumKHeads + qk_head) * kHeadSize;
    const int64_t v_off  = (batch_idx * kNumVHeads + v_head) * kHeadSize;
    const int64_t g_off  = batch_idx * kNumVHeads + v_head;
    const int64_t st_off = ((int64_t)(batch_idx * kNumVHeads + v_head) * kHeadSize + row) * kHeadSize;

    __shared__ __align__(16) float s_q[kHeadSize];
    __shared__ __align__(16) float s_k[kHeadSize];
    __shared__ float s_g, s_beta, s_qk;

    // 32 threads cooperatively load 128 elements (each loads 4)
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        s_q[tid + j*32] = __bfloat162float(q[q_off + tid + j*32]);
        s_k[tid + j*32] = __bfloat162float(k[k_off + tid + j*32]);
    }

    if (tid == 0) {
        const float x = __bfloat162float(a_in[g_off]) + dt_bias[v_head];
        s_g = expf(-expf(A_log[v_head]) * softplusf_stable(x));
        s_beta = sigmoidf_stable(__bfloat162float(b_in[g_off]));
    }
    __syncthreads();

    // Compute qk = Q^T @ K via warp reduction (use the only warp = 32 threads)
    {
        float qk = 0.0f;
        #pragma unroll
        for (int i = tid; i < kHeadSize; i += 32) {
            qk += s_q[i] * s_k[i];
        }
        qk += __shfl_xor_sync(0xffffffff, qk, 16);
        qk += __shfl_xor_sync(0xffffffff, qk, 8);
        qk += __shfl_xor_sync(0xffffffff, qk, 4);
        qk += __shfl_xor_sync(0xffffffff, qk, 2);
        qk += __shfl_xor_sync(0xffffffff, qk, 1);
        if (tid == 0) s_qk = qk;
    }
    __syncthreads();

    const float g = s_g;
    const float beta = s_beta;
    const float qk = s_qk;
    const auto* st_vec = reinterpret_cast<const float4*>(state + st_off);
    auto* ns_vec = reinterpret_cast<float4*>(new_state + st_off);

    float4 sr[kVecsPerRow];
    float ov0 = 0.0f, ov1 = 0.0f, ov2 = 0.0f, ov3 = 0.0f;
    float qs0 = 0.0f, qs1 = 0.0f, qs2 = 0.0f, qs3 = 0.0f;

    #pragma unroll
    for (int i = 0; i < kVecsPerRow; ++i) {
        float4 tmp = __ldg(&st_vec[i]);
        const int b4 = i * 4;
        tmp.x *= g; tmp.y *= g; tmp.z *= g; tmp.w *= g;
        sr[i] = tmp;
        ov0 += s_k[b4+0]*tmp.x; ov1 += s_k[b4+1]*tmp.y;
        ov2 += s_k[b4+2]*tmp.z; ov3 += s_k[b4+3]*tmp.w;
        qs0 += s_q[b4+0]*tmp.x; qs1 += s_q[b4+1]*tmp.y;
        qs2 += s_q[b4+2]*tmp.z; qs3 += s_q[b4+3]*tmp.w;
    }

    const float old_v = ov0 + ov1 + ov2 + ov3;
    const float qs = qs0 + qs1 + qs2 + qs3;
    // V is loaded per-row from global (no smem)
    const float delta = beta * (__bfloat162float(v[v_off + row]) - old_v);
    const float out_acc = qs + delta * qk;

    #pragma unroll
    for (int i = 0; i < kVecsPerRow; ++i) {
        const int b4 = i * 4;
        sr[i].x += s_k[b4+0]*delta; sr[i].y += s_k[b4+1]*delta;
        sr[i].z += s_k[b4+2]*delta; sr[i].w += s_k[b4+3]*delta;
        ns_vec[i] = sr[i];
    }

    output[(batch_idx * kNumVHeads + v_head) * kHeadSize + row] =
        __float2bfloat16(scale * out_acc);
}

// ---------------------------------------------------------------------------
// Prefill kernel — v2: 4-way V-dim split, 32 threads per block (1 warp).
// 16-byte cp.async per thread loads K + Q for each token.
// ---------------------------------------------------------------------------
constexpr int kPrefillSplits = 4;
constexpr int kPrefillThreads = 32;
constexpr int kPrefillRowsPerBlock = kHeadSize / kPrefillSplits;  // 32

__device__ __forceinline__ void cp_async_16b(void* smem, const void* gmem) {
    unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem));
}

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
    const int split = bid % kPrefillSplits;
    const int v_head = (bid / kPrefillSplits) % kNumVHeads;
    const int seq_idx = bid / (kPrefillSplits * kNumVHeads);
    const int tid = threadIdx.x;
    const int row = split * kPrefillRowsPerBlock + tid;  // V-dim row this thread owns

    if (seq_idx >= num_seqs) return;

    const int64_t seq_start = cu_seqlens[seq_idx];
    const int64_t seq_end   = cu_seqlens[seq_idx + 1];
    if (seq_end <= seq_start) return;

    const int qk_head = v_head / (kNumVHeads / kNumQHeads);
    const int64_t st_off = ((int64_t)(seq_idx * kNumVHeads + v_head) * kHeadSize + row) * kHeadSize;

    const float A_log_val = A_log[v_head];
    const float dt_bias_val = dt_bias[v_head];

    float4 sr[kVecsPerRow];
    {
        const auto* sv = reinterpret_cast<const float4*>(state + st_off);
        #pragma unroll
        for (int i = 0; i < kVecsPerRow; ++i)
            sr[i] = __ldg(&sv[i]);
    }

    __shared__ __align__(16) __nv_bfloat16 s_k_bf16[2][kHeadSize];
    __shared__ __align__(16) __nv_bfloat16 s_q_bf16[2][kHeadSize];

    int buf = 0;

    // 32 threads × 16 bytes = 512 bytes = 128 bf16 K + 128 bf16 Q.
    // tid 0-15: K (each loads 8 bf16). tid 16-31: Q (each loads 8 bf16).
    {
        const int64_t k0 = (seq_start * kNumKHeads + qk_head) * kHeadSize;
        const int64_t q0 = (seq_start * kNumQHeads + qk_head) * kHeadSize;
        if (tid < 16) {
            cp_async_16b(&s_k_bf16[0][tid * 8], &k[k0 + tid * 8]);
        } else {
            const int qi = tid - 16;
            cp_async_16b(&s_q_bf16[0][qi * 8], &q[q0 + qi * 8]);
        }
        cp_async_commit();
        cp_async_wait_all();
    }
    __syncwarp();

    for (int64_t t = seq_start; t < seq_end; ++t) {
        if (t + 1 < seq_end) {
            const int nb = 1 - buf;
            const int64_t nk = ((t + 1) * kNumKHeads + qk_head) * kHeadSize;
            const int64_t nq = ((t + 1) * kNumQHeads + qk_head) * kHeadSize;
            if (tid < 16) {
                cp_async_16b(&s_k_bf16[nb][tid * 8], &k[nk + tid * 8]);
            } else {
                const int qi = tid - 16;
                cp_async_16b(&s_q_bf16[nb][qi * 8], &q[nq + qi * 8]);
            }
            cp_async_commit();
        }

        const int64_t g_off = t * kNumVHeads + v_head;
        const float x = __bfloat162float(a_in[g_off]) + dt_bias_val;
        const float g = expf(-expf(A_log_val) * softplusf_stable(x));
        const float beta = sigmoidf_stable(__bfloat162float(b_in[g_off]));

        const __nv_bfloat16* ck = s_k_bf16[buf];
        const __nv_bfloat16* cq = s_q_bf16[buf];

        float ov0 = 0.0f, ov1 = 0.0f, ov2 = 0.0f, ov3 = 0.0f;
        #pragma unroll
        for (int i = 0; i < kVecsPerRow; ++i) {
            const int b4 = i * 4;
            const float4 prev = sr[i];
            ov0 += __bfloat162float(ck[b4+0]) * (g * prev.x);
            ov1 += __bfloat162float(ck[b4+1]) * (g * prev.y);
            ov2 += __bfloat162float(ck[b4+2]) * (g * prev.z);
            ov3 += __bfloat162float(ck[b4+3]) * (g * prev.w);
        }
        const float old_v = ov0 + ov1 + ov2 + ov3;

        const float delta = beta * (__bfloat162float(v[(t * kNumVHeads + v_head) * kHeadSize + row]) - old_v);

        float oa0 = 0.0f, oa1 = 0.0f, oa2 = 0.0f, oa3 = 0.0f;
        #pragma unroll
        for (int i = 0; i < kVecsPerRow; ++i) {
            const int b4 = i * 4;
            float4 upd;
            upd.x = g * sr[i].x + __bfloat162float(ck[b4+0]) * delta;
            upd.y = g * sr[i].y + __bfloat162float(ck[b4+1]) * delta;
            upd.z = g * sr[i].z + __bfloat162float(ck[b4+2]) * delta;
            upd.w = g * sr[i].w + __bfloat162float(ck[b4+3]) * delta;
            sr[i] = upd;
            oa0 += __bfloat162float(cq[b4+0]) * upd.x;
            oa1 += __bfloat162float(cq[b4+1]) * upd.y;
            oa2 += __bfloat162float(cq[b4+2]) * upd.z;
            oa3 += __bfloat162float(cq[b4+3]) * upd.w;
        }
        const float out_acc = oa0 + oa1 + oa2 + oa3;

        output[(t * kNumVHeads + v_head) * kHeadSize + row] = __float2bfloat16(scale * out_acc);

        cp_async_wait_all();
        __syncwarp();
        buf = 1 - buf;
    }

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
    gdn_decode_kernel<<<bs * kNumVHeads * kSplits, kThreadsV13, 0, st>>>(
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
    gdn_prefill_kernel<<<ns * kNumVHeads * kPrefillSplits, kPrefillThreads, 0, st>>>(
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
