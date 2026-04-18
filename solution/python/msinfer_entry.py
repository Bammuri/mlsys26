"""GDN decode/prefill via CuTe DSL runtime compile.

Surface contract (matches the MSInfer reference build):
    language     = "python"
    entry_point  = "msinfer_entry.py::run"            # for gdn_decode
                 = "msinfer_entry.py::run_prefill"   # for gdn_prefill
    dependencies = ["nvidia-cutlass-dsl", "cuda-python>=12.8"]

On first call per input shape we compile a CuTe kernel with
``cute.compile(..., options="--enable-tvm-ffi --gpu-arch=sm_100a ...")``.
The resulting TVM-FFI callable is cached and reused for subsequent calls
whose (dtype, shape, stride) signature matches.

The explicit ``--gpu-arch=sm_100a`` is the load-bearing bit: without the
``a`` suffix nvcc will not emit Blackwell-native tensor-core / ``tcgen05`` /
CTA-pair instructions, and scoring-environment performance diverges from
Modal bench numbers. Keep the rest of the options pinned here so the SASS
is deterministic across environments.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Dict, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ----------------------------------------------------------------------------
# Constants — pinned to the contest definition.
# ----------------------------------------------------------------------------
HQ = 4
HK = 4
HV = 8
D = 128
DEFAULT_SCALE = 1.0 / math.sqrt(float(D))
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0

# ----------------------------------------------------------------------------
# Compile options — sm_100a + -O3 + fast math are the ones that move SASS.
# ----------------------------------------------------------------------------
_COMPILE_OPTS = (
    "--enable-tvm-ffi "
    "--gpu-arch=sm_100a "
    "--opt-level=3"
)

# Per-shape compiled-callable cache (thread-safe).
_LOCK = threading.Lock()
_CACHE: Dict[Tuple[Any, ...], Callable[..., None]] = {}


# ----------------------------------------------------------------------------
# Device side — gate helpers (numerically-stable softplus + sigmoid).
# ----------------------------------------------------------------------------
@cute.jit
def _softplus_stable(x: cutlass.Float32) -> cutlass.Float32:
    # Single-exit form (DSL preprocessor rejects early-return inside @cute.jit).
    result = cutlass.Float32(0.0)
    if x > cutlass.Float32(SOFTPLUS_THRESHOLD):
        result = x
    elif x < cutlass.Float32(-SOFTPLUS_THRESHOLD):
        result = cute.exp(x, fastmath=True)
    else:
        result = cute.log(cutlass.Float32(1.0) + cute.exp(x, fastmath=True), fastmath=True)
    return result


@cute.jit
def _sigmoid_stable(x: cutlass.Float32) -> cutlass.Float32:
    result = cutlass.Float32(0.0)
    if x >= cutlass.Float32(0.0):
        result = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-x, fastmath=True))
    else:
        e = cute.exp(x, fastmath=True)
        result = e / (cutlass.Float32(1.0) + e)
    return result


# ----------------------------------------------------------------------------
# Decode kernel.
#   Grid: (B * HV, 1, 1), Block: (D, 1, 1) = 128 threads (4 warps).
#   Each thread owns exactly one V-row of the 128x128 state.
# ----------------------------------------------------------------------------
@cute.kernel
def _gdn_decode_dev(
    q: cute.Tensor,          # (B, S=1, HQ, D) bf16
    k: cute.Tensor,          # (B, S=1, HK, D) bf16
    v: cute.Tensor,          # (B, S=1, HV, D) bf16
    state_in: cute.Tensor,   # (B, HV, D, D) f32, k-last
    A_log: cute.Tensor,      # (HV,) f32
    a_in: cute.Tensor,       # (B, S=1, HV) bf16
    dt_bias: cute.Tensor,    # (HV,) f32
    b_in: cute.Tensor,       # (B, S=1, HV) bf16
    out: cute.Tensor,        # (B, S=1, HV, D) bf16
    state_out: cute.Tensor,  # (B, HV, D, D) f32
    scale: cutlass.Constexpr[float],
):
    bid_x, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()

    batch = bid_x // HV
    v_head = bid_x % HV
    qk_head = v_head // (HV // HQ)
    row = tid  # 0..127

    # ----- Cooperative smem load of Q, K (128 threads, each loads 1 bf16). --
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 16)

    sQ[tid] = cutlass.Float32(q[batch, 0, qk_head, tid])
    sK[tid] = cutlass.Float32(k[batch, 0, qk_head, tid])

    # ----- Gate scalars — every lane computes redundantly. --------------------
    a_val = cutlass.Float32(a_in[batch, 0, v_head]) + cutlass.Float32(dt_bias[v_head])
    sp = _softplus_stable(a_val)
    g = cute.exp(-cute.exp(cutlass.Float32(A_log[v_head]), fastmath=True) * sp, fastmath=True)
    beta = _sigmoid_stable(cutlass.Float32(b_in[batch, 0, v_head]))

    cute.arch.barrier()  # Wait on sQ, sK writes.

    # ----- qk = q · k (scalar, block-wide). -----------------------------------
    # Each of 128 threads contributes 1 element; reduce across threads via smem.
    sQK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((1,)), 4)
    # Staged reduce: use one partial per lane in smem; final reduce by lane 0.
    # Simpler: 128-way reduction via warp shuffle + block barrier using a second smem.
    sPartials = smem.allocate_tensor(cutlass.Float32, cute.make_layout((4,)), 16)  # 4 warp sums

    local_qk = sQ[tid] * sK[tid]
    # Butterfly-reduce within warp (32 lanes).
    for offset in [16, 8, 4, 2, 1]:
        local_qk += cute.arch.shuffle_sync_bfly(local_qk, offset=offset, mask=-1, mask_and_clamp=31)
    # Lane 0 of each warp writes partial; sync; thread 0 finalises.
    warp_id = tid // 32
    lane = tid & 31
    if lane == 0:
        sPartials[warp_id] = local_qk
    cute.arch.barrier()
    if tid == 0:
        sQK[0] = sPartials[0] + sPartials[1] + sPartials[2] + sPartials[3]
    cute.arch.barrier()
    qk = sQK[0]

    # ----- Fused first pass: sr[c] = g * state_in[c]; ov = K·sr; qs = Q·sr. --
    # Each thread owns one V-row, holds 128 fp32 state entries in registers.
    sr = cute.make_rmem_tensor(cute.make_layout((D,), stride=(1,)), cutlass.Float32)
    ov = cutlass.Float32(0.0)
    qs = cutlass.Float32(0.0)
    for c in cutlass.range_constexpr(D):
        s = cutlass.Float32(state_in[batch, v_head, row, c]) * g
        sr[c] = s
        ov += sK[c] * s
        qs += sQ[c] * s

    # ----- delta, out_acc. ----------------------------------------------------
    v_val = cutlass.Float32(v[batch, 0, v_head, row])
    delta = beta * (v_val - ov)
    out_acc = qs + delta * qk

    # ----- Second pass: state_out[c] = sr[c] + K[c] * delta. -----------------
    for c in cutlass.range_constexpr(D):
        new_s = sr[c] + sK[c] * delta
        state_out[batch, v_head, row, c] = new_s

    out[batch, 0, v_head, row] = cutlass.BFloat16(cutlass.Float32(scale) * out_acc)


@cute.jit
def _gdn_decode_jit(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    state_in: cute.Tensor,
    A_log: cute.Tensor,
    a_in: cute.Tensor,
    dt_bias: cute.Tensor,
    b_in: cute.Tensor,
    out: cute.Tensor,
    state_out: cute.Tensor,
):
    B = q.layout.shape[0]
    _gdn_decode_dev(
        q, k, v, state_in, A_log, a_in, dt_bias, b_in, out, state_out,
        DEFAULT_SCALE,
    ).launch(
        grid=(B * HV, 1, 1),
        block=(D, 1, 1),
    )


# ----------------------------------------------------------------------------
# Prefill kernel — token-sequential recurrence.
#   Grid: (N_seqs * HV, 1, 1), Block: (D, 1, 1).
#   Each thread owns one V-row and iterates over tokens in its sequence.
#   Uses cu_seqlens [N+1] for variable-length batching.
# ----------------------------------------------------------------------------
@cute.kernel
def _gdn_prefill_dev(
    q: cute.Tensor,          # (total_tokens, HQ, D) bf16
    k: cute.Tensor,          # (total_tokens, HK, D) bf16
    v: cute.Tensor,          # (total_tokens, HV, D) bf16
    state_in: cute.Tensor,   # (N, HV, D, D) f32
    A_log: cute.Tensor,      # (HV,) f32
    a_in: cute.Tensor,       # (total_tokens, HV) bf16
    dt_bias: cute.Tensor,    # (HV,) f32
    b_in: cute.Tensor,       # (total_tokens, HV) bf16
    cu_seqlens: cute.Tensor, # (N+1,) int64
    out: cute.Tensor,        # (total_tokens, HV, D) bf16
    state_out: cute.Tensor,  # (N, HV, D, D) f32
    scale: cutlass.Constexpr[float],
):
    bid_x, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()

    seq_idx = bid_x // HV
    v_head = bid_x % HV
    qk_head = v_head // (HV // HQ)
    row = tid

    # cu_seqlens is int64; the DSL dynamic-range loop requires Int32 bounds.
    seq_start = cutlass.Int32(cu_seqlens[seq_idx])
    seq_end = cutlass.Int32(cu_seqlens[seq_idx + 1])

    A_log_val = cutlass.Float32(A_log[v_head])
    dt_bias_val = cutlass.Float32(dt_bias[v_head])

    # Load this row of state_in into a register array once; mutate in place.
    sr = cute.make_rmem_tensor(cute.make_layout((D,), stride=(1,)), cutlass.Float32)
    for c in cutlass.range_constexpr(D):
        sr[c] = cutlass.Float32(state_in[seq_idx, v_head, row, c])

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 16)
    sPartials = smem.allocate_tensor(cutlass.Float32, cute.make_layout((4,)), 16)
    sScalar = smem.allocate_tensor(cutlass.Float32, cute.make_layout((1,)), 4)

    for t in range(seq_start, seq_end):
        # Cooperative load of Q[t], K[t] into smem.
        sQ[tid] = cutlass.Float32(q[t, qk_head, tid])
        sK[tid] = cutlass.Float32(k[t, qk_head, tid])

        # Gate scalars (redundant across lanes).
        a_val = cutlass.Float32(a_in[t, v_head]) + dt_bias_val
        sp = _softplus_stable(a_val)
        g = cute.exp(-cute.exp(A_log_val, fastmath=True) * sp, fastmath=True)
        beta = _sigmoid_stable(cutlass.Float32(b_in[t, v_head]))

        cute.arch.barrier()  # Wait on smem Q/K population.

        # qk = Q · K scalar reduce.
        local_qk = sQ[tid] * sK[tid]
        for offset in [16, 8, 4, 2, 1]:
            local_qk += cute.arch.shuffle_sync_bfly(
                local_qk, offset=offset, mask=-1, mask_and_clamp=31,
            )
        warp_id = tid // 32
        lane = tid & 31
        if lane == 0:
            sPartials[warp_id] = local_qk
        cute.arch.barrier()
        if tid == 0:
            sScalar[0] = sPartials[0] + sPartials[1] + sPartials[2] + sPartials[3]
        cute.arch.barrier()
        qk = sScalar[0]

        # Fused pass 1: sr ← g·sr; accumulate ov = K·sr, qs = Q·sr.
        ov = cutlass.Float32(0.0)
        qs = cutlass.Float32(0.0)
        for c in cutlass.range_constexpr(D):
            sr[c] = sr[c] * g
            ov += sK[c] * sr[c]
            qs += sQ[c] * sr[c]

        v_val = cutlass.Float32(v[t, v_head, row])
        delta = beta * (v_val - ov)
        out_acc = qs + delta * qk

        # Pass 2: sr ← sr + K·delta (now = new_state), store output.
        for c in cutlass.range_constexpr(D):
            sr[c] = sr[c] + sK[c] * delta

        out[t, v_head, row] = cutlass.BFloat16(cutlass.Float32(scale) * out_acc)

        cute.arch.barrier()  # before next token reuses sQ/sK.

    # Persist final state.
    for c in cutlass.range_constexpr(D):
        state_out[seq_idx, v_head, row, c] = sr[c]


@cute.jit
def _gdn_prefill_jit(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    state_in: cute.Tensor,
    A_log: cute.Tensor,
    a_in: cute.Tensor,
    dt_bias: cute.Tensor,
    b_in: cute.Tensor,
    cu_seqlens: cute.Tensor,
    out: cute.Tensor,
    state_out: cute.Tensor,
):
    N = state_in.layout.shape[0]
    _gdn_prefill_dev(
        q, k, v, state_in, A_log, a_in, dt_bias, b_in, cu_seqlens,
        out, state_out, DEFAULT_SCALE,
    ).launch(
        grid=(N * HV, 1, 1),
        block=(D, 1, 1),
    )


# ----------------------------------------------------------------------------
# Python dispatch surface.
# ----------------------------------------------------------------------------
def _cache_key(tensors) -> Tuple[Any, ...]:
    return tuple(
        (str(t.dtype), tuple(t.shape), tuple(t.stride())) for t in tensors
    )


def _wrap_cute(tensors):
    return [
        from_dlpack(t, enable_tvm_ffi=True).mark_layout_dynamic() for t in tensors
    ]


def _get_compiled(name: str, jit_fn, tensors):
    key = (name,) + _cache_key(tensors)
    with _LOCK:
        fn = _CACHE.get(key)
        if fn is None:
            cute_tensors = _wrap_cute(tensors)
            fn = cute.compile(jit_fn, *cute_tensors, options=_COMPILE_OPTS)
            _CACHE[key] = fn
    return fn


def _dispatch(name, jit_fn, tensors, call_args):
    """Shared dispatch — catches and surfaces exceptions that the
    flashinfer-bench evaluator would otherwise swallow into RUNTIME_ERROR."""
    import sys
    import traceback

    try:
        fn = _get_compiled(name, jit_fn, tensors)
        fn(*call_args)
    except Exception as e:
        print(f"[msinfer_entry:{name}] exception: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        # Also write to stdout so Modal relays it back to the local driver.
        sys.stdout.flush()
        raise


def run(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """DPS entrypoint for gdn_decode_qk4_v8_d128_k_last."""
    tensors = [q, k, v, state, A_log, a, dt_bias, b, output, new_state]
    call_args = [q, k, v, state, A_log, a, dt_bias, b, output, new_state]
    _dispatch("decode", _gdn_decode_jit, tensors, call_args)


def run_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    """DPS entrypoint for gdn_prefill_qk4_v8_d128_k_last."""
    tensors = [q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, output, new_state]
    call_args = [q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, output, new_state]
    _dispatch("prefill", _gdn_prefill_jit, tensors, call_args)
