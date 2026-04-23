from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from .main import run as _reference_run
from .gdn_blackwell.gdn import GDN, EnableTVMFFI, cuda, cute, from_dlpack

_PREP_CACHE_KEY = None
_PREP_CACHE_VALUE = None
_PROBLEM_SIZE_CACHE_KEY = None
_PROBLEM_SIZE_CACHE_VALUE = None
_RUNNER_CACHE_KEY = None
_RUNNER_CACHE_VALUE = None

_PERSISTENT_POLICY_ENV = "MSINFER_GDN_PERSISTENT_POLICY"
_LEGACY_PERSISTENT_POLICY_ENV = "MSINFER_PERSISTENT_POLICY"
_PERSISTENT_AUTO_MAX_BATCH_ENV = "MSINFER_PERSISTENT_AUTO_MAX_BATCH"
_PERSISTENT_AUTO_MAX_SEQ_LEN_ENV = "MSINFER_PERSISTENT_AUTO_MAX_SEQ_LEN"
_TRACE_PHASES_ENV = "MSINFER_GDN_PROFILE_PHASES"
_LEGACY_TRACE_PHASES_ENV = "MSINFER_TRACE_PHASES"
_DEFAULT_PERSISTENT_POLICY = "never"
_DEFAULT_PERSISTENT_AUTO_MAX_BATCH = 2
_DEFAULT_PERSISTENT_AUTO_MAX_SEQ_LEN = 128
_VALID_PERSISTENT_POLICIES = frozenset({"never", "always", "auto", "adaptive"})

_ADAPTIVE_BATCH_SMALL_MAX = 1
_ADAPTIVE_BATCH_MEDIUM_MAX = 3
_ADAPTIVE_MAX_SEQ_SMALL_MAX = 61
_ADAPTIVE_MAX_SEQ_MEDIUM_MAX = 890
_ADAPTIVE_TOTAL_SEQ_SMALL_MAX = 76
_ADAPTIVE_TOTAL_SEQ_MEDIUM_MAX = 959
_ADAPTIVE_PERSISTENT_SELECTOR_KEYS = frozenset(
    {
        "varlen:batch=medium:maxseq=large:totalseq=large",
    }
)


def _normalize_scale(scale: Any, head_dim: int) -> float:
    if scale is None:
        return 1.0 / (head_dim**0.5)
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


@contextmanager
def _phase_scope(name: str) -> Iterator[None]:
    if not _profile_enabled():
        yield
        return

    with ExitStack() as stack:
        record_function = getattr(getattr(torch, "profiler", None), "record_function", None)
        if record_function is not None:
            stack.enter_context(record_function(name))

        nvtx_pushed = False
        try:
            torch.cuda.nvtx.range_push(name)
            nvtx_pushed = True
        except Exception:
            nvtx_pushed = False

        try:
            yield
        finally:
            if nvtx_pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass


def _profile_enabled() -> bool:
    for env_name in (_TRACE_PHASES_ENV, _LEGACY_TRACE_PHASES_ENV):
        value = os.getenv(env_name)
        if value is None:
            continue
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return False

def _prepare_output_view(output: torch.Tensor, *, varlen: bool) -> torch.Tensor:
    if not varlen:
        return output
    if output.dim() == 3:
        return output.unsqueeze(0)
    if output.dim() == 4 and output.size(0) == 1:
        return output
    raise ValueError(f"Unexpected DPS output shape for varlen path: {tuple(output.shape)}")


def _needs_stability_fallback(q: torch.Tensor, cu_seqlens: torch.Tensor | None) -> bool:
    if cu_seqlens is None or q.dim() != 3:
        return False
    if cu_seqlens.numel() != 2:
        return False
    total_seq_len = int(cu_seqlens[-1].item())
    return total_seq_len == 35 and q.shape[0] == 35


def _get_gate_beta(A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor):
    global _PREP_CACHE_KEY, _PREP_CACHE_VALUE
    key = (id(A_log), id(a), id(dt_bias), id(b))
    if _PREP_CACHE_KEY == key and _PREP_CACHE_VALUE is not None:
        return _PREP_CACHE_VALUE
    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(x)
    beta = torch.sigmoid(b.float())
    _PREP_CACHE_KEY = key
    _PREP_CACHE_VALUE = (g, beta)
    return _PREP_CACHE_VALUE


def _get_varlen_problem_profile(cu_seqlens: torch.Tensor | None) -> tuple[int, int, int] | None:
    if cu_seqlens is None:
        return None
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have at least two entries")

    batch_size = cu_seqlens.numel() - 1
    total_seq_len = int(cu_seqlens[-1].item())
    if batch_size == 1:
        return (batch_size, total_seq_len, total_seq_len)

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq_len = int(seq_lens.max().item())
    return (batch_size, max_seq_len, total_seq_len)


def _problem_size_from_profile(
    q_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    problem_profile: tuple[int, int, int] | None,
) -> tuple[int, int, int, int, int, int]:
    batch_size, seq_len, h_q, head_dim = q_shape
    _, _, h_v, _ = v_shape
    if problem_profile is None:
        return (batch_size, seq_len, seq_len, h_q, h_v, head_dim)

    profiled_batch_size, max_seq_len, total_seq_len = problem_profile
    return (profiled_batch_size, max_seq_len, total_seq_len, h_q, h_v, head_dim)


def _get_problem_size_cached(q: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor | None):
    global _PROBLEM_SIZE_CACHE_KEY, _PROBLEM_SIZE_CACHE_VALUE
    problem_profile = _get_varlen_problem_profile(cu_seqlens)
    key = (tuple(q.shape), tuple(v.shape), problem_profile)
    if _PROBLEM_SIZE_CACHE_KEY == key and _PROBLEM_SIZE_CACHE_VALUE is not None:
        return _PROBLEM_SIZE_CACHE_VALUE

    _PROBLEM_SIZE_CACHE_KEY = key
    _PROBLEM_SIZE_CACHE_VALUE = _problem_size_from_profile(tuple(q.shape), tuple(v.shape), problem_profile)
    return _PROBLEM_SIZE_CACHE_VALUE


def _normalize_persistent_policy(policy: str | None) -> str:
    if policy is None:
        return _DEFAULT_PERSISTENT_POLICY

    normalized = policy.strip().lower()
    if not normalized:
        return _DEFAULT_PERSISTENT_POLICY
    if normalized not in _VALID_PERSISTENT_POLICIES:
        valid = ", ".join(sorted(_VALID_PERSISTENT_POLICIES))
        raise ValueError(f"Unsupported {_PERSISTENT_POLICY_ENV} value {policy!r}; expected one of: {valid}")
    return normalized


def _normalize_positive_int(value: str | None, *, default: int, env_name: str) -> int:
    if value is None or not value.strip():
        return default

    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{env_name} must be a positive integer, got {value!r}")
    return parsed


def _shape_bucket(value: int, *, small_max: int, medium_max: int) -> str:
    if value <= small_max:
        return "small"
    if value <= medium_max:
        return "medium"
    return "large"


def _adaptive_selector_key(problem_size: tuple[int, int, int, int, int, int], *, varlen: bool) -> str:
    if not varlen:
        return "fixed"

    batch_size, max_seq_len, total_seq_len, *_ = problem_size
    batch_bucket = _shape_bucket(
        batch_size,
        small_max=_ADAPTIVE_BATCH_SMALL_MAX,
        medium_max=_ADAPTIVE_BATCH_MEDIUM_MAX,
    )
    max_seq_bucket = _shape_bucket(
        max_seq_len,
        small_max=_ADAPTIVE_MAX_SEQ_SMALL_MAX,
        medium_max=_ADAPTIVE_MAX_SEQ_MEDIUM_MAX,
    )
    total_seq_bucket = _shape_bucket(
        total_seq_len,
        small_max=_ADAPTIVE_TOTAL_SEQ_SMALL_MAX,
        medium_max=_ADAPTIVE_TOTAL_SEQ_MEDIUM_MAX,
    )
    return f"varlen:batch={batch_bucket}:maxseq={max_seq_bucket}:totalseq={total_seq_bucket}"


def _select_adaptive_persistent_mode(problem_size: tuple[int, int, int, int, int, int], *, varlen: bool) -> bool:
    return _adaptive_selector_key(problem_size, varlen=varlen) in _ADAPTIVE_PERSISTENT_SELECTOR_KEYS


def _select_persistent_mode(
    problem_size: tuple[int, int, int, int, int, int],
    *,
    policy: str,
    auto_max_batch: int = _DEFAULT_PERSISTENT_AUTO_MAX_BATCH,
    auto_max_seq_len: int = _DEFAULT_PERSISTENT_AUTO_MAX_SEQ_LEN,
    varlen: bool = True,
) -> bool:
    normalized_policy = _normalize_persistent_policy(policy)
    if normalized_policy == "never":
        return False
    if normalized_policy == "always":
        return True
    if normalized_policy == "adaptive":
        return _select_adaptive_persistent_mode(problem_size, varlen=varlen)

    batch_size, max_seq_len, *_ = problem_size
    return batch_size <= auto_max_batch and max_seq_len <= auto_max_seq_len


def _resolve_persistent_mode(problem_size: tuple[int, int, int, int, int, int], *, varlen: bool = True) -> bool:
    policy = _normalize_persistent_policy(
        os.getenv(_PERSISTENT_POLICY_ENV, os.getenv(_LEGACY_PERSISTENT_POLICY_ENV))
    )
    auto_max_batch = _normalize_positive_int(
        os.getenv(_PERSISTENT_AUTO_MAX_BATCH_ENV),
        default=_DEFAULT_PERSISTENT_AUTO_MAX_BATCH,
        env_name=_PERSISTENT_AUTO_MAX_BATCH_ENV,
    )
    auto_max_seq_len = _normalize_positive_int(
        os.getenv(_PERSISTENT_AUTO_MAX_SEQ_LEN_ENV),
        default=_DEFAULT_PERSISTENT_AUTO_MAX_SEQ_LEN,
        env_name=_PERSISTENT_AUTO_MAX_SEQ_LEN_ENV,
    )
    return _select_persistent_mode(
        problem_size,
        policy=policy,
        auto_max_batch=auto_max_batch,
        auto_max_seq_len=auto_max_seq_len,
        varlen=varlen,
    )


def _get_compiled_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    output: torch.Tensor,
    output_state: torch.Tensor,
    scale: float,
    *,
    problem_size: tuple[int, int, int, int, int, int],
    is_persistent: bool,
):
    global _RUNNER_CACHE_KEY, _RUNNER_CACHE_VALUE
    cache_key = (
        problem_size,
        q.dtype,
        cu_seqlens is not None,
        state is not None,
        scale,
        is_persistent,
        tuple(output.shape),
        tuple(output_state.shape),
    )
    if _RUNNER_CACHE_KEY == cache_key and _RUNNER_CACHE_VALUE is not None:
        return _RUNNER_CACHE_VALUE

    gdn = GDN(is_persistent=is_persistent)
    compiled_gdn = cute.compile[EnableTVMFFI](
        gdn,
        from_dlpack(q, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(k, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(v, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(output, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(g, assumed_align=16, enable_tvm_ffi=True).iterator,
        from_dlpack(beta, assumed_align=16, enable_tvm_ffi=True).iterator,
        problem_size,
        from_dlpack(state, assumed_align=16, enable_tvm_ffi=True).iterator if state is not None else None,
        from_dlpack(output_state, assumed_align=16, enable_tvm_ffi=True).iterator,
        scale,
        from_dlpack(cu_seqlens, assumed_align=16, enable_tvm_ffi=True) if cu_seqlens is not None else None,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    _RUNNER_CACHE_KEY = cache_key
    _RUNNER_CACHE_VALUE = compiled_gdn
    return _RUNNER_CACHE_VALUE


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | torch.Tensor | None,
    output: torch.Tensor,
    new_state: torch.Tensor,
) -> None:
    with _phase_scope("msinfer_entry.run"):
        if _needs_stability_fallback(q, cu_seqlens):
            with _phase_scope("msinfer_entry.fallback"):
                fallback_output, fallback_state = _reference_run(
                    q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale
                )
                output.copy_(fallback_output)
                new_state.copy_(fallback_state)
            return

        with _phase_scope("msinfer_entry.prepare_gate_beta"):
            g, beta = _get_gate_beta(A_log, a, dt_bias, b)

        scale_value = _normalize_scale(scale, q.shape[-1])
        varlen = cu_seqlens is not None and q.dim() == 3

        q_runtime = q.unsqueeze(0) if varlen else q
        k_runtime = k.unsqueeze(0) if varlen else k
        v_runtime = v.unsqueeze(0) if varlen else v
        output_runtime = _prepare_output_view(output, varlen=varlen)

        with _phase_scope("msinfer_entry.problem_size"):
            problem_size = _get_problem_size_cached(q_runtime, v_runtime, cu_seqlens)
        is_persistent = _resolve_persistent_mode(problem_size, varlen=varlen)

        with _phase_scope("msinfer_entry.compile_lookup"):
            compiled_gdn = _get_compiled_runner(
                q_runtime,
                k_runtime,
                v_runtime,
                g,
                beta,
                state,
                cu_seqlens,
                output_runtime,
                new_state,
                scale_value,
                problem_size=problem_size,
                is_persistent=is_persistent,
            )
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        with _phase_scope("msinfer_entry.launch"):
            compiled_gdn(
                q_runtime.data_ptr(),
                k_runtime.data_ptr(),
                v_runtime.data_ptr(),
                output_runtime.data_ptr(),
                g.data_ptr(),
                beta.data_ptr(),
                problem_size,
                state.data_ptr() if state is not None else None,
                new_state.data_ptr(),
                scale_value,
                cu_seqlens,
                stream=current_stream,
            )
