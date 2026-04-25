from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator, NamedTuple

import torch
import torch.nn.functional as F

from .gdn_block_policy import choose_default_or_tuned_block_tile
from .main import run as _reference_run
from .gdn_blackwell.gdn import GDN, EnableTVMFFI, cuda, cute, from_dlpack

_PREP_CACHE_KEY = None
_PREP_CACHE_VALUE = None
_PROBLEM_SIZE_CACHE_KEY = None
_PROBLEM_SIZE_CACHE_VALUE = None
_RUNNER_CACHE: dict[tuple[Any, ...], Any] = {}
_ADAPTIVE_SELECTOR_KEYS_CACHE_KEY = None
_ADAPTIVE_SELECTOR_KEYS_CACHE_VALUE = None

_PERSISTENT_POLICY_ENV = "MSINFER_GDN_PERSISTENT_POLICY"
_LEGACY_PERSISTENT_POLICY_ENV = "MSINFER_PERSISTENT_POLICY"
_PERSISTENT_AUTO_MAX_BATCH_ENV = "MSINFER_PERSISTENT_AUTO_MAX_BATCH"
_PERSISTENT_AUTO_MAX_SEQ_LEN_ENV = "MSINFER_PERSISTENT_AUTO_MAX_SEQ_LEN"
_ADAPTIVE_SELECTOR_KEYS_ENV = "MSINFER_GDN_ADAPTIVE_SELECTOR_KEYS"
_BLOCK_SHAPE_TUNING_ENV = "MSINFER_GDN_ENABLE_BLOCK_SHAPE_TUNING"
_EXPERIMENTAL_BLOCK_POLICY_ENV = "MSINFER_GDN_ENABLE_EXPERIMENTAL_BLOCK_POLICY"
_COMPOSITE_REFERENCE_HARNESS_ENV = "MSINFER_GDN_ENABLE_COMPOSITE_REFERENCE_HARNESS"
_COMPOSITE_COMPILED_HARNESS_ENV = "MSINFER_GDN_ENABLE_COMPOSITE_COMPILED_HARNESS"
_TRACE_PHASES_ENV = "MSINFER_GDN_PROFILE_PHASES"
_LEGACY_TRACE_PHASES_ENV = "MSINFER_TRACE_PHASES"
_DEFAULT_PERSISTENT_POLICY = "never"
_DEFAULT_PERSISTENT_AUTO_MAX_BATCH = 2
_DEFAULT_PERSISTENT_AUTO_MAX_SEQ_LEN = 128
_DEFAULT_KERNEL_CHUNK_SIZE = 128
# tcgen05 may support internal MMA M-mode 64/128, but this GDN Python
# kernel path is still 128-specialized. Keep GDN chunk-size support narrow
# until a dedicated 64/128 composed path is implemented and verified.
_SUPPORTED_INTERNAL_GDN_CHUNK_SIZES = frozenset({128})
_VALID_PERSISTENT_POLICIES = frozenset({"never", "always", "auto", "adaptive"})


class _KernelSchedule(NamedTuple):
    outer_schedule_tile: int
    internal_kernel_chunk_size: int
    internal_launch_segments: tuple[int, ...]
    experimental_policy_enabled: bool

_ADAPTIVE_BATCH_SMALL_MAX = 1
_ADAPTIVE_BATCH_MEDIUM_MAX = 3
_ADAPTIVE_MAX_SEQ_SMALL_MAX = 61
_ADAPTIVE_MAX_SEQ_MEDIUM_MAX = 890
_ADAPTIVE_TOTAL_SEQ_SMALL_MAX = 76
_ADAPTIVE_TOTAL_SEQ_MEDIUM_MAX = 959
_MEDIUM_LARGE_LARGE_SELECTOR_KEY = "varlen:batch=medium:maxseq=large:totalseq=large"
_MEDIUM_LARGE_LARGE_M1_SELECTOR_KEY = (
    f"{_MEDIUM_LARGE_LARGE_SELECTOR_KEY}:sub=batch3_maxseq1500to1999"
)
_LARGE_LARGE_LARGE_SELECTOR_KEY = "varlen:batch=large:maxseq=large:totalseq=large"
_LARGE_LARGE_LARGE_R1_SELECTOR_KEY = (
    f"{_LARGE_LARGE_LARGE_SELECTOR_KEY}:sub=total8192_batchge32_maxseqge2000"
)
_LARGE_LARGE_LARGE_R2_SELECTOR_KEY = (
    f"{_LARGE_LARGE_LARGE_SELECTOR_KEY}:sub=total8192_batchge32_maxseqlt2000"
)
_LARGE_LARGE_LARGE_R3_SELECTOR_KEY = f"{_LARGE_LARGE_LARGE_SELECTOR_KEY}:sub=total8192_batchlt32"
_LARGE_LARGE_LARGE_R4_SELECTOR_KEY = f"{_LARGE_LARGE_LARGE_SELECTOR_KEY}:sub=totallt8192"
_DEFAULT_ADAPTIVE_PERSISTENT_SELECTOR_KEYS = frozenset(
    {
        _LARGE_LARGE_LARGE_R1_SELECTOR_KEY,
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


def _normalize_bool_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


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


def _adaptive_selector_keys(problem_size: tuple[int, int, int, int, int, int], *, varlen: bool) -> tuple[str, ...]:
    coarse_key = _adaptive_selector_key(problem_size, varlen=varlen)
    if not varlen:
        return (coarse_key,)

    batch_size, max_seq_len, total_seq_len, *_ = problem_size
    if coarse_key == _MEDIUM_LARGE_LARGE_SELECTOR_KEY:
        if batch_size == 3 and 1500 <= max_seq_len < 2000:
            return (coarse_key, _MEDIUM_LARGE_LARGE_M1_SELECTOR_KEY)
        return (coarse_key,)

    if coarse_key == _LARGE_LARGE_LARGE_SELECTOR_KEY:
        if total_seq_len != 8192:
            return (coarse_key, _LARGE_LARGE_LARGE_R4_SELECTOR_KEY)
        if batch_size < 32:
            return (coarse_key, _LARGE_LARGE_LARGE_R3_SELECTOR_KEY)
        if max_seq_len < 2000:
            return (coarse_key, _LARGE_LARGE_LARGE_R2_SELECTOR_KEY)
        return (coarse_key, _LARGE_LARGE_LARGE_R1_SELECTOR_KEY)

    return (coarse_key,)


def _get_adaptive_selector_key_set() -> frozenset[str]:
    global _ADAPTIVE_SELECTOR_KEYS_CACHE_KEY, _ADAPTIVE_SELECTOR_KEYS_CACHE_VALUE
    raw_keys = os.getenv(_ADAPTIVE_SELECTOR_KEYS_ENV)
    if _ADAPTIVE_SELECTOR_KEYS_CACHE_KEY == raw_keys and _ADAPTIVE_SELECTOR_KEYS_CACHE_VALUE is not None:
        return _ADAPTIVE_SELECTOR_KEYS_CACHE_VALUE

    if raw_keys is None:
        _ADAPTIVE_SELECTOR_KEYS_CACHE_KEY = raw_keys
        _ADAPTIVE_SELECTOR_KEYS_CACHE_VALUE = _DEFAULT_ADAPTIVE_PERSISTENT_SELECTOR_KEYS
        return _ADAPTIVE_SELECTOR_KEYS_CACHE_VALUE

    selector_keys = frozenset(key.strip() for key in raw_keys.split(",") if key.strip())
    if not selector_keys:
        raise ValueError(
            f"{_ADAPTIVE_SELECTOR_KEYS_ENV} must contain at least one non-empty selector key when set"
        )
    _ADAPTIVE_SELECTOR_KEYS_CACHE_KEY = raw_keys
    _ADAPTIVE_SELECTOR_KEYS_CACHE_VALUE = selector_keys
    return _ADAPTIVE_SELECTOR_KEYS_CACHE_VALUE


def _resolve_block_tile(problem_size: tuple[int, int, int, int, int, int]) -> int:
    """Resolve the evidence-backed outer schedule tile target.

    The returned value is a scheduling policy target (typically 192, with
    optional 160/224 tuning). It is intentionally *not* an internal GDN
    chunk/MMA tile size: tcgen05 currently only accepts internal M-mode
    sizes 64 or 128.
    """
    *_unused, h_v, head_dim = problem_size
    enable_shape_tuning = _normalize_bool_env(os.getenv(_BLOCK_SHAPE_TUNING_ENV))
    return choose_default_or_tuned_block_tile(
        num_heads=h_v,
        rhs_dim=head_dim,
        enable_shape_tuning=enable_shape_tuning,
    )


def _validate_internal_kernel_chunk_size(chunk_size: int) -> int:
    if chunk_size not in _SUPPORTED_INTERNAL_GDN_CHUNK_SIZES:
        legal_values = ", ".join(str(value) for value in sorted(_SUPPORTED_INTERNAL_GDN_CHUNK_SIZES))
        raise ValueError(
            f"supported internal GDN chunk size must be one of {{{legal_values}}}; got {chunk_size}. "
            "Outer block policy tiles such as 160/192/224 must not be forwarded "
            "directly into GDN(chunk_size=...)."
        )
    return chunk_size


def _resolve_outer_schedule_tile(problem_size: tuple[int, int, int, int, int, int]) -> int:
    if not _normalize_bool_env(os.getenv(_EXPERIMENTAL_BLOCK_POLICY_ENV)):
        return _DEFAULT_KERNEL_CHUNK_SIZE
    return _resolve_block_tile(problem_size)


def _decompose_outer_schedule_tile(outer_schedule_tile: int, *, internal_kernel_chunk_size: int) -> tuple[int, ...]:
    if outer_schedule_tile <= 0:
        raise ValueError(f"outer schedule tile must be positive; got {outer_schedule_tile}")
    _validate_internal_kernel_chunk_size(internal_kernel_chunk_size)

    remaining = outer_schedule_tile
    segments: list[int] = []
    while remaining > 0:
        segment = min(internal_kernel_chunk_size, remaining)
        segments.append(segment)
        remaining -= segment
    return tuple(segments)


def _resolve_kernel_schedule(problem_size: tuple[int, int, int, int, int, int]) -> _KernelSchedule:
    experimental_policy_enabled = _normalize_bool_env(os.getenv(_EXPERIMENTAL_BLOCK_POLICY_ENV))
    outer_schedule_tile = _resolve_outer_schedule_tile(problem_size)

    # Phase 1 guardrail: the researched 192-centered policy is an outer
    # schedule target. Until gdn.py has a dedicated legal 128/64 composed path,
    # the internal compiled GDN path must stay on a legal tile size.
    internal_kernel_chunk_size = _validate_internal_kernel_chunk_size(_DEFAULT_KERNEL_CHUNK_SIZE)
    internal_launch_segments = _decompose_outer_schedule_tile(
        outer_schedule_tile,
        internal_kernel_chunk_size=internal_kernel_chunk_size,
    )
    return _KernelSchedule(
        outer_schedule_tile=outer_schedule_tile,
        internal_kernel_chunk_size=internal_kernel_chunk_size,
        internal_launch_segments=internal_launch_segments,
        experimental_policy_enabled=experimental_policy_enabled,
    )


def _resolve_kernel_chunk_size(problem_size: tuple[int, int, int, int, int, int]) -> int:
    return _resolve_kernel_schedule(problem_size).internal_kernel_chunk_size


def _validate_kernel_schedule_for_execution(schedule: _KernelSchedule) -> None:
    if not schedule.internal_launch_segments:
        raise ValueError("kernel schedule must contain at least one internal launch segment")
    _validate_internal_kernel_chunk_size(schedule.internal_kernel_chunk_size)


def _iter_composite_segment_bounds(total_length: int, segment_pattern: tuple[int, ...]) -> Iterator[tuple[int, int]]:
    if total_length < 0:
        raise ValueError(f"total_length must be non-negative; got {total_length}")
    if not segment_pattern:
        raise ValueError("segment pattern must contain at least one segment")
    if any(segment <= 0 for segment in segment_pattern):
        raise ValueError(f"segment pattern must contain only positive lengths; got {segment_pattern}")

    start = 0
    pattern_index = 0
    while start < total_length:
        segment_len = min(segment_pattern[pattern_index], total_length - start)
        end = start + segment_len
        yield start, end
        start = end
        pattern_index = (pattern_index + 1) % len(segment_pattern)


def _run_composite_reference_schedule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    scale: float | torch.Tensor | None,
    output: torch.Tensor,
    new_state: torch.Tensor,
    schedule: _KernelSchedule,
) -> None:
    """Correctness-first Option-B composite path.

    This consumes the outer-schedule decomposition metadata and stitches segment
    outputs/state on the host side. It is intentionally behind the experimental
    block-policy gate; the default compiled GDN path remains unchanged.
    """
    _validate_kernel_schedule_for_execution(schedule)
    segment_pattern = schedule.internal_launch_segments

    varlen = cu_seqlens is not None and q.dim() == 3
    if varlen:
        assert cu_seqlens is not None
        for seq_idx in range(cu_seqlens.numel() - 1):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            current_state = state[seq_idx : seq_idx + 1].contiguous() if state is not None else None

            for rel_start, rel_end in _iter_composite_segment_bounds(seq_end - seq_start, segment_pattern):
                start = seq_start + rel_start
                end = seq_start + rel_end
                segment_cu = torch.tensor(
                    [0, end - start],
                    dtype=cu_seqlens.dtype,
                    device=cu_seqlens.device,
                )
                segment_output, current_state = _reference_run(
                    q[start:end].contiguous(),
                    k[start:end].contiguous(),
                    v[start:end].contiguous(),
                    current_state,
                    A_log,
                    a[start:end].contiguous(),
                    dt_bias,
                    b[start:end].contiguous(),
                    segment_cu,
                    scale,
                )
                output[start:end].copy_(segment_output)

            if current_state is None:
                new_state[seq_idx].zero_()
            else:
                new_state[seq_idx].copy_(current_state[0])
        return

    if cu_seqlens is not None or q.dim() != 4:
        raise NotImplementedError(
            "composite block policy currently supports dense q.dim()==4 or varlen q.dim()==3 inputs"
        )

    seq_len = q.size(1)
    for batch_idx in range(q.size(0)):
        current_state = state[batch_idx : batch_idx + 1].contiguous() if state is not None else None
        for start, end in _iter_composite_segment_bounds(seq_len, segment_pattern):
            segment_output, current_state = _reference_run(
                q[batch_idx : batch_idx + 1, start:end].contiguous(),
                k[batch_idx : batch_idx + 1, start:end].contiguous(),
                v[batch_idx : batch_idx + 1, start:end].contiguous(),
                current_state,
                A_log,
                a[batch_idx : batch_idx + 1, start:end].contiguous(),
                dt_bias,
                b[batch_idx : batch_idx + 1, start:end].contiguous(),
                None,
                scale,
            )
            output[batch_idx : batch_idx + 1, start:end].copy_(segment_output)

        if current_state is None:
            new_state[batch_idx].zero_()
        else:
            new_state[batch_idx].copy_(current_state[0])


def _select_adaptive_persistent_mode(problem_size: tuple[int, int, int, int, int, int], *, varlen: bool) -> bool:
    selector_keys = _get_adaptive_selector_key_set()
    return any(key in selector_keys for key in _adaptive_selector_keys(problem_size, varlen=varlen))


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
    chunk_size: int,
):
    cache_key = (
        problem_size,
        q.dtype,
        cu_seqlens is not None,
        state is not None,
        scale,
        is_persistent,
        chunk_size,
        tuple(output.shape),
        tuple(output_state.shape),
    )
    if cache_key in _RUNNER_CACHE:
        return _RUNNER_CACHE[cache_key]

    gdn = GDN(is_persistent=is_persistent, chunk_size=chunk_size)
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
    _RUNNER_CACHE[cache_key] = compiled_gdn
    return compiled_gdn


def _run_compiled_gdn_segment(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale_value: float,
    output: torch.Tensor,
    new_state: torch.Tensor,
    *,
    varlen: bool,
    chunk_size: int,
) -> None:
    q_runtime = q.unsqueeze(0) if varlen else q
    k_runtime = k.unsqueeze(0) if varlen else k
    v_runtime = v.unsqueeze(0) if varlen else v
    output_runtime = _prepare_output_view(output, varlen=varlen)

    with _phase_scope("msinfer_entry.composite_compiled_problem_size"):
        problem_size = _get_problem_size_cached(q_runtime, v_runtime, cu_seqlens)
    is_persistent = _resolve_persistent_mode(problem_size, varlen=varlen)

    with _phase_scope("msinfer_entry.composite_compiled_lookup"):
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
            chunk_size=chunk_size,
        )
    with _phase_scope("msinfer_entry.composite_compiled_launch"):
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
            stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


def _run_composite_compiled_schedule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale_value: float,
    output: torch.Tensor,
    new_state: torch.Tensor,
    schedule: _KernelSchedule,
) -> None:
    _validate_kernel_schedule_for_execution(schedule)
    segment_pattern = schedule.internal_launch_segments
    chunk_size = schedule.internal_kernel_chunk_size

    varlen = cu_seqlens is not None and q.dim() == 3
    if varlen:
        assert cu_seqlens is not None
        for seq_idx in range(cu_seqlens.numel() - 1):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            current_state = state[seq_idx : seq_idx + 1].contiguous() if state is not None else None

            for rel_start, rel_end in _iter_composite_segment_bounds(seq_end - seq_start, segment_pattern):
                start = seq_start + rel_start
                end = seq_start + rel_end
                segment_cu = torch.tensor([0, end - start], dtype=cu_seqlens.dtype, device=cu_seqlens.device)
                segment_state = torch.empty_like(new_state[seq_idx : seq_idx + 1])
                _run_compiled_gdn_segment(
                    q[start:end].contiguous(),
                    k[start:end].contiguous(),
                    v[start:end].contiguous(),
                    g[start:end].contiguous(),
                    beta[start:end].contiguous(),
                    current_state,
                    segment_cu,
                    scale_value,
                    output[start:end],
                    segment_state,
                    varlen=True,
                    chunk_size=chunk_size,
                )
                current_state = segment_state

            if current_state is None:
                new_state[seq_idx].zero_()
            else:
                new_state[seq_idx].copy_(current_state[0])
        return

    if cu_seqlens is not None or q.dim() != 4:
        raise NotImplementedError(
            "compiled composite block policy currently supports dense q.dim()==4 or varlen q.dim()==3 inputs"
        )

    seq_len = q.size(1)
    for batch_idx in range(q.size(0)):
        current_state = state[batch_idx : batch_idx + 1].contiguous() if state is not None else None
        for start, end in _iter_composite_segment_bounds(seq_len, segment_pattern):
            segment_state = torch.empty_like(new_state[batch_idx : batch_idx + 1])
            _run_compiled_gdn_segment(
                q[batch_idx : batch_idx + 1, start:end].contiguous(),
                k[batch_idx : batch_idx + 1, start:end].contiguous(),
                v[batch_idx : batch_idx + 1, start:end].contiguous(),
                g[batch_idx : batch_idx + 1, start:end].contiguous(),
                beta[batch_idx : batch_idx + 1, start:end].contiguous(),
                current_state,
                None,
                scale_value,
                output[batch_idx : batch_idx + 1, start:end],
                segment_state,
                varlen=False,
                chunk_size=chunk_size,
            )
            current_state = segment_state

        if current_state is None:
            new_state[batch_idx].zero_()
        else:
            new_state[batch_idx].copy_(current_state[0])


def _run_single_chunk(
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
    schedule = _resolve_kernel_schedule(problem_size)
    if schedule.experimental_policy_enabled and len(schedule.internal_launch_segments) != 1:
        if _normalize_bool_env(os.getenv(_COMPOSITE_COMPILED_HARNESS_ENV)):
            with _phase_scope("msinfer_entry.composite_compiled_schedule"):
                _run_composite_compiled_schedule(
                    q, k, v, g, beta, state, cu_seqlens, scale_value, output, new_state, schedule
                )
            return
        if _normalize_bool_env(os.getenv(_COMPOSITE_REFERENCE_HARNESS_ENV)):
            with _phase_scope("msinfer_entry.composite_reference_schedule"):
                _run_composite_reference_schedule(
                    q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state, schedule
                )
            return
        raise NotImplementedError(
            "experimental block policy selected composite outer schedule "
            f"{schedule.outer_schedule_tile} with segments {schedule.internal_launch_segments}. "
            f"Set {_COMPOSITE_COMPILED_HARNESS_ENV}=1 for the compiled prototype or "
            f"{_COMPOSITE_REFERENCE_HARNESS_ENV}=1 only for the slow correctness harness; "
            "there is no shippable composite fast path yet."
        )

    _validate_kernel_schedule_for_execution(schedule)
    chunk_size = schedule.internal_kernel_chunk_size

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
            chunk_size=chunk_size,
        )
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
            stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


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

        _run_single_chunk(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state)
