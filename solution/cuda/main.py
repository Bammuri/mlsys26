"""Sample-style FlashInfer GDN prefill comparison wrapper.

This mirrors the user-provided FlashInfer/Blackwell flow closely enough for
A/B benchmarking, including the ``scale=None`` fast path used by that wrapper.
It is not the active promoted submission path.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

try:
    from flashinfer import chunk_gated_delta_rule
except Exception:  # pragma: no cover
    chunk_gated_delta_rule = None

_BufferKey = Tuple[str, Tuple[int, ...], torch.dtype]
_GateKey = Tuple[int, int, int, int]
_BUFFER_CACHE: Dict[_BufferKey, torch.Tensor] = {}
_GATE_BETA_CACHE: Dict[_GateKey, tuple[torch.Tensor, torch.Tensor]] = {}


def _buffer_key(reference: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype) -> _BufferKey:
    device_index = reference.device.index if reference.device.index is not None else -1
    return (f"{reference.device.type}:{device_index}", shape, dtype)


def _get_buffer_like(reference: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    key = _buffer_key(reference, shape, dtype)
    buf = _BUFFER_CACHE.get(key)
    if buf is None or buf.device != reference.device:
        buf = torch.empty(shape, dtype=dtype, device=reference.device)
        _BUFFER_CACHE[key] = buf
    return buf


def _gate_key(A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor) -> _GateKey:
    return (id(A_log), id(a), id(dt_bias), id(b))


def _get_gate_beta(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = _gate_key(A_log, a, dt_bias, b)
    cached = _GATE_BETA_CACHE.get(key)
    if cached is not None:
        return cached

    x = a.float() + dt_bias.float()
    g = (-torch.exp(A_log.float()) * F.softplus(x)).contiguous()
    beta = torch.sigmoid(b.float()).contiguous()
    _GATE_BETA_CACHE[key] = (g, beta)
    return g, beta


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
):
    if chunk_gated_delta_rule is None:
        raise RuntimeError("flashinfer.chunk_gated_delta_rule is unavailable")

    g, beta = _get_gate_beta(A_log, a, dt_bias, b)

    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    state = None if state is None else (state if state.is_contiguous() else state.contiguous())
    cu_seqlens = (
        cu_seqlens
        if cu_seqlens.dtype == torch.int64 and cu_seqlens.is_contiguous()
        else cu_seqlens.to(torch.int64).contiguous()
    )

    varlen = cu_seqlens is not None and q.dim() == 3
    output_shape = (q.size(0), v.size(1), q.size(2))
    output_state_shape = (cu_seqlens.numel() - 1, v.size(1), q.size(2), q.size(2))
    output = _get_buffer_like(q, output_shape, q.dtype)
    output_state = _get_buffer_like(q, output_state_shape, torch.float32)
    output_for_call = output

    if varlen:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        output_for_call = output.unsqueeze(0)

    output, new_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=output_for_call,
        output_state=output_state,
    )

    if varlen:
        output = output.squeeze(0)

    return output, new_state
