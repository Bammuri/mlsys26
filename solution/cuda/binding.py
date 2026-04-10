"""
Optimized FlashInfer-backed GDN prefill comparison wrapper.

This file exists for Python-side A/B benchmarking against the active native
submission path in ``binding.cpp::run``. The main optimization over a simple
wrapper is to minimize wrapper overhead: reuse output/state buffers when shapes
repeat, pass preallocated destinations into FlashInfer, and opportunistically
alias the final state to the incoming state buffer when that is safe enough to
try.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

try:
    from flashinfer import chunk_gated_delta_rule
except Exception:  # pragma: no cover - runtime fallback for environments without FlashInfer
    chunk_gated_delta_rule = None

_BufferKey = Tuple[str, Tuple[int, ...], torch.dtype]
_BUFFER_CACHE: Dict[_BufferKey, torch.Tensor] = {}


def _normalize_scale(scale: float | torch.Tensor | None, head_size: int) -> float:
    if scale is None:
        return 1.0 / (head_size**0.5)
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


def _buffer_key(tensor: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype) -> _BufferKey:
    device_index = tensor.device.index if tensor.device.index is not None else -1
    return (f"{tensor.device.type}:{device_index}", shape, dtype)


def _get_buffer_like(reference: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    key = _buffer_key(reference, shape, dtype)
    buf = _BUFFER_CACHE.get(key)
    if buf is None or buf.device != reference.device:
        buf = torch.empty(shape, dtype=dtype, device=reference.device)
        _BUFFER_CACHE[key] = buf
    return buf


def _make_gate_beta(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    A_log_f32 = A_log if A_log.dtype == torch.float32 else A_log.float()
    dt_bias_f32 = dt_bias if dt_bias.dtype == torch.float32 else dt_bias.float()
    x = a.float() + dt_bias_f32
    g = -torch.exp(A_log_f32) * F.softplus(x)
    beta = torch.sigmoid(b.float())
    return g.contiguous(), beta.contiguous()


def _reference_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale_val: float,
):
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    num_seqs = cu_seqlens.numel() - 1
    device = q.device

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1).float()
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1).float()
    v_f32 = v.float()

    output = torch.empty(
        (q.size(0), num_v_heads, head_size),
        dtype=q.dtype,
        device=device,
    )
    new_state = torch.empty(
        (num_seqs, num_v_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if state is not None:
            state_hkv = state[seq_idx].float().transpose(-1, -2).contiguous()
        else:
            state_hkv = torch.zeros(
                (num_v_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        for offset in range(seq_len):
            t = seq_start + offset
            q_h1k = q_exp[t].unsqueeze(1)
            k_h1k = k_exp[t].unsqueeze(1)
            v_h1v = v_f32[t].unsqueeze(1)
            g_h11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_h11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state = g_h11 * state_hkv
            old_v = torch.matmul(k_h1k, old_state)
            new_v = beta_h11 * v_h1v + (1.0 - beta_h11) * old_v
            state_remove = torch.einsum("hkl,hlv->hkv", k_h1k.transpose(-1, -2), old_v)
            state_update = torch.einsum("hkl,hlv->hkv", k_h1k.transpose(-1, -2), new_v)
            state_hkv = old_state - state_remove + state_update

            out = scale_val * torch.matmul(q_h1k, state_hkv)
            output[t] = out.squeeze(1).to(dtype=q.dtype)

        new_state[seq_idx] = state_hkv.transpose(-1, -2)

    return output, new_state


def _flashinfer_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale_val: float,
):
    g, beta = _make_gate_beta(A_log, a, dt_bias, b)
    output = _get_buffer_like(q, (q.size(0), v.size(1), q.size(2)), q.dtype)
    output_state_shape = (cu_seqlens.numel() - 1, v.size(1), q.size(2), q.size(2))

    alias_state = (
        state is not None
        and state.dtype == torch.float32
        and state.is_contiguous()
        and tuple(state.shape) == output_state_shape
    )
    output_state = state if alias_state else _get_buffer_like(q, output_state_shape, torch.float32)

    try:
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale_val,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output,
            output_state=output_state,
        )
    except Exception:
        if not alias_state:
            raise
        output_state = _get_buffer_like(q, output_state_shape, torch.float32)
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale_val,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output,
            output_state=output_state,
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
):
    """Contest entrypoint for GDN prefill."""
    scale_val = _normalize_scale(scale, q.shape[-1])
    q = q if q.is_contiguous() else q.contiguous()
    k = k if k.is_contiguous() else k.contiguous()
    v = v if v.is_contiguous() else v.contiguous()
    state = None if state is None else (state if state.is_contiguous() else state.contiguous())
    cu_seqlens = (
        cu_seqlens
        if cu_seqlens.dtype == torch.int64 and cu_seqlens.is_contiguous()
        else cu_seqlens.to(torch.int64).contiguous()
    )

    if chunk_gated_delta_rule is not None:
        try:
            return _flashinfer_prefill(
                q=q,
                k=k,
                v=v,
                state=state,
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                b=b,
                cu_seqlens=cu_seqlens,
                scale_val=scale_val,
            )
        except Exception:
            pass

    return _reference_gdn_prefill(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        cu_seqlens=cu_seqlens,
        scale_val=scale_val,
    )
