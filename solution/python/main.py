from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

_CUTLASS_IMPORT_ERROR = None
try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401
    import cutlass.pipeline as pipeline  # noqa: F401
    import cuda.bindings.driver as cuda  # noqa: F401
except Exception as exc:  # pragma: no cover
    _CUTLASS_IMPORT_ERROR = exc

try:
    from flashinfer import chunk_gated_delta_rule
except Exception:  # pragma: no cover
    chunk_gated_delta_rule = None


def _require_cutlass_dsl() -> None:
    if _CUTLASS_IMPORT_ERROR is not None:
        raise RuntimeError(f"CUTLASS DSL import failed: {_CUTLASS_IMPORT_ERROR}")


def _normalize_scale(scale: float | torch.Tensor | None, head_size: int) -> float:
    if scale is None:
        return 1.0 / math.sqrt(head_size)
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


def _reference_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
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

    output = torch.empty((q.size(0), num_v_heads, head_size), dtype=q.dtype, device=device)
    new_state = torch.empty(
        (num_seqs, num_v_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())

        if state is not None:
            state_hkv = state[seq_idx].float().transpose(-1, -2).contiguous()
        else:
            state_hkv = torch.zeros(
                (num_v_heads, head_size, head_size), dtype=torch.float32, device=device
            )

        for t in range(seq_start, seq_end):
            q_h1k = q_exp[t].unsqueeze(1)
            k_h1k = k_exp[t].unsqueeze(1)
            v_h1v = v_f32[t].unsqueeze(1)
            g_h11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_h11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state = g_h11 * state_hkv
            old_v = torch.matmul(k_h1k, old_state)
            new_v = beta_h11 * v_h1v + (1.0 - beta_h11) * old_v
            state_remove = torch.matmul(k_h1k.transpose(-1, -2), old_v)
            state_update = torch.matmul(k_h1k.transpose(-1, -2), new_v)
            state_hkv = old_state - state_remove + state_update

            out = scale_val * torch.matmul(q_h1k, state_hkv)
            output[t] = out.squeeze(1).to(dtype=q.dtype)

        new_state[seq_idx] = state_hkv.transpose(-1, -2)

    return output, new_state


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | torch.Tensor | None,
):
    _require_cutlass_dsl()

    scale_val = _normalize_scale(scale, q.shape[-1])
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    state = None if state is None else state.contiguous()
    cu_seqlens = cu_seqlens.to(torch.int64).contiguous()

    if chunk_gated_delta_rule is not None:
        x = a.float() + dt_bias.float()
        g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
        beta = torch.sigmoid(b.float())
        try:
            return chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g.contiguous(),
                beta=beta.contiguous(),
                scale=scale_val,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )
        except Exception:
            pass

    return _reference_gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale_val)
