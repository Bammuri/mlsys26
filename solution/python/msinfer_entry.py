from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .gdn_blackwell.gdn import chunk_gated_delta_rule


def _normalize_scale(scale: Any) -> float | None:
    if scale is None:
        return None
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


def _prepare_output_view(output: torch.Tensor, *, varlen: bool) -> torch.Tensor:
    if not varlen:
        return output
    if output.dim() == 3:
        return output.unsqueeze(0)
    if output.dim() == 4 and output.size(0) == 1:
        return output
    raise ValueError(f"Unexpected DPS output shape for varlen path: {tuple(output.shape)}")


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
    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(x)
    beta = torch.sigmoid(b.float())
    scale_value = _normalize_scale(scale)

    varlen = cu_seqlens is not None and q.dim() == 3
    if varlen:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale_value,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=_prepare_output_view(output, varlen=varlen),
        output_state=new_state,
    )
