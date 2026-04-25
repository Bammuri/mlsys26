from __future__ import annotations

import torch


def block_lower_triangular_solve(
    lower: torch.Tensor,
    rhs: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Solve (I + lower) x = rhs for strictly lower-triangular `lower`.

    Shapes:
    - lower: [H, N, N] or [N, N]
    - rhs:   [H, N, D] or [N, D]
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    squeeze = False
    if lower.ndim == 2:
        lower = lower.unsqueeze(0)
        rhs = rhs.unsqueeze(0)
        squeeze = True

    if lower.ndim != 3 or rhs.ndim != 3:
        raise ValueError("lower must be [H,N,N] and rhs must be [H,N,D]")
    if lower.shape[0] != rhs.shape[0] or lower.shape[1] != lower.shape[2] or lower.shape[1] != rhs.shape[1]:
        raise ValueError("shape mismatch between lower and rhs")

    h, n, _ = lower.shape
    identity = torch.eye(n, dtype=lower.dtype, device=lower.device).expand(h, -1, -1)
    system = identity + lower
    out = rhs.clone()

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        diag = system[:, start:end, start:end]
        block_rhs = out[:, start:end, :]
        solved = torch.linalg.solve_triangular(diag, block_rhs, upper=False)
        out[:, start:end, :] = solved
        if end < n:
            update = torch.matmul(system[:, end:, start:end], solved)
            out[:, end:, :] = out[:, end:, :] - update

    if squeeze:
        return out[0]
    return out
