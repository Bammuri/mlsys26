from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkSummary:
    prefix_P: torch.Tensor
    prefix_H: torch.Tensor
    terminal_P: torch.Tensor
    terminal_H: torch.Tensor


@dataclass(frozen=True)
class CompactChunkSummary:
    gate_prefix: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    k: torch.Tensor


def _token_affine_parts(k_t: torch.Tensor, v_t: torch.Tensor, gate_t: torch.Tensor, beta_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eye = torch.eye(k_t.numel(), dtype=k_t.dtype, device=k_t.device)
    transition = gate_t * (eye - beta_t * torch.outer(k_t, k_t))
    additive = beta_t * torch.outer(v_t, k_t)
    return transition, additive


def sequential_gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = (
        initial_state.clone()
        if initial_state is not None
        else torch.zeros(v.size(-1), k.size(-1), dtype=q.dtype, device=q.device)
    )
    outputs = []
    for idx in range(q.size(0)):
        transition, additive = _token_affine_parts(k[idx], v[idx], gate[idx], beta[idx])
        state = state @ transition + additive
        outputs.append(state @ q[idx])
    return torch.stack(outputs, dim=0), state


def build_chunk_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> ChunkSummary:
    seq_len, d_k = k.shape
    d_v = v.shape[-1]
    prefix_P = []
    prefix_H = []

    current_P = torch.eye(d_k, dtype=k.dtype, device=k.device)
    current_H = torch.zeros(d_v, d_k, dtype=v.dtype, device=v.device)
    for idx in range(seq_len):
        transition, additive = _token_affine_parts(k[idx], v[idx], gate[idx], beta[idx])
        current_P = current_P @ transition
        current_H = current_H @ transition + additive
        prefix_P.append(current_P.clone())
        prefix_H.append(current_H.clone())

    return ChunkSummary(
        prefix_P=torch.stack(prefix_P, dim=0),
        prefix_H=torch.stack(prefix_H, dim=0),
        terminal_P=current_P,
        terminal_H=current_H,
    )


def build_compact_chunk_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> CompactChunkSummary:
    seq_len, d_k = k.shape
    d_v = v.shape[-1]
    gate_prefix = []
    w_rows = []
    u_rows = []

    cumulative_gate = torch.ones((), dtype=k.dtype, device=k.device)
    w = torch.zeros(seq_len, d_k, dtype=k.dtype, device=k.device)
    u = torch.zeros(seq_len, d_v, dtype=v.dtype, device=v.device)
    for idx in range(seq_len):
        cumulative_gate = cumulative_gate * gate[idx]
        if idx > 0:
            coeffs = k[:idx] @ k[idx]
            w_correction = torch.einsum("i,ij->j", coeffs, w[:idx])
            u_correction = torch.einsum("i,ij->j", coeffs, u[:idx])
        else:
            w_correction = torch.zeros(d_k, dtype=k.dtype, device=k.device)
            u_correction = torch.zeros(d_v, dtype=v.dtype, device=v.device)
        w[idx] = beta[idx] * (k[idx] - w_correction)
        u[idx] = beta[idx] * ((v[idx] / cumulative_gate) - u_correction)
        gate_prefix.append(cumulative_gate.clone())
        w_rows.append(w[idx].clone())
        u_rows.append(u[idx].clone())

    return CompactChunkSummary(
        gate_prefix=torch.stack(gate_prefix, dim=0),
        w=torch.stack(w_rows, dim=0),
        u=torch.stack(u_rows, dim=0),
        k=k,
    )


def materialize_compact_chunk_summary(summary: CompactChunkSummary) -> ChunkSummary:
    seq_len, d_k = summary.k.shape
    d_v = summary.u.shape[-1]
    prefix_P = []
    prefix_H = []
    eye = torch.eye(d_k, dtype=summary.k.dtype, device=summary.k.device)
    for idx in range(seq_len):
        k_prefix = summary.k[: idx + 1]
        w_prefix = summary.w[: idx + 1]
        u_prefix = summary.u[: idx + 1]
        P = summary.gate_prefix[idx] * (eye - torch.einsum("id,ik->dk", w_prefix, k_prefix))
        J = torch.einsum("iv,ik->vk", u_prefix, k_prefix)
        H = summary.gate_prefix[idx] * J
        prefix_P.append(P)
        prefix_H.append(H)
    return ChunkSummary(
        prefix_P=torch.stack(prefix_P, dim=0),
        prefix_H=torch.stack(prefix_H, dim=0),
        terminal_P=prefix_P[-1],
        terminal_H=prefix_H[-1],
    )


def outputs_from_compact_chunk_summary(
    summary: CompactChunkSummary,
    q: torch.Tensor,
    *,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    for idx in range(q.size(0)):
        k_prefix = summary.k[: idx + 1]
        alpha = k_prefix @ q[idx]
        p_q = q[idx] - torch.einsum("i,ij->j", alpha, summary.w[: idx + 1])
        j_q = torch.einsum("i,ij->j", alpha, summary.u[: idx + 1])
        output = summary.gate_prefix[idx] * (initial_state @ p_q + j_q)
        outputs.append(output)
    return torch.stack(outputs, dim=0)


def chunk_parallel_from_compact_summary(
    q: torch.Tensor,
    summary: CompactChunkSummary,
    *,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = summary.k
    c = summary.gate_prefix
    w = summary.w
    u = summary.u

    causal_qk = torch.tril(q @ k.T)
    state_term = q @ initial_state.T
    correction = causal_qk @ (u - w @ initial_state.T)
    outputs = c[:, None] * (state_term + correction)

    d_k = k.size(-1)
    eye = torch.eye(d_k, dtype=k.dtype, device=k.device)
    final_state = c[-1] * (initial_state @ (eye - w.T @ k) + u.T @ k)
    return outputs, final_state


def build_parallel_chunk_factors(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = torch.cumprod(gate, dim=0)
    v_hat = v / c[:, None]
    gram = k @ k.T
    lower = torch.tril(beta[:, None] * gram, diagonal=-1)
    lhs = torch.eye(k.size(0), dtype=k.dtype, device=k.device) + lower
    rhs = torch.diag(beta)
    transform = torch.linalg.solve_triangular(lhs, rhs, upper=False)
    w = transform @ k
    u = transform @ v_hat
    return c, w, u


def chunk_parallel_from_triangular_factors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    c, w, u = build_parallel_chunk_factors(k, v, gate, beta)
    summary = CompactChunkSummary(gate_prefix=c, w=w, u=u, k=k)
    return chunk_parallel_from_compact_summary(q, summary, initial_state=initial_state)


def build_parallel_chunk_factors_batched(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError("k and v must have shape [T, H, D]")
    if gate.shape != beta.shape or gate.ndim != 2:
        raise ValueError("gate and beta must have shape [T, H]")
    if k.shape[:2] != gate.shape or v.shape[:2] != gate.shape:
        raise ValueError("sequence/head dimensions must match across k, v, gate, and beta")

    k_h = k.permute(1, 0, 2)
    v_h = v.permute(1, 0, 2)
    gate_h = gate.permute(1, 0)
    beta_h = beta.permute(1, 0)

    c_h = torch.cumprod(gate_h, dim=1)
    v_hat_h = v_h / c_h.unsqueeze(-1)
    gram = torch.matmul(k_h, k_h.transpose(-1, -2))
    lower = torch.tril(beta_h.unsqueeze(-1) * gram, diagonal=-1)
    identity = torch.eye(k_h.size(1), dtype=k.dtype, device=k.device).expand(k_h.size(0), -1, -1)
    rhs = torch.diag_embed(beta_h)
    transform = torch.linalg.solve_triangular(identity + lower, rhs, upper=False)
    w_h = torch.matmul(transform, k_h)
    u_h = torch.matmul(transform, v_hat_h)
    return c_h.permute(1, 0), w_h.permute(1, 0, 2), u_h.permute(1, 0, 2)


def chunk_parallel_from_triangular_factors_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 3:
        raise ValueError("q must have shape [T, H, D]")
    if initial_state.ndim != 3:
        raise ValueError("initial_state must have shape [H, Dv, Dk]")
    c, w, u = build_parallel_chunk_factors_batched(k, v, gate, beta)

    q_h = q.permute(1, 0, 2)
    k_h = k.permute(1, 0, 2)
    w_h = w.permute(1, 0, 2)
    u_h = u.permute(1, 0, 2)
    c_h = c.permute(1, 0)

    causal_qk = torch.tril(torch.matmul(q_h, k_h.transpose(-1, -2)))
    state_term = torch.matmul(q_h, initial_state.transpose(-1, -2))
    correction_rhs = u_h - torch.matmul(w_h, initial_state.transpose(-1, -2))
    correction = torch.matmul(causal_qk, correction_rhs)
    outputs_h = c_h.unsqueeze(-1) * (state_term + correction)

    d_k = k.size(-1)
    identity = torch.eye(d_k, dtype=k.dtype, device=k.device).expand(initial_state.size(0), -1, -1)
    final_state = c_h[:, -1].view(-1, 1, 1) * (
        torch.matmul(initial_state, identity - torch.matmul(w_h.transpose(-1, -2), k_h))
        + torch.matmul(u_h.transpose(-1, -2), k_h)
    )
    return outputs_h.permute(1, 0, 2), final_state


def compose_chunk_summaries(left: ChunkSummary, right: ChunkSummary) -> tuple[torch.Tensor, torch.Tensor]:
    return left.terminal_P @ right.terminal_P, left.terminal_H @ right.terminal_P + right.terminal_H


def segmented_gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("q, k, and v must share the same sequence length")

    seq_len = q.size(0)
    chunk_summaries: list[ChunkSummary] = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_summaries.append(build_chunk_summary(k[start:end], v[start:end], gate[start:end], beta[start:end]))

    state = (
        initial_state.clone()
        if initial_state is not None
        else torch.zeros(v.size(-1), k.size(-1), dtype=q.dtype, device=q.device)
    )
    outputs = []
    offset = 0
    for summary in chunk_summaries:
        chunk_len = summary.prefix_P.size(0)
        q_chunk = q[offset : offset + chunk_len]
        chunk_outputs = torch.einsum("ij,tjk,tk->ti", state, summary.prefix_P, q_chunk)
        chunk_outputs = chunk_outputs + torch.einsum("tij,tj->ti", summary.prefix_H, q_chunk)
        outputs.append(chunk_outputs)
        state = state @ summary.terminal_P + summary.terminal_H
        offset += chunk_len

    return torch.cat(outputs, dim=0), state
