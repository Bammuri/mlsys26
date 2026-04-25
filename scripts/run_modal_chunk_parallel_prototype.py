from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRACE_VOLUME_NAME = "mlsys26-contest"
BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

app = modal.App("flashinfer-gdn-chunk-parallel-prototype")
trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "PATH": BASE_PATH,
        }
    )
    .pip_install("torch", "triton", "numpy")
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Modal B200 GPU prototype for chunk-parallel GDN math.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _dtype_from_name(name: str):
    import torch

    return {"float32": torch.float32, "float64": torch.float64}[name]


def summarize_timings(samples_ms: list[float]) -> dict[str, float | int | None]:
    if not samples_ms:
        return {"count": 0, "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(samples_ms)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "count": len(samples_ms),
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "median_ms": median,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


@dataclass(frozen=True)
class CompactChunkSummary:
    gate_prefix: any
    w: any
    u: any
    k: any


def _build_parallel_chunk_factors_batched(k, v, gate, beta):
    import torch

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


def _chunk_parallel_from_triangular_factors_batched(q, k, v, gate, beta, *, initial_state):
    import torch

    c, w, u = _build_parallel_chunk_factors_batched(k, v, gate, beta)
    k_h = k.permute(1, 0, 2)
    q_h = q.permute(1, 0, 2)
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


def _sequential_gdn_reference(q, k, v, gate, beta, *, initial_state):
    import torch

    state = initial_state.clone()
    outputs = []
    for idx in range(q.size(0)):
        eye = torch.eye(k.size(-1), dtype=q.dtype, device=q.device)
        transition = gate[idx] * (eye - beta[idx] * torch.outer(k[idx], k[idx]))
        additive = beta[idx] * torch.outer(v[idx], k[idx])
        state = state @ transition + additive
        outputs.append(state @ q[idx])
    return torch.stack(outputs, dim=0), state


def _write_output(output_json: Path, result: dict) -> None:
    payload = {"result": result}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={"/data": trace_volume})
def run_chunk_parallel_case(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    iterations: int,
    warmup_iterations: int,
    seed: int,
    dtype_name: str,
):
    import torch

    dtype = _dtype_from_name(dtype_name)
    torch.manual_seed(seed)
    q = 0.1 * torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    k = 0.1 * torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    v = 0.1 * torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    gate = 0.95 + 0.05 * torch.sigmoid(torch.randn(seq_len, num_heads, dtype=dtype, device="cuda"))
    beta = 0.25 * torch.sigmoid(torch.randn(seq_len, num_heads, dtype=dtype, device="cuda"))
    initial_state = 0.1 * torch.randn(num_heads, head_dim, head_dim, dtype=dtype, device="cuda")

    def _sync():
        torch.cuda.synchronize("cuda")

    def _time_ms(fn):
        start = time.perf_counter()
        result = fn()
        _sync()
        return result, (time.perf_counter() - start) * 1000.0

    # correctness
    expected_out = []
    expected_state = []
    for head in range(num_heads):
        out_ref, state_ref = _sequential_gdn_reference(
            q[:, head, :],
            k[:, head, :],
            v[:, head, :],
            gate[:, head],
            beta[:, head],
            initial_state=initial_state[head],
        )
        expected_out.append(out_ref)
        expected_state.append(state_ref)
    expected_out = torch.stack(expected_out, dim=1)
    expected_state = torch.stack(expected_state, dim=0)
    actual_out, actual_state = _chunk_parallel_from_triangular_factors_batched(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=initial_state,
    )
    max_output_abs_diff = float((actual_out - expected_out).abs().max().item())
    max_state_abs_diff = float((actual_state - expected_state).abs().max().item())
    finite_ok = bool(torch.isfinite(actual_out).all().item() and torch.isfinite(actual_state).all().item())

    # timing
    seq_samples = []
    tri_samples = []
    for _ in range(max(1, warmup_iterations)):
        for head in range(num_heads):
            _sequential_gdn_reference(
                q[:, head, :],
                k[:, head, :],
                v[:, head, :],
                gate[:, head],
                beta[:, head],
                initial_state=initial_state[head],
            )
        _chunk_parallel_from_triangular_factors_batched(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
        )
        _sync()

    for _ in range(max(1, iterations)):
        _, seq_ms = _time_ms(
            lambda: [
                _sequential_gdn_reference(
                    q[:, head, :],
                    k[:, head, :],
                    v[:, head, :],
                    gate[:, head],
                    beta[:, head],
                    initial_state=initial_state[head],
                )
                for head in range(num_heads)
            ]
        )
        _, tri_ms = _time_ms(
            lambda: _chunk_parallel_from_triangular_factors_batched(
                q,
                k,
                v,
                gate,
                beta,
                initial_state=initial_state,
            )
        )
        seq_samples.append(seq_ms)
        tri_samples.append(tri_ms)

    seq_summary = summarize_timings(seq_samples)
    tri_summary = summarize_timings(tri_samples)
    speedup = (
        seq_summary["mean_ms"] / tri_summary["mean_ms"]
        if seq_summary["mean_ms"] and tri_summary["mean_ms"]
        else None
    )
    return {
        "config": {
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": dtype_name,
            "seed": seed,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
        },
        "correctness": {
            "max_output_abs_diff": max_output_abs_diff,
            "max_state_abs_diff": max_state_abs_diff,
            "finite_ok": finite_ok,
        },
        "timing": {
            "sequential_ms": seq_summary,
            "chunk_parallel_ms": tri_summary,
            "speedup_vs_sequential": speedup,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with app.run():
        result = run_chunk_parallel_case.remote(
            args.seq_len,
            args.num_heads,
            args.head_dim,
            args.iterations,
            args.warmup_iterations,
            args.seed,
            args.dtype,
        )
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "speedup": result["timing"]["speedup_vs_sequential"]}, indent=2))
    return 0


@app.local_entrypoint()
def modal_main(
    seq_len: int = 128,
    num_heads: int = 4,
    head_dim: int = 64,
    iterations: int = 20,
    warmup_iterations: int = 3,
    seed: int = 0,
    dtype: str = "float32",
    output_json: str = "",
):
    argv = [
        "--seq-len",
        str(seq_len),
        "--num-heads",
        str(num_heads),
        "--head-dim",
        str(head_dim),
        "--iterations",
        str(iterations),
        "--warmup-iterations",
        str(warmup_iterations),
        "--seed",
        str(seed),
        "--dtype",
        dtype,
        "--output-json",
        output_json,
    ]
    args = parse_args(argv)
    result = run_chunk_parallel_case.remote(
        args.seq_len,
        args.num_heads,
        args.head_dim,
        args.iterations,
        args.warmup_iterations,
        args.seed,
        args.dtype,
    )
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "speedup": result["timing"]["speedup_vs_sequential"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
