from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRACE_VOLUME_NAME = "mlsys26-contest"
BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

app = modal.App("flashinfer-gdn-twu-phase-profile")
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
    parser = argparse.ArgumentParser(description="Profile chunk-local T/W/U phase timings on Modal B200.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def summarize_timings(samples_ms: list[float]) -> dict[str, float | int | None]:
    if not samples_ms:
        return {"count": 0, "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(samples_ms)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "count": len(samples_ms),
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "median_ms": median,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _write_output(output_json: Path, result: dict) -> None:
    payload = {"result": result}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={"/data": trace_volume})
def run_twu_phase_case(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    iterations: int,
    warmup_iterations: int,
    seed: int,
):
    import torch

    torch.manual_seed(seed)
    dtype = torch.float32
    k = 0.1 * torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    v = 0.1 * torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    gate = 0.95 + 0.05 * torch.sigmoid(torch.randn(seq_len, num_heads, dtype=dtype, device="cuda"))
    beta = 0.25 * torch.sigmoid(torch.randn(seq_len, num_heads, dtype=dtype, device="cuda"))

    def _sync():
        torch.cuda.synchronize("cuda")

    def _time_ms(fn):
        start = time.perf_counter()
        result = fn()
        _sync()
        return result, (time.perf_counter() - start) * 1000.0

    samples = {
        "cumprod_and_vhat_ms": [],
        "gram_ms": [],
        "lower_ms": [],
        "solve_ms": [],
        "wu_matmul_ms": [],
        "total_ms": [],
    }

    def _run_once():
        k_h = k.permute(1, 0, 2)
        v_h = v.permute(1, 0, 2)
        gate_h = gate.permute(1, 0)
        beta_h = beta.permute(1, 0)

        (c_h, v_hat_h), phase_ms = _time_ms(
            lambda: (
                torch.cumprod(gate_h, dim=1),
                v_h / torch.cumprod(gate_h, dim=1).unsqueeze(-1),
            )
        )
        samples["cumprod_and_vhat_ms"].append(phase_ms)

        gram, phase_ms = _time_ms(lambda: torch.matmul(k_h, k_h.transpose(-1, -2)))
        samples["gram_ms"].append(phase_ms)

        lower, phase_ms = _time_ms(lambda: torch.tril(beta_h.unsqueeze(-1) * gram, diagonal=-1))
        samples["lower_ms"].append(phase_ms)

        identity = torch.eye(k_h.size(1), dtype=k.dtype, device=k.device).expand(k_h.size(0), -1, -1)
        rhs = torch.diag_embed(beta_h)
        transform, phase_ms = _time_ms(lambda: torch.linalg.solve_triangular(identity + lower, rhs, upper=False))
        samples["solve_ms"].append(phase_ms)

        (_, _), phase_ms = _time_ms(lambda: (torch.matmul(transform, k_h), torch.matmul(transform, v_hat_h)))
        samples["wu_matmul_ms"].append(phase_ms)

    for _ in range(max(1, warmup_iterations)):
        _run_once()

    for _ in range(max(1, iterations)):
        total_start = time.perf_counter()
        _run_once()
        samples["total_ms"].append((time.perf_counter() - total_start) * 1000.0)

    return {
        "config": {
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "seed": seed,
        },
        "timing": {name: summarize_timings(values) for name, values in samples.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with app.run():
        result = run_twu_phase_case.remote(
            args.seq_len,
            args.num_heads,
            args.head_dim,
            args.iterations,
            args.warmup_iterations,
            args.seed,
        )
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json)}, indent=2))
    return 0


@app.local_entrypoint()
def modal_main(
    seq_len: int = 128,
    num_heads: int = 4,
    head_dim: int = 64,
    iterations: int = 20,
    warmup_iterations: int = 3,
    seed: int = 0,
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
        "--output-json",
        output_json,
    ]
    args = parse_args(argv)
    result = run_twu_phase_case.remote(
        args.seq_len,
        args.num_heads,
        args.head_dim,
        args.iterations,
        args.warmup_iterations,
        args.seed,
    )
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
