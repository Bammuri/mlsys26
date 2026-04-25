from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

TRACE_VOLUME_NAME = "mlsys26-contest"
BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

app = modal.App("flashinfer-gdn-block-solve-phase-profile")
trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "10.0a", "PATH": BASE_PATH})
    .pip_install("torch", "numpy")
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile flat block solve phase timings on Modal B200.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rhs-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def summarize(samples_ms: list[float]) -> dict[str, float | int | None]:
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
def run_block_solve_phase_case(seq_len: int, num_heads: int, rhs_dim: int, block_size: int, iterations: int, warmup_iterations: int, seed: int):
    import time
    import torch

    torch.manual_seed(seed)
    dtype = torch.float32
    base = 0.01 * torch.randn(num_heads, seq_len, seq_len, dtype=dtype, device="cuda")
    lower = torch.tril(base, diagonal=-1)
    rhs = 0.1 * torch.randn(num_heads, seq_len, rhs_dim, dtype=dtype, device="cuda")
    identity = torch.eye(seq_len, dtype=dtype, device="cuda").expand(num_heads, -1, -1)
    system = identity + lower

    def _sync():
        torch.cuda.synchronize("cuda")

    def _time_ms(fn):
        start = time.perf_counter()
        result = fn()
        _sync()
        return result, (time.perf_counter() - start) * 1000.0

    samples = {
        'diag_solve_ms': [],
        'trailing_update_ms': [],
        'total_ms': [],
    }

    def _run_once():
        out = rhs.clone()
        diag_total = 0.0
        update_total = 0.0
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            diag = system[:, start:end, start:end]
            solved, t_ms = _time_ms(lambda: torch.linalg.solve_triangular(diag, out[:, start:end, :], upper=False))
            diag_total += t_ms
            out[:, start:end, :] = solved
            if end < seq_len:
                update, t_ms = _time_ms(lambda: torch.matmul(system[:, end:, start:end], solved))
                update_total += t_ms
                out[:, end:, :] = out[:, end:, :] - update
        return out, diag_total, update_total

    for _ in range(max(1, warmup_iterations)):
        _run_once()

    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        _, diag_total, update_total = _run_once()
        samples['total_ms'].append((time.perf_counter() - start) * 1000.0)
        samples['diag_solve_ms'].append(diag_total)
        samples['trailing_update_ms'].append(update_total)

    return {
        'config': {
            'seq_len': seq_len,
            'num_heads': num_heads,
            'rhs_dim': rhs_dim,
            'block_size': block_size,
            'iterations': iterations,
            'warmup_iterations': warmup_iterations,
            'seed': seed,
        },
        'timing': {name: summarize(values) for name, values in samples.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with app.run():
        result = run_block_solve_phase_case.remote(args.seq_len, args.num_heads, args.rhs_dim, args.block_size, args.iterations, args.warmup_iterations, args.seed)
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json)}))
    return 0


@app.local_entrypoint()
def modal_main(seq_len: int = 4096, num_heads: int = 4, rhs_dim: int = 128, block_size: int = 192, iterations: int = 8, warmup_iterations: int = 2, seed: int = 0, output_json: str = ''):
    argv = [
        '--seq-len', str(seq_len),
        '--num-heads', str(num_heads),
        '--rhs-dim', str(rhs_dim),
        '--block-size', str(block_size),
        '--iterations', str(iterations),
        '--warmup-iterations', str(warmup_iterations),
        '--seed', str(seed),
        '--output-json', output_json,
    ]
    args = parse_args(argv)
    result = run_block_solve_phase_case.remote(args.seq_len, args.num_heads, args.rhs_dim, args.block_size, args.iterations, args.warmup_iterations, args.seed)
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json)}))


if __name__ == '__main__':
    raise SystemExit(main())
