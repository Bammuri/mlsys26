from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

TRACE_VOLUME_NAME = "mlsys26-contest"
BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

app = modal.App("flashinfer-gdn-diag-solve-sweep")
trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "10.0a", "PATH": BASE_PATH})
    .pip_install("torch", "numpy")
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Modal B200 diagonal solve sweep for TWU blocks.")
    parser.add_argument("--block-sizes", required=True, help="Comma-separated block sizes")
    parser.add_argument("--rhs-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
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
def run_diag_solve_sweep(block_sizes: list[int], rhs_dim: int, num_heads: int, iterations: int, warmup_iterations: int, seed: int):
    import time
    import torch

    torch.manual_seed(seed)
    dtype = torch.float32
    rows = []
    for block_size in block_sizes:
        base = 0.01 * torch.randn(num_heads, block_size, block_size, dtype=dtype, device="cuda")
        lower = torch.tril(base, diagonal=-1)
        rhs = 0.1 * torch.randn(num_heads, block_size, rhs_dim, dtype=dtype, device="cuda")
        identity = torch.eye(block_size, dtype=dtype, device="cuda").expand(num_heads, -1, -1)

        def _sync():
            torch.cuda.synchronize("cuda")

        def _time_ms(fn):
            start = time.perf_counter()
            result = fn()
            _sync()
            return result, (time.perf_counter() - start) * 1000.0

        def _solve():
            return torch.linalg.solve_triangular(identity + lower, rhs, upper=False)

        expected = _solve()
        actual = _solve()
        correctness = {
            'finite_ok': bool(torch.isfinite(actual).all().item()),
            'max_abs_diff': float((actual - expected).abs().max().item()),
        }

        for _ in range(max(1, warmup_iterations)):
            _solve(); _sync()

        samples = []
        for _ in range(max(1, iterations)):
            _, t_ms = _time_ms(_solve)
            samples.append(t_ms)

        rows.append({
            'block_size': block_size,
            'correctness': correctness,
            'timing': summarize(samples),
        })
    return {'config': {'rhs_dim': rhs_dim, 'num_heads': num_heads, 'iterations': iterations, 'warmup_iterations': warmup_iterations, 'seed': seed}, 'rows': rows}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    block_sizes = [int(x.strip()) for x in args.block_sizes.split(',') if x.strip()]
    with app.run():
        result = run_diag_solve_sweep.remote(block_sizes, args.rhs_dim, args.num_heads, args.iterations, args.warmup_iterations, args.seed)
    _write_output(args.output_json, result)
    print(json.dumps({'output_json': str(args.output_json), 'cases': len(result['rows'])}))
    return 0


@app.local_entrypoint()
def modal_main(block_sizes: str = '160,192,224', rhs_dim: int = 128, num_heads: int = 4, iterations: int = 20, warmup_iterations: int = 3, seed: int = 0, output_json: str = ''):
    argv = [
        '--block-sizes', block_sizes,
        '--rhs-dim', str(rhs_dim),
        '--num-heads', str(num_heads),
        '--iterations', str(iterations),
        '--warmup-iterations', str(warmup_iterations),
        '--seed', str(seed),
        '--output-json', output_json,
    ]
    args = parse_args(argv)
    block_sizes_list = [int(x.strip()) for x in args.block_sizes.split(',') if x.strip()]
    result = run_diag_solve_sweep.remote(block_sizes_list, args.rhs_dim, args.num_heads, args.iterations, args.warmup_iterations, args.seed)
    _write_output(args.output_json, result)
    print(json.dumps({'output_json': str(args.output_json), 'cases': len(result['rows'])}))


if __name__ == '__main__':
    raise SystemExit(main())
