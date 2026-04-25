from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

TRACE_VOLUME_NAME = "mlsys26-contest"
BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

app = modal.App("flashinfer-gdn-staged-block-solve-prototype")
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
    .pip_install("torch", "numpy")
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Modal B200 prototype for staged block TWU solve.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rhs-dim", type=int, default=128)
    parser.add_argument("--stage-block", type=int, default=192)
    parser.add_argument("--inner-block", type=int, default=64)
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


def _staged_block_lower_triangular_solve(lower, rhs, *, stage_block, inner_block):
    import torch

    h, n, _ = lower.shape
    identity = torch.eye(n, dtype=lower.dtype, device=lower.device).expand(h, -1, -1)
    system = identity + lower
    out = rhs.clone()
    for stage_start in range(0, n, stage_block):
        stage_end = min(stage_start + stage_block, n)
        for start in range(stage_start, stage_end, inner_block):
            end = min(start + inner_block, stage_end)
            diag = system[:, start:end, start:end]
            solved = torch.linalg.solve_triangular(diag, out[:, start:end, :], upper=False)
            out[:, start:end, :] = solved
            if end < stage_end:
                out[:, end:stage_end, :] = out[:, end:stage_end, :] - torch.matmul(system[:, end:stage_end, start:end], solved)
        if stage_end < n:
            out[:, stage_end:, :] = out[:, stage_end:, :] - torch.matmul(system[:, stage_end:, stage_start:stage_end], out[:, stage_start:stage_end, :])
    return out


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={"/data": trace_volume})
def run_staged_block_solve_case(seq_len: int, num_heads: int, rhs_dim: int, stage_block: int, inner_block: int, iterations: int, warmup_iterations: int, seed: int):
    import time
    import torch

    torch.manual_seed(seed)
    dtype = torch.float32
    base = 0.01 * torch.randn(num_heads, seq_len, seq_len, dtype=dtype, device="cuda")
    lower = torch.tril(base, diagonal=-1)
    rhs = 0.1 * torch.randn(num_heads, seq_len, rhs_dim, dtype=dtype, device="cuda")
    identity = torch.eye(seq_len, dtype=dtype, device="cuda").expand(num_heads, -1, -1)

    def _sync():
        torch.cuda.synchronize("cuda")

    def _time_ms(fn):
        start = time.perf_counter()
        result = fn()
        _sync()
        return result, (time.perf_counter() - start) * 1000.0

    def torch_solve():
        return torch.linalg.solve_triangular(identity + lower, rhs, upper=False)

    expected = torch_solve()
    actual = _staged_block_lower_triangular_solve(lower, rhs, stage_block=stage_block, inner_block=inner_block)
    correctness = {
        'finite_ok': bool(torch.isfinite(actual).all().item()),
        'max_abs_diff': float((actual - expected).abs().max().item()),
    }

    torch_samples = []
    staged_samples = []
    for _ in range(max(1, warmup_iterations)):
        torch_solve()
        _staged_block_lower_triangular_solve(lower, rhs, stage_block=stage_block, inner_block=inner_block)
        _sync()

    for _ in range(max(1, iterations)):
        _, t_ms = _time_ms(torch_solve)
        _, s_ms = _time_ms(lambda: _staged_block_lower_triangular_solve(lower, rhs, stage_block=stage_block, inner_block=inner_block))
        torch_samples.append(t_ms)
        staged_samples.append(s_ms)

    torch_summary = summarize(torch_samples)
    staged_summary = summarize(staged_samples)
    return {
        'config': {
            'seq_len': seq_len,
            'num_heads': num_heads,
            'rhs_dim': rhs_dim,
            'stage_block': stage_block,
            'inner_block': inner_block,
            'iterations': iterations,
            'warmup_iterations': warmup_iterations,
            'seed': seed,
        },
        'correctness': correctness,
        'timing': {
            'torch_solve_ms': torch_summary,
            'staged_block_solve_ms': staged_summary,
            'speedup_vs_torch_solve': torch_summary['mean_ms'] / staged_summary['mean_ms'] if torch_summary['mean_ms'] and staged_summary['mean_ms'] else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with app.run():
        result = run_staged_block_solve_case.remote(args.seq_len, args.num_heads, args.rhs_dim, args.stage_block, args.inner_block, args.iterations, args.warmup_iterations, args.seed)
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "speedup": result['timing']['speedup_vs_torch_solve']}))
    return 0


@app.local_entrypoint()
def modal_main(seq_len: int = 4096, num_heads: int = 4, rhs_dim: int = 128, stage_block: int = 192, inner_block: int = 64, iterations: int = 8, warmup_iterations: int = 2, seed: int = 0, output_json: str = ''):
    argv = [
        '--seq-len', str(seq_len),
        '--num-heads', str(num_heads),
        '--rhs-dim', str(rhs_dim),
        '--stage-block', str(stage_block),
        '--inner-block', str(inner_block),
        '--iterations', str(iterations),
        '--warmup-iterations', str(warmup_iterations),
        '--seed', str(seed),
        '--output-json', output_json,
    ]
    args = parse_args(argv)
    result = run_staged_block_solve_case.remote(args.seq_len, args.num_heads, args.rhs_dim, args.stage_block, args.inner_block, args.iterations, args.warmup_iterations, args.seed)
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "speedup": result['timing']['speedup_vs_torch_solve']}))


if __name__ == '__main__':
    raise SystemExit(main())
