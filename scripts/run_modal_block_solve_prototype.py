from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal
try:
    from scripts.block_solve_policy import choose_default_or_tuned_block_tile
except ModuleNotFoundError:
    def choose_default_or_tuned_block_tile(*, num_heads: int, rhs_dim: int, enable_shape_tuning: bool) -> int:
        if not enable_shape_tuning:
            return 192
        if num_heads <= 2 or rhs_dim >= 512:
            return 224
        if num_heads >= 8 or rhs_dim <= 64:
            return 160
        return 192

TRACE_VOLUME_NAME = "mlsys26-contest"
BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

app = modal.App("flashinfer-gdn-block-solve-prototype")
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
    parser = argparse.ArgumentParser(description="Run a Modal B200 prototype for block-structured TWU solve.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rhs-dim", type=int, default=128)
    parser.add_argument("--block-sizes", default="", help="Comma-separated block sizes")
    parser.add_argument("--use-default-policy", action="store_true", help="Use the practical default block policy (flat 192 unless tuning is enabled).")
    parser.add_argument("--enable-shape-tuning", action="store_true", help="When using policy mode, enable the 160/224 shape-aware fallbacks.")
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


def resolve_block_sizes(*, block_sizes_csv: str, use_default_policy: bool, enable_shape_tuning: bool, num_heads: int, rhs_dim: int) -> list[int]:
    if use_default_policy:
        return [
            choose_default_or_tuned_block_tile(
                num_heads=num_heads,
                rhs_dim=rhs_dim,
                enable_shape_tuning=enable_shape_tuning,
            )
        ]
    if not block_sizes_csv.strip():
        raise ValueError("block-sizes must be provided unless --use-default-policy is set")
    return [int(x.strip()) for x in block_sizes_csv.split(",") if x.strip()]


def _block_lower_triangular_solve(lower, rhs, *, block_size):
    import torch

    h, n, _ = lower.shape
    identity = torch.eye(n, dtype=lower.dtype, device=lower.device).expand(h, -1, -1)
    system = identity + lower
    out = rhs.clone()
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        diag = system[:, start:end, start:end]
        solved = torch.linalg.solve_triangular(diag, out[:, start:end, :], upper=False)
        out[:, start:end, :] = solved
        if end < n:
            out[:, end:, :] = out[:, end:, :] - torch.matmul(system[:, end:, start:end], solved)
    return out


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={"/data": trace_volume})
def run_block_solve_case(seq_len: int, num_heads: int, rhs_dim: int, block_sizes: list[int], iterations: int, warmup_iterations: int, seed: int):
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
    rows = []
    for block_size in block_sizes:
        actual = _block_lower_triangular_solve(lower, rhs, block_size=block_size)
        correctness = {
            "finite_ok": bool(torch.isfinite(actual).all().item()),
            "max_abs_diff": float((actual - expected).abs().max().item()),
        }

        torch_samples = []
        block_samples = []
        for _ in range(max(1, warmup_iterations)):
            torch_solve()
            _block_lower_triangular_solve(lower, rhs, block_size=block_size)
            _sync()

        for _ in range(max(1, iterations)):
            _, t_ms = _time_ms(torch_solve)
            _, b_ms = _time_ms(lambda: _block_lower_triangular_solve(lower, rhs, block_size=block_size))
            torch_samples.append(t_ms)
            block_samples.append(b_ms)

        torch_summary = summarize(torch_samples)
        block_summary = summarize(block_samples)
        rows.append(
            {
                "block_size": block_size,
                "correctness": correctness,
                "timing": {
                    "torch_solve_ms": torch_summary,
                    "block_solve_ms": block_summary,
                    "speedup_vs_torch_solve": (
                        torch_summary["mean_ms"] / block_summary["mean_ms"]
                        if torch_summary["mean_ms"] and block_summary["mean_ms"]
                        else None
                    ),
                },
            }
        )
    return {"config": {"seq_len": seq_len, "num_heads": num_heads, "rhs_dim": rhs_dim, "iterations": iterations, "warmup_iterations": warmup_iterations, "seed": seed}, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    block_sizes = resolve_block_sizes(
        block_sizes_csv=args.block_sizes,
        use_default_policy=args.use_default_policy,
        enable_shape_tuning=args.enable_shape_tuning,
        num_heads=args.num_heads,
        rhs_dim=args.rhs_dim,
    )
    with app.run():
        result = run_block_solve_case.remote(args.seq_len, args.num_heads, args.rhs_dim, block_sizes, args.iterations, args.warmup_iterations, args.seed)
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "cases": len(result['rows'])}))
    return 0


@app.local_entrypoint()
def modal_main(seq_len: int = 512, num_heads: int = 4, rhs_dim: int = 128, block_sizes: str = '16,32,64,128', use_default_policy: bool = False, enable_shape_tuning: bool = False, iterations: int = 20, warmup_iterations: int = 3, seed: int = 0, output_json: str = ''):
    argv = [
        '--seq-len', str(seq_len),
        '--num-heads', str(num_heads),
        '--rhs-dim', str(rhs_dim),
        '--iterations', str(iterations),
        '--warmup-iterations', str(warmup_iterations),
        '--seed', str(seed),
        '--output-json', output_json,
    ]
    if use_default_policy:
        argv.append('--use-default-policy')
    if enable_shape_tuning:
        argv.append('--enable-shape-tuning')
    if block_sizes:
        argv.extend(['--block-sizes', block_sizes])
    args = parse_args(argv)
    block_sizes_list = resolve_block_sizes(
        block_sizes_csv=args.block_sizes,
        use_default_policy=args.use_default_policy,
        enable_shape_tuning=args.enable_shape_tuning,
        num_heads=args.num_heads,
        rhs_dim=args.rhs_dim,
    )
    result = run_block_solve_case.remote(args.seq_len, args.num_heads, args.rhs_dim, block_sizes_list, args.iterations, args.warmup_iterations, args.seed)
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "cases": len(result['rows'])}))


if __name__ == '__main__':
    raise SystemExit(main())
