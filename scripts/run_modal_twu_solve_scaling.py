from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

TRACE_VOLUME_NAME = "mlsys26-contest"
BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

app = modal.App("flashinfer-gdn-twu-solve-scaling")
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
    parser = argparse.ArgumentParser(description="Run Modal B200 scaling study for TWU triangular solve.")
    parser.add_argument("--seq-lens", required=True, help="Comma-separated sequence lengths")
    parser.add_argument("--rhs-dims", required=True, help="Comma-separated rhs dims")
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
def run_twu_solve_scaling_case(seq_lens: list[int], rhs_dims: list[int], num_heads: int, iterations: int, warmup_iterations: int, seed: int):
    import time
    import torch

    rows = []
    for seq_len in seq_lens:
        for rhs_dim in rhs_dims:
            torch.manual_seed(seed + seq_len * 1000 + rhs_dim)
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

            def _solve():
                return torch.linalg.solve_triangular(identity + lower, rhs, upper=False)

            for _ in range(max(1, warmup_iterations)):
                _solve()
                _sync()

            samples = []
            for _ in range(max(1, iterations)):
                _, t_ms = _time_ms(_solve)
                samples.append(t_ms)

            rows.append(
                {
                    "seq_len": seq_len,
                    "rhs_dim": rhs_dim,
                    "num_heads": num_heads,
                    "timing": summarize(samples),
                }
            )
    return {"rows": rows}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    rhs_dims = [int(x.strip()) for x in args.rhs_dims.split(",") if x.strip()]
    with app.run():
        result = run_twu_solve_scaling_case.remote(seq_lens, rhs_dims, args.num_heads, args.iterations, args.warmup_iterations, args.seed)
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "cases": len(result["rows"])}))
    return 0


@app.local_entrypoint()
def modal_main(seq_lens: str = "128,256,512", rhs_dims: str = "64,128,256", num_heads: int = 4, iterations: int = 20, warmup_iterations: int = 3, seed: int = 0, output_json: str = ""):
    argv = [
        "--seq-lens", seq_lens,
        "--rhs-dims", rhs_dims,
        "--num-heads", str(num_heads),
        "--iterations", str(iterations),
        "--warmup-iterations", str(warmup_iterations),
        "--seed", str(seed),
        "--output-json", output_json,
    ]
    args = parse_args(argv)
    seq_lens_list = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    rhs_dims_list = [int(x.strip()) for x in args.rhs_dims.split(",") if x.strip()]
    result = run_twu_solve_scaling_case.remote(
        seq_lens_list,
        rhs_dims_list,
        args.num_heads,
        args.iterations,
        args.warmup_iterations,
        args.seed,
    )
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "cases": len(result["rows"])}))


if __name__ == "__main__":
    raise SystemExit(main())
