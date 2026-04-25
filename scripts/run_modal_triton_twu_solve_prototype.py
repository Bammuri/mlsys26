from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

TRACE_VOLUME_NAME = "mlsys26-contest"
BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

app = modal.App("flashinfer-gdn-triton-twu-solve-prototype")
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
    parser = argparse.ArgumentParser(description="Run a Triton prototype for chunk-local triangular solve on Modal B200.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rhs-dim", type=int, default=128)
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
def run_triton_twu_solve_case(
    seq_len: int,
    num_heads: int,
    rhs_dim: int,
    iterations: int,
    warmup_iterations: int,
    seed: int,
):
    import time
    import torch
    import triton
    import triton.language as tl

    torch.manual_seed(seed)
    dtype = torch.float32
    # Build a stable strictly-lower matrix similar to TWU solve input.
    base = 0.01 * torch.randn(num_heads, seq_len, seq_len, dtype=dtype, device="cuda")
    lower = torch.tril(base, diagonal=-1)
    rhs = 0.1 * torch.randn(num_heads, seq_len, rhs_dim, dtype=dtype, device="cuda")

    @triton.jit
    def _solve_kernel(
        lower_ptr,
        rhs_ptr,
        out_ptr,
        stride_l_h,
        stride_l_i,
        stride_l_j,
        stride_b_h,
        stride_b_i,
        stride_b_d,
        stride_o_h,
        stride_o_i,
        stride_o_d,
        rhs_dim,
        SEQ_LEN: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < rhs_dim

        for i in range(SEQ_LEN):
            b_ptrs = rhs_ptr + pid_h * stride_b_h + i * stride_b_i + offs_d * stride_b_d
            acc = tl.load(b_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            for j in range(i):
                lij_ptr = lower_ptr + pid_h * stride_l_h + i * stride_l_i + j * stride_l_j
                lij = tl.load(lij_ptr)
                prev_ptrs = out_ptr + pid_h * stride_o_h + j * stride_o_i + offs_d * stride_o_d
                prev = tl.load(prev_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                acc -= lij * prev
            out_ptrs = out_ptr + pid_h * stride_o_h + i * stride_o_i + offs_d * stride_o_d
            tl.store(out_ptrs, acc, mask=mask_d)

    def torch_reference():
        identity = torch.eye(seq_len, dtype=dtype, device="cuda").expand(num_heads, -1, -1)
        return torch.linalg.solve_triangular(identity + lower, rhs, upper=False)

    def triton_reference():
        out = torch.empty_like(rhs)
        BLOCK_D = 32
        grid = (num_heads, triton.cdiv(rhs_dim, BLOCK_D))
        _solve_kernel[grid](
            lower,
            rhs,
            out,
            lower.stride(0),
            lower.stride(1),
            lower.stride(2),
            rhs.stride(0),
            rhs.stride(1),
            rhs.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            rhs_dim,
            SEQ_LEN=seq_len,
            BLOCK_D=BLOCK_D,
        )
        return out

    def _sync():
        torch.cuda.synchronize("cuda")

    def _time_ms(fn):
        start = time.perf_counter()
        result = fn()
        _sync()
        return result, (time.perf_counter() - start) * 1000.0

    expected = torch_reference()
    actual = triton_reference()
    correctness = {
        "finite_ok": bool(torch.isfinite(actual).all().item()),
        "max_abs_diff": float((actual - expected).abs().max().item()),
    }

    torch_samples = []
    triton_samples = []
    for _ in range(max(1, warmup_iterations)):
        torch_reference()
        triton_reference()
        _sync()

    for _ in range(max(1, iterations)):
        _, torch_ms = _time_ms(torch_reference)
        _, triton_ms = _time_ms(triton_reference)
        torch_samples.append(torch_ms)
        triton_samples.append(triton_ms)

    torch_summary = summarize_timings(torch_samples)
    triton_summary = summarize_timings(triton_samples)
    speedup = (
        torch_summary["mean_ms"] / triton_summary["mean_ms"]
        if torch_summary["mean_ms"] and triton_summary["mean_ms"]
        else None
    )
    return {
        "config": {
            "seq_len": seq_len,
            "num_heads": num_heads,
            "rhs_dim": rhs_dim,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "seed": seed,
        },
        "correctness": correctness,
        "timing": {
            "torch_solve_ms": torch_summary,
            "triton_solve_ms": triton_summary,
            "speedup_vs_torch_solve": speedup,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with app.run():
        result = run_triton_twu_solve_case.remote(
            args.seq_len,
            args.num_heads,
            args.rhs_dim,
            args.iterations,
            args.warmup_iterations,
            args.seed,
        )
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "speedup": result["timing"]["speedup_vs_torch_solve"]}, indent=2))
    return 0


@app.local_entrypoint()
def modal_main(
    seq_len: int = 128,
    num_heads: int = 4,
    rhs_dim: int = 128,
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
        "--rhs-dim",
        str(rhs_dim),
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
    result = run_triton_twu_solve_case.remote(
        args.seq_len,
        args.num_heads,
        args.rhs_dim,
        args.iterations,
        args.warmup_iterations,
        args.seed,
    )
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "speedup": result["timing"]["speedup_vs_torch_solve"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
