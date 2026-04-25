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

app = modal.App("flashinfer-gdn-triton-chunk-output-prototype")
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
    parser = argparse.ArgumentParser(description="Run a Triton prototype for chunk-parallel GDN output reconstruction on Modal B200.")
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


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={"/data": trace_volume})
def run_triton_chunk_output_case(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    iterations: int,
    warmup_iterations: int,
    seed: int,
):
    import torch
    import triton
    import triton.language as tl

    torch.manual_seed(seed)
    dtype = torch.float32
    q = 0.1 * torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    k = 0.1 * torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    v = 0.1 * torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    gate = 0.95 + 0.05 * torch.sigmoid(torch.randn(seq_len, num_heads, dtype=dtype, device="cuda"))
    beta = 0.25 * torch.sigmoid(torch.randn(seq_len, num_heads, dtype=dtype, device="cuda"))
    initial_state = 0.1 * torch.randn(num_heads, head_dim, head_dim, dtype=dtype, device="cuda")

    c, w, u = _build_parallel_chunk_factors_batched(k, v, gate, beta)
    q_h = q.permute(1, 0, 2)
    k_h = k.permute(1, 0, 2)
    w_h = w.permute(1, 0, 2)
    u_h = u.permute(1, 0, 2)
    c_h = c.permute(1, 0)
    base_h = torch.matmul(q_h, initial_state.transpose(-1, -2))
    rhs_h = u_h - torch.matmul(w_h, initial_state.transpose(-1, -2))
    base = base_h.permute(1, 0, 2).contiguous()
    rhs = rhs_h.permute(1, 0, 2).contiguous()
    c_contig = c.contiguous()
    q_contig = q.contiguous()
    k_contig = k.contiguous()

    def torch_reference_output():
        causal_qk = torch.tril(torch.matmul(q_h, k_h.transpose(-1, -2)))
        corr = torch.matmul(causal_qk, rhs_h)
        return (c_h.unsqueeze(-1) * (base_h + corr)).permute(1, 0, 2).contiguous()

    @triton.jit
    def _kernel(
        q_ptr,
        k_ptr,
        rhs_ptr,
        c_ptr,
        base_ptr,
        out_ptr,
        stride_q_t,
        stride_q_h,
        stride_q_d,
        stride_k_t,
        stride_k_h,
        stride_k_d,
        stride_rhs_t,
        stride_rhs_h,
        stride_rhs_d,
        stride_c_t,
        stride_c_h,
        stride_base_t,
        stride_base_h,
        stride_base_d,
        stride_out_t,
        stride_out_h,
        stride_out_d,
        seq_len,
        head_dim,
        dv_dim,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_dv = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_dk = tl.arange(0, BLOCK_DMODEL)
        offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)

        mask_m = offs_m < seq_len
        mask_dk = offs_dk < head_dim
        mask_dv = offs_dv < dv_dim

        q_ptrs = q_ptr + offs_m[:, None] * stride_q_t + pid_h * stride_q_h + offs_dk[None, :] * stride_q_d
        q_block = tl.load(q_ptrs, mask=mask_m[:, None] & mask_dk[None, :], other=0.0)

        base_ptrs = base_ptr + offs_m[:, None] * stride_base_t + pid_h * stride_base_h + offs_dv[None, :] * stride_base_d
        acc = tl.load(base_ptrs, mask=mask_m[:, None] & mask_dv[None, :], other=0.0).to(tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len
            k_ptrs = k_ptr + offs_n[:, None] * stride_k_t + pid_h * stride_k_h + offs_dk[None, :] * stride_k_d
            k_block = tl.load(k_ptrs, mask=mask_n[:, None] & mask_dk[None, :], other=0.0)
            logits = tl.dot(q_block, tl.trans(k_block))
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            logits = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], logits, 0.0)

            rhs_ptrs = rhs_ptr + offs_n[:, None] * stride_rhs_t + pid_h * stride_rhs_h + offs_dv[None, :] * stride_rhs_d
            rhs_block = tl.load(rhs_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
            acc += tl.dot(logits.to(tl.float32), rhs_block.to(tl.float32))

        c_ptrs = c_ptr + offs_m * stride_c_t + pid_h * stride_c_h
        c_vals = tl.load(c_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        acc = acc * c_vals[:, None]

        out_ptrs = out_ptr + offs_m[:, None] * stride_out_t + pid_h * stride_out_h + offs_dv[None, :] * stride_out_d
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_dv[None, :])

    def triton_output():
        out = torch.empty_like(base)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_DV = 32
        BLOCK_DMODEL = triton.next_power_of_2(head_dim)
        grid = (
            triton.cdiv(seq_len, BLOCK_M),
            num_heads,
            triton.cdiv(head_dim, BLOCK_DV),
        )
        _kernel[grid](
            q_contig,
            k_contig,
            rhs,
            c_contig,
            base,
            out,
            q_contig.stride(0),
            q_contig.stride(1),
            q_contig.stride(2),
            k_contig.stride(0),
            k_contig.stride(1),
            k_contig.stride(2),
            rhs.stride(0),
            rhs.stride(1),
            rhs.stride(2),
            c_contig.stride(0),
            c_contig.stride(1),
            base.stride(0),
            base.stride(1),
            base.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            seq_len,
            head_dim,
            head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DV=BLOCK_DV,
        )
        return out

    def _sync():
        torch.cuda.synchronize("cuda")

    def _time_ms(fn):
        start = time.perf_counter()
        result = fn()
        _sync()
        return result, (time.perf_counter() - start) * 1000.0

    expected = torch_reference_output()
    actual = triton_output()
    correctness = {
        "finite_ok": bool(torch.isfinite(actual).all().item()),
        "max_output_abs_diff": float((actual - expected).abs().max().item()),
    }

    torch_samples = []
    triton_samples = []
    for _ in range(max(1, warmup_iterations)):
        torch_reference_output()
        triton_output()
        _sync()

    for _ in range(max(1, iterations)):
        _, torch_ms = _time_ms(torch_reference_output)
        _, triton_ms = _time_ms(triton_output)
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
            "head_dim": head_dim,
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "seed": seed,
        },
        "correctness": correctness,
        "timing": {
            "torch_chunk_output_ms": torch_summary,
            "triton_chunk_output_ms": triton_summary,
            "speedup_vs_torch_chunk_output": speedup,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with app.run():
        result = run_triton_chunk_output_case.remote(
            args.seq_len,
            args.num_heads,
            args.head_dim,
            args.iterations,
            args.warmup_iterations,
            args.seed,
        )
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "speedup": result["timing"]["speedup_vs_torch_chunk_output"]}, indent=2))
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
    result = run_triton_chunk_output_case.remote(
        args.seq_len,
        args.num_heads,
        args.head_dim,
        args.iterations,
        args.warmup_iterations,
        args.seed,
    )
    _write_output(args.output_json, result)
    print(json.dumps({"output_json": str(args.output_json), "speedup": result["timing"]["speedup_vs_torch_chunk_output"]}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
