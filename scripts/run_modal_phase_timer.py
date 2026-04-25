from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import modal
from flashinfer_bench import Solution, TraceSet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRACE_VOLUME_NAME = "mlsys26-contest"
TRACE_SET_PATH = "/data/data/mlsys26-contest"
BASE_PATH = "/opt/nvidia/nsight-compute:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PACKED_SUBMIT_SOLUTION_SOURCE = "packed-submit"
TRACE_SET_BASELINE_SOLUTION_SOURCE = "trace-set-baseline"
app = modal.App("flashinfer-prefill-phase-timer")
trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "PATH": BASE_PATH,
            "FIB_DATASET_PATH": TRACE_SET_PATH,
        }
    )
    .pip_install("flashinfer-bench", "flashinfer-python", "torch", "triton", "numpy")
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct phase timing for a single prefill workload on Modal B200.")
    parser.add_argument("--trace-set-path", type=Path, default=None)
    parser.add_argument("--definition", default="gdn_prefill_qk4_v8_d128_k_last")
    parser.add_argument(
        "--solution-source",
        choices=(PACKED_SUBMIT_SOLUTION_SOURCE, TRACE_SET_BASELINE_SOLUTION_SOURCE),
        default=PACKED_SUBMIT_SOLUTION_SOURCE,
    )
    parser.add_argument("--baseline-solution-index", type=int, default=0)
    parser.add_argument("--workload-uuid", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _entry_module_name(entry_point: str) -> str:
    module_part, _, _ = entry_point.partition("::")
    module_part = module_part.removesuffix(".py")
    return module_part.replace("/", ".")


def summarize_samples(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"mean_us": None, "median_us": None, "min_us": None, "max_us": None, "count": 0}
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "mean_us": sum(samples) / len(samples),
        "median_us": median,
        "min_us": min(samples),
        "max_us": max(samples),
        "count": len(samples),
    }


def _trace_set_candidates(trace_set_path: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if trace_set_path is not None:
        candidates.append(trace_set_path)
    else:
        candidates.append(Path(TRACE_SET_PATH))
    local_fallback = Path("/home/hyu/flashinfer/mlsys26-contest")
    if local_fallback not in candidates:
        candidates.append(local_fallback)
    return [candidate for candidate in candidates if candidate.exists()]


def _resolve_trace_set_root(trace_set_path: Path | None, *, definition: str) -> Path:
    first_existing: Path | None = None
    for candidate in _trace_set_candidates(trace_set_path):
        if first_existing is None:
            first_existing = candidate
        trace_set = TraceSet.from_path(str(candidate))
        if definition in trace_set.workloads or definition in trace_set.solutions:
            return candidate
    if first_existing is not None:
        return first_existing
    raise FileNotFoundError("No trace-set path exists for workload resolution")


def _solution_provenance_payload(
    solution: Solution,
    *,
    solution_source: str,
    trace_set_path: Path | None,
    baseline_solution_index: int,
) -> dict[str, Any]:
    return {
        "solution_source": solution_source,
        "solution_name": solution.name,
        "solution_author": solution.author,
        "entry_point": solution.spec.entry_point,
        "destination_passing_style": solution.spec.destination_passing_style,
        "trace_set_path": str(trace_set_path) if trace_set_path is not None else None,
        "baseline_solution_index": baseline_solution_index
        if solution_source == TRACE_SET_BASELINE_SOLUTION_SOURCE
        else None,
    }


def _load_solution_payload(
    *,
    definition: str,
    solution_source: str,
    trace_set_path: Path | None,
    baseline_solution_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if solution_source == PACKED_SUBMIT_SOLUTION_SOURCE:
        from scripts.pack_solution import pack_solution

        solution_path = pack_solution(PROJECT_ROOT / "solution.json")
        solution = Solution.model_validate_json(solution_path.read_text())
        provenance = _solution_provenance_payload(
            solution,
            solution_source=solution_source,
            trace_set_path=None,
            baseline_solution_index=baseline_solution_index,
        )
        provenance["packed_solution_json"] = str(solution_path)
        return solution.model_dump(mode="json"), provenance

    if solution_source == TRACE_SET_BASELINE_SOLUTION_SOURCE:
        resolved_path = _resolve_trace_set_root(trace_set_path, definition=definition)
        trace_set = TraceSet.from_path(str(resolved_path))
        solutions = trace_set.solutions.get(definition, [])
        if baseline_solution_index < 0 or baseline_solution_index >= len(solutions):
            raise ValueError(
                f"baseline solution index {baseline_solution_index} out of range for definition "
                f"{definition!r} (available={len(solutions)})"
            )
        solution = solutions[baseline_solution_index]
        return solution.model_dump(mode="json"), _solution_provenance_payload(
            solution,
            solution_source=solution_source,
            trace_set_path=resolved_path,
            baseline_solution_index=baseline_solution_index,
        )

    raise ValueError(f"Unsupported solution_source: {solution_source}")


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={"/data": trace_volume})
def time_prefill_workload(
    solution_payload: dict[str, Any],
    definition_name: str,
    workload_uuid: str,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    import os

    import torch
    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    solution = Solution.model_validate(solution_payload)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[definition_name]
    workload_map = {trace.workload.uuid: trace.workload for trace in trace_set.workloads.get(definition_name, [])}
    workload = workload_map[workload_uuid]

    safe_tensors = (
        load_safetensors(definition, workload, trace_set.root)
        if any(spec.type == "safetensors" for spec in workload.inputs.values())
        else {}
    )
    inputs = gen_inputs(definition, workload, device="cuda:0", safe_tensors=safe_tensors)

    with tempfile.TemporaryDirectory(prefix="fib_phase_timer_") as tmpdir:
        tmp_root = Path(tmpdir)
        package_name = "fib_solution_runtime"
        package_root = tmp_root / package_name
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text("")
        for source in solution.sources:
            source_path = package_root / source.path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(source.content, encoding="utf-8")
            for parent in source_path.parents:
                if parent == package_root:
                    break
                init_path = parent / "__init__.py"
                if not init_path.exists():
                    init_path.write_text("")

        sys.path.insert(0, str(tmp_root))
        try:
            module = importlib.import_module(f"{package_name}.{_entry_module_name(solution.spec.entry_point)}")

            def _sync() -> None:
                torch.cuda.synchronize("cuda:0")

            def _time_block(fn) -> tuple[Any, float]:
                started = time.perf_counter()
                result = fn()
                _sync()
                return result, (time.perf_counter() - started) * 1_000_000.0

            is_submit_style = bool(getattr(solution.spec, "destination_passing_style", False))

            def _call_run_once() -> None:
                if is_submit_style:
                    outputs = allocate_outputs(definition, inputs, "cuda:0")
                    module.run(*inputs, *outputs)
                else:
                    module.run(*inputs)

            for _ in range(max(1, warmup_iterations)):
                _call_run_once()
                _sync()

            samples: dict[str, list[float]] = {
                "total_run_us": [],
                "allocate_outputs_us": [],
                "prepare_gate_beta_us": [],
                "problem_size_us": [],
                "compile_lookup_us": [],
                "launch_sync_us": [],
            }

            if is_submit_style:
                q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale = inputs
                for _ in range(max(1, iterations)):
                    _, allocate_us = _time_block(lambda: allocate_outputs(definition, inputs, "cuda:0"))
                    outputs = allocate_outputs(definition, inputs, "cuda:0")
                    output, new_state = outputs

                    total_started = time.perf_counter()
                    (g, beta), prepare_us = _time_block(lambda: module._get_gate_beta(A_log, a, dt_bias, b))
                    scale_value = module._normalize_scale(scale, q.shape[-1])
                    varlen = cu_seqlens is not None and q.dim() == 3
                    q_runtime = q.unsqueeze(0) if varlen else q
                    k_runtime = k.unsqueeze(0) if varlen else k
                    v_runtime = v.unsqueeze(0) if varlen else v
                    output_runtime = module._prepare_output_view(output, varlen=varlen)
                    problem_size, problem_us = _time_block(lambda: module._get_problem_size_cached(q_runtime, v_runtime, cu_seqlens))
                    is_persistent = module._resolve_persistent_mode(problem_size, varlen=varlen)
                    compiled_gdn, compile_us = _time_block(
                        lambda: module._get_compiled_runner(
                            q_runtime,
                            k_runtime,
                            v_runtime,
                            g,
                            beta,
                            state,
                            cu_seqlens,
                            output_runtime,
                            new_state,
                            scale_value,
                            problem_size=problem_size,
                            is_persistent=is_persistent,
                        )
                    )
                    current_stream = module.cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                    _, launch_us = _time_block(
                        lambda: compiled_gdn(
                            q_runtime.data_ptr(),
                            k_runtime.data_ptr(),
                            v_runtime.data_ptr(),
                            output_runtime.data_ptr(),
                            g.data_ptr(),
                            beta.data_ptr(),
                            problem_size,
                            state.data_ptr() if state is not None else None,
                            new_state.data_ptr(),
                            scale_value,
                            cu_seqlens,
                            stream=current_stream,
                        )
                    )
                    total_us = (time.perf_counter() - total_started) * 1_000_000.0
                    samples["allocate_outputs_us"].append(allocate_us)
                    samples["prepare_gate_beta_us"].append(prepare_us)
                    samples["problem_size_us"].append(problem_us)
                    samples["compile_lookup_us"].append(compile_us)
                    samples["launch_sync_us"].append(launch_us)
                    samples["total_run_us"].append(total_us)
            else:
                q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale = inputs

                def _baseline_prepare_gate_beta():
                    x = a.float() + dt_bias.float()
                    g = -torch.exp(A_log.float()) * torch.nn.functional.softplus(x)
                    beta = torch.sigmoid(b.float())
                    return g, beta

                for _ in range(max(1, iterations)):
                    total_started = time.perf_counter()
                    (g, beta), prepare_us = _time_block(_baseline_prepare_gate_beta)
                    scale_value = None if scale is None else float(scale.item() if isinstance(scale, torch.Tensor) else scale)
                    varlen = cu_seqlens is not None and q.dim() == 3
                    q_runtime = q.unsqueeze(0) if varlen else q
                    k_runtime = k.unsqueeze(0) if varlen else k
                    v_runtime = v.unsqueeze(0) if varlen else v
                    _, launch_us = _time_block(
                        lambda: module.chunk_gated_delta_rule(
                            q=q_runtime,
                            k=k_runtime,
                            v=v_runtime,
                            g=g,
                            beta=beta,
                            scale=scale_value,
                            initial_state=state,
                            output_final_state=True,
                            cu_seqlens=cu_seqlens,
                            use_qk_l2norm_in_kernel=False,
                        )
                    )
                    total_us = (time.perf_counter() - total_started) * 1_000_000.0
                    samples["prepare_gate_beta_us"].append(prepare_us)
                    samples["launch_sync_us"].append(launch_us)
                    samples["total_run_us"].append(total_us)

            return {
                "solution": {
                    "name": solution.name,
                    "author": solution.author,
                    "entry_point": solution.spec.entry_point,
                    "destination_passing_style": solution.spec.destination_passing_style,
                },
                "workload": {
                    "uuid": workload.uuid,
                    "axes": dict(sorted(workload.axes.items())),
                },
                "iterations": iterations,
                "warmup_iterations": warmup_iterations,
                "metrics": {name: summarize_samples(values) for name, values in samples.items() if values},
            }
        finally:
            sys.path = [entry for entry in sys.path if entry != str(tmp_root)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    solution_payload, solution_provenance = _load_solution_payload(
        definition=args.definition,
        solution_source=args.solution_source,
        trace_set_path=args.trace_set_path,
        baseline_solution_index=args.baseline_solution_index,
    )
    with app.run():
        result = time_prefill_workload.remote(
            solution_payload,
            args.definition,
            args.workload_uuid,
            args.iterations,
            args.warmup_iterations,
        )
    payload = {
        "metadata": {
            "definition": args.definition,
            "solution_source": args.solution_source,
            "solution_provenance": solution_provenance,
        },
        "result": result,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "workload_uuid": args.workload_uuid}, indent=2))
    return 0


@app.local_entrypoint()
def modal_main(
    trace_set_path: str = "",
    definition: str = "gdn_prefill_qk4_v8_d128_k_last",
    solution_source: str = PACKED_SUBMIT_SOLUTION_SOURCE,
    baseline_solution_index: int = 0,
    workload_uuid: str = "",
    iterations: int = 5,
    warmup_iterations: int = 2,
    output_json: str = "",
):
    argv = [
        "--definition",
        definition,
        "--solution-source",
        solution_source,
        "--baseline-solution-index",
        str(baseline_solution_index),
        "--workload-uuid",
        workload_uuid,
        "--iterations",
        str(iterations),
        "--warmup-iterations",
        str(warmup_iterations),
        "--output-json",
        output_json,
    ]
    if trace_set_path:
        argv.extend(["--trace-set-path", trace_set_path])
    raise SystemExit(main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
