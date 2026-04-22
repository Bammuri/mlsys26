"""
Targeted workload profiler for the GDN prefill submission.

Supports:
- Perfetto-compatible Chrome traces via torch.profiler
- Nsight Compute text capture via flashinfer_bench_run_ncu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from flashinfer_bench import Solution, TraceSet, Workload
from flashinfer_bench.agents import flashinfer_bench_run_ncu
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pack_solution import pack_solution


def _load_solution(solution_path: Path | None) -> Solution:
    if solution_path is None:
        solution_path = pack_solution()
    return Solution.model_validate_json(Path(solution_path).read_text())


def _resolve_trace_set_path(trace_set_path: str | None) -> str | None:
    if trace_set_path:
        return trace_set_path
    env_path = os.environ.get("FIB_DATASET_PATH")
    if env_path:
        return env_path
    return None


def _resolve_workload(
    solution: Solution,
    *,
    workload_uuid: str | None,
    workload_path: Path | None,
    trace_set_path: str | None,
) -> tuple[Workload, TraceSet]:
    resolved_trace_set_path = _resolve_trace_set_path(trace_set_path)
    if workload_path is not None:
        trace_set = TraceSet.from_path(resolved_trace_set_path)
        return Workload.model_validate_json(workload_path.read_text()), trace_set
    if workload_uuid is None:
        raise ValueError("Either --workload-uuid or --workload-path must be provided")
    trace_set = TraceSet.from_path(resolved_trace_set_path)
    workload_traces = trace_set.workloads.get(solution.definition, [])
    for workload_trace in workload_traces:
        if workload_trace.workload.uuid == workload_uuid:
            return workload_trace.workload, trace_set
    raise ValueError(f"Workload UUID '{workload_uuid}' not found for definition '{solution.definition}'")


def _call_runnable(runnable, definition, inputs: list[Any], device: str) -> None:
    if runnable.metadata.destination_passing_style:
        outputs = allocate_outputs(definition, inputs, device)
        runnable(*inputs, *outputs)
        return
    runnable(*inputs)


def _run_perfetto_profile(
    solution: Solution,
    workload: Workload,
    trace_set: TraceSet,
    *,
    device: str,
    iterations: int,
    output_path: Path,
    enable_wrapper_phases: bool,
) -> None:
    definition = trace_set.definitions[solution.definition]
    safe_tensors = (
        load_safetensors(definition, workload, trace_set.root)
        if any(spec.type == "safetensors" for spec in workload.inputs.values())
        else {}
    )
    inputs = gen_inputs(definition, workload, device=device, safe_tensors=safe_tensors)
    runnable = BuilderRegistry.get_instance().build(definition, solution)

    previous_phase_flag = os.environ.get("MSINFER_GDN_PROFILE_PHASES")
    if enable_wrapper_phases:
        os.environ["MSINFER_GDN_PROFILE_PHASES"] = "1"

    try:
        _call_runnable(runnable, definition, inputs, device)
        torch.cuda.synchronize(device)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            for _ in range(iterations):
                _call_runnable(runnable, definition, inputs, device)
                torch.cuda.synchronize(device)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(output_path))
    finally:
        if enable_wrapper_phases:
            if previous_phase_flag is None:
                os.environ.pop("MSINFER_GDN_PROFILE_PHASES", None)
            else:
                os.environ["MSINFER_GDN_PROFILE_PHASES"] = previous_phase_flag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a specific GDN prefill workload")
    parser.add_argument("--solution-path", type=Path, default=None, help="Optional packed solution JSON path.")
    parser.add_argument("--trace-set-path", default=None, help="Optional trace set path (or rely on FIB_DATASET_PATH).")
    parser.add_argument("--workload-uuid", default=None, help="Workload UUID to profile.")
    parser.add_argument("--workload-path", type=Path, default=None, help="Optional standalone workload JSON path.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to run the profiler on.")
    parser.add_argument("--mode", choices=("perfetto", "ncu"), default="perfetto", help="Profiler backend.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to .omx/profiles/<uuid>.(json|txt).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of profiled iterations for perfetto mode.",
    )
    parser.add_argument(
        "--wrapper-phases",
        action="store_true",
        help="Enable msinfer_entry phase annotations so torch.profiler traces show wrapper phases.",
    )
    parser.add_argument("--ncu-set", default="detailed", help="NCU section set.")
    parser.add_argument("--ncu-page", default="details", help="NCU output page.")
    parser.add_argument("--kernel-name", default=None, help="Optional regex filter for a specific kernel.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout for NCU profiling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solution = _load_solution(args.solution_path)
    workload, trace_set = _resolve_workload(
        solution,
        workload_uuid=args.workload_uuid,
        workload_path=args.workload_path,
        trace_set_path=args.trace_set_path,
    )

    output_path = args.output
    if output_path is None:
        suffix = ".json" if args.mode == "perfetto" else ".txt"
        output_path = Path(".omx/profiles") / f"{workload.uuid}{suffix}"

    if args.mode == "perfetto":
        _run_perfetto_profile(
            solution,
            workload,
            trace_set,
            device=args.device,
            iterations=args.iterations,
            output_path=output_path,
            enable_wrapper_phases=args.wrapper_phases,
        )
        print(f"Saved Perfetto-compatible trace to {output_path}")
        return

    ncu_output = flashinfer_bench_run_ncu(
        solution,
        workload,
        device=args.device,
        trace_set_path=args.trace_set_path,
        set=args.ncu_set,
        page=args.ncu_page,
        kernel_name=args.kernel_name,
        timeout=args.timeout,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ncu_output)
    print(f"Saved NCU output to {output_path}")


if __name__ == "__main__":
    main()
