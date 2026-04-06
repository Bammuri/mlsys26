"""
Profile the GDN prefill solution with NVIDIA Nsight Compute (NCU).

This script is intended to be launched by `ncu`, but it can also be run
directly for harness validation and workload inspection.

Examples:
    python scripts/profile_ncu_prefill.py --list-workloads --list-limit 8

    ncu --set detailed \
      --kernel-name regex:gdn_.* \
      --launch-skip 3 --launch-count 1 \
      python scripts/profile_ncu_prefill.py --workload-uuid 07aa7922
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry

from scripts.pack_solution import pack_solution


def get_trace_set_path(dataset_path: str | None) -> Path:
    """Resolve dataset path from CLI or environment."""
    if dataset_path:
        return Path(dataset_path)

    env_path = os.environ.get("FIB_DATASET_PATH")
    if env_path:
        return Path(env_path)

    raise EnvironmentError(
        "Dataset path not provided. Pass --dataset-path or set FIB_DATASET_PATH."
    )


def load_solution(solution_path: Path | None) -> Solution:
    """Load a solution JSON, or pack the current source tree first."""
    if solution_path is None:
        print("Packing solution from source files...")
        solution_path = pack_solution()
    else:
        print(f"Loading solution from JSON: {solution_path}")

    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded solution: {solution.name} ({solution.definition})")
    return solution


def get_definition_and_workloads(trace_set: TraceSet, definition_name: str):
    """Return the definition object and workload traces for a definition."""
    if definition_name not in trace_set.definitions:
        raise ValueError(f"Definition '{definition_name}' not found in trace set")

    definition = trace_set.definitions[definition_name]
    workloads = trace_set.workloads.get(definition_name, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{definition_name}'")
    return definition, workloads


def format_axes(axes: dict[str, Any]) -> str:
    """Compact JSON formatting for workload axes."""
    return json.dumps(axes, ensure_ascii=False, sort_keys=True)


def list_workloads(workloads: list, limit: int) -> None:
    """Print workload UUIDs and axes for manual selection."""
    shown = workloads[:limit] if limit > 0 else workloads
    print(f"Listing {len(shown)} of {len(workloads)} workloads:")
    for idx, trace in enumerate(shown):
        workload = trace.workload
        print(f"[{idx:03d}] {workload.uuid} | axes={format_axes(workload.axes)}")


def select_workload(workloads: list, workload_uuid: str | None, workload_index: int):
    """Pick one workload by UUID prefix or by list index."""
    if workload_uuid:
        exact = [trace for trace in workloads if trace.workload.uuid == workload_uuid]
        if exact:
            return exact[0]

        prefix = [
            trace for trace in workloads if trace.workload.uuid.startswith(workload_uuid)
        ]
        if len(prefix) == 1:
            return prefix[0]
        if len(prefix) > 1:
            raise ValueError(
                f"UUID prefix '{workload_uuid}' matched multiple workloads; use a longer prefix."
            )
        raise ValueError(f"No workload matched UUID/prefix '{workload_uuid}'")

    if workload_index < 0 or workload_index >= len(workloads):
        raise IndexError(
            f"workload index {workload_index} out of range (0..{len(workloads) - 1})"
        )
    return workloads[workload_index]


def load_workload_inputs(definition, workload, trace_root: Path, device: str):
    """Load one workload's inputs in definition order and as a named mapping."""
    safe_tensors = load_safetensors(definition, workload, trace_root)
    input_list = gen_inputs(definition, workload, device=device, safe_tensors=safe_tensors)
    input_names = list(definition.inputs.keys())
    input_map = dict(zip(input_names, input_list, strict=True))
    return input_list, input_map


def flush_l2_cache(size_mb: int, device: str) -> None:
    """Best-effort L2 flush, matching the general pattern used in reference profiling."""
    if size_mb <= 0:
        return
    num_int32 = size_mb * 1024 * 1024 // 4
    cache = torch.empty(num_int32, dtype=torch.int32, device=device)
    cache.zero_()
    torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile GDN prefill with NCU")
    parser.add_argument(
        "--solution-path",
        type=Path,
        default=None,
        help="Path to an existing solution JSON. If omitted, repack from source.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Trace-set path. Defaults to FIB_DATASET_PATH when set.",
    )
    parser.add_argument(
        "--workload-uuid",
        type=str,
        default=None,
        help="Workload UUID or unique prefix to profile.",
    )
    parser.add_argument(
        "--workload-index",
        type=int,
        default=0,
        help="Workload index to profile when --workload-uuid is not provided.",
    )
    parser.add_argument(
        "--list-workloads",
        action="store_true",
        help="List workloads for the active definition and exit.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=20,
        help="How many workloads to show with --list-workloads (0 = all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for generated inputs (default: cuda).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs before the profiled run.",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        help="Number of profiled runs after warmup.",
    )
    parser.add_argument(
        "--l2-flush-mb",
        type=int,
        default=256,
        help="L2 flush scratch size in MiB before the profiled run (0 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    trace_set_path = get_trace_set_path(args.dataset_path)
    print(f"Loading trace set from: {trace_set_path}")
    trace_set = TraceSet.from_path(str(trace_set_path))

    solution = load_solution(args.solution_path)
    definition, workloads = get_definition_and_workloads(trace_set, solution.definition)

    if args.list_workloads:
        list_workloads(workloads, args.list_limit)
        return

    selected_trace = select_workload(workloads, args.workload_uuid, args.workload_index)
    workload = selected_trace.workload
    print(f"Selected workload: {workload.uuid}")
    print(f"Workload axes: {format_axes(workload.axes)}")

    input_list, input_map = load_workload_inputs(
        definition=definition,
        workload=workload,
        trace_root=trace_set.root,
        device=args.device,
    )
    print(f"Loaded inputs: {', '.join(input_map.keys())}")

    registry = BuilderRegistry.get_instance()
    runnable = None
    try:
        print("Building runnable...")
        runnable = registry.build(definition, solution)
        print(
            "Runnable ready: "
            f"build_type={runnable.metadata.build_type}, "
            f"dps={runnable.metadata.destination_passing_style}"
        )

        print(f"Warming up ({args.warmup_runs} runs)...")
        for _ in range(args.warmup_runs):
            runnable.call_value_returning(*input_list)
        torch.cuda.synchronize()

        if args.l2_flush_mb > 0:
            print(f"Flushing L2 with {args.l2_flush_mb} MiB scratch...")
            flush_l2_cache(args.l2_flush_mb, args.device)

        print(f"Profiling ({args.profile_runs} runs)...")
        for _ in range(args.profile_runs):
            runnable.call_value_returning(*input_list)
        torch.cuda.synchronize()
        print("Done.")
    finally:
        if runnable is not None:
            runnable.cleanup()
        registry.cleanup()


if __name__ == "__main__":
    main()
