from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_modal_ncu_prefill import (
    BASE_PATH,
    PACKED_SUBMIT_SOLUTION_SOURCE,
    TRACE_SET_BASELINE_SOLUTION_SOURCE,
    TRACE_SET_PATH,
    _load_solution_payload,
)

TRACE_VOLUME_NAME = "mlsys26-contest"
app = modal.App("flashinfer-prefill-phase-profile")
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
    parser = argparse.ArgumentParser(description="Run torch.profiler phase summary for a single prefill workload on Modal B200.")
    parser.add_argument("--trace-set-path", type=Path, default=None)
    parser.add_argument("--definition", default="gdn_prefill_qk4_v8_d128_k_last")
    parser.add_argument(
        "--solution-source",
        choices=(PACKED_SUBMIT_SOLUTION_SOURCE, TRACE_SET_BASELINE_SOLUTION_SOURCE),
        default=PACKED_SUBMIT_SOLUTION_SOURCE,
    )
    parser.add_argument("--baseline-solution-index", type=int, default=0)
    parser.add_argument("--workload-uuid", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--wrapper-phases", action="store_true")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _event_time_total(event: Any) -> float:
    for attr in ("cuda_time_total", "device_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def summarize_profiler_events(events: list[Any], *, top_k: int) -> dict[str, Any]:
    rows = []
    for event in events:
        key = getattr(event, "key", None) or getattr(event, "name", None)
        if not key:
            continue
        rows.append(
            {
                "key": str(key),
                "count": int(getattr(event, "count", 0) or 0),
                "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0) or 0.0),
                "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0) or 0.0),
                "device_time_total_us": _event_time_total(event),
            }
        )

    rows.sort(
        key=lambda row: (
            row["device_time_total_us"],
            row["cpu_time_total_us"],
            row["self_cpu_time_total_us"],
        ),
        reverse=True,
    )
    phase_summary = {
        row["key"]: row
        for row in rows
        if row["key"].startswith("msinfer_entry.") or row["key"].startswith("omx.profile.")
    }
    return {
        "top_events": rows[:top_k],
        "phase_summary": phase_summary,
    }


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={"/data": trace_volume})
def profile_prefill_workload(
    solution_payload: dict[str, Any],
    definition_name: str,
    workload_uuid: str,
    iterations: int,
    warmup_iterations: int,
    wrapper_phases: bool,
    top_k: int,
) -> dict[str, Any]:
    import os

    import torch
    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.compile import BuilderRegistry

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
    runnable = BuilderRegistry.get_instance().build(definition, solution)

    def _call_once() -> None:
        if runnable.metadata.destination_passing_style:
            outputs = allocate_outputs(definition, inputs, "cuda:0")
            runnable(*inputs, *outputs)
        else:
            runnable(*inputs)

    previous_phase_flag = os.environ.get("MSINFER_GDN_PROFILE_PHASES")
    if wrapper_phases:
        os.environ["MSINFER_GDN_PROFILE_PHASES"] = "1"

    try:
        for _ in range(max(1, warmup_iterations)):
            _call_once()
            torch.cuda.synchronize("cuda:0")

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        wall_times_us: list[float] = []
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
        ) as profiler:
            for _ in range(max(1, iterations)):
                with torch.profiler.record_function("omx.profile.iteration"):
                    started = time.perf_counter()
                    _call_once()
                    torch.cuda.synchronize("cuda:0")
                    wall_times_us.append((time.perf_counter() - started) * 1_000_000.0)

        summary = summarize_profiler_events(list(profiler.key_averages()), top_k=top_k)
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
            "wrapper_phases": wrapper_phases,
            "wall_time_mean_us": (sum(wall_times_us) / len(wall_times_us)) if wall_times_us else None,
            "wall_time_min_us": min(wall_times_us) if wall_times_us else None,
            "wall_time_max_us": max(wall_times_us) if wall_times_us else None,
            **summary,
        }
    finally:
        if wrapper_phases:
            if previous_phase_flag is None:
                os.environ.pop("MSINFER_GDN_PROFILE_PHASES", None)
            else:
                os.environ["MSINFER_GDN_PROFILE_PHASES"] = previous_phase_flag


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    solution_payload, solution_provenance = _load_solution_payload(
        definition=args.definition,
        solution_source=args.solution_source,
        trace_set_path=args.trace_set_path,
        baseline_solution_index=args.baseline_solution_index,
    )
    with app.run():
        result = profile_prefill_workload.remote(
            solution_payload,
            args.definition,
            args.workload_uuid,
            args.iterations,
            args.warmup_iterations,
            args.wrapper_phases,
            args.top_k,
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
    iterations: int = 3,
    warmup_iterations: int = 1,
    wrapper_phases: bool = False,
    top_k: int = 30,
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
        "--top-k",
        str(top_k),
        "--output-json",
        output_json,
    ]
    if trace_set_path:
        argv.extend(["--trace-set-path", trace_set_path])
    if wrapper_phases:
        argv.append("--wrapper-phases")
    raise SystemExit(main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
