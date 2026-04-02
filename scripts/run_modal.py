"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y build-essential git git-lfs",
        "git lfs install --system",
    )
    .pip_install("flashinfer-bench", "modal", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    max_workloads: int = 0,
    workload_uuid: str = "",
    batch_size_filter: int = 0,
) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = build_benchmark_config()

    trace_root = resolve_trace_root(Path(TRACE_SET_PATH))
    print(f"Using trace root: {trace_root}")
    trace_set = TraceSet.from_path(trace_root)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if workload_uuid:
        workloads = [workload for workload in workloads if workload.workload.uuid == workload_uuid]
    if batch_size_filter > 0:
        workloads = [
            workload
            for workload in workloads
            if workload.workload.axes.get("batch_size") == batch_size_filter
        ]
    if max_workloads > 0:
        workloads = workloads[:max_workloads]

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    print(
        "Benchmark config:",
        {
            "warmup_runs": config.warmup_runs,
            "iterations": config.iterations,
            "num_trials": config.num_trials,
            "workloads": len(workloads),
            "workload_uuid": workload_uuid or None,
            "batch_size_filter": batch_size_filter or None,
        },
    )

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    print_results(results)
    return results


def resolve_trace_root(base_path: Path) -> str:
    """Find the trace-set root that actually contains definitions and workloads."""
    candidates = [base_path]
    candidates.extend(path for path in sorted(base_path.iterdir()) if path.is_dir())

    for candidate in candidates:
        if (candidate / "definitions").exists() and (candidate / "workloads").exists():
            return str(candidate)

    raise FileNotFoundError(f"No trace-set root found under {base_path}")


def build_benchmark_config() -> BenchmarkConfig:
    """Build benchmark config, optionally overridden by environment variables."""
    return BenchmarkConfig(
        warmup_runs=int(os.environ.get("FIB_WARMUP_RUNS", "3")),
        iterations=int(os.environ.get("FIB_ITERATIONS", "100")),
        num_trials=int(os.environ.get("FIB_NUM_TRIALS", "5")),
    )


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(
        solution,
        build_benchmark_config(),
        int(os.environ.get("FIB_MAX_WORKLOADS", "0")),
        os.environ.get("FIB_WORKLOAD_UUID", ""),
        int(os.environ.get("FIB_BATCH_SIZE_FILTER", "0")),
    )

    if not results:
        print("No results returned!")
        return

    print_results(results)
