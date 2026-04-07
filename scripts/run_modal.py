"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("build-essential", "ninja-build")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        }
    )
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

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
                "log": trace.evaluation.log,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def get_git_commit_hash() -> str:
    """Return the current short git hash, if available."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def sanitize_label(label: str) -> str:
    """Convert a free-form label into a filesystem-safe token."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-")
    return sanitized or "run"


def summarize_results(results: dict) -> dict:
    """Compute aggregate statistics from benchmark results."""
    latencies = []
    ref_latencies = []
    speedups = []
    passed = 0
    total = 0

    for traces in results.values():
        for result in traces.values():
            total += 1
            if result.get("status") == "PASSED":
                passed += 1
            if result.get("latency_ms") is not None:
                latencies.append(float(result["latency_ms"]))
            if result.get("reference_latency_ms") is not None:
                ref_latencies.append(float(result["reference_latency_ms"]))
            if result.get("speedup_factor") is not None:
                speedups.append(float(result["speedup_factor"]))

    def mean_or_none(values):
        if not values:
            return None
        return sum(values) / float(len(values))

    return {
        "total_workloads": total,
        "passed_workloads": passed,
        "failed_workloads": total - passed,
        "avg_latency_ms": mean_or_none(latencies),
        "avg_reference_latency_ms": mean_or_none(ref_latencies),
        "avg_speedup_factor": mean_or_none(speedups),
    }


def save_results_artifact(
    *,
    results: dict,
    solution: Solution,
    config: BenchmarkConfig,
    label: str,
    entry_point: str,
) -> Path:
    """Persist raw benchmark results and aggregate summary."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    definition_dir = PROJECT_ROOT / ".omx" / "benchmarks" / solution.definition
    definition_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = definition_dir / f"{timestamp}-{sanitize_label(label)}.json"
    payload = {
        "metadata": {
            "timestamp_utc": timestamp,
            "label": label,
            "definition": solution.definition,
            "solution_name": solution.name,
            "entry_point": entry_point,
            "git_commit": get_git_commit_hash(),
            "benchmark_config": {
                "warmup_runs": config.warmup_runs,
                "iterations": config.iterations,
                "num_trials": config.num_trials,
            },
        },
        "summary": summarize_results(results),
        "results": results,
    }
    artifact_path.write_text(json.dumps(payload, indent=2))
    return artifact_path


def summarize_log(log: str, max_lines: int = 20) -> str:
    """Return a compact multi-line log summary."""
    if not log:
        return ""

    lines = [line.rstrip() for line in log.strip().splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["[log truncated]"])


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        shown_failure_log = False
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

            if status != "PASSED" and not shown_failure_log:
                summary = summarize_log(result.get("log", ""))
                if summary:
                    print("  First failure log:")
                    print(summary)
                    print()
                    shown_failure_log = True

        summary = summarize_results({def_name: traces})
        print(
            "  Summary:",
            f"passed={summary['passed_workloads']}/{summary['total_workloads']}",
            f"avg_latency_ms={summary['avg_latency_ms']:.3f}"
            if summary["avg_latency_ms"] is not None
            else "avg_latency_ms=n/a",
            f"avg_reference_latency_ms={summary['avg_reference_latency_ms']:.3f}"
            if summary["avg_reference_latency_ms"] is not None
            else "avg_reference_latency_ms=n/a",
            f"avg_speedup_factor={summary['avg_speedup_factor']:.2f}"
            if summary["avg_speedup_factor"] is not None
            else "avg_speedup_factor=n/a",
        )


@app.local_entrypoint()
def main(
    definition: str = "",
    entry_point: str = "",
    language: str = "",
    binding: str = "",
    destination_passing_style: str = "",
    output_path: str = "",
    warmup_runs: int = 3,
    iterations: int = 100,
    num_trials: int = 5,
    label: str = "",
):
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    dps_override = None
    if destination_passing_style:
        normalized = destination_passing_style.strip().lower()
        if normalized in {"true", "1", "yes", "dps"}:
            dps_override = True
        elif normalized in {"false", "0", "no", "value"}:
            dps_override = False
        else:
            raise ValueError(
                "destination_passing_style must be one of: true, false, dps, value"
            )

    print("Packing solution from source files...")
    solution_path = pack_solution(
        Path(output_path) if output_path else None,
        definition=definition or None,
        language=language or None,
        entry_point=entry_point or None,
        destination_passing_style=dps_override,
        binding=binding or None,
    )

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    config = BenchmarkConfig(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=num_trials,
    )
    print(
        "Benchmark config:",
        f"warmup_runs={config.warmup_runs}",
        f"iterations={config.iterations}",
        f"num_trials={config.num_trials}",
    )
    results = run_benchmark.remote(solution, config)

    if not results:
        print("No results returned!")
        return

    print_results(results)
    artifact_path = save_results_artifact(
        results=results,
        solution=solution,
        config=config,
        label=label or solution.definition,
        entry_point=solution.spec.entry_point,
    )
    print(f"\nSaved results: {artifact_path}")
