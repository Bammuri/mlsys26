"""Segment benchmark results into fast/mid/tail cohorts."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from flashinfer_bench import TraceSet

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_results import (
    SUCCESS_STATUSES,
    extract_results_map,
    load_results_payload,
    save_results_json,
)

EPSILON_LATENCY_MS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment benchmark results into latency cohorts")
    parser.add_argument("results_json", type=Path, help="Path to benchmark results JSON")
    parser.add_argument(
        "--trace-set-path",
        type=Path,
        default=None,
        help="Optional dataset path for workload-axis enrichment (falls back to results metadata or FIB_DATASET_PATH)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for cohort JSON (default: <results>.cohorts.json)",
    )
    parser.add_argument(
        "--fast-threshold-ms",
        type=float,
        default=None,
        help="Manual fast threshold in ms (overrides auto clustering)",
    )
    parser.add_argument(
        "--tail-threshold-ms",
        type=float,
        default=None,
        help="Manual tail threshold in ms (overrides auto clustering)",
    )
    return parser.parse_args()


def resolve_trace_set_path(cli_path: Path | None, payload: dict[str, Any]) -> Path | None:
    candidates = [cli_path]
    metadata = payload.get("metadata", {})
    metadata_path = metadata.get("trace_set_path")
    if metadata_path:
        candidates.append(Path(metadata_path))
    env_path = os.environ.get("FIB_DATASET_PATH")
    if env_path:
        candidates.append(Path(env_path))
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def load_workload_index(trace_set_path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if trace_set_path is None:
        return {}
    trace_set = TraceSet.from_path(str(trace_set_path))
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for definition, traces in trace_set.workloads.items():
        for trace in traces:
            workload = trace.workload
            input_names = sorted(workload.inputs.keys())
            index[(definition, workload.uuid)] = {
                "axes": dict(sorted(workload.axes.items())),
                "input_names": input_names,
                "shape_fingerprint": ",".join(f"{key}={value}" for key, value in sorted(workload.axes.items())),
                "has_initial_state": any("state" in name.lower() or "init" in name.lower() for name in input_names),
            }
    return index


def _prefix_stats(values: list[float]) -> tuple[list[float], list[float]]:
    prefix_sum = [0.0]
    prefix_sq = [0.0]
    for value in values:
        prefix_sum.append(prefix_sum[-1] + value)
        prefix_sq.append(prefix_sq[-1] + value * value)
    return prefix_sum, prefix_sq


def _segment_cost(prefix_sum: list[float], prefix_sq: list[float], start: int, end: int) -> float:
    count = end - start + 1
    total = prefix_sum[end + 1] - prefix_sum[start]
    total_sq = prefix_sq[end + 1] - prefix_sq[start]
    return total_sq - (total * total) / count


def auto_cluster_boundaries(latencies: list[float]) -> tuple[float | None, float | None]:
    if not latencies:
        return None, None
    if len(latencies) == 1:
        value = latencies[0]
        return value, value
    if len(latencies) == 2:
        ordered = sorted(latencies)
        return ordered[0], ordered[1]

    sorted_latencies = sorted(latencies)
    transformed = [math.log(max(value, EPSILON_LATENCY_MS)) for value in sorted_latencies]
    prefix_sum, prefix_sq = _prefix_stats(transformed)

    best: tuple[float, int, int] | None = None
    n = len(sorted_latencies)
    for first_end in range(0, n - 2):
        for second_end in range(first_end + 1, n - 1):
            cost = (
                _segment_cost(prefix_sum, prefix_sq, 0, first_end)
                + _segment_cost(prefix_sum, prefix_sq, first_end + 1, second_end)
                + _segment_cost(prefix_sum, prefix_sq, second_end + 1, n - 1)
            )
            if best is None or cost < best[0]:
                best = (cost, first_end, second_end)

    assert best is not None
    _, first_end, second_end = best
    return sorted_latencies[first_end], sorted_latencies[second_end + 1]


def assign_cohort(latency_ms: float | None, status: str, fast_threshold: float | None, tail_threshold: float | None) -> str:
    if status not in SUCCESS_STATUSES:
        return "failed"
    if latency_ms is None:
        return "unclassified"
    if fast_threshold is None and tail_threshold is None:
        return "fast"
    if fast_threshold is not None and latency_ms <= fast_threshold:
        return "fast"
    if tail_threshold is not None and latency_ms >= tail_threshold:
        return "tail"
    return "mid"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_workloads": len(rows),
        "status_counts": {},
        "cohort_counts": {},
    }
    for row in rows:
        status = row.get("status", "UNKNOWN")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        cohort = row.get("cohort", "unclassified")
        summary["cohort_counts"][cohort] = summary["cohort_counts"].get(cohort, 0) + 1

    successful = [
        row["latency_ms"]
        for row in rows
        if row.get("status") in SUCCESS_STATUSES and isinstance(row.get("latency_ms"), (int, float))
    ]
    if successful:
        summary["latency_ms"] = {
            "min": min(successful),
            "max": max(successful),
            "mean": mean(successful),
        }

    return summary


def segment_definition_rows(
    definition: str,
    rows: list[dict[str, Any]],
    workload_index: dict[tuple[str, str], dict[str, Any]],
    fast_threshold_override: float | None,
    tail_threshold_override: float | None,
) -> dict[str, Any]:
    successful_latencies = [
        float(row["latency_ms"])
        for row in rows
        if row.get("status") in SUCCESS_STATUSES and isinstance(row.get("latency_ms"), (int, float))
    ]

    if fast_threshold_override is not None or tail_threshold_override is not None:
        fast_threshold = fast_threshold_override
        tail_threshold = tail_threshold_override
    else:
        fast_threshold, tail_threshold = auto_cluster_boundaries(successful_latencies)

    enriched_rows = []
    for row in sorted(
        rows,
        key=lambda item: (
            float(item["latency_ms"]) if isinstance(item.get("latency_ms"), (int, float)) else math.inf,
            item["uuid"],
        ),
    ):
        enriched = dict(row)
        enriched["cohort"] = assign_cohort(
            enriched.get("latency_ms"),
            enriched.get("status", "UNKNOWN"),
            fast_threshold,
            tail_threshold,
        )
        enrichment = workload_index.get((definition, enriched["uuid"]))
        if enrichment:
            enriched.update(enrichment)
        enriched_rows.append(enriched)

    cohorts: dict[str, list[str]] = {}
    for row in enriched_rows:
        cohorts.setdefault(row["cohort"], []).append(row["uuid"])

    return {
        "thresholds": {
            "fast_max_latency_ms": fast_threshold,
            "tail_min_latency_ms": tail_threshold,
            "method": "manual" if fast_threshold_override is not None or tail_threshold_override is not None else "auto_1d_3cluster",
        },
        "summary": summarize_rows(enriched_rows),
        "cohorts": cohorts,
        "workloads": enriched_rows,
    }


def build_output_path(results_json: Path, output_json: Path | None) -> Path:
    if output_json is not None:
        return output_json
    if results_json.suffix == ".json":
        return results_json.with_name(f"{results_json.stem}.cohorts.json")
    return Path(str(results_json) + ".cohorts.json")


def print_summary(definitions: dict[str, dict[str, Any]]) -> None:
    for definition, payload in definitions.items():
        summary = payload["summary"]
        thresholds = payload["thresholds"]
        print(f"\n{definition}:")
        print(f"  workloads: {summary['total_workloads']}")
        print(
            "  status counts: "
            + ", ".join(f"{key}={value}" for key, value in sorted(summary["status_counts"].items()))
        )
        print(
            "  cohort counts: "
            + ", ".join(f"{key}={value}" for key, value in sorted(summary["cohort_counts"].items()))
        )
        latency_summary = summary.get("latency_ms")
        if latency_summary:
            print(
                "  latency ms: "
                f"min={latency_summary['min']:.4f}, mean={latency_summary['mean']:.4f}, max={latency_summary['max']:.4f}"
            )
        print(
            "  thresholds: "
            f"fast<= {thresholds['fast_max_latency_ms']}, tail>= {thresholds['tail_min_latency_ms']}"
        )


def main() -> None:
    args = parse_args()
    payload = load_results_payload(args.results_json)
    results = extract_results_map(payload)
    trace_set_path = resolve_trace_set_path(args.trace_set_path, payload)
    workload_index = load_workload_index(trace_set_path)

    definitions_output: dict[str, dict[str, Any]] = {}
    for definition, workloads in results.items():
        rows = []
        for workload_uuid, entry in workloads.items():
            row = {"definition": definition, "uuid": workload_uuid}
            if isinstance(entry, dict):
                row.update(entry)
            else:
                row["value"] = entry
            rows.append(row)
        definitions_output[definition] = segment_definition_rows(
            definition,
            rows,
            workload_index,
            args.fast_threshold_ms,
            args.tail_threshold_ms,
        )

    output = {
        "metadata": {
            "source_results_json": str(args.results_json),
            "trace_set_path": str(trace_set_path) if trace_set_path else None,
        },
        "definitions": definitions_output,
    }
    output_path = build_output_path(args.results_json, args.output_json)
    save_results_json(
        output_path,
        output["definitions"],
        source="scripts/segment_benchmark_results.py",
        extra_metadata=output["metadata"],
    )
    print_summary(definitions_output)
    print(f"\nSaved cohort JSON to {output_path}")


if __name__ == "__main__":
    main()
