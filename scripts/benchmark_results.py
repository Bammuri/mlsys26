"""Helpers for saving, loading, and iterating benchmark result JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SUCCESS_STATUSES = {"OK", "PASSED"}


ResultsMap = dict[str, dict[str, dict[str, Any]]]


def _jsonable(value: Any) -> Any:
    """Convert common benchmark objects into JSON-serializable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    return str(value)


def create_results_payload(
    results: ResultsMap,
    *,
    source: str,
    solution: Any = None,
    config: Any = None,
    trace_set_path: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap raw results with metadata for reuse by analysis scripts."""
    payload: dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
        },
        "results": _jsonable(results),
    }
    metadata = payload["metadata"]
    if solution is not None:
        metadata["solution"] = {
            "name": getattr(solution, "name", None),
            "definition": getattr(solution, "definition", None),
        }
    if config is not None:
        metadata["config"] = _jsonable(config)
    if trace_set_path:
        metadata["trace_set_path"] = str(trace_set_path)
    if extra_metadata:
        metadata.update(_jsonable(extra_metadata))
    return payload


def save_results_json(
    output_path: str | Path,
    results: ResultsMap,
    *,
    source: str,
    solution: Any = None,
    config: Any = None,
    trace_set_path: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist benchmark results and metadata as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = create_results_payload(
        results,
        source=source,
        solution=solution,
        config=config,
        trace_set_path=trace_set_path,
        extra_metadata=extra_metadata,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_results_payload(path: str | Path) -> dict[str, Any]:
    """Load either a raw results map or an envelope with metadata."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Results payload must be a JSON object: {path}")
    return data


def extract_results_map(payload: dict[str, Any]) -> ResultsMap:
    """Return the raw definition -> workload -> metrics mapping."""
    if "results" in payload and isinstance(payload["results"], dict):
        return payload["results"]
    return payload  # backward-compatible with plain dict output


def iter_result_rows(results: ResultsMap) -> Iterator[dict[str, Any]]:
    """Flatten nested result maps into per-workload rows."""
    for definition, workloads in results.items():
        if not isinstance(workloads, dict):
            continue
        for workload_uuid, entry in workloads.items():
            row = {"definition": definition, "uuid": workload_uuid}
            if isinstance(entry, dict):
                row.update(entry)
            else:
                row["value"] = entry
            yield row


def load_results_rows(path: str | Path) -> tuple[dict[str, Any], ResultsMap, list[dict[str, Any]]]:
    """Load results JSON and return payload, nested map, and flattened rows."""
    payload = load_results_payload(path)
    results = extract_results_map(payload)
    return payload, results, list(iter_result_rows(results))


def filter_workloads_by_uuid(workloads: list[Any], workload_uuids: list[str] | None) -> list[Any]:
    """Filter workload trace objects by UUID while preserving input ordering."""
    if not workload_uuids:
        return workloads
    requested = set(workload_uuids)
    return [workload for workload in workloads if workload.workload.uuid in requested]


def select_workloads_evenly(workloads: list[Any], max_workloads: int) -> list[Any]:
    """Select an evenly spaced subset of workloads."""
    if max_workloads <= 0 or len(workloads) <= max_workloads:
        return workloads
    if max_workloads == 1:
        return [workloads[0]]

    last = len(workloads) - 1
    indices = []
    for i in range(max_workloads):
        idx = round(i * last / (max_workloads - 1))
        if idx not in indices:
            indices.append(idx)
    return [workloads[idx] for idx in indices]


def summarize_trace_entries(traces: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate status/performance/error stats from per-workload result entries."""
    statuses: dict[str, int] = {}
    latency_values: list[float] = []
    speedup_values: list[float] = []
    abs_errors: list[float] = []
    rel_errors: list[float] = []

    for result in traces.values():
        status = result.get("status", "UNKNOWN")
        statuses[status] = statuses.get(status, 0) + 1
        if status in SUCCESS_STATUSES:
            if result.get("latency_ms") is not None:
                latency_values.append(result["latency_ms"])
            if result.get("speedup_factor") is not None:
                speedup_values.append(result["speedup_factor"])
        if result.get("max_abs_error") is not None:
            abs_errors.append(result["max_abs_error"])
        if result.get("max_rel_error") is not None:
            rel_errors.append(result["max_rel_error"])

    return {
        "statuses": statuses,
        "latency_values": latency_values,
        "speedup_values": speedup_values,
        "abs_errors": abs_errors,
        "rel_errors": rel_errors,
    }
