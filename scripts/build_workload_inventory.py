"""Build a full workload inventory by joining trace-set workloads, raw runtime summaries,
benchmark results, and optional NCU capture rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from flashinfer_bench import TraceSet

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_results import load_results_payload, load_results_rows

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - safetensors is expected in real runs
    safe_open = None


ROW_LIST_KEYS = (
    "rows",
    "reports",
    "captures",
    "entries",
    "workloads",
    "scorecards",
    "profiles",
)
BATCH_AXIS_CANDIDATES = ("batch_size", "num_seqs", "batch", "B")
TOTAL_SEQ_AXIS_CANDIDATES = ("total_seq_len", "total_tokens")
MAX_SEQ_AXIS_CANDIDATES = ("max_seq_len", "seq_len", "kv_len", "context_len")
HEAD_DIM_AXIS_CANDIDATES = ("head_dim", "head_size", "d", "D")
Q_HEAD_AXIS_CANDIDATES = ("num_q_heads", "num_heads_q", "num_heads", "h_q")
V_HEAD_AXIS_CANDIDATES = ("num_v_heads", "num_heads_v", "h_v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a workload inventory JSON for one trace-set definition")
    parser.add_argument("--definition", default=None, help="Definition name to inventory")
    parser.add_argument(
        "--trace-set-path",
        type=Path,
        default=None,
        help="Optional trace-set path (falls back to results metadata or FIB_DATASET_PATH)",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional benchmark results JSON created by scripts/run_local.py or scripts/run_modal.py",
    )
    parser.add_argument(
        "--ncu-json",
        type=Path,
        default=None,
        help="Optional JSON file containing NCU manifest/status rows keyed by UUID/workload_uuid",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to also save the emitted JSON payload",
    )
    return parser.parse_args()


def _json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _first_axis_value(axes: dict[str, Any], candidates: tuple[str, ...]) -> int | None:
    for name in candidates:
        value = axes.get(name)
        if isinstance(value, int):
            return value
    return None


def _definition_input_shapes(definition: Any, axes: dict[str, Any]) -> dict[str, list[int] | None]:
    shapes = definition.get_input_shapes(axes)
    return {
        name: (list(shape) if shape is not None else None)
        for name, shape in zip(definition.inputs.keys(), shapes)
    }


def _resolve_input_blob_path(trace_set_root: Path, input_spec: Any) -> Path | None:
    if getattr(input_spec, "type", None) != "safetensors":
        return None
    path = Path(input_spec.path)
    if not path.is_absolute():
        path = trace_set_root / path
    return path


def _load_single_safetensor(trace_set_root: Path, input_spec: Any) -> Any | None:
    path = _resolve_input_blob_path(trace_set_root, input_spec)
    if path is None or not path.exists():
        return None

    tensor_key = getattr(input_spec, "tensor_key", None)
    if not isinstance(tensor_key, str):
        return None

    if safe_open is not None:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if tensor_key not in handle.keys():
                return None
            return handle.get_tensor(tensor_key)

    import safetensors.torch  # pragma: no cover - fallback only

    return safetensors.torch.load_file(str(path), device="cpu").get(tensor_key)


def _compute_max_seq_len(workload_trace: Any, input_shapes: dict[str, list[int] | None], trace_set_root: Path) -> int | None:
    input_spec = workload_trace.workload.inputs.get("cu_seqlens")
    if input_spec is not None:
        cu_seqlens = _load_single_safetensor(trace_set_root, input_spec)
        if cu_seqlens is not None:
            if int(cu_seqlens.numel()) < 2:
                raise ValueError("cu_seqlens must contain at least two entries")
            diffs = cu_seqlens[1:] - cu_seqlens[:-1]
            return int(diffs.max().item())

    axes = workload_trace.workload.axes
    max_seq_len = _first_axis_value(axes, MAX_SEQ_AXIS_CANDIDATES)
    if max_seq_len is not None:
        return max_seq_len

    q_shape = input_shapes.get("q")
    if q_shape is None:
        return None

    if "cu_seqlens" in workload_trace.workload.inputs:
        return q_shape[0] if q_shape else None

    batch_size = _first_axis_value(axes, BATCH_AXIS_CANDIDATES)
    if len(q_shape) >= 4 and batch_size is not None and q_shape[0] == batch_size:
        return q_shape[1]
    if len(q_shape) >= 3 and batch_size is not None and q_shape[0] == batch_size:
        return q_shape[1]
    if len(q_shape) >= 2:
        return q_shape[0]
    return None


def build_runtime_summary(workload_trace: Any, definition: Any, trace_set_root: Path) -> dict[str, Any]:
    axes = dict(sorted(workload_trace.workload.axes.items()))
    input_names = list(definition.inputs.keys())
    input_shapes = _definition_input_shapes(definition, workload_trace.workload.axes)

    q_shape = input_shapes.get("q")
    v_shape = input_shapes.get("v")

    batch_size = _first_axis_value(axes, BATCH_AXIS_CANDIDATES)
    total_seq_len = _first_axis_value(axes, TOTAL_SEQ_AXIS_CANDIDATES)

    max_seq_len = _compute_max_seq_len(workload_trace, input_shapes, trace_set_root)
    if total_seq_len is None and q_shape is not None:
        if "cu_seqlens" in workload_trace.workload.inputs:
            total_seq_len = q_shape[0]
        elif batch_size is not None and max_seq_len is not None:
            total_seq_len = batch_size * max_seq_len
        else:
            total_seq_len = q_shape[0]
    head_dim = _first_axis_value(axes, HEAD_DIM_AXIS_CANDIDATES)
    if head_dim is None and q_shape is not None:
        head_dim = q_shape[-1]

    h_q = _first_axis_value(axes, Q_HEAD_AXIS_CANDIDATES)
    if h_q is None and q_shape is not None and len(q_shape) >= 2:
        h_q = q_shape[-2]

    h_v = _first_axis_value(axes, V_HEAD_AXIS_CANDIDATES)
    if h_v is None and v_shape is not None and len(v_shape) >= 2:
        h_v = v_shape[-2]

    varlen = "cu_seqlens" in definition.inputs or "cu_seqlens" in workload_trace.workload.inputs
    if not varlen and batch_size is not None and total_seq_len is not None and max_seq_len is not None:
        varlen = batch_size * max_seq_len != total_seq_len

    return {
        "axes": axes,
        "batch_size": batch_size,
        "head_dim": head_dim,
        "h_q": h_q,
        "h_v": h_v,
        "input_names": input_names,
        "input_shapes": input_shapes,
        "max_seq_len": max_seq_len,
        "shape_fingerprint": ",".join(f"{key}={value}" for key, value in axes.items()),
        "total_seq_len": total_seq_len,
        "varlen": bool(varlen),
    }


def _load_results_rows_by_uuid(results_json: Path | None, definition: str) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    if results_json is None:
        return None, {}

    payload, _, rows = load_results_rows(results_json)
    filtered = {
        row["uuid"]: row
        for row in rows
        if isinstance(row, dict) and row.get("definition") == definition and isinstance(row.get("uuid"), str)
    }
    return payload, filtered


def _extract_row_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("Expected a list of JSON objects")
        return list(payload)

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object or list")

    for key in ROW_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            if not all(isinstance(item, dict) for item in value):
                raise ValueError(f"Expected '{key}' to contain only JSON objects")
            return list(value)

    rows: list[dict[str, Any]] = []
    for maybe_uuid, value in payload.items():
        if isinstance(value, dict):
            row = dict(value)
            row.setdefault("uuid", row.get("workload_uuid") or maybe_uuid)
            rows.append(row)
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError("Expected nested row lists to contain only JSON objects")
                row = dict(item)
                row.setdefault("uuid", row.get("workload_uuid") or maybe_uuid)
                rows.append(row)

    if rows:
        return rows

    raise ValueError("Could not find any row list in the NCU JSON payload")


def _load_ncu_rows_by_uuid(ncu_json: Path | None) -> tuple[dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    if ncu_json is None:
        return None, {}

    payload = json.loads(ncu_json.read_text(encoding="utf-8"))
    rows = _extract_row_list(payload)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        uuid = row.get("uuid") or row.get("workload_uuid")
        if not isinstance(uuid, str) or not uuid:
            raise ValueError("Each NCU row must contain 'uuid' or 'workload_uuid'")
        normalized = dict(row)
        normalized.setdefault("uuid", uuid)
        grouped.setdefault(uuid, []).append(normalized)
    return payload, grouped


def _infer_definition(
    requested_definition: str | None,
    results_payload: dict[str, Any] | None,
    trace_set: TraceSet,
) -> str:
    if requested_definition:
        return requested_definition

    if results_payload is not None:
        metadata = results_payload.get("metadata", {})
        solution = metadata.get("solution", {})
        definition = solution.get("definition") if isinstance(solution, dict) else None
        if isinstance(definition, str) and definition:
            return definition

        results = results_payload.get("results")
        if isinstance(results, dict) and len(results) == 1:
            only_definition = next(iter(results))
            if isinstance(only_definition, str):
                return only_definition

    if len(trace_set.definitions) == 1:
        return next(iter(trace_set.definitions))

    raise ValueError("Definition could not be inferred; pass --definition")


def resolve_trace_set_path(cli_path: Path | None, results_payload: dict[str, Any] | None) -> Path:
    candidates: list[Path] = []
    if cli_path is not None:
        candidates.append(cli_path)
    if results_payload is not None:
        metadata = results_payload.get("metadata", {})
        metadata_path = metadata.get("trace_set_path")
        if isinstance(metadata_path, str) and metadata_path:
            candidates.append(Path(metadata_path))
    env_path = os.environ.get("FIB_DATASET_PATH")
    if env_path:
        candidates.append(Path(env_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if cli_path is not None:
        raise FileNotFoundError(f"Trace-set path does not exist: {cli_path}")
    raise ValueError("Trace-set path not provided or discoverable from results metadata/FIB_DATASET_PATH")


def build_inventory(
    *,
    definition_name: str | None,
    trace_set_path: Path | None,
    results_json: Path | None,
    ncu_json: Path | None,
) -> dict[str, Any]:
    results_payload = load_results_payload(results_json) if results_json is not None else None
    resolved_trace_set_path = resolve_trace_set_path(trace_set_path, results_payload)
    trace_set = TraceSet.from_path(str(resolved_trace_set_path))
    resolved_definition = _infer_definition(definition_name, results_payload, trace_set)

    definition = trace_set.definitions.get(resolved_definition)
    if definition is None:
        raise ValueError(f"Definition '{resolved_definition}' not found in trace set")

    _, results_by_uuid = _load_results_rows_by_uuid(results_json, resolved_definition)
    ncu_payload, ncu_by_uuid = _load_ncu_rows_by_uuid(ncu_json)

    workload_traces = trace_set.workloads.get(resolved_definition, [])
    rows = []
    workload_uuids: list[str] = []
    matched_results = 0
    matched_ncu_rows = 0
    workloads_with_ncu = 0

    for workload_trace in workload_traces:
        uuid = workload_trace.workload.uuid
        workload_uuids.append(uuid)
        benchmark_result = results_by_uuid.get(uuid)
        ncu_rows = list(ncu_by_uuid.get(uuid, []))
        if benchmark_result is not None:
            matched_results += 1
        if ncu_rows:
            workloads_with_ncu += 1
            matched_ncu_rows += len(ncu_rows)
        rows.append(
            {
                "benchmark_result": benchmark_result,
                "definition": resolved_definition,
                "ncu_rows": ncu_rows,
                "runtime_summary": build_runtime_summary(workload_trace, definition, resolved_trace_set_path),
                "uuid": uuid,
            }
        )

    workload_uuid_set = set(workload_uuids)
    unmatched_result_uuids = sorted(uuid for uuid in results_by_uuid if uuid not in workload_uuid_set)
    unmatched_ncu_uuids = sorted(uuid for uuid in ncu_by_uuid if uuid not in workload_uuid_set)

    return {
        "metadata": {
            "definition": resolved_definition,
            "matched_benchmark_rows": matched_results,
            "matched_ncu_rows": matched_ncu_rows,
            "results_json": str(results_json) if results_json is not None else None,
            "source": "scripts/build_workload_inventory.py",
            "trace_set_path": str(resolved_trace_set_path),
            "unmatched_result_uuids": unmatched_result_uuids,
            "unmatched_ncu_uuids": unmatched_ncu_uuids,
            "workload_count": len(rows),
            "workloads_with_ncu_rows": workloads_with_ncu,
            "ncu_json": str(ncu_json) if ncu_json is not None else None,
            "ncu_json_has_metadata": isinstance(ncu_payload, dict) and "metadata" in ncu_payload,
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    payload = build_inventory(
        definition_name=args.definition,
        trace_set_path=args.trace_set_path,
        results_json=args.results_json,
        ncu_json=args.ncu_json,
    )
    rendered = _json_dump(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
