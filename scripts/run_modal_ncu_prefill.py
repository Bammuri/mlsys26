from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
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
DEFAULT_KERNEL_PATTERN = "regex:kernel_cutlass_kernel_fib_python_gdn_prefill_v1.*"
DEFAULT_BASELINE_KERNEL_PATTERN = "regex:kernel_cutlass_kernel_fib.*"
DEFAULT_SECTIONS = (
    "SpeedOfLight",
    "SchedulerStats",
    "WarpStateStats",
    "Occupancy",
    "MemoryWorkloadAnalysis",
    "LaunchStats",
)
PERSISTENT_POLICY_ENV = "MSINFER_GDN_PERSISTENT_POLICY"
PERSISTENT_AUTO_MAX_BATCH_ENV = "MSINFER_PERSISTENT_AUTO_MAX_BATCH"
PERSISTENT_AUTO_MAX_SEQ_LEN_ENV = "MSINFER_PERSISTENT_AUTO_MAX_SEQ_LEN"
ADAPTIVE_SELECTOR_KEYS_ENV = "MSINFER_GDN_ADAPTIVE_SELECTOR_KEYS"
PACKED_SUBMIT_SOLUTION_SOURCE = "packed-submit"
TRACE_SET_BASELINE_SOLUTION_SOURCE = "trace-set-baseline"

app = modal.App("flashinfer-prefill-ncu")
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
    parser = argparse.ArgumentParser(description="Capture targeted NCU reports for prefill workloads on Modal B200")
    parser.add_argument("--trace-set-path", type=Path, default=None)
    parser.add_argument("--definition", default="gdn_prefill_qk4_v8_d128_k_last")
    parser.add_argument(
        "--solution-source",
        choices=(PACKED_SUBMIT_SOLUTION_SOURCE, TRACE_SET_BASELINE_SOLUTION_SOURCE),
        default=PACKED_SUBMIT_SOLUTION_SOURCE,
        help="Select whether to capture NCU for the local packed submit solution or the TraceSet baseline solution.",
    )
    parser.add_argument(
        "--baseline-solution-index",
        type=int,
        default=0,
        help="Index into TraceSet solutions[definition] when --solution-source=trace-set-baseline.",
    )
    parser.add_argument("--workload-uuids", default="", help="Comma-separated workload UUIDs (empty means all)")
    parser.add_argument(
        "--representative-groups-json",
        type=Path,
        default=None,
        help="Optional structural groups JSON; selects the first UUID from each group",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap after workload selection (0 means all)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--kernel-pattern",
        default="",
        help="Optional kernel regex override. Empty selects a source-aware default.",
    )
    parser.add_argument("--launch-skip", type=int, default=1)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--sections-csv", default=",".join(DEFAULT_SECTIONS))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path, default=Path(".omx/profiles/full-prefill-ncu"))
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--label", default="baseline", help="Capture label written to the manifest")
    parser.add_argument("--persistent-policy", default="", help="Optional MSINFER_GDN_PERSISTENT_POLICY override")
    parser.add_argument("--persistent-auto-max-batch", type=int, default=0)
    parser.add_argument("--persistent-auto-max-seq-len", type=int, default=0)
    parser.add_argument(
        "--adaptive-selector-keys",
        default="",
        help="Optional MSINFER_GDN_ADAPTIVE_SELECTOR_KEYS override (comma-separated selector keys)",
    )
    return parser.parse_args(argv)


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


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


def _resolve_kernel_pattern(solution_source: str, kernel_pattern: str) -> str:
    if kernel_pattern:
        return kernel_pattern
    if solution_source == TRACE_SET_BASELINE_SOLUTION_SOURCE:
        return DEFAULT_BASELINE_KERNEL_PATTERN
    return DEFAULT_KERNEL_PATTERN


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
    from scripts.pack_solution import pack_solution

    if solution_source == PACKED_SUBMIT_SOLUTION_SOURCE:
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


def _build_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    runtime_env: dict[str, str] = {}
    if args.persistent_policy:
        runtime_env[PERSISTENT_POLICY_ENV] = args.persistent_policy
    if args.persistent_auto_max_batch > 0:
        runtime_env[PERSISTENT_AUTO_MAX_BATCH_ENV] = str(args.persistent_auto_max_batch)
    if args.persistent_auto_max_seq_len > 0:
        runtime_env[PERSISTENT_AUTO_MAX_SEQ_LEN_ENV] = str(args.persistent_auto_max_seq_len)
    if args.adaptive_selector_keys:
        runtime_env[ADAPTIVE_SELECTOR_KEYS_ENV] = args.adaptive_selector_keys
    return runtime_env


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    if not overrides:
        yield
        return

    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resolve_workload_uuids(definition: str, trace_set_path: Path | None, explicit_uuids: list[str], limit: int) -> list[str]:
    resolved_path = _resolve_trace_set_root(trace_set_path, definition=definition)
    trace_set = TraceSet.from_path(str(resolved_path))
    workloads = [trace.workload.uuid for trace in trace_set.workloads.get(definition, [])]
    if explicit_uuids:
        requested = set(explicit_uuids)
        workloads = [uuid for uuid in workloads if uuid in requested]
    if limit > 0:
        workloads = workloads[:limit]
    return workloads


def _load_representative_uuids(groups_json: Path) -> list[str]:
    payload = json.loads(groups_json.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("Representative groups JSON must contain a 'groups' list")

    representative_uuids: list[str] = []
    for group in groups:
        uuids = group.get("uuids") if isinstance(group, dict) else None
        if not isinstance(uuids, list) or not uuids or not isinstance(uuids[0], str):
            raise ValueError("Each group must contain at least one UUID")
        representative_uuids.append(uuids[0])
    return representative_uuids


def _manifest_row(result: dict[str, Any], report_filename: str, *, label: str = "baseline") -> dict[str, Any]:
    notes = list(result.get("notes", []))
    return {
        "uuid": result["uuid"],
        "label": label,
        "report_path": report_filename,
        "status": result["status"],
        "notes": notes,
        "axes": result.get("axes"),
        "kernel_pattern": result.get("kernel_pattern"),
    }


def _capture_single_workload(
    *,
    solution_payload: dict[str, Any],
    definition: Any,
    workload: Any,
    kernel_pattern: str,
    section_args: list[str],
    launch_skip: int,
    launch_count: int,
    timeout_seconds: int,
    runtime_env: dict[str, str],
) -> dict[str, Any]:
    from flashinfer_bench import Solution

    solution = Solution.model_validate(solution_payload)
    try:
        with tempfile.TemporaryDirectory(prefix="prefill_ncu_") as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "definition.json").write_text(definition.model_dump_json())
            (data_dir / "solution.json").write_text(solution.model_dump_json())
            (data_dir / "workload.json").write_text(workload.model_dump_json())
            cmd = [
                "ncu",
                "--page",
                "details",
                "-f",
                "--kernel-name",
                kernel_pattern,
                "--launch-skip",
                str(launch_skip),
                "--launch-count",
                str(launch_count),
                *section_args,
                sys.executable,
                "-u",
                "-m",
                "flashinfer_bench.agents._solution_runner",
                "--data-dir",
                str(data_dir),
                "--device",
                "cuda:0",
                "--trace-set-path",
                TRACE_SET_PATH,
            ]
            with _temporary_env(runtime_env):
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            report_text = proc.stdout + proc.stderr
            notes: list[str] = []
            if "No kernels were profiled" in report_text:
                notes.append("no_kernels_profiled")
            return {
                "uuid": workload.uuid,
                "axes": dict(sorted(workload.axes.items())),
                "status": "ok" if proc.returncode == 0 else f"error_{proc.returncode}",
                "notes": notes,
                "kernel_pattern": kernel_pattern,
                "report_text": report_text,
            }
    except subprocess.TimeoutExpired as exc:
        return {
            "uuid": workload.uuid,
            "axes": dict(sorted(workload.axes.items())),
            "status": "timeout",
            "notes": [f"timeout_seconds={timeout_seconds}"],
            "kernel_pattern": kernel_pattern,
            "report_text": (exc.stdout or "") + (exc.stderr or ""),
        }
    except Exception as exc:  # pragma: no cover
        return {
            "uuid": workload.uuid,
            "axes": dict(sorted(workload.axes.items())),
            "status": "exception",
            "notes": [f"{type(exc).__name__}: {exc}"],
            "kernel_pattern": kernel_pattern,
            "report_text": "",
        }


@app.function(image=image, gpu="B200:1", timeout=14400, volumes={"/data": trace_volume})
def capture_prefill_ncu_workload(
    solution_payload,
    definition_name,
    workload_uuid,
    kernel_pattern,
    sections,
    launch_skip,
    launch_count,
    timeout_seconds,
    runtime_env,
):
    from flashinfer_bench import TraceSet

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[definition_name]
    section_args: list[str] = []
    for section in sections:
        section_args.extend(["--section", section])

    workload_map = {trace.workload.uuid: trace.workload for trace in trace_set.workloads.get(definition_name, [])}
    workload = workload_map.get(workload_uuid)
    if workload is None:
        return {
            "uuid": workload_uuid,
            "status": "missing_workload",
            "notes": ["workload_not_found"],
            "report_text": "",
            "kernel_pattern": kernel_pattern,
        }
    return _capture_single_workload(
        solution_payload=solution_payload,
        definition=definition,
        workload=workload,
        kernel_pattern=kernel_pattern,
        section_args=section_args,
        launch_skip=launch_skip,
        launch_count=launch_count,
        timeout_seconds=timeout_seconds,
        runtime_env=runtime_env,
    )


def _collect_workload_results(
    workload_uuids: list[str],
    *,
    batch_size: int,
    spawn_fn,
    result_callback=None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for batch in _chunked(workload_uuids, max(1, batch_size)):
        calls = [(uuid, spawn_fn(uuid)) for uuid in batch]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as executor:
            futures = {executor.submit(call.get): uuid for uuid, call in calls}
            for future in concurrent.futures.as_completed(futures):
                uuid = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - exercised by mocks/local failures
                    result = {
                        "uuid": uuid,
                        "status": "exception",
                        "notes": [f"{type(exc).__name__}: {exc}"],
                        "report_text": "",
                        "kernel_pattern": None,
                    }
                results.append(result)
                if result_callback is not None:
                    result_callback(result)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    explicit_uuids = [uuid.strip() for uuid in args.workload_uuids.split(",") if uuid.strip()]
    if not explicit_uuids and args.representative_groups_json is not None:
        explicit_uuids = _load_representative_uuids(args.representative_groups_json)
    workload_uuids = _resolve_workload_uuids(args.definition, args.trace_set_path, explicit_uuids, args.limit)
    if not workload_uuids:
        raise SystemExit("No workloads selected")

    kernel_pattern = _resolve_kernel_pattern(args.solution_source, args.kernel_pattern)
    solution_payload, solution_provenance = _load_solution_payload(
        definition=args.definition,
        solution_source=args.solution_source,
        trace_set_path=args.trace_set_path,
        baseline_solution_index=args.baseline_solution_index,
    )
    runtime_env = _build_runtime_env(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    sections = [section.strip() for section in args.sections_csv.split(",") if section.strip()]

    def handle_result(result: dict[str, Any]) -> None:
        report_filename = f"{result['uuid']}.txt"
        (output_dir / report_filename).write_text(result.get("report_text", ""), encoding="utf-8")
        manifest_rows.append(_manifest_row(result, report_filename, label=args.label))
        print(f"[{result['status']}] {result['uuid']}", flush=True)

    with app.run():
        _collect_workload_results(
            workload_uuids,
            batch_size=args.batch_size,
            spawn_fn=lambda uuid: capture_prefill_ncu_workload.spawn(
                solution_payload,
                args.definition,
                uuid,
                kernel_pattern,
                sections,
                args.launch_skip,
                args.launch_count,
                args.timeout,
                runtime_env,
            ),
            result_callback=handle_result,
        )

    manifest_path = args.manifest_json or output_dir / "manifest.json"
    manifest_payload = {
        "metadata": {
            "definition": args.definition,
            "kernel_pattern": kernel_pattern,
            "label": args.label,
            "sections": sections,
            "launch_skip": args.launch_skip,
            "launch_count": args.launch_count,
            "runtime_env": runtime_env,
            "workload_count": len(workload_uuids),
            "solution": solution_provenance,
        },
        "reports": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved_manifest={manifest_path}")
    return 0


@app.local_entrypoint()
def modal_main(
    trace_set_path: str = "",
    definition: str = "gdn_prefill_qk4_v8_d128_k_last",
    solution_source: str = PACKED_SUBMIT_SOLUTION_SOURCE,
    baseline_solution_index: int = 0,
    workload_uuids: str = "",
    representative_groups_json: str = "",
    limit: int = 0,
    batch_size: int = 8,
    kernel_pattern: str = "",
    launch_skip: int = 1,
    launch_count: int = 1,
    sections_csv: str = ",".join(DEFAULT_SECTIONS),
    timeout: int = 1800,
    output_dir: str = ".omx/profiles/full-prefill-ncu",
    manifest_json: str = "",
    label: str = "baseline",
    persistent_policy: str = "",
    persistent_auto_max_batch: int = 0,
    persistent_auto_max_seq_len: int = 0,
    adaptive_selector_keys: str = "",
):
    argv = [
        "--definition",
        definition,
        "--solution-source",
        solution_source,
        "--baseline-solution-index",
        str(baseline_solution_index),
        "--workload-uuids",
        workload_uuids,
        "--limit",
        str(limit),
        "--batch-size",
        str(batch_size),
        "--launch-skip",
        str(launch_skip),
        "--launch-count",
        str(launch_count),
        "--sections-csv",
        sections_csv,
        "--timeout",
        str(timeout),
        "--output-dir",
        output_dir,
        "--label",
        label,
    ]
    if kernel_pattern:
        argv.extend(["--kernel-pattern", kernel_pattern])
    if trace_set_path:
        argv.extend(["--trace-set-path", trace_set_path])
    if representative_groups_json:
        argv.extend(["--representative-groups-json", representative_groups_json])
    if manifest_json:
        argv.extend(["--manifest-json", manifest_json])
    if persistent_policy:
        argv.extend(["--persistent-policy", persistent_policy])
    if persistent_auto_max_batch > 0:
        argv.extend(["--persistent-auto-max-batch", str(persistent_auto_max_batch)])
    if persistent_auto_max_seq_len > 0:
        argv.extend(["--persistent-auto-max-seq-len", str(persistent_auto_max_seq_len)])
    if adaptive_selector_keys:
        argv.extend(["--adaptive-selector-keys", adaptive_selector_keys])

    raise SystemExit(main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
