from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_results import load_results_rows
from scripts.ncu_scorecard import detect_sections, parse_ncu_metrics

OFFICIAL_AGGREGATE = {
    "avg_speedup": 2.486,
    "baseline_avg_latency_us": 307.51,
    "submit_avg_latency_us": 192.27,
    "speedup_range": [0.762, 5.431],
}

EXTRA_METRIC_PATTERNS = {
    "dram_frequency_ghz": [r"DRAM Frequency\s+Ghz\s+([0-9]+(?:\.[0-9]+)?)"],
    "sm_frequency_ghz": [r"SM Frequency\s+Ghz\s+([0-9]+(?:\.[0-9]+)?)"],
    "elapsed_cycles": [r"Elapsed Cycles\s+cycle\s+([0-9]+(?:\.[0-9]+)?)"],
    "dram_throughput_pct": [r"DRAM Throughput\s+%\s+([0-9]+(?:\.[0-9]+)?)"],
    "duration_us": [r"Duration\s+us\s+([0-9]+(?:\.[0-9]+)?)"],
    "l1tex_throughput_pct": [r"L1/TEX Cache Throughput\s+%\s+([0-9]+(?:\.[0-9]+)?)"],
    "l2_throughput_pct": [r"L2 Cache Throughput\s+%\s+([0-9]+(?:\.[0-9]+)?)"],
    "sm_active_cycles": [r"SM Active Cycles\s+cycle\s+([0-9]+(?:\.[0-9]+)?)"],
    "memory_throughput_gbps": [r"Memory Throughput\s+Gbyte/s\s+([0-9]+(?:\.[0-9]+)?)"],
    "mem_busy_pct": [r"Mem Busy\s+%\s+([0-9]+(?:\.[0-9]+)?)"],
    "l1tex_hit_rate_pct": [r"L1/TEX Hit Rate\s+%\s+([0-9]+(?:\.[0-9]+)?)"],
    "l2_hit_rate_pct": [r"L2 Hit Rate\s+%\s+([0-9]+(?:\.[0-9]+)?)"],
    "local_spill_requests": [r"Local Memory Spilling Requests\s+([0-9]+(?:\.[0-9]+)?)"],
    "warp_cycles_per_issued_inst": [r"Warp Cycles Per Issued Instruction\s+cycle\s+([0-9]+(?:\.[0-9]+)?)"],
    "block_size": [r"Block Size\s+([0-9]+(?:\.[0-9]+)?)"],
    "grid_size": [r"Grid Size\s+([0-9]+(?:\.[0-9]+)?)"],
    "regs_per_thread": [r"Registers Per Thread\s+register/thread\s+([0-9]+(?:\.[0-9]+)?)"],
    "dynamic_smem_kb": [r"Dynamic Shared Memory Per Block\s+Kbyte/block\s+([0-9]+(?:\.[0-9]+)?)"],
    "waves_per_sm": [r"Waves Per SM\s+([0-9]+(?:\.[0-9]+)?)"],
}

KERNEL_NAME_PATTERNS = (
    re.compile(r'==PROF== Profiling "([^"]+)"'),
    re.compile(r"^\s+(kernel_[^\s]+)\s+\(", flags=re.MULTILINE),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join full-workload submit/baseline NCU artifacts and emit attribution report.")
    parser.add_argument(
        "--inventory-json",
        type=Path,
        default=Path(".omx/results/prefill-workload-inventory-with-benchmark.json"),
    )
    parser.add_argument(
        "--submit-manifest",
        type=Path,
        default=Path(".omx/profiles/full-prefill-ncu/manifest.json"),
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=Path(".omx/profiles/flashinfer-baseline-ncu-full/manifest.json"),
    )
    parser.add_argument(
        "--pair-json",
        type=Path,
        default=Path(".omx/results/flashinfer-baseline-vs-submit-cutlass-ncu-pairs.json"),
    )
    parser.add_argument("--submit-benchmark-json", type=Path, default=None)
    parser.add_argument("--baseline-benchmark-json", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(".omx/results/full-workload-flashinfer-submit-ncu-attribution.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(".omx/reports/full-workload-flashinfer-submit-ncu-attribution.md"),
    )
    return parser.parse_args(argv)


def _extract_first_float(raw_text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_kernel_names(raw_text: str) -> list[str]:
    kernel_names: list[str] = []
    for pattern in KERNEL_NAME_PATTERNS:
        for match in pattern.finditer(raw_text):
            kernel_name = match.group(1)
            if kernel_name not in kernel_names:
                kernel_names.append(kernel_name)
    return kernel_names


def parse_extended_ncu_metrics(raw_text: str) -> dict[str, Any]:
    metrics = parse_ncu_metrics(raw_text)
    for name, patterns in EXTRA_METRIC_PATTERNS.items():
        metrics[name] = _extract_first_float(raw_text, patterns)
    elapsed = metrics.get("elapsed_cycles")
    sm_active = metrics.get("sm_active_cycles")
    metrics["sm_active_ratio"] = (
        float(sm_active) / float(elapsed)
        if isinstance(elapsed, float) and elapsed > 0 and isinstance(sm_active, float)
        else None
    )
    return metrics


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if isinstance(value, (int, float)) and not math.isnan(value)]
    if not filtered:
        return None
    return statistics.fmean(filtered)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_inventory_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Inventory rows must be a list: {path}")
    return {
        row["uuid"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("uuid"), str)
    }


def load_manifest_rows(manifest_path: Path) -> dict[str, Any]:
    payload = _load_json(manifest_path)
    if payload is None:
        return {"metadata": {}, "rows": {}}
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    report_rows = payload.get("reports", []) if isinstance(payload, dict) else []
    base_dir = manifest_path.parent
    rows: dict[str, dict[str, Any]] = {}
    for row in report_rows:
        if not isinstance(row, dict):
            continue
        uuid = row.get("uuid")
        if not isinstance(uuid, str):
            continue
        report_path = base_dir / row["report_path"] if row.get("report_path") else None
        raw_text = report_path.read_text(encoding="utf-8") if report_path and report_path.exists() else ""
        kernel_names = _extract_kernel_names(raw_text) if raw_text else []
        parsed_metrics = parse_extended_ncu_metrics(raw_text) if raw_text else {}
        notes = list(row.get("notes", []))
        if raw_text and "No kernels were profiled" in raw_text and "no_kernels_profiled" not in notes:
            notes.append("no_kernels_profiled")
        rows[uuid] = {
            **row,
            "report_path": str(report_path) if report_path else row.get("report_path"),
            "detected_sections": detect_sections(raw_text) if raw_text else [],
            "kernel_names": kernel_names,
            "first_kernel_name": kernel_names[0] if kernel_names else None,
            "metrics": parsed_metrics,
            "valid_cutlass": bool(kernel_names) and kernel_names[0].startswith("kernel_cutlass_kernel_fib"),
            "raw_text_present": bool(raw_text),
            "notes": notes,
        }
    return {"metadata": metadata, "rows": rows}


def load_pair_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if payload is None:
        return {}
    pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
    return {
        pair["uuid"]: pair
        for pair in pairs
        if isinstance(pair, dict) and isinstance(pair.get("uuid"), str)
    }


def load_benchmark_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"metadata": {}, "rows": {}}
    payload, _, rows = load_results_rows(path)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return {
        "metadata": metadata if isinstance(metadata, dict) else {},
        "rows": {
            row["uuid"]: row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("uuid"), str)
        },
    }


def ncu_baseline_provenance_check(metadata: dict[str, Any]) -> dict[str, Any]:
    solution = metadata.get("solution", {}) if isinstance(metadata, dict) else {}
    return {
        "solution_source": solution.get("solution_source"),
        "solution_name": solution.get("solution_name"),
        "solution_author": solution.get("solution_author"),
        "entry_point": solution.get("entry_point"),
        "destination_passing_style": solution.get("destination_passing_style"),
        "valid": (
            solution.get("solution_source") == "trace-set-baseline"
            and solution.get("solution_name") == "flashinfer_wrapper_123ca6"
            and solution.get("solution_author") == "flashinfer"
            and solution.get("entry_point") == "main.py::run"
            and solution.get("destination_passing_style") is False
        ),
    }


def latency_baseline_provenance_check(metadata: dict[str, Any]) -> dict[str, Any]:
    solution = metadata.get("solution_provenance", {}) if isinstance(metadata, dict) else {}
    return {
        "solution_json": solution.get("solution_json"),
        "solution_name": solution.get("solution_name"),
        "solution_definition": solution.get("solution_definition"),
        "solution_author": solution.get("solution_author"),
        "entry_point": solution.get("entry_point"),
        "destination_passing_style": solution.get("destination_passing_style"),
        "valid": (
            solution.get("solution_name") == "flashinfer_wrapper_123ca6"
            and solution.get("solution_definition") == "gdn_prefill_qk4_v8_d128_k_last"
            and solution.get("solution_author") == "flashinfer"
            and solution.get("entry_point") == "main.py::run"
            and solution.get("destination_passing_style") is False
        ),
    }


def build_latency_entry(
    uuid: str,
    *,
    inventory_row: dict[str, Any] | None,
    submit_benchmark_rows: dict[str, dict[str, Any]],
    baseline_benchmark_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if uuid in submit_benchmark_rows and uuid in baseline_benchmark_rows:
        submit_row = submit_benchmark_rows[uuid]
        baseline_row = baseline_benchmark_rows[uuid]
        submit_latency = submit_row.get("latency_ms")
        baseline_latency = baseline_row.get("latency_ms")
        return {
            "source": "reproduced_flashinfer_baseline_vs_submit",
            "submit_latency_ms": submit_latency,
            "baseline_latency_ms": baseline_latency,
            "speedup_factor": _safe_ratio(baseline_latency, submit_latency),
        }

    benchmark_result = inventory_row.get("benchmark_result") if isinstance(inventory_row, dict) else None
    if isinstance(benchmark_result, dict):
        return {
            "source": "submit_vs_simple_reference",
            "submit_latency_ms": benchmark_result.get("latency_ms"),
            "baseline_latency_ms": benchmark_result.get("reference_latency_ms"),
            "speedup_factor": benchmark_result.get("speedup_factor"),
        }

    return {
        "source": "aggregate_only",
        "submit_latency_ms": None,
        "baseline_latency_ms": None,
        "speedup_factor": None,
    }


def kernel_signal_from_ratios(duration_ratio: float | None, cycle_ratio: float | None) -> str:
    if duration_ratio is None or cycle_ratio is None:
        return "unknown"
    if duration_ratio <= 0.95 and cycle_ratio <= 0.95:
        return "kernel_win"
    if duration_ratio >= 1.05 and cycle_ratio >= 1.05:
        return "kernel_regression"
    if abs(duration_ratio - 1.0) <= 0.05 and abs(cycle_ratio - 1.0) <= 0.05:
        return "kernel_near_parity"
    return "kernel_mixed"


def classify_row(
    *,
    latency: dict[str, Any],
    kernel_signal: str,
    submit_metrics: dict[str, Any],
    kernel_duration_ratio: float | None,
    kernel_cycle_ratio: float | None,
) -> str:
    speedup = latency.get("speedup_factor")
    if latency.get("authoritative") is not True or speedup is None:
        return "unknown"
    if kernel_signal == "kernel_win" and speedup > 1.0:
        return "kernel_win"
    if kernel_signal == "kernel_near_parity" and speedup > 1.0:
        return "path_win"
    if (
        speedup < 1.0
        and (
            (submit_metrics.get("waves_per_sm") or 0.0) <= 0.2
            or (submit_metrics.get("skipped_issue_slots") or 0.0) >= 80.0
        )
    ):
        return "underfilled_regression"
    if kernel_duration_ratio is not None or kernel_cycle_ratio is not None:
        return "mixed"
    return "unknown"


def _merge_metrics(
    manifest_row: dict[str, Any] | None,
    pair_section: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str], str | None]:
    notes: list[str] = []
    if manifest_row is None and pair_section is None:
        return {}, ["missing_ncu_row"], None

    if pair_section is not None:
        if manifest_row is not None:
            notes.append("pair_artifact_override")
        return dict(pair_section), notes, "pair_artifact"

    assert manifest_row is not None
    metrics = dict(manifest_row.get("metrics", {}))
    if not manifest_row.get("raw_text_present"):
        notes.append("missing_report_text")
    if "no_kernels_profiled" in manifest_row.get("notes", []):
        notes.append("no_kernels_profiled")
    if not manifest_row.get("valid_cutlass"):
        notes.append("non_cutlass_or_missing_kernel")
    return metrics, notes, "manifest"


def build_rows(
    *,
    inventory_rows: dict[str, dict[str, Any]],
    submit_manifest: dict[str, Any],
    baseline_manifest: dict[str, Any],
    pair_rows: dict[str, dict[str, Any]],
    submit_benchmark_rows: dict[str, dict[str, Any]],
    baseline_benchmark_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_uuids = sorted(
        set(inventory_rows)
        | set(submit_manifest["rows"])
        | set(baseline_manifest["rows"])
        | set(pair_rows)
        | set(submit_benchmark_rows)
        | set(baseline_benchmark_rows)
    )
    for uuid in all_uuids:
        inventory_row = inventory_rows.get(uuid, {})
        runtime_summary = inventory_row.get("runtime_summary", {}) if isinstance(inventory_row, dict) else {}
        submit_manifest_row = submit_manifest["rows"].get(uuid)
        baseline_manifest_row = baseline_manifest["rows"].get(uuid)
        pair_row = pair_rows.get(uuid, {})

        submit_metrics, submit_notes, submit_source = _merge_metrics(
            submit_manifest_row,
            pair_row.get("submit") if isinstance(pair_row, dict) else None,
        )
        baseline_metrics, baseline_notes, baseline_source = _merge_metrics(
            baseline_manifest_row,
            pair_row.get("baseline") if isinstance(pair_row, dict) else None,
        )
        latency = build_latency_entry(
            uuid,
            inventory_row=inventory_row,
            submit_benchmark_rows=submit_benchmark_rows,
            baseline_benchmark_rows=baseline_benchmark_rows,
        )
        kernel_duration_ratio = _safe_ratio(
            baseline_metrics.get("duration_us"),
            submit_metrics.get("duration_us"),
        )
        kernel_cycle_ratio = _safe_ratio(
            baseline_metrics.get("elapsed_cycles"),
            submit_metrics.get("elapsed_cycles"),
        )
        sm_active_cycle_ratio = _safe_ratio(
            baseline_metrics.get("sm_active_cycles"),
            submit_metrics.get("sm_active_cycles"),
        )
        kernel_signal = kernel_signal_from_ratios(kernel_duration_ratio, kernel_cycle_ratio)
        blockers = []
        if not baseline_metrics:
            blockers.append("missing_baseline_ncu")
        if not submit_metrics:
            blockers.append("missing_submit_ncu")
        rows.append(
            {
                "uuid": uuid,
                "definition": inventory_row.get("definition"),
                "axes": runtime_summary.get("axes", {}),
                "runtime_summary": runtime_summary,
                "latency": latency,
                "submit_ncu": {
                    "source": submit_source,
                    "notes": submit_notes,
                    "metrics": submit_metrics,
                    "manifest_row": submit_manifest_row,
                },
                "baseline_ncu": {
                    "source": baseline_source,
                    "notes": baseline_notes,
                    "metrics": baseline_metrics,
                    "manifest_row": baseline_manifest_row,
                },
                "kernel_ratios": {
                    "duration_ratio": kernel_duration_ratio,
                    "cycle_ratio": kernel_cycle_ratio,
                    "sm_active_cycle_ratio": sm_active_cycle_ratio,
                },
                "kernel_signal": kernel_signal,
                "classification": "pending_latency_gate",
                "blockers": blockers,
                "pair_context": {
                    "label": pair_row.get("label"),
                    "selector": pair_row.get("selector"),
                    "group": pair_row.get("group"),
                }
                if isinstance(pair_row, dict) and pair_row
                else None,
            }
        )
    return rows


def build_summary(
    rows: list[dict[str, Any]],
    baseline_manifest: dict[str, Any],
    baseline_benchmark_metadata: dict[str, Any],
) -> dict[str, Any]:
    kernel_duration_ratios = [row["kernel_ratios"]["duration_ratio"] for row in rows]
    kernel_cycle_ratios = [row["kernel_ratios"]["cycle_ratio"] for row in rows]
    sm_active_cycle_ratios = [row["kernel_ratios"]["sm_active_cycle_ratio"] for row in rows]
    kernel_signal_counts: dict[str, int] = {}
    classification_counts: dict[str, int] = {}
    for row in rows:
        kernel_signal_counts[row["kernel_signal"]] = kernel_signal_counts.get(row["kernel_signal"], 0) + 1
        classification_counts[row["classification"]] = classification_counts.get(row["classification"], 0) + 1
    reproduced_rows = [row for row in rows if row["latency"]["source"] == "reproduced_flashinfer_baseline_vs_submit"]
    reproduced_submit_latencies_us = [
        row["latency"]["submit_latency_ms"] * 1000.0
        for row in reproduced_rows
        if row["latency"].get("submit_latency_ms") is not None
    ]
    reproduced_baseline_latencies_us = [
        row["latency"]["baseline_latency_ms"] * 1000.0
        for row in reproduced_rows
        if row["latency"].get("baseline_latency_ms") is not None
    ]
    reproduced_speedups = [
        row["latency"]["speedup_factor"]
        for row in reproduced_rows
        if row["latency"].get("speedup_factor") is not None
    ]
    reproduced_summary = {
        "row_count": len(reproduced_rows),
        "submit_mean_latency_us": _mean(reproduced_submit_latencies_us),
        "baseline_mean_latency_us": _mean(reproduced_baseline_latencies_us),
        "arithmetic_mean_speedup": _mean(reproduced_speedups),
        "speedup_min": min(reproduced_speedups) if reproduced_speedups else None,
        "speedup_max": max(reproduced_speedups) if reproduced_speedups else None,
    }
    reproduced_gate = {
        "submit_mean_within_10pct": (
            reproduced_summary["submit_mean_latency_us"] is not None
            and abs(reproduced_summary["submit_mean_latency_us"] - OFFICIAL_AGGREGATE["submit_avg_latency_us"])
            / OFFICIAL_AGGREGATE["submit_avg_latency_us"]
            <= 0.10
        ),
        "baseline_mean_within_10pct": (
            reproduced_summary["baseline_mean_latency_us"] is not None
            and abs(reproduced_summary["baseline_mean_latency_us"] - OFFICIAL_AGGREGATE["baseline_avg_latency_us"])
            / OFFICIAL_AGGREGATE["baseline_avg_latency_us"]
            <= 0.10
        ),
        "speedup_mean_within_10pct": (
            reproduced_summary["arithmetic_mean_speedup"] is not None
            and abs(reproduced_summary["arithmetic_mean_speedup"] - OFFICIAL_AGGREGATE["avg_speedup"])
            / OFFICIAL_AGGREGATE["avg_speedup"]
            <= 0.10
        ),
        "speedup_range_brackets_official": (
            reproduced_summary["speedup_min"] is not None
            and reproduced_summary["speedup_max"] is not None
            and reproduced_summary["speedup_min"] <= OFFICIAL_AGGREGATE["speedup_range"][0] * 1.10
            and reproduced_summary["speedup_max"] >= OFFICIAL_AGGREGATE["speedup_range"][1] * 0.90
        ),
    }
    reproduced_gate["passed"] = all(reproduced_gate.values()) if reproduced_rows else False
    authoritative_rows = [row for row in rows if row["classification"] != "pending_latency_gate" and row["latency"].get("authoritative") is True]
    ncu_provenance = ncu_baseline_provenance_check(baseline_manifest.get("metadata", {}))
    latency_provenance = latency_baseline_provenance_check(baseline_benchmark_metadata)
    return {
        "row_count": len(rows),
        "baseline_provenance": {
            "ncu_capture": ncu_provenance,
            "latency_benchmark": latency_provenance,
            "coupled_valid": ncu_provenance["valid"] and latency_provenance["valid"],
        },
        "kernel_pair_count": sum(
            1
            for row in rows
            if row["kernel_ratios"]["duration_ratio"] is not None and row["kernel_ratios"]["cycle_ratio"] is not None
        ),
        "reproduced_latency_pair_count": len(reproduced_rows),
        "authoritative_latency_pair_count": len(authoritative_rows),
        "kernel_signal_counts": kernel_signal_counts,
        "classification_counts": classification_counts,
        "mean_kernel_duration_ratio": _mean(kernel_duration_ratios),
        "mean_kernel_cycle_ratio": _mean(kernel_cycle_ratios),
        "mean_sm_active_cycle_ratio": _mean(sm_active_cycle_ratios),
        "mean_authoritative_speedup": _mean([row["latency"]["speedup_factor"] for row in authoritative_rows]),
        "reproduced_summary": reproduced_summary,
        "reproduced_gate": reproduced_gate,
        "official_aggregate": {
            **OFFICIAL_AGGREGATE,
            "ratio_of_average_latencies": OFFICIAL_AGGREGATE["baseline_avg_latency_us"]
            / OFFICIAL_AGGREGATE["submit_avg_latency_us"],
        },
    }


def finalize_rows(rows: list[dict[str, Any]], reproduced_gate_passed: bool, coupled_baseline_valid: bool) -> None:
    for row in rows:
        latency = row["latency"]
        latency["authoritative"] = (
            latency["source"] == "reproduced_flashinfer_baseline_vs_submit"
            and reproduced_gate_passed
            and coupled_baseline_valid
        )
        if latency["source"] == "reproduced_flashinfer_baseline_vs_submit" and not reproduced_gate_passed:
            if "reproduced_latency_gate_failed" not in row["blockers"]:
                row["blockers"].append("reproduced_latency_gate_failed")
        if latency["source"] == "reproduced_flashinfer_baseline_vs_submit" and not coupled_baseline_valid:
            if "latency_baseline_provenance_invalid" not in row["blockers"]:
                row["blockers"].append("latency_baseline_provenance_invalid")
        elif latency["authoritative"] is not True and f"latency_source={latency['source']}" not in row["blockers"]:
            row["blockers"].append(f"latency_source={latency['source']}")

        row["classification"] = classify_row(
            latency=latency,
            kernel_signal=row["kernel_signal"],
            submit_metrics=row["submit_ncu"]["metrics"],
            kernel_duration_ratio=row["kernel_ratios"]["duration_ratio"],
            kernel_cycle_ratio=row["kernel_ratios"]["cycle_ratio"],
        )


def render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    baseline_provenance = summary["baseline_provenance"]
    ncu_provenance = baseline_provenance["ncu_capture"]
    latency_provenance = baseline_provenance["latency_benchmark"]
    reproduced_summary = summary["reproduced_summary"]
    reproduced_gate = summary["reproduced_gate"]
    top_blocked = [row for row in rows if row["classification"] == "unknown"][:5]
    pair_examples = [row for row in rows if row["kernel_signal"] != "unknown"][:6]
    lines = [
        "# Full-workload FlashInfer baseline vs submit NCU attribution",
        "",
        "## Aggregate framing",
        "",
        f"- Official arithmetic mean speedup: {summary['official_aggregate']['avg_speedup']:.3f}x",
        (
            "- Ratio of average latencies: "
            f"{summary['official_aggregate']['baseline_avg_latency_us']:.2f} / "
            f"{summary['official_aggregate']['submit_avg_latency_us']:.2f} = "
            f"{summary['official_aggregate']['ratio_of_average_latencies']:.3f}x"
        ),
        f"- Joined workload rows: {summary['row_count']}",
        f"- Kernel pairs with duration/cycle ratios: {summary['kernel_pair_count']}",
        f"- Reproduced latency rows: {summary['reproduced_latency_pair_count']}",
        f"- Authoritative baseline-vs-submit latency rows: {summary['authoritative_latency_pair_count']}",
        "",
        "## Provenance gate",
        "",
        f"- NCU baseline provenance valid: {ncu_provenance['valid']}",
        f"- NCU baseline solution source: {ncu_provenance['solution_source']}",
        f"- NCU baseline solution name: {ncu_provenance['solution_name']}",
        f"- NCU baseline author: {ncu_provenance['solution_author']}",
        f"- NCU baseline entry point: {ncu_provenance['entry_point']}",
        (
            "- NCU baseline destination_passing_style: "
            f"{ncu_provenance['destination_passing_style']}"
        ),
        f"- Latency baseline provenance valid: {latency_provenance['valid']}",
        f"- Latency baseline solution json: {latency_provenance['solution_json']}",
        f"- Latency baseline solution name: {latency_provenance['solution_name']}",
        f"- Latency baseline author: {latency_provenance['solution_author']}",
        f"- Latency baseline entry point: {latency_provenance['entry_point']}",
        f"- Coupled baseline provenance valid: {baseline_provenance['coupled_valid']}",
        "",
        "## Current measured kernel signal",
        "",
        f"- Mean kernel duration ratio (baseline/submit): {summary['mean_kernel_duration_ratio']}",
        f"- Mean kernel cycle ratio (baseline/submit): {summary['mean_kernel_cycle_ratio']}",
        f"- Mean SM active cycle ratio (baseline/submit): {summary['mean_sm_active_cycle_ratio']}",
        f"- Kernel signal counts: {json.dumps(summary['kernel_signal_counts'], sort_keys=True)}",
        f"- Authoritative classification counts: {json.dumps(summary['classification_counts'], sort_keys=True)}",
        "",
        "## Reproduced-latency gate",
        "",
        f"- Reproduced submit mean latency (us): {reproduced_summary['submit_mean_latency_us']}",
        f"- Reproduced baseline mean latency (us): {reproduced_summary['baseline_mean_latency_us']}",
        f"- Reproduced arithmetic mean speedup: {reproduced_summary['arithmetic_mean_speedup']}",
        f"- Reproduced speedup range: [{reproduced_summary['speedup_min']}, {reproduced_summary['speedup_max']}]",
        f"- Submit mean within 10%: {reproduced_gate['submit_mean_within_10pct']}",
        f"- Baseline mean within 10%: {reproduced_gate['baseline_mean_within_10pct']}",
        f"- Speedup mean within 10%: {reproduced_gate['speedup_mean_within_10pct']}",
        f"- Speedup range brackets official: {reproduced_gate['speedup_range_brackets_official']}",
        f"- Reproduced-latency gate passed: {reproduced_gate['passed']}",
        "",
        "## Interpretation",
        "",
    ]
    if summary["kernel_pair_count"] > 0 and reproduced_gate["passed"] and baseline_provenance["coupled_valid"]:
        lines.extend(
            [
                "- Measured CUTLASS pairs remain near parity overall; the current evidence does not support the claim that the inner CUTLASS kernel body itself became 2.486x faster.",
                "- The reproduced baseline-vs-submit latency rows passed the aggregate gate, so row-level class attribution is usable for this run.",
                "- Therefore current component-level explanations outside the filtered CUTLASS row remain hypotheses until reproduced/official latency rows and launch accounting are added.",
            ]
        )
    elif summary["kernel_pair_count"] > 0 and baseline_provenance["coupled_valid"] is False:
        lines.extend(
            [
                "- Measured CUTLASS pairs remain near parity overall; the current evidence does not support the claim that the inner CUTLASS kernel body itself became 2.486x faster.",
                "- The latency denominator is not provenance-coupled strongly enough to the same baseline identity used for NCU capture, so whole-run attribution remains blocked.",
                "- Therefore the safe conclusion is limited to kernel-body parity plus blocked whole-run attribution.",
            ]
        )
    elif summary["kernel_pair_count"] > 0:
        lines.extend(
            [
                "- Measured CUTLASS pairs remain near parity overall; the current evidence does not support the claim that the inner CUTLASS kernel body itself became 2.486x faster.",
                "- The reproduced baseline-vs-submit latency rows failed the aggregate gate against the official result, so per-workload speedup attribution remains non-authoritative for this run.",
                "- Therefore the safe conclusion is limited to kernel-body parity plus a blocked attribution for the official 2.486x distribution.",
            ]
        )
    else:
        lines.append("- No valid baseline-vs-submit CUTLASS pairs were available; attribution remains blocked.")

    lines.extend(["", "## Example paired rows", ""])
    if pair_examples:
        for row in pair_examples:
            lines.extend(
                [
                    f"### {row['uuid']}",
                    f"- axes: {json.dumps(row['axes'], sort_keys=True)}",
                    f"- kernel_signal: {row['kernel_signal']}",
                    f"- duration_ratio: {row['kernel_ratios']['duration_ratio']}",
                    f"- cycle_ratio: {row['kernel_ratios']['cycle_ratio']}",
                    f"- sm_active_cycle_ratio: {row['kernel_ratios']['sm_active_cycle_ratio']}",
                    f"- latency_source: {row['latency']['source']}",
                    "",
                ]
            )
    else:
        lines.append("- none")

    lines.extend(["## Blockers", ""])
    if baseline_provenance["coupled_valid"] is False:
        lines.append("- Global blocker: baseline provenance is not coupled strongly enough to the latency benchmark artifact.")
    if reproduced_gate["passed"] is False:
        lines.append("- Global blocker: reproduced latency does not match the official aggregate closely enough, so row-level class responsibility remains blocked.")
    if top_blocked:
        for row in top_blocked:
            lines.append(f"- {row['uuid']}: {', '.join(row['blockers'])}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "- This report intentionally separates the official aggregate result from locally reproduced or non-authoritative per-workload rows.",
            "- `submit_vs_simple_reference` latency rows are useful for workload context, but they are not the official FlashInfer-baseline denominator.",
            "- No adaptive-policy result is used here as explanatory evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inventory_rows = load_inventory_rows(args.inventory_json)
    submit_manifest = load_manifest_rows(args.submit_manifest)
    baseline_manifest = load_manifest_rows(args.baseline_manifest)
    pair_rows = load_pair_rows(args.pair_json)
    submit_benchmark = load_benchmark_payload(args.submit_benchmark_json)
    baseline_benchmark = load_benchmark_payload(args.baseline_benchmark_json)

    rows = build_rows(
        inventory_rows=inventory_rows,
        submit_manifest=submit_manifest,
        baseline_manifest=baseline_manifest,
        pair_rows=pair_rows,
        submit_benchmark_rows=submit_benchmark["rows"],
        baseline_benchmark_rows=baseline_benchmark["rows"],
    )
    interim_summary = build_summary(rows, baseline_manifest, baseline_benchmark["metadata"])
    finalize_rows(
        rows,
        interim_summary["reproduced_gate"]["passed"],
        interim_summary["baseline_provenance"]["coupled_valid"],
    )
    summary = build_summary(rows, baseline_manifest, baseline_benchmark["metadata"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "metadata": {
                    "inventory_json": str(args.inventory_json),
                    "submit_manifest": str(args.submit_manifest),
                    "baseline_manifest": str(args.baseline_manifest),
                    "pair_json": str(args.pair_json),
                    "submit_benchmark_json": str(args.submit_benchmark_json) if args.submit_benchmark_json else None,
                    "baseline_benchmark_json": str(args.baseline_benchmark_json) if args.baseline_benchmark_json else None,
                },
                "summary": summary,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(render_report(summary, rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_report": str(args.output_report),
                "row_count": len(rows),
                "kernel_pair_count": summary["kernel_pair_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
