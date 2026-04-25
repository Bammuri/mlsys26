from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze joined submit/baseline phase-timer cohort outputs.")
    parser.add_argument("--submit-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument(
        "--ncu-attribution-json",
        type=Path,
        default=Path(".omx/results/full-workload-flashinfer-submit-ncu-attribution.json"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(result: dict[str, Any], name: str) -> float | None:
    return result.get("metrics", {}).get(name, {}).get("median_us")


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return statistics.fmean(vals) if vals else None


def _corr(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if len(pairs) < 2:
        return None
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    xmean = statistics.fmean(xvals)
    ymean = statistics.fmean(yvals)
    num = sum((x - xmean) * (y - ymean) for x, y in pairs)
    den_x = sum((x - xmean) ** 2 for x in xvals)
    den_y = sum((y - ymean) ** 2 for y in yvals)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / math.sqrt(den_x * den_y)


def load_phase_manifest(manifest_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _load_json(manifest_path)
    rows = {}
    for row in manifest.get("results", []):
        path = manifest_path.parent / row["result_path"]
        payload = _load_json(path)
        rows[row["uuid"]] = payload.get("result", payload)
    return manifest.get("metadata", {}), rows


def load_ncu_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    return {row["uuid"]: row for row in payload.get("rows", []) if isinstance(row, dict) and row.get("uuid")}


def build_rows(submit_rows: dict[str, dict[str, Any]], baseline_rows: dict[str, dict[str, Any]], ncu_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    uuids = sorted(set(submit_rows) & set(baseline_rows))
    rows = []
    for uuid in uuids:
        s = submit_rows[uuid]
        b = baseline_rows[uuid]
        ncu = ncu_rows.get(uuid, {})
        row = {
            "uuid": uuid,
            "axes": s.get("workload", {}).get("axes"),
            "submit_total_median_us": _metric(s, "total_run_us"),
            "baseline_total_median_us": _metric(b, "total_run_us"),
            "submit_prepare_median_us": _metric(s, "prepare_gate_beta_us"),
            "baseline_prepare_median_us": _metric(b, "prepare_gate_beta_us"),
            "submit_launch_median_us": _metric(s, "launch_sync_us"),
            "baseline_launch_median_us": _metric(b, "launch_sync_us"),
            "submit_allocate_median_us": _metric(s, "allocate_outputs_us"),
            "submit_problem_size_median_us": _metric(s, "problem_size_us"),
            "submit_compile_median_us": _metric(s, "compile_lookup_us"),
            "ncu_duration_ratio": ncu.get("kernel_ratios", {}).get("duration_ratio"),
            "ncu_cycle_ratio": ncu.get("kernel_ratios", {}).get("cycle_ratio"),
            "ncu_waves_per_sm": ncu.get("submit_ncu", {}).get("metrics", {}).get("waves_per_sm"),
            "ncu_no_eligible_pct": ncu.get("submit_ncu", {}).get("metrics", {}).get("skipped_issue_slots"),
        }
        row["total_delta_us"] = (
            row["baseline_total_median_us"] - row["submit_total_median_us"]
            if row["baseline_total_median_us"] is not None and row["submit_total_median_us"] is not None
            else None
        )
        row["prepare_delta_us"] = (
            row["baseline_prepare_median_us"] - row["submit_prepare_median_us"]
            if row["baseline_prepare_median_us"] is not None and row["submit_prepare_median_us"] is not None
            else None
        )
        row["launch_delta_us"] = (
            row["baseline_launch_median_us"] - row["submit_launch_median_us"]
            if row["baseline_launch_median_us"] is not None and row["submit_launch_median_us"] is not None
            else None
        )
        row["submit_nonlaunch_overhead_us"] = sum(
            value or 0.0
            for value in [
                row["submit_allocate_median_us"],
                row["submit_problem_size_median_us"],
                row["submit_compile_median_us"],
                row["submit_prepare_median_us"],
            ]
        )
        rows.append(row)
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_deltas = [row["total_delta_us"] for row in rows]
    prepare_deltas = [row["prepare_delta_us"] for row in rows]
    launch_deltas = [row["launch_delta_us"] for row in rows]
    return {
        "row_count": len(rows),
        "mean_total_delta_us": _mean(total_deltas),
        "mean_prepare_delta_us": _mean(prepare_deltas),
        "mean_launch_delta_us": _mean(launch_deltas),
        "mean_submit_nonlaunch_overhead_us": _mean([row["submit_nonlaunch_overhead_us"] for row in rows]),
        "positive_total_delta_count": sum(1 for row in rows if isinstance(row["total_delta_us"], (int, float)) and row["total_delta_us"] > 0),
        "corr_total_vs_prepare": _corr(total_deltas, prepare_deltas),
        "corr_total_vs_launch": _corr(total_deltas, launch_deltas),
        "corr_total_vs_submit_nonlaunch_overhead": _corr(total_deltas, [row["submit_nonlaunch_overhead_us"] for row in rows]),
    }


def render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    strongest = sorted(
        [row for row in rows if isinstance(row["total_delta_us"], (int, float))],
        key=lambda row: row["total_delta_us"],
        reverse=True,
    )[:10]
    lines = [
        "# Phase-timer cohort analysis",
        "",
        f"- row_count: {summary['row_count']}",
        f"- mean_total_delta_us: {summary['mean_total_delta_us']}",
        f"- mean_prepare_delta_us: {summary['mean_prepare_delta_us']}",
        f"- mean_launch_delta_us: {summary['mean_launch_delta_us']}",
        f"- mean_submit_nonlaunch_overhead_us: {summary['mean_submit_nonlaunch_overhead_us']}",
        f"- positive_total_delta_count: {summary['positive_total_delta_count']}",
        f"- corr(total, prepare): {summary['corr_total_vs_prepare']}",
        f"- corr(total, launch): {summary['corr_total_vs_launch']}",
        f"- corr(total, submit_nonlaunch_overhead): {summary['corr_total_vs_submit_nonlaunch_overhead']}",
        "",
        "## Top positive baseline-submit total deltas",
        "",
    ]
    for row in strongest:
        lines.append(
            f"- {row['uuid']}: total_delta={row['total_delta_us']:.2f}us, "
            f"prepare_delta={row['prepare_delta_us']:.2f}us, "
            f"launch_delta={row['launch_delta_us']:.2f}us, "
            f"submit_nonlaunch_overhead={row['submit_nonlaunch_overhead_us']:.2f}us"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, submit_rows = load_phase_manifest(args.submit_manifest)
    _, baseline_rows = load_phase_manifest(args.baseline_manifest)
    ncu_rows = load_ncu_rows(args.ncu_attribution_json)
    rows = build_rows(submit_rows, baseline_rows, ncu_rows)
    summary = build_summary(rows)
    payload = {"summary": summary, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(render_report(summary, rows), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "row_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
