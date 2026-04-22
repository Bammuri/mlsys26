"""
Generate reusable NCU scorecard templates for canonical workload lanes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


KNOWN_SECTIONS = [
    "SchedulerStats",
    "Occupancy",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
    "LaunchStats",
    "SpeedOfLight",
    "Roofline",
]


def detect_sections(raw_text: str) -> list[str]:
    found = []
    for section in KNOWN_SECTIONS:
        pattern = re.compile(re.escape(section), re.IGNORECASE)
        if pattern.search(raw_text):
            found.append(section)
    return found


def build_scorecard_entry(
    lane_name: str,
    workload_entry: dict[str, Any],
    *,
    source_report: str | None = None,
    detected_sections: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "lane": lane_name,
        "uuid": workload_entry["uuid"],
        "latency_ms_reference": workload_entry.get("latency_ms"),
        "axes": workload_entry.get("axes"),
        "source_report": source_report,
        "detected_sections": detected_sections or [],
        "primary": {
            "scheduler_health": {
                "section": "SchedulerStats / WarpStateStats",
                "metrics": {
                    "issue_efficiency": None,
                    "skipped_issue_slots": None,
                    "eligible_warps_per_scheduler": None,
                    "issued_warps_per_scheduler": None,
                },
                "interpretation": "Focus on stalls only if schedulers fail to issue consistently.",
                "verdict": "unknown",
            },
            "occupancy_effectiveness": {
                "section": "Occupancy / LaunchStats",
                "metrics": {
                    "theoretical_occupancy_pct": None,
                    "achieved_occupancy_pct": None,
                    "occupancy_gap_pct": None,
                },
                "interpretation": "Low occupancy hurts latency hiding; large theory-vs-achieved gap implies imbalance.",
                "verdict": "unknown",
            },
            "bound_classification": {
                "section": "SpeedOfLight / MemoryWorkloadAnalysis / Roofline",
                "metrics": {
                    "compute_utilization_pct": None,
                    "memory_utilization_pct": None,
                },
                "classification": "unknown",
                "interpretation": "Use roofline + SOL to identify whether the kernel is memory- or compute/scheduler-limited.",
                "verdict": "unknown",
            },
            "cross_lane_consistency": {
                "section": "Cross-lane comparison",
                "interpretation": "Tail wins must not create fast-floor structural regressions.",
                "verdict": "unknown",
            },
        },
        "secondary": {
            "paired_latency_ms": None,
            "trial_1_vs_steady_state": "unknown",
            "broad_regression_screen": "unknown",
        },
        "notes": [],
    }


def build_scorecard_payload(
    canonical_lanes: dict[str, Any],
    *,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    scorecards = []
    report_text = {}
    if report_dir is not None and report_dir.exists():
        for report_path in report_dir.glob("*.txt"):
            report_text[report_path.name] = report_path.read_text(encoding="utf-8", errors="ignore")

    for lane_name, entries in canonical_lanes["lanes"].items():
        for entry in entries:
            source_report = None
            detected = []
            if report_text:
                matches = [name for name in report_text if entry["uuid"][:8] in name or entry["uuid"] in name]
                if matches:
                    source_report = matches[0]
                    detected = detect_sections(report_text[source_report])
            scorecards.append(
                build_scorecard_entry(
                    lane_name,
                    entry,
                    source_report=source_report,
                    detected_sections=detected,
                )
            )

    return {
        "metadata": canonical_lanes["metadata"],
        "scorecards": scorecards,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# NCU scorecard template: {payload['metadata']['definition']}",
        "",
        "Use this sheet to evaluate bare-metal NCU captures lane-by-lane.",
        "",
    ]
    for scorecard in payload["scorecards"]:
        lines.extend(
            [
                f"## {scorecard['lane']} — {scorecard['uuid']}",
                f"- reference latency: {scorecard['latency_ms_reference']}",
                f"- axes: {scorecard.get('axes')}",
                f"- source report: {scorecard.get('source_report')}",
                f"- detected sections: {', '.join(scorecard.get('detected_sections', [])) or '(none)'}",
                "",
                "### Primary criteria",
            ]
        )
        for key, section in scorecard["primary"].items():
            lines.append(f"- **{key}** [{section['section']}]")
            lines.append(f"  - interpretation: {section['interpretation']}")
            lines.append(f"  - verdict: {section['verdict']}")
            for metric_name, metric_value in section.get("metrics", {}).items():
                lines.append(f"  - {metric_name}: {metric_value}")
            if "classification" in section:
                lines.append(f"  - classification: {section['classification']}")
        lines.extend(
            [
                "",
                "### Secondary criteria",
                f"- paired_latency_ms: {scorecard['secondary']['paired_latency_ms']}",
                f"- trial_1_vs_steady_state: {scorecard['secondary']['trial_1_vs_steady_state']}",
                f"- broad_regression_screen: {scorecard['secondary']['broad_regression_screen']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NCU scorecard templates from canonical lanes")
    parser.add_argument("canonical_lane_json", type=Path)
    parser.add_argument("--report-dir", type=Path, default=None, help="Optional directory of saved NCU text reports")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_lanes = json.loads(args.canonical_lane_json.read_text())
    payload = build_scorecard_payload(canonical_lanes, report_dir=args.report_dir)
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown)


if __name__ == "__main__":
    main()
