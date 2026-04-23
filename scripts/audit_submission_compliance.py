from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SPEC_PATH = "<spec>"
_DIRECT_FLASHINFER_PATTERNS = (
    (re.compile(r"^\s*from\s+flashinfer(?:\.|\s|$)"), "direct flashinfer import"),
    (re.compile(r"^\s*import\s+flashinfer(?:\.|\s|$)"), "direct flashinfer import"),
    (re.compile(r"\bflashinfer\.[A-Za-z_][A-Za-z0-9_]*"), "flashinfer module usage"),
)
_FLASHINFER_API_SHAPE_PATTERNS = (
    (re.compile(r"\bchunk_gated_delta_rule\b"), "FlashInfer-style API symbol"),
    (re.compile(r"FlashInfer API Layer", re.IGNORECASE), "FlashInfer API compatibility comment"),
)
_CUBLAS_PATTERNS = (
    (re.compile(r"\bcublasLt[A-Za-z0-9_]*\b"), "cuBLASLt symbol"),
    (re.compile(r"\bcublas[A-Za-z0-9_]*\b"), "cuBLAS symbol"),
)
_VENDOR_PATTERNS = (
    (re.compile(r"Copyright\s*\(c\)\s*\d{4}\s+by\s+FlashInfer\s+team", re.IGNORECASE), "FlashInfer copyright header"),
    (re.compile(r"Licensed under the Apache License, Version 2\.0", re.IGNORECASE), "Apache-2.0 license header"),
    (re.compile(r"FlashInfer API Layer", re.IGNORECASE), "FlashInfer API layer comment"),
)
_SM100A_PROOF_PATTERNS = (
    (re.compile(r"\bsm[_-]?100a\b", re.IGNORECASE), "explicit sm_100a token"),
    (re.compile(r"\bcompute[_-]?100a\b", re.IGNORECASE), "explicit compute_100a token"),
)
_SM100A_WEAK_PATTERNS = (
    (re.compile(r"\bSM100\b", re.IGNORECASE), "SM100 reference"),
    (re.compile(r"\bBlackwell\b", re.IGNORECASE), "Blackwell reference"),
    (re.compile(r"\btcgen05\b", re.IGNORECASE), "tcgen05 Blackwell tensorcore reference"),
    (re.compile(r"\bsm100_utils\b", re.IGNORECASE), "sm100 helper reference"),
)


def _load_solution_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))



def _pack_current_solution_json() -> dict[str, Any]:
    from scripts.pack_solution import pack_solution

    with tempfile.NamedTemporaryFile(prefix="packed-solution-", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with redirect_stdout(io.StringIO()):
            pack_solution(tmp_path)
        return _load_solution_json(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)



def _load_or_pack_solution(solution_json_path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if solution_json_path is not None:
        payload = _load_solution_json(solution_json_path)
        metadata = {
            "solution_json": str(solution_json_path),
            "packed_from_repo": False,
            "inspection_scope": "packed_sources_only",
        }
        return payload, metadata

    payload = _pack_current_solution_json()
    metadata = {
        "solution_json": None,
        "packed_from_repo": True,
        "inspection_scope": "packed_sources_only",
    }
    return payload, metadata



def _iter_source_lines(solution_payload: dict[str, Any]):
    for source in solution_payload.get("sources", []):
        path = str(source.get("path", "<unknown>"))
        content = source.get("content", "")
        if not isinstance(content, str):
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            yield path, line_no, line



def _make_evidence(path: str, line: int | None, snippet: str, reason: str) -> dict[str, Any]:
    evidence = {
        "path": path,
        "reason": reason,
        "snippet": snippet.strip(),
    }
    if line is not None:
        evidence["line"] = line
    return evidence



def _collect_source_matches(solution_payload: dict[str, Any], patterns: tuple[tuple[re.Pattern[str], str], ...]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for path, line_no, line in _iter_source_lines(solution_payload):
        for pattern, reason in patterns:
            if not pattern.search(line):
                continue
            key = (path, line_no, reason)
            if key in seen:
                continue
            seen.add(key)
            matches.append(_make_evidence(path, line_no, line, reason))
    return matches



def _collect_spec_dependency_matches(solution_payload: dict[str, Any], pattern: re.Pattern[str], reason: str) -> list[dict[str, Any]]:
    dependencies = solution_payload.get("spec", {}).get("dependencies", [])
    if not isinstance(dependencies, list):
        return []

    matches: list[dict[str, Any]] = []
    for index, dependency in enumerate(dependencies):
        dep_text = str(dependency)
        if not pattern.search(dep_text):
            continue
        matches.append(
            _make_evidence(
                _SPEC_PATH,
                None,
                f"dependencies[{index}] = {dep_text}",
                reason,
            )
        )
    return matches



def _build_runtime_flashinfer_report(solution_payload: dict[str, Any]) -> dict[str, Any]:
    direct_evidence = _collect_source_matches(solution_payload, _DIRECT_FLASHINFER_PATTERNS)
    dependency_clues = _collect_spec_dependency_matches(
        solution_payload,
        re.compile(r"flashinfer", re.IGNORECASE),
        "flashinfer dependency",
    )
    api_shape_clues = _collect_source_matches(solution_payload, _FLASHINFER_API_SHAPE_PATTERNS)
    evidence = [
        *direct_evidence,
        *[item for item in dependency_clues if item not in direct_evidence],
        *[item for item in api_shape_clues if item not in direct_evidence and item not in dependency_clues],
    ]

    inference: list[str] = []
    if direct_evidence:
        inference.append("Packed submission contains direct flashinfer imports/usages in packed source files, so runtime FlashInfer API usage is evidenced.")
    elif dependency_clues and api_shape_clues:
        inference.append(
            "No direct flashinfer import/call was found in packed source files, but flashinfer dependencies plus local API-shape clues suggest vendored or compatibility-layer usage rather than a direct runtime package call."
        )
    elif dependency_clues:
        inference.append(
            "A flashinfer dependency is listed in the packed spec, but no direct packed-source import/call was found; treat this as an inference clue, not direct runtime-call evidence."
        )
    elif api_shape_clues:
        inference.append(
            "No direct flashinfer import/call was found in packed sources, but local `chunk_gated_delta_rule`/`FlashInfer API Layer` references suggest a vendored FlashInfer-style API surface instead of the runtime package."
        )
    else:
        inference.append("No direct FlashInfer runtime API usage was found in the packed submission.")

    return {
        "status": "direct_evidence_found" if direct_evidence else "no_direct_evidence",
        "evidence": evidence,
        "inference": inference,
    }



def _build_cublas_report(solution_payload: dict[str, Any]) -> dict[str, Any]:
    evidence = _collect_source_matches(solution_payload, _CUBLAS_PATTERNS)
    cutlass_evidence = _collect_source_matches(
        solution_payload,
        ((re.compile(r"\bcutlass\b", re.IGNORECASE), "CUTLASS usage"),),
    )

    inference: list[str] = []
    if evidence:
        inference.append("Packed sources contain direct cuBLAS/cuBLASLt symbols/usages.")
    elif cutlass_evidence:
        inference.append("No direct cuBLAS symbols were found; packed sources instead show CUTLASS-based implementation clues.")
    else:
        inference.append("No direct cuBLAS symbols/usages were found in the packed submission.")

    return {
        "status": "direct_evidence_found" if evidence else "no_direct_evidence",
        "evidence": evidence,
        "inference": inference,
    }



def _build_vendored_report(solution_payload: dict[str, Any]) -> dict[str, Any]:
    evidence = _collect_source_matches(solution_payload, _VENDOR_PATTERNS)
    author = str(solution_payload.get("author", "")).strip()
    flashinfer_headers = [item for item in evidence if item["reason"] == "FlashInfer copyright header"]

    inference: list[str] = []
    if flashinfer_headers and author and author.lower() != "flashinfer":
        inference.append(
            f"Packed sources carry FlashInfer attribution while the submission author is {author!r}, which is a concrete vendored/upstream-origin clue."
        )
    elif evidence:
        inference.append("Packed sources contain attribution/license/API-layer clues that suggest vendored or upstream-derived material.")
    else:
        inference.append("No explicit vendored/upstream-origin clues were found in the packed submission.")

    return {
        "status": "clues_found" if evidence else "no_clues_found",
        "evidence": evidence,
        "inference": inference,
    }



def _build_sm100a_report(solution_payload: dict[str, Any]) -> dict[str, Any]:
    exact_evidence = _collect_source_matches(solution_payload, _SM100A_PROOF_PATTERNS)
    weak_evidence = _collect_source_matches(solution_payload, _SM100A_WEAK_PATTERNS)
    evidence = [*exact_evidence, *[item for item in weak_evidence if item not in exact_evidence]]

    inference: list[str] = []
    if exact_evidence:
        inference.append("Packed sources contain explicit sm_100a-style proof tokens.")
        status = "explicit_proof_found"
    elif weak_evidence:
        inference.append(
            "Packed sources reference Blackwell/SM100-specific machinery, but no explicit `sm_100a` token was found, so sm_100a proof remains an unresolved gap."
        )
        status = "unresolved_gap"
    else:
        inference.append("No explicit sm_100a proof was found in the packed submission; compliance remains an unresolved gap.")
        status = "unresolved_gap"

    return {
        "status": status,
        "evidence": evidence,
        "inference": inference,
    }



def audit_submission_compliance(solution_json_path: Path | None = None) -> dict[str, Any]:
    solution_payload, metadata = _load_or_pack_solution(solution_json_path)
    spec = solution_payload.get("spec", {}) if isinstance(solution_payload.get("spec"), dict) else {}
    report = {
        "metadata": {
            **metadata,
            "solution_name": solution_payload.get("name"),
            "solution_author": solution_payload.get("author"),
            "entry_point": spec.get("entry_point"),
            "language": spec.get("language"),
            "dependencies": spec.get("dependencies", []),
            "source_count": len(solution_payload.get("sources", [])),
            "source_paths": [source.get("path") for source in solution_payload.get("sources", [])],
        },
        "runtime_flashinfer_api_usage": _build_runtime_flashinfer_report(solution_payload),
        "cublas_symbols_usages": _build_cublas_report(solution_payload),
        "vendored_upstream_origin_clues": _build_vendored_report(solution_payload),
        "sm_100a_proof": _build_sm100a_report(solution_payload),
    }
    report["summary"] = {
        key: report[key]["status"]
        for key in (
            "runtime_flashinfer_api_usage",
            "cublas_symbols_usages",
            "vendored_upstream_origin_clues",
            "sm_100a_proof",
        )
    }
    return report



def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the packed submission for compliance clues.")
    parser.add_argument(
        "--solution-json",
        type=Path,
        default=None,
        help="Read an existing packed solution.json instead of repacking the current submission.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report to a file instead of stdout.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON instead of indented JSON.",
    )
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = audit_submission_compliance(args.solution_json)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    json_text = json.dumps(report, indent=None if args.compact else 2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(json_text + "\n", encoding="utf-8")
    else:
        print(json_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
