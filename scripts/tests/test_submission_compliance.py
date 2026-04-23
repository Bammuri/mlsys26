import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import audit_submission_compliance as audit


class SubmissionComplianceAuditTest(unittest.TestCase):
    def _write_solution_json(self, directory: Path, payload: dict) -> Path:
        path = directory / "solution.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_audit_detects_direct_runtime_cublas_vendor_and_sm100a_evidence(self) -> None:
        payload = {
            "name": "demo",
            "definition": "demo_def",
            "author": "Submitter",
            "description": "",
            "spec": {
                "language": "python",
                "entry_point": "main.py::run",
                "dependencies": ["flashinfer-python", "cuda-python>=12.8"],
            },
            "sources": [
                {
                    "path": "main.py",
                    "content": "from flashinfer import chunk_prefill\nresult = flashinfer.prefill(q, k, v)\n",
                },
                {
                    "path": "kernel.cu",
                    "content": "// Copyright (c) 2026 by FlashInfer team.\n"
                    "// Licensed under the Apache License, Version 2.0 (the \"License\");\n"
                    "void run() { cublasLtMatmul(handle, a, b, c); }\n"
                    "// target=sm_100a\n",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            solution_json = self._write_solution_json(Path(tmpdir), payload)
            report = audit.audit_submission_compliance(solution_json)

        self.assertEqual(report["runtime_flashinfer_api_usage"]["status"], "direct_evidence_found")
        self.assertEqual(report["cublas_symbols_usages"]["status"], "direct_evidence_found")
        self.assertEqual(report["vendored_upstream_origin_clues"]["status"], "clues_found")
        self.assertEqual(report["sm_100a_proof"]["status"], "explicit_proof_found")
        self.assertEqual(report["metadata"]["inspection_scope"], "packed_sources_only")

        runtime_paths = {item["path"] for item in report["runtime_flashinfer_api_usage"]["evidence"]}
        cublas_paths = {item["path"] for item in report["cublas_symbols_usages"]["evidence"]}
        sm_paths = {item["path"] for item in report["sm_100a_proof"]["evidence"]}
        self.assertIn("main.py", runtime_paths)
        self.assertIn("kernel.cu", cublas_paths)
        self.assertIn("kernel.cu", sm_paths)

    def test_flashinfer_dependency_is_inference_only_without_direct_calls(self) -> None:
        payload = {
            "name": "demo",
            "definition": "demo_def",
            "author": "Submitter",
            "description": "",
            "spec": {
                "language": "python",
                "entry_point": "main.py::run",
                "dependencies": ["flashinfer-python"],
            },
            "sources": [{"path": "main.py", "content": "print('hello')\n"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            solution_json = self._write_solution_json(Path(tmpdir), payload)
            report = audit.audit_submission_compliance(solution_json)

        self.assertEqual(report["runtime_flashinfer_api_usage"]["status"], "no_direct_evidence")
        evidence_paths = {item["path"] for item in report["runtime_flashinfer_api_usage"]["evidence"]}
        self.assertIn(audit._SPEC_PATH, evidence_paths)

    def test_default_mode_packs_current_solution_and_inspects_only_packed_sources(self) -> None:
        packed_payload = {
            "name": "packed-only",
            "definition": "demo_def",
            "author": "PackedAuthor",
            "description": "",
            "spec": {"language": "python", "entry_point": "packed.py::run", "dependencies": []},
            "sources": [
                {
                    "path": "packed.py",
                    "content": "# FlashInfer API Layer\ndef chunk_gated_delta_rule():\n    pass\n",
                }
            ],
        }

        with mock.patch.object(audit, "_pack_current_solution_json", return_value=packed_payload):
            report = audit.audit_submission_compliance()

        self.assertTrue(report["metadata"]["packed_from_repo"])
        self.assertEqual(report["metadata"]["inspection_scope"], "packed_sources_only")
        evidence_paths = {item["path"] for item in report["runtime_flashinfer_api_usage"]["evidence"]}
        self.assertEqual(evidence_paths, {"packed.py"})
        self.assertEqual(report["runtime_flashinfer_api_usage"]["status"], "no_direct_evidence")
        self.assertIn("vendored FlashInfer-style API surface", report["runtime_flashinfer_api_usage"]["inference"][0])

    def test_current_solution_json_smoke_report_is_structured(self) -> None:
        report = audit.audit_submission_compliance(Path("solution.json"))

        self.assertEqual(report["metadata"]["inspection_scope"], "packed_sources_only")
        self.assertFalse(report["metadata"]["packed_from_repo"])
        self.assertGreater(report["metadata"]["source_count"], 0)
        self.assertIn("runtime_flashinfer_api_usage", report["summary"])
        self.assertIn(report["sm_100a_proof"]["status"], {"explicit_proof_found", "unresolved_gap"})
        self.assertIsInstance(report["vendored_upstream_origin_clues"]["evidence"], list)

    def test_cli_writes_json_report(self) -> None:
        payload = {
            "name": "demo",
            "definition": "demo_def",
            "author": "Submitter",
            "description": "",
            "spec": {"language": "python", "entry_point": "main.py::run", "dependencies": []},
            "sources": [{"path": "main.py", "content": "print('hello')\n"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            solution_json = self._write_solution_json(tmpdir_path, payload)
            output_path = tmpdir_path / "report.json"
            exit_code = audit.main(["--solution-json", str(solution_json), "--output", str(output_path)])
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(written["metadata"]["solution_json"], str(solution_json))
        self.assertEqual(written["summary"]["cublas_symbols_usages"], "no_direct_evidence")


if __name__ == "__main__":
    unittest.main()
