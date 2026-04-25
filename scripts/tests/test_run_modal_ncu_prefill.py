import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_modal_ncu_prefill import (
    ADAPTIVE_SELECTOR_KEYS_ENV,
    DEFAULT_BASELINE_KERNEL_PATTERN,
    DEFAULT_KERNEL_PATTERN,
    PACKED_SUBMIT_SOLUTION_SOURCE,
    PERSISTENT_POLICY_ENV,
    TRACE_SET_BASELINE_SOLUTION_SOURCE,
    _build_runtime_env,
    _capture_single_workload,
    _chunked,
    _collect_workload_results,
    _load_solution_payload,
    _load_representative_uuids,
    _manifest_row,
    _resolve_kernel_pattern,
    _resolve_workload_uuids,
)


class ModalNcuPrefillHelpersTest(unittest.TestCase):
    def test_chunked(self) -> None:
        self.assertEqual(_chunked(["a", "b", "c", "d", "e"], 2), [["a", "b"], ["c", "d"], ["e"]])

    def test_manifest_row(self) -> None:
        row = _manifest_row(
            {
                "uuid": "abc",
                "status": "ok",
                "notes": ["baseline"],
                "axes": {"total_seq_len": 32},
                "kernel_pattern": "regex:kernel.*",
            },
            "abc.txt",
        )
        self.assertEqual(row["label"], "baseline")
        self.assertEqual(row["report_path"], "abc.txt")
        self.assertEqual(row["status"], "ok")

    def test_manifest_row_accepts_candidate_label(self) -> None:
        row = _manifest_row(
            {
                "uuid": "abc",
                "status": "ok",
                "notes": [],
                "axes": {},
                "kernel_pattern": "regex:kernel.*",
            },
            "abc.txt",
            label="candidate",
        )
        self.assertEqual(row["label"], "candidate")

    def test_resolve_workload_uuids_filters_and_limits(self) -> None:
        fake_trace = mock.Mock()
        fake_trace.workloads = {
            "demo": [
                mock.Mock(workload=mock.Mock(uuid="u1")),
                mock.Mock(workload=mock.Mock(uuid="u2")),
                mock.Mock(workload=mock.Mock(uuid="u3")),
            ]
        }
        with mock.patch("scripts.run_modal_ncu_prefill.TraceSet.from_path", return_value=fake_trace):
            with tempfile.TemporaryDirectory() as tmpdir:
                uuids = _resolve_workload_uuids("demo", Path(tmpdir), ["u2", "u3"], 1)
        self.assertEqual(uuids, ["u2"])

    def test_load_representative_uuids_reads_first_uuid_per_group(self) -> None:
        payload = {
            "groups": [
                {"selector_key": "a", "uuids": ["u1", "u2"]},
                {"selector_key": "b", "uuids": ["u3"]},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "groups.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_representative_uuids(path), ["u1", "u3"])

    def test_capture_single_workload_timeout_is_localized(self) -> None:
        definition = mock.Mock()
        definition.model_dump_json.return_value = "{}"
        workload = mock.Mock()
        workload.uuid = "u-timeout"
        workload.axes = {"total_seq_len": 32}
        workload.model_dump_json.return_value = "{}"
        timeout_exc = subprocess.TimeoutExpired(cmd=["ncu"], timeout=5, output="partial", stderr="stderr")

        with mock.patch("scripts.run_modal_ncu_prefill.subprocess.run", side_effect=timeout_exc):
            result = _capture_single_workload(
                solution_payload={
                    "name": "demo",
                    "definition": "demo",
                    "author": "x",
                    "spec": {
                        "language": "python",
                        "target_hardware": ["cuda"],
                        "entry_point": "main.py::run",
                        "dependencies": [],
                        "destination_passing_style": True,
                    },
                    "sources": [{"path": "main.py", "content": "def run():\n    pass\n"}],
                },
                definition=definition,
                workload=workload,
                kernel_pattern="regex:kernel.*",
                section_args=[],
                launch_skip=1,
                launch_count=1,
                timeout_seconds=5,
                runtime_env={},
            )

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["uuid"], "u-timeout")

    def test_capture_single_workload_applies_runtime_env_to_ncu_subprocess(self) -> None:
        definition = mock.Mock()
        definition.model_dump_json.return_value = "{}"
        workload = mock.Mock()
        workload.uuid = "u-env"
        workload.axes = {"total_seq_len": 32}
        workload.model_dump_json.return_value = "{}"

        def _fake_run(*args, **kwargs):
            self.assertEqual(__import__("os").environ.get(PERSISTENT_POLICY_ENV), "adaptive")
            return mock.Mock(returncode=0, stdout="Occupancy\n", stderr="")

        with mock.patch("scripts.run_modal_ncu_prefill.subprocess.run", side_effect=_fake_run):
            result = _capture_single_workload(
                solution_payload={
                    "name": "demo",
                    "definition": "demo",
                    "author": "x",
                    "spec": {
                        "language": "python",
                        "target_hardware": ["cuda"],
                        "entry_point": "main.py::run",
                        "dependencies": [],
                        "destination_passing_style": True,
                    },
                    "sources": [{"path": "main.py", "content": "def run():\n    pass\n"}],
                },
                definition=definition,
                workload=workload,
                kernel_pattern="regex:kernel.*",
                section_args=[],
                launch_skip=1,
                launch_count=1,
                timeout_seconds=5,
                runtime_env={PERSISTENT_POLICY_ENV: "adaptive"},
            )

        self.assertEqual(result["status"], "ok")

    def test_build_runtime_env_records_persistent_policy(self) -> None:
        args = mock.Mock(
            persistent_policy="adaptive",
            persistent_auto_max_batch=0,
            persistent_auto_max_seq_len=0,
            adaptive_selector_keys="",
        )

        self.assertEqual(_build_runtime_env(args), {PERSISTENT_POLICY_ENV: "adaptive"})

    def test_build_runtime_env_records_adaptive_selector_keys(self) -> None:
        args = mock.Mock(
            persistent_policy="adaptive",
            persistent_auto_max_batch=0,
            persistent_auto_max_seq_len=0,
            adaptive_selector_keys="key-a,key-b",
        )

        self.assertEqual(
            _build_runtime_env(args),
            {
                PERSISTENT_POLICY_ENV: "adaptive",
                ADAPTIVE_SELECTOR_KEYS_ENV: "key-a,key-b",
            },
        )

    def test_resolve_kernel_pattern_uses_source_aware_defaults(self) -> None:
        self.assertEqual(
            _resolve_kernel_pattern(PACKED_SUBMIT_SOLUTION_SOURCE, ""),
            DEFAULT_KERNEL_PATTERN,
        )
        self.assertEqual(
            _resolve_kernel_pattern(TRACE_SET_BASELINE_SOLUTION_SOURCE, ""),
            DEFAULT_BASELINE_KERNEL_PATTERN,
        )
        self.assertEqual(
            _resolve_kernel_pattern(TRACE_SET_BASELINE_SOLUTION_SOURCE, "regex:custom.*"),
            "regex:custom.*",
        )

    def test_load_solution_payload_uses_trace_set_baseline_solution(self) -> None:
        fake_solution = mock.Mock()
        fake_solution.name = "flashinfer_wrapper_123ca6"
        fake_solution.author = "flashinfer"
        fake_solution.spec = mock.Mock(entry_point="main.py::run", destination_passing_style=False)
        fake_solution.model_dump.return_value = {"name": "flashinfer_wrapper_123ca6"}

        fake_trace = mock.Mock()
        fake_trace.workloads = {}
        fake_trace.solutions = {"demo": [fake_solution]}

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_set_path = Path(tmpdir)
            with mock.patch("scripts.run_modal_ncu_prefill.TraceSet.from_path", return_value=fake_trace):
                payload, provenance = _load_solution_payload(
                    definition="demo",
                    solution_source=TRACE_SET_BASELINE_SOLUTION_SOURCE,
                    trace_set_path=trace_set_path,
                    baseline_solution_index=0,
                )

        self.assertEqual(payload, {"name": "flashinfer_wrapper_123ca6"})
        self.assertEqual(provenance["solution_source"], TRACE_SET_BASELINE_SOLUTION_SOURCE)
        self.assertEqual(provenance["solution_name"], "flashinfer_wrapper_123ca6")
        self.assertEqual(provenance["solution_author"], "flashinfer")
        self.assertEqual(provenance["entry_point"], "main.py::run")
        self.assertFalse(provenance["destination_passing_style"])
        self.assertEqual(provenance["baseline_solution_index"], 0)

    def test_collect_workload_results_localizes_call_failures(self) -> None:
        class _Call:
            def __init__(self, value=None, exc=None) -> None:
                self._value = value
                self._exc = exc

            def get(self):
                if self._exc is not None:
                    raise self._exc
                return self._value

        calls = {
            "u1": _Call({"uuid": "u1", "status": "ok", "notes": [], "report_text": "", "kernel_pattern": "k"}),
            "u2": _Call(exc=RuntimeError("boom")),
            "u3": _Call({"uuid": "u3", "status": "ok", "notes": [], "report_text": "", "kernel_pattern": "k"}),
        }
        streamed = []
        results = _collect_workload_results(
            ["u1", "u2", "u3"],
            batch_size=2,
            spawn_fn=lambda uuid: calls[uuid],
            result_callback=streamed.append,
        )
        by_uuid = {item["uuid"]: item for item in results}
        self.assertEqual(by_uuid["u1"]["status"], "ok")
        self.assertEqual(by_uuid["u3"]["status"], "ok")
        self.assertEqual(by_uuid["u2"]["status"], "exception")
        self.assertEqual({item["uuid"] for item in streamed}, {"u1", "u2", "u3"})


if __name__ == "__main__":
    unittest.main()
