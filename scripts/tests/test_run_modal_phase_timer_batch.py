import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_modal_phase_timer_batch import _chunked, _load_workload_uuids, main


class RunModalPhaseTimerBatchTest(unittest.TestCase):
    def test_chunked(self) -> None:
        self.assertEqual(_chunked(["a", "b", "c"], 2), [["a", "b"], ["c"]])

    def test_load_workload_uuids_dedupes_cli_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "uuids.txt"
            path.write_text("u2\nu3\nu1\n", encoding="utf-8")
            args = mock.Mock(workload_uuids="u1,u2", workload_uuids_file=path)
            self.assertEqual(_load_workload_uuids(args), ["u1", "u2", "u3"])

    def test_main_writes_manifest(self) -> None:
        class _Call:
            def __init__(self, value) -> None:
                self._value = value

            def get(self):
                return self._value

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "out"
            manifest = Path(tmpdir) / "manifest.json"
            fake_result = {
                "solution": {
                    "name": "demo",
                    "entry_point": "main.py::run",
                    "destination_passing_style": True,
                }
            }
            with mock.patch("scripts.run_modal_phase_timer_batch._load_solution_payload", return_value=({}, {"solution_name": "demo"})):
                with mock.patch("scripts.run_modal_phase_timer_batch.app.run") as app_run:
                    app_run.return_value.__enter__.return_value = None
                    app_run.return_value.__exit__.return_value = False
                    with mock.patch("scripts.run_modal_phase_timer_batch.time_prefill_workload.spawn", side_effect=lambda *_args, **_kwargs: _Call(fake_result)):
                        exit_code = main([
                            "--workload-uuids", "u1,u2",
                            "--output-dir", str(outdir),
                            "--manifest-json", str(manifest),
                        ])
            self.assertEqual(exit_code, 0)
            payload = json.loads(manifest.read_text())
            self.assertEqual(payload["metadata"]["workload_count"], 2)
            self.assertEqual(len(payload["results"]), 2)


if __name__ == "__main__":
    unittest.main()
