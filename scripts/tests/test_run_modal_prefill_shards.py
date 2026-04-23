import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_modal_prefill_shards import build_shards, main


class RunModalPrefillShardsTest(unittest.TestCase):
    def test_build_shards(self) -> None:
        uuids = [f"u{i}" for i in range(5)]
        self.assertEqual(build_shards(uuids, 2), [["u0", "u1"], ["u2", "u3"], ["u4"]])

    def test_main_runs_selected_shards_and_merges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "out"
            merge_output = Path(tmpdir) / "merged.json"
            with mock.patch("scripts.run_modal_prefill_shards.resolve_workload_uuids", return_value=[f"u{i}" for i in range(5)]):
                with mock.patch("scripts.run_modal_prefill_shards.run_shard") as run_shard:
                    with mock.patch("scripts.run_modal_prefill_shards.merge_shards") as merge_shards:
                        exit_code = main([
                            "--output-dir", str(outdir),
                            "--merge-output", str(merge_output),
                            "--shard-size", "2",
                            "--start-shard", "1",
                            "--limit-shards", "1",
                        ])
        self.assertEqual(exit_code, 0)
        run_shard.assert_called_once()
        merge_shards.assert_called_once()


if __name__ == "__main__":
    unittest.main()
