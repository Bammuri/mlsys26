import unittest

from scripts.run_modal_chunk_parallel_prototype import summarize_timings


class RunModalChunkParallelPrototypeTest(unittest.TestCase):
    def test_summarize_timings(self) -> None:
        summary = summarize_timings([3.0, 1.0, 2.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 4.0)
        self.assertEqual(summary["mean_ms"], 2.5)


if __name__ == "__main__":
    unittest.main()
