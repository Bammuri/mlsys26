import unittest

from scripts.run_modal_staged_block_solve_prototype import summarize


class RunModalStagedBlockSolvePrototypeTest(unittest.TestCase):
    def test_summarize(self) -> None:
        summary = summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary['count'], 4)
        self.assertEqual(summary['mean_ms'], 2.5)
        self.assertEqual(summary['median_ms'], 2.5)
        self.assertEqual(summary['min_ms'], 1.0)
        self.assertEqual(summary['max_ms'], 4.0)


if __name__ == '__main__':
    unittest.main()
