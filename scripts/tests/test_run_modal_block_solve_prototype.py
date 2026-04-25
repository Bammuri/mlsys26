import unittest

from scripts.run_modal_block_solve_prototype import resolve_block_sizes, summarize


class RunModalBlockSolvePrototypeTest(unittest.TestCase):
    def test_summarize(self) -> None:
        summary = summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary['count'], 4)
        self.assertEqual(summary['mean_ms'], 2.5)
        self.assertEqual(summary['median_ms'], 2.5)
        self.assertEqual(summary['min_ms'], 1.0)
        self.assertEqual(summary['max_ms'], 4.0)

    def test_resolve_block_sizes_uses_policy_when_requested(self) -> None:
        self.assertEqual(
            resolve_block_sizes(
                block_sizes_csv="",
                use_default_policy=True,
                enable_shape_tuning=False,
                num_heads=4,
                rhs_dim=128,
            ),
            [192],
        )
        self.assertEqual(
            resolve_block_sizes(
                block_sizes_csv="",
                use_default_policy=True,
                enable_shape_tuning=True,
                num_heads=8,
                rhs_dim=64,
            ),
            [160],
        )

    def test_resolve_block_sizes_requires_explicit_blocks_without_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "block-sizes"):
            resolve_block_sizes(
                block_sizes_csv="",
                use_default_policy=False,
                enable_shape_tuning=False,
                num_heads=4,
                rhs_dim=128,
            )


if __name__ == '__main__':
    unittest.main()
