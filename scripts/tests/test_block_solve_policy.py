from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.block_solve_policy import choose_block_tile, choose_default_or_tuned_block_tile


class BlockSolvePolicyTest(unittest.TestCase):
    def test_policy_shape_rules(self) -> None:
        self.assertEqual(choose_block_tile(num_heads=4, rhs_dim=128), 192)
        self.assertEqual(choose_block_tile(num_heads=4, rhs_dim=64), 160)
        self.assertEqual(choose_block_tile(num_heads=4, rhs_dim=512), 224)
        self.assertEqual(choose_block_tile(num_heads=2, rhs_dim=128), 224)
        self.assertEqual(choose_block_tile(num_heads=8, rhs_dim=128), 160)

    def test_policy_tracks_measured_best_or_near_best_at_4096(self) -> None:
        files = [
            '.omx/results/task4-block-solve-L4096-rhs32-20260425.json',
            '.omx/results/task4-block-solve-L4096-rhs64-20260425.json',
            '.omx/results/task4-block-solve-L4096-band-20260425.json',
            '.omx/results/task4-block-solve-L4096-rhs256-20260425.json',
            '.omx/results/task4-block-solve-L4096-rhs384-20260425.json',
            '.omx/results/task4-block-solve-L4096-rhs512-20260425.json',
            '.omx/results/task4-block-solve-L4096-H2-20260425.json',
            '.omx/results/task4-block-solve-L4096-H8-20260425.json',
        ]
        for path in files:
            payload = json.loads(Path(path).read_text())['result']
            chosen = choose_block_tile(num_heads=payload['config']['num_heads'], rhs_dim=payload['config']['rhs_dim'])
            by_block = {row['block_size']: row['timing']['speedup_vs_torch_solve'] for row in payload['rows']}
            best = max(by_block.values())
            # Policy should stay within a small delta of the measured optimum.
            self.assertGreaterEqual(by_block[chosen] + 0.02, best, msg=f'{path}: chosen={chosen}, by_block={by_block}')

    def test_default_policy_stays_at_192_unless_shape_tuning_is_enabled(self) -> None:
        self.assertEqual(
            choose_default_or_tuned_block_tile(num_heads=4, rhs_dim=64, enable_shape_tuning=False),
            192,
        )
        self.assertEqual(
            choose_default_or_tuned_block_tile(num_heads=8, rhs_dim=64, enable_shape_tuning=False),
            192,
        )
        self.assertEqual(
            choose_default_or_tuned_block_tile(num_heads=2, rhs_dim=512, enable_shape_tuning=False),
            192,
        )

    def test_shape_tuned_policy_uses_neighbor_fallbacks_when_enabled(self) -> None:
        self.assertEqual(
            choose_default_or_tuned_block_tile(num_heads=8, rhs_dim=64, enable_shape_tuning=True),
            160,
        )
        self.assertEqual(
            choose_default_or_tuned_block_tile(num_heads=2, rhs_dim=512, enable_shape_tuning=True),
            224,
        )
        self.assertEqual(
            choose_default_or_tuned_block_tile(num_heads=4, rhs_dim=256, enable_shape_tuning=True),
            192,
        )


if __name__ == '__main__':
    unittest.main()
