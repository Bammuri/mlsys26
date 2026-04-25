import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from solution.python.gdn_block_policy import choose_block_tile, choose_default_or_tuned_block_tile


class GdnBlockPolicyTest(unittest.TestCase):
    def test_shape_aware_policy(self):
        self.assertEqual(choose_block_tile(num_heads=4, rhs_dim=128), 192)
        self.assertEqual(choose_block_tile(num_heads=8, rhs_dim=64), 160)
        self.assertEqual(choose_block_tile(num_heads=2, rhs_dim=512), 224)

    def test_default_policy_stays_192_when_tuning_disabled(self):
        self.assertEqual(choose_default_or_tuned_block_tile(num_heads=8, rhs_dim=64, enable_shape_tuning=False), 192)
        self.assertEqual(choose_default_or_tuned_block_tile(num_heads=2, rhs_dim=512, enable_shape_tuning=False), 192)
        self.assertEqual(choose_default_or_tuned_block_tile(num_heads=4, rhs_dim=128, enable_shape_tuning=False), 192)

    def test_tuned_policy_matches_shape_aware_policy(self):
        self.assertEqual(choose_default_or_tuned_block_tile(num_heads=8, rhs_dim=64, enable_shape_tuning=True), 160)
        self.assertEqual(choose_default_or_tuned_block_tile(num_heads=2, rhs_dim=512, enable_shape_tuning=True), 224)
        self.assertEqual(choose_default_or_tuned_block_tile(num_heads=4, rhs_dim=128, enable_shape_tuning=True), 192)


if __name__ == '__main__':
    unittest.main()
