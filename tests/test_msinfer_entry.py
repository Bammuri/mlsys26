import os
import sys
import unittest
from unittest import mock

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from solution.python.gdn_blackwell.gdn import _get_problem_size as _legacy_get_problem_size
from solution.python import msinfer_entry


class ProblemProfileTests(unittest.TestCase):
    def test_problem_profile_matches_legacy_varlen_semantics(self):
        cu_seqlens = torch.tensor([0, 16, 80, 96], dtype=torch.int32)
        q_shape = (1, 96, 8, 128)
        v_shape = (1, 96, 16, 128)

        problem_profile = msinfer_entry._get_varlen_problem_profile(cu_seqlens)
        problem_size = msinfer_entry._problem_size_from_profile(q_shape, v_shape, problem_profile)
        legacy_problem_size = _legacy_get_problem_size(q_shape, v_shape, tuple(cu_seqlens.tolist()))

        self.assertEqual(problem_profile, (3, 64, 96))
        self.assertEqual(problem_size, legacy_problem_size)

    def test_problem_profile_matches_dense_semantics(self):
        q_shape = (2, 128, 8, 128)
        v_shape = (2, 128, 16, 128)

        problem_size = msinfer_entry._problem_size_from_profile(q_shape, v_shape, None)

        self.assertEqual(problem_size, (2, 128, 128, 8, 16, 128))

    def test_problem_profile_rejects_too_short_cu_seqlens(self):
        with self.assertRaisesRegex(ValueError, "at least two entries"):
            msinfer_entry._get_varlen_problem_profile(torch.tensor([0], dtype=torch.int32))

    def test_problem_size_cache_uses_profile_summary_not_tensor_identity(self):
        q = torch.empty((1, 96, 8, 128), dtype=torch.float16)
        v = torch.empty((1, 96, 16, 128), dtype=torch.float16)
        cu_seqlens_a = torch.tensor([0, 16, 80, 96], dtype=torch.int32)
        cu_seqlens_b = torch.tensor([0, 16, 32, 96], dtype=torch.int32)

        msinfer_entry._PROBLEM_SIZE_CACHE_KEY = None
        msinfer_entry._PROBLEM_SIZE_CACHE_VALUE = None

        first_problem_size = msinfer_entry._get_problem_size_cached(q, v, cu_seqlens_a)
        second_problem_size = msinfer_entry._get_problem_size_cached(q, v, cu_seqlens_b)

        self.assertEqual(first_problem_size, second_problem_size)
        self.assertIs(first_problem_size, second_problem_size)


class PersistentPolicyTests(unittest.TestCase):
    def test_persistent_policy_variants(self):
        small_problem = (1, 128, 128, 8, 16, 128)
        large_problem = (4, 1024, 4096, 8, 32, 128)

        self.assertFalse(msinfer_entry._select_persistent_mode(small_problem, policy="never"))
        self.assertTrue(msinfer_entry._select_persistent_mode(large_problem, policy="always"))
        self.assertTrue(
            msinfer_entry._select_persistent_mode(
                small_problem,
                policy="auto",
                auto_max_batch=1,
                auto_max_seq_len=128,
            )
        )
        self.assertFalse(
            msinfer_entry._select_persistent_mode(
                large_problem,
                policy="auto",
                auto_max_batch=1,
                auto_max_seq_len=128,
            )
        )

    def test_invalid_persistent_policy_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            msinfer_entry._normalize_persistent_policy("sometimes")

    def test_resolve_persistent_mode_reads_env(self):
        problem_size = (1, 128, 128, 8, 16, 128)
        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._PERSISTENT_POLICY_ENV: "auto",
                msinfer_entry._PERSISTENT_AUTO_MAX_BATCH_ENV: "1",
                msinfer_entry._PERSISTENT_AUTO_MAX_SEQ_LEN_ENV: "64",
            },
            clear=False,
        ):
            self.assertFalse(msinfer_entry._resolve_persistent_mode(problem_size))

        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._PERSISTENT_POLICY_ENV: "always",
                msinfer_entry._PERSISTENT_AUTO_MAX_BATCH_ENV: "1",
                msinfer_entry._PERSISTENT_AUTO_MAX_SEQ_LEN_ENV: "64",
            },
            clear=False,
        ):
            self.assertTrue(msinfer_entry._resolve_persistent_mode(problem_size))

    def test_invalid_auto_threshold_raises(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            msinfer_entry._normalize_positive_int("0", default=16, env_name="TEST_ENV")


class TracingFlagTests(unittest.TestCase):
    def test_tracing_env_defaults_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(msinfer_entry._profile_enabled())

    def test_tracing_env_truthy_values_enable_ranges(self):
        with mock.patch.dict(os.environ, {msinfer_entry._TRACE_PHASES_ENV: "1"}, clear=False):
            self.assertTrue(msinfer_entry._profile_enabled())


if __name__ == "__main__":
    unittest.main()
