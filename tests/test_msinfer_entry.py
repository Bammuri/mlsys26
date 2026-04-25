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
        adaptive_target_problem = (32, 2300, 8192, 8, 16, 128)

        self.assertFalse(msinfer_entry._select_persistent_mode(small_problem, policy="never"))
        self.assertTrue(msinfer_entry._select_persistent_mode(large_problem, policy="always"))
        self.assertTrue(msinfer_entry._select_persistent_mode(adaptive_target_problem, policy="adaptive"))
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

    def test_adaptive_policy_matches_ncu_stable_shape_groups(self):
        promoted_r1_tail = (32, 2300, 8192, 8, 16, 128)
        non_promoted_r2_tail = (56, 1296, 8192, 8, 16, 128)
        non_promoted_r3_tail = (20, 3371, 8192, 8, 16, 128)
        promoted_medium_large_large = (3, 1515, 1800, 8, 16, 128)
        non_promoted_medium_large_large = (3, 1128, 1592, 8, 16, 128)
        stable_mid = (3, 512, 512, 8, 16, 128)
        scheduler_regressed_small_seq = (3, 32, 512, 8, 16, 128)
        fast_floor = (1, 30, 30, 8, 16, 128)
        fixed_shape = (2, 512, 512, 8, 16, 128)

        self.assertTrue(msinfer_entry._select_persistent_mode(promoted_r1_tail, policy="adaptive"))
        self.assertFalse(msinfer_entry._select_persistent_mode(non_promoted_r2_tail, policy="adaptive"))
        self.assertFalse(msinfer_entry._select_persistent_mode(non_promoted_r3_tail, policy="adaptive"))
        self.assertFalse(msinfer_entry._select_persistent_mode(promoted_medium_large_large, policy="adaptive"))
        self.assertFalse(msinfer_entry._select_persistent_mode(non_promoted_medium_large_large, policy="adaptive"))
        self.assertFalse(msinfer_entry._select_persistent_mode(stable_mid, policy="adaptive"))
        self.assertFalse(msinfer_entry._select_persistent_mode(scheduler_regressed_small_seq, policy="adaptive"))
        self.assertFalse(msinfer_entry._select_persistent_mode(fast_floor, policy="adaptive"))
        self.assertFalse(msinfer_entry._select_persistent_mode(fixed_shape, policy="adaptive", varlen=False))

        self.assertEqual(
            msinfer_entry._adaptive_selector_key(promoted_medium_large_large, varlen=True),
            msinfer_entry._MEDIUM_LARGE_LARGE_SELECTOR_KEY,
        )

    def test_large_large_large_adaptive_subkeys_split_by_runtime_visible_shape(self):
        r1 = (32, 2300, 8192, 8, 16, 128)
        r2 = (56, 1296, 8192, 8, 16, 128)
        r3 = (20, 3371, 8192, 8, 16, 128)
        r4 = (13, 2877, 3999, 8, 16, 128)

        self.assertEqual(
            msinfer_entry._adaptive_selector_keys(r1, varlen=True),
            (
                msinfer_entry._LARGE_LARGE_LARGE_SELECTOR_KEY,
                msinfer_entry._LARGE_LARGE_LARGE_R1_SELECTOR_KEY,
            ),
        )
        self.assertEqual(
            msinfer_entry._adaptive_selector_keys(r2, varlen=True),
            (
                msinfer_entry._LARGE_LARGE_LARGE_SELECTOR_KEY,
                msinfer_entry._LARGE_LARGE_LARGE_R2_SELECTOR_KEY,
            ),
        )
        self.assertEqual(
            msinfer_entry._adaptive_selector_keys(r3, varlen=True),
            (
                msinfer_entry._LARGE_LARGE_LARGE_SELECTOR_KEY,
                msinfer_entry._LARGE_LARGE_LARGE_R3_SELECTOR_KEY,
            ),
        )
        self.assertEqual(
            msinfer_entry._adaptive_selector_keys(r4, varlen=True),
            (
                msinfer_entry._LARGE_LARGE_LARGE_SELECTOR_KEY,
                msinfer_entry._LARGE_LARGE_LARGE_R4_SELECTOR_KEY,
            ),
        )

    def test_medium_large_large_adaptive_subkey_identifies_promising_batch3_window(self):
        promoted = (3, 1515, 1800, 8, 16, 128)
        not_promoted_small = (3, 1128, 1592, 8, 16, 128)
        not_promoted_large = (3, 2120, 2284, 8, 16, 128)
        not_promoted_batch2 = (2, 2023, 2040, 8, 16, 128)

        self.assertEqual(
            msinfer_entry._adaptive_selector_keys(promoted, varlen=True),
            (
                msinfer_entry._MEDIUM_LARGE_LARGE_SELECTOR_KEY,
                msinfer_entry._MEDIUM_LARGE_LARGE_M1_SELECTOR_KEY,
            ),
        )
        self.assertEqual(
            msinfer_entry._adaptive_selector_keys(not_promoted_small, varlen=True),
            (msinfer_entry._MEDIUM_LARGE_LARGE_SELECTOR_KEY,),
        )
        self.assertEqual(
            msinfer_entry._adaptive_selector_keys(not_promoted_large, varlen=True),
            (msinfer_entry._MEDIUM_LARGE_LARGE_SELECTOR_KEY,),
        )
        self.assertEqual(
            msinfer_entry._adaptive_selector_keys(not_promoted_batch2, varlen=True),
            (msinfer_entry._MEDIUM_LARGE_LARGE_SELECTOR_KEY,),
        )

    def test_adaptive_selector_key_env_can_target_fine_subkeys_without_uuid_rules(self):
        r1 = (32, 2300, 8192, 8, 16, 128)
        r2 = (56, 1296, 8192, 8, 16, 128)
        m1 = (3, 1515, 1800, 8, 16, 128)
        m_other = (3, 1128, 1592, 8, 16, 128)

        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._ADAPTIVE_SELECTOR_KEYS_ENV: msinfer_entry._LARGE_LARGE_LARGE_R1_SELECTOR_KEY,
            },
            clear=False,
        ):
            self.assertTrue(msinfer_entry._select_persistent_mode(r1, policy="adaptive"))
            self.assertFalse(msinfer_entry._select_persistent_mode(r2, policy="adaptive"))

        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._ADAPTIVE_SELECTOR_KEYS_ENV: ",".join(
                    [
                        msinfer_entry._LARGE_LARGE_LARGE_R1_SELECTOR_KEY,
                        msinfer_entry._LARGE_LARGE_LARGE_R2_SELECTOR_KEY,
                    ]
                ),
            },
            clear=False,
        ):
            self.assertTrue(msinfer_entry._select_persistent_mode(r1, policy="adaptive"))
            self.assertTrue(msinfer_entry._select_persistent_mode(r2, policy="adaptive"))

        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._ADAPTIVE_SELECTOR_KEYS_ENV: msinfer_entry._MEDIUM_LARGE_LARGE_M1_SELECTOR_KEY,
            },
            clear=False,
        ):
            self.assertTrue(msinfer_entry._select_persistent_mode(m1, policy="adaptive"))
            self.assertFalse(msinfer_entry._select_persistent_mode(m_other, policy="adaptive"))

    def test_adaptive_selector_key_env_is_cached_by_raw_env_value(self):
        msinfer_entry._ADAPTIVE_SELECTOR_KEYS_CACHE_KEY = None
        msinfer_entry._ADAPTIVE_SELECTOR_KEYS_CACHE_VALUE = None

        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._ADAPTIVE_SELECTOR_KEYS_ENV: msinfer_entry._LARGE_LARGE_LARGE_R2_SELECTOR_KEY,
            },
            clear=False,
        ):
            first_keys = msinfer_entry._get_adaptive_selector_key_set()
            second_keys = msinfer_entry._get_adaptive_selector_key_set()

        self.assertIs(first_keys, second_keys)

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

        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._PERSISTENT_POLICY_ENV: "adaptive",
            },
            clear=False,
        ):
            self.assertFalse(msinfer_entry._resolve_persistent_mode(problem_size))

    def test_invalid_auto_threshold_raises(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            msinfer_entry._normalize_positive_int("0", default=16, env_name="TEST_ENV")

    def test_empty_adaptive_selector_key_env_raises(self):
        with mock.patch.dict(
            os.environ,
            {msinfer_entry._ADAPTIVE_SELECTOR_KEYS_ENV: "   ,  "},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, msinfer_entry._ADAPTIVE_SELECTOR_KEYS_ENV):
                msinfer_entry._get_adaptive_selector_key_set()

    def test_resolve_block_tile_defaults_to_flat_192(self):
        problem_size = (32, 2300, 8192, 4, 4, 128)
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertEqual(msinfer_entry._resolve_block_tile(problem_size), 192)

    def test_resolve_block_tile_can_enable_shape_tuning(self):
        narrow_high_parallel = (32, 2300, 8192, 4, 8, 64)
        wide_low_head = (32, 2300, 8192, 4, 2, 512)
        with mock.patch.dict(
            os.environ,
            {msinfer_entry._BLOCK_SHAPE_TUNING_ENV: "1"},
            clear=False,
        ):
            self.assertEqual(msinfer_entry._resolve_block_tile(narrow_high_parallel), 160)
            self.assertEqual(msinfer_entry._resolve_block_tile(wide_low_head), 224)

    def test_outer_schedule_tile_is_separate_from_internal_kernel_chunk_size(self):
        problem_size = (32, 2300, 8192, 4, 4, 128)
        with mock.patch.dict(os.environ, {}, clear=False):
            schedule = msinfer_entry._resolve_kernel_schedule(problem_size)
            self.assertEqual(schedule.outer_schedule_tile, 128)
            self.assertEqual(schedule.internal_kernel_chunk_size, 128)
            self.assertEqual(schedule.internal_launch_segments, (128,))
            self.assertFalse(schedule.experimental_policy_enabled)

        with mock.patch.dict(
            os.environ,
            {msinfer_entry._EXPERIMENTAL_BLOCK_POLICY_ENV: "1"},
            clear=False,
        ):
            schedule = msinfer_entry._resolve_kernel_schedule(problem_size)
            self.assertEqual(schedule.outer_schedule_tile, 192)
            self.assertEqual(schedule.internal_kernel_chunk_size, 128)
            self.assertEqual(schedule.internal_launch_segments, (128, 64))
            self.assertTrue(schedule.experimental_policy_enabled)

    def test_kernel_chunk_size_stays_legal_when_shape_tuned_policy_selects_non_mma_tiles(self):
        narrow_high_parallel = (32, 2300, 8192, 4, 8, 64)
        wide_low_head = (32, 2300, 8192, 4, 2, 512)

        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._EXPERIMENTAL_BLOCK_POLICY_ENV: "1",
                msinfer_entry._BLOCK_SHAPE_TUNING_ENV: "1",
            },
            clear=False,
        ):
            narrow_schedule = msinfer_entry._resolve_kernel_schedule(narrow_high_parallel)
            wide_schedule = msinfer_entry._resolve_kernel_schedule(wide_low_head)
            self.assertEqual(narrow_schedule.outer_schedule_tile, 160)
            self.assertEqual(wide_schedule.outer_schedule_tile, 224)
            self.assertEqual(narrow_schedule.internal_kernel_chunk_size, 128)
            self.assertEqual(wide_schedule.internal_kernel_chunk_size, 128)
            self.assertEqual(narrow_schedule.internal_launch_segments, (128, 32))
            self.assertEqual(wide_schedule.internal_launch_segments, (128, 96))

    def test_internal_kernel_chunk_guard_rejects_unsupported_gdn_tiles(self):
        for illegal_tile in (64, 160, 192, 224):
            with self.subTest(illegal_tile=illegal_tile):
                with self.assertRaisesRegex(ValueError, "must not be forwarded"):
                    msinfer_entry._validate_internal_kernel_chunk_size(illegal_tile)

    def test_kernel_chunk_size_remains_legal_under_experimental_policy(self):
        problem_size = (32, 2300, 8192, 4, 4, 128)
        with mock.patch.dict(
            os.environ,
            {msinfer_entry._EXPERIMENTAL_BLOCK_POLICY_ENV: "1"},
            clear=False,
        ):
            self.assertEqual(msinfer_entry._resolve_kernel_chunk_size(problem_size), 128)


    def test_composite_segment_bounds_repeat_the_outer_tile_pattern(self):
        self.assertEqual(
            list(msinfer_entry._iter_composite_segment_bounds(5, (2, 1))),
            [(0, 2), (2, 3), (3, 5)],
        )


    def test_outer_schedule_decomposition_uses_supported_internal_segments(self):
        self.assertEqual(
            msinfer_entry._decompose_outer_schedule_tile(192, internal_kernel_chunk_size=128),
            (128, 64),
        )
        self.assertEqual(
            msinfer_entry._decompose_outer_schedule_tile(160, internal_kernel_chunk_size=128),
            (128, 32),
        )
        self.assertEqual(
            msinfer_entry._decompose_outer_schedule_tile(224, internal_kernel_chunk_size=128),
            (128, 96),
        )


class CompositeScheduleExecutionTests(unittest.TestCase):
    def test_varlen_composite_schedule_stitches_outputs_and_carries_state(self):
        q = torch.arange(10, dtype=torch.float32).reshape(5, 1, 2)
        k = q + 100
        v = q + 200
        state = torch.stack([torch.zeros((1, 2, 2)), torch.full((1, 2, 2), 10.0)])
        A_log = torch.zeros((1,), dtype=torch.float32)
        a = torch.zeros((5, 1), dtype=torch.float32)
        dt_bias = torch.zeros((1,), dtype=torch.float32)
        b = torch.zeros((5, 1), dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        output = torch.empty_like(q)
        new_state = torch.empty_like(state)
        schedule = msinfer_entry._KernelSchedule(
            outer_schedule_tile=3,
            internal_kernel_chunk_size=128,
            internal_launch_segments=(2, 1),
            experimental_policy_enabled=True,
        )
        call_states = []

        def fake_reference_run(q_seg, k_seg, v_seg, state_seg, A_log_arg, a_seg, dt_bias_arg, b_seg, cu_seg, scale):
            call_idx = len(call_states) + 1
            call_states.append(None if state_seg is None else state_seg.clone())
            return torch.full_like(q_seg, float(call_idx)), torch.full((1, 1, 2, 2), float(call_idx))

        with mock.patch.object(msinfer_entry, "_reference_run", side_effect=fake_reference_run):
            msinfer_entry._run_composite_reference_schedule(
                q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, None, output, new_state, schedule
            )

        self.assertEqual(len(call_states), 3)
        self.assertTrue(torch.equal(call_states[0], state[0:1]))
        self.assertTrue(torch.equal(call_states[1], torch.full((1, 1, 2, 2), 1.0)))
        self.assertTrue(torch.equal(call_states[2], state[1:2]))
        self.assertTrue(torch.equal(output[:2], torch.full_like(output[:2], 1.0)))
        self.assertTrue(torch.equal(output[2:3], torch.full_like(output[2:3], 2.0)))
        self.assertTrue(torch.equal(output[3:5], torch.full_like(output[3:5], 3.0)))
        self.assertTrue(torch.equal(new_state[0], torch.full_like(new_state[0], 2.0)))
        self.assertTrue(torch.equal(new_state[1], torch.full_like(new_state[1], 3.0)))

    def test_experimental_outer_schedule_requires_explicit_slow_harness_flag(self):
        q = torch.zeros((192, 1, 2), dtype=torch.float32)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        A_log = torch.zeros((1,), dtype=torch.float32)
        a = torch.zeros((192, 1), dtype=torch.float32)
        dt_bias = torch.zeros((1,), dtype=torch.float32)
        b = torch.zeros((192, 1), dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 192], dtype=torch.int32)
        output = torch.empty_like(q)
        new_state = torch.empty_like(state)

        with mock.patch.dict(os.environ, {msinfer_entry._EXPERIMENTAL_BLOCK_POLICY_ENV: "1"}, clear=False):
            with self.assertRaisesRegex(NotImplementedError, "slow correctness harness"):
                msinfer_entry.run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, None, output, new_state)


    def test_run_uses_composite_path_for_experimental_outer_schedule(self):
        q = torch.zeros((192, 1, 2), dtype=torch.float32)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        A_log = torch.zeros((1,), dtype=torch.float32)
        a = torch.zeros((192, 1), dtype=torch.float32)
        dt_bias = torch.zeros((1,), dtype=torch.float32)
        b = torch.zeros((192, 1), dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 192], dtype=torch.int32)
        output = torch.empty_like(q)
        new_state = torch.empty_like(state)

        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._EXPERIMENTAL_BLOCK_POLICY_ENV: "1",
                msinfer_entry._COMPOSITE_REFERENCE_HARNESS_ENV: "1",
            },
            clear=False,
        ):
            with mock.patch.object(msinfer_entry, "_reference_run") as reference_run:
                reference_run.side_effect = [
                    (torch.full((128, 1, 2), 1.0), torch.full((1, 1, 2, 2), 1.0)),
                    (torch.full((64, 1, 2), 2.0), torch.full((1, 1, 2, 2), 2.0)),
                ]
                msinfer_entry.run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, None, output, new_state)

        self.assertEqual(reference_run.call_count, 2)
        self.assertTrue(torch.equal(output[:128], torch.full_like(output[:128], 1.0)))
        self.assertTrue(torch.equal(output[128:], torch.full_like(output[128:], 2.0)))
        self.assertTrue(torch.equal(new_state, torch.full_like(new_state, 2.0)))


    def test_dense_composite_schedule_stitches_outputs_and_carries_state(self):
        q = torch.arange(12, dtype=torch.float32).reshape(1, 6, 1, 2)
        k = q + 100
        v = q + 200
        state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        A_log = torch.zeros((1,), dtype=torch.float32)
        a = torch.zeros((1, 6, 1), dtype=torch.float32)
        dt_bias = torch.zeros((1,), dtype=torch.float32)
        b = torch.zeros((1, 6, 1), dtype=torch.float32)
        output = torch.empty_like(q)
        new_state = torch.empty_like(state)
        schedule = msinfer_entry._KernelSchedule(
            outer_schedule_tile=3,
            internal_kernel_chunk_size=128,
            internal_launch_segments=(2, 1),
            experimental_policy_enabled=True,
        )

        def fake_reference_run(q_seg, k_seg, v_seg, state_seg, A_log_arg, a_seg, dt_bias_arg, b_seg, cu_seg, scale):
            call_idx = float(fake_reference_run.call_count)
            fake_reference_run.call_count += 1
            return torch.full_like(q_seg, call_idx), torch.full((1, 1, 2, 2), call_idx)

        fake_reference_run.call_count = 1
        with mock.patch.object(msinfer_entry, "_reference_run", side_effect=fake_reference_run):
            msinfer_entry._run_composite_reference_schedule(
                q, k, v, state, A_log, a, dt_bias, b, None, None, output, new_state, schedule
            )

        self.assertTrue(torch.equal(output[:, :2], torch.full_like(output[:, :2], 1.0)))
        self.assertTrue(torch.equal(output[:, 2:3], torch.full_like(output[:, 2:3], 2.0)))
        self.assertTrue(torch.equal(output[:, 3:5], torch.full_like(output[:, 3:5], 3.0)))
        self.assertTrue(torch.equal(output[:, 5:6], torch.full_like(output[:, 5:6], 4.0)))
        self.assertTrue(torch.equal(new_state, torch.full_like(new_state, 4.0)))

    def test_run_uses_shape_tuned_160_and_224_composite_segments_when_harness_enabled(self):
        base_state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        A_log = torch.zeros((1,), dtype=torch.float32)
        dt_bias = torch.zeros((1,), dtype=torch.float32)

        cases = [
            (64, (128, 32), 160),
            (512, (128, 96), 224),
        ]
        for rhs_dim, expected_segments, expected_outer in cases:
            with self.subTest(rhs_dim=rhs_dim):
                q = torch.zeros((expected_outer, 8 if rhs_dim == 64 else 2, 2), dtype=torch.float32)
                k = torch.zeros_like(q)
                v = torch.zeros_like(q)
                a = torch.zeros((expected_outer, q.shape[1]), dtype=torch.float32)
                b = torch.zeros_like(a)
                cu_seqlens = torch.tensor([0, expected_outer], dtype=torch.int32)
                output = torch.empty_like(q)
                new_state = torch.empty_like(base_state)

                with mock.patch.dict(
                    os.environ,
                    {
                        msinfer_entry._EXPERIMENTAL_BLOCK_POLICY_ENV: "1",
                        msinfer_entry._BLOCK_SHAPE_TUNING_ENV: "1",
                        msinfer_entry._COMPOSITE_REFERENCE_HARNESS_ENV: "1",
                    },
                    clear=False,
                ):
                    with mock.patch.object(msinfer_entry, "_reference_run") as reference_run:
                        reference_run.side_effect = [
                            (torch.full((expected_segments[0], q.shape[1], 2), 1.0), torch.full((1, 1, 2, 2), 1.0)),
                            (torch.full((expected_segments[1], q.shape[1], 2), 2.0), torch.full((1, 1, 2, 2), 2.0)),
                        ]
                        msinfer_entry.run(q, k, v, base_state, A_log, a, dt_bias, b, cu_seqlens, None, output, new_state)

                self.assertEqual(reference_run.call_count, 2)
                self.assertTrue(torch.equal(output[: expected_segments[0]], torch.full_like(output[: expected_segments[0]], 1.0)))
                self.assertTrue(torch.equal(output[expected_segments[0] :], torch.full_like(output[expected_segments[0] :], 2.0)))


    def test_run_uses_compiled_composite_path_when_enabled(self):
        q = torch.zeros((192, 1, 2), dtype=torch.float32)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        A_log = torch.zeros((1,), dtype=torch.float32)
        a = torch.zeros((192, 1), dtype=torch.float32)
        dt_bias = torch.zeros((1,), dtype=torch.float32)
        b = torch.zeros((192, 1), dtype=torch.float32)
        cu_seqlens = torch.tensor([0, 192], dtype=torch.int32)
        output = torch.empty_like(q)
        new_state = torch.empty_like(state)

        def fake_get_compiled_runner(*args, **kwargs):
            output_tensor = args[7]
            state_tensor = args[8]
            call_idx = fake_get_compiled_runner.call_count
            fake_get_compiled_runner.call_count += 1

            def compiled(*_compiled_args, **_compiled_kwargs):
                output_tensor.fill_(float(call_idx))
                state_tensor.fill_(float(call_idx))

            return compiled

        fake_get_compiled_runner.call_count = 1
        fake_stream = type("FakeStream", (), {"cuda_stream": 0})()
        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._EXPERIMENTAL_BLOCK_POLICY_ENV: "1",
                msinfer_entry._COMPOSITE_COMPILED_HARNESS_ENV: "1",
            },
            clear=False,
        ):
            with mock.patch.object(msinfer_entry, "_get_compiled_runner", side_effect=fake_get_compiled_runner):
                with mock.patch.object(msinfer_entry.torch.cuda, "current_stream", return_value=fake_stream):
                    with mock.patch.object(msinfer_entry.cuda, "CUstream", side_effect=lambda stream: stream):
                        msinfer_entry.run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, None, output, new_state)

        self.assertEqual(fake_get_compiled_runner.call_count, 3)
        self.assertTrue(torch.equal(output[:128], torch.full_like(output[:128], 1.0)))
        self.assertTrue(torch.equal(output[128:], torch.full_like(output[128:], 2.0)))
        self.assertTrue(torch.equal(new_state, torch.full_like(new_state, 2.0)))


class Round4PortDispatchTests(unittest.TestCase):
    def test_hybrid_split_is_opt_in_with_disable_escape_hatch(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(msinfer_entry._hybrid_split_enabled())

        with mock.patch.dict(os.environ, {msinfer_entry._HYBRID_ENABLE_ENV: "1"}, clear=True):
            self.assertTrue(msinfer_entry._hybrid_split_enabled())

        with mock.patch.dict(
            os.environ,
            {
                msinfer_entry._HYBRID_ENABLE_ENV: "1",
                msinfer_entry._HYBRID_DISABLE_ENV: "1",
            },
            clear=True,
        ):
            self.assertFalse(msinfer_entry._hybrid_split_enabled())

    def test_short_sequential_launcher_passes_identity_seq_map(self):
        q = torch.empty((1, 1), dtype=torch.float32)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        state = torch.empty((2, 1), dtype=torch.float32)
        A_log = torch.empty((1,), dtype=torch.float32)
        a = torch.empty_like(q)
        dt_bias = torch.empty((1,), dtype=torch.float32)
        b = torch.empty_like(q)
        cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int64)
        output = torch.empty_like(q)
        new_state = torch.empty_like(state)
        fake_driver = mock.Mock()

        with mock.patch.object(msinfer_entry, "_is_sequential_short_candidate", return_value=True):
            with mock.patch.object(msinfer_entry, "_get_sequential_kernel", return_value="kernel"):
                with mock.patch.object(msinfer_entry, "_current_driver_stream", return_value="stream"):
                    with mock.patch.object(msinfer_entry, "_cuda_driver", fake_driver):
                        self.assertTrue(
                            msinfer_entry._try_run_sequential_short_path(
                                q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, 0.125, output, new_state
                            )
                        )

        launch_args = fake_driver.cuLaunchKernel.call_args.args
        self.assertEqual(launch_args[:8], ("kernel", 256, 1, 1, 128, 1, 1, 0))
        self.assertEqual(launch_args[8], "stream")
        self.assertEqual(msinfer_entry._SEQUENTIAL_KERNEL_ARGS.p12.value, 2)
        self.assertEqual(msinfer_entry._SEQUENTIAL_KERNEL_ARGS.p13.value, 0)

    def test_hybrid_split_routes_short_map_and_scatters_long_outputs(self):
        total = 5128
        q = torch.zeros((total, 4, 128), dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        v = torch.zeros((total, 8, 128), dtype=torch.bfloat16)
        state = torch.zeros((5, 8, 128, 128), dtype=torch.float32)
        A_log = torch.zeros((8,), dtype=torch.float32)
        a = torch.zeros((total, 8), dtype=torch.bfloat16)
        dt_bias = torch.zeros((8,), dtype=torch.float32)
        b = torch.zeros((total, 8), dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 64, 5064, 5096, 5112, 5128], dtype=torch.int64)
        output = torch.zeros_like(v)
        new_state = torch.zeros_like(state)
        fake_driver = mock.Mock()

        def fake_compiled_segment(*args, **kwargs):
            output_long = args[8]
            new_state_long = args[9]
            output_long.fill_(7)
            new_state_long.fill_(3)

        msinfer_entry._HYBRID_META_CACHE.clear()
        msinfer_entry._HYBRID_LONG_OUTPUT_CACHE.clear()
        msinfer_entry._HYBRID_LONG_STATE_CACHE.clear()
        with mock.patch.object(msinfer_entry, "_is_hybrid_candidate_inputs", return_value=True):
            with mock.patch.object(msinfer_entry, "_get_sequential_kernel", return_value="kernel"):
                with mock.patch.object(msinfer_entry, "_current_driver_stream", return_value="stream"):
                    with mock.patch.object(msinfer_entry, "_cuda_driver", fake_driver):
                        with mock.patch.object(msinfer_entry, "_copy_cu_seqlens_to_host", return_value=tuple(cu_seqlens.tolist())):
                            with mock.patch.object(msinfer_entry, "_run_compiled_gdn_segment", side_effect=fake_compiled_segment):
                                self.assertTrue(
                                    msinfer_entry._try_run_hybrid_split_path(
                                        q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, 0.125, output, new_state
                                    )
                                )

        launch_args = fake_driver.cuLaunchKernel.call_args.args
        self.assertEqual(launch_args[:8], ("kernel", 512, 1, 1, 128, 1, 1, 0))
        self.assertEqual(msinfer_entry._SEQUENTIAL_KERNEL_ARGS.p12.value, 4)
        self.assertNotEqual(msinfer_entry._SEQUENTIAL_KERNEL_ARGS.p13.value, 0)
        self.assertTrue(torch.equal(output[64:5064], torch.full_like(output[64:5064], 7)))
        self.assertTrue(torch.equal(new_state[1], torch.full_like(new_state[1], 3)))


class TracingFlagTests(unittest.TestCase):
    def test_tracing_env_defaults_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(msinfer_entry._profile_enabled())

    def test_tracing_env_truthy_values_enable_ranges(self):
        with mock.patch.dict(os.environ, {msinfer_entry._TRACE_PHASES_ENV: "1"}, clear=False):
            self.assertTrue(msinfer_entry._profile_enabled())


if __name__ == "__main__":
    unittest.main()
