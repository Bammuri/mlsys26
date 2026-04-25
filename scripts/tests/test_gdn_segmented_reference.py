import unittest

import torch

from scripts.gdn_segmented_reference import (
    build_chunk_summary,
    build_compact_chunk_summary,
    build_parallel_chunk_factors,
    build_parallel_chunk_factors_batched,
    chunk_parallel_from_compact_summary,
    chunk_parallel_from_triangular_factors,
    chunk_parallel_from_triangular_factors_batched,
    compose_chunk_summaries,
    materialize_compact_chunk_summary,
    outputs_from_compact_chunk_summary,
    segmented_gdn_reference,
    sequential_gdn_reference,
)


class GdnSegmentedReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_segmented_reference_matches_sequential_reference(self) -> None:
        seq_len = 7
        d_k = 4
        d_v = 3
        q = torch.randn(seq_len, d_k, dtype=torch.float64)
        k = torch.randn(seq_len, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        initial_state = torch.randn(d_v, d_k, dtype=torch.float64)

        expected_out, expected_state = sequential_gdn_reference(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
        )
        actual_out, actual_state = segmented_gdn_reference(
            q,
            k,
            v,
            gate,
            beta,
            chunk_size=3,
            initial_state=initial_state,
        )

        self.assertTrue(torch.allclose(actual_out, expected_out, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(actual_state, expected_state, atol=1e-10, rtol=1e-10))

    def test_chunk_summary_composition_matches_full_chunk_summary(self) -> None:
        seq_len = 6
        d_k = 4
        d_v = 2
        k = torch.randn(seq_len, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))

        left = build_chunk_summary(k[:3], v[:3], gate[:3], beta[:3])
        right = build_chunk_summary(k[3:], v[3:], gate[3:], beta[3:])
        full = build_chunk_summary(k, v, gate, beta)

        composed_P, composed_H = compose_chunk_summaries(left, right)
        self.assertTrue(torch.allclose(composed_P, full.terminal_P, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(composed_H, full.terminal_H, atol=1e-10, rtol=1e-10))

    def test_compact_chunk_summary_materializes_full_summary_exactly(self) -> None:
        seq_len = 6
        d_k = 4
        d_v = 3
        k = torch.randn(seq_len, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))

        full = build_chunk_summary(k, v, gate, beta)
        compact = build_compact_chunk_summary(k, v, gate, beta)
        materialized = materialize_compact_chunk_summary(compact)

        self.assertTrue(torch.allclose(materialized.prefix_P, full.prefix_P, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(materialized.prefix_H, full.prefix_H, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(materialized.terminal_P, full.terminal_P, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(materialized.terminal_H, full.terminal_H, atol=1e-10, rtol=1e-10))

    def test_compact_chunk_outputs_match_sequential_outputs(self) -> None:
        seq_len = 5
        d_k = 4
        d_v = 3
        q = torch.randn(seq_len, d_k, dtype=torch.float64)
        k = torch.randn(seq_len, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        initial_state = torch.randn(d_v, d_k, dtype=torch.float64)

        expected_out, _ = sequential_gdn_reference(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
        )
        compact = build_compact_chunk_summary(k, v, gate, beta)
        actual_out = outputs_from_compact_chunk_summary(compact, q, initial_state=initial_state)

        self.assertTrue(torch.allclose(actual_out, expected_out, atol=1e-10, rtol=1e-10))

    def test_chunk_parallel_from_compact_summary_matches_sequential_outputs_and_state(self) -> None:
        seq_len = 6
        d_k = 4
        d_v = 3
        q = torch.randn(seq_len, d_k, dtype=torch.float64)
        k = torch.randn(seq_len, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        initial_state = torch.randn(d_v, d_k, dtype=torch.float64)

        expected_out, expected_state = sequential_gdn_reference(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
        )
        summary = build_compact_chunk_summary(k, v, gate, beta)
        actual_out, actual_state = chunk_parallel_from_compact_summary(
            q,
            summary,
            initial_state=initial_state,
        )

        self.assertTrue(torch.allclose(actual_out, expected_out, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(actual_state, expected_state, atol=1e-10, rtol=1e-10))

    def test_triangular_factor_construction_matches_recurrent_compact_summary(self) -> None:
        seq_len = 7
        d_k = 4
        d_v = 2
        k = torch.randn(seq_len, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))

        compact = build_compact_chunk_summary(k, v, gate, beta)
        c, w, u = build_parallel_chunk_factors(k, v, gate, beta)

        self.assertTrue(torch.allclose(c, compact.gate_prefix, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(w, compact.w, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(u, compact.u, atol=1e-10, rtol=1e-10))

    def test_chunk_parallel_from_triangular_factors_matches_sequential_outputs_and_state(self) -> None:
        seq_len = 8
        d_k = 5
        d_v = 3
        q = torch.randn(seq_len, d_k, dtype=torch.float64)
        k = torch.randn(seq_len, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        initial_state = torch.randn(d_v, d_k, dtype=torch.float64)

        expected_out, expected_state = sequential_gdn_reference(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
        )
        actual_out, actual_state = chunk_parallel_from_triangular_factors(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
        )

        self.assertTrue(torch.allclose(actual_out, expected_out, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(actual_state, expected_state, atol=1e-10, rtol=1e-10))

    def test_batched_triangular_factor_construction_matches_per_head_reference(self) -> None:
        seq_len = 6
        num_heads = 3
        d_k = 4
        d_v = 4
        k = torch.randn(seq_len, num_heads, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, num_heads, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, num_heads, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, num_heads, dtype=torch.float64))

        c, w, u = build_parallel_chunk_factors_batched(k, v, gate, beta)
        for head in range(num_heads):
            c_ref, w_ref, u_ref = build_parallel_chunk_factors(
                k[:, head, :],
                v[:, head, :],
                gate[:, head],
                beta[:, head],
            )
            self.assertTrue(torch.allclose(c[:, head], c_ref, atol=1e-10, rtol=1e-10))
            self.assertTrue(torch.allclose(w[:, head, :], w_ref, atol=1e-10, rtol=1e-10))
            self.assertTrue(torch.allclose(u[:, head, :], u_ref, atol=1e-10, rtol=1e-10))

    def test_batched_triangular_chunk_parallel_matches_per_head_sequential_reference(self) -> None:
        seq_len = 7
        num_heads = 2
        d_k = 5
        d_v = 5
        q = torch.randn(seq_len, num_heads, d_k, dtype=torch.float64)
        k = torch.randn(seq_len, num_heads, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, num_heads, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, num_heads, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, num_heads, dtype=torch.float64))
        initial_state = torch.randn(num_heads, d_v, d_k, dtype=torch.float64)

        actual_out, actual_state = chunk_parallel_from_triangular_factors_batched(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
        )

        expected_out = []
        expected_state = []
        for head in range(num_heads):
            out_ref, state_ref = sequential_gdn_reference(
                q[:, head, :],
                k[:, head, :],
                v[:, head, :],
                gate[:, head],
                beta[:, head],
                initial_state=initial_state[head],
            )
            expected_out.append(out_ref)
            expected_state.append(state_ref)
        expected_out = torch.stack(expected_out, dim=1)
        expected_state = torch.stack(expected_state, dim=0)

        self.assertTrue(torch.allclose(actual_out, expected_out, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(actual_state, expected_state, atol=1e-10, rtol=1e-10))

    def test_chunk_size_one_reduces_to_sequential(self) -> None:
        seq_len = 5
        d_k = 3
        d_v = 3
        q = torch.randn(seq_len, d_k, dtype=torch.float64)
        k = torch.randn(seq_len, d_k, dtype=torch.float64)
        v = torch.randn(seq_len, d_v, dtype=torch.float64)
        gate = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(seq_len, dtype=torch.float64))

        expected_out, expected_state = sequential_gdn_reference(q, k, v, gate, beta)
        actual_out, actual_state = segmented_gdn_reference(q, k, v, gate, beta, chunk_size=1)

        self.assertTrue(torch.allclose(actual_out, expected_out, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(actual_state, expected_state, atol=1e-10, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
