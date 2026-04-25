import unittest

import torch

from scripts.gdn_block_solve_reference import block_lower_triangular_solve


class GdnBlockSolveReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_block_solve_matches_torch_solve_triangular_single_head(self) -> None:
        n = 9
        d = 4
        lower = torch.tril(0.01 * torch.randn(n, n, dtype=torch.float64), diagonal=-1)
        rhs = torch.randn(n, d, dtype=torch.float64)
        expected = torch.linalg.solve_triangular(torch.eye(n, dtype=torch.float64) + lower, rhs, upper=False)
        for block_size in [1, 2, 4, 8]:
            actual = block_lower_triangular_solve(lower, rhs, block_size=block_size)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=1e-10))

    def test_block_solve_matches_torch_solve_triangular_batched(self) -> None:
        h = 3
        n = 12
        d = 5
        lower = torch.tril(0.01 * torch.randn(h, n, n, dtype=torch.float64), diagonal=-1)
        rhs = torch.randn(h, n, d, dtype=torch.float64)
        expected = torch.linalg.solve_triangular(
            torch.eye(n, dtype=torch.float64).expand(h, -1, -1) + lower,
            rhs,
            upper=False,
        )
        for block_size in [1, 3, 4, 6]:
            actual = block_lower_triangular_solve(lower, rhs, block_size=block_size)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=1e-10))

    def test_invalid_block_size_raises(self) -> None:
        lower = torch.zeros(4, 4)
        rhs = torch.zeros(4, 2)
        with self.assertRaisesRegex(ValueError, "block_size"):
            block_lower_triangular_solve(lower, rhs, block_size=0)


if __name__ == "__main__":
    unittest.main()
