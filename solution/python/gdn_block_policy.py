from __future__ import annotations


def choose_block_tile(*, num_heads: int, rhs_dim: int) -> int:
    """Evidence-backed shape-aware tile selector for the block-solve branch.

    Tuned policy from Task 4 research:
    - default center: 192
    - 160 for narrower / higher-parallel-pressure shapes
    - 224 for wider / lower-head shapes
    """
    if num_heads <= 2 or rhs_dim >= 512:
        return 224
    if num_heads >= 8 or rhs_dim <= 64:
        return 160
    return 192


def choose_default_or_tuned_block_tile(*, num_heads: int, rhs_dim: int, enable_shape_tuning: bool) -> int:
    """Practical implementation policy.

    Keep flat 192 as the strongest simple default and only enable nearby
    shape-aware fallbacks when tuning is explicitly requested.
    """
    if not enable_shape_tuning:
        return 192
    return choose_block_tile(num_heads=num_heads, rhs_dim=rhs_dim)
