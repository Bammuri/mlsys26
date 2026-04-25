from __future__ import annotations


def choose_block_tile(*, num_heads: int, rhs_dim: int) -> int:
    """Evidence-backed tile policy from Ralph Task 4 experiments.

    Default to 192. Use 160 for narrower RHS or higher head count, and 224 for
    very wide RHS or lower head count. This reflects the measured tradeoff that
    192 is the strongest simple default while 160/224 provide small shape-aware
    gains in edge regimes.
    """
    if num_heads <= 2 or rhs_dim >= 512:
        return 224
    if num_heads >= 8 or rhs_dim <= 64:
        return 160
    return 192


def choose_default_or_tuned_block_tile(*, num_heads: int, rhs_dim: int, enable_shape_tuning: bool) -> int:
    """Return the practical default tile policy.

    Current Ralph conclusion:
    - keep flat 192 as the strongest simple default;
    - only use 160/224 fallbacks when shape-aware tuning is explicitly enabled.
    """
    if not enable_shape_tuning:
        return 192
    return choose_block_tile(num_heads=num_heads, rhs_dim=rhs_dim)
