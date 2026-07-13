"""Stats: bootstrap determinism, aggregation, and the above-chance semantics."""

import numpy as np
import pytest

from introspection_scaling.records import (
    CONDITION_INJECTED,
    CONDITION_NO_INJECTION,
    CONDITION_RANDOM,
    SeedRecord,
)
from introspection_scaling.stats import (
    aggregate,
    bootstrap_ci,
    is_above_chance,
    model_points,
)


def _mk(
    model: str, cond: str, seed: int, ns: int, nt: int = 10, concept: str = "oceans"
) -> SeedRecord:
    return SeedRecord(model, concept, cond, seed, ns, nt)


def test_bootstrap_deterministic_with_seed() -> None:
    vals = [0.1, 0.4, 0.9, 0.2, 0.5]
    a = bootstrap_ci(vals, n_boot=1000, rng=np.random.default_rng(0))
    b = bootstrap_ci(vals, n_boot=1000, rng=np.random.default_rng(0))
    assert a == b
    assert a[0] <= a[1]


def test_bootstrap_single_value_is_degenerate_band() -> None:
    assert bootstrap_ci([0.7], n_boot=100, rng=np.random.default_rng(0)) == (0.7, 0.7)


def test_bootstrap_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([], n_boot=10, rng=np.random.default_rng(0))


def test_aggregate_mean_std() -> None:
    recs = [_mk("m", CONDITION_INJECTED, s, ns) for s, ns in [(0, 2), (1, 4), (2, 6)]]
    (agg,) = aggregate(recs, seed=0)
    assert agg.n_seeds == 3
    assert agg.mean == pytest.approx(0.4)  # (0.2+0.4+0.6)/3
    assert agg.std == pytest.approx(np.std([0.2, 0.4, 0.6], ddof=1))
    assert agg.ci_low <= agg.mean <= agg.ci_high


def test_pool_concepts_sums_counts_per_seed() -> None:
    # seed 0: 3/10 (oceans) + 7/10 (dogs) => 10/20 = 0.5
    recs = [
        _mk("m", CONDITION_INJECTED, 0, 3, concept="oceans"),
        _mk("m", CONDITION_INJECTED, 0, 7, concept="dogs"),
        _mk("m", CONDITION_INJECTED, 1, 4, concept="oceans"),
        _mk("m", CONDITION_INJECTED, 1, 6, concept="dogs"),
    ]
    (agg,) = aggregate(recs, pool_concepts=True, seed=0)
    assert agg.concept is None
    assert agg.mean == pytest.approx(0.5)  # both seeds pool to 0.5


def test_above_chance_true_when_injected_clears_both() -> None:
    # injected high & tight, controls low & tight → non-overlapping.
    recs = (
        [_mk("m", CONDITION_INJECTED, s, 9) for s in range(3)]
        + [_mk("m", CONDITION_NO_INJECTION, s, 0) for s in range(3)]
        + [_mk("m", CONDITION_RANDOM, s, 1) for s in range(3)]
    )
    (pt,) = model_points(recs, n_boot=2000, seed=0)
    assert pt.above_chance is True


def test_above_chance_false_when_bands_overlap() -> None:
    # injected ≈ controls → overlapping bands → not above chance.
    recs = (
        [_mk("m", CONDITION_INJECTED, s, ns) for s, ns in [(0, 3), (1, 4), (2, 5)]]
        + [_mk("m", CONDITION_NO_INJECTION, s, ns) for s, ns in [(0, 3), (1, 4), (2, 5)]]
        + [_mk("m", CONDITION_RANDOM, s, ns) for s, ns in [(0, 2), (1, 4), (2, 6)]]
    )
    (pt,) = model_points(recs, n_boot=2000, seed=0)
    assert pt.above_chance is False


def test_above_chance_uses_max_of_both_controls() -> None:
    hi = aggregate([_mk("m", CONDITION_INJECTED, s, 9) for s in range(3)], seed=0)[0]
    lo_ctrl = aggregate([_mk("m", CONDITION_NO_INJECTION, s, 0) for s in range(3)], seed=0)[0]
    # a control whose band overlaps injected must veto, even if the other is low.
    hi_ctrl = aggregate([_mk("m", CONDITION_RANDOM, s, 9) for s in range(3)], seed=0)[0]
    assert is_above_chance(hi, lo_ctrl, lo_ctrl) is True
    assert is_above_chance(hi, lo_ctrl, hi_ctrl) is False


def test_model_points_requires_all_three_conditions() -> None:
    recs = [_mk("m", CONDITION_INJECTED, s, 5) for s in range(3)]
    with pytest.raises(ValueError, match="missing condition"):
        model_points(recs)
