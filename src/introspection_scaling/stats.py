"""Seed aggregation + bootstrap confidence bands (acceptance-critical).

Two distinct numbers per group, kept separate on purpose:

* **Descriptive** ``mean ± std`` over the per-seed rates (RESULTS.md table).
* **Inferential** bootstrap ``[ci_low, ci_high]`` over seeds — this is the band
  used for the "above chance" test.

"Above chance" (SPEC lines 60-63): the injected success rate exceeds **both**
control rates with **non-overlapping** confidence bands. We implement that as
``injected.ci_low > max(control.ci_high for both controls)`` (and injected mean
above both), using the bootstrap band — not ``mean ± std``.

Caveat, stated honestly: with only ~3 seeds the percentile bootstrap band is
coarse and non-overlap fires rarely, so this test is **biased toward the null**.
Per SPEC that is the safe direction ("never above chance … is EQUALLY a
finding") — we do not compensate for it. More seeds tighten the bands.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .records import (
    CONDITION_INJECTED,
    CONDITION_NO_INJECTION,
    CONDITION_RANDOM,
    SeedRecord,
)

DEFAULT_N_BOOT = 10_000
DEFAULT_CI = 0.95


@dataclass(frozen=True)
class Aggregate:
    """Aggregated stats for one ``(model_id, concept, condition)`` group.

    ``concept is None`` means pooled across concepts (per-seed counts summed
    over concepts before rates are formed).
    """

    model_id: str
    concept: str | None
    condition: str
    n_seeds: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class ModelPoint:
    """Everything the scaling curve needs for one model: injected + both controls."""

    model_id: str
    injected: Aggregate
    no_injection: Aggregate
    random_direction: Aggregate
    above_chance: bool


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``, resampling with
    replacement over the sample (here: over seeds).

    Degenerate cases return a zero-width band at the single value / mean so
    callers never crash on 1 seed — but 1-seed bands are meaningless and should
    be flagged upstream.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means: npt.NDArray[np.float64] = arr[idx].mean(axis=1)
    lo = (1.0 - ci) / 2.0 * 100.0
    hi = (1.0 + ci) / 2.0 * 100.0
    return float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi))


def aggregate(
    records: Iterable[SeedRecord],
    *,
    pool_concepts: bool = False,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> list[Aggregate]:
    """Aggregate ``SeedRecord``s into per-(model, concept, condition) stats.

    ``mean``/``std`` are over per-seed rates (std is sample std, ddof=1, or 0.0
    for a single seed). ``ci_low``/``ci_high`` are the bootstrap band over seeds.
    RNG is seeded for reproducibility (reproduce.sh determinism).
    """
    # (model_id, concept) -> {(condition, seed): (n_success, n_trials)}
    grouped: dict[tuple[str, str | None], dict[tuple[str, int], tuple[int, int]]] = defaultdict(
        dict
    )
    for r in records:
        concept: str | None = None if pool_concepts else r.concept
        bucket = grouped[(r.model_id, concept)]
        cs = (r.condition, r.seed)
        s, t = bucket.get(cs, (0, 0))
        bucket[cs] = (s + r.n_success, t + r.n_trials)

    rng = np.random.default_rng(seed)
    out: list[Aggregate] = []
    for (model_id, concept), by_cond_seed in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))
    ):
        # regroup by condition -> {seed: rate}
        by_cond: dict[str, dict[int, float]] = defaultdict(dict)
        for (cond, sd), (s, t) in by_cond_seed.items():
            by_cond[cond][sd] = s / t
        for cond in sorted(by_cond):
            rates = [by_cond[cond][sd] for sd in sorted(by_cond[cond])]
            arr = np.asarray(rates, dtype=np.float64)
            lo, hi = bootstrap_ci(rates, n_boot=n_boot, ci=ci, rng=rng)
            out.append(
                Aggregate(
                    model_id=model_id,
                    concept=concept,
                    condition=cond,
                    n_seeds=arr.size,
                    mean=float(arr.mean()),
                    std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    ci_low=lo,
                    ci_high=hi,
                )
            )
    return out


def is_above_chance(
    injected: Aggregate, no_injection: Aggregate, random_direction: Aggregate
) -> bool:
    """True iff injected band clears BOTH control bands (SPEC lines 60-63).

    Uses the bootstrap CI, not mean ± std. Non-overlapping-and-higher is:
    ``injected.ci_low > max(control.ci_high)`` and injected mean above both.
    """
    ceiling = max(no_injection.ci_high, random_direction.ci_high)
    return (
        injected.ci_low > ceiling
        and injected.mean > no_injection.mean
        and injected.mean > random_direction.mean
    )


def model_points(
    records: Iterable[SeedRecord],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> list[ModelPoint]:
    """Per-model summary (concepts pooled) with the above-chance verdict.

    A model is included only if all three conditions are present; a missing
    control raises, since a point without its controls is unreportable (SPEC:
    report all three on every point).
    """
    aggs = aggregate(records, pool_concepts=True, n_boot=n_boot, ci=ci, seed=seed)
    by_model: dict[str, dict[str, Aggregate]] = defaultdict(dict)
    for a in aggs:
        by_model[a.model_id][a.condition] = a

    points: list[ModelPoint] = []
    for model_id in sorted(by_model):
        conds = by_model[model_id]
        wanted = (CONDITION_INJECTED, CONDITION_NO_INJECTION, CONDITION_RANDOM)
        missing = [c for c in wanted if c not in conds]
        if missing:
            raise ValueError(
                f"{model_id}: missing condition(s) {missing}; "
                "every point needs injected + both controls"
            )
        inj = conds[CONDITION_INJECTED]
        noinj = conds[CONDITION_NO_INJECTION]
        rand = conds[CONDITION_RANDOM]
        points.append(
            ModelPoint(
                model_id=model_id,
                injected=inj,
                no_injection=noinj,
                random_direction=rand,
                above_chance=is_above_chance(inj, noinj, rand),
            )
        )
    return points
