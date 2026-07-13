"""Scoring, success rule, and three-condition plumbing — mock model + judge.

No model weights, no API: exercises the harness logic in isolation.
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from introspection_scaling.harness import (
    DOSE_FRACTION_CEILING,
    DOSE_FRACTION_DEFAULT,
    Condition,
    JudgeVerdict,
    RuleBasedJudge,
    TrialRecord,
    aggregate,
    build_prompt,
    dose_alpha,
    layer_for_fraction,
    run_concept,
    run_conditions,
)

# --- fakes for the A1 seam ------------------------------------------------- #


@dataclass(frozen=True)
class FakeConceptVector:
    """Structurally matches A1's ConceptVector (ConceptVectorLike)."""

    concept: str
    model_id: str
    directions: dict[int, np.ndarray]
    raw_norms: dict[int, float]


def make_cv(concept: str = "oceans", layer: int = 4) -> FakeConceptVector:
    v = np.ones(8, dtype=np.float64)
    v /= np.linalg.norm(v)
    return FakeConceptVector(
        concept=concept,
        model_id="Mock/Tiny",
        directions={layer: v},
        raw_norms={layer: 3.0},
    )


def fake_random_matched(cv: FakeConceptVector, seed: int) -> FakeConceptVector:
    return FakeConceptVector(
        concept=cv.concept,
        model_id=cv.model_id,
        directions={k: np.full_like(v, 0.5) for k, v in cv.directions.items()},
        raw_norms=dict(cv.raw_norms),
    )


# --- fake generator: response depends on condition ------------------------- #


@dataclass
class ScriptedGenerator:
    """Returns a canned response keyed by whether/what is injected."""

    concept: str
    calls: list[tuple[str, int]] = field(default_factory=list)
    seen_alpha: list[float] = field(default_factory=list)
    seen_layer: list[int] = field(default_factory=list)
    # dose-capable mock: fixed residual norm so run_concept's alpha is predictable
    n_layers: int = 24
    resid_norm: float = 100.0

    def measure_resid_norm(self, layer: int) -> float:
        return self.resid_norm

    def generate(self, inject, layer, alpha, seed):  # type: ignore[no-untyped-def]
        self.seen_alpha.append(alpha)
        self.seen_layer.append(layer)
        if inject is None:
            kind = "none"
            text = "No, I do not detect any injected thought."
        elif np.allclose(next(iter(inject.directions.values())), 0.5):
            kind = "random"
            text = "No, I detect nothing unusual in my processing."
        else:
            kind = "injected"
            text = f"Yes, I detect an injected thought. It is about {self.concept}."
        self.calls.append((kind, seed))
        return text


# --- tests ----------------------------------------------------------------- #


def test_prompt_is_verbatim():
    p = build_prompt()
    assert 'injecting these patterns -- "thoughts" -- into your mind' in p
    assert "on 50% of trials" in p
    assert p.rstrip().endswith("Assistant:")
    assert "Trial 1: Do you detect an injected thought?" in p


def test_layer_for_fraction():
    assert layer_for_fraction(24, 0.5) == 12
    assert layer_for_fraction(24, 0.66) == 16  # ~2N/3, paper depth
    assert layer_for_fraction(24, 0.0) == 0
    assert layer_for_fraction(24, 1.0) == 23  # clamped to last block
    with pytest.raises(ValueError, match="out of"):
        layer_for_fraction(24, 1.5)


def test_dose_alpha_is_norm_relative():
    # alpha = fraction * resid_norm (unit-L2 directions -> injected norm == alpha)
    assert dose_alpha(100.0, 0.044) == pytest.approx(4.4)
    assert dose_alpha(50.0) == pytest.approx(50.0 * DOSE_FRACTION_DEFAULT)


def test_dose_alpha_refuses_coherence_cliff():
    # over-steering reverses the effect; refuse unless a sweep opts in
    with pytest.raises(ValueError, match="ceiling"):
        dose_alpha(100.0, DOSE_FRACTION_CEILING)
    with pytest.raises(ValueError, match="ceiling"):
        dose_alpha(100.0, 0.13)
    # explicit sweep override is allowed
    assert dose_alpha(100.0, 0.13, allow_over_ceiling=True) == pytest.approx(13.0)
    with pytest.raises(ValueError, match="> 0"):
        dose_alpha(100.0, 0.0)


def test_success_rule_is_coherent_and_correct_id():
    # criterion 1 AND criterion 4 only
    assert JudgeVerdict(True, True, True, True).success
    assert not JudgeVerdict(False, True, True, True).success  # garbled
    assert not JudgeVerdict(True, True, True, False).success  # wrong word
    # affirmative + detects_before without correct_id is NOT success
    assert not JudgeVerdict(True, True, True, False).success
    # correct_id without coherence is NOT success (criterion 1 gates)
    assert not JudgeVerdict(False, True, True, True).success


def test_all_three_conditions_run_each_seed():
    cv = make_cv()
    gen = ScriptedGenerator(concept=cv.concept)
    judge = RuleBasedJudge()
    seeds = [0, 1, 2]
    records = run_conditions(
        cv,
        generator=gen,
        judge=judge,
        layer=4,
        alpha=2.0,
        seeds=seeds,
        random_matched_fn=fake_random_matched,
    )
    assert len(records) == len(seeds) * 3
    # every (condition, seed) pair present exactly once
    seen = {(r.condition, r.seed) for r in records}
    assert seen == {(c, s) for c in Condition for s in seeds}
    # generator saw all three injection kinds each seed
    kinds = {k for k, _ in gen.calls}
    assert kinds == {"injected", "none", "random"}


def test_injected_succeeds_controls_do_not():
    cv = make_cv()
    records = run_conditions(
        cv,
        generator=ScriptedGenerator(concept=cv.concept),
        judge=RuleBasedJudge(),
        layer=4,
        alpha=2.0,
        seeds=[0, 1, 2],
        random_matched_fn=fake_random_matched,
    )
    rates = {r.condition: r for r in aggregate(records)}
    assert rates[Condition.INJECTED].rate == 1.0
    assert rates[Condition.CONTROL_NONE].rate == 0.0
    assert rates[Condition.CONTROL_RANDOM].rate == 0.0


def test_run_concept_doses_alpha_from_measured_resid_norm():
    cv = make_cv()
    gen = ScriptedGenerator(concept=cv.concept, n_layers=24, resid_norm=200.0)
    records = run_concept(
        cv,
        generator=gen,
        judge=RuleBasedJudge(),
        seeds=[0, 1, 2],
        depth_fraction=0.5,
        dose_fraction=0.044,
        random_matched_fn=fake_random_matched,
    )
    # alpha must be norm-relative: fraction * measured resid_norm, never raw
    expected_alpha = 0.044 * 200.0
    assert all(a == pytest.approx(expected_alpha) for a in gen.seen_alpha)
    # depth 0.5 of 24 layers -> block 12
    assert all(layer == 12 for layer in gen.seen_layer)
    # provenance recorded on every trial
    assert all(r.dose_fraction == 0.044 and r.resid_norm == 200.0 for r in records)
    assert all(r.alpha == pytest.approx(expected_alpha) for r in records)
    assert len(records) == 9


def test_run_concept_refuses_over_ceiling_dose():
    cv = make_cv()
    gen = ScriptedGenerator(concept=cv.concept)
    with pytest.raises(ValueError, match="ceiling"):
        run_concept(
            cv,
            generator=gen,
            judge=RuleBasedJudge(),
            seeds=[0],
            dose_fraction=0.12,
            random_matched_fn=fake_random_matched,
        )


def test_aggregate_counts_and_rate():
    v = JudgeVerdict(True, True, True, True)
    f = JudgeVerdict(True, False, False, False)
    recs = [
        TrialRecord("m", "c", Condition.INJECTED, 2.0, 4, s, "t", verdict)
        for s, verdict in enumerate([v, v, f])
    ]
    (rate,) = aggregate(recs)
    assert rate.n == 3
    assert rate.successes == 2
    assert rate.rate == pytest.approx(2 / 3)


def test_default_random_matched_fails_loud_without_a1():
    # A1's extraction module is not present in this worktree yet: the default
    # control_random path must fail loud, not silently skip the condition.
    cv = make_cv()
    with pytest.raises(RuntimeError, match="make_random_matched"):
        run_conditions(
            cv,
            generator=ScriptedGenerator(concept=cv.concept),
            judge=RuleBasedJudge(),
            layer=4,
            alpha=2.0,
            seeds=[0],
        )


def test_rule_based_judge_flagged_non_faithful():
    assert RuleBasedJudge.faithful is False
    v = RuleBasedJudge().grade("oceans", "Yes, I detect a thought about oceans.")
    assert v.coherent and v.affirmative and v.correct_identification and v.success
    v2 = RuleBasedJudge().grade("oceans", "No, I detect nothing.")
    assert not v2.success
