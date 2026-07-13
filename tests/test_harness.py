"""Scoring, success rule, and three-condition plumbing — mock model + judge.

No model weights, no API: exercises the harness logic in isolation.
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from introspection_scaling.harness import (
    Condition,
    JudgeVerdict,
    RuleBasedJudge,
    TrialRecord,
    aggregate,
    build_prompt,
    default_injection_layer,
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

    def generate(self, inject, layer, alpha, seed):  # type: ignore[no-untyped-def]
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


def test_default_injection_layer():
    assert default_injection_layer(24) == 16  # round(2*24/3)
    assert default_injection_layer(28) == 19  # round(2*28/3=18.67)


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
