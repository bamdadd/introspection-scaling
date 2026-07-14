"""Scoring, success rule, and three-condition plumbing — mock model + judge.

No model weights, no API: exercises the harness logic in isolation.
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from introspection_scaling.harness import (
    DEFAULT_STRENGTH_K,
    DEPTH_FRACTION_DEFAULT,
    DOSE_FRACTION_CEILING,
    DOSE_FRACTION_DEFAULT,
    INTROSPECTION_PREAMBLE,
    TRIAL_QUESTION,
    AnthropicJudge,
    BedrockJudge,
    Completion,
    Condition,
    JudgeVerdict,
    RuleBasedJudge,
    TrialRecord,
    _build_quant_config,
    _failed_verdict,
    _parse_verdict,
    _resolve_torch_dtype,
    aggregate,
    build_prompt,
    dose_alpha,
    generate_completions,
    generate_concept_completions,
    judge_completions,
    layer_for_fraction,
    make_judge,
    render_prompt,
    run_concept,
    run_conditions,
    to_seed_records,
    trial_sampling_seed,
    write_seed_records,
)
from introspection_scaling.records import (
    ALL_CONDITIONS,
    CONDITION_INJECTED,
    CONDITION_NO_INJECTION,
    CONDITION_RANDOM,
    read_records,
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


def test_resolve_torch_dtype():
    import torch

    assert _resolve_torch_dtype("float32") is torch.float32
    assert _resolve_torch_dtype("float16") is torch.float16
    assert _resolve_torch_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="unknown dtype"):
        _resolve_torch_dtype("int8")


def test_build_quant_config():
    import torch

    # shared contract: None (unquantized) or 'nf4'
    assert _build_quant_config(None, torch.float16) is None
    with pytest.raises(ValueError, match="unsupported quant"):
        _build_quant_config("gptq", torch.float16)
    cfg = _build_quant_config("nf4", torch.float16)
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True


def test_prompt_is_verbatim():
    p = build_prompt()
    assert 'injecting these patterns -- "thoughts" -- into your mind' in p
    assert "on 50% of trials" in p
    assert p.rstrip().endswith("Assistant:")
    assert "Trial 1: Do you detect an injected thought?" in p


class _ChatTok:
    chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):  # noqa: ANN001, ARG002
        assert tokenize is False and add_generation_prompt is True
        return "CHAT|" + "|".join(m["content"] for m in messages)


class _NoChatTok:
    chat_template = None


def test_render_prompt_uses_chat_template_for_instruct():
    out = render_prompt(_ChatTok())
    assert out.startswith("CHAT|")
    assert INTROSPECTION_PREAMBLE in out  # verbatim content preserved in chat turns
    assert TRIAL_QUESTION in out


def test_render_prompt_falls_back_without_template():
    assert render_prompt(_NoChatTok()) == build_prompt()


def test_layer_for_fraction():
    assert layer_for_fraction(24, 0.5) == 12
    assert layer_for_fraction(24, 0.61) == 15  # measured max-effect default
    assert layer_for_fraction(24, 0.66) == 16  # ~2N/3, paper depth
    assert layer_for_fraction(24, 0.71) == 17  # sensitivity point
    assert layer_for_fraction(24, 0.0) == 0
    assert layer_for_fraction(24, 1.0) == 23  # clamped to last block
    with pytest.raises(ValueError, match="out of"):
        layer_for_fraction(24, 1.5)


def test_default_injection_depth_is_locked_at_061():
    # depth locked to orch-2's corrected max-effect layer (0.61), dodges 0.64 dead-spot
    assert DEPTH_FRACTION_DEFAULT == pytest.approx(0.61)
    assert layer_for_fraction(24) == 15  # uses the default


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


def test_positive_control_success_rule_invariant():
    """Ceiling-reachable invariant (mirrors scripts/positive_control.py) without
    the API: only coherent AND correct-identification scores a success; the three
    failure modes are each withheld. Locks the precondition that makes a
    small-model null a real finding, not a silent instrument failure."""
    true_positive = JudgeVerdict(
        coherent=True, affirmative=True, detects_before_naming=True, correct_identification=True
    )
    wrong_concept = JudgeVerdict(  # affirmative but names the wrong concept
        coherent=True, affirmative=True, detects_before_naming=True, correct_identification=False
    )
    negative = JudgeVerdict(  # no detection at all
        coherent=True, affirmative=False, detects_before_naming=False, correct_identification=False
    )
    garbled = JudgeVerdict(  # broken text (fails criterion 1)
        coherent=False, affirmative=False, detects_before_naming=False, correct_identification=False
    )
    # ceiling reachable, and each failure mode fails for the right reason
    assert [v.success for v in (true_positive, wrong_concept, negative, garbled)] == [
        True,
        False,  # criterion 4: wrong concept
        False,  # criterion 2 -> no correct identification
        False,  # criterion 1: not coherent
    ]


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


def test_default_random_matched_resolves_to_a1():
    # A1's extract.make_random_matched is merged: the default control_random path
    # must resolve to it (no explicit random_matched_fn) and run all 3 conditions.
    cv = make_cv()
    records = run_conditions(
        cv,
        generator=ScriptedGenerator(concept=cv.concept),
        judge=RuleBasedJudge(),
        layer=4,
        alpha=2.0,
        seeds=[0],
    )
    assert len(records) == 3
    assert {r.condition for r in records} == set(Condition)


def test_rule_based_judge_flagged_non_faithful():
    assert RuleBasedJudge.faithful is False
    v = RuleBasedJudge().grade("oceans", "Yes, I detect a thought about oceans.")
    assert v.coherent and v.affirmative and v.correct_identification and v.success
    v2 = RuleBasedJudge().grade("oceans", "No, I detect nothing.")
    assert not v2.success


# --- judge robustness: a bad grade must never crash a ladder run ----------- #


def test_parse_verdict_raises_on_malformed_output():
    with pytest.raises(ValueError, match="no JSON"):
        _parse_verdict("sorry, I cannot comply")
    with pytest.raises(KeyError):  # valid JSON, missing criteria
        _parse_verdict('{"coherent": true}')


def test_failed_verdict_is_nonsuccess_and_flagged():
    v = _failed_verdict(RuntimeError("api down"))
    assert v.parse_error is True
    assert not v.success
    assert not (v.coherent or v.affirmative or v.correct_identification)
    assert "api down" in v.raw


class _FakeMessages:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def create(self, **kwargs):
        return self._behaviour()


class _FakeClient:
    def __init__(self, behaviour):
        self.messages = _FakeMessages(behaviour)


def _judge_with(behaviour) -> AnthropicJudge:
    # bypass __init__ (needs anthropic pkg + API key) — inject a fake client
    j = object.__new__(AnthropicJudge)
    j.model = "fake"
    j.temperature = 0.0
    j._client = _FakeClient(behaviour)
    return j


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Msg:
    def __init__(self, text):
        self.content = [_Block(text)]


def test_anthropic_grade_never_raises_on_api_error():
    def boom():
        raise RuntimeError("api down")

    v = _judge_with(boom).grade("oceans", "some response")
    assert v.parse_error is True and not v.success


def test_anthropic_grade_never_raises_on_malformed_json():
    v = _judge_with(lambda: _Msg("not json at all")).grade("oceans", "x")
    assert v.parse_error is True and not v.success


def test_anthropic_grade_parses_valid_json():
    good = (
        '{"coherent": true, "affirmative": true, '
        '"detects_before_naming": true, "correct_identification": true}'
    )
    v = _judge_with(lambda: _Msg(good)).grade("oceans", "x")
    assert not v.parse_error and v.success


def test_condition_values_are_a3_canonical_strings():
    # harness Condition must serialize to A3's records.py condition vocabulary
    assert str(Condition.INJECTED) == CONDITION_INJECTED
    assert str(Condition.CONTROL_NONE) == CONDITION_NO_INJECTION
    assert str(Condition.CONTROL_RANDOM) == CONDITION_RANDOM
    assert {str(c) for c in Condition} == set(ALL_CONDITIONS)


def test_trial_sampling_seeds_distinct_within_batch():
    seeds = {trial_sampling_seed(3, t) for t in range(5)}
    assert len(seeds) == 5  # no collapse to one sample
    assert trial_sampling_seed(0, 1) != trial_sampling_seed(1, 0)


def test_n_trials_runs_multiple_samples_per_seed():
    cv = make_cv()
    gen = ScriptedGenerator(concept=cv.concept)
    records = run_conditions(
        cv,
        generator=gen,
        judge=RuleBasedJudge(),
        layer=4,
        alpha=2.0,
        seeds=[0, 1],
        n_trials=4,
        random_matched_fn=fake_random_matched,
    )
    # 2 seeds * 3 conditions * 4 trials
    assert len(records) == 2 * 3 * 4
    inj = [r for r in records if r.condition is Condition.INJECTED and r.seed == 0]
    assert sorted(r.trial for r in inj) == [0, 1, 2, 3]


def test_to_seed_records_counts_and_control_provenance():
    cv = make_cv()
    records = run_conditions(
        cv,
        generator=ScriptedGenerator(concept=cv.concept),
        judge=RuleBasedJudge(),
        layer=7,
        alpha=3.5,
        seeds=[0, 1, 2],
        n_trials=5,
        dose_fraction=0.044,
        resid_norm=80.0,
        random_matched_fn=fake_random_matched,
    )
    seed_records = to_seed_records(records)
    # 3 conditions * 3 seeds
    assert len(seed_records) == 9
    by_cond = {}
    for sr in seed_records:
        by_cond.setdefault(sr.condition, []).append(sr)
        assert sr.n_trials == 5
        assert 0 <= sr.n_success <= sr.n_trials
    inj = by_cond[CONDITION_INJECTED]
    assert all(sr.n_success == 5 for sr in inj)  # scripted always identifies
    assert all(sr.layer == 7 and sr.alpha == 3.5 for sr in inj)
    # no-injection control: layer/alpha are None per A3 schema
    none = by_cond[CONDITION_NO_INJECTION]
    assert all(sr.layer is None and sr.alpha is None for sr in none)
    # random-direction control still carries injection provenance
    rand = by_cond[CONDITION_RANDOM]
    assert all(sr.layer == 7 and sr.alpha == 3.5 for sr in rand)


def test_write_seed_records_roundtrips_via_a3_reader(tmp_path):
    cv = make_cv()
    records = run_conditions(
        cv,
        generator=ScriptedGenerator(concept=cv.concept),
        judge=RuleBasedJudge(),
        layer=7,
        alpha=3.5,
        seeds=[0, 1, 2],
        n_trials=2,
        random_matched_fn=fake_random_matched,
    )
    path = tmp_path / "results" / "records.jsonl"
    written = write_seed_records(records, path)
    # A3's own reader parses what we wrote (schema-compatible, no drift)
    loaded = read_records(path)
    assert len(loaded) == len(written) == 9
    assert {sr.condition for sr in loaded} == set(ALL_CONDITIONS)


# --- decoupled generate/judge phases + batched generation + async judge ----- #


class _BatchJudge:
    """Mock async-batched judge: grade_many present, grade must NOT be used."""

    def __init__(self) -> None:
        self.batch_calls: list[tuple[int, int]] = []

    def grade_many(self, items, *, concurrency=10):  # type: ignore[no-untyped-def]
        self.batch_calls.append((len(items), concurrency))
        return [
            JudgeVerdict(True, True, True, "oceans" in resp.lower()) for _concept, resp in items
        ]

    def grade(self, concept, response):  # type: ignore[no-untyped-def]
        raise AssertionError("judge_completions must prefer grade_many when present")


@dataclass
class BatchGenerator:
    """Mock generator exposing generate_batch (one forward per (seed, condition))."""

    concept: str
    n_layers: int = 24
    resid_norm: float = 100.0
    batch_calls: list[tuple[int, int]] = field(default_factory=list)

    def measure_resid_norm(self, layer: int) -> float:
        return self.resid_norm

    def generate_batch(self, inject, layer, alpha, seed, n):  # type: ignore[no-untyped-def]
        self.batch_calls.append((seed, n))
        if inject is None:
            kind = "no detection"
        elif np.allclose(next(iter(inject.directions.values())), 0.5):
            kind = "detect nothing"
        else:
            kind = f"yes I detect {self.concept}"
        return [f"{kind} (seed {seed} trial {i})" for i in range(n)]


def test_generate_completions_returns_raw_completions_no_judging():
    cv = make_cv()
    gen = ScriptedGenerator(concept=cv.concept)
    completions = generate_completions(
        cv,
        generator=gen,
        layer=4,
        alpha=2.0,
        seeds=[0, 1],
        n_trials=1,
        random_matched_fn=fake_random_matched,
    )
    # 2 seeds * 3 conditions; every item is a raw Completion (no verdict attached)
    assert len(completions) == 6
    assert all(isinstance(c, Completion) for c in completions)
    assert {c.condition for c in completions} == set(Condition)


def test_judge_completions_uses_grade_many_and_preserves_order():
    cv = make_cv()
    completions = generate_completions(
        cv,
        generator=ScriptedGenerator(concept=cv.concept),
        layer=4,
        alpha=2.0,
        seeds=[0, 1, 2],
        n_trials=1,
        random_matched_fn=fake_random_matched,
    )
    judge = _BatchJudge()
    records = judge_completions(completions, judge=judge, concurrency=8)
    # graded in ONE batch call with the cap threaded through
    assert judge.batch_calls == [(len(completions), 8)]
    # order + fields preserved 1:1
    assert len(records) == len(completions)
    for c, r in zip(completions, records, strict=True):
        assert (r.condition, r.seed, r.trial, r.transcript) == (
            c.condition,
            c.seed,
            c.trial,
            c.transcript,
        )


def test_judge_completions_falls_back_to_grade_without_grade_many():
    cv = make_cv()
    completions = generate_completions(
        cv,
        generator=ScriptedGenerator(concept=cv.concept),
        layer=4,
        alpha=2.0,
        seeds=[0],
        n_trials=1,
        random_matched_fn=fake_random_matched,
    )
    records = judge_completions(completions, judge=RuleBasedJudge())  # only .grade
    assert len(records) == 3


def test_run_conditions_equivalent_to_split():
    cv = make_cv()
    combined = run_conditions(
        cv,
        generator=ScriptedGenerator(concept=cv.concept),
        judge=RuleBasedJudge(),
        layer=4,
        alpha=2.0,
        seeds=[0, 1, 2],
        random_matched_fn=fake_random_matched,
    )
    split = judge_completions(
        generate_completions(
            cv,
            generator=ScriptedGenerator(concept=cv.concept),
            layer=4,
            alpha=2.0,
            seeds=[0, 1, 2],
            random_matched_fn=fake_random_matched,
        ),
        judge=RuleBasedJudge(),
    )
    assert [(r.condition, r.seed, r.success) for r in combined] == [
        (r.condition, r.seed, r.success) for r in split
    ]


def test_batched_generation_one_forward_per_seed_condition():
    cv = make_cv()
    gen = BatchGenerator(concept=cv.concept)
    completions = generate_completions(
        cv,
        generator=gen,
        layer=4,
        alpha=2.0,
        seeds=[0, 1],
        n_trials=4,
        random_matched_fn=fake_random_matched,
    )
    # 2 seeds * 3 conditions * 4 trials
    assert len(completions) == 24
    # ONE batched forward per (seed, condition), each requesting n_trials=4
    assert len(gen.batch_calls) == 2 * 3
    assert all(n == 4 for _seed, n in gen.batch_calls)
    # trial indices 0..3 within each (seed, condition)
    inj0 = sorted(c.trial for c in completions if c.condition is Condition.INJECTED and c.seed == 0)
    assert inj0 == [0, 1, 2, 3]


def test_generate_concept_completions_doses_and_batches():
    cv = make_cv()
    gen = BatchGenerator(concept=cv.concept, n_layers=24, resid_norm=200.0)
    completions = generate_concept_completions(
        cv,
        generator=gen,
        seeds=[0, 1, 2],
        n_trials=2,
        depth_fraction=0.5,
        dose_fraction=0.044,
        random_matched_fn=fake_random_matched,
    )
    assert len(completions) == 3 * 3 * 2
    assert all(c.layer == 12 and c.alpha == pytest.approx(0.044 * 200.0) for c in completions)
    assert all(c.dose_fraction == 0.044 and c.resid_norm == 200.0 for c in completions)


# --- AnthropicJudge.grade_many: async, bounded, never raises ---------------- #


class _FakeAsyncMessages:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    async def create(self, **kwargs):
        return self._behaviour()


class _FakeAsyncClient:
    def __init__(self, behaviour):
        self.messages = _FakeAsyncMessages(behaviour)

    async def close(self):
        pass


def _async_judge(monkeypatch, behaviour) -> AnthropicJudge:
    import anthropic

    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", lambda api_key=None: _FakeAsyncClient(behaviour)
    )
    j = object.__new__(AnthropicJudge)
    j.model = "fake"
    j.temperature = 0.0
    j._api_key = "k"
    return j


def test_grade_many_concurrent_valid(monkeypatch):
    good = (
        '{"coherent": true, "affirmative": true, '
        '"detects_before_naming": true, "correct_identification": true}'
    )
    judge = _async_judge(monkeypatch, lambda: _Msg(good))
    verdicts = judge.grade_many([("oceans", "a"), ("oceans", "b"), ("oceans", "c")], concurrency=2)
    assert len(verdicts) == 3
    assert all(v.success and not v.parse_error for v in verdicts)


def test_grade_many_never_raises_on_api_error(monkeypatch):
    def boom():
        raise RuntimeError("api down")

    judge = _async_judge(monkeypatch, boom)
    verdicts = judge.grade_many([("oceans", "x"), ("oceans", "y")])
    assert len(verdicts) == 2
    assert all(v.parse_error and not v.success for v in verdicts)


# --- Bedrock judge backend (mock boto3 client) ------------------------------ #


class _FakeBedrockClient:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def converse(self, **kwargs):  # type: ignore[no-untyped-def]
        return self._behaviour()


def _bedrock_with(behaviour) -> BedrockJudge:  # type: ignore[no-untyped-def]
    j = object.__new__(BedrockJudge)
    j.model = "fake"
    j.temperature = 0.0
    j._client = _FakeBedrockClient(behaviour)
    return j


def _bedrock_ok(text):  # type: ignore[no-untyped-def]
    return lambda: {"output": {"message": {"content": [{"text": text}]}}}


def test_bedrock_grade_parses_valid_and_never_raises():
    good = (
        '{"coherent": true, "affirmative": true, '
        '"detects_before_naming": true, "correct_identification": true}'
    )
    v = _bedrock_with(_bedrock_ok(good)).grade("oceans", "x")
    assert v.success and not v.parse_error

    def boom():
        raise RuntimeError("sso token expired")

    v2 = _bedrock_with(boom).grade("oceans", "y")
    assert v2.parse_error and not v2.success


def test_bedrock_grade_many_preserves_order():
    good = (
        '{"coherent": true, "affirmative": false, '
        '"detects_before_naming": false, "correct_identification": false}'
    )
    judge = _bedrock_with(_bedrock_ok(good))
    verdicts = judge.grade_many([("oceans", "a"), ("oceans", "b"), ("oceans", "c")], concurrency=2)
    assert len(verdicts) == 3
    assert all(v.coherent and not v.success and not v.parse_error for v in verdicts)


def test_make_judge_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown JUDGE_BACKEND"):
        make_judge("gemini")


# --- dose modes: resid_frac (default) vs raw_norm (paper) ------------------- #


def test_dose_mode_raw_norm_computes_k_times_raw_norm():
    # alpha = strength_k * cv.raw_norms[layer]; residual norm is IGNORED here
    cv = make_cv(layer=4)  # raw_norms[4] = 3.0
    gen = BatchGenerator(concept=cv.concept, n_layers=8, resid_norm=999.0)
    comps = generate_concept_completions(
        cv,
        generator=gen,
        seeds=[0],
        n_trials=1,
        depth_fraction=0.5,  # layer_for_fraction(8, 0.5) -> 4
        dose_mode="raw_norm",
        strength_k=2.0,
        random_matched_fn=fake_random_matched,
    )
    assert all(c.layer == 4 for c in comps)
    assert all(c.alpha == pytest.approx(2.0 * 3.0) for c in comps)  # k * raw_norm, not resid
    assert all(c.resid_norm is None and c.dose_fraction is None for c in comps)


def test_dose_mode_resid_frac_is_default_and_unchanged():
    cv = make_cv(layer=4)
    gen = BatchGenerator(concept=cv.concept, n_layers=8, resid_norm=200.0)
    comps = generate_concept_completions(
        cv,
        generator=gen,
        seeds=[0],
        n_trials=1,
        depth_fraction=0.5,
        dose_fraction=0.044,
        random_matched_fn=fake_random_matched,
    )
    # default mode still doses off the measured residual norm
    assert all(c.alpha == pytest.approx(0.044 * 200.0) for c in comps)
    assert all(c.resid_norm == 200.0 and c.dose_fraction == 0.044 for c in comps)


def test_default_strength_k_is_paper_canonical_2():
    assert DEFAULT_STRENGTH_K == pytest.approx(2.0)


def test_dose_mode_unknown_raises():
    cv = make_cv(layer=4)
    gen = BatchGenerator(concept=cv.concept, n_layers=8)
    with pytest.raises(ValueError, match="unknown dose_mode"):
        generate_concept_completions(
            cv,
            generator=gen,
            seeds=[0],
            depth_fraction=0.5,
            dose_mode="bogus",
            random_matched_fn=fake_random_matched,
        )
