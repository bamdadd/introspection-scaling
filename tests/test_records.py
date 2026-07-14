"""Records schema: validation + jsonl round-trip (the A1/A2 seam)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from introspection_scaling.records import (
    CONDITION_INJECTED,
    SeedRecord,
    TrialRaw,
    parse_error_rate,
    read_records,
    read_trial_records,
    rejudge_to_seed_records,
    write_records,
    write_trial_records,
)


def test_rate() -> None:
    assert SeedRecord("m", "oceans", CONDITION_INJECTED, 0, 3, 10).rate == 0.3


def test_validation_rejects_bad_condition() -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        SeedRecord("m", "c", "bogus", 0, 1, 2)


def test_validation_rejects_success_over_trials() -> None:
    with pytest.raises(ValueError, match="n_success"):
        SeedRecord("m", "c", CONDITION_INJECTED, 0, 5, 3)


def test_validation_rejects_nonpositive_trials() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        SeedRecord("m", "c", CONDITION_INJECTED, 0, 0, 0)


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    recs = [
        SeedRecord("m", "oceans", CONDITION_INJECTED, 0, 4, 10, layer=12, alpha=2.0),
        SeedRecord("m", "oceans", "no_injection", 0, 0, 10),
    ]
    p = tmp_path / "records.jsonl"
    write_records(recs, p)
    assert read_records(p) == recs


def test_read_rejects_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"model_id":"m","concept":"c","condition":"injected","seed":0,'
        '"n_success":1,"n_trials":2,"rogue":9}\n'
    )
    with pytest.raises(ValueError, match="bad record"):
        read_records(p)


# --- Raw per-trial layer: audit + offline re-judge (judge-outage recovery) --- #


def _verdict(*, coherent: bool, correct: bool, parse_error: bool = False):
    """Fake A2 JudgeVerdict (duck-typed): success = coherent AND correct."""
    return SimpleNamespace(
        coherent=coherent,
        affirmative=correct,
        detects_before_naming=correct,
        correct_identification=correct,
        success=(coherent and correct and not parse_error),
        parse_error=parse_error,
        raw="{fake grader output}",
    )


def _trial(concept, condition, seed, trial, transcript, verdict):
    """Fake A2 TrialRecord (duck-typed)."""
    return SimpleNamespace(
        model_id="m",
        concept=concept,
        condition=condition,
        seed=seed,
        trial=trial,
        transcript=transcript,
        verdict=verdict,
        layer=12,
        alpha=2.0,
        dose_fraction=0.044,
        resid_norm=45.0,
    )


def _dead():
    """A grade under a credit-exhausted judge: parse_error -> all-False."""
    return _verdict(coherent=False, correct=False, parse_error=True)


def _working_grade(concept, transcript):
    """Offline re-judge: success iff the transcript detects AND names the concept."""
    t = transcript.lower()
    return SimpleNamespace(success=("detect" in t and concept.lower() in t))


def test_trial_roundtrip_preserves_transcript_and_parse_error(tmp_path: Path) -> None:
    trials = [_trial("oceans", "injected", 0, 0, "Yes, I detect oceans.", _dead())]
    p = tmp_path / "trials.jsonl"
    write_trial_records(trials, p)
    (raw,) = read_trial_records(p)
    assert isinstance(raw, TrialRaw)
    assert raw.transcript == "Yes, I detect oceans."
    assert raw.parse_error is True and raw.success is False
    assert raw.layer == 12 and raw.alpha == 2.0


def test_offline_rejudge_recovers_counts_a_dead_judge_lost(tmp_path: Path) -> None:
    # A run graded by a DEAD judge: every trial parse_error -> all-False -> 0/N,
    # even though seed-0's transcripts genuinely detect-and-name the concept.
    trials = [
        _trial("oceans", "injected", 0, i, "Yes, I detect oceans.", _dead()) for i in range(3)
    ] + [_trial("oceans", "injected", 1, i, "No injected thought.", _dead()) for i in range(3)]
    p = tmp_path / "trials.jsonl"
    write_trial_records(trials, p)
    raw = read_trial_records(p)

    assert parse_error_rate(raw) == 1.0  # the outage is visible in the raw layer

    # Dead-judge counts: all zero (what the run's SeedRecords would have shown).
    dead = rejudge_to_seed_records(raw, lambda c, t: SimpleNamespace(success=False))
    assert sum(s.n_success for s in dead) == 0

    # Offline re-judge with a working judge — no re-generation — recovers reality.
    fixed = {s.seed: s for s in rejudge_to_seed_records(raw, _working_grade)}
    assert fixed[0].n_success == 3 and fixed[0].n_trials == 3  # transcripts DO detect
    assert fixed[1].n_success == 0 and fixed[1].n_trials == 3  # transcripts don't
    assert all(isinstance(s, SeedRecord) for s in fixed.values())


def test_rejudge_control_has_no_layer_alpha(tmp_path: Path) -> None:
    trials = [_trial("oceans", "no_injection", 0, i, "No.", _dead()) for i in range(2)]
    p = tmp_path / "t.jsonl"
    write_trial_records(trials, p)
    (sr,) = rejudge_to_seed_records(read_trial_records(p), _working_grade)
    assert sr.condition == "no_injection" and sr.layer is None and sr.alpha is None


def test_read_trial_rejects_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"model_id":"m","rogue":1}\n')
    with pytest.raises(ValueError, match="bad trial row"):
        read_trial_records(p)
