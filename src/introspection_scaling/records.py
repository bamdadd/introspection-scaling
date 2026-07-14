"""Result record schema — the aggregation seam between A1/A2 and A3.

A1/A2 (extraction + harness) emit one :class:`SeedRecord` per
``(model_id, concept, condition, seed)``: how many introspection trials
succeeded out of ``n_trials`` for that seed. Rates are **not** pre-computed —
carrying the raw ``(n_success, n_trials)`` counts keeps every level of the
design available to the stats layer (a count-level / hierarchical bootstrap
stays possible) and loses no information.

Wire format: JSON Lines (one JSON object per line) at ``results/records.jsonl``.
This module is the single source of truth for that format; A3 owns it, A1/A2
consume it. Do not change field names without flagging orch-1.

Conditions (SPEC §Controls):
    * ``injected``          — the real concept vector, injected at the layer.
    * ``no_injection``      — control: nothing injected (false-positive floor).
    * ``random_direction``  — control: random unit vector, matched-norm.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

CONDITION_INJECTED = "injected"
CONDITION_NO_INJECTION = "no_injection"
CONDITION_RANDOM = "random_direction"

#: The two controls reported beside every injected point (SPEC lines 43-49).
CONTROL_CONDITIONS: tuple[str, str] = (CONDITION_NO_INJECTION, CONDITION_RANDOM)
ALL_CONDITIONS: tuple[str, str, str] = (CONDITION_INJECTED, *CONTROL_CONDITIONS)


@dataclass(frozen=True)
class SeedRecord:
    """One (model, concept, condition, seed) outcome as raw trial counts."""

    model_id: str
    concept: str
    condition: str
    seed: int
    n_success: int
    n_trials: int
    # Provenance of the injection that produced these counts (SPEC: layer + α are
    # parameters, never hardcoded folklore). None for the no-injection control.
    layer: int | None = None
    alpha: float | None = None

    def __post_init__(self) -> None:
        if self.condition not in ALL_CONDITIONS:
            raise ValueError(
                f"unknown condition {self.condition!r}; expected one of {ALL_CONDITIONS}"
            )
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got {self.n_trials}")
        if not 0 <= self.n_success <= self.n_trials:
            raise ValueError(
                f"n_success ({self.n_success}) must be in [0, n_trials ({self.n_trials})]"
            )

    @property
    def rate(self) -> float:
        """Per-seed success rate = n_success / n_trials."""
        return self.n_success / self.n_trials


def write_records(records: Iterable[SeedRecord], path: str | Path) -> None:
    """Append-safe write of records as JSON Lines to ``path``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), sort_keys=True))
            fh.write("\n")


def read_records(path: str | Path) -> list[SeedRecord]:
    """Read JSON Lines records written by :func:`write_records`.

    Blank lines are skipped. Unknown extra keys are rejected so a schema drift
    on the A1/A2 side surfaces loudly instead of being silently dropped.
    """
    out: list[SeedRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(SeedRecord(**json.loads(line)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: bad record: {exc}") from exc
    return out


# --------------------------------------------------------------------------- #
# Raw per-trial layer — audit + offline re-judge.
#
# SeedRecord keeps only counts, so a run graded by a DEAD judge (e.g. credits
# exhausted -> every grade a parse_error -> all-False) yields 0/N that is
# indistinguishable from a real null. Persisting the transcript + the full
# verdict (incl. ``parse_error``) alongside the counts makes any run
# OFFLINE-RE-JUDGEABLE (re-grade the saved transcripts, no GPU re-generation) and
# lets an auditor spot a judge outage after the fact.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrialRaw:
    """One trial's transcript + judge verdict — the re-judgeable raw layer.

    Serialized to ``results/trials*.jsonl`` beside the ``SeedRecord`` counts.
    ``parse_error=True`` marks a grade that FAILED (API/parse error); such a trial
    is never a success, and a run where many/all trials are ``parse_error`` is a
    judge-outage artifact, NOT a real null.
    """

    model_id: str
    concept: str
    condition: str
    seed: int
    trial: int
    transcript: str
    coherent: bool
    affirmative: bool
    detects_before_naming: bool
    correct_identification: bool
    success: bool
    parse_error: bool
    raw_judge: str = ""
    alpha: float | None = None
    layer: int | None = None
    dose_fraction: float | None = None
    resid_norm: float | None = None


_TRIAL_RAW_FIELDS = {f.name for f in fields(TrialRaw)}


def to_trial_raw(trial: Any) -> TrialRaw:
    """Convert an A2 ``TrialRecord`` (duck-typed) into a :class:`TrialRaw`.

    Reads ``trial.verdict`` (JudgeVerdict) without importing the harness, so the
    records module stays free of the A1/A2 dependency cycle.
    """
    v = trial.verdict
    return TrialRaw(
        model_id=trial.model_id,
        concept=trial.concept,
        condition=str(trial.condition),
        seed=int(trial.seed),
        trial=int(trial.trial),
        transcript=trial.transcript,
        coherent=bool(v.coherent),
        affirmative=bool(v.affirmative),
        detects_before_naming=bool(v.detects_before_naming),
        correct_identification=bool(v.correct_identification),
        success=bool(v.success),
        parse_error=bool(v.parse_error),
        raw_judge=getattr(v, "raw", ""),
        alpha=getattr(trial, "alpha", None),
        layer=getattr(trial, "layer", None),
        dose_fraction=getattr(trial, "dose_fraction", None),
        resid_norm=getattr(trial, "resid_norm", None),
    )


def write_trial_records(trials: Iterable[Any], path: str | Path) -> list[TrialRaw]:
    """Write raw per-trial rows as JSON Lines. Accepts A2 ``TrialRecord``s or
    :class:`TrialRaw`. Returns the ``TrialRaw`` list written."""
    raws = [t if isinstance(t, TrialRaw) else to_trial_raw(t) for t in trials]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in raws:
            fh.write(json.dumps(asdict(r), sort_keys=True))
            fh.write("\n")
    return raws


def read_trial_records(path: str | Path) -> list[TrialRaw]:
    """Read raw per-trial rows written by :func:`write_trial_records`."""
    out: list[TrialRaw] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                extra = set(data) - _TRIAL_RAW_FIELDS
                if extra:
                    raise ValueError(f"unknown keys {sorted(extra)}")
                out.append(TrialRaw(**data))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: bad trial row: {exc}") from exc
    return out


def parse_error_rate(trials: Iterable[TrialRaw]) -> float:
    """Fraction of trials whose grade failed. ~1.0 => the run's judge was down;
    its zeros are an artifact, not a real null."""
    ts = list(trials)
    if not ts:
        return 0.0
    return sum(t.parse_error for t in ts) / len(ts)


def rejudge_to_seed_records(
    trials: Iterable[TrialRaw], grade: Callable[[str, str], Any]
) -> list[SeedRecord]:
    """Re-grade saved transcripts OFFLINE (no GPU) and re-aggregate to counts.

    ``grade(concept, transcript)`` returns a verdict object exposing ``.success``
    (e.g. a working Anthropic/Bedrock judge). This is how a run graded by a dead
    judge is recovered once credits are restored — reload transcripts, re-judge,
    rebuild the counts, without re-generating a single token.
    """
    buckets: dict[tuple[str, str, str, int], list[TrialRaw]] = defaultdict(list)
    for t in trials:
        buckets[(t.model_id, t.concept, t.condition, t.seed)].append(t)
    out: list[SeedRecord] = []
    for (model_id, concept, condition, seed), group in buckets.items():
        n_success = sum(bool(grade(concept, t.transcript).success) for t in group)
        injected = condition != CONDITION_NO_INJECTION
        out.append(
            SeedRecord(
                model_id=model_id,
                concept=concept,
                condition=condition,
                seed=seed,
                n_success=n_success,
                n_trials=len(group),
                layer=group[0].layer if injected else None,
                alpha=group[0].alpha if injected else None,
            )
        )
    return out
