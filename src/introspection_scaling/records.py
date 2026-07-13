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
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

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
