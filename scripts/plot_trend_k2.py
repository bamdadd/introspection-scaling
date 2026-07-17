"""Render the honest size-trend hero figure from the judged k2 trials.

Reads the ``*_bedrock.jsonl`` trial files for the 7B/14B/32B base / general-instruct
/ code-instruct rungs, rebuilds ``SeedRecord`` counts from the strict ``success``
flag (same as ``scripts/trend_table.py``), runs ``stats.model_points`` (same CI /
above-chance instrument as the published table), classifies each rung into a
``(size, variant)`` cell, and hands the points to ``plot.plot_variant_trend``.

    uv run python scripts/plot_trend_k2.py            # -> results/scaling_trend_k2.png

Only sizes {7, 14, 32} go on the figure; the smaller Instruct rungs are a
coherence-floor aside in RESULTS, not part of the size-trend claim.
"""

from __future__ import annotations

import re
from collections import defaultdict

from introspection_scaling.plot import plot_variant_trend
from introspection_scaling.records import (
    CONDITION_NO_INJECTION,
    SeedRecord,
    TrialRaw,
    read_trial_records,
)
from introspection_scaling.stats import ModelPoint, model_points

_TRIAL_FILES = [
    "results/trials_ladder_k2_bedrock.jsonl",  # general-instruct 0.5B-32B
    "results/trials_base32b_k2_bedrock.jsonl",
    "results/trials_coder32b_k2_bedrock.jsonl",
    "results/trials_base7b_k2_bedrock.jsonl",
    "results/trials_base14b_k2_bedrock.jsonl",
    "results/trials_coder7b_k2_bedrock.jsonl",
    "results/trials_coder14b_k2_bedrock.jsonl",
]
_TREND_SIZES = (7.0, 14.0, 32.0)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)B")


def _classify(model_id: str) -> tuple[float, str]:
    m = _SIZE_RE.search(model_id)
    if m is None:
        raise ValueError(f"cannot parse param size from {model_id!r}")
    size = float(m.group(1))
    if "coder" in model_id.lower():
        variant = "coder"
    elif model_id.endswith("-Instruct"):
        variant = "instruct"
    else:
        variant = "base"
    return size, variant


def _seed_records(trials: list[TrialRaw]) -> list[SeedRecord]:
    buckets: dict[tuple[str, str, str, int], list[TrialRaw]] = defaultdict(list)
    for t in trials:
        buckets[(t.model_id, t.concept, t.condition, t.seed)].append(t)
    out: list[SeedRecord] = []
    for (model_id, concept, condition, seed), grp in buckets.items():
        injected = condition != CONDITION_NO_INJECTION
        out.append(
            SeedRecord(
                model_id=model_id,
                concept=concept,
                condition=condition,
                seed=seed,
                n_success=sum(t.success for t in grp),
                n_trials=len(grp),
                layer=grp[0].layer if injected else None,
                alpha=grp[0].alpha if injected else None,
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    paths = argv if argv else _TRIAL_FILES
    trials: list[TrialRaw] = []
    for p in paths:
        trials.extend(read_trial_records(p))

    points: list[ModelPoint] = model_points(_seed_records(trials))

    series: dict[str, list[tuple[float, ModelPoint]]] = defaultdict(list)
    for pt in points:
        size, variant = _classify(pt.model_id)
        if size in _TREND_SIZES:
            series[variant].append((size, pt))
    for variant in series:
        series[variant].sort(key=lambda sp: sp[0])

    out = plot_variant_trend(dict(series), "results/scaling_trend_k2.png")
    filled = [
        pt.model_id
        for pt in points
        if pt.above_chance and _classify(pt.model_id)[0] in _TREND_SIZES
    ]
    print(f"wrote {out} | above-chance cells: {filled or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
