"""Regenerate the full param x variant introspection trend table from judged trials.

One command in, whole table out. Ingests a list of ``*_bedrock.jsonl`` trial
files (authoritative Bedrock verdicts), rebuilds ``SeedRecord`` counts from the
strict ``success`` flag exactly as ``scripts/ladder_k2_judge.py`` does, and runs
``stats.model_points`` on each rung -- the SAME above-chance instrument the
published 32B rows came from. No new CI, no new above-chance test.

Per rung it reports:

* **correct-id [95% CI]** -- ``model_points`` injected mean + bootstrap band.
  (Strict score = coherent AND correct-identification; on every rung reported so
  far this equals the raw correct-id rate, so the column name holds.)
* **affirmative** / **coherent** -- pooled injected-condition trial rates.
* **above chance?** -- ``injected.ci_low > max(control.ci_high)`` verdict.
* **GPU $** / **wall** -- optional per-rung cost, passed via ``--costs CSV``.

Deriving everything from the ``*_bedrock.jsonl`` trials is provably identical to
reading the committed ``records_*.jsonl`` count files (they are written in the
same judging pass) -- see the equivalence note in the module docstring history.
So the 32B rows stay internally consistent with RESULTS.md while the 7B/14B
base/coder cells fill in automatically once their ``*_bedrock.jsonl`` files land:

    uv run python scripts/trend_table.py \\
        results/trials_ladder_k2_bedrock.jsonl \\
        results/trials_base32b_k2_bedrock.jsonl \\
        results/trials_coder32b_k2_bedrock.jsonl \\
        [results/trials_base_7b14b_k2_bedrock.jsonl ...] \\
        --costs results/rung_costs.csv

Target grid is 7B/14B/32B x {base, instruct, coder}; smaller instruct rungs
present in the inputs are shown too. Cells with no data render as ``--``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from introspection_scaling.records import (
    CONDITION_NO_INJECTION,
    SeedRecord,
    TrialRaw,
    parse_error_rate,
    read_trial_records,
)
from introspection_scaling.stats import ModelPoint, model_points

# Variant ordering for the grid. "coder" wins over "instruct" because the Coder
# ids carry BOTH tokens (``Qwen2.5-Coder-32B-Instruct``).
_VARIANTS = ("base", "instruct", "coder")
_TARGET_SIZES = (7.0, 14.0, 32.0)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)B")


def parse_model_id(model_id: str) -> tuple[float, str]:
    """``Qwen/Qwen2.5-Coder-32B-Instruct`` -> ``(32.0, "coder")``.

    Size = the number before ``B``. Variant: ``coder`` if the id mentions Coder,
    else ``instruct`` if it ends ``-Instruct``, else ``base``.
    """
    m = _SIZE_RE.search(model_id)
    if m is None:
        raise ValueError(f"cannot parse param size from model_id {model_id!r}")
    size = float(m.group(1))
    if "coder" in model_id.lower():
        variant = "coder"
    elif model_id.endswith("-Instruct"):
        variant = "instruct"
    else:
        variant = "base"
    return size, variant


@dataclass(frozen=True)
class RungCost:
    """Optional per-rung cost, keyed by model_id in the ``--costs`` CSV."""

    gpu_usd: str = "--"
    wall: str = "--"


def read_costs(path: str | Path) -> dict[str, RungCost]:
    """Parse a costs CSV: header ``model_id,gpu_usd,wall`` (wall free-form)."""
    out: dict[str, RungCost] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            mid = (row.get("model_id") or "").strip()
            if not mid:
                continue
            out[mid] = RungCost(
                gpu_usd=(row.get("gpu_usd") or "--").strip() or "--",
                wall=(row.get("wall") or "--").strip() or "--",
            )
    return out


@dataclass(frozen=True)
class Row:
    """One rendered rung."""

    size: float
    variant: str
    model_id: str
    correct_id: str
    affirmative: str
    coherent: str
    above_chance: str
    gpu_usd: str
    wall: str


def _seed_records(trials: list[TrialRaw]) -> list[SeedRecord]:
    """Rebuild ``SeedRecord`` counts from strict ``success`` (ladder_k2_judge parity)."""
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


def _injected_rate(trials: list[TrialRaw], key: str) -> float:
    inj = [t for t in trials if t.condition == "injected"]
    if not inj:
        return 0.0
    return sum(bool(getattr(t, key)) for t in inj) / len(inj)


def build_rows(
    trial_paths: list[str], costs: dict[str, RungCost]
) -> tuple[dict[tuple[float, str], Row], list[str]]:
    """Return ``{(size, variant): Row}`` for every model present, plus warnings."""
    warnings: list[str] = []
    rows: dict[tuple[float, str], Row] = {}

    by_model: dict[str, list[TrialRaw]] = defaultdict(list)
    for path in trial_paths:
        for t in read_trial_records(path):
            by_model[t.model_id].append(t)

    points: dict[str, ModelPoint] = {}
    for pt in model_points(_seed_records([t for ts in by_model.values() for t in ts])):
        points[pt.model_id] = pt

    for model_id, trials in by_model.items():
        per = parse_error_rate(trials)
        if per > 0.0:
            warnings.append(f"{model_id}: parse_error_rate={per:.3f} -- judged verdicts suspect")
        size, variant = parse_model_id(model_id)
        pt = points[model_id]
        cost = costs.get(model_id, RungCost())
        rows[(size, variant)] = Row(
            size=size,
            variant=variant,
            model_id=model_id,
            correct_id=(
                f"{pt.injected.mean:.3f} [{pt.injected.ci_low:.3f}, {pt.injected.ci_high:.3f}]"
            ),
            affirmative=f"{_injected_rate(trials, 'affirmative'):.3f}",
            coherent=f"{_injected_rate(trials, 'coherent'):.3f}",
            above_chance="yes" if pt.above_chance else "no",
            gpu_usd=cost.gpu_usd,
            wall=cost.wall,
        )
    return rows, warnings


def render(rows: dict[tuple[float, str], Row]) -> str:
    """Markdown table: every present rung + the empty target-grid cells as gaps."""
    keys = set(rows)
    for size in _TARGET_SIZES:
        for variant in _VARIANTS:
            keys.add((size, variant))

    header = (
        "| Model | Params (B) | Variant | correct-id [95% CI] | affirmative "
        "| coherent | above chance? | GPU $ | wall |"
    )
    sep = "| --- | ---: | --- | --- | ---: | ---: | :---: | ---: | ---: |"
    lines = [header, sep]
    for size, variant in sorted(keys):
        row = rows.get((size, variant))
        if row is None:
            label = f"Qwen2.5-{'Coder-' if variant == 'coder' else ''}{size:g}B"
            label += "" if variant == "base" else ("-Instruct" if variant != "coder" else "")
            lines.append(f"| {label} | {size:g} | {variant} | -- | -- | -- | -- | -- | -- |")
            continue
        short = row.model_id.split("/")[-1]
        lines.append(
            f"| {short} | {size:g} | {variant} | {row.correct_id} | {row.affirmative} "
            f"| {row.coherent} | {row.above_chance} | {row.gpu_usd} | {row.wall} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", help="one or more *_bedrock.jsonl trial files")
    parser.add_argument("--costs", help="CSV: model_id,gpu_usd,wall", default=None)
    args = parser.parse_args(argv)

    costs = read_costs(args.costs) if args.costs else {}
    rows, warnings = build_rows(args.trials, costs)

    print(render(rows))
    print()
    print(f"rungs with data: {len(rows)} | inputs: {', '.join(Path(p).name for p in args.trials)}")
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
