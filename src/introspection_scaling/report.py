"""CLI: records.jsonl -> aggregates + hero scaling curve + RESULTS table.

The real, runnable tail of the pipeline (SPEC: aggregate -> plot). Consumes the
``SeedRecord`` JSON Lines that A1/A2's sweep produces; produces
``results/scaling-curve.png`` and a markdown table on stdout. The sweep that
*writes* records.jsonl is the only deferred piece; everything here is live.

    python -m introspection_scaling.report results/records.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .plot import KNOWN_MODELS, plot_scaling_curve
from .records import read_records
from .stats import ModelPoint, model_points


def _markdown_table(points: list[ModelPoint]) -> str:
    reg = KNOWN_MODELS
    lines = [
        "| Model | Params (B) | Injected | No-inj ctrl | Rand-dir ctrl | Above chance? |",
        "|-------|-----------:|---------:|------------:|--------------:|:-------------:|",
    ]
    for p in sorted(points, key=lambda q: reg.get(q.model_id, ("", 0.0))[1]):
        params = reg.get(p.model_id, ("", float("nan")))[1] / 1e9
        i, n, r = p.injected, p.no_injection, p.random_direction
        lines.append(
            f"| {p.model_id} | {params:.1f} "
            f"| {i.mean:.2f} [{i.ci_low:.2f}, {i.ci_high:.2f}] "
            f"| {n.mean:.2f} [{n.ci_low:.2f}, {n.ci_high:.2f}] "
            f"| {r.mean:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] "
            f"| {'YES' if p.above_chance else 'no'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", type=Path, help="path to records.jsonl (from the sweep)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/scaling-curve.png"),
        help="output PNG path for the scaling curve",
    )
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    records = read_records(args.records)
    if not records:
        raise SystemExit(f"no records in {args.records}")

    points = model_points(records, n_boot=args.n_boot, ci=args.ci, seed=args.seed)
    out = plot_scaling_curve(points, args.out)

    print(_markdown_table(points))
    print(f"\nWrote scaling curve -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
