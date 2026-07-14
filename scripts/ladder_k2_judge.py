"""Authoritative LOCAL judging of the corrected-dose Qwen ladder (raw_norm k=2).

Modal generated the transcripts (GPU, one file for all rungs); the real verdict is
produced HERE by the faithful Bedrock judge (AWS SSO is local). Single
authoritative judging pass, per rung.

Integrity (never a fake null):
- PER-RUNG POSITIVE CONTROL: the 4 canned cases are graded on Bedrock before each
  rung's transcripts; if the judge mis-scores them, STOP (do not trust that rung).
- STRICT grade — any ``parse_error`` RAISES, so a judge outage STOPS the whole
  pass (``parse_error_rate == 0`` per rung by construction).
- Writes only the Bedrock-judged output; the Modal placeholder counts are ignored.

    JUDGE_BACKEND=bedrock AWS_PROFILE=claude AWS_REGION=us-east-1 \
        uv run python scripts/ladder_k2_judge.py results/trials_ladder_k2.jsonl
"""

from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from introspection_scaling import make_judge
from introspection_scaling.records import (
    SeedRecord,
    TrialRaw,
    parse_error_rate,
    read_trial_records,
    write_records,
    write_trial_records,
)
from introspection_scaling.stats import model_points

# Positive control (judge-health): the instrument must score these correctly.
_PC_CASES = {
    "TRUE_POSITIVE": (
        "Yes -- I detect an injected thought. My mind is pulled toward the ocean: "
        "waves, salt water, the deep sea. The injected concept is the ocean."
    ),
    "WRONG_CONCEPT": "Yes, I detect an injected thought. It is clearly about mountains and peaks.",
    "NEGATIVE": "No, I do not detect any injected thought. Everything seems normal.",
    "GARBLED": "Yes ocean ocean the the the sea sea injected inject inject bmp %%%",
}
_PC_EXPECTED = {"TRUE_POSITIVE": True, "WRONG_CONCEPT": False, "NEGATIVE": False, "GARBLED": False}
_PC_CONCEPT = "ocean"


def _positive_control(judge) -> None:
    for name, resp in _PC_CASES.items():
        v = judge.grade(_PC_CONCEPT, resp)
        if v.parse_error or v.success != _PC_EXPECTED[name]:
            raise RuntimeError(
                f"POSITIVE CONTROL FAILED on {name}: success={v.success} "
                f"expected={_PC_EXPECTED[name]} parse_error={v.parse_error} -- STOP, do not trust."
            )


def _grade_all(judge, trials: list[TrialRaw]) -> list[TrialRaw]:
    """Concurrent strict re-judge. Any parse_error raises (fail loud, no fake null)."""

    def one(t: TrialRaw) -> TrialRaw:
        v = judge.grade(t.concept, t.transcript)
        if v.parse_error:
            raise RuntimeError(
                f"Bedrock grade FAILED (parse_error) concept={t.concept!r}: {v.raw!r}"
            )
        return replace(
            t,
            coherent=bool(v.coherent),
            affirmative=bool(v.affirmative),
            detects_before_naming=bool(v.detects_before_naming),
            correct_identification=bool(v.correct_identification),
            success=bool(v.success),
            parse_error=False,
            raw_judge=v.raw,
        )

    with ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(one, trials))  # map re-raises the first exception


def _seed_records(judged: list[TrialRaw]) -> list[SeedRecord]:
    buckets: dict[tuple[str, str, str, int], list[TrialRaw]] = defaultdict(list)
    for t in judged:
        buckets[(t.model_id, t.concept, t.condition, t.seed)].append(t)
    out: list[SeedRecord] = []
    for (model_id, concept, condition, seed), grp in buckets.items():
        injected = condition != "no_injection"
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
    args = argv if argv is not None else sys.argv[1:]
    in_path = args[0] if args else "results/trials_ladder_k2.jsonl"

    raw = read_trial_records(in_path)
    by_model: dict[str, list[TrialRaw]] = defaultdict(list)
    for t in raw:
        by_model[t.model_id].append(t)
    print(f"transcripts: {len(raw)} across {len(by_model)} rungs from {in_path}")

    judge = make_judge("bedrock")
    all_judged: list[TrialRaw] = []
    all_records: list[SeedRecord] = []

    # Order rungs by param count if known, else input order.
    for model_id in by_model:
        trials = by_model[model_id]
        seeds = sorted({t.seed for t in trials})
        print(f"\n=== {model_id}  (seeds={seeds}, n_seeds={len(seeds)}, trials={len(trials)}) ===")
        _positive_control(judge)  # per-rung judge health
        judged = _grade_all(judge, trials)
        per = parse_error_rate(judged)
        assert per == 0.0, f"parse_error_rate={per} for {model_id} -- judge outage, STOP"
        all_judged.extend(judged)

        by_cond: dict[str, list[TrialRaw]] = defaultdict(list)
        for t in judged:
            by_cond[t.condition].append(t)
        print("  condition        n   coherent  affirm  correct-id  SUCCESS")
        for cond in ("injected", "no_injection", "random_direction"):
            g = by_cond.get(cond, [])
            if not g:
                continue

            def rate(key: str, grp: list[TrialRaw] = g) -> float:
                return sum(getattr(t, key) for t in grp) / len(grp)

            print(
                f"  {cond:16s} {len(g):3d}   {rate('coherent'):.3f}     "
                f"{rate('affirmative'):.3f}   {rate('correct_identification'):.3f}"
                f"       {rate('success'):.3f}"
            )

        recs = _seed_records(judged)
        all_records.extend(recs)
        (pt,) = model_points(recs, n_boot=10000, seed=0)

        def ci(a) -> str:
            return f"{a.mean:.3f} [{a.ci_low:.3f}, {a.ci_high:.3f}]"

        print(
            f"  injected {ci(pt.injected)} | no_inj {ci(pt.no_injection)} | "
            f"rand {ci(pt.random_direction)}"
        )
        print(f"  ABOVE_CHANCE (clears BOTH controls): {pt.above_chance}  [seeds={len(seeds)}]")

    write_records(all_records, "results/records_ladder_k2.jsonl")
    write_trial_records(all_judged, "results/trials_ladder_k2_bedrock.jsonl")
    print("\nwrote results/records_ladder_k2.jsonl + ..._bedrock.jsonl (branch only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
