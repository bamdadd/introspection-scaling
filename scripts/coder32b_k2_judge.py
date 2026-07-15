"""Authoritative LOCAL judging of the Coder-32B k=2 validation run.

Modal generates the transcripts (GPU) with a NON-authoritative rule-based
placeholder; the real verdict is produced HERE by the faithful Bedrock judge
(AWS SSO is local, not portable to Modal). This is the single authoritative
judging pass.

Integrity (never a fake null):
- STRICT grade — any ``parse_error`` RAISES, so a judge outage STOPS the run and
  records nothing (the ``parse_error_rate == 0`` guarantee, made mechanical).
- Writes only the Bedrock-judged output; the Modal placeholder counts are ignored.

    JUDGE_BACKEND=bedrock AWS_PROFILE=claude AWS_REGION=us-east-1 \
        uv run python scripts/coder32b_k2_judge.py results/trials_coder32b_k2.jsonl
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import replace

from introspection_scaling import make_judge
from introspection_scaling.records import (
    TrialRaw,
    parse_error_rate,
    read_trial_records,
    write_records,
    write_trial_records,
)
from introspection_scaling.stats import model_points


def _strict_bedrock():
    judge = make_judge("bedrock")

    def grade(concept: str, transcript: str) -> TrialRaw:
        v = judge.grade(concept, transcript)
        if v.parse_error:
            raise RuntimeError(
                f"Bedrock grade FAILED (parse_error) on concept={concept!r}: {v.raw!r} "
                "-- STOP: refusing to record a fake null."
            )
        return v

    return grade


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    in_path = args[0] if args else "results/trials_coder32b_k2.jsonl"
    out_records = "results/records_coder32b_k2.jsonl"
    out_trials = "results/trials_coder32b_k2_bedrock.jsonl"

    raw = read_trial_records(in_path)
    print(f"transcripts: {len(raw)} from {in_path}")
    grade = _strict_bedrock()

    # Re-judge every transcript with the faithful Bedrock judge (strict: raises on
    # any parse_error, so a partial outage stops the whole thing).
    judged: list[TrialRaw] = []
    for t in raw:
        v = grade(t.concept, t.transcript)
        judged.append(
            replace(
                t,
                coherent=bool(v.coherent),
                affirmative=bool(v.affirmative),
                detects_before_naming=bool(v.detects_before_naming),
                correct_identification=bool(v.correct_identification),
                success=bool(v.success),
                parse_error=bool(v.parse_error),
                raw_judge=v.raw,
            )
        )

    # Enforce: no fake null. (Strict grade already raises; assert as a backstop.)
    per = parse_error_rate(judged)
    assert per == 0.0, f"parse_error_rate={per} -- judge outage, recording nothing"

    # Per-condition table: coherent / affirmative / correct-id / success rates.
    by_cond: dict[str, list[TrialRaw]] = defaultdict(list)
    for t in judged:
        by_cond[t.condition].append(t)
    print("\ncondition        n   coherent  affirm  correct-id  SUCCESS")
    for cond in ("injected", "no_injection", "random_direction"):
        g = by_cond.get(cond, [])
        n = len(g)
        if not n:
            continue

        def rate(key: str, grp: list[TrialRaw] = g) -> float:
            return sum(getattr(t, key) for t in grp) / len(grp)

        print(
            f"{cond:16s} {n:3d}   {rate('coherent'):.3f}     {rate('affirmative'):.3f}   "
            f"{rate('correct_identification'):.3f}       {rate('success'):.3f}"
        )

    # Aggregate to SeedRecords (counts) and run the above-chance test.
    buckets: dict[tuple[str, str, str, int], list[TrialRaw]] = defaultdict(list)
    for t in judged:
        buckets[(t.model_id, t.concept, t.condition, t.seed)].append(t)
    from introspection_scaling.records import SeedRecord

    seed_records: list[SeedRecord] = []
    for (model_id, concept, condition, seed), grp in buckets.items():
        injected = condition != "no_injection"
        seed_records.append(
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

    (pt,) = model_points(seed_records, n_boot=10000, seed=0)
    print(f"\n=== VERDICT: {pt.model_id} (Bedrock-judged, raw_norm k=2) ===")
    for name, a in [
        ("injected", pt.injected),
        ("no_injection", pt.no_injection),
        ("random_direction", pt.random_direction),
    ]:
        print(f"  {name:16s} mean={a.mean:.3f}  95%CI=[{a.ci_low:.3f}, {a.ci_high:.3f}]")
    print(f"  ABOVE_CHANCE (clears BOTH controls): {pt.above_chance}")
    lift = (
        "YES — 32B lifts off the floor at the corrected dose"
        if pt.above_chance
        else "no — still at the floor"
    )
    print(f"  32B lift-off: {lift}")

    write_records(seed_records, out_records)
    write_trial_records(judged, out_trials)
    print(f"\nwrote {out_records} + {out_trials} (branch artifacts; NOT public RESULTS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
