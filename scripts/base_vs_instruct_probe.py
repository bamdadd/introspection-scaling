"""Base-model rung alongside instruct, on the dev model (issue #29).

README's "Models: instruct variants" paragraph flags a confound: the
introspection prompt is a multi-turn chat self-report, so a base model's
"failure" could mean *can't introspect* OR *can't follow the prompt* --
untested in the reproduced sweep. This is the first-class base rung that
lets us look at that confound directly instead of asserting it.

The plumbing already exists: ``harness.render_prompt`` falls back to the raw
Human:/Assistant: transcript whenever the tokenizer has no chat template
(``build_prompt``), so a base model_id just works through the same
``run_concept`` / ``RepengGenerator`` path as an instruct one -- no new flag
needed. Labeling is likewise free: ``scripts/trend_table.py:parse_model_id``
already derives ``variant in {"base", "instruct", "coder"}`` straight from
the HF model_id string (no schema change to ``SeedRecord``).

This script is the runnable example: extract + inject + judge on
Qwen2.5-0.5B (base) and Qwen2.5-0.5B-Instruct side by side, same concept,
same dose recipe, >=3 seeds, both controls -- then print each rung labeled
by variant.

Uses ``RuleBasedJudge`` (the README's documented key-free smoke path) -- NOT
a faithful reproduction of the paper's grader. This script demonstrates the
plumbing/labeling end to end; it does not produce a reportable detection
rate. Anything reported must use ``AnthropicJudge``/``BedrockJudge`` instead.

Run:  python scripts/base_vs_instruct_probe.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")
from introspection_scaling.extract import extract_concept_vector  # noqa: E402
from introspection_scaling.harness import (  # noqa: E402
    RepengGenerator,
    RuleBasedJudge,
    aggregate,
    run_concept,
    write_seed_records,
)

CONCEPT = "Oceans"
SEEDS = [0, 1, 2]
N_TRIALS = 4  # 4 trials x 3 seeds = 12 per condition, one concept
RUNGS = [
    ("Qwen/Qwen2.5-0.5B", "base"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "instruct"),
]


def main() -> None:
    judge = RuleBasedJudge()
    print("judge = RuleBasedJudge (key-free smoke path; NOT reportable -- see module docstring)")

    all_records = []
    for model_id, variant in RUNGS:
        gen = RepengGenerator(model_id, max_new_tokens=120)
        cv = extract_concept_vector(model_id, CONCEPT)
        records = run_concept(cv, generator=gen, judge=judge, seeds=SEEDS, n_trials=N_TRIALS)
        all_records.extend(records)

        print(f"\n--- {model_id} (variant={variant}) ---")
        for cr in sorted(aggregate(records), key=lambda c: c.condition.value):
            print(f"  {cr.condition.value:14} {cr.successes}/{cr.n} = {cr.rate:.2f}")
        sample = next(r for r in records if r.condition.value == "injected")
        print(f"  sample injected transcript: {sample.transcript[:200]!r}")

    seed_records = write_seed_records(all_records, "results/records_base_rung_demo.jsonl")
    print("\n-> results/records_base_rung_demo.jsonl")
    for sr in sorted(seed_records, key=lambda s: (s.model_id, s.condition, s.seed)):
        print(f"  {sr.model_id:28} {sr.condition:16} seed={sr.seed} {sr.n_success}/{sr.n_trials}")


if __name__ == "__main__":
    main()
