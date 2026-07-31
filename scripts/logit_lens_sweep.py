"""Logit-lens layer sweep on the dev model (issue #32).

RESULTS.md's existing logit-lens finding looks at ONE residual point (the
injection layer) in 32B models and shows a legibility dissociation: the
injected concept is linearly decodable in Coder-32B but not Instruct-32B.
This sweeps EVERY layer instead of one, on the 0.5B dev model (base vs
instruct) — forward-pass only, no generation, no judge, effectively free on
CPU. Not a replacement for the 32B finding; a cheap, reusable tool
(``harness.layer_logit_lens`` / ``RepengGenerator.logit_lens_layer_sweep``)
exercised end to end, with real dev-model numbers instead of a stub.

Run:  python scripts/logit_lens_sweep.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")
from introspection_scaling.extract import extract_concept_vector  # noqa: E402
from introspection_scaling.harness import (  # noqa: E402
    RepengGenerator,
    layer_for_fraction,
    resolve_dose,
)

CONCEPT = "Oceans"
RUNGS = [
    ("base", "Qwen/Qwen2.5-0.5B"),
    ("instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
]


def _concept_token_id(tokenizer: object, concept: str) -> int:
    # First token of the concept word as it appears after a space in running
    # text — the standard BPE convention (leading-space token != bare token).
    ids = tokenizer(" " + concept, add_special_tokens=False)["input_ids"]  # type: ignore[operator]
    return int(ids[0])


def run_rung(label: str, model_id: str) -> list[dict[str, float]]:
    gen = RepengGenerator(model_id, max_new_tokens=1)
    cv = extract_concept_vector(model_id, CONCEPT)
    layer = layer_for_fraction(gen.n_layers)
    alpha, resid_norm = resolve_dose(cv, gen, layer)
    concept_token_id = _concept_token_id(gen.tokenizer, CONCEPT)

    resid_norm_str = f"{resid_norm:.2f}" if resid_norm is not None else "n/a"
    print(
        f"[{label}] model={model_id} n_layers={gen.n_layers} layer={layer} "
        f"resid_norm={resid_norm_str} alpha={alpha:.3f} concept_token_id={concept_token_id}"
    )

    records = gen.logit_lens_layer_sweep(cv, layer, alpha, concept_token_id)

    out_path = f"results/logit_lens_sweep_{label}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_id": model_id,
                "concept": CONCEPT,
                "layer": layer,
                "alpha": alpha,
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"  -> {out_path}")

    print(f"  {'layer':>5}  {'rank_base':>9}  {'rank_inj':>8}  {'lift':>8}")
    for r in records:
        print(
            f"  {int(r['layer']):>5}  {int(r['rank_baseline']):>9}  "
            f"{int(r['rank_injected']):>8}  {r['logit_lift']:>8.3f}"
        )
    return records


def main() -> None:
    for label, model_id in RUNGS:
        run_rung(label, model_id)


if __name__ == "__main__":
    main()
