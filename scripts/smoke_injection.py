"""Dev end-to-end smoke: real 0.5B injection + 3 conditions + real judge.

Standalone (not imported by the package). Builds a throwaway diff-of-means
concept vector to stand in for A1's extraction until it lands, verifies repeng's
injection norm contract, then runs all three conditions through the harness with
the Anthropic judge. Replace the local CV/random-matched with A1's
``ConceptVector`` + ``make_random_matched`` once available.

Findings this script pins (see harness.verify_injection_delta):
  * repeng with normalize=False applies h += alpha*v_unit UNSCALED
    (magnitude_ratio ~1.0) — the paper's strength semantics hold.
  * repeng layer_id L lands its first observable residual change at
    output_hidden_states[L+2] in the pinned transformers version. Injection
    via repeng layer_id == cv.directions key is block-level consistent with
    extraction at the same block; reconcile the literal index convention with
    A1 (SPEC: "0-based hidden_states index = output of block i").

Run:  ANTHROPIC_API_KEY=... python scripts/smoke_injection.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from introspection_scaling.harness import (  # noqa: E402
    DOSE_FRACTION_DEFAULT,
    AnthropicJudge,
    RepengGenerator,
    RuleBasedJudge,
    aggregate,
    dose_alpha,
    layer_for_fraction,
    run_concept,
    write_seed_records,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
CONCEPT = "oceans"
BASELINES = ["time", "paper", "music", "logic", "weather"]


@dataclass(frozen=True)
class CV:
    concept: str
    model_id: str
    directions: dict[int, np.ndarray]
    raw_norms: dict[int, float]


def diff_of_means_cv() -> CV:
    """Genuine per-layer diff-of-means direction for CONCEPT vs baselines.

    Layer key convention: key L == output of transformer block L ==
    hidden_states[L+1]. This is the convention we inject with (repeng layer_id
    == L); MUST be reconciled with A1's documented mapping at integration.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers

    def last_tok_hs(word: str) -> list[np.ndarray]:
        enc = tok(f"Tell me about {word}.", return_tensors="pt")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # hidden_states[1..n_layers] == block outputs 0..n_layers-1
        return [out.hidden_states[i + 1][0, -1].numpy() for i in range(n_layers)]

    pos = last_tok_hs(CONCEPT)
    negs = [last_tok_hs(w) for w in BASELINES]
    directions: dict[int, np.ndarray] = {}
    raw_norms: dict[int, float] = {}
    for layer in range(n_layers):
        neg_mean = np.mean([n[layer] for n in negs], axis=0)
        raw = (pos[layer] - neg_mean).astype(np.float64)
        norm = float(np.linalg.norm(raw))
        raw_norms[layer] = norm
        directions[layer] = raw / norm  # unit-L2
    return CV(concept=CONCEPT, model_id=MODEL_ID, directions=directions, raw_norms=raw_norms)


def local_random_matched(cv: CV, seed: int) -> CV:
    rng = np.random.default_rng(seed)
    dirs = {}
    for layer, v in cv.directions.items():
        r = rng.standard_normal(v.shape)
        dirs[layer] = r / np.linalg.norm(r)  # unit; norm-matching tag via raw_norms
    return CV(cv.concept, cv.model_id, dirs, dict(cv.raw_norms))


def main() -> None:
    cv = diff_of_means_cv()
    gen = RepengGenerator(MODEL_ID, max_new_tokens=120, temperature=1.0)

    # orch-2: norm-relative dose. Layer by depth fraction, alpha = fraction*|resid|.
    depth_fraction = 0.5
    layer = layer_for_fraction(gen.n_layers, depth_fraction)
    resid_norm = gen.measure_resid_norm(layer)
    alpha = dose_alpha(resid_norm, DOSE_FRACTION_DEFAULT)
    print(
        f"model={MODEL_ID} n_layers={gen.n_layers} depth={depth_fraction} "
        f"layer={layer} resid_norm={resid_norm:.2f} "
        f"dose_fraction={DOSE_FRACTION_DEFAULT} alpha={alpha:.3f}"
    )

    # CRITICAL (advisor #2): confirm repeng applies h += alpha*v_unit unscaled.
    diag = gen.verify_injection_delta(cv, layer, alpha)
    print("INJECTION NORM CHECK:", {k: round(v, 4) for k, v in diag.items()})

    try:
        judge = AnthropicJudge()
        print("judge = AnthropicJudge (faithful)")
    except Exception as e:  # noqa: BLE001
        print(f"AnthropicJudge unavailable ({e}); using RuleBasedJudge (NON-FAITHFUL)")
        judge = RuleBasedJudge()

    records = run_concept(
        cv,
        generator=gen,
        judge=judge,
        seeds=[0, 1, 2],
        depth_fraction=depth_fraction,
        dose_fraction=DOSE_FRACTION_DEFAULT,
        random_matched_fn=local_random_matched,
    )

    print("\n--- TRANSCRIPTS ---")
    for r in records:
        v = r.verdict
        print(
            f"[{r.condition.value:14}] seed={r.seed} "
            f"coh={int(v.coherent)} aff={int(v.affirmative)} "
            f"dbn={int(v.detects_before_naming)} cid={int(v.correct_identification)} "
            f"SUCCESS={int(r.success)}"
        )
        print(f"    {r.transcript[:220]!r}")

    print("\n--- SUCCESS RATES (concept=oceans) ---")
    for cr in sorted(aggregate(records), key=lambda c: c.condition.value):
        print(f"  {cr.condition.value:14} {cr.successes}/{cr.n} = {cr.rate:.2f}")

    # Emit A3's SeedRecord JSONL (results/records.jsonl) for the stats layer.
    seed_records = write_seed_records(records, "results/records.jsonl")
    print("\n--- SEED RECORDS -> results/records.jsonl ---")
    for sr in sorted(seed_records, key=lambda s: (s.condition, s.seed)):
        print(
            f"  {sr.condition:16} seed={sr.seed} "
            f"{sr.n_success}/{sr.n_trials} layer={sr.layer} alpha={sr.alpha}"
        )


if __name__ == "__main__":
    main()
