# Results

> Fill from `reproduce.sh`. Every point = mean ± std over ≥3 seeds, plus a
> percentile bootstrap band over seeds. Report both controls (no-injection,
> random-direction) beside every number. Hero figure: `results/scaling-curve.png`.

| Model | Params | Detection rate | Control (rand dir) | Δ over chance |
|-------|-------:|---------------:|-------------------:|--------------:|
| _tbd_ | _tbd_  | _tbd_          | _tbd_              | _tbd_         |

**Compute:** _GPU type, hours._  **Seeds:** _list._  **Wall-clock:** _tbd._

---

## Cost estimate (write BEFORE any full Modal sweep) — budget < $200

Sweep dimensions (planning defaults; final concept/trial counts land with the
A2 harness interface):

| Dimension  | Count | Notes |
|------------|------:|-------|
| Models     | 8     | Qwen2.5 {0.5, 1.5, 3, 7, 14}B + Llama3.x {1, 3, 8}B |
| Conditions | 3     | injected + no_injection + random_direction (all reported) |
| Seeds      | 3     | ≥3 required for bands |
| Concepts   | 10    | subset of the paper's 50 |
| Trials     | 20    | per (concept, condition, seed); temperature 1 |

Introspection generations per model = 3 × 3 × 10 × 20 = **1,800**; ×8 models =
**14,400** generations, plus one control-vector extraction per model.

**GPU-hours (A100, fp16, ~150-token responses)** — 14B and 8B dominate:

| Model      | s/gen | gen-hours (1,800 gen) |
|------------|------:|----------------------:|
| Qwen 0.5B  | 0.4   | 0.20 |
| Qwen 1.5B  | 0.7   | 0.35 |
| Qwen 3B    | 1.0   | 0.50 |
| Qwen 7B    | 2.5   | 1.25 |
| Qwen 14B   | 4.5   | 2.25 |
| Llama 1B   | 0.6   | 0.30 |
| Llama 3B   | 1.0   | 0.50 |
| Llama 8B   | 2.8   | 1.40 |
| **gen total** |    | **≈ 6.75** |

**Extraction (repeng diff-of-means) — separate line item (per super-orch FLAG-3).**
repeng runs **2 forward passes per concept** (positive `"Tell me about {concept}"`
+ negative `"Tell me about {baseline}"`). Nominal passes = 2 × 10 concepts ×
8 models = **160** short forward passes. If each concept word is paired against
`B` baseline words (the paper's diff-of-means uses a 100-word baseline list),
multiply by `B`: upper bound 2 × 10 × **100** × 8 = **16,000** short passes.
These are short prompts (no long generation), so even the paired upper bound is
cheap — dominated by 14B/8B, ~**1.0–1.5 GPU-h** total across the ladder. Use
`B ≈ 20` baselines/concept in practice (3,200 passes ≈ 0.5 GPU-h) unless A1's
extraction quality needs the full list.

| Cost line             | GPU-h |
|-----------------------|------:|
| Trial inference (gen) | 6.75  |
| Extraction (repeng)   | ~1.0  |
| Model-load / overhead | ~0.5  |
| **Subtotal**          | **~8.3** |
| **× 2 safety factor** | **~17** |

At ~$3/A100-h (Modal): **≈ $50** nominal, **≈ $100** with the 2× headroom.
**Well under the $200 budget.**

**LLM-judge (Anthropic API, A2's grader)** — 14,400 grades × ~(1k in + 200 out)
tokens ≈ 14M in / 3M out. On a Haiku-class grader this is order **~$30**
(confirm against current Anthropic pricing before running — do not trust this
figure blind). Total envelope **≈ $140**.

**If the total ever exceeds $200, cut in this order (named, not silent):**
concepts 10 → 6, then trials 20 → 12. Both shrink GPU *and* judge cost linearly.
Do not shrink models or seeds — the ladder and the bands are the contribution.

Dev on Qwen2.5-0.5B locally (CPU ok); Modal only for the ladder.

## Method notes

### What was underspecified in the paper (and how we resolved it)

We reproduce Lindsey et al. (2025) as faithfully as the public materials allow.
Where the paper leaves something unspecified, we disclose the choice rather than
hide it:

- **Baseline word list — reconstructed substitute.** The paper's 100-word
  baseline appendix was not released publicly. We use a fixed, documented set of
  100 common concrete nouns, disjoint from the 50 concept words, defined in
  `BASELINE_WORDS` (`src/introspection_scaling/extract.py`) — the marked swap-in
  point should the verbatim list ever surface. The 50 concept words are verbatim
  from the paper. Concept vectors are diff-of-means over concept-vs-baseline
  contrasts, so the baseline set is a broad neutral reference; the substitution
  does not bias toward a positive result.
- **Extraction estimator — diff-of-means.** The paper describes "systematic
  diff-of-means" concept vectors. With our constant-positive contrast, an
  off-the-shelf centered-PCA extractor (`repeng`'s `pca_diff`) removes the
  concept signal and returns the top PC of the baseline activations
  (`|cos|` with diff-of-means ~0.1-0.4; `|cos|` with PCA1 of the baselines
  ~1.0; split-half stability ~0.3-0.8). We use diff-of-means directly
  (split-half stability ~0.98, deterministic), which is the paper's stated
  method. Documented upstream: vgel/repeng#77.
