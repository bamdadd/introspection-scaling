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

Add extraction (~0.2 h/model ≈ 1.6 h) + model-load/overhead (~0.5 h) ≈ **9 GPU-h**.
With a 2× safety factor → **~18 GPU-h**. At ~$3/A100-h (Modal): **≈ $55**, or
**≈ $110** with the 2× headroom. **Well under the $200 budget.**

**LLM-judge (Anthropic API, A2's grader)** — 14,400 grades × ~(1k in + 200 out)
tokens ≈ 14M in / 3M out. On a Haiku-class grader this is order **~$30**
(confirm against current Anthropic pricing before running — do not trust this
figure blind). Total envelope **≈ $140**.

**If the total ever exceeds $200, cut in this order (named, not silent):**
concepts 10 → 6, then trials 20 → 12. Both shrink GPU *and* judge cost linearly.
Do not shrink models or seeds — the ladder and the bands are the contribution.

Dev on Qwen2.5-0.5B locally (CPU ok); Modal only for the ladder.
