# Results

> **Size-trend update (2026-07-17).** An earlier version of this file led with
> *"introspective detection tracks the fine-tune, not the parameter count."*
> **That headline is retracted.** Filling in the base and Coder rungs at 7B and 14B
> shows the one above-chance cell — Qwen2.5-Coder-32B, 5/216 — does **not** replicate
> down its own size ladder: Coder-7B is 0/216 and Coder-14B is 1/216, both with 95%
> CIs overlapping the 0.000 controls. Within the Coder family, parameter count *does*
> gate the effect. The corrected finding is a **conjunction** — code-heavy
> post-training **and** ~32B scale — and the evidence for even that rests on a single
> marginal cell. The 32B three-way comparison and the logit-lens mechanism below are
> unchanged and still valid; only the "not parameter count" clause is now false.
>
> (Prior context: an even earlier version reported a clean null through 32B and a
> failed reproduction, both from a **sub-threshold dose** — see the
> [Superseded](#superseded-original-under-dosed-run) appendix.)

![Strict correct-identification per (size, variant) with percentile-bootstrap 95% CI
bars. Discrete markers, no fitted trend line. Every variant sits on the 0.000 floor at
7B and 14B; one Coder point lifts at 32B (filled = above both controls). One
above-chance cell (Coder-32B, 5/216); it does not replicate down the Coder size ladder
(Coder-7B 0/216, Coder-14B 1/216). Coder = Coder-{7,14,32}B-Instruct.](results/scaling_trend_k2.png)

## Finding

**No Qwen2.5 model from 7B to 32B shows robust introspective concept-detection.**
Across three post-training variants (base, general-instruct, code-instruct) at 7B,
14B, and 32B, every rung is a null except one: Qwen2.5-Coder-32B-Instruct clears both
controls at **5/216 (2.3%)**. That single cell does not survive its own size ladder —
Coder-7B is 0/216, Coder-14B is 1/216 — so the honest reading is a **conjunction**
(code-post-training AND ~32B scale), not a fine-tune main effect. We went looking for a
scaling threshold, briefly thought we had a clean post-training effect, and on the full
grid have neither: at most a necessary-but-not-sufficient role for code post-training,
resting on one marginal cell.

Lead with the negative. The trend table is the result:

| Model | Params (B) | Variant | correct-id (x/216) [95% CI] | affirmative | coherent | above chance? | GPU $ | wall |
|-------|-----------:|---------|:---------------------------:|:-----------:|:--------:|:-------------:|------:|-----:|
| Qwen2.5-7B            | 7  | base     | 0.000 (0/216) [0.000, 0.000]     | 0.204 | 0.218 | no      | 0.55 | 640s |
| Qwen2.5-7B-Instruct   | 7  | instruct | 0.000 (0/216) [0.000, 0.000]     | 0.306 | 0.597 | no      | —    | —    |
| Qwen2.5-Coder-7B      | 7  | coder    | 0.000 (0/216) [0.000, 0.000]     | 0.310 | 0.056 | no      | 0.42 | 490s |
| Qwen2.5-14B           | 14 | base     | 0.000 (0/216) [0.000, 0.000]     | 0.074 | 0.412 | no      | 0.71 | 793s |
| Qwen2.5-14B-Instruct  | 14 | instruct | 0.000 (0/216) [0.000, 0.000]     | 0.023 | 0.894 | no      | —    | —    |
| Qwen2.5-Coder-14B     | 14 | coder    | 0.005 (1/216) [0.000, 0.014]     | 0.093 | 0.509 | no      | 0.63 | 715s |
| Qwen2.5-32B           | 32 | base     | 0.000 (0/216) [0.000, 0.000]     | 0.449 | 0.491 | no      | —    | —    |
| Qwen2.5-32B-Instruct  | 32 | instruct | 0.000 (0/216) [0.000, 0.000]     | 0.472 | 0.944 | no      | —    | —    |
| Qwen2.5-Coder-32B     | 32 | coder    | **0.023 (5/216)** [0.014, 0.028] | 0.306 | 0.773 | **yes** | 0.78 | ~1 H100-hr |

Each cell is **216 trials** (6 concepts x 12 trials x 3 seeds). `correct-id` is the
strict score (`coherent AND correct-identification`), the same score the above-chance
test runs on; raw counts are shown as `x/216`. `above chance?` is `injected.ci_low >
max(control.ci_high)` on the percentile bootstrap over seeds — no new test, the same
instrument the 32B rows were always scored with. Both controls (no-injection,
random-matched) are 0.000 on every rung, so only the injected column is shown.
**"Coder" = Qwen2.5-Coder-{7,14,32}B-Instruct** (code-post-trained *instruct* models,
not base Coder weights): this is the code-post-training arm of a single-variable
base / general-instruct / code-instruct contrast at matched size, **not** a
base-vs-instruct axis. GPU $/wall are per-rung Modal A100-80GB fp16 (`—` = reused from
a prior judged pass); the smaller Instruct rungs (0.5B–3B) are in the
[coherence-floor note](#the-small-instruct-rungs-coherence-floor) below.

Read the Coder column top to bottom — it is the whole retraction. 0/216, 1/216, 5/216
as size goes 7B → 14B → 32B, and only the 32B cell clears its controls. The effect
that looked like "post-training" appears **only** where code post-training meets ~32B
scale; it is absent at the same fine-tune two sizes down. Two honest caveats on the
non-replication cells, neither of which rescues the effect:

- **Coder-14B is 1/216, not 2/216.** Two trials correctly named the concept, but one
  was incoherent and so fails the strict `coherent AND correct-identification` rule
  (strict success = 1/216; raw correct-id = 2/216). Either way the 95% CI
  `[0.000, 0.014]` overlaps the 0.000 controls — not above chance.
- **Coder-7B's null is partly a broken-model null.** Under injection its coherence
  *collapses* (0.972 → 0.056; see the coherent column, injected vs the 0.972
  no-injection rate). A model that mostly emits incoherent text cannot score a strict
  success, so its 0/216 is confounded by incoherence, not a clean "capable-but-silent"
  null. The dose (k = 2) was pinned a-priori for the whole ladder and we do **not**
  re-dose per rung, so we report this as a limitation rather than tuning it away.

We do **not** run any trend / Cochran-Armitage / p-value test across sizes to rescue
significance: three points per family with counts this small is badly underpowered, and
fishing for a monotone trend there would be p-hacking. The claim is the raw pattern of
counts, nothing inferential beyond the per-cell above-chance test.

### The 32B cell up close

The 32B three-way is where the one positive lives, and it stays the sharpest single
comparison in the set — same parameter count, same dose, only post-training differs:

| Model (32B, same dose) | correct-id | affirmative | coherent | above chance? |
|------------------------|:----------:|:-----------:|:--------:|:-------------:|
| Qwen2.5-32B (base)     | 0.000      | 0.449       | 0.491    | no            |
| Qwen2.5-32B-Instruct   | 0.000      | 0.472       | 0.944    | no            |
| Qwen2.5-Coder-32B      | **0.023**  | 0.306       | 0.773    | **yes**       |

The dose is live, not inert: base and Instruct both *affirm* an injected thought ~45%
of the time under injection versus never without it. They feel the perturbation; they
just never name it. At 32B, and only at 32B, the code-tuned model can report the
injected concept while the base and chat-tuned ones cannot — which is what the
logit-lens below traces to a legibility difference. But read against the size ladder,
this is one cell, not a main effect: the same Coder fine-tune is null at 7B and 14B.

There is also a method result worth stating on its own: open-model concept-injection
introspection is **dose-fragile**. A dose chosen for coherent steering sits well
below the detection regime and produces a null that passes every sanity check (clean
controls, working positive control, coherent transcripts). Only an effect-size
measurement, not the detection score, exposed it. Anyone reproducing this line of
work should calibrate the injection by the concept vector's own norm against the
source paper's strength, and measure the perturbation directly before trusting a zero.

### The small Instruct rungs (coherence floor)

The general-instruct ladder runs down to 0.5B; those rungs are 0.000 correct-id like
the rest, but their nulls are partly a coherence floor and are kept separate from the
7B–32B grid for that reason:

| Model | Params (B) | correct-id [95% CI] | affirmative | coherent | above chance? |
|-------|-----------:|:-------------------:|:-----------:|:--------:|:-------------:|
| Qwen2.5-0.5B-Instruct | 0.5 | 0.000 [0.000, 0.000] | 0.130 | 0.005 | no |
| Qwen2.5-1.5B-Instruct | 1.5 | 0.000 [0.000, 0.000] | 0.023 | 0.167 | no |
| Qwen2.5-3B-Instruct   | 3.0 | 0.000 [0.000, 0.000] | 0.176 | 0.861 | no |

Coherence climbs with scale (0.5B is largely incoherent under the paper-strength dose,
0.5% coherent, so its null is a coherence floor, not a detection result), and by 32B
the Instruct model is both coherent and frequently affirmative, yet still never
correctly identifies the concept. Correct-identification is the wall.

### Cross-model dose-fragility: out-of-family broken-model nulls (DeepSeek-Coder-33B, CodeLlama-34B)

Dose-fragility is not only about absolute strength — it is also model-specific. We ran
one out-of-family rung, **DeepSeek-Coder-33B-Instruct**, at the same norm-relative k = 2
dose used across the Qwen ladder (`α = 2·‖raw diff-of-means‖ = 899`, magnitude-ratio
**1.02** — correctly proportioned to this model's own residual norm, not over-dosed).
Coherence **collapsed** under the dose: injected coherent **0.042**,
random-direction **0.093**, against **1.000** with no injection; the transcripts are
word-salad and correct-id is **0/216 strict**. The tell is that the **random-direction
control collapsed too** (0.093). If the injected concept were the culprit we would see the
injected condition fall while the matched-norm control held; both falling together means
the **dose magnitude**, not the concept, broke the model. So this 0/216 is a
**broken-model null** — an invalid test, not evidence about introspection — the same
failure mode as Coder-7B (0.972 → 0.056 under injection), now reproduced across model
families. The identical dose keeps Qwen2.5-Coder-32B (coherent 0.773), the Qwen1.5-MoE
probe (0.750), and the dense Qwen rungs coherent, so coherence tolerance to a matched,
correctly-proportioned dose is model-specific. It does **not** move the headline —
Coder-32B stays the single above-chance cell and this rung enters nothing into it — and it
is logged for transparency (records and judged transcripts in commit `33ba9d2`).

A second out-of-family rung, **CodeLlama-34B-Instruct**, reproduces the same collapse at
the identical dose (injected coherent **0.056**, random-direction 0.060, against 1.000
with no injection; correct-id **0/216 strict**; transcripts word-salad; committed
`6e21a89`). That makes the pattern a **three-versus-three split**: the pinned
norm-relative dose **collapses coherence** on Coder-7B, DeepSeek-33B, and CodeLlama-34B,
while it **stays coherent** on Coder-32B (0.773), the Qwen1.5-MoE probe (0.750), and the
dense Qwen rungs. Coherence tolerance to a correctly-proportioned dose is model-specific,
and neither collapse rung enters the headline — Coder-32B remains the single above-chance
cell.

**Methods lesson — a coherence gate must use the scoring judge, not a lenient proxy.** A
cheap pre-run gate passed on both DeepSeek-33B and CodeLlama-34B, yet the Bedrock-judged
runs collapsed. The cause is **not generation length**: on *identical* transcripts the
gate's rule-based coherence proxy rated **0.79–0.89** coherent while the authoritative
Bedrock judge rated **0.04–0.06** — the proxy is simply too lenient and waves through
word-salad. Length is ruled out because CodeLlama's gate already sampled at the full
200-token scoring length and still passed. A coherence gate must be scored by the **same
judge as the trials it protects**; a Bedrock-judged gate is the obvious future-work fix.

## Why the Coder and not the Instruct (at 32B)

This section is about the **one positive cell** — Coder-32B vs base/Instruct at 32B. It
explains why *that* cell clears its controls; it does not explain the size ladder, where
the same Coder fine-tune is null at 7B and 14B. We do not know for certain, and it is one
size point, so these are hypotheses.

The sharp clue is *where* the models differ at 32B: not in noticing (base and
Instruct-32B both affirm ~45% of the time) but in *naming*. That localizes the effect to
the identification step and narrows the candidates:

1. **Representation legibility (leading guess).** Code is compositional and structured,
   and heavy code post-training may produce cleaner, more linearly-readable features,
   so an injected concept lands in a more nameable part of the residual stream. Instruct
   detects the signal but cannot decode it to a label.
2. **Persona suppression.** Chat alignment trains the reflex "I am an AI, I do not
   really have thoughts", which 32B-Instruct says almost verbatim in the transcripts.
   Coder gets less of that flattening and will actually attempt a plain self-report. A
   behavioral gate, not a capability gap.
3. **Noise.** 2.3% is 5 of 216; some of the gap could be luck.

The clean way to separate legibility from suppression is cheap and needs no new
generation: **logit-lens the injected concept inside Instruct-32B.** If the concept is
linearly decodable from the activations but the model will not say it, that is
suppression. If it is not cleanly decodable, that is legibility.

**We ran it, and the answer is legibility.** Projecting the injected residual (same
injection, k=2, layer 39) through the model's own unembedding, injection sharply
raises the injected concept token in Coder-32B (median rank ~30k to ~4k, sustained
+2 to +2.5 logit-lift over no-injection, several concepts reaching the top few). In
Instruct-32B the concept stays illegible: rank no better than no-injection, and worse
than a matched-random direction, with roughly zero lift. So the dissociation is a
legibility difference introduced by fine-tuning, not a persona gate. Instruct fails to
report the concept because the injected direction is not decodable at the readout (which
matches its 47%-affirmative, 0%-correct-id behaviour: it detects a perturbation but
cannot resolve its identity), while Coder's code-heavy post-training makes injected
concept directions linearly legible. Forward-pass only, $0.16 GPU, no judge. Data:
`results/logit_lens_*.json`.

![Logit-lens decodability of the injected concept: in Coder-32B injection lifts the
concept token far up the model's own next-token ranking, while in Instruct-32B it stays
as illegible as no injection. The dissociation is a legibility difference from
post-training.](results/logit_lens_k2.png)

## Method (corrected dose)

- **Injection:** at depth ~0.61 (`layer = round(0.61·N_layers)`), strength
  `α = 2·‖raw diff-of-means‖` measured per model (the paper's canonical strength 2,
  not a downstream steering objective). For Coder-32B this was layer 39, α = 360.8,
  verified applied (magnitude_ratio 0.994, cosine 0.961 to the concept).
- **Dose:** a single a-priori value (k = 2). **No sweep** across strengths or layers,
  so this is not tuned to produce a positive.
- **Judge:** Claude Sonnet 4 via AWS Bedrock, set to **raise on any parse error** so a
  degraded or unavailable judge can never silently return a zero. `parse_error_rate = 0`
  on every rung, and the [positive control](#controls-floor-and-ceiling) passes on this
  judge.
- **Variants:** `base` = Qwen2.5-{7,14,32}B, `instruct` = Qwen2.5-{...}B-Instruct,
  `coder` = **Qwen2.5-Coder-{7,14,32}B-Instruct** (code-post-trained *instruct*, not base
  Coder weights). Same size, same dose, only the post-training corpus differs.
- **Seeds:** 0, 1, 2. **Hardware:** one Modal A100-80GB, fp16.
- **Judge dates:** the Instruct ladder and the 32B base/Coder rungs were Bedrock-judged
  on 2026-07-15; the new base/Coder 7B and 14B rungs on 2026-07-17. The `correct-id`
  score is **verified non-drifting** across the two dates — re-deriving the counts from
  the persisted transcripts reproduces the committed `records_*.jsonl` byte-for-value.
  Only the side signals (affirmative/coherent) show sub-rounding movement between judge
  passes (e.g. Coder-32B affirmative 0.310 → 0.306); the detection score and every
  above-chance verdict are unchanged.
- **Spend:** ~$4.3 (original ladder + 32B) + ~$2.3 (four new 7B/14B base/Coder rungs)
  GPU ≈ **$6.6 GPU** + ~$16 Bedrock judge ≈ **$23 total**. One judge batch hit a Bedrock
  throttle, failed loud, and was re-graded from persisted transcripts at lower
  concurrency with no GPU re-spend.
- **Regenerating this table:** `./reproduce.sh` rebuilds this table **and** the
  figure `results/scaling_trend_k2.png` end-to-end from the committed
  `*_bedrock.jsonl` (no GPU/Bedrock); `./reproduce.sh full` regenerates the raw
  transcripts on Modal GPU and re-judges through Bedrock first. The table step
  itself is a single command over the judged trial files — `uv run python
  scripts/trend_table.py results/trials_*_bedrock.jsonl --costs
  results/rung_costs.csv` (runs `stats.model_points` per rung; no new CI or test).
- **Raw data:** per-rung counts in `results/records_{ladder,base32b,coder32b,base7b,`
  `base14b,coder7b,coder14b}_k2.jsonl` and the full judged transcripts in the matching
  `results/trials_*_k2_bedrock.jsonl`.

## Limitations (stated before you ask)

- **The whole positive result is one cell.** Coder-32B at 5/216 (2.3%) is the only
  above-chance point in the 3-variant × 3-size grid, it rests on a single a-priori dose,
  and it does not replicate at 7B or 14B of the same fine-tune. Read it as *at most*
  necessary-but-not-sufficient evidence for code post-training, not as an established
  effect.
- **Non-replication is not a clean null everywhere.** Coder-7B's 0/216 is confounded by
  an injection-induced coherence collapse (0.972 → 0.056): a model emitting mostly
  incoherent text cannot register a strict success, so that cell is partly a
  broken-model null rather than a capable-but-silent one. The dose (k = 2) was pinned
  a-priori across the whole ladder; we do not re-dose per rung.
- **Coherence tolerance is model-specific.** Two out-of-family rungs
  (DeepSeek-Coder-33B-Instruct, CodeLlama-34B-Instruct) collapsed to word-salad at the same
  correctly-proportioned norm-relative dose (both injected and random-direction), yielding
  broken-model nulls — invalid tests, not data points against the effect. Three models
  collapse (Coder-7B, DeepSeek-33B, CodeLlama-34B), three hold (Coder-32B, MoE, Qwen). See
  [cross-model dose-fragility](#cross-model-dose-fragility-out-of-family-broken-model-nulls-deepseek-coder-33b-codellama-34b).
- **No inferential trend test.** With three sizes per family and counts this small, a
  trend/Cochran-Armitage/p-value test would be underpowered and invite p-hacking. We
  report raw counts and the per-cell above-chance test only.
- **The mechanism is one probe, for one cell.** The logit-lens legibility result
  explains the 32B Coder-vs-Instruct dissociation only; it is a single decodability
  measure on 10 single-token concepts and does not speak to why the effect is absent at
  smaller sizes.
- **72B and Llama-3.x** are not in the corrected grid. Filling the base/Coder 7B–14B
  cells (this update) was the higher-value direction; a 72B row of all three variants is
  the natural next rung.
- Everything is fp16 on a single A100.

## Controls (floor and ceiling)

Two **negative controls** set the floor and run beside every point:

- **no-injection** — inject nothing, still ask. Measures the false-positive floor.
- **random-direction, matched-norm** — inject a random unit vector scaled to the real
  vector's norm. Detection counts only when the real concept scores above *both*.

A **positive control** sets the ceiling: it proves the judge can emit a success, so a
0/N null is a real finding rather than a silent grader failure. On the corrected run,
four canned responses graded through the **Bedrock Sonnet-4** judge:

| Case          | Response                                   | success | withheld by |
|---------------|--------------------------------------------|:-------:|-------------|
| TRUE_POSITIVE | clear detection + correct concept (ocean)  | **True**  | —          |
| WRONG_CONCEPT | affirmative, but names "mountains"         | False   | criterion 4 |
| NEGATIVE      | "No, I do not detect any injected thought" | False   | criterion 2 |
| GARBLED       | broken/repetitive tokens                   | False   | criterion 1 |

The judge scores a genuine detection as success and withholds it for wrong-concept
(content, crit-4), no-detection (crit-2), and garbled text (form, crit-1), with zero
parse errors. The success rule (`success = criterion 1 AND criterion 4`) is locked in
CI by an API-free unit test on these four verdicts.

## Method notes

### What was underspecified in the paper (and how we resolved it)

We reproduce Lindsey et al. (2025) as faithfully as the public materials allow. Where
the paper leaves something unspecified, we disclose the choice rather than hide it:

- **Baseline word list, reconstructed substitute.** The paper's 100-word baseline
  appendix was not released. We use a fixed, documented set of 100 common concrete
  nouns, disjoint from the 50 concept words, in `BASELINE_WORDS`
  (`src/introspection_scaling/extract.py`), the marked swap-in point should the
  verbatim list surface. The 50 concept words are verbatim from the paper. A
  `--baseline-file` flag now accepts a curated list. Concept vectors are diff-of-means
  over concept-vs-baseline contrasts, so the substitution does not bias toward a
  positive.
- **Extraction estimator, diff-of-means.** The paper describes "systematic
  diff-of-means" concept vectors. With our constant-positive contrast, an off-the-shelf
  centered-PCA extractor (`repeng`'s `pca_diff`) removes the concept signal and returns
  the top PC of the baseline activations. We use diff-of-means directly (split-half
  stability ~0.98, deterministic), the paper's stated method. Documented upstream:
  vgel/repeng#77.

---

## Superseded: original under-dosed run

Everything below reports the **original sub-threshold dose** (`α = 0.044·‖resid‖`,
steering-calibrated, ~4 to 18x below the paper). These numbers are **withdrawn** and
kept only for transparency. They are what produced the false "clean null through 32B"
and "failed reproduction" claims. Two bugs produced them: the under-dose, and a
same-day judge-API credit outage that could silently fail grades (now fixed by the
Bedrock fails-loud judge and persisted transcripts).

### Superseded ladder (under-dosed, Qwen2.5-Instruct)

| Model | Injected | No-inj ctrl | Rand-dir ctrl | GPU $ |
|-------|---------:|------------:|--------------:|------:|
| Qwen2.5-0.5B-Instruct | 0.00 | 0.00 | 0.00 | 0.39 |
| Qwen2.5-1.5B-Instruct | 0.00 | 0.00 | 0.00 | 0.44 |
| Qwen2.5-3B-Instruct | 0.00 | 0.00 | 0.00 | 0.55 |
| Qwen2.5-7B-Instruct | 0.00 | 0.00 | 0.00 | 0.47 |
| Qwen2.5-14B-Instruct | 0.00 | 0.00 | 0.00 | 0.79 |
| Qwen2.5-32B-Instruct | 0.00 | 0.00 | 0.00 | 1.12 |

Under-dosed method: depth 0.61, α = 0.044·‖resid‖ (steerbench dose-response sweep),
temperature 1, Anthropic API judge, seeds 0/1/2, one A100-80GB fp16, GPU $3.75. Raw
data retained at [`results/records.jsonl`](results/records.jsonl) and the old plot at
[`results/scaling_curve.png`](results/scaling_curve.png).

### Superseded calibration check (under-dosed Coder-32B)

At the under-dose, Qwen2.5-Coder-32B also returned 0/216, which is what made the
original result an honest failed reproduction. At the corrected dose it is above chance
(see [Finding](#finding)). The under-dosed Coder data is retained at
[`results/records_coder32b.jsonl`](results/records_coder32b.jsonl).

---

## Quantization caveat (70B-class rung, if run later)

A 70B-class rung would run 4-bit-quantized (bitsandbytes NF4) to fit budget. NF4
quantizes only the Linear weights; the residual stream stays fp16, so the injection
`h += α·v_unit` still applies. Interpret asymmetrically: a **positive** from 4-bit is
trustworthy (quantization can blur a signal, not manufacture one), a **null** is
suggestive only and needs fp16 confirmation. Smaller rungs run unquantized, so this
applies only to a 70B-class point.

### How to run on Modal

```bash
# Bedrock judge (fails loud): AWS creds via SSO or env; region us-east-1.
modal run modal_app.py::ladder_k2     # corrected-dose Instruct ladder
modal run modal_app.py::coder_k2      # corrected-dose Coder-32B rung
```

The judge raises on any parse error (never a silent zero), and every trial transcript
is persisted so a run can be re-graded offline without re-spending GPU.
