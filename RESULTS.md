# Results

> **Corrected-dose result (2026-07-14).** An earlier version of this file reported a
> clean null through 32B and a failed reproduction. Both used a **sub-threshold
> injection dose** (`α = 0.044·‖resid‖`, chosen for steering coherence, ~4 to 18x
> below the paper's absolute strength). At the corrected dose
> (`α = 2·‖raw diff-of-means‖`, the paper's regime) with a fails-loud judge,
> **Qwen2.5-Coder-32B reproduces the effect** (2.3%, 5/216, above both controls,
> non-overlapping 95% CIs), while every Qwen2.5-Instruct rung stays null. The old
> under-dosed numbers are kept in the [Superseded](#superseded-original-under-dosed-run)
> appendix for transparency.

![Corrected-dose scaling curve: every Qwen2.5-Instruct rung is a flat null from 0.5B
to 32B, while Qwen2.5-Coder-32B sits above both controls at 32B. Detection tracks the
fine-tune, not parameter count.](results/scaling_curve_k2.png)

## Finding

**At the paper's dose, introspective detection tracks the fine-tune, not the
parameter count.** We went looking for a scaling threshold and found a
post-training effect instead.

Two results, and the second is the important one:

1. **The reproduction works.** Injecting at the paper's absolute strength
   (`α = 2·‖raw diff-of-means‖`) with a fails-loud judge, Qwen2.5-Coder-32B scores
   **0.023 (5/216)** injected-concept detection against **0.000** for both controls
   (no-injection and random-matched), with non-overlapping 95% CIs. Modest, but real
   and clean: the model sometimes notices and correctly names a concept that was
   never in its input.

2. **The scaling ladder is flat, and that is the finding.** Every Qwen2.5-Instruct
   rung from 0.5B to 32B stays at **0.000** correct-identification at the same
   corrected dose. The dose is live, not inert: at 32B the Instruct model *affirms*
   an injected thought 47% of the time under injection versus never without it. It
   feels the perturbation. It just never names it. So the interesting contrast is not
   on the ladder, it is between two same-size models:

   | Model (32B, same dose) | correct-id | affirmative | coherent | above chance? |
   |------------------------|:----------:|:-----------:|:--------:|:-------------:|
   | Qwen2.5-32B-Instruct   | 0.000      | 0.472       | 0.944    | no            |
   | Qwen2.5-Coder-32B      | **0.023**  | 0.310       | 0.770    | **yes**       |

   Same parameter count, same dose, same everything except post-training. The
   code-tuned model can report the injected concept; the chat-tuned one cannot.

There is also a method result worth stating on its own: open-model concept-injection
introspection is **dose-fragile**. A dose chosen for coherent steering sits well
below the detection regime and produces a null that passes every sanity check (clean
controls, working positive control, coherent transcripts). Only an effect-size
measurement, not the detection score, exposed it. Anyone reproducing this line of
work should calibrate the injection by the concept vector's own norm against the
source paper's strength, and measure the perturbation directly before trusting a zero.

## Corrected-dose ladder (Qwen2.5-Instruct, one A100-80GB, fp16)

Each cell is over **216 trials** (6 concepts x 12 trials x 3 seeds). Correct-id is the
strict score (`coherent AND correct-identification`). Affirmative and coherent are the
side signals that show the dose is doing something. Both controls (no-injection,
random-matched) are 0.000 correct-id on every rung.

| Model | Params (B) | correct-id [95% CI] | affirmative | coherent | above chance? |
|-------|-----------:|:-------------------:|:-----------:|:--------:|:-------------:|
| Qwen2.5-0.5B-Instruct | 0.5 | 0.000 [0.00, 0.00] | 0.130 | 0.005 | no |
| Qwen2.5-1.5B-Instruct | 1.5 | 0.000 [0.00, 0.00] | 0.023 | 0.167 | no |
| Qwen2.5-3B-Instruct | 3.0 | 0.000 [0.00, 0.00] | 0.176 | 0.861 | no |
| Qwen2.5-7B-Instruct | 7.0 | 0.000 [0.00, 0.00] | 0.306 | 0.597 | no |
| Qwen2.5-14B-Instruct | 14.0 | 0.000 [0.00, 0.00] | 0.023 | 0.894 | no |
| Qwen2.5-32B-Instruct | 32.0 | 0.000 [0.00, 0.00] | 0.472 | 0.944 | no |

Read across: coherence climbs with scale (0.5B is largely incoherent under the
paper-strength dose, 0.5% coherent, so its null is a coherence floor, not a detection
result), and by 32B the model is both coherent and frequently affirmative, yet still
never correctly identifies the concept. Correct-identification is the wall.

## Why the Coder and not the Instruct

We do not know for certain, and it is one model pair, so these are hypotheses.

The sharp clue is *where* the two models differ: not in noticing (Instruct-32B affirms
47% of the time) but in *naming*. That localizes the effect to the identification step
and narrows the candidates:

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
suppression. If it is not cleanly decodable, that is legibility. That is the next
experiment.

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
- **Seeds:** 0, 1, 2. **Hardware:** one Modal A100-80GB, fp16.
- **Spend:** ~$4.3 GPU (A100-80GB) + ~$16 Bedrock judge (about 4,500 grades) ≈ **$20
  total**. One judge batch hit a Bedrock throttle, failed loud, and was re-graded from
  persisted transcripts at lower concurrency with no GPU re-spend.
- **Raw data:** [`results/records_ladder_k2.jsonl`](results/records_ladder_k2.jsonl),
  [`results/records_coder32b_k2.jsonl`](results/records_coder32b_k2.jsonl), and the
  full judged transcripts in
  [`results/trials_ladder_k2_bedrock.jsonl`](results/trials_ladder_k2_bedrock.jsonl)
  and [`results/trials_coder32b_k2_bedrock.jsonl`](results/trials_coder32b_k2_bedrock.jsonl).

## Limitations (stated before you ask)

- **The dissociation is one model pair at one size.** It is an observation, not yet a
  claim. It needs more Instruct / Coder / Base pairs across sizes before it earns the
  word "finding" without a hedge.
- **The Coder-32B effect is modest** (2.3%) and rests on a single a-priori dose.
- **The logit-lens follow-up** that would explain the dissociation is not done yet.
- **72B and Llama-3.x** are not in the corrected ladder. The finding is about
  fine-tune, not scale, so extending scale is a low priority; a Base-model rung at
  each size would be more informative than a bigger one.
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
