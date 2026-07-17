# Concept-Injection Introspection in Open Models Is Dose-Fragile and Fine-Tune-Dependent, Not Scale-Dependent

**Bamdad Dashtban**

---

## Abstract

We reproduce the concept-injection introspection protocol of Lindsey et al. (2025, "Emergent Introspective Awareness in Large Language Models") on open Qwen2.5 models. Two findings result. First, the effect is dose-fragile: an injection strength calibrated for coherent activation steering sits roughly 4 to 18 times below the paper's absolute strength, and at that under-dose every model returns a clean null that passes every sanity check (flat controls, a working positive control, coherent transcripts). Correcting the dose to the paper's regime and adding a fails-loud judge is what surfaces any signal at all. Second, at the corrected dose the ability to notice and correctly name an injected concept at 32B tracks the fine-tune, not the parameter count. In a controlled three-way at fixed 32B and fixed dose, Qwen2.5-32B base scores 0/216 correct-identification and Qwen2.5-32B-Instruct scores 0/216, while Qwen2.5-Coder-32B scores 2.3% (5/216), above both a no-injection and a random-direction control with non-overlapping 95% CIs. The base control rules out both parameter count (all three are 32B) and fine-tuning in general (Instruct is a fine-tune and is null). A logit-lens localizes the mechanism: the injected concept is linearly decodable at the unembedding in Coder but not in base or Instruct, so the split is a legibility difference introduced by code-heavy post-training, not suppression. The effect is small and rests on a single a-priori dose, and we report it with that caveat.

## 1. Introduction

Lindsey et al. inject a known concept vector into a model's residual stream and ask whether the model can report that an injected thought is present and name it. On frontier closed models the effect is real but unreliable. We ask a scaling question on open models: does this ability emerge with parameter count. The answer we reach is that at 32B it does not depend on scale at all; it depends on how the model was post-trained. Getting to that answer required first noticing that our own initial null was an artifact of dose calibration, which is a result in its own right for anyone trying to replicate this line of work.

Contributions: (1) a controllable, deterministic open-model reproduction with a fails-loud judge; (2) the dose-fragility result, with the exact under-dose that fakes a clean null; (3) a base/instruct/coder three-way at fixed 32B that isolates code post-training as the differentiator; (4) a logit-lens that identifies the mechanism as representational legibility.

## 2. Method

**Concept vectors.** Diff-of-means over concept-versus-baseline prompts (the paper's stated estimator), taken as a unit direction with its raw norm retained.

**Injection.** A forward hook adds the concept direction to the residual stream at depth 0.61, at strength `alpha = 2 * ||raw diff-of-means||` (the paper's canonical strength of 2). This is a single a-priori dose; we run no strength or layer sweep, so a positive cannot be an artifact of tuning to it. We log the applied magnitude ratio and cosine to confirm the injection is live rather than a no-op or a coherence-destroyer.

**Judge.** Detection is scored `coherent AND correct-identification` by Claude Sonnet 4 via AWS Bedrock, configured to raise on any parse error so a degraded or unavailable judge can never silently return a zero. Every transcript is persisted for offline re-judging.

**Controls.** Every point carries a no-injection control and a random-direction control matched to the concept vector's norm; detection counts only when the injected condition clears both. A positive control grades four canned responses through the real judge to prove it can emit a success and withholds it otherwise.

## 3. Results

**Corrected-dose Instruct ladder (Qwen2.5-Instruct, 216 trials per condition, seeds 0/1/2, one A100-80GB, fp16).** Correct-identification is 0.000 at every rung from 0.5B to 32B, with both controls flat at 0.000. The dose is live, not inert: at 32B the model affirms an injected thought 47% of the time under injection versus never without it, and coherence climbs with scale (0.5% coherent at 0.5B, 94% at 32B). The models feel the perturbation; they do not name it.

**The 32B three-way (fixed size, fixed dose).**

| Model (32B) | correct-id [95% CI] | affirmative | coherent | above chance |
|-------------|:-------------------:|:-----------:|:--------:|:------------:|
| Qwen2.5-32B (base)   | 0.000 [0.000, 0.000] | 0.449 | 0.491 | no  |
| Qwen2.5-32B-Instruct | 0.000 [0.000, 0.000] | 0.472 | 0.944 | no  |
| Qwen2.5-Coder-32B    | 0.023 [0.014, 0.028] | 0.310 | 0.770 | yes |

Base and Instruct sit at the floor and behave alike (they affirm around 45 to 47% and never correctly identify). Only the code-tuned model lifts off. The base rung is the control that matters: it removes parameter count and fine-tuning in general as explanations and leaves code-heavy post-training as the differentiator.

**Mechanism (logit-lens).** Projecting the injected residual through the model's own unembedding, injection sharply raises the injected concept token in Coder-32B (median rank about 30k to 4k, a sustained lift of roughly 2 to 2.5 over no injection, several concepts reaching the top few). In Instruct-32B and base the concept stays illegible: rank no better than no injection, worse than a matched-random direction. The dissociation is a legibility difference from post-training, not a persona gate, which matches the behavioural signature of affirming a thought while naming the wrong one.

## 4. The dose-fragility result

Our first pass reported a clean null on every model, including Coder-32B where the effect is claimed. It was wrong for two compounding reasons. The injection dose was inherited from a companion steering study and tuned for coherent output, which put it roughly 4 to 18 times below the paper's absolute strength; the effect-size measurement, not the detection score, is what exposed this. Separately, a same-day judge-API credit outage silently turned grades into false negatives. Correcting the dose to the paper's regime and moving to a fails-loud judge is what surfaced the real signal. We keep the superseded numbers in the repository, marked, because the wrong turn is half the story: a steering-calibrated dose plus a quiet judge will hand you a null you will believe.

## 5. Limitations

The Coder-32B effect is 2.3%, modest, and rests on a single a-priori dose with no sweep. The base/instruct/coder three-way is at one size so far; 7B and 14B triples would turn it into a trend. We identify the mechanism as legibility but not its cause: we do not yet know which layers or features code post-training changes, or whether the driver is code specifically or a correlate of it. Everything is fp16 on a single A100, dense Qwen only; a cross-architecture check (a mixture-of-experts model) is the obvious generalization test.

## 6. Reproducibility

Deterministic, fixed seeds (0/1/2), one Modal A100-80GB, fp16, corrected dose `alpha = 2 * ||raw diff-of-means||`, Bedrock Sonnet 4 judge that fails loud. Total spend about 20 USD in GPU and judge calls (the base rung was 0.78 USD of GPU). Code, raw transcripts, and the exact dose calibration are in the repository; a companion write-up is at https://bamdad.substack.com/p/same-size-different-mind.

## Related work

Lindsey et al., "Emergent Introspective Awareness in Large Language Models" (Anthropic, 2025; arXiv:2601.01828), the protocol we reproduce. Prior open-model replications on Qwen and Llama report the effect on code-tuned checkpoints; our base control and logit-lens give a mechanism for why those checkpoints and not their instruct siblings.
