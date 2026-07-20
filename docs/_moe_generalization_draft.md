<!--
DRAFT for review — a candidate `## Generalization` block for docs/note.md.
Do NOT slot into note.md until approved. Placeholders marked [FILL: …] are for the
run to fill; no numbers are invented here.
-->

## Generalization: a cross-architecture (MoE) probe

**State the confound before the result.** The cross-architecture probe runs the same
concept-injection protocol on a **mixture-of-experts** model — a ~3B-*active*
general-instruct MoE (target TBD, e.g. Qwen3-30B-A3B-Instruct — confirm in STEP 1).
That model sits **far below the ~32B-active scale** where the one above-chance cell
appeared, **and** it carries **no code-specific post-training**. On both axes that
made the Coder-32B cell light up (code-heavy post-training AND ~32B scale), this MoE
is on the *null* side. It is the architectural analog of our **dense general-instruct
arm, which is null at every size we tested.** So a null here is **predetermined by
scale and post-training, not by architecture** — and it is **not** a verdict that
"MoE models cannot introspect." Reading a null as an architecture failure would be
exactly the over-claim this study exists to avoid.

Given that, the probe is **not** run to see whether an MoE "passes." It has two
narrower, honest jobs:

**STEP 1 — feasibility: is the injection hook live on MoE expert routing?** The
injection is a `repeng.ControlModel` forward hook, and `repeng` assumes a
mistral/Qwen-shaped `model.model.layers` stack (`control.py:204`). An MoE decoder
layer routes through experts and its `forward` can return a `(hidden, router_logits)`
tuple, so the hook may attach to the wrong tensor and **silently no-op**. STEP 1 runs
the existing magnitude-ratio + cosine fit-check on the MoE and requires a **live
perturbation** — a non-trivial applied magnitude and high cosine to the intended
direction — before anything else counts. This is the same gate the [K2 estimate](k2_estimate.md)
demands, run on a model small enough to check cheaply first.

> STEP-1 routing / hook-liveness (to fill from the run):
> - Layer stack resolved by `model_layer_list`: [FILL: attached path, e.g. `model.model.layers`, N layers]
> - Injection depth (fraction 0.61): [FILL: block index]
> - Applied **magnitude ratio**: [FILL] (live if non-trivial; ~0 = silent no-op)
> - **Cosine** to intended direction: [FILL]
> - Hook writes to the post-expert residual, not a discarded copy: [FILL: pass/fail]

**STEP 2 — a same-scale dense-PARITY check.** With the hook proven live, STEP 2 asks
one comparative question: does the MoE behave **like the dense models at comparable
active scale** — feel the perturbation but fail to name it? Concretely, does it show
the **affirmative-up-under-injection** signature (the model reacts to the injected
thought) while returning the **correct-identification null** (it cannot name the
thought), the same dissociation the dense 7B/14B rungs show? If yes, the MoE is
**parity** with dense of its size, and architecture buys nothing extra at this scale —
which is the expected, honest outcome. STEP 2 is a parity check against the dense
null, **not** a search for a positive.

> STEP-2 detection (to fill from the run, same columns as the dense grid):
>
> | Model (MoE, ~3B active) | correct-id (x/216) [95% CI] | affirmative | coherent | above chance? |
> |-------------------------|:---------------------------:|:-----------:|:--------:|:-------------:|
> | [FILL: model id] | [FILL] | [FILL] | [FILL] | [FILL] |
>
> - Both controls (no-injection, random-matched) flat at 0.000: [FILL: yes/no]
> - Dense-parity read: affirmative up under injection AND correct-id null,
>   like dense 7B/14B — [FILL: holds / does not hold]

**What the probe can and cannot conclude.** It can show (1) the injection machinery
transfers to an expert-routed architecture at all, and (2) whether an MoE at
small-active-scale sits on the dense null curve. It **cannot** speak to whether a
*large-active, code-post-trained* MoE (e.g. Kimi-K2 scale) would introspect — that is
the [K2 estimate](k2_estimate.md)'s question, gated on its own STEP-1, and left open.
The predetermined null here narrows the confound; it does not test the architecture at
the scale where the effect actually lives.
