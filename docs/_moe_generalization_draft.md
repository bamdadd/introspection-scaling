<!--
FINAL — filled with the real Qwen1.5-MoE-A2.7B-Chat run numbers.
This is the exact block slotted into docs/note.md (near Generalization/Limitations).
-->

## Generalization: a cross-architecture (MoE) probe

**State the confound before the result.** The cross-architecture probe runs the same
concept-injection protocol on a **mixture-of-experts** model, **Qwen1.5-MoE-A2.7B-Chat**
(**2.7B active / 14.3B total, 24 layers**), a general-instruct MoE. That model sits
**far below the ~32B-active scale** where the one above-chance cell appeared, **and**
it carries **no code-specific post-training**. On both axes that made the Coder-32B
cell light up (code-heavy post-training AND ~32B scale), this MoE is on the *null*
side. It is the architectural analog of our **dense general-instruct arm, which is null
at every size we tested.** So a null here is **predetermined by scale and
post-training, not by architecture** — and it is **not** a verdict that "MoE models
cannot introspect." Reading a null as an architecture failure would be exactly the
over-claim this study exists to avoid.

Given that, the probe is **not** run to see whether an MoE "passes." It has two
narrower, honest jobs.

**STEP 1 — feasibility: is the injection hook live on MoE expert routing?** The
injection is a `repeng.ControlModel` forward hook, and `repeng` assumes a
mistral/Qwen-shaped `model.model.layers` stack (`control.py:204`). An MoE decoder
layer routes through experts and its `forward` can return a `(hidden, router_logits)`
tuple, so the hook may attach to the wrong tensor and **silently no-op**. STEP 1 runs
the existing magnitude-ratio + cosine fit-check on the MoE — plus an MoE-specific
router-shift check — and requires a **live perturbation** before anything else counts.

**STEP 1 PASSED.** The layer stack resolved via `repeng`'s `model.model.layers` (the
mistral-shaped path attached cleanly on this MoE), and the injected direction is a
live, correctly-oriented perturbation that **demonstrably changes expert routing** —
the silent-no-op failure mode is ruled out and the run is feasibility-cleared.

> STEP-1 routing / hook-liveness (PASSED):
> - Injection **layer 15** (depth fraction 0.61 of 24 layers), stack resolved via
>   `repeng` `model.model.layers`
> - Dose **alpha = 8.893** = 2 × `raw_norm` (4.447) — the corrected a-priori `k = 2`
> - Applied **magnitude ratio 1.061** (live — non-trivial; a silent no-op reads ~0)
> - **Cosine 0.885** to the intended direction
> - **Routing changed on 0.786** of MoE-layer/token positions — the top-k expert
>   *set* flips at **79%** of positions under injection, so the hook perturbs expert
>   *selection*, not a discarded copy
> - Expert-**gate L1 shift 0.312** over **1422** MoE token-positions
> - Hook writes to the post-expert residual: **pass** (the 0.786 routing change is the
>   direct proof — not a mistral-shaped mis-attach)

**STEP 2 — a same-scale dense-parity check.** With the hook proven live, STEP 2 asks
one comparative question: does the MoE sit **on the dense null curve at its own active
scale**? It does.

> STEP-2 detection (216 trials/condition × 3 seeds, corrected dose):
>
> | Model (MoE, 2.7B active / 14.3B total) | correct-id (x/216) [95% CI] | affirmative | coherent | above chance? |
> |----------------------------------------|:---------------------------:|:-----------:|:--------:|:-------------:|
> | Qwen1.5-MoE-A2.7B-Chat | 0.000 (0/216) [0.000, 0.000] | 0.005 (1/216) | 0.750 | **no** |
>
> Both controls (no-injection, random-direction) are **0.000** on correct-id, so only
> the injected column carries a number. Coherence by condition: no-injection **0.931**,
> injected **0.750**, random-direction **0.648**.

**Read the result carefully — do not over-read it in either direction.**

- **The headline is a dissociation between mechanism and behaviour.** The injection is
  *mechanistically live* on MoE expert routing (79% of positions re-route, magnitude
  ratio ~1.06) and yet produces **zero introspective detection** (0/216, not above
  chance). The machinery reaches the experts; the report does not follow. This is the
  **same null as the dense sub-32B rungs** — a small general-instruct MoE sits **on the
  dense null curve**, not off it.
- **The near-floor affirmative is a scale fact, not an architecture break.**
  Affirmative is 0.005 (1/216), essentially the floor. Do **not** read this as an MoE
  "failure to feel" the perturbation: the strong affirmative signature (a model
  affirming an injected thought ~45% of the time) was **scale-dependent in the dense
  models too** — 32B affirms ~45%, but 14B-Instruct affirms only 0.023 (5/216). At
  2.7B active, a near-floor affirmative is exactly what small *dense* models do, so it
  is attributable to **scale, not to MoE architecture.** We report the number and note
  it tracks small dense; the data cannot isolate an architecture-specific dissociation
  and we do not claim one.
- **This is a genuine null, not a broken-model null.** Coherence stays at **0.750**
  under injection (vs 0.931 no-injection), so the model is largely capable-but-silent —
  unlike Coder-7B, whose 0/216 was confounded by an injection-induced coherence
  collapse to 0.056. The MoE null is clean.

**What the probe can and cannot conclude.** The **positive, publishable finding is
STEP 1 as a methods result**: the residual-injection hook transfers cleanly to an
expert-routed architecture and demonstrably perturbs routing (79% expert-set change) —
the technique is not dense-only. The **behavioural** effect stays absent, exactly as
predetermined: introspective detection needed the conjunction of code-heavy
post-training AND ~32B scale, and this model has neither, so its null is **not an
architecture verdict**. The probe cannot speak to whether a *large-active,
code-post-trained* MoE (e.g. Kimi-K2 scale) would introspect — that is the
[K2 estimate](k2_estimate.md)'s question, gated on its own STEP-1, and left open.
