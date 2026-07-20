# STEP 3 — Kimi-K2 cost & feasibility estimate

> **Naming, first, to kill a grep collision.** This document is about **Kimi-K2**,
> Moonshot's ~1T-parameter mixture-of-experts model. It has **nothing to do with**
> the `_k2` suffix all over this repo (`trials_*_k2_bedrock.jsonl`,
> `CODER_K2_STRENGTH_K`, etc.), which means the **corrected dose `k = 2`**
> (`alpha = 2 * ||raw diff-of-means||`). Same two characters, unrelated things. Do
> not conflate them.

This is a **feasibility and order-of-magnitude estimate**, not a quote. Nothing
here has been run. The purpose is to decide whether scaling the concept-injection
introspection protocol to a ~1T-param MoE is worth authorizing — and the honest
answer is that the **dollar figure is not the gating input; the STEP-1
hook-attachment gate is** (see [Feasibility gate](#feasibility-gate-read-this-first)).

## The trial budget (unchanged from the dense rungs)

Every rung in this study is the same fixed budget, and K2 would be no different:

```
6 concepts  x  12 trials  x  3 seeds  x  3 conditions  =  648 generations / model
```

The three conditions are injected / no-injection control / random-direction control.
On top of the 648 generations sit the cheap parts: one hidden-state **extraction**
forward pass over the contrast set per concept (6 passes), and the **fit-check**
(magnitude-ratio + cosine) that confirms the injection is live. Extraction and
fit-check are a rounding error next to the 648 decodes.

## GPU / memory

K2 is ~1T parameters. Weights alone:

| Precision | Bytes/param | Weight memory | Fits on… |
|-----------|:-----------:|:-------------:|----------|
| bf16/fp16 | 2 | ~2.0 TB | **multi-node** (≥2× 8-GPU nodes) |
| fp8       | 1 | ~1.0 TB | one 8×H200 node **at the edge**, realistically 2 nodes |

One 8×H200 node is 8 × 141 GB = **1128 GB**. fp8 weights (~1.0 TB) technically fit
that number, but with **zero headroom** for KV cache, activations, expert-routing
buffers, and the ControlModel/extraction copies this protocol holds resident — so
plan on **two nodes (16×H200 ≈ 2.25 TB)** for fp8, and unambiguously multi-node for
bf16. For reference, the entire dense study ran on **one A100-80GB**; K2's weights
alone are **~13× a single A100's total VRAM at fp8, ~25× at bf16.** This is a
multi-GPU, near-certainly multi-node job. It is **not** a single-A100 rung.

## Wall-clock

**Anchor to a measured point, not a from-scratch throughput guess.** K2 is ~32B
*active* params, and this repo already has an empirical 648-generation rung at
32B-active: **Qwen2.5-Coder-32B ≈ 1 H100-hr** (RESULTS.md). Single-stream decode is
memory-bandwidth-bound by *active* params, so K2's raw generation work is the
**same order of magnitude** as that measured rung. What inflates K2 over it:

1. **Multi-node interconnect tax.** Tensor/expert-parallel across nodes adds an
   all-to-all every MoE layer over the inter-node fabric. Budget **~2–5×** the
   single-node-equivalent decode time.
2. **Weight load dominates end-to-end.** ~1 TB (fp8) of weights must be pulled and
   loaded before a single token decodes. At ~1 GB/s effective that is ~17 min; on a
   slower or cold cache, **1–2 hr** just to become ready. This, not inference,
   is the long pole.

Arithmetic: measured 32B-active decode work ≈ **~1 GPU-hr** of compute → × (2–5×
interconnect) ≈ **2–5 GPU-hr of compute**, wall-clocked down by multi-node
parallelism to **tens of minutes of actual decode**, then **added to 0.3–2 hr of
load/provisioning**. End-to-end per-run wall-clock:

> **~1–4 hours, most of it weight-load and node provisioning, not inference.**

(Theoretical cross-check: 648 gens × ~150 decode tokens ≈ ~100k decode tokens; even
at a conservative ~500 tok/s aggregate that is ~200 s of pure decode — confirming
compute is not the bottleneck. The estimate is dominated by setup, exactly as the
anchor implies.)

## Dollars

Multi-node H200 on-demand, assumptions stated:

- **Per-GPU H200 on-demand:** ~$3–4/hr (spot/discount clouds) to ~$8–11/hr
  (hyperscaler list). Take **~$3–11/GPU-hr**.
- **8×H200 node:** ~$24–88/hr. **Two nodes (16 GPUs):** ~$48–176/hr.
- **Billed wall-clock** (you pay for load + provisioning too): **1–4 hr**.

| Scenario | Node-hr rate | Wall-clock | GPU cost |
|----------|:------------:|:----------:|:--------:|
| Cheap (2 nodes, spot, fast load) | ~$48/hr | ~1 hr | **~$50** |
| Mid (2 nodes, mixed)             | ~$110/hr | ~2 hr | **~$220** |
| Expensive (2 nodes, list, slow load) | ~$176/hr | ~4 hr | **~$700** |

> **GPU range: ~$50–700, most-likely ~$150–400 per run.** Add the fixed Bedrock
> judge cost (~$2–5 for the 648 transcripts at the rung rate) — negligible next to
> GPU. Contrast with the whole dense study to date at **~$23 total**: a single K2
> run is **an order of magnitude more expensive than every dense rung combined.**

## Feasibility gate — read this first

**The dollars are not what decides this. A K2-specific STEP-1 hook-attachment gate
is, and it must pass before any full run is authorized.**

The injection is a `repeng.ControlModel` forward hook. `repeng`'s
`model_layer_list` (`control.py:204`) resolves the layer stack as **`model.model.layers`
— a mistral/Qwen-shaped assumption.** K2 ships **custom `trust_remote_code`
modeling** (DeepSeek-V3-style), which is **unverified** here. The dangerous failure
mode is not a loud crash — it is a **silent no-op**:

- `hasattr(model, "model")` is very likely **True** on K2's CausalLM wrapper
  (DeepSeek-V3 *does* name its decoder stack `model.model.layers`), so
  `model_layer_list` returns a list and `ControlModel` wraps each layer's `forward`
  **without error**.
- But an MoE decoder layer's `forward` can return **`(hidden_states, router_logits)`**
  (or a richer tuple) rather than a bare hidden-state tensor. `ControlModel`'s
  "add the concept vector to `output[0]`" assumption can then hit the wrong tensor,
  a routing artefact, or a copy that is discarded — **injecting nothing while
  raising no exception.** A silent no-op scores a clean, believable null.

**The gate is the fit-check this protocol already runs** (note.md §Method: *"log the
applied magnitude ratio and cosine to confirm the injection is live rather than a
no-op or a coherence-destroyer"*). Run **STEP 1 only** on K2:

1. Load K2, attach `ControlModel`, inject one concept at `k = 2`.
2. Read back the applied **magnitude ratio** and **cosine** against the intended
   direction. A live injection shows a non-trivial magnitude ratio and high cosine;
   a silent no-op shows **~0 magnitude change** — caught here, before spend.
3. **Recommended primary gate for an MoE: the router-shift metric**
   (`RepengGenerator.router_shift`). Magnitude ratio and cosine can look healthy even
   when the write lands on a routing artefact; `router_shift` measures whether the
   **top-k expert set actually changes** under injection, so it catches a silent no-op
   *on routing specifically* — the exact failure mode a custom MoE modeling file
   introduces. On the Qwen1.5-MoE probe this read **0.786** (the expert set flipped at
   79% of positions), the unambiguous "hook is live on the experts" signal; require a
   comparably non-trivial `router_shift` on K2 before authorizing spend.
4. Confirm the perturbation actually moves the residual (e.g. affirmative-rate lift
   under injection vs control on a handful of trials), so we know the hook writes to
   the real forward path and not a dead copy.

**Decision rule: no STEP-1 pass, no STEP-2/full run.** If the hook does not attach
cleanly to K2's expert-routing residual path, the ~$50–700 buys a null that means
*"our tool didn't attach"*, not *"K2 can't introspect"* — the worst possible outcome
for a study whose whole discipline is leading with honest negatives. The STEP-1 gate
is the real deliverable of this estimate.

## Summary (the numbers to decide on)

| Axis | Estimate |
|------|----------|
| **Memory** | ~1.0 TB weights (fp8) / ~2.0 TB (bf16); multi-GPU, near-certainly **multi-node** (2× 8×H200); ~13–25× a single A100. **Not** a single-A100 rung. |
| **Wall-clock** | **~1–4 hr** end-to-end per run, dominated by ~1 TB weight load + node provisioning, not the ~minutes of actual decode (anchored to the measured Coder-32B ≈ 1 H100-hr rung × 2–5× multi-node tax). |
| **Dollars** | **~$50–700 GPU** per run (most-likely ~$150–400) + ~$2–5 Bedrock; an order of magnitude over the entire ~$23 dense study. |
| **Gate** | **A K2-specific STEP-1 hook-attachment check MUST pass first.** `repeng` assumes `model.model.layers`; K2 is custom `trust_remote_code` MoE and can **silently no-op**. Primary gate is the **`RepengGenerator.router_shift`** metric (top-k expert-set change) — sharper than magnitude alone, which can look healthy on a routing artefact; the Qwen1.5-MoE probe read 0.786. No non-trivial router-shift, no authorized run. |
