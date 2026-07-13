# introspection-scaling

> **Does the ability to introspect on injected concepts emerge with scale?**
> Reproduce concept-injection detection, then chart detection rate vs
> parameter count across two model-size ladders.

![Detection rate vs parameter count — Qwen2.5-Instruct 0.5–32B. Injected-concept
detection and both controls (no-injection, random-matched) sit at 0 across the
whole ladder: no concept-injection introspection emerges at or below 32B.](results/scaling_curve.png)

**First result:** across Qwen2.5-Instruct 0.5→32B, injected-concept detection is
**0/216 at every rung — and so are both controls.** A clean null with clean
controls (the [positive control](scripts/positive_control.py) confirms the judge
*can* score a hit, so this is a real absence, not a dead instrument). See
[RESULTS.md](RESULTS.md). 72B and the Llama-3.x family are pending; details and
limitations there.

## The question
Large models can sometimes *detect* when a concept has been injected into
their own activations (shown at 30B+, replicated at 70B). Unanswered: **at
what scale does this appear, and how does it degrade as models shrink?**

## Quickstart
```bash
uv sync                        # pinned env from uv.lock
export ANTHROPIC_API_KEY=...    # the introspection judge
./reproduce.sh                 # clean env → per-model detection rates + scaling curve
```

Or drive the pipeline directly:
```python
from introspection_scaling import (
    extract_concept_vector, make_random_matched,
    RepengGenerator, AnthropicJudge, run_concept, aggregate,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
cv    = extract_concept_vector(MODEL, "oceans", device="cpu")  # repeng diff-of-means → ConceptVector
gen   = RepengGenerator(MODEL, device="cpu")                   # injects h += α·v_unit via ControlModel
judge = AnthropicJudge()                                       # scores the self-report (needs API key)

records = run_concept(                                         # inject @0.61 depth, α = 0.044·‖resid‖
    cv, generator=gen, judge=judge, seeds=(0, 1, 2),
    random_matched_fn=make_random_matched,                     # random-matched-norm control
)
for r in aggregate(records):                                   # detection rate per condition
    print(r.condition, r.successes, "/", r.n)
```

## Architecture
```mermaid
flowchart LR
  W["50 concept words<br/>vs baseline"] --> E["repeng<br/>diff-of-means"] --> CV["ConceptVector<br/>unit-L2 per layer"]
  CV --> INJ["inject @0.61 depth<br/>α = 0.044·‖resid‖"]
  P["introspection prompt<br/>native chat template"] --> INJ
  INJ --> GEN["generate<br/>Qwen2.5 + Llama-3.x ladders"] --> J["Anthropic judge<br/>coherent AND correct-ID"] --> R["detection rate<br/>vs parameter count"]
  CTRL["controls:<br/>no-injection ·<br/>random-matched norm"] --> INJ
```
Both controls run the **identical** path — only the injected vector differs
(real concept · random of matched norm · nothing). Detection = the real concept
scoring above *both* controls.

## How it works (plain language)

**1. A concept is a direction.** As a model reads text, every layer keeps its
running "thoughts" as a big list of numbers — the *residual stream*, a conveyor
belt each layer adds to. A concept like *ocean* or *formality* shows up as a
**direction** in that space. We use [repeng](https://github.com/vgel/repeng) to
extract that direction (we do not reimplement it).

**2. The nudge.** We **add** that direction into the model's live internal state
mid-generation — plain vector addition — so it leans toward the concept without
us ever typing the word:

```
current thoughts  +  α · (ocean direction)  →  nudged thoughts
```

Two dials: **where** (which layer — depth) and **how much** (`α` — strength).
Turn `α` too high and the output degrades into nonsense — the *coherence cliff*.

**3. Introspection.** After nudging, we **ask** the model: *"Do you detect an
injected thought, and what is it about?"* If it correctly flags *and* names the
concept — reading its own internal state, not its own output — that is a
primitive form of introspection: the model reporting on its own internals.

**4. Controls (why this is science, not an artifact).** Every result carries two
controls beside it: **no-injection** (inject nothing, still ask) and
**random-direction of matched norm** (inject junk of the same size). Real
introspection means detecting the true concept **above both controls** — that
gap is the result, not the raw hit rate.

**5. The scaling question.** We chart detection rate vs parameter count across
the Qwen2.5 and Llama-3.x size ladders. If it climbs and crosses the controls at
some size, that is a **scaling threshold**. If it stays at noise up to 14B, that
is an **honest negative**. Both are findings; we publish whichever we get.

## Method (one paragraph)
Extract concept vectors via contrastive/PCA extraction (repeng, not
reimplemented), inject at a chosen layer/strength, run the introspection
prompt. **Controls are non-negotiable:** no-injection and random-direction
of matched norm, reported beside every result.

### Injection depth & strength
We normalize each per-layer direction to unit L2 and inject `h ← h + α·v_unit`.
**Strength is norm-relative:** `α = 0.044 · ‖resid‖`, where `‖resid‖` is the
residual-stream L2 norm measured at the injection block for *that* model — raw α
does not transfer across sizes (residual norm scales with architecture). We
target a fraction of ~0.044 and hard-cap it below 0.09 (a coherence cliff, where
over-steering degrades and can reverse the effect).
**Depth = 0.61 fraction-of-depth** (`layer = round(0.61·N)`), the default.
*Provisional* — the depth and dose defaults come from our companion steering-dose
study ([steerbench], a separate repo; see Methods), which reports a max-effect
layer near 0.61 (bracketing the paper's ~0.66) inside a usable band with a
dead-spot near 0.64. These numbers are **not yet reproduced in this repo** (our
[RESULTS](RESULTS.md) are tbd); we treat them as preliminary until the artifact
is linked. Depth stays a parameter; 0.5 and 0.71 are cheap sensitivity points on
0.5B so the choice isn't depth-cherry-picked.

**Models: instruct variants** (Qwen2.5-\*-Instruct, Llama-3.x-\*-Instruct). The
introspection prompt is a multi-turn *chat* self-report; base models don't follow
instructions, so a base "failure" confounds *can't introspect* with *can't follow
the prompt* — a fatal confound for a scaling claim. The paper used RLHF chat
models, so instruct is the faithful analog; our companion steerbench study also
reports instruct steering more cleanly than base (provisional, same caveat). We
render the prompt with each model's native chat template.

[steerbench]: # "companion steering-dose study — link when published"

## Rigor bar
3+ seeds · mean ± std on every point · pinned lockfile · fixed seeds ·
published hardware + wall-clock · negative results reported plainly.

## Status
Scaffold. See [RESULTS.md](RESULTS.md) and open issues.

## License
MIT.
