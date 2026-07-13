# introspection-scaling

> **Does the ability to introspect on injected concepts emerge with scale?**
> Reproduce concept-injection detection, then chart detection rate vs
> parameter count across two model-size ladders.

<!-- HERO FIGURE: results/scaling-curve.png — put it above the fold once you have it. -->

## The question
Large models can sometimes *detect* when a concept has been injected into
their own activations (shown at 30B+, replicated at 70B). Unanswered: **at
what scale does this appear, and how does it degrade as models shrink?**

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

## Reproduce
```bash
uv sync
./reproduce.sh          # clean env → results table
```

## Method (one paragraph)
Extract concept vectors via contrastive/PCA extraction (repeng, not
reimplemented), inject at a chosen layer/strength, run the introspection
prompt. **Controls are non-negotiable:** no-injection and random-direction
of matched norm, reported beside every result.

## Rigor bar
3+ seeds · mean ± std on every point · pinned lockfile · fixed seeds ·
published hardware + wall-clock · negative results reported plainly.

## Status
Scaffold. See [RESULTS.md](RESULTS.md) and open issues.

## License
MIT.
