# introspection-scaling

> **Does the ability to introspect on injected concepts emerge with scale?**
> Reproduce concept-injection detection, then chart detection rate vs
> parameter count across two model-size ladders.

<!-- HERO FIGURE: results/scaling-curve.png — put it above the fold once you have it. -->

## The question
Large models can sometimes *detect* when a concept has been injected into
their own activations (shown at 30B+, replicated at 70B). Unanswered: **at
what scale does this appear, and how does it degrade as models shrink?**

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
