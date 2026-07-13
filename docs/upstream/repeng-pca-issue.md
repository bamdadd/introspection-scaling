# `pca_diff` returns a concept-independent axis when the positive prompt is held constant across pairs

First, thanks for repeng — the `ControlVector` / `ControlModel` API made it easy
to build a faithful concept-injection reproduction, and being able to extract
per-layer directions for a whole model in a couple of lines is great.

We hit a subtle case we wanted to document, in case it helps others building
diff-of-means-style concept vectors. It is specific to one dataset shape; it is
**not** a claim about repeng's normal (varying-positive) usage.

## Summary

When every `DatasetEntry` shares the **same `positive` string** and only the
`negative` varies (a "concept vs. many baselines" contrast), `pca_diff` returns a
direction that is essentially independent of the concept — it tracks variation
among the *negative* examples instead. A plain diff-of-means on the same data
recovers a stable concept direction.

## When it happens

Our contrast holds the positive constant and varies the negative:

```python
DatasetEntry(positive="Tell me about Oceans.", negative="Tell me about table.")
DatasetEntry(positive="Tell me about Oceans.", negative="Tell me about window.")
...  # one pair per baseline word; positive never changes
```

This is a natural way to build a single-concept vector (the paper we reproduce,
Lindsey et al. 2025, describes "systematic diff-of-means" vectors of exactly this
shape).

## Why (mechanism)

`read_representations` with `method="pca_diff"` forms `train = h[::2] - h[1::2]`
(positives − negatives) and fits `PCA(n_components=1)` on it. Two things combine:

1. **The concept lives in the *mean* of the difference vectors.** With a constant
   positive, `mean(h_pos − h_neg_i) = h_pos − mean(h_neg)` — exactly the
   diff-of-means, i.e. the concept direction. But PCA centers its input before
   the SVD, so this mean is subtracted away.
2. **The constant positive contributes zero variance.** After centering, the
   difference vectors reduce to `−(h_neg_i − mean(h_neg))` — purely the variation
   among the baseline activations. So PC1 tracks *baseline-word* variation, which
   is unrelated to the concept.

The net effect: `pca_diff` ≈ `PCA1` of the negative examples alone.

## Minimal repro

Self-contained, deterministic, CPU-only, ~seconds on `Qwen/Qwen2.5-0.5B`
(full script attached as `repeng_pca_repro.py`). Metric is `|cos|` throughout, so
sign conventions don't affect the conclusion. Versions: repeng 0.4.0,
transformers 5.13.1, scikit-learn 1.9.0, torch 2.13.0, numpy 1.26.4, Python 3.13.

Measured output (reproduces exactly on rerun):

```
model=Qwen/Qwen2.5-0.5B  concept='Oceans'  #pairs=40  seed=0
positive prompt held CONSTANT: 'Tell me about Oceans.'

 layer  |cos(pca,dom)|  |cos(pca,negPC1)|   split(dom)   split(pca)
     4           0.257              1.000        0.964        0.684
     8           0.195              1.000        0.948        0.513
    12           0.146              1.000        0.952        0.800
    16           0.382              1.000        0.953        0.626
    20           0.076              1.000        0.944        0.309
```

- **`|cos(pca, dom)|` = 0.08–0.38**: the `pca_diff` direction is nearly orthogonal
  to the diff-of-means concept direction.
- **`|cos(pca, negPC1)|` = 1.000 at every layer**: the `pca_diff` direction *is*
  the top principal component of the negative examples — the mechanism above,
  measured directly rather than argued.

More baseline pairs don't change this: the mean is removed exactly regardless of
`n`, so adding negatives only refines the baseline covariance PC1 tracks.

## Split-half stability

Splitting the 40 pairs into two disjoint halves and extracting independently
(the `split(...)` columns above): diff-of-means directions agree across halves
(**0.94–0.96**), while `pca_diff` directions are much less stable
(**0.31–0.80**) — consistent with them capturing incidental baseline variation
rather than a stable concept axis.

## Secondary note (not central)

On these large-magnitude LM activations, sklearn selects the randomized SVD
solver and emits float-overflow `RuntimeWarning`s. In our runs the resulting
`pca_diff` direction happened to be stable run-to-run (one dominant singular
direction), so we're not claiming non-determinism — just flagging the warnings in
case they're unexpected. The three findings above are solver-independent.

## Proposed fix (either is fine)

1. **Add a `method="diff_of_means"` option** to `read_representations` /
   `ControlVector.train` — `unit(mean(h_pos) − mean(h_neg))` per layer. It's the
   right estimator for constant-positive / single-concept contrasts, is
   deterministic, and needs no SVD.
2. **Or a short doc note**: `pca_diff` assumes variation across *both* sides of
   the contrast; for a constant positive, prefer a diff-of-means vector.

Happy to open a PR for (1) with a test if that would be useful — just let us know
the API shape you'd prefer.
