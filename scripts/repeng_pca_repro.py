"""Minimal repro: repeng's pca_diff on a *constant-positive* contrast returns a
concept-independent axis (the top PC of the negative examples), not the
concept's diff-of-means.

Self-contained and deterministic. Depends only on repeng + transformers +
scikit-learn + numpy + torch. CPU is fine. Runs in ~seconds on Qwen2.5-0.5B.

    python scripts/repeng_pca_repro.py

What it demonstrates (all scoped to: the positive prompt is held CONSTANT across
the contrast pairs, e.g. "Tell me about Oceans." vs "Tell me about {baseline}."):

  1. |cos(pca_diff, diff-of-means)| is low  -> pca_diff misses the concept axis.
  2. |cos(pca_diff, PCA1(negatives))| ~ 1   -> pca_diff *is* the top PC of the
     negative (baseline) activations. This is the mechanism, not an analogy:
       - The concept lives in the MEAN of the difference vectors (diff-of-means);
         PCA centers before the SVD, so the mean is removed.
       - The constant positive contributes ZERO variance, so the only variance
         left in {h_pos - h_neg_i} is baseline-word variation. PC1 tracks that.
  3. Split-half stability: diff-of-means directions from two disjoint halves of
     the pairs agree; pca_diff directions do not.

Secondary / numerical note: on these large-magnitude LM activations sklearn
selects the randomized SVD solver and emits float-overflow RuntimeWarnings (on
stderr) from its matmuls. Run-to-run the pca_diff direction happens to be stable
here (one dominant singular direction), so this is NOT a determinism claim — it
is reported as measured. The three findings above are solver-independent.

The script prints its numbers; rerunning reproduces them (seeded).
"""

from __future__ import annotations

import numpy as np
import torch
from repeng import ControlVector, DatasetEntry
from repeng.extract import batched_get_hiddens
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 0
MODEL_ID = "Qwen/Qwen2.5-0.5B"
CONCEPT = "Oceans"
LAYERS = [4, 8, 12, 16, 20]
BATCH_SIZE = 16

# 40 common concrete nouns as the baseline (negative) words.
BASELINES = [
    "table",
    "window",
    "pencil",
    "clock",
    "button",
    "chair",
    "bottle",
    "ladder",
    "carpet",
    "curtain",
    "spoon",
    "kettle",
    "drawer",
    "candle",
    "basket",
    "pillow",
    "blanket",
    "napkin",
    "saucer",
    "shelf",
    "nail",
    "screw",
    "hammer",
    "wrench",
    "bucket",
    "shovel",
    "rope",
    "wire",
    "brick",
    "plank",
    "fence",
    "gate",
    "door",
    "roof",
    "wall",
    "floor",
    "engine",
    "wheel",
    "pedal",
    "lever",
]


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def abs_cos(a: np.ndarray, b: np.ndarray) -> float:
    # |cos|: directions are lines, and pca_diff's sign is a heuristic — comparing
    # magnitudes is the charitable metric and removes any sign-flip objection.
    return float(abs(np.dot(unit(a), unit(b))))


def dataset(baselines: list[str]) -> list[DatasetEntry]:
    return [
        DatasetEntry(positive=f"Tell me about {CONCEPT}.", negative=f"Tell me about {b}.")
        for b in baselines
    ]


def pca_diff_dirs(model, tokenizer, baselines: list[str]) -> dict[int, np.ndarray]:
    """repeng's own pca_diff output, at the API level (method='pca_diff' default)."""
    cv = ControlVector.train(
        model, tokenizer, dataset(baselines), hidden_layers=LAYERS, batch_size=BATCH_SIZE
    )
    return {layer: np.asarray(vec, dtype=np.float64) for layer, vec in cv.directions.items()}


def hiddens(model, tokenizer, baselines: list[str]) -> dict[int, np.ndarray]:
    """Last-token hidden states for [pos, neg, pos, neg, ...] via repeng's own pass."""
    strs = [s for e in dataset(baselines) for s in (e.positive, e.negative)]
    return batched_get_hiddens(model, tokenizer, strs, LAYERS, BATCH_SIZE)


def diff_of_means_dirs(h: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {L: (h[L][::2].mean(0) - h[L][1::2].mean(0)).astype(np.float64) for L in h}


def baseline_pc1(h: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Top PC of the negative (baseline) activations alone, per layer."""
    out = {}
    for L in h:
        negatives = h[L][1::2]  # the "Tell me about {baseline}." rows
        out[L] = PCA(n_components=1).fit(negatives).components_[0].astype(np.float64)
    return out


def detected_solver(h: dict[int, np.ndarray]) -> str:
    diffs = h[LAYERS[0]][::2] - h[LAYERS[0]][1::2]
    return str(PCA(n_components=1).fit(diffs)._fit_svd_solver)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    half1, half2 = BASELINES[: len(BASELINES) // 2], BASELINES[len(BASELINES) // 2 :]

    h_full = hiddens(model, tokenizer, BASELINES)
    dom = diff_of_means_dirs(h_full)
    neg_pc1 = baseline_pc1(h_full)
    pca = pca_diff_dirs(model, tokenizer, BASELINES)

    # split-half
    dom_h1 = diff_of_means_dirs(hiddens(model, tokenizer, half1))
    dom_h2 = diff_of_means_dirs(hiddens(model, tokenizer, half2))
    pca_h1 = pca_diff_dirs(model, tokenizer, half1)
    pca_h2 = pca_diff_dirs(model, tokenizer, half2)

    print(f"\nmodel={MODEL_ID}  concept={CONCEPT!r}  #pairs={len(BASELINES)}  seed={SEED}")
    print(f"positive prompt held CONSTANT: 'Tell me about {CONCEPT}.'\n")

    header = ("layer", "|cos(pca,dom)|", "|cos(pca,negPC1)|", "split(dom)", "split(pca)")
    print("{:>6} {:>15} {:>18} {:>12} {:>12}".format(*header))
    for L in LAYERS:
        c_dom = abs_cos(pca[L], dom[L])
        c_neg = abs_cos(pca[L], neg_pc1[L])
        s_dom = abs_cos(dom_h1[L], dom_h2[L])
        s_pca = abs_cos(pca_h1[L], pca_h2[L])
        print(f"{L:>6} {c_dom:>15.3f} {c_neg:>18.3f} {s_dom:>12.3f} {s_pca:>12.3f}")

    print(
        "\nLegend:"
        "\n  |cos(pca,dom)|     pca_diff vs diff-of-means         (low  -> misses concept)"
        "\n  |cos(pca,negPC1)|  pca_diff vs PCA1(negatives only)  (~1   -> IS baseline axis)"
        "\n  split(dom)         diff-of-means, disjoint halves    (high -> stable)"
        "\n  split(pca)         pca_diff, disjoint halves         (low  -> unstable)"
    )

    # Secondary numerical note (reported as measured, not a determinism claim).
    solver = detected_solver(h_full)
    pca_a = pca_diff_dirs(model, tokenizer, BASELINES)
    pca_b = pca_diff_dirs(model, tokenizer, BASELINES)
    dom_b = diff_of_means_dirs(hiddens(model, tokenizer, BASELINES))
    print(f"\nsklearn SVD solver selected: {solver!r} (float-overflow RuntimeWarnings on stderr)")
    print(
        "run-to-run |cos| (same data, back-to-back), layer "
        f"{LAYERS[2]}: pca_diff={abs_cos(pca_a[LAYERS[2]], pca_b[LAYERS[2]]):.3f}  "
        f"diff-of-means={abs_cos(dom[LAYERS[2]], dom_b[LAYERS[2]]):.3f}"
    )


if __name__ == "__main__":
    main()
