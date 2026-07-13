"""Concept-vector extraction (A1).

Builds the paper's systematic **diff-of-means** contrast (concept vs baseline
words) and packages the per-layer directions as a :class:`ConceptVector` for the
introspection harness (A2) to inject.

Estimator — diff-of-means (the paper's method)
----------------------------------------------
For each layer, ``direction = unit(mean(h_positive) - mean(h_negative))`` and
``raw_norm = ||mean(h_positive) - mean(h_negative)||`` — computed from ONE
hidden-state pass over the contrast set. ``mean(pos) - mean(neg)`` already points
toward the concept, so no sign correction is needed.

We deliberately do NOT use ``repeng.ControlVector.train`` (method ``"pca_diff"``)
for the direction. Our dataset uses a *constant* positive prompt, so the per-pair
differences are ``h_concept - h_baseline_i`` whose mean is exactly the
diff-of-means; sklearn's PCA centers the data before the SVD, subtracting that
mean — i.e. it removes the concept signal and returns the top PC of baseline-word
variation, a ~concept-independent axis (measured cos with diff-of-means ~0.1-0.4,
and non-deterministic via randomized SVD). diff-of-means is the paper's stated
estimator, is deterministic, and needs one forward pass. We still use `repeng`
for the plumbing that matters for the seam: hidden-state extraction
(``batched_get_hiddens``) and the layer-index convention below. (Estimator swap
ruled in by orch-1; the :class:`ConceptVector` interface is unchanged.)

``raw_norms`` is reported/auxiliary — it does NOT drive injection magnitude
(injection is fixed at ``h <- h + alpha * v_unit``, so the injected norm is
``alpha``); it is retained per the interface contract and now matches the
direction it accompanies.

Layer-index convention (A1 owns this; A2 MUST agree)
----------------------------------------------------
`repeng`'s ``batched_get_hiddens`` keys layer ``L`` to ``hidden_states[L + 1]``,
i.e. the *output of transformer block L* (``hidden_states[0]`` is the embedding
output). We adopt those keys verbatim: ``ConceptVector.directions[i]`` is the
unit direction at the output of block ``i``, 0-based. These are repeng-native
block-output indices — A2 passes them straight into ``ControlModel`` with **no
offset**. Passing ``hidden_layers=range(num_hidden_layers)`` extracts every
block (0..N-1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import torch
from repeng import DatasetEntry  # type: ignore[import-untyped]
from repeng.extract import batched_get_hiddens  # type: ignore[import-untyped]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Per SPEC the seam type is ``np.ndarray``; we pin the dtype for mypy-strict.
Array = npt.NDArray[np.float32]

# ---------------------------------------------------------------------------
# Word lists (paper appendix).
# ---------------------------------------------------------------------------

# 50 concept words, verbatim from the task brief / paper appendix.
CONCEPT_WORDS: tuple[str, ...] = (
    "Dust",
    "Satellites",
    "Trumpets",
    "Origami",
    "Illusions",
    "Cameras",
    "Lightning",
    "Constellations",
    "Treasures",
    "Phones",
    "Trees",
    "Avalanches",
    "Mirrors",
    "Fountains",
    "Quarries",
    "Sadness",
    "Xylophones",
    "Secrecy",
    "Oceans",
    "Information",
    "Deserts",
    "Kaleidoscopes",
    "Sugar",
    "Vegetables",
    "Poetry",
    "Aquariums",
    "Bags",
    "Peace",
    "Caverns",
    "Memories",
    "Frosts",
    "Volcanoes",
    "Boulders",
    "Harmonies",
    "Masquerades",
    "Rubber",
    "Plastic",
    "Blood",
    "Amphitheaters",
    "Contraptions",
    "Youths",
    "Dynasties",
    "Snow",
    "Dirigibles",
    "Algorithms",
    "Denim",
    "Monoliths",
    "Milk",
    "Bread",
    "Silver",
)

# 100 baseline words.
#
# RECONSTRUCTED SUBSTITUTE (orch-1 ruling): this is NOT the paper's verbatim
# 100-word baseline appendix — that appendix was not released publicly. It is a
# fixed, documented set of 100 common concrete nouns, disjoint from
# CONCEPT_WORDS, so each concept is contrasted against a broad, neutral bag of
# unrelated words. This substitution is disclosed in RESULTS.md under "what was
# underspecified in the paper" — surfaced, not hidden.
#
# >>> SWAP-IN POINT <<< if the verbatim baseline list is ever recovered, replace
# the tuple below with it; nothing else changes (the interface is unaffected).
BASELINE_WORDS: tuple[str, ...] = (
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
    "mirror",
    "candle",
    "basket",
    "pillow",
    "blanket",
    "napkin",
    "saucer",
    "shelf",
    "hinge",
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
    "ceiling",
    "stair",
    "engine",
    "wheel",
    "pedal",
    "handle",
    "lever",
    "switch",
    "cable",
    "socket",
    "battery",
    "magnet",
    "lens",
    "prism",
    "ruler",
    "compass",
    "envelope",
    "stamp",
    "folder",
    "binder",
    "stapler",
    "eraser",
    "marker",
    "crayon",
    "notebook",
    "calendar",
    "receipt",
    "ticket",
    "coin",
    "wallet",
    "purse",
    "glove",
    "scarf",
    "jacket",
    "sweater",
    "sandal",
    "sneaker",
    "helmet",
    "goggles",
    "umbrella",
    "raincoat",
    "backpack",
    "suitcase",
    "trolley",
    "cart",
    "wagon",
    "sled",
    "canoe",
    "paddle",
    "anchor",
    "sail",
    "mast",
    "rudder",
    "propeller",
    "cockpit",
    "runway",
    "hangar",
    "turbine",
    "piston",
    "gasket",
    "valve",
    "nozzle",
)


# ---------------------------------------------------------------------------
# Interface: ConceptVector (shared with A2 — see SPEC.md).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptVector:
    """Per-layer concept directions for one (model, concept).

    Attributes:
        concept: the concept word used to build the positive prompt (the join
            key A2 aggregates on; use the strings in :data:`CONCEPT_WORDS`).
        model_id: HF id the vector was extracted on, e.g.
            ``"Qwen/Qwen2.5-0.5B-Instruct"``.
        directions: block-output index -> unit-L2 direction, shape ``(hidden,)``.
        raw_norms: block-output index -> ``||raw diff-of-means||`` (auxiliary;
            see module docstring). Copied unchanged by :func:`make_random_matched`.
    """

    concept: str
    model_id: str
    directions: dict[int, Array]
    raw_norms: dict[int, float]


# ---------------------------------------------------------------------------
# Extraction.
# ---------------------------------------------------------------------------


def build_dataset(concept: str, baseline_words: tuple[str, ...]) -> list[DatasetEntry]:
    """Paper's contrast dataset: concept vs each baseline word.

    positive = ``"Tell me about {concept}."``, negative =
    ``"Tell me about {baseline}."`` — one pair per baseline word.
    """
    return [
        DatasetEntry(
            positive=f"Tell me about {concept}.",
            negative=f"Tell me about {baseline}.",
        )
        for baseline in baseline_words
    ]


def extract_concept_vector(
    model_id: str,
    concept: str,
    *,
    baseline_words: tuple[str, ...] = BASELINE_WORDS,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    hidden_layers: list[int] | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> ConceptVector:
    """Extract a :class:`ConceptVector` for ``concept`` on ``model_id``.

    ``hidden_layers`` defaults to every block (``range(num_hidden_layers)``).
    ``model``/``tokenizer`` may be supplied to reuse a loaded model across
    concepts; otherwise they are loaded on ``device`` in float32.

    One forward pass over the contrast set (``batched_get_hiddens``); both the
    unit directions and ``raw_norms`` come from the same diff-of-means, so they
    are consistent and the result is deterministic.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        # transformers' typed `.to` overload confuses mypy-strict; runtime is fine.
        model = model.to(torch.device(device))  # type: ignore[arg-type]

    if hidden_layers is None:
        hidden_layers = list(range(int(model.config.num_hidden_layers)))

    dataset = build_dataset(concept, baseline_words)
    directions, raw_norms = _diff_of_means(
        model, tokenizer, dataset, list(hidden_layers), batch_size
    )
    return ConceptVector(
        concept=concept, model_id=model_id, directions=directions, raw_norms=raw_norms
    )


def _diff_of_means(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[DatasetEntry],
    hidden_layers: list[int],
    batch_size: int,
) -> tuple[dict[int, Array], dict[int, float]]:
    """Per-layer ``(unit direction, raw norm)`` from the diff-of-means.

    Uses repeng's ``batched_get_hiddens`` (last-token hidden states; layer ``L``
    -> output of block ``L``). Contrast order is [pos, neg, pos, neg, ...], so
    ``mean(pos) - mean(neg)`` points toward the concept — no sign flip needed.
    """
    strs = [s for entry in dataset for s in (entry.positive, entry.negative)]
    hiddens = batched_get_hiddens(model, tokenizer, strs, hidden_layers, batch_size)
    directions: dict[int, Array] = {}
    raw_norms: dict[int, float] = {}
    for layer, h in hiddens.items():
        diff_of_means = (h[::2].mean(axis=0) - h[1::2].mean(axis=0)).astype(np.float32)
        raw_norms[layer] = float(np.linalg.norm(diff_of_means))
        directions[layer] = _unit(diff_of_means)
    return directions, raw_norms


# ---------------------------------------------------------------------------
# Control: random-direction, matched-norm (owned by A1).
# ---------------------------------------------------------------------------


def make_random_matched(cv: ConceptVector, seed: int) -> ConceptVector:
    """Random-direction control: a random unit direction per layer.

    Same concept/model_id/raw_norms/keys/shapes as ``cv`` (A2 tracks the
    condition separately, so the join key ``concept`` is preserved). Directions
    are replaced with i.i.d. Gaussian unit vectors. Deterministic in ``seed``.
    At injection the harness unit-normalizes and scales by ``alpha`` — so the
    random and real injected vectors share the same norm (``alpha``); this is
    the matched-norm control.
    """
    rng = np.random.default_rng(seed)
    directions = {
        layer: _unit(rng.standard_normal(vec.shape).astype(np.float32))
        for layer, vec in cv.directions.items()
    }
    return replace(cv, directions=directions)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _unit(vec: Array) -> Array:
    """Unit-L2 normalize a 1-D vector (raises on a zero vector)."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("cannot unit-normalize a zero vector")
    return (vec / norm).astype(np.float32)
