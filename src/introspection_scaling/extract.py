"""Concept-vector extraction (A1).

Thin wrapper around `repeng` that builds the paper's systematic diff-of-means
contrast dataset, trains per-layer directions, and packages them as a
:class:`ConceptVector` for the introspection harness (A2) to inject.

Extraction is NOT reimplemented here: the injected *direction* always comes from
``repeng.ControlVector.train`` (method ``"pca_diff"``: PCA of the per-pair
activation differences, which is unit-norm and, for strong single-concept
contrasts, ~parallel to the normalized diff-of-means). We additionally compute
``raw_norms`` — the L2 norm of the raw diff-of-means per layer — from a single
`repeng` hidden-state pass. That scalar is reported/auxiliary; it does NOT drive
injection magnitude (injection is fixed at ``h <- h + alpha * v_unit``, so the
injected norm is ``alpha``). It is retained per the interface contract.

Layer-index convention (A1 owns this; A2 MUST agree)
----------------------------------------------------
`repeng` keys each direction by ``L`` and reads ``hidden_states[L + 1]``, i.e.
the *output of transformer block L* (``hidden_states[0]`` is the embedding
output). We adopt those keys verbatim: ``ConceptVector.directions[i]`` is the
unit direction at the output of block ``i``, 0-based. These are repeng-native
block-output indices — A2 passes them straight into ``ControlModel`` with **no
offset**. Passing ``hidden_layers=range(num_hidden_layers)`` extracts every
block (0..N-1); repeng's default would skip block 0 and yield only N-1 layers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import torch
from repeng import ControlModel, ControlVector, DatasetEntry  # type: ignore[import-untyped]
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
# NOTE: this is a *documented substitute*, not the paper's verbatim 100-word
# baseline appendix (not recovered verbatim at build time — flagged to orch-1).
# It is a fixed set of common, concrete, high-frequency nouns disjoint from
# CONCEPT_WORDS, so each concept is contrasted against a broad, neutral bag of
# unrelated words. Swap in the verbatim list here if/when recovered; the
# interface is unaffected.
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
        model_id: HF id the vector was extracted on, e.g. ``"Qwen/Qwen2.5-0.5B"``.
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

    Passes ``hidden_layers`` through to repeng (default: every block,
    ``range(num_hidden_layers)``). ``model``/``tokenizer`` may be supplied to
    reuse a loaded model across concepts; otherwise they are loaded on ``device``
    in float32.

    Does two forward passes over the contrast set: one inside
    ``ControlVector.train`` (for the directions) and one via
    ``batched_get_hiddens`` (for ``raw_norms``). Fine for dev-scale models;
    flagged to orch-1/A3 as an optimization target for the large ladder.
    """
    owns_model = model is None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        # transformers' typed `.to` overload confuses mypy-strict; runtime is fine.
        model = model.to(torch.device(device))  # type: ignore[arg-type]

    try:
        n_blocks = int(model.config.num_hidden_layers)
        if hidden_layers is None:
            hidden_layers = list(range(n_blocks))
        control_model = ControlModel(model, list(hidden_layers))

        dataset = build_dataset(concept, baseline_words)

        # Direction: repeng PCA-of-differences (unit-norm). Do NOT reimplement.
        cv = ControlVector.train(
            control_model,
            tokenizer,
            dataset,
            hidden_layers=hidden_layers,
            batch_size=batch_size,
        )
        directions = {layer: _unit(vec.astype(np.float32)) for layer, vec in cv.directions.items()}

        # raw_norms: L2 norm of the raw diff-of-means, same repeng hidden pass.
        raw_norms = _raw_diff_of_means_norms(
            control_model, tokenizer, dataset, list(hidden_layers), batch_size
        )
    finally:
        if owns_model:
            # Restore the underlying model in case the caller reuses it.
            control_model.reset()

    return ConceptVector(
        concept=concept, model_id=model_id, directions=directions, raw_norms=raw_norms
    )


def _raw_diff_of_means_norms(
    model: ControlModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[DatasetEntry],
    hidden_layers: list[int],
    batch_size: int,
) -> dict[int, float]:
    """``||mean(h_pos) - mean(h_neg)||`` per layer, via repeng's hidden pass.

    Uses the same last-token, same layer keys as ``ControlVector.train`` so the
    norms line up with the directions. Order is [pos, neg, pos, neg, ...].
    """
    strs = [s for entry in dataset for s in (entry.positive, entry.negative)]
    hiddens = batched_get_hiddens(model, tokenizer, strs, hidden_layers, batch_size)
    norms: dict[int, float] = {}
    for layer, h in hiddens.items():
        diff_of_means = h[::2].mean(axis=0) - h[1::2].mean(axis=0)
        norms[layer] = float(np.linalg.norm(diff_of_means))
    return norms


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
