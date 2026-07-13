"""Tests for concept-vector extraction (A1)."""

from __future__ import annotations

import numpy as np
import pytest

from introspection_scaling.extract import (
    BASELINE_WORDS,
    CONCEPT_WORDS,
    ConceptVector,
    _unit,
    build_dataset,
    extract_concept_vector,
    make_random_matched,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"


# --- word lists ------------------------------------------------------------


def test_concept_word_count() -> None:
    assert len(CONCEPT_WORDS) == 50
    assert len(set(CONCEPT_WORDS)) == 50


def test_baseline_word_count() -> None:
    assert len(BASELINE_WORDS) == 100
    assert len(set(BASELINE_WORDS)) == 100


def test_concept_and_baseline_disjoint() -> None:
    concepts = {w.lower() for w in CONCEPT_WORDS}
    baselines = {w.lower() for w in BASELINE_WORDS}
    assert concepts.isdisjoint(baselines)


# --- dataset ---------------------------------------------------------------


def test_build_dataset_pairs_concept_against_every_baseline() -> None:
    ds = build_dataset("Oceans", BASELINE_WORDS)
    assert len(ds) == len(BASELINE_WORDS)
    assert ds[0].positive == "Tell me about Oceans."
    assert ds[0].negative == f"Tell me about {BASELINE_WORDS[0]}."
    # positive prompt is the concept for every pair
    assert {e.positive for e in ds} == {"Tell me about Oceans."}


# --- _unit helper ----------------------------------------------------------


def test_unit_normalizes() -> None:
    v = _unit(np.array([3.0, 4.0], dtype=np.float32))
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert v.dtype == np.float32


def test_unit_rejects_zero_vector() -> None:
    with pytest.raises(ValueError):
        _unit(np.zeros(4, dtype=np.float32))


# --- make_random_matched (pure numpy, no model) ----------------------------


def _synthetic_cv() -> ConceptVector:
    rng = np.random.default_rng(0)
    directions = {i: _unit(rng.standard_normal(8).astype(np.float32)) for i in range(4)}
    raw_norms = {i: float(i) + 1.5 for i in range(4)}
    return ConceptVector(
        concept="oceans", model_id=MODEL_ID, directions=directions, raw_norms=raw_norms
    )


def test_random_matched_directions_are_unit() -> None:
    rm = make_random_matched(_synthetic_cv(), seed=7)
    for vec in rm.directions.values():
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_random_matched_preserves_metadata_and_shapes() -> None:
    cv = _synthetic_cv()
    rm = make_random_matched(cv, seed=7)
    assert rm.concept == cv.concept
    assert rm.model_id == cv.model_id
    assert rm.raw_norms == cv.raw_norms  # matched-norm: copied unchanged
    assert rm.directions.keys() == cv.directions.keys()
    for layer, vec in rm.directions.items():
        assert vec.shape == cv.directions[layer].shape


def test_random_matched_is_deterministic_in_seed() -> None:
    cv = _synthetic_cv()
    a = make_random_matched(cv, seed=42)
    b = make_random_matched(cv, seed=42)
    for layer in cv.directions:
        assert np.array_equal(a.directions[layer], b.directions[layer])


def test_random_matched_differs_across_seeds_and_from_real() -> None:
    cv = _synthetic_cv()
    a = make_random_matched(cv, seed=1)
    b = make_random_matched(cv, seed=2)
    for layer in cv.directions:
        assert not np.array_equal(a.directions[layer], b.directions[layer])
        assert not np.array_equal(a.directions[layer], cv.directions[layer])


# --- end-to-end extraction on the dev model --------------------------------


@pytest.mark.slow
def test_extract_concept_vector_on_qwen() -> None:
    """Real ConceptVector on the 0.5B dev model (loads weights; ~seconds)."""
    baselines = BASELINE_WORDS[:6]  # subset for speed
    cv = extract_concept_vector(MODEL_ID, "Oceans", baseline_words=baselines)

    assert cv.concept == "Oceans"
    assert cv.model_id == MODEL_ID
    # every block extracted (Qwen2.5-0.5B has 24 blocks)
    assert set(cv.directions) == set(range(24))
    assert cv.directions.keys() == cv.raw_norms.keys()

    for vec in cv.directions.values():
        assert vec.shape == (896,)
        assert vec.dtype == np.float32
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    for norm in cv.raw_norms.values():
        assert norm >= 0.0
    assert any(n > 0.0 for n in cv.raw_norms.values())

    # control is norm-matched and lives here
    rm = make_random_matched(cv, seed=0)
    assert rm.raw_norms == cv.raw_norms
    assert set(rm.directions) == set(cv.directions)


@pytest.mark.slow
def test_extraction_is_deterministic() -> None:
    """diff-of-means has no randomized solver: identical inputs -> identical dirs."""
    a = extract_concept_vector(MODEL_ID, "Silver", baseline_words=BASELINE_WORDS[:20])
    b = extract_concept_vector(MODEL_ID, "Silver", baseline_words=BASELINE_WORDS[:20])
    assert all(np.array_equal(a.directions[layer], b.directions[layer]) for layer in a.directions)


@pytest.mark.slow
def test_direction_captures_concept_split_half_stability() -> None:
    """The direction must actually encode the concept, not baseline noise.

    Split the baselines in two, extract independently, and require the two
    directions to agree. A concept-blind estimator (e.g. centered PCA on a
    constant-positive contrast) fails this; diff-of-means scores ~0.98 here.
    Threshold 0.90 sits well below the observed min (~0.96 across concepts).
    """
    concepts = ["Oceans", "Sadness"]
    check_layers = [8, 16, 20]  # spans the default injection layer (16)
    for concept in concepts:
        first = extract_concept_vector(MODEL_ID, concept, baseline_words=BASELINE_WORDS[:50])
        second = extract_concept_vector(MODEL_ID, concept, baseline_words=BASELINE_WORDS[50:])
        for layer in check_layers:
            cos = float(first.directions[layer] @ second.directions[layer])
            assert cos > 0.90, f"{concept} layer {layer}: split-half cos {cos:.3f}"
