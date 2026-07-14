"""Tests for concept-vector extraction (A1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import introspection_scaling.extract as extract_mod
from introspection_scaling.extract import (
    BASELINE_WORDS,
    CONCEPT_WORDS,
    ConceptVector,
    _unit,
    build_dataset,
    extract_concept_vector,
    load_baseline_words,
    load_extraction_model,
    make_random_matched,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


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


def test_load_baseline_words_strips_comments_blanks_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "baseline.txt"
    path.write_text("# curated list\n ocean \n\ncloud # inline comment\nocean\nrain\n")

    assert load_baseline_words(path) == ("ocean", "cloud", "rain")


@pytest.mark.parametrize("contents", ["", "# comments only\n  # another comment\n"])
def test_load_baseline_words_rejects_files_without_words(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "baseline.txt"
    path.write_text(contents)

    with pytest.raises(ValueError, match=f"Baseline file contains no words: {path}"):
        load_baseline_words(path)


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
    # every block extracted (Qwen2.5-0.5B-Instruct has 24 blocks)
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


# --- dtype / quant loader (issue #10, shared contract dtype:str, quant:str|None) ---


class _FakeModel:
    """Stand-in for a HF model: records .to(); has a minimal .config."""

    def __init__(self) -> None:
        self.to_device: object = None
        self.config = type("cfg", (), {"num_hidden_layers": 4})()

    def to(self, device: object) -> _FakeModel:
        self.to_device = device
        return self


def _patch_from_pretrained(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake(model_id: str, **kwargs: object) -> _FakeModel:
        captured["model_id"] = model_id
        captured.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(extract_mod.AutoModelForCausalLM, "from_pretrained", fake)
    return captured


def test_load_extraction_model_rejects_bad_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        load_extraction_model(MODEL_ID, dtype="float8")


def test_load_extraction_model_rejects_bad_quant() -> None:
    with pytest.raises(ValueError, match="quant"):
        load_extraction_model(MODEL_ID, quant="int8")


def test_load_extraction_model_plain_threads_dtype_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_from_pretrained(monkeypatch)
    model = load_extraction_model(MODEL_ID, dtype="bfloat16", device="cpu")
    assert captured["torch_dtype"] is torch.bfloat16
    assert "quantization_config" not in captured
    assert isinstance(model, _FakeModel)
    assert model.to_device == torch.device("cpu")  # moved to device on the plain path


def test_load_extraction_model_nf4_builds_bnb_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_from_pretrained(monkeypatch)
    model = load_extraction_model(MODEL_ID, dtype="float16", quant="nf4")
    cfg = captured["quantization_config"]
    assert isinstance(cfg, extract_mod.BitsAndBytesConfig)
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_compute_dtype is torch.float16
    assert captured["device_map"] == "auto"  # accelerate places it; no .to()
    assert isinstance(model, _FakeModel)
    assert model.to_device is None


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 forward needs a GPU")
def test_extract_float16_on_gpu() -> None:
    cv = extract_concept_vector(
        MODEL_ID, "Oceans", baseline_words=BASELINE_WORDS[:8], device="cuda", dtype="float16"
    )
    for vec in cv.directions.values():
        assert vec.dtype == np.float32  # float32 accumulation regardless of fp16 weights
        assert np.isfinite(vec).all()
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)
