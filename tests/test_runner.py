"""Runner orchestration, exercised with fakes (no models, no harness, no GPU).

Verifies the loop structure, the two-phase (extract-then-generate) ordering, and
that the orch-2 injection parameters are threaded to A2's run_concept — without
importing the (not-yet-merged) harness.
"""

from pathlib import Path

import numpy as np
import pytest

from introspection_scaling import runner as runner_mod
from introspection_scaling.extract import ConceptVector
from introspection_scaling.records import SeedRecord
from introspection_scaling.runner import run_ladder

_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
_CONCEPTS = ["Oceans", "Dust", "Snow"]


def _fake_cv(model_id: str, concept: str) -> ConceptVector:
    v = np.ones(4, dtype=np.float32)
    return ConceptVector(concept, model_id, {0: v / np.linalg.norm(v)}, {0: 1.0})


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.run_concept_calls: list[dict] = []
        self.loads: list[str] = []
        self.written: list = []

    def load_model(self, model_id: str, device: str):
        self.loads.append(model_id)
        self.events.append(f"load:{model_id}")
        return (f"model:{model_id}", "tok")

    def extract(self, model_id, concept, *, model, tokenizer, device):
        assert model == f"model:{model_id}"  # uses the Phase-1 loaded model
        return _fake_cv(model_id, concept)

    def make_generator(self, model_id: str):
        self.events.append(f"gen:{model_id}")
        return f"generator:{model_id}"

    def run_concept(self, cv, **kwargs):
        self.run_concept_calls.append({"cv": cv, **kwargs})
        return [f"trial:{cv.model_id}:{cv.concept}"]

    def write(self, trials, path):
        self.written = list(trials)
        return [
            SeedRecord("m", "Oceans", "injected", 0, 1, 1),
        ]


def _run(rec: _Recorder, path: Path) -> list[SeedRecord]:
    fake_judge = object()
    return run_ladder(
        _MODELS,
        concepts=_CONCEPTS,
        seeds=[0, 1, 2],
        n_trials=5,
        out_path=path,
        judge=fake_judge,  # skip harness judge resolution
        extract_fn=rec.extract,
        make_generator=rec.make_generator,
        run_concept_fn=rec.run_concept,
        write_seed_records_fn=rec.write,
        load_model_fn=rec.load_model,
    )


def test_run_concept_called_per_model_and_concept(tmp_path: Path) -> None:
    rec = _Recorder()
    _run(rec, tmp_path / "records.jsonl")
    assert len(rec.run_concept_calls) == len(_MODELS) * len(_CONCEPTS)
    assert rec.loads == _MODELS  # exactly one extraction load per model


def test_orch2_injection_params_threaded(tmp_path: Path) -> None:
    rec = _Recorder()
    _run(rec, tmp_path / "records.jsonl")
    for call in rec.run_concept_calls:
        assert call["depth_fraction"] == pytest.approx(0.61)
        assert call["dose_fraction"] == pytest.approx(0.044)
        assert call["seeds"] == [0, 1, 2]
        assert call["n_trials"] == 5
        # generator matches the model of the concept vector (right ControlModel)
        assert call["generator"] == f"generator:{call['cv'].model_id}"


def test_two_phase_extract_before_generate(tmp_path: Path) -> None:
    rec = _Recorder()
    _run(rec, tmp_path / "records.jsonl")
    # Per model: the extraction load must precede building that model's generator,
    # and the generator for model N must come before loading model N+1.
    assert rec.events == [
        "load:Qwen/Qwen2.5-0.5B-Instruct",
        "gen:Qwen/Qwen2.5-0.5B-Instruct",
        "load:Qwen/Qwen2.5-1.5B-Instruct",
        "gen:Qwen/Qwen2.5-1.5B-Instruct",
    ]


def test_write_receives_all_trials(tmp_path: Path) -> None:
    rec = _Recorder()
    out = _run(rec, tmp_path / "records.jsonl")
    assert len(rec.written) == len(_MODELS) * len(_CONCEPTS)  # one fake trial each
    assert out and isinstance(out[0], SeedRecord)


def test_cli_seam_threads_args(monkeypatch, tmp_path: Path) -> None:
    """`runner.main` argparse -> run_ladder: flags bind, defaults (depth 0.61) hold."""
    captured: dict = {}

    def fake_run_ladder(models, **kwargs):
        captured["models"] = list(models)
        captured.update(kwargs)
        return [SeedRecord("m", "Oceans", "injected", 0, 1, 1)]

    monkeypatch.setattr(runner_mod, "run_ladder", fake_run_ladder)
    out = tmp_path / "records.jsonl"
    rc = runner_mod.main(
        [
            "--models",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "--seeds",
            "0",
            "1",
            "2",
            "--n-concepts",
            "3",
            "--n-trials",
            "7",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert captured["models"] == ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
    assert captured["seeds"] == [0, 1, 2]
    assert captured["n_trials"] == 7
    assert len(captured["concepts"]) == 3  # first 3 concept words
    assert captured["out_path"] == out
    assert captured["depth_fraction"] == pytest.approx(0.61)
    assert captured["dose_fraction"] == pytest.approx(0.044)
