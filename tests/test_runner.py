"""Runner orchestration, exercised with fakes (no models, no GPU, no real money).

Covers the loop structure, two-phase ordering, orch-2 param threading, per-model
precision threading, and the money cost-guard self-stop (with a fake clock).
"""

from pathlib import Path

import numpy as np
import pytest

from introspection_scaling import runner as runner_mod
from introspection_scaling.extract import ConceptVector
from introspection_scaling.records import SeedRecord
from introspection_scaling.runner import LadderRun, run_ladder

_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
_CONCEPTS = ["Oceans", "Dust", "Snow"]


def _fake_cv(model_id: str, concept: str) -> ConceptVector:
    v = np.ones(4, dtype=np.float32)
    return ConceptVector(concept, model_id, {0: v / np.linalg.norm(v)}, {0: 1.0})


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.gen_completion_calls: list[dict] = []
        self.judge_calls: list[list] = []
        self.loads: list[tuple] = []
        self.gen_calls: list[tuple] = []
        self.written: list = []
        self.commits = 0

    def load_model(self, model_id, device, dtype="float32", quant=None):
        self.loads.append((model_id, dtype, quant))
        self.events.append(f"load:{model_id}")
        return (f"model:{model_id}", "tok")

    def extract(self, model_id, concept, *, model, tokenizer, device):
        assert model == f"model:{model_id}"  # uses the Phase-1 loaded model
        return _fake_cv(model_id, concept)

    def make_generator(self, model_id, dtype, quant):
        self.gen_calls.append((model_id, dtype, quant))
        self.events.append(f"gen:{model_id}")
        return f"generator:{model_id}"

    def gen_completions(self, cv, **kwargs):
        # GPU phase: return raw completions (strings stand in for Completion objs)
        self.gen_completion_calls.append({"cv": cv, **kwargs})
        self.events.append(f"gencompl:{cv.model_id}")
        return [f"compl:{cv.model_id}:{cv.concept}"]

    def judge(self, completions, **kwargs):
        # off-GPU phase: runs AFTER the generator is freed
        self.judge_calls.append(list(completions))
        self.events.append("judge")
        return [f"trial:{c.split(':')[1]}:{c.split(':')[2]}" for c in completions]

    def write(self, trials, path):
        self.written = list(trials)
        return [
            SeedRecord(m, "Oceans", "injected", 0, 1, 1) for m in {t.split(":")[1] for t in trials}
        ]

    def commit(self) -> None:
        self.commits += 1


def _run(rec: _Recorder, path: Path, **kw) -> LadderRun:
    return run_ladder(
        _MODELS,
        concepts=_CONCEPTS,
        seeds=[0, 1, 2],
        n_trials=5,
        out_path=path,
        judge=object(),  # skip harness judge resolution
        extract_fn=rec.extract,
        make_generator=rec.make_generator,
        generate_completions_fn=rec.gen_completions,
        judge_completions_fn=rec.judge,
        write_seed_records_fn=rec.write,
        load_model_fn=rec.load_model,
        on_model_done=rec.commit,
        **kw,
    )


def test_generate_called_per_model_and_concept(tmp_path: Path) -> None:
    rec = _Recorder()
    res = _run(rec, tmp_path / "records.jsonl")
    assert len(rec.gen_completion_calls) == len(_MODELS) * len(_CONCEPTS)
    assert len(rec.judge_calls) == len(_MODELS)  # judged once per rung, post-generation
    assert [m for m, _, _ in rec.loads] == _MODELS  # one extraction load per model
    assert res.ran == _MODELS and res.stopped_reason is None


def test_orch2_injection_params_threaded(tmp_path: Path) -> None:
    rec = _Recorder()
    _run(rec, tmp_path / "records.jsonl")
    for call in rec.gen_completion_calls:
        assert call["depth_fraction"] == pytest.approx(0.61)
        assert call["dose_fraction"] == pytest.approx(0.044)
        assert call["seeds"] == [0, 1, 2]
        assert call["n_trials"] == 5
        assert call["generator"] == f"generator:{call['cv'].model_id}"
        assert "judge" not in call  # generation phase does NOT judge


def test_precision_map_threaded_to_load_and_generator(tmp_path: Path) -> None:
    rec = _Recorder()
    pmap = {
        "Qwen/Qwen2.5-0.5B-Instruct": ("float16", None),
        "Qwen/Qwen2.5-1.5B-Instruct": ("bfloat16", "nf4"),
    }
    _run(rec, tmp_path / "records.jsonl", precision_map=pmap)
    assert rec.loads == [
        ("Qwen/Qwen2.5-0.5B-Instruct", "float16", None),
        ("Qwen/Qwen2.5-1.5B-Instruct", "bfloat16", "nf4"),
    ]
    assert rec.gen_calls == [
        ("Qwen/Qwen2.5-0.5B-Instruct", "float16", None),
        ("Qwen/Qwen2.5-1.5B-Instruct", "bfloat16", "nf4"),
    ]


def test_default_precision_is_float32(tmp_path: Path) -> None:
    rec = _Recorder()
    _run(rec, tmp_path / "records.jsonl")  # no precision_map
    assert all(dtype == "float32" and quant is None for _, dtype, quant in rec.loads)


def test_two_phase_extract_generate_then_judge(tmp_path: Path) -> None:
    rec = _Recorder()
    _run(rec, tmp_path / "records.jsonl")
    # per model: load (extract) -> build generator -> all completions -> judge.
    # Judging comes AFTER every completion for the rung (GPU freed before judge).
    assert rec.events == [
        "load:Qwen/Qwen2.5-0.5B-Instruct",
        "gen:Qwen/Qwen2.5-0.5B-Instruct",
        "gencompl:Qwen/Qwen2.5-0.5B-Instruct",
        "gencompl:Qwen/Qwen2.5-0.5B-Instruct",
        "gencompl:Qwen/Qwen2.5-0.5B-Instruct",
        "judge",
        "load:Qwen/Qwen2.5-1.5B-Instruct",
        "gen:Qwen/Qwen2.5-1.5B-Instruct",
        "gencompl:Qwen/Qwen2.5-1.5B-Instruct",
        "gencompl:Qwen/Qwen2.5-1.5B-Instruct",
        "gencompl:Qwen/Qwen2.5-1.5B-Instruct",
        "judge",
    ]


def test_incremental_write_and_commit_per_model(tmp_path: Path) -> None:
    rec = _Recorder()
    res = _run(rec, tmp_path / "records.jsonl")
    assert rec.commits == len(_MODELS)  # committed after each rung, not just at end
    assert isinstance(res, LadderRun) and res.records


def test_cost_guard_stops_before_breaching_rung(tmp_path: Path) -> None:
    """Each model 'costs' 1h; cap $25 at $10/h, $10 est/rung -> exactly 2 rungs fit.

    Clock advances 1h only when a model actually loads, so `spent` is deterministic:
    guard(m0)=$0+$10 ok; guard(m1)=$10+$10 ok; guard(m2)=$20+$10=$30 > $25 -> stop.
    """
    rec = _Recorder()
    models = [f"m{i}" for i in range(5)]
    now = {"t": 0.0}
    loader_calls = {"n": 0}

    def clock() -> float:
        return now["t"]

    def loader(model_id, device, dtype="float32", quant=None):
        loader_calls["n"] += 1
        now["t"] += 3600.0  # this rung's wall-hour accrues on load
        return rec.load_model(model_id, device, dtype, quant)

    res = run_ladder(
        models,
        concepts=_CONCEPTS,
        seeds=[0],
        n_trials=1,
        out_path=tmp_path / "records.jsonl",
        judge=object(),
        extract_fn=rec.extract,
        make_generator=rec.make_generator,
        generate_completions_fn=rec.gen_completions,
        judge_completions_fn=rec.judge,
        write_seed_records_fn=rec.write,
        load_model_fn=loader,
        on_model_done=rec.commit,
        cost_rate_per_hour=10.0,
        cost_cap_usd=25.0,
        rung_gpu_hours={m: 1.0 for m in models},
        clock=clock,
    )
    assert res.ran == ["m0", "m1"]
    assert res.skipped == ["m2", "m3", "m4"]
    assert res.stopped_reason is not None and "cost guard" in res.stopped_reason
    assert rec.commits == 2  # only the two completed rungs were committed
    assert loader_calls["n"] == 2  # never even loaded the breaching rung


def test_no_guard_when_cap_unset(tmp_path: Path) -> None:
    rec = _Recorder()
    res = _run(rec, tmp_path / "records.jsonl")  # no cost args
    assert res.stopped_reason is None and res.ran == _MODELS


def test_cli_seam_threads_args(monkeypatch, tmp_path: Path) -> None:
    """`runner.main` argparse -> run_ladder: flags bind, defaults (depth 0.61) hold."""
    captured: dict = {}

    def fake_run_ladder(models, **kwargs):
        captured["models"] = list(models)
        captured.update(kwargs)
        return LadderRun(records=[SeedRecord("m", "Oceans", "injected", 0, 1, 1)], ran=list(models))

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
    assert len(captured["concepts"]) == 3
    assert captured["out_path"] == out
    assert captured["depth_fraction"] == pytest.approx(0.61)
    assert captured["dose_fraction"] == pytest.approx(0.044)
