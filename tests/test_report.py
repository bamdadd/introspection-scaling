"""End-to-end aggregate→plot chain over a synthetic fixture (no GPU, no network).

Guards the reproduce.sh tail: records.jsonl -> report.main -> scaling-curve.png.
Uses fake numbers only; nothing here is committed to results/.
"""

from pathlib import Path

from introspection_scaling.plot import KNOWN_MODELS, plot_scaling_curve
from introspection_scaling.records import (
    CONDITION_INJECTED,
    CONDITION_NO_INJECTION,
    CONDITION_RANDOM,
    SeedRecord,
    write_records,
)
from introspection_scaling.report import main
from introspection_scaling.stats import model_points

_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]


def _synthetic_records() -> list[SeedRecord]:
    """Small models look like noise; the big model shows a signal (fake)."""
    recs: list[SeedRecord] = []
    for model in _MODELS:
        big = "7B" in model
        for concept in ("oceans", "dogs"):
            for seed in range(3):
                inj = 8 if big else 2
                recs.append(SeedRecord(model, concept, CONDITION_INJECTED, seed, inj, 10))
                recs.append(SeedRecord(model, concept, CONDITION_NO_INJECTION, seed, 1, 10))
                recs.append(SeedRecord(model, concept, CONDITION_RANDOM, seed, 2, 10))
    return recs


def test_plot_writes_png(tmp_path: Path) -> None:
    points = model_points(_synthetic_records(), n_boot=1000, seed=0)
    out = plot_scaling_curve(points, tmp_path / "curve.png")
    assert out.exists() and out.stat().st_size > 0


def test_all_plotted_models_are_in_registry() -> None:
    for m in _MODELS:
        assert m in KNOWN_MODELS


def test_report_main_end_to_end(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    write_records(_synthetic_records(), records)
    out = tmp_path / "scaling-curve.png"
    rc = main([str(records), "--out", str(out), "--n-boot", "500", "--seed", "0"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
