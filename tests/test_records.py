"""Records schema: validation + jsonl round-trip (the A1/A2 seam)."""

from pathlib import Path

import pytest

from introspection_scaling.records import (
    CONDITION_INJECTED,
    SeedRecord,
    read_records,
    write_records,
)


def test_rate() -> None:
    assert SeedRecord("m", "oceans", CONDITION_INJECTED, 0, 3, 10).rate == 0.3


def test_validation_rejects_bad_condition() -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        SeedRecord("m", "c", "bogus", 0, 1, 2)


def test_validation_rejects_success_over_trials() -> None:
    with pytest.raises(ValueError, match="n_success"):
        SeedRecord("m", "c", CONDITION_INJECTED, 0, 5, 3)


def test_validation_rejects_nonpositive_trials() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        SeedRecord("m", "c", CONDITION_INJECTED, 0, 0, 0)


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    recs = [
        SeedRecord("m", "oceans", CONDITION_INJECTED, 0, 4, 10, layer=12, alpha=2.0),
        SeedRecord("m", "oceans", "no_injection", 0, 0, 10),
    ]
    p = tmp_path / "records.jsonl"
    write_records(recs, p)
    assert read_records(p) == recs


def test_read_rejects_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"model_id":"m","concept":"c","condition":"injected","seed":0,'
        '"n_success":1,"n_trials":2,"rogue":9}\n'
    )
    with pytest.raises(ValueError, match="bad record"):
        read_records(p)
