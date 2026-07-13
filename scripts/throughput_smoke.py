"""HARD GATE before any paid Modal relaunch: prove generation is FAST and
injection still works on 0.5B-Instruct locally (CPU/MPS, tiny counts).

Reports:
  (a) generation throughput (tokens/sec + wall-clock) for N completions,
      use_cache=False (before) vs use_cache=True + batched (after);
  (b) injection integrity AFTER the change (verify_injection_delta:
      magnitude_ratio ~1, cosine sane).

Run:  python scripts/throughput_smoke.py
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "src")
from introspection_scaling import extract_concept_vector  # noqa: E402
from introspection_scaling.harness import (  # noqa: E402
    RepengGenerator,
    dose_alpha,
    layer_for_fraction,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CONCEPT = "oceans"
MAX_NEW = 64
N = 8  # completions for the before/after comparison


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tok_count(gen: RepengGenerator, texts: list[str]) -> int:
    return sum(len(gen.tokenizer(t).input_ids) for t in texts)


def _time_batch(gen: RepengGenerator, layer: int, n: int) -> tuple[float, float]:
    """Generate n no-injection completions in one batched forward; return
    (wall_seconds, tokens_per_sec)."""
    t0 = time.perf_counter()
    texts = gen.generate_batch(None, layer, 0.0, seed=0, n=n)
    wall = time.perf_counter() - t0
    toks = _tok_count(gen, texts)
    return wall, toks / wall if wall else 0.0


def main() -> int:
    device = _device()
    print(f"device={device} model={MODEL_ID} max_new_tokens={MAX_NEW} N={N}")

    # BEFORE: KV-cache OFF (the O(n^2) decode that hangs the ladder).
    gen_off = RepengGenerator(
        MODEL_ID, device=device, max_new_tokens=MAX_NEW, temperature=1.0, use_cache=False
    )
    layer = layer_for_fraction(gen_off.n_layers)
    off_wall, off_tps = _time_batch(gen_off, layer, N)
    print(f"[BEFORE use_cache=False] {N} gens: {off_wall:.1f}s  {off_tps:.1f} tok/s")
    del gen_off

    # AFTER: KV-cache ON + batched.
    gen_on = RepengGenerator(
        MODEL_ID, device=device, max_new_tokens=MAX_NEW, temperature=1.0, use_cache=True
    )
    on_wall, on_tps = _time_batch(gen_on, layer, N)
    print(f"[AFTER  use_cache=True ] {N} gens: {on_wall:.1f}s  {on_tps:.1f} tok/s")
    speedup = off_wall / on_wall if on_wall else float("inf")
    print(f"SPEEDUP (wall): {speedup:.1f}x")

    # Throughput at a ladder-ish batch (24 completions in one forward).
    big_wall, big_tps = _time_batch(gen_on, layer, 24)
    print(f"[AFTER batched x24] {big_wall:.1f}s  {big_tps:.1f} tok/s")

    # Injection integrity AFTER the change.
    cv = extract_concept_vector(MODEL_ID, CONCEPT, device=device)
    resid_norm = gen_on.measure_resid_norm(layer)
    alpha = dose_alpha(resid_norm, 0.044)
    diag = gen_on.verify_injection_delta(cv, layer, alpha)
    print(
        f"[INJECTION] layer={layer} alpha={alpha:.3f} "
        f"magnitude_ratio={diag['magnitude_ratio']:.3f} cosine={diag['cosine_to_v_unit']:.3f}"
    )
    ratio_ok = 0.7 <= diag["magnitude_ratio"] <= 1.4
    print("GATE:", "PASS" if (ratio_ok and on_tps > off_tps) else "REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
