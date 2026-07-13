#!/usr/bin/env bash
# reproduce.sh — clean environment → hero scaling curve, one command.
# A stranger runs THIS and gets our numbers. Everything except the sweep that
# PRODUCES results/records.jsonl is wired and real; the sweep is the one hook
# A1/A2 fill in (their extraction + injection + judge interface is pending).
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B}"
RECORDS="${RECORDS:-results/records.jsonl}"
CURVE="${CURVE:-results/scaling-curve.png}"

echo "[reproduce] syncing pinned env from committed lockfile…"
uv sync --frozen

mkdir -p results

# ─── SWEEP (deferred hook — owned by A1/A2) ──────────────────────────────────
# Produces $RECORDS: one JSON line per (model, concept, condition, seed) as
# introspection_scaling.records.SeedRecord (raw n_success / n_trials counts).
# Locally develop on Qwen2.5-0.5B (CPU ok); run the full ladder on Modal.
#   Local dev  : uv run python -m introspection_scaling.sweep --models "$MODELS" --seeds "$SEEDS" --out "$RECORDS"
#   Modal ladder: modal run modal_app.py::run_ladder  (wired once the interface lands)
# Controls (SPEC, non-negotiable): injected + no_injection + random_direction.
if [ ! -f "$RECORDS" ]; then
  echo "[reproduce] MISSING $RECORDS — the A1/A2 sweep hook has not run yet." >&2
  echo "[reproduce] models: $MODELS" >&2
  echo "[reproduce] seeds:  $SEEDS" >&2
  echo "[reproduce] TODO(A1/A2): write the sweep that emits SeedRecord JSONL to $RECORDS." >&2
  exit 2
fi

# ─── AGGREGATE + PLOT (real, A3) ─────────────────────────────────────────────
# Bootstrap confidence bands over seeds; above-chance = injected band clears
# BOTH control bands; renders the hero scaling curve.
echo "[reproduce] aggregating seeds + rendering scaling curve…"
uv run python -m introspection_scaling.report "$RECORDS" --out "$CURVE" --seed 0

echo "[reproduce] done → $CURVE"
