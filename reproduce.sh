#!/usr/bin/env bash
# reproduce.sh — clean environment → hero scaling curve, one command.
# A stranger runs THIS and gets our numbers. Everything except the sweep that
# PRODUCES results/records.jsonl is wired and real; the sweep is the one hook
# A1/A2 fill in (their extraction + injection + judge interface is pending).
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2}"
# Instruct variants: the introspection task is a chat self-report; base models
# can't follow the protocol and would manufacture false nulls.
MODELS="${MODELS:-Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct}"
RECORDS="${RECORDS:-results/records.jsonl}"
CURVE="${CURVE:-results/scaling-curve.png}"

echo "[reproduce] syncing pinned env from committed lockfile…"
uv sync --frozen

mkdir -p results

# ─── SWEEP (runner wired; needs A2 harness on the path) ──────────────────────
# introspection_scaling.runner.run_ladder drives A1 extraction + A2 harness and
# emits SeedRecord JSONL (raw n_success / n_trials). Controls are handled inside
# run_concept: injected + no_injection + random_direction (SPEC, non-negotiable).
# Injection params: depth 0.61; dose defaults to the paper's absolute strength
# (dose_mode=raw_norm, alpha = 2 * ||raw diff-of-means||) — the corrected regime.
# Override with --dose-mode resid_frac for the superseded residual-relative dose.
#   Local dev (0.5B, CPU): the command below (small n-trials/concepts for speed).
#   Full ladder (GPU)    : modal run modal_app.py::ladder   (A100-80GB; needs
#                          huggingface-secret + anthropic-secret).
# BLOCKED until A2 harness (wt/agent2 49cc0ee) is on main; the runner fails loud
# with an actionable message if harness is not importable.
if [ ! -f "$RECORDS" ]; then
  echo "[reproduce] generating $RECORDS via the ladder runner…"
  uv run python -m introspection_scaling.runner \
    --models $MODELS --seeds $SEEDS --out "$RECORDS" \
    --n-concepts "${N_CONCEPTS:-10}" --n-trials "${N_TRIALS:-20}" \
    --device "${DEVICE:-cpu}"
fi

# ─── AGGREGATE + PLOT (real, A3) ─────────────────────────────────────────────
# Bootstrap confidence bands over seeds; above-chance = injected band clears
# BOTH control bands; renders the hero scaling curve.
echo "[reproduce] aggregating seeds + rendering scaling curve…"
uv run python -m introspection_scaling.report "$RECORDS" --out "$CURVE" --seed 0

echo "[reproduce] done → $CURVE"
