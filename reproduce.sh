#!/usr/bin/env bash
# reproduce.sh — clean environment → results table, one command.
# Pin seeds, versions, hardware. A stranger runs THIS and gets our numbers.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B}"

echo "[reproduce] syncing pinned env…"
uv sync --frozen || uv sync

echo "[reproduce] TODO: extract concept vectors (repeng), inject, run introspection prompt."
echo "[reproduce] TODO: controls = no-injection + random-direction-matched-norm."
echo "[reproduce] TODO: sweep MODELS × SEEDS, write results/RESULTS.md + scaling curve."
echo "[reproduce] models: $MODELS"
echo "[reproduce] seeds:  $SEEDS"
