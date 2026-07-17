#!/usr/bin/env bash
# reproduce.sh — clean env → the corrected-dose SIZE-TREND hero, one command.
#
# The published claim (RESULTS.md#finding) is the 7B/14B/32B × {base, instruct,
# coder} introspection trend table and the figure results/scaling_trend_k2.png.
# BOTH are DERIVED, by the exact same instruments RESULTS documents, from the
# committed authoritative verdicts results/trials_*_k2_bedrock.jsonl. This script
# rebuilds them — and, on demand, regenerates those verdicts from scratch on GPU.
#
# ─── STAGES (cheap → expensive) ──────────────────────────────────────────────
#   ./reproduce.sh              DEFAULT. Rebuild the table + figure from the
#                               committed *_bedrock.jsonl. No GPU, no Bedrock,
#                               no cloud auth — this is the reproduction bar a
#                               stranger hits from a clean clone.
#
#   ./reproduce.sh full         FROM ZERO. Regenerate every raw transcript on
#                               Modal GPU (A100-80GB, fp16), re-judge locally
#                               through Bedrock, then rebuild table + figure.
#                               Needs Modal auth AND AWS Bedrock SSO; spends
#                               real GPU + Bedrock money (see the estimate it
#                               echoes up front). Each rung keeps its own baked
#                               $ cap; the run self-stops rather than overspend.
#
# Env knobs (all optional): SEEDS (default "0 1 2"), N_CONCEPTS (6), N_TRIALS
# (12), MODAL_PROFILE (passed through to Modal), AWS_PROFILE (claude),
# AWS_REGION (us-east-1), VOLUME (introspection-results).
#
# ─── MANUAL PARTIAL RERUNS (what the run actually issued, rung by rung) ───────
# Generate ONE rung on Modal (writes trials to the Modal volume):
#     MODAL_PROFILE=bamdad uv run modal run modal_app.py::coder7b_k2
#     uv run modal volume get introspection-results trials_coder7b_k2.jsonl \
#         results/trials_coder7b_k2.jsonl --force
# Re-judge ONE raw trials file locally (fails loud on any parse error):
#     JUDGE_BACKEND=bedrock AWS_PROFILE=claude AWS_REGION=us-east-1 \
#         uv run python scripts/rung_k2_judge.py results/trials_coder7b_k2.jsonl
#   (32B/ladder rungs use scripts/{ladder,base32b,coder32b}_k2_judge.py instead.)
# Rebuild just the table / figure:
#     uv run python scripts/trend_table.py results/trials_*_k2_bedrock.jsonl \
#         --costs results/rung_costs.csv
#     uv run python scripts/plot_trend_k2.py
#
# NOTE: this reproduces the SIZE-TREND claim only. The logit-lens legibility
# probe (RESULTS.md, mechanism section) is a separate artifact, not wired here.
set -euo pipefail
cd "$(dirname "$0")"

STAGE="${1:-plot}"
case "$STAGE" in
  plot | full) ;;
  *)
    echo "[reproduce] unknown stage '$STAGE' — use: ./reproduce.sh [plot|full]" >&2
    exit 2
    ;;
esac

SEEDS="${SEEDS:-0 1 2}"
N_CONCEPTS="${N_CONCEPTS:-6}"
N_TRIALS="${N_TRIALS:-12}"
VOLUME="${VOLUME:-introspection-results}"
AWS_PROFILE="${AWS_PROFILE:-claude}"
AWS_REGION="${AWS_REGION:-us-east-1}"
COSTS="results/rung_costs.csv"
FIGURE="results/scaling_trend_k2.png"

# The seven corrected-dose rungs (raw_norm, k=2, depth 0.61), as three parallel
# columns: Modal entrypoint | raw trials file on the volume | tag for the local
# judge. Ordered cheap→expensive so a cost-guard stop preserves the most rungs.
RUNGS=(
  # entrypoint      raw_trials_basename            judge_script
  "base7b_k2        trials_base7b_k2.jsonl         rung"
  "coder7b_k2       trials_coder7b_k2.jsonl        rung"
  "base14b_k2       trials_base14b_k2.jsonl        rung"
  "coder14b_k2      trials_coder14b_k2.jsonl       rung"
  "base_k2          trials_base32b_k2.jsonl        base32b"
  "coder_k2         trials_coder32b_k2.jsonl       coder32b"
  "ladder_k2        trials_ladder_k2.jsonl         ladder"
)

# The seven authoritative Bedrock verdict files the table + figure consume.
BEDROCK_FILES=(
  results/trials_ladder_k2_bedrock.jsonl
  results/trials_base32b_k2_bedrock.jsonl
  results/trials_coder32b_k2_bedrock.jsonl
  results/trials_base7b_k2_bedrock.jsonl
  results/trials_base14b_k2_bedrock.jsonl
  results/trials_coder7b_k2_bedrock.jsonl
  results/trials_coder14b_k2_bedrock.jsonl
)

die() { echo "[reproduce] FATAL: $*" >&2; exit 1; }

echo "[reproduce] syncing pinned env from committed lockfile…"
uv sync --frozen
mkdir -p results

# ─── STAGE 1 — GPU GENERATE (only in 'full') ─────────────────────────────────
# Fail-loud Modal preflight, echo the cost envelope, then generate each rung and
# pull its transcripts off the Modal volume to results/. Each entrypoint carries
# its own baked $ cap (ladder $15, 32B base/coder $3 each, 7B/14B rungs $3 each);
# a rung self-stops at its cap rather than overspend.
if [ "$STAGE" = full ]; then
  echo "[reproduce] Modal auth preflight…"
  uv run modal profile current >/dev/null 2>&1 \
    || die "Modal not authenticated. Run 'uv run modal setup' (or export MODAL_PROFILE=<profile>) and retry."

  cat <<'COST'
[reproduce] ── GPU COST ENVELOPE ────────────────────────────────────────────
  This regenerates 7 A100-80GB fp16 rungs. Per-rung HARD caps sum to ≈$33
  (ladder 0.5–32B $15 + 32B base $3 + 32B coder $3 + four 7B/14B rungs 4×$3);
  the actual reference run spent ≈$6.6 GPU. Local Bedrock judging adds ≈$16.
  Each rung self-stops at its cap. Ctrl-C now to abort.
──────────────────────────────────────────────────────────────────────────────
COST

  for spec in "${RUNGS[@]}"; do
    read -r ep raw _judge <<<"$spec"
    echo "[reproduce] generate: modal run modal_app.py::$ep  (→ volume:$raw)"
    uv run modal run "modal_app.py::$ep" \
      --n-concepts "$N_CONCEPTS" --n-trials "$N_TRIALS" \
      || die "Modal generation failed for $ep — inspect the log, do NOT retry-with-tweaks."
    echo "[reproduce] fetch: $VOLUME:$raw → results/$raw"
    uv run modal volume get "$VOLUME" "$raw" "results/$raw" --force \
      || die "could not fetch $raw from volume $VOLUME"
  done
fi

# ─── STAGE 2 — LOCAL BEDROCK JUDGE (only in 'full') ──────────────────────────
# Fail-loud AWS preflight, then re-grade each raw transcript file through the
# STRICT Bedrock judge (any parse_error RAISES — never a fake null). Produces the
# results/trials_*_k2_bedrock.jsonl + records the table/figure consume.
if [ "$STAGE" = full ]; then
  echo "[reproduce] AWS Bedrock preflight (profile=$AWS_PROFILE region=$AWS_REGION)…"
  aws sts get-caller-identity --profile "$AWS_PROFILE" --region "$AWS_REGION" >/dev/null 2>&1 \
    || die "AWS Bedrock SSO not live. Run 'aws sso login --profile $AWS_PROFILE' and retry."

  for spec in "${RUNGS[@]}"; do
    read -r _ep raw judge <<<"$spec"
    [ -f "results/$raw" ] || die "raw trials results/$raw missing — run './reproduce.sh full' from stage 1."
    case "$judge" in
      rung)     script=scripts/rung_k2_judge.py ;;
      base32b)  script=scripts/base32b_k2_judge.py ;;
      coder32b) script=scripts/coder32b_k2_judge.py ;;
      ladder)   script=scripts/ladder_k2_judge.py ;;
      *)        die "unknown judge tag '$judge'" ;;
    esac
    echo "[reproduce] judge: $script results/$raw"
    JUDGE_BACKEND=bedrock AWS_PROFILE="$AWS_PROFILE" AWS_REGION="$AWS_REGION" \
      uv run python "$script" "results/$raw" \
      || die "Bedrock judge failed on results/$raw (strict-raise: outage or parse error)."
  done
fi

# ─── STAGE 3 — TABLE + FIGURE (always) ───────────────────────────────────────
# Rebuild the hero table + figure from the authoritative *_bedrock.jsonl. In the
# default 'plot' stage these are the committed verdicts; in 'full' they are the
# ones just regenerated above.
missing=()
for f in "${BEDROCK_FILES[@]}"; do
  [ -f "$f" ] || missing+=("$f")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "[reproduce] missing Bedrock verdict files:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  die "run './reproduce.sh full' to regenerate them on GPU (needs Modal + AWS auth)."
fi

echo "[reproduce] rebuilding trend table (→ results/trend_table.md)…"
uv run python scripts/trend_table.py "${BEDROCK_FILES[@]}" --costs "$COSTS" \
  | tee results/trend_table.md

echo "[reproduce] rendering size-trend figure (→ $FIGURE)…"
uv run python scripts/plot_trend_k2.py

echo "[reproduce] done → results/trend_table.md + $FIGURE"
