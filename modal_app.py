"""Modal app + reproducible pinned image for the model-size ladder.

The image is built **from the committed lockfile** (`uv.lock` + `pyproject.toml`)
via `Image.uv_sync(frozen=True)`, so a run resolves to exactly the pinned deps a
stranger gets from `uv sync --frozen` locally — the lockfile is the
reproducibility guarantee. `--no-install-project` is passed because the build
context carries only the lock + pyproject, not `src/`, and the smoke path needs
only torch/transformers, not our own package.

Two entrypoints:

* ``smoke`` (``modal run modal_app.py``) — loads Qwen2.5-0.5B-Instruct on an A100
  and returns its hidden size; proves the pinned image works end-to-end on GPU.
* ``run_ladder`` (``modal run modal_app.py::ladder``) — the emergence sweep:
  extract (A1) + inject/judge (A2) across the instruct ladder (Qwen first, Llama
  preflight-gated, 72B held), writing ``SeedRecord`` JSONL to a Modal Volume.
  Sized **A100-80GB** — fp16 for ≤32B (32B fp16 ≈ 64 GB) via the two-phase load;
  the $80 cost guard self-stops and commits a partial curve.

Required Modal secrets (create once; values never live in this repo). Names are
configurable via ``HF_SECRET_NAME`` / ``ANTHROPIC_SECRET_NAME`` — defaults match
the current workspace (``huggingface`` + ``anthropic-secret``):

* HF secret (default name ``huggingface``) must expose ``HF_TOKEN``, on an
  account that has **accepted the gated meta-llama/Llama-3.x licenses**.
* Anthropic secret (default name ``anthropic-secret``) must expose
  ``ANTHROPIC_API_KEY`` for the faithful judge.

Pinned (from uv.lock at commit time): torch 2.13.0 (CUDA), transformers 5.13.1,
modal 1.5.2. Python 3.11. Base: modal debian_slim.
"""

from __future__ import annotations

import os

import modal

APP_NAME = "introspection-scaling"

# Base image pinned by Python version; deps pinned by the frozen lockfile.
# uv_sync bundles pyproject.toml + uv.lock from the project dir into the build
# and runs `uv sync --frozen`, so the image == the committed lock.
image = modal.Image.debian_slim(python_version="3.11").uv_sync(
    frozen=True,
    # --no-install-project: build ctx carries only lock + pyproject, not src/,
    #   and the smoke path needs torch/transformers, not our own package.
    # --no-dev: the runtime image needs no ruff/mypy/pytest — smaller, faster cold start.
    extra_options="--no-install-project --no-dev",
    uv_version="0.10.8",  # match the local uv that authored the lock
)

app = modal.App(APP_NAME, image=image)

# Cache HF downloads across runs so the ladder doesn't re-download weights.
_hf_cache = modal.Volume.from_name("introspection-hf-cache", create_if_missing=True)
_HF_CACHE_DIR = "/root/.cache/huggingface"

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@app.function(gpu="A100", volumes={_HF_CACHE_DIR: _hf_cache}, timeout=900)
def smoke(model_id: str = SMOKE_MODEL) -> dict[str, object]:
    """Load ``model_id`` on the GPU and report its hidden size — proves the image.

    Returns a small dict (JSON-friendly) with the CUDA device, dtype, and the
    model's hidden size, so the caller can assert the GPU path actually ran.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    assert torch.cuda.is_available(), "CUDA not available — image is not GPU-capable"
    device = torch.cuda.get_device_name(0)

    config = AutoConfig.from_pretrained(model_id, cache_dir=_HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, cache_dir=_HF_CACHE_DIR
    ).to("cuda")

    # Tiny forward pass to confirm weights are actually on-device and usable.
    with torch.no_grad():
        out = model(torch.tensor([[0, 1, 2]], device="cuda"))
    _hf_cache.commit()

    return {
        "model_id": model_id,
        "cuda_device": device,
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "logits_shape": list(out.logits.shape),
        "param_dtype": str(next(model.parameters()).dtype),
    }


# --------------------------------------------------------------------------- #
# Full ladder sweep. Needs the two Modal secrets above; run with `modal run`.
# --------------------------------------------------------------------------- #

# The sweep needs our package (extract + harness + runner). Mount it into the
# image; the pinned deps still come from the frozen lockfile.
_ladder_image = image.add_local_python_source("introspection_scaling")

# Persist records.jsonl across runs; download with `modal volume get`.
_results_vol = modal.Volume.from_name("introspection-results", create_if_missing=True)
_RESULTS_DIR = "/results"

# Secret NAMES are configurable so we are not hard-locked to one workspace's
# naming. Defaults match what the workspace actually has today: an HF secret
# named "huggingface" and an Anthropic secret named "anthropic-secret".
# Override per-run, e.g.:  HF_SECRET_NAME=my-hf ANTHROPIC_SECRET_NAME=my-anthropic
# Required KEYS inside the secrets (values never live in this repo):
#   HF secret        -> HF_TOKEN         (account must have accepted the GATED
#                                          meta-llama/Llama-3.x licenses)
#   Anthropic secret -> ANTHROPIC_API_KEY (faithful judge; fails loud if absent)
HF_SECRET_NAME = os.environ.get("HF_SECRET_NAME", "huggingface")
ANTHROPIC_SECRET_NAME = os.environ.get("ANTHROPIC_SECRET_NAME", "anthropic-secret")
_SECRETS = [
    modal.Secret.from_name(HF_SECRET_NAME),
    modal.Secret.from_name(ANTHROPIC_SECRET_NAME),
]

# ------------------------------- Ladder ------------------------------------- #
# Ascending by params so a cost-guard stop preserves the MOST rungs — and the low
# end is where the emergence threshold likely sits, so the partial curve is the
# useful part. Qwen rungs are all UNGATED; Llama is gated (preflight below).
QWEN_LADDER: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
)
LLAMA_LADDER: tuple[str, ...] = (
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)
# 72B anchor is HELD — excluded from the fireable set. Double-blocked: needs
# bitsandbytes/nf4 AND A2's DE-RISK verdict that repeng injection works in 4-bit.
HELD_ANCHOR_72B = "Qwen/Qwen2.5-72B-Instruct"

# One-off calibration (human-authorized single run, NOT part of the fireable
# ladder): Qwen2.5-Coder-32B-Instruct — the third-party KNOWN-POSITIVE open model.
# Our ladder only ran the Instruct family; this checks whether the pipeline
# reproduces the reported effect at all. fp16, separate path, HARD $10 cap.
CALIBRATION_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
CALIBRATION_RECORDS = f"{_RESULTS_DIR}/records_coder32b.jsonl"
CALIBRATION_CAP_USD = 10.0

# Coder-32B validation rung at the corrected paper dose (raw_norm, k=2). Modal
# GENERATES transcripts only; the authoritative judge is the LOCAL Bedrock judge
# (AWS SSO is local, not portable to Modal). SEPARATE _k2 path — must NOT touch
# the suspect records_coder32b.jsonl. HARD $3 cap; est 0.7h so the guard's
# projection ($2.8) stays under the cap and the single rung actually starts.
CODER_K2_RECORDS = f"{_RESULTS_DIR}/records_coder32b_k2.jsonl"
CODER_K2_TRIALS = f"{_RESULTS_DIR}/trials_coder32b_k2.jsonl"
CODER_K2_CAP_USD = 3.0
CODER_K2_STRENGTH_K = 2.0
CODER_K2_RUNG_HOURS = {CALIBRATION_MODEL: 0.7}

# Base-model rung at the same corrected dose (raw_norm, k=2) as run_coder_k2 —
# Qwen2.5-32B BASE (NO fine-tune) so Base vs Instruct vs Coder is comparable at
# 32B. Same generate-on-Modal / judge-locally-with-Bedrock split. SEPARATE _k2
# path; HARD $3 cap; est 0.7h so the guard projection ($2.8) stays under the cap.
BASE_MODEL_K2 = "Qwen/Qwen2.5-32B"
BASE_K2_RECORDS = f"{_RESULTS_DIR}/records_base32b_k2.jsonl"
BASE_K2_TRIALS = f"{_RESULTS_DIR}/trials_base32b_k2.jsonl"
BASE_K2_CAP_USD = 3.0
BASE_K2_RUNG_HOURS = {BASE_MODEL_K2: 0.7}

# Corrected-dose FULL Qwen ladder (raw_norm k=2). Same generate-on-Modal /
# judge-locally-with-Bedrock split. SEPARATE _k2 path. GPU cap $15. Per-rung
# estimates are REALISTIC (based on the actual fp16 costs, biased slightly high)
# — the biased-high RUNG_GPU_HOURS would false-trip the guard before 32B at a $15
# cap, so use tighter values here. The guard still stops if a rung runs wildly over.
LADDER_K2_RECORDS = f"{_RESULTS_DIR}/records_ladder_k2.jsonl"
LADDER_K2_TRIALS = f"{_RESULTS_DIR}/trials_ladder_k2.jsonl"
LADDER_K2_CAP_USD = 15.0
LADDER_K2_RUNG_HOURS: dict[str, float] = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.2,
    "Qwen/Qwen2.5-1.5B-Instruct": 0.25,
    "Qwen/Qwen2.5-3B-Instruct": 0.3,
    "Qwen/Qwen2.5-7B-Instruct": 0.35,
    "Qwen/Qwen2.5-14B-Instruct": 0.5,
    "Qwen/Qwen2.5-32B-Instruct": 0.7,
}

# Small-model rungs at the SAME corrected dose (raw_norm k=2) as run_base_k2 /
# run_coder_k2 — they fill in Base vs Coder-Instruct at 7B and 14B below the 32B
# anchor. Same generate-on-Modal / judge-locally-with-Bedrock split; SEPARATE
# per-model _k2 paths (never cross-wire a model to another's path). NONE are in
# QWEN_LADDER, so each needs an EXPLICIT PRECISION_MAP + RUNG_GPU_HOURS entry: the
# default precision would be float32 (breaks the fp16 contract), and a missing
# GPU-hours est makes the cost guard project $0 and skip the projection.
RUNG_K2_BASE_7B = "Qwen/Qwen2.5-7B"  # base (no fine-tune)
RUNG_K2_BASE_14B = "Qwen/Qwen2.5-14B"  # base (no fine-tune)
RUNG_K2_CODER_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"  # coder = the -Instruct variant
RUNG_K2_CODER_14B = "Qwen/Qwen2.5-Coder-14B-Instruct"  # coder = the -Instruct variant
RUNG_K2_MODELS: tuple[str, ...] = (
    RUNG_K2_BASE_7B,
    RUNG_K2_BASE_14B,
    RUNG_K2_CODER_7B,
    RUNG_K2_CODER_14B,
)
RUNG_K2_CAP_USD = 3.0  # per-rung HARD cap, same as the Coder/Base 32B rungs
RUNG_K2_RECORDS = {
    RUNG_K2_BASE_7B: f"{_RESULTS_DIR}/records_base7b_k2.jsonl",
    RUNG_K2_BASE_14B: f"{_RESULTS_DIR}/records_base14b_k2.jsonl",
    RUNG_K2_CODER_7B: f"{_RESULTS_DIR}/records_coder7b_k2.jsonl",
    RUNG_K2_CODER_14B: f"{_RESULTS_DIR}/records_coder14b_k2.jsonl",
}
RUNG_K2_TRIALS = {
    RUNG_K2_BASE_7B: f"{_RESULTS_DIR}/trials_base7b_k2.jsonl",
    RUNG_K2_BASE_14B: f"{_RESULTS_DIR}/trials_base14b_k2.jsonl",
    RUNG_K2_CODER_7B: f"{_RESULTS_DIR}/trials_coder7b_k2.jsonl",
    RUNG_K2_CODER_14B: f"{_RESULTS_DIR}/trials_coder14b_k2.jsonl",
}

# Cross-architecture MoE probe (issue #39). Qwen1.5-MoE-A2.7B-Chat: a sparse MoE
# (60 experts, top-4, ~2.7B active / ~14B total) at the SAME corrected dose
# (raw_norm k=2, depth 0.61) as the dense _k2 rungs — the point is to show the
# repeng control is live on the EXPERTS, not just the dense residual. NOT in
# QWEN_LADDER, so it needs its own explicit PRECISION_MAP + RUNG_GPU_HOURS entry
# (default float32 would break the fp16 contract; a missing hours est projects $0
# and skips the cost-guard projection). Same generate-on-Modal / judge-locally
# split as the other _k2 rungs. SEPARATE _k2 paths; HARD $3 cap.
MOE_MODEL_K2 = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
MOE_K2_RECORDS = f"{_RESULTS_DIR}/records_moe_k2.jsonl"
MOE_K2_TRIALS = f"{_RESULTS_DIR}/trials_moe_k2.jsonl"
MOE_K2_CAP_USD = 3.0
MOE_K2_RUNG_HOURS = {MOE_MODEL_K2: 0.5}  # biased-high; ~2.7B active projects $2.0 < $3

# Per-model precision (SHARED CONTRACT). <=32B -> fp16; 72B anchor -> bf16 + nf4.
PRECISION_MAP: dict[str, tuple[str, str | None]] = {
    **{m: ("float16", None) for m in QWEN_LADDER + LLAMA_LADDER},
    CALIBRATION_MODEL: ("float16", None),  # explicit: default would be float32 (OOMs 32B)
    BASE_MODEL_K2: ("float16", None),  # 32B base, same fp16 contract as the Coder rung
    # Small-model _k2 rungs: NOT in QWEN_LADDER -> explicit fp16 (default float32 breaks fp16).
    **{m: ("float16", None) for m in RUNG_K2_MODELS},
    MOE_MODEL_K2: ("float16", None),  # MoE probe, same fp16 contract (default would be float32)
    HELD_ANCHOR_72B: ("bfloat16", "nf4"),
}

# Money cost guard. Rate biased HIGH: erring high trips the guard early (=under-
# spend), the safe direction for a cap we cannot verify live from here.
A100_80GB_USD_PER_HOUR = 4.00  # guard correctness depends on this being >= real
MODAL_GPU_CAP_USD = 80.0  # hard self-stop for THIS run (workspace caps $100)

# GPU-hours per rung (fp16, 6 concepts x 12 trials x 3 seeds = 648 gen/model, plus
# extraction + two model loads). See RESULTS "Cost estimate"; biased high.
RUNG_GPU_HOURS: dict[str, float] = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.3,
    "Qwen/Qwen2.5-1.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-3B-Instruct": 0.8,
    "Qwen/Qwen2.5-7B-Instruct": 1.4,
    "Qwen/Qwen2.5-14B-Instruct": 2.4,
    "Qwen/Qwen2.5-32B-Instruct": 4.5,
    "meta-llama/Llama-3.2-1B-Instruct": 0.4,
    "meta-llama/Llama-3.2-3B-Instruct": 0.8,
    "meta-llama/Llama-3.1-8B-Instruct": 1.5,
    # 72B 4-bit anchor (bf16+nf4). Biased high so the $80 guard projection is real
    # on a single-rung run (else it defaults to 0.0 and only the timeout backstops).
    HELD_ANCHOR_72B: 8.0,
    # Coder-32B calibration: same size as Qwen 32B-Instruct (actual ~$1.12); 2.4h
    # is biased-high yet projects $9.6 < the $10 cap so the guard allows the start.
    CALIBRATION_MODEL: 2.4,
    # Base-32B rung: same size as the Coder rung; biased-high mirror of its 2.4h.
    BASE_MODEL_K2: 2.4,
    # Small-model _k2 rungs (generate-only, single rung each). Biased-high yet each
    # projects under the $3 per-rung cap so the guard STARTS the rung: 7B 0.4h->$1.6,
    # 14B 0.7h->$2.8 (both < $3). A missing entry would default to 0.0 and skip the projection.
    RUNG_K2_BASE_7B: 0.4,
    RUNG_K2_BASE_14B: 0.7,
    RUNG_K2_CODER_7B: 0.4,
    RUNG_K2_CODER_14B: 0.7,
    # MoE probe (~2.7B active): generate-only single rung. Biased-high 0.5h projects
    # $2.0 < the $3 cap so the guard STARTS the rung; a missing entry would default
    # to 0.0 and skip the projection.
    MOE_MODEL_K2: 0.5,
}

# 72B anchor records go to a SEPARATE volume path — write_records opens mode 'w'
# (overwrite), so it must NOT share records.jsonl with the committed 0.5-32B run.
ANCHOR_72B_RECORDS = f"{_RESULTS_DIR}/records_72b.jsonl"


def _llama_preflight() -> tuple[bool, str]:
    """Cheap gated-access probe: pull a Llama config + tokenizer (no weights).

    On 401/403 (license not accepted yet) returns ``(False, reason)`` so the
    caller runs the Qwen curve and defers Llama — the Llama license must NOT
    block the Qwen curve.
    """
    probe = LLAMA_LADDER[0]
    try:
        from transformers import AutoConfig, AutoTokenizer

        AutoConfig.from_pretrained(probe, cache_dir=_HF_CACHE_DIR)
        AutoTokenizer.from_pretrained(probe, cache_dir=_HF_CACHE_DIR)
        return True, f"gated access OK ({probe})"
    except Exception as exc:  # noqa: BLE001 - any gate/auth failure defers Llama
        return False, f"Llama gated access not live ({type(exc).__name__}); deferring Llama rungs"


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",  # 32B fp16 ≈ 64 GB (extract + ControlModel, two-phase)
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=_SECRETS,
    timeout=24 * 3600,
)
def run_ladder(
    concepts: list[str],
    seeds: list[int],
    n_trials: int = 12,
    depth_fraction: float = 0.61,
    dose_fraction: float = 0.044,
    models: list[str] | None = None,
) -> dict[str, object]:
    """Run the emergence ladder on GPU and persist ``records.jsonl`` incrementally.

    Order: Qwen rungs (ungated) ALWAYS run first; Llama rungs are appended only if
    the gated-access preflight passes, else deferred (logged). The 72B anchor is
    NOT included (held). Per-model precision from ``PRECISION_MAP`` (fp16 <=32B).
    The $80 cost guard self-stops before a rung that would breach the cap and
    commits the partial curve. Faithful Anthropic judge (no silent fallback).
    """
    from introspection_scaling.runner import run_ladder as _run

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    # huggingface_hub reads HF_TOKEN (canonical) for gated Llama pulls; alias the
    # legacy name so auth works regardless of which the loader/version checks.
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    deferred_llama: list[str] = []
    if models is None:
        ok, reason = _llama_preflight()
        print(f"[preflight] {reason}")
        models = list(QWEN_LADDER) + (list(LLAMA_LADDER) if ok else [])
        if not ok:
            deferred_llama = list(LLAMA_LADDER)

    out = f"{_RESULTS_DIR}/records.jsonl"
    result = _run(
        models,
        concepts=concepts,
        seeds=seeds,
        n_trials=n_trials,
        out_path=out,
        trials_path=f"{_RESULTS_DIR}/trials.jsonl",  # re-judgeable raw layer
        depth_fraction=depth_fraction,
        dose_fraction=dose_fraction,
        device="cuda",
        precision_map=PRECISION_MAP,
        cost_rate_per_hour=A100_80GB_USD_PER_HOUR,
        cost_cap_usd=MODAL_GPU_CAP_USD,
        rung_gpu_hours=RUNG_GPU_HOURS,
        on_model_done=_results_vol.commit,
    )
    _results_vol.commit()
    _hf_cache.commit()
    return {
        "n_records": len(result.records),
        "out": out,
        "ran": result.ran,
        "skipped_by_cost_guard": result.skipped,
        "deferred_llama_gated": deferred_llama,
        "stopped_reason": result.stopped_reason,
        "spent_usd_gpu": round(result.spent_usd, 2),
    }


# --------------------------- 72B anchor (4-bit) ----------------------------- #
# DE-RISK gate + the held anchor rung, both A100-80GB with the bitsandbytes image.


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache},
    secrets=_SECRETS,
    timeout=3600,
)
def verify_4bit(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> dict[str, object]:
    """DE-RISK: does repeng injection survive nf4? (image gate + injection gate.)

    Compares ``verify_injection_delta`` under fp16 vs fp16+nf4 on ``model_id``
    (default 0.5B — a cheap proxy; the 'nf4 leaves the residual stream in compute
    dtype' argument is architecture-general). PASS iff nf4 magnitude_ratio ∈
    [0.8,1.3] and cosine within 0.15 of fp16. Reports WHICH gate failed so a FAIL
    distinguishes a broken bnb/CUDA image from injection breaking under nf4.
    """
    import numpy as np
    import torch

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # Image gate: can bitsandbytes import and see the GPU on this CUDA image?
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {
            "verdict": "FAIL",
            "failure_kind": "image/bitsandbytes-import",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not torch.cuda.is_available():
        return {
            "verdict": "FAIL",
            "failure_kind": "image/no-cuda",
            "error": "torch.cuda unavailable",
        }

    from introspection_scaling import extract_concept_vector
    from introspection_scaling.harness import RepengGenerator, dose_alpha, layer_for_fraction

    def _one(dtype: str, quant: str | None) -> dict[str, float]:
        gen = RepengGenerator(model_id, device="cuda", dtype=dtype, quant=quant, max_new_tokens=8)
        cv = extract_concept_vector(model_id, "oceans", device="cuda")
        layer = layer_for_fraction(gen.n_layers)
        alpha = dose_alpha(gen.measure_resid_norm(layer), 0.044)
        return {k: float(v) for k, v in gen.verify_injection_delta(cv, layer, alpha).items()}

    try:
        fp16 = _one("float16", None)
        nf4 = _one("float16", "nf4")
    except Exception as exc:  # noqa: BLE001
        return {
            "verdict": "FAIL",
            "failure_kind": "injection/runtime",
            "error": f"{type(exc).__name__}: {exc}",
        }

    ratio_ok = 0.8 <= nf4["magnitude_ratio"] <= 1.3
    cos_ok = abs(nf4["cosine_to_v_unit"] - fp16["cosine_to_v_unit"]) < 0.15
    ok = ratio_ok and cos_ok and bool(np.isfinite(nf4["delta_norm"]))
    return {
        "verdict": "PASS" if ok else "FAIL",
        "failure_kind": None if ok else "injection-breaks-under-nf4",
        "model_id": model_id,
        "fp16": fp16,
        "nf4": nf4,
        "ratio_ok": ratio_ok,
        "cos_ok": cos_ok,
    }


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=_SECRETS,
    timeout=24 * 3600,
)
def run_anchor_72b(concepts: list[str], seeds: list[int], n_trials: int = 12) -> dict[str, object]:
    """Run the Qwen2.5-72B-Instruct anchor rung (bf16+nf4) to a SEPARATE path.

    Writes ``records_72b.jsonl`` (never the committed records.jsonl). $80 cost
    guard active. The caller inspects these records for the decision trigger
    (quantized positive -> do NOT auto-commit) before any repo write.
    """
    from introspection_scaling.runner import run_ladder as _run

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    result = _run(
        [HELD_ANCHOR_72B],
        concepts=concepts,
        seeds=seeds,
        n_trials=n_trials,
        out_path=ANCHOR_72B_RECORDS,
        trials_path=f"{_RESULTS_DIR}/trials_72b.jsonl",  # re-judgeable raw layer
        depth_fraction=0.61,
        dose_fraction=0.044,
        device="cuda",
        precision_map=PRECISION_MAP,
        cost_rate_per_hour=A100_80GB_USD_PER_HOUR,
        cost_cap_usd=MODAL_GPU_CAP_USD,
        rung_gpu_hours=RUNG_GPU_HOURS,
        on_model_done=_results_vol.commit,
    )
    _results_vol.commit()
    _hf_cache.commit()
    return {
        "n_records": len(result.records),
        "out": ANCHOR_72B_RECORDS,
        "ran": result.ran,
        "stopped_reason": result.stopped_reason,
        "spent_usd_gpu": round(result.spent_usd, 2),
    }


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=_SECRETS,
    timeout=6 * 3600,
)
def run_calibration(
    concepts: list[str],
    seeds: list[int],
    n_trials: int = 12,
    dose_mode: str = "raw_norm",
    strength_k: float = 2.0,
) -> dict[str, object]:
    """One-off fp16 calibration on Coder-32B (human-authorized single run).

    Uses the CORRECTED paper dose: ``dose_mode='raw_norm'``, ``strength_k=2``
    (the paper's canonical self-report injection strength, PINNED A PRIORI — one
    value, no sweep). alpha = 2 * ||raw diff-of-means|| at the injection layer.

    FIT CHECK FIRST: load the fp16 RepengGenerator and take one resid-norm forward
    pass. If it OOMs / can't run, return ``fit_ok=False`` and run NO sweep (no
    crash-loop). If it fits, run the $10-capped single-rung sweep to a SEPARATE
    path (records_coder32b.jsonl — never the committed records.jsonl).
    """
    import gc

    import torch

    from introspection_scaling.harness import RepengGenerator, layer_for_fraction
    from introspection_scaling.runner import run_ladder as _run

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # --- FIT CHECK (fp16 32B ~ 64 GB; Instruct-32B fit at $1.12). Fail clean. ---
    try:
        gen = RepengGenerator(
            CALIBRATION_MODEL, device="cuda", dtype="float16", quant=None, max_new_tokens=4
        )
        layer = layer_for_fraction(gen.n_layers)
        resid_norm = gen.measure_resid_norm(layer)
        if not (resid_norm == resid_norm and resid_norm > 0):  # finite + positive
            raise RuntimeError(f"non-finite resid_norm {resid_norm}")
    except Exception as exc:  # noqa: BLE001 - report the fit failure, run no sweep
        return {
            "fit_ok": False,
            "model_id": CALIBRATION_MODEL,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(f"[fit] {CALIBRATION_MODEL} fp16 OK: layer={layer} resid_norm={resid_norm:.2f}")
    del gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Sweep (single rung, HARD $10 cap, separate path) ---
    result = _run(
        [CALIBRATION_MODEL],
        concepts=concepts,
        seeds=seeds,
        n_trials=n_trials,
        out_path=CALIBRATION_RECORDS,
        trials_path=f"{_RESULTS_DIR}/trials_coder32b.jsonl",  # re-judgeable raw layer
        depth_fraction=0.61,
        dose_mode=dose_mode,  # 'raw_norm' (paper dose) for the Coder validation
        strength_k=strength_k,  # k=2 pinned a priori
        device="cuda",
        precision_map=PRECISION_MAP,
        cost_rate_per_hour=A100_80GB_USD_PER_HOUR,
        cost_cap_usd=CALIBRATION_CAP_USD,
        rung_gpu_hours=RUNG_GPU_HOURS,
        on_model_done=_results_vol.commit,
    )
    _results_vol.commit()
    _hf_cache.commit()
    return {
        "fit_ok": True,
        "n_records": len(result.records),
        "out": CALIBRATION_RECORDS,
        "ran": result.ran,
        "stopped_reason": result.stopped_reason,
        "spent_usd_gpu": round(result.spent_usd, 2),
        "fit_resid_norm": round(resid_norm, 2),
    }


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],  # Qwen ungated; NO judge secret on Modal
    timeout=2 * 3600,
)
def run_coder_k2(concepts: list[str], seeds: list[int], n_trials: int = 12) -> dict[str, object]:
    """GENERATE-ONLY Coder-32B transcripts at the corrected paper dose (raw_norm,
    k=2). The authoritative judge is the LOCAL Bedrock judge (AWS SSO is local);
    the RuleBasedJudge here is a NON-authoritative placeholder so PR#24 can persist
    the transcripts. Committed result = local Bedrock re-judge of these transcripts.

    Fit-check first: log the ACTUAL alpha (= k·‖raw diff-of-means‖ at the 0.61
    layer) and one ``verify_injection_delta`` so the dose is observably LIVE (not a
    no-op / coherence-destroyer) — observe, do NOT tune. $3 cap; separate _k2 path.
    """
    import gc

    import torch

    from introspection_scaling import extract_concept_vector
    from introspection_scaling.harness import RepengGenerator, RuleBasedJudge, layer_for_fraction
    from introspection_scaling.runner import run_ladder as _run

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # --- FIT CHECK + DOSE OBSERVABILITY (observe, never tune) ---
    # ONE model resident at a time (32B fp16 ~64 GB; two copies OOM an 80 GB A100):
    # Phase 1 extract (fp16) -> free -> Phase 2 generator + verify -> free.
    from introspection_scaling.runner import _load_causal_lm

    def _free() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        emodel, etok = _load_causal_lm(CALIBRATION_MODEL, "cuda", "float16", None)
        layer = layer_for_fraction(int(emodel.config.num_hidden_layers))  # depth 0.61
        cv = extract_concept_vector(
            CALIBRATION_MODEL, "oceans", model=emodel, tokenizer=etok, device="cuda"
        )
        del emodel, etok
        _free()
        if layer not in cv.raw_norms:
            raise RuntimeError(f"raw_norms missing injection layer {layer}")
        raw_norm = float(cv.raw_norms[layer])
        alpha = CODER_K2_STRENGTH_K * raw_norm
        gen = RepengGenerator(
            CALIBRATION_MODEL, device="cuda", dtype="float16", quant=None, max_new_tokens=8
        )
        diag = {k: float(v) for k, v in gen.verify_injection_delta(cv, layer, alpha).items()}
        del gen, cv
        _free()
    except Exception as exc:  # noqa: BLE001 - report the fit/dose failure, run no sweep
        return {
            "fit_ok": False,
            "model_id": CALIBRATION_MODEL,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(
        f"[dose] layer={layer} raw_norm={raw_norm:.3f} k={CODER_K2_STRENGTH_K} alpha={alpha:.3f} "
        f"ratio={diag['magnitude_ratio']:.3f} cos={diag['cosine_to_v_unit']:.3f}"
    )

    # --- Generate (RuleBasedJudge PLACEHOLDER; transcripts are the payload) ---
    result = _run(
        [CALIBRATION_MODEL],
        concepts=concepts,
        seeds=seeds,
        n_trials=n_trials,
        out_path=CODER_K2_RECORDS,
        trials_path=CODER_K2_TRIALS,
        depth_fraction=0.61,
        dose_mode="raw_norm",
        strength_k=CODER_K2_STRENGTH_K,
        device="cuda",
        precision_map=PRECISION_MAP,
        judge=RuleBasedJudge(),  # NON-authoritative; real verdict = local Bedrock re-judge
        cost_rate_per_hour=A100_80GB_USD_PER_HOUR,
        cost_cap_usd=CODER_K2_CAP_USD,
        rung_gpu_hours=CODER_K2_RUNG_HOURS,
        on_model_done=_results_vol.commit,
    )
    _results_vol.commit()
    _hf_cache.commit()
    return {
        "fit_ok": True,
        "dose": {
            "layer": layer,
            "raw_norm": round(raw_norm, 3),
            "k": CODER_K2_STRENGTH_K,
            "alpha": round(alpha, 3),
            **{k: round(v, 3) for k, v in diag.items()},
        },
        "trials_out": CODER_K2_TRIALS,
        "n_trials_persisted": len(result.records),
        "ran": result.ran,
        "stopped_reason": result.stopped_reason,
        "spent_usd_gpu": round(result.spent_usd, 2),
        "note": "Modal verdicts are placeholders; authoritative = local Bedrock re-judge",
    }


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],  # Qwen ungated; NO judge secret on Modal
    timeout=2 * 3600,
)
def run_base_k2(concepts: list[str], seeds: list[int], n_trials: int = 12) -> dict[str, object]:
    """GENERATE-ONLY Qwen2.5-32B BASE transcripts at the corrected paper dose
    (raw_norm, k=2) — the same corrected dose as run_coder_k2, so Base vs Instruct
    vs Coder is comparable at 32B. The authoritative judge is the LOCAL Bedrock
    judge (AWS SSO is local); the RuleBasedJudge here is a NON-authoritative
    placeholder so the transcripts persist. Committed result = local Bedrock
    re-judge of these transcripts.

    Fit-check first: log the ACTUAL alpha (= k·‖raw diff-of-means‖ at the 0.61
    layer) and one ``verify_injection_delta`` so the dose is observably LIVE (not a
    no-op / coherence-destroyer) — observe, do NOT tune. $3 cap; separate _k2 path.
    """
    import gc

    import torch

    from introspection_scaling import extract_concept_vector
    from introspection_scaling.harness import RepengGenerator, RuleBasedJudge, layer_for_fraction
    from introspection_scaling.runner import run_ladder as _run

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # --- FIT CHECK + DOSE OBSERVABILITY (observe, never tune) ---
    # ONE model resident at a time (32B fp16 ~64 GB; two copies OOM an 80 GB A100):
    # Phase 1 extract (fp16) -> free -> Phase 2 generator + verify -> free.
    from introspection_scaling.runner import _load_causal_lm

    def _free() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        emodel, etok = _load_causal_lm(BASE_MODEL_K2, "cuda", "float16", None)
        layer = layer_for_fraction(int(emodel.config.num_hidden_layers))  # depth 0.61
        cv = extract_concept_vector(
            BASE_MODEL_K2, "oceans", model=emodel, tokenizer=etok, device="cuda"
        )
        del emodel, etok
        _free()
        if layer not in cv.raw_norms:
            raise RuntimeError(f"raw_norms missing injection layer {layer}")
        raw_norm = float(cv.raw_norms[layer])
        alpha = CODER_K2_STRENGTH_K * raw_norm
        gen = RepengGenerator(
            BASE_MODEL_K2, device="cuda", dtype="float16", quant=None, max_new_tokens=8
        )
        diag = {k: float(v) for k, v in gen.verify_injection_delta(cv, layer, alpha).items()}
        del gen, cv
        _free()
    except Exception as exc:  # noqa: BLE001 - report the fit/dose failure, run no sweep
        return {
            "fit_ok": False,
            "model_id": BASE_MODEL_K2,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(
        f"[dose] layer={layer} raw_norm={raw_norm:.3f} k={CODER_K2_STRENGTH_K} alpha={alpha:.3f} "
        f"ratio={diag['magnitude_ratio']:.3f} cos={diag['cosine_to_v_unit']:.3f}"
    )

    # --- Generate (RuleBasedJudge PLACEHOLDER; transcripts are the payload) ---
    result = _run(
        [BASE_MODEL_K2],
        concepts=concepts,
        seeds=seeds,
        n_trials=n_trials,
        out_path=BASE_K2_RECORDS,
        trials_path=BASE_K2_TRIALS,
        depth_fraction=0.61,
        dose_mode="raw_norm",
        strength_k=CODER_K2_STRENGTH_K,
        device="cuda",
        precision_map=PRECISION_MAP,
        judge=RuleBasedJudge(),  # NON-authoritative; real verdict = local Bedrock re-judge
        cost_rate_per_hour=A100_80GB_USD_PER_HOUR,
        cost_cap_usd=BASE_K2_CAP_USD,
        rung_gpu_hours=BASE_K2_RUNG_HOURS,
        on_model_done=_results_vol.commit,
    )
    _results_vol.commit()
    _hf_cache.commit()
    return {
        "fit_ok": True,
        "dose": {
            "layer": layer,
            "raw_norm": round(raw_norm, 3),
            "k": CODER_K2_STRENGTH_K,
            "alpha": round(alpha, 3),
            **{k: round(v, 3) for k, v in diag.items()},
        },
        "trials_out": BASE_K2_TRIALS,
        "n_trials_persisted": len(result.records),
        "ran": result.ran,
        "stopped_reason": result.stopped_reason,
        "spent_usd_gpu": round(result.spent_usd, 2),
        "note": "Modal verdicts are placeholders; authoritative = local Bedrock re-judge",
    }


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],  # Qwen ungated; NO judge secret on Modal
    timeout=1800,  # two single forward passes; 30 min bounds cost well under the $3 cap
)
def run_moe_fitcheck(model_id: str = MOE_MODEL_K2) -> dict[str, object]:
    """STEP 1 (issue #39): is the repeng control live on the EXPERTS of an MoE?

    Loads ``model_id`` (Qwen1.5-MoE-A2.7B-Chat) fp16 at the corrected dose
    (raw_norm, k=2, depth 0.61) and reports TWO things on the introspection prompt:

    1. ``verify_injection_delta`` — magnitude_ratio + cosine, i.e. repeng's residual
       arithmetic is live. NECESSARY but architecture-agnostic: it says nothing
       MoE-specific.
    2. ``router_shift`` — the EXPERT-ROUTING DIFF (control OFF vs ON): the fraction
       of (MoE-layer, token) top-k expert SETS that move, and the mean L1 shift in
       the gate softmax, for every MoE layer at/after the injection site. NONZERO
       => injection re-routes which experts fire => the control reaches the MoE
       machinery, not just the dense residual. This is what earns STEP 1.

    Two-phase load (extract -> free -> generator) so only one copy is resident.
    """
    import gc

    import torch

    from introspection_scaling import extract_concept_vector
    from introspection_scaling.harness import RepengGenerator, layer_for_fraction
    from introspection_scaling.runner import _load_causal_lm

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    def _free() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        emodel, etok = _load_causal_lm(model_id, "cuda", "float16", None)
        layer = layer_for_fraction(int(emodel.config.num_hidden_layers))  # depth 0.61
        cv = extract_concept_vector(model_id, "oceans", model=emodel, tokenizer=etok, device="cuda")
        del emodel, etok
        _free()
        if layer not in cv.raw_norms:
            raise RuntimeError(f"raw_norms missing injection layer {layer}")
        raw_norm = float(cv.raw_norms[layer])
        alpha = 2.0 * raw_norm  # raw_norm dose, k=2 (same corrected dose as the _k2 rungs)
        gen = RepengGenerator(
            model_id, device="cuda", dtype="float16", quant=None, max_new_tokens=8
        )
        delta = {k: float(v) for k, v in gen.verify_injection_delta(cv, layer, alpha).items()}
        routing = {k: float(v) for k, v in gen.router_shift(cv, layer, alpha).items()}
        del gen, cv
        _free()
    except Exception as exc:  # noqa: BLE001 - report the probe failure, run nothing
        return {"fit_ok": False, "model_id": model_id, "error": f"{type(exc).__name__}: {exc}"}

    print(
        f"[moe-fitcheck] layer={layer} raw_norm={raw_norm:.3f} alpha={alpha:.3f} "
        f"ratio={delta['magnitude_ratio']:.3f} cos={delta['cosine_to_v_unit']:.3f} "
        f"routing_changed_frac={routing['routing_changed_frac']:.3f} "
        f"gate_l1_shift={routing['gate_l1_shift']:.4f} "
        f"n_moe_positions={routing['n_moe_positions']:.0f}"
    )
    return {
        "fit_ok": True,
        "model_id": model_id,
        "layer": layer,
        "raw_norm": round(raw_norm, 3),
        "k": 2.0,
        "alpha": round(alpha, 3),
        "verify_injection_delta": {k: round(v, 4) for k, v in delta.items()},
        "router_shift": {k: round(v, 4) for k, v in routing.items()},
        "note": "nonzero router_shift => injection perturbs expert routing (live on the experts)",
    }


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],  # Qwen ungated; NO judge secret on Modal
    timeout=2 * 3600,
)
def run_moe_k2(concepts: list[str], seeds: list[int], n_trials: int = 12) -> dict[str, object]:
    """STEP 2 rung (issue #39): GENERATE-ONLY Qwen1.5-MoE-A2.7B-Chat transcripts at
    the corrected paper dose (raw_norm, k=2) — an EXACT mirror of run_base_k2, so the
    MoE sits on the same dose/protocol as the dense rungs. The authoritative judge is
    the LOCAL Bedrock judge (AWS SSO is local); the RuleBasedJudge here is a
    NON-authoritative placeholder so the transcripts persist. Committed result =
    local Bedrock re-judge of these transcripts.

    Fit-check first: log the ACTUAL alpha (= k·‖raw diff-of-means‖ at the 0.61
    layer) and one ``verify_injection_delta`` so the dose is observably LIVE (not a
    no-op / coherence-destroyer) — observe, do NOT tune. $3 cap; separate _k2 path.
    """
    import gc

    import torch

    from introspection_scaling import extract_concept_vector
    from introspection_scaling.harness import RepengGenerator, RuleBasedJudge, layer_for_fraction
    from introspection_scaling.runner import _load_causal_lm
    from introspection_scaling.runner import run_ladder as _run

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # --- FIT CHECK + DOSE OBSERVABILITY (observe, never tune) ---
    # ONE model resident at a time: Phase 1 extract -> free -> Phase 2 generator.
    def _free() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        emodel, etok = _load_causal_lm(MOE_MODEL_K2, "cuda", "float16", None)
        layer = layer_for_fraction(int(emodel.config.num_hidden_layers))  # depth 0.61
        cv = extract_concept_vector(
            MOE_MODEL_K2, "oceans", model=emodel, tokenizer=etok, device="cuda"
        )
        del emodel, etok
        _free()
        if layer not in cv.raw_norms:
            raise RuntimeError(f"raw_norms missing injection layer {layer}")
        raw_norm = float(cv.raw_norms[layer])
        alpha = CODER_K2_STRENGTH_K * raw_norm
        gen = RepengGenerator(
            MOE_MODEL_K2, device="cuda", dtype="float16", quant=None, max_new_tokens=8
        )
        diag = {k: float(v) for k, v in gen.verify_injection_delta(cv, layer, alpha).items()}
        del gen, cv
        _free()
    except Exception as exc:  # noqa: BLE001 - report the fit/dose failure, run no sweep
        return {
            "fit_ok": False,
            "model_id": MOE_MODEL_K2,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(
        f"[dose] layer={layer} raw_norm={raw_norm:.3f} k={CODER_K2_STRENGTH_K} alpha={alpha:.3f} "
        f"ratio={diag['magnitude_ratio']:.3f} cos={diag['cosine_to_v_unit']:.3f}"
    )

    # --- Generate (RuleBasedJudge PLACEHOLDER; transcripts are the payload) ---
    result = _run(
        [MOE_MODEL_K2],
        concepts=concepts,
        seeds=seeds,
        n_trials=n_trials,
        out_path=MOE_K2_RECORDS,
        trials_path=MOE_K2_TRIALS,
        depth_fraction=0.61,
        dose_mode="raw_norm",
        strength_k=CODER_K2_STRENGTH_K,
        device="cuda",
        precision_map=PRECISION_MAP,
        judge=RuleBasedJudge(),  # NON-authoritative; real verdict = local Bedrock re-judge
        cost_rate_per_hour=A100_80GB_USD_PER_HOUR,
        cost_cap_usd=MOE_K2_CAP_USD,
        rung_gpu_hours=MOE_K2_RUNG_HOURS,
        on_model_done=_results_vol.commit,
    )
    _results_vol.commit()
    _hf_cache.commit()
    return {
        "fit_ok": True,
        "dose": {
            "layer": layer,
            "raw_norm": round(raw_norm, 3),
            "k": CODER_K2_STRENGTH_K,
            "alpha": round(alpha, 3),
            **{k: round(v, 3) for k, v in diag.items()},
        },
        "trials_out": MOE_K2_TRIALS,
        "n_trials_persisted": len(result.records),
        "ran": result.ran,
        "stopped_reason": result.stopped_reason,
        "spent_usd_gpu": round(result.spent_usd, 2),
        "note": "Modal verdicts are placeholders; authoritative = local Bedrock re-judge",
    }


def _run_rung_k2(
    model_id: str,
    records_path: str,
    trials_path: str,
    cap_usd: float,
    rung_hours_map: dict[str, float],
    concepts: list[str],
    seeds: list[int],
    n_trials: int,
) -> dict[str, object]:
    """GENERATE-ONLY ``model_id`` transcripts at the corrected paper dose (raw_norm,
    k=2) — the SAME corrected dose as run_base_k2 / run_coder_k2. Shared body for the
    four small-model _k2 rungs (base 7B/14B, Coder-Instruct 7B/14B) so the
    fit-check + dose observability + generate path lives in ONE place — this kills
    the copy-paste bug class the four thin wrappers would otherwise reintroduce.

    The authoritative judge is the LOCAL Bedrock judge (AWS SSO is local, not
    portable to Modal); the RuleBasedJudge here is a NON-authoritative placeholder
    so the transcripts persist. Committed result = local Bedrock re-judge.

    Fit-check first: log the ACTUAL alpha (= k·‖raw diff-of-means‖ at the 0.61
    layer) and one ``verify_injection_delta`` so the dose is observably LIVE (not a
    no-op / coherence-destroyer) — observe, do NOT tune. Per-rung ``cap_usd``;
    SEPARATE per-model ``records_path`` / ``trials_path`` (never cross-wired).
    """
    import gc

    import torch

    from introspection_scaling import extract_concept_vector
    from introspection_scaling.harness import RepengGenerator, RuleBasedJudge, layer_for_fraction
    from introspection_scaling.runner import _load_causal_lm
    from introspection_scaling.runner import run_ladder as _run

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # --- FIT CHECK + DOSE OBSERVABILITY (observe, never tune) ---
    # Two-phase load (extract -> free -> generator + verify -> free), mirroring the
    # 32B rungs. These 7B/14B models fit an 80 GB A100 with room to spare; the phased
    # free is kept only for a uniform code path with run_base_k2 / run_coder_k2.
    def _free() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        emodel, etok = _load_causal_lm(model_id, "cuda", "float16", None)
        layer = layer_for_fraction(int(emodel.config.num_hidden_layers))  # depth 0.61
        cv = extract_concept_vector(model_id, "oceans", model=emodel, tokenizer=etok, device="cuda")
        del emodel, etok
        _free()
        if layer not in cv.raw_norms:
            raise RuntimeError(f"raw_norms missing injection layer {layer}")
        raw_norm = float(cv.raw_norms[layer])
        alpha = CODER_K2_STRENGTH_K * raw_norm
        gen = RepengGenerator(
            model_id, device="cuda", dtype="float16", quant=None, max_new_tokens=8
        )
        diag = {k: float(v) for k, v in gen.verify_injection_delta(cv, layer, alpha).items()}
        del gen, cv
        _free()
    except Exception as exc:  # noqa: BLE001 - report the fit/dose failure, run no sweep
        return {
            "fit_ok": False,
            "model_id": model_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(
        f"[dose] model={model_id} layer={layer} raw_norm={raw_norm:.3f} k={CODER_K2_STRENGTH_K} "
        f"alpha={alpha:.3f} ratio={diag['magnitude_ratio']:.3f} cos={diag['cosine_to_v_unit']:.3f}"
    )

    # --- Generate (RuleBasedJudge PLACEHOLDER; transcripts are the payload) ---
    result = _run(
        [model_id],
        concepts=concepts,
        seeds=seeds,
        n_trials=n_trials,
        out_path=records_path,
        trials_path=trials_path,
        depth_fraction=0.61,
        dose_mode="raw_norm",
        strength_k=CODER_K2_STRENGTH_K,
        device="cuda",
        precision_map=PRECISION_MAP,
        judge=RuleBasedJudge(),  # NON-authoritative; real verdict = local Bedrock re-judge
        cost_rate_per_hour=A100_80GB_USD_PER_HOUR,
        cost_cap_usd=cap_usd,
        rung_gpu_hours=rung_hours_map,
        on_model_done=_results_vol.commit,
    )
    _results_vol.commit()
    _hf_cache.commit()
    return {
        "fit_ok": True,
        "model_id": model_id,
        "dose": {
            "layer": layer,
            "raw_norm": round(raw_norm, 3),
            "k": CODER_K2_STRENGTH_K,
            "alpha": round(alpha, 3),
            **{k: round(v, 3) for k, v in diag.items()},
        },
        "trials_out": trials_path,
        "n_trials_persisted": len(result.records),
        "ran": result.ran,
        "stopped_reason": result.stopped_reason,
        "spent_usd_gpu": round(result.spent_usd, 2),
        "note": "Modal verdicts are placeholders; authoritative = local Bedrock re-judge",
    }


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],  # Qwen ungated; NO judge secret on Modal
    timeout=2 * 3600,
)
def run_base7b_k2(concepts: list[str], seeds: list[int], n_trials: int = 12) -> dict[str, object]:
    """GENERATE-ONLY Qwen2.5-7B BASE transcripts (raw_norm k=2). See _run_rung_k2."""
    m = RUNG_K2_BASE_7B
    return _run_rung_k2(
        m,
        RUNG_K2_RECORDS[m],
        RUNG_K2_TRIALS[m],
        RUNG_K2_CAP_USD,
        {m: RUNG_GPU_HOURS[m]},
        concepts,
        seeds,
        n_trials,
    )


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],  # Qwen ungated; NO judge secret on Modal
    timeout=2 * 3600,
)
def run_base14b_k2(concepts: list[str], seeds: list[int], n_trials: int = 12) -> dict[str, object]:
    """GENERATE-ONLY Qwen2.5-14B BASE transcripts (raw_norm k=2). See _run_rung_k2."""
    m = RUNG_K2_BASE_14B
    return _run_rung_k2(
        m,
        RUNG_K2_RECORDS[m],
        RUNG_K2_TRIALS[m],
        RUNG_K2_CAP_USD,
        {m: RUNG_GPU_HOURS[m]},
        concepts,
        seeds,
        n_trials,
    )


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],  # Qwen ungated; NO judge secret on Modal
    timeout=2 * 3600,
)
def run_coder7b_k2(concepts: list[str], seeds: list[int], n_trials: int = 12) -> dict[str, object]:
    """GENERATE-ONLY Qwen2.5-Coder-7B-Instruct transcripts (raw_norm k=2). See _run_rung_k2."""
    m = RUNG_K2_CODER_7B
    return _run_rung_k2(
        m,
        RUNG_K2_RECORDS[m],
        RUNG_K2_TRIALS[m],
        RUNG_K2_CAP_USD,
        {m: RUNG_GPU_HOURS[m]},
        concepts,
        seeds,
        n_trials,
    )


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],  # Qwen ungated; NO judge secret on Modal
    timeout=2 * 3600,
)
def run_coder14b_k2(concepts: list[str], seeds: list[int], n_trials: int = 12) -> dict[str, object]:
    """GENERATE-ONLY Qwen2.5-Coder-14B-Instruct transcripts (raw_norm k=2). See _run_rung_k2."""
    m = RUNG_K2_CODER_14B
    return _run_rung_k2(
        m,
        RUNG_K2_RECORDS[m],
        RUNG_K2_TRIALS[m],
        RUNG_K2_CAP_USD,
        {m: RUNG_GPU_HOURS[m]},
        concepts,
        seeds,
        n_trials,
    )


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],  # Qwen ungated; NO judge secret on Modal
    timeout=6 * 3600,
)
def run_ladder_k2(concepts: list[str], seeds: list[int], n_trials: int = 12) -> dict[str, object]:
    """GENERATE-ONLY the full Qwen ladder (0.5-32B) at the corrected paper dose
    (raw_norm, k=2). Transcripts judged LOCALLY by the faithful Bedrock judge.

    run_ladder's per-rung two-phase (extract fp16 -> free -> generate) IS the
    one-model-at-a-time fit-check; records + transcripts are written and the volume
    committed AFTER each rung, so an OOM/crash on a later rung keeps the earlier
    rungs' transcripts (partial ladder still judgeable). $15 GPU cost guard.
    """
    from introspection_scaling.harness import RuleBasedJudge
    from introspection_scaling.runner import run_ladder as _run

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    models = list(QWEN_LADDER)  # 0.5/1.5/3/7/14/32B — no Llama, no 72B
    try:
        result = _run(
            models,
            concepts=concepts,
            seeds=seeds,
            n_trials=n_trials,
            out_path=LADDER_K2_RECORDS,
            trials_path=LADDER_K2_TRIALS,
            depth_fraction=0.61,
            dose_mode="raw_norm",
            strength_k=CODER_K2_STRENGTH_K,  # k=2, same a-priori pin as the Coder run
            device="cuda",
            precision_map=PRECISION_MAP,
            judge=RuleBasedJudge(),  # NON-authoritative; real verdict = local Bedrock re-judge
            cost_rate_per_hour=A100_80GB_USD_PER_HOUR,
            cost_cap_usd=LADDER_K2_CAP_USD,
            rung_gpu_hours=LADDER_K2_RUNG_HOURS,
            on_model_done=_results_vol.commit,
        )
    except Exception as exc:  # noqa: BLE001 - persist partial, report which rung broke
        _results_vol.commit()
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "note": "partial ladder persisted to the volume; earlier rungs' transcripts survive",
            "trials_out": LADDER_K2_TRIALS,
        }
    _results_vol.commit()
    _hf_cache.commit()
    return {
        "trials_out": LADDER_K2_TRIALS,
        "n_trials_persisted": len(result.records),
        "ran": result.ran,
        "skipped_by_cost_guard": result.skipped,
        "stopped_reason": result.stopped_reason,
        "spent_usd_gpu": round(result.spent_usd, 2),
        "note": "Modal verdicts are placeholders; authoritative = local Bedrock re-judge",
    }


@app.local_entrypoint()
def ladder_k2(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::ladder_k2` — generate the corrected-dose Qwen ladder
    (raw_norm k=2, $15 GPU cap). Judge LOCALLY afterwards (scripts/ladder_k2_judge.py)."""
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_ladder_k2.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("ladder_k2:", result)


@app.local_entrypoint()
def coder_k2(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::coder_k2` — generate Coder-32B transcripts (raw_norm
    k=2, $3 cap). Judge LOCALLY with Bedrock afterwards (scripts/coder32b_k2_judge.py)."""
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_coder_k2.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("coder_k2:", result)


@app.local_entrypoint()
def base_k2(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::base_k2` — generate Qwen2.5-32B base transcripts
    (raw_norm k=2, $3 cap). Judge LOCALLY with Bedrock afterwards (scripts/base32b_k2_judge.py)."""
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_base_k2.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("base_k2:", result)


@app.local_entrypoint()
def moe_fitcheck(model_id: str = MOE_MODEL_K2) -> None:
    """`modal run modal_app.py::moe_fitcheck` — STEP 1 (issue #39): prove the repeng
    control is live on the EXPERTS of Qwen1.5-MoE-A2.7B-Chat (magnitude_ratio +
    expert-routing diff). No sweep, no judge; ~$2 GPU bound by the 30-min timeout."""
    result = run_moe_fitcheck.remote(model_id)
    print("moe_fitcheck:", result)


@app.local_entrypoint()
def moe_k2(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::moe_k2` — STEP 2 (issue #39): generate Qwen1.5-MoE
    transcripts (raw_norm k=2, $3 cap). Judge LOCALLY with Bedrock afterwards
    (scripts/rung_k2_judge.py results/trials_moe_k2.jsonl)."""
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_moe_k2.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("moe_k2:", result)


@app.local_entrypoint()
def base7b_k2(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::base7b_k2` — generate Qwen2.5-7B base transcripts (raw_norm
    k=2, $3 cap). Judge LOCALLY: scripts/rung_k2_judge.py results/trials_base7b_k2.jsonl."""
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_base7b_k2.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("base7b_k2:", result)


@app.local_entrypoint()
def base14b_k2(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::base14b_k2` — generate Qwen2.5-14B base transcripts (raw_norm
    k=2, $3 cap). Judge LOCALLY: scripts/rung_k2_judge.py results/trials_base14b_k2.jsonl."""
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_base14b_k2.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("base14b_k2:", result)


@app.local_entrypoint()
def coder7b_k2(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::coder7b_k2` — generate Qwen2.5-Coder-7B-Instruct transcripts
    (raw_norm k=2, $3 cap). Judge LOCALLY: scripts/rung_k2_judge.py trials_coder7b_k2.jsonl."""
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_coder7b_k2.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("coder7b_k2:", result)


@app.local_entrypoint()
def coder14b_k2(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::coder14b_k2` — generate Qwen2.5-Coder-14B-Instruct transcripts
    (raw_norm k=2, $3 cap). Judge LOCALLY: scripts/rung_k2_judge.py trials_coder14b_k2.jsonl."""
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_coder14b_k2.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("coder14b_k2:", result)


@app.local_entrypoint()
def calibration(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::calibration` — Coder-32B fp16 calibration ($10 cap).

    Uses the CORRECTED paper dose (dose_mode='raw_norm', k=2 pinned a priori).
    """
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_calibration.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("calibration:", result)


@app.local_entrypoint()
def verify4bit(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
    """`modal run modal_app.py::verify4bit` — DE-RISK gate for the 72B 4-bit anchor."""
    result = verify_4bit.remote(model_id)
    print("4-bit verify:", result)


@app.local_entrypoint()
def anchor72b(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::anchor72b` — 72B 4-bit anchor to records_72b.jsonl."""
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_anchor_72b.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("72B anchor:", result)


@app.local_entrypoint()
def main() -> None:
    """`modal run modal_app.py` — run the GPU smoke check and print the result."""
    result = smoke.remote()
    print("smoke result:", result)
    assert result["hidden_size"] == 896, result  # Qwen2.5-0.5B-Instruct hidden size
    print("OK: image is GPU-capable and loads", result["model_id"])


@app.local_entrypoint()
def ladder(n_concepts: int = 6, n_trials: int = 12) -> None:
    """`modal run modal_app.py::ladder` — the emergence ladder on GPU.

    Config for the flagship run: 6 concepts x 12 trials x 3 seeds. Qwen rungs run
    first (ungated); Llama appended iff the gated-access preflight passes; 72B
    anchor held. The $80 cost guard self-stops + commits a partial curve. Requires
    the two Modal secrets. Writes records.jsonl to the ``introspection-results``
    volume (fetch with ``modal volume get``).
    """
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_ladder.remote(concepts, [0, 1, 2], n_trials=n_trials)
    print("ladder result:", result)
