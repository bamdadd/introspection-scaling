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
* ``run_ladder`` (``modal run modal_app.py::run_ladder``) — the full sweep:
  extract (A1) + inject/judge (A2) across the instruct ladder, writing
  ``SeedRecord`` JSONL to a Modal Volume. Sized **A100-80GB** because both
  extraction and the ControlModel hold the model in float32 (14B fp32 ≈ 56 GB;
  see ``runner.run_ladder`` two-phase docs).

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

# Instruct ladder (chat self-report — base models can't follow the protocol).
DEFAULT_LADDER: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)


@app.function(
    image=_ladder_image,
    gpu="A100-80GB",  # 14B float32 ≈ 56 GB (extract + ControlModel, two-phase)
    volumes={_HF_CACHE_DIR: _hf_cache, _RESULTS_DIR: _results_vol},
    secrets=_SECRETS,
    timeout=24 * 3600,
)
def run_ladder(
    models: list[str],
    concepts: list[str],
    seeds: list[int],
    n_trials: int = 20,
    depth_fraction: float = 0.61,
    dose_fraction: float = 0.044,
) -> dict[str, object]:
    """Run the ladder sweep on GPU and persist ``records.jsonl`` to the volume.

    Delegates all science to ``runner.run_ladder`` (A1 extract + A2 harness); the
    faithful Anthropic judge is used (needs ``ANTHROPIC_API_KEY`` from the
    Anthropic secret) — no silent fallback.
    """
    from introspection_scaling.runner import run_ladder as _run

    os.environ.setdefault("HF_HOME", _HF_CACHE_DIR)
    # huggingface_hub reads HF_TOKEN (canonical) for gated Llama pulls; alias the
    # legacy name so auth works regardless of which the loader/version checks.
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    out = f"{_RESULTS_DIR}/records.jsonl"
    records = _run(
        models,
        concepts=concepts,
        seeds=seeds,
        n_trials=n_trials,
        out_path=out,
        depth_fraction=depth_fraction,
        dose_fraction=dose_fraction,
        device="cuda",
    )
    _results_vol.commit()
    _hf_cache.commit()
    return {"n_records": len(records), "out": out, "models": models}


@app.local_entrypoint()
def main() -> None:
    """`modal run modal_app.py` — run the GPU smoke check and print the result."""
    result = smoke.remote()
    print("smoke result:", result)
    assert result["hidden_size"] == 896, result  # Qwen2.5-0.5B-Instruct hidden size
    print("OK: image is GPU-capable and loads", result["model_id"])


@app.local_entrypoint()
def ladder(n_concepts: int = 10, n_trials: int = 20) -> None:
    """`modal run modal_app.py::ladder` — full instruct-ladder sweep on GPU.

    Requires A2 harness on the path + the two Modal secrets. Writes records.jsonl
    to the ``introspection-results`` volume (fetch with ``modal volume get``).
    """
    from introspection_scaling.extract import CONCEPT_WORDS

    concepts = list(CONCEPT_WORDS[:n_concepts])
    result = run_ladder.remote(list(DEFAULT_LADDER), concepts, [0, 1, 2], n_trials=n_trials)
    print("ladder result:", result)
