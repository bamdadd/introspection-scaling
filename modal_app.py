"""Modal app + reproducible pinned image for the model-size ladder.

The image is built **from the committed lockfile** (`uv.lock` + `pyproject.toml`)
via `Image.uv_sync(frozen=True)`, so a run resolves to exactly the pinned deps a
stranger gets from `uv sync --frozen` locally — the lockfile is the
reproducibility guarantee. `--no-install-project` is passed because the build
context carries only the lock + pyproject, not `src/`, and the smoke path needs
only torch/transformers, not our own package.

The full ladder RUNNER is deferred until the A1/A2 interface (ConceptVector +
harness) lands — see SPEC "ladder runner (deferred until interface lands)". This
file currently proves the image works end-to-end on GPU via `smoke`, which loads
Qwen2.5-0.5B on an A100 and returns its hidden size.

Run the smoke check:
    modal run modal_app.py

Pinned (from uv.lock at commit time): torch 2.13.0 (CUDA), transformers 5.13.1,
modal 1.5.2. Python 3.11. Base: modal debian_slim.
"""

from __future__ import annotations

import modal

APP_NAME = "introspection-scaling"

# Base image pinned by Python version; deps pinned by the frozen lockfile.
# uv_sync bundles pyproject.toml + uv.lock from the project dir into the build
# and runs `uv sync --frozen`, so the image == the committed lock.
image = modal.Image.debian_slim(python_version="3.11").uv_sync(
    frozen=True,
    extra_options="--no-install-project",
    uv_version="0.10.8",  # match the local uv that authored the lock
)

app = modal.App(APP_NAME, image=image)

# Cache HF downloads across runs so the ladder doesn't re-download weights.
_hf_cache = modal.Volume.from_name("introspection-hf-cache", create_if_missing=True)
_HF_CACHE_DIR = "/root/.cache/huggingface"

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B"


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


@app.local_entrypoint()
def main() -> None:
    """`modal run modal_app.py` — run the GPU smoke check and print the result."""
    result = smoke.remote()
    print("smoke result:", result)
    assert result["hidden_size"] == 896, result  # Qwen2.5-0.5B hidden size
    print("OK: image is GPU-capable and loads", result["model_id"])
