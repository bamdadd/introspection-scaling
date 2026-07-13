"""Export one diff-of-means concept vector as a .pt dict for external cross-check.

Extracts a single concept on Qwen2.5-7B-Instruct (see extract.py) and saves the
unit-L2 per-layer directions as a torch dict, so orch-2 can run our vector
through their sweep and compare estimators 1:1.

    python scripts/export_concept_vector.py [CONCEPT] [MODEL_ID] [OUT_PATH]

Saved payload (torch.save):
    {
      "concept": str,
      "model_id": str,
      "layer_convention": str,          # block-output index; see extract.py
      "directions": {int: FloatTensor}, # unit-L2, shape (hidden,)
      "raw_norms":  {int: float},       # ||diff-of-means|| (auxiliary)
    }
"""

from __future__ import annotations

import sys

import torch

from introspection_scaling.extract import BASELINE_WORDS, extract_concept_vector

DEFAULT_CONCEPT = "Oceans"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUT = "artifacts/concept_oceans_qwen2.5-7b-instruct.pt"

LAYER_CONVENTION = (
    "key i = unit-L2 diff-of-means at the output of transformer block i "
    "(0-based; hidden_states[i+1]). Feed straight into repeng ControlModel, no offset."
)


def main() -> None:
    concept = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONCEPT
    model_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    out_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT

    print(f"extracting {concept!r} on {model_id} (fp32, CPU, {len(BASELINE_WORDS)} baselines)...")
    cv = extract_concept_vector(model_id, concept, baseline_words=BASELINE_WORDS)

    directions = {layer: torch.from_numpy(vec).clone() for layer, vec in cv.directions.items()}
    payload = {
        "concept": cv.concept,
        "model_id": cv.model_id,
        "layer_convention": LAYER_CONVENTION,
        "directions": directions,
        "raw_norms": cv.raw_norms,
    }
    torch.save(payload, out_path)

    layers = sorted(directions)
    norms = [float(directions[layer].norm()) for layer in layers]
    hidden = directions[layers[0]].numel()
    print(f"saved -> {out_path}")
    print(f"layers {layers[0]}..{layers[-1]} ({len(layers)}), hidden {hidden}")
    print(f"unit-L2 check: min={min(norms):.5f} max={max(norms):.5f}")


if __name__ == "__main__":
    main()
