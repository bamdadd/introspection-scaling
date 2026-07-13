"""DE-RISK: does repeng ControlModel injection still work under 4-bit (nf4)?

Gates the 70B rung. NF4 quantizes only the Linear *weights* to 4-bit; the residual
stream stays in the fp16/bf16 compute dtype, so `h += alpha * v_unit` should still
apply. This confirms it EMPIRICALLY: load a small model in nf4 and check
`verify_injection_delta` reports magnitude_ratio ~1 and a sane cosine, matching an
fp16 baseline on the same model.

Requires a CUDA GPU + bitsandbytes (cannot run on CPU/MPS). Run on the Modal/A3
GPU box BEFORE committing the 72B-4bit anchor:

    python scripts/verify_4bit_injection.py            # default Qwen2.5-0.5B-Instruct

PASS = nf4 magnitude_ratio in ~[0.8, 1.3] and cosine within ~0.1 of the fp16 run.
FAIL (ratio ~0, or cosine collapses) => injection breaks under 4-bit; drop the
72B-4bit anchor and flag orch-2.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from introspection_scaling import extract_concept_vector  # noqa: E402
from introspection_scaling.harness import (  # noqa: E402
    RepengGenerator,
    dose_alpha,
    layer_for_fraction,
)

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
CONCEPT = "oceans"


def run(dtype: str, quant: str | None) -> dict[str, float]:
    gen = RepengGenerator(MODEL_ID, device="cuda", dtype=dtype, quant=quant, max_new_tokens=8)
    cv = extract_concept_vector(MODEL_ID, CONCEPT, device="cuda")
    layer = layer_for_fraction(gen.n_layers)
    resid_norm = gen.measure_resid_norm(layer)
    alpha = dose_alpha(resid_norm, 0.044)
    diag = gen.verify_injection_delta(cv, layer, alpha)
    tag = dtype + (f"+{quant}" if quant else "")
    print(
        f"[{tag}] layer={layer} resid_norm={resid_norm:.2f} alpha={alpha:.3f} "
        f"ratio={diag['magnitude_ratio']:.3f} cos={diag['cosine_to_v_unit']:.3f} "
        f"first_idx={int(diag['first_changed_index'])}"
    )
    return diag


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA. nf4 needs a GPU + bitsandbytes; run on the Modal/A3 box.")
        return 2
    fp16 = run("float16", None)
    nf4 = run("float16", "nf4")

    ratio_ok = 0.8 <= nf4["magnitude_ratio"] <= 1.3
    cos_ok = abs(nf4["cosine_to_v_unit"] - fp16["cosine_to_v_unit"]) < 0.15
    ok = ratio_ok and cos_ok and np.isfinite(nf4["delta_norm"])
    print(f"\nnf4 ratio_ok={ratio_ok} cos_ok={cos_ok}")
    print(
        "4-BIT INJECTION VERDICT:",
        "PASS — injection survives nf4" if ok else "FAIL — drop 72B-4bit",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
