"""Ladder runner (A3): drive A1 extraction + A2 harness across the model ladder
and emit the ``SeedRecord`` JSONL that ``report.py`` aggregates into the curve.

Orchestration only — the science lives in A1 (`extract`) and A2 (`harness`):

    for each model:                       # instruct ladder (chat self-report)
        Phase 1  extract every concept vector   (one model load, float32)
        free the extraction model               (fp32 14B ≈ 56 GB — see below)
        Phase 2  run_concept per vector          (RepengGenerator, ControlModel)
    write_seed_records(all_trials) -> results/records.jsonl

**Two-phase / memory (load-bearing for the top of the ladder).** Both
`extract.extract_concept_vector` and `harness.RepengGenerator` hold the model in
**float32**. 14B fp32 ≈ 56 GB, so the extraction model and the ControlModel must
**not** be resident together (112 GB > any single A100). We extract all concept
vectors first (they're tiny), free the model, then build the generator — peak
≈ 56 GB, fits an A100-80GB. This is why the Modal ladder is sized 80GB.

**Injection parameters (orch-2):** depth fraction **0.61**, dose fraction
**0.044** (α = 0.044 · measured residual-stream norm). Threaded explicitly —
never the harness defaults (which are 0.5 / 0.044).

**Dependency-injected seam.** The A2 harness (`introspection_scaling.harness`,
pinned agent2 ``49cc0ee``) is not yet merged to main, so it is resolved at call
time (not a static import) and every collaborator is an injectable parameter.
The orchestration therefore imports, type-checks, and unit-tests standalone with
fakes; the real end-to-end integration runs once harness is on the path. A1's
`extract` IS on main and is imported statically.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .extract import (
    CONCEPT_WORDS,
    ConceptVector,
    extract_concept_vector,
    make_random_matched,
)
from .records import SeedRecord

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# orch-2 injection parameters (see module docstring / RESULTS.md).
DEPTH_FRACTION = 0.61
DOSE_FRACTION = 0.044


@runtime_checkable
class DoseGeneratorLike(Protocol):
    """The subset of A2's ``RepengGenerator`` the runner drives."""

    n_layers: int

    def measure_resid_norm(self, layer: int) -> float: ...
    def generate(
        self, inject: ConceptVector | None, layer: int, alpha: float, seed: int
    ) -> str: ...


@runtime_checkable
class JudgeLike(Protocol):
    """The subset of A2's judges the runner needs (``grade`` -> verdict)."""

    def grade(self, concept: str, response: str) -> Any: ...


# Injected collaborators (defaults resolve the real A1/A2 code lazily).
GeneratorFactory = Callable[[str], DoseGeneratorLike]
RunConceptFn = Callable[..., list[Any]]
WriteFn = Callable[[Sequence[Any], str | Path], list[SeedRecord]]
ExtractFn = Callable[..., ConceptVector]


def _harness() -> Any:
    """Resolve the A2 harness at call time (it is not a static dependency).

    Fails loud with an actionable message if harness is not yet on the path —
    aggregate/plot still work standalone, only the sweep needs it.
    """
    import importlib

    try:
        return importlib.import_module("introspection_scaling.harness")
    except ModuleNotFoundError as exc:  # pragma: no cover - trivial guard
        raise RuntimeError(
            "ladder runner needs introspection_scaling.harness (A2, pinned agent2 "
            "49cc0ee) importable. It is not yet merged to main — merge wt/agent2 "
            "first. report.py aggregate/plot work without it."
        ) from exc


def _default_generator(model_id: str, *, device: str) -> DoseGeneratorLike:
    gen: DoseGeneratorLike = _harness().RepengGenerator(model_id, device=device)
    return gen


def _resolve_judge(judge: JudgeLike | None, *, allow_rule_based: bool) -> JudgeLike:
    """Faithful Anthropic judge by default; a NON-faithful fallback only if the
    caller explicitly opts in (SPEC: never grade silently with a fallback)."""
    if judge is not None:
        return judge
    h = _harness()
    try:
        anthropic_judge: JudgeLike = h.AnthropicJudge()
        return anthropic_judge
    except h.MissingJudgeCredentialsError:
        if not allow_rule_based:
            raise
        rule_judge: JudgeLike = h.RuleBasedJudge()
        return rule_judge


def _load_causal_lm(model_id: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load once per model for extraction (float32, matching A1/A2).

    Loading here (rather than letting ``extract_concept_vector`` load per
    concept) means one model load per model, and lets us free it before the
    generator is built — the two-phase memory contract.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model = model.to(torch.device(device))  # type: ignore[arg-type]
    return model, tokenizer


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch always present in this project
        pass


def run_ladder(
    models: Sequence[str],
    *,
    concepts: Sequence[str],
    seeds: Sequence[int],
    n_trials: int,
    out_path: str | Path = "results/records.jsonl",
    depth_fraction: float = DEPTH_FRACTION,
    dose_fraction: float = DOSE_FRACTION,
    device: str = "cpu",
    judge: JudgeLike | None = None,
    allow_rule_based_judge: bool = False,
    # Injected seams — defaults resolve real A1/A2 code (see module docstring).
    extract_fn: ExtractFn = extract_concept_vector,
    make_generator: GeneratorFactory | None = None,
    run_concept_fn: RunConceptFn | None = None,
    write_seed_records_fn: WriteFn | None = None,
    random_matched_fn: Callable[[ConceptVector, int], ConceptVector] = make_random_matched,
    load_model_fn: Callable[[str, str], tuple[Any, Any]] = _load_causal_lm,
) -> list[SeedRecord]:
    """Run the full ladder and write ``SeedRecord`` JSONL to ``out_path``.

    ``models`` are HF ids (instruct variants). ``concepts`` are the words to
    inject (subset of :data:`extract.CONCEPT_WORDS`). ``seeds`` should be ≥3.
    All collaborators are injectable for testing; defaults wire A1 + A2.
    """

    def _default_gen(model_id: str) -> DoseGeneratorLike:
        return _default_generator(model_id, device=device)

    # Resolve collaborators; only touch the (possibly-unmerged) harness when a
    # real default is actually needed, so tests with fakes never import it.
    gen_factory: GeneratorFactory = make_generator if make_generator is not None else _default_gen
    rc_fn: RunConceptFn
    resolved_judge: JudgeLike | None
    if run_concept_fn is None:
        rc_fn = _harness().run_concept
        resolved_judge = _resolve_judge(judge, allow_rule_based=allow_rule_based_judge)
    else:
        rc_fn = run_concept_fn
        resolved_judge = judge
    write_fn: WriteFn = (
        write_seed_records_fn
        if write_seed_records_fn is not None
        else _harness().write_seed_records
    )

    all_trials: list[Any] = []
    for model_id in models:
        # Phase 1: extract every concept vector with a single model load, then free.
        model, tokenizer = load_model_fn(model_id, device)
        concept_vectors = [
            extract_fn(model_id, concept, model=model, tokenizer=tokenizer, device=device)
            for concept in concepts
        ]
        del model, tokenizer
        _free_cuda()

        # Phase 2: inject + sample + judge per concept (fresh ControlModel).
        generator = gen_factory(model_id)
        for cv in concept_vectors:
            all_trials.extend(
                rc_fn(
                    cv,
                    generator=generator,
                    judge=resolved_judge,
                    seeds=seeds,
                    n_trials=n_trials,
                    depth_fraction=depth_fraction,
                    dose_fraction=dose_fraction,
                    random_matched_fn=random_matched_fn,
                )
            )
        del generator
        _free_cuda()

    return write_fn(all_trials, out_path)


def _default_concepts(n: int) -> list[str]:
    """First ``n`` concept words from A1's list (deterministic subset)."""
    return list(CONCEPT_WORDS[:n])


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run the introspection-scaling ladder sweep.")
    ap.add_argument("--models", nargs="+", required=True, help="HF instruct model ids")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--concepts", nargs="+", default=None, help="default: first --n-concepts")
    ap.add_argument("--n-concepts", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/records.jsonl"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth-fraction", type=float, default=DEPTH_FRACTION)
    ap.add_argument("--dose-fraction", type=float, default=DOSE_FRACTION)
    ap.add_argument(
        "--allow-rule-based-judge",
        action="store_true",
        help="opt into the NON-faithful keyword judge when no Anthropic key (flagged, not silent)",
    )
    args = ap.parse_args(argv)

    concepts = args.concepts if args.concepts is not None else _default_concepts(args.n_concepts)
    records = run_ladder(
        args.models,
        concepts=concepts,
        seeds=args.seeds,
        n_trials=args.n_trials,
        out_path=args.out,
        depth_fraction=args.depth_fraction,
        dose_fraction=args.dose_fraction,
        device=args.device,
        allow_rule_based_judge=args.allow_rule_based_judge,
    )
    print(f"wrote {len(records)} seed records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
