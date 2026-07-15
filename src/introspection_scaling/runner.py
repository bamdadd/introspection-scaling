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

**Injection parameters:** depth fraction **0.61**; dose defaults to the paper's
absolute strength **α = k · ‖raw diff-of-means‖** with **k = 2** (``dose_mode=
'raw_norm'``, the corrected/published regime). The superseded residual-relative
dose (``'resid_frac'``, α = 0.044·‖resid‖) stays available as an explicit option.

**Dependency-injected seam.** The real A1 (`extract`) and A2 (`harness`) callables
are the defaults; every collaborator is also an injectable parameter, so the
orchestration unit-tests with fakes (no models / GPU) while production wires the
live `extract_concept_vector` + `run_concept` + `write_seed_records`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .extract import (
    BASELINE_WORDS,
    CONCEPT_WORDS,
    ConceptVector,
    extract_concept_vector,
    load_baseline_words,
    make_random_matched,
)
from .harness import (
    DEFAULT_STRENGTH_K,
    DOSE_MODE_DEFAULT,
    AnthropicJudge,
    MissingJudgeCredentialsError,
    RepengGenerator,
    RuleBasedJudge,
    generate_concept_completions,
    judge_completions,
    write_seed_records,
)
from .records import SeedRecord, write_trial_records

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# orch-2 injection parameters (see module docstring / RESULTS.md).
DEPTH_FRACTION = 0.61
# Dose defaults. DEFAULT = 'raw_norm' (paper) = STRENGTH_K * ||raw diff-of-means||,
# STRENGTH_K = 2 (paper canonical self-report strength). 'resid_frac' (superseded
# steerbench regime) = DOSE_FRACTION * resid norm, available as an explicit option.
DOSE_MODE = DOSE_MODE_DEFAULT
DOSE_FRACTION = 0.044
STRENGTH_K = DEFAULT_STRENGTH_K

#: model_id -> (dtype, quant). dtype ∈ {float16, bfloat16, float32}; quant ∈
#: {None, "nf4"}. This is the SHARED CONTRACT threaded to A1 `extract` (via the
#: extraction model load, which A3 owns) and A2 `RepengGenerator` (which A2 must
#: teach to accept ``dtype`` / ``quant`` — pending). Default per model is
#: ("float32", None), preserving prior behaviour when no map is given.
PrecisionMap = Mapping[str, "tuple[str, str | None]"]
_DEFAULT_PRECISION: tuple[str, str | None] = ("float32", None)


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


# Injected collaborators (defaults are the real A1/A2 callables).
# GeneratorFactory takes (model_id, dtype, quant) so precision is per model.
GeneratorFactory = Callable[[str, str, "str | None"], DoseGeneratorLike]
GenerateCompletionsFn = Callable[..., list[Any]]  # GPU phase -> Completions
JudgeCompletionsFn = Callable[..., list[Any]]  # off-GPU phase -> TrialRecords
WriteFn = Callable[..., list[SeedRecord]]
ExtractFn = Callable[..., ConceptVector]


@dataclass
class LadderRun:
    """Outcome of :func:`run_ladder` — enough to report a partial curve honestly."""

    records: list[SeedRecord]
    ran: list[str] = field(default_factory=list)  # models completed
    skipped: list[str] = field(default_factory=list)  # models not started (cost stop)
    stopped_reason: str | None = None  # set iff the cost guard tripped
    spent_usd: float = 0.0  # measured GPU $ at the pinned rate


def _default_generator(
    model_id: str, dtype: str, quant: str | None, *, device: str
) -> DoseGeneratorLike:
    """Build A2's RepengGenerator at the requested precision (SHARED CONTRACT).

    ``dtype`` / ``quant`` are now first-class on RepengGenerator, so no fp32
    fallback exists: the requested precision is used or the load raises.
    """
    gen: DoseGeneratorLike = RepengGenerator(model_id, device=device, dtype=dtype, quant=quant)
    return gen


def _resolve_judge(judge: JudgeLike | None, *, allow_rule_based: bool) -> JudgeLike:
    """Faithful Anthropic judge by default; a NON-faithful fallback only if the
    caller explicitly opts in (SPEC: never grade silently with a fallback)."""
    if judge is not None:
        return judge
    try:
        anthropic_judge: JudgeLike = AnthropicJudge()
        return anthropic_judge
    except MissingJudgeCredentialsError:
        if not allow_rule_based:
            raise
        rule_judge: JudgeLike = RuleBasedJudge()
        return rule_judge


def _load_causal_lm(
    model_id: str, device: str, dtype: str = "float32", quant: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load once per model for extraction at the requested precision (A3 owns this).

    Loading here (rather than letting ``extract_concept_vector`` load per concept)
    means one model load per model, and lets us free it before the generator is
    built — the two-phase memory contract. ``extract_concept_vector`` uses the
    passed-in model as-is, so its extraction dtype is whatever we load here.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"unknown dtype {dtype!r}; expected float16 | bfloat16 | float32")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if quant == "nf4":
        # 4-bit nf4 (72B anchor). Needs bitsandbytes + CUDA; device_map handles
        # placement, so we do NOT .to(device) afterwards.
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - dep not yet in lock
            raise RuntimeError(
                "nf4 quantization needs bitsandbytes (not in the lockfile). The 72B "
                "anchor is HELD pending A2's DE-RISK verdict on 4-bit injection — do "
                "not run this rung until both land."
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype
        )
        kwargs["device_map"] = {"": device}
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return model, tokenizer

    if quant is not None:
        raise ValueError(f"unknown quant {quant!r}; expected None | 'nf4'")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = model.to(torch.device(device))  # type: ignore[arg-type]
    return model, tokenizer


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch always present in this project
        pass


def _noop() -> None:  # default on_model_done
    pass


def run_ladder(
    models: Sequence[str],
    *,
    concepts: Sequence[str],
    baseline_words: tuple[str, ...] = BASELINE_WORDS,
    seeds: Sequence[int],
    n_trials: int,
    out_path: str | Path = "results/records.jsonl",
    # Raw per-trial layer (transcripts + verdicts incl parse_error) for audit +
    # offline re-judge. Written beside the counts when set. None = counts only.
    trials_path: str | Path | None = None,
    depth_fraction: float = DEPTH_FRACTION,
    dose_mode: str = DOSE_MODE,
    dose_fraction: float = DOSE_FRACTION,
    strength_k: float = STRENGTH_K,
    device: str = "cpu",
    judge: JudgeLike | None = None,
    allow_rule_based_judge: bool = False,
    precision_map: PrecisionMap | None = None,
    # Cost guard (money): stop BEFORE a rung whose projected total would breach the
    # cap; incremental write + on_model_done after each rung preserve a partial curve.
    cost_rate_per_hour: float | None = None,
    cost_cap_usd: float | None = None,
    rung_gpu_hours: Mapping[str, float] | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_model_done: Callable[[], None] = _noop,
    # Injected seams — defaults are the real A1/A2 callables (see module docstring).
    extract_fn: ExtractFn = extract_concept_vector,
    make_generator: GeneratorFactory | None = None,
    generate_completions_fn: GenerateCompletionsFn = generate_concept_completions,
    judge_completions_fn: JudgeCompletionsFn = judge_completions,
    write_seed_records_fn: WriteFn = write_seed_records,
    write_trials_fn: Callable[..., Any] = write_trial_records,
    random_matched_fn: Callable[[ConceptVector, int], ConceptVector] = make_random_matched,
    load_model_fn: Callable[..., tuple[Any, Any]] = _load_causal_lm,
) -> LadderRun:
    """Run the ladder (ascending order recommended) and write records incrementally.

    ``models`` are HF ids (instruct variants). ``concepts`` are the words to inject
    (subset of :data:`extract.CONCEPT_WORDS`). ``seeds`` should be ≥3. Per model,
    ``precision_map`` gives ``(dtype, quant)`` (default float32/None).

    Cost guard: when ``cost_rate_per_hour`` + ``cost_cap_usd`` are set, before each
    rung we project ``spent + rung_gpu_hours[model]·rate`` and STOP (committing the
    partial curve) if it would exceed the cap. Records are written and
    ``on_model_done`` (e.g. a Modal volume commit) is called after every rung, so a
    stop or crash never loses completed rungs.
    """

    def _default_gen(model_id: str, dtype: str, quant: str | None) -> DoseGeneratorLike:
        return _default_generator(model_id, dtype, quant, device=device)

    gen_factory: GeneratorFactory = make_generator if make_generator is not None else _default_gen
    resolved_judge = _resolve_judge(judge, allow_rule_based=allow_rule_based_judge)
    gen_fn = generate_completions_fn
    judge_fn = judge_completions_fn
    write_fn = write_seed_records_fn
    prec = precision_map or {}
    rung_h = rung_gpu_hours or {}
    guard_on = cost_rate_per_hour is not None and cost_cap_usd is not None

    model_list = list(models)
    all_trials: list[Any] = []
    ran: list[str] = []
    stopped_reason: str | None = None
    t0 = clock()

    def _spent() -> float:
        return (clock() - t0) / 3600.0 * cost_rate_per_hour if cost_rate_per_hour else 0.0

    for model_id in model_list:
        # Cost guard: project this rung BEFORE paying for it.
        if guard_on:
            assert cost_cap_usd is not None and cost_rate_per_hour is not None
            spent = _spent()
            rung_cost = rung_h.get(model_id, 0.0) * cost_rate_per_hour
            if spent + rung_cost > cost_cap_usd:
                stopped_reason = (
                    f"cost guard: projected ${spent + rung_cost:.2f} (spent ${spent:.2f} "
                    f"+ est ${rung_cost:.2f} for {model_id}) > cap ${cost_cap_usd:.2f}"
                )
                print(f"[cost-guard] STOP before {model_id}: {stopped_reason}")
                break

        dtype, quant = prec.get(model_id, _DEFAULT_PRECISION)
        spent_before = _spent()

        # Phase 1: extract every concept vector with a single model load, then free.
        model, tokenizer = load_model_fn(model_id, device, dtype, quant)
        concept_vectors = [
            extract_fn(
                model_id,
                concept,
                baseline_words=baseline_words,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            for concept in concepts
        ]
        del model, tokenizer
        _free_cuda()

        # Phase 2a (GPU): inject + sample all completions per concept, then FREE
        # the GPU before any (network-bound) judging.
        generator = gen_factory(model_id, dtype, quant)
        completions: list[Any] = []
        for cv in concept_vectors:
            completions.extend(
                gen_fn(
                    cv,
                    generator=generator,
                    seeds=seeds,
                    n_trials=n_trials,
                    depth_fraction=depth_fraction,
                    dose_mode=dose_mode,
                    dose_fraction=dose_fraction,
                    strength_k=strength_k,
                    random_matched_fn=random_matched_fn,
                )
            )
        del generator
        _free_cuda()

        # Phase 2b (off-GPU): grade the completions concurrently — no GPU held.
        all_trials.extend(judge_fn(completions, judge=resolved_judge))
        ran.append(model_id)

        # Incremental write + commit so a later stop/crash keeps this rung. Write
        # BOTH the counts and (when requested) the re-judgeable raw per-trial layer.
        write_fn(all_trials, out_path)
        if trials_path is not None:
            write_trials_fn(all_trials, trials_path)
        on_model_done()
        if cost_rate_per_hour is not None:
            spent_after = _spent()
            print(
                f"[cost] {model_id} ({dtype}"
                + (f"/{quant}" if quant else "")
                + f") done: ${spent_after - spent_before:.2f} | running total "
                f"${spent_after:.2f}" + (f" / cap ${cost_cap_usd:.2f}" if cost_cap_usd else "")
            )

    records = write_fn(all_trials, out_path)
    if trials_path is not None:
        write_trials_fn(all_trials, trials_path)
    return LadderRun(
        records=records,
        ran=ran,
        skipped=model_list[len(ran) :] if stopped_reason else [],
        stopped_reason=stopped_reason,
        spent_usd=_spent(),
    )


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
    ap.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help="newline-delimited baseline words; blank lines and # comments are ignored",
    )
    ap.add_argument("--out", type=Path, default=Path("results/records.jsonl"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth-fraction", type=float, default=DEPTH_FRACTION)
    ap.add_argument(
        "--dose-mode",
        choices=["resid_frac", "raw_norm"],
        default=DOSE_MODE,
        help="raw_norm (default)=k*||raw diff|| (paper); resid_frac=fraction*resid_norm",
    )
    ap.add_argument("--dose-fraction", type=float, default=DOSE_FRACTION)
    ap.add_argument(
        "--strength-k",
        type=float,
        default=STRENGTH_K,
        help="raw_norm dose strength (paper canonical = 2); ignored in resid_frac mode",
    )
    ap.add_argument(
        "--allow-rule-based-judge",
        action="store_true",
        help="opt into the NON-faithful keyword judge when no Anthropic key (flagged, not silent)",
    )
    args = ap.parse_args(argv)

    concepts = args.concepts if args.concepts is not None else _default_concepts(args.n_concepts)
    baseline_words = (
        load_baseline_words(args.baseline_file)
        if args.baseline_file is not None
        else BASELINE_WORDS
    )
    result = run_ladder(
        args.models,
        concepts=concepts,
        baseline_words=baseline_words,
        seeds=args.seeds,
        n_trials=args.n_trials,
        out_path=args.out,
        depth_fraction=args.depth_fraction,
        dose_mode=args.dose_mode,
        dose_fraction=args.dose_fraction,
        strength_k=args.strength_k,
        device=args.device,
        allow_rule_based_judge=args.allow_rule_based_judge,
    )
    print(f"wrote {len(result.records)} seed records -> {args.out}  (ran: {result.ran})")
    if result.stopped_reason:
        print(f"STOPPED: {result.stopped_reason}; skipped {result.skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
