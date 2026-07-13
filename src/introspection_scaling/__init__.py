"""introspection-scaling — see README."""

from introspection_scaling.harness import (
    DEPTH_FRACTION_DEFAULT,
    DOSE_FRACTION_CEILING,
    DOSE_FRACTION_DEFAULT,
    AnthropicJudge,
    ConceptVectorLike,
    Condition,
    ConditionRate,
    DoseGenerator,
    JudgeVerdict,
    MissingJudgeCredentialsError,
    RepengGenerator,
    RuleBasedJudge,
    TrialRecord,
    aggregate,
    build_prompt,
    dose_alpha,
    layer_for_fraction,
    run_concept,
    run_conditions,
)

__version__ = "0.0.1"

__all__ = [
    "DEPTH_FRACTION_DEFAULT",
    "DOSE_FRACTION_CEILING",
    "DOSE_FRACTION_DEFAULT",
    "AnthropicJudge",
    "Condition",
    "ConceptVectorLike",
    "ConditionRate",
    "DoseGenerator",
    "JudgeVerdict",
    "MissingJudgeCredentialsError",
    "RepengGenerator",
    "RuleBasedJudge",
    "TrialRecord",
    "__version__",
    "aggregate",
    "build_prompt",
    "dose_alpha",
    "layer_for_fraction",
    "run_concept",
    "run_conditions",
]
