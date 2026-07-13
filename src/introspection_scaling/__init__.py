"""introspection-scaling — see README."""

from introspection_scaling.harness import (
    AnthropicJudge,
    ConceptVectorLike,
    Condition,
    ConditionRate,
    JudgeVerdict,
    MissingJudgeCredentialsError,
    RepengGenerator,
    RuleBasedJudge,
    TrialRecord,
    aggregate,
    build_prompt,
    default_injection_layer,
    run_conditions,
)

__version__ = "0.0.1"

__all__ = [
    "AnthropicJudge",
    "Condition",
    "ConceptVectorLike",
    "ConditionRate",
    "JudgeVerdict",
    "MissingJudgeCredentialsError",
    "RepengGenerator",
    "RuleBasedJudge",
    "TrialRecord",
    "__version__",
    "aggregate",
    "build_prompt",
    "default_injection_layer",
    "run_conditions",
]
