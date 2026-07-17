"""The one hero figure: detection-rate-above-chance vs parameter count.

We plot the *injected* success rate per model (one line per family, log-x =
parameter count) with its bootstrap band shaded, and draw **both** controls —
no-injection and random-direction — as their own per-model bands. We do NOT plot
a subtraction: with two controls a subtraction is ill-defined, and SPEC line 63
requires all three reported on every point. "Above chance" is the visible gap
where the injected band clears both control bands; those models get a filled
marker, the rest an open marker.

Controls are drawn **per model, not as one pooled horizontal band** — the
no-injection false-positive floor may itself scale with model size (SPEC 43-46),
and a flat band would hide that. If the data turns out flat, it will simply look
flat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe; no display required
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from .stats import ModelPoint  # noqa: E402

#: model_id -> (family, nominal parameter count). Nominal = the label size, used
#: only to position points on the log-x axis.
#: Instruct variants: the introspection task is a chat self-report, so base
#: models (which can't follow the protocol) would manufacture false nulls.
KNOWN_MODELS: dict[str, tuple[str, float]] = {
    "Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5", 0.5e9),
    "Qwen/Qwen2.5-1.5B-Instruct": ("Qwen2.5", 1.5e9),
    "Qwen/Qwen2.5-3B-Instruct": ("Qwen2.5", 3.0e9),
    "Qwen/Qwen2.5-7B-Instruct": ("Qwen2.5", 7.0e9),
    "Qwen/Qwen2.5-14B-Instruct": ("Qwen2.5", 14.0e9),
    "Qwen/Qwen2.5-32B-Instruct": ("Qwen2.5", 32.0e9),
    "Qwen/Qwen2.5-72B-Instruct": ("Qwen2.5", 72.0e9),
    "meta-llama/Llama-3.2-1B-Instruct": ("Llama3.x", 1.0e9),
    "meta-llama/Llama-3.2-3B-Instruct": ("Llama3.x", 3.0e9),
    "meta-llama/Llama-3.1-8B-Instruct": ("Llama3.x", 8.0e9),
    # Coder-32B calibration point: own family so it renders distinct from the
    # Instruct line at the same 32B size (fine-tune comparison).
    "Qwen/Qwen2.5-Coder-32B-Instruct": ("Qwen2.5-Coder", 32.0e9),
}

_FAMILY_COLORS: dict[str, str] = {"Qwen2.5": "#1f77b4", "Llama3.x": "#d62728"}
_FALLBACK_COLORS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def _family_and_params(
    model_id: str, registry: Mapping[str, tuple[str, float]]
) -> tuple[str, float]:
    if model_id in registry:
        return registry[model_id]
    raise KeyError(
        f"{model_id!r} not in the model registry; add it to KNOWN_MODELS "
        f"(family, nominal param count) so it can be placed on the log-x axis"
    )


def plot_scaling_curve(
    points: Sequence[ModelPoint],
    out_path: str | Path = "results/scaling-curve.png",
    *,
    registry: Mapping[str, tuple[str, float]] = KNOWN_MODELS,
    title: str = "Introspective detection vs parameter count",
    caption: str | None = None,
    ymax: float | None = None,
) -> Path:
    """Render the hero scaling curve to ``out_path`` (PNG). Returns the path.

    Standalone-readable: the caption text is baked into the figure so the PNG
    explains itself without the paper. Pass ``caption`` to override the default.
    """
    if not points:
        raise ValueError("no model points to plot")

    # Group points by family, ordered by parameter count.
    by_family: dict[str, list[tuple[float, ModelPoint]]] = {}
    for p in points:
        fam, params = _family_and_params(p.model_id, registry)
        by_family.setdefault(fam, []).append((params, p))
    for fam in by_family:
        by_family[fam].sort(key=lambda xp: xp[0])

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fallback = iter(_FALLBACK_COLORS)

    for fam in sorted(by_family):
        color = _FAMILY_COLORS.get(fam) or next(fallback, "#333333")
        xs = [params for params, _ in by_family[fam]]
        pts = [p for _, p in by_family[fam]]

        inj_mean = [p.injected.mean for p in pts]
        inj_lo = [p.injected.ci_low for p in pts]
        inj_hi = [p.injected.ci_high for p in pts]

        # Injected line + bootstrap band.
        ax.plot(xs, inj_mean, "-", color=color, lw=2.0, label=f"{fam} — injected", zorder=3)
        ax.fill_between(xs, inj_lo, inj_hi, color=color, alpha=0.20, zorder=1)

        # Above-chance points get a filled marker; others open.
        for x, p in zip(xs, pts, strict=True):
            filled = p.above_chance
            ax.plot(
                [x],
                [p.injected.mean],
                marker="o",
                markersize=8,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.6,
                zorder=4,
            )

        # Both controls, per model, in the family color but faint + dashed/dotted.
        noinj_mean = [p.no_injection.mean for p in pts]
        rand_mean = [p.random_direction.mean for p in pts]
        ax.plot(
            xs,
            noinj_mean,
            ":",
            color=color,
            lw=1.3,
            alpha=0.7,
            label=f"{fam} — no-injection ctrl",
            zorder=2,
        )
        ax.fill_between(
            xs,
            [p.no_injection.ci_low for p in pts],
            [p.no_injection.ci_high for p in pts],
            color=color,
            alpha=0.07,
            zorder=0,
        )
        ax.plot(
            xs,
            rand_mean,
            "--",
            color=color,
            lw=1.3,
            alpha=0.7,
            label=f"{fam} — random-dir ctrl",
            zorder=2,
        )
        ax.fill_between(
            xs,
            [p.random_direction.ci_low for p in pts],
            [p.random_direction.ci_high for p in pts],
            color=color,
            alpha=0.07,
            zorder=0,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (nominal, log scale)")
    ax.set_ylabel("Introspective detection rate")
    # Default full 0-1 scale; ymax zooms into a near-floor regime (small effects).
    ax.set_ylim((-0.02 * ymax, ymax * 1.02) if ymax is not None else (-0.02, 1.02))
    ax.set_title(title)
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    caption_text = caption or (
        "Injected concept-detection rate (solid, bootstrap 95% band) vs both "
        "controls per model:\nno-injection (dotted) and random-direction "
        "matched-norm (dashed). Filled marker = injected\nband clears BOTH "
        "controls (above chance); open marker = not distinguishable from chance.\n"
        "Rates pooled over concepts; band = percentile bootstrap over ≥3 seeds."
    )
    fig.text(0.5, -0.02, caption_text, ha="center", va="top", fontsize=8, wrap=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


#: Post-training variants of the size trend, in legend order, with plot colors.
#: "coder" = Qwen2.5-Coder-*-Instruct (code-post-trained *instruct*), the
#: code-post-training arm of a same-size base / general-instruct / code-instruct
#: contrast — NOT a base-vs-instruct axis.
_VARIANT_STYLE: dict[str, tuple[str, str]] = {
    "base": ("Qwen2.5 (base)", "#7f7f7f"),
    "instruct": ("Qwen2.5-Instruct (general)", "#1f77b4"),
    "coder": ("Qwen2.5-Coder-Instruct (code)", "#2ca02c"),
}


def plot_variant_trend(
    series: Mapping[str, Sequence[tuple[float, ModelPoint]]],
    out_path: str | Path = "results/scaling_trend_k2.png",
    *,
    n_trials: int = 216,
    title: str = "Introspective detection across size and post-training",
    caption: str | None = None,
) -> Path:
    """Render the honest size-trend hero: discrete (size, variant) points with
    bootstrap 95% CI bars, filled marker = above chance.

    ``series`` maps a variant key (``base`` / ``instruct`` / ``coder``) to its
    ``(param_count_B, ModelPoint)`` points. We deliberately draw **markers only,
    no connecting line** — three points per family do not justify a fitted trend,
    and the paper text explicitly disclaims one. Both controls are 0.000 on every
    rung, so they are shown as a single flat reference at zero rather than nine
    redundant bands. The story is visual: every variant flat-zero at 7B/14B, one
    Coder point lifted at 32B — non-replication made visible.
    """
    if not series:
        raise ValueError("no series to plot")

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    # Dodge the three variants around each size tick so overlapping zeros stay
    # legible on the log-x axis (multiplicative offset = symmetric in log space).
    dodge = {"base": 1.0 / 1.08, "instruct": 1.0, "coder": 1.08}

    for variant, (label, color) in _VARIANT_STYLE.items():
        pts = series.get(variant)
        if not pts:
            continue
        off = dodge.get(variant, 1.0)
        xs = [size * off for size, _ in pts]
        means = [p.injected.mean for _, p in pts]
        lo = [p.injected.mean - p.injected.ci_low for _, p in pts]
        hi = [p.injected.ci_high - p.injected.mean for _, p in pts]
        # CI bars, no line (fmt="none"); markers drawn separately per point so
        # above-chance ones can be filled and the rest left open.
        ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor=color, capsize=4, zorder=2)
        ax.plot([], [], "o", color=color, label=label)  # legend proxy
        for x, (_, p) in zip(xs, pts, strict=True):
            ax.plot(
                [x],
                [p.injected.mean],
                marker="o",
                markersize=9,
                markerfacecolor=color if p.above_chance else "white",
                markeredgecolor=color,
                markeredgewidth=1.8,
                zorder=4,
            )
            if p.injected.mean > 0.0:
                n_succ = round(p.injected.mean * n_trials)
                ax.annotate(
                    f"{n_succ}/{n_trials}",
                    (x, p.injected.mean),
                    textcoords="offset points",
                    xytext=(9, 0),
                    va="center",
                    fontsize=8,
                    color=color,
                )

    ax.axhline(0.0, ls="--", color="#555555", lw=1.2, alpha=0.8, zorder=1)
    ax.annotate(
        "no-injection & random-direction controls: 0.000 on every rung",
        (0.5, 0.0),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=8,
        color="#555555",
    )

    ax.set_xscale("log")
    sizes = sorted({size for pts in series.values() for size, _ in pts})
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:g}B" for s in sizes])
    # Kill the log minor ticks/labels (e.g. "3x10^1") that otherwise collide with
    # the clean 7B/14B/32B major labels.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(min(sizes) / 1.25, max(sizes) * 1.25)
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel(f"Strict correct-id rate (of {n_trials} trials)")
    # Zoom near the floor so the lone ~0.023 Coder-32B point and its CI are visible.
    top = max(
        (p.injected.ci_high for pts in series.values() for _, p in pts),
        default=0.03,
    )
    ax.set_ylim(-0.002, max(0.035, top * 1.25))
    ax.set_title(title)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    caption_text = caption or (
        "Strict correct-identification (coherent AND correct-id) per (size, variant), "
        "with percentile-bootstrap\n95% CI over seeds. Markers only — no trend line is "
        "fitted through three points. Filled = above both\ncontrols; open = not "
        "distinguishable from chance. One above-chance cell (Coder-32B, 5/216); it does "
        "not\nreplicate down the Coder size ladder (Coder-7B 0/216, Coder-14B 1/216). "
        "Coder = Coder-{7,14,32}B-Instruct."
    )
    fig.text(0.5, -0.02, caption_text, ha="center", va="top", fontsize=8, wrap=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
