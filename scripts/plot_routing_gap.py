"""Render the supplied routing results: python scripts/plot_routing_gap.py."""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


BLUE = "#1f77b4"
RED = "#d62728"
GRAY = "#7f7f7f"
OUTPUT = Path(__file__).resolve().parents[1] / "docs/assets/routing-gap.png"

# Preserve the supplied percentages and Wilson bounds, including their rounding.
# Each observation is (count label, accuracy, lower bound, upper bound).
ROWS = (
    ("Qwen3.5-4B base\n(no summary)",
     ("37/50", 74.0, 60.4, 84.1), ("70/87", 80.5, 70.9, 87.4)),
    ("LoRA v1: 497 rows\n(no summary)",
     ("49/50", 98.0, 89.5, 99.6), ("71/87", 81.6, 72.2, 88.4)),
    ("LoRA v2: 637 rows\n(+ repo summary)",
     ("48/50", 96.0, 86.5, 98.9), ("82/87", 94.3, 87.2, 97.5)),
    ("Qwen3.5-9B base\n(+ repo summary)",
     None, ("82/87", 94.3, 87.2, 97.5)),
    ("DeepSeek Flash API\n(+ repo summary)",
     ("48/50", 96.0, 86.5, 98.9), ("83/87", 95.4, 88.8, 98.2)),
)
SERIES = (
    (1, BLUE, "o", 0.23,
     "In-distribution: 50 questions from the same generator as the training data"),
    (2, RED, "s", -0.23,
     "My repository: 87 hand-written RepoRouter questions"),
)
TITLE = (
    "v1 scored 98% on its own test set but 81.6% on my repository;\n"
    "v2 closed most of the gap"
)
XLABEL = "Routing decision accuracy, % (Wilson 95% CI)"


def build_figure():
    font = (
        "Segoe UI"
        if any(item.name == "Segoe UI" for item in font_manager.fontManager.ttflist)
        else "DejaVu Sans"
    )
    with plt.rc_context({
        "font.family": font,
        "font.size": 11,
        "text.color": GRAY,
        "axes.labelcolor": GRAY,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(10.8, 7.4), dpi=150)
        fig.subplots_adjust(left=0.295, right=0.975, bottom=0.28, top=0.865)
        fig.text(0.035, 0.965, TITLE, fontsize=16, fontweight="semibold",
                 ha="left", va="top", linespacing=1.35)

        for column, color, marker, offset, _ in SERIES:
            observations = [(4 - i, row) for i, row in enumerate(ROWS)
                            if row[column] is not None]
            values = [row[column] for _, row in observations]
            y = [position + (offset if row[1] is not None else 0)
                 for position, row in observations]
            x = [value[1] for value in values]
            # One errorbar call per series batches the interval segments.
            ax.errorbar(
                x, y,
                xerr=([value[1] - value[2] for value in values],
                      [value[3] - value[1] for value in values]),
                fmt=marker, color=color, markersize=8, markeredgewidth=1.2,
                elinewidth=1.8, capsize=4, capthick=1.4, linestyle="none",
                zorder=3,
            )
            for value, position in zip(values, y):
                ax.annotate(value[0], (value[1], position), xytext=(0, 8),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=10.5, color=color)

        ax.set_xlim(50, 100)
        ax.set_ylim(-0.65, 4.75)
        ax.set_xticks(range(50, 101, 10))
        ax.set_yticks(range(4, -1, -1), [row[0] for row in ROWS], fontsize=12)
        ax.tick_params(axis="y", length=0, pad=18)
        ax.tick_params(axis="x", length=0, pad=9)
        ax.set_xlabel(XLABEL, labelpad=14, fontsize=11, linespacing=1.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRAY, alpha=0.18, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRAY)
            ax.spines[side].set_alpha(0.4)
            ax.spines[side].set_linewidth(0.8)

        handles = [Line2D([], [], color=color, marker=marker, linestyle="none",
                          markersize=8, label=label)
                   for _, color, marker, _, label in SERIES]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.035, 0.045),
                   frameon=False, fontsize=10.8, handletextpad=0.7,
                   labelspacing=1.1, borderaxespad=0)
    return fig


def main() -> None:
    start = perf_counter()
    fig = build_figure()
    try:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", pad_inches=0.22,
                    facecolor="white")
    finally:
        plt.close(fig)
    print(f"Saved {OUTPUT} ({perf_counter() - start:.3f}s)")


if __name__ == "__main__":
    main()
