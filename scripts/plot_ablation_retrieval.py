"""Render the ablation and retrieval figures from the supplied rounded values."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from chart_style import BLUE, GRAY, RED, chart_style, save_figure, style_axes


ABLATION_LABELS = (
    "v1\n497 rows",
    "+140 boundary\nquestions (637 rows)",
    "+ repo summary\nin training only",
    "+ repo summary\nat inference (v2)",
)
ABLATION_COUNTS = ((71, 73, 69), (78, 83, 81), (81, 79, 80), (82, 83, 83))
ABLATION_STEP_CONTROL = 74
ABLATION_TITLE = (
    "Most of the gain came from 140 boundary questions\n"
    "(3 seeds each, same 87 questions; dots = individual seeds)"
)
RETRIEVAL_LABELS = (
    "Exact keyword\n(n=5)",
    "Chinese semantic\n(n=5)",
    "English semantic\n(n=3)",
    "Chinese–English mixed\n(n=5)",
)
# Preserve the two-decimal category means and three-decimal totals.
RETRIEVAL_SERIES = (
    ("Dense (vector): MRR 0.903 overall", BLUE, (0.90, 0.85, 1.00, 0.90)),
    ("BM25: MRR 0.350 overall", GRAY, (0.71, 0.00, 0.17, 0.45)),
    ("Hybrid (RRF): MRR 0.565 overall", RED, (0.80, 0.40, 0.61, 0.47)),
)
RETRIEVAL_TITLE = (
    "BM25 scored 0 on every Chinese query (its tokenizer drops Chinese),\n"
    "pulling Hybrid below Dense"
)


def build_ablation_figure():
    with chart_style():
        fig, ax = plt.subplots(figsize=(10.8, 7.4), dpi=150)
        fig.subplots_adjust(left=0.105, right=0.975, bottom=0.18, top=0.835)
        fig.text(0.035, 0.965, ABLATION_TITLE, fontsize=16, fontweight="semibold",
                 ha="left", va="top", linespacing=1.35)

        #1.柱高用三个seed的平均值，原始分数留给圆点和范围。
        means = np.mean(ABLATION_COUNTS, axis=1)
        colors = (GRAY, BLUE, BLUE, BLUE)
        # The left-hand lane keeps both reference labels away from the bars.
        ax.bar(range(4), means, width=0.64, color=colors, zorder=3)
        ax.set_xlim(-1.85, 3.6)
        ax.set_ylim(60, 87)
        ax.set_yticks((60, 65, 70, 75, 80, 85, 87))
        ax.set_xticks(range(4), ABLATION_LABELS, fontsize=11.5)
        ax.set_ylabel("Correct routing decisions (of 87)", labelpad=15, fontsize=12)
        ax.tick_params(axis="x", length=0, pad=13)
        ax.tick_params(axis="y", length=0, pad=8)
        style_axes(ax)

        #2.圆点横向错开，文字放在最低分下面，避免互相遮挡。
        for x, (counts, mean, color) in enumerate(zip(ABLATION_COUNTS, means, colors)):
            ax.scatter(x + np.array((-0.2, 0, 0.2)), counts, s=40,
                       color="#1E293B", edgecolors="white", linewidths=1.2, zorder=6)
            ax.text(x, min(counts) - 1.15, f"{mean:.1f}", ha="center", va="top",
                    color="white", fontsize=16, fontweight="bold", zorder=5)
            ax.text(x, min(counts) - 3.2, f"{min(counts)}–{max(counts)}",
                    ha="center", va="top", color="white", fontsize=13, zorder=5)

        #3.单seed的步数对照单独标出，不参与四组平均。
        ax.scatter(-0.48, ABLATION_STEP_CONTROL, marker="D", s=48,
                   facecolors="none", edgecolors=GRAY, linewidths=1.4, zorder=6)
        ax.annotate(f"v1, 240 steps (seed 42): {ABLATION_STEP_CONTROL}",
                    xy=(-0.48, ABLATION_STEP_CONTROL), xytext=(-1.77, 76.0),
                    color=GRAY, fontsize=10.5, ha="left", va="bottom",
                    arrowprops={"arrowstyle": "-", "color": GRAY, "linewidth": 0.8})

        for value, label, offset, alignment in (
            (82, "Qwen3.5-9B base: 82/87 (single run)", -0.23, "top"),
            (83, "DeepSeek Flash API: 83/87 (single run)", 0.23, "bottom"),
        ):
            ax.axhline(value, color=RED, linewidth=1.5, linestyle=(0, (5, 4)), zorder=4)
            ax.text(-1.77, value + offset, label, color=RED, fontsize=10.5,
                    ha="left", va=alignment, zorder=5)
    return fig


def build_retrieval_figure():
    with chart_style():
        fig, ax = plt.subplots(figsize=(10.8, 7.4), dpi=150)
        fig.subplots_adjust(left=0.105, right=0.975, bottom=0.18, top=0.735)
        fig.text(0.035, 0.965, RETRIEVAL_TITLE, fontsize=16, fontweight="semibold",
                 ha="left", va="top", linespacing=1.35)

        positions = np.arange(4)
        width = 0.23
        for index, (label, color, values) in enumerate(RETRIEVAL_SERIES):
            bars = ax.bar(positions + (index - 1) * width, values, width=width,
                          color=color, label=label, zorder=3)
            ax.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                         padding=5, fontsize=11.5, color=color)

        ax.set_xlim(-0.58, 3.58)
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xticks(positions, RETRIEVAL_LABELS, fontsize=11.5)
        ax.set_ylabel("Mean reciprocal rank of the expected file (top 8)",
                      labelpad=15, fontsize=12)
        ax.tick_params(axis="x", length=0, pad=13)
        ax.tick_params(axis="y", length=0, pad=8)
        style_axes(ax)
        # A dedicated header band leaves the upper-right legend clear of all bars.
        fig.legend(*ax.get_legend_handles_labels(), loc="upper right",
                   bbox_to_anchor=(0.975, 0.868), frameon=False, fontsize=10.8,
                   handlelength=1.2, labelspacing=0.6, borderaxespad=0)
    return fig


def main() -> None:
    start = perf_counter()
    for builder, filename in (
        (build_ablation_figure, "ablation.png"),
        (build_retrieval_figure, "retrieval-by-type.png"),
    ):
        path = save_figure(builder(), filename)
        print(f"Saved {path}")
    print(f"Rendered both figures in {perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()
