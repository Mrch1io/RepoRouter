"""Plot the supplied paired Q4 versus native-call results."""

from time import perf_counter

import matplotlib.pyplot as plt

from chart_style import BLUE, GRAY, RED, chart_style, save_figure, style_axes


TITLE = (
    "Same 90 questions on a clean index, paired: the Q4 router was faster and cheaper;\n"
    "the quality difference is uncertain"
)
FOOTNOTE = (
    "Judged correct: native 81/90, Q4 81/90 (8 only native right, 8 only Q4 right; "
    "exact McNemar p = 1.0).   Searched when required: 32/35 vs 32/35.   "
    "First run (index contained the question files): 83 vs 79, p = 0.34."
)
# Use the user-supplied rounded estimates and bootstrap bounds verbatim.
METRICS = (
    (
        "Latency per question (s):  native 5.43 → Q4 3.97   Δ −1.46 [−1.72, −1.21]",
        -1.46, -1.72, -1.21, (-1.8, 0.45), (-1.5, -1.0, -0.5, 0.0),
    ),
    (
        "DeepSeek API cost per question (US$ × 10⁻⁴):  native 12.15 → Q4 7.46   Δ −4.68 [−5.56, −3.87]",
        -4.68, -5.56, -3.87, (-6.4, 1.6), (-6.0, -4.0, -2.0, 0.0),
    ),
    (
        "Answer length (characters):  native 1606 → Q4 1037   Δ −569 [−664, −477]",
        -569, -664, -477, (-760, 190), (-600, -400, -200, 0),
    ),
    (
        "Judge score (1–5):  native 4.64 → Q4 4.58   Δ −0.07 [−0.26, +0.11]",
        -0.07, -0.26, 0.11, (-0.32, 0.18), (-0.3, -0.2, -0.1, 0.0, 0.1),
    ),
)


def build_e2e_figure():
    with chart_style():
        fig, axes = plt.subplots(4, 1, figsize=(10.8, 8.5), dpi=150)
        fig.subplots_adjust(left=0.055, right=0.975, top=0.815, bottom=0.15,
                            hspace=0.8)
        fig.text(0.035, 0.965, TITLE, ha="left", va="top", fontsize=16,
                 fontweight="semibold", linespacing=1.35)

        for ax, (label, delta, lower, upper, limits, ticks) in zip(axes, METRICS):
            color = RED if lower <= 0 <= upper else BLUE
            style_axes(ax, grid="x")
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_xlim(*limits)
            ax.set_ylim(0, 1)
            ax.set_xticks(ticks)
            ax.tick_params(axis="x", length=0, pad=8, labelsize=10.5)
            ax.axvline(0, color=GRAY, linestyle=(0, (4, 4)), linewidth=1.2,
                       zorder=2)
            ax.errorbar(delta, 0.48, xerr=[[delta - lower], [upper - delta]],
                        fmt="o", color=color, markersize=9, elinewidth=2.2,
                        capsize=5, capthick=1.6, zorder=3)
            ax.set_title(label, loc="left", color=color, fontsize=11.3, pad=13)

        fig.text(0.035, 0.045, FOOTNOTE, ha="left", va="bottom", fontsize=8.7)
    return fig


def main() -> None:
    start = perf_counter()
    path = save_figure(build_e2e_figure(), "e2e-tradeoff.png")
    print(f"Saved {path} ({perf_counter() - start:.3f}s)")


if __name__ == "__main__":
    main()
