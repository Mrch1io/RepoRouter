"""Render figures 2–5: python scripts/plot_report_figures.py."""

from time import perf_counter

from chart_style import BLUE, GRAY, RED, chart_style, plt, save_figure, style_axes


# These are the supplied rounded distances, not estimates from the plot.
DISTANCES = (
    ("Repository questions (40)", BLUE, (
        0.418, 0.483, 0.486, 0.511, 0.512, 0.515, 0.553, 0.583,
        0.600, 0.604, 0.606, 0.615, 0.640, 0.643, 0.645, 0.647,
        0.649, 0.654, 0.656, 0.669, 0.680, 0.684, 0.706, 0.725,
        0.726, 0.732, 0.756, 0.761, 0.771, 0.792, 0.821, 0.826,
        0.856, 0.870, 0.881, 0.885, 0.905, 0.964, 0.980, 1.043,
    )),
    ("Clearly general (28)", RED, (
        0.831, 0.844, 0.902, 0.945, 0.995, 1.002, 1.040, 1.055,
        1.086, 1.089, 1.095, 1.105, 1.113, 1.117, 1.117, 1.128,
        1.135, 1.139, 1.146, 1.178, 1.185, 1.205, 1.209, 1.219,
        1.222, 1.235, 1.256, 1.293,
    )),
    ("Repo-like terms,\ngeneral question (12)", RED, (
        0.573, 0.632, 0.756, 0.761, 0.819, 0.900, 0.950, 1.060,
        1.115, 1.191, 1.210, 1.248,
    )),
    ("Misleading edge cases (10)", RED, (
        0.601, 0.620, 0.633, 0.646, 0.664, 0.761, 0.831, 0.880,
        0.907, 0.941,
    )),
)
THRESHOLD = 0.905
JITTER = (0.0, 0.10, -0.10, 0.20, -0.20)
SOFT_FUSE_TITLE = (
    "The groups overlap: ROC-AUC 0.85; at a 7.5% false-block rate\n"
    "only 66% of off-topic searches are stopped"
)
SOFT_FUSE_XLABEL = (
    "Top-1 distance (squared L2 between unit vectors = 2 − 2·cosθ);\n"
    "right of the line is blocked"
)


def distance_to_cosine(distance):
    return 1 - distance / 2


def cosine_to_distance(cosine):
    return 2 * (1 - cosine)


def build_soft_fuse_figure():
    with chart_style():
        fig, ax = plt.subplots(figsize=(10.8, 7.0), dpi=150)
        fig.subplots_adjust(left=0.31, right=0.975, bottom=0.20, top=0.79)
        fig.text(0.035, 0.965, SOFT_FUSE_TITLE, ha="left", va="top",
                 fontsize=16, fontweight="semibold", linespacing=1.35)

        for index, (_, color, values) in enumerate(DISTANCES):
            positions = [3 - index + JITTER[i % len(JITTER)]
                         for i in range(len(values))]
            ax.scatter(values, positions, s=42, color=color, linewidths=0,
                       zorder=3)

        ax.axvline(THRESHOLD, color="black", linewidth=1.3, zorder=2)
        ax.text(1.30, 0.4,
                "threshold 0.905: blocks 33/50 off-topic\nand 3/40 repository questions",
                color=GRAY, ha="right", va="center", fontsize=10.5,
                linespacing=1.4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 5},
                zorder=4)
        ax.set_xlim(0.38, 1.32)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xticks((0.4, 0.6, 0.8, 1.0, 1.2))
        ax.set_yticks(range(3, -1, -1), [row[0] for row in DISTANCES], fontsize=11.5)
        ax.set_xlabel(SOFT_FUSE_XLABEL, fontsize=11, labelpad=15, linespacing=1.5)
        style_axes(ax, grid="x")
        ax.tick_params(axis="y", pad=18)

        cosine_axis = ax.secondary_xaxis(
            "top", functions=(distance_to_cosine, cosine_to_distance))
        cosine_axis.set_xlabel("Cosine similarity", labelpad=12, color=GRAY)
        cosine_axis.set_xticks((0.8, 0.7, 0.6, 0.5, 0.4))
        cosine_axis.tick_params(length=0, pad=8, colors=GRAY)
        cosine_axis.spines["top"].set_visible(False)
    return fig


def main():
    from plot_ablation_retrieval import build_ablation_figure, build_retrieval_figure
    from plot_e2e_tradeoff import build_e2e_figure

    figures = (
        ("ablation.png", build_ablation_figure),
        ("e2e-tradeoff.png", build_e2e_figure),
        ("retrieval-by-type.png", build_retrieval_figure),
        ("soft-fuse-distance.png", build_soft_fuse_figure),
    )
    for filename, build in figures:
        start = perf_counter()
        path = save_figure(build(), filename)
        print(f"Saved {path} ({perf_counter() - start:.3f}s)")


if __name__ == "__main__":
    main()
