"""Shared matplotlib styling for the routing evaluation figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


BLUE = "#1f77b4"
RED = "#d62728"
GRAY = "#7f7f7f"
ASSETS = Path(__file__).resolve().parents[1] / "docs/assets"


def chart_style():
    font = (
        "Segoe UI"
        if any(item.name == "Segoe UI" for item in font_manager.fontManager.ttflist)
        else "DejaVu Sans"
    )
    return plt.rc_context({
        "font.family": font,
        "font.size": 11,
        "text.color": GRAY,
        "axes.labelcolor": GRAY,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.unicode_minus": True,
    })


def style_axes(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
        ax.spines[side].set_alpha(0.4)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRAY, alpha=0.18, linewidth=0.8)


def save_figure(fig, filename):
    path = ASSETS / filename
    try:
        ASSETS.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.22,
                    facecolor="white")
    finally:
        plt.close(fig)
    return path
