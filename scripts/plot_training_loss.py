import re
from pathlib import Path
from time import perf_counter

from chart_style import BLUE, GRAY, chart_style, save_figure, style_axes

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/assets/training-loss.png"
LOSS_PATTERN = re.compile(
    r"\{\s*'loss':\s*'([^']+)'[^{}\r\n]*?'epoch':\s*'([^']+)'\s*\}"
)
TITLE = (
    "Training loss ends at about 0.1 for both — generalization does not\n"
    "(v1 vs v2, seeds 43 and 44; loss counts only the routing-JSON tokens)"
)


def read_loss(path : Path)->list[tuple[float, float]]:
    #1.逐行匹配loss和epoch，不把train_loss等汇总记录算进来。
    points = []
    with path.open(encoding="utf-8") as log:
        for line in log:
            for match in LOSS_PATTERN.finditer(line):
                loss, epoch = match.groups()
                points.append((float(epoch), float(loss)))

    #2.没有训练记录时直接报错，避免画出空曲线。
    if not points:
        raise ValueError(f"No training loss records: {path}")
    return points


def build_figure():
    #1.复用现有样式，右侧留出说明的位置。
    with chart_style():
        fig, ax = plt.subplots(figsize=(11.8, 6.4), dpi=150)
        fig.subplots_adjust(left=0.08, right=0.72, bottom=0.14, top=0.80)
        fig.text(0.035, 0.965, TITLE, fontsize=16, fontweight="semibold",
                 ha="left", va="top", linespacing=1.35)

        #2.每个seed保留一条原始曲线，同一版本只显示一次图例。
        for version, color in (("v1", GRAY), ("v2", BLUE)):
            for seed in (43, 44):
                name = f"{version}_s{seed}"
                points = read_loss(ROOT / "data/v2/seed_step" / name / "train.log")
                print(f"{name}/train.log: {len(points)} points, last loss = {points[-1][1]:g}")
                epochs, losses = zip(*points)
                ax.plot(epochs, losses, color=color, linewidth=1.2, alpha=0.85,
                        label=version if seed == 43 else "_nolegend_")

        #3.用epoch对齐不同总步数，说明文字与曲线分开放。
        style_axes(ax)
        ax.set_xlim(0, 3)
        ax.set_ylim(bottom=0)
        ax.set_xticks((0, 0.5, 1, 1.5, 2, 2.5, 3))
        ax.set_xlabel("Epoch", labelpad=12)
        ax.set_ylabel("Training loss", labelpad=12)
        ax.legend(loc="upper right", frameon=False)
        fig.text(0.765, 0.55,
                 "87 repo questions\n(3-seed mean):\nv1 81.6% · v2 95.0%",
                 ha="left", va="center", fontsize=12, linespacing=1.6)
    return fig


def main()->None:
    #1.已有目标图片就停止，避免覆盖历史结果。
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")

    #2.统一保存图片，并打印本次绘图耗时。
    start = perf_counter()
    path = save_figure(build_figure(), OUTPUT.name)
    print(f"Saved {path.relative_to(ROOT)} ({perf_counter() - start:.3f}s)")


if __name__ == "__main__":
    main()
