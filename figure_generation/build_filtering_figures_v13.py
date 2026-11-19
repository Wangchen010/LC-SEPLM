#!/usr/bin/env python3
"""Build standalone, large-font filtering workflow and retention heatmap."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/figure_source_data/figureS1_retention.csv"
OUT = ROOT / "results/generated_figures/filtering_standalone"

INK = "#242A31"
MUTED = "#6E747B"
LINE = "#C8D0D8"
BLUE = "#2B6A9F"
BLUE_FILL = "#EAF2F8"
GOLD = "#C98A1B"
GOLD_FILL = "#FBF3E3"
GREY = "#68717A"
GREY_FILL = "#F3F5F7"
VIOLET = "#7562A8"
VIOLET_FILL = "#F1EEF8"
GREEN = "#2E8F57"
GREEN_FILL = "#EAF5EF"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 11.2,
        "axes.titlesize": 14.0,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.8,
        "ytick.labelsize": 10.8,
    }
)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("png", 400), ("svg", 300), ("pdf", 300), ("tiff", 600)):
        kwargs = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def arrow(ax, x1: float, x2: float, y: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=16,
            linewidth=1.7, color=MUTED, shrinkA=0, shrinkB=0, zorder=2,
        )
    )


def pill(ax, x: float, y: float, text: str) -> None:
    width = 0.060
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.032), width, 0.064,
            boxstyle="round,pad=0.006,rounding_size=0.025",
            facecolor="white", edgecolor=VIOLET, linewidth=1.0, zorder=5,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=8.8, color=VIOLET, fontweight="bold", zorder=6)


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 3.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.006, 0.988, "a", transform=ax.transAxes,
            ha="left", va="top", fontsize=15.0, fontweight="bold", color=INK)

    ax.text(0.025, 0.955, "Accession and sequence-identity filtering workflow",
            ha="left", va="top", fontsize=15.0, fontweight="bold", color=INK)
    ax.text(0.025, 0.855, "Training–evaluation overlap is removed before downstream assessment",
            ha="left", va="top", fontsize=10.5, color=MUTED)

    centers = [0.105, 0.3025, 0.500, 0.6975, 0.895]
    width, height, bottom = 0.158, 0.46, 0.245
    stages = [
        ("INPUT SPLITS", "Fixed task-specific\ntrain/evaluation splits", BLUE, BLUE_FILL),
        ("ACCESSION FILTER", "Canonical accession\nmatching and removal", GOLD, GOLD_FILL),
        ("SEQUENCE AUDIT", "Exact-sequence hash\noverlap audit", GREY, GREY_FILL),
        ("IDENTITY FILTER", "MMseqs2 search\nqcov ≥ 80%", VIOLET, VIOLET_FILL),
        ("EVALUATION", "Held-out evaluation\nacross 5 seeds", GREEN, GREEN_FILL),
    ]

    for index, (center, stage) in enumerate(zip(centers, stages), start=1):
        upper, body, color, fill = stage
        x = center - width / 2
        ax.add_patch(
            FancyBboxPatch(
                (x, bottom), width, height,
                boxstyle="round,pad=0.010,rounding_size=0.020",
                facecolor=fill, edgecolor=color, linewidth=1.45, zorder=3,
            )
        )
        ax.add_patch(Rectangle((x, bottom + height - 0.080), width, 0.080, facecolor=color, edgecolor="none", alpha=0.92, zorder=4))
        ax.text(center, bottom + height - 0.040, upper, ha="center", va="center",
                fontsize=8.3, color="white", fontweight="bold", zorder=5)
        ax.text(center, bottom + 0.245, body, ha="center", va="center",
                fontsize=9.8, color=INK, fontweight="bold", linespacing=1.12, zorder=5)
        number_x, number_y = x + 0.018, bottom + height + 0.035
        ax.add_patch(Circle((number_x, number_y), 0.029, facecolor=color, edgecolor="white", linewidth=1.1, zorder=6))
        ax.text(number_x, number_y, str(index), ha="center", va="center", fontsize=9.1,
                color="white", fontweight="bold", zorder=7)
        if index < len(centers):
            arrow(ax, x + width + 0.009, centers[index] - width / 2 - 0.009, bottom + height / 2)

    pill(ax, centers[3] - 0.038, bottom + 0.085, "40% ID")
    pill(ax, centers[3] + 0.038, bottom + 0.085, "30% ID")
    ax.text(centers[4], bottom + 0.085, "Seeds: 7, 17, 23, 42, 101",
            ha="center", va="center", fontsize=7.8, color=GREEN, fontweight="bold")

    ax.text(0.5, 0.095,
            "Structural-adaptation proteins are excluded by accession or sequence identity before task-specific evaluation.",
            ha="center", va="center", fontsize=10.0, color=MUTED)
    save_bundle(fig, "Filtering_workflow_standalone")


def prepare_heatmap() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(INPUT)
    protocols = ["Exact accession", "40% identity", "30% identity"]
    tasks = ["EC", "Subcellular", "Membrane", "Strict membrane", "GO-MF", "Pfam family", "Remote homology"]
    detail = data.pivot(index="task_label", columns="protocol", values="retained_fraction").loc[tasks, protocols] * 100

    totals = data.groupby("protocol", as_index=False).agg(
        original=("eval_original", "sum"), retained=("eval_kept", "sum")
    )
    totals["fraction"] = totals.retained / totals.original * 100
    totals = totals.set_index("protocol").loc[protocols]
    pooled = pd.DataFrame([totals.fraction.to_numpy()], index=["Pooled task-level entries"], columns=protocols)
    matrix = pd.concat([detail, pooled])

    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "source_data_retention_by_task.csv", index=False)
    totals.reset_index().to_csv(OUT / "source_data_retention_pooled.csv", index=False)
    matrix.to_csv(OUT / "source_data_retention_heatmap_percent.csv")
    return matrix, totals


def heatmap() -> None:
    matrix, totals = prepare_heatmap()
    values = matrix.to_numpy()
    cmap = LinearSegmentedColormap.from_list(
        "retention_blue", ["#F7FAFC", "#DCEAF4", "#A8C8DE", "#6F9FBE", "#2B6A9F"]
    )
    norm = Normalize(vmin=0, vmax=17)

    fig, ax = plt.subplots(figsize=(7.0, 5.45))
    fig.text(0.075, 0.985, "b", ha="left", va="top",
             fontsize=14.2, fontweight="bold", color=INK)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=False, bottom=False, labeltop=True, labelbottom=False, pad=9)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.tick_params(axis="y", left=False, right=False, length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # White gutters improve cell separation without a heavy table frame.
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(matrix.shape[0] - 1.5, color=INK, linewidth=1.5)

    protocol_order = list(matrix.columns)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = values[row, col]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.60 else INK
            if row == matrix.shape[0] - 1:
                protocol = protocol_order[col]
                retained = int(totals.loc[protocol, "retained"])
                label = f"{value:.1f}%\n(n={retained:,})"
                fontsize = 9.6
                weight = "bold"
            else:
                label = f"{value:.1f}%"
                fontsize = 10.8
                weight = "bold" if value >= 10 else "normal"
            ax.text(col, row, label, ha="center", va="center", fontsize=fontsize,
                    color=color, fontweight=weight, linespacing=1.15)

    cbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.035, shrink=0.86)
    cbar.set_label("Retained evaluations (%)", rotation=90, labelpad=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=3, width=0.8)

    fig.text(0.16, 0.965, "Task-level evaluation retention after strict filtering",
             ha="left", va="top", fontsize=14.2, fontweight="bold", color=INK)
    fig.text(0.16, 0.915, "Cell values are percentages of each task's original evaluation set",
             ha="left", va="top", fontsize=10.1, color=MUTED)
    fig.text(0.16, 0.025, "Pooled counts are task-level evaluation entries, not unique proteins.",
             ha="left", va="bottom", fontsize=9.3, color=MUTED)
    fig.subplots_adjust(left=0.31, right=0.88, top=0.82, bottom=0.12)
    save_bundle(fig, "Filtering_retention_heatmap_standalone")


def write_readme() -> None:
    (OUT / "README.md").write_text(
        """# Filtering figures v13

Two standalone figures replace the former sparse composite:

1. `Filtering_workflow_standalone`: five-stage accession and sequence-identity filtering workflow.
2. `Filtering_retention_heatmap_standalone`: annotated 7-task × 3-protocol heatmap plus pooled task-level entry totals.

Suggested assembled legend:

**Supplementary Figure S1 | Accession-overlap and sequence-identity filtering.** a, Fixed downstream splits were filtered by canonical accession, exact-sequence hash and MMseqs2 sequence identity at 40% and 30% with query coverage of at least 80%. Held-out evaluations used five classifier seeds. b, Percentage of each task's original evaluation set retained under each protocol. The final row pools task-level evaluation entries across tasks; counts are not unique proteins.
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workflow()
    heatmap()
    write_readme()
    print(OUT)


if __name__ == "__main__":
    main()
