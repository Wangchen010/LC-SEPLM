"""Draw a clean, fully vector LC-SEPLM architecture schematic."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/generated_figures"
STEM = OUT / "Figure1_LC_SEPLM_clean_architecture"

BLUE = "#2B6A9F"
BLUE_FILL = "#E9F2F8"
TEAL = "#2C8C87"
TEAL_FILL = "#E8F4F1"
PURPLE = "#7562A8"
PURPLE_FILL = "#F0EDF7"
ORANGE = "#D59121"
ORANGE_FILL = "#FBF3E2"
RED = "#B94A48"
RED_FILL = "#F9EDEC"
GREY = "#69717A"
GREY_FILL = "#F4F5F6"
DARK = "#23282F"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
    }
)


def rounded_box(ax, x, y, w, h, text, *, fc, ec, fontsize=6.7, weight="normal", pad=0.012, linespacing=1.15):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size=0.014",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=DARK,
        linespacing=linespacing,
        zorder=3,
    )
    return patch


def module_box(ax, x, y, w, h, title, body="", *, fc, ec, title_y=0.72, body_y=0.38, body_fs=5.8):
    """Draw a module with a uniform 6.8-pt heading and 5.8-pt body."""
    patch = rounded_box(ax, x, y, w, h, "", fc=fc, ec=ec)
    ax.text(
        x + w / 2,
        y + h * title_y,
        title,
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=DARK,
        linespacing=1.08,
        zorder=5,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * body_y,
            body,
            ha="center",
            va="center",
            fontsize=body_fs,
            color=DARK,
            linespacing=1.12,
            zorder=5,
        )
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=GREY, style="-", lw=1.15, mutation=9, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def sequence_tokens(ax, x, y, n=7, radius=0.0105, gap=0.026):
    colors = [BLUE_FILL, TEAL_FILL, ORANGE_FILL, PURPLE_FILL, BLUE_FILL, RED_FILL, TEAL_FILL]
    letters = list("MKLPYDC")
    for i in range(n):
        cx = x + i * gap
        ax.add_patch(Circle((cx, y), radius, facecolor=colors[i], edgecolor=GREY, linewidth=0.6, zorder=4))
        ax.text(cx, y, letters[i], ha="center", va="center", fontsize=5.2, color=DARK, zorder=5)
    ax.text(x + n * gap + 0.002, y, "…", ha="left", va="center", fontsize=8, color=DARK)


def residue_matrix(ax, x, y, w, h, rows=5, cols=8):
    cell_w, cell_h = w / cols, h / rows
    palette = ["#EEF3F7", "#DCE8F1", "#C7DCEA", "#E8F2EF", "#D6E9E4"]
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (x + c * cell_w, y + r * cell_h),
                    cell_w,
                    cell_h,
                    facecolor=palette[(r + 2 * c) % len(palette)],
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
            )
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=GREY, linewidth=0.7, zorder=4))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.55), facecolor="white")
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.96])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel containers.
    ax.add_patch(FancyBboxPatch((0.01, 0.49), 0.98, 0.49, boxstyle="round,pad=0.008,rounding_size=0.018", facecolor="white", edgecolor="#C9D6E2", linewidth=1.0))
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.42, boxstyle="round,pad=0.008,rounding_size=0.018", facecolor="white", edgecolor="#C9D6E2", linewidth=1.0))
    ax.text(0.025, 0.955, "a", fontsize=10, fontweight="bold", va="top")
    ax.text(0.055, 0.955, "Contact-supervised adaptation", fontsize=8.4, fontweight="bold", va="top", color=DARK)
    ax.text(0.025, 0.415, "b", fontsize=10, fontweight="bold", va="top")
    ax.text(0.055, 0.415, "Sequence-only downstream inference", fontsize=8.4, fontweight="bold", va="top", color=DARK)

    # Panel a: five bounded modules.
    xs = [0.035, 0.225, 0.415, 0.615, 0.815]
    ws = [0.145, 0.145, 0.155, 0.155, 0.145]
    y, h = 0.60, 0.245
    module_box(ax, xs[0], y, ws[0], h, "AF2–Swiss-Prot", "structure + sequence\n500k training proteins", fc=BLUE_FILL, ec=BLUE, title_y=0.70, body_y=0.43)
    sequence_tokens(ax, xs[0] + 0.018, y + 0.055, n=5, radius=0.008, gap=0.022)
    module_box(ax, xs[1], y, ws[1], h, "ESM2-150M\n+ Tail-8 LoRA", "$H\\in\\mathbb{R}^{L\\times640}$", fc=TEAL_FILL, ec=TEAL, title_y=0.67, body_y=0.35)
    module_box(ax, xs[2], y, ws[2], h, "Pair descriptor", "$x_{ij}=[h_i;h_j;|h_i-h_j|;$\n$h_i\\odot h_j;\\Delta_{ij}]$\n$x_{ij}\\in\\mathbb{R}^{2561}$", fc=ORANGE_FILL, ec=ORANGE, title_y=0.72, body_y=0.39, body_fs=5.6)
    module_box(ax, xs[3], y, ws[3], h, "Pair-to-sequence\ncross-attention", "$Q=q_{ij}\\in\\mathbb{R}^{640}$\n$K=V=H$\n$r_{ij}\\in\\mathbb{R}^{640}$", fc=PURPLE_FILL, ec=PURPLE, title_y=0.72, body_y=0.36)
    module_box(ax, xs[4], y, ws[4], h, "Contact predictor", "$\\hat{p}_{ij}=\\sigma(\\mathrm{MLP}(r_{ij}))$\n$\\mathcal{L}_{contact}+0.6\\mathcal{L}_{MLM}$", fc=RED_FILL, ec=RED, title_y=0.72, body_y=0.38, body_fs=5.6)
    for i in range(4):
        arrow(ax, xs[i] + ws[i] + 0.006, y + h / 2, xs[i + 1] - 0.006, y + h / 2)

    # Training-only structural label path, placed below the modules and kept inside panel a.
    label_y = 0.545
    ax.text(0.075, label_y + 0.018, "training-only label", fontsize=5.8, color=RED, ha="center")
    ax.text(0.245, label_y + 0.018, r"$y_{ij}=\mathbb{I}(|i-j|\geq12,\ d_{ij}^{C_\alpha}\leq8\ \AA)$", fontsize=6.2, color=RED, ha="center")
    arrow(ax, 0.105, y - 0.005, 0.105, label_y + 0.035, color=RED, style="--", lw=1.0, mutation=8)
    arrow(ax, 0.36, label_y + 0.014, 0.885, y - 0.004, color=RED, style="--", lw=1.0, mutation=8, connectionstyle="arc3,rad=-0.16")

    # Panel b: sequence-only flow with explicit mean pooling.
    y2, h2 = 0.12, 0.18
    x2 = [0.04, 0.205, 0.385, 0.53, 0.675, 0.82]
    w2 = [0.12, 0.13, 0.10, 0.105, 0.10, 0.135]
    module_box(ax, x2[0], y2, w2[0], h2, "Amino-acid\nsequence", "$S$", fc=BLUE_FILL, ec=BLUE, title_y=0.69, body_y=0.42)
    sequence_tokens(ax, x2[0] + 0.015, y2 + 0.045, n=4, radius=0.0075, gap=0.021)
    module_box(ax, x2[1], y2, w2[1], h2, "Adapted ESM2\n+ Tail-8 LoRA", fc=TEAL_FILL, ec=TEAL, title_y=0.52)
    rounded_box(ax, x2[2], y2, w2[2], h2, "", fc=GREY_FILL, ec=GREY, fontsize=6.4, weight="bold")
    ax.text(x2[2] + w2[2] / 2, y2 + 0.142, "Residue states", ha="center", va="center", fontsize=6.4, fontweight="bold", color=DARK, zorder=5)
    residue_matrix(ax, x2[2] + 0.018, y2 + 0.063, 0.064, 0.048)
    ax.text(x2[2] + w2[2] / 2, y2 + 0.032, r"$H\in\mathbb{R}^{L\times640}$", ha="center", va="center", fontsize=5.8, color=DARK, zorder=5)
    module_box(ax, x2[3], y2, w2[3], h2, "Mean pooling", "valid residues only", fc=ORANGE_FILL, ec=ORANGE, title_y=0.62, body_y=0.37)
    module_box(ax, x2[4], y2, w2[4], h2, "Protein\nembedding", "$z\\in\\mathbb{R}^{640}$", fc=PURPLE_FILL, ec=PURPLE, title_y=0.66, body_y=0.34)
    module_box(ax, x2[5], y2, w2[5], h2, "Downstream heads", "EC · GO-MF · Pfam\nremote homology\nlocalization", fc=BLUE_FILL, ec=BLUE, title_y=0.73, body_y=0.38, body_fs=5.5)
    for i in range(5):
        arrow(ax, x2[i] + w2[i] + 0.006, y2 + h2 / 2, x2[i + 1] - 0.006, y2 + h2 / 2)
    ax.text(0.505, 0.075, "Contact predictor removed; no structure is required at inference", ha="center", va="center", fontsize=6.2, color=TEAL, fontweight="bold")

    fig.savefig(STEM.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(STEM)


if __name__ == "__main__":
    main()
