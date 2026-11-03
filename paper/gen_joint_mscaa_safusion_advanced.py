"""Draw the hierarchy-rich MSCAA + SAFusion architecture preview."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT / ".python_packages"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

import gen_joint_mscaa_safusion_clean as base


OUT_DIR = ROOT / "paper_overleaf_revised_20260815"
INK = base.INK
MUTED = base.MUTED
WHITE = base.WHITE
BLUE = base.BLUE
BLUE_LIGHT = base.BLUE_LIGHT
PURPLE = base.PURPLE
GRAY = base.GRAY
TEAL = base.TEAL
TEAL_LIGHT = base.TEAL_LIGHT
ORANGE = base.ORANGE
ORANGE_LIGHT = base.ORANGE_LIGHT
PINK = base.PINK
PINK_LIGHT = base.PINK_LIGHT


def group(ax, x, y, w, h, title, *, edge="#A8B1B8", fill="none",
          title_color=MUTED, title_x=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.035,rounding_size=0.07",
        facecolor=fill, edgecolor=edge, linewidth=0.75,
        linestyle=(0, (3, 2)), zorder=0,
    )
    ax.add_patch(box)
    ax.text(title_x if title_x is not None else x + 0.18, y + h - 0.18,
            title, ha="left", va="top", fontsize=4.6,
            color=title_color, fontweight="bold", zorder=8)


def junction(ax, x, y, symbol, *, radius=0.16, fill=WHITE, edge=INK):
    ax.add_patch(Circle((x, y), radius, facecolor=fill, edgecolor=edge,
                        linewidth=0.8, zorder=6))
    ax.text(x, y, symbol, ha="center", va="center", fontsize=6.2,
            color=INK, fontweight="bold", zorder=7)


def joint_cell(ax, x, y, stage):
    """A repeated composite cell, exposing the true order SAFusion -> C2f."""
    box = FancyBboxPatch(
        (x, y), 1.84, 0.78,
        boxstyle="round,pad=0.025,rounding_size=0.055",
        facecolor="#FAFBFB", edgecolor="#8E9AA4", linewidth=0.65, zorder=2,
    )
    ax.add_patch(box)
    base.op(ax, x + 0.09, y + 0.14, 0.62, 0.48, "SAF",
            fill=ORANGE_LIGHT, edge="#B97B20", fontsize=5.0)
    base.arrow(ax, (x + 0.74, y + 0.38), (x + 0.88, y + 0.38), mutation=5)
    base.op(ax, x + 0.92, y + 0.14, 0.78, 0.48, "C2f-Star",
            fill=TEAL_LIGHT, edge="#4D8D7C", fontsize=4.35)
    badge = FancyBboxPatch(
        (x + 1.48, y + 0.48), 0.16, 0.13,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        facecolor=PINK, edgecolor="#8F4C72", linewidth=0.55, zorder=8,
    )
    ax.add_patch(badge)
    ax.text(x + 1.56, y + 0.545, "M", ha="center", va="center",
            fontsize=3.35, color=WHITE, fontweight="bold", zorder=9)
    ax.text(x + 0.92, y - 0.10, stage, ha="center", va="top",
            fontsize=4.1, color=MUTED)


def draw_overview(ax):
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 5.55)
    ax.axis("off")
    base.panel_title(ax, "a", "Joint content-adaptive pyramid calibration", 0.05, 5.47)

    base.feature(ax, 0.28, 1.78, 0.54, 1.62, 0.17, "#D5E0E5", "Input", fontsize=5.3)
    base.scale_label(ax, 0.64, 1.58, r"$640\times640$")
    base.op(ax, 1.22, 1.76, 0.90, 1.68, "StarNet\nS050", fill="#F2F4F5", fontsize=5.8)
    base.arrow(ax, (1.01, 2.58), (1.18, 2.58))

    ax.text(3.38, 5.00, "Backbone pyramid", ha="center", fontsize=5.6,
            color=MUTED, fontweight="bold")
    ax.plot([2.42, 2.42], [1.34, 4.38], color=INK, linewidth=0.82, zorder=2)
    base.arrow(ax, (2.12, 2.58), (2.42, 2.58))
    levels = [
        (3.02, 4.00, 0.70, BLUE_LIGHT, "P3", r"$80\times80$"),
        (3.02, 2.50, 0.62, BLUE, "P4", r"$40\times40$"),
        (3.02, 1.02, 0.55, PURPLE, "P5", r"$20\times20$"),
    ]
    ends = {}
    for x, y, h, color, label, dims in levels:
        _, ends[label] = base.feature_sequence(ax, x, y, h, color, label, dims)
        base.arrow(ax, (2.42, y + h / 2), (x - 0.04, y + h / 2))

    group(ax, 4.72, 0.72, 9.05, 4.18, "Four repeated joint calibration cells")
    base.op(ax, 4.90, 1.06, 0.68, 0.48, "SPPF", fill="#E8F1EF",
            edge="#4D8D7C", fontsize=4.8)
    base.arrow(ax, (ends["P5"] + 0.02, 1.29), (4.86, 1.29))

    joint_cell(ax, 5.10, 2.42, r"$\mathcal{J}_{4}^{td}$")
    joint_cell(ax, 5.10, 3.90, r"$\mathcal{J}_{3}^{td}$")
    joint_cell(ax, 8.38, 2.42, r"$\mathcal{J}_{4}^{bu}$")
    joint_cell(ax, 10.98, 0.98, r"$\mathcal{J}_{5}^{bu}$")

    base.arrow(ax, (ends["P4"] + 0.02, 2.81), (5.06, 2.81))
    base.arrow(ax, (ends["P3"] + 0.02, 4.34), (5.06, 4.29))
    base.path_arrow(ax, [(5.62, 1.29), (4.84, 1.29), (4.84, 2.60), (5.06, 2.60)])
    ax.text(4.66, 2.02, r"$\uparrow 2$", fontsize=4.4, color=MUTED)
    base.path_arrow(ax, [(6.98, 2.81), (7.25, 2.81), (7.25, 4.12), (5.06, 4.12)])
    ax.text(7.08, 3.52, r"$\uparrow 2$", fontsize=4.4, color=MUTED)
    base.arrow(ax, (6.98, 2.81), (8.34, 2.81))
    base.path_arrow(ax, [(6.98, 4.29), (8.03, 4.29), (8.03, 3.02), (8.34, 3.02)])
    ax.text(7.83, 3.52, r"$\downarrow 2$", fontsize=4.4, color=MUTED)
    base.arrow(ax, (5.62, 1.29), (10.94, 1.29))
    base.path_arrow(ax, [(10.26, 2.81), (10.68, 2.81), (10.68, 1.48), (10.94, 1.48)])
    ax.text(10.48, 2.05, r"$\downarrow 2$", fontsize=4.4, color=MUTED)

    out_specs = [
        (13.24, 4.00, BLUE_LIGHT, r"$N_3$"),
        (13.24, 2.50, BLUE, r"$N_4$"),
        (13.24, 1.02, PURPLE, r"$N_5$"),
    ]
    for x, y, color, label in out_specs:
        base.feature(ax, x, y, 0.17, 0.60, 0.06, color, "")
        ax.text(x + 0.10, y - 0.08, label, ha="center", va="top",
                fontsize=4.1, color=MUTED)
    base.arrow(ax, (6.98, 4.29), (13.20, 4.29))
    base.arrow(ax, (10.26, 2.81), (13.20, 2.81))
    base.arrow(ax, (12.86, 1.37), (13.20, 1.31))
    base.op(ax, 14.18, 1.50, 0.82, 2.72, "SCBN\nHead", fill="#F2F4F5", fontsize=5.6)
    base.arrow(ax, (13.55, 4.29), (14.14, 3.88))
    base.arrow(ax, (13.55, 2.81), (14.14, 2.86))
    base.arrow(ax, (13.55, 1.31), (14.14, 1.86))
    base.feature(ax, 15.56, 2.40, 0.48, 0.78, 0.16, GRAY, "Det", fontsize=5.6)
    base.arrow(ax, (15.04, 2.84), (15.52, 2.84))
    base.scale_label(ax, 15.86, 2.17, "P3 / P4 / P5")

    ax.text(8.75, 0.42,
            r"$\mathcal{J}_{i}=\mathrm{C2f\!-\!Star}_{\mathrm{MSCAA}}"
            r"(\mathrm{SAFusion}(F_i,F_j))$",
            ha="center", va="center", fontsize=5.0, color=MUTED)


def draw_star_mscaa(ax):
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    base.panel_title(ax, "b", "C2f-Star block with embedded MSCAA", 0.04, 5.53)

    # Outer C2f routing makes the repeated block hierarchy explicit.
    group(ax, 0.10, 3.95, 15.85, 1.18, "C2f wrapper", fill="#FCFDFD")
    base.feature(ax, 0.34, 4.18, 0.28, 0.56, 0.09, TEAL, r"$X$", fontsize=4.6)
    base.arrow(ax, (0.76, 4.48), (1.08, 4.48), mutation=5)
    base.op(ax, 1.12, 4.20, 0.72, 0.54, r"$1\!\times\!1$", fill=WHITE, fontsize=4.4)
    junction(ax, 2.18, 4.47, "Y", radius=0.14, fill="#F2F4F5")
    base.arrow(ax, (1.88, 4.47), (2.00, 4.47), mutation=5)
    base.arrow(ax, (2.34, 4.47), (2.66, 4.47), mutation=5)
    base.op(ax, 2.70, 4.20, 2.10, 0.54, r"Star--MSCAA $\times n$",
            fill=TEAL_LIGHT, edge="#4D8D7C", fontsize=4.4)
    base.arrow(ax, (4.84, 4.47), (5.30, 4.47), mutation=5)
    base.op(ax, 5.34, 4.20, 0.78, 0.54, "Concat", fill=WHITE, fontsize=4.2)
    base.path_arrow(ax, [(2.18, 4.28), (2.18, 4.04), (5.16, 4.04), (5.30, 4.30)],
                    style="--", color=MUTED, lw=0.65)
    base.arrow(ax, (6.16, 4.47), (6.48, 4.47), mutation=5)
    base.op(ax, 6.52, 4.20, 0.72, 0.54, r"$1\!\times\!1$", fill=WHITE, fontsize=4.4)
    base.arrow(ax, (7.28, 4.47), (7.62, 4.47), mutation=5)
    base.feature(ax, 7.66, 4.18, 0.28, 0.56, 0.09, TEAL, r"$Y$", fontsize=4.6)
    ax.text(3.72, 4.00, "bypass", ha="center", va="top",
            fontsize=3.7, color=MUTED)

    # Magnified Star_Block_MSCAA implementation.
    group(ax, 0.10, 0.18, 15.85, 3.42, "Magnified Star_Block_MSCAA", edge="#7B8790")
    base.feature(ax, 0.34, 1.54, 0.28, 0.70, 0.09, BLUE_LIGHT, r"$U$", fontsize=4.6)
    base.arrow(ax, (0.76, 1.90), (1.02, 1.90), mutation=5)
    base.op(ax, 1.06, 1.62, 0.86, 0.56, "DW 7x7", fill="#E7EEF5", edge="#718CA8", fontsize=4.2)
    ax.plot([2.10, 2.10], [1.25, 2.62], color=INK, linewidth=0.75)
    base.arrow(ax, (1.96, 1.90), (2.10, 1.90), mutation=5)

    base.op(ax, 2.42, 2.34, 0.72, 0.50, r"$f_1$ 1x1", fill=WHITE, fontsize=4.0)
    base.op(ax, 3.40, 2.34, 0.76, 0.50, "ReLU6", fill="#EAF1DE", edge="#7D9868", fontsize=3.9)
    base.arrow(ax, (2.10, 2.58), (2.38, 2.58), mutation=5)
    base.arrow(ax, (3.18, 2.58), (3.36, 2.58), mutation=5)
    base.op(ax, 2.42, 0.98, 0.72, 0.50, r"$f_2$ 1x1", fill=WHITE, fontsize=4.0)
    base.arrow(ax, (2.10, 1.24), (2.38, 1.24), mutation=5)
    junction(ax, 4.58, 1.90, r"$\odot$", radius=0.18, fill=ORANGE_LIGHT, edge="#B97B20")
    base.path_arrow(ax, [(4.20, 2.58), (4.58, 2.58), (4.58, 2.12)], lw=0.75)
    base.path_arrow(ax, [(3.18, 1.24), (4.58, 1.24), (4.58, 1.68)], lw=0.75)
    ax.text(4.58, 3.03, "Star interaction", ha="center", fontsize=4.0,
            color="#B97B20", fontweight="bold")

    # The inset expands the real MSCAA path inside the Star block.
    group(ax, 5.02, 0.60, 6.36, 2.66, "Embedded MSCAA", edge=PINK,
          fill="#FEFBFC", title_color=PINK)
    base.arrow(ax, (4.80, 1.90), (5.20, 1.90), color=PINK, mutation=5)
    base.op(ax, 5.28, 1.64, 0.88, 0.52, "Pool 7x7\n$\mathcal{C}_0$", fill="#F2F4F5", fontsize=3.7)
    ax.plot([6.36, 6.36], [1.18, 2.60], color=INK, linewidth=0.65)
    base.arrow(ax, (6.20, 1.90), (6.36, 1.90), mutation=5)
    base.op(ax, 6.58, 2.28, 0.66, 0.44, "1x11", fill=PINK_LIGHT, edge="#9C5179", fontsize=3.7)
    base.op(ax, 7.46, 2.28, 0.66, 0.44, "11x1", fill=PINK_LIGHT, edge="#9C5179", fontsize=3.7)
    base.arrow(ax, (6.36, 2.50), (6.54, 2.50), mutation=4)
    base.arrow(ax, (7.28, 2.50), (7.42, 2.50), mutation=4)
    base.op(ax, 6.58, 1.02, 0.66, 0.44, "1x7", fill="#E4ECF5", edge="#718CA8", fontsize=3.7)
    base.op(ax, 7.46, 1.02, 0.66, 0.44, "7x1", fill="#E4ECF5", edge="#718CA8", fontsize=3.7)
    base.arrow(ax, (6.36, 1.24), (6.54, 1.24), mutation=4)
    base.arrow(ax, (7.28, 1.24), (7.42, 1.24), mutation=4)
    base.path_arrow(ax, [(8.16, 2.50), (8.40, 2.50), (8.40, 2.04), (8.58, 2.04)], lw=0.65)
    base.path_arrow(ax, [(8.16, 1.24), (8.40, 1.24), (8.40, 1.78), (8.58, 1.78)], lw=0.65)
    base.op(ax, 8.62, 1.62, 0.58, 0.54, "Concat", fill=WHITE, fontsize=3.0)
    base.arrow(ax, (9.22, 1.90), (9.34, 1.90), mutation=3)
    base.op(ax, 9.36, 1.62, 0.34, 0.54, r"$\mathcal{C}_f$", fill=WHITE, fontsize=3.8)
    base.arrow(ax, (9.73, 1.90), (9.88, 1.90), mutation=3)
    base.op(ax, 9.90, 1.62, 0.34, 0.54, r"$\mathcal{C}_2$", fill=WHITE, fontsize=3.8)
    base.arrow(ax, (10.27, 1.90), (10.42, 1.90), mutation=3)
    base.op(ax, 10.44, 1.62, 0.36, 0.54, r"$\sigma$", fill=PINK_LIGHT,
            edge="#9C5179", fontsize=5.1)
    junction(ax, 11.05, 1.90, r"$\odot$", radius=0.14, fill=WHITE, edge=PINK)
    base.arrow(ax, (10.84, 1.90), (10.89, 1.90), color=PINK, mutation=4)
    base.path_arrow(ax, [(5.18, 1.66), (5.12, 1.66), (5.12, 0.76),
                         (11.05, 0.76), (11.05, 1.72)], style="--",
                    color=MUTED, lw=0.58)

    base.arrow(ax, (11.24, 1.90), (11.52, 1.90), mutation=5)
    base.op(ax, 11.56, 1.62, 0.66, 0.56, r"$g$ 1x1", fill=WHITE, fontsize=4.0)
    base.arrow(ax, (12.26, 1.90), (12.50, 1.90), mutation=5)
    base.op(ax, 12.54, 1.62, 0.84, 0.56, "DW 7x7", fill="#E7EEF5", edge="#718CA8", fontsize=4.0)
    junction(ax, 14.02, 1.90, "+", radius=0.18, fill="#F2F4F5")
    base.arrow(ax, (13.42, 1.90), (13.80, 1.90), mutation=5)
    base.path_arrow(ax, [(0.76, 1.64), (0.88, 1.64), (0.88, 0.38),
                         (14.02, 0.38), (14.02, 1.68)], style="--",
                    color=MUTED, lw=0.68)
    base.arrow(ax, (14.24, 1.90), (14.54, 1.90), mutation=5)
    base.feature(ax, 14.58, 1.54, 0.28, 0.70, 0.09, TEAL, r"$V$", fontsize=4.6)
    ax.text(12.62, 0.69, "residual projection", ha="center", fontsize=3.9, color=MUTED)


def draw_safusion(ax):
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    base.panel_title(ax, "c", "SAFusion gate-conditioned fusion", 0.04, 5.53)

    group(ax, 0.10, 0.22, 9.25, 4.88, "Cross-level fusion cell", fill="#FCFDFD")
    base.feature(ax, 0.36, 3.50, 0.42, 0.72, 0.13, BLUE_LIGHT, r"$F_l$", fontsize=5.0)
    base.feature(ax, 0.36, 1.24, 0.42, 0.64, 0.13, PURPLE, r"$F_h$", fontsize=5.0)
    base.op(ax, 1.20, 1.28, 0.68, 0.50, "Resize", fill=WHITE, fontsize=4.0)
    base.arrow(ax, (0.94, 1.56), (1.16, 1.54), mutation=5)
    base.arrow(ax, (0.94, 3.88), (2.22, 3.12), mutation=5)
    base.arrow(ax, (1.92, 1.54), (2.22, 2.54), mutation=5)
    base.feature(ax, 2.26, 2.28, 0.52, 1.00, 0.16, GRAY, r"$F$", fontsize=5.2)
    ax.text(2.60, 2.08, "Concat", ha="center", va="top", fontsize=4.0, color=MUTED)

    group(ax, 3.25, 2.60, 5.55, 1.82, "Gate generator  $\phi(\cdot)$",
          edge="#B97B20", fill="#FFFDF8", title_color="#B97B20")
    base.arrow(ax, (2.98, 3.00), (3.52, 3.54), color="#B97B20", mutation=5)
    base.op(ax, 3.58, 3.24, 0.64, 0.52, "GAP", fill=TEAL_LIGHT,
            edge="#4D8D7C", fontsize=4.0)
    base.arrow(ax, (4.26, 3.50), (4.48, 3.50), mutation=4)
    base.op(ax, 4.52, 3.24, 0.72, 0.52, r"$W_1$", fill=WHITE, fontsize=4.0)
    ax.text(4.88, 3.06, r"$C\to r$", ha="center", va="top", fontsize=3.5, color=MUTED)
    base.arrow(ax, (5.28, 3.50), (5.48, 3.50), mutation=4)
    base.op(ax, 5.52, 3.24, 0.66, 0.52, "SiLU", fill=WHITE, fontsize=3.9)
    base.arrow(ax, (6.22, 3.50), (6.42, 3.50), mutation=4)
    base.op(ax, 6.46, 3.24, 0.72, 0.52, r"$W_2$", fill=WHITE, fontsize=4.0)
    ax.text(6.82, 3.06, r"$r\to C$", ha="center", va="top", fontsize=3.5, color=MUTED)
    base.arrow(ax, (7.22, 3.50), (7.42, 3.50), mutation=4)
    base.op(ax, 7.46, 3.24, 0.48, 0.52, r"$\sigma$", fill=ORANGE_LIGHT,
            edge="#B97B20", fontsize=5.2)
    base.arrow(ax, (7.98, 3.50), (8.16, 3.50), color="#B97B20", mutation=4)
    base.segmented_gate(ax, 8.20, 3.31, w=0.48, h=0.38)

    junction(ax, 6.48, 1.62, r"$\odot$", radius=0.18, fill=ORANGE_LIGHT, edge="#B97B20")
    base.path_arrow(ax, [(8.70, 3.50), (8.96, 3.50), (8.96, 1.62),
                         (6.70, 1.62)], color="#B97B20", lw=0.75)
    base.path_arrow(ax, [(2.98, 2.66), (3.12, 2.66), (3.12, 1.62),
                         (6.26, 1.62)], style="--", color=MUTED, lw=0.68)
    base.arrow(ax, (6.70, 1.62), (7.16, 1.62), color="#B97B20", mutation=5)
    base.feature(ax, 7.20, 1.12, 0.48, 1.00, 0.16, TEAL, r"$F_o$", fontsize=5.0)
    ax.text(6.20, 0.66,
            r"$F_o=F\odot\sigma(W_2\,\mathrm{SiLU}(W_1\,\mathrm{GAP}(F)))$",
            ha="center", va="center", fontsize=4.4, color=MUTED)
    ax.text(5.98, 4.68, r"$r=\max(\lfloor C/8\rfloor,8)$",
            ha="center", va="center", fontsize=4.2, color=MUTED)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.16, 4.72), facecolor=WHITE)
    ax_top = fig.add_axes([0.018, 0.51, 0.964, 0.475])
    ax_left = fig.add_axes([0.018, 0.035, 0.61, 0.44])
    ax_right = fig.add_axes([0.645, 0.035, 0.337, 0.44])
    draw_overview(ax_top)
    draw_star_mscaa(ax_left)
    draw_safusion(ax_right)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"joint_mscaa_safusion_advanced.{ext}", dpi=500,
                    bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Saved advanced architecture preview to {OUT_DIR}")


if __name__ == "__main__":
    main()
