"""PPT-ready overview: one MD loop, three sources of the potential."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from common import INK, DARK_GRAY, LINE_GRAY, NAVY, LayoutRegistry, new_static_figure, save_static
from responsive_story import place_render_cropped


ROOT = Path(__file__).resolve().parent
STEM = "00_md_aimd_nnmd_overview"
ASSETS = {
    "md": ROOT / "_qa" / "02_classical_lj" / "source" / "mattervis_v3" / "lj_force.png",
    "aimd": ROOT / "_qa" / "03_aimd_scf" / "source" / "mattervis_multistep_v3" / "ion_00_scf_06_density.png",
    "nnmd": ROOT / "_qa" / "04_dpmd_native" / "mattervis_v3" / "focus_force.png",
}
BLUE = "#4E9BB5"
OLIVE = "#A89B52"
GREEN = "#2F8562"


def _panel(ax, reg, x0, x1, title, subtitle):
    ax.add_patch(Rectangle((x0, 0.13), x1 - x0, 0.72, fc="white", ec=LINE_GRAY, lw=1.5))
    reg.text(ax, (x0 + x1) / 2, 0.82, title, ha="center", va="center", fontsize=14, color=INK, weight="bold")
    reg.text(ax, (x0 + x1) / 2, 0.775, subtitle, ha="center", va="center", fontsize=10, color=DARK_GRAY)


def _node(ax, reg, x, y, label, colour=NAVY, radius=0.025):
    circle = plt.Circle((x, y), radius, fc="white", ec=colour, lw=1.8, zorder=5)
    ax.add_patch(circle)
    reg.text(ax, x, y, label, ha="center", va="center", fontsize=10, color=colour, weight="bold", zorder=6)


def _flow(ax, reg, xs, y, labels, colours=None):
    colours = colours or [NAVY] * len(labels)
    for idx, (x, label, colour) in enumerate(zip(xs, labels, colours)):
        _node(ax, reg, x, y, label, colour)
        if idx < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.032, y), (xs[idx + 1] - 0.032, y), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=LINE_GRAY, zorder=3))


def main() -> None:
    fig = new_static_figure()
    reg = LayoutRegistry(min_font_pt=10, max_font_pt=16, edge_pad_px=18)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    reg.text(ax, 0.05, 0.95, "FROM MOLECULAR DYNAMICS TO NEURAL-NETWORK MD", ha="left", va="top", fontsize=16, color=INK, weight="bold")
    reg.text(ax, 0.05, 0.915, "same outer loop; only the force provider changes", ha="left", va="top", fontsize=11, color=DARK_GRAY)
    columns = [(0.035, 0.315, "MD", "explicit potential"), (0.36, 0.64, "AIMD", "SCF electronic structure"), (0.685, 0.965, "NNMD / DPMD", "learned local potential")]
    for x0, x1, title, subtitle in columns:
        _panel(ax, reg, x0, x1, title, subtitle)
    # The images are independent MatterVis renders; only their paper placement
    # is composed here, preserving their native geometry and transparency.
    for key, rect in (("md", (0.065, 0.41, 0.285, 0.74)), ("aimd", (0.39, 0.41, 0.61, 0.74)), ("nnmd", (0.715, 0.41, 0.935, 0.74))):
        place_render_cropped(ax, ASSETS[key], rect, padding=0.04, zorder=2)
    _flow(ax, reg, [0.08, 0.17, 0.26], 0.28, ["r", "U", "F"], [BLUE, NAVY, OLIVE])
    _flow(ax, reg, [0.385, 0.455, 0.525, 0.595], 0.28, ["r", "SCF", "E", "F"], [BLUE, NAVY, NAVY, OLIVE])
    _flow(ax, reg, [0.70, 0.765, 0.83, 0.895, 0.95], 0.28, ["env", "Dᵢ", "NN", "Σ ε", "F"], [BLUE, BLUE, NAVY, GREEN, OLIVE])
    reg.text(ax, 0.895, 0.235, "Σ εᵢ = E", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    reg.text(ax, 0.50, 0.18, "rₙ  →  potential  →  Fₙ = −∇U  →  Velocity Verlet  →  rₙ₊₁", ha="center", va="center", fontsize=13, color=INK, weight="bold")
    ax.add_patch(FancyArrowPatch((0.17, 0.075), (0.83, 0.075), arrowstyle="-|>", mutation_scale=13, lw=1.8, color=LINE_GRAY))
    reg.text(ax, 0.50, 0.045, "the integrator is shared; only the force provider changes", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    errors = reg.validate(fig)
    if errors:
        raise RuntimeError("overview layout failed: " + "; ".join(errors))
    save_static(fig, STEM)


if __name__ == "__main__":
    main()
