"""Vector-only PPT still for the DeepMD local-neighbourhood story.

The video keeps the native MatterVis raster frames, but this static plate is
drawn directly from the same saved Cartesian snapshot.  Nothing in the
structure/magnifier panel is pasted from a PNG: atoms, bonds, neighbour edges,
the box, and the environment matrix are vector primitives in the SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from common import INK, DARK_GRAY, LINE_GRAY, NAVY, new_static_figure, save_static
from render_dpmd_native import _descriptor_data, _focus_indices, load_data
from responsive_story import EMERALD, LAKE_BLUE, PALE_OLIVE


ROOT = Path(__file__).resolve().parents[2] / "product"
OUT_PNG = ROOT / "figures" / "04_deep_potential_md.png"
OUT_SVG = ROOT / "figures" / "04_deep_potential_md.svg"


def _panel(ax, title: str) -> None:
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.axis("off")
    ax.add_patch(Rectangle((0.015, 0.015), 0.97, 0.97, fill=False, ec="#C4C9C9", lw=1.15))
    ax.text(0.5, 0.945, title, ha="center", va="center", fontsize=15, weight="bold", color=INK)


def _draw_vv(ax) -> None:
    _panel(ax, "ONE MD STEP")
    position = ax.get_position(); fig = ax.figure
    panel_ratio = (position.width * fig.get_figwidth()) / (position.height * fig.get_figheight())
    centre = np.array([0.5, 0.58]); radius_x = 0.37; radius = radius_x * panel_ratio
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    ax.plot(centre[0] + radius_x * np.cos(theta), centre[1] + radius * np.sin(theta), color="#C7CACC", lw=2.0)
    # Active position->velocity arc and three vector arrows are purely vector
    # paper primitives; no native image is involved in this panel.
    active = theta[(theta >= 2.45) & (theta <= 4.85)]
    ax.plot(centre[0] + radius_x * np.cos(active), centre[1] + radius * np.sin(active), color=INK, lw=2.8)
    for angle, label, colour in [(np.pi / 2, "r", "#C7CACC"), (3.8, "v", INK), (5.9, "a", "#C7CACC")]:
        p = centre + np.array([radius_x * np.cos(angle), radius * np.sin(angle)])
        ax.add_patch(__import__("matplotlib").patches.Ellipse(p, 0.055 / panel_ratio, 0.055, fc=INK if label == "v" else "white", ec=INK if label == "v" else "#C7CACC", lw=1.8))
        ax.text(*p, label, ha="center", va="center", fontsize=13, color="white" if label == "v" else DARK_GRAY, weight="bold")
    ax.text(0.5, 0.60, r"$r_{n+1}=r_n+v_n\Delta t+\frac{1}{2}a_n\Delta t^2$", ha="center", va="center", fontsize=11, color=INK)
    ax.text(0.5, 0.77, "position", ha="center", fontsize=10, color=DARK_GRAY)
    ax.text(0.20, 0.30, "velocity", ha="center", fontsize=10, color=INK)
    ax.text(0.80, 0.30, "acceleration", ha="center", fontsize=10, color=DARK_GRAY)
    ax.text(0.09, 0.19, "VV STATE", fontsize=10, color=DARK_GRAY, weight="bold")
    for y, label, colour in [(0.12, "r, v / input", LAKE_BLUE), (0.08, "F_DP", PALE_OLIVE), (0.04, "r′, v′", EMERALD)]:
        ax.plot([0.09, 0.20], [y, y], color=colour, lw=3.0, solid_capstyle="round")
        ax.text(0.25, y, label, va="center", fontsize=10, color=INK)


def _project_yz(points: np.ndarray, box: float, rect: tuple[float, float, float, float]) -> np.ndarray:
    """Project along +x into a square y-z box using the actual coordinates."""
    x0, y0, w, h = rect
    yz = np.asarray(points, dtype=float)[:, [1, 2]]
    uv = yz / float(box)
    return np.column_stack([x0 + w * uv[:, 0], y0 + h * uv[:, 1]])


def _draw_water(ax, centre: np.ndarray, h1: np.ndarray, h2: np.ndarray, *, scale: float = 1.0, alpha: float = 1.0) -> None:
    ax.plot([centre[0], h1[0]], [centre[1], h1[1]], color="#B7BDBE", lw=0.55 * scale, alpha=alpha, zorder=3)
    ax.plot([centre[0], h2[0]], [centre[1], h2[1]], color="#B7BDBE", lw=0.55 * scale, alpha=alpha, zorder=3)
    ax.add_patch(Circle(centre, 0.0075 * scale, fc="#A32035", ec="none", alpha=alpha, zorder=4))
    ax.add_patch(Circle(h1, 0.0042 * scale, fc="#F4F4F0", ec="#A7ABAB", lw=0.25, alpha=alpha, zorder=4))
    ax.add_patch(Circle(h2, 0.0042 * scale, fc="#F4F4F0", ec="#A7ABAB", lw=0.25, alpha=alpha, zorder=4))


def _draw_central_neighbour(ax, data: dict[str, object]) -> None:
    positions = np.asarray(data["positions_wrapped"], dtype=float)
    elements = np.asarray(data["elements"]).astype(str)
    molecules = np.asarray(data["molecule_ids"], dtype=int)
    box = float(np.asarray(data["box_length"]).reshape(-1)[0])
    central = int(np.asarray(data["central_index"]).reshape(-1)[0])
    delta = positions - positions[central]
    delta -= box * np.round(delta / box)
    distances = np.linalg.norm(delta, axis=1)
    focus = _focus_indices(data, distances, central)
    central_pos = positions[central]

    # Whole periodic square, viewed down +x.
    position = ax.get_position(); fig = ax.figure
    panel_ratio = (position.width * fig.get_figwidth()) / (position.height * fig.get_figheight())
    box_rect = (0.045, 0.28, 0.52 / panel_ratio, 0.52)
    ax.add_patch(Rectangle((box_rect[0], box_rect[1]), box_rect[2], box_rect[3], fill=False, ec="#AEB9BC", lw=1.2, zorder=1))
    oxygen = np.flatnonzero(elements == "O")
    for oi in oxygen:
        members = np.flatnonzero(molecules == molecules[oi])
        if len(members) < 3:
            continue
        mol = positions[members]
        q = _project_yz(mol, box, box_rect)
        alpha = 0.22 if int(oi) != central else 1.0
        _draw_water(ax, q[0], q[1], q[2], scale=0.85, alpha=alpha)
    ax.text(0.32, 0.245, "periodic water box · view along +x", ha="center", va="top", fontsize=10, color=DARK_GRAY)

    source_uv = _project_yz(central_pos[None, :], box, box_rect)[0]
    source_r = 0.055; source_rx = source_r / panel_ratio
    ax.add_patch(__import__("matplotlib").patches.Ellipse(source_uv, 2*source_rx, 2*source_r, fc="white", ec=PALE_OLIVE, lw=1.8, zorder=10))
    # Re-draw the central local water inside the locator circle.  It is a
    # genuine projection of O126 and its two saved H neighbours, not a blank
    # mask over the source point.
    cmembers = np.flatnonzero(molecules == molecules[central])
    local = [source_uv]
    for index in cmembers:
        if int(index) == central:
            continue
        d = positions[int(index)] - central_pos
        d -= box * np.round(d / box)
        local.append(source_uv + 0.05 * np.array([d[1], d[2]]))
    _draw_water(ax, local[0], local[1], local[2], scale=1.4, alpha=1.0)
    ax.add_patch(__import__("matplotlib").patches.Ellipse(source_uv, 0.018 / panel_ratio, 0.018, fc="#A32035", ec="none", zorder=12))
    ax.text(source_uv[0], source_uv[1] - source_r - 0.018, "O126", ha="center", va="top", fontsize=10, color=DARK_GRAY)

    # Magnifier: the same atoms are re-projected around O126, but the bond
    # layer is intentionally removed. Only real centre-to-neighbour edges are
    # shown, with both endpoints at atom centres.
    mag_c = np.array([0.67, 0.57]); mag_r = 0.19; mag_rx = mag_r / panel_ratio
    ax.add_patch(__import__("matplotlib").patches.Ellipse(mag_c, 2*mag_rx, 2*mag_r, fc="white", ec="#466C7A", lw=1.8, zorder=20))
    focus_list = sorted(int(i) for i in focus if int(i) != central)
    focus_pos = positions[central] + delta[focus_list]
    scale_x = mag_rx / 6.0; scale_y = mag_r / 6.0
    def mag_point(p):
        d = p - central_pos
        return mag_c + np.array([scale_x * d[1], scale_y * d[2]])
    origin = mag_c.copy()
    ax.add_patch(__import__("matplotlib").patches.Ellipse(origin, 0.028/panel_ratio, 0.028, fc="#183153", ec="#183153", lw=1.0, zorder=30))
    for index, p in zip(focus_list, focus_pos):
        end = mag_point(p)
        ax.plot([origin[0], end[0]], [origin[1], end[1]], color=EMERALD, lw=0.65, alpha=0.82, zorder=22)
        radius = 0.009 if elements[index] == "O" else 0.006
        face = "#A32035" if elements[index] == "O" else "#F4F4F0"
        edge = "none" if elements[index] == "O" else "#A7ABAB"
        ax.add_patch(__import__("matplotlib").patches.Ellipse(end, 2*radius/panel_ratio, 2*radius, fc=face, ec=edge, lw=0.3, zorder=25))
    ax.text(mag_c[0], mag_c[1] + mag_r + 0.02, "magnified local · neighbor view", ha="center", va="bottom", fontsize=11, color=INK, weight="bold")
    ax.text(mag_c[0], mag_c[1] - mag_r - 0.02, "23 real j shown · 83 in Nᵢ(r_c)", ha="center", va="top", fontsize=10, color=DARK_GRAY)
    # Leaders terminate on the circles, not on the atoms or edges.
    ax.plot([source_uv[0] + source_rx, mag_c[0] - mag_rx], [source_uv[1] + source_r * 0.65, mag_c[1] + mag_r * 0.62], color=PALE_OLIVE, lw=1.2, zorder=15)
    ax.plot([source_uv[0] + source_rx, mag_c[0] - mag_rx], [source_uv[1] - source_r * 0.65, mag_c[1] - mag_r * 0.62], color=PALE_OLIVE, lw=1.2, zorder=15)


def _draw_right(ax, data: dict[str, object]) -> None:
    _panel(ax, "LOCAL FRAME")
    ax.text(0.5, 0.88, "O126 · MIC neighbours", ha="center", fontsize=10, color=DARK_GRAY)
    origin = np.array([0.16, 0.75])
    for end, colour, label in [((0.30, 0.75), LAKE_BLUE, "x"), ((0.16, 0.86), EMERALD, "y"), ((0.24, 0.82), PALE_OLIVE, "z")]:
        ax.add_patch(FancyArrowPatch(origin, end, arrowstyle="-|>", mutation_scale=10, lw=1.4, color=colour))
        ax.text(*end, label, fontsize=10, color=colour)
    values = _descriptor_data(data)
    matrix = np.asarray(values["matrix"], dtype=float)
    ax.text(0.67, 0.79, "Nᵢⱼ", ha="center", fontsize=11, color=NAVY, weight="bold")
    ax.text(0.67, 0.735, "Nᵢⱼ = |rᵢ − rⱼ|", ha="center", fontsize=10, color=DARK_GRAY)
    left, bottom, size = 0.50, 0.55, 0.28; vmax = max(float(matrix.max()), 1.0e-9)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            weight = float(matrix[row, col]) / vmax
            ax.add_patch(Rectangle((left + col * size / matrix.shape[1], bottom + (matrix.shape[0]-1-row) * size / matrix.shape[0]), size / matrix.shape[1] - 0.002, size / matrix.shape[0] - 0.002, fc=(0.25, 0.60, 0.70, 0.18 + 0.65 * (1.0 - weight)), ec="white", lw=0.4))
    ax.add_patch(Rectangle((0.035, 0.035), 0.93, 0.40, fc="white", ec=LINE_GRAY, lw=1.15))
    ax.text(0.5, 0.385, "ENVIRONMENT  Rᵢ", ha="center", fontsize=11, color=INK, weight="bold")
    env = np.asarray(values["environment"], dtype=float); emin, emax = float(env.min()), float(env.max())
    l, b, w, h = 0.20, 0.17, 0.60, 0.15
    for row in range(env.shape[0]):
        for col in range(env.shape[1]):
            q = (float(env[row, col]) - emin) / max(emax-emin, 1.0e-12)
            ax.add_patch(Rectangle((l + col*w/4, b + (env.shape[0]-1-row)*h/env.shape[0]), w/4-0.003, h/env.shape[0]-0.003, fc=(0.18,0.52,0.38,0.18+0.68*q), ec="white", lw=0.4))
    for col, label in enumerate(("s", "sx/r", "sy/r", "sz/r")):
        ax.text(l + (col+0.5)*w/4, 0.14, label, ha="center", va="top", fontsize=10, color=EMERALD)
    ax.text(0.5, 0.095, r"Rᵢⱼ=[s, sx/r, sy/r, sz/r]", ha="center", fontsize=10, color=DARK_GRAY)
    ax.text(0.5, 0.055, r"Dᵢ=(Gᵢ)ᵀRᵢRᵢᵀGᵢ/Nc²", ha="center", fontsize=10, color=DARK_GRAY)


def main() -> None:
    data = load_data()
    fig = plt.figure(figsize=(14, 10), dpi=250, facecolor="white")
    left = fig.add_axes([0.035, 0.09, 0.25, 0.83])
    middle = fig.add_axes([0.30, 0.09, 0.46, 0.83])
    right = fig.add_axes([0.78, 0.09, 0.20, 0.83])
    _draw_vv(left); _panel(middle, "DEEP POTENTIAL · local neighbourhood"); _draw_central_neighbour(middle, data); _draw_right(right, data)
    fig.savefig(OUT_SVG, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PNG, format="png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()

