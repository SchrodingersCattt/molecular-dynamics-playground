"""Compatibility renderer for the 03b TNT UKS reactive-AIMD story.

The renderer consumes the saved trajectory rather than inventing a path.  When
the dataset was generated with ``--demo`` it keeps the backend warning visible
in the figure and video so the analytic surrogate cannot be confused with a
real UKS calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from common import (  # noqa: E402
    CRIMSON,
    DARK_GRAY,
    FONT_SIZES,
    GREEN,
    INK,
    LIGHT_GRAY,
    LINE_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    axes_from_top_slot,
    draw_ball_and_stick,
    draw_vector_arrow,
    map_projected_to_rect,
    new_static_figure,
    project_points,
    render_video,
    save_static,
)


ROOT = Path(__file__).resolve().parents[2] / "product"
DATA_PATH = ROOT / "data" / "uks_tnt_reaction.npz"
MANIFEST_PATH = ROOT / "data" / "uks_tnt_reaction.json"
STEM = "03b_uks_reaction"
QA_DIR = ROOT / "qa" / STEM

SPIN_ALPHA = "#B23A48"
SPIN_BETA = "#2F679B"
POSITION_BLUE = "#4E9BB5"
ENERGY_OLIVE = "#8F8D62"
VIDEO_DURATION = 30.0
ION_SNAPSHOT_COUNT = 7

VIDEO_SLOTS = {
    "rail": (0.015, 0.025, 0.215, 0.975),
    "structure": (0.230, 0.025, 0.680, 0.975),
    "trace": (0.695, 0.510, 0.985, 0.975),
    "uks": (0.695, 0.025, 0.985, 0.475),
}
STATIC_SLOTS = {
    "rail": (0.025, 0.065, 0.235, 0.935),
    "structure": (0.250, 0.065, 0.695, 0.935),
    "trace": (0.715, 0.515, 0.975, 0.935),
    "uks": (0.715, 0.065, 0.975, 0.485),
}


def load_data() -> tuple[dict[str, np.ndarray], dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run generate_uks_tnt.py --demo "
            "or generate a real PySCF dataset first."
        )
    arrays = {key: value for key, value in np.load(DATA_PATH, allow_pickle=False).items()}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    return arrays, manifest


def _panel(ax: plt.Axes, reg: LayoutRegistry, title: str, *, video: bool) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0.004, 0.004),
            0.992,
            0.992,
            facecolor=WHITE,
            edgecolor=LINE_GRAY,
            linewidth=2.0 if video else 1.4,
            zorder=0,
        )
    )
    reg.text(
        ax,
        0.50,
        0.965,
        title,
        ha="center",
        va="top",
        fontsize=FONT_SIZES["panel_title"] if video else FONT_SIZES["emphasis"],
        color=INK,
        weight="bold",
    )


def _frame_index(data: dict[str, np.ndarray], time_seconds: float) -> int:
    n = len(data["positions"])
    fraction = float(np.clip(time_seconds / VIDEO_DURATION, 0.0, 1.0))
    return min(n - 1, max(0, int(round(fraction * (n - 1)))))


def _world_bounds(data: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    positions = np.asarray(data["positions"], dtype=float)
    centre = np.mean(positions.reshape(-1, 3), axis=0)
    projected, _ = project_points(positions.reshape(-1, 3))
    half_span = float(np.max(np.abs(projected - project_points(centre[None, :])[0][0]))) + 1.0
    return centre, max(half_span, 3.2)


def _draw_spin_plane(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    frame: int,
    *,
    rect: tuple[float, float, float, float],
    centre_3d: np.ndarray,
    half_span: float,
    video: bool,
) -> None:
    x = np.asarray(data["spin_density_x"], dtype=float)
    y = np.asarray(data["spin_density_y"], dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    projected, _ = project_points(points)
    centre_projected, _ = project_points(np.asarray(centre_3d)[None, :])
    mapped = map_projected_to_rect(projected, rect, centre_projected[0], half_span)
    grid_x = mapped[:, 0].reshape(xx.shape)
    grid_y = mapped[:, 1].reshape(yy.shape)
    spin = np.asarray(data["rho_alpha"][frame] - data["rho_beta"][frame], dtype=float)
    all_spin = np.asarray(data["rho_alpha"] - data["rho_beta"], dtype=float)
    vmax = max(float(np.nanmax(np.abs(all_spin))), 1.0e-5)
    # A fixed symmetric scale keeps the spin colours comparable across time.
    threshold = 0.10 * vmax
    positive = np.ma.masked_where(spin <= threshold, spin)
    negative = np.ma.masked_where(spin >= -threshold, spin)
    ax.contourf(
        grid_x,
        grid_y,
        positive,
        levels=[threshold, vmax],
        colors=[SPIN_ALPHA],
        alpha=0.28 if video else 0.34,
        antialiased=True,
        zorder=1,
    )
    ax.contourf(
        grid_x,
        grid_y,
        negative,
        levels=[-vmax, -threshold],
        colors=[SPIN_BETA],
        alpha=0.28 if video else 0.34,
        antialiased=True,
        zorder=1,
    )
    ax.contour(
        grid_x,
        grid_y,
        positive,
        levels=[0.55 * vmax],
        colors=[SPIN_ALPHA],
        linewidths=1.1 if video else 0.8,
        alpha=0.85,
        zorder=2,
    )
    ax.contour(
        grid_x,
        grid_y,
        negative,
        levels=[-0.55 * vmax],
        colors=[SPIN_BETA],
        linewidths=1.1 if video else 0.8,
        alpha=0.85,
        zorder=2,
    )


def _draw_vv_loop(ax: plt.Axes, reg: LayoutRegistry, *, video: bool, active_stage: int) -> None:
    """A compact VV loop that remains inside the narrow 03b rail."""

    radius = 0.105 if video else 0.085
    nodes = [
        (0.50, 0.79, "position", r"$\mathbf{r}_{n+1}$"),
        (0.70, 0.35, "accel.", r"$\mathbf{a}_{n+1}$"),
        (0.30, 0.35, "vel.", r"$\mathbf{v}_{n+1}$"),
    ]
    paths = [
        ((0.57, 0.74), (0.64, 0.45), -0.12),
        ((0.62, 0.35), (0.38, 0.35), -0.10),
        ((0.36, 0.45), (0.43, 0.74), -0.12),
    ]
    for start, end, rad in paths:
        reg.arrow(
            ax,
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=22 if video else 14,
            lw=3.0 if video else 2.0,
            color=LINE_GRAY,
            zorder=2,
        )
    for index, (x, y, label, symbol) in enumerate(nodes):
        weight = 1.0 if index == active_stage else 0.0
        fill = "#252525" if weight else "#F4F4F4"
        text_color = WHITE if weight else DARK_GRAY
        ax.add_patch(Circle((x, y), radius, fc=fill, ec=LINE_GRAY, lw=2.0 if video else 1.4, zorder=5))
        reg.text(ax, x, y + 0.018, label, ha="center", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=text_color, weight="bold", zorder=6)
        reg.text(ax, x, y - 0.040, symbol, ha="center", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=text_color, zorder=6)
    reg.text(ax, 0.50, 0.60, "kick → force → update", ha="center", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=DARK_GRAY)
    reg.text(ax, 0.50, 0.53, r"$\Delta t=0.10\,\mathrm{fs}$", ha="center", va="center", fontsize=FONT_SIZES["emphasis"] if video else FONT_SIZES["body"], color=INK, weight="bold")


def _draw_structure(
    ax: plt.Axes,
    reg: LayoutRegistry,
    data: dict[str, np.ndarray],
    frame: int,
    *,
    video: bool,
    centre_3d: np.ndarray,
    half_span: float,
) -> None:
    positions = np.asarray(data["positions"][frame], dtype=float)
    elements = data["elements"].astype(str)
    bonds = np.asarray([[0, 1], [1, 2], [0, 3], [0, 4], [0, 5]], dtype=int)
    rect = (0.10, 0.13, 0.92, 0.87)
    _draw_spin_plane(
        ax,
        data,
        frame,
        rect=rect,
        centre_3d=centre_3d,
        half_span=half_span,
        video=video,
    )
    xy, _ = draw_ball_and_stick(
        ax,
        positions,
        elements,
        bonds,
        rect=rect,
        centre_3d=centre_3d,
        half_span=half_span,
        atom_scale=0.72 if video else 0.66,
        bond_alpha=0.78,
    )
    # Explicitly highlight the reactive Cl--O(H) bond.
    ax.plot(
        [xy[0, 0], xy[1, 0]],
        [xy[0, 1], xy[1, 1]],
        color=CRIMSON,
        lw=7.0 if video else 5.0,
        alpha=0.92,
        solid_capstyle="round",
        zorder=12,
    )
    oh_centre = positions[[1, 2]].mean(axis=0)
    direction = positions[1] - positions[0]
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    draw_vector_arrow(
        ax,
        reg,
        oh_centre,
        direction * 0.55,
        colour=GREEN,
        rect=rect,
        centre_3d=centre_3d,
        half_span=half_span,
        display_scale=1.0,
        video=video,
        alpha=0.95,
    )
    reg.text(
        ax,
        0.08,
        0.115,
        r"red/blue: spin density  $m(r)=\rho_\alpha-\rho_\beta$",
        fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"],
        color=DARK_GRAY,
    )
    reg.text(
        ax,
        0.08,
        0.072,
        f"rC–N = {float(data['r_cn'][frame]):.2f} Å   ·   rN–O = {float(data['r_no'][frame]):.2f} Å",
        fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"],
        color=INK,
        weight="bold",
    )
    reg.text(
        ax,
        0.50,
        0.875,
        r"$\mathrm{TNT\ \rightarrow\ aryl\ radical + \cdot NO_2}$",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZES["emphasis"] if video else FONT_SIZES["body"],
        color=INK,
        weight="bold",
    )
    ax.plot([0.76, 0.82], [0.825, 0.825], color=SPIN_ALPHA, lw=4.0 if video else 3.0, solid_capstyle="round", zorder=13)
    ax.plot([0.76, 0.82], [0.785, 0.785], color=SPIN_BETA, lw=4.0 if video else 3.0, solid_capstyle="round", zorder=13)
    reg.text(ax, 0.835, 0.825, r"$\alpha$", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=SPIN_ALPHA)
    reg.text(ax, 0.835, 0.785, r"$\beta$", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=SPIN_BETA)


def _plot_traces(
    ax: plt.Axes,
    reg: LayoutRegistry,
    data: dict[str, np.ndarray],
    frame: int,
    *,
    video: bool,
) -> None:
    time = np.arange(len(data["r_cn"]), dtype=float) * float(data["dt_fs"])
    shown = slice(0, frame + 1)
    inner = ax.inset_axes([0.16, 0.54, 0.77, 0.34])
    inner.plot(time, data["r_cn"], color=LINE_GRAY, lw=1.2, alpha=0.65)
    inner.plot(time[shown], data["r_cn"][shown], color=POSITION_BLUE, lw=2.3 if video else 1.8)
    inner.scatter([time[frame]], [data["r_cn"][frame]], color=CRIMSON, s=38 if video else 24, zorder=5)
    inner.set_ylabel("Cl-O / Ang", fontsize=12 if video else 9, color=DARK_GRAY)
    inner.set_xlim(time[0], time[-1])
    inner.set_ylim(1.1, max(3.8, float(np.max(data["r_cn"]) + 0.25)))
    inner.tick_params(labelsize=12 if video else 8, colors=DARK_GRAY, labelbottom=False)
    inner.spines[["top", "right"]].set_visible(False)
    inner.spines[["left", "bottom"]].set_color(LINE_GRAY)

    lower = ax.inset_axes([0.16, 0.18, 0.77, 0.25])
    s2 = np.asarray(data["spin_square"], dtype=float)
    energy = np.asarray(data["energy_ev"], dtype=float)
    relative_energy = energy - energy[0]
    lower.plot(time, s2, color=SPIN_ALPHA, lw=1.2, alpha=0.55)
    lower.plot(time[shown], s2[shown], color=SPIN_ALPHA, lw=2.2 if video else 1.7)
    lower.set_ylabel(r"$\langle S^2\rangle$", fontsize=12 if video else 9, color=SPIN_ALPHA)
    lower.set_xlabel("time / fs", fontsize=12 if video else 8, color=DARK_GRAY)
    lower.set_xlim(time[0], time[-1])
    lower.set_ylim(-0.05, max(1.05, float(np.max(s2) + 0.1)))
    lower.set_yticks([0.0, 0.5, 1.0])
    lower.tick_params(labelsize=12 if video else 8, colors=DARK_GRAY)
    lower.spines[["top", "right"]].set_visible(False)
    lower.spines[["left", "bottom"]].set_color(LINE_GRAY)
    energy_axis = lower.twinx()
    energy_axis.plot(time[shown], relative_energy[shown], color=ENERGY_OLIVE, lw=1.5, ls="--")
    energy_axis.set_ylabel(r"$\Delta E$ / eV", fontsize=12 if video else 8, color=ENERGY_OLIVE, labelpad=-12)
    span = max(float(np.max(np.abs(relative_energy))), 0.1)
    energy_axis.set_ylim(-1.1 * span, 0.2 * span)
    energy_axis.tick_params(labelsize=12 if video else 7, colors=ENERGY_OLIVE)
    energy_axis.spines[["top", "left"]].set_visible(False)
    reg.text(
        ax,
        0.50,
        0.475,
        "reaction coordinate + spin localization",
        ha="center",
        va="center",
        fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"],
        color=DARK_GRAY,
    )


def _draw_uks_loop(ax: plt.Axes, reg: LayoutRegistry, *, video: bool) -> None:
    positions = [(0.17, 0.59), (0.50, 0.59), (0.83, 0.59)]
    labels = [(r"$D_\alpha,D_\beta$", "densities"), (r"$F_\alpha,F_\beta$", "potentials"), (r"$C_\alpha,C_\beta$", "orbitals")]
    for index, ((x, y), (title, caption)) in enumerate(zip(positions, labels)):
        width, height = (0.245, 0.22) if video else (0.25, 0.22)
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=LIGHT_GRAY if index != 1 else "#E8EEF2",
                edgecolor=LINE_GRAY,
                linewidth=1.7 if video else 1.2,
            )
        )
        reg.text(ax, x, y + 0.025, title, ha="center", va="center", fontsize=FONT_SIZES["emphasis"] if video else FONT_SIZES["body"], color=NAVY, weight="bold")
        reg.text(ax, x, y - 0.055, caption, ha="center", va="center", fontsize=FONT_SIZES["micro"], color=DARK_GRAY)
        if index < 2:
            reg.arrow(ax, (x + width / 2 + 0.018, y), (positions[index + 1][0] - width / 2 - 0.018, y), arrowstyle="-|>", mutation_scale=18 if video else 12, lw=2.3 if video else 1.5, color=LINE_GRAY)
    reg.arrow(ax, (0.83, 0.46), (0.17, 0.46), connectionstyle="arc3,rad=-0.23", arrowstyle="-|>", mutation_scale=17 if video else 12, lw=2.0 if video else 1.4, color=GREEN)
    reg.text(ax, 0.50, 0.30, r"$\mathrm{SCF}:\ \rho_\alpha,\rho_\beta\ \rightarrow\ \mathrm{self\text{-}consistent}\ \alpha/\beta\ \mathrm{orbitals}$", ha="center", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=INK)
    reg.text(ax, 0.50, 0.17, r"$\mathrm{charge}\ 0\ \cdot\ 50e^-\ \cdot\ \mathrm{BS\ singlet}$", ha="center", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=DARK_GRAY)
    reg.text(ax, 0.50, 0.08, r"UKS allows $\alpha/\beta$ spatial separation", ha="center", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=SPIN_ALPHA, weight="bold")


def compose(
    fig: plt.Figure,
    data: dict[str, np.ndarray],
    manifest: dict,
    time_seconds: float,
    *,
    video: bool,
    reg: LayoutRegistry,
) -> list[dict]:
    slots = VIDEO_SLOTS if video else STATIC_SLOTS
    frame = _frame_index(data, time_seconds)
    centre, half_span = _world_bounds(data)
    rail = axes_from_top_slot(fig, slots["rail"])
    structure = axes_from_top_slot(fig, slots["structure"])
    trace = axes_from_top_slot(fig, slots["trace"])
    uks = axes_from_top_slot(fig, slots["uks"])
    _panel(rail, reg, "VELOCITY VERLET", video=video)
    _panel(structure, reg, "REACTIVE STRUCTURE", video=video)
    _panel(trace, reg, "REACTION OBSERVABLES", video=video)
    _panel(uks, reg, "UKS / SCF", video=video)
    stage = min(2, int((frame / max(len(data["positions"]) - 1, 1)) * 3.0))
    _draw_vv_loop(rail, reg, video=video, active_stage=stage)
    reg.arrow(rail, (0.19, 0.16), (0.81, 0.16), arrowstyle="-|>", mutation_scale=20 if video else 13, lw=3.0 if video else 2.0, color=GREEN)
    reg.text(rail, 0.50, 0.105, "NO2 outward kick", ha="center", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=GREEN, weight="bold")
    reg.text(rail, 0.50, 0.055, f"step {frame:03d} / {len(data['positions']) - 1:03d}", ha="center", va="center", fontsize=FONT_SIZES["body"] if video else FONT_SIZES["micro"], color=DARK_GRAY)
    _draw_structure(structure, reg, data, frame, video=video, centre_3d=centre, half_span=half_span)
    _plot_traces(trace, reg, data, frame, video=video)
    _draw_uks_loop(uks, reg, video=video)
    backend = str(manifest.get("backend", "unknown"))
    footer = fig.add_axes([0, 0, 1, 1])
    footer.axis("off")
    footer_text = "03b UKS reactive AIMD · qualitative kick-started Cl–O(H) homolysis"
    if backend != "pyscf_uks":
        footer_text += " · ANALYTIC SURROGATE / NOT UKS DATA"
    reg.text(footer, 0.50, 0.035, footer_text, ha="center", va="bottom", fontsize=FONT_SIZES["micro"], color=CRIMSON if backend != "pyscf_uks" else DARK_GRAY, weight="bold" if backend != "pyscf_uks" else "normal")
    return [
        {"id": "spin_alpha", "color": SPIN_ALPHA, "min_pixels": 60},
        {"id": "spin_beta", "color": SPIN_BETA, "min_pixels": 60},
        {"id": "reactive_bond", "color": CRIMSON, "min_pixels": 100},
        {"id": "kick", "color": GREEN, "min_pixels": 80},
    ]


def _audit_config() -> dict:
    return {
        "panels": [
            {"id": "rail", "rect": list(VIDEO_SLOTS["rail"]), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            {"id": "structure", "rect": list(VIDEO_SLOTS["structure"]), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            {"id": "trace", "rect": list(VIDEO_SLOTS["trace"]), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            {"id": "uks", "rect": list(VIDEO_SLOTS["uks"]), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.012, "min_panel_bbox_fill": 0.13, "grid_rows": 12, "grid_columns": 24},
        "bands": [
            {"id": "rail_structure_gap", "rect": [0.215, 0.025, 0.230, 0.975], "max_ink_pixels": 5000},
            {"id": "structure_trace_gap", "rect": [0.680, 0.025, 0.695, 0.975], "max_ink_pixels": 5000},
        ],
    }


def main() -> None:
    global DATA_PATH, MANIFEST_PATH
    parser = argparse.ArgumentParser(description="Render the 03b UKS reactive AIMD story")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    DATA_PATH = args.data.resolve()
    MANIFEST_PATH = DATA_PATH.with_suffix(".json")
    data, manifest = load_data()
    fig = new_static_figure()
    static_registry = LayoutRegistry(min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=18, font_family=None, coerce_min_font=True)
    compose(fig, data, manifest, VIDEO_DURATION, video=False, reg=static_registry)
    errors = static_registry.validate(fig)
    if errors:
        raise RuntimeError("03b static layout failed: " + "; ".join(errors))
    save_static(fig, STEM)
    if args.static_only:
        return

    def draw_frame(frame_fig: plt.Figure, time_seconds: float, _index: int, registry: LayoutRegistry) -> list[dict]:
        return compose(frame_fig, data, manifest, time_seconds, video=True, reg=registry)

    render_video(
        stem=STEM,
        duration_seconds=VIDEO_DURATION,
        draw_frame=draw_frame,
        audit_config=_audit_config(),
        qa_directory=QA_DIR / "_qa",
        representative_times=[0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 15.5],
    )
    strict = {
        "stem": STEM,
        "backend": manifest.get("backend"),
        "static": {"width": 3508, "height": 2480, "path": str(ROOT / "figures" / f"{STEM}.png")},
        "video": {"width": 1920, "height": 600, "fps": 24, "path": str(ROOT / "videos" / f"{STEM}.mp4")},
        "passed": True,
    }
    (QA_DIR / "qa_report_strict.json").parent.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "qa_report_strict.json").write_text(json.dumps(strict, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    # 03b must follow the canonical 03 AIMD/MatterVis renderer exactly:
    # nested SCF -> force -> velocity -> position timing, native MatterVis
    # assets, and the 30 s/24 fps timeline.  Keep this historical entry point
    # as a compatibility wrapper for callers that already use its filename.
    from render_uks_aimd import main as exact_aimd_main

    exact_aimd_main()

