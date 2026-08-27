"""Render the independent A4 figure and 16:9 AIMD/SCF video."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
from skimage.measure import marching_cubes

from common import (
    CRIMSON,
    DARK_GRAY,
    GREEN,
    INK,
    LIGHT_GRAY,
    LINE_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    add_footer,
    add_page_title,
    axes_from_top_slot,
    draw_ball_and_stick,
    draw_three_step_loop,
    draw_vector_arrow,
    map_projected_to_rect,
    mix_hex,
    new_static_figure,
    project_points,
    render_source_panel,
    render_video,
    save_static,
    smoothstep,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "aimd_h2o_dimer.npz"
QA_DIR = ROOT / "_qa" / "03_aimd_scf"
STEM = "03_aimd_scf"

STATIC_LEFT = (0.035, 0.27, 0.265, 0.79)
STATIC_MIDDLE = (0.30, 0.20, 0.72, 0.88)
STATIC_RIGHT = (0.76, 0.22, 0.97, 0.83)
VIDEO_LEFT = (0.030, 0.20, 0.270, 0.88)
VIDEO_MIDDLE = (0.305, 0.20, 0.705, 0.88)
VIDEO_RIGHT = (0.750, 0.20, 0.970, 0.86)

FORCE_DISPLAY_SCALE = 0.25
MOVE_DISPLAY_SCALE = 60.0
TIME_STEP_FS = 0.50
ACCELERATION_FACTOR = 0.00964853322  # (eV/A)/amu -> A/fs^2


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def precompute_surfaces(data: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    grid_x = data["grid_x"]
    grid_y = data["grid_y"]
    grid_z = data["grid_z"]
    spacing = (float(np.diff(grid_x).mean()), float(np.diff(grid_y).mean()), float(np.diff(grid_z).mean()))
    origin = np.array([grid_x[0], grid_y[0], grid_z[0]], dtype=float)
    result = []
    for volume in data["density_volumes"]:
        vertices, faces, _, _ = marching_cubes(volume, level=0.030, spacing=spacing, step_size=2)
        result.append((vertices + origin, faces.astype(int)))
    return result


def _surface_collection(
    ax: plt.Axes,
    surface: tuple[np.ndarray, np.ndarray],
    *,
    rect: tuple[float, float, float, float],
    centre_3d: np.ndarray,
    half_span: float,
    alpha: float,
) -> None:
    vertices, faces = surface
    projected, depth = project_points(vertices)
    centre_projected, _ = project_points(centre_3d[None, :])
    mapped = map_projected_to_rect(projected, rect, centre_projected[0], half_span)
    order = np.argsort(depth[faces].mean(axis=1))
    polygons = mapped[faces[order]]
    collection = PolyCollection(
        polygons,
        facecolors=NAVY,
        edgecolors=NAVY,
        linewidths=0.16,
        alpha=alpha,
        zorder=2,
    )
    ax.add_collection(collection)


def _motion_vectors(data: dict[str, np.ndarray]) -> np.ndarray:
    masses = np.array([15.999 if element == "O" else 1.008 for element in data["elements"]])
    acceleration = data["forces"] / masses[:, None] * ACCELERATION_FACTOR
    return 0.5 * acceleration * TIME_STEP_FS**2


def draw_dimer_density(
    ax: plt.Axes,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    surfaces: list[tuple[np.ndarray, np.ndarray]],
    *,
    surface_index: int,
    video: bool,
    density_alpha: float = 0.18,
    force_weight: float = 1.0,
    move_weight: float = 0.0,
) -> None:
    centre = np.asarray(data["positions"], dtype=float).mean(axis=0)
    rect = (0.025, 0.20 if video else 0.18, 0.975, 0.84 if video else 0.86)
    half_span = 2.65
    positions = np.asarray(data["positions"], dtype=float)
    movement = _motion_vectors(data) * MOVE_DISPLAY_SCALE * move_weight
    if density_alpha > 0.001:
        _surface_collection(
            ax,
            surfaces[surface_index],
            rect=rect,
            centre_3d=centre,
            half_span=half_span,
            alpha=density_alpha,
        )
    if move_weight > 0.02:
        draw_ball_and_stick(
            ax,
            positions,
            data["elements"],
            data["bonds"],
            rect=rect,
            centre_3d=centre,
            half_span=half_span,
            alpha=0.22,
            atom_scale=1.00 if video else 0.65,
            bond_alpha=0.25,
            edge_color=LINE_GRAY,
        )
    atom_xy, _ = draw_ball_and_stick(
        ax,
        positions + movement,
        data["elements"],
        data["bonds"],
        rect=rect,
        centre_3d=centre,
        half_span=half_span,
        atom_scale=1.15 if video else 0.72,
    )
    first, second = (int(value) for value in data["hydrogen_bond"])
    ax.plot(
        [atom_xy[first, 0], atom_xy[second, 0]],
        [atom_xy[first, 1], atom_xy[second, 1]],
        color=GREEN,
        lw=2.8 if video else 1.8,
        ls=(0, (4, 5)),
        alpha=0.60,
        zorder=5,
    )
    if force_weight > 0.02:
        for position, force in zip(positions, data["forces"]):
            draw_vector_arrow(
                ax,
                registry,
                position,
                force,
                colour=CRIMSON,
                rect=rect,
                centre_3d=centre,
                half_span=half_span,
                display_scale=FORCE_DISPLAY_SCALE,
                video=video,
                alpha=force_weight,
            )
    if move_weight > 0.02:
        for position, displacement in zip(positions, _motion_vectors(data)):
            draw_vector_arrow(
                ax,
                registry,
                position,
                displacement,
                colour=GREEN,
                rect=rect,
                centre_3d=centre,
                half_span=half_span,
                display_scale=MOVE_DISPLAY_SCALE * move_weight,
                video=video,
                alpha=move_weight,
            )


def draw_md_loop(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    video: bool,
) -> None:
    ax.set_aspect("equal", adjustable="box")
    radius = 0.125 if video else 0.105
    nodes = [
        (0.50, 0.79, r"$\mathbf{r}$", "position", DARK_GRAY),
        (0.78, 0.31, r"$\mathbf{a}$", "acceleration", CRIMSON),
        (0.22, 0.31, r"$\mathbf{v}$", "velocity", DARK_GRAY),
    ]
    paths = [
        ((0.57, 0.72), (0.72, 0.42), -0.10),
        ((0.66, 0.27), (0.34, 0.27), -0.10),
        ((0.27, 0.42), (0.43, 0.72), -0.10),
    ]
    for start, end, curve in paths:
        registry.arrow(
            ax,
            start,
            end,
            connectionstyle=f"arc3,rad={curve}",
            arrowstyle="-|>",
            mutation_scale=27 if video else 18,
            lw=4.0 if video else 2.4,
            color=LINE_GRAY,
            zorder=2,
        )
    for index, (x, y, symbol, label, label_colour) in enumerate(nodes):
        active = index == 1
        ax.add_patch(Circle((x, y), radius, fc=INK if active else LIGHT_GRAY, ec=LINE_GRAY, lw=2.8 if video else 1.8, zorder=4))
        registry.text(ax, x, y, symbol, ha="center", va="center", fontsize=25 if video else 15, color=WHITE if active else DARK_GRAY, weight="bold", zorder=5)
        label_y = y + radius + (0.060 if index == 0 else -2.0 * radius - 0.055)
        registry.text(ax, x, label_y, label, ha="center", va="center", fontsize=18 if video else 10, color=label_colour, weight="bold" if active else "normal")
    registry.text(ax, 0.50, 0.55, "one MD step", ha="center", va="center", fontsize=19 if video else 11, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.47, r"$\Delta t$", ha="center", va="center", fontsize=24 if video else 14, color=INK, weight="bold")


def draw_scf_loop(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    video: bool,
    active_stage: int | None,
    active_weight: float,
    iteration: int,
    delta_energy: float,
    converged: bool,
) -> None:
    ax.set_aspect("equal", adjustable="box")
    nodes = [
        (0.50, 0.82, r"$\rho^k$", "density", (0.50, 0.925)),
        (0.80, 0.64, r"$F$", "build Fock", (0.80, 0.825)),
        (0.69, 0.29, r"$C$", "solve", (0.84, 0.18)),
        (0.31, 0.29, r"$\rho'$", "new density", (0.25, 0.12)),
        (0.20, 0.64, r"$\Delta E$", "converged?", (0.24, 0.825)),
    ]
    radius = 0.080 if video else 0.061
    paths = [
        ((0.57, 0.78), (0.73, 0.68), -0.04),
        ((0.78, 0.57), (0.72, 0.37), -0.04),
        ((0.61, 0.29), (0.39, 0.29), -0.04),
        ((0.28, 0.37), (0.22, 0.57), -0.04),
        ((0.27, 0.68), (0.43, 0.78), -0.04),
    ]
    for start, end, curve in paths:
        registry.arrow(
            ax,
            start,
            end,
            connectionstyle=f"arc3,rad={curve}",
            arrowstyle="-|>",
            mutation_scale=26 if video else 17,
            lw=3.8 if video else 2.2,
            color=LINE_GRAY,
            zorder=2,
        )
    for index, (x, y, symbol, label, label_position) in enumerate(nodes):
        weight = active_weight if active_stage == index else 0.0
        fill = mix_hex(LIGHT_GRAY, INK, weight)
        edge = GREEN if converged and index == 4 else LINE_GRAY
        text_color = WHITE if weight > 0.48 else DARK_GRAY
        ax.add_patch(Circle((x, y), radius, fc=fill, ec=edge, lw=3.0 if video else 2.0, zorder=4))
        registry.text(
            ax,
            x,
            y,
            symbol,
            ha="center",
            va="center",
            fontsize=18 if video else 10,
            color=text_color,
            weight="bold",
            zorder=5,
        )
        registry.text(ax, label_position[0], label_position[1], label, ha="center", va="center", fontsize=18 if video else 10, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.58, "SCF", ha="center", va="center", fontsize=25 if video else 14, color=INK, weight="bold")
    registry.text(ax, 0.50, 0.49, f"iteration {iteration:02d}/19", ha="center", va="center", fontsize=18 if video else 10, color=DARK_GRAY)
    delta_text = f"|ΔE| = {delta_energy:.1e} Ha" if np.isfinite(delta_energy) else "initial density"
    registry.text(ax, 0.50, 0.025, delta_text, ha="center", va="center", fontsize=18 if video else 10, color=GREEN if converged else DARK_GRAY, weight="bold" if converged else "normal")


def _draw_md_source(ax: plt.Axes, registry: LayoutRegistry) -> None:
    draw_md_loop(ax, registry, video=False)


def _draw_density_source(ax: plt.Axes, registry: LayoutRegistry, data, surfaces) -> None:
    registry.text(ax, 0.50, 0.975, "one concrete H₂O dimer", ha="center", va="top", fontsize=13, color=INK, weight="bold")
    draw_dimer_density(ax, registry, data, surfaces, surface_index=-1, video=False, density_alpha=0.18, force_weight=1.0)
    registry.text(ax, 0.50, 0.115, "3D electron-density isosurface · force arrows ×0.25", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.055, r"$\mathbf{F}=-\nabla_{\mathbf{R}}E_{\mathrm{SCF}}$", ha="center", va="center", fontsize=12, color=INK)


def _draw_scf_source(ax: plt.Axes, registry: LayoutRegistry, data) -> None:
    energies = data["energies"]
    final_delta = abs(float(energies[-1] - energies[-2]))
    draw_scf_loop(
        ax,
        registry,
        video=False,
        active_stage=4,
        active_weight=0.0,
        iteration=len(energies),
        delta_energy=final_delta,
        converged=True,
    )


def render_static(data, surfaces) -> None:
    render_source_panel(QA_DIR / "source" / "md_loop.png", _draw_md_source, width_px=900, height_px=1400)
    render_source_panel(QA_DIR / "source" / "dimer_density.png", lambda ax, reg: _draw_density_source(ax, reg, data, surfaces), width_px=1550, height_px=1450)
    render_source_panel(QA_DIR / "source" / "scf_loop.png", lambda ax, reg: _draw_scf_source(ax, reg, data), width_px=900, height_px=1350)
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    add_page_title(fig, "03", "Ab initio molecular dynamics", "one nuclear force requires a self-consistent electronic calculation", video=False, registry=registry)
    left = axes_from_top_slot(fig, STATIC_LEFT)
    middle = axes_from_top_slot(fig, STATIC_MIDDLE)
    right = axes_from_top_slot(fig, STATIC_RIGHT)
    _draw_md_source(left, registry)
    _draw_density_source(middle, registry, data, surfaces)
    _draw_scf_source(right, registry, data)
    add_footer(fig, "H₂O dimer · pedagogical RHF/STO-3G · 19 SCF iterations · repeated at every MD step", video=False, registry=registry)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    png, svg = save_static(fig, STEM)
    print(f"figure: {png}")
    print(f"vector: {svg}")


def _video_state(time_seconds: float, energies: np.ndarray) -> tuple[str, int, int | None, float, bool]:
    if time_seconds < 1.5:
        return "request", 1, None, 0.0, False
    if time_seconds < 14.8:
        elapsed = time_seconds - 1.5
        iteration_float = min(elapsed / 13.3 * len(energies), len(energies) - 1e-6)
        iteration_zero = int(iteration_float)
        within = iteration_float - iteration_zero
        stage = min(int(within * 5.0), 4)
        edge = min((within * 5.0) % 1.0 / 0.22, 1.0)
        return "scf", iteration_zero + 1, stage, smoothstep(edge), False
    return "converged", len(energies), None, 0.0, True


def _surface_index(iteration_one_based: int, density_iterations: np.ndarray) -> int:
    iteration_zero = max(iteration_one_based - 1, 0)
    candidates = np.where(density_iterations <= iteration_zero)[0]
    return int(candidates[-1]) if len(candidates) else 0


def _draw_video_frame(fig, time_seconds, frame_index, registry, data, surfaces):
    energies = np.asarray(data["energies"], dtype=float)
    state, iteration, stage, active_weight, converged = _video_state(time_seconds, energies)
    add_page_title(fig, "03", "Ab initio MD: the SCF loop", "a single nuclear force query repeatedly rebuilds the electronic state", video=True, registry=registry)
    left = axes_from_top_slot(fig, VIDEO_LEFT)
    middle = axes_from_top_slot(fig, VIDEO_MIDDLE)
    right = axes_from_top_slot(fig, VIDEO_RIGHT)
    draw_md_loop(left, registry, video=True)

    if iteration <= 1:
        delta = float("nan")
    else:
        delta = abs(float(energies[iteration - 1] - energies[iteration - 2]))
    density_index = _surface_index(iteration, data["density_iterations"])
    if converged:
        force_in = smoothstep((time_seconds - 14.8) / 0.45)
        force_out = smoothstep((time_seconds - 15.65) / 0.45)
        force_weight = force_in * (1.0 - force_out)
        move_weight = smoothstep((time_seconds - 15.65) / 0.70)
        final_weight = max(force_in, move_weight)
    else:
        force_weight = 0.0
        move_weight = 0.0
        final_weight = 0.0
    draw_dimer_density(
        middle,
        registry,
        data,
        surfaces,
        surface_index=density_index,
        video=True,
        density_alpha=0.18 * (1.0 - 0.70 * final_weight),
        force_weight=force_weight,
        move_weight=move_weight,
    )
    if state == "request":
        headline, colour = "nuclei fixed: start electronic solve", NAVY
    elif state == "scf":
        headline, colour = f"SCF iteration {iteration:02d}/19", INK
    elif time_seconds < 15.65:
        headline, colour = "SCF converged → nuclear forces", CRIMSON
    else:
        headline, colour = "forces → move nuclei", GREEN
    registry.text(middle, 0.50, 0.965, headline, ha="center", va="top", fontsize=24, color=colour, weight="bold")
    if converged and time_seconds < 15.65:
        registry.text(middle, 0.50, 0.105, "red force arrows ×0.25", ha="center", va="center", fontsize=18, color=DARK_GRAY)
    elif converged:
        registry.text(middle, 0.50, 0.105, "green nuclear move ×60 · old positions grey", ha="center", va="center", fontsize=18, color=DARK_GRAY)
    else:
        registry.text(middle, 0.50, 0.105, "true 3D density isosurface · nuclei remain fixed", ha="center", va="center", fontsize=18, color=DARK_GRAY)

    draw_scf_loop(
        right,
        registry,
        video=True,
        active_stage=stage,
        active_weight=active_weight,
        iteration=iteration,
        delta_energy=delta,
        converged=converged,
    )
    add_footer(fig, "real H₂O dimer · RHF/STO-3G teaching calculation · fixed asymmetric orthographic camera", video=True, registry=registry)
    semantics = [
        {"id": "density", "color": NAVY, "min_pixels": 350},
        {"id": "oxygen", "color": CRIMSON, "min_pixels": 260},
    ]
    if converged:
        semantics.append({"id": "converged", "color": GREEN, "min_pixels": 260})
    return semantics


def render_animation(data, surfaces) -> None:
    audit_config = {
        "panels": [
            {"id": "md_loop", "rect": list(VIDEO_LEFT), "min_clearance_px": 16},
            {"id": "dimer_density", "rect": list(VIDEO_MIDDLE), "min_clearance_px": 16},
            {"id": "scf_loop", "rect": list(VIDEO_RIGHT), "min_clearance_px": 16},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.020, "min_panel_bbox_fill": 0.30, "grid_rows": 12, "grid_columns": 24},
        "bands": [
            {"id": "left_gap", "rect": [0.278, 0.205, 0.295, 0.88], "max_ink_pixels": 0},
            {"id": "right_gap", "rect": [0.715, 0.205, 0.738, 0.88], "max_ink_pixels": 0},
        ],
    }
    output = render_video(
        stem=STEM,
        duration_seconds=17.0,
        draw_frame=lambda fig, t, i, reg: _draw_video_frame(fig, t, i, reg, data, surfaces),
        audit_config=audit_config,
        qa_directory=QA_DIR / "_qa",
        representative_times=[0.8, 4.2, 9.0, 14.2, 16.2],
    )
    print(f"video: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    surfaces = precompute_surfaces(data)
    render_static(data, surfaces)
    if not args.static_only:
        render_animation(data, surfaces)


if __name__ == "__main__":
    main()
