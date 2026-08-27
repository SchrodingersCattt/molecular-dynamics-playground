"""Render the independent A4 figure and 16:9 Deep Potential MD video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

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
BASE_PATH = ROOT / "data" / "water_box_64.npz"
RESULT_PATH = ROOT / "data" / "dpmd_water_box_results.npz"
META_PATH = ROOT / "data" / "dpmd_eval.json"
QA_DIR = ROOT / "_qa" / "04_dpmd"
STEM = "04_deep_potential_md"

STATIC_LEFT = (0.035, 0.245, 0.255, 0.835)
STATIC_MIDDLE = (0.285, 0.185, 0.715, 0.895)
STATIC_RIGHT = (0.750, 0.215, 0.970, 0.855)
VIDEO_LEFT = (0.030, 0.205, 0.270, 0.875)
VIDEO_MIDDLE = (0.305, 0.185, 0.720, 0.895)
VIDEO_RIGHT = (0.755, 0.205, 0.970, 0.875)

FORCE_DISPLAY_SCALE = 1.8
VIDEO_DURATION_SECONDS = 16.0
STAGE_SECONDS = VIDEO_DURATION_SECONDS / 4.0


def load_data() -> dict[str, np.ndarray | float | int | dict]:
    with np.load(BASE_PATH, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(RESULT_PATH, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    positions = np.asarray(data["positions_wrapped"], dtype=float)
    if not np.allclose(positions, result["positions"], atol=1.0e-12):
        raise RuntimeError("DeepMD result positions do not match the prepared water box")
    if len(positions) != 192 or int(np.count_nonzero(data["neighbor_mask"])) != 83:
        raise RuntimeError("Water-box atom or exact-neighbor count changed")
    if int(data["central_index"]) != 126:
        raise RuntimeError("Stable central source ID changed")
    if not np.all(np.isfinite(result["forces_ev_per_angstrom"])):
        raise RuntimeError("DeepMD result contains a non-finite force")
    if not np.allclose(np.sum(result["forces_ev_per_angstrom"], axis=0), metadata["net_force_ev_per_angstrom"], atol=1.0e-12):
        raise RuntimeError("DeepMD metadata and retained force array disagree")
    return {**data, **{f"result_{key}": value for key, value in result.items()}, "metadata": metadata}


def _map_points(points: np.ndarray, rect, centre_3d: np.ndarray, half_span: float) -> tuple[np.ndarray, np.ndarray]:
    projected, depth = project_points(points)
    centre_projected, _ = project_points(centre_3d[None, :])
    return map_projected_to_rect(projected, rect, centre_projected[0], half_span), depth


def _cell_corners(length: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    corners = np.array([[x, y, z] for x in (0.0, length) for y in (0.0, length) for z in (0.0, length)])
    edges = []
    for first in range(8):
        for second in range(first + 1, 8):
            if np.count_nonzero(corners[first] != corners[second]) == 1:
                edges.append((first, second))
    return corners, edges


def _draw_cell(ax: plt.Axes, length: float, rect, centre: np.ndarray, half_span: float, *, video: bool, alpha: float = 1.0) -> None:
    corners, edges = _cell_corners(length)
    xy, depth = _map_points(corners, rect, centre, half_span)
    for first, second in sorted(edges, key=lambda edge: float(depth[list(edge)].mean())):
        ax.plot(
            [xy[first, 0], xy[second, 0]],
            [xy[first, 1], xy[second, 1]],
            color=DARK_GRAY,
            lw=2.2 if video else 1.25,
            alpha=0.72 * alpha,
            zorder=2,
        )


def _draw_sphere(
    ax: plt.Axes,
    centre: np.ndarray,
    radius: float,
    rect,
    half_span: float,
    *,
    video: bool,
    alpha: float,
    reveal: float,
) -> None:
    centre_xy, _ = _map_points(centre[None, :], rect, centre, half_span)
    scale = min(rect[2] - rect[0], rect[3] - rect[1]) / (2.0 * half_span)
    ax.add_patch(Circle(tuple(centre_xy[0]), radius * scale, fc=NAVY, ec="none", alpha=0.035 * alpha, zorder=1))
    ax.add_patch(
        Circle(
            tuple(centre_xy[0]),
            radius * scale,
            fc="none",
            ec=NAVY,
            lw=2.5 if video else 1.4,
            alpha=0.58 * alpha,
            zorder=3,
        )
    )
    latitudes = np.deg2rad([-60, -30, 0, 30, 60])
    longitudes = np.deg2rad(np.arange(0, 180, 22.5))
    curves = []
    theta = np.linspace(0, 2 * np.pi, 160)
    for latitude in latitudes:
        curves.append(
            centre
            + radius
            * np.column_stack(
                (
                    np.cos(latitude) * np.cos(theta),
                    np.cos(latitude) * np.sin(theta),
                    np.full_like(theta, np.sin(latitude)),
                )
            )
        )
    polar = np.linspace(-np.pi / 2, np.pi / 2, 120)
    for longitude in longitudes:
        curves.append(
            centre
            + radius
            * np.column_stack(
                (
                    np.cos(polar) * np.cos(longitude),
                    np.cos(polar) * np.sin(longitude),
                    np.sin(polar),
                )
            )
        )
    visible = max(1, int(np.ceil(len(curves) * float(np.clip(reveal, 0.0, 1.0)))))
    for points in curves[:visible]:
        xy, _ = _map_points(points, rect, centre, half_span)
        ax.plot(xy[:, 0], xy[:, 1], color=NAVY, lw=2.0 if video else 1.15, alpha=0.38 * alpha, zorder=3)


def _draw_atoms(
    ax: plt.Axes,
    positions: np.ndarray,
    elements: np.ndarray,
    rect,
    centre: np.ndarray,
    half_span: float,
    *,
    video: bool,
    alpha: np.ndarray,
    central_index: int,
) -> np.ndarray:
    xy, depth = _map_points(positions, rect, centre, half_span)
    order = np.argsort(depth)
    for index in order:
        element = str(elements[index])
        colour = CRIMSON if element == "O" else NAVY
        base_size = 170 if element == "O" else 62
        size = base_size if video else base_size * 0.72
        edge = INK if index == central_index else WHITE
        edge_width = 2.8 if index == central_index and video else 1.8 if index == central_index else 0.8
        ax.scatter(
            xy[index, 0],
            xy[index, 1],
            s=size * (1.75 if index == central_index else 1.0),
            c=colour,
            alpha=float(alpha[index]),
            edgecolors=edge,
            linewidths=edge_width,
            zorder=10 + int(index == central_index),
        )
    return xy


def _draw_bonds(ax: plt.Axes, xy: np.ndarray, bonds: np.ndarray, depth: np.ndarray, *, video: bool, alpha: float) -> None:
    for first, second in sorted(np.asarray(bonds, dtype=int), key=lambda edge: float(depth[edge].mean())):
        ax.plot(
            [xy[first, 0], xy[second, 0]],
            [xy[first, 1], xy[second, 1]],
            color=DARK_GRAY,
            lw=2.8 if video else 1.45,
            alpha=alpha,
            solid_capstyle="round",
            zorder=5,
        )


def _force_indices(data) -> np.ndarray:
    mask = np.asarray(data["neighbor_mask"], dtype=bool)
    central = int(data["central_index"])
    forces = np.asarray(data["result_forces_ev_per_angstrom"], dtype=float)
    local = np.where(mask)[0]
    ranked = local[np.argsort(np.linalg.norm(forces[local], axis=1))[::-1]]
    return np.unique(np.concatenate(([central], ranked[:11])))


def draw_water_scene(
    ax: plt.Axes,
    registry: LayoutRegistry,
    data,
    *,
    rect,
    video: bool,
    mode: str,
    reveal: float = 1.0,
) -> None:
    length = float(data["box_length"])
    cutoff = float(data["cutoff"])
    centre = np.full(3, length / 2.0)
    half_span = 9.15
    elements = np.asarray(data["elements"]).astype(str)
    central = int(data["central_index"])
    neighbor_mask = np.asarray(data["neighbor_mask"], dtype=bool)
    if mode == "bonds":
        positions = np.asarray(data["positions_bond_view"], dtype=float)
        alpha = np.ones(len(positions))
        _draw_cell(ax, length, rect, centre, half_span, video=video)
        xy, depth = _map_points(positions, rect, centre, half_span)
        _draw_bonds(ax, xy, data["bonds"], depth, video=video, alpha=0.72)
        _draw_atoms(ax, positions, elements, rect, centre, half_span, video=video, alpha=alpha, central_index=central)
        return

    positions = np.asarray(data["positions_neighbor_view"], dtype=float)
    outside_alpha = 1.0 - 0.92 * smoothstep(reveal)
    alpha = np.where(neighbor_mask | (np.arange(len(positions)) == central), 1.0, outside_alpha)
    _draw_cell(ax, length, rect, centre, half_span, video=video, alpha=0.78)
    _draw_sphere(ax, centre, cutoff, rect, half_span, video=video, alpha=smoothstep(reveal), reveal=reveal)
    xy, depth = _map_points(positions, rect, centre, half_span)
    if mode in {"neighbors", "forces"}:
        for index in np.where(neighbor_mask)[0][np.argsort(depth[neighbor_mask])]:
            ax.plot(
                [xy[central, 0], xy[index, 0]],
                [xy[central, 1], xy[index, 1]],
                color=NAVY,
                lw=1.5 if video else 0.85,
                alpha=0.18 + 0.22 * smoothstep(reveal),
                zorder=4,
            )
    _draw_atoms(ax, positions, elements, rect, centre, half_span, video=video, alpha=alpha, central_index=central)
    if mode == "forces":
        forces = np.asarray(data["result_forces_ev_per_angstrom"], dtype=float)
        for index in _force_indices(data):
            start = positions[index]
            end = start + FORCE_DISPLAY_SCALE * forces[index]
            points, _ = _map_points(np.vstack((start, end)), rect, centre, half_span)
            registry.arrow(
                ax,
                tuple(points[0]),
                tuple(points[1]),
                arrowstyle="-|>",
                mutation_scale=24 if video else 14,
                lw=3.5 if video else 2.0,
                color=GREEN,
                alpha=smoothstep(reveal),
                zorder=18,
            )


def _rounded_node(ax, x, y, width, height, fill, edge, *, lw=2.0, radius=0.025, zorder=4):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        fc=fill,
        ec=edge,
        lw=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_dp_pipeline(
    ax: plt.Axes,
    registry: LayoutRegistry,
    data,
    *,
    video: bool,
    active_stage: int | None,
    active_weight: float,
) -> None:
    stages = [
        (0.88, "local neighbors", r"$\{s_j,\,\mathbf{r}_{ij}\}_{r_{ij}<r_c}$"),
        (0.71, "descriptor", r"$\mathcal{D}_i$"),
        (0.54, "shared atomic network", r"$\mathrm{NN}_{\theta}$"),
        (0.37, "atomic energies", r"$\varepsilon_1\quad\varepsilon_i\quad\varepsilon_N$"),
        (0.19, "sum and differentiate", r"$E=\sum_i\varepsilon_i$   ·   $\mathbf{F}=-\nabla_{\mathbf{R}}E$"),
    ]
    for index in range(len(stages) - 1):
        registry.arrow(
            ax,
            (0.50, stages[index][0] - 0.073),
            (0.50, stages[index + 1][0] + 0.073),
            arrowstyle="-|>",
            mutation_scale=27 if video else 18,
            lw=4.0 if video else 2.35,
            color=INK if active_stage == index + 1 and active_weight > 0.45 else LINE_GRAY,
            zorder=2,
        )
    for index, (y, label, symbol) in enumerate(stages):
        weight = active_weight if active_stage == index else 0.0
        fill = mix_hex(LIGHT_GRAY, INK, weight)
        edge = GREEN if index >= 3 else LINE_GRAY
        text_colour = WHITE if weight > 0.48 else GREEN if index == 3 else INK
        width = 0.84 if index in {0, 2, 4} else 0.70
        _rounded_node(ax, 0.50, y, width, 0.105, fill, edge, lw=2.7 if video else 1.7)
        registry.text(ax, 0.50, y + 0.028, label, ha="center", va="center", fontsize=18 if video else 10, color=text_colour, weight="bold", zorder=5)
        registry.text(ax, 0.50, y - 0.033, symbol, ha="center", va="center", fontsize=18 if video else 10, color=WHITE if weight > 0.48 else INK, zorder=5)
    energy = float(data["result_total_energy_ev"])
    max_force = float(data["metadata"]["max_force_ev_per_angstrom"])
    registry.text(
        ax,
        0.50,
        0.005,
        f"real inference: E = {energy:.3f} eV\nmax |F| = {max_force:.3f} eV Å⁻¹",
        ha="center",
        va="bottom",
        fontsize=18 if video else 10,
        color=GREEN,
        weight="bold",
    )


def draw_md_loop(ax: plt.Axes, registry: LayoutRegistry, *, video: bool) -> None:
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


def _draw_md_source(ax: plt.Axes, registry: LayoutRegistry) -> None:
    draw_md_loop(ax, registry, video=False)


def _draw_environment_source(ax: plt.Axes, registry: LayoutRegistry, data) -> None:
    registry.text(ax, 0.50, 0.985, "one concrete 64-water periodic case", ha="center", va="top", fontsize=13, color=INK, weight="bold")
    registry.text(ax, 0.025, 0.935, "chemical-bond view", ha="left", va="top", fontsize=10, color=DARK_GRAY, weight="bold")
    draw_water_scene(ax, registry, data, rect=(0.02, 0.54, 0.98, 0.925), video=False, mode="bonds")
    registry.text(ax, 0.025, 0.495, "neighbor view", ha="left", va="top", fontsize=10, color=DARK_GRAY, weight="bold")
    draw_water_scene(ax, registry, data, rect=(0.02, 0.085, 0.98, 0.475), video=False, mode="neighbors")
    registry.text(
        ax,
        0.50,
        0.020,
        "central source ID 126 · exact 6.0 Å sphere · 83 minimum-image neighbors",
        ha="center",
        va="bottom",
        fontsize=10,
        color=NAVY,
        weight="bold",
    )


def _draw_pipeline_source(ax: plt.Axes, registry: LayoutRegistry, data) -> None:
    registry.text(ax, 0.50, 0.985, "Deep Potential", ha="center", va="top", fontsize=13, color=INK, weight="bold")
    draw_dp_pipeline(ax, registry, data, video=False, active_stage=None, active_weight=0.0)


def render_static(data) -> None:
    render_source_panel(QA_DIR / "source" / "md_loop.png", _draw_md_source, width_px=900, height_px=1400)
    render_source_panel(
        QA_DIR / "source" / "water_environment.png",
        lambda ax, reg: _draw_environment_source(ax, reg, data),
        width_px=1650,
        height_px=1600,
    )
    render_source_panel(
        QA_DIR / "source" / "learned_model.png",
        lambda ax, reg: _draw_pipeline_source(ax, reg, data),
        width_px=900,
        height_px=1450,
    )
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    add_page_title(
        fig,
        "04",
        "Deep Potential molecular dynamics",
        "replace the electronic solve with a local, differentiable energy model",
        video=False,
        registry=registry,
    )
    left = axes_from_top_slot(fig, STATIC_LEFT)
    middle = axes_from_top_slot(fig, STATIC_MIDDLE)
    right = axes_from_top_slot(fig, STATIC_RIGHT)
    _draw_md_source(left, registry)
    _draw_environment_source(middle, registry, data)
    _draw_pipeline_source(right, registry, data)
    add_footer(
        fig,
        "prepared 64-water box (not equilibrated) · DeepMD-kit 3.1.3 inference · fixed asymmetric orthographic camera",
        video=False,
        registry=registry,
    )
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    png, svg = save_static(fig, STEM)
    print(f"figure: {png}")
    print(f"vector: {svg}")


def _video_stage(time_seconds: float) -> tuple[int, float]:
    bounded = min(max(time_seconds, 0.0), VIDEO_DURATION_SECONDS - 1.0e-9)
    stage = int(bounded // STAGE_SECONDS)
    local = (bounded - stage * STAGE_SECONDS) / STAGE_SECONDS
    return stage, local


def _draw_video_frame(fig, time_seconds, frame_index, registry, data):
    stage, local = _video_stage(time_seconds)
    fade = smoothstep(min(local / 0.24, 1.0))
    add_page_title(
        fig,
        "04",
        "Deep Potential MD: from neighbors to forces",
        "one real 64-water box · one fixed 3D camera · one exact 6 Å cutoff",
        video=True,
        registry=registry,
    )
    left = axes_from_top_slot(fig, VIDEO_LEFT)
    middle = axes_from_top_slot(fig, VIDEO_MIDDLE)
    right = axes_from_top_slot(fig, VIDEO_RIGHT)
    draw_md_loop(left, registry, video=True)

    if stage == 0:
        mode = "bonds"
        headline = "1 · chemical-bond view"
        detail = "64 H₂O · 192 atoms · periodic cube"
        colour = DARK_GRAY
    elif stage == 1:
        mode = "cutoff"
        headline = "2 · draw the true cutoff sphere"
        detail = "central atom fixed at source ID 126 · rcut = 6.0 Å"
        colour = NAVY
    elif stage == 2:
        mode = "neighbors"
        headline = "3 · switch to the neighbor view"
        detail = "83 exact minimum-image neighbors · outside atoms fade"
        colour = NAVY
    else:
        mode = "forces"
        headline = "4 · the learned energy returns real forces"
        detail = "12 local force vectors shown · one fixed display scale"
        colour = GREEN
    registry.text(middle, 0.50, 0.985, headline, ha="center", va="top", fontsize=25, color=colour, weight="bold")
    draw_water_scene(
        middle,
        registry,
        data,
        rect=(0.015, 0.115, 0.985, 0.905),
        video=True,
        mode=mode,
        reveal=fade if stage in {1, 2, 3} else 1.0,
    )
    registry.text(middle, 0.50, 0.035, detail, ha="center", va="bottom", fontsize=18, color=DARK_GRAY)

    if stage == 0:
        active = 0
        active_weight = fade
    elif stage == 1:
        active = 0
        active_weight = 1.0
    elif stage == 2:
        active = 1
        active_weight = fade
    else:
        active = min(4, 1 + int(min(local, 0.999) * 4.0))
        within = (local * 4.0) % 1.0
        active_weight = smoothstep(min(within / 0.22, 1.0))
    draw_dp_pipeline(right, registry, data, video=True, active_stage=active, active_weight=active_weight)
    add_footer(
        fig,
        "model: H2O-Phase-Diagram-model_compressed.pb · inference executed in a Bohrium sandbox · no 111 view",
        video=True,
        registry=registry,
    )
    return [
        {"id": "hydrogen_cutoff", "color": NAVY, "min_pixels": 500},
        {"id": "oxygen", "color": CRIMSON, "min_pixels": 350},
        {"id": "atomic_energy", "color": GREEN, "min_pixels": 300},
    ]


def render_animation(data) -> None:
    audit_config = {
        "panels": [
            {"id": "md_loop", "rect": list(VIDEO_LEFT), "min_clearance_px": 16},
            {"id": "water_environment", "rect": list(VIDEO_MIDDLE), "min_clearance_px": 16},
            {"id": "learned_model", "rect": list(VIDEO_RIGHT), "min_clearance_px": 16},
        ],
        "whitespace": {
            "background_threshold": 245,
            "min_ink_fraction": 0.022,
            "min_panel_bbox_fill": 0.31,
            "grid_rows": 12,
            "grid_columns": 24,
        },
        "bands": [
            {"id": "left_gap", "rect": [0.278, 0.19, 0.295, 0.895], "max_ink_pixels": 0},
            {"id": "right_gap", "rect": [0.728, 0.19, 0.745, 0.895], "max_ink_pixels": 0},
        ],
    }
    output = render_video(
        stem=STEM,
        duration_seconds=VIDEO_DURATION_SECONDS,
        draw_frame=lambda fig, t, i, reg: _draw_video_frame(fig, t, i, reg, data),
        audit_config=audit_config,
        qa_directory=QA_DIR / "_qa",
        representative_times=[1.8, 5.8, 9.8, 13.0, 15.4],
    )
    print(f"video: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    render_static(data)
    if not args.static_only:
        render_animation(data)


if __name__ == "__main__":
    main()
