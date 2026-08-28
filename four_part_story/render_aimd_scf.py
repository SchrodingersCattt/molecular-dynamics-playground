"""Render the multi-step AIMD/RHF density story."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image
from scipy import ndimage

from common import (
    DARK_GRAY,
    GREEN,
    INK,
    LINE_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    axes_from_top_slot,
    json_dump,
    mix_hex,
    new_static_figure,
    new_video_figure,
    render_source_panel,
    render_video,
    save_static,
    smoothstep,
)
from mattervis_story import (
    SceneCamera,
    add_story_title,
    camera_for_source,
    draw_vv_loop,
    make_vector_group,
    place_render,
    place_render_blend,
    project_world,
    render_structure,
    write_provenance_index,
)


ROOT = Path(__file__).resolve().parent
STEM = "03_aimd_scf"
QA_DIR = ROOT / "_qa" / STEM
MATTERVIS_DIR = QA_DIR / "source" / "mattervis_multistep"
DATA_PATH = ROOT / "data" / "aimd_multistep_h2o_dimer.npz"
MOTION_SOURCE = ROOT / "data" / "aimd_multistep_h2o_dimer.extxyz"

STATIC_SCENE_RECT = (0.04, 0.10, 0.96, 0.87)
VIDEO_SCENE_RECT = (0.04, 0.10, 0.96, 0.87)

STATIC_A = (0.035, 0.20, 0.310, 0.91)
STATIC_B = (0.325, 0.18, 0.715, 0.91)
STATIC_C = (0.745, 0.18, 0.965, 0.49)
STATIC_D = (0.755, 0.52, 0.955, 0.91)

VIDEO_A = (0.035, 0.20, 0.310, 0.91)
VIDEO_B = (0.325, 0.18, 0.715, 0.91)
VIDEO_C = (0.745, 0.18, 0.965, 0.49)
VIDEO_D = (0.755, 0.52, 0.955, 0.91)

CAMERA_SCALE = 1.72
POSITION_LAKE = "#4E9BB5"
FORCE_OLIVE = "#A99C50"
VELOCITY_EMERALD = "#2F8562"
FORCE_DISPLAY_SCALE = 42.0
VELOCITY_DISPLAY_SCALE = 34.0
DISPLACEMENT_ARROW_SCALE = 75.0

DENSITY_LEVELS = np.asarray(
    [0.003, 0.007, 0.015, 0.035, 0.080, 0.180, 0.400, 0.900]
)
DENSITY_COLORS = (
    "#DDE5EA",
    "#CEDAE1",
    "#BCCDD7",
    "#A8BDCA",
    "#91AABB",
    "#7894A8",
    "#607E94",
    "#49687F",
)

VIDEO_DURATION = 15.0
DETAILED_BLOCK_SECONDS = 5.0
RAPID_BLOCK_SECONDS = 1.25

POSITION_EQUATION = (
    r"$\mathbf{r}_{n+1}=\mathbf{r}_n$"
    "\n"
    r"$+\mathbf{v}_{n+1/2}\Delta t$"
)
ACCELERATION_EQUATION = (
    r"$\mathbf{a}_{n}=\mathbf{F}_{n}/m$"
    "\n"
    r"$\mathbf{F}_{n}=-\nabla_R E(\mathbf{r}_n)$"
)
VELOCITY_EQUATION = (
    r"$\mathbf{v}_{n+1/2}=\mathbf{v}_{n}$"
    "\n"
    r"$+\frac{1}{2}\mathbf{a}_{n}\Delta t$"
)


def scf_visual_progress(residuals: np.ndarray, iteration: int) -> float:
    """Map the real residual decrease to a bounded display-resolution progress."""
    finite = np.asarray(residuals, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size <= 1:
        return 1.0
    index = min(max(int(iteration), 0), finite.size - 1)
    start = float(np.log10(finite[0]))
    finish = float(np.log10(finite[-1]))
    current = float(np.log10(finite[index]))
    if abs(start - finish) < 1.0e-12:
        return index / max(finite.size - 1, 1)
    return float(np.clip((start - current) / (start - finish), 0.0, 1.0))


def render_density_plane_array_scene(
    density: np.ndarray,
    structure_image: Path,
    output: Path,
    *,
    plane_centre: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    residuals: np.ndarray,
    camera: SceneCamera,
    ion_index: int,
    scf_index: int,
) -> dict:
    """Render one real SCF density plane beneath an aligned MatterVis structure."""
    progress = scf_visual_progress(residuals, scf_index)
    # Keep the contour topology stable between cached SCF states. The real
    # density and residual-driven blur still evolve continuously, while a
    # fixed level set prevents contour bands from popping into existence.
    level_indices = np.arange(len(DENSITY_LEVELS), dtype=int)
    levels = DENSITY_LEVELS[level_indices]
    colors = [DENSITY_COLORS[index] for index in level_indices]
    blur_sigma = 5.5 * (1.0 - progress) ** 1.35
    displayed_density = ndimage.gaussian_filter(
        np.asarray(density, dtype=float),
        sigma=blur_sigma,
        mode="nearest",
    )

    with Image.open(structure_image) as structure_source:
        width, height = structure_source.size
    plane_x, plane_y = np.meshgrid(
        np.asarray(u_axis, dtype=float),
        np.asarray(v_axis, dtype=float),
    )
    points = (
        plane_centre
        + plane_x[:, :, None] * plane_u
        + plane_y[:, :, None] * plane_v
    )
    projected = project_world(
        points.reshape(-1, 3),
        camera=camera,
        rect=(0.0, 0.0, float(width), float(height)),
        image_aspect=width / height,
    ).reshape(points.shape[:2] + (2,))
    grid_x = projected[:, :, 0]
    grid_y = height - projected[:, :, 1]

    signature = {
        "pipeline_version": 5,
        "density_source": str(DATA_PATH),
        "structure_source": str(structure_image),
        "ion_index": int(ion_index),
        "scf_index": int(scf_index),
        "residual": float(np.asarray(residuals)[scf_index]),
        "visual_progress": progress,
        "blur_sigma_pixels": blur_sigma,
        "levels": levels.tolist(),
        "camera": {
            "target": list(camera.target),
            "direction": list(camera.direction),
            "up": list(camera.up),
            "ortho_scale": camera.ortho_scale,
        },
    }
    sidecar = output.with_suffix(".json")
    if output.exists() and sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if previous == {**signature, "output": str(output)}:
            with Image.open(output) as cached:
                if cached.size == (width, height):
                    return previous

    figure = plt.figure(
        figsize=(width / 100.0, height / 100.0),
        dpi=100,
        facecolor=WHITE,
    )
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.set_xlim(0.0, width - 1.0)
    axes.set_ylim(height - 1.0, 0.0)
    axes.contour(
        grid_x,
        grid_y,
        displayed_density,
        levels=levels,
        colors=colors,
        linewidths=np.linspace(
            3.0 - 0.9 * progress,
            1.8 - 0.5 * progress,
            len(levels),
        ),
        antialiased=True,
    )
    axes.axis("off")
    figure.canvas.draw()
    base = Image.fromarray(
        np.asarray(figure.canvas.buffer_rgba()).copy(),
        mode="RGBA",
    )
    plt.close(figure)
    structure = Image.open(structure_image).convert("RGBA")
    base.alpha_composite(structure)
    output.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(output)
    payload = {**signature, "output": str(output)}
    json_dump(sidecar, payload)
    return payload


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def prepare_mattervis(
    data: dict[str, np.ndarray],
) -> tuple[dict[str, object], SceneCamera]:
    """Render all structures and atom-centred vectors with one fixed camera."""
    positions = np.asarray(data["positions"], dtype=float)
    plane_centre = np.asarray(data["plane_centre_angstrom"], dtype=float)
    plane_v = np.asarray(data["plane_v"], dtype=float)
    plane_normal = np.asarray(data["plane_normal"], dtype=float)
    scf_counts = np.asarray(data["scf_counts"], dtype=int)
    residuals = np.asarray(data["scf_residuals"], dtype=float)
    density_planes = np.asarray(data["density_planes"], dtype=float)

    view_direction = plane_normal - 0.72 * data["plane_u"] - 0.55 * plane_v
    view_direction /= np.linalg.norm(view_direction)
    camera = camera_for_source(
        MOTION_SOURCE,
        target=plane_centre,
        ortho_scale=CAMERA_SCALE,
        frame=0,
        direction=tuple(view_direction),
        up=tuple(plane_v),
    )
    records: list[dict] = []
    structure_paths: list[Path] = []
    for ion_index in range(len(positions)):
        path = MATTERVIS_DIR / f"ion_{ion_index:02d}_structure.png"
        records.append(
            render_structure(
                MOTION_SOURCE,
                path,
                camera=camera,
                frame=ion_index,
                width=1500,
                height=950,
                atom_scale=0.90,
                bond_radius=0.102,
            )
        )
        structure_paths.append(path)

    density_paths: list[list[Path]] = []
    for ion_index, count in enumerate(scf_counts):
        ion_paths: list[Path] = []
        for scf_index in range(int(count)):
            output = MATTERVIS_DIR / (
                f"ion_{ion_index:02d}_scf_{scf_index:02d}_density.png"
            )
            records.append(
                render_density_plane_array_scene(
                    density_planes[ion_index, scf_index],
                    structure_paths[ion_index],
                    output,
                    plane_centre=plane_centre,
                    plane_u=data["plane_u"],
                    plane_v=plane_v,
                    u_axis=data["plane_u_axis_angstrom"],
                    v_axis=data["plane_v_axis_angstrom"],
                    residuals=residuals[ion_index, : int(count)],
                    camera=camera,
                    ion_index=ion_index,
                    scf_index=scf_index,
                )
            )
            ion_paths.append(output)
        density_paths.append(ion_paths)

    arrow_style = {
        "shaft_radius": 0.052,
        "head_length_ratio": 0.28,
        "head_radius_ratio": 2.25,
        "sides": 18,
    }
    force_paths: list[Path] = []
    velocity_paths: list[Path] = []
    movement_paths: list[Path] = []
    update_count = len(positions) - 1
    for ion_index in range(update_count):
        force_vectors = make_vector_group(
            f"rhf-force-ion-{ion_index:02d}",
            positions[ion_index],
            data["forces_eh_per_bohr"][ion_index],
            scale=FORCE_DISPLAY_SCALE,
            color=FORCE_OLIVE,
            tail_offset=0.0,
            style=arrow_style,
        )
        force_path = MATTERVIS_DIR / f"ion_{ion_index:02d}_force.png"
        records.append(
            render_structure(
                MOTION_SOURCE,
                force_path,
                camera=camera,
                frame=ion_index,
                width=1500,
                height=950,
                atom_scale=0.90,
                bond_radius=0.102,
                vector_overlays=force_vectors,
            )
        )
        force_paths.append(force_path)

        velocity_vectors = make_vector_group(
            f"half-step-velocity-ion-{ion_index:02d}",
            positions[ion_index],
            data["half_velocities"][ion_index],
            scale=VELOCITY_DISPLAY_SCALE,
            color=VELOCITY_EMERALD,
            tail_offset=0.0,
            style=arrow_style,
        )
        velocity_path = MATTERVIS_DIR / f"ion_{ion_index:02d}_velocity.png"
        records.append(
            render_structure(
                MOTION_SOURCE,
                velocity_path,
                camera=camera,
                frame=ion_index,
                width=1500,
                height=950,
                atom_scale=0.90,
                bond_radius=0.102,
                vector_overlays=velocity_vectors,
            )
        )
        velocity_paths.append(velocity_path)

        displacement_vectors = make_vector_group(
            f"position-drift-ion-{ion_index:02d}",
            positions[ion_index],
            positions[ion_index + 1] - positions[ion_index],
            scale=DISPLACEMENT_ARROW_SCALE,
            color=POSITION_LAKE,
            tail_offset=0.0,
            style=arrow_style,
        )
        movement_path = MATTERVIS_DIR / f"ion_{ion_index:02d}_move.png"
        records.append(
            render_structure(
                MOTION_SOURCE,
                movement_path,
                camera=camera,
                frame=ion_index + 1,
                width=1500,
                height=950,
                atom_scale=0.90,
                bond_radius=0.102,
                vector_overlays=displacement_vectors,
            )
        )
        movement_paths.append(movement_path)

    write_provenance_index(MATTERVIS_DIR, records)
    return {
        "density": density_paths,
        "structure": structure_paths,
        "force": force_paths,
        "velocity": velocity_paths,
        "movement": movement_paths,
    }, camera


def _axes_aspect(ax: plt.Axes) -> float:
    figure_width, figure_height = ax.figure.canvas.get_width_height()
    position = ax.get_position()
    return (position.width * figure_width) / (position.height * figure_height)


def draw_scf_loop(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    video: bool,
    stage_weights: tuple[float, float, float, float],
    iteration: int,
    iteration_count: int,
    converged: bool,
) -> None:
    """Draw the electronic loop in its own lower-right panel."""
    aspect = _axes_aspect(ax)
    symbols = [r"$F$", r"$C$", r"$\rho$", r"$?$"]
    positions = [
        (0.50, 0.78),
        (0.78, 0.55),
        (0.50, 0.32),
        (0.22, 0.55),
    ]
    arrows = [
        ((0.565, 0.725), (0.715, 0.605)),
        ((0.715, 0.495), (0.565, 0.375)),
        ((0.435, 0.375), (0.285, 0.495)),
        ((0.285, 0.605), (0.435, 0.725)),
    ]
    radius_x = 0.072 if video else 0.064
    centre_x = 0.50
    for index, (start, end) in enumerate(arrows):
        arrow_weight = max(stage_weights[index], stage_weights[(index + 1) % 4])
        registry.arrow(
            ax,
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=19 if video else 13,
            lw=2.6 if video else 1.6,
            color=mix_hex(LINE_GRAY, INK, arrow_weight),
            zorder=3,
        )
    radius_y = radius_x * aspect
    for index, ((x, y), symbol) in enumerate(zip(positions, symbols)):
        weight = stage_weights[index]
        fill = mix_hex(WHITE, INK, weight)
        ax.add_patch(
            Ellipse(
                (x, y),
                2.0 * radius_x,
                2.0 * radius_y,
                fc=fill,
                ec=mix_hex(LINE_GRAY, INK, weight),
                lw=2.4 if video else 1.5,
                zorder=4,
            )
        )
        registry.text(
            ax,
            x,
            y,
            symbol,
            ha="center",
            va="center",
            fontsize=20 if video else 10,
            color=WHITE if weight > 0.48 else DARK_GRAY,
            weight="bold",
            zorder=5,
        )
    registry.text(
        ax,
        centre_x,
        0.55,
        "SCF",
        ha="center",
        va="center",
        fontsize=20 if video else 14,
        color=INK,
        weight="bold",
    )
    status = "converged" if converged else f"SCF {iteration:02d} / {iteration_count:02d}"
    registry.text(
        ax,
        centre_x,
        0.10,
        status,
        ha="center",
        va="center",
        fontsize=18 if video else 10,
        color=GREEN if converged else DARK_GRAY,
        weight="bold" if converged else "normal",
    )


def draw_left(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    video: bool,
    stage: int,
) -> None:
    equations = [POSITION_EQUATION, ACCELERATION_EQUATION, VELOCITY_EQUATION]
    equation = equations[stage]
    draw_vv_loop(
        ax,
        registry,
        video=video,
        active_stage=stage,
        centre_text=equation,
        centre_y=0.54,
        radius_x=0.40,
    )


def _nested_paths(assets: dict[str, object], key: str) -> list:
    paths = assets[key]
    if not isinstance(paths, list):
        raise TypeError(f"Expected a list for asset group {key}")
    return paths


def draw_case(
    ax: plt.Axes,
    registry: LayoutRegistry,
    assets: dict[str, object],
    *,
    video: bool,
    mode: str,
    ion_index: int,
    iteration_index: int,
    density_blend: float,
    scf_stage_weights: tuple[float, float, float, float],
    phase_progress: float,
) -> None:
    all_density_paths = _nested_paths(assets, "density")
    structure_paths = _nested_paths(assets, "structure")
    force_paths = _nested_paths(assets, "force")
    velocity_paths = _nested_paths(assets, "velocity")
    movement_paths = _nested_paths(assets, "movement")
    density_paths = all_density_paths[ion_index]
    if not isinstance(density_paths, list):
        raise TypeError("Density assets must be nested by ionic step")
    scene_rect = VIDEO_SCENE_RECT if video else STATIC_SCENE_RECT
    ax.add_patch(
        Rectangle(
            (0.04, 0.04),
            0.92,
            0.92,
            fill=False,
            ec=LINE_GRAY,
            lw=2.0 if video else 1.1,
            zorder=20,
        )
    )

    if mode in {"scf", "pause"}:
        if mode == "pause":
            current = len(density_paths) - 1
            following = current
            blend = 0.0
        else:
            current = min(iteration_index, len(density_paths) - 1)
            following = min(current + 1, len(density_paths) - 1)
            blend = density_blend
        place_render_blend(
            ax,
            density_paths[current],
            density_paths[following],
            scene_rect,
            blend=blend,
            zorder=4,
        )
        if mode == "pause":
            stage_text = f"Iteration {iteration_index + 1:02d} · SCF converged"
        else:
            actions = (
                "build Fock matrix",
                "solve orbitals",
                "update density",
                "check convergence",
            )
            action = actions[int(np.argmax(scf_stage_weights))]
            stage_text = f"Iteration {iteration_index + 1:02d} · {action}"
    elif mode == "force":
        place_render(ax, force_paths[ion_index], scene_rect, zorder=5)
        stage_text = f"Ionic step {ion_index + 1:02d} · evaluate nuclear forces"
    elif mode == "velocity":
        place_render(ax, velocity_paths[ion_index], scene_rect, zorder=5)
        stage_text = f"Ionic step {ion_index + 1:02d} · update velocity"
    elif mode == "move":
        fade = smoothstep(np.clip(phase_progress / 0.35, 0.0, 1.0))
        old_fade = 1.0 - smoothstep(
            np.clip((phase_progress - 0.42) / 0.42, 0.0, 1.0)
        )
        place_render(
            ax,
            movement_paths[ion_index],
            scene_rect,
            alpha=0.72 + 0.28 * fade,
            zorder=5,
        )
        place_render(
            ax,
            structure_paths[ion_index],
            scene_rect,
            alpha=0.44 * old_fade,
            zorder=6,
        )
        stage_text = (
            f"Ionic step {ion_index + 1:02d} → {ion_index + 2:02d} · "
            "update position"
        )
    else:
        raise ValueError(f"Unknown AIMD mode: {mode}")

    registry.text(
        ax,
        0.50,
        0.925,
        stage_text,
        ha="center",
        va="center",
        fontsize=22 if video else 11,
        color=INK,
        weight="normal",
        zorder=21,
    )


def draw_energy_curve(
    ax: plt.Axes,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    *,
    video: bool,
    mode: str,
    ion_index: int,
    iteration_index: int,
    density_blend: float,
) -> None:
    """Draw the real SCF energy convergence for the current ionic step."""
    ax.add_patch(
        Rectangle(
            (0.04, 0.04),
            0.92,
            0.92,
            fill=False,
            ec=LINE_GRAY,
            lw=2.0 if video else 1.1,
            zorder=2,
        )
    )
    registry.text(
        ax,
        0.50,
        0.90,
        f"Ionic step {ion_index + 1:02d} · SCF energy",
        ha="center",
        va="center",
        fontsize=18 if video else 10,
        color=INK,
        weight="normal",
        zorder=4,
    )

    count = int(data["scf_counts"][ion_index])
    energies = np.asarray(data["scf_energies_eh"][ion_index, :count], dtype=float)
    error = np.abs(energies - energies[-1])
    display_error = np.maximum(error, 1.0e-10)
    iterations = np.arange(1, count + 1, dtype=float)

    if mode == "scf":
        progress_index = min(
            iteration_index + float(density_blend),
            count - 1.0,
        )
    else:
        progress_index = count - 1.0
    current = int(np.floor(progress_index))
    fraction = progress_index - current
    visible_x = list(iterations[: current + 1])
    visible_y = list(display_error[: current + 1])
    if current < count - 1 and fraction > 1.0e-6:
        visible_x.append(current + 1.0 + fraction)
        log_y = (1.0 - fraction) * np.log(display_error[current])
        log_y += fraction * np.log(display_error[current + 1])
        visible_y.append(float(np.exp(log_y)))

    plot_ax = ax.inset_axes((0.30, 0.22, 0.61, 0.55))
    plot_ax.set_yscale("log")
    plot_ax.plot(
        iterations,
        display_error,
        color="#D5D8DC",
        lw=2.0 if video else 1.1,
        marker="o",
        markersize=3.5 if video else 2.0,
        zorder=1,
    )
    plot_ax.plot(
        visible_x,
        visible_y,
        color=NAVY,
        lw=2.8 if video else 1.5,
        zorder=2,
    )
    marker_color = GREEN if mode == "pause" else NAVY
    plot_ax.scatter(
        [visible_x[-1]],
        [visible_y[-1]],
        s=55 if video else 18,
        color=marker_color,
        edgecolors=WHITE,
        linewidths=1.0 if video else 0.5,
        zorder=3,
    )
    plot_ax.set_xlim(0.6, count + 0.4)
    plot_ax.set_ylim(5.0e-11, 2.0)
    plot_ax.set_xticks(sorted(set((1, 4, 8, count))))
    plot_ax.set_yticks((1.0, 1.0e-3, 1.0e-6, 1.0e-9))
    font_size = 18 if video else 10
    plot_ax.tick_params(axis="both", labelsize=font_size, colors=DARK_GRAY, width=1.0)
    plot_ax.set_xlabel("SCF iteration", fontsize=font_size, color=INK, labelpad=3)
    plot_ax.set_ylabel(r"$|E^k-E^*|$ / Eh", fontsize=font_size, color=INK, labelpad=3)
    plot_ax.grid(axis="y", color="#E6E8EA", lw=0.8, zorder=0)
    for spine in plot_ax.spines.values():
        spine.set_color(LINE_GRAY)
        spine.set_linewidth(1.2 if video else 0.8)


def render_static(
    data: dict[str, np.ndarray],
    assets: dict[str, object],
) -> None:
    first_count = int(data["scf_counts"][0])
    render_source_panel(
        QA_DIR / "source" / "integrator.png",
        lambda ax, registry: draw_left(ax, registry, video=False, stage=1),
        width_px=900,
        height_px=1400,
    )
    render_source_panel(
        QA_DIR / "source" / "case.png",
        lambda ax, registry: draw_case(
            ax,
            registry,
            assets,
            video=False,
            mode="pause",
            ion_index=0,
            iteration_index=first_count - 1,
            density_blend=0.0,
            scf_stage_weights=(0.0, 0.0, 0.0, 0.0),
            phase_progress=1.0,
        ),
        width_px=1800,
        height_px=1400,
    )
    render_source_panel(
        QA_DIR / "source" / "energy.png",
        lambda ax, registry: draw_energy_curve(
            ax,
            registry,
            data,
            video=False,
            mode="pause",
            ion_index=0,
            iteration_index=first_count - 1,
            density_blend=0.0,
        ),
        width_px=1100,
        height_px=800,
    )
    render_source_panel(
        QA_DIR / "source" / "scf_loop.png",
        lambda ax, registry: draw_scf_loop(
            ax,
            registry,
            video=False,
            stage_weights=(0.0, 0.0, 0.0, 0.0),
            iteration=first_count,
            iteration_count=first_count,
            converged=True,
        ),
        width_px=900,
        height_px=1000,
    )
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    add_story_title(
        fig,
        registry,
        "Ab initio MD: every ionic step contains an electronic solve",
        "Real RHF/STO-3G water-dimer density converges before the nuclei advance",
        video=False,
    )
    panel_a = axes_from_top_slot(fig, STATIC_A)
    panel_b = axes_from_top_slot(fig, STATIC_B)
    panel_c = axes_from_top_slot(fig, STATIC_C)
    panel_d = axes_from_top_slot(fig, STATIC_D)
    draw_left(panel_a, registry, video=False, stage=1)
    draw_case(
        panel_b,
        registry,
        assets,
        video=False,
        mode="pause",
        ion_index=0,
        iteration_index=first_count - 1,
        density_blend=0.0,
        scf_stage_weights=(0.0, 0.0, 0.0, 0.0),
        phase_progress=1.0,
    )
    draw_energy_curve(
        panel_c,
        registry,
        data,
        video=False,
        mode="pause",
        ion_index=0,
        iteration_index=first_count - 1,
        density_blend=0.0,
    )
    draw_scf_loop(
        panel_d,
        registry,
        video=False,
        stage_weights=(0.0, 0.0, 0.0, 0.0),
        iteration=first_count,
        iteration_count=first_count,
        converged=True,
    )
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)


def _scf_state(
    *,
    ion_index: int,
    progress: float,
    iteration_count: int,
    rapid: bool,
) -> dict:
    progress = float(np.clip(progress, 0.0, 1.0))
    iteration_float = min(
        progress * (iteration_count - 1),
        iteration_count - 1.0e-6,
    )
    iteration = int(iteration_float)
    within = iteration_float - iteration
    loop_count = 1.0 if rapid else 2.0
    stage_float = progress * loop_count * 4.0
    active_stage = int(stage_float) % 4
    following_stage = (active_stage + 1) % 4
    stage_within = stage_float - int(stage_float)
    stage_blend = smoothstep(stage_within)
    stage_weights = [0.0, 0.0, 0.0, 0.0]
    stage_weights[active_stage] = 1.0 - stage_blend
    stage_weights[following_stage] = stage_blend
    return {
        "mode": "scf",
        "ion": ion_index,
        "iteration": iteration,
        "blend": smoothstep(within),
        "stage_weights": tuple(float(weight) for weight in stage_weights),
        "progress": progress,
        "scf_progress": progress,
        "rapid": rapid,
    }


def _phase_state(
    mode: str,
    *,
    ion_index: int,
    progress: float,
    iteration_count: int,
    rapid: bool,
) -> dict:
    return {
        "mode": mode,
        "ion": ion_index,
        "iteration": iteration_count - 1,
        "blend": 0.0,
        "stage_weights": (0.0, 0.0, 0.0, 0.0),
        "progress": float(np.clip(progress, 0.0, 1.0)),
        "scf_progress": 1.0,
        "rapid": rapid,
    }


def video_state(time_seconds: float, scf_counts: np.ndarray) -> dict:
    """Map 15 s to two detailed and four rapid, real ionic updates."""
    bounded = float(np.clip(time_seconds, 0.0, VIDEO_DURATION - 1.0e-9))
    if bounded < 2.0 * DETAILED_BLOCK_SECONDS:
        ion_index = int(bounded // DETAILED_BLOCK_SECONDS)
        local = bounded - ion_index * DETAILED_BLOCK_SECONDS
        count = int(scf_counts[ion_index])
        if local < 2.92:
            return _scf_state(
                ion_index=ion_index,
                progress=local / 2.92,
                iteration_count=count,
                rapid=False,
            )
        if local < 3.58:
            return _phase_state(
                "pause",
                ion_index=ion_index,
                progress=(local - 2.92) / 0.66,
                iteration_count=count,
                rapid=False,
            )
        if local < 4.04:
            return _phase_state(
                "force",
                ion_index=ion_index,
                progress=(local - 3.58) / 0.46,
                iteration_count=count,
                rapid=False,
            )
        if local < 4.48:
            return _phase_state(
                "velocity",
                ion_index=ion_index,
                progress=(local - 4.04) / 0.44,
                iteration_count=count,
                rapid=False,
            )
        return _phase_state(
            "move",
            ion_index=ion_index,
            progress=(local - 4.48) / 0.52,
            iteration_count=count,
            rapid=False,
        )

    rapid_time = bounded - 2.0 * DETAILED_BLOCK_SECONDS
    rapid_index = min(int(rapid_time // RAPID_BLOCK_SECONDS), 3)
    ion_index = 2 + rapid_index
    local = rapid_time - rapid_index * RAPID_BLOCK_SECONDS
    count = int(scf_counts[ion_index])
    if local < 0.50:
        return _scf_state(
            ion_index=ion_index,
            progress=local / 0.50,
            iteration_count=count,
            rapid=True,
        )
    if local < 0.70:
        return _phase_state(
            "pause",
            ion_index=ion_index,
            progress=(local - 0.50) / 0.20,
            iteration_count=count,
            rapid=True,
        )
    if local < 0.88:
        return _phase_state(
            "force",
            ion_index=ion_index,
            progress=(local - 0.70) / 0.18,
            iteration_count=count,
            rapid=True,
        )
    if local < 1.04:
        return _phase_state(
            "velocity",
            ion_index=ion_index,
            progress=(local - 0.88) / 0.16,
            iteration_count=count,
            rapid=True,
        )
    return _phase_state(
        "move",
        ion_index=ion_index,
        progress=(local - 1.04) / 0.21,
        iteration_count=count,
        rapid=True,
    )


def draw_video_frame(
    fig: plt.Figure,
    time_seconds: float,
    frame_index: int,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    assets: dict[str, object],
) -> list[dict]:
    del frame_index
    state = video_state(time_seconds, data["scf_counts"])
    add_story_title(
        fig,
        registry,
        "AIMD: electrons converge before nuclei move",
        (
            "real RHF/STO-3G H₂O dimer · two complete SCF cycles, "
            "then four rapid ionic steps"
        ),
        video=True,
    )
    panel_a = axes_from_top_slot(fig, VIDEO_A)
    panel_b = axes_from_top_slot(fig, VIDEO_B)
    panel_c = axes_from_top_slot(fig, VIDEO_C)
    panel_d = axes_from_top_slot(fig, VIDEO_D)
    stage_for_mode = {
        "scf": 1,
        "pause": 1,
        "force": 1,
        "velocity": 2,
        "move": 0,
    }
    draw_left(panel_a, registry, video=True, stage=stage_for_mode[state["mode"]])
    draw_case(
        panel_b,
        registry,
        assets,
        video=True,
        mode=state["mode"],
        ion_index=state["ion"],
        iteration_index=state["iteration"],
        density_blend=state["blend"],
        scf_stage_weights=state["stage_weights"],
        phase_progress=state["progress"],
    )
    draw_energy_curve(
        panel_c,
        registry,
        data,
        video=True,
        mode=state["mode"],
        ion_index=state["ion"],
        iteration_index=state["iteration"],
        density_blend=state["blend"],
    )
    count = int(data["scf_counts"][state["ion"]])
    draw_scf_loop(
        panel_d,
        registry,
        video=True,
        stage_weights=state["stage_weights"],
        iteration=state["iteration"] + 1,
        iteration_count=count,
        converged=state["mode"] != "scf",
    )
    if state["mode"] in {"scf", "pause"}:
        return [
            {
                "id": "density_contours",
                "color": DENSITY_COLORS[-1],
                "min_pixels": 35,
            }
        ]
    if state["mode"] == "force":
        return [
            {
                "id": "nuclear_force",
                "color": FORCE_OLIVE,
                "min_pixels": 120,
            }
        ]
    if state["mode"] == "velocity":
        return [
            {
                "id": "half_step_velocity",
                "color": VELOCITY_EMERALD,
                "min_pixels": 120,
            }
        ]
    return [
        {
            "id": "nuclear_displacement",
            "color": POSITION_LAKE,
            "min_pixels": 120,
        }
    ]


KEYFRAME_TIMES = [
    0.10,
    1.45,
    3.12,
    3.80,
    4.25,
    4.75,
    5.10,
    6.45,
    8.12,
    8.80,
    9.25,
    9.75,
    10.12,
    11.18,
    12.42,
    14.82,
]


def render_representative_frames(
    data: dict[str, np.ndarray],
    assets: dict[str, object],
) -> Path:
    """Render phase-boundary keyframes before the expensive full animation."""
    output_dir = QA_DIR / "_qa" / "multistep_keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []
    records: list[dict] = []
    for index, time_seconds in enumerate(KEYFRAME_TIMES):
        fig = new_video_figure()
        registry = LayoutRegistry(min_font_pt=18, edge_pad_px=12)
        semantics = draw_video_frame(
            fig,
            time_seconds,
            int(round(time_seconds * 24)),
            registry,
            data,
            assets,
        )
        errors = registry.validate(fig)
        if errors:
            plt.close(fig)
            raise RuntimeError(
                f"Keyframe {time_seconds:.2f} s failed layout:\n"
                + "\n".join(errors)
            )
        path = output_dir / f"frame_{index:02d}_{time_seconds:05.2f}s.png"
        fig.savefig(path, dpi=100, facecolor=WHITE)
        plt.close(fig)
        images.append(Image.open(path).convert("RGB").resize((480, 270)))
        state = video_state(time_seconds, data["scf_counts"])
        records.append(
            {
                "time_seconds": time_seconds,
                "path": str(path),
                "state": state,
                "semantics": semantics,
                "layout_passed": True,
            }
        )

    columns = 4
    rows = int(np.ceil(len(images) / columns))
    contact = Image.new("RGB", (columns * 480, rows * 270), WHITE)
    for index, item in enumerate(images):
        contact.paste(item, ((index % columns) * 480, (index // columns) * 270))
    contact_path = output_dir / "_contact.png"
    contact.save(contact_path)
    json_dump(output_dir / "keyframes.json", {"frames": records})
    return contact_path


def render_animation(
    data: dict[str, np.ndarray],
    assets: dict[str, object],
) -> None:
    audit_config = {
        "panels": [
            {
                "id": "integrator",
                "rect": list(VIDEO_A),
                "min_clearance_px": 12,
            },
            {
                "id": "aimd_case",
                "rect": list(VIDEO_B),
                "min_clearance_px": 12,
            },
            {
                "id": "scf_energy",
                "rect": list(VIDEO_C),
                "min_clearance_px": 12,
            },
            {
                "id": "scf_loop",
                "rect": list(VIDEO_D),
                "min_clearance_px": 12,
            },
        ],
        "whitespace": {
            "background_threshold": 245,
            "min_ink_fraction": 0.020,
            "min_panel_bbox_fill": 0.22,
            "grid_rows": 12,
            "grid_columns": 20,
        },
        "bands": [
            {
                "id": "gap_a_b",
                "rect": [0.310, 0.20, 0.325, 0.91],
                "max_ink_pixels": 0,
            },
            {
                "id": "gap_b_right",
                "rect": [0.715, 0.18, 0.745, 0.91],
                "max_ink_pixels": 0,
            },
            {
                "id": "gap_c_d",
                "rect": [0.745, 0.49, 0.965, 0.52],
                "max_ink_pixels": 0,
            }
        ],
    }
    render_video(
        stem=STEM,
        duration_seconds=VIDEO_DURATION,
        draw_frame=lambda fig, time, index, registry: draw_video_frame(
            fig,
            time,
            index,
            registry,
            data,
            assets,
        ),
        audit_config=audit_config,
        qa_directory=QA_DIR / "_qa",
        representative_times=KEYFRAME_TIMES,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    assets, _camera = prepare_mattervis(data)
    if args.assets_only:
        return
    render_static(data, assets)
    if args.preview_only:
        render_representative_frames(data, assets)
        return
    if not args.static_only:
        render_animation(data, assets)


if __name__ == "__main__":
    main()
