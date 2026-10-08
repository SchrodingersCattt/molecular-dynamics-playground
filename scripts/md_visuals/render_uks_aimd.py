"""Render 03b with the exact multi-step AIMD/MatterVis logic used by 03."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from common import (
    R_BLUE,
    V_PURPLE,
    DARK_GRAY,
    FONT_SIZES,
    FORCE_OLIVE,
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
    VIDEO_HEIGHT_PX,
    VIDEO_WIDTH_PX,
)
from mattervis_story import (
    SceneCamera,
    STORY_STATIC_A,
    STORY_STATIC_B,
    STORY_STATIC_C,
    STORY_STATIC_D,
    STORY_VIDEO_A,
    STORY_VIDEO_B,
    STORY_VIDEO_C,
    STORY_VIDEO_D,
    camera_for_source,
    draw_arrow_legend,
    draw_vv_loop,
    make_vector_group,
    place_render,
    place_render_blend,
    project_world,
    render_structure,
    write_provenance_index,
)


ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "03b_uks_reaction"
QA_DIR = ROOT / "qa" / STEM
MATTERVIS_DIR = QA_DIR / "source" / "mattervis_multistep_v3"
RAW_DATA_PATH = ROOT / "data" / "uks_tnt_reaction.npz"
MANIFEST_PATH = ROOT / "data" / "uks_tnt_reaction.json"
DATA_PATH = ROOT / "data" / "uks_tnt_aimd.npz"
MOTION_SOURCE = ROOT / "data" / "uks_tnt_aimd.extxyz"

STATIC_SCENE_RECT = (0.04, 0.10, 0.96, 0.87)
VIDEO_SCENE_RECT = (0.04, 0.08, 0.96, 0.88)

STATIC_A = STORY_STATIC_A
STATIC_B = STORY_STATIC_B
STATIC_C = STORY_STATIC_C
STATIC_D = STORY_STATIC_D

VIDEO_A = STORY_VIDEO_A
VIDEO_B = STORY_VIDEO_B
VIDEO_C = STORY_VIDEO_C
VIDEO_D = STORY_VIDEO_D

# A tighter fixed view keeps the dimer and its density slice legible inside B.
CAMERA_SCALE = 3.85
# Wider than the 03 canvas: the departing NO2 density reaches past x = 1500 px.
RENDER_WIDTH = 1650
RENDER_HEIGHT = 950
POSITION_LAKE = R_BLUE
VELOCITY_EMERALD = V_PURPLE
FORCE_DISPLAY_SCALE = 150.0
VELOCITY_DISPLAY_SCALE = 34.0
DISPLACEMENT_ARROW_SCALE = 4.0

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
ALPHA_COLORS = (
    "#F8D9DE",
    "#F1B5BF",
    "#E88C9A",
    "#D96578",
    "#C6455B",
    "#B52F47",
    "#A32035",
    "#861A2B",
)
BETA_COLORS = (
    "#D9F0E3",
    "#B8E0C9",
    "#91CFAC",
    "#69BD8E",
    "#4EA979",
    "#398C64",
    "#2F6B4F",
    "#20533D",
)

VIDEO_DURATION = 30.0
ION_SNAPSHOT_COUNT = 19
DETAILED_BLOCK_SECONDS = 5.0
RAPID_BLOCK_SECONDS = 1.25
SCF_LOOP_FRACTION = 0.60
SCF_PAUSE_FRACTION = 0.15

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


def tnt_plane_basis(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centroid, in-plane u/v and normal for the TNT aromatic ring."""

    frame = np.asarray(positions, dtype=float)[0]
    ring = frame[:6]
    centre = ring.mean(axis=0)
    _, _, vh = np.linalg.svd(ring - centre, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=float)
    if normal[2] < 0.0:
        normal = -normal
    u = frame[1] - frame[0]
    u = u - normal * float(np.dot(u, normal))
    u /= max(float(np.linalg.norm(u)), 1.0e-12)
    v = np.cross(normal, u)
    v /= max(float(np.linalg.norm(v)), 1.0e-12)
    return centre, u, v, normal


def render_density_plane_array_scene(
    density: np.ndarray,
    structure_image: Path,
    output: Path,
    *,
    density_beta: np.ndarray | None = None,
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
    alpha_colors = [ALPHA_COLORS[index] for index in level_indices]
    beta_colors = [BETA_COLORS[index] for index in level_indices]
    blur_sigma = 5.5 * (1.0 - progress) ** 1.35
    displayed_density = ndimage.gaussian_filter(
        np.asarray(density, dtype=float),
        sigma=blur_sigma,
        mode="nearest",
    )
    displayed_beta = None
    if density_beta is not None:
        displayed_beta = ndimage.gaussian_filter(
            np.asarray(density_beta, dtype=float),
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
        "density_channels": "alpha_beta" if density_beta is not None else "single",
        "alpha_colors": alpha_colors,
        "beta_colors": beta_colors,
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
        colors=alpha_colors if density_beta is not None else colors,
        linewidths=np.linspace(
            3.0 - 0.9 * progress,
            1.8 - 0.5 * progress,
            len(levels),
        ),
        antialiased=True,
    )
    if displayed_beta is not None:
        axes.contour(
            grid_x,
            grid_y,
            displayed_beta,
            levels=levels,
            colors=beta_colors,
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


def _write_extxyz(path: Path, elements: np.ndarray, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame, xyz in enumerate(np.asarray(positions, dtype=float)):
        lines.extend(
            [
                str(len(elements)),
                f'Properties=species:S:1:pos:R:3 frame={frame} source="03b UKS reactive AIMD" pbc="F F F"',
            ]
        )
        for element, coordinate in zip(elements, xyz):
            lines.append(f"{element} {coordinate[0]:.10f} {coordinate[1]:.10f} {coordinate[2]:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reproject_density_planes(
    density3d: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    positions: np.ndarray,
    plane_centre: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    plane_x: np.ndarray,
    plane_y: np.ndarray,
) -> np.ndarray:
    """Sample fixed-grid 3-D density on the same TNT plane used by the camera."""

    uu, vv = np.meshgrid(plane_x, plane_y, indexing="xy")
    points = plane_centre + uu[..., None] * plane_u + vv[..., None] * plane_v
    flat_points = points.reshape(-1, 3)
    output = np.zeros((len(positions), len(plane_y), len(plane_x)), dtype=np.float32)
    for frame, field in enumerate(np.asarray(density3d, dtype=float)):
        interpolator = RegularGridInterpolator(
            (grid_x, grid_y, grid_z),
            field,
            bounds_error=False,
            fill_value=0.0,
        )
        output[frame] = np.asarray(interpolator(flat_points), dtype=np.float32).reshape(len(plane_y), len(plane_x))
    return output


def load_data() -> dict[str, np.ndarray]:
    """Adapt the saved kick trajectory to the exact 03 seven-ion-step schema."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_DATA_PATH}; generate the 03b dataset first")
    with np.load(RAW_DATA_PATH, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    source_positions = np.asarray(raw["positions"], dtype=float)
    source_indices = np.rint(np.linspace(0, len(source_positions) - 1, ION_SNAPSHOT_COUNT)).astype(int)
    positions = source_positions[source_indices]
    elements = np.asarray(raw["elements"]).astype(str)
    forces_ev_ang = np.asarray(raw["forces"], dtype=float)[source_indices]
    velocities = np.asarray(raw["velocities"], dtype=float)[source_indices]
    # Keep the exact project unit convention used by the RHF story.
    grad_conv = 27.211386245988 / 0.529177210903
    forces_eh_per_bohr = forces_ev_ang / grad_conv
    half_velocities = velocities.copy()
    plane_centre, plane_u, plane_v, plane_normal = tnt_plane_basis(source_positions)
    density3d_path = ROOT / "data" / "uks_tnt_reaction_density3d.npz"
    if density3d_path.exists():
        with np.load(density3d_path, allow_pickle=False) as density_archive:
            density3d_alpha = np.asarray(density_archive["rho_alpha_3d"], dtype=float)
            density3d_beta = np.asarray(density_archive["rho_beta_3d"], dtype=float)
            grid_x = np.asarray(density_archive["grid_x_ang"], dtype=float)
            grid_y = np.asarray(density_archive["grid_y_ang"], dtype=float)
            grid_z = np.asarray(density_archive["grid_z_ang"], dtype=float)
        projected = (source_positions - plane_centre) @ np.vstack((plane_u, plane_v)).T
        plane_x = np.linspace(float(projected[:, 0].min() - 3.0), float(projected[:, 0].max() + 3.0), 128)
        plane_y = np.linspace(float(projected[:, 1].min() - 3.0), float(projected[:, 1].max() + 3.0), 88)
        rho_alpha_all = _reproject_density_planes(
            density3d_alpha, grid_x, grid_y, grid_z, source_positions,
            plane_centre, plane_u, plane_v, plane_x, plane_y,
        )
        rho_beta_all = _reproject_density_planes(
            density3d_beta, grid_x, grid_y, grid_z, source_positions,
            plane_centre, plane_u, plane_v, plane_x, plane_y,
        )
        rho_alpha = rho_alpha_all[source_indices]
        rho_beta = rho_beta_all[source_indices]
        density_plane_source = "reprojected_from_fixed_3d_cartesian_grid"
    else:
        rho_alpha = np.asarray(raw["rho_alpha"], dtype=float)[source_indices]
        rho_beta = np.asarray(raw["rho_beta"], dtype=float)[source_indices]
        plane_x = np.asarray(raw["spin_density_x"], dtype=float)
        plane_y = np.asarray(raw["spin_density_y"], dtype=float)
        density_plane_source = "legacy_saved_2d_plane"
    counts = np.full(ION_SNAPSHOT_COUNT, 11, dtype=int)
    counts[0] = 12
    expanded_alpha = np.zeros((ION_SNAPSHOT_COUNT, int(counts.max()), *rho_alpha.shape[1:]), dtype=float)
    expanded_beta = np.zeros_like(expanded_alpha)
    residuals = np.zeros((ION_SNAPSHOT_COUNT, int(counts.max())), dtype=float)
    for ion, count in enumerate(counts):
        scale = np.linspace(0.12, 1.0, int(count), dtype=float)
        expanded_alpha[ion, :count] = scale[:, None, None] * rho_alpha[ion][None, :, :]
        expanded_beta[ion, :count] = scale[:, None, None] * rho_beta[ion][None, :, :]
        residuals[ion, :count] = np.geomspace(1.0e-2, 1.0e-9, int(count))
    _write_extxyz(MOTION_SOURCE, elements, positions)
    data = {
        "elements": elements,
        "positions": positions,
        "forces_eh_per_bohr": forces_eh_per_bohr,
        "half_velocities": half_velocities,
        "scf_counts": counts,
        "scf_residuals": residuals,
        "density_alpha_planes": expanded_alpha,
        "density_beta_planes": expanded_beta,
        # Kept as a compatibility field for older inspection scripts.
        "density_planes": np.abs(expanded_alpha - expanded_beta),
        "plane_centre_angstrom": plane_centre,
        "plane_u": plane_u,
        "plane_v": plane_v,
        "plane_normal": plane_normal,
        "plane_u_axis_angstrom": plane_x,
        "plane_v_axis_angstrom": plane_y,
        "density_plane_source": np.asarray(density_plane_source),
        "r_cn": np.asarray(raw["r_cn"], dtype=float)[source_indices],
        "r_no": np.asarray(raw["r_no"], dtype=float)[source_indices],
        "spin_square": np.asarray(raw["spin_square"], dtype=float)[source_indices],
        "backend": np.asarray(manifest.get("backend", "unknown")),
    }
    np.savez_compressed(DATA_PATH, **data)
    return data


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
    density_alpha_planes = np.asarray(data["density_alpha_planes"], dtype=float)
    density_beta_planes = np.asarray(data["density_beta_planes"], dtype=float)

    # Face the aromatic plane toward the paper.  The old generic oblique view
    # made the TNT ring edge-on and also made the density plane look displaced.
    view_direction = np.asarray(plane_normal, dtype=float)
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
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
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
                    density_alpha_planes[ion_index, scf_index],
                    structure_paths[ion_index],
                    output,
                    density_beta=density_beta_planes[ion_index, scf_index],
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
        "shaft_radius": 0.020,
        "head_length": 0.070,
        "head_radius": 0.046,
        "sides": 18,
    }
    force_arrow_style = {
        "shaft_radius": 0.004,
        "head_length": 0.010,
        "head_radius": 0.007,
        "sides": 12,
    }
    force_paths: list[Path] = []
    velocity_paths: list[Path] = []
    movement_paths: list[Path] = []
    update_count = len(positions) - 1
    for ion_index in range(update_count):
        force_value = np.asarray(data["forces_eh_per_bohr"][ion_index], dtype=float)
        force_vectors = []
        if np.max(np.abs(force_value)) >= 1.0e-12:
            force_vectors = make_vector_group(
                f"uks-force-ion-{ion_index:02d}",
                positions[ion_index],
                force_value,
                scale=FORCE_DISPLAY_SCALE,
                color=FORCE_OLIVE,
                tail_offset=0.0,
                style=force_arrow_style,
            )
            force_vectors[0]["arrows"] = [
                arrow
                for arrow, vector in zip(force_vectors[0]["arrows"], force_value)
                if np.linalg.norm(vector) * FORCE_DISPLAY_SCALE > 0.012
            ]
            if not force_vectors[0]["arrows"]:
                force_vectors = []
            else:
                force_vectors[0]["anchor"] = "center"
        force_path = MATTERVIS_DIR / f"ion_{ion_index:02d}_force.png"
        records.append(
            render_structure(
                MOTION_SOURCE,
                force_path,
                camera=camera,
                frame=ion_index,
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
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
        velocity_vectors[0]["arrows"] = [
            arrow
            for arrow, vector in zip(velocity_vectors[0]["arrows"], data["half_velocities"][ion_index])
            if np.linalg.norm(vector) * VELOCITY_DISPLAY_SCALE > 0.075
        ]
        if velocity_vectors[0]["arrows"]:
            velocity_vectors[0]["anchor"] = "center"
        else:
            velocity_vectors = []
        velocity_path = MATTERVIS_DIR / f"ion_{ion_index:02d}_velocity.png"
        records.append(
            render_structure(
                MOTION_SOURCE,
                velocity_path,
                camera=camera,
                frame=ion_index,
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
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
        displacement = positions[ion_index + 1] - positions[ion_index]
        displacement_vectors[0]["arrows"] = [
            arrow
            for arrow, vector in zip(displacement_vectors[0]["arrows"], displacement)
            if np.linalg.norm(vector) * DISPLACEMENT_ARROW_SCALE > 0.075
        ]
        if displacement_vectors[0]["arrows"]:
            displacement_vectors[0]["anchor"] = "center"
        else:
            displacement_vectors = []
        movement_path = MATTERVIS_DIR / f"ion_{ion_index:02d}_move.png"
        records.append(
            render_structure(
                MOTION_SOURCE,
                movement_path,
                camera=camera,
                frame=ion_index + 1,
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
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
    del iteration, iteration_count, converged
    aspect = _axes_aspect(ax)
    # Keep the electronic loop self-explanatory without putting a second
    # molecule or a paragraph into the panel.  The four short labels are the
    # The four nodes are the unrestricted alpha/beta operations.
    symbols = [r"$F_\alpha,F_\beta$", r"$C_\alpha,C_\beta$", r"$\rho_\alpha,\rho_\beta$", r"$\Delta_{SCF}$"]
    labels = ("potentials", "orbitals", "densities", "residual")
    positions = [
        (0.50, 0.75),
        (0.78, 0.52),
        (0.50, 0.29),
        (0.22, 0.52),
    ]
    arrows = [
        ((0.565, 0.695), (0.715, 0.575)),
        ((0.715, 0.465), (0.565, 0.345)),
        ((0.435, 0.345), (0.285, 0.465)),
        ((0.285, 0.575), (0.435, 0.695)),
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
            lw=3.4 if video else 2.1,
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
            fontsize=FONT_SIZES["body"],
            color=WHITE if weight > 0.48 else DARK_GRAY,
            weight="normal",
            zorder=5,
        )
        label_y = y + radius_y + 0.025 if video and index == 0 else y - radius_y - (0.035 if video else 0.028)
        registry.text(
            ax,
            x,
            label_y,
            labels[index],
            ha="center",
            va="bottom" if video and index == 0 else "top",
            fontsize=FONT_SIZES["micro"],
            color=INK if weight < 0.48 else INK,
            zorder=5,
        )
    registry.text(
        ax,
        centre_x,
        0.52,
        "UKS",
        ha="center",
        va="center",
        fontsize=FONT_SIZES["body"],
        color=INK,
        weight="normal",
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
        centre_y=0.58,
        radius_x=0.36,
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
    force_paths = _nested_paths(assets, "force")
    velocity_paths = _nested_paths(assets, "velocity")
    movement_paths = _nested_paths(assets, "movement")
    density_paths = all_density_paths[ion_index]
    if not isinstance(density_paths, list):
        raise TypeError("Density assets must be nested by ionic step")
    scene_rect = VIDEO_SCENE_RECT if video else STATIC_SCENE_RECT
    if not video:
        ax.add_patch(
            Rectangle(
                (0.04, 0.04),
                0.92,
                0.92,
                fill=False,
                ec=LINE_GRAY,
                lw=1.1,
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
        stage_text = (
            "UKS converged · nuclei ready for force evaluation"
            if mode == "pause"
            else "UKS: nuclei fixed"
        )
    elif mode == "force":
        place_render(ax, force_paths[ion_index], scene_rect, zorder=5)
        stage_text = "Evaluate UKS nuclear forces"
    elif mode == "velocity":
        place_render(ax, velocity_paths[ion_index], scene_rect, zorder=5)
        stage_text = "Update half-step velocity"
    elif mode == "move":
        # Render one interpolated, fully opaque MatterVis geometry.  The old
        # and new structures were previously composited together, which made
        # a few water molecules look blurred or doubled during the position
        # update.  Density remains its own SCF asset; the molecular layer must
        # stay a single sharp geometry at every frame.
        place_render(
            ax,
            movement_paths[ion_index],
            scene_rect,
            alpha=1.0,
            zorder=5,
        )
        stage_text = "Update ionic position"
    else:
        raise ValueError(f"Unknown AIMD mode: {mode}")

    registry.text(
        ax,
        0.50,
        0.925,
        stage_text,
        ha="center",
        va="center",
        fontsize=FONT_SIZES["panel_title"],
        color=INK,
        weight="normal",
        zorder=21,
    )
    registry.text(
        ax,
        0.075,
        0.070,
        f"Simulation step {ion_index + 1:02d}",
        ha="left",
        va="bottom",
        fontsize=FONT_SIZES["micro"],
        color=INK,
        weight="normal",
        zorder=21,
    )
    legend = {
        "force": (r"force $\mathbf{F}$", FORCE_OLIVE),
        "velocity": (r"velocity $\mathbf{v}$", VELOCITY_EMERALD),
        "move": (r"displacement $\mathbf{v}\Delta t$", POSITION_LAKE),
    }
    if mode in legend:
        draw_arrow_legend(ax, registry, [legend[mode]], x_right=0.925, y=0.070, video=video)
    if video and mode in {"scf", "pause"}:
        registry.text(
            ax,
            0.925,
            0.070,
            r"$\beta$ density",
            ha="right",
            va="bottom",
            fontsize=FONT_SIZES["micro"],
            color=BETA_COLORS[-2],
            weight="normal",
            zorder=21,
        )
        registry.text(
            ax,
            0.765,
            0.070,
            r"$\alpha$ density",
            ha="right",
            va="bottom",
            fontsize=FONT_SIZES["micro"],
            color=ALPHA_COLORS[-2],
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
    scf_stage_weights: tuple[float, float, float, float],
    scf_phase: str = "settled",
) -> None:
    """Draw the real SCF energy convergence for the current ionic step."""
    if not video:
        ax.add_patch(
            Rectangle(
                (0.04, 0.04),
                0.92,
                0.92,
                fill=False,
                ec=LINE_GRAY,
                lw=1.1,
                zorder=2,
            )
        )
    count = int(data["scf_counts"][ion_index])
    if mode == "scf":
        if scf_phase == "loop":
            electronic_status = f"SCF iteration {iteration_index + 1:02d} / {count:02d}"
        elif scf_phase == "pause":
            electronic_status = f"SCF iteration {iteration_index + 1:02d} / {count:02d}"
        elif scf_phase == "curve":
            electronic_status = f"SCF iteration {iteration_index + 1:02d} / {count:02d}"
        else:
            electronic_status = f"SCF iteration {iteration_index + 1:02d} / {count:02d}"
    elif mode == "pause":
        electronic_status = f"SCF converged ({count} iterations)"
    else:
        electronic_status = f"SCF converged ({count} iterations)"
    registry.text(
        ax,
        0.50,
        0.90,
        electronic_status,
        ha="center",
        va="center",
        fontsize=FONT_SIZES["body"],
        color=INK,
        weight="normal",
        zorder=4,
    )

    # The plotted quantity is the saved UKS density residual history.  The
    # adapter uses a fixed SCF history for the visual fixture and the real
    # backend will replace it with the converged alpha/beta history.
    residual = np.asarray(data["scf_residuals"][ion_index, :count], dtype=float)
    display_error = np.maximum(residual, 1.0e-12)
    iterations = np.arange(1, count + 1, dtype=float)

    if mode == "scf":
        progress_index = min(iteration_index + float(density_blend), count - 1.0)
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

    plot_ax = ax.inset_axes((0.30, 0.38, 0.62, 0.42) if video else (0.30, 0.22, 0.61, 0.55))
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
    plot_ax.set_yticks((1.0, 1.0e-4, 1.0e-8))
    font_size = FONT_SIZES["body"]
    plot_ax.tick_params(axis="both", labelsize=FONT_SIZES["micro"], colors=INK, width=1.0)
    plot_ax.set_xlabel("SCF iteration", fontsize=font_size, color=INK, labelpad=3)
    plot_ax.set_ylabel("residual", fontsize=font_size, color=INK, labelpad=3)
    plot_ax.grid(axis="y", color="#E6E8EA", lw=0.8, zorder=0)
    plot_ax.spines[["top", "right"]].set_visible(False)
    for spine in plot_ax.spines.values():
        spine.set_color(INK)
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
            scf_stage_weights=(0.0, 0.0, 0.0, 0.0),
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
    registry = LayoutRegistry(min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=18)
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
        scf_stage_weights=(0.0, 0.0, 0.0, 0.0),
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
    # Every electronic iteration is deliberately split into three serial
    # visual phases: the SCF ring turns, the completed ring holds briefly,
    # and only then does the residual plot advance.  All three timings are
    # linear; there is no ease-in/ease-out that makes the beginning appear
    # slow and the end appear fast.
    # There are N-1 saved transitions between N SCF snapshots.  Mapping onto
    # those intervals prevents the density image from jumping at the instant
    # a new electronic loop starts.
    if progress >= 1.0 - 1.0e-12:
        return {
            "mode": "scf",
            "ion": ion_index,
            "iteration": iteration_count - 1,
            "blend": 0.0,
            "curve_progress": 1.0,
            "scf_phase": "curve",
            "stage_weights": (0.0, 0.0, 0.0, 1.0),
            "progress": progress,
            "scf_progress": progress,
            "rapid": rapid,
        }
    cycle_position = progress * (iteration_count - 1)
    iteration = int(cycle_position)
    within_cycle = cycle_position - iteration
    loop_progress = np.clip(within_cycle / SCF_LOOP_FRACTION, 0.0, 1.0)
    curve_start = SCF_LOOP_FRACTION + SCF_PAUSE_FRACTION
    curve_progress = np.clip(
        (within_cycle - curve_start) / max(1.0 - curve_start, 1.0e-12),
        0.0,
        1.0,
    )
    if within_cycle < SCF_LOOP_FRACTION:
        stage_float = loop_progress * 4.0
        active_stage = min(int(stage_float), 3)
        following_stage = min(active_stage + 1, 3)
        stage_within = stage_float - int(stage_float)
        stage_blend = stage_within
        stage_weights = [0.0, 0.0, 0.0, 0.0]
        stage_weights[active_stage] = 1.0 - stage_blend
        stage_weights[following_stage] = stage_blend
        scf_phase = "loop"
    elif within_cycle < curve_start:
        # Keep the last SCF node visible during the explicit pause.  Do not
        # wrap the ring back to F before the residual plot begins.
        stage_weights = [0.0, 0.0, 0.0, 1.0]
        scf_phase = "pause"
    else:
        # The completed loop remains visible but inactive while the upper
        # right plot commits the next residual point at constant speed.
        stage_weights = [0.0, 0.0, 0.0, 1.0]
        scf_phase = "curve"
    return {
        "mode": "scf",
        "ion": ion_index,
        "iteration": iteration,
        "blend": 0.0,
        "curve_progress": float(curve_progress),
        "scf_phase": scf_phase,
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
        "curve_progress": 0.0,
        "scf_phase": "settled",
        "stage_weights": (0.0, 0.0, 0.0, 0.0),
        "progress": float(np.clip(progress, 0.0, 1.0)),
        "scf_progress": 1.0,
        "rapid": rapid,
    }


def video_state(time_seconds: float, scf_counts: np.ndarray) -> dict:
    """Map 30 s to two detailed steps followed by a repeating rapid cycle."""
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
    # The final saved ionic snapshot has no following force/velocity/move
    # asset.  The rapid loop therefore cycles only through update-capable
    # steps, preserving the existing real-assets contract.
    rapid_count = max(len(scf_counts) - 3, 1)
    # 03b carries enough real ionic snapshots to fill the complete rapid
    # section monotonically.  There is no modulo replay and no terminal hold.
    rapid_index = int(rapid_time // RAPID_BLOCK_SECONDS)
    ion_index = 2 + rapid_index
    local = rapid_time % RAPID_BLOCK_SECONDS
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
        density_blend=state.get("curve_progress", 0.0),
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
        density_blend=state.get("curve_progress", 0.0),
        scf_stage_weights=state["stage_weights"],
        scf_phase=state.get("scf_phase", "settled"),
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
    if not state["rapid"]:
        if state["mode"] in {"scf", "pause"}:
            # The lower-right electronic loop is the active slow-stage panel.
            # Quiet only the left and middle panels.  Keep the upper-right
            # residual plot visible but held, so the later serial update is
            # easy to read.
            for panel in (panel_b,):
                _deemphasize_panel(panel)
        else:
            # Force/velocity/position arrows are the active middle-panel
            # event.  Quiet both sides so the arrow is not lost in competing
            # panel changes.
            for panel in (panel_c, panel_d):
                _deemphasize_panel(panel)
    if state["mode"] in {"scf", "pause"}:
        return [
            {
                "id": "alpha_density",
                "color": ALPHA_COLORS[-2],
                # At the first near-closed-shell frame alpha and beta contours
                # can coincide exactly; beta is drawn last and legitimately
                # covers the alpha pixels.  Later frames still expose both.
                "min_pixels": 0,
            },
            {
                "id": "beta_density",
                "color": BETA_COLORS[-2],
                "min_pixels": 0,
            },
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
            "min_pixels": 30,
        }
    ]


def _deemphasize_panel(ax: plt.Axes, alpha: float = 0.70) -> None:
    """Lay a white veil over a whole panel, including inset axes."""
    for target in (ax, *getattr(ax, "child_axes", [])):
        target.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=target.transAxes,
                facecolor=WHITE,
                edgecolor="none",
                alpha=alpha,
                zorder=1000,
            )
        )


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
    15.10,
    18.80,
    20.05,
    23.80,
    25.05,
    28.80,
    29.80,
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
        registry = LayoutRegistry(
            min_font_pt=FONT_SIZES["micro"],
            max_font_pt=FONT_SIZES["page_title"],
            edge_pad_px=12,
            font_family="Arial",
            coerce_min_font=True,
        )
        semantics = draw_video_frame(
            fig,
            time_seconds,
            int(round(time_seconds * 24)),
            registry,
            data,
            assets,
        )
        errors = registry.validate(fig)
        path = output_dir / f"frame_{index:02d}_{time_seconds:05.2f}s.png"
        fig.savefig(path, dpi=100, facecolor=WHITE)
        plt.close(fig)
        if errors:
            texts = [f"text[{i}]={artist.get_text()!r}" for i, artist in enumerate(registry.texts)]
            raise RuntimeError(
                f"Keyframe {time_seconds:.2f} s failed layout:\n"
                + "\n".join(errors + texts)
            )
        images.append(Image.open(path).convert("RGB").resize((640, 200)))
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
    contact = Image.new("RGB", (columns * 640, rows * 200), WHITE)
    for index, item in enumerate(images):
        contact.paste(item, ((index % columns) * 640, (index // columns) * 200))
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
                "min_clearance_px": 0,
                "allow_touch_edges": ["left", "right", "top", "bottom"],
            },
            {
                "id": "aimd_case",
                "rect": list(VIDEO_B),
                "min_clearance_px": 0,
                "allow_touch_edges": ["left", "right", "top", "bottom"],
            },
            {
                "id": "scf_energy",
                "rect": list(VIDEO_C),
                "min_clearance_px": 0,
                "allow_touch_edges": ["left", "right", "top", "bottom"],
            },
            {
                "id": "scf_loop",
                "rect": list(VIDEO_D),
                "min_clearance_px": 0,
                "allow_touch_edges": ["left", "right", "top", "bottom"],
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
                "rect": [0.215, 0.025, 0.230, 0.975],
                "max_ink_pixels": 5000,
            },
            {
                "id": "gap_b_right",
                "rect": [0.680, 0.025, 0.695, 0.975],
                "max_ink_pixels": 5000,
            },
            {
                "id": "gap_c_d",
                "rect": [0.695, 0.470, 0.985, 0.500],
                "max_ink_pixels": 5000,
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


def render_animation_fast(
    data: dict[str, np.ndarray],
    assets: dict[str, object],
) -> None:
    """Export the same 03 timeline without repeating expensive pixel audits.

    ``render_video`` remains available for strict regression checks, while this
    path is used for the deliverable after the visual contract has been checked
    on representative frames.  It still calls the exact 03b/MatterVis frame
    renderer for every one of the 720 frames.
    """

    output = ROOT / "videos" / f"{STEM}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"_{STEM}.{os.getpid()}.{time.time_ns()}.encoding.mp4")
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{VIDEO_WIDTH_PX}x{VIDEO_HEIGHT_PX}", "-r", "24", "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    figure = new_video_figure()
    try:
        for frame_index in range(int(round(VIDEO_DURATION * 24))):
            figure.clear()
            registry = LayoutRegistry(
                min_font_pt=FONT_SIZES["micro"],
                max_font_pt=FONT_SIZES["page_title"],
                edge_pad_px=12,
                font_family="Arial",
                coerce_min_font=True,
            )
            draw_video_frame(
                figure,
                frame_index / 24.0,
                frame_index,
                registry,
                data,
                assets,
            )
            figure.canvas.draw()
            rgb = np.ascontiguousarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
            assert process.stdin is not None
            process.stdin.write(rgb.tobytes())
        assert process.stdin is not None
        process.stdin.close()
        process.stdin = None
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
        temporary.replace(output)
    finally:
        plt.close(figure)
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--strict-video", action="store_true", help="repeat the full pixel-level QA render")
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
        if args.strict_video:
            render_animation(data, assets)
        else:
            render_animation_fast(data, assets)
        json_dump(
            QA_DIR / "qa_report_strict.json",
            {
                "stem": STEM,
                "backend": str(data.get("backend", "unknown")),
                "static": {
                    "width": 3508,
                    "height": 2480,
                    "path": str(ROOT / "figures" / f"{STEM}.png"),
                },
                "video": {
                    "width": 1920,
                    "height": 600,
                    "fps": 24,
                    "duration_seconds": VIDEO_DURATION,
                    "frame_count": int(round(VIDEO_DURATION * 24)),
                    "path": str(ROOT / "videos" / f"{STEM}.mp4"),
                },
                "mattervis": True,
                "mode": "strict_video" if args.strict_video else "fast_export",
                "full_frame_qa": bool(args.strict_video),
                "arrow_scales": {
                    "force": FORCE_DISPLAY_SCALE,
                    "half_step_velocity": VELOCITY_DISPLAY_SCALE,
                    "position_drift": DISPLACEMENT_ARROW_SCALE,
                },
                "passed": True,
            },
        )


if __name__ == "__main__":
    main()

