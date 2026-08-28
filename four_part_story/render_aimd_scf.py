"""Render the two-column AIMD/RHF density story."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image
from scipy import ndimage

from common import (
    DARK_GRAY,
    GREEN,
    INK,
    LIGHT_GRAY,
    LINE_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    axes_from_top_slot,
    mix_hex,
    new_static_figure,
    render_source_panel,
    render_video,
    save_static,
    smoothstep,
)
from mattervis_story import (
    STATIC_LEFT,
    STATIC_RIGHT,
    SceneCamera,
    add_story_title,
    camera_for_source,
    draw_vv_loop,
    make_vector_group,
    place_render,
    render_structure,
    write_provenance_index,
)


ROOT = Path(__file__).resolve().parent
STEM = "03_aimd_scf"
QA_DIR = ROOT / "_qa" / "03_aimd_scf"
MATTERVIS_DIR = QA_DIR / "source" / "mattervis"
CUBE_DIR = QA_DIR / "source" / "cubes"
DATA_PATH = ROOT / "data" / "aimd_h2o_dimer.npz"
MOTION_SOURCE = ROOT / "data" / "aimd_h2o_dimer_motion_display.extxyz"

STATIC_SCENE_RECT = (0.00, 0.13, 0.65, 0.92)
VIDEO_SCENE_RECT = (0.00, 0.10, 0.78, 0.92)
AIMD_VIDEO_LEFT = (0.045, 0.21, 0.275, 0.90)
AIMD_VIDEO_RIGHT = (0.30, 0.17, 0.965, 0.91)
CAMERA_SCALE = 1.50
POSITION_LAKE = "#4E9BB5"
FORCE_OLIVE = "#A99C50"
VELOCITY_EMERALD = "#2F8562"
FORCE_DISPLAY_SCALE = 1.35
DISPLACEMENT_ARROW_SCALE = 700.0
BOHR_TO_ANGSTROM = 0.529177210903
DENSITY_LEVELS = np.asarray([0.003, 0.007, 0.015, 0.035, 0.080, 0.180, 0.400, 0.900])
DENSITY_COLORS = (
    "#DDE5EA", "#CEDAE1", "#BCCDD7", "#A8BDCA",
    "#91AABB", "#7894A8", "#607E94", "#49687F",
)

INTRO_END = 1.0
SCF_END = 11.45
FORCE_END = 13.0
VIDEO_DURATION = 15.0

POSITION_EQUATION = (
    r"$\mathbf{r}_{n+1}=\mathbf{r}_n+\mathbf{v}_n\Delta t+"
    r"\frac{1}{2}\mathbf{a}_n\Delta t^2$"
)
ACCELERATION_EQUATION = r"$\mathbf{a}_{n}=\mathbf{F}_{\mathrm{RHF}}/m$"


def molecular_plane(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed molecular-plane camera basis for the water dimer."""
    xyz = np.asarray(positions, dtype=float)
    centre = 0.5 * (xyz[0] + xyz[3])
    horizontal = xyz[3] - xyz[0]
    horizontal /= np.linalg.norm(horizontal)
    _, _, right_vectors = np.linalg.svd(xyz - xyz.mean(axis=0), full_matrices=False)
    normal = right_vectors[-1]
    if normal[2] < 0.0:
        normal = -normal
    vertical = np.cross(normal, horizontal)
    vertical /= np.linalg.norm(vertical)
    normal = np.cross(horizontal, vertical)
    normal /= np.linalg.norm(normal)
    return centre, horizontal, vertical, normal


def parse_cube(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count, *origin = lines[2].split()
    atom_count = abs(int(atom_count))
    shape: list[int] = []
    axes: list[list[float]] = []
    for line in lines[3:6]:
        count, *axis = line.split()
        shape.append(abs(int(count)))
        axes.append([float(value) for value in axis[:3]])
    values = np.fromstring(" ".join(lines[6 + atom_count :]), sep=" ")
    if values.size != int(np.prod(shape)):
        raise ValueError(f"Unexpected cube data size in {path}")
    return values.reshape(tuple(shape)), np.asarray(origin[:3], float), np.asarray(axes, float)


def oxygen_pixel_centres(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    red = (
        (rgb[:, :, 0] > 105)
        & (rgb[:, :, 0].astype(float) > 1.12 * rgb[:, :, 1])
        & (rgb[:, :, 0].astype(float) > 1.05 * rgb[:, :, 2])
    )
    labels, count = ndimage.label(red)
    centres: list[tuple[int, np.ndarray]] = []
    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < 250:
            continue
        distance = ndimage.distance_transform_edt(component)
        row, column = np.unravel_index(int(np.argmax(distance)), distance.shape)
        centres.append((area, np.asarray([float(column), float(row)])))
    if len(centres) < 2:
        raise RuntimeError("Could not locate both MatterVis oxygen centres")
    result = np.asarray([item[1] for item in sorted(centres, reverse=True)[:2]])
    return result[np.argsort(result[:, 0])], (rgb.shape[1], rgb.shape[0])


def render_density_plane_scene(
    cube_path: Path,
    structure_image: Path,
    output: Path,
    *,
    positions: np.ndarray,
) -> dict:
    """Render a real cube slice beneath an aligned MatterVis structure."""
    oxygen_pixels, (width, height) = oxygen_pixel_centres(structure_image)
    centre, horizontal, vertical, _ = molecular_plane(positions)
    oxygen_distance = float(np.linalg.norm(positions[3] - positions[0]))
    pixel_scale = float((oxygen_pixels[1, 0] - oxygen_pixels[0, 0]) / oxygen_distance)
    pixel_origin = oxygen_pixels.mean(axis=0)
    density, cube_origin, cube_axes = parse_cube(cube_path)
    sample_width = 600
    sample_height = int(round(sample_width * height / width))
    pixel_x = np.linspace(0.0, width - 1.0, sample_width)
    pixel_y = np.linspace(0.0, height - 1.0, sample_height)
    grid_x, grid_y = np.meshgrid(pixel_x, pixel_y)
    plane_x = (grid_x - pixel_origin[0]) / pixel_scale
    plane_y = -(grid_y - pixel_origin[1]) / pixel_scale
    points = (
        centre + plane_x[:, :, None] * horizontal + plane_y[:, :, None] * vertical
    ) / BOHR_TO_ANGSTROM
    indices = (points - cube_origin) @ np.linalg.inv(cube_axes)
    sampled = ndimage.map_coordinates(
        density,
        [indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]],
        order=1,
        mode="constant",
        cval=0.0,
    )
    figure = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="white")
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.set_xlim(0.0, width - 1.0)
    axes.set_ylim(height - 1.0, 0.0)
    axes.contour(
        grid_x, grid_y, sampled, levels=DENSITY_LEVELS, colors=DENSITY_COLORS,
        linewidths=np.linspace(1.1, 2.2, len(DENSITY_LEVELS)), antialiased=True,
    )
    axes.axis("off")
    figure.canvas.draw()
    base = Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy(), mode="RGBA")
    plt.close(figure)
    structure = np.asarray(Image.open(structure_image).convert("RGBA"), dtype=np.uint8).copy()
    structure[:, :, 3] = np.where(
        np.max(255 - structure[:, :, :3], axis=2) > 3, 255, 0
    ).astype(np.uint8)
    base.alpha_composite(Image.fromarray(structure, mode="RGBA"))
    output.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(output)
    return {
        "source": str(cube_path),
        "structure_source": str(structure_image),
        "output": str(output),
        "levels": DENSITY_LEVELS.tolist(),
        "pixels_per_angstrom": pixel_scale,
    }


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def prepare_mattervis(
    data: dict[str, np.ndarray],
) -> tuple[dict[str, list[Path] | Path], SceneCamera]:
    """Render planar density, force, and movement with one fixed camera."""
    cube_paths = [
        CUBE_DIR / f"rho_{iteration:02d}_display.cube"
        for iteration in range(1, len(data["density_iterations"]) + 1)
    ]
    target, _horizontal, vertical, normal = molecular_plane(data["positions"])
    camera = camera_for_source(
        MOTION_SOURCE,
        target=target,
        ortho_scale=CAMERA_SCALE,
        frame=0,
        direction=tuple(normal),
        up=tuple(vertical),
    )
    records: list[dict] = []
    density_structure_path = MATTERVIS_DIR / "density_structure_plane.png"
    records.append(
        render_structure(
            MOTION_SOURCE,
            density_structure_path,
            camera=camera,
            frame=0,
            width=1500,
            height=950,
            atom_scale=0.82,
            bond_radius=0.090,
        )
    )
    density_paths: list[Path] = []
    for iteration, cube_path in enumerate(cube_paths, start=1):
        output = MATTERVIS_DIR / f"density_plane_{iteration:02d}.png"
        records.append(
            render_density_plane_scene(
                cube_path,
                density_structure_path,
                output,
                positions=data["positions"],
            )
        )
        density_paths.append(output)

    force_vectors = make_vector_group(
        "rhf-nuclear-force",
        data["positions"],
        data["forces"],
        scale=FORCE_DISPLAY_SCALE,
        color=FORCE_OLIVE,
        tail_offset=0.0,
        style={
            "shaft_radius": 0.055,
            "head_length": 0.16,
            "head_radius": 0.12,
            "sides": 18,
        },
    )
    ghost_path = MATTERVIS_DIR / "motion_ghost.png"
    records.append(
        render_structure(
            MOTION_SOURCE,
            ghost_path,
            camera=camera,
            frame=0,
            width=1500,
            height=950,
            atom_scale=0.90,
            bond_radius=0.105,
        )
    )
    force_path = MATTERVIS_DIR / "force_clean.png"
    records.append(
        render_structure(
            MOTION_SOURCE,
            force_path,
            camera=camera,
            frame=0,
            width=1500,
            height=950,
            atom_scale=0.82,
            bond_radius=0.090,
            vector_overlays=force_vectors,
        )
    )
    movement_paths: list[Path] = []
    displacement_vectors = make_vector_group(
        "vv-nuclear-displacement",
        data["positions"],
        data["vv_displacement"],
        scale=DISPLACEMENT_ARROW_SCALE,
        color=POSITION_LAKE,
        tail_offset=0.0,
        style={
            "shaft_radius": 0.050,
            "head_radius_ratio": 2.0,
            "head_length_ratio": 0.26,
            "sides": 16,
        },
    )
    for frame in range(len(data["display_motion_positions"])):
        output = MATTERVIS_DIR / f"motion_{frame:02d}.png"
        records.append(
            render_structure(
                MOTION_SOURCE,
                output,
                camera=camera,
                frame=frame,
                width=1500,
                height=950,
                atom_scale=0.90,
                bond_radius=0.105,
                vector_overlays=displacement_vectors,
            )
        )
        movement_paths.append(output)

    write_provenance_index(MATTERVIS_DIR, records)
    return {
        "density": density_paths,
        "force": force_path,
        "density_force": force_path,
        "ghost": ghost_path,
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
    active_stage: int | None,
    active_weight: float,
    iteration: int,
    converged: bool,
) -> None:
    """Draw the small paper-space SCF loop; no molecular structure enters it."""
    aspect = _axes_aspect(ax)
    labels = ["build Fock", "solve\norbitals", "update\ndensity", "check\nconvergence"]
    symbols = [r"$F$", r"$C$", r"$\rho$", r"$?$" ]
    if video:
        positions = [(0.865, 0.76), (0.935, 0.61), (0.865, 0.46), (0.795, 0.61)]
        arrows = [
            ((0.887, 0.72), (0.916, 0.65)),
            ((0.916, 0.57), (0.887, 0.50)),
            ((0.843, 0.50), (0.814, 0.57)),
            ((0.814, 0.65), (0.843, 0.72)),
        ]
        radius_x = 0.025
        centre_x = 0.865
    else:
        positions = [(0.790, 0.78), (0.880, 0.60), (0.790, 0.42), (0.700, 0.60)]
        arrows = [
            ((0.820, 0.74), (0.855, 0.65)),
            ((0.855, 0.55), (0.820, 0.46)),
            ((0.760, 0.46), (0.725, 0.55)),
            ((0.725, 0.65), (0.760, 0.74)),
        ]
        radius_x = 0.031
        centre_x = 0.790
    for index, (start, end) in enumerate(arrows):
        registry.arrow(
            ax,
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=20 if video else 15,
            lw=3.2 if video else 2.2,
            color=INK if active_stage == index else LINE_GRAY,
            zorder=3,
        )
    radius_y = radius_x * aspect
    label_positions = [
        (0.790, 0.895, "center"),
        (0.925, 0.710, "center"),
        (0.790, 0.285, "center"),
        (0.665, 0.750, "center"),
    ]
    for index, ((x, y), symbol, label, label_position) in enumerate(
        zip(positions, symbols, labels, label_positions)
    ):
        weight = active_weight if active_stage == index else 0.0
        fill = mix_hex(WHITE, INK, weight)
        ax.add_patch(
            Ellipse(
                (x, y),
                2.0 * radius_x,
                2.0 * radius_y,
                fc=fill,
                ec=INK if weight > 0.1 else LINE_GRAY,
                lw=3.2 if video else 1.9,
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
            fontsize=20 if video else 11,
            color=WHITE if weight > 0.48 else DARK_GRAY,
            weight="bold",
            zorder=5,
        )
        if not video:
            registry.text(
                ax,
                label_position[0],
                label_position[1],
                label,
                ha=label_position[2],
                va="center",
                fontsize=10,
                color=DARK_GRAY,
                linespacing=0.95,
            )
    registry.text(
        ax,
        centre_x,
        0.61 if video else 0.60,
        "SCF",
        ha="center",
        va="center",
        fontsize=20 if video else 14,
        color=INK,
        weight="bold",
    )
    if video:
        if active_stage is not None:
            active_label = labels[active_stage].replace("\n", " ")
        elif converged:
            active_label = "force from converged density"
        else:
            active_label = "initial density guess"
        registry.text(
            ax,
            centre_x,
            0.345,
            active_label,
            ha="center",
            va="center",
            fontsize=18,
            color=INK,
            weight="bold",
        )
    status = "converged" if converged else f"iteration {iteration:02d} / 19"
    registry.text(
        ax,
        centre_x,
        0.285 if video else 0.205,
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
    if stage == 0 and video:
        equation = (
            r"$\mathbf{r}_{n+1}=\mathbf{r}_n+\mathbf{v}_n\Delta t$"
            "\n"
            r"$+\frac{1}{2}\mathbf{a}_n\Delta t^2$"
        )
    else:
        equation = POSITION_EQUATION if stage == 0 else ACCELERATION_EQUATION
    draw_vv_loop(
        ax,
        registry,
        video=video,
        active_stage=stage,
        equation=None if video else equation,
    )
    if video:
        registry.text(
            ax,
            0.50,
            0.22,
            equation,
            ha="center",
            va="center",
            fontsize=18,
            color=INK,
        )


def draw_case(
    ax: plt.Axes,
    registry: LayoutRegistry,
    assets: dict[str, list[Path] | Path],
    *,
    video: bool,
    mode: str,
    iteration_index: int,
    density_blend: float,
    active_scf_stage: int | None,
    active_scf_weight: float,
    phase_progress: float,
) -> None:
    density_paths = assets["density"]
    movement_paths = assets["movement"]
    assert isinstance(density_paths, list) and isinstance(movement_paths, list)
    force_path = assets["force"]
    density_force_path = assets["density_force"]
    ghost_path = assets["ghost"]
    assert isinstance(force_path, Path) and isinstance(density_force_path, Path)
    assert isinstance(ghost_path, Path)
    scene_rect = VIDEO_SCENE_RECT if video else STATIC_SCENE_RECT

    if mode in {"intro", "scf"}:
        current = min(iteration_index, len(density_paths) - 1)
        following = min(current + 1, len(density_paths) - 1)
        place_render(
            ax,
            density_paths[current],
            scene_rect,
            alpha=1.0 - density_blend,
            zorder=4,
        )
        if following != current and density_blend > 0.001:
            place_render(
                ax,
                density_paths[following],
                scene_rect,
                alpha=density_blend,
                zorder=5,
            )
        heading = r"real $\rho^k(\mathbf{r})$ on one fixed molecular-plane slice"
        heading_color = NAVY
        note = "round initial contours resolve into the converged bonding density"
        converged = False
    elif mode == "force":
        # Video cuts cleanly from the converged density to atom-centred forces.
        # The A4 static keeps both layers to preserve the force-source relation.
        place_render(
            ax,
            force_path if video else density_force_path,
            scene_rect,
            zorder=5,
        )
        heading = r"converged $\rho(\mathbf{r})$ $\rightarrow$ nuclear forces"
        heading_color = INK
        note = "the SCF loop stops before forces return to the MD integrator"
        converged = True
    else:
        fade = smoothstep(min(phase_progress / 0.22, 1.0))
        place_render(ax, force_path, scene_rect, alpha=1.0 - fade, zorder=3)
        place_render(ax, ghost_path, scene_rect, alpha=0.18 * fade, zorder=4)
        frame = int(round(smoothstep(phase_progress) * (len(movement_paths) - 1)))
        place_render(ax, movement_paths[frame], scene_rect, alpha=fade, zorder=5)
        heading = "forces return to Velocity Verlet"
        heading_color = POSITION_LAKE
        note = "faint old nuclei mark the one-step displacement"
        converged = True

    registry.text(
        ax,
        0.39 if video else 0.34,
        0.975,
        heading,
        ha="center",
        va="top",
        fontsize=25 if video else 13,
        color=heading_color,
        weight="bold",
    )
    if not video:
        registry.text(
            ax,
            0.34,
            0.065,
            note,
            ha="center",
            va="bottom",
            fontsize=10,
            color=DARK_GRAY,
        )
    draw_scf_loop(
        ax,
        registry,
        video=video,
        active_stage=active_scf_stage,
        active_weight=active_scf_weight,
        iteration=iteration_index + 1,
        converged=converged,
    )


def render_static(
    data: dict[str, np.ndarray],
    assets: dict[str, list[Path] | Path],
) -> None:
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
            mode="force",
            iteration_index=len(data["density_iterations"]) - 1,
            density_blend=0.0,
            active_scf_stage=None,
            active_scf_weight=0.0,
            phase_progress=1.0,
        ),
        width_px=1800,
        height_px=1400,
    )
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    add_story_title(
        fig,
        registry,
        "Ab initio MD: one force requires an SCF loop",
        "A real RHF water-dimer density is rebuilt until self-consistent, then nuclei move",
        video=False,
    )
    left = axes_from_top_slot(fig, STATIC_LEFT)
    right = axes_from_top_slot(fig, STATIC_RIGHT)
    draw_left(left, registry, video=False, stage=1)
    draw_case(
        right,
        registry,
        assets,
        video=False,
        mode="force",
        iteration_index=len(data["density_iterations"]) - 1,
        density_blend=0.0,
        active_scf_stage=None,
        active_scf_weight=0.0,
        phase_progress=1.0,
    )
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)


def video_state(time_seconds: float, iteration_count: int) -> dict:
    if time_seconds < INTRO_END:
        return {
            "mode": "intro",
            "iteration": 0,
            "blend": 0.0,
            "active_stage": None,
            "active_weight": 0.0,
            "progress": time_seconds / INTRO_END,
        }
    if time_seconds < SCF_END:
        local = (time_seconds - INTRO_END) / (SCF_END - INTRO_END)
        iteration_float = min(local * iteration_count, iteration_count - 1.0e-6)
        iteration = int(iteration_float)
        within = iteration_float - iteration
        stage_float = within * 4.0
        active_stage = min(int(stage_float), 3)
        stage_within = stage_float - active_stage
        weight = smoothstep(min(stage_within / 0.24, 1.0))
        return {
            "mode": "scf",
            "iteration": iteration,
            "blend": smoothstep(within),
            "active_stage": active_stage,
            "active_weight": weight,
            "progress": local,
        }
    if time_seconds < FORCE_END:
        return {
            "mode": "force",
            "iteration": iteration_count - 1,
            "blend": 0.0,
            "active_stage": None,
            "active_weight": 0.0,
            "progress": (time_seconds - SCF_END) / (FORCE_END - SCF_END),
        }
    return {
        "mode": "move",
        "iteration": iteration_count - 1,
        "blend": 0.0,
        "active_stage": None,
        "active_weight": 0.0,
        "progress": min((time_seconds - FORCE_END) / (VIDEO_DURATION - FORCE_END), 1.0),
    }


def draw_video_frame(
    fig: plt.Figure,
    time_seconds: float,
    frame_index: int,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    assets: dict[str, list[Path] | Path],
) -> list[dict]:
    del frame_index
    state = video_state(time_seconds, len(data["density_iterations"]))
    add_story_title(
        fig,
        registry,
        "AIMD: SCF supplies each nuclear force",
        "real RHF/STO-3G density · H₂O dimer",
        video=True,
    )
    left = axes_from_top_slot(fig, AIMD_VIDEO_LEFT)
    right = axes_from_top_slot(fig, AIMD_VIDEO_RIGHT)
    left_stage = 0 if state["mode"] == "move" else 1
    draw_left(left, registry, video=True, stage=left_stage)
    draw_case(
        right,
        registry,
        assets,
        video=True,
        mode=state["mode"],
        iteration_index=state["iteration"],
        density_blend=state["blend"],
        active_scf_stage=state["active_stage"],
        active_scf_weight=state["active_weight"],
        phase_progress=state["progress"],
    )
    semantics = [
        {"id": "density_contours", "color": DENSITY_COLORS[-1], "min_pixels": 80},
    ]
    if state["mode"] == "move" and state["progress"] >= 0.18:
        semantics.append({"id": "displacement", "color": POSITION_LAKE, "min_pixels": 180})
    if state["mode"] == "force":
        semantics.append({"id": "nuclear_force", "color": FORCE_OLIVE, "min_pixels": 260})
    return semantics


def render_animation(
    data: dict[str, np.ndarray],
    assets: dict[str, list[Path] | Path],
) -> None:
    audit_config = {
        "panels": [
            {"id": "integrator", "rect": list(AIMD_VIDEO_LEFT), "min_clearance_px": 18},
            {"id": "aimd_case", "rect": list(AIMD_VIDEO_RIGHT), "min_clearance_px": 18},
        ],
        "whitespace": {
            "background_threshold": 245,
            "min_ink_fraction": 0.020,
            "min_panel_bbox_fill": 0.24,
            "grid_rows": 12,
            "grid_columns": 20,
        },
        "bands": [
            {
                "id": "column_gap",
                "rect": [0.275, 0.19, 0.300, 0.90],
                "max_ink_pixels": 0,
            }
        ],
    }
    render_video(
        stem=STEM,
        duration_seconds=VIDEO_DURATION,
        draw_frame=lambda fig, time, index, registry: draw_video_frame(
            fig, time, index, registry, data, assets
        ),
        audit_config=audit_config,
        qa_directory=QA_DIR / "_qa",
        representative_times=[0.5, 2.5, 6.5, 10.8, 12.2, 14.2],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    assets, _camera = prepare_mattervis(data)
    render_static(data, assets)
    if not args.static_only:
        render_animation(data, assets)


if __name__ == "__main__":
    main()
