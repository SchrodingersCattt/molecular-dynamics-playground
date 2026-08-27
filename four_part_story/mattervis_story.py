"""Shared MatterVis-first visual language for the four MD stories.

MatterVis owns atoms, chemical bonds, unit cells and density isosurfaces.  This
module only composes those verified renders with paper-space loops and
world-space annotations whose projection is fixed by the same camera contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image

from mat_viewer import load_structure, render
from mat_viewer.render.contracts import CameraSpec, RenderSpec, ViewSpec

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
    json_dump,
    sha256_file,
)


CAMERA_DIRECTION = np.array([1.55, -1.0, 0.62], dtype=float)
CAMERA_DIRECTION /= np.linalg.norm(CAMERA_DIRECTION)
CAMERA_UP = np.array([0.0, 0.0, 1.0], dtype=float)

STATIC_LEFT = (0.045, 0.185, 0.355, 0.89)
STATIC_RIGHT = (0.40, 0.185, 0.965, 0.89)
VIDEO_LEFT = (0.045, 0.19, 0.355, 0.90)
VIDEO_RIGHT = (0.40, 0.19, 0.965, 0.90)


@dataclass(frozen=True)
class SceneCamera:
    target: tuple[float, float, float]
    ortho_scale: float
    direction: tuple[float, float, float] = tuple(float(x) for x in CAMERA_DIRECTION)
    up: tuple[float, float, float] = tuple(float(x) for x in CAMERA_UP)
    scene_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def mattervis(self) -> CameraSpec:
        return CameraSpec.looking_along(
            self.direction,
            target=tuple(np.asarray(self.target) + np.asarray(self.scene_offset)),
            up=self.up,
            distance=24.0,
            projection="orthographic",
            ortho_scale=float(self.ortho_scale),
        )


def camera_for_source(
    source: Path,
    *,
    target: Iterable[float],
    ortho_scale: float,
    frame: int = 0,
    direction: Iterable[float] = CAMERA_DIRECTION,
    up: Iterable[float] = CAMERA_UP,
) -> SceneCamera:
    """Resolve MatterVis's canonical origin shift for an ASE-backed source."""
    from ase.io import read

    loaded = load_structure(source, frame=frame)
    scene_atoms = loaded.frames[0].bundle.scene.get("selected_atoms", [])
    source_atoms = read(source, index=frame)
    if not scene_atoms or len(source_atoms) == 0:
        offset = np.zeros(3)
    else:
        first = scene_atoms[0]
        source_index = int(first.get("_source_index", 0))
        offset = np.asarray(first["cart"], dtype=float) - source_atoms.positions[source_index]
    return SceneCamera(
        target=tuple(float(x) for x in target),
        ortho_scale=float(ortho_scale),
        direction=tuple(float(x) for x in direction),
        up=tuple(float(x) for x in up),
        scene_offset=tuple(float(x) for x in offset),
    )


def _rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))


def render_structure(
    source: Path,
    output: Path,
    *,
    camera: SceneCamera,
    frame: int = 0,
    view: str = "cluster",
    width: int = 1500,
    height: int = 1120,
    atom_scale: float = 1.10,
    bond_radius: float = 0.13,
    show_cell: bool = False,
) -> dict:
    """Render one transparent structure frame through public MatterVis APIs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    loaded = load_structure(source, frame=frame)
    result = render(
        loaded,
        output=output,
        backend="cpu",
        view=ViewSpec(display=view),
        camera=camera.mattervis(),
        render_spec=RenderSpec(
            representation="ball_stick",
            shading="smooth",
            backend="cpu",
            width=width,
            height=height,
            scale=1,
            background=(1.0, 1.0, 1.0, 0.0),
            atom_scale=atom_scale,
            bond_radius=bond_radius,
            show_hydrogen=True,
            show_cell=show_cell,
            show_labels=False,
            sphere_detail=(18, 28),
            cylinder_sides=18,
        ),
    )
    payload = {
        "schema": result.schema,
        "backend": result.backend,
        "format": result.format,
        "width": result.width,
        "height": result.height,
        "plan_sha256": result.plan_sha256,
        "output_sha256": result.output_sha256,
        "output": str(output),
        "source_sha256": sha256_file(source),
        "frame": frame,
        "view": view,
        "camera": asdict(camera),
        "warnings": list(result.warnings),
        "metadata": dict(result.metadata),
    }
    if int(np.count_nonzero(_rgba(output)[:, :, 3])) == 0:
        raise RuntimeError(f"MatterVis produced an empty transparent render: {output}")
    json_dump(output.with_suffix(".json"), payload)
    return payload


def render_density_cube(
    source: Path,
    output: Path,
    *,
    camera: SceneCamera,
    isovalue: float,
    opacity: float = 0.38,
    width: int = 1500,
    height: int = 1120,
) -> dict:
    """Render a positive electron-density Cube with a navy MatterVis mesh."""
    from mat_viewer.cube.cpu import cube_isosurface_meshes

    output.parent.mkdir(parents=True, exist_ok=True)
    loaded = load_structure(source, input_format="cube")
    for frame in loaded.frames:
        bundle = frame.bundle
        meshes = cube_isosurface_meshes(
            bundle.cube_data,
            isovalue=isovalue,
            stride=1,
            positive_color=NAVY,
            negative_color=GREEN,
            opacity=opacity,
        )
        bundle.cube_data.surface_meshes = meshes
        bundle.scene["isosurfaces"] = meshes
    result = render(
        loaded,
        output=output,
        backend="cpu",
        view=ViewSpec(display="cluster"),
        camera=camera.mattervis(),
        render_spec=RenderSpec(
            representation="ball_stick",
            shading="smooth",
            backend="cpu",
            width=width,
            height=height,
            scale=1,
            background=(1.0, 1.0, 1.0, 0.0),
            atom_scale=0.90,
            bond_radius=0.105,
            show_hydrogen=True,
            show_cell=False,
            show_labels=False,
            sphere_detail=(18, 28),
            cylinder_sides=18,
        ),
    )
    payload = {
        "schema": result.schema,
        "backend": result.backend,
        "format": result.format,
        "width": result.width,
        "height": result.height,
        "plan_sha256": result.plan_sha256,
        "output_sha256": result.output_sha256,
        "output": str(output),
        "source_sha256": sha256_file(source),
        "isovalue": float(isovalue),
        "opacity": float(opacity),
        "camera": asdict(camera),
        "warnings": list(result.warnings),
        "metadata": dict(result.metadata),
    }
    json_dump(output.with_suffix(".json"), payload)
    return payload


def place_render(
    ax: plt.Axes,
    image_path: Path,
    rect: tuple[float, float, float, float],
    *,
    alpha: float = 1.0,
    zorder: float = 4.0,
) -> None:
    x0, y0, x1, y1 = rect
    ax.imshow(
        _rgba(image_path),
        extent=(x0, x1, y0, y1),
        origin="upper",
        interpolation="lanczos",
        alpha=float(alpha),
        zorder=zorder,
        aspect="auto",
    )


def camera_basis(camera: SceneCamera) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(camera.direction, dtype=float)
    direction /= np.linalg.norm(direction)
    forward = -direction
    up_hint = np.asarray(camera.up, dtype=float)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    screen_up = np.cross(right, forward)
    screen_up /= np.linalg.norm(screen_up)
    return right, screen_up


def project_world(
    points: np.ndarray,
    *,
    camera: SceneCamera,
    rect: tuple[float, float, float, float],
    image_aspect: float = 1500.0 / 1120.0,
) -> np.ndarray:
    """Project world coordinates exactly under the shared orthographic camera."""
    xyz = np.asarray(points, dtype=float)
    right, screen_up = camera_basis(camera)
    centred = xyz - np.asarray(camera.target, dtype=float)
    normalized = np.column_stack(
        (
            0.5 + centred @ right / (2.0 * camera.ortho_scale * image_aspect),
            0.5 + centred @ screen_up / (2.0 * camera.ortho_scale),
        )
    )
    x0, y0, x1, y1 = rect
    return np.column_stack(
        (x0 + normalized[:, 0] * (x1 - x0), y0 + normalized[:, 1] * (y1 - y0))
    )


def draw_world_arrow(
    ax: plt.Axes,
    registry: LayoutRegistry,
    origin: np.ndarray,
    vector: np.ndarray,
    *,
    camera: SceneCamera,
    rect: tuple[float, float, float, float],
    color: str,
    display_scale: float,
    video: bool,
    alpha: float = 1.0,
    zorder: float = 15.0,
) -> None:
    xy = project_world(
        np.vstack((origin, np.asarray(origin) + float(display_scale) * np.asarray(vector))),
        camera=camera,
        rect=rect,
    )
    linewidth = 7.5 if video else 4.2
    head = 34 if video else 22
    halo = FancyArrowPatch(
        tuple(xy[0]),
        tuple(xy[1]),
        arrowstyle="-|>",
        mutation_scale=head + (8 if video else 5),
        linewidth=linewidth + (6 if video else 3),
        color=WHITE,
        alpha=0.96 * alpha,
        zorder=zorder,
    )
    arrow = FancyArrowPatch(
        tuple(xy[0]),
        tuple(xy[1]),
        arrowstyle="-|>",
        mutation_scale=head,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        zorder=zorder + 0.2,
    )
    ax.add_patch(halo)
    ax.add_patch(arrow)
    registry.arrows.extend((halo, arrow))


def draw_world_segment(
    ax: plt.Axes,
    first: np.ndarray,
    second: np.ndarray,
    *,
    camera: SceneCamera,
    rect: tuple[float, float, float, float],
    color: str = NAVY,
    linewidth: float = 5.0,
    linestyle: str = "--",
    alpha: float = 1.0,
    zorder: float = 12.0,
) -> np.ndarray:
    xy = project_world(np.vstack((first, second)), camera=camera, rect=rect)
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=WHITE,
        lw=linewidth + 4.0,
        ls=linestyle,
        solid_capstyle="round",
        alpha=alpha,
        zorder=zorder,
    )
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=color,
        lw=linewidth,
        ls=linestyle,
        solid_capstyle="round",
        alpha=alpha,
        zorder=zorder + 0.2,
    )
    return xy


def draw_vv_loop(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    video: bool,
    active_stage: int | None,
    equation: str | None = None,
) -> None:
    """Draw the shared empty three-stage Velocity Verlet loop."""
    nodes = [(0.50, 0.79, r"$\mathbf{r}$", "position"), (0.78, 0.43, r"$\mathbf{a}$", "acceleration"), (0.22, 0.43, r"$\mathbf{v}$", "velocity")]
    paths = [((0.56, 0.75), (0.73, 0.50), -0.12), ((0.70, 0.39), (0.30, 0.39), -0.10), ((0.27, 0.50), (0.44, 0.75), -0.12)]
    for index, (start, end, bend) in enumerate(paths):
        active = active_stage == index
        registry.arrow(
            ax,
            start,
            end,
            connectionstyle=f"arc3,rad={bend}",
            arrowstyle="-|>",
            mutation_scale=28 if video else 18,
            lw=4.2 if video else 2.5,
            color=INK if active else LINE_GRAY,
            zorder=2,
        )
    for index, (x, y, symbol, label) in enumerate(nodes):
        active = active_stage == index
        ax.add_patch(
            Circle(
                (x, y),
                0.082 if video else 0.068,
                fc=INK if active else WHITE,
                ec=INK if active else LINE_GRAY,
                lw=4.0 if video else 2.3,
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
            fontsize=25 if video else 14,
            color=WHITE if active else DARK_GRAY,
            weight="bold",
            zorder=5,
        )
        registry.text(
            ax,
            x,
            y - (0.125 if video else 0.108),
            label,
            ha="center",
            va="top",
            fontsize=18 if video else 10,
            color=INK if active else DARK_GRAY,
            zorder=5,
        )
    registry.text(
        ax,
        0.50,
        0.57,
        "Velocity\nVerlet",
        ha="center",
        va="center",
        fontsize=25 if video else 15,
        color=INK,
        weight="bold",
        linespacing=1.0,
    )
    if equation:
        registry.text(
            ax,
            0.50,
            0.12,
            equation,
            ha="center",
            va="center",
            fontsize=19 if video else 10,
            color=INK,
        )


def add_story_title(
    fig: plt.Figure,
    registry: LayoutRegistry,
    title: str,
    subtitle: str,
    *,
    video: bool,
) -> None:
    title_artist = fig.text(
        0.048,
        0.945,
        title,
        ha="left",
        va="top",
        fontsize=36 if video else 23,
        color=INK,
        weight="bold",
    )
    subtitle_artist = fig.text(
        0.050,
        0.890,
        subtitle,
        ha="left",
        va="top",
        fontsize=20 if video else 11,
        color=DARK_GRAY,
    )
    registry.texts.extend((title_artist, subtitle_artist))


def draw_cutoff_sphere(
    ax: plt.Axes,
    centre: np.ndarray,
    radius: float,
    *,
    camera: SceneCamera,
    rect: tuple[float, float, float, float],
    alpha: float,
) -> None:
    """Draw an orthographic projection of a real 3D sphere and great circles."""
    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    loops: list[np.ndarray] = []
    for latitude in (-45.0, 0.0, 45.0):
        polar = np.deg2rad(latitude)
        loops.append(
            np.column_stack(
                (
                    radius * np.cos(polar) * np.cos(angles),
                    radius * np.cos(polar) * np.sin(angles),
                    np.full_like(angles, radius * np.sin(polar)),
                )
            )
            + centre
        )
    for longitude in (0.0, 60.0, 120.0):
        phi = np.deg2rad(longitude)
        loops.append(
            np.column_stack(
                (
                    radius * np.cos(angles) * np.cos(phi),
                    radius * np.cos(angles) * np.sin(phi),
                    radius * np.sin(angles),
                )
            )
            + centre
        )
    for loop in loops:
        xy = project_world(loop, camera=camera, rect=rect)
        ax.plot(xy[:, 0], xy[:, 1], color=NAVY, lw=2.2, alpha=0.48 * alpha, zorder=8)
    right, screen_up = camera_basis(camera)
    disk = np.vstack(
        [
            centre + radius * (np.cos(angle) * right + np.sin(angle) * screen_up)
            for angle in angles
        ]
    )
    xy = project_world(disk, camera=camera, rect=rect)
    ax.fill(xy[:, 0], xy[:, 1], color=NAVY, alpha=0.055 * alpha, zorder=3)
    ax.plot(xy[:, 0], xy[:, 1], color=NAVY, lw=4.0, alpha=0.86 * alpha, zorder=9)


def write_provenance_index(directory: Path, records: Iterable[dict]) -> None:
    rows = list(records)
    json_dump(
        directory / "provenance.json",
        {
            "renderer": "MatterVis public CPU backend",
            "camera_direction": CAMERA_DIRECTION.tolist(),
            "camera_up": CAMERA_UP.tolist(),
            "records": rows,
        },
    )
