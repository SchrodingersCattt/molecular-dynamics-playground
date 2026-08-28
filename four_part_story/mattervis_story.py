"""Shared MatterVis-first visual language for the four MD stories.

MatterVis owns atoms, chemical bonds, unit cells, density isosurfaces, and all
world-space vector arrows.  This module only composes complete MatterVis scene
renders with paper-space explanatory diagrams.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch
from PIL import Image

from mat_viewer import load_structure, render
from mat_viewer.loader import build_bundle_scene
from mat_viewer.render.contracts import CameraSpec, RenderSpec, ViewSpec
from mat_viewer.renderer import resolve_vector_overlays

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


@lru_cache(maxsize=128)
def _composition_rgba(path: Path) -> np.ndarray:
    """Decode immutable source art once for repeated animation composition."""
    return _rgba(path)


def make_vector_group(
    group_id: str,
    origins: np.ndarray,
    vectors: np.ndarray,
    *,
    scale: float,
    color: str,
    viewport_policy: str = "clip",
    tail_offset: float = 0.0,
    opacity: float = 1.0,
    style: dict | None = None,
) -> list[dict]:
    """Create one validated MatterVis native Cartesian vector group."""
    origins_array = np.asarray(origins, dtype=float)
    vectors_array = np.asarray(vectors, dtype=float)
    if origins_array.shape != vectors_array.shape or origins_array.ndim != 2:
        raise ValueError("origins and vectors must have matching shape (N, 3)")
    return [
        {
            "id": group_id,
            "name": group_id,
            "magnitude_mode": "scaled",
            "scale": float(scale),
            "viewport_policy": viewport_policy,
            "color": color,
            "opacity": float(opacity),
            "style": dict(style or {}),
            "arrows": [
                {
                    "id": f"{group_id}-{index}",
                    "origin": origin.tolist(),
                    "vector": vector.tolist(),
                    "tail_offset": float(tail_offset),
                }
                for index, (origin, vector) in enumerate(
                    zip(origins_array, vectors_array)
                )
            ],
        }
    ]


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
    vector_overlays: list[dict] | None = None,
) -> dict:
    """Render a complete structure/vector scene through public MatterVis APIs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    render_settings = {
        "width": int(width),
        "height": int(height),
        "atom_scale": float(atom_scale),
        "bond_radius": float(bond_radius),
        "show_cell": bool(show_cell),
    }
    resolved_vectors = resolve_vector_overlays(vector_overlays or [])
    vector_signature = json.dumps(
        [
            {
                "id": item["arrow_id"],
                "origin": np.asarray(item["origin"], dtype=float).tolist(),
                "vector": np.asarray(item["display_vector"], dtype=float).tolist(),
                "color": item["color"],
                "opacity": item["opacity"],
                "style": item["style"],
            }
            for item in resolved_vectors
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    sidecar = output.with_suffix(".json")
    if output.exists() and sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            previous.get("source_sha256") == source_hash
            and previous.get("frame") == frame
            and previous.get("view") == view
            and previous.get("camera") == asdict(camera)
            and previous.get("render_settings") == render_settings
            and previous.get("vector_signature") == vector_signature
            and previous.get("pipeline_version") == 3
            and int(np.count_nonzero(_rgba(output)[:, :, 3])) > 0
        ):
            return previous
    loaded = load_structure(source, frame=frame)
    native_vectors = [
        {
            "id": "story-vectors",
            "name": "story vectors",
            "visible": True,
            "magnitude_mode": "absolute",
            "scale": 1.0,
            "arrows": [
                {
                    "id": item["arrow_id"],
                    "origin": (
                        np.asarray(item["origin"], dtype=float)
                        + np.asarray(camera.scene_offset, dtype=float)
                    ).tolist(),
                    "vector": np.asarray(item["display_vector"], dtype=float).tolist(),
                    "color": item["color"],
                    "opacity": item["opacity"],
                    "style": item["style"],
                    "metadata": item["metadata"],
                    "visible": True,
                }
                for item in resolved_vectors
            ],
        }
    ] if resolved_vectors else []
    scene = build_bundle_scene(
        loaded.frames[0].bundle,
        display_mode=view,
        show_hydrogen=True,
    )
    scene["vector_overlays"] = native_vectors
    result = render(
        scene,
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
        "source_sha256": source_hash,
        "frame": frame,
        "view": view,
        "camera": asdict(camera),
        "render_settings": render_settings,
        "vector_signature": vector_signature,
        "vector_count": len(resolved_vectors),
        "vector_renderer": "MatterVis native world-space vector_overlays",
        "pipeline_version": 3,
        "warnings": list(result.warnings),
        "metadata": dict(result.metadata),
    }
    if int(np.count_nonzero(_rgba(output)[:, :, 3])) == 0:
        raise RuntimeError(f"MatterVis produced an empty transparent render: {output}")
    json_dump(sidecar, payload)
    return payload


def render_density_cube(
    source: Path,
    output: Path,
    *,
    camera: SceneCamera,
    isovalue: float,
    opacity: float = 0.38,
    positive_color: str = NAVY,
    width: int = 1500,
    height: int = 1120,
    atom_scale: float = 0.90,
    bond_radius: float = 0.105,
    vector_overlays: list[dict] | None = None,
) -> dict:
    """Render one complete Cube/structure/vector scene through MatterVis."""
    from mat_viewer.cube.cpu import cube_isosurface_meshes

    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    render_settings = {
        "width": int(width),
        "height": int(height),
        "atom_scale": float(atom_scale),
        "bond_radius": float(bond_radius),
        "isovalue": float(isovalue),
        "opacity": float(opacity),
        "positive_color": positive_color,
    }
    resolved_vectors = resolve_vector_overlays(vector_overlays or [])
    vector_signature = json.dumps(
        [
            {
                "id": item["arrow_id"],
                "origin": np.asarray(item["origin"], dtype=float).tolist(),
                "vector": np.asarray(item["display_vector"], dtype=float).tolist(),
                "color": item["color"],
                "opacity": item["opacity"],
                "style": item["style"],
            }
            for item in resolved_vectors
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    sidecar = output.with_suffix(".json")
    if output.exists() and sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            previous.get("source_sha256") == source_hash
            and previous.get("camera") == asdict(camera)
            and previous.get("render_settings") == render_settings
            and previous.get("vector_signature") == vector_signature
            and previous.get("pipeline_version") == 3
            and int(np.count_nonzero(_rgba(output)[:, :, 3])) > 0
        ):
            return previous
    loaded = load_structure(source, input_format="cube")
    bundle = loaded.frames[0].bundle
    offset = np.asarray(camera.scene_offset, dtype=float)
    meshes = cube_isosurface_meshes(
        bundle.cube_data,
        isovalue=isovalue,
        stride=1,
        positive_color=positive_color,
        negative_color=GREEN,
        opacity=opacity,
    )
    native_meshes = [
        {
            **mesh,
            "vertices": (np.asarray(mesh["vertices"], dtype=float) + offset).tolist(),
            "triangles": np.asarray(mesh["triangles"], dtype=np.int64).tolist(),
            "normals": np.asarray(mesh["normals"], dtype=float).tolist(),
        }
        for mesh in meshes
    ]
    native_vectors = [
        {
            "id": "story-vectors",
            "name": "story vectors",
            "visible": True,
            "magnitude_mode": "absolute",
            "scale": 1.0,
            "arrows": [
                {
                    "id": item["arrow_id"],
                    "origin": (
                        np.asarray(item["origin"], dtype=float) + offset
                    ).tolist(),
                    "vector": np.asarray(
                        item["display_vector"], dtype=float
                    ).tolist(),
                    "color": item["color"],
                    "opacity": item["opacity"],
                    "style": item["style"],
                    "metadata": item["metadata"],
                    "visible": True,
                }
                for item in resolved_vectors
            ],
        }
    ] if resolved_vectors else []
    scene = build_bundle_scene(
        bundle,
        display_mode="cluster",
        show_hydrogen=True,
    )
    scene["isosurfaces"] = native_meshes
    scene["vector_overlays"] = native_vectors
    result = render(
        scene,
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
            atom_scale=atom_scale,
            bond_radius=bond_radius,
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
        "source_sha256": source_hash,
        "camera": asdict(camera),
        "render_settings": render_settings,
        "vector_signature": vector_signature,
        "vector_count": len(resolved_vectors),
        "vector_renderer": "MatterVis native world-space vector_overlays",
        "isosurface_renderer": "MatterVis native Cube isosurface mesh",
        "isosurface_mesh_count": len(native_meshes),
        "pipeline_version": 3,
        "warnings": list(result.warnings),
        "metadata": dict(result.metadata),
    }
    if int(np.count_nonzero(_rgba(output)[:, :, 3])) == 0:
        raise RuntimeError(f"MatterVis produced an empty transparent render: {output}")
    json_dump(sidecar, payload)
    return payload


def place_render(
    ax: plt.Axes,
    image_path: Path,
    rect: tuple[float, float, float, float],
    *,
    alpha: float = 1.0,
    zorder: float = 4.0,
) -> tuple[float, float, float, float]:
    image = _composition_rgba(image_path)
    image_aspect = image.shape[1] / image.shape[0]
    figure_width, figure_height = ax.figure.canvas.get_width_height()
    position = ax.get_position()
    axes_aspect = (position.width * figure_width) / (position.height * figure_height)
    x0, y0, x1, y1 = rect
    width = x1 - x0
    height = y1 - y0
    required_normalized_ratio = image_aspect / axes_aspect
    if width / height > required_normalized_ratio:
        fitted_width = height * required_normalized_ratio
        centre = 0.5 * (x0 + x1)
        x0, x1 = centre - 0.5 * fitted_width, centre + 0.5 * fitted_width
    else:
        fitted_height = width / required_normalized_ratio
        centre = 0.5 * (y0 + y1)
        y0, y1 = centre - 0.5 * fitted_height, centre + 0.5 * fitted_height
    ax.imshow(
        image,
        extent=(x0, x1, y0, y1),
        origin="upper",
        interpolation="lanczos",
        alpha=float(alpha),
        zorder=zorder,
        aspect="auto",
    )
    return (x0, y0, x1, y1)


def place_render_blend(
    ax: plt.Axes,
    first_path: Path,
    second_path: Path,
    rect: tuple[float, float, float, float],
    *,
    blend: float,
    zorder: float = 4.0,
) -> tuple[float, float, float, float]:
    """Place one exact pixel blend without fading the shared scene to white."""
    weight = float(np.clip(blend, 0.0, 1.0))
    if first_path == second_path or weight <= 1.0e-6:
        return place_render(ax, first_path, rect, zorder=zorder)
    if weight >= 1.0 - 1.0e-6:
        return place_render(ax, second_path, rect, zorder=zorder)

    first = _composition_rgba(first_path)
    second = _composition_rgba(second_path)
    if first.shape != second.shape:
        raise ValueError(
            "Transition renders must have identical pixel dimensions: "
            f"{first.shape} != {second.shape}"
        )
    image = np.rint(
        (1.0 - weight) * first.astype(np.float32)
        + weight * second.astype(np.float32)
    ).astype(np.uint8)

    image_aspect = image.shape[1] / image.shape[0]
    figure_width, figure_height = ax.figure.canvas.get_width_height()
    position = ax.get_position()
    axes_aspect = (position.width * figure_width) / (position.height * figure_height)
    x0, y0, x1, y1 = rect
    width = x1 - x0
    height = y1 - y0
    required_normalized_ratio = image_aspect / axes_aspect
    if width / height > required_normalized_ratio:
        fitted_width = height * required_normalized_ratio
        centre = 0.5 * (x0 + x1)
        x0, x1 = centre - 0.5 * fitted_width, centre + 0.5 * fitted_width
    else:
        fitted_height = width / required_normalized_ratio
        centre = 0.5 * (y0 + y1)
        y0, y1 = centre - 0.5 * fitted_height, centre + 0.5 * fitted_height
    ax.imshow(
        image,
        extent=(x0, x1, y0, y1),
        origin="upper",
        interpolation="lanczos",
        zorder=zorder,
        aspect="auto",
    )
    return (x0, y0, x1, y1)


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
    linewidth = 7.5 if video else 2.8
    head = 34 if video else 12
    halo = FancyArrowPatch(
        tuple(xy[0]),
        tuple(xy[1]),
        arrowstyle="-|>",
        mutation_scale=head + (8 if video else 4),
        linewidth=linewidth + (6 if video else 2.5),
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
    image_aspect: float = 1500.0 / 1120.0,
) -> np.ndarray:
    xy = project_world(
        np.vstack((first, second)),
        camera=camera,
        rect=rect,
        image_aspect=image_aspect,
    )
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
    centre_text: str | None = None,
    centre_y: float = 0.64,
    radius_x: float = 0.31,
) -> None:
    """Draw the shared empty three-stage Velocity Verlet loop."""
    figure_width, figure_height = ax.figure.canvas.get_width_height()
    position = ax.get_position()
    axes_aspect = (position.width * figure_width) / (position.height * figure_height)
    centre_x = 0.50
    radius_y = radius_x * axes_aspect
    arc_ranges = ((-30, 90), (210, 330), (90, 210))
    ax.add_patch(Arc((centre_x, centre_y), 2 * radius_x, 2 * radius_y, theta1=0, theta2=360, color=LINE_GRAY, lw=4.0 if video else 2.4, zorder=1))
    if active_stage is not None:
        theta1, theta2 = arc_ranges[active_stage]
        ax.add_patch(Arc((centre_x, centre_y), 2 * radius_x, 2 * radius_y, theta1=theta1, theta2=theta2, color=INK, lw=5.0 if video else 3.0, zorder=2))
    tangent_angles = ((38, 24), (-82, -98), (-202, -218))
    for index, (start_angle, end_angle) in enumerate(tangent_angles):
        def point(angle: float) -> tuple[float, float]:
            radians = np.deg2rad(angle)
            return centre_x + radius_x * np.cos(radians), centre_y + radius_y * np.sin(radians)
        registry.arrow(
            ax,
            point(start_angle),
            point(end_angle),
            arrowstyle="-|>",
            mutation_scale=25 if video else 16,
            lw=4.5 if video else 2.6,
            color=INK if active_stage == index else LINE_GRAY,
            zorder=3,
        )
    nodes = [
        (centre_x, centre_y + radius_y, r"$\mathbf{r}$", "position"),
        (centre_x + radius_x * np.cos(np.deg2rad(-30)), centre_y + radius_y * np.sin(np.deg2rad(-30)), r"$\mathbf{a}$", "acceleration"),
        (centre_x + radius_x * np.cos(np.deg2rad(210)), centre_y + radius_y * np.sin(np.deg2rad(210)), r"$\mathbf{v}$", "velocity"),
    ]
    node_half_width = 0.075 if video else 0.062
    node_half_height = node_half_width * axes_aspect
    for index, (x, y, symbol, label) in enumerate(nodes):
        active = active_stage == index
        ax.add_patch(
            Ellipse(
                (x, y),
                width=2.0 * node_half_width,
                height=2.0 * node_half_height,
                fc=INK if active else WHITE,
                ec=INK if active else LINE_GRAY,
                lw=4.0 if video else 2.3,
                zorder=4,
            )
        )
        label_x = x
        if index == 1:
            label_x -= 0.018
        elif index == 2:
            label_x += 0.018
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
            label_x,
            y - node_half_height - (0.035 if video else 0.025),
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
        centre_y,
        centre_text or "Velocity\nVerlet",
        ha="center",
        va="center",
        fontsize=(18 if video else 10) if centre_text else (25 if video else 15),
        color=INK,
        weight="normal" if centre_text else "bold",
        linespacing=1.15 if centre_text else 1.0,
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
