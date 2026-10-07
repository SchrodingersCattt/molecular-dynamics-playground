"""01 Velocity Verlet with a pile of identical hard plastic balls.

Layout is shared with 03/04: velocity-Verlet loop (left), the real system
(centre) and one narrow vision column (right).  The right column holds three
Gaussian histograms, one per Cartesian velocity component.  The balls are in
place from the first frame (positions).  All balls then receive velocities at
once: every ball's three sampled numbers (one dot per histogram) fly to the
tip of that ball's velocity arrow in a vector star drawn over the histograms,
and the nine arrows travel together onto their balls.  One velocity-Verlet
step follows (a, v, r) and six more are cycled quickly.

Spheres are analytic MatterVis sphere meshes with a hard-plastic highlight;
velocity, acceleration and displacement arrows are MatterVis world-space
vector overlays.  Arrow images are switched, never cross-faded; only the
ball motion keeps a short ghost.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from common import (
    A_ORANGE,
    FONT_SIZES,
    INK,
    LINE_GRAY,
    R_BLUE,
    V_PURPLE,
    WHITE,
    LayoutRegistry,
    axes_from_top_slot,
    json_dump,
    new_static_figure,
    render_video,
    save_static,
    smoothstep,
)
from mattervis_story import (
    STORY_STATIC_A,
    STORY_STATIC_B,
    STORY_VIDEO_A,
    STORY_VIDEO_B,
    SceneCamera,
    camera_basis,
    draw_arrow_legend,
    draw_vv_loop,
    make_sphere_mesh,
    make_vector_group,
    place_render,
    place_render_blend,
    project_world,
    render_structure,
    write_provenance_index,
)

ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "01_velocity_verlet"
QA_DIR = ROOT / "qa" / STEM
ASSET_DIR = QA_DIR / "source" / "plastic_balls_v1"
DATA_PATH = ROOT / "data" / "vv_plastic_balls.npz"

IMAGE_SIZE = (1700, 1000)
STATIC_SCENE_RECT = (0.03, 0.08, 0.97, 0.88)
VIDEO_SCENE_RECT = (0.0, 0.07, 1.0, 0.875)
STATIC_R = (0.745, 0.045, 0.965, 0.955)
VIDEO_R = (0.695, 0.025, 0.985, 0.975)

RESIN = "#CBD7E2"

FAST_STEPS = 6
FAST_START = 13.2
DURATION = FAST_START + FAST_STEPS * 1.0
DISPLAY_RADIUS = 1.30  # display only; LJ sigma is 3.40 Angstrom
SIGMA_RANGE = 3.2
HIST_BINS = 16
GROWTH_FRAMES = 6

POSITION_EQUATION = (
    r"$\mathbf{r}_{n+1}=\mathbf{r}_n$" "\n" r"$+\mathbf{v}_{n+1/2}\Delta t$"
)
ACCELERATION_EQUATION = (
    r"$\mathbf{a}_{n}=\mathbf{F}_{n}/m$" "\n" r"$\mathbf{F}_{n}=-\nabla E(\mathbf{r}_n)$"
)
VELOCITY_EQUATION = (
    r"$\mathbf{v}_{n+1/2}=\mathbf{v}_{n}$" "\n" r"$+\frac{1}{2}\mathbf{a}_{n}\Delta t$"
)
SAMPLE_EQUATION = r"$\mathbf{v}=(v_x, v_y, v_z)$"
EQUATIONS = (POSITION_EQUATION, ACCELERATION_EQUATION, VELOCITY_EQUATION)

# (name, start, end, loop stage)
PHASES = (
    ("pos", 0.0, 1.6, 0),
    ("sample", 1.6, 3.0, 2),
    ("combine", 3.0, 4.4, 2),
    ("assign", 4.4, 5.8, 2),
    ("ready", 5.8, 6.6, 2),
    ("acc", 6.6, 8.8, 1),
    ("vel", 8.8, 11.0, 2),
    ("move", 11.0, FAST_START, 0),
    ("fast", FAST_START, DURATION, None),
)
ASSIGN_STAGGER = 0.035  # fraction of the assign phase between two arrivals
ASSIGN_TRAVEL = 0.62
TITLES = {
    "pos": "Initial positions",
    "sample": "Sample each velocity component from a Gaussian",
    "combine": r"Three components $\rightarrow$ one velocity arrow",
    "assign": "Give every ball its velocity",
    "ready": "Initial velocities",
    "acc": "Force \u2192 acceleration",
    "vel": "Update velocity",
    "move": "Update position",
}
FAST_TITLES = ("Force \u2192 acceleration", "Update velocity", "Update position")
LEGEND = {
    "v": (r"velocity $\mathbf{v}$", V_PURPLE),
    "a": (r"acceleration $\mathbf{a}$", A_ORANGE),
    "dv": (r"$\frac{1}{2}\mathbf{a}\Delta t$", A_ORANGE),
    "r": (r"displacement $\mathbf{v}\Delta t$", R_BLUE),
}


# ---------------------------------------------------------------------------
# data and display scales
# ---------------------------------------------------------------------------
def load_data() -> dict[str, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}; run scripts/build_box/generate_plastic_vv.py")
    with np.load(DATA_PATH) as archive:
        data = {key: archive[key] for key in archive.files}
    data["radius"] = DISPLAY_RADIUS
    data["dt"] = float(data["dt_fs"])
    data["sigma"] = float(data["sigma_t"])
    v0 = data["velocities"][0]
    # Display scales (Angstrom of arrow per Angstrom/fs): recorded in the manifest.
    data["scale_v"] = 1.45 / float(np.linalg.norm(v0, axis=1).mean())
    data["scale_a"] = 5.0 * data["scale_v"] * data["dt"] / 2.0
    data["scale_d"] = 1.55 / float(np.linalg.norm(data["half_velocities"][0] * data["dt"], axis=1).mean())
    return data


# ---------------------------------------------------------------------------
# MatterVis scenes
# ---------------------------------------------------------------------------
def build_camera(positions: np.ndarray, radius: float, reach: np.ndarray) -> SceneCamera:
    """Shared view direction; the roll about it is free (no floor, no gravity)
    and is chosen so the pile fills the wide panel."""
    right0, up0 = camera_basis(SceneCamera(target=(0.0, 0.0, 0.0), ortho_scale=1.0))
    flat = positions.reshape(-1, 3)
    aspect = IMAGE_SIZE[0] / IMAGE_SIZE[1]
    best = None
    for degrees in range(0, 180, 5):
        angle = np.deg2rad(degrees)
        up_hint = np.cos(angle) * up0 + np.sin(angle) * right0
        probe = SceneCamera(target=(0.0, 0.0, 0.0), ortho_scale=1.0, up=tuple(float(x) for x in up_hint))
        right, up = camera_basis(probe)
        cloud = np.vstack([flat + radius * right, flat - radius * right, flat + radius * up, flat - radius * up, reach])
        sx, sy = cloud @ right, cloud @ up
        half_h = 0.5 * (sy.max() - sy.min()) * 1.03
        half_w = 0.5 * (sx.max() - sx.min()) * 1.03
        ortho = max(half_h, half_w / aspect)
        if best is None or ortho < best[0] - 1.0e-9:
            centre = 0.5 * (sx.min() + sx.max()) * right + 0.5 * (sy.min() + sy.max()) * up
            best = (ortho, centre, probe.up)
    ortho, target, up_hint = best
    return SceneCamera(target=tuple(float(x) for x in target), ortho_scale=float(ortho), up=up_hint)


def write_source(path: Path, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(positions)), 'Properties=species:S:1:pos:R:3 source="01 plastic balls" pbc="F F F"']
    for xyz in positions:
        lines.append(f"Ar {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ball_meshes(positions: np.ndarray, radius: float) -> list[dict]:
    meshes = []
    for index, centre in enumerate(positions):
        mesh = make_sphere_mesh(
            centre, radius, color=RESIN, opacity=1.0, lat_steps=8, lon_steps=16, mesh_id=f"ball-{index:02d}"
        )
        mesh["metadata"].update(
            {
                "_raster_light": (-0.38, 0.50, 1.0),
                "_raster_ambient": 0.60,
                "_raster_diffuse": 0.40,
                "_raster_specular": 0.55,
                "_raster_shininess": 52.0,
                "_raster_alpha_front_factor": 1.0,
                "_raster_alpha_back_factor": 1.0,
                "_raster_two_sided": False,
            }
        )
        mesh["phase"] = "ball"
        meshes.append(mesh)
    return meshes


SHAFT_RADIUS = 0.06


def arrow_style(length: float) -> dict:
    head = float(min(0.36, 0.55 * length))
    return {
        "shaft_radius": SHAFT_RADIUS,
        "head_length": head,
        "head_radius": float(max(0.07, min(0.16, 0.42 * head + 0.02))),
        "sides": 16,
    }


def vector_groups(
    group_id: str,
    origins: np.ndarray,
    vectors: np.ndarray,
    *,
    scale: float,
    color: str,
    tail: float,
) -> list[dict]:
    vectors = np.asarray(vectors, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1) * scale
    keep = lengths > 0.10
    if not np.any(keep):
        return []
    group = make_vector_group(
        group_id, np.asarray(origins)[keep], vectors[keep], scale=scale, color=color, tail_offset=tail
    )
    for arrow, length in zip(group[0]["arrows"], lengths[keep]):
        arrow["style"] = arrow_style(float(length))
    return group


class SceneBuilder:
    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self.data = data
        self.radius = float(data["radius"])
        self.positions = np.asarray(data["positions"], dtype=float)
        reach = []
        for k in range(self.positions.shape[0]):
            vel = data["velocities"][k]
            unit = vel / np.linalg.norm(vel, axis=1, keepdims=True)
            reach.append(self.positions[k] + (self.radius + data["scale_v"] * np.linalg.norm(vel, axis=1, keepdims=True)) * unit)
        self.camera = build_camera(self.positions, self.radius, np.vstack(reach))
        right, up = camera_basis(self.camera)
        r0 = self.positions[0]
        # Arrivals go first to balls that no nearer ball overlaps on screen,
        # each group top-down, so the cumulative arrow images stay readable.
        screen = np.column_stack((r0 @ right, r0 @ up))
        depth = r0 @ np.asarray(self.camera.direction, dtype=float)
        occluded = [
            any(
                depth[j] > depth[i] and np.linalg.norm(screen[j] - screen[i]) < 2.0 * self.radius
                for j in range(len(r0))
                if j != i
            )
            for i in range(len(r0))
        ]
        self.assign_order = sorted(range(len(r0)), key=lambda i: (occluded[i], -screen[i, 1]))
        self.records: list[dict] = []

    def render(self, name: str, positions: np.ndarray, groups: list[dict] | None = None) -> Path:
        out = ASSET_DIR / f"{name}.png"
        source = ASSET_DIR / "sources" / f"{name}.extxyz"
        write_source(source, positions)
        record = render_structure(
            source,
            out,
            camera=self.camera,
            frame=0,
            view="cluster",
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            atom_scale=1.0,
            bond_radius=0.1,
            show_bonds=False,
            atom_opacity_scales={i: 0.0 for i in range(len(positions))},
            mesh_overlays=ball_meshes(positions, self.radius),
            vector_overlays=[g for item in (groups or []) for g in ([item] if isinstance(item, dict) else item)],
        )
        record["name"] = name
        self.records.append(record)
        return out


def prepare_assets(data: dict[str, np.ndarray]) -> tuple[dict[str, object], SceneBuilder]:
    """Render every MatterVis scene used by the movie (cached on disk)."""
    builder = SceneBuilder(data)
    data["assign_order"] = builder.assign_order
    positions = builder.positions
    radius = builder.radius
    sv, sa, sd = data["scale_v"], data["scale_a"], data["scale_d"]
    r0, r1 = positions[0], positions[1]
    v0 = data["velocities"][0]
    growth = [(k + 1) / GROWTH_FRAMES for k in range(GROWTH_FRAMES)]
    assets: dict[str, object] = {}

    assets["plain"] = [builder.render(f"plain_{k}", positions[k]) for k in range(FAST_STEPS + 2)]
    assets["move"] = [
        builder.render(f"move_{k}", r0 + (k / 4.0) * (r1 - r0)) for k in (1, 2, 3)
    ]

    def velocity_scene(name: str, vectors: np.ndarray, at: np.ndarray = r0) -> Path:
        return builder.render(name, at, vector_groups(name, at, vectors, scale=sv, color=V_PURPLE, tail=radius))

    assign = builder.assign_order
    assets["assign"] = [assets["plain"][0]]
    for n in range(1, len(assign) + 1):
        partial = np.zeros_like(v0)
        partial[assign[:n]] = v0[assign[:n]]
        assets["assign"].append(velocity_scene(f"assign_{n}", partial))
    accel = data["accelerations"][0]
    assets["acc"] = [
        builder.render(f"acc_{k}", r0, vector_groups(f"acc_{k}", r0, accel * g, scale=sa, color=A_ORANGE, tail=radius))
        for k, g in enumerate(growth, start=1)
    ]
    # velocity stage: old arrows plus Delta v placed tip to tail, then the sum
    accel_half = 0.5 * accel * data["dt"]
    directions = v0 / np.linalg.norm(v0, axis=1, keepdims=True)
    tips = r0 + (radius + sv * np.linalg.norm(v0, axis=1, keepdims=True)) * directions
    add_groups = vector_groups("vadd-v", r0, v0, scale=sv, color=V_PURPLE, tail=radius) + vector_groups(
        "vadd-dv", tips, accel_half, scale=sv, color=A_ORANGE, tail=0.0
    )
    assets["vadd"] = builder.render("vadd", r0, add_groups)
    assets["vhalf"] = velocity_scene("vhalf", data["half_velocities"][0])
    displacement = data["half_velocities"][0] * data["dt"]
    assets["disp"] = [
        builder.render(f"disp_{k}", r0, vector_groups(f"disp_{k}", r0, displacement * g, scale=sd, color=R_BLUE, tail=radius))
        for k, g in enumerate(growth, start=1)
    ]
    assets["fast_acc"] = [
        builder.render(
            f"facc_{k}",
            positions[k],
            vector_groups(f"facc_{k}", positions[k], data["accelerations"][k], scale=sa, color=A_ORANGE, tail=radius),
        )
        for k in range(1, FAST_STEPS + 1)
    ]
    assets["fast_half"] = [
        velocity_scene(f"fhalf_{k}", data["half_velocities"][k], at=positions[k]) for k in range(1, FAST_STEPS + 1)
    ]
    assets["fast_disp"] = [
        builder.render(
            f"fdisp_{k}",
            positions[k],
            vector_groups(
                f"fdisp_{k}", positions[k], data["half_velocities"][k] * data["dt"], scale=sd, color=R_BLUE, tail=radius
            ),
        )
        for k in range(1, FAST_STEPS + 1)
    ]
    write_provenance_index(ASSET_DIR, builder.records)
    return assets, builder


# ---------------------------------------------------------------------------
# state of the movie
# ---------------------------------------------------------------------------
def phase_at(t: float) -> tuple[str, float, int | None]:
    t = float(np.clip(t, 0.0, DURATION - 1.0e-9))
    for name, start, end, stage in PHASES:
        if t < end:
            return name, (t - start) / (end - start), stage
    name, start, end, stage = PHASES[-1]
    return name, 1.0, stage


def lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    return a + (b - a) * float(np.clip(u, 0.0, 1.0))


def fast_cell(u: float) -> tuple[int, float]:
    """Cell index (0-based, state k = index + 1) and progress inside the cell."""
    position = u * FAST_STEPS
    cell = min(int(position), FAST_STEPS - 1)
    return cell, position - cell


def ramp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def assign_travel(u: float, n_balls: int) -> np.ndarray:
    """Travel progress (0..1) of each assignment slot during the assign phase."""
    return np.asarray([ramp((u - 0.04 - ASSIGN_STAGGER * slot) / ASSIGN_TRAVEL) for slot in range(n_balls)])


def movie_state(t: float, data: dict[str, np.ndarray]) -> dict:
    """Everything the compositor needs for one time stamp."""
    name, u, stage = phase_at(t)
    half = data["half_velocities"]
    full = data["velocities"]
    positions = data["positions"]
    n_balls = positions.shape[1]
    state = {
        "name": name,
        "u": u,
        "stage": stage,
        "title": TITLES.get(name, ""),
        "equation": EQUATIONS[stage] if stage is not None else None,
        "step": 0,
        "dots": full[0],
        "dot_alpha": 1.0,
        "hist_veil": 0.0,
        "positions": positions[0],
        "position_active": name in {"pos", "move"},
        "fast_cell": None,
        "dot_flight": 0.0,
        "star_grow": 0.0,
        "travel": None,
        "legend": (),
    }
    if name == "pos":
        state["dots"] = None
        state["hist_veil"] = 0.68
    elif name == "sample":
        state["equation"] = SAMPLE_EQUATION
        state["dot_alpha"] = smoothstep(u / 0.45)
    elif name == "combine":
        state["equation"] = SAMPLE_EQUATION
        state["hist_veil"] = 0.80 * smoothstep(u / 0.25)
        state["dot_flight"] = smoothstep(ramp((u - 0.10) / 0.55))
        state["star_grow"] = smoothstep(ramp((u - 0.45) / 0.40))
        state["legend"] = ("v",) if u > 0.45 else ()
    elif name == "assign":
        state["equation"] = SAMPLE_EQUATION
        state["hist_veil"] = 0.80 * (1.0 - smoothstep((u - 0.70) / 0.30))
        state["travel"] = assign_travel(u, n_balls)
        state["legend"] = ("v",)
    elif name == "ready":
        state["legend"] = ("v",)
    elif name == "acc":
        state["hist_veil"] = 0.68
        state["legend"] = ("a",) if u >= 0.10 else ("v",)
    elif name == "vel":
        state["dots"] = lerp(full[0], half[0], smoothstep((u - 0.60) / 0.25))
        state["legend"] = ("a",) if u < 0.20 else ("v", "dv") if u < 0.60 else ("v",)
    elif name == "move":
        q = smoothstep(ramp((u - 0.55) / 0.35))
        state["positions"] = lerp(positions[0], positions[1], q)
        state["dots"] = half[0]
        state["hist_veil"] = 0.68
        state["legend"] = ("v",) if u < 0.10 else ("r",) if u < 0.55 else ()
    else:  # fast
        cell, local = fast_cell(u)
        k = cell + 1
        sub = int(min(local * 3.0, 2.999))
        inner = local * 3.0 - sub
        state["fast_cell"] = (cell, local)
        state["stage"] = (1, 2, 0)[sub]
        state["equation"] = EQUATIONS[state["stage"]]
        state["title"] = FAST_TITLES[sub]
        state["step"] = k
        move_q = smoothstep(ramp((inner - 0.45) / 0.45)) if sub == 2 else 0.0
        state["positions"] = lerp(positions[k], positions[k + 1], move_q)
        if sub == 0:
            dots = full[k]
        elif sub == 1:
            dots = lerp(full[k], half[k], smoothstep(inner / 0.6))
        else:
            dots = half[k]
        state["dots"] = dots
        state["hist_veil"] = 0.68 if sub != 1 else 0.0
        state["position_active"] = sub == 2
        state["legend"] = (("a",), ("v",), ("r",) if inner < 0.45 else ())[sub]
    if state["dots"] is not None:
        state["dots"] = np.asarray(state["dots"]) / data["sigma"]
    return state


def image_for_state(state: dict, assets: dict[str, object]) -> tuple[list[Path], float, bool]:
    """Return an ordered image list, the position along it, and whether to blend.

    Arrow images are always picked, never blended; only frames that move the
    balls without arrows are blended.
    """
    name, u = state["name"], state["u"]
    plain = assets["plain"]
    if name in {"pos", "sample", "combine"}:
        return [plain[0]], 0.0, False
    if name == "assign":
        arrived = int(np.sum(np.asarray(state["travel"]) >= 1.0))
        return [assets["assign"][arrived]], 0.0, False
    if name == "ready":
        return [assets["assign"][-1]], 0.0, False
    if name == "acc":
        if u < 0.10:
            return [assets["assign"][-1]], 0.0, False
        return assets["acc"], ramp((u - 0.10) / 0.35) * (GROWTH_FRAMES - 1), False
    if name == "vel":
        if u < 0.20:
            return [assets["acc"][-1]], 0.0, False
        if u < 0.60:
            return [assets["vadd"]], 0.0, False
        return [assets["vhalf"]], 0.0, False
    if name == "move":
        if u < 0.10:
            return [assets["vhalf"]], 0.0, False
        if u < 0.55:
            return assets["disp"], ramp((u - 0.10) / 0.25) * (GROWTH_FRAMES - 1), False
        seq = [plain[0], *assets["move"], plain[1]]
        return seq, smoothstep(ramp((u - 0.55) / 0.35)) * (len(seq) - 1), True
    # fast: a_k -> v_(k+1/2) -> dr, then the balls move to r_(k+1)
    cell, local = state["fast_cell"]
    third = min(int(local * 3.0), 2)
    inner = local * 3.0 - third
    if third == 0:
        return [assets["fast_acc"][cell]], 0.0, False
    if third == 1:
        return [assets["fast_half"][cell]], 0.0, False
    if inner < 0.45:
        return [assets["fast_disp"][cell]], 0.0, False
    return [plain[cell + 1], plain[cell + 2]], smoothstep(ramp((inner - 0.45) / 0.45)), True


def place_sequence(ax: plt.Axes, images: list[Path], position: float, rect, *, blend: bool, zorder: float = 5.0):
    if len(images) == 1:
        return place_render(ax, images[0], rect, zorder=zorder)
    position = float(np.clip(position, 0.0, len(images) - 1))
    if not blend:
        return place_render(ax, images[int(round(position))], rect, zorder=zorder)
    index = min(int(position), len(images) - 2)
    return place_render_blend(ax, images[index], images[index + 1], rect, blend=position - index, zorder=zorder)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def draw_left(ax: plt.Axes, registry: LayoutRegistry, *, video: bool, state: dict) -> None:
    draw_vv_loop(
        ax, registry, video=video, active_stage=int(state["stage"]), centre_text=state["equation"], centre_y=0.58, radius_x=0.36
    )


def _veil(ax: plt.Axes, alpha: float) -> None:
    ax.add_patch(
        Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax.transAxes, facecolor=WHITE, edgecolor="none", alpha=alpha, zorder=30)
    )


def hist_geometry() -> list[dict[str, float]]:
    blocks = []
    for k in range(3):
        top = 0.945 - 0.315 * k
        blocks.append({"top": top, "base": top - 0.225, "height": 0.150})
    return blocks


def sigma_to_x(value: np.ndarray | float) -> np.ndarray | float:
    return 0.14 + 0.80 * (np.asarray(value) + SIGMA_RANGE) / (2.0 * SIGMA_RANGE)


def dot_position(block: dict[str, float], value: float, stack: int) -> tuple[float, float]:
    centres = np.linspace(-SIGMA_RANGE, SIGMA_RANGE, HIST_BINS + 1)
    middle = 0.5 * (centres[:-1] + centres[1:])
    index = int(np.clip(np.searchsorted(centres, value, side="right") - 1, 0, HIST_BINS - 1))
    bar = block["height"] * float(np.exp(-0.5 * middle[index] ** 2))
    y = block["base"] + max(0.5 * bar, 0.016) + 0.018 * stack
    return float(sigma_to_x(np.clip(value, -SIGMA_RANGE + 0.05, SIGMA_RANGE - 0.05))), float(y)


def dot_layout(values: np.ndarray, order: list[int]) -> list[list[tuple[float, float]]]:
    """Positions (axes coords) of the dots in each of the 3 histograms, indexed by ball."""
    blocks = hist_geometry()
    layout = []
    edges = np.linspace(-SIGMA_RANGE, SIGMA_RANGE, HIST_BINS + 1)
    for axis in range(3):
        counts: dict[int, int] = {}
        points: list[tuple[float, float]] = [(0.0, 0.0)] * values.shape[0]
        for ball in order:
            value = float(values[ball, axis])
            index = int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, HIST_BINS - 1))
            stack = counts.get(index, 0)
            counts[index] = stack + 1
            points[ball] = dot_position(blocks[axis], value, stack)
        layout.append(points)
    return layout


def draw_histograms(
    ax: plt.Axes,
    registry: LayoutRegistry,
    state: dict,
    order: list[int],
    *,
    video: bool,
) -> list[list[tuple[float, float]]] | None:
    if not video:
        ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, ec=LINE_GRAY, lw=1.1, zorder=20))
    blocks = hist_geometry()
    edges = np.linspace(-SIGMA_RANGE, SIGMA_RANGE, HIST_BINS + 1)
    middle = 0.5 * (edges[:-1] + edges[1:])
    x_edges = sigma_to_x(edges)
    for block, label in zip(blocks, (r"$v_x$", r"$v_y$", r"$v_z$")):
        for index in range(HIST_BINS):
            height = block["height"] * float(np.exp(-0.5 * middle[index] ** 2))
            ax.add_patch(
                Rectangle(
                    (x_edges[index], block["base"]),
                    x_edges[index + 1] - x_edges[index],
                    height,
                    facecolor="#D6DEE5",
                    edgecolor=WHITE,
                    linewidth=1.0 if video else 0.6,
                    zorder=2,
                )
            )
        ax.plot([0.11, 0.97], [block["base"]] * 2, color=INK, lw=1.4 if video else 1.0, zorder=3)
        zero = float(sigma_to_x(0.0))
        ax.plot([zero, zero], [block["base"] - 0.012, block["base"]], color=INK, lw=1.4 if video else 1.0, zorder=3)
        registry.text(ax, 0.02, block["base"] + 0.5 * block["height"], label, ha="left", va="center", fontsize=FONT_SIZES["body"], color=V_PURPLE, zorder=6)
    layout = None
    if state["dots"] is not None:
        layout = dot_layout(np.asarray(state["dots"]), order)
        alpha = float(np.clip(state["dot_alpha"], 0.0, 1.0))
        size = 10.5 if video else 6.5
        if alpha > 0.0:
            for points in layout:
                xs, ys = zip(*points)
                ax.plot(
                    xs, ys, linestyle="none", marker="o", markersize=size,
                    markerfacecolor=V_PURPLE, markeredgecolor=WHITE,
                    markeredgewidth=1.0 if video else 0.6, alpha=alpha, zorder=8,
                )
    if state["hist_veil"] > 0.0:
        _veil(ax, float(state["hist_veil"]))
    return layout


def velocity_arrows_xy(builder: SceneBuilder, rect, data: dict, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projected tail/tip (scene axes coords) of the initial-velocity arrows."""
    v0 = data["velocities"][0]
    unit = v0 / np.linalg.norm(v0, axis=1, keepdims=True)
    tails = positions + builder.radius * unit
    tips = tails + data["scale_v"] * v0
    aspect = IMAGE_SIZE[0] / IMAGE_SIZE[1]
    return (
        project_world(tails, camera=builder.camera, rect=rect, image_aspect=aspect),
        project_world(tips, camera=builder.camera, rect=rect, image_aspect=aspect),
    )


def paper_arrow(fig: plt.Figure, tail, tip, *, lw: float, head: float, alpha: float = 1.0) -> None:
    fig.add_artist(
        FancyArrowPatch(
            tuple(tail), tuple(tip), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=head,
            lw=lw, color=V_PURPLE, alpha=alpha, shrinkA=0.0, shrinkB=0.0, zorder=60,
        )
    )


def draw_velocity_star(
    fig: plt.Figure,
    ax_hist: plt.Axes,
    ax_scene: plt.Axes,
    builder: SceneBuilder,
    fitted,
    data: dict,
    state: dict,
    layout: list[list[tuple[float, float]]] | None,
    *,
    video: bool,
) -> None:
    """Combine the three dots of every ball into one arrow, then fly it onto the ball."""
    name = state["name"]
    if name not in {"combine", "assign"}:
        return
    to_fig = fig.transFigure.inverted().transform
    tails, tips = velocity_arrows_xy(builder, fitted, data, data["positions"][0])
    tails_fig = to_fig(ax_scene.transData.transform(tails))
    tips_fig = to_fig(ax_scene.transData.transform(tips))
    vectors = tips_fig - tails_fig
    centre = to_fig(ax_hist.transAxes.transform((0.52, 0.50)))
    lw, head = (3.4, 15.0) if video else (1.6, 8.0)
    if name == "combine":
        grow = float(state["star_grow"])
        flight = float(state["dot_flight"])
        if grow > 0.0:
            for vector in vectors:
                paper_arrow(fig, centre, centre + max(grow, 0.02) * vector, lw=lw, head=head * min(1.0, 0.4 + grow))
        if layout is not None and 0.0 < flight < 1.0:
            size = (10.5 if video else 6.5) * (1.0 - 0.55 * flight)
            xs, ys = [], []
            for ball, vector in enumerate(vectors):
                target = centre + vector
                for points in layout:
                    start = to_fig(ax_hist.transData.transform(points[ball]))
                    xs.append(start[0] + (target[0] - start[0]) * flight)
                    ys.append(start[1] + (target[1] - start[1]) * flight)
            fig.add_artist(
                Line2D(
                    xs, ys, transform=fig.transFigure, linestyle="none", marker="o", markersize=size,
                    markerfacecolor=V_PURPLE, markeredgecolor=WHITE, markeredgewidth=0.8, zorder=61,
                )
            )
        return
    order = data["assign_order"]
    travel = np.asarray(state["travel"])
    for slot, ball in enumerate(order):
        progress = float(travel[slot])
        if progress >= 1.0:
            continue
        eased = smoothstep(progress)
        tail = centre + (tails_fig[ball] - centre) * eased
        paper_arrow(fig, tail, tail + vectors[ball], lw=lw, head=head)


def draw_scene_overlay(ax: plt.Axes, builder: SceneBuilder, rect, state: dict, *, video: bool) -> None:
    """Dark centre markers while the position stage is active."""
    if not state["position_active"]:
        return
    aspect = IMAGE_SIZE[0] / IMAGE_SIZE[1]
    xy = project_world(np.asarray(state["positions"], dtype=float), camera=builder.camera, rect=rect, image_aspect=aspect)
    ax.plot(xy[:, 0], xy[:, 1], linestyle="none", marker="o", markersize=5.5 if video else 3.4, color=R_BLUE, zorder=9)


def draw_scene(
    fig: plt.Figure,
    ax_scene: plt.Axes,
    ax_hist: plt.Axes,
    registry: LayoutRegistry,
    builder: SceneBuilder,
    assets: dict[str, object],
    data: dict,
    state: dict,
    layout: list[list[tuple[float, float]]] | None,
    *,
    video: bool,
) -> list[dict]:
    if not video:
        ax_scene.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, ec=LINE_GRAY, lw=1.1, zorder=20))
    scene_rect = VIDEO_SCENE_RECT if video else STATIC_SCENE_RECT
    images, position, blend = image_for_state(state, assets)
    name = state["name"]
    ghost = None
    if name == "move" and state["u"] > 0.55:
        ghost = 0.30 * smoothstep((state["u"] - 0.55) / 0.15)
    if name == "fast" and state["fast_cell"] is not None:
        cell, local = state["fast_cell"]
        if local > 2.0 / 3.0 + 0.45 / 3.0:
            ghost = 0.25
            ghost_image = assets["plain"][cell + 1]
    fitted = place_sequence(ax_scene, images, position, scene_rect, blend=blend, zorder=5.0)
    if ghost:
        place_render(ax_scene, assets["plain"][0] if name == "move" else ghost_image, scene_rect, alpha=float(ghost), zorder=4.0)
    draw_scene_overlay(ax_scene, builder, fitted, state, video=video)
    registry.text(
        ax_scene, 0.50, 0.965, state["title"], ha="center", va="top",
        fontsize=FONT_SIZES["panel_title"], color=INK, zorder=21,
    )
    registry.text(
        ax_scene, 0.0 if video else 0.075, 0.005 if video else 0.070, f"Simulation step {state['step'] + 1:02d}", ha="left", va="bottom",
        fontsize=FONT_SIZES["micro"], color=INK, zorder=21,
    )
    draw_arrow_legend(
        ax_scene, registry, [LEGEND[key] for key in state["legend"]],
        x_right=1.0 if video else 0.925, y=0.005 if video else 0.070, video=video,
    )
    draw_velocity_star(fig, ax_hist, ax_scene, builder, fitted, data, state, layout, video=video)
    semantics: list[dict] = []
    if name == "ready":
        semantics = [{"id": "velocity", "color": V_PURPLE, "min_pixels": 60, "tolerance": 70}]
    elif name == "acc" and state["u"] > 0.5:
        semantics = [{"id": "acceleration", "color": A_ORANGE, "min_pixels": 60, "tolerance": 70}]
    elif name == "move" and 0.37 < state["u"] < 0.55:
        semantics = [{"id": "displacement", "color": R_BLUE, "min_pixels": 60, "tolerance": 70}]
    return semantics


def compose(fig: plt.Figure, t: float, registry: LayoutRegistry, assets, builder, data, *, video: bool) -> list[dict]:
    state = movie_state(t, data)
    slots = (
        (STORY_VIDEO_A, STORY_VIDEO_B, VIDEO_R) if video else (STORY_STATIC_A, STORY_STATIC_B, STATIC_R)
    )
    ax_a = axes_from_top_slot(fig, slots[0])
    ax_b = axes_from_top_slot(fig, slots[1])
    ax_r = axes_from_top_slot(fig, slots[2])
    draw_left(ax_a, registry, video=video, state=state)
    layout = draw_histograms(ax_r, registry, state, builder.assign_order, video=video)
    return draw_scene(fig, ax_b, ax_r, registry, builder, assets, data, state, layout, video=video)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
STATIC_TIME = 6.2
KEYFRAME_TIMES = [
    0.8, 2.0, 2.6, 3.5, 3.9, 4.3, 4.9, 5.3, 6.2, 7.2, 8.4, 9.3,
    10.2, 11.0, 11.6, 12.6, 13.5, 13.9, 14.4, 14.9, 18.9,
]


def render_static(assets, builder, data) -> None:
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=18)
    compose(fig, STATIC_TIME, registry, assets, builder, data, video=False)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)


def video_registry() -> LayoutRegistry:
    return LayoutRegistry(
        min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=12,
        font_family="Arial", coerce_min_font=True,
    )


def render_keyframes(assets, builder, data) -> Path:
    from PIL import Image

    from common import new_video_figure

    out_dir = QA_DIR / "_qa" / "keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):
        old.unlink()
    thumbs = []
    for index, t in enumerate(KEYFRAME_TIMES):
        fig = new_video_figure()
        registry = video_registry()
        compose(fig, t, registry, assets, builder, data, video=True)
        errors = registry.validate(fig)
        path = out_dir / f"frame_{index:02d}_{t:05.2f}s.png"
        fig.savefig(path, dpi=100, facecolor=WHITE)
        plt.close(fig)
        if errors:
            raise RuntimeError(f"Keyframe {t:.2f}s failed layout:\n" + "\n".join(errors))
        thumbs.append(Image.open(path).convert("RGB").resize((640, 200)))
    columns = 3
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * 640, rows * 200), WHITE)
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 640, (index // columns) * 200))
    contact = out_dir / "_contact.png"
    sheet.save(contact)
    return contact


def render_animation(assets, builder, data) -> None:
    panels = [
        {"id": "integrator", "rect": list(STORY_VIDEO_A), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
        {"id": "balls", "rect": list(STORY_VIDEO_B), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
        {"id": "histograms", "rect": list(VIDEO_R), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
    ]
    audit = {
        "panels": panels,
        "whitespace": {
            "background_threshold": 245,
            "min_ink_fraction": 0.020,
            "min_panel_bbox_fill": 0.20,
            "grid_rows": 12,
            "grid_columns": 20,
        },
        "bands": [
            {"id": "gap_a_b", "rect": [0.215, 0.025, 0.230, 0.975], "max_ink_pixels": 5000},
            {"id": "gap_b_r", "rect": [0.680, 0.025, 0.695, 0.975], "max_ink_pixels": 5000},
        ],
    }
    render_video(
        stem=STEM,
        duration_seconds=DURATION,
        draw_frame=lambda fig, t, i, registry: compose(fig, t, registry, assets, builder, data, video=True),
        audit_config=audit,
        qa_directory=QA_DIR / "_qa",
        representative_times=KEYFRAME_TIMES,
    )


def render_animation_fast(assets, builder, data) -> None:
    """Export every frame through the same compositor, skipping pixel audits."""
    import os
    import subprocess
    import time

    from common import VIDEO_HEIGHT_PX, VIDEO_WIDTH_PX, new_video_figure

    output = ROOT / "videos" / f"{STEM}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"_{STEM}.{os.getpid()}.{time.time_ns()}.encoding.mp4")
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{VIDEO_WIDTH_PX}x{VIDEO_HEIGHT_PX}", "-r", "24", "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    figure = new_video_figure()
    try:
        for index in range(int(round(DURATION * 24))):
            figure.clear()
            compose(figure, index / 24.0, video_registry(), assets, builder, data, video=True)
            figure.canvas.draw()
            rgb = np.ascontiguousarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
            process.stdin.write(rgb.tobytes())
        process.stdin.close()
        process.stdin = None
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
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


def write_manifest(data, builder) -> None:
    json_dump(
        QA_DIR / "asset_manifest.json",
        {
            "stem": STEM,
            "data": str(DATA_PATH),
            "n_balls": int(data["positions"].shape[1]),
            "ball_radius_ang": float(data["radius"]),
            "ball_material": "MatterVis analytic sphere mesh, hard-plastic highlight",
            "dt_fs": float(data["dt"]),
            "arrow_scales": {
                "velocity_ang_per_ang_fs": float(data["scale_v"]),
                "acceleration_stage_ang_per_ang_fs2": float(data["scale_a"]),
                "acceleration_stage_note": "a-stage arrows show 5x (a dt/2) in velocity display units; the v-stage tip-to-tail arrows use the true velocity scale",
                "displacement_ang_per_ang": float(data["scale_d"]),
            },
            "arrow_colours": {"position": R_BLUE, "velocity": V_PURPLE, "acceleration": A_ORANGE},
            "arrow_shaft_radius_ang": SHAFT_RADIUS,
            "camera": {"target": list(builder.camera.target), "ortho_scale": builder.camera.ortho_scale},
            "image_size": list(IMAGE_SIZE),
            "timeline_seconds": {name: [start, end] for name, start, end, _ in PHASES},
            "velocity_recipe": "LAMMPS velocity create dist gaussian mom yes (shown values are the final, momentum-free, temperature-scaled velocities)",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--strict-video", action="store_true")
    args = parser.parse_args()
    data = load_data()
    assets, builder = prepare_assets(data)
    write_manifest(data, builder)
    if args.assets_only:
        return
    if args.preview_only:
        print(render_keyframes(assets, builder, data))
        return
    render_static(assets, builder, data)
    if not args.static_only:
        if args.strict_video:
            render_animation(assets, builder, data)
        else:
            render_animation_fast(assets, builder, data)


if __name__ == "__main__":
    main()
