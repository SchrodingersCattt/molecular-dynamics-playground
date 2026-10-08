"""02b: schematic ReaxFF dynamics of TNT C2--NO2 dissociation (same slots as 03b).

Left   shared velocity-Verlet loop (r -> a -> v).
Centre the six atoms.  The breaking Cl-O(H) bond fades with its bond order;
rings mark under-coordinated atoms, an arc marks the weighted valence angle,
delta+/delta- mark the equilibrated charges (all energy-term cues are teal).
Upper right  bond order versus distance, with one dot per bond.
Lower right  the explicit energy expressions of the schematic model
(BO(r), E_bond, E_over, E_angle, E_Coul, F); the active one is highlighted.

Data come from ``generate_reaxff_tnt.py`` (schematic ReaxFF-style energy,
autograd forces, velocity Verlet on the 03b TNT geometry and kick).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

from common import (
    A_ORANGE,
    DARK_GRAY,
    ENERGY_TEAL,
    FONT_SIZES,
    INK,
    LINE_GRAY,
    V_PURPLE,
    VIDEO_HEIGHT_PX,
    VIDEO_WIDTH_PX,
    WHITE,
    LayoutRegistry,
    axes_from_top_slot,
    json_dump,
    mix_hex,
    new_static_figure,
    new_video_figure,
    render_video,
    save_static,
    smoothstep,
)
from mattervis_story import (
    STORY_STATIC_A,
    STORY_VIDEO_A,
    STORY_VIDEO_B,
    SceneCamera,
    camera_basis,
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build_box"))
from generate_reaxff_tnt import BONDS as TNT_BONDS, BREAKING_BOND_INDEX  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "02b_reaxff"
QA_DIR = ROOT / "qa" / STEM
ASSET_DIR = QA_DIR / "source" / "reaxff_v1"
DATA_PATH = ROOT / "data" / "02b_reaxff.npz"
MANIFEST_PATH = ROOT / "data" / "02b_reaxff.json"
MOTION_SOURCE = ROOT / "data" / "02b_reaxff_snapshots.extxyz"

IMAGE_SIZE = (1500, 950)
ATOM_SCALE = 0.90  # same atom and bond sizes as 03 / 03b
BOND_RADIUS = 0.102
STATIC_SCENE_RECT = (0.04, 0.10, 0.96, 0.87)
VIDEO_SCENE_RECT = (0.04, 0.09, 0.96, 0.86)
STATIC_B = (0.325, 0.045, 0.715, 0.640)
STATIC_C = (0.735, 0.045, 0.965, 0.640)
STATIC_D = (0.325, 0.665, 0.965, 0.955)
VIDEO_C = (0.695, 0.025, 0.985, 0.430)
VIDEO_D = (0.695, 0.455, 0.985, 0.975)

BREAKING = "#2E3338"  # the C2-NO2 bond that fades, and its BO curve
TUBE = "#9AA3AB"
TERM = ENERGY_TEAL  # coordination ring, valence angle and charge cues
LEGEND = {
    "F": (r"force $\mathbf{F}$", A_ORANGE),
    "v": (r"velocity $\mathbf{v}$", V_PURPLE),
}

SNAP_STRIDE = 25
N_SNAP = 17  # steps 0, 25, ..., 400
FORCE_SCALE = 1.6  # Angstrom of arrow per eV/Angstrom
VELOCITY_SCALE = 40.0  # Angstrom of arrow per Angstrom/fs
CELL_SECONDS = 0.8
DETAIL_END = 9.6
DURATION = DETAIL_END + CELL_SECONDS * (N_SNAP - 2)

POSITION_EQUATION = r"$\mathbf{r}_{n+1}=\mathbf{r}_n$" "\n" r"$+\mathbf{v}_{n+1/2}\Delta t$"
ACCELERATION_EQUATION = r"$\mathbf{a}_{n}=\mathbf{F}_{n}/m$" "\n" r"$\mathbf{F}_{n}=-\nabla E(\mathbf{r}_n)$"
VELOCITY_EQUATION = r"$\mathbf{v}_{n+1/2}=\mathbf{v}_{n}$" "\n" r"$+\frac{1}{2}\mathbf{a}_{n}\Delta t$"
EQUATIONS = (POSITION_EQUATION, ACCELERATION_EQUATION, VELOCITY_EQUATION)

BONDS = tuple(TNT_BONDS)
ANGLE_AT_O1 = (0, 1, 12)  # ring C1 - ring C2 - selected nitro N

# (name, start, end, stage, chain node, title)
DETAIL = (
    ("geometry", 0.0, 1.4, 0, 0, "Distances"),
    ("order", 1.4, 3.2, 1, 1, "Bond orders"),
    ("energy", 3.2, 5.6, 1, 2, "Energy terms"),
    ("force", 5.6, 7.2, 1, 3, "Forces"),
    ("velocity", 7.2, 8.4, 2, None, "Update velocity"),
    ("position", 8.4, DETAIL_END, 0, None, "Update position"),
)
RAPID_TITLES = ("Forces", "Update velocity", "Update position")
STATIC_TIME = 6.5
KEYFRAME_TIMES = [0.6, 2.0, 3.6, 4.4, 5.2, 6.5, 7.8, 9.0, 10.2, 11.6, 13.0, 15.0, 17.0, 19.0, 21.0]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_data() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}; run scripts/build_box/generate_reaxff_tnt.py")
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    idx = np.rint(np.linspace(0, len(raw["positions"]) - 1, N_SNAP)).astype(int)
    data = {
        "elements": raw["elements"].astype(str),
        "positions": raw["positions"][idx],
        "velocities": raw["velocities"][idx],
        "forces": raw["forces"][idx],
        "bo": raw["bo"][idx],
        "delta": raw["delta"][idx],
        "theta": raw.get("theta", np.zeros((len(raw["positions"]), 1), dtype=float))[idx],
        "q": raw["q"][idx],
        "r_cn": raw["r_cn"][idx],
        "manifest": manifest,
    }
    return data


def write_extxyz(elements: np.ndarray, positions: np.ndarray) -> None:
    MOTION_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame, xyz in enumerate(positions):
        lines.extend([str(len(elements)), f'Properties=species:S:1:pos:R:3 frame={frame} source="02b schematic ReaxFF" pbc="F F F"'])
        for element, c in zip(elements, xyz):
            lines.append(f"{element} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}")
    MOTION_SOURCE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bo_curves(manifest: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    r = np.linspace(1.15, 3.35, 220)
    breaking = np.exp(-3.2 * np.maximum(r - 1.46, 0.0))
    aromatic = 1.5 * np.exp(-2.4 * np.maximum(r - 1.40, 0.0))
    return {"CNO2": (r, breaking), "CC": (r, aromatic)}


# ---------------------------------------------------------------------------
# MatterVis scenes
# ---------------------------------------------------------------------------
def build_camera(positions: np.ndarray) -> SceneCamera:
    direction = np.asarray([0.30, -0.55, 1.0])
    direction /= np.linalg.norm(direction)
    probe = SceneCamera(target=(0.0, 0.0, 0.0), ortho_scale=1.0, direction=tuple(direction), up=(0.0, 1.0, 0.0))
    right, up = camera_basis(probe)
    flat = positions.reshape(-1, 3)
    pad = 0.7
    cloud = np.vstack([flat + pad * right, flat - pad * right, flat + pad * up, flat - pad * up])
    sx, sy = cloud @ right, cloud @ up
    aspect = IMAGE_SIZE[0] / IMAGE_SIZE[1]
    half_w = 0.5 * (sx.max() - sx.min()) * 1.03
    half_h = 0.5 * (sy.max() - sy.min()) * 1.03
    ortho = max(half_h, half_w / aspect)
    target = 0.5 * (sx.max() + sx.min()) * right + 0.5 * (sy.max() + sy.min()) * up
    return camera_for_source(
        MOTION_SOURCE,
        target=tuple(float(x) for x in target),
        ortho_scale=float(ortho),
        frame=0,
        direction=tuple(float(x) for x in direction),
        up=(0.0, 1.0, 0.0),
    )


def bond_styles(bo: np.ndarray) -> dict[tuple[int, int], dict]:
    """Native MatterVis bonds; only the breaking C2-NO2 bond is restyled:
    orange (matches the BO curve), opacity and thickness follow its bond order,
    and it is kept past MatterVis's distance cutoff until BO vanishes."""
    value = float(bo[BREAKING_BOND_INDEX])
    if value < 0.05:
        return {BONDS[BREAKING_BOND_INDEX]: {"opacity": 0.0}}
    return {
        BONDS[BREAKING_BOND_INDEX]: {
            "opacity": float(np.clip(0.15 + 0.85 * min(value, 1.0), 0.0, 1.0)),
            "color": BREAKING,
            "radius_scale": float(0.45 + 0.55 * min(value, 1.0)),
        }
    }


def arrow_groups(group_id: str, origins: np.ndarray, vectors: np.ndarray, *, scale: float, color: str) -> list[dict]:
    lengths = np.linalg.norm(vectors, axis=1) * scale
    keep = lengths > 0.16
    if not np.any(keep):
        return []
    group = make_vector_group(group_id, origins[keep], vectors[keep], scale=scale, color=color)
    for arrow, length in zip(group[0]["arrows"], lengths[keep]):
        head = float(min(0.34, 0.5 * length))
        arrow["style"] = {
            "shaft_radius": 0.045,
            "head_length": head,
            "head_radius": float(max(0.046, min(0.15, 0.4 * head + 0.03))),
            "sides": 16,
        }
    return group


def prepare_assets(data: dict) -> tuple[dict[str, object], SceneCamera]:
    write_extxyz(data["elements"], data["positions"])
    camera = build_camera(data["positions"])
    records: list[dict] = []
    assets: dict[str, object] = {"base": [], "force": []}

    def render(name: str, frame: int, groups: list[dict], bonds: dict[tuple[int, int], dict]) -> Path:
        out = ASSET_DIR / f"{name}.png"
        # MatterVis re-centres each frame; resolve that shift per frame so the
        # world-space tubes and arrows stay on the atoms.
        frame_camera = camera_for_source(
            MOTION_SOURCE,
            target=camera.target,
            ortho_scale=camera.ortho_scale,
            frame=frame,
            direction=camera.direction,
            up=camera.up,
        )
        records.append(
            render_structure(
                MOTION_SOURCE,
                out,
                camera=frame_camera,
                frame=frame,
                view="cluster",
                width=IMAGE_SIZE[0],
                height=IMAGE_SIZE[1],
                atom_scale=ATOM_SCALE,
                bond_radius=BOND_RADIUS,
                show_bonds=True,
                bond_styles=bonds,
                vector_overlays=groups,
            )
        )
        return out

    for k in range(N_SNAP):
        pos = data["positions"][k]
        bonds = bond_styles(data["bo"][k])
        assets["base"].append(render(f"base_{k:02d}", k, [], bonds))
        if k < N_SNAP - 1:
            groups = arrow_groups(f"force-{k}", pos, data["forces"][k], scale=FORCE_SCALE, color=A_ORANGE)
            assets["force"].append(render(f"force_{k:02d}", k, groups, bonds))
    pos = data["positions"][0]
    groups = arrow_groups("vel-0", pos, data["velocities"][0], scale=VELOCITY_SCALE, color=V_PURPLE)
    assets["vel0"] = render("vel_00", 0, groups, bond_styles(data["bo"][0]))
    write_provenance_index(ASSET_DIR, records)
    return assets, camera


# ---------------------------------------------------------------------------
# state of the movie
# ---------------------------------------------------------------------------
def lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    return a + (b - a) * float(np.clip(u, 0.0, 1.0))


def ramp(value: float, start: float, width: float) -> float:
    return smoothstep((value - start) / width)


def movie_state(t: float) -> dict:
    t = float(np.clip(t, 0.0, DURATION - 1e-9))
    state: dict = {"k": 0, "q": 0.0, "step": 0, "images": None, "chain": [0.0] * 4, "sat": [0.0] * 3, "overlay": {}}
    if t < DETAIL_END:
        for name, start, end, stage, node, title in DETAIL:
            if t < end:
                break
        u = (t - start) / (end - start)
        state.update(name=name, u=u, stage=stage, title=title, k=0)
        chain = [0.0] * 4
        sat = [0.0] * 3
        if node is not None:
            for n in range(node):
                chain[n] = 0.40
            chain[node] = 1.0
        if name == "energy":
            sat = [ramp(u, 0.10, 0.18), ramp(u, 0.36, 0.18), ramp(u, 0.62, 0.18)]
        elif name in {"force", "velocity", "position"}:
            sat = [0.80, 0.80, 0.80] if name == "force" else [0.55, 0.55, 0.55]
            chain = [0.40, 0.40, 0.40, 0.40] if name != "force" else [0.40, 0.40, 0.40, 1.0]
            if name == "velocity":
                chain = [0.25] * 4
            if name == "position":
                chain = [0.25] * 4
                sat = [0.25] * 3
        state["chain"], state["sat"] = chain, sat
        overlay = {"rij": 0.0, "coord": 0.0, "angle": 0.0, "charge": 0.0}
        overlay["rij"] = {"geometry": 1.0, "order": 0.35}.get(name, 0.0)
        if name == "energy":
            overlay.update(coord=sat[0], angle=sat[1], charge=sat[2])
        elif name in {"force", "velocity", "position"}:
            hold = 1.0 if name == "force" else 0.6
            overlay.update(coord=hold, angle=hold, charge=hold)
        state["overlay"] = overlay
        if name == "position":
            state["q"] = smoothstep((u - 0.30) / 0.70)
        return state
    cell_time = (t - DETAIL_END) / CELL_SECONDS
    cell = min(int(cell_time), N_SNAP - 3)
    local = cell_time - cell
    k = cell + 1
    sub = 0 if local < 0.34 else (1 if local < 0.60 else 2)
    state.update(
        name="rapid",
        u=local,
        k=k,
        step=k,
        stage=(1, 2, 0)[sub],
        title=RAPID_TITLES[sub],
    )
    q = smoothstep((local - 0.60) / 0.40)
    state["q"] = q
    chain = [0.25] * 4
    sat = [0.25] * 3
    if sub == 0:
        position = local / 0.34 * 4.0
        index = min(int(position), 3)
        chain = [0.40 if n < index else (1.0 if n == index else 0.0) for n in range(4)]
        if index >= 2:
            sat = [0.9, 0.9, 0.9]
    state["chain"], state["sat"] = chain, sat
    state["overlay"] = {"rij": 0.0, "coord": 1.0, "angle": 1.0, "charge": 1.0}
    return state


def image_for_state(state: dict, assets: dict) -> tuple[list[Path], float]:
    """Arrow images are switched, never blended; only atom motion is blended."""
    name, u, k = state["name"], state["u"], state["k"]
    base, force = assets["base"], assets["force"]
    if name in {"geometry", "order", "energy"}:
        return [base[0]], 0.0
    if name == "force":
        return [base[0] if u < 0.15 else force[0]], 0.0
    if name == "velocity":
        return [force[0] if u < 0.12 else assets["vel0"]], 0.0
    if name == "position":
        if u < 0.25:
            return [assets["vel0"]], 0.0
        return [base[0], base[1]], smoothstep((u - 0.30) / 0.70)
    if u < 0.04:
        return [base[k]], 0.0
    if u < 0.34:
        return [force[k]], 0.0
    if u < 0.60:
        return [base[k]], 0.0
    return [base[k], base[k + 1]], smoothstep((u - 0.60) / 0.40)


def legend_for_state(state: dict) -> tuple[str, ...]:
    name, u = state["name"], state["u"]
    if (name == "force" and u >= 0.15) or (name == "velocity" and u < 0.12):
        return ("F",)
    if (name == "velocity" and u >= 0.12) or (name == "position" and u < 0.25):
        return ("v",)
    if name == "rapid" and 0.04 <= u < 0.34:
        return ("F",)
    return ()


def interp_state(data: dict, state: dict) -> dict[str, np.ndarray]:
    k, q = state["k"], state["q"]
    nxt = min(k + 1, N_SNAP - 1)
    return {
        key: lerp(data[key][k], data[key][nxt], q)
        for key in ("positions", "bo", "delta", "q", "r_cn")
    }


def place_sequence(ax: plt.Axes, images: list[Path], position: float, rect, *, zorder: float = 5.0):
    if len(images) == 1:
        return place_render(ax, images[0], rect, zorder=zorder)
    return place_render_blend(ax, images[0], images[1], rect, blend=position, zorder=zorder)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def draw_left(ax: plt.Axes, registry: LayoutRegistry, *, video: bool, stage: int) -> None:
    draw_vv_loop(ax, registry, video=video, active_stage=stage, centre_text=EQUATIONS[stage], centre_y=0.58, radius_x=0.36)


def _axes_aspect(ax: plt.Axes) -> float:
    w, h = ax.figure.canvas.get_width_height()
    pos = ax.get_position()
    return (pos.width * w) / (pos.height * h)


def screen_xy(fitted, camera: SceneCamera, points: np.ndarray) -> np.ndarray:
    return project_world(np.asarray(points, dtype=float), camera=camera, rect=fitted, image_aspect=IMAGE_SIZE[0] / IMAGE_SIZE[1])


def draw_overlays(
    ax: plt.Axes,
    registry: LayoutRegistry,
    camera: SceneCamera,
    fitted,
    current: dict[str, np.ndarray],
    overlay: dict[str, float],
    *,
    video: bool,
) -> None:
    aspect = _axes_aspect(ax)
    xy = screen_xy(fitted, camera, current["positions"])
    lw = 1.0 if video else 0.7

    # r_ij: a dimension line beside the selected C2-NO2 bond
    weight = overlay.get("rij", 0.0)
    if weight > 0.01:
        a, b = xy[1], xy[12]
        along = (b - a) / max(float(np.linalg.norm((b - a) * [aspect, 1.0])), 1e-9)
        normal = np.asarray([-along[1] / aspect, along[0] * aspect])
        normal = normal / max(float(np.linalg.norm(normal * [aspect, 1.0])), 1e-9) * 0.050
        p0, p1 = a + normal, b + normal
        colour = mix_hex(WHITE, INK, weight)
        registry.arrow(ax, tuple(p0), tuple(p1), arrowstyle="<|-|>", mutation_scale=11 if video else 7, lw=1.6 * lw * 1.6, color=colour, zorder=8)
        for end, base in ((p0, a), (p1, b)):
            ax.plot([base[0], end[0]], [base[1], end[1]], color=colour, lw=1.0 * lw, ls=(0, (2, 2)), zorder=7)

    # coordination rings on under-coordinated TNT atoms
    weight = overlay.get("coord", 0.0)
    if weight > 0.01:
        radius = 0.050
        for atom in range(len(current["delta"])):
            open_valence = float(np.clip(-current["delta"][atom], 0.0, 1.2))
            if open_valence < 0.10:
                continue
            ax.add_patch(
                Ellipse(
                    tuple(xy[atom]),
                    2 * radius,
                    2 * radius * aspect,
                    fill=False,
                    ec=TERM,
                    lw=(1.2 + 3.2 * open_valence) * (1.0 if video else 0.7),
                    ls=(0, (3.0, 2.2)),
                    alpha=float(weight) * float(np.clip(0.35 + 0.65 * open_valence, 0.0, 1.0)),
                    zorder=8,
                )
            )

    # valence angle at ring C2, weighted by the two bond orders
    weight = overlay.get("angle", 0.0)
    if weight > 0.01:
        centre, left, nitro = xy[1], xy[0], xy[12]
        u = (left - centre) * [aspect, 1.0]
        w = (nitro - centre) * [aspect, 1.0]
        u, w = u / np.linalg.norm(u), w / np.linalg.norm(w)
        strength = float(current["bo"][0] * current["bo"][BREAKING_BOND_INDEX])
        radius = 0.060
        t = np.linspace(0.0, 1.0, 24)
        directions = (1 - t)[:, None] * u + t[:, None] * w
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        arc = centre + (directions * radius) / [aspect, 1.0]
        ax.plot(arc[:, 0], arc[:, 1], color=TERM, lw=(1.2 + 3.0 * min(strength, 1.0)) * (1.0 if video else 0.7), alpha=float(weight) * float(np.clip(0.25 + 0.75 * strength, 0, 1)), zorder=8, solid_capstyle="round")

    # equilibrated (EEM) charges as partial-charge signs, each placed in the
    # emptiest direction around its atom and outside the coordination ring
    weight = overlay.get("charge", 0.0)
    if weight > 0.01:
        pixel = np.asarray([aspect, 1.0])
        centroid = xy.mean(axis=0)
        placed_charge_anchors: list[np.ndarray] = []
        for atom, charge in enumerate(current["q"]):
            if abs(float(charge)) < 0.05:
                continue
            angles = sorted(
                float(np.arctan2(*((xy[j if i == atom else i] - xy[atom]) * pixel)[::-1]))
                for (i, j), order in zip(BONDS, current["bo"])
                if atom in (i, j) and order >= 0.05
            )
            if not angles:
                outward = (xy[atom] - centroid) * pixel
                angle = float(np.arctan2(outward[1], outward[0]))
            else:
                # middle of the widest angular gap between this atom's bonds
                wrapped = angles + [angles[0] + 2.0 * np.pi]
                gaps = np.diff(wrapped)
                k = int(np.argmax(gaps))
                angle = wrapped[k] + 0.5 * gaps[k]
            # TNT has many more charged atoms than the six-atom HClO4 fixture.
            # Keep every δ marker, but choose an alternate nearby angular gap
            # when the first candidate would overlap a previously placed label.
            candidates = []
            for offset in np.linspace(-np.pi, np.pi, 17)[:-1]:
                direction = np.asarray([np.cos(angle + offset), np.sin(angle + offset)])
                candidates.append(xy[atom] + direction / pixel * 0.16)
            reserved_free = [
                candidate
                for candidate in candidates
                if 0.10 < candidate[0] < 0.90 and 0.14 < candidate[1] < 0.82
            ]
            if reserved_free:
                candidates = reserved_free
            if placed_charge_anchors:
                anchor = max(
                    candidates,
                    key=lambda candidate: min(
                        float(np.linalg.norm((candidate - other) * pixel))
                        for other in placed_charge_anchors
                    ),
                )
            else:
                anchor = candidates[8]
            placed_charge_anchors.append(anchor)
            registry.text(
                ax,
                float(anchor[0]),
                float(anchor[1]),
                r"$\delta^{+}$" if charge > 0 else r"$\delta^{-}$",
                ha="center",
                va="center",
                fontsize=FONT_SIZES["body"],
                color=TERM,
                alpha=float(weight),
                zorder=9,
            )


def draw_scene(
    ax: plt.Axes,
    registry: LayoutRegistry,
    assets: dict,
    camera: SceneCamera,
    data: dict,
    state: dict,
    *,
    video: bool,
) -> None:
    if not video:
        ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, ec=LINE_GRAY, lw=1.1, zorder=20))
    rect = VIDEO_SCENE_RECT if video else STATIC_SCENE_RECT
    images, position = image_for_state(state, assets)
    fitted = place_sequence(ax, images, position, rect, zorder=5.0)
    current = interp_state(data, state)
    draw_overlays(ax, registry, camera, fitted, current, state["overlay"], video=video)
    registry.text(ax, 0.50, 0.965, state["title"], ha="center", va="top", fontsize=FONT_SIZES["panel_title"], color=INK, zorder=21)
    registry.text(ax, 0.0 if video else 0.075, 0.005 if video else 0.070, f"Simulation step {state['step'] + 1:02d}", ha="left", va="bottom", fontsize=FONT_SIZES["micro"], color=INK, zorder=21)
    draw_arrow_legend(
        ax, registry, [LEGEND[key] for key in legend_for_state(state)],
        x_right=1.0 if video else 0.925, y=0.005 if video else 0.070, video=video,
    )


def draw_curve(ax: plt.Axes, registry: LayoutRegistry, data: dict, state: dict, current: dict, *, video: bool) -> None:
    if not video:
        ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, ec=LINE_GRAY, lw=1.1, zorder=2))
    curves = bo_curves(data["manifest"])
    plot = ax.inset_axes((0.17, 0.28, 0.79, 0.68) if video else (0.19, 0.20, 0.72, 0.60))
    r, sigma = curves["CNO2"]
    _, terminal = curves["CC"]
    plot.plot(r, terminal, color=TUBE, lw=2.6 if video else 1.4, zorder=2, label="aromatic C–C")
    plot.plot(r, sigma, color=BREAKING, lw=2.8 if video else 1.5, zorder=3, label="C2–NO2 (breaking)")
    r_cn = float(current["r_cn"])
    bo_cn = float(current["bo"][BREAKING_BOND_INDEX])
    r_aromatic = float(np.linalg.norm(current["positions"][1] - current["positions"][2]))
    bo_aromatic = float(current["bo"][0])
    plot.plot([r_cn, r_cn], [0.0, bo_cn], color=BREAKING, lw=1.2 if video else 0.7, ls=(0, (2, 2)), zorder=3)
    plot.scatter([r_cn], [bo_cn], s=90 if video else 26, color=BREAKING, edgecolors=WHITE, linewidths=1.2 if video else 0.6, zorder=5)
    plot.scatter([r_aromatic], [bo_aromatic], s=70 if video else 20, color=TUBE, edgecolors=WHITE, linewidths=1.0 if video else 0.5, zorder=4)
    plot.set_xlim(1.15, 3.35)
    plot.set_ylim(0.0, 2.0)
    plot.set_xticks((1.5, 2.0, 2.5, 3.0))
    plot.set_yticks((0.0, 1.0, 2.0))
    plot.tick_params(axis="both", labelsize=FONT_SIZES["micro"], colors=INK, width=1.2 if video else 0.8, length=5 if video else 3)
    plot.set_xlabel(r"$r_{ij}$ (Å)", fontsize=FONT_SIZES["body"], color=INK, labelpad=2)
    plot.set_ylabel(r"$BO_{ij}$", fontsize=FONT_SIZES["body"], color=INK, labelpad=4)
    for side in ("top", "right"):
        plot.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        plot.spines[side].set_color(INK)
        plot.spines[side].set_linewidth(1.4 if video else 0.8)
    # The narrow static plot has no free corner, so its key sits above the axes.
    placement = dict(loc="upper right") if video else dict(loc="lower left", bbox_to_anchor=(0.0, 1.02))
    plot.legend(
        **placement, frameon=False, fontsize=FONT_SIZES["micro"], handlelength=1.4,
        borderaxespad=0.1, labelspacing=0.3, handletextpad=0.5,
    )


ENERGY_LINES = (
    (r"$BO_{ij}=\exp[\,p_1\,(r_{ij}/r_0)^{p_2}\,]$", INK),
    (r"$E_\mathrm{bond}=-D_e\,BO_{ij}\,\exp[\,p_\mathrm{be}(1-BO_{ij})\,]$", INK),
    (r"$E_\mathrm{over}=k_o\,\Delta_i^2\,\sigma(\lambda\Delta_i),\ \ \Delta_i=\Sigma_j BO_{ij}-V_i$", TERM),
    (r"$E_\mathrm{angle}=k_\theta\,f(BO_{ij})\,f(BO_{jk})\,(\theta_{ijk}-\theta_0)^2$", TERM),
    (r"$E_\mathrm{Coul}=\Sigma_{i<j}\,q_i q_j\,/\,(r_{ij}^3+\gamma^{-3})^{1/3}$", TERM),
    (r"$\mathbf{F}_i=-\partial E/\partial\mathbf{r}_i,\quad E=\Sigma\,E_\mathrm{term}$", A_ORANGE),
)
IDLE_TEXT = "#A3A9AE"


def draw_energy_terms(ax: plt.Axes, registry: LayoutRegistry, state: dict, *, video: bool) -> None:
    """Explicit expressions of the schematic model; the active ones are coloured."""
    if not video:
        ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, ec=LINE_GRAY, lw=1.1, zorder=2))
    chain, sat = state["chain"], state["sat"]
    weights = (max(chain[0], chain[1]), chain[2], sat[0], sat[1], sat[2], chain[3])
    x0 = 0.02 if video else 0.08
    registry.text(ax, x0, 0.985 if video else 0.92, "Schematic ReaxFF energy", ha="left", va="top", fontsize=FONT_SIZES["micro"], color=DARK_GRAY, zorder=5)
    top, bottom = (0.82, 0.06) if video else (0.76, 0.12)
    for index, ((text, colour), weight) in enumerate(zip(ENERGY_LINES, weights)):
        y = top - index * (top - bottom) / (len(ENERGY_LINES) - 1)
        shade = float(np.clip((weight - 0.25) / 0.75, 0.0, 1.0))
        registry.text(ax, x0, y, text, ha="left", va="center", fontsize=FONT_SIZES["micro"], color=mix_hex(IDLE_TEXT, colour, shade), zorder=5)


def compose(fig: plt.Figure, t: float, registry: LayoutRegistry, assets: dict, camera: SceneCamera, data: dict, *, video: bool) -> list[dict]:
    state = movie_state(t)
    slots = (
        (STORY_VIDEO_A, STORY_VIDEO_B, VIDEO_C, VIDEO_D)
        if video
        else (STORY_STATIC_A, STATIC_B, STATIC_C, STATIC_D)
    )
    axes = [axes_from_top_slot(fig, slot) for slot in slots]
    current = interp_state(data, state)
    draw_left(axes[0], registry, video=video, stage=int(state["stage"]))
    draw_scene(axes[1], registry, assets, camera, data, state, video=video)
    draw_curve(axes[2], registry, data, state, current, video=video)
    draw_energy_terms(axes[3], registry, state, video=video)
    return []


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def render_static(assets, camera, data) -> None:
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=18)
    compose(fig, STATIC_TIME, registry, assets, camera, data, video=False)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)


def render_keyframes(assets, camera, data) -> Path:
    from PIL import Image

    out_dir = QA_DIR / "_qa" / "keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for index, t in enumerate(KEYFRAME_TIMES):
        fig = new_video_figure()
        registry = LayoutRegistry(min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=12, font_family="Arial", coerce_min_font=True)
        compose(fig, t, registry, assets, camera, data, video=True)
        errors = registry.validate(fig)
        path = out_dir / f"frame_{index:02d}_{t:05.2f}s.png"
        fig.savefig(path, dpi=100, facecolor=WHITE)
        plt.close(fig)
        if errors:
            raise RuntimeError(f"Keyframe {t:.2f}s failed layout:\n" + "\n".join(errors))
        thumbs.append(Image.open(path).convert("RGB").resize((640, 200)))
    columns = 4
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * 640, rows * 200), WHITE)
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 640, (index // columns) * 200))
    contact = out_dir / "_contact.png"
    sheet.save(contact)
    return contact


def render_animation_fast(assets, camera, data) -> None:
    import os
    import subprocess
    import time

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
            registry = LayoutRegistry(min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=12, font_family="Arial", coerce_min_font=True)
            compose(figure, index / 24.0, registry, assets, camera, data, video=True)
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


def render_animation_strict(assets, camera, data) -> None:
    def full(rect):
        return {"rect": list(rect), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]}

    audit = {
        "panels": [
            {"id": "integrator", **full(STORY_VIDEO_A)},
            {"id": "molecule", **full(STORY_VIDEO_B)},
            {"id": "bond_order", **full(VIDEO_C)},
            {"id": "energy_terms", **full(VIDEO_D)},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.020, "min_panel_bbox_fill": 0.20, "grid_rows": 12, "grid_columns": 20},
        "bands": [
            {"id": "gap_a_b", "rect": [0.215, 0.025, 0.230, 0.975], "max_ink_pixels": 5000},
            {"id": "gap_b_right", "rect": [0.680, 0.025, 0.695, 0.975], "max_ink_pixels": 5000},
            {"id": "gap_c_d", "rect": [0.695, 0.432, 0.985, 0.453], "max_ink_pixels": 5000},
        ],
    }
    render_video(
        stem=STEM,
        duration_seconds=DURATION,
        draw_frame=lambda fig, t, i, registry: compose(fig, t, registry, assets, camera, data, video=True),
        audit_config=audit,
        qa_directory=QA_DIR / "_qa",
        representative_times=KEYFRAME_TIMES,
    )


def write_manifest(data: dict, camera: SceneCamera) -> None:
    json_dump(
        QA_DIR / "asset_manifest.json",
        {
            "stem": STEM,
            "data": str(DATA_PATH),
            "labelled": "schematic ReaxFF-style model; not a published parameter set; no LAMMPS run",
            "snapshots": {"stride_steps": SNAP_STRIDE, "count": N_SNAP},
            "display_scales": {
                "force_ang_per_ev_ang": FORCE_SCALE,
                "velocity_ang_per_ang_fs": VELOCITY_SCALE,
            },
            "bond_tube": {"radius_ang": "0.020 + 0.040 * min(BO, 1.7)", "opacity": "0.22 + 0.78 * min(BO, 1)", "hidden_below_bo": 0.05},
            "camera": {"target": list(camera.target), "ortho_scale": camera.ortho_scale, "direction": list(camera.direction)},
            "image_size": list(IMAGE_SIZE),
            "duration_seconds": DURATION,
            "detailed_phases": {name: [start, end] for name, start, end, *_ in DETAIL},
            "rapid_cell_seconds": CELL_SECONDS,
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
    assets, camera = prepare_assets(data)
    write_manifest(data, camera)
    if args.assets_only:
        return
    if args.preview_only:
        print(render_keyframes(assets, camera, data))
        return
    render_static(assets, camera, data)
    if not args.static_only:
        if args.strict_video:
            render_animation_strict(assets, camera, data)
        else:
            render_animation_fast(assets, camera, data)


if __name__ == "__main__":
    main()
