"""End-to-end neural-network MD stories (DeepMD and DPA4C) in the AIMD layout.

Both stories share the outer Velocity-Verlet loop of the AIMD movie and the
same retained 64-water periodic box.  What differs is the pluggable module
that turns positions into accelerations:

* DeepMD: minimum-image neighbours -> DeepPot-SE environment matrix R_i ->
  embedding + fitting network -> atomic energies eps_i -> E
* DPA4C: minimum-image neighbours -> relative unit vectors and radial
  envelopes -> equivariant features (l <= 2) -> fitting network -> eps_i -> E

After E the two movies are identical: F = -dE/dr, a = F/m, velocity and
position updates, and a return arrow back into the integrator loop.  Every
number shown is read from the retained trajectory (energies, atomic energies,
forces, velocities) or computed exactly from the retained geometry (r_ij,
s(r_ij), unit vectors).  Atoms, cutoff spheres, neighbour edges and vectors
are rendered by MatterVis in world space; matplotlib only composes panels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from PIL import Image

from common import (
    CRIMSON,
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
    render_video,
    save_static,
    sha256_file,
    smoothstep,
)
from mattervis_story import (
    SceneCamera,
    _composition_rgba,
    camera_for_source,
    draw_vv_loop,
    make_sphere_mesh,
    make_torus_mesh,
    make_vector_group,
    project_world,
    render_structure,
    write_provenance_index,
)
from mat_viewer.render.geometry import cylinder_mesh


ROOT = Path(__file__).resolve().parents[2] / "product"
DATA_DIR = ROOT / "data"

MODELS = {
    "deepmd": {
        "stem": "04_deep_potential_md",
        "data": DATA_DIR / "dpmd_water_box_trajectory.npz",
        "metadata": DATA_DIR / "dpmd_water_box_trajectory.json",
        "provider": "DeepMD · DeepPot-SE",
        "descriptor_title": r"Relative positions $\rightarrow$ environment matrix $R_i$",
        "descriptor_header": r"$R_i$ rows: $s(r)$, $s\,x/r$, $s\,y/r$, $s\,z/r$",
        "network_title": r"Embedding + fitting network $\rightarrow$ atomic energies $\varepsilon_i$",
        "network_note": "two networks: embedding (25-50-100) and fitting (240-240-240)",
    },
    "dpa4c": {
        "stem": "04_4c_dpa4c",
        "data": DATA_DIR / "dpa4c_water_box_trajectory.npz",
        "metadata": DATA_DIR / "dpa4c_water_box_trajectory.json",
        "provider": "DPA4C · equivariant descriptor",
        "descriptor_title": r"Relative positions $\rightarrow$ equivariant features ($l \leq 2$)",
        "descriptor_header": r"$\hat{u}_{ij} = r_{ij}/r$ feeds radial $\times$ $Y_{lm}(\hat{u})$ channels",
        "network_title": r"Equivariant layers + fitting network $\rightarrow$ $\varepsilon_i$",
        "network_note": "64 channels, l = 0, 1, 2; fitting network 256-256-256 (SiLU)",
    },
}

# Layout: identical slot grammar to the AIMD story.  The bottom 7 % of the
# canvas is reserved for the return arrow from the force provider back into
# the integrator loop.
VIDEO_A = (0.015, 0.025, 0.215, 0.925)
VIDEO_B = (0.230, 0.025, 0.680, 0.925)
VIDEO_C = (0.695, 0.025, 0.985, 0.455)
VIDEO_D = (0.695, 0.480, 0.985, 0.925)
STATIC_A = (0.035, 0.055, 0.235, 0.905)
STATIC_B = (0.250, 0.045, 0.705, 0.905)
STATIC_C = (0.720, 0.045, 0.965, 0.400)
STATIC_D = (0.720, 0.430, 0.965, 0.905)
RETURN_Y_VIDEO = 0.045
RETURN_Y_STATIC = 0.030

# Shared colour semantics with the AIMD movie.
POSITION_LAKE = "#4E9BB5"
FORCE_OLIVE = "#A99C50"
VELOCITY_EMERALD = "#2F8562"
ACCEL_PLUM = "#7B5EA7"
EDGE_NAVY = NAVY
SPHERE_TEAL = "#397F99"
CENTRE_NAVY = "#183153"
FADE_OUTSIDE = 0.15

# Display scales (declared in provenance; identical for both models).
# Arrow lengths are normalised per quantity so that the longest arrow drawn in
# any state inside r_c measures LONGEST_ARROW angstrom on screen.  The scale
# factors are therefore data-driven (recorded in asset_manifest.json) and the
# legends quote the real magnitude that the longest arrow stands for.
LONGEST_ARROW = 1.9  # angstrom (world space)
MIN_ARROW = 0.14  # angstrom; shorter arrows are not drawn
# The eps colour scale saturates at the largest |eps_j - mean eps(species)| seen
# inside r_c over all states (rounded up to two significant figures), so both
# models fill the same navy-grey-crimson ramp with their own real spread.

EV_A_TO_A_FS2 = 0.00964853399
O_MASS = 15.9994
H_MASS = 1.008
R_CUT_SMOOTH = 0.5  # DeepPot-SE rcut_smth of the retained water model

FOCUS_RADIUS = 9.0  # angstrom; atoms kept in the local-environment source
# House orthographic direction shared with the other parts of the story.  The
# prepared 64-water box is lattice-like, so a view along a cell axis would
# stack whole columns of molecules; the oblique view separates them.
VIEW_DIRECTION = (1.55, -1.0, 0.62)
BOX_ORTHO = 10.0
FOCUS_ORTHO = 6.75
RENDER_PX = 1100

VIDEO_DURATION = 30.0
DETAILED_BLOCK = 9.0
RAPID_BLOCK = 1.5
DETAILED_PHASES = (
    ("positions", 0.8),
    ("neighbours", 1.2),
    ("descriptor", 1.6),
    ("network", 1.2),
    ("energy", 0.8),
    ("force", 1.0),
    ("accel", 0.8),
    ("velocity", 0.7),
    ("move", 0.9),
)
RAPID_PHASES = (
    ("neighbours", 0.20),
    ("descriptor", 0.30),
    ("network", 0.20),
    ("energy", 0.15),
    ("force", 0.20),
    ("accel", 0.15),
    ("velocity", 0.15),
    ("move", 0.15),
)
MODE_TO_LOOP_STAGE = {
    "positions": 1,
    "neighbours": 1,
    "descriptor": 1,
    "network": 1,
    "energy": 1,
    "force": 1,
    "accel": 1,
    "velocity": 2,
    "move": 0,
}
CHAIN_ACTIVE = {
    "positions": None,
    "neighbours": None,
    "descriptor": None,
    "network": None,
    "energy": 0,
    "force": 2,
    "accel": 3,
    "velocity": 4,
    "move": 4,
}

POSITION_EQUATION = r"$\mathbf{r}_{n+1}=\mathbf{r}_n$" "\n" r"$+\mathbf{v}_{n+1/2}\Delta t$"
ACCELERATION_EQUATION = r"$\mathbf{a}_{n}=\mathbf{F}_{n}/m$" "\n" r"$\mathbf{F}_{n}=-\nabla_R E(\mathbf{r}_n)$"
VELOCITY_EQUATION = r"$\mathbf{v}_{n+1/2}=\mathbf{v}_{n}$" "\n" r"$+\frac{1}{2}\mathbf{a}_{n}\Delta t$"

EPS_CMAP = LinearSegmentedColormap.from_list("eps", [NAVY, "#D9D9D9", CRIMSON])


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def minimum_image(delta: np.ndarray, box: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    return delta - box * np.rint(delta / box)


def load_data(model: str) -> dict[str, object]:
    config = MODELS[model]
    path = config["data"]
    if not path.exists():
        raise FileNotFoundError(f"Missing retained trajectory for {model}: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    required = (
        "elements", "box_length", "central_index", "positions", "velocities",
        "half_velocities", "forces_ev_per_angstrom", "atomic_energy_ev",
        "total_energy_ev", "neighbour_ids", "neighbour_counts", "dt_fs",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} lacks {missing}")
    positions = data["positions"]
    if positions.ndim != 3 or positions.shape[1:] != (192, 3):
        raise ValueError(f"Expected (n_states, 192, 3) positions, got {positions.shape}")
    if not np.allclose(data["atomic_energy_ev"].sum(axis=1), data["total_energy_ev"], atol=1.0e-3):
        raise ValueError("Atomic energies do not sum to the total energy")
    data["elements"] = data["elements"].astype(str)
    data["box_length"] = float(data["box_length"])
    data["central_index"] = int(data["central_index"])
    data["dt_fs"] = float(data["dt_fs"])
    data["cutoff"] = float(data["cutoff_angstrom"]) if "cutoff_angstrom" in data else 6.0
    metadata = {}
    if config["metadata"].exists():
        metadata = json.loads(config["metadata"].read_text(encoding="utf-8"))
    data["metadata"] = metadata
    data["masses"] = np.where(data["elements"] == "O", O_MASS, H_MASS).astype(float)
    data["accelerations"] = data["forces_ev_per_angstrom"] * EV_A_TO_A_FS2 / data["masses"][None, :, None]
    # Recompute minimum-image displacements between retained states so the
    # position-update arrows never inherit a wrapped jump.
    disp = positions[1:] - positions[:-1]
    data["mic_displacements"] = minimum_image(disp, data["box_length"])
    return data


def smooth_switch(r: np.ndarray, rcs: float, rc: float) -> np.ndarray:
    """DeepPot-SE s(r): 1/r below rcs, polynomial switch to zero at rc."""
    r = np.asarray(r, dtype=float)
    u = (r - rcs) / max(rc - rcs, 1.0e-12)
    poly = u**3 * (-6.0 * u**2 + 15.0 * u - 10.0) + 1.0
    s = np.where(r < rcs, 1.0 / np.maximum(r, 1.0e-9), poly / np.maximum(r, 1.0e-9))
    s = np.where(r >= rc, 0.0, s)
    return s


def local_environment(data: dict[str, object], state: int) -> dict[str, object]:
    """Exact minimum-image neighbourhood of the centre atom at one state."""
    positions = data["positions"][state]
    centre = data["central_index"]
    box = data["box_length"]
    ids = data["neighbour_ids"][state]
    ids = ids[ids >= 0]
    ids = ids[ids != centre]
    vectors = minimum_image(positions[ids] - positions[centre], box)
    distances = np.linalg.norm(vectors, axis=1)
    order = np.argsort(distances)
    ids, vectors, distances = ids[order], vectors[order], distances[order]
    unit = vectors / distances[:, None]
    s = smooth_switch(distances, R_CUT_SMOOTH, data["cutoff"])
    return {
        "ids": ids,
        "vectors": vectors,
        "distances": distances,
        "unit": unit,
        "s": s,
        "deep_r": np.column_stack((s, s[:, None] * unit)),
    }


def species_mean_deviation(data: dict[str, object], state: int) -> np.ndarray:
    eps = data["atomic_energy_ev"][state]
    elements = data["elements"]
    deviation = np.zeros_like(eps)
    for species in ("O", "H"):
        mask = elements == species
        deviation[mask] = eps[mask] - eps[mask].mean()
    return deviation


# --------------------------------------------------------------------------
# MatterVis assets
# --------------------------------------------------------------------------
def _edge_meshes(origin: np.ndarray, endpoints: np.ndarray, *, color: str, opacity: float, radius: float) -> list[dict]:
    meshes = []
    for index, end in enumerate(np.asarray(endpoints, dtype=float)):
        if float(np.linalg.norm(end - origin)) <= 1.0e-9:
            continue
        vertices, triangles, normals = cylinder_mesh(origin, end, radius, sides=6, capped=False)
        meshes.append(
            {
                "id": f"neighbour-edge-{index}",
                "vertices": vertices,
                "triangles": triangles,
                "normals": normals,
                "color": color,
                "opacity": opacity,
                "metadata": {"semantic": "minimum-image neighbour edge i-j"},
            }
        )
    return meshes


def write_sources(data: dict[str, object], source_dir: Path) -> tuple[Path, Path, list[int]]:
    """Write the periodic box and the O126-centred local source (all states)."""
    from ase import Atoms
    from ase.io import write

    source_dir.mkdir(parents=True, exist_ok=True)
    positions = data["positions"]
    elements = data["elements"]
    box = data["box_length"]
    centre = data["central_index"]
    origin = positions[0, centre]
    delta0 = minimum_image(positions[0] - origin, box)
    keep = np.flatnonzero(np.linalg.norm(delta0, axis=1) <= FOCUS_RADIUS)
    keep = [int(centre)] + [int(index) for index in keep if int(index) != centre]
    box_frames = []
    focus_frames = []
    for state in range(positions.shape[0]):
        box_frames.append(Atoms(symbols=elements.tolist(), positions=positions[state], cell=np.eye(3) * box, pbc=True))
        local = minimum_image(positions[state][keep] - origin, box)
        focus_frames.append(Atoms(symbols=elements[keep].tolist(), positions=local))
    box_source = source_dir / "water_box_trajectory.extxyz"
    focus_source = source_dir / "local_environment_trajectory.extxyz"
    write(box_source, box_frames, format="extxyz")
    write(focus_source, focus_frames, format="extxyz")
    return box_source, focus_source, keep


def arrow_scales(data: dict[str, object]) -> dict[str, dict[str, float]]:
    """Data-driven display scale per vector quantity.

    The maximum magnitude is taken over every state and every atom inside the
    cutoff of the central atom, so the same scale holds for the whole story and
    arrows can be compared between states.
    """
    n_states = data["positions"].shape[0]
    centre = data["central_index"]
    quantities = {
        "force": (data["forces_ev_per_angstrom"], n_states, "eV/Å"),
        "accel": (data["accelerations"], n_states, "Å/fs²"),
        "velocity": (data["half_velocities"], n_states - 1, "Å/fs"),
        "move": (data["mic_displacements"], n_states - 1, "Å"),
    }
    out = {}
    for key, (array, count, unit) in quantities.items():
        peak = 0.0
        for state in range(count):
            ids = [centre] + [int(j) for j in local_environment(data, state)["ids"]]
            peak = max(peak, float(np.linalg.norm(array[state][ids], axis=1).max()))
        out[key] = {"max_magnitude": peak, "scale": LONGEST_ARROW / peak if peak > 0 else 1.0, "unit": unit}
    eps_peak = 0.0
    for state in range(n_states):
        ids = [centre] + [int(j) for j in local_environment(data, state)["ids"]]
        eps_peak = max(eps_peak, float(np.abs(species_mean_deviation(data, state)[ids]).max()))
    exponent = math.floor(math.log10(eps_peak)) - 1 if eps_peak > 0 else 0
    out["eps"] = {"max_deviation": eps_peak, "range": math.ceil(eps_peak / 10**exponent) * 10**exponent if eps_peak > 0 else 1.0, "unit": "eV"}
    return out


def prepare_assets(model: str, data: dict[str, object], qa_dir: Path) -> dict[str, object]:
    source_dir = qa_dir / "source" / "trajectory_sources"
    asset_dir = qa_dir / "source" / "mattervis_v1"
    box_source, focus_source, keep = write_sources(data, source_dir)
    positions = data["positions"]
    box = data["box_length"]
    centre = data["central_index"]
    n_states = positions.shape[0]
    origin = positions[0, centre]
    box_centre = np.full(3, 0.5 * box)
    direction = VIEW_DIRECTION
    up = (0.0, 0.0, 1.0)
    # MatterVis recentres each loaded frame; the canonical origin shift is
    # therefore resolved per frame.  Target, direction and ortho scale are
    # identical, so every state maps to the same screen coordinates.
    box_cameras = [camera_for_source(box_source, target=box_centre, ortho_scale=BOX_ORTHO, frame=state, direction=direction, up=up) for state in range(n_states)]
    focus_cameras = [camera_for_source(focus_source, target=(0.0, 0.0, 0.0), ortho_scale=FOCUS_ORTHO, frame=state, direction=direction, up=up) for state in range(n_states)]
    box_camera = box_cameras[0]
    focus_camera = focus_cameras[0]

    keep_index = {atom: local for local, atom in enumerate(keep)}
    cutoff = data["cutoff"]
    sphere = make_sphere_mesh(np.zeros(3), cutoff, color=SPHERE_TEAL, opacity=0.07, lat_steps=18, lon_steps=36, mesh_id="r_c_sphere")
    equator = make_torus_mesh(np.zeros(3), cutoff * 0.998, 0.035, normal=direction, color="#1F536B", opacity=0.62, major_steps=72, tube_steps=6, mesh_id="r_c_equator")
    meridian = make_torus_mesh(np.zeros(3), cutoff * 0.998, 0.024, normal=up, color="#2E89A7", opacity=0.42, major_steps=72, tube_steps=6, mesh_id="r_c_meridian")
    shell = [sphere, equator, meridian]
    arrow_style = {"shaft_radius": 0.055, "head_length_ratio": 0.30, "head_radius_ratio": 2.3, "sides": 16}
    scales = arrow_scales(data)

    def arrows(name: str, origins: np.ndarray, vectors: np.ndarray, scale: float, color: str):
        """Vector overlay with sub-legible arrows dropped."""
        lengths = np.linalg.norm(vectors, axis=1) * scale
        mask = lengths >= MIN_ARROW
        return make_vector_group(name, origins[mask], vectors[mask], scale=scale, color=color, style=arrow_style)

    records = []
    assets: dict[str, list[Path]] = {key: [] for key in ("box", "plain", "cut", "eps", "force", "accel", "velocity", "move")}
    for state in range(n_states):
        common_focus = dict(camera=focus_cameras[state], view="cluster", width=RENDER_PX, height=RENDER_PX, atom_scale=1.0, bond_radius=0.095, show_cell=False, include_boundary_replicas=False)
        common_box = dict(camera=box_cameras[state], view="unit_cell", width=RENDER_PX, height=RENDER_PX, atom_scale=0.72, bond_radius=0.075, show_cell=True, cell_color="#9AA5AA", cell_width_px=1.2)
        local = local_environment(data, state)
        inside = set(int(index) for index in local["ids"])
        focus_positions = minimum_image(positions[state][keep] - origin, box)
        fade = {keep_index[atom]: (1.0 if (atom in inside or atom == centre) else FADE_OUTSIDE) for atom in keep}
        centre_colour = {keep_index[centre]: CENTRE_NAVY}
        edges = _edge_meshes(focus_positions[0], focus_positions[[keep_index[int(j)] for j in local["ids"]]], color=EDGE_NAVY, opacity=0.70, radius=0.020)
        deviation = species_mean_deviation(data, state)
        eps_colours = {}
        for atom in keep:
            if atom in inside or atom == centre:
                value = float(np.clip(deviation[atom] / scales["eps"]["range"], -1.0, 1.0))
                eps_colours[keep_index[atom]] = to_hex(EPS_CMAP(0.5 + 0.5 * value))
        inside_list = [centre] + [int(j) for j in local["ids"]]
        rows = [keep_index[atom] for atom in inside_list]
        inside_pos = focus_positions[rows]

        def path(name: str) -> Path:
            return asset_dir / f"{name}_{state:02d}.png"

        records.append(render_structure(box_source, path("box"), frame=state, atom_color_overrides={centre: CENTRE_NAVY}, **common_box))
        records.append(render_structure(focus_source, path("plain"), frame=state, atom_color_overrides=centre_colour, **common_focus))
        records.append(render_structure(focus_source, path("cut"), frame=state, show_bonds=False, mesh_overlays=shell + edges, atom_opacity_scales=fade, atom_color_overrides=centre_colour, **common_focus))
        records.append(render_structure(focus_source, path("eps"), frame=state, show_bonds=False, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=eps_colours, **common_focus))
        force = data["forces_ev_per_angstrom"][state][inside_list]
        records.append(render_structure(focus_source, path("force"), frame=state, show_bonds=False, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=centre_colour, vector_overlays=arrows(f"force-{state}", inside_pos, force, scales["force"]["scale"], FORCE_OLIVE), **common_focus))
        accel = data["accelerations"][state][inside_list]
        records.append(render_structure(focus_source, path("accel"), frame=state, show_bonds=False, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=centre_colour, vector_overlays=arrows(f"accel-{state}", inside_pos, accel, scales["accel"]["scale"], ACCEL_PLUM), **common_focus))
        if state < n_states - 1:
            half = data["half_velocities"][state][inside_list]
            records.append(render_structure(focus_source, path("velocity"), frame=state, show_bonds=False, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=centre_colour, vector_overlays=arrows(f"velocity-{state}", inside_pos, half, scales["velocity"]["scale"], VELOCITY_EMERALD), **common_focus))
            disp = data["mic_displacements"][state][inside_list]
            next_focus = {**common_focus, "camera": focus_cameras[state + 1]}
            records.append(render_structure(focus_source, path("move"), frame=state + 1, show_bonds=False, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=centre_colour, vector_overlays=arrows(f"move-{state}", inside_pos, disp, scales["move"]["scale"], POSITION_LAKE), **next_focus))
            assets["velocity"].append(path("velocity"))
            assets["move"].append(path("move"))
        for key in ("box", "plain", "cut", "eps", "force", "accel"):
            assets[key].append(path(key))
    write_provenance_index(asset_dir, records)
    json_dump(
        qa_dir / "asset_manifest.json",
        {
            "schema": "nnmd_end_to_end_assets/v1",
            "model": model,
            "trajectory": str(MODELS[model]["data"]),
            "trajectory_sha256": sha256_file(MODELS[model]["data"]),
            "box_source": str(box_source),
            "focus_source": str(focus_source),
            "focus_atoms": keep,
            "focus_radius_angstrom": FOCUS_RADIUS,
            "camera": {"direction": list(direction), "up": list(up), "box_ortho_scale": BOX_ORTHO, "focus_ortho_scale": FOCUS_ORTHO},
            "display_scales": {**scales, "longest_arrow_angstrom": LONGEST_ARROW, "min_arrow_angstrom": MIN_ARROW},
            "fade_outside_cutoff": FADE_OUTSIDE,
            "renders": len(records),
        },
    )
    return {"assets": assets, "box_camera": box_camera, "focus_camera": focus_camera, "keep": keep, "scales": scales}


# --------------------------------------------------------------------------
# composition helpers
# --------------------------------------------------------------------------
def _axes_aspect(ax: plt.Axes) -> float:
    width, height = ax.figure.canvas.get_width_height()
    position = ax.get_position()
    return (position.width * width) / (position.height * height)


def _blend(first: Path, second: Path, weight: float) -> np.ndarray:
    a = _composition_rgba(first)
    if weight <= 1.0e-6 or first == second:
        return a
    b = _composition_rgba(second)
    if weight >= 1.0 - 1.0e-6:
        return b
    return np.rint((1.0 - weight) * a.astype(np.float32) + weight * b.astype(np.float32)).astype(np.uint8)


def _square_rect(ax: plt.Axes, centre: tuple[float, float], radius_x: float) -> tuple[float, float, float, float]:
    """Axes rect of a square image whose on-screen half-width is radius_x."""
    aspect = _axes_aspect(ax)
    radius_y = radius_x * aspect
    return (centre[0] - radius_x, centre[1] - radius_y, centre[0] + radius_x, centre[1] + radius_y)


def _place_image(ax: plt.Axes, image: np.ndarray, rect: tuple[float, float, float, float], *, zorder: float, clip: Ellipse | None = None, alpha: float = 1.0):
    x0, y0, x1, y1 = rect
    artist = ax.imshow(image, extent=(x0, x1, y0, y1), origin="upper", interpolation="lanczos", zorder=zorder, aspect="auto", alpha=alpha)
    if clip is not None:
        artist.set_clip_path(clip)
    return artist


def _deemphasize(ax: plt.Axes, alpha: float = 0.70) -> None:
    for target in (ax, *getattr(ax, "child_axes", [])):
        target.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=target.transAxes, facecolor=WHITE, edgecolor="none", alpha=alpha, zorder=1000))


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}".replace("-", "\u2212")


def _neg(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace("-", "\u2212")


# --------------------------------------------------------------------------
# panel A: integrator loop
# --------------------------------------------------------------------------
def draw_left(ax: plt.Axes, registry: LayoutRegistry, *, video: bool, mode: str, provider: str) -> None:
    stage = MODE_TO_LOOP_STAGE[mode]
    equation = (POSITION_EQUATION, ACCELERATION_EQUATION, VELOCITY_EQUATION)[stage]
    draw_vv_loop(ax, registry, video=video, active_stage=stage, centre_text=equation, centre_y=0.58, radius_x=0.40)
    # Two short left-aligned lines keep clear of the return arrow entering the
    # 'a' node from below on the right-hand side of the panel.
    name, _, detail = provider.partition(" · ")
    registry.text(ax, 0.03, 0.185, "force provider:", ha="left", va="center", fontsize=12 if video else 10, color=DARK_GRAY, zorder=5)
    registry.text(ax, 0.03, 0.120, name, ha="left", va="center", fontsize=12 if video else 10, color=INK, weight="bold", zorder=5)
    if detail:
        registry.text(ax, 0.03, 0.055, detail, ha="left", va="center", fontsize=12 if video else 10, color=INK, zorder=5)


# --------------------------------------------------------------------------
# panel B: the real system with the in-situ operator flow
# --------------------------------------------------------------------------
def draw_case(
    ax: plt.Axes,
    registry: LayoutRegistry,
    data: dict[str, object],
    scene: dict[str, object],
    *,
    model: str,
    video: bool,
    mode: str,
    state: int,
    progress: float,
    rapid: bool,
) -> None:
    config = MODELS[model]
    assets = scene["assets"]
    box_camera: SceneCamera = scene["box_camera"]
    focus_camera: SceneCamera = scene["focus_camera"]
    centre = data["central_index"]
    positions = data["positions"]
    box = data["box_length"]
    local = local_environment(data, state)
    n_states = positions.shape[0]
    if not video:
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, ec=LINE_GRAY, lw=1.1, zorder=20))

    # --- periodic box (left) with the in-situ cutoff circle -----------------
    box_x, box_y, box_r = (0.165, 0.515, 0.135) if video else (0.175, 0.560, 0.150)
    box_rect = _square_rect(ax, (box_x, box_y), box_r)
    _place_image(ax, _composition_rgba(assets["box"][min(state, n_states - 1)]), box_rect, zorder=4)
    sphere_alpha = 1.0 if mode not in {"positions", "neighbours"} else (smoothstep(progress) if mode == "neighbours" else 0.0)
    centre_world = positions[state, centre]
    projected_centre = project_world(centre_world[None, :], camera=box_camera, rect=box_rect, image_aspect=1.0)[0]
    aspect = _axes_aspect(ax)
    radius_axes = data["cutoff"] / (2.0 * BOX_ORTHO) * (box_rect[2] - box_rect[0])
    if sphere_alpha > 0.0:
        ax.add_patch(Ellipse(tuple(projected_centre), 2.0 * radius_axes, 2.0 * radius_axes * aspect, fill=False, ec=SPHERE_TEAL, lw=2.4 if video else 1.4, alpha=0.9 * sphere_alpha, zorder=9))
    registry.text(ax, box_x, box_rect[3] + (0.035 if video else 0.015), "64 H$_2$O periodic box", ha="center", va="bottom", fontsize=12 if video else 10, color=DARK_GRAY, zorder=21)

    # --- magnifier (right) --------------------------------------------------
    mag_centre = (0.665, 0.515) if video else (0.655, 0.560)
    mag_rx = 0.262 if video else 0.305
    mag_rect = _square_rect(ax, mag_centre, mag_rx)
    mag_ry = mag_rx * aspect
    clip = Ellipse(mag_centre, 2.0 * mag_rx, 2.0 * mag_ry, transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)
    ax.add_patch(Ellipse(mag_centre, 2.0 * mag_rx, 2.0 * mag_ry, fc=WHITE, ec=LINE_GRAY, lw=2.0 if video else 1.2, zorder=3))
    # guide lines from the in-situ circle to the magnifier
    guide_alpha = 0.55 * (sphere_alpha if mode in {"positions", "neighbours"} else 1.0)
    if guide_alpha > 0.0:
        for sign in (1.0, -1.0):
            start = (projected_centre[0] + radius_axes * 0.35, projected_centre[1] + sign * radius_axes * aspect * 0.94)
            end = (mag_centre[0] - mag_rx * 0.70, mag_centre[1] + sign * mag_ry * 0.71)
            ax.add_line(Line2D([start[0], end[0]], [start[1], end[1]], color=LINE_GRAY, lw=1.6 if video else 1.0, alpha=guide_alpha, zorder=2))

    if mode == "positions":
        image = _composition_rgba(assets["plain"][state])
    elif mode == "neighbours":
        image = _blend(assets["plain"][state], assets["cut"][state], smoothstep(progress))
    elif mode == "descriptor":
        image = _composition_rgba(assets["cut"][state])
    elif mode == "network":
        image = _blend(assets["cut"][state], assets["eps"][state], smoothstep(min(progress / 0.45, 1.0)))
    elif mode == "energy":
        image = _composition_rgba(assets["eps"][state])
    elif mode == "force":
        image = _composition_rgba(assets["force"][state])
    elif mode == "accel":
        image = _composition_rgba(assets["accel"][state])
    elif mode == "velocity":
        image = _composition_rgba(assets["velocity"][state])
    elif mode == "move":
        image = _composition_rgba(assets["move"][state])
    else:
        raise ValueError(mode)
    _place_image(ax, image, mag_rect, zorder=5, clip=clip)

    # --- in-situ descriptor annotation ------------------------------------
    focus_origin = positions[0, centre]
    local_vectors = local["vectors"]
    centre_local = minimum_image(positions[state, centre] - focus_origin, box)
    labelled = min(3, len(local_vectors))
    if mode == "descriptor":
        reveal = progress
        for k in range(labelled):
            if reveal < (k + 0.15) / labelled:
                break
            endpoint = centre_local + local_vectors[k]
            xy = project_world(np.vstack((centre_local, endpoint)), camera=focus_camera, rect=mag_rect, image_aspect=1.0)
            ax.plot(xy[:, 0], xy[:, 1], color=WHITE, lw=7.0 if video else 3.5, solid_capstyle="round", zorder=11, alpha=0.9)
            ax.plot(xy[:, 0], xy[:, 1], color=EDGE_NAVY, lw=3.6 if video else 1.8, solid_capstyle="round", zorder=12)
            offset = np.sign(xy[1] - xy[0]) * np.array([0.028, 0.045])
            registry.text(ax, xy[1, 0] + offset[0], xy[1, 1] + offset[1], f"$j_{k + 1}$", ha="center", va="center", fontsize=14 if video else 10, color=EDGE_NAVY, weight="bold", zorder=13, bbox=dict(boxstyle="round,pad=0.15", fc=WHITE, ec="none", alpha=0.85))

    # --- stage title and step label ---------------------------------------
    title_colour = {
        "positions": INK,
        "neighbours": SPHERE_TEAL,
        "descriptor": EDGE_NAVY,
        "network": CRIMSON,
        "energy": GREEN,
        "force": FORCE_OLIVE,
        "accel": ACCEL_PLUM,
        "velocity": VELOCITY_EMERALD,
        "move": POSITION_LAKE,
    }[mode]
    n_neigh = len(local["ids"])
    cutoff = data["cutoff"]
    title = {
        "positions": r"Current positions $\mathbf{r}_n$ of all 192 atoms",
        "neighbours": rf"Collect neighbours of O126 within $r_c$ = {cutoff:.0f} Å: {n_neigh} atoms",
        "descriptor": config["descriptor_title"],
        "network": config["network_title"],
        "energy": r"$E = \sum_i \varepsilon_i$ over all 192 atoms",
        "force": r"$\mathbf{F}_i = -\partial E/\partial \mathbf{r}_i$ by automatic differentiation",
        "accel": r"$\mathbf{a}_i = \mathbf{F}_i / m_i$: light H atoms respond 16× more than O",
        "velocity": r"Update velocity $\mathbf{v}_{n+1/2}$",
        "move": r"Update position $\mathbf{r}_{n+1}$",
    }[mode]
    registry.text(ax, 0.50, 0.945, title, ha="center", va="center", fontsize=18 if video else 11, color=title_colour, weight="bold", zorder=21)
    step_xy = (0.03, 0.055) if video else (0.05, 0.875)
    registry.text(ax, *step_xy, f"MD step {state + 1:02d} · Δt = {data['dt_fs']:.1f} fs", ha="left", va="bottom", fontsize=12 if video else 10, color=INK, zorder=21)

    # --- descriptor rows under the box -------------------------------------
    rows_y = (0.225, 0.175, 0.125)
    if mode in {"descriptor", "network", "energy", "force", "accel", "velocity", "move"} or not video:
        header_alpha = 1.0 if mode != "descriptor" else min(progress / 0.15, 1.0)
        registry.text(ax, 0.03, 0.275, config["descriptor_header"], ha="left", va="center", fontsize=14 if video else 10, color=EDGE_NAVY if mode == "descriptor" else DARK_GRAY, alpha=header_alpha, zorder=21)
        for k in range(labelled):
            if mode == "descriptor" and progress < (k + 0.15) / labelled:
                break
            r = float(local["distances"][k])
            element = data["elements"][int(local["ids"][k])]
            if model == "deepmd":
                row = local["deep_r"][k]
                text = rf"$j_{k + 1}$ {element} {r:.2f} Å → [{_fmt(row[0])}, {_fmt(row[1])}, {_fmt(row[2])}, {_fmt(row[3])}]"
            else:
                u = local["unit"][k]
                text = rf"$j_{k + 1}$ {element} {r:.2f} Å → $\hat{{u}}$ ({_fmt(u[0])}, {_fmt(u[1])}, {_fmt(u[2])})"
            registry.text(ax, 0.03, rows_y[k], text, ha="left", va="center", fontsize=14 if video else 10, color=INK if mode == "descriptor" else DARK_GRAY, zorder=21)

    # --- legends under the magnifier ---------------------------------------
    legend_y = 0.070
    if mode in {"network", "energy"}:
        gradient = np.linspace(0.0, 1.0, 256)[None, :]
        ax.imshow(gradient, cmap=EPS_CMAP, extent=(0.55, 0.78, legend_y - 0.012, legend_y + 0.012), origin="lower", aspect="auto", zorder=8)
        eps_range = scene["scales"]["eps"]["range"]
        registry.text(ax, 0.54, legend_y, f"−{eps_range:.2f}", ha="right", va="center", fontsize=12 if video else 10, color=DARK_GRAY, zorder=21)
        registry.text(ax, 0.79, legend_y, f"+{eps_range:.2f} eV", ha="left", va="center", fontsize=12 if video else 10, color=DARK_GRAY, zorder=21)
        registry.text(ax, 0.975, legend_y + 0.055, r"colour: $\varepsilon_j$ − species mean $\varepsilon$", ha="right", va="center", fontsize=12 if video else 10, color=DARK_GRAY, zorder=21)
    elif mode == "force":
        peak = scene["scales"]["force"]["max_magnitude"]
        registry.text(ax, 0.975, legend_y, rf"$\mathbf{{F}}_i$ on all atoms in $r_c$ · longest arrow = {peak:.2f} eV/Å", ha="right", va="center", fontsize=12 if video else 10, color=FORCE_OLIVE, zorder=21)
    elif mode == "accel":
        peak = scene["scales"]["accel"]["max_magnitude"]
        registry.text(ax, 0.975, legend_y, rf"$\mathbf{{a}}_i$ on all atoms in $r_c$ · longest arrow = {peak * 1.0e3:.1f}×10$^{{-3}}$ Å/fs²", ha="right", va="center", fontsize=12 if video else 10, color=ACCEL_PLUM, zorder=21)
    elif mode == "velocity":
        peak = scene["scales"]["velocity"]["max_magnitude"]
        registry.text(ax, 0.975, legend_y, rf"$\mathbf{{v}}_{{n+1/2}}$ on all atoms in $r_c$ · longest arrow = {peak * 1.0e2:.1f}×10$^{{-2}}$ Å/fs", ha="right", va="center", fontsize=12 if video else 10, color=VELOCITY_EMERALD, zorder=21)
    elif mode == "move":
        peak = scene["scales"]["move"]["max_magnitude"]
        registry.text(ax, 0.975, legend_y, rf"$\mathbf{{r}}_{{n+1}} - \mathbf{{r}}_n$ (atoms at $\mathbf{{r}}_{{n+1}}$) · longest = {peak * 1.0e2:.1f}×10$^{{-2}}$ Å", ha="right", va="center", fontsize=12 if video else 10, color=POSITION_LAKE, zorder=21)
    elif mode in {"neighbours", "descriptor"}:
        registry.text(ax, 0.975, legend_y, f"faded: outside $r_c$ · navy edges: O126 → {n_neigh} neighbours", ha="right", va="center", fontsize=12 if video else 10, color=DARK_GRAY, alpha=sphere_alpha if mode == "neighbours" else 1.0, zorder=21)
    elif mode == "positions":
        registry.text(ax, 0.975, legend_y, "magnified view around O126 (navy)", ha="right", va="center", fontsize=12 if video else 10, color=DARK_GRAY, zorder=21)


# --------------------------------------------------------------------------
# panel C: model energy
# --------------------------------------------------------------------------
def draw_energy(ax: plt.Axes, registry: LayoutRegistry, data: dict[str, object], *, model: str, video: bool, mode: str, state: int, progress: float) -> None:
    config = MODELS[model]
    if not video:
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, ec=LINE_GRAY, lw=1.1, zorder=2))
    energies = data["total_energy_ev"]
    eps_centre = float(data["atomic_energy_ev"][state, data["central_index"]])
    pre_energy = mode in {"positions", "neighbours", "descriptor", "network"}
    if pre_energy:
        status = config["provider"] + r": building $\varepsilon_i$ for step " + f"{state + 1:02d}"
    else:
        status = rf"$\varepsilon_{{\mathrm{{O126}}}}$ = {_neg(eps_centre)} eV · $E$ = {_neg(float(energies[state]))} eV"
    registry.text(ax, 0.50, 0.90, status, ha="center", va="center", fontsize=14 if video else 10, color=INK, zorder=4)

    steps = np.arange(1, len(energies) + 1, dtype=float)
    visible = state if pre_energy else state + 1
    plot_ax = ax.inset_axes((0.27, 0.24, 0.66, 0.52))
    plot_ax.plot(steps, energies, color="#D5D8DC", lw=2.0 if video else 1.1, marker="o", markersize=3.5 if video else 2.0, zorder=1)
    if visible > 0:
        plot_ax.plot(steps[:visible], energies[:visible], color=NAVY, lw=2.8 if video else 1.5, marker="o", markersize=4.0 if video else 2.2, zorder=2)
        marker_colour = GREEN if mode == "energy" else NAVY
        plot_ax.scatter([steps[visible - 1]], [energies[visible - 1]], s=70 if video else 20, color=marker_colour, edgecolors=WHITE, linewidths=1.0 if video else 0.5, zorder=3)
    lo, hi = float(energies.min()), float(energies.max())
    pad = max(0.12 * (hi - lo), 0.05)
    plot_ax.set_xlim(0.6, len(energies) + 0.4)
    plot_ax.set_ylim(lo - pad, hi + pad)
    plot_ax.set_xticks(steps)
    ticks = np.linspace(lo, hi, 3)
    plot_ax.set_yticks(ticks)
    plot_ax.set_yticklabels([f"{tick:.1f}".replace("-", "\u2212") for tick in ticks])
    font_size = 16 if video else 10
    plot_ax.tick_params(axis="both", labelsize=font_size, colors=DARK_GRAY, width=1.0)
    plot_ax.set_xlabel("MD step", fontsize=font_size, color=INK, labelpad=3)
    plot_ax.set_ylabel(r"$E$ (eV)", fontsize=font_size, color=INK, labelpad=3)
    plot_ax.grid(axis="y", color="#E6E8EA", lw=0.8, zorder=0)
    for spine in plot_ax.spines.values():
        spine.set_color(LINE_GRAY)
        spine.set_linewidth(1.2 if video else 0.8)


# --------------------------------------------------------------------------
# panel D: from energy to motion
# --------------------------------------------------------------------------
CHAIN_ROWS_Y = (0.88, 0.70, 0.52, 0.34, 0.16)
CHAIN_NODE_X = 0.085
CHAIN_NODE_RX = 0.028
CHAIN_NODE_RX_STATIC = 0.030


def draw_chain(ax: plt.Axes, registry: LayoutRegistry, data: dict[str, object], *, video: bool, mode: str, state: int) -> None:
    if not video:
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, ec=LINE_GRAY, lw=1.1, zorder=2))
    centre = data["central_index"]
    energy = float(data["total_energy_ev"][state])
    force = data["forces_ev_per_angstrom"][state, centre]
    accel = data["accelerations"][state, centre]
    dt = data["dt_fs"]
    active = CHAIN_ACTIVE[mode]
    reached = -1 if active is None else active
    if mode == "move":
        reached = 4
    a_norm = float(np.linalg.norm(accel))
    exponent = int(np.floor(np.log10(a_norm))) if a_norm > 0 else 0
    mantissa = a_norm / 10**exponent if a_norm > 0 else 0.0
    rows = (
        (r"$E$", r"$E = \sum_i \varepsilon_i$", f"{_neg(energy)} eV", GREEN),
        (r"$\nabla$", r"$\partial E/\partial \mathbf{r}_i$ (autodiff)", "every atom at once", INK),
        (r"$\mathbf{F}$", r"$\mathbf{F}_i = -\partial E/\partial \mathbf{r}_i$", rf"$|\mathbf{{F}}_{{\mathrm{{O126}}}}|$ = {np.linalg.norm(force):.3f} eV/Å", FORCE_OLIVE),
        (r"$\mathbf{a}$", r"$\mathbf{a}_i = \mathbf{F}_i / m_i$", rf"$|\mathbf{{a}}_{{\mathrm{{O126}}}}|$ = {mantissa:.1f}×10$^{{{exponent}}}$ Å/fs$^2$", ACCEL_PLUM),
        (r"$\Delta t$", r"$\mathbf{v}$ += ½$\mathbf{a}\Delta t$,  $\mathbf{r}$ += $\mathbf{v}\Delta t$", f"Δt = {dt:.1f} fs → step {min(state + 2, len(data['total_energy_ev'])):02d}", POSITION_LAKE),
    )
    aspect = _axes_aspect(ax)
    node_rx = CHAIN_NODE_RX if video else CHAIN_NODE_RX_STATIC
    node_ry = node_rx * aspect
    # One vertical spine carries the information downward; the part already
    # computed at this instant is drawn in ink, the rest in construction grey.
    ax.add_line(Line2D([CHAIN_NODE_X, CHAIN_NODE_X], [CHAIN_ROWS_Y[0], 0.0], color=LINE_GRAY, lw=2.4 if video else 1.5, zorder=2))
    if reached >= 0:
        ax.add_line(Line2D([CHAIN_NODE_X, CHAIN_NODE_X], [CHAIN_ROWS_Y[0], CHAIN_ROWS_Y[reached] if reached < 4 else 0.0], color=INK, lw=2.8 if video else 1.7, zorder=3))
    for index, (symbol, label, value, colour) in enumerate(rows):
        y = CHAIN_ROWS_Y[index]
        weight = 1.0 if (active is not None and index == active) else 0.0
        done = index <= reached
        fill = mix_hex(WHITE, INK, weight)
        edge = INK if done else LINE_GRAY
        ax.add_patch(Ellipse((CHAIN_NODE_X, y), 2.0 * node_rx, 2.0 * node_ry, fc=fill, ec=edge, lw=2.2 if video else 1.4, zorder=4))
        registry.text(ax, CHAIN_NODE_X, y, symbol, ha="center", va="center", fontsize=14 if video else 10, color=WHITE if weight > 0.48 else (INK if done else DARK_GRAY), weight="bold", zorder=5)
        # The A4 column is too narrow for label and value on one line, so the
        # still stacks the value under its label.
        label_y, value_xy, value_ha = (y, (0.985, y), "right") if video else (y + 0.030, (0.16, y - 0.035), "left")
        registry.text(ax, 0.16, label_y, label, ha="left", va="center", fontsize=14 if video else 10, color=INK if done else DARK_GRAY, weight="bold" if weight > 0.48 else "normal", zorder=5)
        registry.text(ax, *value_xy, value, ha=value_ha, va="center", fontsize=14 if video else 10, color=colour if done else DARK_GRAY, zorder=5)


def draw_return_path(fig: plt.Figure, registry: LayoutRegistry, panel_a: plt.Axes, panel_d: plt.Axes, *, video: bool, mode: str, provider: str) -> None:
    """Arrow from the bottom chain node back into the integrator 'a' node."""
    active = mode in {"velocity", "move"}
    colour = INK if active else "#A9B0B4" if video else LINE_GRAY
    y_return = RETURN_Y_VIDEO if video else RETURN_Y_STATIC
    to_fig = fig.transFigure.inverted().transform
    node_fig = to_fig(panel_d.transAxes.transform((CHAIN_NODE_X, 0.0)))
    # The 'a' node of the shared loop sits at angle -30 deg on the ellipse.
    a_aspect = _axes_aspect(panel_a)
    a_x = 0.50 + 0.40 * np.cos(np.deg2rad(-30.0))
    a_y = 0.58 + 0.40 * a_aspect * np.sin(np.deg2rad(-30.0))
    label_bottom = a_y - 0.075 * a_aspect - 0.035 - (0.075 if video else 0.055)
    a_fig = to_fig(panel_a.transAxes.transform((a_x, label_bottom)))
    lw = 2.6 if video else 1.6
    fig.add_artist(Line2D([node_fig[0], node_fig[0]], [node_fig[1], y_return], transform=fig.transFigure, color=colour, lw=lw, zorder=50))
    fig.add_artist(Line2D([node_fig[0], a_fig[0]], [y_return, y_return], transform=fig.transFigure, color=colour, lw=lw, zorder=50))
    arrow = FancyArrowPatch((a_fig[0], y_return), (a_fig[0], a_fig[1]), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=20 if video else 14, lw=lw, color=colour, zorder=50)
    fig.add_artist(arrow)
    registry.arrows.append(arrow)
    label = fig.text(0.50, y_return, f"updated velocities and positions return to the integrator; only the {provider.split(' ·')[0]} module computed $\\mathbf{{a}}$", ha="center", va="center", fontsize=16 if video else 10, fontfamily=registry.font_family, color=INK if active else DARK_GRAY, bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec="none"), zorder=51)
    registry.texts.append(label)


# --------------------------------------------------------------------------
# timeline
# --------------------------------------------------------------------------
def _phase_at(local: float, phases: tuple[tuple[str, float], ...]) -> tuple[str, float]:
    cursor = 0.0
    for name, length in phases:
        if local < cursor + length:
            return name, (local - cursor) / length
        cursor += length
    name, length = phases[-1]
    return name, 1.0


def video_state(time_seconds: float, n_states: int) -> dict[str, object]:
    bounded = float(np.clip(time_seconds, 0.0, VIDEO_DURATION - 1.0e-9))
    n_updates = n_states - 1
    if bounded < 2.0 * DETAILED_BLOCK:
        state = int(bounded // DETAILED_BLOCK)
        mode, progress = _phase_at(bounded - state * DETAILED_BLOCK, DETAILED_PHASES)
        return {"mode": mode, "state": state, "progress": float(np.clip(progress, 0.0, 1.0)), "rapid": False}
    rapid_time = bounded - 2.0 * DETAILED_BLOCK
    rapid_states = list(range(2, n_updates))
    if not rapid_states:
        rapid_states = [n_updates - 1]
    index = int(rapid_time // RAPID_BLOCK) % len(rapid_states)
    mode, progress = _phase_at(rapid_time % RAPID_BLOCK, RAPID_PHASES)
    return {"mode": mode, "state": rapid_states[index], "progress": float(np.clip(progress, 0.0, 1.0)), "rapid": True}


def semantics_for(mode: str) -> list[dict]:
    return {
        "positions": [{"id": "centre_atom", "color": CENTRE_NAVY, "min_pixels": 30, "tolerance": 60}],
        "neighbours": [{"id": "centre_atom", "color": CENTRE_NAVY, "min_pixels": 30, "tolerance": 60}],
        "descriptor": [{"id": "neighbour_edges", "color": EDGE_NAVY, "min_pixels": 60, "tolerance": 60}],
        "network": [{"id": "centre_atom_or_eps", "color": CENTRE_NAVY, "min_pixels": 30, "tolerance": 60}],
        "energy": [{"id": "energy_marker", "color": GREEN, "min_pixels": 40}],
        "force": [{"id": "model_force", "color": FORCE_OLIVE, "min_pixels": 120}],
        "accel": [{"id": "acceleration", "color": ACCEL_PLUM, "min_pixels": 120}],
        "velocity": [{"id": "half_step_velocity", "color": VELOCITY_EMERALD, "min_pixels": 120}],
        "move": [{"id": "displacement", "color": POSITION_LAKE, "min_pixels": 120}],
    }[mode]


def draw_frame(fig: plt.Figure, time_seconds: float, registry: LayoutRegistry, data: dict[str, object], scene: dict[str, object], *, model: str) -> list[dict]:
    state_info = video_state(time_seconds, data["positions"].shape[0])
    mode, state, progress, rapid = state_info["mode"], state_info["state"], state_info["progress"], state_info["rapid"]
    provider = MODELS[model]["provider"]
    panel_a = axes_from_top_slot(fig, VIDEO_A)
    panel_b = axes_from_top_slot(fig, VIDEO_B)
    panel_c = axes_from_top_slot(fig, VIDEO_C)
    panel_d = axes_from_top_slot(fig, VIDEO_D)
    draw_left(panel_a, registry, video=True, mode=mode, provider=provider)
    draw_case(panel_b, registry, data, scene, model=model, video=True, mode=mode, state=state, progress=progress, rapid=rapid)
    draw_energy(panel_c, registry, data, model=model, video=True, mode=mode, state=state, progress=progress)
    draw_chain(panel_d, registry, data, video=True, mode=mode, state=state)
    draw_return_path(fig, registry, panel_a, panel_d, video=True, mode=mode, provider=provider)
    if not rapid and mode in {"positions", "neighbours", "descriptor", "network"}:
        _deemphasize(panel_d)
    return semantics_for(mode)


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def render_static(model: str, data: dict[str, object], scene: dict[str, object]) -> None:
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    provider = MODELS[model]["provider"]
    panel_a = axes_from_top_slot(fig, STATIC_A)
    panel_b = axes_from_top_slot(fig, STATIC_B)
    panel_c = axes_from_top_slot(fig, STATIC_C)
    panel_d = axes_from_top_slot(fig, STATIC_D)
    draw_left(panel_a, registry, video=False, mode="force", provider=provider)
    draw_case(panel_b, registry, data, scene, model=model, video=False, mode="force", state=1, progress=1.0, rapid=False)
    draw_energy(panel_c, registry, data, model=model, video=False, mode="force", state=1, progress=1.0)
    draw_chain(panel_d, registry, data, video=False, mode="force", state=1)
    draw_return_path(fig, registry, panel_a, panel_d, video=False, mode="force", provider=provider)
    errors = registry.validate(fig)
    if errors:
        debug = ROOT / "qa" / MODELS[model]["stem"] / "_qa" / "static_failed.png"
        debug.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(debug, dpi=100, facecolor=WHITE)
        texts = [f"text[{i}]={a.get_text()!r}" for i, a in enumerate(registry.texts)]
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors) + f"\n(debug image {debug})\n" + "\n".join(texts))
    save_static(fig, MODELS[model]["stem"])


KEYFRAME_TIMES = (0.30, 1.20, 1.85, 2.60, 3.40, 4.30, 4.90, 5.60, 6.40, 7.10, 7.90, 8.70, 9.30, 11.00, 12.60, 14.10, 15.80, 17.60, 18.10, 18.60, 19.10, 19.40, 20.90, 24.20, 27.70, 29.80)


def render_keyframes(model: str, data: dict[str, object], scene: dict[str, object], qa_dir: Path) -> Path:
    output_dir = qa_dir / "_qa" / "keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    records = []
    failures: list[str] = []
    for index, time_seconds in enumerate(KEYFRAME_TIMES):
        fig = new_video_figure()
        registry = LayoutRegistry(min_font_pt=16, max_font_pt=18, edge_pad_px=12, font_family="Arial", coerce_min_font=True)
        semantics = draw_frame(fig, time_seconds, registry, data, scene, model=model)
        errors = registry.validate(fig)
        path = output_dir / f"frame_{index:02d}_{time_seconds:05.2f}s.png"
        fig.savefig(path, dpi=100, facecolor=WHITE)
        plt.close(fig)
        if errors:
            failures.append(f"Keyframe {time_seconds:.2f} s failed layout:\n" + "\n".join(errors))
        images.append(Image.open(path).convert("RGB").resize((640, 200)))
        records.append({"time_seconds": time_seconds, "path": str(path), "state": video_state(time_seconds, data["positions"].shape[0]), "semantics": semantics})
    columns = 3
    rows = int(np.ceil(len(images) / columns))
    contact = Image.new("RGB", (columns * 640, rows * 200), WHITE)
    for index, item in enumerate(images):
        contact.paste(item, ((index % columns) * 640, (index // columns) * 200))
    contact_path = output_dir / "_contact.png"
    contact.save(contact_path)
    json_dump(output_dir / "keyframes.json", {"frames": records})
    if failures:
        raise RuntimeError(f"{len(failures)} keyframe(s) failed layout (contact sheet at {contact_path}):\n" + "\n".join(failures))
    return contact_path


def render_animation(model: str, data: dict[str, object], scene: dict[str, object], qa_dir: Path) -> None:
    audit_config = {
        "panels": [
            {"id": "integrator", "rect": list(VIDEO_A), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            {"id": "system", "rect": list(VIDEO_B), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            {"id": "energy", "rect": list(VIDEO_C), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            {"id": "chain", "rect": list(VIDEO_D), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.020, "min_panel_bbox_fill": 0.22, "grid_rows": 12, "grid_columns": 20},
        "bands": [
            {"id": "gap_a_b", "rect": [0.215, 0.025, 0.230, 0.900], "max_ink_pixels": 5000},
            {"id": "gap_b_right", "rect": [0.680, 0.025, 0.695, 0.900], "max_ink_pixels": 5000},
        ],
    }
    render_video(
        stem=MODELS[model]["stem"],
        duration_seconds=VIDEO_DURATION,
        draw_frame=lambda fig, time, _index, registry: draw_frame(fig, time, registry, data, scene, model=model),
        audit_config=audit_config,
        qa_directory=qa_dir / "_qa",
        representative_times=KEYFRAME_TIMES,
    )


def write_provenance(model: str, data: dict[str, object], scene: dict[str, object], qa_dir: Path) -> None:
    centre = data["central_index"]
    local0 = local_environment(data, 0)
    payload = {
        "schema": "nnmd_end_to_end_story/v2",
        "model": model,
        "stem": MODELS[model]["stem"],
        "trajectory": str(MODELS[model]["data"]),
        "trajectory_sha256": sha256_file(MODELS[model]["data"]),
        "trajectory_metadata": data["metadata"],
        "n_states": int(data["positions"].shape[0]),
        "centre_atom": f"{data['elements'][centre]}{centre}",
        "cutoff_angstrom": data["cutoff"],
        "neighbour_counts": [int(value) for value in data["neighbour_counts"]],
        "labelled_neighbours_state0": [
            {"j": int(local0["ids"][k]), "element": str(data["elements"][int(local0["ids"][k])]), "r_angstrom": float(local0["distances"][k]), "s": float(local0["s"][k]), "unit": local0["unit"][k].tolist(), "deep_r_row": local0["deep_r"][k].tolist()}
            for k in range(min(3, len(local0["ids"])))
        ],
        "total_energy_ev": data["total_energy_ev"].tolist(),
        "centre_atomic_energy_ev": data["atomic_energy_ev"][:, centre].tolist(),
        "centre_force_ev_per_angstrom": data["forces_ev_per_angstrom"][:, centre].tolist(),
        "centre_acceleration_angstrom_fs2": data["accelerations"][:, centre].tolist(),
        "smooth_switch": {"form": "DeepPot-SE s(r) = 1/r (r < rcs), (1/r)[u^3(-6u^2+15u-10)+1] (rcs <= r < rc), 0 (r >= rc)", "rcs_angstrom": R_CUT_SMOOTH, "rc_angstrom": data["cutoff"], "note": "shown as a geometry transform for both stories; DPA4C's own radial basis is not reproduced"},
        "eps_colouring": "atom colour = eps_j minus the mean eps of its species at that state, clipped to +-%.3f eV" % scene["scales"]["eps"]["range"],
        "display_scales": scene["scales"],
        "timeline": {"duration_seconds": VIDEO_DURATION, "detailed_block_seconds": DETAILED_BLOCK, "detailed_phases": DETAILED_PHASES, "rapid_block_seconds": RAPID_BLOCK, "rapid_phases": RAPID_PHASES},
    }
    json_dump(qa_dir / "story_provenance.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    models = tuple(MODELS) if args.model == "all" else (args.model,)
    for model in models:
        data = load_data(model)
        qa_dir = ROOT / "qa" / MODELS[model]["stem"]
        scene = prepare_assets(model, data, qa_dir)
        write_provenance(model, data, scene, qa_dir)
        if args.assets_only:
            continue
        if args.preview_only:
            print(render_keyframes(model, data, scene, qa_dir))
            continue
        render_static(model, data, scene)
        if not args.static_only:
            render_animation(model, data, scene, qa_dir)


if __name__ == "__main__":
    main()
