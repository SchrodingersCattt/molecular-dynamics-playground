"""End-to-end neural-network MD stories (DeepMD and DPA4C) in the AIMD layout.

Both stories share the outer Velocity-Verlet loop of the AIMD movie and the
same retained 64-water periodic box.  What differs is the pluggable module
that turns positions into accelerations:

* DeepMD: neighbours -> DeepPot-SE environment matrix R~_i -> embedding net
  (25-50-100) per row -> T = R~^T G / N -> D = T^T T< -> fitting net -> eps_i
* DPA4C: neighbours -> radial basis e(r) and harmonics Y_lm(u) per row ->
  radial MLP and pair FiLM -> amplitude phi -> sum_j moments X^(l) ->
  invariants -> fitting net -> eps_i

The right-hand panel draws every one of these tensors for the centre atom
O126 from the dumped forward pass of the model (``*_internals.npz``), so the
matrix that the neighbours fly into, each embedding layer, the contraction,
the descriptor and the fitting activations are model output.  After E the two
movies are identical: F = -dE/dr is sent back through the same blocks, a =
F/m enters the integrator loop.  Atoms, cutoff spheres, neighbour edges and
vectors are rendered by MatterVis in world space; matplotlib only composes.
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
    A_ORANGE,
    CRIMSON,
    DARK_GRAY,
    ENERGY_TEAL,
    FONT_SIZES,
    INK,
    LINE_GRAY,
    NAVY,
    R_BLUE,
    V_PURPLE,
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
    text_scale_of,
)
from mattervis_story import (
    SceneCamera,
    _composition_rgba,
    camera_for_source,
    draw_arrow_legend,
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
        "internals": DATA_DIR / "dpmd_internals.npz",
        "provider": "DeepMD · DeepPot-SE",
        "titles": {
            "gather": r"Neighbour rows → environment matrix $\tilde{R}_i$",
            "embed": "Embedding net on every row",
        },
    },
    "dpa4c": {
        "stem": "04_4c_dpa4c",
        "data": DATA_DIR / "dpa4c_water_box_trajectory.npz",
        "metadata": DATA_DIR / "dpa4c_water_box_trajectory.json",
        "internals": DATA_DIR / "dpa4c_internals.npz",
        "provider": "DPA4C · equivariant descriptor",
        "titles": {
            "gather": r"Neighbour rows → $Y_{lm}(\hat{u})$ and $e(r)$",
            "embed": r"Radial MLP → amplitude $\phi$ on every row",
        },
    },
}

# Layout: integrator loop (A), real system (B), operator pipeline (D).  The
# bottom strip of the canvas carries the return arrow into the 'a' node.
VIDEO_A = (0.015, 0.025, 0.200, 0.925)
VIDEO_B = (0.210, 0.025, 0.590, 0.925)
VIDEO_D = (0.600, 0.025, 0.985, 0.925)
STATIC_A = (0.035, 0.055, 0.215, 0.905)
STATIC_B = (0.225, 0.045, 0.595, 0.905)
STATIC_D = (0.605, 0.045, 0.965, 0.905)
RETURN_Y_VIDEO = 0.045
RETURN_Y_STATIC = 0.030

# Shared r/v/a palette of every movie: structure and descriptor in navy,
# energy in teal, force and acceleration in orange.
POSITION_LAKE = R_BLUE
VELOCITY_EMERALD = V_PURPLE
ACCEL_PLUM = A_ORANGE
EDGE_NAVY = NAVY
SPHERE_TEAL = "#5B7A99"
CENTRE_NAVY = "#183153"
INACTIVE_PATH = "#B4BABE"
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

FORCE_OLIVE = A_ORANGE
LOOP_CENTRE_Y = 0.58
LOOP_RADIUS_X = 0.36
LEGEND = {
    "force": (r"force $\mathbf{F}$", FORCE_OLIVE),
    "accel": (r"acceleration $\mathbf{a}$", ACCEL_PLUM),
    "velocity": (r"velocity $\mathbf{v}$", VELOCITY_EMERALD),
    "move": (r"displacement $\mathbf{v}\Delta t$", POSITION_LAKE),
}

VIDEO_DURATION = 25.0
DETAILED_PHASES = (
    ("positions", 0.8),
    ("neighbours", 1.0),
    ("gather", 1.7),
    ("embed", 2.1),
    ("contract", 1.7),
    ("fit", 1.4),
    ("energy", 0.7),
    ("force", 1.5),
    ("accel", 0.7),
    ("velocity", 0.6),
    ("move", 0.7),
)
DETAILED_BLOCK = sum(length for _, length in DETAILED_PHASES)
# The rapid pass is a separate storyboard.  Each state is a direct visual
# checkpoint; neighbouring stages are intentionally short and never blend into
# one another as the slow pass does.
SLOW_HOLD_SECONDS = 3.0
RAPID_CYCLE_SECONDS = 2.0
RAPID_STATES = (1, 2, 3, 4)
RAPID_PHASES = (
    ("positions", 0.04),
    ("neighbours", 0.04),
    ("gather", 0.06),
    ("embed", 0.22),
    ("contract", 0.12),
    ("fit", 0.07),
    ("energy", 0.03),
    ("force", 0.12),
    ("accel", 0.08),
    ("velocity", 0.08),
    ("move", 0.14),
)
# Fractions of one rapid cycle: the horizontal wipe spans "embed", the
# vertical wipe spans "contract" + "fit" + "energy".
_RAPID_EDGES = np.cumsum([0.0] + [length for _, length in RAPID_PHASES])
_RAPID_INDEX = {name: index for index, (name, _) in enumerate(RAPID_PHASES)}
SWEEP_H = (_RAPID_EDGES[_RAPID_INDEX["embed"]], _RAPID_EDGES[_RAPID_INDEX["embed"] + 1])
SWEEP_V = (_RAPID_EDGES[_RAPID_INDEX["contract"]], _RAPID_EDGES[_RAPID_INDEX["energy"] + 1])
SWEEP_TOP = 0.815
SWEEP_BOTTOM = 0.10
FINAL_STATE_HOLD_SECONDS = 1.0
FLOW_ORDER = tuple(name for name, _ in DETAILED_PHASES)
MODE_TO_LOOP_STAGE = {name: 1 for name in FLOW_ORDER} | {"velocity": 2, "move": 0}

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
    data["owner"] = water_owner(data)
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


def water_owner(data: dict[str, object]) -> np.ndarray:
    """Index of the O that owns each atom (an O owns itself), from state 0."""
    positions = data["positions"][0]
    elements = data["elements"]
    box = data["box_length"]
    oxygens = np.flatnonzero(elements == "O")
    owner = np.arange(len(elements))
    for atom in np.flatnonzero(elements == "H"):
        distances = np.linalg.norm(minimum_image(positions[oxygens] - positions[atom], box), axis=1)
        owner[atom] = int(oxygens[np.argmin(distances)])
    return owner


def focus_coordinates(data: dict[str, object], state: int, keep: list[int]) -> np.ndarray:
    """Positions of the kept atoms around O126 with every water left whole."""
    positions = data["positions"][state]
    box = data["box_length"]
    origin = data["positions"][0, data["central_index"]]
    owner = data["owner"]
    out = np.empty((len(keep), 3))
    for row, atom in enumerate(keep):
        oxygen = int(owner[atom])
        o_local = minimum_image(positions[oxygen] - origin, box)
        out[row] = o_local + minimum_image(positions[atom] - positions[oxygen], box)
    return out


def write_sources(data: dict[str, object], source_dir: Path) -> tuple[Path, Path, list[int]]:
    """Write the periodic box and the O126-centred local source (all states).

    The local source keeps whole water molecules, so no H is drawn without
    its O inside the magnifier.
    """
    from ase import Atoms
    from ase.io import write

    source_dir.mkdir(parents=True, exist_ok=True)
    positions = data["positions"]
    elements = data["elements"]
    box = data["box_length"]
    centre = data["central_index"]
    owner = data["owner"]
    origin = positions[0, centre]
    delta0 = minimum_image(positions[0] - origin, box)
    near_oxygens = set(int(owner[index]) for index in np.flatnonzero(np.linalg.norm(delta0, axis=1) <= FOCUS_RADIUS))
    keep = [int(index) for index in range(len(elements)) if int(owner[index]) in near_oxygens]
    keep = [int(centre)] + [index for index in keep if index != centre]
    box_frames = []
    focus_frames = []
    for state in range(positions.shape[0]):
        box_frames.append(Atoms(symbols=elements.tolist(), positions=positions[state], cell=np.eye(3) * box, pbc=True))
        local = focus_coordinates(data, state, keep)
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
    equator = make_torus_mesh(np.zeros(3), cutoff * 0.998, 0.035, normal=direction, color="#2B4766", opacity=0.62, major_steps=72, tube_steps=6, mesh_id="r_c_equator")
    meridian = make_torus_mesh(np.zeros(3), cutoff * 0.998, 0.024, normal=up, color=SPHERE_TEAL, opacity=0.42, major_steps=72, tube_steps=6, mesh_id="r_c_meridian")
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
        focus_positions = focus_coordinates(data, state, keep)
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
        records.append(render_structure(focus_source, path("plain"), frame=state, mesh_overlays=shell, atom_color_overrides=centre_colour, **common_focus))
        records.append(render_structure(focus_source, path("cut"), frame=state, show_bonds=False, mesh_overlays=shell + edges, atom_opacity_scales=fade, atom_color_overrides=centre_colour, **common_focus))
        records.append(render_structure(focus_source, path("eps"), frame=state, show_bonds=False, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=eps_colours, **common_focus))
        force = data["forces_ev_per_angstrom"][state][inside_list]
        records.append(render_structure(focus_source, path("force"), frame=state, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=centre_colour, vector_overlays=arrows(f"force-{state}", inside_pos, force, scales["force"]["scale"], FORCE_OLIVE), **common_focus))
        accel = data["accelerations"][state][inside_list]
        records.append(render_structure(focus_source, path("accel"), frame=state, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=centre_colour, vector_overlays=arrows(f"accel-{state}", inside_pos, accel, scales["accel"]["scale"], ACCEL_PLUM), **common_focus))
        if state < n_states - 1:
            half = data["half_velocities"][state][inside_list]
            records.append(render_structure(focus_source, path("velocity"), frame=state, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=centre_colour, vector_overlays=arrows(f"velocity-{state}", inside_pos, half, scales["velocity"]["scale"], VELOCITY_EMERALD), **common_focus))
            disp = data["mic_displacements"][state][inside_list]
            next_focus = {**common_focus, "camera": focus_cameras[state + 1]}
            records.append(render_structure(focus_source, path("move"), frame=state + 1, mesh_overlays=shell, atom_opacity_scales=fade, atom_color_overrides=centre_colour, vector_overlays=arrows(f"move-{state}", inside_pos, disp, scales["move"]["scale"], POSITION_LAKE), **next_focus))
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
    draw_vv_loop(ax, registry, video=video, active_stage=stage, centre_text=equation, centre_y=LOOP_CENTRE_Y, radius_x=LOOP_RADIUS_X)


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
) -> dict[str, object]:
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
    box_x, box_y, box_r = (0.140, 0.560, 0.120) if video else (0.150, 0.580, 0.130)
    box_rect = _square_rect(ax, (box_x, box_y), box_r)
    _place_image(ax, _composition_rgba(assets["box"][min(state, n_states - 1)]), box_rect, zorder=4)
    centre_world = positions[state, centre]
    projected_centre = project_world(centre_world[None, :], camera=box_camera, rect=box_rect, image_aspect=1.0)[0]
    aspect = _axes_aspect(ax)
    radius_axes = data["cutoff"] / (2.0 * BOX_ORTHO) * (box_rect[2] - box_rect[0])
    ax.add_patch(Ellipse(tuple(projected_centre), 2.0 * radius_axes, 2.0 * radius_axes * aspect, fill=False, ec=SPHERE_TEAL, lw=2.4 if video else 1.4, alpha=0.9, zorder=9))
    registry.text(ax, box_x, box_rect[3] + (0.035 if video else 0.015), "64 H$_2$O periodic box", ha="center", va="bottom", fontsize=FONT_SIZES["micro"], color=DARK_GRAY, zorder=21)

    # --- magnifier (right) --------------------------------------------------
    mag_centre = (0.640, 0.505) if video else (0.640, 0.540)
    mag_rx = 0.295 if video else 0.335
    mag_rect = _square_rect(ax, mag_centre, mag_rx)
    mag_ry = mag_rx * aspect
    clip = Ellipse(mag_centre, 2.0 * mag_rx, 2.0 * mag_ry, transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)
    ax.add_patch(Ellipse(mag_centre, 2.0 * mag_rx, 2.0 * mag_ry, fc=WHITE, ec=LINE_GRAY, lw=2.0 if video else 1.2, zorder=3))
    # The in-situ circle, its guide lines and the magnified cutoff sphere are
    # one object and are always drawn together.
    for sign in (1.0, -1.0):
        start = (projected_centre[0] + radius_axes * 0.35, projected_centre[1] + sign * radius_axes * aspect * 0.94)
        end = (mag_centre[0] - mag_rx * 0.70, mag_centre[1] + sign * mag_ry * 0.71)
        ax.add_line(Line2D([start[0], end[0]], [start[1], end[1]], color=SPHERE_TEAL, lw=1.6 if video else 1.0, alpha=0.55, zorder=2))

    if mode == "positions":
        image = _composition_rgba(assets["plain"][state])
    elif mode == "neighbours":
        image = _composition_rgba(assets["cut"][state]) if rapid else _blend(assets["plain"][state], assets["cut"][state], smoothstep(progress))
    elif mode in {"gather", "embed", "contract"}:
        image = _composition_rgba(assets["cut"][state])
    elif mode == "fit":
        image = _composition_rgba(assets["eps"][state]) if rapid else _blend(assets["cut"][state], assets["eps"][state], smoothstep(min(progress / 0.6, 1.0)))
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

    focus_origin = positions[0, centre]
    local_vectors = local["vectors"]
    centre_local = minimum_image(positions[state, centre] - focus_origin, box)

    # --- stage title and step label ---------------------------------------
    title_colour = INK
    n_neigh = len(local["ids"])
    cutoff = data["cutoff"]
    title = {
        "positions": r"Current positions $\mathbf{r}_n$",
        "neighbours": rf"Neighbours of O126 within $r_c$ = {cutoff:.0f} Å",
        "gather": config["titles"]["gather"],
        "embed": config["titles"]["embed"],
        "contract": "Sum over neighbours → descriptor",
        "fit": r"Fitting net → atomic energy $\varepsilon_i$",
        "energy": r"Total energy $E = \Sigma_i\,\varepsilon_i$",
        "force": r"$\mathbf{F}_i = -\partial E/\partial \mathbf{r}_i$",
        "accel": r"$\mathbf{a}_i = \mathbf{F}_i / m_i$",
        "velocity": r"Update velocity $\mathbf{v}_{n+1/2}$",
        "move": r"Update position $\mathbf{r}_{n+1}$",
    }["fit" if rapid and mode == "energy" else mode]
    registry.text(ax, 0.50, 0.955, title, ha="center", va="center", fontsize=FONT_SIZES["panel_title"], color=title_colour, zorder=21)
    step_xy = (0.035, 0.03) if video else (0.05, 0.05)
    registry.text(ax, *step_xy, f"Simulation step {state + 1:02d} ({state * data['dt_fs']:.1f} fs)", ha="left", va="bottom", fontsize=FONT_SIZES["micro"], color=INK, zorder=21)
    if mode in LEGEND:
        draw_arrow_legend(ax, registry, [LEGEND[mode]], x_right=0.97 if video else 0.95, y=step_xy[1], video=video)

    neighbour_world = np.vstack([centre_local + vector for vector in local_vectors])
    neighbour_xy = project_world(neighbour_world, camera=focus_camera, rect=mag_rect, image_aspect=1.0)
    return {
        "neighbour_xy": {int(atom): tuple(xy) for atom, xy in zip(local["ids"], neighbour_xy)},
        "mag_centre": mag_centre,
        "mag_radius": (mag_rx, mag_ry),
    }


# --------------------------------------------------------------------------
# panel D: operator pipeline built from the dumped forward pass of O126
# --------------------------------------------------------------------------
ROW_TOP = 0.815
ROW_BOTTOM = 0.16
LABEL_Y = 0.865
RIGHT_X = (0.54, 0.985)
EPS_Y = 0.215
ENERGY_Y = 0.135
OP_CMAP = LinearSegmentedColormap.from_list("op", [NAVY, "#F4F5F6", CRIMSON])
SPECIES_DOT = {"O": "#D2453A", "H": "#C9CED3"}
# DPA4C moment layout (model constants of the checkpoint): degree zero is the
# 64-channel amplitude, X^(1) uses harmonics 1-3 x channels 0-7 and X^(2)
# harmonics 4-8 x channels 0-3.
DPA4C_CHANNEL_INDEX = np.array(list(range(8)) * 3 + list(range(4)) * 5)
DPA4C_HARMONIC_INDEX = np.array([1] * 8 + [2] * 8 + [3] * 8 + [4] * 4 + [5] * 4 + [6] * 4 + [7] * 4 + [8] * 4)
DPA4C_DEGREE_NORM_FLOOR = 0.25
DPMD_N_SEL = 600.0  # sel = 200 O + 400 H


def _reached(mode: str, stage: str) -> bool:
    return FLOW_ORDER.index(mode) >= FLOW_ORDER.index(stage)


def _stage_p(mode: str, stage: str, progress: float) -> float:
    if mode == stage:
        return progress
    return 1.0 if _reached(mode, stage) else 0.0


def load_internals(model: str, data: dict[str, object]) -> dict[str, object]:
    """Per-state tensors of the centre atom, arranged as display blocks."""
    path = MODELS[model]["internals"]
    with np.load(path, allow_pickle=False) as archive:
        raw = {key: np.asarray(archive[key]) for key in archive.files}
    states = []
    for state in range(data["positions"].shape[0]):
        if model == "deepmd":
            count = int(raw["neighbour_count"][state])
            ids = raw["order"][state, :count].astype(int)
            rmat = raw["rmat"][state, :count]
            g = raw["embed_3"][state, :count]
            rows = [
                ("R", r"$\tilde{R}$", "4", rmat, "gather"),
                ("G1", r"$G^{1}$", "25", raw["embed_1"][state, :count], "embed"),
                ("G2", r"$G^{2}$", "50", raw["embed_2"][state, :count], "embed"),
                ("G3", r"$G$", "100", g, "embed"),
            ]
            T = raw["T"][state]
            # Real-row contributions of the neighbour sum; the padded slots
            # close it, so the last partial sum equals the dumped T.
            real = np.einsum("jk,jc->jkc", rmat, g) / DPMD_N_SEL
            partial = [("T", "", real, T - real.sum(axis=0))]
            invariant = (r"Descriptor $D=T^{\top}T_{<}$", raw["D"][state].T)
            fits = [raw["fit_1"][state], raw["fit_2"][state], raw["fit_3"][state]]
            fit_label = "Fitting net"
            epsilon = float(raw["epsilon"][state])
            pad_rows = int(raw["n_pad"][state])
        else:
            count = int(np.sum(np.isfinite(raw["distance"][state])))
            ids = raw["neighbour_ids"][state, :count].astype(int)
            rows = [
                ("Y", r"$Y_{lm}$", "9", raw["harmonics"][state, :count], "gather"),
                ("e", r"$e(r)$", "16", raw["radial_basis"][state, :count], "gather"),
                ("h", r"$h$", "176", raw["radial_hidden"][state, :count], "embed"),
                ("g", r"$g$", "64", raw["radial"][state, :count], "embed"),
                ("phi", r"$\phi$", "64", raw["amplitude"][state, :count], "embed"),
            ]
            amp = raw["amplitude"][state, :count]
            basis = raw["harmonics"][state, :count]
            env = raw["envelope"][state, :count]
            div = np.sqrt(np.array([np.sum(env**2), np.sum(env**4)]) + DPA4C_DEGREE_NORM_FLOOR)
            c0 = amp / div[0]
            c_hi = amp[:, DPA4C_CHANNEL_INDEX] * basis[:, DPA4C_HARMONIC_INDEX] * env[:, None] / div[1]
            if np.abs(np.concatenate([c0.sum(0), c_hi.sum(0)]) - raw["moments"][state]).max() > 1.0e-5:
                raise ValueError("DPA4C moment partial sums do not close to the dumped moments")
            partial = [
                ("X0", "", c0[:, None, :], np.zeros((1, 64))),
                ("X1", r"$X^{(1)}$", c_hi[:, :24].reshape(count, 3, 8), np.zeros((3, 8))),
                ("X2", r"$X^{(2)}$", c_hi[:, 24:].reshape(count, 5, 4), np.zeros((5, 4))),
            ]
            invariant = (r"Descriptor $D_i$", raw["descriptor"][state][None, :])
            sizes = raw["fit_layer_sizes"][state].astype(int)
            fits = np.split(raw["fit_layers"][state], np.cumsum(sizes)[:-1])[:-1]
            fit_label = "Fitting net"
            epsilon = float(raw["epsilon_model"][state])
            pad_rows = 0
        expected = set(int(j) for j in local_environment(data, state)["ids"])
        if set(ids.tolist()) != expected:
            raise ValueError(f"{model} internals state {state}: neighbour set differs from the trajectory")
        if abs(epsilon - float(data["atomic_energy_ev"][state, data["central_index"]])) > 1.0e-3:
            raise ValueError(f"{model} internals state {state}: epsilon differs from the trajectory")
        states.append({"ids": ids, "species": data["elements"][ids], "rows": rows, "partial": partial, "invariant": invariant, "fits": fits, "fit_label": fit_label, "epsilon": epsilon, "pad_rows": pad_rows})
    return {"states": states, "path": path}


def _heat(ax, matrix, rect, *, alpha=1.0, zorder=3.0, vmax=None, clip=None):
    """Signed heatmap; colour saturates at the block's 98th-percentile |value| unless vmax is given.

    ``clip`` (x0, y0, x1, y1) shows only that part of the block (sweep reveal).
    """
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    vmax = vmax or float(np.nanpercentile(np.abs(matrix), 98.0)) or float(np.nanmax(np.abs(matrix))) or 1.0
    rgba = OP_CMAP(0.5 + 0.5 * np.clip(matrix / vmax, -1.0, 1.0))
    rgba[..., 3] = alpha
    x0, y0, x1, y1 = rect
    image = ax.imshow(rgba, extent=(x0, x1, y0, y1), origin="upper", aspect="auto", interpolation="nearest", zorder=zorder)
    if clip is not None:
        cx0, cy0, cx1, cy1 = clip
        image.set_clip_path(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, transform=ax.transData))
    return image


def _frame(ax, rect, *, colour, lw=1.1, zorder=4.0):
    x0, y0, x1, y1 = rect
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=colour, lw=lw, zorder=zorder))


def _arrow(ax, start, end, *, colour, video, on=True, dashed=False, zorder=6.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14 if video else 9, lw=2.0 if video else 1.2, color=colour if on else LINE_GRAY, linestyle=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0, zorder=zorder))


def draw_pipeline(
    fig: plt.Figure,
    ax: plt.Axes,
    registry: LayoutRegistry,
    data: dict[str, object],
    internals: dict[str, object],
    case_geometry: dict[str, object],
    panel_b: plt.Axes,
    *,
    model: str,
    video: bool,
    mode: str,
    state: int,
    progress: float,
    rapid: bool = False,
    sweeps: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Row-aligned operator pipeline of O126 for one state.

    Returns the bottom centre of the E label in figure coordinates, where the
    force path starts.

    Left: one matrix row per neighbour in the model's own order; each block to
    the right is the model's activation for the same rows.  Right column: the
    neighbour sum, the invariant descriptor, the fitting layers and eps_i.
    During the gather stage every neighbour in the magnifier flies into its
    row at the same time, so the matrix is assembled in one step.
    """
    fs = FONT_SIZES["body"]
    record = internals["states"][state]
    rows = record["rows"]
    n = len(record["ids"])
    p = progress if rapid and mode in {"embed", "contract"} else (1.0 if rapid else (progress if video else 1.0))
    gather_p = _stage_p(mode, "gather", p)
    embed_p = _stage_p(mode, "embed", p)
    contract_p = _stage_p(mode, "contract", p)
    fit_p = _stage_p(mode, "fit", p)
    backward = mode == "force"
    filled = 1.0 if rapid and gather_p > 0.0 else smoothstep(float(np.clip((gather_p - 0.62) / 0.25, 0.0, 1.0)))

    if model == "deepmd":
        widths = {"R": (0.07, 0.13), "G1": (0.17, 0.23), "G2": (0.27, 0.34), "G3": (0.38, 0.47)}
    else:
        widths = {"Y": (0.07, 0.11), "e": (0.15, 0.20), "h": (0.24, 0.31), "g": (0.35, 0.40), "phi": (0.44, 0.49)}
    registry.text(ax, 0.5, 0.955, MODELS[model]["provider"], ha="center", va="center", fontsize=FONT_SIZES["panel_title"], color=INK, zorder=8)
    keys = [key for key, *_ in rows]
    embed_keys = [key for key, *_rest, stage in rows if stage == "embed"]
    row_h = (ROW_TOP - ROW_BOTTOM) / n
    mid_y = 0.5 * (ROW_TOP + ROW_BOTTOM)

    def row_y(k: int) -> float:
        return ROW_TOP - (k + 0.5) * row_h

    # Rapid pass: two continuous wipes instead of staged fades.  The first runs
    # left -> right across the per-neighbour blocks, the second top -> bottom
    # down the right column (sum, descriptor, fitting, eps, E).
    rx0, rx1 = RIGHT_X
    swept = rapid and sweeps is not None
    if swept:
        h_sweep, v_sweep = sweeps
        x_start = widths[embed_keys[0]][0] - 0.02
        x_front = x_start + h_sweep * (rx0 - x_start)
        y_front = SWEEP_TOP - v_sweep * (SWEEP_TOP - SWEEP_BOTTOM)

    def reveal_x(rect) -> tuple[float, tuple[float, float, float, float] | None]:
        x0, y0, x1, y1 = rect
        frac = float(np.clip((x_front - x0) / (x1 - x0), 0.0, 1.0))
        return (1.0 if frac > 0.0 else 0.0), (x0, y0, x0 + frac * (x1 - x0), y1)

    def reveal_y(rect) -> tuple[float, tuple[float, float, float, float] | None]:
        x0, y0, x1, y1 = rect
        frac = float(np.clip((y1 - y_front) / (y1 - y0), 0.0, 1.0))
        return (1.0 if frac > 0.0 else 0.0), (x0, y1 - frac * (y1 - y0), x1, y1)

    # --- row-aligned blocks ----------------------------------------------
    for key, label, size, matrix, stage in rows:
        rect = (widths[key][0], ROW_BOTTOM, widths[key][1], ROW_TOP)
        clip = None
        if stage == "gather":
            alpha = filled
        elif swept:
            alpha, clip = reveal_x(rect)
        else:
            alpha = 1.0 if rapid and embed_p > 0.0 else smoothstep(float(np.clip(embed_p * len(embed_keys) - embed_keys.index(key), 0.0, 1.0)))
        if alpha > 0.0:
            _heat(ax, matrix, rect, alpha=alpha, clip=clip)
        _frame(ax, rect, colour=FORCE_OLIVE if backward else (EDGE_NAVY if alpha > 0 else LINE_GRAY), lw=1.8 if backward else 1.1)
        cx = 0.5 * (rect[0] + rect[2])
        registry.text(ax, cx, LABEL_Y, label, ha="center", va="center", fontsize=fs, color=EDGE_NAVY if alpha > 0 else DARK_GRAY, zorder=8)
    for left, right in zip(keys[:-1], keys[1:]):
        if right not in embed_keys:
            on = filled > 0
        elif swept:
            on = x_front >= widths[right][0]
        else:
            on = embed_p > 0
        _arrow(ax, (widths[left][1] + 0.004, mid_y), (widths[right][0] - 0.004, mid_y), colour=EDGE_NAVY, video=video, on=on)
    if swept and 0.0 < h_sweep < 1.0:
        ax.add_line(Line2D([x_front, x_front], [ROW_BOTTOM - 0.015, ROW_TOP + 0.015], color=EDGE_NAVY, lw=3.0, alpha=0.55, solid_capstyle="round", zorder=9))

    left_x = widths[keys[0]][0]
    if model == "deepmd":
        for element in ("O", "H"):
            members = np.flatnonzero(record["species"] == element)
            if len(members):
                top, bottom = ROW_TOP - members[0] * row_h, ROW_TOP - (members[-1] + 1) * row_h
                ax.add_line(Line2D([left_x - 0.012] * 2, [bottom + 0.004, top - 0.004], color=DARK_GRAY, lw=1.4, zorder=5))
                registry.text(ax, left_x - 0.035, 0.5 * (top + bottom), element, ha="center", va="center", fontsize=fs, color=DARK_GRAY, zorder=8)

    # --- neighbours fly from the magnifier into their rows -----------------
    if video and mode == "gather" and p < 0.75:
        t = smoothstep(float(np.clip(p / 0.62, 0.0, 1.0)))
        to_fig = fig.transFigure.inverted().transform
        groups: dict[str, tuple[list[float], list[float]]] = {}
        for k, atom in enumerate(record["ids"]):
            start_b = case_geometry["neighbour_xy"].get(int(atom))
            if start_b is None:
                continue
            start = to_fig(panel_b.transData.transform(start_b))
            end = to_fig(ax.transData.transform((left_x, row_y(k))))
            lift = 0.06 * np.sin(np.pi * t)
            xs, ys = groups.setdefault(str(record["species"][k]), ([], []))
            xs.append(start[0] + (end[0] - start[0]) * t)
            ys.append(start[1] + (end[1] - start[1]) * t + lift)
        for element, (xs, ys) in groups.items():
            fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, ls="none", marker="o", markersize=6.5, markerfacecolor=SPECIES_DOT[element], markeredgecolor=DARK_GRAY, markeredgewidth=0.6, zorder=60))

    # --- right column: neighbour sum ---------------------------------------
    if swept:
        sum_on = y_front < SWEEP_TOP
        sweep = n
    else:
        sum_on = contract_p > 0
        sweep = int(round(contract_p * n))
    sum_colour = EDGE_NAVY if sum_on else DARK_GRAY
    if model == "deepmd":
        layout = [(rx0, 0.715, rx1, ROW_TOP)]
    else:
        layout = [(rx0, 0.775, rx1, ROW_TOP), (rx0, 0.67, rx0 + 0.15, 0.745), (rx0 + 0.25, 0.655, rx0 + 0.325, 0.745)]
    _arrow(ax, (widths[keys[-1]][1] + 0.006, 0.5 * (layout[0][1] + layout[0][3])), (rx0 - 0.008, 0.5 * (layout[0][1] + layout[0][3])), colour=EDGE_NAVY, video=video, on=(x_front >= rx0 - 0.01) if swept else contract_p > 0)
    registry.text(ax, rx0, LABEL_Y, r"$\Sigma_j$ over all rows", ha="left", va="center", fontsize=fs, color=sum_colour, zorder=8)
    for (key, label, contributions, closing), rect in zip(record["partial"], layout):
        alpha, clip = reveal_y(rect) if swept else (1.0 if contract_p > 0 else 0.0, None)
        if alpha > 0:
            value = contributions[:sweep].sum(axis=0) + (closing if sweep >= n else 0.0)
            vmax = float(np.abs(contributions.sum(axis=0) + closing).max()) or 1.0
            _heat(ax, value, rect, vmax=vmax, clip=clip)
        _frame(ax, rect, colour=FORCE_OLIVE if backward else (EDGE_NAVY if alpha > 0 else LINE_GRAY), lw=1.8 if backward else 1.1)
        if label:
            registry.text(ax, rect[2] + 0.012, 0.5 * (rect[1] + rect[3]), label, ha="left", va="center", fontsize=fs, color=sum_colour, zorder=8)

    # --- invariant descriptor ------------------------------------------------
    inv_label, inv_matrix = record["invariant"]
    inv_rect = (rx0, 0.535, rx1, 0.60) if model == "deepmd" else (rx0, 0.535, rx1, 0.575)
    if swept:
        inv_alpha, inv_clip = reveal_y(inv_rect)
    else:
        inv_alpha, inv_clip = (1.0 if rapid and contract_p > 0.0 else smoothstep(float(np.clip((contract_p - 0.8) / 0.2, 0.0, 1.0)))), None
    if inv_alpha > 0:
        _heat(ax, inv_matrix, inv_rect, alpha=inv_alpha, clip=inv_clip)
    _frame(ax, inv_rect, colour=FORCE_OLIVE if backward else (EDGE_NAVY if inv_alpha > 0 else LINE_GRAY), lw=1.8 if backward else 1.1)
    registry.text(ax, rx0, inv_rect[3] + 0.04, inv_label, ha="left", va="center", fontsize=fs, color=EDGE_NAVY if inv_alpha > 0 else DARK_GRAY, zorder=8)

    # --- fitting layers, eps_i and E ----------------------------------------
    fit_on = y_front < 0.42 if swept else fit_p > 0
    registry.text(ax, rx0, 0.465, record["fit_label"], ha="left", va="center", fontsize=fs, color=ENERGY_TEAL if fit_on else DARK_GRAY, zorder=8)
    for index, (activation, y0) in enumerate(zip(record["fits"], (0.38, 0.33, 0.28))):
        rect = (rx0, y0, rx1, y0 + 0.035)
        if swept:
            alpha, clip = reveal_y(rect)
        else:
            alpha, clip = (1.0 if rapid and fit_p > 0.0 else smoothstep(float(np.clip(fit_p * 4.0 - index, 0.0, 1.0)))), None
        if alpha > 0:
            _heat(ax, activation[None, :], rect, alpha=alpha, clip=clip)
        _frame(ax, rect, colour=FORCE_OLIVE if backward else (ENERGY_TEAL if alpha > 0 else LINE_GRAY), lw=1.8 if backward else 1.1)
    eps_on = y_front <= EPS_Y + 0.015 if swept else fit_p >= 0.75
    registry.text(ax, rx0, EPS_Y, rf"$\varepsilon_{{\mathrm{{O126}}}}$ = {_neg(record['epsilon'])} eV" if eps_on else r"$\varepsilon_{\mathrm{O126}}$ = …", ha="left", va="center", fontsize=fs, color=ENERGY_TEAL if eps_on else DARK_GRAY, zorder=8)
    if swept and 0.0 < v_sweep < 1.0:
        ax.add_line(Line2D([rx0 - 0.015, rx1 + 0.005], [y_front, y_front], color=EDGE_NAVY, lw=3.0, alpha=0.55, solid_capstyle="round", zorder=9))
    e_on = (y_front <= SWEEP_BOTTOM + 0.005) if swept else _reached(mode, "energy")
    energy = float(data["total_energy_ev"][state])
    energy_label = rf"$E=\Sigma_i\,\varepsilon_i$ = {_neg(energy, 1)} eV"
    # The force path leaves the middle of the finished E label, so its anchor
    # does not move when the value is revealed.
    probe = ax.text(rx0, ENERGY_Y, energy_label, ha="left", va="center", fontsize=fs, fontfamily=registry.font_family or plt.rcParams["font.family"])
    box = probe.get_window_extent(fig.canvas.get_renderer())
    probe.remove()
    scale = text_scale_of(fig)
    left, middle = ax.transData.transform((rx0, ENERGY_Y))
    centre_x = left + 0.5 * box.width * scale
    bottom_y = middle - 0.5 * box.height * scale - (6.0 if video else 4.0)
    registry.text(ax, rx0, ENERGY_Y, energy_label if e_on else r"$E=\Sigma_i\,\varepsilon_i$ = …", ha="left", va="center", fontsize=fs, color=ENERGY_TEAL if e_on else DARK_GRAY, zorder=8)
    return tuple(fig.transFigure.inverted().transform((centre_x, bottom_y)))


def _polyline_prefix(points: np.ndarray, fraction: float) -> np.ndarray:
    """Leading part of a polyline covering ``fraction`` of its length."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    target = float(np.clip(fraction, 0.0, 1.0)) * lengths.sum()
    out = [points[0]]
    for start, end, length in zip(points[:-1], points[1:], lengths):
        if target >= length:
            out.append(end)
            target -= length
            continue
        out.append(start + (end - start) * (target / max(length, 1.0e-12)))
        break
    return np.array(out)


def draw_return_path(
    fig: plt.Figure,
    registry: LayoutRegistry,
    panel_a: plt.Axes,
    panel_b: plt.Axes,
    panel_d: plt.Axes,
    energy_anchor: tuple[float, float],
    *,
    video: bool,
    mode: str,
    progress: float,
) -> None:
    """F = -dE/dr leaves the middle of E and enters the integrator's 'a' node.

    One solid path in figure coordinates: down from E, left along the bottom
    strip, up the A|B gutter and into the right side of the 'a' node.  It is
    grey until the force stage, grows in orange during it and stays orange
    while the acceleration is formed.
    """
    y_return = RETURN_Y_VIDEO if video else RETURN_Y_STATIC
    to_fig = fig.transFigure.inverted().transform
    a_aspect = _axes_aspect(panel_a)
    node_half_width = 0.092 if video else 0.070
    a_x = 0.50 + LOOP_RADIUS_X * np.cos(np.deg2rad(-30.0)) + node_half_width
    a_y = LOOP_CENTRE_Y + LOOP_RADIUS_X * a_aspect * np.sin(np.deg2rad(-30.0))
    node_fig = to_fig(panel_a.transAxes.transform((a_x, a_y)))
    gutter_x = panel_b.get_position().x0 - (0.001 if video else 0.002)
    points = np.array([
        energy_anchor,
        (energy_anchor[0], y_return),
        (gutter_x, y_return),
        (gutter_x, node_fig[1]),
        (node_fig[0] + 0.003, node_fig[1]),
    ])
    lw = 2.6 if video else 1.6
    head = 18 if video else 12
    if mode == "force":
        fraction = smoothstep(progress)
    elif mode == "accel":
        fraction = 1.0
    else:
        fraction = 0.0

    def polyline(path: np.ndarray, colour: str, zorder: float) -> None:
        if len(path) < 2:
            return
        fig.add_artist(Line2D(path[:, 0], path[:, 1], transform=fig.transFigure, color=colour, lw=lw, solid_joinstyle="miter", zorder=zorder))
        tail = path[-1] - path[-2]
        if np.linalg.norm(tail) > 1.0e-6:
            start = path[-1] - tail / np.linalg.norm(tail) * 1.0e-3
            arrow = FancyArrowPatch(tuple(start), tuple(path[-1]), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=head, lw=lw, color=colour, shrinkA=0, shrinkB=0, zorder=zorder)
            fig.add_artist(arrow)
            registry.arrows.append(arrow)

    polyline(points, INACTIVE_PATH, 49)
    if fraction > 0.0:
        polyline(_polyline_prefix(points, fraction), FORCE_OLIVE, 50)
    label_xy = panel_d.transAxes.inverted().transform(fig.transFigure.transform((energy_anchor[0] + (0.006 if video else 0.004), 0.5 * (energy_anchor[1] + y_return))))
    registry.text(panel_d, *label_xy, r"$\mathbf{F}=-\partial E/\partial\mathbf{r}$", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=FORCE_OLIVE if fraction > 0.0 else DARK_GRAY, transform=panel_d.transAxes, zorder=50)


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
    if bounded < DETAILED_BLOCK:
        mode, progress = _phase_at(bounded, DETAILED_PHASES)
        return {"mode": mode, "state": 0, "progress": float(np.clip(progress, 0.0, 1.0)), "rapid": False}
    slow_hold_end = DETAILED_BLOCK + SLOW_HOLD_SECONDS
    if bounded < slow_hold_end:
        return {"mode": "move", "state": 0, "progress": 1.0, "rapid": False}

    rapid_time = bounded - slow_hold_end
    rapid_total = len(RAPID_STATES) * RAPID_CYCLE_SECONDS
    if rapid_time < rapid_total:
        index = min(int(rapid_time // RAPID_CYCLE_SECONDS), len(RAPID_STATES) - 1)
        local = (rapid_time - index * RAPID_CYCLE_SECONDS) / RAPID_CYCLE_SECONDS
        mode, stage_progress = _phase_at(local, RAPID_PHASES)
        sweeps = tuple(smoothstep(float(np.clip((local - a) / (b - a), 0.0, 1.0))) for a, b in (SWEEP_H, SWEEP_V))
        return {"mode": mode, "state": RAPID_STATES[index], "progress": float(np.clip(stage_progress, 0.0, 1.0)), "rapid": True, "sweeps": sweeps}

    # The final saved snapshot has no following force/velocity/move asset.
    # Hold its positions directly so the fast scan ends on a clean state.
    return {"mode": "positions", "state": n_states - 1, "progress": 1.0, "rapid": True}


def semantics_for(mode: str) -> list[dict]:
    return {
        "positions": [{"id": "centre_atom", "color": CENTRE_NAVY, "min_pixels": 30, "tolerance": 60}],
        "neighbours": [{"id": "centre_atom", "color": CENTRE_NAVY, "min_pixels": 30, "tolerance": 60}],
        "gather": [{"id": "neighbour_edges", "color": EDGE_NAVY, "min_pixels": 60, "tolerance": 60}],
        "embed": [{"id": "neighbour_edges", "color": EDGE_NAVY, "min_pixels": 60, "tolerance": 60}],
        "contract": [{"id": "neighbour_sum", "color": EDGE_NAVY, "min_pixels": 40}],
        "fit": [{"id": "centre_atom_or_eps", "color": CENTRE_NAVY, "min_pixels": 30, "tolerance": 60}],
        "energy": [{"id": "energy_text", "color": ENERGY_TEAL, "min_pixels": 40}],
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
    panel_d = axes_from_top_slot(fig, VIDEO_D)
    draw_left(panel_a, registry, video=True, mode=mode, provider=provider)
    geometry = draw_case(panel_b, registry, data, scene, model=model, video=True, mode=mode, state=state, progress=progress, rapid=rapid)
    energy_anchor = draw_pipeline(fig, panel_d, registry, data, scene["internals"], geometry, panel_b, model=model, video=True, mode=mode, state=state, progress=progress, rapid=rapid, sweeps=state_info.get("sweeps"))
    draw_return_path(fig, registry, panel_a, panel_b, panel_d, energy_anchor, video=True, mode=mode, progress=progress)
    if not rapid and mode == "positions":
        _deemphasize(panel_d)
    return semantics_for(mode)


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------
def render_static(model: str, data: dict[str, object], scene: dict[str, object]) -> None:
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=18)
    provider = MODELS[model]["provider"]
    panel_a = axes_from_top_slot(fig, STATIC_A)
    panel_b = axes_from_top_slot(fig, STATIC_B)
    panel_d = axes_from_top_slot(fig, STATIC_D)
    draw_left(panel_a, registry, video=False, mode="force", provider=provider)
    geometry = draw_case(panel_b, registry, data, scene, model=model, video=False, mode="force", state=1, progress=1.0, rapid=False)
    energy_anchor = draw_pipeline(fig, panel_d, registry, data, scene["internals"], geometry, panel_b, model=model, video=False, mode="force", state=1, progress=1.0)
    draw_return_path(fig, registry, panel_a, panel_b, panel_d, energy_anchor, video=False, mode="force", progress=1.0)
    errors = registry.validate(fig)
    if errors:
        debug = ROOT / "qa" / MODELS[model]["stem"] / "_qa" / "static_failed.png"
        debug.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(debug, dpi=100, facecolor=WHITE)
        texts = [f"text[{i}]={a.get_text()!r}" for i, a in enumerate(registry.texts)]
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors) + f"\n(debug image {debug})\n" + "\n".join(texts))
    save_static(fig, MODELS[model]["stem"])


KEYFRAME_TIMES = (0.40, 1.40, 2.20, 2.70, 3.30, 4.20, 5.20, 6.20, 7.00, 7.80, 8.50, 9.00, 10.00, 11.20, 12.50, 13.10, 14.50, 16.40, 16.80, 17.60, 18.50, 19.30, 20.30, 21.40, 22.50, 23.60, 24.50)


def render_keyframes(model: str, data: dict[str, object], scene: dict[str, object], qa_dir: Path) -> Path:
    output_dir = qa_dir / "_qa" / "keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    records = []
    failures: list[str] = []
    for index, time_seconds in enumerate(KEYFRAME_TIMES):
        fig = new_video_figure()
        registry = LayoutRegistry(min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=12, font_family="Arial", coerce_min_font=True)
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
            {"id": "pipeline", "rect": list(VIDEO_D), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.020, "min_panel_bbox_fill": 0.22, "grid_rows": 12, "grid_columns": 20},
        "bands": [
            {"id": "gap_a_b", "rect": [0.200, 0.025, 0.210, 0.900], "max_ink_pixels": 5000},
        ],
    }
    render_video(
        stem=MODELS[model]["stem"],
        duration_seconds=VIDEO_DURATION,
        draw_frame=lambda fig, time, _index, registry: draw_frame(fig, time, registry, data, scene, model=model),
        audit_config=audit_config,
        qa_directory=qa_dir / "_qa",
        representative_times=KEYFRAME_TIMES,
        image_sequence=True,
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
        "timeline": {"duration_seconds": VIDEO_DURATION, "detailed_block_seconds": DETAILED_BLOCK, "slow_hold_seconds": SLOW_HOLD_SECONDS, "rapid_cycle_seconds": RAPID_CYCLE_SECONDS, "rapid_states": RAPID_STATES, "final_state_hold_seconds": FINAL_STATE_HOLD_SECONDS, "detailed_phases": DETAILED_PHASES, "rapid_phases": RAPID_PHASES},
        "internals": {
            "file": str(MODELS[model]["internals"]),
            "sha256": sha256_file(MODELS[model]["internals"]),
            "checks": json.loads(MODELS[model]["internals"].with_suffix(".json").read_text(encoding="utf-8")).get("checks"),
            "heatmap_normalisation": "each block saturates at its own 98th-percentile |value| (navy negative, crimson positive); the neighbour-sum blocks use the largest |value| of their final sum",
        },
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
        scene["internals"] = load_internals(model, data)
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
