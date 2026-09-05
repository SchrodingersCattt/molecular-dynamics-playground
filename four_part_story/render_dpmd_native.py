"""Native MatterVis Deep-Potential neighbourhood and frozen-force VV preview.

This module is intentionally isolated from the production renderers.  MatterVis
owns every atom, bond, periodic-cell pixel, cutoff sphere, and neighbour vector;
the paper layer only adds concise labels and the explanatory rails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch

from common import DARK_GRAY, INK, LINE_GRAY, NAVY, LayoutRegistry, json_dump, new_static_figure, render_video, save_static, sha256_file
from mattervis_story import camera_for_source, make_sphere_mesh, make_torus_mesh, make_vector_group, project_world, render_structure
from responsive_story import EMERALD, LAKE_BLUE, PALE_OLIVE, draw_legend, panel_box, place_main, simple_audit, stage_rail, story_axes
from PIL import Image


ROOT = Path(__file__).resolve().parent
STEM = "04_deep_potential_md"
QA_DIR = ROOT / "_qa" / "04_dpmd_native"
SOURCE = ROOT / "data" / "water_box_64.extxyz"
BASE_PATH = ROOT / "data" / "water_box_64.npz"
RESULT_PATH = ROOT / "data" / "dpmd_water_box_results.npz"
META_PATH = ROOT / "data" / "dpmd_eval.json"
VV_SOURCE = QA_DIR / "vv_snapshot.extxyz"
MATTERVIS_DIR = QA_DIR / "mattervis_v3"
BOX_IMAGE = MATTERVIS_DIR / "box_initial.png"
BOX_UPDATED_IMAGE = MATTERVIS_DIR / "box_updated.png"
LOCATOR_IMAGE = MATTERVIS_DIR / "box_locator.png"
FORCE_IMAGE = MATTERVIS_DIR / "box_force.png"
VELOCITY_IMAGE = MATTERVIS_DIR / "box_velocity.png"
CUTOFF_IMAGE = MATTERVIS_DIR / "box_cutoff.png"
INSIDE_IMAGE = MATTERVIS_DIR / "box_inside.png"
NEIGHBOR_IMAGE = MATTERVIS_DIR / "box_neighbors.png"
FOCUS_CUTOFF_IMAGE = MATTERVIS_DIR / "focus_cutoff.png"
FOCUS_INSIDE_IMAGE = MATTERVIS_DIR / "focus_inside.png"
FOCUS_NEIGHBOR_IMAGE = MATTERVIS_DIR / "focus_neighbors.png"
FOCUS_FORCE_IMAGE = MATTERVIS_DIR / "focus_force.png"
FOCUS_VELOCITY_IMAGE = MATTERVIS_DIR / "focus_velocity.png"
FOCUS_FOREGROUND_IMAGE = MATTERVIS_DIR / "focus_foreground.png"

DT_FS = 0.5
VV_SEED = 260829
EV_A_TO_A_FS2 = 0.00964853399


def load_data() -> dict[str, object]:
    with np.load(BASE_PATH, allow_pickle=False) as b:
        data = {key: b[key] for key in b.files}
    with np.load(RESULT_PATH, allow_pickle=False) as r:
        data.update({f"result_{key}": r[key] for key in r.files})
    data["metadata"] = json.loads(META_PATH.read_text(encoding="utf-8"))
    return data


def _make_vv_snapshot(data: dict[str, object]) -> dict[str, object]:
    """Generate a deterministic frozen-force VV pair from this exact snapshot."""
    positions = np.asarray(data["positions_wrapped"], dtype=float)
    elements = np.asarray(data["elements"]).astype(str)
    box = float(np.asarray(data["box_length"]).reshape(-1)[0])
    forces = np.asarray(data["result_forces_ev_per_angstrom"], dtype=float)
    masses = np.where(elements == "O", 15.9994, 1.008)
    rng = np.random.default_rng(VV_SEED)
    velocities = rng.normal(0.0, 0.010, size=positions.shape)
    velocities -= np.average(velocities, axis=0, weights=masses)
    acceleration = forces * EV_A_TO_A_FS2 / masses[:, None]
    half_velocity = velocities + 0.5 * acceleration * DT_FS
    updated = (positions + half_velocity * DT_FS) % box
    # Only one force snapshot is supplied; make the approximation explicit and
    # reproducible instead of silently inventing a second force evaluation.
    final_velocity = half_velocity + 0.5 * acceleration * DT_FS
    QA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from ase import Atoms
        from ase.io import write
        atoms0 = Atoms(symbols=elements.tolist(), positions=positions, cell=np.eye(3) * box, pbc=True)
        atoms1 = Atoms(symbols=elements.tolist(), positions=updated, cell=np.eye(3) * box, pbc=True)
        write(VV_SOURCE, [atoms0, atoms1], format="extxyz")
    except Exception:
        # Minimal extxyz fallback for environments without ASE.
        header = f'Lattice="{box} 0 0 0 {box} 0 0 0 {box}" Properties=species:S:1:pos:R:3 pbc="T T T"'
        with VV_SOURCE.open("w", encoding="utf-8") as out:
            for frame in (positions, updated):
                out.write(f"{len(elements)}\n{header}\n")
                for symbol, xyz in zip(elements, frame):
                    out.write(f"{symbol} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")
    metadata = {
        "seed": VV_SEED,
        "dt_fs": DT_FS,
        "force_units": "eV/Angstrom",
        "velocity_units": "Angstrom/fs",
        "acceleration_conversion": EV_A_TO_A_FS2,
        "force_model": str(data["metadata"].get("model", "DP snapshot")),
        "approximation": "frozen force for one reproducible VV step; no second DP call",
        "central_index": int(np.asarray(data["central_index"]).reshape(-1)[0]),
    }
    (QA_DIR / "vv_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"positions": positions, "updated": updated, "velocities": velocities, "updated_velocities": final_velocity, "acceleration": acceleration, "metadata": metadata}


def _focus_indices(data: dict[str, object], distances: np.ndarray, central: int) -> set[int]:
    """Return complete, real water molecules for the local-environment view.

    The rendered focus is deliberately sparse, but it is not a synthetic
    cluster: each selected oxygen is accompanied by its two nearest hydrogen
    atoms from the same periodic snapshot.  The full 83-atom count remains in
    the provenance and right-hand annotation; only eight representative
    molecules are shown so the cutoff geometry can be read at a glance.
    """
    positions = np.asarray(data["positions_wrapped"], dtype=float)
    elements = np.asarray(data["elements"]).astype(str)
    box = float(np.asarray(data["box_length"]).reshape(-1)[0])
    oxygen = np.flatnonzero(elements == "O")
    ordered = oxygen[np.argsort(distances[oxygen])]
    chosen = [int(central)] + [int(index) for index in ordered if int(index) != int(central)][:7]
    keep: set[int] = set()
    for oxygen_index in chosen:
        keep.add(oxygen_index)
        delta = positions - positions[oxygen_index]
        delta -= box * np.round(delta / box)
        hydrogen = np.flatnonzero(elements == "H")
        h_distances = np.linalg.norm(delta[hydrogen], axis=1)
        bonded = hydrogen[np.argsort(h_distances)[:2]]
        keep.update(
            int(index)
            for index in bonded
            if float(h_distances[np.where(hydrogen == index)[0][0]]) < 1.35
        )
    return keep


def _render_assets(data: dict[str, object], vv: dict[str, object]) -> dict[str, object]:
    MATTERVIS_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(data["positions_wrapped"], dtype=float)
    central = int(np.asarray(data["central_index"]).reshape(-1)[0])
    cutoff = float(np.asarray(data.get("cutoff_angstrom", 6.0)).reshape(-1)[0])
    target = positions[central]
    # Centre the camera on the selected atom, not on the box centroid.  This
    # makes the 6 Å neighbourhood the visual subject while keeping the full
    # periodic cell in frame.
    # Keep an oblique box view, but avoid looking down the selected force
    # vector.  This preserves the 3-D cell while leaving a substantial screen
    # projection for the native force arrow.
    camera = camera_for_source(
        VV_SOURCE, target=target, ortho_scale=15.2, frame=0,
        direction=(0.20, -1.0, 0.70),
    )
    force = np.asarray(data["result_forces_ev_per_angstrom"], dtype=float)[central]
    velocity = np.asarray(vv["velocities"], dtype=float)[central]
    style = {
        "shaft_radius": 0.030,
        "head_length": 0.120,
        "head_radius": 0.078,
        "sides": 18,
    }
    # The physical force is small compared with the 12.43 Å box.  The native
    # MatterVis arrow is therefore amplified for legibility, while the exact
    # unscaled vector remains in the NPZ/sidecar provenance.
    # The display length is deliberately long enough to emerge from the
    # translucent cutoff shell; the numerical force itself remains unchanged
    # in the provenance sidecar and NPZ snapshot.
    force_vectors = make_vector_group("F_DP", positions[central:central + 1], force[None, :], scale=70.0, color=PALE_OLIVE, style=style)
    velocity_vectors = make_vector_group("v_seed", positions[central:central + 1], velocity[None, :], scale=180.0, color=EMERALD, style=style)
    # A native radius vector makes the cutoff measurement unambiguous: it
    # starts at O126 and terminates on the actual 6 Å sphere, rather than
    # relying on a paper-space circle or a floating caption.
    radius_vector = make_vector_group(
        "r_c", positions[central:central + 1], np.asarray([[0.0, 0.0, cutoff]]),
        scale=1.0, color="#2E89A7", style={
            "shaft_radius": 0.024, "head_length": 0.28,
            "head_radius": 0.060, "sides": 14,
        },
    )
    sphere = make_sphere_mesh(
        positions[central], cutoff, color="#397F99", opacity=0.22,
        lat_steps=18, lon_steps=36, mesh_id="r_c_6A_sphere",
    )
    # The radial MatterVis shading carries the actual 3-D depth.  Keep only
    # one equator and one pale meridian as restrained near/far cues; a full
    # globe grid would make the local environment read as a flat diagram.
    equator = make_torus_mesh(
        positions[central], cutoff * 0.998, 0.035,
        normal=camera.direction, color="#1F536B", opacity=0.74,
        major_steps=72, tube_steps=6, mesh_id="r_c_equator",
    )
    meridian = make_torus_mesh(
        positions[central], cutoff * 0.998, 0.024,
        normal=np.asarray(camera.up, dtype=float), color="#2E89A7", opacity=0.64,
        major_steps=72, tube_steps=6, mesh_id="r_c_meridian",
    )
    camera_right = np.cross(np.asarray(camera.direction, dtype=float), np.asarray(camera.up, dtype=float))
    camera_right /= max(float(np.linalg.norm(camera_right)), 1.0e-12)
    oblique_ring = make_torus_mesh(
        positions[central], cutoff * 0.998, 0.020,
        normal=np.asarray(camera.up, dtype=float) + 0.48 * camera_right,
        color="#6AAFC0", opacity=0.46,
        major_steps=72, tube_steps=6, mesh_id="r_c_oblique",
    )
    centre_marker = make_torus_mesh(
        positions[central], 0.46, 0.055,
        normal=camera.direction, color="#183153", opacity=0.92,
        major_steps=48, tube_steps=6, mesh_id="O126_selection_ring",
    )
    sphere_overlays = [sphere, equator, meridian, oblique_ring, centre_marker]
    # Neighbour/force stages need the world-space arrows to remain legible
    # through the shell.  A lighter shell is still a true 3-D sphere, while
    # the same depth cues and centre marker preserve the spatial reference.
    soft_sphere = dict(sphere, opacity=0.19)
    soft_overlays = [soft_sphere, equator, meridian, oblique_ring, centre_marker]
    # Distances use the same minimum-image convention as the saved neighbour
    # mask.  The opacity map is applied inside MatterVis, so fading is not a
    # paper mask and remains depth-consistent with the sphere and bonds.
    box = float(np.asarray(data["box_length"]).reshape(-1)[0])
    deltas = positions - positions[central]
    deltas -= box * np.round(deltas / box)
    distances = np.linalg.norm(deltas, axis=1)
    neighbour_mask = np.asarray(data["neighbor_mask"], dtype=bool)
    outside_scales = {
        int(index): (1.0 if distance <= cutoff + 1.0e-8 else 0.025)
        for index, distance in enumerate(distances)
    }
    selected_scales = {
        int(index): (
            1.0 if (bool(neighbour_mask[index]) or index == central)
            else (0.12 if distance <= cutoff + 1.0e-8 else 0.025)
        )
        for index, distance in enumerate(distances)
    }
    focus_indices = _focus_indices(data, distances, central)
    # Focus renders retain real MatterVis atom/bond geometry, but suppress
    # non-representative waters so the sphere and selected local environment
    # remain the first thing visible at thumbnail size.
    # Preserve a true depth cue in the local MatterVis scene: atoms on the
    # camera-facing side remain crisp, while the far side is gently softened.
    scene_positions = positions + np.asarray(camera.scene_offset, dtype=float)
    camera_coords = camera.mattervis()
    from mat_viewer.render.camera import CameraTransform
    camera_transform = CameraTransform(camera_coords, width=1700, height=1180)
    camera_depth = -camera_transform.world_to_camera(scene_positions)[:, 2]
    depth_span = max(float(np.ptp(camera_depth)), 1.0e-8)
    depth_factor = 0.78 + 0.22 * (camera_depth - float(camera_depth.min())) / depth_span
    focus_scales = {
        int(index): (float(0.92 * depth_factor[index]) if int(index) in focus_indices else 0.0)
        for index in range(len(positions))
    }
    # A second, native MatterVis pass keeps the camera-facing local molecules
    # crisp above the translucent shell. It uses the same coordinates and
    # source-index mask; the compositing is pixel-aligned before paper layout.
    foreground_scales = {
        int(index): (
            1.0 if int(index) == central
            else (1.0 if (int(index) in focus_indices and float(depth_factor[index]) > 0.72) else 0.0)
        )
        for index in range(len(positions))
    }
    locator_scales = {
        int(index): (1.0 if int(index) == central else 0.0)
        for index in range(len(positions))
    }
    context_scales = {
        int(index): (0.98 if int(index) in focus_indices else 0.025)
        for index in range(len(positions))
    }
    centre_colour = {central: "#234A67"}
    neighbour_indices = np.flatnonzero(neighbour_mask)
    neighbour_indices = neighbour_indices[np.argsort(np.asarray(data["neighbor_distances"], dtype=float)[neighbour_indices])]
    neighbour_indices = neighbour_indices[:8]
    neighbour_origins = np.repeat(positions[central:central + 1], len(neighbour_indices), axis=0)
    neighbour_vectors = deltas[neighbour_indices]
    neighbour_style = {"shaft_radius": 0.022, "head_length": 0.24, "head_radius": 0.050, "sides": 14}
    neighbour_vectors_native = make_vector_group(
        "MIC-neighbours", neighbour_origins, neighbour_vectors,
        scale=1.65, color="#2E89A7", opacity=0.98, style=neighbour_style,
    )
    common_kwargs = dict(camera=camera, frame=0, view="unit_cell", width=1700,
                         height=1180, atom_scale=0.72, bond_radius=0.075,
                         show_cell=True, cell_color="#9AA5AA", cell_width_px=1.15)
    render_structure(VV_SOURCE, BOX_IMAGE, **common_kwargs)
    render_structure(VV_SOURCE, BOX_UPDATED_IMAGE, frame=1, camera=camera,
                     view="unit_cell", width=1700, height=1180, atom_scale=0.72,
                     bond_radius=0.075, show_cell=True,
                     cell_color="#9AA5AA", cell_width_px=1.15)
    render_structure(VV_SOURCE, LOCATOR_IMAGE, **common_kwargs,
                     mesh_overlays=[centre_marker],
                     atom_opacity_scales=locator_scales,
                     atom_color_overrides=centre_colour)
    render_structure(VV_SOURCE, CUTOFF_IMAGE, **common_kwargs,
                     mesh_overlays=sphere_overlays, atom_opacity_scales=selected_scales,
                     vector_overlays=radius_vector,
                     atom_color_overrides=centre_colour)
    render_structure(VV_SOURCE, INSIDE_IMAGE, **common_kwargs,
                     mesh_overlays=sphere_overlays, atom_opacity_scales=outside_scales,
                     vector_overlays=radius_vector,
                     atom_color_overrides=centre_colour)
    render_structure(VV_SOURCE, NEIGHBOR_IMAGE, **common_kwargs,
                     mesh_overlays=soft_overlays, atom_opacity_scales=selected_scales,
                     atom_color_overrides=centre_colour,
                     vector_overlays=neighbour_vectors_native)
    render_structure(VV_SOURCE, FORCE_IMAGE, **common_kwargs,
                     mesh_overlays=soft_overlays, atom_opacity_scales=selected_scales,
                     atom_color_overrides=centre_colour, vector_overlays=force_vectors)
    render_structure(VV_SOURCE, VELOCITY_IMAGE, frame=1, camera=camera,
                     view="unit_cell", width=1700, height=1180, atom_scale=0.72,
                     bond_radius=0.075, show_cell=True, mesh_overlays=soft_overlays,
                     atom_opacity_scales=selected_scales,
                     atom_color_overrides=centre_colour, vector_overlays=velocity_vectors)
    focus_kwargs = dict(camera=camera, frame=0, view="cluster", width=1700,
                        height=1180, atom_scale=1.02, bond_radius=0.095,
                        show_cell=False, include_boundary_replicas=False,
                        atom_opacity_scales=focus_scales,
                        atom_color_overrides=centre_colour)
    render_structure(VV_SOURCE, FOCUS_CUTOFF_IMAGE, **{**focus_kwargs, "atom_opacity_scales": context_scales},
                     mesh_overlays=sphere_overlays, vector_overlays=radius_vector)
    render_structure(VV_SOURCE, FOCUS_INSIDE_IMAGE, **focus_kwargs,
                     mesh_overlays=sphere_overlays, vector_overlays=radius_vector)
    render_structure(VV_SOURCE, FOCUS_NEIGHBOR_IMAGE, **focus_kwargs,
                     mesh_overlays=soft_overlays, vector_overlays=neighbour_vectors_native)
    render_structure(VV_SOURCE, FOCUS_FORCE_IMAGE, **focus_kwargs,
                     mesh_overlays=soft_overlays, vector_overlays=force_vectors)
    render_structure(VV_SOURCE, FOCUS_VELOCITY_IMAGE, frame=1, camera=camera,
                     view="cluster", width=1700, height=1180, atom_scale=0.88,
                     bond_radius=0.085, show_cell=False,
                     include_boundary_replicas=False,
                     mesh_overlays=soft_overlays,
                     atom_opacity_scales=focus_scales,
                     atom_color_overrides=centre_colour,
                     vector_overlays=velocity_vectors)
    render_structure(VV_SOURCE, FOCUS_FOREGROUND_IMAGE, camera=camera, frame=0,
                     view="cluster", width=1700, height=1180, atom_scale=1.04,
                     bond_radius=0.098, show_cell=False,
                     include_boundary_replicas=False,
                     atom_opacity_scales=foreground_scales,
                     atom_color_overrides=centre_colour)
    # Foreground atoms are still rendered by MatterVis; alpha-compositing only
    # combines the two same-camera native passes and records the provenance.
    for target_image in (FOCUS_CUTOFF_IMAGE, FOCUS_INSIDE_IMAGE,
                         FOCUS_NEIGHBOR_IMAGE, FOCUS_FORCE_IMAGE,
                         FOCUS_VELOCITY_IMAGE):
        with Image.open(target_image).convert("RGBA") as base, Image.open(FOCUS_FOREGROUND_IMAGE).convert("RGBA") as foreground:
            composed = Image.alpha_composite(base, foreground)
            composed.save(target_image)
        sidecar = target_image.with_suffix(".json")
        if sidecar.exists():
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            payload["output_sha256"] = sha256_file(target_image)
            payload["composite_layers"] = [str(FOCUS_FOREGROUND_IMAGE)]
            json_dump(sidecar, payload)
    return {"initial": BOX_IMAGE, "updated": BOX_UPDATED_IMAGE, "cutoff": CUTOFF_IMAGE,
            "inside": INSIDE_IMAGE, "neighbors": NEIGHBOR_IMAGE, "force": FORCE_IMAGE,
            "velocity": VELOCITY_IMAGE, "locator": LOCATOR_IMAGE,
            "focus_cutoff": FOCUS_CUTOFF_IMAGE, "focus_inside": FOCUS_INSIDE_IMAGE,
            "focus_neighbors": FOCUS_NEIGHBOR_IMAGE, "focus_force": FOCUS_FORCE_IMAGE,
            "focus_velocity": FOCUS_VELOCITY_IMAGE, "focus_foreground": FOCUS_FOREGROUND_IMAGE,
            "camera": camera, "central": central, "radius_vector": radius_vector}


def _mic_overlay(ax: plt.Axes, registry: LayoutRegistry, data: dict[str, object], camera, rect, *, video: bool, reveal: float, outside_fade: float = 0.0) -> None:
    """Add a short label to the native MatterVis cutoff sphere.

    The sphere and MIC vectors are rendered in the same world-space MatterVis
    scene.  No projected ellipse, rectangular frame, or paper-space clipping
    mask is used here.
    """
    positions = np.asarray(data["positions_wrapped"], dtype=float)
    central = int(np.asarray(data["central_index"]).reshape(-1)[0])
    cutoff = float(np.asarray(data.get("cutoff_angstrom", 6.0)).reshape(-1)[0])
    c = positions[central]
    centre = project_world(np.asarray([c]), camera=camera, rect=rect, image_aspect=1700.0 / 1180.0)[0]
    if reveal > 0.05:
        registry.text(ax, float(centre[0]), float(centre[1] + 0.24), f"r_c = {cutoff:g} Å",
                      ha="center", va="bottom", fontsize=12 if video else 11,
                      color=NAVY, weight="bold", zorder=16)


def _focus_scene(
    ax: plt.Axes,
    registry: LayoutRegistry,
    data: dict[str, object],
    camera,
    image: Path,
    locator: Path,
    *,
    video: bool,
) -> tuple[float, float, float, float]:
    """Place a sparse local scene and a small, semantic whole-box locator."""
    rect = place_main(ax, image, rect=(0.035, 0.075, 0.965, 0.925))
    # The inset is intentionally an image of the same MatterVis snapshot, not
    # a paper-space rectangle: it answers "where in the periodic box?" while
    # leaving the local sphere as the visual subject.
    inset = place_main(ax, locator, rect=(0.755, 0.715, 0.955, 0.865), alpha=0.82)
    registry.text(ax, 0.76, 0.695, "periodic box", ha="left", va="top",
                  fontsize=10, color=DARK_GRAY, zorder=30)
    positions = np.asarray(data["positions_wrapped"], dtype=float)
    central = int(np.asarray(data["central_index"]).reshape(-1)[0])
    centre = project_world(
        positions[central:central + 1], camera=camera, rect=rect,
        image_aspect=1700.0 / 1180.0,
    )[0]
    # A quiet leader line makes the locator relationship explicit without
    # turning the scene into another framed panel.
    ax.plot([inset[0] + 0.01, centre[0] + 0.16],
            [inset[1] + 0.01, centre[1] + 0.16],
            color=LINE_GRAY, lw=1.2 if video else 0.9, zorder=25,
            solid_capstyle="round")
    ax.scatter([centre[0]], [centre[1]], s=18 if video else 12,
               facecolor=NAVY, edgecolor="white", linewidth=0.7, zorder=26)
    registry.text(ax, float(centre[0] - 0.075), float(centre[1] + 0.075),
                  "O126", ha="right", va="bottom", fontsize=10,
                  color=NAVY, weight="bold", zorder=30)
    return rect


def _info(ax, registry: LayoutRegistry, data: dict[str, object], vv: dict[str, object], *, video: bool, stage: int | None, returning: bool) -> None:
    panel_box(ax, registry, "DP SNAPSHOT", video=video)
    # Keep the right rail subordinate to the 3-D scene: two compact facts and
    # one explicit local-environment count are easier to parse than stacked
    # dashboard cards.  The active fact is marked by a short coloured rule.
    facts = [
        (0.80, "centre", "O126 · centre atom", LAKE_BLUE, 0),
        (0.67, "cutoff", "r_c = 6 Å", NAVY, 2),
        (0.54, "neighbours", "83 atoms within r_c", PALE_OLIVE, 4),
    ]
    for y, label, value, colour, active_stage in facts:
        active = returning or (stage is not None and stage >= active_stage)
        ax.plot([0.10, 0.19], [y, y], color=colour if active else LINE_GRAY,
                lw=4.0 if video else 3.0, solid_capstyle="round", zorder=3)
        registry.text(ax, 0.24, y + 0.018, label, ha="left", va="center",
                      fontsize=11 if video else 10, color=colour if active else DARK_GRAY,
                      weight="bold")
        registry.text(ax, 0.24, y - 0.028, value, ha="left", va="center",
                      fontsize=10, color=INK)
    # A tiny native-colour key keeps the red/white water representation
    # legible after the local sphere is composited over the atoms.
    ax.plot([0.10, 0.19], [0.47, 0.47], color="#A32035", lw=4.0 if video else 3.0,
            solid_capstyle="round", zorder=3)
    registry.text(ax, 0.24, 0.47, "O/H₂O · MatterVis", ha="left", va="center",
                  fontsize=10, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.42, "minimum-image displacement", ha="center",
                  va="center", fontsize=10, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.375, f"VV  Δt = {DT_FS:g} fs", ha="center",
                  va="center", fontsize=10, color=NAVY, weight="bold")
    metadata = data.get("metadata", {})
    energy = metadata.get("total_energy_ev")
    max_force = metadata.get("max_force_ev_per_angstrom")
    if stage is not None and stage >= 6 and energy is not None and max_force is not None:
        registry.text(ax, 0.50, 0.205, f"E {float(energy):.3f} eV · |F|max {float(max_force):.3f}",
                      ha="center", va="center", fontsize=10, color=DARK_GRAY)
    # A compact visual model pipeline replaces a paragraph of prose.
    registry.text(ax, 0.50, 0.30, "Dᵢ  →  NN  →  εᵢ  →  E, F", ha="center",
                  va="center", fontsize=10, color=INK, weight="bold")
    nodes = [(0.16, "Dᵢ"), (0.39, "NN"), (0.62, "εᵢ"), (0.84, "E,F")]
    for idx, (x, label) in enumerate(nodes):
        ax.add_patch(Ellipse((x, 0.145), 0.10, 0.052, fc="#F7F8F6", ec=LINE_GRAY, lw=1.0, zorder=3))
        registry.text(ax, x, 0.145, label, ha="center", va="center", fontsize=10, color=INK, weight="bold", zorder=4)
        if idx < len(nodes) - 1:
            registry.arrow(ax, (x + 0.055, 0.145), (nodes[idx + 1][0] - 0.055, 0.145), arrowstyle="-|>", mutation_scale=10, lw=1.4, color=LINE_GRAY, zorder=3)
    registry.text(ax, 0.50, 0.070, "r, v → F_DP → r′, v′", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    if returning:
        registry.text(ax, 0.50, 0.025, r"$n\;\rightarrow\;n+1$", ha="center", va="center", fontsize=11, color=NAVY, weight="bold")
    else:
        registry.text(ax, 0.50, 0.025, "MIC links · not chemical bonds", ha="center", va="center", fontsize=10, color=DARK_GRAY)


def _phase(t: float, duration: float = 16.0) -> tuple[int | None, float, bool]:
    """Eight short, legible reveals followed by a two-second loop pause."""
    if t >= duration - 2.0:
        return None, min((t - (duration - 2.0)) / 2.0, 1.0), True
    active = duration - 2.0
    segment = active / 7.0
    stage = min(int(t // segment), 6)
    return stage, (t - stage * segment) / segment, False


def compose(fig, t: float, registry: LayoutRegistry, data: dict[str, object], vv: dict[str, object], a: dict[str, object], *, video: bool) -> list[dict]:
    stage, progress, returning = _phase(t)
    rail, main, info = story_axes(fig)
    # The explanatory rail has three VV nodes; DP's seven visual reveals are
    # finer-grained substeps, so keep the rail index clamped to its 0/1/2
    # semantic stages instead of indexing it with the reveal number.
    rail_stage = None if returning else min(int(stage), 2)
    stage_rail(rail, registry, active=rail_stage, video=video, equation=None, return_phase=returning)
    titles = ("structure", "select centre", "cutoff sphere", "inside / outside", "MIC neighbours", "descriptor", r"$F_{\rm DP}$", "r′, v′")
    panel_box(main, registry, "DEEP POTENTIAL MD" if returning else f"DEEP POTENTIAL · {titles[stage]}", video=video)
    rect = (0.03, 0.08, 0.97, 0.92)
    if returning:
        registry.text(main, 0.035, 0.035, "r′, v′ · pause then repeat", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        _focus_scene(main, registry, data, a["camera"], a["focus_velocity"], a["locator"], video=video)
    elif stage == 0:
        registry.text(main, 0.035, 0.035, "chemical-bond view · 64 H₂O", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, a["initial"], rect=rect)
    elif stage == 1:
        registry.text(main, 0.035, 0.035, "centre O126 · input r, v", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, a["initial"], rect=rect)
    elif stage == 2:
        registry.text(main, 0.035, 0.035, "draw real r_c = 6.0 Å", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, a["initial"], rect=rect, alpha=float(1.0 - np.clip(progress, 0.0, 1.0)))
        fitted = _focus_scene(main, registry, data, a["camera"], a["focus_cutoff"], a["locator"], video=video)
        _mic_overlay(main, registry, data, a["camera"], fitted, video=video, reveal=float(np.clip(progress, 0.0, 1.0)))
    elif stage == 3:
        registry.text(main, 0.035, 0.035, "inside normal · outside faded", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        fitted = _focus_scene(main, registry, data, a["camera"], a["focus_inside"], a["locator"], video=video)
        _mic_overlay(main, registry, data, a["camera"], fitted, video=video, reveal=1.0)
    elif stage == 4:
        registry.text(main, 0.035, 0.035, "minimum-image neighbour vectors", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        fitted = _focus_scene(main, registry, data, a["camera"], a["focus_neighbors"], a["locator"], video=video)
        _mic_overlay(main, registry, data, a["camera"], fitted, video=video, reveal=1.0)
        registry.text(main, 0.035, 0.065, "blue vectors = MIC displacement · not chemical bonds", ha="left", va="bottom", fontsize=10, color=DARK_GRAY)
    elif stage == 5:
        registry.text(main, 0.035, 0.035, "Dᵢ(rᵢⱼ) → shared network", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        _focus_scene(main, registry, data, a["camera"], a["focus_neighbors"], a["locator"], video=video)
    elif stage == 6:
        registry.text(main, 0.035, 0.035, "F_DP(r) · one force query", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        fitted = _focus_scene(main, registry, data, a["camera"], a["focus_force"], a["locator"], video=video)
        _mic_overlay(main, registry, data, a["camera"], fitted, video=video, reveal=1.0)
        registry.text(main, 0.035, 0.065, "F_DP starts at O126 · display length amplified", ha="left", va="bottom", fontsize=10, color=DARK_GRAY)
    else:
        registry.text(main, 0.035, 0.035, "Velocity Verlet: r, v → r′, v′", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        fitted = _focus_scene(main, registry, data, a["camera"], a["focus_velocity"], a["locator"], video=video)
    _info(info, registry, data, vv, video=video, stage=None if returning else stage, returning=returning)
    if not returning:
        draw_legend(rail, registry, (("r, v / input", LAKE_BLUE), ("F_DP", PALE_OLIVE), ("r′, v′", EMERALD)), video=video, y0=0.205)
    return [{"id": "DP-force", "color": PALE_OLIVE, "min_pixels": 100}, {"id": "cutoff", "color": NAVY, "min_pixels": 100}]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--static-only", action="store_true"); args = parser.parse_args()
    data = load_data(); vv = _make_vv_snapshot(data); a = _render_assets(data, vv)
    fig = new_static_figure(); reg = LayoutRegistry(min_font_pt=10, max_font_pt=16, edge_pad_px=18)
    # The still freezes the force-to-propagation transition so the title,
    # native arrow, cutoff sphere and VV return all refer to one state.
    compose(fig, 13.0, reg, data, vv, a, video=False)
    errors = reg.validate(fig)
    if errors: raise RuntimeError("static native DP layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)
    if not args.static_only:
        render_video(stem=STEM, duration_seconds=16.0,
                     draw_frame=lambda f,t,i,r: compose(f,t,r,data,vv,a,video=True),
                     audit_config=simple_audit(("rail","structure","dp_info")), qa_directory=QA_DIR / "_qa",
                     representative_times=[1.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0,15.0,15.5])


if __name__ == "__main__": main()
