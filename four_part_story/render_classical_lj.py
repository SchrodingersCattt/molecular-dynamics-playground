from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    CRIMSON,
    DARK_GRAY,
    GREEN,
    INK,
    NAVY,
    LayoutRegistry,
    axes_from_top_slot,
    new_static_figure,
    render_source_panel,
    render_video,
    save_static,
    smoothstep,
)
from mattervis_story import (
    STATIC_LEFT,
    STATIC_RIGHT,
    VIDEO_LEFT,
    VIDEO_RIGHT,
    SceneCamera,
    add_story_title,
    camera_for_source,
    draw_vv_loop,
    draw_world_segment,
    make_vector_group,
    place_render,
    render_structure,
    write_provenance_index,
)


ROOT = Path(__file__).resolve().parent
STEM = "02_classical_lj"
QA_DIR = ROOT / "_qa" / "02_classical_lj"
MATTERVIS_DIR = QA_DIR / "source" / "mattervis"
SCENE_RECT = (0.03, 0.09, 0.97, 0.91)
CAMERA_SCALE = 1.50
EQUATIONS = (
    r"$r_{\mathrm{OO}}=|\mathbf{r}_{\mathrm{O}_2}-\mathbf{r}_{\mathrm{O}_1}|$",
    r"$U_{\mathrm{LJ}}(r_{\mathrm{OO}})\;\rightarrow\;\mathbf{F}=-\frac{\partial U}{\partial r}\,\hat{\mathbf{r}}\;\rightarrow\;\mathbf{a}$",
    r"$\mathbf{v}_{n+1}=\mathbf{v}_n+\frac{1}{2}(\mathbf{a}_n+\mathbf{a}_{n+1})\Delta t$",
)


def load_data() -> dict[str, np.ndarray]:
    source = np.load(ROOT / "data" / "classical_lj.npz", allow_pickle=True)
    return {key: source[key] for key in source.files}


def prepare_mattervis(
    data: dict[str, np.ndarray],
) -> tuple[dict[str, list[Path] | Path], SceneCamera]:
    source = ROOT / "data" / "classical_lj_motion.extxyz"
    target = np.mean(data["motion_atomic_positions"], axis=(0, 1))
    overlay_camera = SceneCamera(target=tuple(target), ortho_scale=CAMERA_SCALE)
    q0, q1 = data["oxygen_positions"]
    displacement = q1 - q0
    position_overlays = make_vector_group(
        "oxygen-displacement",
        q0,
        displacement,
        scale=float(data["display_displacement_scale"]),
        color=NAVY,
    )
    plain_paths: list[Path] = []
    position_paths: list[Path] = []
    records: list[dict] = []
    frame_count = len(data["motion_atomic_positions"])
    for frame in range(frame_count):
        camera = camera_for_source(
            source,
            target=target,
            ortho_scale=CAMERA_SCALE,
            frame=frame,
        )
        plain_path = MATTERVIS_DIR / f"motion_{frame:02d}.png"
        position_path = MATTERVIS_DIR / f"position_{frame:02d}.png"
        records.append(
            render_structure(
                source,
                plain_path,
                camera=camera,
                frame=frame,
                view="cluster",
                width=1800,
                height=700,
            )
        )
        records.append(
            render_structure(
                source,
                position_path,
                camera=camera,
                frame=frame,
                view="cluster",
                width=1800,
                height=700,
                vector_overlays=position_overlays,
            )
        )
        plain_paths.append(plain_path)
        position_paths.append(position_path)

    final_frame = frame_count - 1
    final_camera = camera_for_source(
        source,
        target=target,
        ortho_scale=CAMERA_SCALE,
        frame=final_frame,
    )
    force_path = MATTERVIS_DIR / "lj_force.png"
    velocity_path = MATTERVIS_DIR / "velocity.png"
    records.append(
        render_structure(
            source,
            force_path,
            camera=final_camera,
            frame=final_frame,
            view="cluster",
            width=1800,
            height=700,
            vector_overlays=make_vector_group(
                "tip3p-oo-force",
                q1,
                data["molecule_forces"][1],
                scale=float(data["display_force_scale"]),
                color=CRIMSON,
            ),
        )
    )
    records.append(
        render_structure(
            source,
            velocity_path,
            camera=final_camera,
            frame=final_frame,
            view="cluster",
            width=1800,
            height=700,
            vector_overlays=make_vector_group(
                "rigid-water-velocity",
                q1,
                data["molecule_velocities"][1],
                scale=float(data["display_velocity_scale"]),
                color=GREEN,
            ),
        )
    )
    write_provenance_index(MATTERVIS_DIR, records)
    return (
        {
            "plain": plain_paths,
            "position": position_paths,
            "force": force_path,
            "velocity": velocity_path,
        },
        overlay_camera,
    )


def draw_formula(
    ax,
    registry: LayoutRegistry,
    *,
    video: bool,
    active_stage: int | None,
) -> None:
    if video:
        if active_stage is not None:
            registry.text(
                ax,
                0.50,
                0.12,
                EQUATIONS[active_stage],
                ha="center",
                va="center",
                fontsize=18,
                color=INK,
            )
        return
    registry.text(
        ax,
        0.50,
        0.155,
        r"$r_{\mathrm{OO}}\;\rightarrow\;U_{\mathrm{LJ}}(r_{\mathrm{OO}})$",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    registry.text(
        ax,
        0.50,
        0.085,
        r"$\mathbf{F}=-\partial U/\partial r\;\rightarrow\;\mathbf{a}=\mathbf{F}/M_{\mathrm{H_2O}}$",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )


def draw_left(
    ax,
    registry: LayoutRegistry,
    *,
    video: bool,
    active_stage: int | None,
) -> None:
    draw_vv_loop(ax, registry, video=video, active_stage=active_stage)
    draw_formula(ax, registry, video=video, active_stage=active_stage)


def draw_oo_distance(
    ax,
    registry: LayoutRegistry,
    oxygen_positions: np.ndarray,
    *,
    camera: SceneCamera,
    rect: tuple[float, float, float, float],
    video: bool,
) -> None:
    xy = draw_world_segment(
        ax,
        oxygen_positions[0],
        oxygen_positions[1],
        camera=camera,
        rect=rect,
        color=NAVY,
        linewidth=5.5 if video else 3.0,
        linestyle="--",
        zorder=11,
        image_aspect=1800.0 / 700.0,
    )
    midpoint = 0.5 * (xy[0] + xy[1])
    registry.text(
        ax,
        float(midpoint[0]),
        float(midpoint[1] + (0.055 if video else 0.032)),
        r"$r_{\mathrm{OO}}$",
        ha="center",
        va="bottom",
        fontsize=19 if video else 10,
        color=NAVY,
        weight="bold",
        zorder=14,
    )


def draw_stage(
    ax,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    camera: SceneCamera,
    *,
    stage: int,
    rect: tuple[float, float, float, float],
    video: bool,
    progress: float = 1.0,
) -> None:
    plain_paths = scenes["plain"]
    position_paths = scenes["position"]
    assert isinstance(plain_paths, list) and isinstance(position_paths, list)
    if stage == 0:
        place_render(ax, plain_paths[0], rect, alpha=0.15, zorder=3)
        index = int(round(smoothstep(progress) * (len(position_paths) - 1)))
        fitted = place_render(ax, position_paths[index], rect, zorder=5)
        draw_oo_distance(
            ax,
            registry,
            data["motion_oxygen_positions"][index],
            camera=camera,
            rect=fitted,
            video=video,
        )
    elif stage == 1:
        force_path = scenes["force"]
        assert isinstance(force_path, Path)
        fitted = place_render(ax, force_path, rect, zorder=5)
        draw_oo_distance(
            ax,
            registry,
            data["oxygen_positions"][1],
            camera=camera,
            rect=fitted,
            video=video,
        )
    else:
        velocity_path = scenes["velocity"]
        assert isinstance(velocity_path, Path)
        place_render(ax, plain_paths[0], rect, alpha=0.15, zorder=3)
        place_render(ax, velocity_path, rect, zorder=5)


def draw_static_case(
    ax,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    camera: SceneCamera,
) -> None:
    labels = (
        ("O···O geometry", NAVY),
        ("TIP3P O–O LJ → force", CRIMSON),
        ("one Velocity Verlet step", GREEN),
    )
    rows = (
        (0.035, 0.685, 0.965, 0.965),
        (0.035, 0.365, 0.965, 0.645),
        (0.035, 0.045, 0.965, 0.325),
    )
    for stage, (label, color) in enumerate(labels):
        registry.text(
            ax,
            0.055,
            rows[stage][3] - 0.012,
            label,
            ha="left",
            va="top",
            fontsize=12,
            color=color,
            weight="bold",
        )
        draw_stage(
            ax,
            registry,
            data,
            scenes,
            camera,
            stage=stage,
            rect=rows[stage],
            video=False,
        )
    registry.text(
        ax,
        0.50,
        0.025,
        "Highlighted term only; TIP3P water also contains electrostatics.",
        ha="center",
        va="bottom",
        fontsize=10,
        color=DARK_GRAY,
    )


def render_static(
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    camera: SceneCamera,
) -> None:
    render_source_panel(
        QA_DIR / "source" / "integrator.png",
        lambda ax, registry: draw_left(
            ax, registry, video=False, active_stage=None
        ),
        width_px=1000,
        height_px=1500,
    )
    render_source_panel(
        QA_DIR / "source" / "case.png",
        lambda ax, registry: draw_static_case(
            ax, registry, data, scenes, camera
        ),
        width_px=1800,
        height_px=1500,
    )
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    add_story_title(
        fig,
        registry,
        "Classical MD: Lennard–Jones",
        "On a real water dimer, O···O geometry becomes energy, force, and motion",
        video=False,
    )
    draw_left(
        axes_from_top_slot(fig, STATIC_LEFT),
        registry,
        video=False,
        active_stage=None,
    )
    draw_static_case(
        axes_from_top_slot(fig, STATIC_RIGHT),
        registry,
        data,
        scenes,
        camera,
    )
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)


def draw_video_frame(
    fig,
    time_seconds: float,
    frame_index: int,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    camera: SceneCamera,
) -> list[dict]:
    del frame_index
    stage = min(int(time_seconds // 3.0), 2)
    progress = (time_seconds - 3.0 * stage) / 3.0
    add_story_title(
        fig,
        registry,
        "Classical MD: Lennard–Jones",
        "The TIP3P O–O LJ term supplies a force inside the same Velocity Verlet loop",
        video=True,
    )
    left = axes_from_top_slot(fig, VIDEO_LEFT)
    right = axes_from_top_slot(fig, VIDEO_RIGHT)
    draw_left(left, registry, video=True, active_stage=stage)
    headings = (
        "O···O geometry",
        "LJ force → acceleration",
        "velocity update",
    )
    colors = (NAVY, CRIMSON, GREEN)
    registry.text(
        right,
        0.50,
        0.965,
        headings[stage],
        ha="center",
        va="top",
        fontsize=27,
        color=colors[stage],
        weight="bold",
    )
    draw_stage(
        right,
        registry,
        data,
        scenes,
        camera,
        stage=stage,
        rect=SCENE_RECT,
        video=True,
        progress=progress,
    )
    registry.text(
        right,
        0.50,
        0.075,
        "Highlighted LJ term only · TIP3P also contains electrostatics",
        ha="center",
        va="bottom",
        fontsize=18,
        color=DARK_GRAY,
    )
    return [{"id": headings[stage], "color": colors[stage], "min_pixels": 150}]


def render_animation(
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    camera: SceneCamera,
) -> None:
    audit = {
        "panels": [
            {"id": "integrator", "rect": list(VIDEO_LEFT), "min_clearance_px": 18},
            {"id": "mattervis_dimer", "rect": list(VIDEO_RIGHT), "min_clearance_px": 18},
        ],
        "whitespace": {
            "background_threshold": 245,
            "min_ink_fraction": 0.02,
            "min_panel_bbox_fill": 0.20,
            "grid_rows": 12,
            "grid_columns": 20,
        },
        "bands": [
            {
                "id": "column_gap",
                "rect": [0.365, 0.19, 0.385, 0.90],
                "max_ink_pixels": 0,
            }
        ],
    }
    render_video(
        stem=STEM,
        duration_seconds=9.0,
        draw_frame=lambda fig, time, index, registry: draw_video_frame(
            fig, time, index, registry, data, scenes, camera
        ),
        audit_config=audit,
        qa_directory=QA_DIR / "_qa",
        representative_times=[1.5, 4.5, 7.5],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    scenes, camera = prepare_mattervis(data)
    render_static(data, scenes, camera)
    if not args.static_only:
        render_animation(data, scenes, camera)


if __name__ == "__main__":
    main()
