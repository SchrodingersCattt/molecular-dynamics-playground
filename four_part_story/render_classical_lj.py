from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse, Rectangle

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
    new_static_figure,
    render_source_panel,
    render_video,
    save_static,
    smoothstep,
)
from mattervis_story import (
    STORY_STATIC_A,
    STORY_STATIC_B,
    STORY_STATIC_C,
    STORY_STATIC_D,
    STORY_VIDEO_A,
    STORY_VIDEO_B,
    STORY_VIDEO_C,
    STORY_VIDEO_D,
    SceneCamera,
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
                fontsize=14,
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
    centre_text = EQUATIONS[active_stage] if active_stage is not None else "\n".join(EQUATIONS)
    draw_vv_loop(
        ax,
        registry,
        video=video,
        active_stage=active_stage,
        centre_text=centre_text,
        centre_y=0.52,
        radius_x=0.40,
    )


def draw_relation_panel(ax, registry: LayoutRegistry, *, video: bool) -> None:
    ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, ec=LINE_GRAY, lw=2.0 if video else 1.1))
    registry.text(ax, 0.50, 0.82, "TIP3P O–O term", ha="center", va="center", fontsize=14 if video else 10, color=INK)
    registry.text(ax, 0.50, 0.54, r"$r_{\mathrm{OO}}\;\rightarrow\;U_{\mathrm{LJ}}(r_{\mathrm{OO}})$", ha="center", va="center", fontsize=14 if video else 10, color=NAVY)
    registry.text(ax, 0.50, 0.34, r"$\mathbf{F}=-\partial U/\partial r\;\rightarrow\;\mathbf{a}$", ha="center", va="center", fontsize=12 if video else 10, color=CRIMSON)
    registry.text(ax, 0.50, 0.10, "LJ term shown · electrostatics omitted", ha="center", va="center", fontsize=10, color=DARK_GRAY)


def draw_lj_loop(ax, registry: LayoutRegistry, *, video: bool, active_stage: int | None) -> None:
    figure_width, figure_height = ax.figure.canvas.get_width_height()
    position = ax.get_position()
    aspect = (position.width * figure_width) / (position.height * figure_height)
    nodes = ((0.50, 0.78, r"$r_{\mathrm{OO}}$"), (0.76, 0.37, r"$U_{\mathrm{LJ}}$"), (0.24, 0.37, r"$\mathbf{F}$"))
    paths = (((0.56, 0.70), (0.70, 0.47)), ((0.65, 0.34), (0.35, 0.34)), ((0.29, 0.47), (0.44, 0.70)))
    for index, (start, end) in enumerate(paths):
        registry.arrow(ax, start, end, arrowstyle="-|>", mutation_scale=18 if video else 12, lw=2.6 if video else 1.6, color=INK if active_stage == index else LINE_GRAY)
    for index, (x, y, symbol) in enumerate(nodes):
        active = active_stage == index
        ax.add_patch(Ellipse((x, y), 0.18, 0.18 * aspect, fc=INK if active else WHITE, ec=INK if active else LINE_GRAY, lw=2.2 if video else 1.3))
        registry.text(ax, x, y, symbol, ha="center", va="center", fontsize=14 if video else 10, color=WHITE if active else INK, weight="bold" if active else "normal")
    registry.text(ax, 0.50, 0.54, "LJ evaluation", ha="center", va="center", fontsize=14 if video else 10, color=INK)


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
        fontsize=14 if video else 10,
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


def draw_case_panel(
    ax,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    camera: SceneCamera,
    *,
    stage: int,
    video: bool,
    progress: float = 1.0,
) -> None:
    ax.add_patch(Rectangle((0.04, 0.04), 0.92, 0.92, fill=False, ec=LINE_GRAY, lw=2.0 if video else 1.1, zorder=20))
    headings = ("Measure O···O geometry", "Evaluate LJ force", "Update velocity")
    registry.text(ax, 0.50, 0.925, headings[stage], ha="center", va="center", fontsize=16 if video else 11, color=INK, zorder=21)
    registry.text(ax, 0.075, 0.070, "Simulation step 01", ha="left", va="bottom", fontsize=12 if video else 10, color=INK, zorder=21)
    draw_stage(ax, registry, data, scenes, camera, stage=stage, rect=SCENE_RECT, video=video, progress=progress)


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
        lambda ax, registry: draw_case_panel(
            ax, registry, data, scenes, camera, stage=1, video=False
        ),
        width_px=1800,
        height_px=1500,
    )
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    draw_left(
        axes_from_top_slot(fig, STORY_STATIC_A),
        registry,
        video=False,
        active_stage=None,
    )
    draw_case_panel(
        axes_from_top_slot(fig, STORY_STATIC_B),
        registry,
        data,
        scenes,
        camera,
        stage=1,
        video=False,
    )
    draw_relation_panel(axes_from_top_slot(fig, STORY_STATIC_C), registry, video=False)
    draw_lj_loop(axes_from_top_slot(fig, STORY_STATIC_D), registry, video=False, active_stage=1)
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
    left = axes_from_top_slot(fig, STORY_VIDEO_A)
    middle = axes_from_top_slot(fig, STORY_VIDEO_B)
    upper_right = axes_from_top_slot(fig, STORY_VIDEO_C)
    lower_right = axes_from_top_slot(fig, STORY_VIDEO_D)
    draw_left(left, registry, video=True, active_stage=stage)
    headings = (
        "O···O geometry",
        "LJ force → acceleration",
        "velocity update",
    )
    colors = (NAVY, CRIMSON, GREEN)
    draw_case_panel(middle, registry, data, scenes, camera, stage=stage, video=True, progress=progress)
    draw_relation_panel(upper_right, registry, video=True)
    draw_lj_loop(lower_right, registry, video=True, active_stage=stage)
    return [{"id": headings[stage], "color": colors[stage], "min_pixels": 150}]


def render_animation(
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    camera: SceneCamera,
) -> None:
    audit = {
        "panels": [
            {"id": "integrator", "rect": list(STORY_VIDEO_A), "min_clearance_px": 12},
            {"id": "mattervis_dimer", "rect": list(STORY_VIDEO_B), "min_clearance_px": 12},
            {"id": "lj_relation", "rect": list(STORY_VIDEO_C), "min_clearance_px": 12},
            {"id": "lj_loop", "rect": list(STORY_VIDEO_D), "min_clearance_px": 12},
        ],
        "whitespace": {
            "background_threshold": 245,
            "min_ink_fraction": 0.02,
            "min_panel_bbox_fill": 0.20,
            "grid_rows": 12,
            "grid_columns": 20,
        },
        "bands": [
            {"id": "gap_a_b", "rect": [0.310, 0.055, 0.325, 0.955], "max_ink_pixels": 0},
            {"id": "gap_b_right", "rect": [0.715, 0.045, 0.745, 0.955], "max_ink_pixels": 0},
            {"id": "gap_c_d", "rect": [0.745, 0.405, 0.965, 0.445], "max_ink_pixels": 0},
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
