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
    add_story_title,
    camera_for_source,
    draw_vv_loop,
    place_render,
    render_structure,
    write_provenance_index,
)


ROOT = Path(__file__).resolve().parent
STEM = "01_velocity_verlet"
QA_DIR = ROOT / "_qa" / "01_velocity_verlet"
MATTERVIS_DIR = QA_DIR / "source" / "mattervis"
SCENE_RECT = (0.04, 0.09, 0.96, 0.91)
EQUATIONS = (
    r"$\mathbf{r}_{n+1}=\mathbf{r}_n+\mathbf{v}_n\Delta t+\frac{1}{2}\mathbf{a}_n\Delta t^2$",
    r"$\mathbf{a}_{n+1}=\mathbf{F}(\mathbf{r}_{n+1})/m$",
    r"$\mathbf{v}_{n+1}=\mathbf{v}_n+\frac{1}{2}(\mathbf{a}_n+\mathbf{a}_{n+1})\Delta t$",
)


def load_data() -> dict[str, np.ndarray]:
    source = np.load(ROOT / "data" / "vv_h2o_step.npz", allow_pickle=True)
    return {key: source[key] for key in source.files}


def vector_group(
    group_id: str,
    origins: np.ndarray,
    vectors: np.ndarray,
    *,
    scale: float,
    color: str,
) -> list[dict]:
    return [
        {
            "id": group_id,
            "name": group_id,
            "magnitude_mode": "scaled",
            "scale": float(scale),
            "viewport_policy": "clip",
            "color": color,
            "arrows": [
                {
                    "id": f"{group_id}-{index}",
                    "origin": np.asarray(origin, dtype=float).tolist(),
                    "vector": np.asarray(vector, dtype=float).tolist(),
                }
                for index, (origin, vector) in enumerate(zip(origins, vectors))
            ],
        }
    ]


def prepare_mattervis(data: dict[str, np.ndarray]) -> dict[str, list[Path] | Path]:
    source = ROOT / "data" / "vv_h2o_motion.extxyz"
    target = np.mean(data["positions"], axis=(0, 1))
    plain_paths: list[Path] = []
    position_paths: list[Path] = []
    records: list[dict] = []
    displacement_overlays = vector_group(
        "displacement",
        data["positions"][0],
        data["positions"][1] - data["positions"][0],
        scale=float(data["display_displacement_scale"]),
        color=NAVY,
    )
    for frame in range(len(data["motion_positions"])):
        camera = camera_for_source(source, target=target, ortho_scale=1.55, frame=frame)
        plain_path = MATTERVIS_DIR / f"motion_{frame:02d}.png"
        position_path = MATTERVIS_DIR / f"position_{frame:02d}.png"
        records.append(render_structure(source, plain_path, camera=camera, frame=frame))
        records.append(
            render_structure(
                source,
                position_path,
                camera=camera,
                frame=frame,
                vector_overlays=displacement_overlays,
            )
        )
        plain_paths.append(plain_path)
        position_paths.append(position_path)
    final_frame = len(data["motion_positions"]) - 1
    final_camera = camera_for_source(
        source,
        target=target,
        ortho_scale=1.55,
        frame=final_frame,
    )
    acceleration_path = MATTERVIS_DIR / "acceleration.png"
    velocity_path = MATTERVIS_DIR / "velocity.png"
    records.append(
        render_structure(
            source,
            acceleration_path,
            camera=final_camera,
            frame=final_frame,
            vector_overlays=vector_group(
                "acceleration",
                data["positions"][1],
                data["accelerations"][1],
                scale=float(data["display_acceleration_scale"]),
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
            vector_overlays=vector_group(
                "velocity",
                data["positions"][1],
                data["velocities"][1],
                scale=float(data["display_velocity_scale"]),
                color=GREEN,
            ),
        )
    )
    write_provenance_index(MATTERVIS_DIR, records)
    return {
        "plain": plain_paths,
        "position": position_paths,
        "acceleration": acceleration_path,
        "velocity": velocity_path,
    }


def draw_formula_stack(ax, registry: LayoutRegistry, *, video: bool, active: int | None) -> None:
    if video:
        if active is not None:
            registry.text(ax, 0.50, 0.12, EQUATIONS[active], ha="center", va="center", fontsize=19, color=INK)
        return
    for index, equation in enumerate(EQUATIONS):
        registry.text(
            ax,
            0.50,
            0.205 - 0.075 * index,
            equation,
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
        )


def draw_static_left(ax, registry: LayoutRegistry) -> None:
    draw_vv_loop(ax, registry, video=False, active_stage=None)
    draw_formula_stack(ax, registry, video=False, active=None)


def draw_stage(
    ax,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
    *,
    stage: int,
    rect: tuple[float, float, float, float],
    video: bool,
    progress: float = 1.0,
) -> None:
    del registry, data, video
    plain_paths = scenes["plain"]
    position_paths = scenes["position"]
    assert isinstance(plain_paths, list) and isinstance(position_paths, list)
    if stage == 0:
        place_render(ax, plain_paths[0], rect, alpha=0.16, zorder=3)
        index = int(round(smoothstep(progress) * (len(position_paths) - 1)))
        place_render(ax, position_paths[index], rect, alpha=1.0, zorder=5)
    elif stage == 1:
        acceleration_path = scenes["acceleration"]
        assert isinstance(acceleration_path, Path)
        place_render(ax, acceleration_path, rect, zorder=5)
    else:
        velocity_path = scenes["velocity"]
        assert isinstance(velocity_path, Path)
        place_render(ax, velocity_path, rect, zorder=5)


def draw_static_case(
    ax,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    scenes: dict[str, list[Path] | Path],
) -> None:
    labels = (("position", NAVY), ("force → acceleration", CRIMSON), ("velocity", GREEN))
    rows = ((0.045, 0.685, 0.955, 0.965), (0.045, 0.365, 0.955, 0.645), (0.045, 0.045, 0.955, 0.325))
    for stage, (label, color) in enumerate(labels):
        registry.text(ax, 0.055, rows[stage][3] - 0.015, label, ha="left", va="top", fontsize=12, color=color, weight="bold")
        draw_stage(ax, registry, data, scenes, stage=stage, rect=rows[stage], video=False)


def render_static(data: dict[str, np.ndarray], scenes: dict[str, list[Path] | Path]) -> None:
    render_source_panel(QA_DIR / "source" / "integrator.png", draw_static_left, width_px=1000, height_px=1500)
    render_source_panel(QA_DIR / "source" / "case.png", lambda ax, reg: draw_static_case(ax, reg, data, scenes), width_px=1800, height_px=1500)
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    add_story_title(fig, registry, "Velocity Verlet", "One exact molecular step: position → acceleration → velocity", video=False)
    draw_static_left(axes_from_top_slot(fig, STATIC_LEFT), registry)
    draw_static_case(axes_from_top_slot(fig, STATIC_RIGHT), registry, data, scenes)
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
) -> list[dict]:
    del frame_index
    stage = min(int(time_seconds // 3.0), 2)
    progress = (time_seconds - 3.0 * stage) / 3.0
    add_story_title(fig, registry, "Velocity Verlet", "The same three operations advance a real H₂O structure", video=True)
    left = axes_from_top_slot(fig, VIDEO_LEFT)
    right = axes_from_top_slot(fig, VIDEO_RIGHT)
    draw_vv_loop(left, registry, video=True, active_stage=stage)
    draw_formula_stack(left, registry, video=True, active=stage)
    headings = ("position update", "force → acceleration", "velocity update")
    colors = (NAVY, CRIMSON, GREEN)
    registry.text(right, 0.50, 0.965, headings[stage], ha="center", va="top", fontsize=27, color=colors[stage], weight="bold")
    draw_stage(right, registry, data, scenes, stage=stage, rect=SCENE_RECT, video=True, progress=progress)
    return [{"id": headings[stage], "color": colors[stage], "min_pixels": 160}]


def render_animation(data: dict[str, np.ndarray], scenes: dict[str, list[Path] | Path]) -> None:
    audit = {
        "panels": [
            {"id": "integrator", "rect": list(VIDEO_LEFT), "min_clearance_px": 18},
            {"id": "mattervis_h2o", "rect": list(VIDEO_RIGHT), "min_clearance_px": 18},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.018, "min_panel_bbox_fill": 0.20, "grid_rows": 12, "grid_columns": 20},
        "bands": [{"id": "column_gap", "rect": [0.365, 0.19, 0.385, 0.90], "max_ink_pixels": 0}],
    }
    render_video(
        stem=STEM,
        duration_seconds=9.0,
        draw_frame=lambda fig, t, i, reg: draw_video_frame(fig, t, i, reg, data, scenes),
        audit_config=audit,
        qa_directory=QA_DIR / "_qa",
        representative_times=[1.5, 4.5, 7.5],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    scenes = prepare_mattervis(data)
    render_static(data, scenes)
    if not args.static_only:
        render_animation(data, scenes)


if __name__ == "__main__":
    main()
