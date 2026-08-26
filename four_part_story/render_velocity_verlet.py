from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    CRIMSON,
    DARK_GRAY,
    GREEN,
    INK,
    LIGHT_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    add_footer,
    add_page_title,
    axes_from_top_slot,
    draw_ball_and_stick,
    draw_three_step_loop,
    draw_vector_arrow,
    new_static_figure,
    render_source_panel,
    render_video,
    save_static,
    smoothstep,
)


ROOT = Path(__file__).resolve().parent
STEM = "01_velocity_verlet"
QA_DIR = ROOT / "_qa" / "01_velocity_verlet"
STATIC_LEFT = (0.045, 0.20, 0.39, 0.86)
STATIC_RIGHT = (0.43, 0.20, 0.965, 0.88)
VIDEO_LEFT = (0.035, 0.20, 0.375, 0.88)
VIDEO_RIGHT = (0.425, 0.20, 0.965, 0.88)


def load_data() -> dict[str, np.ndarray]:
    source = np.load(ROOT / "data" / "vv_h2o_step.npz", allow_pickle=True)
    return {key: source[key] for key in source.files}


def _draw_loop_source(ax, registry: LayoutRegistry) -> None:
    draw_three_step_loop(ax, registry, video=False, active_stage=None)
    registry.text(ax, 0.50, 0.965, "abstract update", ha="center", va="top", fontsize=13, color=INK, weight="bold")
    registry.text(ax, 0.50, 0.075, r"$\mathbf{r}\rightarrow\mathbf{a}\rightarrow\mathbf{v}\rightarrow\mathbf{r}$", ha="center", va="center", fontsize=12, color=DARK_GRAY)


def _draw_static_concrete(ax, registry: LayoutRegistry, data: dict[str, np.ndarray]) -> None:
    r0, r1 = data["positions"]
    v0, v1 = data["velocities"]
    _, a1 = data["accelerations"]
    elements = data["elements"]
    bonds = data["bonds"]
    centre = np.mean(np.vstack((r0, r1)), axis=0)
    columns = [
        (0.035, 0.335, "1  position", NAVY, r"$\mathbf{r}_{n+1}=\mathbf{r}_n+\mathbf{v}_n\Delta t+\frac{1}{2}\mathbf{a}_n\Delta t^2$"),
        (0.350, 0.650, "2  acceleration", CRIMSON, r"$\mathbf{a}_{n+1}=\mathbf{F}(\mathbf{r}_{n+1})/m$"),
        (0.670, 0.970, "3  velocity", GREEN, r"$\mathbf{v}_{n+1}=\mathbf{v}_n+\frac{1}{2}(\mathbf{a}_n+\mathbf{a}_{n+1})\Delta t$"),
    ]
    for index, (x0, x1, label, colour, equation) in enumerate(columns):
        registry.text(ax, (x0 + x1) / 2, 0.965, label, ha="center", va="top", fontsize=13, color=colour, weight="bold")
        molecule_rect = (x0 + 0.01, 0.23, x1 - 0.01, 0.87)
        if index == 0:
            draw_ball_and_stick(ax, r0, elements, bonds, rect=molecule_rect, centre_3d=centre, half_span=1.30, alpha=0.18, atom_scale=0.72, bond_alpha=0.32, edge_color=LIGHT_GRAY)
            draw_ball_and_stick(ax, r1, elements, bonds, rect=molecule_rect, centre_3d=centre, half_span=1.30, alpha=1.0, atom_scale=0.72)
            for atom in range(3):
                draw_vector_arrow(ax, registry, r0[atom], r1[atom] - r0[atom], colour=NAVY, rect=molecule_rect, centre_3d=centre, half_span=1.30, display_scale=float(data["display_displacement_scale"]), video=False)
            registry.text(ax, (x0 + x1) / 2, 0.185, "old → new · Δr arrows ×12", ha="center", va="center", fontsize=10, color=DARK_GRAY)
        elif index == 1:
            draw_ball_and_stick(ax, r1, elements, bonds, rect=molecule_rect, centre_3d=centre, half_span=1.30, atom_scale=0.72)
            for atom in range(3):
                draw_vector_arrow(ax, registry, r1[atom], a1[atom], colour=CRIMSON, rect=molecule_rect, centre_3d=centre, half_span=1.30, display_scale=float(data["display_acceleration_scale"]), video=False)
            registry.text(ax, (x0 + x1) / 2, 0.185, "a arrows ×25", ha="center", va="center", fontsize=10, color=DARK_GRAY)
        else:
            draw_ball_and_stick(ax, r1, elements, bonds, rect=molecule_rect, centre_3d=centre, half_span=1.30, atom_scale=0.72)
            for atom in range(3):
                draw_vector_arrow(ax, registry, r1[atom], v0[atom], colour=LIGHT_GRAY, rect=molecule_rect, centre_3d=centre, half_span=1.30, display_scale=float(data["display_velocity_scale"]), video=False, alpha=0.95)
                draw_vector_arrow(ax, registry, r1[atom], v1[atom], colour=GREEN, rect=molecule_rect, centre_3d=centre, half_span=1.30, display_scale=float(data["display_velocity_scale"]), video=False)
            registry.text(ax, (x0 + x1) / 2, 0.185, "v old grey · new green · ×5", ha="center", va="center", fontsize=10, color=DARK_GRAY)
        registry.text(ax, (x0 + x1) / 2, 0.085, equation, ha="center", va="center", fontsize=10, color=INK)


def render_static(data: dict[str, np.ndarray]) -> None:
    render_source_panel(QA_DIR / "source" / "abstract_loop.png", _draw_loop_source, width_px=1000, height_px=1400)
    render_source_panel(QA_DIR / "source" / "concrete_h2o.png", lambda ax, reg: _draw_static_concrete(ax, reg, data), width_px=1900, height_px=1450)
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    add_page_title(fig, "01", "Velocity Verlet", "one exact step, reduced to position → acceleration → velocity", video=False, registry=registry)
    left = axes_from_top_slot(fig, STATIC_LEFT)
    right = axes_from_top_slot(fig, STATIC_RIGHT)
    _draw_loop_source(left, registry)
    _draw_static_concrete(right, registry, data)
    add_footer(fig, r"H$_2$O · $\Delta t=0.5$ fs · equation residuals $<6\times10^{-17}$ in stored units", video=False, registry=registry)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    png, svg = save_static(fig, STEM)
    print(f"figure: {png}")
    print(f"vector: {svg}")


def _draw_video_frame(fig, time_seconds: float, frame_index: int, registry: LayoutRegistry, data: dict[str, np.ndarray]) -> list[dict]:
    stage = min(int(time_seconds // 3.0), 2)
    local = (time_seconds - stage * 3.0) / 3.0
    active_weight = smoothstep(min(local / 0.18, 1.0))
    add_page_title(fig, "01", "Velocity Verlet", "the integrator is abstract; the H₂O step on the right is concrete", video=True, registry=registry)
    left = axes_from_top_slot(fig, VIDEO_LEFT)
    right = axes_from_top_slot(fig, VIDEO_RIGHT)
    draw_three_step_loop(left, registry, video=True, active_stage=stage, active_weight=active_weight)

    r0, r1 = data["positions"]
    v0, v1 = data["velocities"]
    _, a1 = data["accelerations"]
    elements = data["elements"]
    bonds = data["bonds"]
    centre = np.mean(np.vstack((r0, r1)), axis=0)
    molecule_rect = (0.04, 0.20, 0.96, 0.92)
    labels = ["update position", "evaluate force → acceleration", "finish velocity"]
    colours = [NAVY, CRIMSON, GREEN]
    equations = [
        r"$\mathbf{r}_{n+1}=\mathbf{r}_n+\mathbf{v}_n\Delta t+\frac{1}{2}\mathbf{a}_n\Delta t^2$",
        r"$\mathbf{a}_{n+1}=\mathbf{F}(\mathbf{r}_{n+1})/m$",
        r"$\mathbf{v}_{n+1}=\mathbf{v}_n+\frac{1}{2}(\mathbf{a}_n+\mathbf{a}_{n+1})\Delta t$",
    ]
    registry.text(right, 0.50, 0.965, labels[stage], ha="center", va="top", fontsize=26, color=colours[stage], weight="bold")
    registry.text(right, 0.50, 0.115, equations[stage], ha="center", va="center", fontsize=20, color=INK)
    if stage == 0:
        move = smoothstep(min(local / 0.72, 1.0))
        current = r0 + move * (r1 - r0)
        draw_ball_and_stick(right, r0, elements, bonds, rect=molecule_rect, centre_3d=centre, half_span=1.15, alpha=0.16, atom_scale=1.40, bond_alpha=0.25, edge_color=LIGHT_GRAY)
        draw_ball_and_stick(right, current, elements, bonds, rect=molecule_rect, centre_3d=centre, half_span=1.15, atom_scale=1.40)
        for atom in range(3):
            draw_vector_arrow(right, registry, r0[atom], r1[atom] - r0[atom], colour=NAVY, rect=molecule_rect, centre_3d=centre, half_span=1.15, display_scale=float(data["display_displacement_scale"]), video=True, alpha=max(0.35, move))
        registry.text(right, 0.50, 0.185, "physical move ≤ 0.034 Å · displacement arrows ×12", ha="center", va="center", fontsize=18, color=DARK_GRAY)
    elif stage == 1:
        draw_ball_and_stick(right, r1, elements, bonds, rect=molecule_rect, centre_3d=centre, half_span=1.15, atom_scale=1.40)
        for atom in range(3):
            draw_vector_arrow(right, registry, r1[atom], a1[atom], colour=CRIMSON, rect=molecule_rect, centre_3d=centre, half_span=1.15, display_scale=float(data["display_acceleration_scale"]), video=True, alpha=active_weight)
        registry.text(right, 0.50, 0.185, r"$\mathbf{a}_{n+1}$ arrows ×25 · positions fixed", ha="center", va="center", fontsize=18, color=DARK_GRAY)
    else:
        draw_ball_and_stick(right, r1, elements, bonds, rect=molecule_rect, centre_3d=centre, half_span=1.15, atom_scale=1.40)
        for atom in range(3):
            draw_vector_arrow(right, registry, r1[atom], v0[atom], colour=LIGHT_GRAY, rect=molecule_rect, centre_3d=centre, half_span=1.15, display_scale=float(data["display_velocity_scale"]), video=True, alpha=0.95)
            draw_vector_arrow(right, registry, r1[atom], v1[atom], colour=GREEN, rect=molecule_rect, centre_3d=centre, half_span=1.15, display_scale=float(data["display_velocity_scale"]), video=True, alpha=active_weight)
        registry.text(right, 0.50, 0.185, r"grey $\mathbf{v}_n$ · green $\mathbf{v}_{n+1}$ · arrows ×5", ha="center", va="center", fontsize=18, color=DARK_GRAY)
    add_footer(fig, r"verified H$_2$O step · $\Delta t=0.5$ fs · fixed asymmetric orthographic camera", video=True, registry=registry)
    semantics = [
        {"id": "hydrogen_or_position", "color": NAVY, "min_pixels": 150},
        {"id": "oxygen", "color": CRIMSON, "min_pixels": 150},
    ]
    if stage == 2:
        semantics.append({"id": "velocity", "color": GREEN, "min_pixels": 120})
    return semantics


def render_animation(data: dict[str, np.ndarray]) -> None:
    audit_config = {
        "panels": [
            {"id": "abstract_loop", "rect": list(VIDEO_LEFT), "min_clearance_px": 18},
            {"id": "concrete_h2o", "rect": list(VIDEO_RIGHT), "min_clearance_px": 18},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.02, "min_panel_bbox_fill": 0.38, "grid_rows": 12, "grid_columns": 20},
        "bands": [{"id": "column_gap", "rect": [0.387, 0.20, 0.413, 0.89], "max_ink_pixels": 0}],
    }
    output = render_video(
        stem=STEM,
        duration_seconds=9.0,
        draw_frame=lambda fig, t, i, reg: _draw_video_frame(fig, t, i, reg, data),
        audit_config=audit_config,
        qa_directory=QA_DIR / "_qa",
        representative_times=[1.5, 4.5, 7.5],
    )
    print(f"video: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    render_static(data)
    if not args.static_only:
        render_animation(data)


if __name__ == "__main__":
    main()
