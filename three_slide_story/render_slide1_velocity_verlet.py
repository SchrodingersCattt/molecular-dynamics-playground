"""Render slide 1: Velocity Verlet as a circular loop around a real H2O step."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

from common import (
    CRIMSON,
    DARK_GRAY,
    GREEN,
    INK,
    LINE_GRAY,
    NAVY,
    ROOT,
    VIDEO_DIR,
    add_title,
    draw_node,
    draw_ring_arrows,
    draw_vectors,
    draw_water,
    new_figure,
    ring_points,
    save_static,
    smoothstep,
)


FPS = 24
DURATION = 12.0
STEM = "01_velocity_verlet"
STAGES = [
    ("state", r"$\mathbf{R}_n,\;\mathbf{v}_n,\;\mathbf{a}_n$"),
    ("half kick", r"$\mathbf{v}_{n+1/2}=\mathbf{v}_n+\frac{1}{2}\mathbf{a}_n\Delta t$"),
    ("drift", r"$\mathbf{R}_{n+1}=\mathbf{R}_n+\mathbf{v}_{n+1/2}\Delta t$"),
    ("force query", r"$\mathbf{a}_{n+1}=\mathbf{M}^{-1}[-\nabla U(\mathbf{R}_{n+1})]$"),
    ("half kick", r"$\mathbf{v}_{n+1}=\mathbf{v}_{n+1/2}+\frac{1}{2}\mathbf{a}_{n+1}\Delta t$"),
]


def load_step() -> dict[str, np.ndarray]:
    path = ROOT / "md_workflows" / "classical_data.npz"
    with np.load(path) as data:
        index = 4
        return {
            "r0": np.asarray(data["positions"][index]),
            "r1": np.asarray(data["positions"][index + 1]),
            "v0": np.asarray(data["velocities"][index])[:, [0, 2]],
            "v1": np.asarray(data["velocities"][index + 1])[:, [0, 2]],
            "f0": np.asarray(data["forces"][index])[:, [0, 2]],
            "f1": np.asarray(data["forces"][index + 1])[:, [0, 2]],
        }


def stage_state(time_s: float | None) -> tuple[int, float, float]:
    if time_s is None:
        return -1, 0.0, 0.0
    stage_duration = DURATION / len(STAGES)
    active = min(len(STAGES) - 1, int(time_s / stage_duration))
    local = (time_s - active * stage_duration) / stage_duration
    enter = smoothstep(local / 0.18)
    exit_weight = smoothstep((1.0 - local) / 0.18)
    return active, min(enter, exit_weight), local


def draw_frame(fig, data: dict[str, np.ndarray], time_s: float | None) -> list:
    fig.clear()
    fig.patch.set_facecolor("white")
    add_title(fig, "01", "Velocity Verlet", "one real H2O step · the force query closes the loop")

    ax = fig.add_axes([0.03, 0.055, 0.94, 0.77])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

    center = (0.50, 0.49)
    radius = 0.31
    nodes = ring_points(center, radius, len(STAGES))
    draw_ring_arrows(ax, center, radius, len(STAGES))

    active, active_weight, local = stage_state(time_s)
    for index, node in enumerate(nodes):
        weight = active_weight if index == active else 0.0
        draw_node(ax, node, index, weight)

        direction = node - np.asarray(center)
        direction /= np.linalg.norm(direction)
        label_xy = node + direction * 0.075
        alignment = "center"
        if direction[0] > 0.28:
            alignment = "left"
        elif direction[0] < -0.28:
            alignment = "right"
        text_color = INK if index == active and active_weight > 0.45 else DARK_GRAY
        ax.text(
            label_xy[0], label_xy[1] + 0.020, STAGES[index][0],
            ha=alignment, va="bottom", fontsize=22,
            color=text_color, weight="bold" if index == active else "normal",
        )
        ax.text(
            label_xy[0], label_xy[1] - 0.004, STAGES[index][1],
            ha=alignment, va="top", fontsize=19, color=text_color,
        )

    drift = 0.0
    if active > 2:
        drift = 1.0
    elif active == 2:
        drift = smoothstep(local)
    positions = (1.0 - drift) * data["r0"] + drift * data["r1"]

    molecule_center = (0.50, 0.49)
    old_points = None
    if active == 2 and drift > 0.02:
        old_points = draw_water(ax, data["r0"], center=molecule_center, scale=0.18, alpha=0.18)
    points = draw_water(ax, positions, center=molecule_center, scale=0.18, alpha=1.0)

    velocity_alpha = 0.0
    force_alpha = 0.0
    if time_s is None:
        velocity_alpha = 0.55
        force_alpha = 0.55
    elif active in (0, 1, 4):
        velocity_alpha = 0.25 + 0.75 * active_weight
    elif active == 3:
        force_alpha = 0.25 + 0.75 * active_weight
    elif active == 2 and old_points is not None:
        for old, new in zip(old_points, points):
            ax.plot([old[0], new[0]], [old[1], new[1]], color=NAVY, lw=2.0, alpha=0.7)

    velocities = (1.0 - drift) * data["v0"] + drift * data["v1"]
    forces = (1.0 - drift) * data["f0"] + drift * data["f1"]
    draw_vectors(ax, points, velocities, color=GREEN, scale=0.085, alpha=velocity_alpha)
    draw_vectors(ax, points, forces, color=CRIMSON, scale=0.085, alpha=force_alpha)

    ax.text(0.50, 0.365, "H2O", ha="center", va="top", color=DARK_GRAY, fontsize=18)
    ax.text(0.075, 0.66, "position", color=NAVY, fontsize=18, ha="left")
    ax.plot([0.075, 0.115], [0.635, 0.635], color=NAVY, lw=4)
    ax.text(0.075, 0.58, "velocity", color=GREEN, fontsize=18, ha="left")
    ax.plot([0.075, 0.115], [0.555, 0.555], color=GREEN, lw=4)
    ax.text(0.075, 0.50, "force", color=CRIMSON, fontsize=18, ha="left")
    ax.plot([0.075, 0.115], [0.475, 0.475], color=CRIMSON, lw=4)

    fig.text(
        0.50, 0.035,
        "same integrator · any differentiable potential-energy surface",
        ha="center", va="bottom", fontsize=20, color=DARK_GRAY,
    )
    return []


def render_static(data: dict[str, np.ndarray]) -> None:
    fig = new_figure()
    draw_frame(fig, data, None)
    png, svg = save_static(fig, STEM)
    plt.close(fig)
    print(f"figure: {png}")
    print(f"vector: {svg}")


def render_video(data: dict[str, np.ndarray]) -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    output = VIDEO_DIR / f"{STEM}.mp4"
    fig = new_figure()
    frames = int(DURATION * FPS)
    animation = FuncAnimation(
        fig,
        lambda frame: draw_frame(fig, data, frame / FPS),
        frames=frames,
        interval=1000 / FPS,
        blit=False,
        cache_frame_data=False,
    )
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=3500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Velocity Verlet — H2O"},
    )
    animation.save(output, writer=writer, dpi=100)
    plt.close(fig)
    print(f"video: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_step()
    render_static(data)
    if not args.static_only:
        render_video(data)


if __name__ == "__main__":
    main()
