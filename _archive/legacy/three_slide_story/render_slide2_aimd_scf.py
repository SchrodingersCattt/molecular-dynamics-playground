"""Render slide 2: an AIMD force query expanded into the real SCF loop."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from common import (
    CRIMSON,
    DARK_GRAY,
    GREEN,
    INK,
    LIGHT_GRAY,
    LINE_GRAY,
    MID_GRAY,
    NAVY,
    VIDEO_DIR,
    WHITE,
    add_title,
    draw_node,
    draw_ring_arrows,
    draw_vectors,
    new_figure,
    ring_points,
    save_static,
    smoothstep,
)


FPS = 24
DURATION = 16.0
LOOP_DURATION = 14.0
STEM = "02_aimd_scf"
OUTER_LABELS = ["state", "half kick", "drift", "force query", "half kick"]
SCF_LABELS = [
    r"density  $\rho^k$",
    r"build  $F[\rho^k]$",
    r"solve  $FC=SC\varepsilon$",
    r"update  $\rho^{k+1}$",
    r"converged?",
]


def load_data() -> dict[str, np.ndarray]:
    path = __import__("pathlib").Path(__file__).resolve().parent / "data" / "h2o_dimer_scf.npz"
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def project_dimer(positions: np.ndarray, center: tuple[float, float], scale: float) -> np.ndarray:
    xy = np.asarray(positions)[:, :2].copy()
    xy -= xy.mean(axis=0)
    return xy * scale + np.asarray(center)


def draw_dimer(
    ax,
    positions: np.ndarray,
    *,
    center: tuple[float, float],
    scale: float,
) -> np.ndarray:
    points = project_dimer(positions, center, scale)
    for oxygen, hydrogen in ((0, 1), (0, 2), (3, 4), (3, 5)):
        ax.plot(
            [points[oxygen, 0], points[hydrogen, 0]],
            [points[oxygen, 1], points[hydrogen, 1]],
            color=DARK_GRAY,
            lw=3.2,
            solid_capstyle="round",
            zorder=8,
        )
    ax.plot(
        [points[1, 0], points[3, 0]],
        [points[1, 1], points[3, 1]],
        color=LINE_GRAY,
        lw=1.8,
        ls=(0, (3, 4)),
        zorder=6,
    )
    hydrogen = np.array([1, 2, 4, 5])
    oxygen = np.array([0, 3])
    ax.scatter(
        points[hydrogen, 0], points[hydrogen, 1],
        s=250, c=NAVY, edgecolors=WHITE, linewidths=1.8, zorder=10,
    )
    ax.scatter(
        points[oxygen, 0], points[oxygen, 1],
        s=500, c=CRIMSON, edgecolors=WHITE, linewidths=2.0, zorder=11,
    )
    return points


def draw_density(
    ax,
    data: dict[str, np.ndarray],
    density_index: int,
    *,
    center: tuple[float, float],
    scale: float,
) -> None:
    density = np.asarray(data["density"][density_index], dtype=float)
    transformed = np.log1p(density)
    transformed /= max(float(transformed.max()), 1.0e-12)
    x = center[0] + data["grid_x"] * scale
    y = center[1] + data["grid_y"] * scale
    ax.contourf(
        x,
        y,
        transformed,
        levels=[0.04, 0.09, 0.16, 0.28, 0.48, 1.01],
        colors=["#FAFAFA", "#F4F4F4", "#EBEBEB", "#DEDEDE", "#D0D0D0"],
        alpha=0.82,
        zorder=4,
    )
    ax.contour(
        x,
        y,
        transformed,
        levels=[0.13, 0.25, 0.42, 0.62, 0.80],
        colors=[LINE_GRAY, LINE_GRAY, MID_GRAY, NAVY, NAVY],
        linewidths=[1.0, 1.0, 1.15, 1.2, 1.2],
        alpha=0.56,
        zorder=5,
    )


def sampled_iterations(count: int) -> np.ndarray:
    raw = np.rint(np.linspace(0, count - 1, 7)).astype(int)
    return np.unique(raw)


def animation_state(time_s: float | None, iteration_count: int) -> tuple[int, int, float, bool]:
    samples = sampled_iterations(iteration_count)
    if time_s is None:
        return int(samples[-1]), -1, 0.0, True
    if time_s >= LOOP_DURATION:
        return int(samples[-1]), 4, 1.0, True
    total_stages = len(samples) * 5
    stage_position = time_s / LOOP_DURATION * total_stages
    stage_number = min(total_stages - 1, int(stage_position))
    local = stage_position - stage_number
    sample_number = stage_number // 5
    active_stage = stage_number % 5
    enter = smoothstep(local / 0.18)
    leave = smoothstep((1.0 - local) / 0.18)
    return int(samples[sample_number]), active_stage, min(enter, leave), False


def draw_outer_loop(ax) -> np.ndarray:
    center = (0.48, 0.535)
    radius = 0.185
    points = ring_points(center, radius, 5)
    draw_ring_arrows(ax, center, radius, 5)
    for index, point in enumerate(points):
        draw_node(ax, point, index, 0.88 if index == 3 else 0.0)
        direction = point - np.asarray(center)
        direction /= np.linalg.norm(direction)
        label = point + direction * 0.052
        horizontal = "center"
        if direction[0] > 0.3:
            horizontal = "left"
        elif direction[0] < -0.3:
            horizontal = "right"
        ax.text(
            label[0], label[1], OUTER_LABELS[index],
            ha=horizontal, va="center", fontsize=17,
            color=INK if index == 3 else MID_GRAY,
            weight="bold" if index == 3 else "normal",
        )
    ax.text(center[0], center[1], "nuclei", ha="center", va="center", fontsize=20, color=DARK_GRAY)
    ax.text(center[0], center[1] - 0.043, "Velocity Verlet", ha="center", va="center", fontsize=17, color=MID_GRAY)
    return points


def draw_scf_loop(
    ax,
    active_stage: int,
    active_weight: float,
    converged: bool,
) -> np.ndarray:
    center = (1.45, 0.535)
    radius = 0.285
    points = ring_points(center, radius, 5)
    draw_ring_arrows(ax, center, radius, 5)
    for index, point in enumerate(points):
        weight = active_weight if index == active_stage else 0.0
        draw_node(ax, point, index, weight)
        direction = point - np.asarray(center)
        direction /= np.linalg.norm(direction)
        label = point + direction * 0.058
        horizontal = "center"
        if direction[0] > 0.3:
            horizontal = "left"
        elif direction[0] < -0.3:
            horizontal = "right"
        color = INK if index == active_stage and active_weight > 0.35 else DARK_GRAY
        ax.text(
            label[0], label[1], SCF_LABELS[index],
            ha=horizontal, va="center", fontsize=18,
            color=color, weight="bold" if index == active_stage else "normal",
        )
    if converged:
        ax.add_patch(Circle(points[4], 0.044, fill=False, ec=GREEN, lw=3.0, zorder=13))
    return points


def draw_frame(fig, data: dict[str, np.ndarray], time_s: float | None) -> list:
    fig.clear()
    fig.patch.set_facecolor(WHITE)
    add_title(
        fig,
        "02",
        "Ab initio molecular dynamics",
        "one force query = a self-consistent electronic problem",
    )
    ax = fig.add_axes([0.03, 0.055, 0.94, 0.77])
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    outer_points = draw_outer_loop(ax)
    iteration, active_stage, active_weight, converged = animation_state(
        time_s, len(data["energies"])
    )
    draw_scf_loop(ax, active_stage, active_weight, converged)

    ax.add_patch(
        FancyArrowPatch(
            outer_points[3] + np.array([0.025, -0.005]),
            (1.145, 0.330),
            connectionstyle="arc3,rad=-0.22",
            arrowstyle="-|>",
            mutation_scale=15,
            lw=2.0,
            color=LINE_GRAY,
            zorder=2,
        )
    )
    ax.text(0.86, 0.255, r"evaluate  $-\nabla_{\mathbf{R}} E$", fontsize=18, color=DARK_GRAY, ha="center")

    scf_center = (1.45, 0.535)
    geometry_scale = 0.055
    draw_density(ax, data, iteration, center=scf_center, scale=geometry_scale)
    atom_points = draw_dimer(ax, data["positions"], center=scf_center, scale=geometry_scale)

    if converged:
        force_alpha = 1.0 if time_s is None else smoothstep((time_s - LOOP_DURATION) / 0.7)
        draw_vectors(
            ax,
            atom_points,
            data["forces"][:, :2],
            color=CRIMSON,
            scale=0.060,
            alpha=force_alpha,
            width=2.3,
        )

    energies = data["energies"]
    if iteration == 0:
        delta_text = "initial density"
    else:
        delta = abs(float(energies[iteration] - energies[iteration - 1]))
        delta_text = rf"$|\Delta E|={delta:.1e}$ Ha"
    ax.text(
        scf_center[0], 0.405,
        f"SCF iter {iteration + 1:02d}/{len(energies):02d}",
        ha="center", va="center", fontsize=18, color=INK, weight="bold",
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 2.5, "alpha": 0.88},
        zorder=14,
    )
    ax.text(
        scf_center[0], 0.366, delta_text,
        ha="center", va="center", fontsize=17, color=DARK_GRAY,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 2.0, "alpha": 0.88},
        zorder=14,
    )

    if converged:
        ax.text(
            1.45, 0.165, r"converged  $\checkmark$  →  $\mathbf{F}=-\nabla E$",
            ha="center", va="center", fontsize=19, color=GREEN, weight="bold",
        )
    elif active_stage == 4:
        ax.text(1.45, 0.165, "not yet  →  next SCF iteration", ha="center", va="center", fontsize=18, color=DARK_GRAY)

    final_delta = abs(float(energies[-1] - energies[-2]))
    fig.text(
        0.50,
        0.035,
        rf"real H2O dimer · RHF/STO-3G · {len(energies)} iterations · final $|\Delta E|={final_delta:.1e}$ Ha",
        ha="center",
        va="bottom",
        fontsize=20,
        color=DARK_GRAY,
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
        bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "AIMD SCF loop — H2O dimer"},
    )
    animation.save(output, writer=writer, dpi=100)
    plt.close(fig)
    print(f"video: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    render_static(data)
    if not args.static_only:
        render_video(data)


if __name__ == "__main__":
    main()
