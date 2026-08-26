"""Render slide 3: Deep Potential on a real 3D periodic water box."""

from __future__ import annotations

import argparse
from itertools import product

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
    mix_hex,
    new_figure,
    ring_points,
    save_static,
    smoothstep,
)


FPS = 24
DURATION = 15.0
STEM = "03_deep_potential_md"
VIEW_DIRECTION = np.array([1.55, -1.0, 0.62], dtype=float)
CAMERA_UP = np.array([0.0, 0.0, 1.0], dtype=float)
OUTER_LABELS = ["state", "half kick", "drift", "force query", "half kick"]
PIPELINE_LABELS = ["structure", "neighbors", "descriptor", r"$\varepsilon_i$", r"$E,\,\mathbf{F}$"]


def load_data() -> dict[str, np.ndarray]:
    path = __import__("pathlib").Path(__file__).resolve().parent / "data" / "water_box_64.npz"
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = VIEW_DIRECTION / np.linalg.norm(VIEW_DIRECTION)
    right = np.cross(CAMERA_UP, view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    return right, up, view


def projection_contract(data: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    length = float(data["box_length"])
    center = np.full(3, length / 2.0)
    right, up, view = camera_basis()
    margin = 1.15
    fit_points = np.asarray(
        list(product([-margin, length + margin], repeat=3)), dtype=float
    )
    relative = fit_points - center
    projected = np.column_stack([relative @ right, relative @ up])
    width = float(np.ptp(projected[:, 0]))
    height = float(np.ptp(projected[:, 1]))
    scale = min(0.72 / width, 0.59 / height)
    return {
        "world_center": center,
        "screen_center": np.array([0.99, 0.535]),
        "right": right,
        "up": up,
        "view": view,
        "scale": scale,
    }


def project(points: np.ndarray, contract: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray]:
    relative = np.asarray(points) - contract["world_center"]
    xy = np.column_stack([relative @ contract["right"], relative @ contract["up"]])
    xy = xy * float(contract["scale"]) + contract["screen_center"]
    depth = relative @ contract["view"]
    return xy, depth


def cell_geometry(length: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    corners = np.asarray(list(product([0.0, length], repeat=3)), dtype=float)
    edges: list[tuple[int, int]] = []
    for i, a in enumerate(corners):
        for j in range(i + 1, len(corners)):
            b = corners[j]
            if np.count_nonzero(np.abs(a - b) > 1.0e-8) == 1:
                edges.append((i, j))
    return corners, edges


def stage_state(time_s: float | None) -> tuple[int, float, float]:
    if time_s is None:
        return -1, 0.0, 1.0
    stage_duration = DURATION / 5.0
    active = min(4, int(time_s / stage_duration))
    local = (time_s - active * stage_duration) / stage_duration
    enter = smoothstep(local / 0.16)
    leave = smoothstep((1.0 - local) / 0.16)
    return active, min(enter, leave), local


def draw_outer_loop(ax) -> np.ndarray:
    center = (0.35, 0.545)
    radius = 0.15
    points = ring_points(center, radius, 5)
    draw_ring_arrows(ax, center, radius, 5)
    for index, point in enumerate(points):
        draw_node(ax, point, index, 0.88 if index == 3 else 0.0)
        direction = point - np.asarray(center)
        direction /= np.linalg.norm(direction)
        label = point + direction * 0.047
        horizontal = "center"
        if direction[0] > 0.3:
            horizontal = "left"
        elif direction[0] < -0.3:
            horizontal = "right"
        ax.text(
            label[0], label[1], OUTER_LABELS[index],
            ha=horizontal, va="center", fontsize=16,
            color=INK if index == 3 else MID_GRAY,
            weight="bold" if index == 3 else "normal",
        )
    ax.text(center[0], center[1], "nuclei", ha="center", va="center", fontsize=20, color=DARK_GRAY)
    ax.text(center[0], center[1] - 0.040, "Velocity Verlet", ha="center", va="center", fontsize=16, color=MID_GRAY)
    return points


def draw_sphere(
    ax,
    data: dict[str, np.ndarray],
    contract: dict[str, np.ndarray | float],
    radius_weight: float,
    alpha: float,
) -> None:
    if radius_weight <= 1.0e-3 or alpha <= 1.0e-3:
        return
    center = np.full(3, float(data["box_length"]) / 2.0)
    radius = float(data["cutoff"]) * radius_weight
    center_2d, _ = project(center[None, :], contract)
    radius_2d = radius * float(contract["scale"])
    ax.add_patch(
        Circle(
            center_2d[0], radius_2d,
            facecolor=NAVY, edgecolor=NAVY,
            lw=2.1, alpha=0.035 * alpha, zorder=3,
        )
    )
    theta = np.linspace(0, 2 * np.pi, 181)
    axes = np.eye(3)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        points = (
            center
            + radius * np.cos(theta)[:, None] * axes[first]
            + radius * np.sin(theta)[:, None] * axes[second]
        )
        projected, _ = project(points, contract)
        ax.plot(
            projected[:, 0], projected[:, 1],
            color=NAVY, lw=1.25, alpha=0.38 * alpha, zorder=4,
        )
    ax.text(
        center_2d[0, 0] + radius_2d * 0.70,
        center_2d[0, 1] + radius_2d * 0.72,
        r"$r_{cut}=6.0$ Å",
        color=NAVY, fontsize=17, ha="left", va="bottom",
    )


def draw_water_box(
    ax,
    data: dict[str, np.ndarray],
    contract: dict[str, np.ndarray | float],
    active_stage: int,
    local: float,
) -> None:
    static = active_stage < 0
    length = float(data["box_length"])
    corners, edges = cell_geometry(length)
    corner_xy, corner_depth = project(corners, contract)
    for first, second in sorted(edges, key=lambda edge: (corner_depth[edge[0]] + corner_depth[edge[1]]) / 2):
        ax.plot(
            [corner_xy[first, 0], corner_xy[second, 0]],
            [corner_xy[first, 1], corner_xy[second, 1]],
            color=LINE_GRAY, lw=1.4, alpha=0.95, zorder=2,
        )

    if static:
        sphere_weight = 1.0
        sphere_alpha = 1.0
        outside_alpha = 0.10
        inside_alpha = 0.78
        bond_alpha = 0.14
        neighbor_line_alpha = 0.52
        positions = data["positions_neighbor_view"]
    elif active_stage == 0:
        sphere_weight = 0.0
        sphere_alpha = 0.0
        outside_alpha = 0.46
        inside_alpha = 0.46
        bond_alpha = 0.72
        neighbor_line_alpha = 0.0
        positions = data["positions_bond_view"]
    elif active_stage == 1:
        sphere_weight = smoothstep(local)
        sphere_alpha = smoothstep(local)
        outside_alpha = 0.46 * (1.0 - smoothstep(local)) + 0.08 * smoothstep(local)
        inside_alpha = 0.46 + 0.32 * smoothstep(local)
        bond_alpha = 0.45 * (1.0 - smoothstep(local))
        neighbor_line_alpha = 0.12 * smoothstep(local)
        positions = data["positions_neighbor_view"]
    else:
        sphere_weight = 1.0
        sphere_alpha = 1.0
        outside_alpha = 0.07
        inside_alpha = 0.78
        bond_alpha = 0.0
        neighbor_line_alpha = 0.82 if active_stage == 2 else 0.34
        positions = data["positions_neighbor_view"]

    draw_sphere(ax, data, contract, sphere_weight, sphere_alpha)
    atom_xy, depth = project(positions, contract)
    central = int(data["central_index"])
    central_molecule = int(data["molecule_ids"][central])
    central_atoms = np.where(data["molecule_ids"] == central_molecule)[0]

    if bond_alpha > 0.001:
        bond_positions = data["positions_bond_view"]
        bond_xy, bond_depth = project(bond_positions, contract)
        for first, second in sorted(
            data["bonds"].astype(int),
            key=lambda edge: (bond_depth[edge[0]] + bond_depth[edge[1]]) / 2,
        ):
            ax.plot(
                [bond_xy[first, 0], bond_xy[second, 0]],
                [bond_xy[first, 1], bond_xy[second, 1]],
                color=DARK_GRAY, lw=1.65, alpha=bond_alpha, zorder=5,
                solid_capstyle="round",
            )

    if neighbor_line_alpha > 0.001:
        candidates = np.where(data["neighbor_mask"])[0]
        closest = candidates[np.argsort(data["neighbor_distances"][candidates])[:14]]
        for index in closest:
            ax.plot(
                [atom_xy[central, 0], atom_xy[index, 0]],
                [atom_xy[central, 1], atom_xy[index, 1]],
                color=DARK_GRAY, lw=1.0, alpha=neighbor_line_alpha, zorder=6,
            )

    order = np.argsort(depth)
    for index in order:
        element = str(data["elements"][index])
        inside = bool(data["neighbor_mask"][index]) or index == central
        if index in central_atoms:
            color = CRIMSON if element == "O" else NAVY
            alpha = 1.0
            # Keep the selected oxygen visually dominant: it is the atom whose
            # local environment is encoded by the cutoff sphere and D_i.
            size = 168 if element == "O" else 58
            edge = WHITE
            width = 1.5
        elif inside:
            color = CRIMSON if element == "O" else NAVY
            alpha = inside_alpha
            size = 48 if element == "O" else 25
            edge = WHITE
            width = 0.7
        else:
            color = LIGHT_GRAY
            alpha = outside_alpha
            size = 37 if element == "O" else 19
            edge = LINE_GRAY
            width = 0.4
        ax.scatter(
            atom_xy[index, 0], atom_xy[index, 1], s=size,
            c=color, alpha=alpha, edgecolors=edge, linewidths=width, zorder=8,
        )

    ax.text(0.99, 0.850, "64 H2O · periodic box", ha="center", va="center", fontsize=18, color=DARK_GRAY)
    if static or active_stage >= 1:
        ax.text(
            0.99, 0.247, "83 neighbors · outside context faded",
            ha="center", va="center", fontsize=17, color=DARK_GRAY,
        )
    elif active_stage == 0:
        ax.text(0.99, 0.247, "chemical-bond view", ha="center", va="center", fontsize=17, color=DARK_GRAY)


def draw_network(ax, active_stage: int, active_weight: float) -> None:
    static = active_stage < 0
    descriptor_active = static or active_stage == 2
    network_active = static or active_stage == 3
    output_active = static or active_stage == 4

    descriptor_weight = 0.72 if static else active_weight if descriptor_active else 0.0
    descriptor_fill = mix_hex(LIGHT_GRAY, INK, descriptor_weight)
    descriptor_text = mix_hex(DARK_GRAY, WHITE, descriptor_weight)
    ax.add_patch(Circle((1.455, 0.545), 0.053, fc=descriptor_fill, ec=LINE_GRAY, lw=1.7, zorder=9))
    ax.text(1.455, 0.545, r"$D_i$", ha="center", va="center", fontsize=20, color=descriptor_text, weight="bold", zorder=10)
    ax.text(1.455, 0.645, "local descriptor", ha="center", va="center", fontsize=17, color=DARK_GRAY)

    ax.add_patch(FancyArrowPatch((1.36, 0.545), (1.395, 0.545), arrowstyle="-|>", mutation_scale=13, color=LINE_GRAY, lw=1.7))
    ax.add_patch(FancyArrowPatch((1.512, 0.545), (1.565, 0.545), arrowstyle="-|>", mutation_scale=13, color=LINE_GRAY, lw=1.7))

    columns = [
        (1.62, np.linspace(0.43, 0.66, 4)),
        (1.73, np.linspace(0.40, 0.69, 5)),
        (1.84, np.linspace(0.45, 0.64, 3)),
    ]
    network_weight = 0.70 if static else active_weight if network_active else 0.0
    edge_color = mix_hex(LINE_GRAY, INK, network_weight)
    node_color = mix_hex(LIGHT_GRAY, INK, network_weight)
    for column_index in range(len(columns) - 1):
        x_a, ys_a = columns[column_index]
        x_b, ys_b = columns[column_index + 1]
        for y_a in ys_a:
            for y_b in ys_b:
                ax.plot([x_a, x_b], [y_a, y_b], color=edge_color, lw=0.65, alpha=0.55, zorder=6)
    for x, ys in columns:
        ax.scatter(
            np.full_like(ys, x), ys, s=86, c=node_color,
            edgecolors=WHITE, linewidths=0.7, zorder=10,
        )
    ax.text(1.73, 0.758, "shared atomic network", ha="center", va="center", fontsize=17, color=DARK_GRAY)

    output_weight = 0.72 if static else active_weight if output_active else 0.0
    output_fill = mix_hex(LIGHT_GRAY, GREEN, output_weight)
    ax.add_patch(Circle((1.965, 0.545), 0.047, fc=output_fill, ec=LINE_GRAY, lw=1.6, zorder=10))
    ax.text(1.965, 0.545, r"$\varepsilon_i$", ha="center", va="center", fontsize=18, color=WHITE if output_weight > 0.45 else DARK_GRAY, weight="bold", zorder=11)
    ax.add_patch(FancyArrowPatch((1.885, 0.545), (1.912, 0.545), arrowstyle="-|>", mutation_scale=13, color=LINE_GRAY, lw=1.7))

    equation_color = INK if output_active and not static else DARK_GRAY
    equation_weight = "bold" if active_stage == 4 else "normal"
    ax.text(1.73, 0.328, r"$E=\sum_i \varepsilon_i$", ha="center", va="center", fontsize=22, color=equation_color, weight=equation_weight)
    ax.text(1.73, 0.276, r"$\mathbf{F}=-\nabla_{\mathbf{R}}E$", ha="center", va="center", fontsize=22, color=equation_color, weight=equation_weight)


def draw_pipeline(ax, active_stage: int, active_weight: float) -> None:
    xs = np.array([0.64, 0.93, 1.20, 1.52, 1.84])
    y = 0.145
    ax.plot([xs[0], xs[-1]], [y, y], color=LINE_GRAY, lw=1.8, zorder=1)
    for index, x in enumerate(xs):
        weight = active_weight if index == active_stage else 0.0
        fill = mix_hex(LIGHT_GRAY, INK, weight)
        text = mix_hex(DARK_GRAY, WHITE, weight)
        ax.add_patch(Circle((x, y), 0.027, fc=fill, ec=LINE_GRAY, lw=1.4, zorder=11))
        ax.text(x, y, str(index + 1), ha="center", va="center", fontsize=15, color=text, weight="bold", zorder=12)
        ax.text(
            x, 0.095, PIPELINE_LABELS[index],
            ha="center", va="center", fontsize=16,
            color=INK if index == active_stage else DARK_GRAY,
            weight="bold" if index == active_stage else "normal",
        )


def draw_frame(fig, data: dict[str, np.ndarray], time_s: float | None) -> list:
    fig.clear()
    fig.patch.set_facecolor(WHITE)
    add_title(
        fig,
        "03",
        "Deep Potential molecular dynamics",
        "replace every SCF loop with one learned local-energy evaluation",
    )
    ax = fig.add_axes([0.03, 0.055, 0.94, 0.77])
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    active_stage, active_weight, local = stage_state(time_s)
    outer_points = draw_outer_loop(ax)
    ax.add_patch(
        FancyArrowPatch(
            outer_points[3] + np.array([0.025, -0.006]),
            (0.605, 0.340),
            connectionstyle="arc3,rad=-0.18",
            arrowstyle="-|>", mutation_scale=14,
            lw=1.8, color=LINE_GRAY, zorder=2,
        )
    )
    ax.text(0.49, 0.275, "learned force query", ha="center", va="center", fontsize=17, color=DARK_GRAY)

    contract = projection_contract(data)
    draw_water_box(ax, data, contract, active_stage, local)
    draw_network(ax, active_stage, active_weight)
    draw_pipeline(ax, active_stage, active_weight)

    fig.text(
        0.50,
        0.035,
        r"64 H2O · $L=12.43$ Å · $\rho=0.997$ g cm$^{-3}$ · $r_{cut}=6.0$ Å · 83 neighbors",
        ha="center", va="bottom", fontsize=19, color=DARK_GRAY,
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
        bitrate=4200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Deep Potential MD — periodic water box"},
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
