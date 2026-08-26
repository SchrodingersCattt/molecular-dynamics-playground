"""Shared visual grammar for the three molecular-dynamics slides."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
VIDEO_DIR = HERE / "videos"
DATA_DIR = HERE / "data"

WHITE = "#FFFFFF"
INK = "#161616"
DARK_GRAY = "#4B4B4B"
MID_GRAY = "#8C8C8C"
LINE_GRAY = "#C9C9C9"
LIGHT_GRAY = "#ECECEC"
NAVY = "#183153"
CRIMSON = "#A32035"
GREEN = "#2F6B4F"


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def mix_hex(a: str, b: str, weight: float) -> str:
    weight = float(np.clip(weight, 0.0, 1.0))
    aa = np.array([int(a[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    bb = np.array([int(b[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    cc = np.rint((1.0 - weight) * aa + weight * bb).astype(int)
    return "#" + "".join(f"{channel:02X}" for channel in cc)


def new_figure():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=WHITE)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def add_title(fig, number: str, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.935, number, color=DARK_GRAY, fontsize=20, weight="bold", va="top")
    fig.text(0.055, 0.890, title, color=INK, fontsize=39, weight="bold", va="top")
    fig.text(0.055, 0.835, subtitle, color=DARK_GRAY, fontsize=21, va="top")


def ring_points(center: tuple[float, float], radius: float, n: int, start_deg: float = 90.0) -> np.ndarray:
    angles = np.deg2rad(start_deg - np.arange(n) * 360.0 / n)
    return np.column_stack(
        [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]
    )


def draw_ring_arrows(ax, center: tuple[float, float], radius: float, n: int) -> None:
    step = 360.0 / n
    for index in range(n):
        start = 90.0 - index * step - 10.0
        end = 90.0 - (index + 1) * step + 10.0
        arc = Arc(
            center,
            2 * radius,
            2 * radius,
            theta1=end,
            theta2=start,
            color=LINE_GRAY,
            lw=2.0,
            zorder=1,
        )
        ax.add_patch(arc)
        theta = np.deg2rad(end)
        tangent = np.array([-np.sin(theta), np.cos(theta)])
        point = np.array(center) + radius * np.array([np.cos(theta), np.sin(theta)])
        ax.add_patch(
            FancyArrowPatch(
                point - 0.012 * tangent,
                point + 0.012 * tangent,
                arrowstyle="-|>",
                mutation_scale=12,
                color=LINE_GRAY,
                lw=1.5,
                zorder=2,
            )
        )


def draw_node(ax, xy: np.ndarray, index: int, active_weight: float) -> None:
    fill = mix_hex(LIGHT_GRAY, INK, active_weight)
    edge = mix_hex(LINE_GRAY, INK, active_weight)
    text = mix_hex(DARK_GRAY, WHITE, active_weight)
    ax.add_patch(Circle(xy, 0.035, facecolor=fill, edgecolor=edge, lw=1.7, zorder=4))
    ax.text(
        xy[0],
        xy[1],
        str(index + 1),
        ha="center",
        va="center",
        color=text,
        fontsize=19,
        weight="bold",
        zorder=5,
    )


def project_water(positions: np.ndarray, center: tuple[float, float], scale: float) -> np.ndarray:
    projection = np.asarray(positions)[:, [0, 2]].copy()
    projection -= projection.mean(axis=0)
    return projection * scale + np.asarray(center)


def draw_water(
    ax,
    positions: np.ndarray,
    *,
    center: tuple[float, float],
    scale: float,
    alpha: float = 1.0,
) -> np.ndarray:
    points = project_water(positions, center, scale)
    for hydrogen in (1, 2):
        ax.plot(
            [points[0, 0], points[hydrogen, 0]],
            [points[0, 1], points[hydrogen, 1]],
            color=DARK_GRAY,
            lw=4.0,
            alpha=0.82 * alpha,
            solid_capstyle="round",
            zorder=7,
        )
    ax.scatter(
        points[1:, 0], points[1:, 1], s=420, c=NAVY,
        edgecolors=WHITE, linewidths=2.0, alpha=alpha, zorder=9,
    )
    ax.scatter(
        points[0, 0], points[0, 1], s=760, c=CRIMSON,
        edgecolors=WHITE, linewidths=2.2, alpha=alpha, zorder=10,
    )
    return points


def draw_vectors(
    ax,
    origins: np.ndarray,
    vectors: np.ndarray,
    *,
    color: str,
    scale: float,
    alpha: float,
    width: float = 2.6,
) -> None:
    norms = np.linalg.norm(vectors, axis=1)
    denominator = max(float(norms.max()), 1e-12)
    for origin, vector in zip(origins, vectors):
        direction = vector / denominator * scale
        ax.add_patch(
            FancyArrowPatch(
                origin,
                origin + direction,
                arrowstyle="-|>",
                mutation_scale=16,
                color=color,
                lw=width,
                alpha=alpha,
                zorder=12,
            )
        )


def save_static(fig, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{stem}.png"
    svg = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=100, facecolor=WHITE)
    fig.savefig(svg, facecolor=WHITE)
    return png, svg
