"""04a — how a frame is added when several models disagree.

Same 16:5 canvas and three-column layout as 04: a loop on the left, the
water box in the middle, and the descriptor chain on the right. 04 shows
one forward pass. This movie shows the step that comes before it: four
models read the same fingerprint, and a structure they do not agree on is
sent for a first-principles label.

The middle panel is one frame of the existing 04 water box. The four
energies are marked schematic; they are not a committee dump.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image

from common import (
    A_ORANGE,
    DARK_GRAY,
    ENERGY_TEAL,
    FONT_SIZES,
    INK,
    LINE_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    VIDEO_HEIGHT_PX,
    VIDEO_WIDTH_PX,
    axes_from_top_slot,
    new_video_figure,
)


ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "04a_training"
VIDEO_DURATION = 16.0
SOURCE_04 = ROOT / "videos" / "04_deep_potential_md.mp4"
STILL = ROOT / "qa" / STEM / "water_box_still.png"

VIDEO_A = (0.015, 0.025, 0.200, 0.925)
VIDEO_B = (0.210, 0.025, 0.590, 0.925)
VIDEO_D = (0.600, 0.025, 0.985, 0.925)

# Four schematic energies, in eV, spread on purpose. Not a model dump.
ENERGIES = (-1884.2, -1881.6, -1886.9, -1879.4)
DFT_ENERGY = -1883.1


def water_box_still() -> Image.Image:
    STILL.parent.mkdir(parents=True, exist_ok=True)
    if not STILL.exists():
        if not SOURCE_04.exists():
            raise FileNotFoundError(SOURCE_04)
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-ss", "6", "-i", str(SOURCE_04),
                "-frames:v", "1", str(STILL),
            ],
            check=True,
        )
    image = Image.open(STILL).convert("RGB")
    width, height = image.size
    # Panel B of 04, the periodic box and its magnifier.
    return image.crop((int(0.205 * width), int(0.04 * height), int(0.595 * width), int(0.96 * height)))


def _loop(ax, registry: LayoutRegistry, weights: tuple[float, float, float, float]) -> None:
    labels = ("four\nmodels", "compare", "DFT", "add\nframe")
    positions = ((0.50, 0.78), (0.78, 0.50), (0.50, 0.22), (0.22, 0.50))
    arrows = (
        ((0.62, 0.70), (0.70, 0.60)),
        ((0.70, 0.40), (0.62, 0.30)),
        ((0.38, 0.30), (0.30, 0.40)),
        ((0.30, 0.60), (0.38, 0.70)),
    )
    figure_w, figure_h = ax.figure.canvas.get_width_height()
    position = ax.get_position()
    aspect = (position.width * figure_w) / (position.height * figure_h)
    for start, end in arrows:
        registry.arrow(
            ax, start, end, arrowstyle="-|>", mutation_scale=16,
            lw=2.4, color=LINE_GRAY, zorder=3,
        )
    radius_x = 0.16
    for index, ((x, y), label) in enumerate(zip(positions, labels)):
        weight = weights[index]
        ax.add_patch(
            Ellipse(
                (x, y), 2 * radius_x, 2 * radius_x * aspect,
                fc="#171717" if weight > 0.5 else WHITE,
                ec=INK, lw=1.6, zorder=4,
            )
        )
        registry.text(
            ax, x, y, label, ha="center", va="center",
            fontsize=FONT_SIZES["micro"],
            color=WHITE if weight > 0.5 else INK, zorder=5,
        )


def _right(ax, registry: LayoutRegistry, phase: float) -> None:
    registry.text(
        ax, 0.04, 0.94, "Same fingerprint, four answers",
        ha="left", va="top", fontsize=FONT_SIZES["body"], color=INK,
    )
    registry.text(
        ax, 0.96, 0.94, "schematic",
        ha="right", va="top", fontsize=FONT_SIZES["micro"], color=DARK_GRAY,
    )
    # Neighbour rows become one numerical description.
    show_fp = phase >= 0.15
    ax.add_patch(Rectangle((0.06, 0.62), 0.28, 0.22, fc="#E7EEF6", ec=NAVY, lw=1.2, zorder=2))
    registry.text(ax, 0.20, 0.73, "neighbours", ha="center", va="center", fontsize=FONT_SIZES["micro"], color=NAVY)
    ax.annotate(
        "", xy=(0.42, 0.73), xytext=(0.36, 0.73),
        arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.4),
    )
    face = "#D7EBEB" if show_fp else WHITE
    ax.add_patch(Rectangle((0.44, 0.58), 0.50, 0.28, fc=face, ec=ENERGY_TEAL, lw=1.6, zorder=2))
    registry.text(
        ax, 0.69, 0.72, "chemical fingerprint",
        ha="center", va="center", fontsize=FONT_SIZES["body"], color=INK,
    )
    if phase < 0.35:
        return
    registry.text(
        ax, 0.04, 0.50, "Four models, one structure",
        ha="left", va="center", fontsize=FONT_SIZES["micro"], color=DARK_GRAY,
    )
    reveal = min(int((phase - 0.35) / 0.28 * 4) + 1, 4)
    for index, energy in enumerate(ENERGIES[:reveal]):
        y = 0.40 - index * 0.07
        color = A_ORANGE if phase >= 0.72 else INK
        registry.text(
            ax, 0.08, y, f"model {index + 1}    {energy:.1f} eV",
            ha="left", va="center", fontsize=FONT_SIZES["body"], color=color,
        )
    if phase < 0.72:
        return
    registry.text(
        ax, 0.08, 0.10, f"first principles    {DFT_ENERGY:.1f} eV",
        ha="left", va="center", fontsize=FONT_SIZES["body"], color=ENERGY_TEAL,
    )


def draw_frame(fig, time_seconds: float, still: Image.Image, registry: LayoutRegistry) -> None:
    phase = (time_seconds % 8.0) / 8.0
    weights = tuple(1.0 if phase >= 0.18 * (index + 1) else 0.0 for index in range(4))
    panel_a = axes_from_top_slot(fig, VIDEO_A)
    panel_b = axes_from_top_slot(fig, VIDEO_B)
    panel_d = axes_from_top_slot(fig, VIDEO_D)
    for panel in (panel_a, panel_b, panel_d):
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)
        panel.axis("off")
    _loop(panel_a, registry, weights)  # type: ignore[arg-type]
    panel_b.imshow(still, extent=(0.0, 1.0, 0.0, 1.0), aspect="auto", zorder=1)
    _right(panel_d, registry, phase)


def render() -> Path:
    still = water_box_still()
    output = ROOT / "videos" / f"{STEM}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"_{STEM}.{os.getpid()}.{time.time_ns()}.encoding.mp4")
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{VIDEO_WIDTH_PX}x{VIDEO_HEIGHT_PX}", "-r", "24", "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    figure = new_video_figure()
    total = int(round(VIDEO_DURATION * 24))
    try:
        for frame_index in range(total):
            figure.clear()
            registry = LayoutRegistry(
                min_font_pt=FONT_SIZES["micro"],
                max_font_pt=FONT_SIZES["page_title"],
                edge_pad_px=8,
                font_family="Arial",
                coerce_min_font=True,
            )
            draw_frame(figure, frame_index / 24.0, still, registry)
            figure.canvas.draw()
            rgb = np.ascontiguousarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
            assert process.stdin is not None
            process.stdin.write(rgb.tobytes())
        assert process.stdin is not None
        process.stdin.close()
        process.stdin = None
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {code}")
        temporary.replace(output)
    finally:
        plt.close(figure)
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        if temporary.exists():
            temporary.unlink()
    return output


if __name__ == "__main__":
    print(render())
