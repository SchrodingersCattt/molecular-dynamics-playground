"""Compact responsive composition shared by the four MD stories.

MatterVis supplies all atom/bond/cell/vector pixels.  This module only lays
those immutable renders into a small, readable three-rail composition and
adds short paper-space labels.  The same normalized slots are used for the
300-dpi A4 still and the 16:9 movie, so a stage has the same visual grammar in
both deliverables while the main structure is allowed to grow on video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from common import (
    DARK_GRAY,
    INK,
    LIGHT_GRAY,
    LINE_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    axes_from_top_slot,
    new_static_figure,
    new_video_figure,
    render_video,
    save_static,
    smoothstep,
)
from mattervis_story import _composition_rgba, place_render, place_render_blend, draw_vv_loop


# A restrained palette shared by all four stories.  Use shape/labels as the
# primary semantics; colours are only a redundant cue.
PALE_OLIVE = "#A89B52"       # force / acceleration
LAKE_BLUE = "#4E9BB5"        # position / displacement
EMERALD = "#2F8562"          # velocity / model output
PALE_BLUE = "#B8D7E2"         # density contours
PALE_OLIVE_FILL = "#EEEBDD"

# Normalized top-coordinate slots.  The rail is intentionally narrow and the
# structure slot is dominant; the right slot holds one concise explanation.
RAIL_SLOT = (0.030, 0.105, 0.270, 0.905)
MAIN_SLOT = (0.290, 0.075, 0.755, 0.905)
INFO_SLOT = (0.775, 0.105, 0.970, 0.905)


def story_axes(fig: plt.Figure) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    return (
        axes_from_top_slot(fig, RAIL_SLOT),
        axes_from_top_slot(fig, MAIN_SLOT),
        axes_from_top_slot(fig, INFO_SLOT),
    )


def panel_box(
    ax: plt.Axes,
    registry: LayoutRegistry,
    title: str,
    *,
    video: bool,
    title_y: float = 0.965,
) -> None:
    ax.add_patch(
        Rectangle(
            (0.005, 0.005), 0.99, 0.99,
            facecolor=WHITE,
            edgecolor=LINE_GRAY,
            linewidth=2.1 if video else 1.35,
            zorder=0,
        )
    )
    registry.text(
        ax, 0.50, title_y, title,
        ha="center", va="top", fontsize=15 if video else 14,
        color=INK, weight="bold", zorder=50,
    )


def stage_rail(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    active: int | None,
    video: bool,
    equation: str | None = None,
    return_phase: bool = False,
) -> None:
    """Draw the abstract position → acceleration → velocity loop."""
    panel_box(ax, registry, "ONE MD STEP", video=video)
    centre_formula = equation or "rₙ₊₁ = rₙ + vₙΔt + ½aₙΔt²"
    draw_vv_loop(
        ax,
        registry,
        video=video,
        active_stage=active,
        centre_text=centre_formula,
        centre_y=0.55,
        radius_x=0.38,
    )
    if return_phase:
        ax.add_patch(
            FancyBboxPatch(
                (0.18, 0.035), 0.64, 0.055,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                facecolor=LIGHT_GRAY, edgecolor=LAKE_BLUE,
                linewidth=1.6, zorder=4,
            )
        )
        registry.text(
            ax, 0.50, 0.062, r"$n\;\rightarrow\;n+1$  next step",
            ha="center", va="center", fontsize=11 if video else 10,
            color=NAVY, weight="bold", zorder=5,
        )


def stage_title(
    ax: plt.Axes,
    registry: LayoutRegistry,
    title: str,
    *,
    video: bool,
    step: str | None = None,
) -> None:
    registry.text(
        ax, 0.50, 0.965, title,
        ha="center", va="top", fontsize=16 if video else 14,
        color=INK, weight="bold", zorder=50,
    )
    if step:
        registry.text(
            ax, 0.035, 0.035, step,
            ha="left", va="bottom", fontsize=11 if video else 10,
            color=DARK_GRAY, zorder=50,
        )


def draw_legend(
    ax: plt.Axes,
    registry: LayoutRegistry,
    entries: Iterable[tuple[str, str]],
    *,
    y0: float = 0.12,
    video: bool,
) -> None:
    entries = list(entries)
    step = 0.065 if video else 0.07
    for index, (label, colour) in enumerate(entries):
        y = y0 - index * step
        ax.plot([0.10, 0.22], [y, y], color=colour, lw=4.5 if video else 3.2, solid_capstyle="round", zorder=8)
        registry.text(ax, 0.28, y, label, ha="left", va="center", fontsize=11 if video else 10, color=INK, zorder=8)


def draw_horizontal_key(
    ax: plt.Axes,
    registry: LayoutRegistry,
    entries: Iterable[tuple[str, str]],
    *,
    y: float = 0.075,
    video: bool,
) -> None:
    """A compact key kept out of the three-step rail.

    The rail is reserved for the integrator state machine.  A horizontal key
    in the large physical panel keeps the color semantics visible without
    colliding with the velocity node or the return marker.
    """
    items = list(entries)
    if not items:
        return
    xs = np.linspace(0.20, 0.80, len(items))
    for x, (label, colour) in zip(xs, items):
        ax.plot([x - 0.065, x - 0.025], [y, y], color=colour,
                lw=4.0 if video else 3.0, solid_capstyle="round", zorder=30)
        registry.text(ax, x - 0.012, y, label, ha="left", va="center",
                      fontsize=11 if video else 10, color=INK, zorder=30)


def place_main(
    ax: plt.Axes,
    image: Path,
    *,
    alpha: float = 1.0,
    rect: tuple[float, float, float, float] = (0.03, 0.09, 0.97, 0.91),
) -> tuple[float, float, float, float]:
    return place_render_cropped(ax, image, rect, alpha=alpha, zorder=10)


def place_render_cropped(
    ax: plt.Axes,
    image_path: Path,
    rect: tuple[float, float, float, float],
    *,
    alpha: float = 1.0,
    zorder: float = 10.0,
    padding: float = 0.06,
) -> tuple[float, float, float, float]:
    """Place a transparent MatterVis render after alpha-bbox cropping.

    MatterVis deliberately exports transparent safety margins.  Cropping only
    the transparent border (never the opaque atom/bond pixels) keeps the
    central structure dominant in both the paper figure and the movie.
    """
    image = np.asarray(_composition_rgba(image_path)).copy()
    alpha_mask = image[:, :, 3] > 8
    if np.any(alpha_mask):
        ys, xs = np.where(alpha_mask)
        h, w = image.shape[:2]
        pad_x = max(2, int(round(float(padding) * (xs.max() - xs.min() + 1))))
        pad_y = max(2, int(round(float(padding) * (ys.max() - ys.min() + 1))))
        x0 = max(0, int(xs.min()) - pad_x)
        x1 = min(w, int(xs.max()) + 1 + pad_x)
        y0 = max(0, int(ys.min()) - pad_y)
        y1 = min(h, int(ys.max()) + 1 + pad_y)
        image = image[y0:y1, x0:x1]
    image_aspect = image.shape[1] / image.shape[0]
    figure_width, figure_height = ax.figure.canvas.get_width_height()
    position = ax.get_position()
    axes_aspect = (position.width * figure_width) / (position.height * figure_height)
    x0, y0, x1, y1 = rect
    width = x1 - x0
    height = y1 - y0
    required_ratio = image_aspect / axes_aspect
    if width / height > required_ratio:
        fitted_width = height * required_ratio
        centre = 0.5 * (x0 + x1)
        x0, x1 = centre - 0.5 * fitted_width, centre + 0.5 * fitted_width
    else:
        fitted_height = width / required_ratio
        centre = 0.5 * (y0 + y1)
        y0, y1 = centre - 0.5 * fitted_height, centre + 0.5 * fitted_height
    ax.imshow(image, extent=(x0, x1, y0, y1), origin="upper",
              interpolation="lanczos", alpha=float(alpha), zorder=zorder, aspect="auto")
    return (x0, y0, x1, y1)


def blend_main(
    ax: plt.Axes,
    first: Path,
    second: Path,
    *,
    blend: float,
    rect: tuple[float, float, float, float] = (0.03, 0.09, 0.97, 0.91),
) -> tuple[float, float, float, float]:
    return place_render_blend(ax, first, second, rect, blend=blend, zorder=10)


def make_static_and_video(
    *,
    stem: str,
    draw_static,
    draw_frame,
    audit_config: dict,
    qa_directory: Path,
    duration: float,
    representative_times: Iterable[float],
) -> None:
    """Publish one clean A4 still and one 16:9 MP4 with the same grammar."""
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10.0, max_font_pt=16.0, edge_pad_px=18)
    draw_static(fig, registry)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("static responsive layout failed:\n" + "\n".join(errors))
    save_static(fig, stem)

    def wrapped(fig, t, i, reg):
        return draw_frame(fig, t, i, reg)

    render_video(
        stem=stem,
        duration_seconds=float(duration),
        draw_frame=wrapped,
        audit_config=audit_config,
        qa_directory=qa_directory,
        representative_times=list(representative_times),
    )


def simple_audit(panel_ids: tuple[str, str, str]) -> dict:
    panels = [
        {"id": panel_ids[0], "rect": list(RAIL_SLOT), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
        {"id": panel_ids[1], "rect": list(MAIN_SLOT), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
        {"id": panel_ids[2], "rect": list(INFO_SLOT), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
    ]
    return {
        "panels": panels,
        "whitespace": {
            "background_threshold": 245,
            "min_ink_fraction": 0.012,
            "min_panel_bbox_fill": 0.16,
            "grid_rows": 12,
            "grid_columns": 24,
        },
        "bands": [
            {"id": "rail_main_gap", "rect": [0.270, 0.06, 0.290, 0.95], "max_ink_pixels": 5000},
            {"id": "main_info_gap", "rect": [0.755, 0.06, 0.775, 0.95], "max_ink_pixels": 5000},
        ],
    }


def timeline_stage(t: float, duration: float, n_stages: int = 3, return_seconds: float = 1.4) -> tuple[int | None, float, bool]:
    """Three equal action stages followed by a legible n→n+1 return pause."""
    active_duration = max(duration - return_seconds, 0.1)
    if t >= active_duration:
        return None, min((t - active_duration) / max(return_seconds, 1e-6), 1.0), True
    segment = active_duration / n_stages
    stage = min(int(t // segment), n_stages - 1)
    local = (t - stage * segment) / segment
    return stage, smoothstep(local), False
