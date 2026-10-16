"""Shared drawing helpers for the 08 (t-SNE/UMAP) and 09 (multi-task fine-tuning) explainers.

Palette, fonts, canvas, fps and the every-frame QA all come from common.py (the 04/05/05a series):
soft rounded cards (round, rounding 0.018, lw 1.1) on pale tints with softened edges, NNMD teal/orange,
R_BLUE/V_PURPLE/A_ORANGE for categories, INK text, LINE_GRAY construction lines. No bold (deck rule).
One draw function serves both outputs: the 1920x600 VideoFigure (slide points via PPT_TEXT_SCALE) and a
13.333 x 4.167 in static figure at 300 dpi (4000 x 1250 px), whose text is in true slide points.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from common import (  # noqa: F401  (re-exported for the renderers)
    A_ORANGE, DARK_GRAY, FONT_SIZES, INK, LIGHT_GRAY, LINE_GRAY, MID_GRAY, NAVY, NNMD_ENERGY_TEAL,
    NNMD_ORANGE, R_BLUE, SIGNED_CMAP, STATIC_DPI, V_PURPLE, WHITE, LayoutRegistry, FONT_MIN_PT,
    FONT_MAX_PT, VIDEO_FONT_FAMILY, json_dump, mix_hex, smoothstep,
)

# 05a card tints (render_05a_tnt_training.py), copied verbatim
STRUCTURE_BG = "#F7F9FA"
INPUT_FILL = "#E7EEF6"
PREDICTION_FILL = "#EEF6F5"
SOFT_TEAL = "#9CC6C2"
LOSS_FILL = "#FFF9F4"
SOFT_ORANGE = "#E9BC96"
UPDATED_FILL = "#E5F0E8"
TEAL = NNMD_ENERGY_TEAL
ORANGE = NNMD_ORANGE
SOFT_NAVY = mix_hex(NAVY, WHITE, 0.55)
CLUSTER = (R_BLUE, ORANGE, V_PURPLE)
STATIC_SIZE_IN = (13.333, 13.333 * 600 / 1920)


def sf(fig) -> float:
    """Physical scale: 1.0 on the 13.333 in static figure, 1.44 on the 19.2 in video figure."""
    return fig.get_figwidth() / 13.333


def canvas(fig):
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return ax


def card(ax, rect, *, fill, edge, lw=1.1, z=2, alpha=1.0, fig=None, rounding=0.018):
    x0, y0, x1, y1 = rect
    k = sf(fig) if fig is not None else 1.0
    aspect = (fig.get_figwidth() / fig.get_figheight()) if fig is not None else 3.2
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={rounding}",
                       mutation_aspect=aspect, fc=fill, ec=edge, lw=lw * k, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def number_disc(ax, reg, fig, x, y, n, colour, alpha=1.0):
    """Stage number in a filled disc (Arial has no circled digits)."""
    disc(ax, fig, x, y, 0.0125, colour, alpha=alpha, z=6)
    reg.text(ax, x, y, str(n), ha="center", va="center_baseline", fontsize=FONT_SIZES["micro"], color=WHITE, alpha=alpha, zorder=7)


def disc(ax, fig, x, y, r, colour, alpha=1.0, z=6):
    from matplotlib.patches import Ellipse
    aspect = fig.get_figwidth() / fig.get_figheight()
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * aspect, fc=colour, ec="none", alpha=alpha, zorder=z))


def big_arrow(ax, reg, fig, a, b, colour=MID_GRAY, lw=3.0, ms=22, alpha=1.0, ls="-", rad=0.0, z=5):
    k = sf(fig)
    return reg.arrow(ax, a, b, arrowstyle="-|>", mutation_scale=ms * k, lw=lw * k, color=colour, alpha=alpha,
                     linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter")


def sub_axes(fig, rect):
    x0, y0, x1, y1 = rect
    ax = fig.add_axes((x0, y0, x1 - x0, y1 - y0)); ax.axis("off")
    return ax


def save_static(draw, path: Path) -> Path:
    """Render the final state of a draw function at 300 dpi on the static canvas (text in slide pt)."""
    fig = plt.figure(figsize=STATIC_SIZE_IN, dpi=STATIC_DPI, facecolor=WHITE)
    reg = LayoutRegistry(min_font_pt=FONT_MIN_PT, max_font_pt=FONT_MAX_PT, edge_pad_px=12 * STATIC_DPI / 144,
                         font_family=VIDEO_FONT_FAMILY, coerce_min_font=True)
    draw(fig, reg)
    errors = reg.validate(fig)
    if errors:
        raise RuntimeError(f"static layout failed: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STATIC_DPI, facecolor=WHITE)
    plt.close(fig)
    return path


def full_panel_audit(max_ws=48) -> dict:
    """audit_config for common.render_video: one full-canvas panel (layout is a fixed grid of cards)."""
    return {
        "panels": [{"id": "canvas", "rect": [0.005, 0.01, 0.995, 0.99], "min_clearance_px": 0,
                    "allow_touch_edges": ["left", "right", "top", "bottom"]}],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.01, "min_panel_bbox_fill": 0.045,
                       "grid_rows": 12, "grid_columns": 20},
        "bands": [],
        "max_vertical_border_whitespace_px": max_ws,
    }
