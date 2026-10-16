"""Shared drawing helpers for the 08 (PCA / t-SNE / UMAP) and 09 (multi-task fine-tuning) explainers.

Palette, Arial and the frame QA come from common.py (the 04/05/05a series). Card style per the user
(2026-10-17): dark-grey border (DARK_GRAY), light-grey fill, dark-grey step discs, black text only;
colour is used for lines, arrows and points.
One draw function serves both outputs. Both use a plain figure whose size is the final printed size,
so text sizes are true points: static at 300 dpi; video at 1920 px wide (about 144 dpi).
The 08 canvas is 338 x 125 mm (not 16:5), so the video loop here mirrors common.render_video
(same LayoutRegistry, the same common._frame_checks pixel checks and vertical-whitespace limit,
the same every_frame_qa.json and contact-sheet layout) on a free canvas size.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from PIL import Image

import common
from common import (  # noqa: F401  (re-exported for the renderers)
    A_ORANGE, DARK_GRAY, FONT_SIZES, INK, LIGHT_GRAY, LINE_GRAY, MID_GRAY, NAVY, NNMD_ENERGY_TEAL,
    NNMD_ORANGE, R_BLUE, SIGNED_CMAP, STATIC_DPI, V_PURPLE, WHITE, FPS, LayoutRegistry, FONT_MIN_PT,
    FONT_MAX_PT, VIDEO_FONT_FAMILY, json_dump, mix_hex, smoothstep,
)

TEAL = NNMD_ENERGY_TEAL
ORANGE = NNMD_ORANGE
CLUSTER = (R_BLUE, ORANGE, V_PURPLE)
CARD_FILL = mix_hex(LIGHT_GRAY, WHITE, 0.45)   # light grey
CARD_EDGE = DARK_GRAY                            # dark grey
IDLE_FILL = WHITE
IDLE_EDGE = LINE_GRAY
DISC = DARK_GRAY
MM = 1 / 25.4
VIDEO_WIDTH = 1920


def sf(fig) -> float:
    """Line/marker scale; 1.0 because every figure is drawn at its printed size."""
    return 1.0


def new_figure(size_in, dpi):
    return plt.figure(figsize=size_in, dpi=dpi, facecolor=WHITE)


def canvas(fig):
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return ax


def card(ax, rect, *, fill=CARD_FILL, edge=CARD_EDGE, lw=1.1, z=2, alpha=1.0, fig=None, rounding=0.016):
    x0, y0, x1, y1 = rect
    aspect = fig.get_figwidth() / fig.get_figheight()
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={rounding}",
                       mutation_aspect=aspect, fc=fill, ec=edge, lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def disc(ax, fig, x, y, r, colour, alpha=1.0, z=6):
    aspect = fig.get_figwidth() / fig.get_figheight()
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * aspect, fc=colour, ec="none", alpha=alpha, zorder=z))


def number_disc(ax, reg, fig, x, y, n, colour, alpha=1.0):
    """Stage number in a filled disc (Arial has no circled digits); the digit is white on dark grey."""
    disc(ax, fig, x, y, 0.0125, colour, alpha=alpha, z=6)
    reg.text(ax, x, y, str(n), ha="center", va="center_baseline", fontsize=FONT_SIZES["micro"], color=WHITE, zorder=7)


def big_arrow(ax, reg, fig, a, b, colour=MID_GRAY, lw=3.0, ms=22, alpha=1.0, ls="-", rad=0.0, z=5):
    return reg.arrow(ax, a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=colour, alpha=alpha,
                     linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter")


def sub_axes(fig, rect):
    x0, y0, x1, y1 = rect
    ax = fig.add_axes((x0, y0, x1 - x0, y1 - y0)); ax.axis("off")
    return ax


def registry(dpi):
    return LayoutRegistry(min_font_pt=FONT_MIN_PT, max_font_pt=FONT_MAX_PT, edge_pad_px=12 * dpi / 144,
                          font_family=VIDEO_FONT_FAMILY, coerce_min_font=True)


def save_static(draw, path: Path, size_in) -> Path:
    fig = new_figure(size_in, STATIC_DPI); reg = registry(STATIC_DPI)
    draw(fig, reg)
    errors = reg.validate(fig)
    if errors:
        raise RuntimeError(f"static layout failed: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=STATIC_DPI, facecolor=WHITE)
    plt.close(fig)
    return path


def full_panel_audit(max_ws=48) -> dict:
    return {
        "panels": [{"id": "canvas", "rect": [0.005, 0.01, 0.995, 0.99], "min_clearance_px": 0,
                    "allow_touch_edges": ["left", "right", "top", "bottom"]}],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.01, "min_panel_bbox_fill": 0.045,
                       "grid_rows": 12, "grid_columns": 20},
        "bands": [],
        "max_vertical_border_whitespace_px": max_ws,
    }


def render_video_canvas(*, stem, size_in, duration_seconds, draw_frame, qa_directory: Path, representative_times,
                        audit_config=None, frame_dump: Path | None = None) -> Path:
    """common.render_video on a free canvas (width 1920 px, height from size_in), same QA and outputs."""
    audit_config = audit_config or full_panel_audit()
    dpi = VIDEO_WIDTH / size_in[0]
    height_px = int(round(size_in[1] * dpi)); height_px += height_px % 2
    size_in = (size_in[0], height_px / dpi)
    video_dir = common.VIDEO_DIR; video_dir.mkdir(parents=True, exist_ok=True)
    qa = Path(qa_directory); (qa / "video_frames").mkdir(parents=True, exist_ok=True)
    out = video_dir / f"{stem}.mp4"; tmp = video_dir / f"_{stem}.{os.getpid()}.{time.time_ns()}.encoding.mp4"
    n = int(round(duration_seconds * FPS)); rep = {int(round(v * FPS)) for v in representative_times}
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_WIDTH}x{height_px}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frame_checks = getattr(common, "_frame_checks", None)
    vws = getattr(common, "_vertical_border_whitespace", None)
    limit = int(audit_config.get("max_vertical_border_whitespace_px", 48))
    records, thumbs, failures = [], [], []
    tw, th = 320, int(round(320 * height_px / VIDEO_WIDTH))
    fig = new_figure(size_in, dpi)
    try:
        for k in range(n):
            fig.clear(); t = k / FPS; reg = registry(dpi)
            semantics = draw_frame(fig, t, k, reg) or []
            errors = reg.validate(fig)
            rgb = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
            if rgb.shape[:2] != (height_px, VIDEO_WIDTH):
                raise RuntimeError(f"frame size {rgb.shape} != {(height_px, VIDEO_WIDTH)}")
            checks = frame_checks(rgb, audit_config, semantics) if frame_checks else []
            errors += [f"{r.check}: {f.message}" for r in checks for f in r.findings if f.level == "error"]
            top, bottom = vws(rgb) if vws else (0, 0)
            if top > limit: errors.append(f"top border whitespace {top}px exceeds {limit}px")
            if bottom > limit: errors.append(f"bottom border whitespace {bottom}px exceeds {limit}px")
            records.append({"frame": k, "time_seconds": t, "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
                            "passed": not errors, "layout_errors": errors,
                            "vertical_border_whitespace_px": {"top": top, "bottom": bottom, "maximum": limit},
                            "checks": [{"check": r.check, "passed": r.passed, "findings": [f.to_dict() for f in r.findings]} for r in checks]})
            if errors:
                fail = qa / f"failed_frame_{k:04d}.png"; Image.fromarray(rgb).save(fail)
                failures.append({"frame": k, "errors": errors, "image": str(fail)})
                raise RuntimeError(f"Frame {k} failed strict frame QA: " + "; ".join(errors))
            if k % 12 == 0:
                thumbs.append(Image.fromarray(rgb).resize((tw, th), Image.Resampling.LANCZOS))
            if k in rep:
                Image.fromarray(rgb).save(qa / "video_frames" / f"frame_{k:04d}.png")
            if frame_dump is not None and k % 2 == 0:
                frame_dump.mkdir(parents=True, exist_ok=True); Image.fromarray(rgb).save(frame_dump / f"f{k:04d}.png")
            proc.stdin.write(rgb.tobytes())
    finally:
        plt.close(fig)
        proc.stdin.close(); rc = proc.wait()
        report = {"version": 1, "stem": stem, "frame_count_expected": n, "frame_count_audited": len(records), "fps": FPS,
                  "dimensions": [VIDEO_WIDTH, height_px], "size_mm": [round(size_in[0] / MM, 1), round(size_in[1] / MM, 1)],
                  "font_family": VIDEO_FONT_FAMILY, "minimum_font_pt": FONT_MIN_PT, "maximum_font_pt": FONT_MAX_PT,
                  "pixel_checks": "common._frame_checks" if frame_checks else "unavailable (box preview)",
                  "passed": len(records) == n and not failures and rc == 0, "ffmpeg_return_code": rc,
                  "failures": failures, "frames": records}
        (qa / "every_frame_qa.json").write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
        cd = qa / "contact_sheets"; cd.mkdir(parents=True, exist_ok=True)
        for old in cd.glob("contact_*.jpg"):
            old.unlink()
        for pi, s0 in enumerate(range(0, len(thumbs), 48)):
            page = Image.new("RGB", (6 * tw, 8 * th), WHITE)
            for o, im in enumerate(thumbs[s0:s0 + 48]):
                page.paste(im, ((o % 6) * tw, (o // 6) * th))
            page.save(cd / f"contact_{pi:02d}.jpg", quality=92, subsampling=0)
    if rc != 0 or failures or len(records) != n:
        raise RuntimeError("Video was not published because one or more frames failed QA")
    tmp.replace(out)
    return out
