"""A compact, reproducible well-tempered metadynamics teaching demo.

The trajectory is one-dimensional so the two things that matter remain visible:
the particle crossing a double-well barrier and the deposited bias recovering
the underlying free-energy profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    DARK_GRAY,
    INK,
    LINE_GRAY,
    NAVY,
    LayoutRegistry,
    axes_from_top_slot,
    new_static_figure,
    render_video,
    save_static,
)


ROOT = Path(__file__).resolve().parent
STEM = "05_well_tempered_metadynamics"
QA_DIR = ROOT / "_qa" / "05_metadynamics"
BLUE = "#4E9BB5"
OLIVE = "#A89B52"
CRIMSON = "#A32035"
GREEN = "#2F8562"


def potential(x: np.ndarray) -> np.ndarray:
    return 1.6 * (x * x - 1.0) ** 2


def simulate() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260905)
    n = 4200
    dt = 0.01
    x = np.zeros(n)
    x[0] = -1.0
    bias = np.zeros(n)
    hills_x: list[float] = []
    hills_h: list[float] = []
    width = 0.16
    hill0 = 0.055
    temperature = 0.23
    delta_t = 1.0
    for step in range(n - 1):
        current = x[step]
        grid = np.asarray(hills_x, dtype=float)
        heights = np.asarray(hills_h, dtype=float)
        if len(grid):
            delta = current - grid
            vb = np.sum(heights * np.exp(-0.5 * (delta / width) ** 2))
            db = np.sum(heights * np.exp(-0.5 * (delta / width) ** 2) * (-delta / width**2))
        else:
            vb, db = 0.0, 0.0
        force = -6.4 * current * (current * current - 1.0) - db
        noise = np.sqrt(2.0 * temperature * dt) * rng.normal()
        # A restrained Langevin step gives a readable left-well residence and
        # a barrier crossing after hills start accumulating at 5 s.
        x[step + 1] = np.clip(current + 0.14 * force * dt + noise * 0.16, -1.55, 1.55)
        if step >= 1250 and step % 24 == 0:
            hill_height = hill0 * np.exp(-vb / (temperature * delta_t))
            hills_x.append(float(current))
            hills_h.append(float(hill_height))
        bias[step] = vb
    if hills_x:
        grid = np.asarray(hills_x)
        heights = np.asarray(hills_h)
        delta = x[-1] - grid
        bias[-1] = np.sum(heights * np.exp(-0.5 * (delta / width) ** 2))
    return {"time": np.arange(n) * dt, "x": x, "bias": bias, "hill_x": np.asarray(hills_x), "hill_h": np.asarray(hills_h)}


def _panel(ax: plt.Axes, reg: LayoutRegistry, title: str, *, video: bool) -> None:
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.005, 0.005), 0.99, 0.99, fc="white", ec=LINE_GRAY, lw=2.0 if video else 1.3))
    reg.text(ax, 0.50, 0.965, title, ha="center", va="top", fontsize=15 if video else 14, color=INK, weight="bold")


def _plot_axes(ax: plt.Axes, data: dict[str, np.ndarray], *, upto: float, video: bool, reg: LayoutRegistry, meta: bool) -> None:
    ax.set_position([0.09, 0.22, 0.84, 0.58]); ax.set_facecolor("white")
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-0.18, 1.95)
    ax.spines[["top", "right"]].set_visible(False); ax.spines["left"].set_color(LINE_GRAY); ax.spines["bottom"].set_color(LINE_GRAY)
    ax.tick_params(labelsize=10, colors=DARK_GRAY)
    grid = np.linspace(-1.55, 1.55, 400)
    ax.plot(grid, potential(grid), color=NAVY, lw=2.4, label="V(x)")
    mask = data["time"] <= upto
    if meta:
        ax.plot(data["x"][mask], potential(data["x"][mask]), color=BLUE, lw=1.5, alpha=0.8)
        px = float(data["x"][np.flatnonzero(mask)[-1]])
        ax.scatter([px], [potential(np.asarray([px]))[0]], s=78, color=CRIMSON, zorder=6)
        reg.text(ax, -1.48, 1.78, "particle crosses the barrier", fontsize=11 if video else 10, color=DARK_GRAY)
    else:
        ax.plot(data["x"][mask], potential(data["x"][mask]), color=BLUE, lw=1.5, alpha=0.85)
        px = float(data["x"][np.flatnonzero(mask)[-1]])
        ax.scatter([px], [potential(np.asarray([px]))[0]], s=78, color=CRIMSON, zorder=6)
        reg.text(ax, -1.48, 1.78, "ordinary MD remains in one well", fontsize=11 if video else 10, color=DARK_GRAY)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([-1.0, 0.0, 1.0]); ax.set_yticks([0.0, 0.75, 1.5])


def compose(fig: plt.Figure, data: dict[str, np.ndarray], t: float, *, video: bool) -> list[dict]:
    reg = LayoutRegistry(min_font_pt=10, max_font_pt=16, edge_pad_px=12)
    left = axes_from_top_slot(fig, (0.04, 0.08, 0.49, 0.93)); right = axes_from_top_slot(fig, (0.53, 0.08, 0.96, 0.93))
    _panel(left, reg, "DOUBLE-WELL DYNAMICS", video=video); _panel(right, reg, "WELL-TEMPERED BIAS", video=video)
    active_time = min(t, float(data["time"][-1]))
    ax = fig.add_axes([0.09, 0.22, 0.38, 0.58]); _plot_axes(ax, data, upto=active_time, video=video, reg=reg, meta=active_time >= 5.0)
    ax2 = fig.add_axes([0.58, 0.22, 0.34, 0.58]); ax2.set_xlim(-1.55, 1.55); ax2.set_ylim(-0.18, 1.95); ax2.spines[["top", "right"]].set_visible(False); ax2.tick_params(labelsize=10, colors=DARK_GRAY)
    grid = np.linspace(-1.55, 1.55, 400); ax2.plot(grid, potential(grid), color=NAVY, lw=2.0, label="V(x)")
    hills = data["hill_x"][data["hill_x"] <= 0.0 if active_time < 8.0 else np.ones(len(data["hill_x"]), dtype=bool)]
    heights = data["hill_h"][: len(hills)] if len(hills) else np.array([])
    if len(hills): ax2.vlines(hills, 0.0, np.minimum(heights * 10.0, 1.5), color=OLIVE, lw=1.3, alpha=0.8)
    mask = data["time"] <= active_time; ax2.plot(grid, potential(grid) - np.nanmax(data["bias"][mask]) * 0.25 if np.any(mask) else potential(grid), color=GREEN, lw=1.8)
    ax2.set_xlabel(""); ax2.set_ylabel(""); ax2.set_xticks([-1.0, 0.0, 1.0]); ax2.set_yticks([0.0, 0.75, 1.5])
    reg.text(left, 0.50, 0.095, "x: collective variable", ha="center", fontsize=10, color=DARK_GRAY)
    reg.text(right, 0.50, 0.095, "green: recovered F(x)  ·  olive: deposited hills", ha="center", fontsize=10, color=DARK_GRAY)
    reg.text(left, 0.05, 0.045, "t = %.1f ps" % active_time, fontsize=11 if video else 10, color=DARK_GRAY)
    reg.text(right, 0.05, 0.045, "Gaussian hills temper as bias accumulates", fontsize=11 if video else 10, color=DARK_GRAY)
    errors = reg.validate(fig)
    if errors: raise RuntimeError("metadynamics layout failed: " + "; ".join(errors))
    return [{"id": "potential", "color": NAVY, "min_pixels": 120}, {"id": "hills", "color": OLIVE, "min_pixels": 100}, {"id": "particle", "color": CRIMSON, "min_pixels": 50}]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--static-only", action="store_true"); args = parser.parse_args()
    data = simulate(); fig = new_static_figure(); compose(fig, data, 16.0, video=False); save_static(fig, STEM)
    if args.static_only:
        return
    def draw_frame(fig, t, _index, _registry):
        return compose(fig, data, t, video=True)
    audit = {
        "panels": [{"id": "left", "rect": [0.04,0.08,0.49,0.93], "min_clearance_px": 0, "allow_touch_edges": ["left","right","top","bottom"]}, {"id":"right", "rect":[0.53,0.08,0.96,0.93], "min_clearance_px":0, "allow_touch_edges":["left","right","top","bottom"]}],
        "whitespace": {"background_threshold":245, "min_ink_fraction":0.012, "min_panel_bbox_fill":0.16, "grid_rows":12, "grid_columns":24},
        "bands": [{"id":"gap", "rect":[0.49,0.08,0.53,0.93], "max_ink_pixels":5000}],
    }
    render_video(stem=STEM, duration_seconds=16.0, draw_frame=draw_frame, audit_config=audit, qa_directory=QA_DIR / "_qa", representative_times=[1,4,6,8,10,12,14,15.5])


if __name__ == "__main__": main()
