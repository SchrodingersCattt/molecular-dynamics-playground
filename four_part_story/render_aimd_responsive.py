"""Compact AIMD/RHF composition using the existing MatterVis density scenes.

The density PNGs are real fixed-grid 2-D RHF slices produced by the source
pipeline; this wrapper changes only the paper/video composition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch

from common import DARK_GRAY, INK, LINE_GRAY, NAVY, LayoutRegistry, new_static_figure, render_video, save_static
from responsive_story import EMERALD, LAKE_BLUE, PALE_OLIVE, PALE_BLUE, draw_legend, panel_box, place_main, simple_audit, stage_rail, story_axes, timeline_stage

ROOT = Path(__file__).resolve().parent
STEM = "03_aimd_scf"
QA_DIR = ROOT / "_qa" / STEM
ASSET_DIR = QA_DIR / "source" / "mattervis_multistep"
DATA_PATH = ROOT / "data" / "aimd_multistep_h2o_dimer.npz"


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATA_PATH, allow_pickle=True) as source:
        return {key: source[key] for key in source.files}


def assets() -> dict[str, object]:
    return {
        "structure": [ASSET_DIR / f"ion_{i:02d}_structure.png" for i in range(7)],
        "move": [ASSET_DIR / f"ion_{i:02d}_move.png" for i in range(6)],
        "force": [ASSET_DIR / f"ion_{i:02d}_force.png" for i in range(6)],
        "velocity": [ASSET_DIR / f"ion_{i:02d}_velocity.png" for i in range(6)],
        "density": [[ASSET_DIR / f"ion_{i:02d}_scf_{k:02d}_density.png" for k in range(12)] for i in range(7)],
    }


def info_panel(ax, registry: LayoutRegistry, data: dict[str, np.ndarray], *, video: bool, stage: int | None, returning: bool) -> None:
    panel_box(ax, registry, "RHF / AIMD", video=video)
    cards = [
        (0.78, "ion step", r"$R_n\;\rightarrow\;R_{n+1}$", LAKE_BLUE),
        (0.60, "SCF k", "electrons at fixed R", PALE_BLUE),
        (0.42, "residual", "stop at threshold", PALE_OLIVE),
    ]
    for idx, (y, label, value, colour) in enumerate(cards):
        selected = stage == idx
        ax.add_patch(FancyBboxPatch((0.08, y - 0.065), 0.84, 0.13, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor="#EAF2EE" if selected else "#F7F8F6", edgecolor=colour if selected else LINE_GRAY,
                                    linewidth=2.1 if video else 1.4, zorder=2))
        registry.text(ax, 0.17, y + 0.018, label, ha="left", va="center", fontsize=11 if video else 10, color=colour, weight="bold")
        registry.text(ax, 0.17, y - 0.028, value, ha="left", va="center", fontsize=10, color=INK)
    counts = int(data["scf_counts"][0])
    residual = float(data["scf_residuals"][0, counts - 1])
    registry.text(ax, 0.50, 0.22, f"k = {counts} iterations", ha="center", va="center", fontsize=11 if video else 10, color=NAVY, weight="bold")
    registry.text(ax, 0.50, 0.15, f"final residual {residual:.2e}", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.085, "fixed grid · fixed colour scale", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    if returning:
        registry.text(ax, 0.50, 0.025, r"$n\;\rightarrow\;n+1$", ha="center", va="center", fontsize=11, color=NAVY, weight="bold")


def compose(fig, t: float, registry: LayoutRegistry, data: dict[str, np.ndarray], a: dict[str, object], *, video: bool) -> list[dict]:
    stage, progress, returning = timeline_stage(t, 12.0, n_stages=3, return_seconds=2.0)
    rail, main, info = story_axes(fig)
    stage_rail(rail, registry, active=None if returning else stage, video=video, equation=None, return_phase=returning)
    labels = ("ion step → SCF", "SCF iterations at fixed R", "force → velocity → position")
    panel_box(main, registry, "AIMD / RHF" if returning else f"AIMD / RHF · {labels[stage]}", video=video)
    rect = (0.03, 0.10, 0.97, 0.91)
    if returning:
        registry.text(main, 0.035, 0.035, "next ion step · pause", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, a["structure"][-1], rect=rect)
    elif stage == 0:
        registry.text(main, 0.035, 0.035, "R_n → R_{n+1} · real atom positions", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        idx = min(int(round(progress * 5)), 5)
        place_main(main, a["structure"][idx], rect=rect, alpha=0.18)
        place_main(main, a["move"][idx], rect=rect)
    elif stage == 1:
        registry.text(main, 0.035, 0.035, r"fixed R · real $\rho^k$ slice", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        k = min(int(round(progress * 11)), 11)
        place_main(main, a["density"][0][k], rect=rect)
    else:
        registry.text(main, 0.035, 0.035, "F(R) → v → R'", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        idx = min(int(round(progress * 5)), 5)
        place_main(main, a["structure"][idx], rect=rect, alpha=0.16)
        place_main(main, a["force"][idx], rect=rect, alpha=0.85)
        place_main(main, a["velocity"][idx], rect=rect)
    info_panel(info, registry, data, video=video, stage=None if returning else stage, returning=returning)
    if not returning:
        draw_legend(rail, registry, (("position / R", LAKE_BLUE), ("force / SCF", PALE_OLIVE), ("velocity", EMERALD)), video=video, y0=0.205)
    return [{"id": "density", "color": PALE_BLUE, "min_pixels": 120}, {"id": "force", "color": PALE_OLIVE, "min_pixels": 100}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data(); a = assets()
    fig = new_static_figure(); reg = LayoutRegistry(min_font_pt=10, max_font_pt=16, edge_pad_px=18)
    compose(fig, 4.0, reg, data, a, video=False)
    errors = reg.validate(fig)
    if errors: raise RuntimeError("static responsive layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)
    if not args.static_only:
        render_video(stem=STEM, duration_seconds=12.0,
                     draw_frame=lambda f,t,i,r: compose(f,t,r,data,a,video=True),
                     audit_config=simple_audit(("rail","structure","aimd_info")),
                     qa_directory=QA_DIR / "_qa", representative_times=[0.5,4.5,8.5,11.0])


if __name__ == "__main__": main()
