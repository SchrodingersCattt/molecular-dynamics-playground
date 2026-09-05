"""Responsive Deep-Potential MD neighbourhood story.

The atomistic box and force scene are rendered through MatterVis.  The
neighbourhood reveal is kept as a separate explanatory layer so the cutoff
sphere and MIC selection remain legible at paper scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch

from common import DARK_GRAY, INK, LINE_GRAY, NAVY, LayoutRegistry, new_static_figure, render_video, save_static
from mattervis_story import camera_for_source, make_vector_group, render_structure
from responsive_story import EMERALD, LAKE_BLUE, PALE_OLIVE, draw_legend, panel_box, place_main, simple_audit, stage_rail, story_axes, timeline_stage

ROOT = Path(__file__).resolve().parent
STEM = "04_deep_potential_md"
QA_DIR = ROOT / "_qa" / "04_dpmd"
SOURCE = ROOT / "data" / "water_box_64.extxyz"
BASE = ROOT / "data" / "water_box_64.npz"
RESULT = ROOT / "data" / "dpmd_water_box_results.npz"
BOX_IMAGE = QA_DIR / "mattervis" / "water_box.png"
ENV_IMAGE = QA_DIR / "source" / "water_environment.png"
FORCE_IMAGE = QA_DIR / "mattervis" / "water_box_dp_force.png"


def load_data() -> dict[str, object]:
    with np.load(BASE, allow_pickle=False) as b:
        data = {key: b[key] for key in b.files}
    with np.load(RESULT, allow_pickle=False) as r:
        data.update({f"result_{key}": r[key] for key in r.files})
    meta = json.loads((ROOT / "data" / "dpmd_eval.json").read_text(encoding="utf-8"))
    data["metadata"] = meta
    return data


def mattervis_assets(data: dict[str, object]) -> dict[str, object]:
    """Ensure a fixed-camera MatterVis box/force pair exists."""
    BOX_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    target = np.mean(np.asarray(data["positions_wrapped"], dtype=float), axis=0)
    camera = camera_for_source(SOURCE, target=target, ortho_scale=16.0, frame=0)
    # Refresh the force image when the old box is absent; retaining the
    # existing public MatterVis box avoids needless full-box re-rendering.
    if not FORCE_IMAGE.exists():
        central = int(np.asarray(data["central_index"]).reshape(-1)[0])
        force = np.asarray(data["result_forces_ev_per_angstrom"], dtype=float)[central]
        vectors = make_vector_group("DP-force", np.asarray(data["positions_wrapped"])[central:central + 1], force[None, :],
                                    scale=1.8, color=PALE_OLIVE,
                                    style={"shaft_radius": 0.045, "head_length_ratio": 0.30, "head_radius_ratio": 2.2, "sides": 16})
        render_structure(SOURCE, FORCE_IMAGE, camera=camera, frame=0, view="unit_cell", width=1700, height=1180,
                         atom_scale=0.72, bond_radius=0.075, show_cell=True, vector_overlays=vectors)
    return {"box": BOX_IMAGE, "force": FORCE_IMAGE, "environment": ENV_IMAGE, "camera": camera}


def info_panel(ax, registry: LayoutRegistry, data: dict[str, object], *, video: bool, stage: int | None, returning: bool) -> None:
    panel_box(ax, registry, "DP LOCAL FIELD", video=video)
    cards = [
        (0.78, "centre", "O126 · r, v", LAKE_BLUE),
        (0.59, "cutoff", "r_c = 6.0 Å", NAVY),
        (0.40, "selected", "83 atomic neighbors", PALE_OLIVE),
    ]
    for idx, (y, label, value, colour) in enumerate(cards):
        selected = stage == idx
        ax.add_patch(FancyBboxPatch((0.08, y - 0.065), 0.84, 0.13, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor="#EAF2EE" if selected else "#F7F8F6", edgecolor=colour if selected else LINE_GRAY,
                                    linewidth=2.1 if video else 1.4, zorder=2))
        registry.text(ax, 0.17, y + 0.018, label, ha="left", va="center", fontsize=11 if video else 10, color=colour, weight="bold")
        registry.text(ax, 0.17, y - 0.028, value, ha="left", va="center", fontsize=10, color=INK)
    meta = data["metadata"]
    max_force = float(meta.get("max_force_ev_per_angstrom", np.nan))
    registry.text(ax, 0.50, 0.22, f"max |F_DP| = {max_force:.2f} eV/Å", ha="center", va="center", fontsize=10 if video else 10, color=NAVY, weight="bold")
    registry.text(ax, 0.50, 0.14, "MIC images · not chemical bonds", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.075, "local environment only", ha="center", va="center", fontsize=10, color=DARK_GRAY)
    if returning:
        registry.text(ax, 0.50, 0.025, r"$n\;\rightarrow\;n+1$", ha="center", va="center", fontsize=11, color=NAVY, weight="bold")


def compose(fig, t: float, registry: LayoutRegistry, data: dict[str, object], a: dict[str, object], *, video: bool) -> list[dict]:
    stage, progress, returning = timeline_stage(t, 12.0, n_stages=3, return_seconds=2.0)
    rail, main, info = story_axes(fig)
    stage_rail(rail, registry, active=None if returning else stage, video=video, equation=None, return_phase=returning)
    labels = ("r, v · central atom", "6 Å cutoff reveal", "F_DP → r′, v′")
    panel_box(main, registry, "DEEP POTENTIAL MD" if returning else f"DEEP POTENTIAL MD · {labels[stage]}", video=video)
    rect = (0.03, 0.08, 0.97, 0.92)
    if returning:
        registry.text(main, 0.035, 0.035, "r′, v′ · next step", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, a["box"], rect=rect)
    elif stage == 0:
        registry.text(main, 0.035, 0.035, "central atom O126 · input r, v", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, a["box"], rect=rect)
    elif stage == 1:
        registry.text(main, 0.035, 0.035, "reveal 6 Å sphere → select 83 neighbours", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        # The explanatory environment image contains the fixed cell, sphere,
        # MIC links, central O126, and the exact neighbour mask.
        place_main(main, a["environment"], rect=rect)
    else:
        registry.text(main, 0.035, 0.035, "F_DP(r) → updated r′, v′", ha="left", va="bottom", fontsize=11 if video else 10, color=DARK_GRAY)
        place_main(main, a["box"], rect=rect, alpha=0.18)
        place_main(main, a["force"], rect=rect)
    info_panel(info, registry, data, video=video, stage=None if returning else stage, returning=returning)
    if not returning:
        draw_legend(rail, registry, (("r, v / input", LAKE_BLUE), ("F_DP", PALE_OLIVE), ("r′, v′", EMERALD)), video=video, y0=0.205)
    return [{"id": "DP-force", "color": PALE_OLIVE, "min_pixels": 100}, {"id": "cutoff", "color": NAVY, "min_pixels": 100}]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--static-only", action="store_true"); args = parser.parse_args()
    data = load_data(); a = mattervis_assets(data)
    fig = new_static_figure(); reg = LayoutRegistry(min_font_pt=10, max_font_pt=16, edge_pad_px=18)
    compose(fig, 4.0, reg, data, a, video=False)
    errors = reg.validate(fig)
    if errors: raise RuntimeError("static responsive layout failed:\n" + "\n".join(errors))
    save_static(fig, STEM)
    if not args.static_only:
        render_video(stem=STEM, duration_seconds=12.0,
                     draw_frame=lambda f,t,i,r: compose(f,t,r,data,a,video=True),
                     audit_config=simple_audit(("rail","structure","dp_info")), qa_directory=QA_DIR / "_qa",
                     representative_times=[0.5,4.5,8.5,11.0])


if __name__ == "__main__": main()
