"""04a — one transparent neural-network training step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Ellipse, Rectangle

from common import (
    A_ORANGE, ENERGY_TEAL, FONT_SIZES, INK, LINE_GRAY, LIGHT_GRAY,
    NAVY, WHITE, LayoutRegistry, axes_from_top_slot, json_dump,
    new_video_figure, render_video, smoothstep,
)
from mattervis_story import _composition_rgba
from render_nnmd_end_to_end import _axes_aspect, camera_for_source, draw_box_magnifier, load_data, prepare_assets

ROOT = Path(__file__).resolve().parents[2] / "product"
DATA_DIR = ROOT / "data"
STEM = "04a_training"
MANIFEST = DATA_DIR / "04a_training_demo.json"
QA_ROOT = ROOT / "qa" / STEM
VIDEO_DURATION = 16.0

PANEL_STRUCTURE = (0.015, 0.035, 0.335, 0.955)
PANEL_NETWORK = (0.350, 0.035, 0.665, 0.955)
PANEL_METRICS = (0.680, 0.035, 0.985, 0.955)
STRUCTURE_BG = "#F7F9FA"
INPUT_FILL = "#E7EEF6"
PREDICTION_FILL = "#D7EBEB"
LOSS_FILL = "#FFF1E6"
UPDATED_FILL = "#E5F0E8"
GRADIENT_ORANGE = A_ORANGE


def _fmt_ev(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace("-", "\N{MINUS SIGN}")


def _load_demo() -> dict[str, object]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    atoms = payload.get("atoms", [])
    if len(atoms) != 3:
        raise ValueError("04a demo manifest must contain three teaching atoms")
    required = {"id", "element", "reference_ev", "prediction_initial_ev", "prediction_updated_ev"}
    for atom in atoms:
        if not required.issubset(atom):
            raise ValueError(f"Missing fields in teaching atom: {atom}")
    ref = np.asarray([float(atom["reference_ev"]) for atom in atoms], dtype=float)
    initial = np.asarray([float(atom["prediction_initial_ev"]) for atom in atoms], dtype=float)
    updated = np.asarray([float(atom["prediction_updated_ev"]) for atom in atoms], dtype=float)
    delta = initial - ref
    updated_delta = updated - ref
    if not np.all(np.abs(updated_delta) < np.abs(delta)):
        raise ValueError("Updated predictions must reduce every teaching error")
    loss = 0.5 * float(delta[0] ** 2)
    updated_loss = 0.5 * float(updated_delta[0] ** 2)
    if not updated_loss < loss:
        raise ValueError("Updated focus loss must be smaller than the initial loss")
    payload["derived"] = {
        "delta_ev": delta.tolist(),
        "updated_delta_ev": updated_delta.tolist(),
        "focus_loss_ev2": loss,
        "updated_focus_loss_ev2": updated_loss,
        "sum_reference_ev": float(ref.sum()),
        "sum_prediction_initial_ev": float(initial.sum()),
        "sum_prediction_updated_ev": float(updated.sum()),
    }
    payload["_arrays"] = {"reference": ref, "initial": initial, "updated": updated}
    return payload


def _slot_axes(fig: plt.Figure, slot: tuple[float, float, float, float]) -> plt.Axes:
    ax = axes_from_top_slot(fig, slot)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    return ax


def _panel_background(ax: plt.Axes) -> None:
    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, fc=WHITE, ec="none", zorder=-10))


def _draw_structure(ax: plt.Axes, registry: LayoutRegistry, demo: dict[str, object], scene: dict[str, object], time_seconds: float) -> None:
    _panel_background(ax)
    registry.text(
        ax, 0.03, 0.965, r"DFT training sample: structure $\mathbf{R}$",
        ha="left", va="top", fontsize=FONT_SIZES["panel_title"], color=INK,
    )
    reveal = smoothstep(np.clip((time_seconds - 0.6) / 0.8, 0.0, 1.0))
    image = _composition_rgba(scene["assets"]["cut"][0])
    draw_box_magnifier(
        ax, scene["data"], scene, image=image, state=0,
        box_center=(0.185, 0.615), box_radius=0.125,
        magnifier_center=(0.675, 0.570), magnifier_radius=0.275,
        reveal=reveal, video=True, zorder=2.0,
    )
    registry.text(
        ax, 0.185, 0.330, "64 H$_2$O periodic box",
        ha="center", va="top", fontsize=FONT_SIZES["micro"], color=INK,
    )
    registry.text(
        ax, 0.675, 0.245, "local environment around O126",
        ha="center", va="top", fontsize=FONT_SIZES["micro"], color=INK,
    )
    registry.text(
        ax, 0.05, 0.105, "DFT reference · schematic",
        ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK,
    )
    registry.text(
        ax, 0.05, 0.045, "one structure  →  one target",
        ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK,
    )

def _heat_block(ax: plt.Axes, rect: tuple[float, float, float, float], *, fill: str, edge: str, active: float) -> None:
    x0, y0, x1, y1 = rect
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fill, ec=edge, lw=1.6, zorder=3))
    cols, rows = 5, 8
    for col in range(cols):
        for row in range(rows):
            weight = ((col * 7 + row * 3) % 11) / 10.0
            colour = LIGHT_GRAY if fill == WHITE else "#4F8D8D" if active > 0.5 and weight > 0.48 else "#B5D7D7" if active > 0.5 else "#C8D2DD"
            pad_x = (x1 - x0) * 0.06
            pad_y = (y1 - y0) * 0.035
            cell_w = (x1 - x0 - 2 * pad_x) / cols
            cell_h = (y1 - y0 - 2 * pad_y) / rows
            ax.add_patch(Rectangle((x0 + pad_x + col * cell_w, y0 + pad_y + row * cell_h), cell_w * 0.80, cell_h * 0.72, fc=colour, ec="none", alpha=0.35 + 0.65 * active, zorder=4))


def _network_arrow(ax: plt.Axes, registry: LayoutRegistry, start: tuple[float, float], end: tuple[float, float], *, colour: str, alpha: float = 1.0) -> None:
    registry.arrow(ax, start, end, arrowstyle="-|>", mutation_scale=17, lw=2.5, color=colour, alpha=alpha, zorder=5)


def _draw_network(ax: plt.Axes, registry: LayoutRegistry, demo: dict[str, object], time_seconds: float) -> None:
    _panel_background(ax)
    registry.text(ax, 0.03, 0.965, "Neural network", ha="left", va="top", fontsize=FONT_SIZES["panel_title"], color=INK)
    # The forward pass is staged explicitly: descriptor, G1, G2, fit, then
    # the output energy. This prevents the output value from appearing while
    # an earlier layer is still being evaluated.
    forward_phase = float(np.clip((time_seconds - 2.0) / 2.5, 0.0, 1.0))
    if time_seconds >= 13.5:
        forward_phase = 1.0
    back_t = np.clip((time_seconds - 8.5) / 3.0, 0.0, 1.0)
    backward_p = smoothstep(back_t)
    backprop_active = 8.5 <= time_seconds < 11.5
    update_t = smoothstep(np.clip((time_seconds - 11.5) / 2.0, 0.0, 1.0))
    update_on = time_seconds >= 11.5
    if time_seconds < 8.5:
        stage, stage_colour = "forward pass", NAVY
    elif time_seconds < 11.5:
        stage, stage_colour = "backpropagation", GRADIENT_ORANGE
    elif time_seconds < 13.5:
        stage, stage_colour = "update weights", GRADIENT_ORANGE
    else:
        stage, stage_colour = "forward again", ENERGY_TEAL
    registry.text(ax, 0.97, 0.965, stage, ha="right", va="top", fontsize=FONT_SIZES["body"], color=stage_colour)

    input_rect = (0.035, 0.420, 0.205, 0.755)
    hidden_rects = [(0.280, 0.385, 0.385, 0.795), (0.455, 0.385, 0.560, 0.795), (0.630, 0.385, 0.735, 0.795)]
    # Scalar energy output: a coloured node, with the numeric value labelled
    # below in black. Tensor layers keep the matrix/grid treatment.
    output_center = (0.860, 0.630)
    output_radius = 0.065
    layer_centres = [0.5 * (rect[0] + rect[2]) for rect in hidden_rects]
    # Keep the descriptor quiet during the sample/zoom phase.  It is the
    # network's first evaluated block, so the navy border should appear only
    # when the forward pass reaches the network at 2 s.
    input_active = 1.0 if time_seconds >= 2.0 else 0.0
    _heat_block(
        ax,
        input_rect,
        fill=INPUT_FILL if input_active > 0.2 else WHITE,
        edge=NAVY if time_seconds >= 2.0 else LINE_GRAY,
        active=input_active,
    )
    registry.text(ax, 0.120, 0.825, "descriptor", ha="center", va="bottom", fontsize=FONT_SIZES["micro"], color=INK)
    registry.text(ax, 0.120, 0.350, r"$\mathbf{R}_i$", ha="center", va="top", fontsize=FONT_SIZES["body"], color=INK)

    for index, rect in enumerate(hidden_rects):
        # G1, G2, and fit occupy successive windows of the forward pass.
        forward_arrival = float(np.clip((forward_phase - (0.10 + 0.22 * index)) / 0.20, 0.0, 1.0))
        reverse_offset = (2 - index) * 0.30
        reverse_arrival = float(np.clip((back_t * 1.30 - reverse_offset) / 0.36, 0.0, 1.0))
        active = max(forward_arrival, reverse_arrival if backprop_active else 0.0)
        if backprop_active and backward_p > 0.0:
            edge = GRADIENT_ORANGE if reverse_arrival > 0.02 else LINE_GRAY
            fill = LOSS_FILL if reverse_arrival > 0.25 else WHITE
        else:
            edge = ENERGY_TEAL if forward_arrival > 0.35 else LINE_GRAY
            fill = PREDICTION_FILL if forward_arrival > 0.35 else WHITE
        _heat_block(ax, rect, fill=fill, edge=edge, active=active)
        label = ("G¹", "G²", "fit")[index]
        registry.text(ax, layer_centres[index], 0.825, label, ha="center", va="bottom", fontsize=FONT_SIZES["micro"], color=INK)
        if not backprop_active:
            registry.text(ax, layer_centres[index], 0.350, label, ha="center", va="top", fontsize=FONT_SIZES["micro"], color=INK)

    output_active = float(np.clip((forward_phase - 0.76) / 0.20, 0.0, 1.0))
    if time_seconds >= 13.5:
        output_active = 1.0
    output_ready = output_active > 0.35
    if output_ready:
        aspect = _axes_aspect(ax)
        output_edge = GRADIENT_ORANGE if backprop_active and backward_p > 0.0 else ENERGY_TEAL
        output_fill = LOSS_FILL if backprop_active and backward_p > 0.0 else PREDICTION_FILL
        ax.add_patch(Ellipse(
            output_center,
            2.0 * output_radius,
            2.0 * output_radius * aspect,
            fc=output_fill,
            ec=output_edge,
            lw=1.8,
            zorder=3,
        ))
        # The scalar node stays a simple coloured circle.  Its symbol and
        # value are black labels below it, so the node is not confused with a
        # tensor layer or a card.
        output_x = output_center[0]
        registry.text(ax, output_x, 0.565, r"$\hat{\varepsilon}_{i}$", ha="center", va="top", fontsize=FONT_SIZES["body"], color=INK)
        value = float(demo["_arrays"]["updated"][0] if time_seconds >= 13.5 else demo["_arrays"]["initial"][0])
        registry.text(ax, output_x, 0.505, f"{_fmt_ev(value)} eV", ha="center", va="top", fontsize=FONT_SIZES["body"], color=INK)

    links = [
        ((input_rect[2], 0.600), (hidden_rects[0][0], 0.600)),
        ((hidden_rects[0][2], 0.600), (hidden_rects[1][0], 0.600)),
        ((hidden_rects[1][2], 0.600), (hidden_rects[2][0], 0.600)),
        ((hidden_rects[2][2], 0.600), (output_center[0] - output_radius, output_center[1])),
    ]
    for link_index, (start, end) in enumerate(links):
        if link_index == 3 and not output_ready:
            continue
        if backprop_active and backward_p > 0.0:
            _network_arrow(ax, registry, end, start, colour=GRADIENT_ORANGE, alpha=0.35 + 0.55 * backward_p)
        else:
            forward_colour = ENERGY_TEAL if time_seconds >= 13.5 else NAVY
            _network_arrow(ax, registry, start, end, colour=forward_colour, alpha=max(0.25, forward_phase))

    if 8.5 <= time_seconds < 11.5:
        pulse = np.clip(back_t * 4.0, 0.0, 3.999)
        link_index = min(int(pulse), 3)
        local_p = pulse - link_index
        start, end = links[3 - link_index]
        x = end[0] + local_p * (start[0] - end[0])
        y = end[1] + local_p * (start[1] - end[1])
        ax.plot([x], [y], marker="o", markersize=7.0, color=GRADIENT_ORANGE, zorder=8)

    card_y0, card_y1 = 0.015, 0.195
    card_w = 0.145
    old_values = (0.42, -0.18, 0.31)
    new_values = (0.47, -0.12, 0.28)
    if backprop_active or update_on:
        registry.text(ax, 0.035, 0.240, "parameters", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)
        for index, x in enumerate(layer_centres):
            reverse_offset = (2 - index) * 0.30
            arrival = float(np.clip((back_t * 1.30 - reverse_offset) / 0.36, 0.0, 1.0))
            edge = GRADIENT_ORANGE if not update_on else ENERGY_TEAL
            fill = LOSS_FILL if not update_on else UPDATED_FILL
            ax.add_patch(Rectangle((x - card_w / 2, card_y0), card_w, card_y1 - card_y0, fc=fill, ec=edge, lw=1.3, zorder=2))
            connector_alpha = max(0.20, arrival) if not update_on else 1.0
            ax.plot([x, x], [hidden_rects[index][1] - 0.01, card_y1], color=edge, lw=2.0, alpha=connector_alpha, zorder=1)
            registry.text(ax, x, 0.165, f"w{index + 1}", ha="center", va="center", fontsize=FONT_SIZES["micro"], color=INK)

            def swatch(y: float, value: float, colour: str, alpha: float) -> None:
                # Magnitude is carried by block length and shade; sign is
                # carried by the teal/navy hue, so numeric values are not
                # needed in every parameter card.
                magnitude = float(np.clip(abs(value) / 0.50, 0.0, 1.0))
                width = card_w * (0.22 + 0.58 * magnitude)
                ax.add_patch(Rectangle((x - width / 2, y), width, 0.027, fc=colour, ec="none", alpha=alpha, zorder=4))

            if not update_on:
                swatch(0.075, old_values[index], GRADIENT_ORANGE, 0.25 + 0.60 * arrival)
            else:
                current = old_values[index] + update_t * (new_values[index] - old_values[index])
                swatch(0.108, old_values[index], NAVY if old_values[index] < 0 else ENERGY_TEAL, 0.32)
                registry.arrow(ax, (x, 0.101), (x, 0.077), arrowstyle="-|>", mutation_scale=8, lw=1.1, color=GRADIENT_ORANGE, zorder=6)
                swatch(0.040, current, NAVY if current < 0 else ENERGY_TEAL, 0.35 + 0.55 * update_t)
        registry.text(ax, 0.965, 0.240, r"$w' = w - \eta\,\partial L/\partial w$", ha="right", va="center", fontsize=FONT_SIZES["micro"], color=INK)
        if backprop_active and backward_p > 0.12:
            registry.text(ax, 0.50, 0.240, r"$\partial L/\partial w$", ha="center", va="center", fontsize=FONT_SIZES["body"], color=INK)

def _metric_card(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    rect: tuple[float, float, float, float],
    colour: str,
    title: str,
    value: str,
    fill: str,
) -> None:
    x0, y0, x1, y1 = rect
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fill, ec=colour, lw=1.6, zorder=2))
    registry.text(ax, x0 + 0.025, y1 - 0.005, title, ha="left", va="top", fontsize=FONT_SIZES["micro"], color=INK)
    registry.text(ax, x0 + 0.025, y0 + 0.010, value, ha="left", va="bottom", fontsize=FONT_SIZES["body"], color=INK)


def _draw_metrics(ax: plt.Axes, registry: LayoutRegistry, demo: dict[str, object], time_seconds: float) -> None:
    _panel_background(ax)
    registry.text(ax, 0.03, 0.965, "Compare → correct", ha="left", va="top", fontsize=FONT_SIZES["panel_title"], color=INK)
    registry.text(ax, 0.97, 0.965, "O126", ha="right", va="top", fontsize=FONT_SIZES["micro"], color=INK)
    ref = float(demo["_arrays"]["reference"][0])
    initial = float(demo["_arrays"]["initial"][0])
    updated = float(demo["_arrays"]["updated"][0])
    delta = initial - ref
    updated_delta = updated - ref
    loss = float(demo["derived"]["focus_loss_ev2"])
    updated_loss = float(demo["derived"]["updated_focus_loss_ev2"])
    # Keep the compare panel quiet during sample/zoom and the forward pass.
    # Once the prediction is ready, use three stacked teaching elements only:
    # truth, prediction, and the resulting error.
    prediction_on = time_seconds >= 4.5
    final_on = time_seconds >= 13.5

    if prediction_on:
        _metric_card(
            ax, registry, rect=(0.05, 0.765, 0.95, 0.910), colour=ENERGY_TEAL,
            title="truth · DFT reference", value=rf"$\varepsilon_i^{{ref}} = {_fmt_ev(ref)}$ eV", fill=STRUCTURE_BG,
        )
        prediction_value = updated if final_on else initial
        prediction_label = "prediction · after update" if final_on else "prediction · network output"
        _metric_card(
            ax, registry, rect=(0.05, 0.535, 0.95, 0.705), colour=ENERGY_TEAL if final_on else NAVY,
            title=prediction_label, value=rf"$\hat{{\varepsilon}}_i = {_fmt_ev(prediction_value)}$ eV", fill=PREDICTION_FILL,
        )
        error_colour = ENERGY_TEAL if final_on else GRADIENT_ORANGE
        error_fill = UPDATED_FILL if final_on else LOSS_FILL
        error_value = abs(updated_delta) if final_on else abs(delta)
        loss_value = updated_loss if final_on else loss
        _metric_card(
            ax, registry, rect=(0.05, 0.275, 0.95, 0.455), colour=error_colour,
            title="error · prediction − truth",
            value=rf"$|\delta_i| = {error_value:.2f}$ eV   ·   $L_i = {loss_value:.3f}$ eV$^2$",
            fill=error_fill,
        )


def _draw_story_links(
    fig: plt.Figure,
    structure_ax: plt.Axes,
    network_ax: plt.Axes,
    metrics_ax: plt.Axes,
    time_seconds: float,
) -> None:
    """Connect the physical centre atom, network entrance, output, and loss."""
    # The descriptor is computed from the central O126 environment.  Start
    # the cross-panel relation at the atom in the magnified view, rather than
    # at the structure panel edge, so the source of R_i is unambiguous.
    input_reveal = smoothstep(np.clip((time_seconds - 1.15) / 0.55, 0.0, 1.0))
    if input_reveal > 0.0:
        input_link = ConnectionPatch(
            xyA=(0.675, 0.57), coordsA="data", axesA=structure_ax,
            xyB=(0.035, 0.60), coordsB="data", axesB=network_ax,
            arrowstyle="-|>", mutation_scale=13, lw=1.6,
            color=NAVY, alpha=0.25 + 0.65 * input_reveal, zorder=12,
            connectionstyle="arc3,rad=-0.16",
        )
        input_link.set_clip_on(False)
        fig.add_artist(input_link)

    if 6.5 <= time_seconds < 8.5:
        output_link = ConnectionPatch(
            xyA=(0.925, 0.63), coordsA="data", axesA=network_ax,
            xyB=(0.05, 0.62), coordsB="data", axesB=metrics_ax,
            arrowstyle="-|>", mutation_scale=13, lw=1.5,
            color=NAVY, alpha=0.75, zorder=12,
        )
        output_link.set_clip_on(False)
        fig.add_artist(output_link)
    elif 8.5 <= time_seconds < 11.5:
        # Once the loss is formed, the error travels back to the scalar
        # output first.  The network then continues the orange pulse through
        # fit, G², G¹, and the descriptor in _draw_network.
        error_link = ConnectionPatch(
            xyA=(0.05, 0.36), coordsA="data", axesA=metrics_ax,
            xyB=(0.795, 0.63), coordsB="data", axesB=network_ax,
            arrowstyle="-|>", mutation_scale=14, lw=2.0,
            color=GRADIENT_ORANGE, alpha=0.88, zorder=12,
            connectionstyle="arc3,rad=0.10",
        )
        error_link.set_clip_on(False)
        fig.add_artist(error_link)

def _stage_label(time_seconds: float) -> str:
    if time_seconds < 2.0:
        return "sample_zoom"
    if time_seconds < 4.5:
        return "forward"
    if time_seconds < 6.5:
        return "compare"
    if time_seconds < 8.5:
        return "loss"
    if time_seconds < 11.5:
        return "backprop"
    if time_seconds < 13.5:
        return "update"
    return "forward_again"


def draw_frame(fig: plt.Figure, time_seconds: float, registry: LayoutRegistry, demo: dict[str, object], scene: dict[str, object]) -> list[dict]:
    structure_ax = _slot_axes(fig, PANEL_STRUCTURE)
    network_ax = _slot_axes(fig, PANEL_NETWORK)
    metrics_ax = _slot_axes(fig, PANEL_METRICS)
    _draw_structure(structure_ax, registry, demo, scene, time_seconds)
    _draw_network(network_ax, registry, demo, time_seconds)
    _draw_metrics(metrics_ax, registry, demo, time_seconds)
    _draw_story_links(fig, structure_ax, network_ax, metrics_ax, time_seconds)
    stage = _stage_label(time_seconds)
    semantics = [{"id": "structure", "color": NAVY, "min_pixels": 20, "tolerance": 70}, {"id": "reference", "color": ENERGY_TEAL, "min_pixels": 20, "tolerance": 70}]
    if stage in {"backprop", "update"}:
        semantics.append({"id": "gradient", "color": GRADIENT_ORANGE, "min_pixels": 20, "tolerance": 70})
    return semantics


def _audit_config() -> dict[str, object]:
    return {
        "panels": [
            {"id": "structure", "rect": list(PANEL_STRUCTURE), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            {"id": "network", "rect": list(PANEL_NETWORK), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            {"id": "metrics", "rect": list(PANEL_METRICS), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.012, "min_panel_bbox_fill": 0.045, "grid_rows": 12, "grid_columns": 20},
        "bands": [
            {"id": "gap_structure_network", "rect": [0.335, 0.035, 0.350, 0.955], "max_ink_pixels": 5000},
            {"id": "gap_network_metrics", "rect": [0.665, 0.035, 0.680, 0.955], "max_ink_pixels": 5000},
        ],
    }


REPRESENTATIVE_TIMES = (0.0, 1.0, 2.0, 4.5, 7.0, 9.0, 10.5, 12.0, 14.0, 15.5)


def _prepare_scene() -> dict[str, object]:
    data = load_data("deepmd")
    # The 04 story already contains the canonical MatterVis render of state 0.
    # Reuse those immutable assets so a training-video preview does not spend
    # several minutes regenerating all six force/velocity/move variants.
    existing = ROOT / "qa" / "04_deep_potential_md" / "source" / "mattervis_v1"
    box = existing / "box_00.png"
    cut = existing / "cut_00.png"
    if box.exists() and cut.exists():
        asset_manifest = ROOT / "qa" / "04_deep_potential_md" / "asset_manifest.json"
        manifest = json.loads(asset_manifest.read_text(encoding="utf-8"))
        camera_data = manifest["camera"]
        box_source = Path(manifest["box_source"])
        focus_source = Path(manifest["focus_source"])
        scene = {
            "assets": {"box": [box], "cut": [cut]},
            "asset_root": existing,
            "box_camera": camera_for_source(
                box_source, target=np.full(3, 0.5 * float(data["box_length"])),
                ortho_scale=float(camera_data["box_ortho_scale"]),
                direction=camera_data["direction"], up=camera_data["up"],
            ),
            "focus_camera": camera_for_source(
                focus_source, target=(0.0, 0.0, 0.0),
                ortho_scale=float(camera_data["focus_ortho_scale"]),
                direction=camera_data["direction"], up=camera_data["up"],
            ),
        }
    else:
        scene = prepare_assets("deepmd", data, QA_ROOT)
    scene["data"] = data
    return scene


def render_preview(demo: dict[str, object], scene: dict[str, object]) -> Path:
    output_dir = QA_ROOT / "_qa" / "keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for index, time_seconds in enumerate(REPRESENTATIVE_TIMES):
        fig = new_video_figure()
        registry = LayoutRegistry(min_font_pt=FONT_SIZES["micro"], max_font_pt=FONT_SIZES["page_title"], edge_pad_px=12, font_family="Arial", coerce_min_font=True)
        draw_frame(fig, time_seconds, registry, demo, scene)
        errors = registry.validate(fig)
        path = output_dir / f"frame_{index:02d}_{time_seconds:04.1f}s.png"
        fig.savefig(path, dpi=100, facecolor=WHITE)
        plt.close(fig)
        records.append({"time_seconds": time_seconds, "path": str(path), "stage": _stage_label(time_seconds)})
        if errors:
            failures.append({"time_seconds": time_seconds, "errors": errors})
    json_dump(output_dir / "keyframes.json", {"frames": records, "failures": failures})
    if failures:
        raise RuntimeError(f"04a preview layout failed: {failures}")
    return output_dir / "keyframes.json"


def render_animation(demo: dict[str, object], scene: dict[str, object]) -> Path:
    return render_video(
        stem=STEM,
        duration_seconds=VIDEO_DURATION,
        draw_frame=lambda fig, time_seconds, _frame_index, registry: draw_frame(fig, time_seconds, registry, demo, scene),
        audit_config=_audit_config(),
        qa_directory=QA_ROOT / "_qa",
        representative_times=REPRESENTATIVE_TIMES,
        image_sequence=True,
    )


def write_provenance(demo: dict[str, object], scene: dict[str, object]) -> None:
    data = scene["data"]
    json_dump(
        QA_ROOT / "story_provenance.json",
        {
            "schema": "nnmd_training_story/v1",
            "stem": STEM,
            "manifest": str(MANIFEST),
            "structure_source": str(DATA_DIR / "water_box_64.npz"),
            "trajectory_source": str(DATA_DIR / "dpmd_water_box_trajectory.npz"),
            "structure_status": demo["structure_status"],
            "label_kind": demo["label_kind"],
            "label_note": demo["label_note"],
            "focus_atom": demo["focus_atom"],
            "centre_atom_index": int(data["central_index"]),
            "box_length_angstrom": float(data["box_length"]),
            "cutoff_angstrom": float(data["cutoff"]),
            "derived": demo["derived"],
            "timeline_seconds": demo["timeline_seconds"],
            "video": {"duration_seconds": VIDEO_DURATION, "canvas": [1920, 600], "fps": 24, "aspect_ratio": "16:5"},
            "mattervis_assets": str(scene.get("asset_root", QA_ROOT / "asset_manifest.json")),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    demo = _load_demo()
    scene = _prepare_scene()
    write_provenance(demo, scene)
    if args.assets_only:
        print(QA_ROOT)
    elif args.preview_only:
        print(render_preview(demo, scene))
    else:
        print(render_animation(demo, scene))


if __name__ == "__main__":
    main()
