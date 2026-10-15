"""04a — one transparent neural-network training step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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
GRID_CMAP = LinearSegmentedColormap.from_list(
    "training_grid",
    ["#E1EBF8", "#C7EAFB", "#9ED4D0", "#1F7A7A", "#183153"],
)


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
        ax, 0.03, 0.965, "Training dataset",
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
        ax, 0.675, 0.245, "local environment",
        ha="center", va="top", fontsize=FONT_SIZES["micro"], color=INK,
    )

def _heat_block(
    ax: plt.Axes,
    rect: tuple[float, float, float, float],
    *,
    fill: str,
    edge: str,
    active: float,
    value_phase: float = 0.0,
    mode: str = "activation",
) -> None:
    x0, y0, x1, y1 = rect
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fill, ec=edge, lw=1.6, zorder=3))
    cols, rows = 5, 8
    for col in range(cols):
        for row in range(rows):
            # Deterministic mock values make the grid a readable teaching
            # fixture.  The same -1 to +1 colour mapping is used for forward
            # activations, gradients, and the update transition.
            base = 2.0 * (((col * 7 + row * 3) % 11) / 10.0) - 1.0
            if mode == "gradient":
                value = base + 0.85 * (float(value_phase) - 0.5)
            elif mode == "update":
                value = base + 0.65 * (float(value_phase) - 0.5)
            else:
                value = base + 0.25 * float(value_phase)
            colour = GRID_CMAP(float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))) if active > 0.05 else LIGHT_GRAY
            pad_x = (x1 - x0) * 0.06
            pad_y = (y1 - y0) * 0.035
            cell_w = (x1 - x0 - 2 * pad_x) / cols
            cell_h = (y1 - y0 - 2 * pad_y) / rows
            ax.add_patch(Rectangle((x0 + pad_x + col * cell_w, y0 + pad_y + row * cell_h), cell_w * 0.80, cell_h * 0.72, fc=colour, ec="none", alpha=0.22 + 0.78 * active, zorder=4))


def _draw_grid_colorbar(ax: plt.Axes, registry: LayoutRegistry, *, label: str) -> None:
    x0, x1 = 0.64, 0.96
    y0, y1 = 0.105, 0.145
    steps = 24
    for index in range(steps):
        left = x0 + (x1 - x0) * index / steps
        right = x0 + (x1 - x0) * (index + 1) / steps
        ax.add_patch(Rectangle((left, y0), right - left, y1 - y0, fc=GRID_CMAP(index / (steps - 1)), ec="none", zorder=3))
    registry.text(ax, x0, 0.070, "−1", ha="left", va="top", fontsize=FONT_SIZES["micro"], color=INK)
    registry.text(ax, x1, 0.070, "+1", ha="right", va="top", fontsize=FONT_SIZES["micro"], color=INK)
    registry.text(ax, 0.80, 0.180, label, ha="center", va="bottom", fontsize=FONT_SIZES["micro"], color=INK)


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
        value_phase=input_active,
        mode="activation",
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
        if backprop_active:
            grid_mode = "gradient"
            grid_phase = reverse_arrival
        elif update_on and time_seconds < 13.5:
            grid_mode = "update"
            grid_phase = update_t
        else:
            grid_mode = "activation"
            grid_phase = forward_arrival
        _heat_block(ax, rect, fill=fill, edge=edge, active=active, value_phase=grid_phase, mode=grid_mode)
        label = ("G¹", "G²", "fit")[index]
        registry.text(ax, layer_centres[index], 0.825, label, ha="center", va="bottom", fontsize=FONT_SIZES["micro"], color=INK)
        if not backprop_active:
            registry.text(ax, layer_centres[index], 0.350, label, ha="center", va="top", fontsize=FONT_SIZES["micro"], color=INK)

    links = [
        ((input_rect[2], 0.600), (hidden_rects[0][0], 0.600)),
        ((hidden_rects[0][2], 0.600), (hidden_rects[1][0], 0.600)),
        ((hidden_rects[1][2], 0.600), (hidden_rects[2][0], 0.600)),
    ]
    for start, end in links:
        if backprop_active and backward_p > 0.0:
            _network_arrow(ax, registry, end, start, colour=GRADIENT_ORANGE, alpha=0.35 + 0.55 * backward_p)
        else:
            forward_colour = ENERGY_TEAL if time_seconds >= 13.5 else NAVY
            _network_arrow(ax, registry, start, end, colour=forward_colour, alpha=max(0.25, forward_phase))

    if 8.5 <= time_seconds < 11.5:
        pulse = np.clip(back_t * 3.0, 0.0, 2.999)
        link_index = min(int(pulse), 2)
        local_p = pulse - link_index
        start, end = links[2 - link_index]
        x = end[0] + local_p * (start[0] - end[0])
        y = end[1] + local_p * (start[1] - end[1])
        ax.plot([x], [y], marker="o", markersize=7.0, color=GRADIENT_ORANGE, zorder=8)

    if backprop_active:
        colourbar_label = "gradient value (mock)"
    elif update_on and time_seconds < 13.5:
        colourbar_label = "updated activation (mock)"
    else:
        colourbar_label = "activation value (mock)"
    _draw_grid_colorbar(ax, registry, label=colourbar_label)

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


def _prediction_card(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    value: float,
    final_on: bool,
) -> None:
    """Merge the scalar output node and prediction panel into one element."""
    # Let the merged prediction element extend a little into the inter-panel
    # gap so the fit block and output node do not leave a large white void.
    rect = (-0.20, 0.515, 0.95, 0.720)
    edge = ENERGY_TEAL if final_on else NAVY
    card = Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], fc=PREDICTION_FILL, ec=edge, lw=1.6, zorder=2)
    card.set_clip_on(False)
    ax.add_patch(card)
    aspect = _axes_aspect(ax)
    circle_center = (-0.12, 0.635)
    radius = 0.045
    circle = Ellipse(
        circle_center,
        2.0 * radius,
        2.0 * radius * aspect,
        fc=PREDICTION_FILL,
        ec=edge,
        lw=1.8,
        zorder=4,
    )
    circle.set_clip_on(False)
    ax.add_patch(circle)
    # Keep the requested black label below the circle while the numeric value
    # sits in the same card, so the network output is no longer duplicated.
    registry.text(ax, circle_center[0], 0.575, r"$\hat{\varepsilon}_{i}$", ha="center", va="top", fontsize=FONT_SIZES["body"], color=INK)
    title = "prediction · after update" if final_on else "prediction · network output"
    registry.text(ax, 0.08, 0.685, title, ha="left", va="top", fontsize=FONT_SIZES["micro"], color=INK)
    registry.text(ax, 0.08, 0.575, f"{_fmt_ev(value)} eV", ha="left", va="top", fontsize=FONT_SIZES["body"], color=INK)


def _draw_metrics(ax: plt.Axes, registry: LayoutRegistry, demo: dict[str, object], time_seconds: float) -> None:
    _panel_background(ax)
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
        _prediction_card(ax, registry, value=prediction_value, final_on=final_on)
        registry.arrow(
            # Anchor the double arrow on the horizontal centre lines of the
            # truth and prediction bands, so the relation remains long and
            # legible even when the inter-band gap is narrow.
            ax, (0.90, 0.8375), (0.90, 0.6175),
            arrowstyle="<->", mutation_scale=16, lw=2.0,
            color=GRADIENT_ORANGE, zorder=6,
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
            color=NAVY, alpha=1.0, zorder=12,
            connectionstyle="arc3,rad=-0.16",
        )
        input_link.set_clip_on(False)
        fig.add_artist(input_link)

    # The dataset relation is structural, so draw it from the first frame as
    # a solid inverted-Π connector: local atom up, across the headers, then
    # down into the DFT truth element.
    dft_segments = (
        ConnectionPatch(
            xyA=(0.675, 0.57), coordsA="data", axesA=structure_ax,
            xyB=(0.675, 0.985), coordsB="data", axesB=structure_ax,
            arrowstyle="-", lw=1.3, color=ENERGY_TEAL, alpha=1.0, zorder=10,
        ),
        ConnectionPatch(
            xyA=(0.675, 0.985), coordsA="data", axesA=structure_ax,
            xyB=(0.50, 0.985), coordsB="data", axesB=metrics_ax,
            arrowstyle="-", lw=1.3, color=ENERGY_TEAL, alpha=1.0, zorder=10,
        ),
        ConnectionPatch(
            xyA=(0.50, 0.985), coordsA="data", axesA=metrics_ax,
            xyB=(0.50, 0.885), coordsB="data", axesB=metrics_ax,
            arrowstyle="-|>", mutation_scale=11, lw=1.3, color=ENERGY_TEAL, alpha=1.0, zorder=10,
        ),
    )
    for dft_link in dft_segments:
        dft_link.set_clip_on(False)
        fig.add_artist(dft_link)

    if (4.5 <= time_seconds < 8.5) or time_seconds >= 13.5:
        output_link = ConnectionPatch(
            xyA=(0.735, 0.60), coordsA="data", axesA=network_ax,
            xyB=(-0.165, 0.635), coordsB="data", axesB=metrics_ax,
            arrowstyle="-|>", mutation_scale=13, lw=1.5,
            color=ENERGY_TEAL if time_seconds >= 13.5 else NAVY, alpha=0.85, zorder=12,
        )
        output_link.set_clip_on(False)
        fig.add_artist(output_link)
    elif 8.5 <= time_seconds < 11.5:
        # The error first returns to the merged prediction/output node, then
        # crosses back to fit and continues through G², G¹, and the descriptor.
        metrics_ax.annotate(
            "", xy=(-0.12, 0.585), xytext=(-0.12, 0.46),
            arrowprops={"arrowstyle": "-|>", "mutation_scale": 12, "lw": 1.8, "color": GRADIENT_ORANGE},
            zorder=12,
        )
        error_link = ConnectionPatch(
            xyA=(-0.165, 0.635), coordsA="data", axesA=metrics_ax,
            xyB=(0.735, 0.60), coordsB="data", axesB=network_ax,
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
