from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    CRIMSON,
    DARK_GRAY,
    GREEN,
    INK,
    LIGHT_GRAY,
    LINE_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    add_footer,
    add_page_title,
    axes_from_top_slot,
    draw_ball_and_stick,
    new_static_figure,
    project_points,
    map_projected_to_rect,
    render_source_panel,
    render_video,
    save_static,
    smoothstep,
)


ROOT = Path(__file__).resolve().parent
STEM = "02_classical_lj"
QA_DIR = ROOT / "_qa" / "02_classical_lj"
STATIC_LEFT = (0.045, 0.20, 0.53, 0.86)
STATIC_CURVE_INNER = (0.065, 0.23, 0.51, 0.83)
STATIC_RIGHT = (0.57, 0.18, 0.965, 0.86)
VIDEO_LEFT = (0.045, 0.20, 0.55, 0.86)
VIDEO_CURVE_INNER = (0.068, 0.23, 0.53, 0.83)
VIDEO_RIGHT = (0.60, 0.18, 0.965, 0.87)


def load_data() -> dict[str, np.ndarray]:
    source = np.load(ROOT / "data" / "classical_lj.npz")
    return {key: source[key] for key in source.files}


def potential(r: np.ndarray | float, sigma: float, epsilon: float):
    ratio = sigma / np.asarray(r)
    return 4.0 * epsilon * (ratio**12 - ratio**6)


def force(r: np.ndarray | float, sigma: float, epsilon: float):
    ratio = sigma / np.asarray(r)
    return 24.0 * epsilon / np.asarray(r) * (2.0 * ratio**12 - ratio**6)


def _style_curve_axis(ax, *, video: bool) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(DARK_GRAY)
    ax.spines[["left", "bottom"]].set_linewidth(1.8 if video else 1.2)
    ax.tick_params(axis="both", labelsize=18 if video else 10, colors=DARK_GRAY, length=5, width=1.2)
    ax.set_xlabel(r"separation  $r$  (Å)", fontsize=20 if video else 11, color=DARK_GRAY, labelpad=8)
    ax.set_ylabel(r"pair energy  $U$  (eV)", fontsize=20 if video else 11, color=DARK_GRAY, labelpad=8)
    ax.grid(False)


def _draw_curve(ax, registry: LayoutRegistry, data: dict[str, np.ndarray], *, video: bool, current_r: float | None = None) -> None:
    sigma = float(data["sigma_angstrom"])
    epsilon = float(data["epsilon_ev"])
    r_min = float(data["r_min_angstrom"])
    r = data["r_curve"]
    u = data["u_curve"]
    ax.set_xlim(float(r.min()), float(r.max()))
    ax.set_ylim(-0.0125, 0.0145)
    _style_curve_axis(ax, video=video)
    left = r <= r_min
    right = r >= r_min
    ax.plot(r[left], u[left], color=CRIMSON, lw=4.5 if video else 2.8)
    ax.plot(r[right], u[right], color=GREEN, lw=4.5 if video else 2.8)
    ax.axhline(0.0, color=LINE_GRAY, lw=1.5)
    ax.axvline(r_min, color=LINE_GRAY, lw=1.6, ls="--")
    ax.scatter([r_min], [-epsilon], s=170 if video else 90, c=NAVY, edgecolors=WHITE, linewidths=1.5, zorder=6)
    registry.text(ax, r_min, -0.0118, r"$r_m=2^{1/6}\sigma$", ha="center", va="top", fontsize=18 if video else 10, color=NAVY)
    registry.text(ax, 0.90 * r_min, 0.0118, "repulsive", ha="center", va="center", fontsize=19 if video else 10, color=CRIMSON, weight="bold")
    registry.text(ax, 1.38 * r_min, -0.0067, "attractive", ha="center", va="center", fontsize=19 if video else 10, color=GREEN, weight="bold")
    if current_r is not None:
        current_u = float(potential(current_r, sigma, epsilon))
        ax.scatter([current_r], [current_u], s=300 if video else 130, c=INK, edgecolors=WHITE, linewidths=2.0, zorder=8)


def _draw_pair(
    ax,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    *,
    r: float,
    rect: tuple[float, float, float, float],
    video: bool,
    label: str,
    colour: str,
) -> None:
    axis = data["pair_axis"]
    positions = np.vstack((-0.5 * r * axis, 0.5 * r * axis))
    centre = np.zeros(3)
    half_span = 3.10
    xy, _ = draw_ball_and_stick(ax, positions, ["Ar", "Ar"], np.empty((0, 2), dtype=int), rect=rect, centre_3d=centre, half_span=half_span, atom_scale=1.35 if video else 0.58)
    scalar_force = float(force(r, float(data["sigma_angstrom"]), float(data["epsilon_ev"])))
    projected_axis, _ = project_points(np.vstack((np.zeros(3), axis)))
    centre_projected, _ = project_points(np.zeros((1, 3)))
    screen = map_projected_to_rect(projected_axis, rect, centre_projected[0], half_span)
    direction_2d = screen[1] - screen[0]
    direction_2d /= np.linalg.norm(direction_2d)
    force_length = 0.115 if video else 0.075
    if abs(scalar_force) > 1.0e-8:
        sign = np.sign(scalar_force)
        for atom, atom_sign in [(0, -1.0), (1, 1.0)]:
            start = xy[atom]
            end = start + atom_sign * sign * force_length * direction_2d
            registry.arrow(ax, tuple(start), tuple(end), arrowstyle="-|>", mutation_scale=30 if video else 16, lw=5.0 if video else 3.0, color=colour, zorder=15)
        force_text = f"|F| = {abs(scalar_force):.4f} eV/Å"
    else:
        force_text = "F = 0"
    x0, y0, x1, y1 = rect
    registry.text(ax, (x0 + x1) / 2, y1 + (0.045 if video else 0.025), label, ha="center", va="bottom", fontsize=24 if video else 12, color=colour, weight="bold")
    registry.text(ax, (x0 + x1) / 2, y0 - (0.045 if video else 0.022), f"r = {r:.3f} Å · {force_text}", ha="center", va="top", fontsize=18 if video else 10, color=DARK_GRAY)


def _draw_curve_source(ax, registry: LayoutRegistry, data: dict[str, np.ndarray]) -> None:
    _draw_curve(ax, registry, data, video=False, current_r=float(data["r_min_angstrom"]))


def _draw_pair_source(ax, registry: LayoutRegistry, data: dict[str, np.ndarray]) -> None:
    states = data["state_r"]
    rows = [
        ((0.08, 0.72, 0.92, 0.90), "repulsive", CRIMSON),
        ((0.08, 0.40, 0.92, 0.58), "equilibrium", NAVY),
        ((0.08, 0.08, 0.92, 0.26), "attractive", GREEN),
    ]
    for r, (rect, label, colour) in zip(states, rows):
        _draw_pair(ax, registry, data, r=float(r), rect=rect, video=False, label=label, colour=colour)


def render_static(data: dict[str, np.ndarray]) -> None:
    render_source_panel(QA_DIR / "source" / "lj_curve.png", lambda ax, reg: _draw_curve_source(ax, reg, data), width_px=1500, height_px=1350)
    render_source_panel(QA_DIR / "source" / "argon_pair.png", lambda ax, reg: _draw_pair_source(ax, reg, data), width_px=1250, height_px=1450)
    fig = new_static_figure()
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=18)
    add_page_title(fig, "02", "Classical potential", "Lennard-Jones turns geometry into energy and force without an electronic loop", video=False, registry=registry)
    left = axes_from_top_slot(fig, STATIC_CURVE_INNER)
    right = axes_from_top_slot(fig, STATIC_RIGHT)
    _draw_curve_source(left, registry, data)
    _draw_pair_source(right, registry, data)
    add_footer(fig, r"Ar–Ar · $\sigma=3.405$ Å · $\varepsilon=0.0103$ eV · $\mathbf{F}=-\nabla U$", video=False, registry=registry)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Static layout failed:\n" + "\n".join(errors))
    png, svg = save_static(fig, STEM)
    print(f"figure: {png}")
    print(f"vector: {svg}")


def _state_r(time_seconds: float, states: np.ndarray) -> tuple[int, float]:
    stage = min(int(time_seconds // 3.0), 2)
    local = (time_seconds - stage * 3.0) / 3.0
    target = float(states[stage])
    previous = float(states[stage - 1]) if stage > 0 else target
    interpolation = smoothstep(min(local / 0.35, 1.0))
    return stage, previous + interpolation * (target - previous)


def _draw_video_frame(fig, time_seconds: float, frame_index: int, registry: LayoutRegistry, data: dict[str, np.ndarray]) -> list[dict]:
    stage, r = _state_r(time_seconds, data["state_r"])
    add_page_title(fig, "02", "Classical potential: Lennard-Jones", "one analytic function supplies both energy and force", video=True, registry=registry)
    left = axes_from_top_slot(fig, VIDEO_CURVE_INNER)
    right = axes_from_top_slot(fig, VIDEO_RIGHT)
    _draw_curve(left, registry, data, video=True, current_r=r)
    labels = ["repulsive", "equilibrium", "attractive"]
    colours = [CRIMSON, NAVY, GREEN]
    _draw_pair(right, registry, data, r=r, rect=(0.02, 0.24, 0.98, 0.75), video=True, label=labels[stage], colour=colours[stage])
    registry.text(right, 0.50, 0.94, r"$U(r)=4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right]$", ha="center", va="top", fontsize=24, color=INK)
    registry.text(right, 0.50, 0.10, r"$\mathbf{F}_{ij}=-\frac{\mathrm{d}U}{\mathrm{d}r}\,\hat{\mathbf{r}}_{ij}$", ha="center", va="center", fontsize=22, color=INK)
    add_footer(fig, r"same $r$ drives the curve point, 3D separation and force arrows · fixed axes and camera", video=True, registry=registry)
    return [
        {"id": "potential", "color": NAVY, "min_pixels": 120},
        {"id": "repulsive", "color": CRIMSON, "min_pixels": 140},
        {"id": "attractive", "color": GREEN, "min_pixels": 140},
    ]


def render_animation(data: dict[str, np.ndarray]) -> None:
    audit_config = {
        "panels": [
            {"id": "lj_curve", "rect": list(VIDEO_LEFT), "min_clearance_px": 18},
            {"id": "argon_pair", "rect": list(VIDEO_RIGHT), "min_clearance_px": 18},
        ],
        "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.02, "min_panel_bbox_fill": 0.40, "grid_rows": 12, "grid_columns": 20},
        "bands": [{"id": "column_gap", "rect": [0.565, 0.18, 0.585, 0.88], "max_ink_pixels": 0}],
    }
    output = render_video(
        stem=STEM,
        duration_seconds=9.0,
        draw_frame=lambda fig, t, i, reg: _draw_video_frame(fig, t, i, reg, data),
        audit_config=audit_config,
        qa_directory=QA_DIR / "_qa",
        representative_times=[1.5, 4.5, 7.5],
    )
    print(f"video: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    data = load_data()
    render_static(data)
    if not args.static_only:
        render_animation(data)


if __name__ == "__main__":
    main()
