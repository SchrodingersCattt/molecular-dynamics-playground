"""Shared end-to-end MD teaching composition for DeepMD and DPA4C.

Both models use the same outer Velocity--Verlet loop.  The lower-right panel
changes the force provider, while the centre scene, energy output and feedback
arrow keep the reading order identical to the AIMD plate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import (
    DARK_GRAY,
    GREEN,
    INK,
    LINE_GRAY,
    NAVY,
    WHITE,
    LayoutRegistry,
    axes_from_top_slot,
    json_dump,
    new_static_figure,
    render_video,
    save_static,
)
from mattervis_story import (
    STORY_STATIC_A,
    STORY_STATIC_B,
    STORY_STATIC_C,
    STORY_STATIC_D,
    STORY_VIDEO_A,
    STORY_VIDEO_B,
    STORY_VIDEO_C,
    STORY_VIDEO_D,
    draw_vv_loop,
)
from responsive_story import EMERALD, LAKE_BLUE, PALE_OLIVE, panel_box, place_main


ROOT = Path(__file__).resolve().parents[2] / "product"
TRAJECTORY = ROOT / "data" / "dpmd_water_box_trajectory.npz"
ASSET_DIR = ROOT / "qa" / "04_dpmd_native" / "mattervis_trajectory_v1"
QA_DIR = ROOT / "qa" / "end_to_end_md"

MODELS = {
    "deepmd": {
        "stem": "04_deep_potential_md",
        "title": "DEEP POTENTIAL MD",
        "subtitle": "local environment → learned energy → force feedback",
        "chain": (
            "local neighbours",
            r"descriptor $D_i$",
            "shared atomic NN",
            r"atomic energies $\varepsilon_i$",
            "sum atomic energies",
            "differentiate energy",
            "acceleration",
            "Velocity–Verlet update",
        ),
        "chain_symbols": (
            r"$\{s_j,\mathbf{r}_{ij}\}_{r_{ij}<r_c}$",
            r"$D_i$",
            r"$NN_\theta$",
            r"$\varepsilon_1,\ldots,\varepsilon_N$",
            r"$E$",
            r"$-\partial E/\partial r_k$",
            r"$a_k$",
            r"$\mathbf{r}',\mathbf{v}'$",
        ),
    },
    "dpa4c": {
        "stem": "04_4c_dpa4c",
        "title": "DPA4C MD",
        "subtitle": "equivariant local operators → energy → force feedback",
        "chain": (
            r"neighbour vectors $\mathbf{r}_{ij}$",
            "radial + angular features",
            r"equivariant blocks $\ell=0,1,2$",
            "rotational invariants",
            r"descriptor $D_i$",
            r"energy readout $\varepsilon_i$",
            "energy → force",
            "Velocity–Verlet update",
        ),
        "chain_symbols": (
            r"$r_{ij},\hat{\mathbf{r}}_{ij}$",
            r"$f_n(r),B_{\ell m}(\hat r)$",
            r"$X_{\ell mc}$",
            r"$G,J,\Pi$",
            r"$D_i$",
            r"$\varepsilon_i$",
            r"$\mathbf{F}_k=-\partial E/\partial\mathbf{r}_k$",
            r"$\mathbf{r}',\mathbf{v}'$",
        ),
    },
}


def _load_data() -> dict[str, np.ndarray]:
    if not TRAJECTORY.exists():
        raise FileNotFoundError(f"Missing shared MD trajectory: {TRAJECTORY}")
    with np.load(TRAJECTORY, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    required = ("positions", "velocities", "forces_ev_per_angstrom", "atomic_energy_ev", "total_energy_ev")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Trajectory is missing {missing}")
    if not np.all(np.isfinite(data["total_energy_ev"])):
        raise ValueError("Trajectory contains non-finite total energies")
    if not np.all(np.isfinite(data["forces_ev_per_angstrom"])):
        raise ValueError("Trajectory contains non-finite forces")
    if not np.allclose(
        data["atomic_energy_ev"].sum(axis=1), data["total_energy_ev"], atol=1.0e-7
    ):
        raise ValueError("Atomic energies do not sum to total energy")
    return data


def _assets() -> dict[str, list[Path]]:
    def paths(prefix: str, count: int = 6) -> list[Path]:
        return [ASSET_DIR / f"{prefix}_{index:02d}.png" for index in range(count)]

    return {
        "box": paths("box"),
        "focus": paths("focus"),
        "force": paths("focus_force"),
        "velocity": paths("focus_velocity", 5),
        "move": paths("focus_move", 5),
    }


def _font(video: bool, static: float = 10.0, video_size: float = 16.0) -> float:
    return video_size if video else static


def _state_for_time(time_seconds: float, n_states: int) -> tuple[int, str, float]:
    """Return (trajectory state, semantic phase, phase progress)."""
    t = float(max(time_seconds, 0.0))
    detailed = ("input", "neighbours", "descriptor", "energy", "gradient", "force", "update", "return")
    if t < 10.0:
        cursor = t / 1.25
        index = min(int(cursor), len(detailed) - 1)
        return min(int(t / 2.0), n_states - 1), detailed[index], cursor - index
    fast = (t - 10.0) % 4.0
    phases = ("neighbours", "descriptor", "energy", "gradient", "update", "return")
    cursor = fast / (4.0 / len(phases))
    index = min(int(cursor), len(phases) - 1)
    state = min(1 + int((t - 10.0) // 4.0), n_states - 1)
    return state, phases[index], cursor - index


def _loop(ax: plt.Axes, registry: LayoutRegistry, *, video: bool, phase: str) -> None:
    panel_box(ax, registry, "ONE MD STEP", video=video)
    active = 2 if phase in {"update", "return"} else 1 if phase in {"gradient", "force", "energy"} else 0
    equation = {
        "input": r"$\mathbf{r}_n$" + "\ninput geometry",
        "neighbours": r"$\mathbf{r}_n$" + "\nselect $N_i(r_c)$",
        "descriptor": r"$D_i(\mathbf{r}_n)$" + "\nlocal features",
        "energy": r"$E(\mathbf{r}_n)$" + "\nmodel output",
        "gradient": r"$\mathbf{F}=-\nabla E$" + "\nbackpropagate",
        "force": r"$\mathbf{a}=\mathbf{F}/m$" + "\nacceleration",
        "update": r"$\mathbf{r}',\mathbf{v}'$" + "\nVV update",
        "return": r"$n\rightarrow n+1$" + "\nrepeat",
    }[phase]
    draw_vv_loop(
        ax,
        registry,
        video=video,
        active_stage=active,
        centre_text=equation,
        centre_y=0.54,
        radius_x=0.39,
    )
    registry.text(
        ax,
        0.50,
        0.08,
        r"$\mathbf{F}=-\nabla_{\mathbf{R}}E\;\rightarrow\;\mathbf{a}=\mathbf{F}/m$",
        ha="center",
        va="center",
        fontsize=_font(video, 10, 16),
        color=PALE_OLIVE,
        weight="bold",
    )


def _centre(
    ax: plt.Axes,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    assets: dict[str, list[Path]],
    *,
    model: str,
    video: bool,
    phase: str,
    state: int,
    progress: float,
) -> None:
    spec = MODELS[model]
    panel_box(ax, registry, spec["title"], video=video)
    if phase in {"force", "gradient"}:
        image = assets["force"][state]
    elif phase == "update" and state < len(assets["move"]):
        image = assets["move"][state]
    elif phase == "return":
        image = assets["box"][state]
    elif phase in {"neighbours", "descriptor", "energy"}:
        image = assets["focus"][state]
    else:
        image = assets["box"][state]
    if not image.exists():
        image = assets["box"][0]
    place_main(ax, image, rect=(0.04, 0.12, 0.96, 0.89), alpha=1.0)
    phase_label = {
        "input": "input positions",
        "neighbours": "select local neighbours within 6 Å",
        "descriptor": "encode the local environment",
        "energy": "read out atomic energies and total E",
        "gradient": "differentiate E with respect to positions",
        "force": "force → acceleration",
        "update": "update velocity and position",
        "return": "next MD step",
    }[phase]
    colour = PALE_OLIVE if phase in {"gradient", "force"} else EMERALD if phase == "update" else NAVY
    registry.text(
        ax,
        0.50,
        0.885 if video else 0.925,
        phase_label,
        ha="center",
        va="center",
        fontsize=_font(video, 11, 18),
        color=colour,
        weight="bold",
    )
    energy = float(data["total_energy_ev"][state])
    force = np.asarray(data["forces_ev_per_angstrom"][state], dtype=float)
    max_force = float(np.linalg.norm(force, axis=1).max())
    registry.text(
        ax,
        0.05,
        0.075,
        f"step {state:02d} · O126 · 83 MIC neighbours",
        ha="left",
        va="center",
        fontsize=_font(video, 10, 16),
        color=DARK_GRAY,
    )
    if not video:
        registry.text(
            ax,
            0.95,
            0.035,
            f"E={energy:+.3f} · Fmax={max_force:.3f}",
            ha="right",
            va="center",
            fontsize=_font(video, 10, 16),
            color=GREEN,
            weight="bold",
        )
    if model == "dpa4c" and phase in {"descriptor", "energy"}:
        registry.text(
            ax,
            0.50,
            0.15,
            "radial + angular channels → rotation-aware local descriptor",
            ha="center",
            va="center",
            fontsize=_font(video, 10, 16),
            color=NAVY,
            weight="bold",
            zorder=40,
        )


def _energy_panel(
    ax: plt.Axes,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    *,
    model: str,
    video: bool,
    state: int,
    phase: str,
) -> None:
    panel_box(ax, registry, "ENERGY OUTPUT", video=video)
    energies = np.asarray(data["total_energy_ev"], dtype=float)
    lo, hi = float(energies.min()), float(energies.max())
    span = max(hi - lo, 1.0e-6)
    left, right, bottom, top = (0.15, 0.91, 0.25, 0.70) if video else (0.15, 0.91, 0.27, 0.78)
    ax.plot([left, right], [bottom, bottom], color=LINE_GRAY, lw=1.3)
    ax.plot([left, left], [bottom, top], color=LINE_GRAY, lw=1.3)
    for fraction in (0.0, 0.5, 1.0):
        y = bottom + fraction * (top - bottom)
        ax.plot([left, right], [y, y], color="#E6E8E8", lw=0.8, zorder=0)
    x = np.linspace(left, right, len(energies))
    y = bottom + (energies - lo) / span * (top - bottom)
    ax.plot(x, y, color=NAVY, lw=2.2 if video else 1.5, marker="o", ms=5 if video else 3.5, zorder=5)
    current = min(max(state, 0), len(energies) - 1)
    ax.plot(x[current], y[current], marker="o", ms=9 if video else 6, color=PALE_OLIVE, zorder=6)
    registry.text(ax, left, 0.20, "step 0", ha="left", va="top", fontsize=_font(video, 10, 16), color=DARK_GRAY)
    registry.text(ax, right, 0.20, f"step {len(energies)-1}", ha="right", va="top", fontsize=_font(video, 10, 16), color=DARK_GRAY)
    registry.text(ax, 0.50, 0.78 if video else 0.84, "E = Σ atomic energies", ha="center", va="center", fontsize=_font(video, 11, 16), color=GREEN, weight="bold")
    registry.text(ax, 0.50, 0.16 if video else 0.13, f"current E = {energies[current]:+.3f} eV", ha="center", va="center", fontsize=_font(video, 10, 16), color=INK, weight="bold")
    registry.text(ax, 0.50, 0.07 if video else 0.065, "energy → gradient → force", ha="center", va="center", fontsize=_font(video, 10, 16), color=PALE_OLIVE, weight="bold")


def _node(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    y: float,
    label: str,
    symbol: str,
    active: bool,
    video: bool,
    colour: str,
) -> None:
    height = 0.085 if video else 0.09
    width = 0.84
    fill = colour if active else "#F7F8F6"
    text_colour = WHITE if active else INK
    ax.add_patch(
        FancyBboxPatch(
            (0.08, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            fc=fill,
            ec=colour if active else LINE_GRAY,
            lw=2.0 if video else 1.25,
            zorder=2,
        )
    )
    if video:
        registry.text(ax, 0.50, y, symbol, ha="center", va="center", fontsize=16, color=text_colour, weight="bold", zorder=4)
    else:
        registry.text(ax, 0.50, y + 0.017, label, ha="center", va="center", fontsize=10, color=text_colour, weight="bold", zorder=4)
        registry.text(ax, 0.50, y - 0.018, symbol, ha="center", va="center", fontsize=10, color=text_colour, zorder=4)


def _operator_panel(
    ax: plt.Axes,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    *,
    model: str,
    video: bool,
    phase: str,
    state: int,
) -> None:
    spec = MODELS[model]
    panel_box(ax, registry, "FORCE PROVIDER", video=video)
    chain = spec["chain"]
    symbols = spec["chain_symbols"]
    n = len(chain)
    top = 0.84 if video else 0.86
    bottom = 0.33 if video else 0.22
    ys = np.linspace(top, bottom, n)
    active_index = {
        "input": 0,
        "neighbours": 0,
        "descriptor": 2 if model == "dpa4c" else 1,
        "energy": 5 if model == "dpa4c" else 4,
        "gradient": 6,
        "force": 6,
        "update": 7,
        "return": 7,
    }[phase]
    colours = [LAKE_BLUE, NAVY, NAVY, GREEN, GREEN, PALE_OLIVE, PALE_OLIVE, EMERALD]
    compact_symbols = ("r_ij", "f,B", "X_l", "G,J", "D_i", "eps_i", "E→F", "r′,v′")
    for index, (y, label, symbol) in enumerate(zip(ys, chain, symbols)):
        active = index <= active_index
        _node(ax, registry, y=float(y), label=label, symbol=compact_symbols[index] if video else symbol, active=active, video=video, colour=colours[index])
        if index < n - 1:
            registry.arrow(
                ax,
                (0.50, float(y - (0.047 if video else 0.050))),
                (0.50, float(ys[index + 1] + (0.047 if video else 0.050))),
                arrowstyle="-|>",
                mutation_scale=12 if video else 9,
                lw=1.8 if video else 1.2,
                color=colours[index + 1] if active else LINE_GRAY,
                zorder=1,
            )
    energy = float(data["total_energy_ev"][state])
    force = np.asarray(data["forces_ev_per_angstrom"][state], dtype=float)
    max_force = float(np.linalg.norm(force, axis=1).max())
    registry.text(ax, 0.50, 0.23 if video else 0.15, f"E={energy:+.3f} · Fmax={max_force:.3f}", ha="center", va="center", fontsize=_font(video, 10, 16), color=GREEN, weight="bold")
    if not video:
        registry.text(ax, 0.50, 0.09, r"$\mathbf{E}\;\rightarrow\;-\nabla E\;\rightarrow\;\mathbf{F}\;\rightarrow\;\mathbf{a}$", ha="center", va="center", fontsize=10, color=PALE_OLIVE, weight="bold")
    registry.arrow(ax, (0.18, 0.075 if video else 0.035), (0.82, 0.075 if video else 0.035), arrowstyle="-|>", mutation_scale=11 if video else 8, lw=1.8 if video else 1.2, color=LAKE_BLUE if phase in {"update", "return"} else LINE_GRAY)
    registry.text(ax, 0.50, 0.04 if video else 0.018, "feedback to the left loop → next position", ha="center", va="bottom", fontsize=_font(video, 10, 16), color=NAVY, weight="bold")


def compose(
    fig: plt.Figure,
    registry: LayoutRegistry,
    data: dict[str, np.ndarray],
    assets: dict[str, list[Path]],
    *,
    model: str,
    time_seconds: float,
    video: bool,
    static: bool = False,
) -> list[dict]:
    state, phase, progress = _state_for_time(time_seconds, len(data["positions"]))
    if static:
        state, phase, progress = min(2, len(data["positions"]) - 1), "gradient", 1.0
    slots = (STORY_VIDEO_A, STORY_VIDEO_B, STORY_VIDEO_C, STORY_VIDEO_D) if video else (STORY_STATIC_A, STORY_STATIC_B, STORY_STATIC_C, STORY_STATIC_D)
    left = axes_from_top_slot(fig, slots[0])
    centre = axes_from_top_slot(fig, slots[1])
    upper = axes_from_top_slot(fig, slots[2])
    lower = axes_from_top_slot(fig, slots[3])
    _loop(left, registry, video=video, phase=phase)
    _centre(centre, registry, data, assets, model=model, video=video, phase=phase, state=state, progress=progress)
    _energy_panel(upper, registry, data, model=model, video=video, state=state, phase=phase)
    _operator_panel(lower, registry, data, model=model, video=video, phase=phase, state=state)
    return [
        {"id": "energy", "color": GREEN, "min_pixels": 80 if not video else 250},
        {"id": "gradient", "color": PALE_OLIVE, "min_pixels": 80 if not video else 250},
        {"id": "feedback", "color": LAKE_BLUE, "min_pixels": 40 if not video else 120},
    ]


def render_model(model: str, *, static_only: bool = False, video_only: bool = False) -> None:
    if model not in MODELS:
        raise ValueError(model)
    data = _load_data()
    assets = _assets()
    spec = MODELS[model]
    QA_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema": "md_end_to_end_story/v1",
        "model": model,
        "trajectory_source": str(TRAJECTORY),
        "trajectory_source_role": "shared kinematic and energy/force carrier for the teaching composition",
        "n_states": int(len(data["positions"])),
        "n_atoms": int(data["positions"].shape[1]),
        "operators": list(spec["chain"]),
    }
    json_dump(QA_DIR / f"{model}_provenance.json", provenance)
    if not video_only:
        fig = new_static_figure()
        registry = LayoutRegistry(min_font_pt=10, max_font_pt=16, edge_pad_px=18)
        compose(fig, registry, data, assets, model=model, time_seconds=6.0, video=False, static=True)
        errors = registry.validate(fig)
        if errors:
            raise RuntimeError("Static end-to-end layout failed:\n" + "\n".join(errors))
        save_static(fig, spec["stem"])
    if not static_only:
        audit = {
            "panels": [
                {"id": "integrator", "rect": list(STORY_VIDEO_A), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
                {"id": "atomistic", "rect": list(STORY_VIDEO_B), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
                {"id": "energy", "rect": list(STORY_VIDEO_C), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
                {"id": "force_provider", "rect": list(STORY_VIDEO_D), "min_clearance_px": 0, "allow_touch_edges": ["left", "right", "top", "bottom"]},
            ],
            "whitespace": {"background_threshold": 245, "min_ink_fraction": 0.02, "min_panel_bbox_fill": 0.20, "grid_rows": 12, "grid_columns": 24},
            "bands": [
                {"id": "gap_a_b", "rect": [0.215, 0.025, 0.230, 0.975], "max_ink_pixels": 5000},
                {"id": "gap_b_right", "rect": [0.680, 0.025, 0.695, 0.975], "max_ink_pixels": 5000},
                {"id": "gap_c_d", "rect": [0.695, 0.470, 0.985, 0.500], "max_ink_pixels": 5000},
            ],
        }
        render_video(
            stem=spec["stem"],
            duration_seconds=16.0,
            draw_frame=lambda fig, t, _i, registry: compose(fig, registry, data, assets, model=model, time_seconds=t, video=True),
            audit_config=audit,
            qa_directory=QA_DIR / model / "_qa",
            representative_times=(0.2, 2.0, 4.0, 6.0, 8.0, 10.5, 12.5, 14.5, 15.8),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("deepmd", "dpa4c", "all"), default="all")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--video-only", action="store_true")
    args = parser.parse_args()
    models = ("deepmd", "dpa4c") if args.model == "all" else (args.model,)
    for model in models:
        render_model(model, static_only=args.static_only, video_only=args.video_only)


if __name__ == "__main__":
    main()
