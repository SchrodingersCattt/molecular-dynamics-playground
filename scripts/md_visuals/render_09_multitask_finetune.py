"""09 - multi-task fine-tuning: one shared DPA3 descriptor, two heads, one head per step (static + video).

Config: product/data/09_multitask_config.json (values copied from exp7a_fold0/input.json on a800).
Focus: the shared descriptor in the centre. Each training step draws one head with p = 0.5
(model_prob 0.5 / 0.5); that head lights up and its gradient flows back into the shared descriptor,
whose RepFlow activations update. Inputs and the step ribbon stay in idle (white/light-border) cards. Style: grey cards, black text.
Outputs: product/figures/09_multitask_finetune.png (4000x1250, 300 dpi at 13.33 in),
product/videos/09_multitask_finetune.mp4 (1920x600, 24 fps) + every-frame QA and contact sheets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from explainer_common import (
    CARD_EDGE, CARD_FILL, DARK_GRAY, FONT_SIZES, IDLE_EDGE, IDLE_FILL, INK, LINE_GRAY, MID_GRAY, MM, ORANGE, SIGNED_CMAP,
    TEAL, WHITE, big_arrow, canvas, card, mix_hex, render_video_canvas, save_static, sf, smoothstep,
)

ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "09_multitask_finetune"
CONFIG = ROOT / "data" / "09_multitask_config.json"
QA_ROOT = ROOT / "qa" / STEM
SIZE_IN = (338 * MM, 338 * MM * 5 / 16)  # 338 x 105.6 mm, same width as 08
INTRO, STEP_SECONDS, HOLD = 1.0, 2.0, 1.5
REPRESENTATIVE_TIMES = (0.5, 1.3, 1.8, 2.3, 2.8, 4.5, 6.5, 8.5, 10.5, 12.8, 13.8)

DESC = (0.335, 0.045, 0.655, 0.955)
HEAD = {0: (0.745, 0.560, 0.990, 0.955), 1: (0.745, 0.045, 0.990, 0.440)}
HEAD_COLOUR = {0: TEAL, 1: ORANGE}
HEAD_TEXT = {0: ("$E$ / $F$ head", "pretrained"), 1: ("$V_{\\mathrm{det}}$ head", "random init")}
INPUT = {0: (0.012, 0.560, 0.200, 0.780), 1: (0.012, 0.220, 0.200, 0.440)}
INPUT_TEXT = {0: "DFT frames", 1: "Ion clusters"}
RIBBON_Y = 0.875
LAYER_Y = (0.585, 0.385, 0.185)
CELLS = 16
GRID_X1 = DESC[2] - 0.062


def load():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    probs = [cfg["heads"][h]["model_prob"] for h in ("deepems_vanilla", "pems_vdet_kj")]
    rng = np.random.default_rng(cfg["animation"]["schedule_seed"])
    n = cfg["animation"]["steps_shown"]
    schedule = rng.choice(2, size=n, p=probs).tolist()
    heat = rng.uniform(-1, 1, size=(n + 1, 3, 2, CELLS))
    return {"cfg": cfg, "schedule": schedule, "heat": heat, "duration": INTRO + n * STEP_SECONDS + HOLD}


def state(t, data):
    """Return (step index or None, head, phase dict)."""
    n = len(data["schedule"])
    if t is None:
        return None
    s = (t - INTRO) / STEP_SECONDS
    if s < 0:
        return {"step": -1, "fwd": 0, "head": 0, "grad": 0, "pulse": 0, "upd": 0.0}
    k = min(int(s), n - 1); u = s - k if s < n else 1.0
    return {"step": k, "fwd": smoothstep(u / 0.25), "head": smoothstep((u - 0.25) / 0.2), "grad": smoothstep((u - 0.45) / 0.35),
            "pulse": np.sin(np.pi * min(max((u - 0.78) / 0.22, 0), 1)), "upd": smoothstep((u - 0.78) / 0.22)}


def descriptor(ax, reg, fig, data, st, glow):
    k = sf(fig)
    # focus: same grey card as everything else, but a heavy dark-grey border that flashes the head colour on update
    edge = CARD_EDGE if glow is None else mix_hex(CARD_EDGE, glow[0], 0.8 * glow[1])
    card(ax, DESC, fill=CARD_FILL, edge=edge, lw=3.2 + 1.6 * (glow[1] if glow else 0), fig=fig, z=2)
    reg.text(ax, 0.5 * (DESC[0] + DESC[2]), DESC[3] - 0.075, "Shared DPA3 descriptor", ha="center", va="center",
             fontsize=FONT_SIZES["panel_title"], color=INK)
    heat = data["heat"]
    if st is None:
        cur = heat[-1]
    else:
        k0 = max(st["step"], 0); cur = heat[k0] * (1 - st["upd"]) + heat[k0 + 1] * st["upd"] if st["step"] >= 0 else heat[0]
    gx0, gx1 = DESC[0] + 0.095, GRID_X1; cw = (gx1 - gx0) / CELLS; ch = 0.062
    for li, yc in enumerate(LAYER_Y):
        reg.text(ax, DESC[0] + 0.012, yc, f"RepFlow {li + 1}", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)
        for r in range(2):
            for c in range(CELLS):
                v = 0.5 + 0.5 * float(np.clip(cur[li, r, c], -1, 1))
                ax.add_patch(plt_rect(gx0 + c * cw, yc - ch + r * ch, cw * 0.92, ch * 0.88, SIGNED_CMAP(v), k))


def plt_rect(x, y, w, h, colour, k):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fc=colour, ec=mix_hex(LINE_GRAY, WHITE, 0.3), lw=0.5 * k, zorder=3)


def head(ax, reg, fig, h, level):
    """level 0 = idle (grey), 1 = active."""
    col = HEAD_COLOUR[h]
    # idle: white card, light border; active: grey card, border in the head colour (a line, not text)
    fill = mix_hex(IDLE_FILL, CARD_FILL, level); edge = mix_hex(IDLE_EDGE, col, level)
    card(ax, HEAD[h], fill=fill, edge=edge, lw=1.1 + 1.9 * level, fig=fig, z=2)
    x0, y0, x1, y1 = HEAD[h]; xc = 0.5 * (x0 + x1)
    reg.text(ax, xc, 0.5 * (y0 + y1) + 0.075, HEAD_TEXT[h][0], ha="center", va="center", fontsize=FONT_SIZES["panel_title"],
             color=INK)
    reg.text(ax, xc, 0.5 * (y0 + y1) - 0.085, HEAD_TEXT[h][1], ha="center", va="center", fontsize=FONT_SIZES["body"],
             color=INK)


def grad_arrow(ax, reg, fig, h, progress, label=None, fan=True):
    if progress <= 0.01:
        return
    y = 0.665 if h == 0 else 0.335
    xs, xe = HEAD[h][0] - 0.006, (DESC[2] + 0.006) if fan else (GRID_X1 + 0.010)
    x = xs + (xe - xs) * progress
    p1 = min(progress / 0.7, 1.0) if fan else progress; x = xs + (xe - xs) * p1
    big_arrow(ax, reg, fig, (xs, y), (x, y), colour=HEAD_COLOUR[h], lw=5.0, ms=30, z=6)
    p2 = smoothstep((progress - 0.7) / 0.3)
    if fan and p2 > 0.01:  # back-propagation fans into every RepFlow layer of the shared descriptor
        for yc in LAYER_Y:
            ye = yc - 0.002
            big_arrow(ax, reg, fig, (xe - 0.004, y), (xe - 0.004 + (GRID_X1 + 0.006 - xe + 0.004) * p2, y + (ye - y) * p2),
                      colour=HEAD_COLOUR[h], lw=2.4, ms=16, z=7)
    if label:
        reg.text(ax, 0.5 * (xs + DESC[2]), y + (0.055 if h == 0 else -0.055), label, ha="center", va="center",
                 fontsize=FONT_SIZES["body"], color=INK)


def draw(fig, reg, t, data):
    ax = canvas(fig); st = state(t, data); sched = data["schedule"]
    active = None if st is None or st["step"] < 0 else sched[st["step"]]
    # left column: step ribbon + inputs (de-emphasised)
    reg.text(ax, 0.012, RIBBON_Y + 0.045, "One head per step,", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)
    reg.text(ax, 0.012, RIBBON_Y - 0.035, "$p$ = 0.5 / 0.5", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)
    from explainer_common import disc
    for i, hh in enumerate(sched):
        shown = st is None or i <= st["step"]
        cx = 0.150 + i * 0.026; current = st is not None and i == st["step"]
        disc(ax, fig, cx, RIBBON_Y - 0.035, 0.0105 if current else 0.0075, HEAD_COLOUR[hh] if shown else LINE_GRAY, z=4)
    for h in (0, 1):
        lit = 1.0 if st is None else (st["fwd"] if active == h else 0.0)
        card(ax, INPUT[h], fill=mix_hex(IDLE_FILL, CARD_FILL, lit), edge=mix_hex(IDLE_EDGE, CARD_EDGE, lit), lw=1.1, fig=fig, z=1)
        x0, y0, x1, y1 = INPUT[h]
        reg.text(ax, 0.5 * (x0 + x1), 0.5 * (y0 + y1), INPUT_TEXT[h], ha="center", va="center", fontsize=FONT_SIZES["body"],
                 color=INK)
        yin = 0.5 * (y0 + y1)
        big_arrow(ax, reg, fig, (x1 + 0.006, yin), (DESC[0] - 0.006, yin), colour=mix_hex(LINE_GRAY, MID_GRAY, lit), lw=2.2, ms=18)
        yout = 0.81 if h == 0 else 0.19
        big_arrow(ax, reg, fig, (DESC[2] + 0.006, yout), (HEAD[h][0] - 0.006, yout), colour=mix_hex(LINE_GRAY, MID_GRAY, lit), lw=2.2, ms=18)
    # heads
    for h in (0, 1):
        level = 1.0 if st is None else (0.15 + 0.85 * st["head"] if active == h else 0.15)
        head(ax, reg, fig, h, level)
    # gradients into the shared descriptor
    if st is None:
        grad_arrow(ax, reg, fig, 0, 1.0, "step $t$", fan=False); grad_arrow(ax, reg, fig, 1, 1.0, "step $t$ + 1", fan=False)
        glow = None
    else:
        glow = None
        if active is not None:
            grad_arrow(ax, reg, fig, active, st["grad"], "gradient")
            glow = (HEAD_COLOUR[active], float(st["pulse"]))
    descriptor(ax, reg, fig, data, st, glow)
    return []


def static(data):
    return save_static(lambda fig, reg: draw(fig, reg, None, data), ROOT / "figures" / f"{STEM}.png", SIZE_IN)


def video(data, frame_dump=None):
    return render_video_canvas(stem=STEM, size_in=SIZE_IN, duration_seconds=data["duration"],
                               draw_frame=lambda fig, t, _i, reg: draw(fig, reg, t, data),
                               qa_directory=QA_ROOT / "_qa", representative_times=REPRESENTATIVE_TIMES, frame_dump=frame_dump)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--frame-dump", type=Path, help="also save every 2nd frame (review only)"); args = ap.parse_args()
    data = load(); print("schedule", data["schedule"]); print(static(data))
    if not args.static_only:
        print(video(data, args.frame_dump))


if __name__ == "__main__":
    main()
