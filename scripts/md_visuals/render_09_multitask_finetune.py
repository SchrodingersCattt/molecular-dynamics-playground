"""09 - multi-task fine-tuning as DeePMD-kit actually runs it (static + video).

Verified on a800 (read-only), deepmd-kit editable install /aisi-nas/guomingyu/SOFTWARE/deepmd-kit-250729 @ 04e3517b,
deepmd/pt/train/training.py, Trainer.run() -> step():
  * each step draws ONE model_key: dp_random.choice(arange(num_model), p=self.model_prob)   (l.700-704)
  * zero_grad -> get_data(task_key) -> wrapper(**input, label, task_key) -> loss.backward()
    -> clip_grad_norm_(gradient_max_norm) -> optimizer.step() -> scheduler.step()             (l.714-740)
  * model_prob is normalised from training.model_prob (l.642-653); the two descriptors are tied by
    wrapper.share_params(shared_links) (l.588), so the update reaches the one shared DPA3 descriptor.
exp7a_fold0/input.json: model_prob {deepems_vanilla: 0.5, pems_vdet_kj: 0.5}; gradient_max_norm 1.0; 400k steps.
=> serial stochastic alternation: one head, one batch, one loss, one backward pass per step; losses are not summed.

Figure: per sampled step the active lane shows the forward pass (input -> shared descriptor -> head -> prediction ->
loss vs label, dark-grey arrows) and the backward pass (gradient from the loss back through the head into every
RepFlow layer of the shared descriptor, arrows in the head colour), then the update. The other lane stays idle.
Size 338 x 125 mm (same as 08), text in true points (>= 14 pt). Style: grey cards, dark-grey borders, black text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from explainer_common import (
    CARD_EDGE, CARD_FILL, DARK_GRAY, FONT_SIZES, IDLE_EDGE, IDLE_FILL, INK, LINE_GRAY, MM, ORANGE, SIGNED_CMAP,
    TEAL, WHITE, big_arrow, canvas, card, disc, mix_hex, render_video_canvas, save_static, smoothstep,
)

ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "09_multitask_finetune"
CONFIG = ROOT / "data" / "09_multitask_config.json"
QA_ROOT = ROOT / "qa" / STEM
SIZE_IN = (338 * MM, 125 * MM)
INTRO, STEP_SECONDS, HOLD = 1.0, 3.0, 1.5
REPRESENTATIVE_TIMES = (0.5, 1.6, 2.4, 3.0, 3.7, 5.5, 8.5, 11.5, 14.5, 17.5, 19.0, 20.4)

COLOUR = {0: TEAL, 1: ORANGE}
LANE_Y = {0: 0.640, 1: 0.220}          # centre line of each task lane
LANE_H = 0.260
DESC = (0.185, 0.045, 0.425, 0.835)
COLS = {"input": (0.010, 0.140), "head": (0.490, 0.640), "pred": (0.690, 0.815), "loss": (0.865, 0.992)}
TEXT = {
    "input": ("DFT frames", "PEMs clusters"),
    "head": (("$E$ / $F$ head", "pretrained"), ("$V_{\\mathrm{det}}$ head", "random init")),
    "pred": ("$\\hat{E}$, $\\hat{F}$", "$\\hat{V}_{\\mathrm{det}}$"),
    "loss": (("ener loss", "vs DFT $E$, $F$"), ("MAE loss", "vs $V_{\\mathrm{det}}$ label")),
}
LAYER_Y = (0.600, 0.435, 0.270)
CELLS = 12
FWD_DY, BWD_DY = 0.045, -0.045


def load():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    probs = [cfg["heads"][h]["model_prob"] for h in ("deepems_vanilla", "pems_vdet_kj")]
    rng = np.random.default_rng(cfg["animation"]["schedule_seed"])
    n = cfg["animation"]["steps_shown"]
    schedule = rng.choice(2, size=n, p=probs).tolist()
    heat = rng.uniform(-1, 1, size=(n + 1, 3, 2, CELLS))
    return {"cfg": cfg, "schedule": schedule, "heat": heat, "duration": INTRO + n * STEP_SECONDS + HOLD}


def phases(u):
    """Within one step (u in 0..1): forward 0-0.38, loss 0.38-0.48, backward 0.48-0.82, update 0.82-1."""
    return {"fwd": smoothstep(u / 0.38), "loss": smoothstep((u - 0.36) / 0.10), "bwd": smoothstep((u - 0.48) / 0.34),
            "upd": smoothstep((u - 0.82) / 0.18), "pulse": float(np.sin(np.pi * min(max((u - 0.82) / 0.18, 0), 1)))}


def state(t, data):
    n = len(data["schedule"])
    if t is None:
        return None
    s = (t - INTRO) / STEP_SECONDS
    if s < 0:
        return {"step": -1, "lane": None, **phases(0.0)}
    k = min(int(s), n - 1); u = s - k if s < n else 1.0
    return {"step": k, "lane": data["schedule"][k], **phases(u)}


def lane_card(ax, reg, fig, key, h, level):
    x0, x1 = COLS[key]; yc = LANE_Y[h]; r = (x0, yc - LANE_H / 2, x1, yc + LANE_H / 2)
    edge = mix_hex(IDLE_EDGE, COLOUR[h] if key == "head" else CARD_EDGE, level)
    card(ax, r, fill=mix_hex(IDLE_FILL, CARD_FILL, level), edge=edge, lw=1.1 + (1.9 if key == "head" else 0.4) * level, fig=fig, z=2)
    xc = 0.5 * (x0 + x1); txt = TEXT[key][h]
    if isinstance(txt, tuple):
        reg.text(ax, xc, yc + 0.045, txt[0], ha="center", va="center", fontsize=FONT_SIZES["body"], color=INK)
        reg.text(ax, xc, yc - 0.050, txt[1], ha="center", va="center", fontsize=FONT_SIZES["micro"], color=INK)
    else:
        reg.text(ax, xc, yc, txt, ha="center", va="center", fontsize=FONT_SIZES["body"], color=INK)


def descriptor(ax, reg, fig, data, st, glow):
    edge = CARD_EDGE if glow is None else mix_hex(CARD_EDGE, glow[0], 0.8 * glow[1])
    card(ax, DESC, fill=CARD_FILL, edge=edge, lw=3.2 + 1.6 * (glow[1] if glow else 0), fig=fig, z=2)
    xc = 0.5 * (DESC[0] + DESC[2])
    reg.text(ax, xc, DESC[3] - 0.060, "Shared DPA3", ha="center", va="center", fontsize=FONT_SIZES["panel_title"], color=INK)
    reg.text(ax, xc, DESC[3] - 0.135, "descriptor", ha="center", va="center", fontsize=FONT_SIZES["panel_title"], color=INK)
    heat = data["heat"]
    if st is None or st["step"] < 0:
        cur = heat[-1] if st is None else heat[0]
    else:
        k0 = st["step"]; cur = heat[k0] * (1 - st["upd"]) + heat[k0 + 1] * st["upd"]
    gx0, gx1 = DESC[0] + 0.088, DESC[2] - 0.012; cw = (gx1 - gx0) / CELLS; ch = 0.042
    for li, yc in enumerate(LAYER_Y):
        reg.text(ax, DESC[0] + 0.010, yc, f"RepFlow {li + 1}", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)
        for r in range(2):
            for c in range(CELLS):
                v = 0.5 + 0.5 * float(np.clip(cur[li, r, c], -1, 1))
                ax.add_patch(Rectangle((gx0 + c * cw, yc - ch + r * ch), cw * 0.9, ch * 0.86, fc=SIGNED_CMAP(v),
                                       ec=mix_hex(LINE_GRAY, WHITE, 0.3), lw=0.4, zorder=3))
    reg.text(ax, xc, DESC[1] + 0.045, "shared by both heads", ha="center", va="center", fontsize=FONT_SIZES["micro"], color=INK)


def segments(h):
    """Forward path pieces (x0, x1, y) of lane h; descriptor edge -> head etc."""
    y = LANE_Y[h]
    return [(COLS["input"][1], DESC[0], y), (DESC[2], COLS["head"][0], y), (COLS["head"][1], COLS["pred"][0], y),
            (COLS["pred"][1], COLS["loss"][0], y)]


def forward(ax, reg, fig, h, p, colour=DARK_GRAY):
    segs = segments(h); n = len(segs)
    for i, (x0, x1, y) in enumerate(segs):
        q = min(max(p * n - i, 0.0), 1.0)
        if q > 0.01:
            big_arrow(ax, reg, fig, (x0 + 0.005, y + FWD_DY), (x0 + 0.005 + (x1 - x0 - 0.010) * q, y + FWD_DY),
                      colour=colour, lw=2.6, ms=18, z=6)


def backward(ax, reg, fig, h, p):
    segs = segments(h)[1:][::-1]; n = len(segs) + 1; col = COLOUR[h]
    for i, (x0, x1, y) in enumerate(segs):
        q = min(max(p * n - i, 0.0), 1.0)
        if q > 0.01:
            big_arrow(ax, reg, fig, (x1 - 0.005, y + BWD_DY), (x1 - 0.005 - (x1 - x0 - 0.010) * q, y + BWD_DY),
                      colour=col, lw=4.0, ms=24, z=6)
    q = min(max(p * n - (n - 1), 0.0), 1.0)
    if q > 0.01:  # into every RepFlow layer of the shared descriptor
        xs, y = DESC[2] - 0.004, LANE_Y[h] + BWD_DY
        for yc in LAYER_Y:
            big_arrow(ax, reg, fig, (xs, y), (xs - 0.050 * q, y + (yc - y) * q), colour=col, lw=2.2, ms=14, z=7)


def ribbon(ax, reg, fig, st, sched):
    reg.text(ax, 0.010, 0.935, "Each step: sample one task with model_prob 0.5 / 0.5", ha="left", va="center",
             fontsize=FONT_SIZES["micro"], color=INK)
    for i, hh in enumerate(sched):
        shown = st is None or i <= st["step"]; cur = st is not None and i == st["step"]
        disc(ax, fig, 0.395 + i * 0.024, 0.935, 0.0100 if cur else 0.0070, COLOUR[hh] if shown else LINE_GRAY, z=4)
    # legend: line colour carries the meaning, text stays black
    big_arrow(ax, reg, fig, (0.640, 0.935), (0.690, 0.935), colour=DARK_GRAY, lw=2.6, ms=18, z=6)
    reg.text(ax, 0.698, 0.935, "forward", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)
    big_arrow(ax, reg, fig, (0.835, 0.935), (0.785, 0.935), colour=TEAL, lw=4.0, ms=22, z=6)
    reg.text(ax, 0.843, 0.935, "gradient → update", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)


def draw(fig, reg, t, data):
    ax = canvas(fig); st = state(t, data); sched = data["schedule"]
    ribbon(ax, reg, fig, st, sched)
    glow = None
    for h in (0, 1):
        if st is None:
            lv = {k: 1.0 for k in COLS}; fwd, bwd = 1.0, 1.0
        elif st["lane"] == h:
            f = st["fwd"]
            lv = {"input": 1.0, "head": smoothstep(f * 2 - 0.5), "pred": smoothstep(f * 2 - 1.0), "loss": st["loss"]}
            fwd, bwd = f, st["bwd"]
            glow = (COLOUR[h], st["pulse"])
        else:
            lv = {k: 0.0 for k in COLS}; fwd, bwd = 0.0, 0.0
        for key in COLS:
            lane_card(ax, reg, fig, key, h, lv[key])
        if fwd <= 0.01:  # idle lane: faint guide arrows only
            for x0, x1, y in segments(h):
                big_arrow(ax, reg, fig, (x0 + 0.005, y + FWD_DY), (x1 - 0.005, y + FWD_DY), colour=LINE_GRAY, lw=1.6, ms=14, z=4)
        forward(ax, reg, fig, h, fwd)
        backward(ax, reg, fig, h, bwd)
    if st is None:
        pass
    elif st["lane"] is not None:
        reg.text(ax, 0.5 * (COLS["head"][1] + COLS["pred"][0]) + 0.005, 0.5 * (LANE_Y[0] + LANE_Y[1]),
                 f"step {st['step'] + 1}", ha="center", va="center", fontsize=FONT_SIZES["body"], color=INK)
    descriptor(ax, reg, fig, data, st, glow if glow and glow[1] > 0.01 else None)
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
