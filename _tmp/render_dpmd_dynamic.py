"""04: a moving water box, exact compressed DeepMD tensors, and a VV loop."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts/md_visuals"))
import common
from common import LayoutRegistry, new_video_figure, render_video, smoothstep
from mattervis_story import draw_vv_loop
from deepmd_trace import export_trace
from verify_dpmd_video import verify
from dpmd_motion import DIRECTORY as MOTION_DIR, DIRECTION, TARGET, BOX_SCALE, FOCUS_SCALE, prepare, schedule

OUTPUT = HERE / "04_deep_potential_md_dynamic.mp4"
QA = HERE / "04_deep_potential_md_dynamic_qa"
TRACE = QA / "deepmd_trace"
PREVIEWS = QA / "preview_current"
INK, GREY, FAINT = "#12243C", "#7A868A", "#EEF1F2"
GREEN, BLUE, GOLD, RED = "#246249", "#205A74", "#766D29", "#941B32"
BOX_RECT = (343, 142, 425)
FOCUS_RECT = (784, 200, 298)
KEY_TIMES = (0.1, 1.7, 3.5, 6.5, 8.8, 11.8, 16.5, 19.4, 20.6, 21.4, 29.4)


def load_case():
    if not (TRACE / "deepmd_trace.npz").exists():
        export_trace(TRACE)
    trace_report = json.loads((TRACE / "deepmd_trace.json").read_text(encoding="utf-8"))
    if not trace_report.get("passed"):
        raise RuntimeError("DeepMD numeric verification did not pass")
    with np.load(TRACE / "deepmd_trace.npz", allow_pickle=False) as source:
        trace = {key: source[key] for key in source.files}
    manifest = prepare()
    with np.load(MOTION_DIR / "display_geometry.npz", allow_pickle=False) as source:
        geometry = source["positions"].copy()
        focus_ids = source["focus_ids"].copy()
    shutil.copy2(TRACE / "deepmd_trace.npz", HERE / "dpmd_dynamic_calculation_record.npz")
    record = {**trace_report, "motion_manifest": str(MOTION_DIR / "manifest.json"),
              "display_sampling": manifest["interpolation"]}
    (HERE / "dpmd_dynamic_calculation_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"trace": trace, "geometry": geometry, "focus_ids": focus_ids,
            "manifest": manifest, "trace_report": trace_report}


@lru_cache(maxsize=6)
def image_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def project(points, rect, scale):
    right = np.cross(-DIRECTION, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, -DIRECTION)
    delta = np.asarray(points) - TARGET
    x, y, size = rect
    return np.column_stack((x + size / 2 + delta @ right * size / (2 * scale),
                            y + size / 2 - delta @ up * size / (2 * scale)))


def text(reg, ax, x, y, value, color=INK, size=16, ha="center", weight="normal"):
    return reg.text(ax, x, y, value, fontsize=size, color=color, ha=ha,
                    va="center", weight=weight, zorder=40)


def arrow(ax, start, end, color=GREY, width=1.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                color=color, lw=width, zorder=20))


def stage_weights(t):
    if t < 20:
        return tuple(smoothstep((t - start) / length) for start, length in
                     ((1.2, .6), (3.0, .6), (4.2, .6), (5.2, .6),
                      (6.2, .6), (7.3, .6), (8.4, .6), (9.5, .6)))
    _, _, _, phase = schedule(t)
    local = (t - 20) % 2
    if phase in ("half_kick", "drift"):
        return (.15,) * 8
    return tuple(smoothstep((local - start) / .12) for start in
                 (.95, .98, 1.04, 1.12, 1.20, 1.33, 1.45, 1.55))


def active(color, weight):
    return common.mix_hex(GREY, color, float(weight))


def scene(ax, reg, data, t, state, phase, weights):
    frame = 0 if t < 20 else min(int(round((t - 20) * 24)), 239)
    positions = data["geometry"][frame]
    box = float(data["trace"]["cell"][0, 0])
    delta = positions - positions[126]
    delta -= box * np.rint(delta / box)
    distance = np.linalg.norm(delta, axis=1)
    neighbors = np.flatnonzero((distance > 1e-10) & (distance < 6.0))
    bonded = t < 1.2
    for kind, rect, scale in (("box", BOX_RECT, BOX_SCALE), ("focus", FOCUS_RECT, FOCUS_SCALE)):
        path = MOTION_DIR / f"{kind}_bonded.png" if bonded else MOTION_DIR / f"{kind}_frames/{frame+1:04d}.png"
        x, y, size = rect
        ax.imshow(image_array(path), extent=(x, x + size, y + size, y),
                  origin="upper", interpolation="lanczos", zorder=3)
        local_points = positions[126] + delta if kind == "box" else TARGET + delta
        centre = positions[126] if kind == "box" else TARGET
        centre_xy = project(centre[None], rect, scale)[0]
        if not bonded:
            growth = smoothstep((t - 1.2) / 1.8) if t < 3 else 1.0
            radius = 6 * size / (2 * scale) * growth
            ax.add_patch(Circle(centre_xy, radius, fill=False, ec=BLUE,
                                lw=1.5, linestyle=(0, (3, 3)), zorder=15))
            for j in neighbors:
                if distance[j] > 6 * growth:
                    continue
                endpoint = project(local_points[j:j+1], rect, scale)[0]
                color = GREEN if j == 127 else "#ADBCC0"
                ax.plot([centre_xy[0], endpoint[0]], [centre_xy[1], endpoint[1]],
                        lw=2.2 if j == 127 else .50, color=color,
                        alpha=.95 if j == 127 else .58, zorder=8)
            endpoint = project(local_points[127:128], rect, scale)[0]
            ax.add_patch(Circle(endpoint, 6.5, fill=False, ec=GREEN, lw=1.8, zorder=17))
            if kind == "focus":
                text(reg, ax, 930, 521, r"$j=127$", GREEN)
                text(reg, ax, 553, 573, rf"$N_i={len(neighbors)}$", BLUE)
        ax.add_patch(Circle(centre_xy, 8, fill=False, ec=INK, lw=1.7, zorder=16))
        if weights[7] > .01:
            force = data["trace"]["forces_ev_per_angstrom"][state, 126]
            end = project((centre + 4.0 * force)[None], rect, scale)[0]
            arrow(ax, centre_xy, end, active(GOLD, weights[7]), 2.6)
    text(reg, ax, 552, 75, r"$64\,H_2O$", INK, size=18)
    text(reg, ax, 930, 95, r"$O_{126}$", INK, size=18)
    text(reg, ax, 930, 134, r"$r_c=6.0\ \mathrm{\AA}$", BLUE)


def matrix(ax, reg, data, t, state, weights):
    trace = data["trace"]
    positions = trace["positions"][state]
    box = float(trace["cell"][0, 0])
    xs = (1126, 1200, 1308, 1418, 1528, 1638)
    for x, label in zip(xs, ("k", "j", r"$r/\mathrm{\AA}$", r"$\Delta x$", r"$\Delta y$", r"$\Delta z$")):
        text(reg, ax, x, 43, label, GREY)
    for row, j in enumerate((78, 127, 128)):
        index = int(np.flatnonzero(trace["nlist"][state] == j)[0])
        delta = positions[j] - positions[126]
        delta -= box * np.rint(delta / box)
        radius = float(np.linalg.norm(delta))
        y = 86 + row * 39
        color = GREEN if j == 127 else BLUE
        weight = weights[1] if j == 127 else .45 * weights[1]
        ax.add_patch(Rectangle((1111, y - 15), 584, 30, fc=active("#E7F0EC", weight), ec="none", alpha=.33, zorder=1))
        values = (str(index), f"{trace['elements'][j]}{j}", f"{radius:.4f}", *(f"{v:+.3f}" for v in delta))
        for x, value in zip(xs, values):
            text(reg, ax, x, y, value, active(color, weight))
    for x, symbol, dimensions, weight in (
        (1163, r"$\bar R_i$", r"$600\times4$", weights[2]),
        (1305, r"$G_i$", r"$600\times100$", weights[3]),
        (1450, r"$A_i$", r"$4\times100$", weights[4]),
    ):
        ax.add_patch(Rectangle((x-48, 211), 96, 35, fill=False, ec=active(BLUE, weight), lw=1.3))
        text(reg, ax, x, 228, symbol, active(BLUE, weight))
        text(reg, ax, x, 262, dimensions, active(BLUE, weight))
    arrow(ax, (1216, 228), (1246, 228), active(BLUE, weights[3]))
    arrow(ax, (1358, 228), (1395, 228), active(BLUE, weights[4]))
    arrow(ax, (1504, 228), (1730, 194), active(RED, weights[4]))
    d = trace["D"][state]
    vmax = max(float(np.percentile(np.abs(trace["D"]), 97)), 1e-12)
    rgba = np.empty((100, 12, 4))
    for sign, color in ((True, GREEN), (False, RED)):
        mask = d >= 0 if sign else d < 0
        rgba[mask, :3] = plt.matplotlib.colors.to_rgb(color)
    rgba[:, :, 3] = (.08 + .85*np.clip(np.abs(d)/vmax, 0, 1)) * (.16 + .84*weights[4])
    ax.imshow(rgba, extent=(1750, 1882, 205, 65), origin="upper", aspect="auto", zorder=8, interpolation="nearest")
    text(reg, ax, 1816, 38, r"$D_i$", active(RED, weights[4]), size=18)
    text(reg, ax, 1816, 235, r"$100\times12$", active(RED, weights[4]))
    ax.add_patch(Rectangle((1749, 64), 13, 5, fill=False, ec=RED, lw=1.5, zorder=12))
    row = int(np.flatnonzero(trace["nlist"][state] == 127)[0])
    if t < 6:
        label, values, color = r"$R_{" + str(row) + "}$", trace["R"][state, row], BLUE
    elif t < 8:
        label, values, color = r"$\bar R=(R-\mu)/\sigma$", trace["Rbar"][state, row], BLUE
    elif t < 10:
        label, values, color = r"$G_{" + str(row) + ",0:4}$", trace["G"][state, row, :4], GREEN
    else:
        label, values, color = r"$D_{0,0}$", trace["A"][state, :, 0], GOLD
    text(reg, ax, 1130, 308, label, active(color, max(weights[1:5])), ha="left")
    if t < 10:
        for x, value in zip((1370, 1510, 1650, 1790), values):
            text(reg, ax, x, 308, f"{value:+.3f}", active(color, max(weights[1:5])))
    else:
        for col, (x, value) in enumerate(zip((1285, 1405, 1525, 1645), values)):
            text(reg, ax, x, 308, "$" + f"{value:.4f}" + r"^2$", active(color, weights[4]))
            if col < 3:
                text(reg, ax, x+60, 308, "+", active(color, weights[4]))
        text(reg, ax, 1878, 308, r"$\simeq " + f"{d[0,0]:.6f}" + "$", active(color, weights[4]), ha="right")


def fitting(ax, reg, data, state, weights):
    ax.plot((1110, 1890), (344, 344), color="#D6DDDF", lw=.9)
    # Three compact hidden layers make the fitting network immediately
    # recognizable; each column is a small visual sample of the real dense
    # layer, while the trace records retain the full 240-neuron widths.
    columns = (1152, 1320, 1460, 1600, 1790)
    node_rows = (371, 386, 401)
    active_weight = smoothstep(weights[5] * 3.0)
    for left, right in zip(columns[:-1], columns[1:]):
        for y1 in node_rows:
            for y2 in node_rows:
                level = active(GREEN, active_weight)
                ax.plot((left + 15, right - 15), (y1, y2), color=level, lw=1.0, alpha=.22 + .60 * active_weight, zorder=1)
    for index, x in enumerate(columns):
        if index == 0:
            ys = (386,)
            color = RED
            labels = (r"$D_i$",)
        elif index == len(columns) - 1:
            ys = (386,)
            color = GREEN
            labels = (r"$\epsilon_i$",)
        else:
            ys = node_rows
            color = GREEN
            labels = ("", "", "")
        for y, label in zip(ys, labels):
            ax.add_patch(Circle((x, y), 11 if index in (0, len(columns)-1) else 7,
                                fc=active(color, active_weight), ec=color, lw=1.3, zorder=3))
            if label:
                text(reg, ax, x, y, label, "#FFFFFF" if active_weight > .5 else color, size=13, weight="bold")
    text(reg, ax, 1152, 420, "D", active(RED, weights[5]), size=14)
    trace = data["trace"]
    values = (
        (453, r"$\epsilon_{O126}$", f"{trace['atomic_energy_ev'][state,126]:+.6f} eV", GREEN, weights[5]),
        (503, r"$U=\sum_i\epsilon_i$", f"{trace['total_energy_ev'][state]:+.6f} eV", INK, weights[6]),
        (555, r"$F_{O126}=-\nabla_{O126} U$", "(" + ", ".join(f"{v:+.3f}" for v in trace['forces_ev_per_angstrom'][state,126]) + r") eV/$\mathrm{\AA}$", GOLD, weights[7]),
    )
    for y, label, value, color, weight in values:
        text(reg, ax, 1124, y, label, active(color, weight), ha="left")
        text(reg, ax, 1880, y, value, active(color, weight), ha="right")


def draw_frame(fig, time, index, reg, data):
    step, alpha, state, phase = schedule(time)
    weights = stage_weights(time)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920)
    ax.set_ylim(600, 0)
    ax.axis("off")
    ring = common.axes_from_top_slot(fig, (.015, .025, .17, .97))
    active_stage = 0 if time < 1.2 or phase == "drift" else 2 if phase in ("half_kick", "final_kick") else 1
    equation = (r"$r_{n+1}=r_n$" + "\n" + r"$+v_{n+1/2}\Delta t$") if active_stage == 0 else (r"$v_{n+1}=v_{n+1/2}$" + "\n" + r"$+\frac{1}{2}a_{n+1}\Delta t$") if active_stage == 2 else r"$a_i=F_i/m_i$"
    draw_vv_loop(ring, reg, video=True, active_stage=active_stage, centre_text=equation, centre_y=.53, radius_x=.38)
    if time >= 20:
        text(reg, ax, 165, 547, f"VV {step+1}/5", GOLD)
    scene(ax, reg, data, time, state, phase, weights)
    matrix(ax, reg, data, time, state, weights)
    fitting(ax, reg, data, state, weights)
    return []


def preview(data):
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    records, thumbnails = [], []
    for i, time in enumerate(KEY_TIMES):
        fig = new_video_figure()
        registry = LayoutRegistry(min_font_pt=16, max_font_pt=18, edge_pad_px=12,
                                  font_family="Arial", coerce_min_font=True)
        draw_frame(fig, time, round(time*24), registry, data)
        errors = registry.validate(fig)
        path = PREVIEWS / f"{i:02d}_{time:05.2f}.png"
        fig.savefig(path, dpi=100)
        records.append({"time": time, "errors": errors, "path": str(path)})
        thumbnails.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).resize((640, 200)))
        plt.close(fig)
    sheet = Image.new("RGB", (1920, 200*int(np.ceil(len(thumbnails)/3))), "white")
    for i, thumb in enumerate(thumbnails):
        sheet.paste(thumb, ((i%3)*640, (i//3)*200))
    sheet.save(PREVIEWS / "contact.png")
    (PREVIEWS / "report.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    failed = [r for r in records if r["errors"]]
    if failed:
        raise RuntimeError(json.dumps(failed, indent=2))
    return PREVIEWS / "contact.png"


def render():
    data = load_case()
    print("Preview", preview(data), flush=True)
    common.VIDEO_DIR = QA / "candidate"
    output = render_video(stem="04_deep_potential_md_dynamic", duration_seconds=30,
                          draw_frame=lambda f,t,i,r: draw_frame(f,t,i,r,data),
                          audit_config={"panels": [], "bands": [],
                            "whitespace": {"background_threshold":245, "min_ink_fraction":.01, "min_panel_bbox_fill":0, "grid_rows":12, "grid_columns":24},
                            "max_vertical_border_whitespace_px":60},
                          qa_directory=QA, representative_times=KEY_TIMES)
    print("Acceptance", verify(output), flush=True)
    output.replace(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-only", action="store_true")
    options = parser.parse_args()
    print(preview(load_case()) if options.preview_only else render())
