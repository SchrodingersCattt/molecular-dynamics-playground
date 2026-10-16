"""08 - t-SNE and UMAP as two five-stage pipelines on one toy data set (static + video).

Data: product/data/08_tsne_umap_toy.json from scripts/build_box/generate_tsne_umap_toy.py (seed 20261032):
24 points in 10-D, exact t-SNE (perplexity 6) and a full-batch UMAP-style cross-entropy optimisation
(k = 6, a = 1.577, b = 0.895), both started from the PCA view of the 10-D cloud.
Rows: t-SNE (top), UMAP (bottom). Stages: 1 points, 2 high-dim similarities, 3 low-dim kernel,
4 loss falling, 5 embedding. The video reveals the stages in order and moves the points along the
recorded optimisation while the loss dot runs down the curve.
Outputs: product/figures/08_tsne_umap_principle.png (4000x1250, 300 dpi at 13.33 in),
product/videos/08_tsne_umap_principle.mp4 (1920x600, 24 fps) + every-frame QA and contact sheets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import render_video
from explainer_common import (
    CLUSTER, FONT_SIZES, INK, INPUT_FILL, LINE_GRAY, LOSS_FILL, MID_GRAY, NAVY, PREDICTION_FILL, SOFT_NAVY,
    ORANGE, SOFT_ORANGE, SOFT_TEAL, STRUCTURE_BG, TEAL, WHITE, mix_hex, big_arrow, canvas, card, full_panel_audit,
    number_disc, save_static, sf, smoothstep, sub_axes,
)

ORANGE_TXT = mix_hex(ORANGE, INK, 0.25)
ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "08_tsne_umap_principle"
DATA = ROOT / "data" / "08_tsne_umap_toy.json"
QA_ROOT = ROOT / "qa" / STEM
ROW_SECONDS = 7.5
DURATION = 2 * ROW_SECONDS + 1.5
STAGE_T = (0.0, 0.8, 1.8, 2.8)       # stage 1, 2, 3 reveal; optimisation from 2.8 to 6.8 s
OPT_SECONDS = 4.0
REPRESENTATIVE_TIMES = (0.4, 1.3, 2.3, 3.5, 5.0, 6.9, 8.0, 9.5, 11.0, 12.5, 14.4, 16.4)

ROWS = ((0.515, 0.985), (0.015, 0.485))
X_START, X_END, GAP = 0.078, 0.994, 0.030
PANEL_W = (X_END - X_START - 4 * GAP) / 5
TITLES = (
    ("10-D points", "Neighbours $p_{ij}$", "Student-$t$  $q_{ij}$", "KL($P\\,\\|\\,Q$)", "Embedding"),
    ("10-D points", "Fuzzy $k$NN graph", "Low-dim  $q_{ij}$", "Cross-entropy", "Embedding"),
)
STYLE = ((INPUT_FILL, SOFT_NAVY), (INPUT_FILL, SOFT_NAVY), (PREDICTION_FILL, SOFT_TEAL), (LOSS_FILL, SOFT_ORANGE), (PREDICTION_FILL, SOFT_TEAL))


def load():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    out = {"labels": np.array(d["labels"]), "pca": np.array(d["pca"])}
    for key, loss in (("tsne", "kl"), ("umap", "ce")):
        m = d[key]
        out[key] = {"traj": np.array(m["traj"]), "loss": np.array(m[loss]),
                    "edges": np.array(m["P" if key == "tsne" else "W"], dtype=float)}
    return out


def panel_rect(row, col):
    y0, y1 = ROWS[row]; x0 = X_START + col * (PANEL_W + GAP)
    return x0, y0, x0 + PANEL_W, y1


def content_rect(row, col):
    x0, y0, x1, y1 = panel_rect(row, col)
    return x0 + 0.010, y0 + 0.030, x1 - 0.010, y1 - 0.105


def fit(ax, fig, rect, Y, pad=1.12):
    """Centre and scale a 2-D layout into the axes, keeping it undistorted."""
    w = (rect[2] - rect[0]) * fig.get_figwidth(); h = (rect[3] - rect[1]) * fig.get_figheight(); r = w / h
    Y = Y - 0.5 * (Y.max(0) + Y.min(0)); s = max(np.abs(Y[:, 0]).max() / r, np.abs(Y[:, 1]).max()) or 1.0
    ax.set_xlim(-pad * r, pad * r); ax.set_ylim(-pad, pad)
    return Y / s


def points(ax, fig, Y, labels, alpha=1.0, size=78):
    k = sf(fig)
    ax.scatter(Y[:, 0], Y[:, 1], s=size * k * k, c=[CLUSTER[i] for i in labels], edgecolors=WHITE,
               linewidths=1.0 * k, alpha=alpha, zorder=4)


def edges(ax, fig, Y, E, colour, alpha, top=None):
    k = sf(fig); w = E[:, 2]
    if top is not None:
        E = E[np.argsort(w)[::-1][:top]]; w = E[:, 2]
    wn = w / w.max()
    for (i, j, _), v in zip(E, wn):
        i, j = int(i), int(j)
        ax.plot([Y[i, 0], Y[j, 0]], [Y[i, 1], Y[j, 1]], color=colour, lw=(0.6 + 2.6 * v) * k,
                alpha=alpha * (0.25 + 0.6 * v), zorder=2, solid_capstyle="round")


def kernel(ax, fig, row, alpha, reg):
    k = sf(fig); d = np.linspace(0, 4, 200)
    ax.set_xlim(-0.15, 4.2); ax.set_ylim(-0.12, 1.12)
    ax.plot([0, 4.1], [0, 0], color=LINE_GRAY, lw=1.0 * k, alpha=alpha)
    ax.plot([0, 0], [0, 1.08], color=LINE_GRAY, lw=1.0 * k, alpha=alpha)
    if row == 0:
        ax.plot(d, np.exp(-d ** 2), color=MID_GRAY, lw=2.4 * k, ls=(0, (3, 2)), alpha=alpha)
        ax.plot(d, 1 / (1 + d ** 2), color=TEAL, lw=3.0 * k, alpha=alpha)
        ax.fill_between(d, np.exp(-d ** 2), 1 / (1 + d ** 2), where=d > 1.0, color=TEAL, alpha=0.14 * alpha, lw=0)
        reg.text(ax, 1.45, 0.60, "Gaussian", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=MID_GRAY, alpha=alpha)
        reg.text(ax, 1.45, 0.86, "heavy tail", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=TEAL, alpha=alpha)
    else:
        q = 1 / (1 + 1.577 * d ** (2 * 0.895))
        ax.plot(d, q, color=TEAL, lw=3.0 * k, alpha=alpha)
        ax.plot(d, 1 - q, color=SOFT_ORANGE, lw=3.0 * k, ls=(0, (3, 2)), alpha=alpha)
        reg.text(ax, 1.55, 0.34, "edges attract", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=TEAL, alpha=alpha)
        reg.text(ax, 1.55, 0.60, "others repel", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=ORANGE_TXT, alpha=alpha)


def loss_curve(ax, fig, loss, u, alpha):
    # x: sqrt-spaced iterations (the video maps time to iteration as u^2)
    k = sf(fig); n = len(loss); x = np.sqrt(np.linspace(0, 1, n)); y = (loss - loss.min()) / (loss.max() - loss.min())
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.12, 1.12)
    ax.plot([0, 1.04], [-0.06, -0.06], color=LINE_GRAY, lw=1.0 * k, alpha=alpha)
    ax.plot([0, 0], [-0.06, 1.08], color=LINE_GRAY, lw=1.0 * k, alpha=alpha)
    ax.plot(x, y, color=LINE_GRAY, lw=2.0 * k, alpha=alpha)
    m = max(1, int(round(u * u * (n - 1))) + 1)
    ax.plot(x[:m], y[:m], color=TEAL, lw=3.2 * k, alpha=alpha)
    ax.scatter([x[m - 1]], [y[m - 1]], s=90 * k * k, color=TEAL, edgecolors=WHITE, linewidths=1.4 * k, zorder=5, alpha=alpha)


def draw(fig, reg, t, data):
    ax = canvas(fig); labels = data["labels"]
    for row, key in enumerate(("tsne", "umap")):
        lt = t - row * ROW_SECONDS
        y0, y1 = ROWS[row]
        reg.text(ax, 0.010, 0.5 * (y0 + y1), ("t-SNE", "UMAP")[row], ha="left", va="center",
                 fontsize=FONT_SIZES["panel_title"], color=INK)
        reveal = [smoothstep((lt - s) / 0.5) for s in STAGE_T] + [smoothstep((lt - STAGE_T[3]) / 0.5)]
        u = smoothstep((lt - STAGE_T[3]) / OPT_SECONDS) if lt > STAGE_T[3] else 0.0
        m = data[key]; traj = m["traj"]; idx = int(round(u * u * (len(traj) - 1)))
        for col in range(5):
            a = reveal[col]; rect = panel_rect(row, col); fill, edge = STYLE[col]
            card(ax, rect, fill=STRUCTURE_BG if a < 0.01 else fill, edge=LINE_GRAY if a < 0.5 else edge, fig=fig, z=1)
            x0, _, x1, yt = rect
            active = a > 0.5
            number_disc(ax, reg, fig, x0 + 0.018, yt - 0.052, col + 1, NAVY if active else LINE_GRAY)
            reg.text(ax, x0 + 0.036, yt - 0.052, TITLES[row][col], ha="left", va="center_baseline",
                     fontsize=FONT_SIZES["body"], color=INK if active else MID_GRAY)
            if col < 4:
                ym = 0.5 * (rect[1] + rect[3]) - 0.03
                big_arrow(ax, reg, fig, (x1 + 0.004, ym), (x1 + GAP - 0.004, ym),
                          colour=NAVY if reveal[col + 1] > 0.5 else LINE_GRAY, lw=3.2, ms=20)
            if a < 0.01:
                continue
            cr = content_rect(row, col)
            if row == 1 and col == 4:  # leave a caption band under the UMAP embedding
                cr = (cr[0], cr[1] + 0.055, cr[2], cr[3])
            sub = sub_axes(fig, cr)
            if col == 0:
                points(sub, fig, fit(sub, fig, cr, data["pca"]), labels, alpha=a)
            elif col == 1:
                Y = fit(sub, fig, cr, data["pca"])
                edges(sub, fig, Y, m["edges"], NAVY, a, top=48 if key == "tsne" else None)
                points(sub, fig, Y, labels, alpha=a)
            elif col == 2:
                kernel(sub, fig, row, a, reg)
            elif col == 3:
                loss_curve(sub, fig, m["loss"], u, a)
            else:
                Y = fit(sub, fig, cr, traj[idx])
                points(sub, fig, Y, labels, alpha=a)
                if key == "umap":
                    reg.text(ax, 0.5 * (x0 + x1), rect[1] + 0.020, "more global layout", ha="center", va="bottom",
                             fontsize=FONT_SIZES["micro"], color=TEAL, alpha=a)
    return []


def static(data):
    return save_static(lambda fig, reg: draw(fig, reg, DURATION, data), ROOT / "figures" / f"{STEM}.png")


def video(data):
    return render_video(stem=STEM, duration_seconds=DURATION,
                        draw_frame=lambda fig, t, _i, reg: draw(fig, reg, t, data),
                        audit_config=full_panel_audit(), qa_directory=QA_ROOT / "_qa",
                        representative_times=REPRESENTATIVE_TIMES)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--static-only", action="store_true"); args = ap.parse_args()
    data = load(); print(static(data))
    if not args.static_only:
        print(video(data))


if __name__ == "__main__":
    main()
