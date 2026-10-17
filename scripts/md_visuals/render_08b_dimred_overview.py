"""08b - one-row overview of PCA / t-SNE / UMAP for a short talk slot (static + video).

Same toy data as 08 (product/data/08_tsne_umap_toy.json, scripts/build_box/generate_tsne_umap_toy.py).
Five cards = the shared abstraction; inside each card the three methods are stacked as small variants,
aligned with the method labels on the left:
  1 High-dim data (one shared 10-D cloud)
  2 Structure      PCA covariance ellipse | t-SNE neighbour probabilities p_ij | UMAP fuzzy kNN graph
  3 Objective      max variance | min KL(P||Q) | min cross-entropy
  4 Solve          eigen-decomposition (scree, one shot) | gradient descent (loss) | SGD (loss)
  5 2-D embedding  linear projection | t-SNE layout | UMAP layout
Video: cards light up left to right; then all three methods solve together (PCA rotates onto PC1/PC2,
t-SNE/UMAP follow their recorded optimisation while the loss dot runs down).
Size: 338 x 125 mm, text in true points (>= 14 pt). Static 300 dpi; video 1920 px wide, 24 fps.
Style: grey cards (dark-grey border, light-grey fill), dark-grey step discs and arrows, black text.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse

from explainer_common import (
    CARD_EDGE, CARD_FILL, DARK_GRAY, DISC, FONT_SIZES, IDLE_EDGE, IDLE_FILL, INK, LINE_GRAY, MM,
    big_arrow, canvas, card, number_disc, render_video_canvas, save_static, smoothstep, sub_axes,
)
from render_08_tsne_umap import edges, fit, load, loss_curve, pca_proj, points, scree

ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "08b_dimred_overview"
QA_ROOT = ROOT / "qa" / STEM
SIZE_IN = (338 * MM, 125 * MM)
REVEAL_T = (0.0, 1.2, 2.4, 3.6, 4.8)
OPT_T, OPT_SECONDS = 4.8, 5.0
DURATION = 11.5
REPRESENTATIVE_TIMES = (0.5, 1.7, 2.9, 4.1, 5.5, 7.0, 8.5, 10.0, 11.4)

X_START, X_END, GAP = 0.086, 0.995, 0.028
PANEL_W = (X_END - X_START - 4 * GAP) / 5
Y0, Y1 = 0.012, 0.988
TITLE_Y = Y1 - 0.052
BANDS = ((0.618, 0.878), (0.335, 0.595), (0.040, 0.312))  # PCA, t-SNE, UMAP (inside every card)
TITLES = ("High-dim data", "Structure", "Objective", "Solve", "2-D embedding")
NAMES = ("PCA", "t-SNE", "UMAP")
OBJECTIVE = ("max variance", "min KL($P\\,\\|\\,Q$)", "min cross-entropy")
TITLE_PT = 16.0


def rect(col):
    x0 = X_START + col * (PANEL_W + GAP)
    return x0, Y0, x0 + PANEL_W, Y1


def band_rect(col, b, pad=0.012):
    x0, _, x1, _ = rect(col); y0, y1 = BANDS[b]
    return x0 + pad, y0, x1 - pad, y1


def draw(fig, reg, t, data):
    ax = canvas(fig); labels = data["labels"]
    reveal = [smoothstep((t - s) / 0.5) for s in REVEAL_T]
    u = smoothstep((t - OPT_T) / OPT_SECONDS) if t > OPT_T else 0.0
    for b, name in enumerate(NAMES):
        reg.text(ax, 0.010, 0.5 * sum(BANDS[b]), name, ha="left", va="center", fontsize=FONT_SIZES["panel_title"], color=INK)
    for col in range(5):
        a = reveal[col]; r = rect(col); active = a > 0.5
        card(ax, r, fill=CARD_FILL if active else IDLE_FILL, edge=CARD_EDGE if active else IDLE_EDGE, fig=fig, z=1)
        x0, _, x1, _ = r
        number_disc(ax, reg, fig, x0 + 0.016, TITLE_Y, col + 1, DISC if active else LINE_GRAY)
        reg.text(ax, x0 + 0.031, TITLE_Y, TITLES[col], ha="left", va="center_baseline", fontsize=TITLE_PT, color=INK)
        if col < 4:
            ym = 0.5 * (Y0 + Y1) - 0.04
            big_arrow(ax, reg, fig, (x1 + 0.004, ym), (x1 + GAP - 0.004, ym),
                      colour=DARK_GRAY if reveal[col + 1] > 0.5 else LINE_GRAY, lw=3.0, ms=18)
        if col > 0:  # faint separators keep the three variants aligned with the method labels
            for yb in (0.5 * (BANDS[0][0] + BANDS[1][1]), 0.5 * (BANDS[1][0] + BANDS[2][1])):
                ax.plot([x0 + 0.010, x1 - 0.010], [yb, yb], color=LINE_GRAY, lw=0.9, alpha=max(a, 0.35), zorder=2)
        if a < 0.01:
            continue
        if col == 0:
            cr = (x0 + 0.012, BANDS[2][0] + 0.06, x1 - 0.012, BANDS[0][1] - 0.06)
            sub = sub_axes(fig, cr); points(sub, fit(sub, fig, cr, data["view"])[0], labels, alpha=a, size=80)
            reg.text(ax, 0.5 * (x0 + x1), BANDS[2][0] + 0.010, "same input for all three", ha="center", va="bottom",
                     fontsize=FONT_SIZES["micro"], color=INK)
            continue
        for b, key in enumerate(("pca", "tsne", "umap")):
            cr = band_rect(col, b); sub = sub_axes(fig, cr)
            if col == 1:
                Y, s = fit(sub, fig, cr, data["view"], centre=(b > 0), pad=1.08)
                if b == 0:
                    pr = data["pcarow"]; C = np.cov((pr["Xc"] @ pr["view"]).T) / s ** 2; ev, V = np.linalg.eigh(C)
                    points(sub, Y, labels, alpha=a * 0.7, size=26)
                    sub.add_patch(Ellipse((0, 0), 3.2 * np.sqrt(ev[1]), 3.2 * np.sqrt(ev[0]),
                                          angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])), fc="none", ec=DARK_GRAY,
                                          lw=1.8, ls=(0, (4, 2)), alpha=a, zorder=5))
                else:
                    edges(sub, Y, data[key]["edges"], DARK_GRAY, a, top=48 if key == "tsne" else None)
                    points(sub, Y, labels, alpha=a, size=26)
            elif col == 2:
                sub.set_xlim(0, 1); sub.set_ylim(0, 1)
                reg.text(sub, 0.5, 0.5, OBJECTIVE[b], ha="center", va="center", fontsize=FONT_SIZES["body"], color=INK)
            elif col == 3:
                if b == 0:
                    scree(sub, data["pcarow"]["var_ratio"], u, a)
                else:
                    loss_curve(sub, data[key]["loss"], u, a)
            else:
                if b == 0:
                    Y = pca_proj(data, u)
                else:
                    traj = data[key]["traj"]; Y = traj[int(round(u * u * (len(traj) - 1)))]
                points(sub, fit(sub, fig, cr, Y, pad=1.10)[0], labels, alpha=a, size=34)
    return []


def static(data):
    return save_static(lambda fig, reg: draw(fig, reg, DURATION, data), ROOT / "figures" / f"{STEM}.png", SIZE_IN)


def video(data, frame_dump=None):
    return render_video_canvas(stem=STEM, size_in=SIZE_IN, duration_seconds=DURATION,
                               draw_frame=lambda fig, t, _i, reg: draw(fig, reg, t, data),
                               qa_directory=QA_ROOT / "_qa", representative_times=REPRESENTATIVE_TIMES, frame_dump=frame_dump)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--frame-dump", type=Path, help="also save every 2nd frame (review only)"); args = ap.parse_args()
    data = load(); print(static(data))
    if not args.static_only:
        print(video(data, args.frame_dump))


if __name__ == "__main__":
    main()
