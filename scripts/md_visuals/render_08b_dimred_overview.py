"""08b - generic dimensionality-reduction pipeline, one row of five stages (static + video).

Deliberately method-agnostic: no method names and no method-specific symbols. The presenter adds per-method notes
in PowerPoint, so the bottom of every card is left empty for that text.
  1 Data        point cloud with a fan of many axes (high-dimensional)
  2 Relations   pairwise neighbour graph between the points
  3 Objective   high-dim relation matrix vs low-dim relation matrix, matched (the low-dim one converges in the video)
  4 Optimise    points move along their trajectories while an error curve goes down
  5 Embedding   the final 2-D layout with clusters
Toy data: the seeded 24-point, 10-D, three-cluster set from scripts/build_box/generate_tsne_umap_toy.py
(product/data/08_tsne_umap_toy.json); one recorded neighbour-embedding run provides the trajectory and the error curve.
Size 338 x 125 mm, text in true points (>= 14 pt); static 300 dpi; video 1920 px wide, 24 fps, every-frame QA.
Style: light-grey cards, dark-grey borders, dark-grey numbered discs and arrows, black text; colour only on points,
lines and curves.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse

from explainer_common import (
    CARD_EDGE, CARD_FILL, CLUSTER, DARK_GRAY, DISC, FONT_SIZES, IDLE_EDGE, IDLE_FILL, INK, LINE_GRAY, MID_GRAY, MM,
    TEAL, WHITE, big_arrow, canvas, card, mix_hex, number_disc, render_video_canvas, save_static, smoothstep, sub_axes,
)
from render_08_tsne_umap import edges, fit, load, points

ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "08b_dimred_overview"
QA_ROOT = ROOT / "qa" / STEM
SIZE_IN = (338 * MM, 125 * MM)
REVEAL_T = (0.0, 1.2, 2.4, 3.6, 4.8)
OPT_T, OPT_SECONDS = 3.6, 5.5
DURATION = 10.6
REPRESENTATIVE_TIMES = (0.5, 1.7, 2.9, 4.0, 5.5, 7.0, 8.5, 10.5)

X_START, X_END, GAP = 0.012, 0.988, 0.030
PANEL_W = (X_END - X_START - 4 * GAP) / 5
Y0, Y1 = 0.012, 0.988
TITLE_Y = Y1 - 0.060
VIS = (0.385, 0.865)          # visual band inside each card; below it stays empty for the presenter's notes
NOTE_GUIDE_Y = 0.345          # faint divider between visual and note area
TITLES = ("Data", "Relations", "Objective", "Optimise", "Embedding")
TITLE_PT = 18.0


def rect(col):
    x0 = X_START + col * (PANEL_W + GAP)
    return x0, Y0, x0 + PANEL_W, Y1


def vis_rect(col, pad=0.014):
    x0, _, x1, _ = rect(col)
    return x0 + pad, VIS[0], x1 - pad, VIS[1]


def affinity(Y, order):
    D = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    s = np.median(D[D > 0])
    A = np.exp(-D / s); np.fill_diagonal(A, 0.0)
    A = A[np.ix_(order, order)]
    return A / A.max()


def matrix(ax, A, alpha):
    cmap = __import__("matplotlib").colors.LinearSegmentedColormap.from_list("rel", [WHITE, mix_hex(TEAL, WHITE, 0.55), TEAL])
    ax.imshow(A, cmap=cmap, vmin=0, vmax=1, alpha=alpha, interpolation="nearest", zorder=2)
    n = A.shape[0]
    ax.plot([-0.5, n - 0.5, n - 0.5, -0.5, -0.5], [-0.5, -0.5, n - 0.5, n - 0.5, -0.5], color=DARK_GRAY, lw=1.2, alpha=alpha, zorder=3)
    ax.set_xlim(-1, n); ax.set_ylim(n, -1)


def traj_at(data, u):
    tr = data["tsne"]["traj"]; f = u * u * (len(tr) - 1); i = int(f); j = min(i + 1, len(tr) - 1); w = f - i
    return tr[i] * (1 - w) + tr[j] * w


def draw(fig, reg, t, data):
    ax = canvas(fig); labels = data["labels"]; order = np.argsort(labels, kind="stable")
    reveal = [smoothstep((t - s) / 0.5) for s in REVEAL_T]
    u = smoothstep((t - OPT_T) / OPT_SECONDS) if t > OPT_T else 0.0
    X = data["pcarow"]["Xc"]; V = data["view"]
    for col in range(5):
        a = reveal[col]; r = rect(col); active = a > 0.5
        card(ax, r, fill=CARD_FILL if active else IDLE_FILL, edge=CARD_EDGE if active else IDLE_EDGE, fig=fig, z=1)
        x0, _, x1, _ = r
        number_disc(ax, reg, fig, x0 + 0.020, TITLE_Y, col + 1, DISC if active else LINE_GRAY)
        reg.text(ax, x0 + 0.038, TITLE_Y, TITLES[col], ha="left", va="center_baseline", fontsize=TITLE_PT, color=INK)
        ax.plot([x0 + 0.014, x1 - 0.014], [NOTE_GUIDE_Y, NOTE_GUIDE_Y], color=LINE_GRAY, lw=0.9, ls=(0, (3, 3)),
                alpha=max(a, 0.4), zorder=2)
        if col < 4:
            ym = 0.5 * sum(VIS)
            big_arrow(ax, reg, fig, (x1 + 0.004, ym), (x1 + GAP - 0.004, ym),
                      colour=DARK_GRAY if reveal[col + 1] > 0.5 else LINE_GRAY, lw=3.0, ms=18)
        if a < 0.01:
            continue
        cr = vis_rect(col); sub = sub_axes(fig, cr)
        if col == 0:
            Y = fit(sub, fig, cr, V, pad=1.15)[0]
            for k, ang in enumerate(np.linspace(0, 2 * np.pi, 9, endpoint=False) + 0.35):  # many axes: a star of directions
                d = np.array([np.cos(ang) * 0.92 * sub.get_xlim()[1], np.sin(ang) * 0.92 * sub.get_ylim()[1]])
                sub.annotate("", xy=d, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=mix_hex(LINE_GRAY, MID_GRAY, 0.3),
                             mutation_scale=9, shrinkA=0, shrinkB=0, alpha=a), zorder=1)
            points(sub, Y, labels, alpha=a, size=55)
        elif col == 1:
            Y = fit(sub, fig, cr, V, pad=1.12)[0]
            edges(sub, Y, data["umap"]["edges"], DARK_GRAY, a)
            points(sub, Y, labels, alpha=a, size=55)
        elif col == 2:
            w = cr[2] - cr[0]; m = 0.41 * w
            left = (cr[0], cr[1] + 0.06, cr[0] + m, cr[3] - 0.02)
            right = (cr[2] - m, cr[1] + 0.06, cr[2], cr[3] - 0.02)
            sub.remove()
            for rr, A in ((left, affinity(X, order)), (right, affinity(traj_at(data, u) if u > 0 else V, order))):
                s2 = sub_axes(fig, rr); matrix(s2, A, a)
            xm = 0.5 * (cr[0] + cr[2]); ym = 0.5 * (cr[1] + 0.06 + cr[3] - 0.02) - 0.015
            big_arrow(ax, reg, fig, (xm - 0.009, ym + 0.03), (xm + 0.009, ym + 0.03), colour=DARK_GRAY, lw=2.0, ms=14, alpha=a, z=6)
            big_arrow(ax, reg, fig, (xm + 0.009, ym - 0.0), (xm - 0.009, ym - 0.0), colour=DARK_GRAY, lw=2.0, ms=14, alpha=a, z=6)
            for xx, lab in ((0.5 * (left[0] + left[2]), "high-D"), (0.5 * (right[0] + right[2]), "low-D")):
                reg.text(ax, xx, cr[1] + 0.040, lab, ha="center", va="top", fontsize=FONT_SIZES["micro"], color=INK)
        elif col == 3:
            h = cr[3] - cr[1]
            top = (cr[0], cr[1] + 0.42 * h, cr[2], cr[3]); bot = (cr[0] + 0.01, cr[1], cr[2] - 0.01, cr[1] + 0.36 * h)
            sub.remove(); st = sub_axes(fig, top)
            tr = data["tsne"]["traj"]; n = len(tr); m = max(1, int(round(u * u * (n - 1))) + 1)
            Yall = np.concatenate([tr[k] for k in range(0, n, max(1, n // 30))])
            _, s = fit(st, fig, top, Yall, pad=1.10, centre=False); c = 0.5 * (Yall.max(0) + Yall.min(0))
            for i in range(tr.shape[1]):
                path = (tr[:m, i, :] - c) / s
                st.plot(path[:, 0], path[:, 1], color=mix_hex(CLUSTER[labels[i]], WHITE, 0.35), lw=1.1, alpha=a, zorder=2)
            points(st, (traj_at(data, u) - c) / s, labels, alpha=a, size=34)
            sb = sub_axes(fig, bot); loss = data["tsne"]["loss"]; k = len(loss)
            x = np.sqrt(np.linspace(0, 1, k)); y = (loss - loss.min()) / (loss.max() - loss.min())
            sb.set_xlim(-0.05, 1.05); sb.set_ylim(-0.15, 1.15)
            sb.plot([0, 1.03], [-0.08, -0.08], color=LINE_GRAY, lw=1.0, alpha=a); sb.plot([0, 0], [-0.08, 1.1], color=LINE_GRAY, lw=1.0, alpha=a)
            sb.plot(x, y, color=LINE_GRAY, lw=1.6, alpha=a); mm = max(1, int(round(u * u * (k - 1))) + 1)
            sb.plot(x[:mm], y[:mm], color=TEAL, lw=2.8, alpha=a)
            sb.scatter([x[mm - 1]], [y[mm - 1]], s=60, color=TEAL, edgecolors=WHITE, linewidths=1.1, zorder=5, alpha=a)
        else:
            Y = traj_at(data, max(u, 0.0)) if t < DURATION else data["tsne"]["traj"][-1]
            Yn = fit(sub, fig, cr, Y, pad=1.18)[0]
            show = smoothstep((u - 0.75) / 0.25) if t < DURATION else 1.0
            for c_ in range(3):
                P = Yn[labels == c_]; cen = P.mean(0); rad = np.abs(P - cen).max(0) * 1.35 + 0.12
                sub.add_patch(Ellipse(cen, 2 * rad[0], 2 * rad[1], fc=mix_hex(CLUSTER[c_], WHITE, 0.82), ec=CLUSTER[c_], lw=1.2,
                                      alpha=a * show, zorder=1))
            points(sub, Yn, labels, alpha=a, size=55)
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
