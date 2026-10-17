"""08b - generic dimensionality-reduction pipeline, one row of five stages (static + video).

Deliberately method-agnostic: no method names and no method-specific symbols. The presenter adds per-method text
in PowerPoint OUTSIDE the cards, so each card is only as tall as its title and picture (no reserved note area).
  1 Data        point cloud with a fan of many axes (high-dimensional)
  2 Relations   pairwise neighbor graph between the points
  3 Objective   high-dim relation matrix vs low-dim relation matrix, matched (the low-dim one converges in the video)
  4 Optimize    points move along their trajectories while an error curve goes down
  5 Low-D map   the final low-dimensional (here 2-D) layout with clusters
Toy data: the seeded 24-point, 10-D, three-cluster set from scripts/build_box/generate_tsne_umap_toy.py
(product/data/08_tsne_umap_toy.json); one recorded neighbor-embedding run provides the trajectory and the error curve.
Size 338 x 72 mm, text in true points (>= 14 pt); static 300 dpi; video 1920 px wide, 24 fps, every-frame QA.
Layout is in millimeters converted to figure fractions. Every picture is framed by frame_to(): the data extent of
everything drawn in a card (points, trails, ellipses, axis arrows) is fitted with equal aspect inside the card's
picture box with a fixed margin, so nothing can reach a card border in any frame.
Style: light-gray cards, dark-gray borders, dark-gray numbered discs and arrows, black text; color only on points,
lines and curves.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from explainer_common import (
    CARD_EDGE, CARD_FILL, CLUSTER, DARK_GRAY, DISC, FONT_SIZES, IDLE_EDGE, IDLE_FILL, INK, LINE_GRAY, MID_GRAY, MM,
    TEAL, WHITE, big_arrow, canvas, card, mix_hex, number_disc, render_video_canvas, save_static, smoothstep, sub_axes,
)
from render_08_tsne_umap import edges, load, points

ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "08b_dimred_overview"
QA_ROOT = ROOT / "qa" / STEM
W_MM, H_MM = 338.0, 72.0
SIZE_IN = (W_MM * MM, H_MM * MM)
REVEAL_T = (0.0, 1.2, 2.4, 3.6, 4.8)
OPT_T, OPT_SECONDS = 3.6, 5.5
DURATION = 10.6
REPRESENTATIVE_TIMES = (0.5, 1.7, 2.9, 4.0, 5.5, 7.0, 8.5, 10.5)

TITLES = ("Data", "Relations", "Objective", "Optimize", "Low-D map")
TITLE_PT = 18.0
# millimeter layout
EDGE_MM, GAP_MM = 4.0, 10.0               # outer margin, gap between cards (holds the arrow)
CARD_Y0_MM, CARD_Y1_MM = 1.5, H_MM - 1.5
TITLE_MM = CARD_Y1_MM - 6.5               # title / disc center line
PIC_PAD_MM = 3.0                          # picture box inset from the card border (left/right/bottom)
PIC_TOP_MM = CARD_Y1_MM - 13.0            # picture box top (below the title row)
CARD_W_MM = (W_MM - 2 * EDGE_MM - 4 * GAP_MM) / 5
FRAME_MARGIN_MM = 2.5                     # extra inset of drawn content inside the picture box


def fx(mm): return mm / W_MM
def fy(mm): return mm / H_MM


def rect(col):
    x0 = EDGE_MM + col * (CARD_W_MM + GAP_MM)
    return fx(x0), fy(CARD_Y0_MM), fx(x0 + CARD_W_MM), fy(CARD_Y1_MM)


def pic_rect(col):
    x0 = EDGE_MM + col * (CARD_W_MM + GAP_MM)
    return fx(x0 + PIC_PAD_MM), fy(CARD_Y0_MM + PIC_PAD_MM), fx(x0 + CARD_W_MM - PIC_PAD_MM), fy(PIC_TOP_MM)


def frame_to(ax, r, lo, hi, margin_mm=FRAME_MARGIN_MM):
    """Equal-aspect limits so the box [lo, hi] sits centered inside rect r with margin_mm all round."""
    w_mm = (r[2] - r[0]) * W_MM; h_mm = (r[3] - r[1]) * H_MM
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); span = np.maximum(hi - lo, 1e-9); c = 0.5 * (lo + hi)
    k = min((w_mm - 2 * margin_mm) / span[0], (h_mm - 2 * margin_mm) / span[1])   # mm per data unit
    ax.set_xlim(c[0] - 0.5 * w_mm / k, c[0] + 0.5 * w_mm / k); ax.set_ylim(c[1] - 0.5 * h_mm / k, c[1] + 0.5 * h_mm / k)
    return k


def affinity(Y, order):
    D = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    s = np.median(D[D > 0])
    A = np.exp(-D / s); np.fill_diagonal(A, 0.0)
    A = A[np.ix_(order, order)]
    return A / A.max()


REL_CMAP = LinearSegmentedColormap.from_list("rel", [WHITE, mix_hex(TEAL, WHITE, 0.55), TEAL])


def matrix(ax, A, alpha):
    ax.imshow(A, cmap=REL_CMAP, vmin=0, vmax=1, alpha=alpha, interpolation="nearest", zorder=2)
    n = A.shape[0]
    ax.plot([-0.5, n - 0.5, n - 0.5, -0.5, -0.5], [-0.5, -0.5, n - 0.5, n - 0.5, -0.5], color=DARK_GRAY, lw=1.2, alpha=alpha, zorder=3)
    ax.set_xlim(-0.8, n - 0.2); ax.set_ylim(n - 0.2, -0.8)


def traj_at(data, u):
    tr = data["tsne"]["traj"]; f = u * u * (len(tr) - 1); i = int(f); j = min(i + 1, len(tr) - 1); w = f - i
    return tr[i] * (1 - w) + tr[j] * w


def ellipses(Y, labels):
    out = []
    for c_ in range(3):
        P = Y[labels == c_]; cen = P.mean(0); rad = np.abs(P - cen).max(0) * 1.35 + 0.12 * np.ptp(Y, 0).max() / 2
        out.append((c_, cen, rad))
    return out


def draw(fig, reg, t, data):
    ax = canvas(fig); labels = data["labels"]; order = np.argsort(labels, kind="stable")
    reveal = [smoothstep((t - s) / 0.5) for s in REVEAL_T]
    u = smoothstep((t - OPT_T) / OPT_SECONDS) if t > OPT_T else 0.0
    X = data["pcarow"]["Xc"]; V = data["view"]; V = V - 0.5 * (V.max(0) + V.min(0))
    tr = data["tsne"]["traj"]; final = tr[-1]
    for col in range(5):
        a = reveal[col]; r = rect(col); active = a > 0.5
        card(ax, r, fill=CARD_FILL if active else IDLE_FILL, edge=CARD_EDGE if active else IDLE_EDGE, fig=fig, z=1)
        x0, _, x1, _ = r
        number_disc(ax, reg, fig, x0 + fx(6.5), fy(TITLE_MM), col + 1, DISC if active else LINE_GRAY)
        reg.text(ax, x0 + fx(12.5), fy(TITLE_MM), TITLES[col], ha="left", va="center_baseline", fontsize=TITLE_PT, color=INK)
        pr = pic_rect(col)
        if col < 4:
            ym = 0.5 * (pr[1] + pr[3])
            big_arrow(ax, reg, fig, (x1 + fx(1.5), ym), (x1 + fx(GAP_MM - 1.5), ym),
                      color=DARK_GRAY if reveal[col + 1] > 0.5 else LINE_GRAY, lw=3.0, ms=18)
        if a < 0.01:
            continue
        if col == 0:
            sub = sub_axes(fig, pr); R = np.abs(V).max() * 1.15
            frame_to(sub, pr, (-R, -R), (R, R))
            for ang in np.linspace(0, 2 * np.pi, 9, endpoint=False) + 0.35:   # many axes: a star of directions
                d = R * np.array([np.cos(ang), np.sin(ang)])
                sub.annotate("", xy=d, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=mix_hex(LINE_GRAY, MID_GRAY, 0.3),
                             mutation_scale=9, shrinkA=0, shrinkB=0, alpha=a), zorder=1)
            points(sub, V, labels, alpha=a, size=55)
        elif col == 1:
            sub = sub_axes(fig, pr); frame_to(sub, pr, V.min(0), V.max(0))
            edges(sub, V, data["umap"]["edges"], DARK_GRAY, a)
            points(sub, V, labels, alpha=a, size=55)
        elif col == 2:
            w_mm = (pr[2] - pr[0]) * W_MM; h_mm = (pr[3] - pr[1]) * H_MM
            lab_mm = 7.0; m_mm = min(0.42 * w_mm, h_mm - lab_mm - 2.0)          # matrix side
            yb = pr[1] + fy(lab_mm + 0.5 * (h_mm - lab_mm - m_mm)); yt = yb + fy(m_mm)
            left = (pr[0] + fx(1.0), yb, pr[0] + fx(1.0 + m_mm), yt)
            right = (pr[2] - fx(1.0 + m_mm), yb, pr[2] - fx(1.0), yt)
            for rr, A in ((left, affinity(X, order)), (right, affinity(traj_at(data, u) if u > 0 else V, order))):
                matrix(sub_axes(fig, rr), A, a)
            xm = 0.5 * (pr[0] + pr[2]); ymid = 0.5 * (yb + yt)
            big_arrow(ax, reg, fig, (xm - fx(3.5), ymid + fy(3.0)), (xm + fx(3.5), ymid + fy(3.0)), color=DARK_GRAY, lw=2.0, ms=14, alpha=a, z=6)
            big_arrow(ax, reg, fig, (xm + fx(3.5), ymid - fy(3.0)), (xm - fx(3.5), ymid - fy(3.0)), color=DARK_GRAY, lw=2.0, ms=14, alpha=a, z=6)
            for rr, lab in ((left, "high-D"), (right, "low-D")):
                reg.text(ax, 0.5 * (rr[0] + rr[2]), yb - fy(1.2), lab, ha="center", va="top", fontsize=FONT_SIZES["micro"], color=INK)
        elif col == 3:
            h = pr[3] - pr[1]
            top = (pr[0], pr[1] + 0.40 * h, pr[2], pr[3]); bot = (pr[0] + fx(2.0), pr[1], pr[2] - fx(2.0), pr[1] + 0.34 * h)
            st = sub_axes(fig, top); n = len(tr); m = max(1, int(round(u * u * (n - 1))) + 1)
            Yall = tr.reshape(-1, 2)
            frame_to(st, top, Yall.min(0), Yall.max(0), margin_mm=2.0)
            for i in range(tr.shape[1]):
                st.plot(tr[:m, i, 0], tr[:m, i, 1], color=mix_hex(CLUSTER[labels[i]], WHITE, 0.35), lw=1.1, alpha=a, zorder=2)
            points(st, traj_at(data, u), labels, alpha=a, size=30)
            sb = sub_axes(fig, bot); loss = data["tsne"]["loss"]; k = len(loss)
            x = np.sqrt(np.linspace(0, 1, k)); y = (loss - loss.min()) / (loss.max() - loss.min())
            sb.set_xlim(-0.08, 1.08); sb.set_ylim(-0.25, 1.2)
            sb.plot([0, 1.03], [-0.08, -0.08], color=LINE_GRAY, lw=1.0, alpha=a); sb.plot([0, 0], [-0.08, 1.1], color=LINE_GRAY, lw=1.0, alpha=a)
            sb.plot(x, y, color=LINE_GRAY, lw=1.6, alpha=a); mm = max(1, int(round(u * u * (k - 1))) + 1)
            sb.plot(x[:mm], y[:mm], color=TEAL, lw=2.8, alpha=a)
            sb.scatter([x[mm - 1]], [y[mm - 1]], s=60, color=TEAL, edgecolors=WHITE, linewidths=1.1, zorder=5, alpha=a)
        else:
            Y = traj_at(data, u) if t < DURATION else final
            show = smoothstep((u - 0.75) / 0.25) if t < DURATION else 1.0
            # frame on the union of the current points and the FINAL ellipses, so the view is stable and never clips
            ell_now = ellipses(Y, labels); ext = ellipses(final, labels) + (ell_now if show > 0 else [])
            lo = np.min([Y.min(0)] + [cen - rad for _, cen, rad in ext], axis=0)
            hi = np.max([Y.max(0)] + [cen + rad for _, cen, rad in ext], axis=0)
            sub = sub_axes(fig, pr); frame_to(sub, pr, lo, hi)
            for c_, cen, rad in ell_now:
                sub.add_patch(Ellipse(cen, 2 * rad[0], 2 * rad[1], fc=mix_hex(CLUSTER[c_], WHITE, 0.82), ec=CLUSTER[c_], lw=1.2,
                                      alpha=a * show, zorder=1))
            points(sub, Y, labels, alpha=a, size=55)
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
