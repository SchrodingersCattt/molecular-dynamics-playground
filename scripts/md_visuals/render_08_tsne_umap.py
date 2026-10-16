"""08 - PCA, t-SNE and UMAP as three parallel five-stage pipelines on one toy data set (static + video).

Data: product/data/08_tsne_umap_toy.json from scripts/build_box/generate_tsne_umap_toy.py (seed in SEED):
24 points in 10-D (three clusters), an oblique 2-D linear view, the PCA eigen-decomposition, exact t-SNE
(perplexity 6) and a full-batch UMAP-style cross-entropy run (k = 6, a = 1.577, b = 0.895); the two
non-linear runs start from the PCA projection.
Rows: PCA, t-SNE, UMAP. Stage 1 is the same 10-D cloud (an oblique 2-D linear view) in every row; stage 2 of t-SNE/UMAP draws the graph on it.
  PCA:   points -> centre + covariance -> principal axes PC1/PC2 -> variance explained -> linear projection
  t-SNE: points -> neighbours p_ij -> Student-t q_ij -> KL(P||Q) -> embedding
  UMAP:  points -> fuzzy kNN graph -> low-dim q_ij -> cross-entropy -> embedding
Video: rows play in order; stages light up; PCA rotates the view onto PC1/PC2 while the captured variance
rises, t-SNE/UMAP move the points along the recorded optimisation while the loss dot runs down.
Size: 338 x 125 mm (text in true points, >= 14 pt). Static 300 dpi; video 1920 px wide, 24 fps.
Style: grey cards (dark-grey border, light-grey fill), dark-grey step discs, black text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse

from explainer_common import (
    CARD_EDGE, CARD_FILL, CLUSTER, DARK_GRAY, DISC, FONT_SIZES, IDLE_EDGE, IDLE_FILL, INK, LINE_GRAY, MID_GRAY, MM,
    TEAL, ORANGE, WHITE, big_arrow, canvas, card, number_disc, render_video_canvas, save_static, smoothstep, sub_axes,
)

ROOT = Path(__file__).resolve().parents[2] / "product"
STEM = "08_tsne_umap_principle"
DATA = ROOT / "data" / "08_tsne_umap_toy.json"
QA_ROOT = ROOT / "qa" / STEM
SIZE_IN = (338 * MM, 125 * MM)
ROW_SECONDS = (6.0, 7.5, 7.5)
ROW_START = (0.0, 6.0, 13.5)
DURATION = 21.0 + 1.5
STAGE_T = (0.0, 0.8, 1.8, 2.8)       # stage 1, 2, 3 reveal; stage 4/5 from 2.8 s
OPT_SECONDS = (2.8, 4.0, 4.0)
REPRESENTATIVE_TIMES = (0.4, 1.3, 2.3, 3.5, 5.0, 6.4, 8.5, 10.5, 13.0, 14.0, 16.5, 19.5, 22.4)

LABEL_W = 0.080
ROWS = ((0.676, 0.990), (0.343, 0.657), (0.010, 0.324))
X_START, X_END, GAP = 0.086, 0.995, 0.028
PANEL_W = (X_END - X_START - 4 * GAP) / 5
TITLES = (
    ("10-D points", "Centre, covariance", "Principal axes", "Variance explained", "Projection"),
    ("10-D points", "Neighbours $p_{ij}$", "Student-$t$  $q_{ij}$", "KL($P\\,\\|\\,Q$)", "Embedding"),
    ("10-D points", "Fuzzy $k$NN graph", "Low-dim  $q_{ij}$", "Cross-entropy", "Embedding"),
)
ROW_NAMES = ("PCA", "t-SNE", "UMAP")
KEYS = ("pca", "tsne", "umap")
TITLE_PT = 15.0  # stage titles; >= 14 pt at 338 mm


def load():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    out = {"labels": np.array(d["labels"]), "pca": np.array(d["pca"])}
    r = d["pca_row"]
    out["pcarow"] = {k: np.array(v) for k, v in r.items()}
    out["view"] = out["pcarow"]["Xc"] @ out["pcarow"]["view"]
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
    return x0 + 0.010, y0 + 0.022, x1 - 0.010, y1 - 0.092


def fit(ax, fig, rect, Y, pad=1.12, centre=True):
    w = (rect[2] - rect[0]) * fig.get_figwidth(); h = (rect[3] - rect[1]) * fig.get_figheight(); r = w / h
    if centre:
        Y = Y - 0.5 * (Y.max(0) + Y.min(0))
    s = max(np.abs(Y[:, 0]).max() / r, np.abs(Y[:, 1]).max()) or 1.0
    ax.set_xlim(-pad * r, pad * r); ax.set_ylim(-pad, pad)
    return Y / s, s


def points(ax, Y, labels, alpha=1.0, size=60):
    ax.scatter(Y[:, 0], Y[:, 1], s=size, c=[CLUSTER[i] for i in labels], edgecolors=WHITE, linewidths=0.9, alpha=alpha, zorder=4)


def edges(ax, Y, E, colour, alpha, top=None):
    w = E[:, 2]
    if top is not None:
        E = E[np.argsort(w)[::-1][:top]]; w = E[:, 2]
    wn = w / w.max()
    for (i, j, _), v in zip(E, wn):
        i, j = int(i), int(j)
        ax.plot([Y[i, 0], Y[j, 0]], [Y[i, 1], Y[j, 1]], color=colour, lw=0.5 + 2.2 * v, alpha=alpha * (0.25 + 0.6 * v),
                zorder=2, solid_capstyle="round")


def axes_l(ax, x1, y1, alpha, x0=0.0, y0=0.0):
    ax.plot([x0, x1], [y0, y0], color=LINE_GRAY, lw=1.0, alpha=alpha)
    ax.plot([x0, x0], [y0, y1], color=LINE_GRAY, lw=1.0, alpha=alpha)


def kernel(ax, reg, row, alpha):
    d = np.linspace(0, 4, 200)
    ax.set_xlim(-0.15, 4.2); ax.set_ylim(-0.12, 1.15); axes_l(ax, 4.1, 1.1, alpha)
    if row == 1:
        ax.plot(d, np.exp(-d ** 2), color=MID_GRAY, lw=2.2, ls=(0, (3, 2)), alpha=alpha)
        ax.plot(d, 1 / (1 + d ** 2), color=TEAL, lw=2.8, alpha=alpha)
        ax.fill_between(d, np.exp(-d ** 2), 1 / (1 + d ** 2), where=d > 1.0, color=TEAL, alpha=0.14 * alpha, lw=0)
        reg.text(ax, 1.45, 0.90, "heavy tail", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)
        reg.text(ax, 1.45, 0.58, "Gaussian", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)
    else:
        q = 1 / (1 + 1.577 * d ** (2 * 0.895))
        ax.plot(d, q, color=TEAL, lw=2.8, alpha=alpha)
        ax.plot(d, 1 - q, color=ORANGE, lw=2.8, ls=(0, (3, 2)), alpha=alpha)
        reg.text(ax, 1.55, 0.60, "others repel", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)
        reg.text(ax, 1.55, 0.32, "edges attract", ha="left", va="center", fontsize=FONT_SIZES["micro"], color=INK)


def loss_curve(ax, loss, u, alpha):
    n = len(loss); x = np.sqrt(np.linspace(0, 1, n)); y = (loss - loss.min()) / (loss.max() - loss.min())  # sqrt-spaced iterations
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.14, 1.12); axes_l(ax, 1.04, 1.08, alpha, y0=-0.06)
    ax.plot(x, y, color=LINE_GRAY, lw=1.8, alpha=alpha)
    m = max(1, int(round(u * u * (n - 1))) + 1)
    ax.plot(x[:m], y[:m], color=TEAL, lw=3.0, alpha=alpha)
    ax.scatter([x[m - 1]], [y[m - 1]], s=70, color=TEAL, edgecolors=WHITE, linewidths=1.2, zorder=5, alpha=alpha)


def scree(ax, ratio, u, alpha):
    k = len(ratio); xs = np.arange(k)
    ax.set_xlim(-0.8, k - 0.2); ax.set_ylim(-0.08, 1.0); axes_l(ax, k - 0.4, 0.95, alpha, x0=-0.6, y0=0.0)
    show = ratio / ratio.max() * 0.9
    for i in xs:
        col = TEAL if i < 2 else LINE_GRAY
        ax.add_patch(__import__("matplotlib").patches.Rectangle((i - 0.32, 0), 0.64, show[i] * min(1, max(0, u * 3 - i * 0.2)),
                     fc=col, ec="none", alpha=alpha, zorder=3))
    ax.plot(xs, show, color=DARK_GRAY, lw=1.6, alpha=alpha * smoothstep(u * 2), zorder=4, marker="o", ms=3.5)


def pca_proj(data, u):
    """Rotate the random view basis onto PC1/PC2 (orthonormalised linear blend)."""
    r = data["pcarow"]; B = (1 - u) * r["view"] + u * r["evecs2"]
    q, rr = np.linalg.qr(B); q = q * np.sign(np.diag(rr))
    return r["Xc"] @ q


def draw(fig, reg, t, data):
    ax = canvas(fig); labels = data["labels"]
    for row, key in enumerate(KEYS):
        lt = t - ROW_START[row]
        y0, y1 = ROWS[row]
        reg.text(ax, 0.010, 0.5 * (y0 + y1), ROW_NAMES[row], ha="left", va="center", fontsize=FONT_SIZES["panel_title"], color=INK)
        reveal = [smoothstep((lt - s) / 0.5) for s in STAGE_T] + [smoothstep((lt - STAGE_T[3]) / 0.5)]
        u = smoothstep((lt - STAGE_T[3]) / OPT_SECONDS[row]) if lt > STAGE_T[3] else 0.0
        for col in range(5):
            a = reveal[col]; rect = panel_rect(row, col); active = a > 0.5
            card(ax, rect, fill=CARD_FILL if active else IDLE_FILL, edge=CARD_EDGE if active else IDLE_EDGE, fig=fig, z=1)
            x0, _, x1, yt = rect
            number_disc(ax, reg, fig, x0 + 0.016, yt - 0.046, col + 1, DISC if active else LINE_GRAY)
            reg.text(ax, x0 + 0.031, yt - 0.046, TITLES[row][col], ha="left", va="center_baseline",
                     fontsize=TITLE_PT, color=INK)
            if col < 4:
                ym = 0.5 * (rect[1] + rect[3]) - 0.025
                big_arrow(ax, reg, fig, (x1 + 0.004, ym), (x1 + GAP - 0.004, ym),
                          colour=DARK_GRAY if reveal[col + 1] > 0.5 else LINE_GRAY, lw=3.0, ms=18)
            if a < 0.01:
                continue
            cr = content_rect(row, col)
            if row == 2 and col == 4:
                cr = (cr[0], cr[1] + 0.050, cr[2], cr[3])
            sub = sub_axes(fig, cr)
            if col == 0:
                points(sub, fit(sub, fig, cr, data["view"])[0], labels, alpha=a)
            elif row == 0:
                _pca_panel(sub, reg, fig, cr, data, col, a, u)
            elif col == 1:
                Y = fit(sub, fig, cr, data["view"])[0]
                edges(sub, Y, data[key]["edges"], DARK_GRAY, a, top=48 if key == "tsne" else None)
                points(sub, Y, labels, alpha=a)
            elif col == 2:
                kernel(sub, reg, row, a)
            elif col == 3:
                loss_curve(sub, data[key]["loss"], u, a)
            else:
                traj = data[key]["traj"]; idx = int(round(u * u * (len(traj) - 1)))
                points(sub, fit(sub, fig, cr, traj[idx])[0], labels, alpha=a)
                if row == 2:
                    reg.text(ax, 0.5 * (x0 + x1), rect[1] + 0.016, "more global layout", ha="center", va="bottom",
                             fontsize=FONT_SIZES["micro"], color=INK)
    return []


def _pca_panel(sub, reg, fig, cr, data, col, a, u):
    labels = data["labels"]; r = data["pcarow"]
    if col in (1, 2):
        Yv, s = fit(sub, fig, cr, data["view"], centre=False)  # data are centred: origin = mean
        points(sub, Yv, labels, alpha=a * (0.55 if col == 2 else 1.0))
        C = np.cov((r["Xc"] @ r["view"]).T) / s ** 2; ev, V = np.linalg.eigh(C)
        if col == 1:
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            sub.add_patch(Ellipse((0, 0), 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), angle=ang, fc="none", ec=DARK_GRAY,
                                  lw=1.8, ls=(0, (4, 2)), alpha=a, zorder=5))
            sub.plot([0], [0], marker="+", ms=14, mew=2.4, color=INK, alpha=a, zorder=6)
        else:
            P = r["evecs2"].T @ r["view"]  # PC1/PC2 directions seen in the random view
            for i, (vec, lab) in enumerate(zip(P, ("PC1", "PC2"))):
                L = 2.0 * np.sqrt(r["var_ratio"][i] / r["var_ratio"][0])
                v = vec / (np.linalg.norm(vec) + 1e-12) * L
                xl, yl = sub.get_xlim()[1], sub.get_ylim()[1]
                v = v * min(1.0, 0.62 / max(abs(v[0]) / xl, abs(v[1]) / yl))  # keep arrow + label inside the card
                sub.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="-|>", lw=2.6, color=TEAL if i == 0 else ORANGE, mutation_scale=16, shrinkA=0, shrinkB=0),
                             zorder=7)
                reg.text(sub, v[0] * 1.08, v[1] * 1.08 + (0.12 if v[1] >= 0 else -0.12), lab, ha="center", va="center",
                         fontsize=FONT_SIZES["micro"], color=INK, zorder=8)
    elif col == 3:
        scree(sub, r["var_ratio"], max(u, 0.0) if u > 0 else 0.0, a)
    else:
        points(sub, fit(sub, fig, cr, pca_proj(data, u))[0], labels, alpha=a)


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
