"""Toy data for the 08 t-SNE / UMAP explainer: exact t-SNE and a full-batch UMAP-style optimization.

24 points in 10-D (also a random 2-D view and the PCA eigen-decomposition for the PCA row) (three clusters of 8; A and B close, C far) plus one large non-informative
dimension, so the PCA projection mixes the clusters. Both methods start from that PCA layout.
Writes product/data/08_tsne_umap_toy.json (positions per recorded iteration and the loss).
Deterministic: numpy seed 20261032 (see SEED).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SEED = 20261032  # picked from a 40-seed scan (20261000-39): clean clusters in both embeddings, A/B overlapping in the PCA view
OUT = Path(__file__).resolve().parents[2] / "product" / "data" / "08_tsne_umap_toy.json"


def make_points(rng):
    """Three clusters on orthogonal axes (C twice as far), plus broad within-cluster spread in two
    shared dimensions, so the 2-D linear (PCA) view of the 10-D cloud overlaps A and B."""
    centers = np.zeros((3, 10)); centers[0, 0] = 4.0; centers[1, 1] = 4.0; centers[2, 2] = 8.0
    scale = np.full(10, 0.35); scale[5] = scale[6] = 2.4
    X = np.vstack([centers[k] + scale * rng.standard_normal((8, 10)) for k in range(3)])
    return X, np.repeat(np.arange(3), 8)


def pca2(X):
    Xc = X - X.mean(0); _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ vt[:2].T; return Y / Y.std()


def tsne(X, Y0, perplexity=6.0, iters=600, lr=6.0, record=4):
    n = len(X); D = ((X[:, None] - X[None]) ** 2).sum(-1); P = np.zeros((n, n))
    for i in range(n):
        lo, hi = 1e-6, 1e4
        for _ in range(80):
            beta = np.sqrt(lo * hi); p = np.exp(-np.maximum(D[i] - D[i][D[i] > 0].min(), 0) * beta); p[i] = 0; p /= p.sum()
            H = -(p[p > 0] * np.log(p[p > 0])).sum()
            if H > np.log(perplexity): lo = beta
            else: hi = beta
        P[i] = p
    P = (P + P.T) / (2 * n); P = np.maximum(P, 1e-12)
    Y = Y0.copy() * 2.0; vel = np.zeros_like(Y); traj, loss = [], []
    for it in range(iters + 1):
        num = 1.0 / (1.0 + ((Y[:, None] - Y[None]) ** 2).sum(-1)); np.fill_diagonal(num, 0); Q = np.maximum(num / num.sum(), 1e-12)
        if it % record == 0:
            traj.append(Y.tolist()); loss.append(float((P * np.log(P / Q)).sum()))
        ex = 1.0  # no early exaggeration: keeps the recorded KL monotone for the teaching curve
        g = 4 * (((ex * P - Q) * num)[:, :, None] * (Y[:, None] - Y[None])).sum(1)
        vel = 0.6 * vel - lr * g; Y = Y + vel; Y -= Y.mean(0)
    return P, traj, loss


def umap_like(X, Y0, k=6, iters=600, lr=0.05, record=4, a=1.577, b=0.895):
    n = len(X); D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); W = np.zeros((n, n))
    for i in range(n):
        nn = np.argsort(D[i])[1:k + 1]; rho = D[i, nn[0]]; lo, hi = 1e-3, 1e3
        for _ in range(80):
            s = np.sqrt(lo * hi); v = np.exp(-(D[i, nn] - rho) / s).sum()
            if v > np.log2(k): hi = s
            else: lo = s
        W[i, nn] = np.exp(-(D[i, nn] - rho) / s)
    W = W + W.T - W * W.T; np.fill_diagonal(W, 0)
    Y = Y0.copy() * 3.0; traj, loss = [], []
    for it in range(iters + 1):
        d2 = ((Y[:, None] - Y[None]) ** 2).sum(-1) + 1e-3; q = 1.0 / (1.0 + a * d2 ** b); np.fill_diagonal(q, 0.5)
        if it % record == 0:
            m = ~np.eye(n, dtype=bool); qq = np.clip(q[m], 1e-6, 1 - 1e-6); w = W[m]
            traj.append(Y.tolist()); loss.append(float(-(w * np.log(qq) + (1 - w) * np.log(1 - qq)).sum() / 2))
        gq = a * b * d2 ** (b - 1) * q
        g = -(W * (-gq) + (1 - W) * (b / d2 - gq)); np.fill_diagonal(g, 0)
        grad = np.clip(2 * (g[:, :, None] * (Y[:, None] - Y[None])).sum(1), -4, 4)
        Y = Y - lr * grad * (1 - 0.9 * it / (iters + 1)); Y -= Y.mean(0)
    return W, traj, loss


def main():
    rng = np.random.default_rng(SEED); X, lab = make_points(rng); Y0 = pca2(X)
    P, tt, tl = tsne(X, Y0); W, ut, ul = umap_like(X, Y0)
    pos = lambda arr: [[i, j, round(float(arr[i, j]), 5)] for i in range(24) for j in range(i + 1, 24) if arr[i, j] > 0]
    Xc = X - X.mean(0); evals, evecs = np.linalg.eigh(np.cov(Xc.T)); order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evecs = evecs * np.sign(evecs[np.abs(evecs).argmax(0), np.arange(evecs.shape[1])])  # fixed sign convention
    R = np.random.default_rng(SEED + 1).standard_normal((X.shape[1], 2))
    view, _ = np.linalg.qr(np.cos(1.0) * evecs[:, :2] + np.sin(1.0) * R / np.linalg.norm(R, axis=0))  # an oblique 2-D linear view (57 deg off the PC plane)
    pca_row = {"Xc": Xc.round(6).tolist(), "view": view.tolist(), "evecs2": evecs[:, :2].tolist(),
               "var_ratio": (evals / evals.sum()).tolist()}
    out = {"seed": SEED, "labels": lab.tolist(), "pca": Y0.tolist(), "pca_row": pca_row, "tsne": {"P": pos(P), "traj": tt, "kl": tl},
           "umap": {"W": pos(W), "traj": ut, "ce": ul, "a": 1.577, "b": 0.895, "k": 6}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out), encoding="utf-8")
    print(OUT, "KL", round(tl[0], 3), "->", round(tl[-1], 3), "CE", round(ul[0], 2), "->", round(ul[-1], 2))


if __name__ == "__main__":
    main()
