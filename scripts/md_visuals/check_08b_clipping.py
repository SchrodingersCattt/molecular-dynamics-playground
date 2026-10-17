"""Every-frame clipping check for 08b_dimred_overview (run by build_08_09_explainers.py after the render).

Decodes every frame of the master video and, for each card, finds the "ink" (pixels that differ from the card fill)
below the title row. Each ink pixel's distance to the card's rounded border is the signed distance to the rounded
rectangle that render_08b draws, using that script's own layout constants. Every frame and card must keep its ink at
least LIMIT_MM inside the border. Frames are decoded into the QA folder and deleted afterwards, so nothing goes to a
system temp dir. Writes product/qa/08b_dimred_overview/_qa/clip_check.json.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import render_08b_dimred_overview as F

LIMIT_MM = 2.0
CORNER_MM = 0.016 * F.W_MM          # explainer_common.card rounding_size (x-fraction) in mm


def check(master: Path, qa_dir: Path) -> dict:
    tmp = qa_dir / "_clip_frames"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(master), str(tmp / "f%04d.png")], check=True)
        files = sorted(tmp.glob("f*.png")); worst = [float("inf")] * 5; bad = []; grid = None
        for f in files:
            im = np.asarray(Image.open(f).convert("RGB")).astype(int); h, w, _ = im.shape
            if grid is None:
                yy, xx = np.mgrid[0:h, 0:w]; X = (xx + 0.5) * F.W_MM / w; Y = F.H_MM - (yy + 0.5) * F.H_MM / h; grid = (X, Y)
            X, Y = grid
            for c in range(5):
                x0 = F.EDGE_MM + c * (F.CARD_W_MM + F.GAP_MM); cx = x0 + F.CARD_W_MM / 2; cy = 0.5 * (F.CARD_Y0_MM + F.CARD_Y1_MM)
                qx = np.abs(X - cx) - (F.CARD_W_MM / 2 - CORNER_MM); qy = np.abs(Y - cy) - ((F.CARD_Y1_MM - F.CARD_Y0_MM) / 2 - CORNER_MM)
                sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - CORNER_MM
                inner = (sdf < -0.6) & (Y < F.PIC_TOP_MM)
                fill = np.median(im[inner], 0); ink = inner & (np.abs(im - fill).max(-1) > 40)
                if not ink.any():
                    continue
                d = float(-sdf[ink].max()); worst[c] = min(worst[c], d)
                if d < LIMIT_MM:
                    bad.append({"frame": f.name, "card": c + 1, "distance_mm": round(d, 2)})
        out = {"stem": F.STEM, "frames_checked": len(files), "limit_mm": LIMIT_MM,
               "min_distance_mm_per_card": [round(v, 2) if np.isfinite(v) else None for v in worst],
               "violations": bad, "passed": not bad and len(files) > 0}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (qa_dir / "clip_check.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    r = check(F.ROOT / "videos" / f"{F.STEM}.mp4", F.QA_ROOT / "_qa")
    print(json.dumps({k: v for k, v in r.items() if k != "violations"}), "violations:", len(r["violations"]))
    sys.exit(0 if r["passed"] else 1)
