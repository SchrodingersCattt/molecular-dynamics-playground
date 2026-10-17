"""One-command rebuild of the 08 (PCA/t-SNE/UMAP, 3 rows), 08b (one-row overview) and 09 (multi-task fine-tuning) explainers.

    .venv311\\Scripts\\python.exe scripts\\md_visuals\\build_08_09_explainers.py [--deck-assets DIR]

Steps: regenerate the toy data (seeded) -> static PNGs (300 dpi) -> master videos with every-frame QA
and contact sheets (common.render_video) -> PowerPoint copies via export_ppt_media.export_one
settings (24 fps, CRF 30) at 1280 px wide; all three figures are 338 x 125 mm since round 11 (not 16:5),
so export_wide keeps their aspect (1280 x 474) -> optional copy of PNG/MP4/poster into a deck asset folder.
Writes build_08_09_manifest.json with SHA-256 of every output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRODUCT = REPO / "product"
STEMS = ("08_tsne_umap_principle", "08b_dimred_overview", "09_multitask_finetune")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export_wide(master: Path) -> Path:
    """export_one's settings (24 fps, H.264 CRF 30, yuv420p, faststart) without forcing 16:5."""
    out = master.parent / "ppt" / master.name; out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(master), "-vf", "scale=1280:-2:flags=lanczos,fps=24",
                    "-c:v", "libx264", "-crf", "30", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-an", str(out)], check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--deck-assets", type=Path); args = ap.parse_args()
    py = sys.executable
    subprocess.run([py, str(REPO / "scripts" / "build_box" / "generate_tsne_umap_toy.py")], check=True)
    subprocess.run([py, str(HERE / "render_08_tsne_umap.py")], check=True, cwd=HERE)
    subprocess.run([py, str(HERE / "render_08b_dimred_overview.py")], check=True, cwd=HERE)
    subprocess.run([py, str(HERE / "render_09_multitask_finetune.py")], check=True, cwd=HERE)
    sys.path.insert(0, str(HERE))
    out = {}
    for stem in STEMS:
        master = PRODUCT / "videos" / f"{stem}.mp4"; export_wide(master)
        ppt = PRODUCT / "videos" / "ppt" / f"{stem}.mp4"; png = PRODUCT / "figures" / f"{stem}.png"
        poster = PRODUCT / "videos" / "ppt" / f"{stem}_poster.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", str(ppt), "-frames:v", "1", str(poster)], check=True)
        qa = json.loads((PRODUCT / "qa" / stem / "_qa" / "every_frame_qa.json").read_text(encoding="utf-8"))
        if not qa["passed"]:
            raise SystemExit(f"{stem}: every-frame QA failed")
        out[stem] = {"png": str(png), "master": str(master), "ppt": str(ppt), "poster": str(poster),
                     "frames": qa.get("frame_count_audited", qa.get("frames")), "qa_passed": True,
                     "sha256": {"png": sha(png), "master": sha(master), "ppt": sha(ppt)}}
        if args.deck_assets:
            args.deck_assets.mkdir(parents=True, exist_ok=True)
            for f in (png, ppt, poster):
                shutil.copy2(f, args.deck_assets / f.name)
    (PRODUCT / "qa" / "build_08_09_manifest.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
