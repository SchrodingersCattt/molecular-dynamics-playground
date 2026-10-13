"""Write a PowerPoint MP4 and a GIF next to each master movie.

The master stays at product/videos/<stem>.mp4. Slide copies go to
product/videos/ppt/.

MP4: 1280x400, 24 fps, H.264 CRF 30, faststart. These frames are mostly
white with thin lines, so CRF 30 keeps the labels and contour dashes and
brings a 30 s clip down to a few megabytes. CRF 23 is cleaner but only
modestly smaller than the master. CRF 35 starts to break the small type.

GIF: 960x300, 8 fps, 128 colours. PowerPoint plays it without a codec, but
the palette softens red/green contours and the file is often larger than
the CRF 30 MP4. Use the MP4 on the slide; keep the GIF as a fallback.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "product" / "videos"
OUTPUT_DIR = SOURCE_DIR / "ppt"


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"{command[0]} failed:\n{detail}")


def export_one(source: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    mp4 = OUTPUT_DIR / f"{stem}.mp4"
    gif = OUTPUT_DIR / f"{stem}.gif"
    palette = OUTPUT_DIR / f"_{stem}.palette.png"
    _run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-an", "-vf", "scale=1280:400:flags=lanczos", "-r", "24",
            "-c:v", "libx264", "-preset", "slow", "-crf", "30",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(mp4),
        ]
    )
    _run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-an", "-vf", "fps=8,scale=960:300:flags=lanczos,palettegen=max_colors=128:stats_mode=diff",
            str(palette),
        ]
    )
    _run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(source), "-i", str(palette),
            "-an", "-lavfi", "fps=8,scale=960:300:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
            str(gif),
        ]
    )
    palette.unlink(missing_ok=True)
    print(f"{stem}  mp4 {mp4.stat().st_size / 1e6:.2f} MB  gif {gif.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    sources = sorted(
        path for path in SOURCE_DIR.glob("*.mp4") if not path.name.startswith("_")
    )
    if not sources:
        raise SystemExit(f"No master videos in {SOURCE_DIR}")
    for source in sources:
        export_one(source)


if __name__ == "__main__":
    main()
