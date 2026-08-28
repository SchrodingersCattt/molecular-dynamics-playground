from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
VIDEO_DIR = ROOT / "videos"
DATA_DIR = ROOT / "data"

WHITE = "#FFFFFF"
INK = "#171717"
DARK_GRAY = "#505050"
MID_GRAY = "#8E8E8E"
LINE_GRAY = "#C8C8C8"
LIGHT_GRAY = "#EEEEEE"
NAVY = "#183153"
CRIMSON = "#A32035"
GREEN = "#2F6B4F"

STATIC_DPI = 300
STATIC_WIDTH_PX = 3508
STATIC_HEIGHT_PX = 2480
VIDEO_DPI = 100
VIDEO_WIDTH_PX = 1920
VIDEO_HEIGHT_PX = 1080
FPS = 24
FONT_MIN_PT = 10.0
FONT_MAX_PT = 16.0

CAMERA_DIRECTION = np.array([1.55, -1.0, 0.62], dtype=float)
CAMERA_UP = np.array([0.0, 0.0, 1.0], dtype=float)

VD_ROOT = Path(r"D:\DP_Internship\AIagents\AgentDev\visualize-data")
if str(VD_ROOT) not in sys.path:
    sys.path.insert(0, str(VD_ROOT))

from visualize_data.checks.pixels import (  # noqa: E402
    audit_whitespace,
    check_boundaries,
    check_clipping,
    check_colors,
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
    }
)


def mix_hex(first: str, second: str, weight: float) -> str:
    weight = float(np.clip(weight, 0.0, 1.0))
    a = np.array([int(first[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(second[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    rgb = np.rint((1.0 - weight) * a + weight * b).astype(int)
    return "#" + "".join(f"{value:02X}" for value in rgb)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def axes_from_top_slot(fig: plt.Figure, slot: Iterable[float]) -> plt.Axes:
    x0, y0, x1, y1 = (float(item) for item in slot)
    ax = fig.add_axes([x0, 1.0 - y1, x1 - x0, y1 - y0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("auto")
    ax.axis("off")
    return ax


def new_static_figure() -> plt.Figure:
    return plt.figure(
        figsize=(STATIC_WIDTH_PX / STATIC_DPI, STATIC_HEIGHT_PX / STATIC_DPI),
        dpi=STATIC_DPI,
        facecolor=WHITE,
    )


def new_video_figure() -> plt.Figure:
    return plt.figure(
        figsize=(VIDEO_WIDTH_PX / VIDEO_DPI, VIDEO_HEIGHT_PX / VIDEO_DPI),
        dpi=VIDEO_DPI,
        facecolor=WHITE,
    )


def add_page_title(
    fig: plt.Figure,
    number: str,
    title: str,
    subtitle: str,
    *,
    video: bool,
    registry: "LayoutRegistry | None" = None,
) -> None:
    number_size = 12 if video else 11
    title_size = 16
    subtitle_size = 12
    artists = [
        fig.text(0.048, 0.942, number, ha="left", va="top", fontsize=number_size, color=DARK_GRAY, weight="bold"),
        fig.text(0.048, 0.902, title, ha="left", va="top", fontsize=title_size, color=INK, weight="bold"),
        fig.text(0.050, 0.842, subtitle, ha="left", va="top", fontsize=subtitle_size, color=DARK_GRAY),
    ]
    if registry is not None:
        registry.texts.extend(artists)


def add_footer(
    fig: plt.Figure,
    text: str,
    *,
    video: bool,
    registry: "LayoutRegistry | None" = None,
) -> None:
    artist = fig.text(
        0.50,
        0.036 if video else 0.042,
        text,
        ha="center",
        va="bottom",
        fontsize=12 if video else 10,
        color=DARK_GRAY,
    )
    if registry is not None:
        registry.texts.append(artist)


@dataclass
class LayoutRegistry:
    min_font_pt: float
    max_font_pt: float = FONT_MAX_PT
    edge_pad_px: float = 12.0
    texts: list = field(default_factory=list)
    arrows: list = field(default_factory=list)

    def text(self, ax: plt.Axes, x: float, y: float, value: str, **kwargs):
        fontsize = float(kwargs.get("fontsize", self.min_font_pt))
        if fontsize < self.min_font_pt:
            raise ValueError(f"Font {fontsize:g} pt is below the contract minimum {self.min_font_pt:g} pt")
        if fontsize > self.max_font_pt:
            raise ValueError(f"Font {fontsize:g} pt is above the contract maximum {self.max_font_pt:g} pt")
        artist = ax.text(x, y, value, **kwargs)
        self.texts.append(artist)
        return artist

    def arrow(self, ax: plt.Axes, start, end, **kwargs):
        patch = FancyArrowPatch(start, end, **kwargs)
        ax.add_patch(patch)
        self.arrows.append(patch)
        return patch

    def validate(self, fig: plt.Figure) -> list[str]:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        errors: list[str] = []
        boxes = []
        for index, artist in enumerate(self.texts):
            if float(artist.get_fontsize()) < self.min_font_pt:
                errors.append(f"text[{index}] font below minimum")
            if float(artist.get_fontsize()) > self.max_font_pt:
                errors.append(f"text[{index}] font above maximum")
            bbox = artist.get_window_extent(renderer=renderer)
            boxes.append((index, bbox))
            if bbox.x0 < self.edge_pad_px or bbox.y0 < self.edge_pad_px:
                errors.append(f"text[{index}] crosses left/bottom canvas pad")
            if bbox.x1 > width - self.edge_pad_px or bbox.y1 > height - self.edge_pad_px:
                errors.append(f"text[{index}] crosses right/top canvas pad")
        for left in range(len(boxes)):
            i, a = boxes[left]
            for right in range(left + 1, len(boxes)):
                j, b = boxes[right]
                overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
                overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
                if overlap_x > 2.0 and overlap_y > 2.0:
                    errors.append(f"text[{i}] overlaps text[{j}] by {overlap_x:.1f}x{overlap_y:.1f}px")
        for index, artist in enumerate(self.arrows):
            bbox = artist.get_window_extent(renderer=renderer)
            if bbox.x0 < 4 or bbox.y0 < 4 or bbox.x1 > width - 4 or bbox.y1 > height - 4:
                errors.append(f"arrow[{index}] crosses canvas edge")
        return errors


def draw_three_step_loop(
    ax: plt.Axes,
    registry: LayoutRegistry,
    *,
    video: bool,
    active_stage: int | None,
    active_weight: float = 1.0,
    centre_lines: tuple[str, str] = ("one time step", r"$\Delta t$"),
) -> None:
    node_radius = 0.145 if video else 0.094
    top_y = 0.82 if video else 0.84
    nodes = [
        (0.50, top_y, "position", r"$\mathbf{r}_{n+1}$"),
        (0.79, 0.35, "acceleration", r"$\mathbf{a}_{n+1}$"),
        (0.21, 0.35, "velocity", r"$\mathbf{v}_{n+1}$"),
    ]
    arrow_lw = 4.0 if video else 2.5
    arrow_head = 28 if video else 20
    paths = [
        ((0.58, 0.79), (0.76, 0.46), -0.12),
        ((0.69, 0.31), (0.31, 0.31), -0.10),
        ((0.24, 0.45), (0.43, 0.79), -0.12),
    ]
    for start, end, rad in paths:
        registry.arrow(
            ax,
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=arrow_head,
            lw=arrow_lw,
            color=LINE_GRAY,
            zorder=2,
        )

    for index, (x, y, label, symbol) in enumerate(nodes):
        weight = active_weight if active_stage == index else 0.0
        fill = mix_hex(LIGHT_GRAY, INK, weight)
        text_color = WHITE if weight > 0.48 else DARK_GRAY
        ax.add_patch(Circle((x, y), node_radius, fc=fill, ec=LINE_GRAY, lw=2.6 if video else 1.8, zorder=5))
        registry.text(ax, x, y + 0.018, label, ha="center", va="center", fontsize=14 if video else 11, color=text_color, weight="bold", zorder=6)
        registry.text(ax, x, y - 0.040, symbol, ha="center", va="center", fontsize=14 if video else 10, color=text_color, zorder=6)

    registry.text(ax, 0.50, 0.60, centre_lines[0], ha="center", va="center", fontsize=14 if video else 11, color=DARK_GRAY)
    registry.text(ax, 0.50, 0.53, centre_lines[1], ha="center", va="center", fontsize=16 if video else 14, color=INK, weight="bold")


def camera_basis(direction=CAMERA_DIRECTION, up=CAMERA_UP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    up_vec = np.asarray(up, dtype=float)
    up_vec -= np.dot(up_vec, forward) * forward
    up_vec /= np.linalg.norm(up_vec)
    right = np.cross(forward, up_vec)
    right /= np.linalg.norm(right)
    return right, up_vec, forward


def project_points(points: np.ndarray, direction=CAMERA_DIRECTION, up=CAMERA_UP) -> tuple[np.ndarray, np.ndarray]:
    right, up_vec, forward = camera_basis(direction, up)
    points = np.asarray(points, dtype=float)
    return np.column_stack((points @ right, points @ up_vec)), points @ forward


def map_projected_to_rect(
    projected: np.ndarray,
    rect: tuple[float, float, float, float],
    centre_xy: np.ndarray,
    half_span: float,
) -> np.ndarray:
    x0, y0, x1, y1 = rect
    scale = min((x1 - x0), (y1 - y0)) / (2.0 * half_span)
    centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    return centre + (projected - centre_xy) * scale


def draw_ball_and_stick(
    ax: plt.Axes,
    positions: np.ndarray,
    elements: Iterable[str],
    bonds: np.ndarray,
    *,
    rect: tuple[float, float, float, float],
    centre_3d: np.ndarray,
    half_span: float,
    alpha: float = 1.0,
    atom_scale: float = 1.0,
    bond_alpha: float = 0.75,
    edge_color: str = WHITE,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    centre_projected, _ = project_points(np.asarray(centre_3d, dtype=float)[None, :])
    projected, depth = project_points(positions)
    xy = map_projected_to_rect(projected, rect, centre_projected[0], half_span)
    for first, second in sorted(np.asarray(bonds, dtype=int), key=lambda edge: float(depth[edge].mean())):
        ax.plot(
            [xy[first, 0], xy[second, 0]],
            [xy[first, 1], xy[second, 1]],
            color=DARK_GRAY,
            lw=8.0 * atom_scale,
            alpha=bond_alpha * alpha,
            solid_capstyle="round",
            zorder=4,
        )
    order = np.argsort(depth)
    for index in order:
        element = str(list(elements)[index])
        base = CRIMSON if element == "O" else NAVY if element == "H" else DARK_GRAY
        size = (1850 if element == "O" else 930 if element == "H" else 2300) * atom_scale
        ax.scatter(xy[index, 0], xy[index, 1], s=size, c=base, alpha=alpha, edgecolors=edge_color, linewidths=2.2 * atom_scale, zorder=8)
        ax.scatter(xy[index, 0] - 0.010 * atom_scale, xy[index, 1] + 0.014 * atom_scale, s=size * 0.13, c=WHITE, alpha=0.52 * alpha, edgecolors="none", zorder=9)
    return xy, depth


def draw_vector_arrow(
    ax: plt.Axes,
    registry: LayoutRegistry,
    start_3d: np.ndarray,
    vector_3d: np.ndarray,
    *,
    colour: str,
    rect: tuple[float, float, float, float],
    centre_3d: np.ndarray,
    half_span: float,
    display_scale: float,
    video: bool,
    alpha: float = 1.0,
) -> None:
    points = np.vstack((start_3d, start_3d + display_scale * vector_3d))
    projected, _ = project_points(points)
    centre_projected, _ = project_points(np.asarray(centre_3d)[None, :])
    xy = map_projected_to_rect(projected, rect, centre_projected[0], half_span)
    registry.arrow(
        ax,
        tuple(xy[0]),
        tuple(xy[1]),
        arrowstyle="-|>",
        mutation_scale=28 if video else 14,
        lw=5.0 if video else 3.0,
        color=colour,
        alpha=alpha,
        zorder=15,
    )


def render_source_panel(
    path: Path,
    draw: Callable[[plt.Axes, LayoutRegistry], None],
    *,
    width_px: int = 1400,
    height_px: int = 1200,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(width_px / 150, height_px / 150), dpi=150, facecolor=WHITE)
    ax = fig.add_axes([0.055, 0.055, 0.89, 0.89])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    registry = LayoutRegistry(min_font_pt=10, edge_pad_px=8)
    draw(ax, registry)
    errors = registry.validate(fig)
    if errors:
        raise RuntimeError("Source-panel layout failed:\n" + "\n".join(errors))
    fig.savefig(path, dpi=150, facecolor=WHITE)
    plt.close(fig)


def save_static(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{stem}.png"
    svg = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=STATIC_DPI, facecolor=WHITE)
    fig.savefig(svg, facecolor=WHITE)
    plt.close(fig)
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    return png, svg


def _frame_checks(image: np.ndarray, config: dict, semantics: list[dict]) -> list:
    whitespace = dict(config["whitespace"])
    whitespace.setdefault("panels", [{"id": item["id"], "rect": item["rect"]} for item in config["panels"]])
    clipping_panels = [
        {
            "id": item["id"],
            "rect": item["rect"],
            "min_clearance_px": item.get("min_clearance_px", 16),
            "allow_touch_edges": item.get("allow_touch_edges", []),
        }
        for item in config["panels"]
    ]
    return [
        audit_whitespace(image, whitespace),
        check_clipping(image, clipping_panels, threshold=245),
        check_boundaries(image, config.get("bands", []), threshold=245),
        check_colors(image, semantics),
    ]


def render_video(
    *,
    stem: str,
    duration_seconds: float,
    draw_frame: Callable[[plt.Figure, float, int, LayoutRegistry], list[dict]],
    audit_config: dict,
    qa_directory: Path,
    representative_times: Iterable[float],
) -> Path:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    qa_directory.mkdir(parents=True, exist_ok=True)
    representative_dir = qa_directory / "video_frames"
    representative_dir.mkdir(parents=True, exist_ok=True)
    output = VIDEO_DIR / f"{stem}.mp4"
    temporary = VIDEO_DIR / f"_{stem}.encoding.mp4"
    frames = int(round(duration_seconds * FPS))
    representative_indices = {int(round(value * FPS)): value for value in representative_times}
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{VIDEO_WIDTH_PX}x{VIDEO_HEIGHT_PX}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    fig = new_video_figure()
    frame_records = []
    failures = []
    contact_thumbnails: list[Image.Image] = []
    try:
        for frame_index in range(frames):
            fig.clear()
            time_seconds = frame_index / FPS
            registry = LayoutRegistry(min_font_pt=FONT_MIN_PT, max_font_pt=FONT_MAX_PT, edge_pad_px=12)
            semantics = draw_frame(fig, time_seconds, frame_index, registry)
            layout_errors = registry.validate(fig)
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgb = np.ascontiguousarray(rgba[:, :, :3])
            thumbnail = Image.fromarray(rgb).resize((240, 135), Image.Resampling.LANCZOS)
            contact_thumbnails.append(thumbnail)
            checks = _frame_checks(rgb, audit_config, semantics)
            check_errors = [
                f"{result.check}: {finding.message}"
                for result in checks
                for finding in result.findings
                if finding.level == "error"
            ]
            errors = layout_errors + check_errors
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            frame_records.append(
                {
                    "frame": frame_index,
                    "time_seconds": time_seconds,
                    "sha256_rgb": digest,
                    "passed": not errors,
                    "layout_errors": layout_errors,
                    "checks": [
                        {
                            "check": result.check,
                            "passed": result.passed,
                            "findings": [finding.to_dict() for finding in result.findings],
                        }
                        for result in checks
                    ],
                }
            )
            if errors:
                failure_path = qa_directory / f"failed_frame_{frame_index:04d}.png"
                fig.savefig(failure_path, dpi=VIDEO_DPI, facecolor=WHITE)
                failures.append({"frame": frame_index, "time_seconds": time_seconds, "errors": errors, "image": str(failure_path)})
                raise RuntimeError(f"Frame {frame_index} failed strict frame QA: " + "; ".join(errors))
            if frame_index in representative_indices:
                fig.savefig(representative_dir / f"frame_{frame_index:04d}.png", dpi=VIDEO_DPI, facecolor=WHITE)
            assert process.stdin is not None
            process.stdin.write(rgb.tobytes())
    finally:
        plt.close(fig)
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        report = {
            "version": 1,
            "stem": stem,
            "visualize_data_root": str(VD_ROOT),
            "frame_count_expected": frames,
            "frame_count_audited": len(frame_records),
            "fps": FPS,
            "dimensions": [VIDEO_WIDTH_PX, VIDEO_HEIGHT_PX],
            "minimum_font_pt": FONT_MIN_PT,
            "maximum_font_pt": FONT_MAX_PT,
            "passed": len(frame_records) == frames and not failures and return_code == 0,
            "ffmpeg_return_code": return_code,
            "failures": failures,
            "frames": frame_records,
        }
        (qa_directory / "every_frame_qa.json").write_text(
            json.dumps(report, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        contact_dir = qa_directory / "contact_sheets"
        contact_dir.mkdir(parents=True, exist_ok=True)
        per_page = 48
        columns = 8
        rows = 6
        for page_index, start in enumerate(range(0, len(contact_thumbnails), per_page)):
            page = Image.new("RGB", (columns * 240, rows * 135), WHITE)
            for offset, thumb in enumerate(contact_thumbnails[start : start + per_page]):
                page.paste(thumb, ((offset % columns) * 240, (offset // columns) * 135))
            page.save(contact_dir / f"contact_{page_index:02d}.jpg", quality=92, subsampling=0)
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    if failures or len(frame_records) != frames:
        raise RuntimeError("Video was not published because one or more frames failed QA")
    temporary.replace(output)
    return output


def cosine_stage(time_seconds: float, stage_seconds: float, stage_count: int) -> tuple[int, float]:
    bounded = min(max(time_seconds, 0.0), stage_seconds * stage_count - 1e-9)
    stage = int(bounded // stage_seconds)
    local = (bounded - stage * stage_seconds) / stage_seconds
    fade = smoothstep(min(local / 0.18, 1.0)) * smoothstep(min((1.0 - local) / 0.18, 1.0))
    return stage, max(0.30, fade)


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
