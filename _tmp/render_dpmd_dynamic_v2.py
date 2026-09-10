"""Compatibility alias; the canonical renderer is render_dpmd_dynamic.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_dpmd_dynamic import draw_frame, load_case, render

if __name__ == "__main__":
    print(render())
