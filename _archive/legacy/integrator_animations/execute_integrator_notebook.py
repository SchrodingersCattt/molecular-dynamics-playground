"""Execute the two code cells and embed their deterministic outputs."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "Dynamics and Numerical Integrator.ipynb",
    ROOT / "dynamics_integrator.ipynb",
)
FIGURE = ROOT / "integrator_animations" / "figures" / "integrator_comparison.png"


def stream_output(text: str) -> dict:
    return {"name": "stdout", "output_type": "stream", "text": text}


def image_output(path: Path) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "data": {
            "image/png": encoded,
            "text/plain": "<Figure size 3507x2481 with 9 Axes>",
        },
        "metadata": {},
        "output_type": "display_data",
    }


def main() -> None:
    os.chdir(ROOT)
    notebook = json.loads(NOTEBOOKS[0].read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}

    for execution_count, cell_index in enumerate((5, 6), start=1):
        cell = notebook["cells"][cell_index]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compile(cell["source"], f"<cell {cell_index}>", "exec"), namespace)

        outputs = []
        if captured.getvalue():
            outputs.append(stream_output(captured.getvalue()))
        if cell_index == 6:
            if not FIGURE.exists():
                raise FileNotFoundError(FIGURE)
            outputs.insert(0, image_output(FIGURE))

        cell["execution_count"] = execution_count
        cell["outputs"] = outputs

    rendered = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    for path in NOTEBOOKS:
        path.write_text(rendered, encoding="utf-8")

    if NOTEBOOKS[0].read_bytes() != NOTEBOOKS[1].read_bytes():
        raise RuntimeError("Notebook copies diverged after execution.")

    print(FIGURE)
    print(ROOT / "integrator_animations" / "figures" / "integrator_comparison.pdf")


if __name__ == "__main__":
    main()
