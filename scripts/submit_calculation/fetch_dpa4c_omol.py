"""Fetch and record a public DPA4C checkpoint from AIS Square.

The download is intentionally explicit: the model is not vendored in git.
The default route uses the public ``aissq-explorer`` client and AIS Square;
the Hugging Face route is retained as an optional fallback.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://huggingface.co/deepmodelingcommunity/DPA4C-OMol/resolve/main/"
VARIANTS = {"nano": "DPA4C-Nano", "mini": "DPA4C-Mini", "neo": "DPA4C-Neo", "air": "DPA4C-Air", "plus": "DPA4C-Plus"}
VERSION = "v20260820"

def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(url, headers={"User-Agent": "0314-dpa4c-fetch/1"}), timeout=120) as src, target.open("wb") as dst:
        while chunk := src.read(1024 * 1024): dst.write(chunk)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="neo")
    ap.add_argument("--resource-name", default="DPA4C-OMat24", help="Exact AIS Square model name")
    ap.add_argument("--source", choices=("aissq", "huggingface"), default="aissq")
    ap.add_argument("--output", type=Path, default=Path("product/data/dpa4c"))
    args = ap.parse_args()
    if args.source == "aissq":
        try:
            from aissq.client import AissqClient
        except ImportError as exc:
            raise SystemExit("Install aissq-explorer first: python -m pip install git+https://github.com/SchrodingersCattt/aissq-explorer.git") from exc
        client = AissqClient(timeout=60)
        candidates = client.search_by_keyword(args.resource_name or "DPA4C", "models")
        if not candidates and not args.resource_name:
            candidates = client.search_by_keyword("DPA", "models")
        if not candidates:
            raise SystemExit("No AIS Square model matched. Run with --resource-name and the exact model name.")
        resource = next((x for x in candidates if x.get("name") == args.resource_name), candidates[0])
        name = resource.get("name")
        paths = client.download_resource(name, "models", output_dir=args.output, show_progress=True)
        manifest = {"source": "AIS Square via aissq-explorer", "resource_name": name,
                    "resource_id": resource.get("ID"), "resource": resource,
                    "downloaded_files": [str(x) for x in paths], "backend": "DeePMD-kit"}
        (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, default=str)); return
    stem = f"{VARIANTS[args.variant]}-OMol25-100M-{VERSION}"
    model, config = args.output / f"{stem}.pt", args.output / f"{stem}.json"
    download(BASE + model.name, model); download(BASE + config.name, config)
    manifest = {"source": "deepmodelingcommunity/DPA4C-OMol", "variant": args.variant, "version": VERSION,
                "model": model.name, "config": config.name, "model_sha256": sha256(model), "config_sha256": sha256(config),
                "license": "cc-by-nc-4.0", "backend": "DeePMD-kit >=3.2 PyTorch Exportable"}
    (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
