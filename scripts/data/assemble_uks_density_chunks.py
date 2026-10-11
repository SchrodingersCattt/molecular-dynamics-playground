"""Split or reassemble the large 03b fixed-grid density NPZ.

The GitHub repository stores the 3-D NPZ as ordinary chunks below
``product/data/chunks`` so a clone does not depend on Git LFS.  The renderer
can call ``assemble`` automatically when the full NPZ is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_NAME = "uks_tnt_reaction_density3d.npz"
DEFAULT_CHUNK_BYTES = 50 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_file(source: Path, directory: Path, *, chunk_bytes: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, object]] = []
    with source.open("rb") as handle:
        index = 0
        while True:
            block = handle.read(int(chunk_bytes))
            if not block:
                break
            chunk = directory / f"{source.name}.part-{index:03d}"
            chunk.write_bytes(block)
            chunks.append({"path": chunk.name, "bytes": len(block), "sha256": sha256(chunk)})
            index += 1
    manifest = directory / f"{source.name}.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "chunked_npz/v1",
                "file": source.name,
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
                "chunk_bytes": int(chunk_bytes),
                "chunks": chunks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def assemble(manifest: Path, output: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as target:
        for item in payload["chunks"]:
            chunk = manifest.parent / str(item["path"])
            if sha256(chunk) != item["sha256"]:
                raise RuntimeError(f"Chunk SHA256 mismatch: {chunk}")
            with chunk.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
    if output.stat().st_size != int(payload["bytes"]) or sha256(output) != payload["sha256"]:
        raise RuntimeError(f"Reassembled NPZ SHA256 mismatch: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("split")
    split.add_argument("source", type=Path)
    split.add_argument("directory", type=Path)
    split.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    join = sub.add_parser("assemble")
    join.add_argument("manifest", type=Path)
    join.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "split":
        print(split_file(args.source, args.directory, chunk_bytes=args.chunk_bytes))
    else:
        assemble(args.manifest, args.output)


if __name__ == "__main__":
    main()
