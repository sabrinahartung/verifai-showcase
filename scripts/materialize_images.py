#!/usr/bin/env python
"""Fetch the HAM10000 JPEGs that the manifests point at.

The manifests (small, versioned, in git) are the source of truth for *which*
images belong to which split. This script fetches the pixels they name into a
gitignored directory, so training and evaluation both work off plain files and
the existing manifest loader needs no changes.

Reads the image bytes straight out of the dataset's parquet, deduplicated by
image_id — the dataset stores ~13.4k rows for 10,015 distinct images, so going
through the rows would download a third more than necessary.

Storage: the originals are 600x450 at ~264 KB, so all 10,015 come to ~2.7 GB.
`--max-size 320` re-encodes them on the way in and costs ~140 MB instead, which
loses nothing for training because the first transform resizes to 224 anyway.
Note it does shift the robustness metric slightly (a blur radius or a JPEG quality
means something different at a different resolution), so keep one choice for a
whole comparison. The choice is recorded in `_materialize.json` beside the images.

Usage:
    python scripts/materialize_images.py --max-size 320      # ~140 MB, recommended
    python scripts/materialize_images.py                     # originals, ~2.7 GB
    python scripts/materialize_images.py --split test        # just one split (~400 MB)
    python scripts/materialize_images.py --limit 20          # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from build_splits import DATASET, parquet_urls   # same directory

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data" / "raw" / "ham10000"


def wanted_ids(manifest_dir: Path, prefix: str, splits: list[str]) -> dict[str, str]:
    """image_id -> filename, from the manifests."""
    want = {}
    for s in splits:
        p = manifest_dir / f"{prefix}_{s}.csv"
        if not p.exists():
            sys.exit(f"missing {p} — run scripts/build_splits.py first")
        for row in csv.DictReader(open(p, encoding="utf-8")):
            want[row["image_id"]] = row["filename"]
    return want


def _shrink(data: bytes, max_size: int, quality: int) -> bytes:
    """Re-encode with the longest side capped. Aspect ratio preserved."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--manifests", default=str(REPO / "data" / "manifests"))
    ap.add_argument("--prefix", default="ham10000")
    ap.add_argument("--split", action="append", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (smoke test)")
    ap.add_argument("--max-size", type=int, default=0,
                    help="longest side in px; 0 keeps the original (320 -> ~19x smaller)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality when resizing")
    a = ap.parse_args()

    try:
        import duckdb
    except ImportError:
        sys.exit("needs duckdb:  pip install duckdb")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # A directory of full-size images and one of downscaled images must not be
    # silently mixed — the robustness numbers would not be comparable.
    stamp = out / "_materialize.json"
    want_cfg = {"max_size": a.max_size or None, "quality": a.quality if a.max_size else None,
                "source": DATASET}
    if stamp.exists():
        have = json.loads(stamp.read_text())
        if have.get("max_size") != want_cfg["max_size"]:
            sys.exit(f"{out} already holds images at max_size={have.get('max_size')}, "
                     f"but --max-size {a.max_size or 'None'} was requested. Use a different "
                     f"--out, or delete that directory first.")
    want = wanted_ids(Path(a.manifests), a.prefix, a.split or ["train", "val", "test"])
    todo = {i: f for i, f in want.items() if not (out / f).exists()}
    print(f"▶ {len(want):,} images wanted, {len(want) - len(todo):,} already on disk, "
          f"{len(todo):,} to fetch -> {out}")
    if not todo:
        print("✓ nothing to do"); return

    con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
    files = "['" + "','".join(parquet_urls()) + "']"
    ids = "(" + ",".join(f"'{i}'" for i in todo) + ")"
    con.execute(f"""
        SELECT image_id, any_value(image) AS im
        FROM read_parquet({files})
        WHERE image_id IN {ids}
        GROUP BY image_id
    """)

    got = 0
    while True:
        batch = con.fetchmany(64)
        if not batch:
            break
        for image_id, im in batch:
            data = im["bytes"] if isinstance(im, dict) else im
            if a.max_size:
                data = _shrink(data, a.max_size, a.quality)
            (out / todo[image_id]).write_bytes(data)
            got += 1
            if got % 250 == 0:
                print(f"   {got:,}/{len(todo):,}")
            if a.limit and got >= a.limit:
                print(f"✓ stopped at --limit {a.limit}"); return
    stamp.write_text(json.dumps(want_cfg, indent=2), encoding="utf-8")
    size_mb = sum(f.stat().st_size for f in out.glob("*.jpg")) / 1e6
    print(f"✓ wrote {got:,} images to {out}  ({size_mb:,.0f} MB on disk"
          f"{f', max side {a.max_size}px' if a.max_size else ', originals'})")
    missing = [i for i, f in todo.items() if not (out / f).exists()]
    if missing:
        print(f"! {len(missing)} still missing, e.g. {missing[:3]}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
