#!/usr/bin/env python
"""Materialize the ISIC 2019 JPEGs that isic_train.csv / isic_val.csv name.

The manifests (small, versioned, in git) say *which* images belong to the
training corpus; this fetches the pixels they name into a gitignored directory,
so the existing manifest loader needs no changes. Same contract as
materialize_images.py, different source.

Reads straight out of ISIC_2019_Training_Input.zip without extracting it. The
zip is 9.1 GB; extracting first would cost another 9.1 GB for no reason, since
every member is read exactly once.

Two sources, on purpose
-----------------------
8,393 of the 23,278 images the corpus needs are HAM10000 images we already hold
at 320px in `data/raw/ham10000`. Those are **copied byte-for-byte** rather than
re-encoded from the ISIC zip.

That is not only to save time. `train_model.py` selects a checkpoint on the
validation split read from `training.images_dir`, while `tune_decision.py` tunes
the decision weight on validation read from `dataset.images_dir`. If the shared
images were re-encoded here, those two would be looking at different pixels for
the same rows — the threshold tuned on one encoding and the checkpoint selected
on another. Copying makes them identical, and also makes every shared image
identical to what the five published runs were scored on.

Only the 14,885 genuinely new ISIC images are encoded, at the same 320px /
quality 90 recorded in `data/raw/ham10000/_materialize.json`. Keep one choice
for a whole comparison: a blur radius or a JPEG quality means something
different at a different resolution, so mixing them would shift the robustness
metric for reasons that have nothing to do with the model.

Usage:
    # 1. get the zip (9.1 GB, official ISIC S3 bucket, no account needed)
    curl -O https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip

    python scripts/materialize_isic.py --zip ISIC_2019_Training_Input.zip
    python scripts/materialize_isic.py --zip ... --limit 50   # smoke test first
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data" / "raw" / "isic2019"
DEFAULT_REUSE = REPO / "data" / "raw" / "ham10000"
SOURCE = "ISIC 2019 Training Input"


def wanted(manifest_dir: Path, prefix: str, splits: list[str]) -> dict[str, str]:
    """image_id -> filename, from the manifests. They are the source of truth."""
    want: dict[str, str] = {}
    for s in splits:
        p = manifest_dir / f"{prefix}_{s}.csv"
        if not p.exists():
            sys.exit(f"missing {p} — run scripts/build_isic_train.py first")
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                want[row["image_id"]] = row["filename"]
    return want


def _shrink(data: bytes, max_size: int, quality: int) -> bytes:
    """Re-encode with the longest side capped. Aspect ratio preserved.

    Same transform as materialize_images.py, so the two directories agree.
    """
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def zip_index(zf: zipfile.ZipFile) -> dict[str, str]:
    """basename-without-extension -> member name.

    Matched on basename because the challenge zip nests everything under
    `ISIC_2019_Training_Input/`, and a future repackaging might not.
    """
    index = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        stem = Path(name).stem
        if stem:
            index[stem] = name
    return index


def check_stamp(out: Path, max_size: int, quality: int) -> None:
    """Refuse to mix encodings in one directory — materialize_images.py's rule.

    One directory, one encoding. A directory holding two is unusable for a
    comparison and there is nothing in the files themselves to reveal it.
    """
    stamp = out / "_materialize.json"
    if not stamp.exists():
        return
    have = json.loads(stamp.read_text(encoding="utf-8"))
    if have.get("max_size") != (max_size or None) or have.get("quality") != (quality if max_size else None):
        sys.exit(f"{out} already holds images at max_size={have.get('max_size')} "
                 f"quality={have.get('quality')}, but max_size={max_size or None} "
                 f"quality={quality if max_size else None} was requested.\n"
                 f"Use a different --out, or delete that directory first.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True,
                    help="ISIC_2019_Training_Input.zip (read in place, not extracted)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--reuse-from", default=str(DEFAULT_REUSE),
                    help="directory of already-materialized images to copy instead of "
                         "re-encoding; pass '' to disable")
    ap.add_argument("--manifests", default=str(REPO / "data" / "manifests"))
    ap.add_argument("--prefix", default="isic")
    ap.add_argument("--split", action="append", choices=["train", "val"],
                    help="default: both, because training reads train and val")
    ap.add_argument("--max-size", type=int, default=320,
                    help="longest side; 320 matches data/raw/ham10000")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (smoke test)")
    a = ap.parse_args()

    zpath = Path(a.zip)
    if not zpath.exists():
        sys.exit(f"not found: {zpath}\nFetch it with:\n  curl -O "
                 f"https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip")

    out = Path(a.out)
    check_stamp(out, a.max_size, a.quality)
    out.mkdir(parents=True, exist_ok=True)

    splits = a.split or ["train", "val"]
    want = wanted(Path(a.manifests), a.prefix, splits)
    print(f"▶ {len(want):,} images named by {a.prefix}_{'/'.join(splits)}.csv")

    reuse_dir = Path(a.reuse_from) if a.reuse_from else None
    todo = {i: f for i, f in want.items() if not (out / f).exists()}
    print(f"  {len(want) - len(todo):,} already present, {len(todo):,} to do")

    # Copy what we already hold at this exact encoding, rather than re-encoding
    # it from the zip: byte-identical beats merely equivalent, and it keeps the
    # shared rows identical to what the published runs were scored on.
    copied = 0
    if reuse_dir and reuse_dir.exists():
        for image_id, fname in sorted(todo.items()):
            src = reuse_dir / fname
            if src.exists():
                shutil.copyfile(src, out / fname)
                copied += 1
                if a.limit and copied >= a.limit:
                    break
        print(f"▶ copied {copied:,} byte-identical from {reuse_dir}")
        todo = {i: f for i, f in todo.items() if not (out / f).exists()}
    elif reuse_dir:
        print(f"  note: {reuse_dir} does not exist, so everything is encoded from the zip")

    if a.limit:
        todo = dict(sorted(todo.items())[:max(0, a.limit - copied)])

    encoded = failed = 0
    if todo:
        print(f"▶ encoding {len(todo):,} from {zpath.name} "
              f"at max_size={a.max_size or 'original'} quality={a.quality}", flush=True)
        with zipfile.ZipFile(zpath) as zf:
            index = zip_index(zf)
            missing = [i for i in todo if i not in index]
            if missing:
                sys.exit(f"{len(missing)} manifest image(s) are not in the zip, "
                         f"e.g. {missing[0]}. Wrong archive?")
            for n, (image_id, fname) in enumerate(sorted(todo.items()), 1):
                try:
                    with zf.open(index[image_id]) as fh:
                        data = fh.read()
                    if a.max_size:
                        data = _shrink(data, a.max_size, a.quality)
                    (out / fname).write_bytes(data)
                    encoded += 1
                except Exception as e:                      # noqa: BLE001
                    failed += 1
                    print(f"  ! {image_id}: {e}")
                if n % 2000 == 0:
                    print(f"  {n:,}/{len(todo):,}", flush=True)

    # Record what this directory holds, and that it has two provenances. The
    # stamp is the only place that distinction survives.
    (out / "_materialize.json").write_text(json.dumps({
        "max_size": a.max_size or None,
        "quality": a.quality if a.max_size else None,
        "source": SOURCE,
        "reused_from": str(reuse_dir) if copied else None,
        "reused_count": copied,
        "encoded_count": encoded,
    }, indent=2) + "\n", encoding="utf-8")

    have = len(list(out.glob("*.jpg")))
    size_mb = sum(f.stat().st_size for f in out.glob("*.jpg")) / 1e6
    print(f"\n✓ {have:,} images in {out}  ({size_mb:,.0f} MB)")
    print(f"  {copied:,} copied byte-identical, {encoded:,} encoded from the zip"
          + (f", {failed:,} failed" if failed else ""))

    if not a.limit:
        absent = [f for f in want.values() if not (out / f).exists()]
        if absent:
            sys.exit(f"REFUSING to report success: {len(absent)} manifest image(s) "
                     f"are still missing, e.g. {absent[0]}")
        print("  every image the manifests name is present.")
        print(f"\n  Next: python scripts/train_model.py scenarios/skin_cancer_isic.yaml")


if __name__ == "__main__":
    main()
