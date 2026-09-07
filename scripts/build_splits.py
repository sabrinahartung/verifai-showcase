#!/usr/bin/env python
"""Build leak-free train/val/test manifests for HAM10000.

Why this exists
---------------
The splits shipped with `marmal88/skin_cancer` are contaminated: 80% of its test
images have an image_id that also appears in train, and train+validation together
cover 99.5% of HAM10000. Any accuracy measured on that test split is largely a
memorisation check. See docs/ROADMAP.md for the audit.

What this does instead
----------------------
HAM10000 photographs the same lesion more than once, so splitting on rows leaks
even when image_ids are unique. We split on **lesion_id**, and because every
lesion carries exactly one diagnosis (verified), we can stratify at lesion level
too — class balance is preserved *and* no lesion appears on both sides.

Only the metadata columns are read, over HTTP range requests, so this costs a few
MB rather than the 3.6 GB the images would. Run scripts/materialize_images.py
afterwards to fetch the JPEGs the manifests point at.

Usage:
    python scripts/build_splits.py --out data/manifests --prefix ham10000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASET = "marmal88/skin_cancer"
COLUMNS = ["image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization"]


def _ssl_context():
    """python.org builds on macOS ship without root certs; use certifi when present."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def parquet_urls(dataset: str = DATASET) -> list[str]:
    """Ask the HF datasets-server where the parquet lives, rather than guessing."""
    url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset.replace('/', '%2F')}"
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as r:
        return [f["url"] for f in json.loads(r.read())["parquet_files"]]


def load_metadata(urls: list[str]) -> list[dict]:
    """One row per unique image_id, metadata columns only (no image bytes)."""
    try:
        import duckdb
    except ImportError:
        sys.exit("needs duckdb:  pip install duckdb   (only this script uses it)")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    files = "['" + "','".join(urls) + "']"
    cols = ", ".join(f"any_value({c}) AS {c}" for c in COLUMNS if c != "image_id")
    rows = con.execute(
        f"SELECT image_id, {cols} FROM read_parquet({files}) GROUP BY image_id"
    ).fetchall()
    names = ["image_id"] + [c for c in COLUMNS if c != "image_id"]
    return [dict(zip(names, r)) for r in rows]


def split_lesions(rows: list[dict], ratios=(0.70, 0.15, 0.15), seed: int = 42):
    """Stratified split of *lesions* (never of images) into train/val/test.

    Every lesion has exactly one dx, so stratifying at lesion level is well
    defined: shuffle each class's lesions with a seeded RNG and slice.
    """
    by_class: dict[str, set] = defaultdict(set)
    for r in rows:
        by_class[r["dx"]].add(r["lesion_id"])

    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for dx, lesions in sorted(by_class.items()):
        ls = sorted(lesions)          # sort first so the shuffle is reproducible
        rng.shuffle(ls)
        n = len(ls)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        for i, lid in enumerate(ls):
            assign[lid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return assign


def write_manifests(rows: list[dict], assign: dict[str, str], out: Path, prefix: str):
    out.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "image_id", "lesion_id", "label", "dx_type", "age", "sex", "localization"]
    written = {}
    for split in ("train", "val", "test"):
        sel = [r for r in rows if assign[r["lesion_id"]] == split]
        sel.sort(key=lambda r: r["image_id"])          # deterministic order
        path = out / f"{prefix}_{split}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            for r in sel:
                w.writerow({
                    "filename": f"{r['image_id']}.jpg", "image_id": r["image_id"],
                    "lesion_id": r["lesion_id"], "label": r["dx"],
                    "dx_type": r["dx_type"], "age": r["age"], "sex": r["sex"],
                    "localization": r["localization"],
                })
        written[split] = (path, sel)
    return written


def verify(written) -> bool:
    """The check the original splits would have failed."""
    les = {s: {r["lesion_id"] for _, rows in [written[s]] for r in rows} for s in written}
    img = {s: {r["image_id"] for _, rows in [written[s]] for r in rows} for s in written}
    ok = True
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        shared_l, shared_i = les[a] & les[b], img[a] & img[b]
        flag = "OK " if not (shared_l or shared_i) else "LEAK"
        ok &= not (shared_l or shared_i)
        print(f"  [{flag}] {a:<5} vs {b:<5}: {len(shared_l)} shared lesions, {len(shared_i)} shared images")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--prefix", default="ham10000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.70,0.15,0.15")
    a = ap.parse_args()
    ratios = tuple(float(x) for x in a.ratios.split(","))

    print(f"▶ reading metadata for {DATASET} (columns only, no images)")
    rows = load_metadata(parquet_urls())
    print(f"  {len(rows):,} unique images / {len({r['lesion_id'] for r in rows}):,} unique lesions")

    assign = split_lesions(rows, ratios, a.seed)
    written = write_manifests(rows, assign, Path(a.out), a.prefix)

    print("\n▶ split (grouped by lesion, stratified by diagnosis)")
    for split, (path, sel) in written.items():
        lesions = len({r["lesion_id"] for r in sel})
        print(f"  {split:<5} {len(sel):>6,} images  {lesions:>6,} lesions  -> {path}")

    print("\n▶ leakage check")
    if not verify(written):
        sys.exit("REFUSING: the split leaks. Do not train on this.")
    print("\n✓ manifests are leak-free. Next: python scripts/materialize_images.py")


if __name__ == "__main__":
    main()
