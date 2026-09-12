#!/usr/bin/env python
"""Build an ISIC 2019 training manifest that excludes our frozen val/test lesions.

Why this exists
---------------
ISIC 2019 *incorporates* HAM10000, so training on all of it would train on
`ham10000_test.csv` — the exact mistake documented in docs/ROADMAP.md, and the
one `_enforce_split_integrity` exists to refuse. But the fix is not "remove
HAM10000 from ISIC": that would throw away 7,014 usable training images.

Only the **val and test** lesions have to be held back — 3,001 images, 11.8% of
ISIC 2019 and 7.5% of its melanoma. Everything else is fair training data,
including HAM10000's own training portion.

The join is exact rather than approximate: HAM10000's `image_id` *is* an ISIC id
(`ISIC_0024342`), so this is a string anti-join, not image hashing. And because
our three manifests sum to exactly 10,015 — the whole of HAM10000 — every image
of every held-back lesion is already known to us by ISIC id.

What this preserves
-------------------
`ham10000_test.csv` is not touched, so its content hash is unchanged, so a run
against the new model lands in the *same* comparability group as the five
existing configurations and can be ranked against them. That is the whole point:
a bigger training set is only interesting if the comparison survives it.

Validation is the frozen HAM10000 val set, copied under the new prefix rather
than resampled. `tune_decision.py` tunes the decision rule on val, so changing
val would silently change what the tuned threshold means.

What this cannot verify
-----------------------
Exclusion is by `image_id` and by `lesion_id`. If ISIC holds another photograph
of a held-back lesion contributed from a different source under an unrelated id
and no shared lesion_id, nothing here can detect it. The script reports how much
of ISIC carries a lesion_id at all, so the residual risk is stated rather than
assumed away — the same `verifiable` vs `clean` distinction `audit_split` makes.

Usage:
    python scripts/build_isic_train.py \
        --groundtruth data/raw/isic2019/ISIC_2019_Training_GroundTruth.csv \
        --metadata    data/raw/isic2019/ISIC_2019_Training_Metadata.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ISIC's short codes -> the label vocabulary already in data/manifests/*.csv.
# Spelling matters: these strings are the model's class names, and `classes` in a
# scenario must match the sorted order of whatever ends up in the manifest.
ISIC_TO_LABEL = {
    "MEL": "melanoma",
    "NV": "melanocytic_Nevi",
    "BCC": "basal_cell_carcinoma",
    "AK": "actinic_keratoses",
    "BKL": "benign_keratosis-like_lesions",
    "DF": "dermatofibroma",
    "VASC": "vascular_lesions",
    # SCC has no HAM10000 equivalent. Dropped unless --keep-scc, because the
    # frozen test set has seven classes: an eighth head could never be scored on
    # it, and would make the new model incomparable for no measurable gain.
    "SCC": "squamous_cell_carcinoma",
}
# UNK is "none of the above", not a diagnosis. Never a training label.
ALWAYS_DROP = {"UNK"}

MANIFEST_FIELDS = ["filename", "image_id", "lesion_id", "label",
                   "dx_type", "age", "sex", "localization"]

DEFAULT_EXCLUDE = ["data/manifests/ham10000_val.csv",
                   "data/manifests/ham10000_test.csv"]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO / path)


def read_groundtruth(path: Path, keep_scc: bool) -> tuple[dict[str, str], dict[str, int]]:
    """One-hot ISIC ground truth -> {image_id: label}, plus a drop tally.

    ISIC_2019_Training_GroundTruth.csv is one column per class with a 1.0 in the
    winning column. Reading it as one-hot rather than trusting a single
    `diagnosis` column keeps this working on the challenge files as published.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        id_col = next((c for c in ("image", "image_id", "isic_id") if c in cols), None)
        if id_col is None:
            sys.exit(f"{path}: no image id column (looked for image/image_id/isic_id); got {cols}")
        class_cols = [c for c in cols if c.upper() in ISIC_TO_LABEL or c.upper() in ALWAYS_DROP]
        if not class_cols:
            sys.exit(f"{path}: no recognisable class columns in {cols}")

        labels: dict[str, str] = {}
        dropped: dict[str, int] = defaultdict(int)
        for row in reader:
            image_id = (row.get(id_col) or "").strip()
            if not image_id:
                continue
            hot = [c for c in class_cols if _is_one(row.get(c))]
            if len(hot) != 1:
                # No winner or several: not a usable training label. Counted, not guessed.
                dropped["ambiguous or unlabelled"] += 1
                continue
            code = hot[0].upper()
            if code in ALWAYS_DROP:
                dropped[code] += 1
                continue
            if code == "SCC" and not keep_scc:
                dropped["SCC (no HAM10000 equivalent)"] += 1
                continue
            labels[image_id] = ISIC_TO_LABEL[code]
    return labels, dict(dropped)


def _is_one(v) -> bool:
    try:
        return float(v) == 1.0
    except (TypeError, ValueError):
        return False


def read_metadata(path: Path | None) -> dict[str, dict]:
    """{image_id: {lesion_id, age, sex, localization}} — all optional.

    ISIC 2019's metadata leaves `lesion_id` empty for much of the archive. That
    is reported rather than worked around: a missing lesion_id is exactly the
    case where lesion-level exclusion cannot be verified.
    """
    if path is None:
        return {}
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        id_col = next((c for c in ("image", "image_id", "isic_id") if c in cols), None)
        if id_col is None:
            sys.exit(f"{path}: no image id column; got {cols}")
        age_col = next((c for c in ("age_approx", "age") if c in cols), None)
        site_col = next((c for c in ("anatom_site_general", "localization",
                                     "anatom_site_general_challenge") if c in cols), None)
        for row in reader:
            image_id = (row.get(id_col) or "").strip()
            if not image_id:
                continue
            out[image_id] = {
                "lesion_id": (row.get("lesion_id") or "").strip(),
                "age": (row.get(age_col) or "").strip() if age_col else "",
                "sex": (row.get("sex") or "").strip(),
                "localization": (row.get(site_col) or "").strip() if site_col else "",
            }
    return out


def read_exclusions(paths: list[Path]) -> tuple[set[str], set[str]]:
    """Image ids and lesion ids that must never appear in training."""
    images: set[str] = set()
    lesions: set[str] = set()
    for p in paths:
        if not p.exists():
            sys.exit(f"exclusion manifest not found: {p}")
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("image_id"):
                    images.add(row["image_id"].strip())
                if row.get("lesion_id"):
                    lesions.add(row["lesion_id"].strip())
    return images, lesions


def build_rows(labels: dict[str, str], meta: dict[str, dict],
               excl_images: set[str], excl_lesions: set[str]):
    """Apply the anti-join. Returns (kept rows, tally of why rows were dropped)."""
    kept = []
    by_image = by_lesion = 0
    for image_id in sorted(labels):
        m = meta.get(image_id, {})
        lesion_id = m.get("lesion_id", "")
        if image_id in excl_images:
            by_image += 1
            continue
        # Only meaningful when ISIC actually supplies a lesion_id; an empty one
        # is not a match, it is an absence of evidence.
        if lesion_id and lesion_id in excl_lesions:
            by_lesion += 1
            continue
        kept.append({
            "filename": f"{image_id}.jpg",
            "image_id": image_id,
            # Fall back to the image id so every row has a grouping key. Stated
            # in the output: these rows are lesion-unverifiable.
            "lesion_id": lesion_id or image_id,
            "label": labels[image_id],
            "dx_type": "",
            "age": m.get("age", ""),
            "sex": m.get("sex", ""),
            "localization": m.get("localization", ""),
        })
    return kept, {"excluded by image_id": by_image, "excluded by lesion_id": by_lesion}


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["image_id"]):   # deterministic
            w.writerow(r)


def verify(train_path: Path, exclude_paths: list[Path]) -> bool:
    """Re-read what was written and prove the anti-join held.

    Deliberately reads the file back rather than trusting the in-memory set: the
    manifest is the contract, so the manifest is what gets checked.
    """
    with open(train_path, newline="", encoding="utf-8") as f:
        train = list(csv.DictReader(f))
    t_img = {r["image_id"] for r in train if r.get("image_id")}
    t_les = {r["lesion_id"] for r in train if r.get("lesion_id")}
    ok = True
    for p in exclude_paths:
        with open(p, newline="", encoding="utf-8") as f:
            held = list(csv.DictReader(f))
        h_img = {r["image_id"] for r in held if r.get("image_id")}
        h_les = {r["lesion_id"] for r in held if r.get("lesion_id")}
        si, sl = t_img & h_img, t_les & h_les
        flag = "OK " if not (si or sl) else "LEAK"
        ok &= not (si or sl)
        print(f"  [{flag}] train vs {p.name:<22} {len(si)} shared images, {len(sl)} shared lesions")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groundtruth", required=True,
                    help="ISIC_2019_Training_GroundTruth.csv (one-hot labels)")
    ap.add_argument("--metadata", default=None,
                    help="ISIC_2019_Training_Metadata.csv (lesion_id, age, sex, site)")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help="manifests whose lesions/images must never be trained on")
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--prefix", default="isic")
    ap.add_argument("--val-from", default="data/manifests/ham10000_val.csv",
                    help="frozen val manifest to copy under the new prefix")
    ap.add_argument("--keep-scc", action="store_true",
                    help="keep SCC as an 8th class (incomparable with the 7-class test set)")
    a = ap.parse_args()

    gt_path, md_path = _resolve(a.groundtruth), (_resolve(a.metadata) if a.metadata else None)
    excl_paths = [_resolve(p) for p in a.exclude]
    out_dir = _resolve(a.out)

    print(f"▶ reading ISIC labels from {gt_path.name}")
    labels, dropped = read_groundtruth(gt_path, a.keep_scc)
    print(f"  {len(labels):,} labelled images")
    for reason, n in sorted(dropped.items()):
        print(f"  dropped {n:>6,}  {reason}")

    meta = read_metadata(md_path)
    if md_path:
        with_lesion = sum(1 for i in labels if meta.get(i, {}).get("lesion_id"))
        pct = 100 * with_lesion / max(len(labels), 1)
        print(f"▶ metadata from {md_path.name}: {with_lesion:,} of {len(labels):,} "
              f"({pct:.1f}%) carry a lesion_id")
        if pct < 50:
            print("  note: lesion-level exclusion is therefore only partly verifiable.")
            print("  image_id exclusion is exact; a second photo of a held-back lesion")
            print("  under an unrelated id cannot be detected. Say so in the report.")
    else:
        print("▶ no --metadata: excluding by image_id only, and no age/sex/site columns")

    excl_images, excl_lesions = read_exclusions(excl_paths)
    print(f"▶ holding back {len(excl_images):,} image ids / {len(excl_lesions):,} lesion ids "
          f"from {', '.join(p.name for p in excl_paths)}")

    rows, why = build_rows(labels, meta, excl_images, excl_lesions)
    for reason, n in why.items():
        print(f"  {reason}: {n:,}")

    train_path = out_dir / f"{a.prefix}_train.csv"
    write_manifest(rows, train_path)

    # Copy val verbatim rather than resampling: train_model.py reads
    # <prefix>_val.csv, and tune_decision.py tunes on val, so a different val
    # would silently change what a tuned decision weight means.
    val_src = _resolve(a.val_from)
    val_path = out_dir / f"{a.prefix}_val.csv"
    val_path.write_bytes(val_src.read_bytes())

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["label"]] += 1
    print(f"\n▶ wrote {len(rows):,} training images -> {train_path}")
    print(f"  {len({r['lesion_id'] for r in rows}):,} distinct lesion keys")
    for label in sorted(counts):
        print(f"    {label:<32} {counts[label]:>7,}")
    print(f"  classes ({len(counts)}), in the order a scenario must declare:")
    for label in sorted(counts):
        print(f"    - {label}")
    print(f"▶ copied {val_src.name} -> {val_path}  (frozen val, unchanged)")

    print("\n▶ leakage check (the manifest is the contract, so the manifest is checked)")
    if not verify(train_path, excl_paths):
        train_path.unlink(missing_ok=True)
        val_path.unlink(missing_ok=True)
        sys.exit("REFUSING: the training manifest overlaps held-back data. Nothing written.")

    print(f"\n✓ {train_path.name} is disjoint from val and test.")
    print("  Next: materialize the ISIC images, then train with")
    print(f"        training.manifest_prefix: \"{a.prefix}\"")
    print("  Evaluate against data/manifests/ham10000_test.csv — unchanged, so the run")
    print("  stays in the same comparability group as the existing five.")


if __name__ == "__main__":
    main()
