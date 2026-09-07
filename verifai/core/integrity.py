"""Split-integrity auditing — is this evaluation worth believing at all?

HAM10000 photographs the same lesion repeatedly, so a split made on rows leaks
even when every image_id is unique. The dataset this project started from splits
that way: 80% of its test image_ids also appear in train. Every accuracy computed
on it was really a memorisation check.

This module is the single implementation of that check. `core/run.py` calls it as
a precondition (refusing to evaluate a contaminated split) and
`metrics/integrity/split_leakage.py` turns the same result into a published
finding — so the guard and the report can never disagree.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _read(manifest: str | Path, keys: tuple[str, ...]) -> list[dict[str, str]]:
    path = _resolve(manifest)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: (row.get(k) or "").strip() for k in keys} for row in csv.DictReader(f)]


def audit_split(test_manifest: str | Path,
                train_manifests: list[str | Path],
                group_key: str = "lesion_id",
                id_key: str = "image_id") -> dict[str, Any]:
    """Compare a test manifest against everything the model was trained on.

    Two kinds of contamination, in increasing severity:
      - `shared_groups`: the same lesion appears on both sides (a different photo
        of a lesion the model already learned),
      - `shared_ids`: literally the same image.

    Returns counts plus `contamination` — the share of test rows touched by either.
    """
    test = _read(test_manifest, (id_key, group_key))
    train: list[dict[str, str]] = []
    for m in train_manifests:
        train.extend(_read(m, (id_key, group_key)))

    train_groups = {r[group_key] for r in train if r[group_key]}
    train_ids = {r[id_key] for r in train if r[id_key]}

    shared_groups = sorted({r[group_key] for r in test
                            if r[group_key] and r[group_key] in train_groups})
    shared_ids = sorted({r[id_key] for r in test
                         if r[id_key] and r[id_key] in train_ids})
    affected = [r for r in test
                if (r[group_key] and r[group_key] in train_groups)
                or (r[id_key] and r[id_key] in train_ids)]

    n = len(test)
    has_groups = any(r[group_key] for r in test) and any(r[group_key] for r in train)
    has_ids = any(r[id_key] for r in test) and any(r[id_key] for r in train)
    # Without a usable identifier on both sides nothing was actually compared.
    # Reporting that as "clean" would publish an unverified split as a verified
    # one — the precise failure this pillar exists to catch.
    verifiable = has_groups or has_ids

    return {
        "n_test": n,
        "n_train": len(train),
        "group_key": group_key if has_groups else None,
        "id_key": id_key if has_ids else None,
        "shared_groups": len(shared_groups),
        "shared_ids": len(shared_ids),
        "affected_rows": len(affected),
        "contamination": round(len(affected) / n, 4) if n else 0.0,
        "verifiable": verifiable,
        "clean": (not shared_groups and not shared_ids) if verifiable else None,
        # a few examples, so a failure is actionable rather than just a number
        "example_shared_groups": shared_groups[:5],
        "example_shared_ids": shared_ids[:5],
    }


def train_manifests_from_scenario(scenario: dict[str, Any]) -> list[str]:
    """Where to look for 'what the model was trained on', by convention.

    Explicit `integrity.train_manifests` wins; otherwise derive from the
    `training:` block, which already names the manifest prefix and directory.
    """
    integ = scenario.get("integrity") or {}
    if integ.get("train_manifests"):
        return list(integ["train_manifests"])
    tcfg = scenario.get("training") or {}
    prefix = tcfg.get("manifest_prefix")
    if not prefix:
        return []
    d = tcfg.get("manifest_dir", "data/manifests")
    # validation counts as seen: the model was selected on it
    return [f"{d}/{prefix}_train.csv", f"{d}/{prefix}_val.csv"]
