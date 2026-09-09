#!/usr/bin/env python
"""Choose a cost-sensitive decision rule — on validation, never on test.

`argmax` maximises expected accuracy, which on imbalanced data means quietly
under-calling the rare classes. Weighting a class scales its probability by the
cost of missing it, so melanoma can win on 0.3 against a nevus on 0.5 — the model
is untouched, only the rule that reads it changes.

The weight has to be chosen on data the model was selected on anyway (validation).
Sweeping it on the test set and then reporting test numbers would be fitting the
decision rule to the test set — a quieter form of the leakage this project exists
to catch, and it would not show up in the integrity check.

Inference runs once; the sweep afterwards is pure arithmetic.

Usage:
    python scripts/tune_decision.py scenarios/skin_cancer_clean.yaml --class melanoma
    python scripts/tune_decision.py scenarios/skin_cancer_clean.yaml --class melanoma --target 0.85
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from verifai.datasets.loaders import load_image_manifest   # noqa: E402
from verifai.models.image import load as load_model        # noqa: E402
from verifai.metrics._stats import wilson                  # noqa: E402


def sweep(probs_and_labels, target_class: str, weights: list[float]):
    """Sensitivity / PPV / accuracy for the target class at each weight."""
    rows = []
    for w in weights:
        tp = fn = fp = correct = 0
        for probs, label in probs_and_labels:
            pred = max(probs, key=lambda c: probs[c] * (w if c == target_class else 1.0))
            correct += int(pred == label)
            if label == target_class:
                tp += int(pred == target_class)
                fn += int(pred != target_class)
            elif pred == target_class:
                fp += 1
        n = len(probs_and_labels)
        rows.append({
            "weight": w,
            "sensitivity": tp / (tp + fn) if (tp + fn) else None,
            "sens_ci": wilson(tp, tp + fn),
            "ppv": tp / (tp + fp) if (tp + fp) else None,
            "accuracy": correct / n if n else None,
            "false_alarms": fp,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--class", dest="target", required=True, help="class to protect")
    ap.add_argument("--target", dest="goal", type=float, default=None,
                    help="desired sensitivity, e.g. 0.85")
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="tune here — never on test")
    ap.add_argument("--grid", default=None, help="comma-separated weights to try")
    a = ap.parse_args()

    sc = yaml.safe_load(Path(a.scenario).read_text(encoding="utf-8"))
    tcfg = sc.get("training") or {}
    prefix = tcfg.get("manifest_prefix")
    if not prefix:
        sys.exit("scenario has no training.manifest_prefix, so the tuning split is unknown")
    man = f"{tcfg.get('manifest_dir', 'data/manifests')}/{prefix}_{a.split}.csv"

    base = {k: v for k, v in sc["dataset"].items() if k != "loader"}
    ds = load_image_manifest({**base, "manifest": man})
    model = load_model(sc["model"])
    if a.target not in model.classes:
        sys.exit(f"{a.target!r} is not one of {model.classes}")

    print(f"▶ tuning on {man} ({len(ds)} images) — the test manifest is not read")
    data = [(model.predict_probs(ds.load(s)), s.label) for s in ds if s.label]
    print(f"  inference done on {len(data)} labeled images\n")

    grid = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
    if a.grid:
        grid = [float(x) for x in a.grid.split(",")]
    rows = sweep(data, a.target, grid)

    print(f"{'weight':>7} {'sensitivity':>22} {'PPV':>7} {'accuracy':>9} {'false alarms':>13}")
    for r in rows:
        ci = r["sens_ci"]
        cis = f"[{ci[0]:.2f}-{ci[1]:.2f}]" if ci else ""
        print(f"  {r['weight']:>5.2f} {r['sensitivity']:>8.3f} {cis:>13} "
              f"{r['ppv']:>7.3f} {r['accuracy']:>9.3f} {r['false_alarms']:>13}")

    if a.goal:
        hit = next((r for r in rows if r["sensitivity"] and r["sensitivity"] >= a.goal), None)
        base_row = rows[0]
        print()
        if not hit:
            print(f"✗ no weight in the grid reaches sensitivity {a.goal:.2f} for {a.target}.")
        else:
            print(f"✓ weight {hit['weight']:g} reaches {a.target} sensitivity "
                  f"{hit['sensitivity']:.3f} (target {a.goal:.2f}) on {a.split}.")
            print(f"  cost: PPV {base_row['ppv']:.3f} -> {hit['ppv']:.3f}, "
                  f"overall accuracy {base_row['accuracy']:.3f} -> {hit['accuracy']:.3f}, "
                  f"false alarms {base_row['false_alarms']} -> {hit['false_alarms']}")
            print(f"\n  Put this in the scenario's model: block, then re-run run_scenario.py:\n"
                  f"    decision_weights:\n      {a.target}: {hit['weight']:g}")


if __name__ == "__main__":
    main()
