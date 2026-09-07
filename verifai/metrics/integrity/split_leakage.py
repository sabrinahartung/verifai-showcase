"""Integrity (any domain): is the test set actually held out?

Signature: run(model, dataset, ctx) -> Finding

Every other number in the report is conditional on this one. If the test set
shares lesions or images with the training set, the accuracy below it is a
memorisation check wearing a benchmark's clothes — which is exactly what happened
to the dataset this project started from (80% of its test image_ids also appeared
in train).

Uses the same `core.integrity.audit_split` the runner uses as a precondition, so
the published finding and the guard can never disagree.
"""
from __future__ import annotations

from typing import Any

from verifai.core.findings import Finding
from verifai.core.integrity import audit_split, train_manifests_from_scenario

EXPLAIN = {
    "what": ("A test set only means something if the model has never seen it. This "
             "compares the evaluation set against everything the model trained on — "
             "not just image by image, but by lesion, because the same lesion is "
             "often photographed several times and a second photo of a memorised "
             "lesion is not a fair question."),
    "how": ("The marker sits on the share of test images that are contaminated. At 0 "
            "the split is clean and every other number in this report can be read at "
            "face value. Anything above 0 means part of the score is recall rather "
            "than generalisation, and the accuracy above is inflated by roughly that "
            "share."),
    "limits": ("This checks identifiers, not pixels. Two different photographs of "
               "genuinely different lesions that happen to look alike are not caught, "
               "and neither is duplication that carries no shared lesion_id. A clean "
               "result here means no *declared* overlap, which is a floor on "
               "trustworthiness, not a ceiling."),
}


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    scenario = ctx.get("scenario", {}) or {}
    test_manifest = (dataset.meta or {}).get("manifest")
    train_manifests = train_manifests_from_scenario(scenario)

    # Honest about what it cannot check, in the same spirit as privacy/mia.py.
    if not test_manifest or not train_manifests:
        missing = "the dataset was not loaded from a manifest" if not test_manifest \
            else "no training manifests are declared"
        return Finding(
            pillar="integrity", metric="split_leakage", domain=scenario.get("domain", "image"),
            value={"status": "not_checkable", "reason": missing},
            verdict="info",
            summary=(f"Split integrity could not be verified: {missing}. Every number in "
                     f"this report therefore rests on an unverified assumption — that the "
                     f"model never saw this data."),
            details={"explain": EXPLAIN,
                     "note": "Declare integrity.train_manifests (or a training: block) to enable."},
        )

    cfg = scenario.get("integrity") or {}
    a = audit_split(test_manifest, train_manifests,
                    group_key=cfg.get("group_key", "lesion_id"),
                    id_key=cfg.get("id_key", "image_id"))

    if not a["verifiable"]:
        return Finding(
            pillar="integrity", metric="split_leakage", domain=scenario.get("domain", "image"),
            value=dict(a, status="not_checkable"),
            verdict="info",
            summary=("Split integrity could not be verified: the manifests carry no shared "
                     "identifier (image_id / lesion_id) to compare on. This is not a clean "
                     "bill of health — it means the question was never answered, so every "
                     "number in this report rests on the assumption that the model never "
                     "saw this data."),
            details={"explain": EXPLAIN,
                     "train_manifests": [str(m) for m in train_manifests],
                     "test_manifest": str(test_manifest)},
        )

    pct = a["contamination"] * 100
    if a["clean"]:
        verdict, summary = "pass", (
            f"Clean split: none of the {a['n_test']:,} test images shares a lesion or an "
            f"image with the {a['n_train']:,} the model trained on.")
    else:
        verdict = "fail" if pct >= 1 else "warn"
        summary = (
            f"{a['affected_rows']:,} of {a['n_test']:,} test images ({pct:.1f}%) were seen "
            f"in training: {a['shared_ids']:,} identical images and {a['shared_groups']:,} "
            f"shared lesions. Accuracy on this split is inflated by roughly that share.")

    return Finding(
        pillar="integrity", metric="split_leakage", domain=scenario.get("domain", "image"),
        value=a, verdict=verdict, summary=summary,
        details={
            "explain": EXPLAIN,
            "train_manifests": [str(m) for m in train_manifests],
            "test_manifest": str(test_manifest),
            "chart": {
                "kind": "scale", "title": "Share of the test set seen during training",
                "value": round(pct, 2), "min": 0, "max": 100,
                "ticks": [0, 1, 10, 100], "tick_labels": ["0%", "1%", "10%", "100%"],
                "bands": [
                    {"to": 1, "label": "clean", "color": "#CDE8D5"},
                    {"to": 10, "label": "contaminated", "color": "#FAECC8"},
                    {"to": 100, "label": "invalid", "color": "#F5D3CE"},
                ],
                "value_label": ("0% means every other number in this report can be read at "
                                "face value. Above 0, part of the score is memorisation."),
            },
        },
    )
