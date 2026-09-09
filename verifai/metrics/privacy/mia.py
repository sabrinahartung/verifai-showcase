"""Privacy (image): membership-inference risk.

Signature: run(model, dataset, ctx) -> Finding

A membership-inference attack asks: given one image, can an attacker tell whether
that patient was in the training set? The standard cheap attack uses the model's
own confidence — a model tends to be more certain on what it memorised. If members
and non-members separate cleanly, the model leaks who it was trained on.

This needs a members set (images the model trained on) and non-members (images it
never saw). Without both it reports that honestly rather than inventing a number,
which is what happens for scenarios whose split provenance is unknown.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from verifai.core.findings import Finding
from verifai.metrics._stats import auc_ci

EXPLAIN = {
    "what": ("A model is usually more confident on images it memorised during training. "
             "A membership-inference attack exploits exactly that: given one image, it "
             "guesses whether that patient was in the training set. This runs the attack "
             "and reports how well it works."),
    "how": ("The result is an AUC. At 0.5 the attacker does no better than a coin flip, "
            "which is the good case — members and non-members are indistinguishable. "
            "Towards 1.0 the model reliably betrays who was in its training data, which "
            "for medical images is a real patient-privacy problem."),
    "limits": ("This is one cheap, standard attack using confidence alone. A determined "
               "attacker with shadow models or per-class calibration can do better, so a "
               "low score here is evidence of low risk rather than a guarantee of privacy. "
               "It also says nothing about what could be reconstructed from the weights."),
}


def _rank_auc(pos: list[float], neg: list[float]) -> float:
    """AUC via rank statistics, with ties averaged. No sklearn dependency."""
    import numpy as np
    a = np.asarray(pos + neg, dtype=float)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    r = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0        # average rank across the tie group
        i = j + 1
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = r
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _confidences(model, ds, cap: int, seed: int) -> list[float]:
    """Confidence in the *true* class — the attacker's signal."""
    samples = [s for s in ds if s.label is not None]
    rng = random.Random(seed)
    if cap and len(samples) > cap:
        samples = rng.sample(samples, cap)
    out = []
    for s in samples:
        probs = model.predict_probs(ds.load(s))
        out.append(float(probs.get(s.label, 0.0)))
    return out


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    scenario = ctx.get("scenario", {}) or {}
    cfg = scenario.get("privacy") or {}
    members_manifest = cfg.get("members")

    if not members_manifest:
        return Finding(
            pillar="privacy", metric="membership_inference_auc", domain="image",
            value={"mia_auc": None, "status": "requires_members"},
            verdict="info",
            summary=("Membership inference needs training vs. holdout images. This scenario "
                     "declares no members set — so no number is invented here. Point "
                     "privacy.members at the training manifest to enable it."),
            details={"explain": EXPLAIN,
                     "note": "Set scenario.privacy.members (and optionally non_members)."},
        )

    from verifai.datasets.loaders import load_image_manifest

    base = {k: v for k, v in (scenario.get("dataset") or {}).items() if k != "loader"}
    seed = int(ctx.get("seed", 42))
    cap = int(cfg.get("sample", 750))

    member_ds = load_image_manifest({**base, "manifest": members_manifest})
    non_manifest = cfg.get("non_members")
    non_ds = load_image_manifest({**base, "manifest": non_manifest}) if non_manifest else dataset

    member_conf = _confidences(model, member_ds, cap, seed)
    non_conf = _confidences(model, non_ds, cap, seed + 1)

    if len(member_conf) < 50 or len(non_conf) < 50:
        return Finding(
            pillar="privacy", metric="membership_inference_auc", domain="image",
            value={"mia_auc": None, "status": "too_few_samples",
                   "n_members": len(member_conf), "n_non_members": len(non_conf)},
            verdict="info",
            summary=(f"Too few images to run the attack meaningfully "
                     f"({len(member_conf)} members, {len(non_conf)} non-members; 50 needed "
                     f"on each side). No AUC is claimed."),
            details={"explain": EXPLAIN},
        )

    auc = round(_rank_auc(member_conf, non_conf), 4)
    ci = auc_ci(auc, len(member_conf), len(non_conf))
    # A verdict on the interval, not the point estimate: an AUC of 0.59 whose
    # interval reaches 0.68 has not been shown to be low-risk.
    verdict = "pass" if (ci and ci[1] < 0.60) else ("warn" if auc < 0.75 else "fail")
    mean_m = round(sum(member_conf) / len(member_conf), 4)
    mean_n = round(sum(non_conf) / len(non_conf), 4)

    return Finding(
        pillar="privacy", metric="membership_inference_auc", domain="image",
        value={"mia_auc": auc, "mia_auc_ci": ci,
               "n_members": len(member_conf), "n_non_members": len(non_conf),
               "mean_confidence_members": mean_m, "mean_confidence_non_members": mean_n},
        verdict=verdict,
        summary=(f"Membership-inference AUC {auc}"
                 f"{f' [{ci[0]:.2f}-{ci[1]:.2f}]' if ci else ''} "
                 f"from {len(member_conf)} training and "
                 f"{len(non_conf)} held-out images (0.5 = an attacker cannot tell them "
                 f"apart). Mean confidence in the true class was {mean_m} on training "
                 f"images against {mean_n} on unseen ones."),
        details={
            "better": {"mia_auc": "lower"},
            "explain": EXPLAIN,
            "members_manifest": str(members_manifest),
            "non_members_manifest": str(non_manifest or (dataset.meta or {}).get("manifest")),
            "chart": {
                "kind": "scale", "title": "Membership-inference AUC",
                "value": auc, "min": 0.5, "max": 1.0,
                "ticks": [0.5, 0.6, 0.75, 1.0], "tick_labels": ["0.5", "0.6", "0.75", "1.0"],
                "bands": [
                    {"to": 0.60, "label": "low risk", "color": "#CDE8D5"},
                    {"to": 0.75, "label": "moderate", "color": "#FAECC8"},
                    {"to": 1.00, "label": "high risk", "color": "#F5D3CE"},
                ],
                "value_label": ("0.5 means the attacker is only guessing (good). Towards 1.0 "
                                "the model reveals who was in its training set."),
            },
        },
    )
