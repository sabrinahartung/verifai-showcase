"""Robustness (image): stability under common corruptions.

Signature: run(model, dataset, ctx) -> Finding

For each image we compare the clean top-1 prediction to the prediction under
each corruption (noise, blur, brightness, JPEG). We report:
  - PREDICTION STABILITY: share of images whose top-1 class is unchanged,
  - mean confidence of the clean-top class after each corruption.

A model that flips its call under mild, clinically-irrelevant perturbations is
fragile — a real concern for a decision-support tool.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verifai.core.findings import Finding
from verifai.metrics._common import CORRUPTIONS
from verifai.metrics._stats import fmt, wilson


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    seed = ctx.get("seed", 42)
    rng = np.random.default_rng(seed)

    names = list(CORRUPTIONS)
    stable = {c: 0 for c in names}
    conf_after = {c: [] for c in names}
    n = 0

    for s in dataset:
        img = dataset.load(s)
        clean = model.predict_probs(img)
        clean_top = max(clean, key=clean.get)
        n += 1
        for c in names:
            corrupted = CORRUPTIONS[c](img, rng=rng) if c == "noise" else CORRUPTIONS[c](img)
            probs = model.predict_probs(corrupted)
            if max(probs, key=probs.get) == clean_top:
                stable[c] += 1
            conf_after[c].append(probs[clean_top])

    stability = {c: round(stable[c] / n, 3) for c in names} if n else {}
    stability_ci = {c: wilson(stable[c], n) for c in names} if n else {}
    mean_stability = round(float(np.mean(list(stability.values()))), 3) if stability else None

    verdict = "info"
    if n >= 20 and mean_stability is not None:
        verdict = "pass" if mean_stability >= 0.85 else ("warn" if mean_stability >= 0.7 else "fail")

    note = "" if n >= 20 else f" Small sample (n={n}) — illustrative only."
    return Finding(
        pillar="robustness", metric="corruption_stability", domain="image",
        value={"prediction_stability": stability, "stability_ci": stability_ci,
               "mean_stability": mean_stability, "n": n},
        verdict=verdict,
        summary=(f"The prediction stays stable under corruptions for {mean_stability*100:.0f}% "
                 f"of the images on average.{note}" if mean_stability is not None
                 else "No images evaluated."),
        details={
            "explain": {
                "what": ("Real images are never clean: sensor noise, a slightly out-of-"
                         "focus shot, harsh lighting, heavy JPEG compression. None of that "
                         "changes the diagnosis, so the model's answer should not change "
                         "either. This applies each distortion and checks whether it does."),
                "how": ("Each bar is one kind of distortion. The height is the share of "
                        "images whose top-1 class stayed the same after it was applied — "
                        "1.0 means the model never changed its mind, 0.5 means it flipped "
                        "on half the images. Short bars point at the distortion this model "
                        "is most brittle against."),
                "limits": ("Stability is not correctness: a model that is confidently wrong "
                           "both before and after a distortion scores a perfect 1.0 here. "
                           "Read this next to the performance pillar, never on its own."),
            },
            "chart": {
                "kind": "bar", "title": "Prediction stability per corruption",
                "x": names, "y": [stability[c] for c in names], "color": "#1F8A70",
                "y_lo": [stability_ci[c][0] for c in names],
                "y_hi": [stability_ci[c][1] for c in names],
                "hover": [f"{stability[c]:.3f} "
                          f"[{stability_ci[c][0]:.2f}-{stability_ci[c][1]:.2f}] of {n}"
                          for c in names],
                "x_title": "Corruption", "y_title": "Share of unchanged top-1",
            },
        },
    )
