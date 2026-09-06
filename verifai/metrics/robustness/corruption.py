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
    mean_stability = round(float(np.mean(list(stability.values()))), 3) if stability else None

    verdict = "info"
    if n >= 20 and mean_stability is not None:
        verdict = "pass" if mean_stability >= 0.85 else ("warn" if mean_stability >= 0.7 else "fail")

    note = "" if n >= 20 else f" Kleine Stichprobe (n={n}) — illustrativ."
    return Finding(
        pillar="robustness", metric="corruption_stability", domain="image",
        value={"prediction_stability": stability, "mean_stability": mean_stability, "n": n},
        verdict=verdict,
        summary=(f"Vorhersage bleibt im Schnitt bei {mean_stability*100:.0f}% der Bilder "
                 f"unter Störungen stabil.{note}" if mean_stability is not None
                 else "Keine Bilder ausgewertet."),
        details={
            "chart": {
                "kind": "bar", "title": "Vorhersage-Stabilität je Störung",
                "x": names, "y": [stability[c] for c in names], "color": "#1F8A70",
                "x_title": "Störung", "y_title": "Anteil unveränderter Top-1",
            },
        },
    )
