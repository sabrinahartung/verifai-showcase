"""Performance (image): top-1 accuracy + per-example confidence.

Signature: run(model, dataset, ctx) -> Finding

Honest about sample size: with a handful of labeled examples this is a sanity
check, not a benchmark. The summary always states n. Run the larger subset
(free-GPU notebook) for a statistically meaningful number.
"""
from __future__ import annotations

from typing import Any

from verifai.core.findings import Finding


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    labeled = [s for s in dataset if s.label is not None]
    per_example = []
    correct = 0
    for s in labeled:
        probs = model.predict_probs(dataset.load(s))
        top = max(probs, key=probs.get)
        ok = (top == s.label)
        correct += int(ok)
        per_example.append({
            "id": s.id, "true": s.label, "pred": top,
            "confidence": round(probs[top], 4), "correct": ok,
        })

    n = len(labeled)
    acc = correct / n if n else None

    verdict = "info"
    if n >= 30:  # only claim a verdict once the sample is big enough to mean something
        verdict = "pass" if acc >= 0.75 else ("warn" if acc >= 0.6 else "fail")

    note = (f"Small sample (n={n}) — a plausibility check, not a benchmark."
            if n < 30 else f"n={n}.")

    return Finding(
        pillar="performance",
        metric="top1_accuracy",
        domain="image",
        value={"accuracy": round(acc, 4) if acc is not None else None, "n": n,
               "correct": correct},
        verdict=verdict,
        summary=(f"Top-1 accuracy {acc*100:.0f}% on {n} labeled examples. {note}"
                 if acc is not None else "No labeled examples."),
        details={
            "explain": {
                "what": ("Top-1 accuracy is the share of images where the model's most "
                         "confident class is the correct one — the plainest measure of "
                         "'does it get the answer right'."),
                "how": ("Each bar is one image. The height is how confident the model was "
                        "in its single best guess. Green means that guess was correct, red "
                        "means it was wrong — so a tall red bar is the worst case: "
                        "confidently wrong. Hover a bar to see the predicted and true class."),
                "limits": ("Accuracy alone hides *which* classes fail. A model can look "
                           "good overall while missing most melanomas, because melanoma is "
                           "rare in the data. It also says nothing about whether the "
                           "accuracy is evenly spread across patient groups — that is what "
                           "the fairness pillar is for."),
            },
            "per_example": per_example,
            "chart": {
                "kind": "bar",
                "title": "Confidence of the top prediction per example",
                "x": [e["id"] for e in per_example],
                "y": [e["confidence"] for e in per_example],
                "colors": ["#2E9E5B" if e["correct"] else "#C0392B" for e in per_example],
                "x_title": "Example", "y_title": "Confidence (top class)",
                "hover": [f"{e['pred']} (true: {e['true']})" for e in per_example],
            },
        },
    )
