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

    # per-class recall + confusion matrix: what actually matters once n is large
    classes = list(getattr(model, "classes", [])) or sorted({e["true"] for e in per_example})
    idx = {c: i for i, c in enumerate(classes)}
    cm = [[0] * len(classes) for _ in classes]
    for e in per_example:
        if e["true"] in idx and e["pred"] in idx:
            cm[idx[e["true"]]][idx[e["pred"]]] += 1
    support = [sum(row) for row in cm]
    recall = {c: (round(cm[i][i] / support[i], 4) if support[i] else None)
              for i, c in enumerate(classes)}
    seen = [recall[c] for c in classes if recall[c] is not None]
    balanced_acc = round(sum(seen) / len(seen), 4) if seen else None

    return Finding(
        pillar="performance",
        metric="top1_accuracy",
        domain="image",
        value={"accuracy": round(acc, 4) if acc is not None else None, "n": n,
               "correct": correct, "balanced_accuracy": balanced_acc,
               "per_class_recall": recall, "support": dict(zip(classes, support))},
        verdict=verdict,
        summary=(f"Top-1 accuracy {acc*100:.0f}% on {n} labeled examples. {note}"
                 if acc is not None else "No labeled examples."),
        details=_details(per_example, classes, cm, support, recall, n),
    )


# One bar per image is readable for a handful of examples and useless for a
# thousand. Past that a confusion matrix and per-class recall say far more — an
# average of 80% hides which classes are actually being missed.
PER_EXAMPLE_CHART_LIMIT = 50

_EXPLAIN = {
    "what": ("Top-1 accuracy is the share of images where the model's most confident class "
             "is the correct one — the plainest measure of 'does it get the answer right'."),
    "how": ("Each bar is one image. The height is how confident the model was in its single "
            "best guess. Green means that guess was correct, red means it was wrong — so a "
            "tall red bar is the worst case: confidently wrong. Hover a bar to see the "
            "predicted and true class."),
    "limits": ("Accuracy alone hides *which* classes fail. A model can look good overall "
               "while missing most melanomas, because melanoma is rare in the data. It also "
               "says nothing about whether the accuracy is evenly spread across patient "
               "groups — that is what the fairness pillar is for."),
}


def _details(per_example, classes, cm, support, recall, n) -> dict[str, Any]:
    explain = dict(_EXPLAIN)

    if n <= PER_EXAMPLE_CHART_LIMIT:
        return {
            "explain": explain,
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
        }

    # row-normalised: each row is one true class, so the diagonal is its recall
    z = [[(c / s if s else 0.0) for c in row] for row, s in zip(cm, support)]
    hover = [[f"true {classes[i]}<br>predicted {classes[j]}<br>{cm[i][j]} of {support[i]}"
              for j in range(len(classes))] for i in range(len(classes))]
    explain["how"] = (
        "The matrix reads row by row: each row is one true diagnosis, and the columns show "
        "what the model called those images. A perfect model is a bright diagonal. Bright "
        "cells *off* the diagonal are the confusions that matter — read the melanoma row to "
        "see what melanomas get mistaken for. The bar chart below gives each class's recall "
        "next to its support, because a class with only a handful of images produces a noisy "
        "percentage.")
    return {
        "explain": explain,
        "per_example": per_example,
        "chart": {
            "kind": "heatmap",
            "title": "Confusion matrix (row-normalised: each row is one true class)",
            "z": z, "x": classes, "y": classes, "text": hover, "zmin": 0, "zmax": 1,
            "x_title": "Predicted", "y_title": "True",
        },
        "chart2": {
            "kind": "bar", "title": "Recall per class (share of that class the model finds)",
            "x": classes, "y": [recall[c] or 0.0 for c in classes], "color": "#5B3FD6",
            "x_title": "Class", "y_title": "Recall",
            "hover": [f"recall {recall[c]} on {support[i]} images"
                      for i, c in enumerate(classes)],
        },
    }
