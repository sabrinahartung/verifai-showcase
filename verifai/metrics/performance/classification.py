"""Performance (image): per-class diagnostic accuracy with uncertainty.

Signature: run(model, dataset, ctx) -> Finding

Reports what a clinician would ask for rather than only what a benchmark reports:
top-1 accuracy, but also **per-class sensitivity, specificity and PPV**, top-3
differential accuracy, and a 95% interval on every one of them.

The per-class view is not decoration. This model scores ~0.80 top-1 while finding
only ~0.64 of melanomas: the headline is dominated by the commonest class, and the
rare, dangerous one fails underneath it without moving the average.

Honest about sample size: with a handful of labeled examples this is a sanity
check, not a benchmark. The summary always states n, and no verdict is claimed
below n=30.
"""
from __future__ import annotations

from typing import Any

from verifai.core.findings import Finding
from verifai.metrics._stats import auc_ci, fmt, ppv_at_prevalence, wilson  # noqa: F401

# One bar per image is readable for a handful of examples and useless for a
# thousand. Past this, a confusion matrix and per-class recall say far more.
PER_EXAMPLE_CHART_LIMIT = 50
VERDICT_MIN_N = 30

# Which direction is an improvement, for the comparison view. Patterns may use *.
_BETTER = {
    "accuracy": "higher", "balanced_accuracy": "higher", "top3_accuracy": "higher",
    "per_class.*.sensitivity": "higher", "per_class.*.specificity": "higher",
    "per_class.*.ppv_test_prevalence": "higher",
    "per_class_recall.*": "higher",
}


def _per_class(cm: list[list[int]], classes: list[str]) -> dict[str, dict[str, Any]]:
    """One-vs-rest sensitivity / specificity / PPV per class, each with a 95% CI."""
    total = sum(sum(row) for row in cm)
    out: dict[str, dict[str, Any]] = {}
    for i, c in enumerate(classes):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp                       # true c, called something else
        fp = sum(cm[r][i] for r in range(len(classes))) - tp
        tn = total - tp - fn - fp
        sens_n, spec_n, prec_n = tp + fn, tn + fp, tp + fp
        out[c] = {
            "support": sens_n,
            "sensitivity": round(tp / sens_n, 4) if sens_n else None,
            "sensitivity_ci": wilson(tp, sens_n),
            "specificity": round(tn / spec_n, 4) if spec_n else None,
            "specificity_ci": wilson(tn, spec_n),
            # precision == PPV at *this test set's* prevalence, not at deployment's
            "ppv_test_prevalence": round(tp / prec_n, 4) if prec_n else None,
            "ppv_ci": wilson(tp, prec_n),
        }
    return out


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    classes = list(getattr(model, "classes", []))
    labeled = [s for s in dataset if s.label is not None]

    per_example, correct, top3_correct = [], 0, 0
    idx = {c: i for i, c in enumerate(classes)}
    cm = [[0] * len(classes) for _ in classes]

    for s in labeled:
        probs = model.predict_probs(dataset.load(s))
        ranked = model.rank(probs)          # honours the configured decision rule
        top = ranked[0]
        ok = (top == s.label)
        in3 = s.label in ranked[:3]
        correct += int(ok)
        top3_correct += int(in3)
        if s.label in idx and top in idx:
            cm[idx[s.label]][idx[top]] += 1
        per_example.append({
            "id": s.id, "true": s.label, "pred": top,
            "confidence": round(probs[top], 4), "correct": ok, "in_top3": in3,
        })

    n = len(labeled)
    if not n:
        return Finding(pillar="performance", metric="top1_accuracy", domain="image",
                       value={"accuracy": None, "n": 0}, verdict="info",
                       summary="No labeled examples.", details={"explain": _EXPLAIN})

    acc = correct / n
    acc_ci = wilson(correct, n)
    top3 = top3_correct / n
    top3_ci_ = wilson(top3_correct, n)
    per_class = _per_class(cm, classes)

    sens = [v["sensitivity"] for v in per_class.values() if v["sensitivity"] is not None]
    balanced = round(sum(sens) / len(sens), 4) if sens else None

    # PPV at a stated deployment prevalence, when the scenario declares one
    prevalence = ((ctx.get("scenario", {}) or {}).get("clinical", {}) or {}).get("prevalence") or {}
    for c, p in prevalence.items():
        v = per_class.get(c)
        if v and v["sensitivity"] is not None and v["specificity"] is not None:
            v["stated_prevalence"] = p
            v["ppv_at_stated_prevalence"] = ppv_at_prevalence(
                v["sensitivity"], v["specificity"], float(p))

    verdict = "info"
    if n >= VERDICT_MIN_N:                 # only claim a verdict once n means something
        verdict = "pass" if acc >= 0.75 else ("warn" if acc >= 0.6 else "fail")

    # name the weakest class in the summary — the headline hides it by construction
    scored = [(c, v) for c, v in per_class.items()
              if v["sensitivity"] is not None and v["support"] > 0]
    worst = min(scored, key=lambda kv: kv[1]["sensitivity"])[0] if scored else None
    note = (f" Small sample (n={n}) — a plausibility check, not a benchmark."
            if n < VERDICT_MIN_N else "")
    summary = (f"Top-1 accuracy {fmt(acc, acc_ci)} on {n} labeled examples; "
               f"balanced {balanced}. ")
    if worst:
        w = per_class[worst]
        summary += (f"Weakest class {worst}: sensitivity "
                    f"{fmt(w['sensitivity'], w['sensitivity_ci'])} on {w['support']} images.")
    summary += note

    return Finding(
        pillar="performance", metric="top1_accuracy", domain="image",
        value={"accuracy": round(acc, 4), "accuracy_ci": acc_ci, "n": n, "correct": correct,
               "balanced_accuracy": balanced,
               "top3_accuracy": round(top3, 4), "top3_accuracy_ci": top3_ci_,
               "per_class": per_class,
               "per_class_recall": {c: v["sensitivity"] for c, v in per_class.items()},
               "support": {c: v["support"] for c, v in per_class.items()}},
        verdict=verdict, summary=summary,
        details=_details(per_example, classes, cm, per_class, n),
    )


_EXPLAIN = {
    "what": ("Top-1 accuracy is the share of images whose most confident class is correct. "
             "Reported per class as well, because one averaged number hides the classes that "
             "matter most: a model can look strong overall while missing most melanomas, "
             "since melanoma is rare in the data."),
    "how": ("Each bar is one image. Height is the model's confidence in its single best "
            "guess. Green means that guess was right, red means wrong — a tall red bar is "
            "the worst case, confidently wrong."),
    "limits": ("Every figure carries a 95% interval, and the interval is the point: a class "
               "with 13 images produces a recall whose true value could plausibly sit "
               "anywhere across half the range. Two numbers whose intervals overlap have not "
               "been shown to differ. PPV additionally depends on prevalence — precision "
               "measured here is PPV at *this test set's* case mix, which is rarely the one "
               "at deployment."),
}


def _details(per_example, classes, cm, per_class, n) -> dict[str, Any]:
    explain = dict(_EXPLAIN)

    if n <= PER_EXAMPLE_CHART_LIMIT:
        return {
            "explain": explain, "better": _BETTER, "per_example": per_example,
            "chart": {
                "kind": "bar", "title": "Confidence of the top prediction per example",
                "x": [e["id"] for e in per_example], "y": [e["confidence"] for e in per_example],
                "colors": ["#2E9E5B" if e["correct"] else "#C0392B" for e in per_example],
                "x_title": "Example", "y_title": "Confidence (top class)",
                "hover": [f"{e['pred']} (true: {e['true']})" for e in per_example],
            },
        }

    support = [per_class[c]["support"] for c in classes]
    z = [[(v / s if s else 0.0) for v in row] for row, s in zip(cm, support)]
    hover = [[f"true {classes[i]}<br/>predicted {classes[j]}<br/>{cm[i][j]} of {support[i]}"
              for j in range(len(classes))] for i in range(len(classes))]
    sens = [per_class[c]["sensitivity"] or 0.0 for c in classes]
    ci = [per_class[c]["sensitivity_ci"] or (0.0, 0.0) for c in classes]

    explain["how"] = (
        "The matrix reads row by row: each row is one true diagnosis, the columns are what the "
        "model called those images. A perfect model is a bright diagonal; bright cells *off* it "
        "are the confusions that matter — read the melanoma row to see what melanomas get "
        "mistaken for. Below it, sensitivity per class with its 95% interval. **Compare the "
        "error bars, not the bar heights**: a short bar on a class with a thousand images is a "
        "finding, while a tall bar on a class with thirteen may be luck.")
    return {
        "explain": explain, "better": _BETTER, "per_example": per_example,
        "chart": {
            "kind": "heatmap",
            "title": "Confusion matrix (row-normalised: each row is one true class)",
            "z": z, "x": classes, "y": classes, "text": hover, "zmin": 0, "zmax": 1,
            "x_title": "Predicted", "y_title": "True",
        },
        "chart2": {
            "kind": "bar", "title": "Sensitivity per class, with 95% intervals",
            "x": classes, "y": sens, "color": "#5B3FD6",
            "y_lo": [c[0] for c in ci], "y_hi": [c[1] for c in ci],
            "x_title": "Class", "y_title": "Sensitivity (recall)",
            "hover": [f"{sens[i]:.3f} [{ci[i][0]:.2f}–{ci[i][1]:.2f}] on {support[i]} images"
                      for i in range(len(classes))],
        },
    }
