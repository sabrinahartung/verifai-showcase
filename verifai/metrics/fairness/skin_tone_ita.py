"""Fairness (image): skin-tone coverage & subgroup accuracy via ITA.

Signature: run(model, dataset, ctx) -> Finding

HAM10000 has no skin-type labels, so we estimate the Individual Typology Angle
(ITA) from each image and bin it into coarse skin-tone groups — a label-free way
to probe fairness. Two things are reported:

  - COVERAGE: how the sample distributes across skin tones. HAM10000 is known to
    skew light; a skewed sample is itself a documented fairness limitation.
  - SUBGROUP ACCURACY: top-1 accuracy per skin-tone bin *when the sample is big
    enough*. With only a few images per bin this is not claimed as a result —
    the summary says so. Run the full subset (notebook) for a real gap.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from verifai.core.findings import Finding
from verifai.metrics._common import estimate_ita, ita_bin, ITA_BINS


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    bins_order = [b[0] for b in ITA_BINS]
    per_bin_total = defaultdict(int)
    per_bin_correct = defaultdict(int)
    per_example = []

    for s in dataset:
        img = dataset.load(s)
        ita = estimate_ita(img)
        b = ita_bin(ita)
        per_bin_total[b] += 1
        entry = {"id": s.id, "ita": round(ita, 1), "bin": b, "true": s.label}
        if s.label is not None:
            probs = model.predict_probs(img)
            top = max(probs, key=probs.get)
            ok = (top == s.label)
            per_bin_correct[b] += int(ok)
            entry["pred"] = top
            entry["correct"] = ok
        per_example.append(entry)

    coverage = [per_bin_total.get(b, 0) for b in bins_order]
    n = len(per_example)
    # smallest bin count among *populated* bins tells us if accuracy is meaningful
    populated = [c for c in coverage if c > 0]
    enough_per_bin = populated and min(populated) >= 10

    # coverage skew: is any real skin tone essentially absent?
    dark_share = per_bin_total.get("dunkel (V–VI)", 0) / n if n else 0
    verdict = "warn" if dark_share < 0.15 else "info"

    summary = (f"Hauttyp-Abdeckung (ITA-geschätzt) über n={n}: "
               + ", ".join(f"{b.split(' ')[0]} {per_bin_total.get(b,0)}" for b in bins_order)
               + ". Die Stichprobe ist klein und schräg zu hellen Hauttypen — eine "
                 "dokumentierte HAM10000-Limitation. Für eine belastbare Subgruppen-"
                 "Trefferquote den vollen Subset-Lauf nutzen.")

    chart_bar = {
        "kind": "bar", "title": "Hauttyp-Abdeckung der Stichprobe (ITA-Bins)",
        "x": [b for b in bins_order], "y": coverage, "color": "#C77700",
        "x_title": "geschätzter Hauttyp", "y_title": "Anzahl Bilder",
    }

    details: dict[str, Any] = {"per_example": per_example, "chart": chart_bar,
                               "enough_per_bin": bool(enough_per_bin)}
    value: dict[str, Any] = {"coverage": dict(zip(bins_order, coverage)), "n": n}

    if enough_per_bin:
        acc = {b: round(per_bin_correct[b] / per_bin_total[b], 3)
               for b in bins_order if per_bin_total.get(b)}
        vals = list(acc.values())
        gap = round(max(vals) - min(vals), 3) if len(vals) > 1 else 0.0
        value["subgroup_accuracy"] = acc
        value["accuracy_gap"] = gap
        verdict = "fail" if gap > 0.15 else ("warn" if gap > 0.08 else "pass")
        details["chart2"] = {
            "kind": "bar", "title": "Trefferquote nach Hauttyp (ITA)",
            "x": list(acc.keys()), "y": list(acc.values()), "color": "#5B3FD6",
            "x_title": "geschätzter Hauttyp", "y_title": "Top-1-Trefferquote",
        }
        summary = (f"Subgruppen-Trefferquote nach ITA-Hauttyp; größter Abstand "
                   f"{gap*100:.0f} Punkte (n={n}).")

    return Finding(
        pillar="fairness", metric="skin_tone_ita", domain="image",
        value=value, verdict=verdict, summary=summary, details=details,
    )
