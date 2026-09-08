"""Uncertainty helpers shared by the metrics.

A number without an interval is not a claim. `dermatofibroma` has 13 images in
the test set: a recall of 0.77 there and a recall of 0.77 on 1,009 nevi are not
the same statement, and only the interval says so.

Pure Python + math — no scipy, so the engine stays light.
"""
from __future__ import annotations

import math

Z95 = 1.959963984540054          # two-sided 95%


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation, which misbehaves precisely where
    these metrics live: small samples and proportions near 0 or 1 (where it can
    produce bounds outside [0, 1], or a zero-width interval at exactly 0 or 1).
    """
    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def auc_ci(auc: float, n_pos: int, n_neg: int, z: float = Z95) -> tuple[float, float] | None:
    """Hanley–McNeil interval for an AUC.

    Approximate — it assumes an exponential score distribution — but adequate for
    saying whether a membership-inference result is distinguishable from 0.5.
    """
    if n_pos <= 0 or n_neg <= 0:
        return None
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return (round(max(0.0, auc - z * se), 4), round(min(1.0, auc + z * se), 4))


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float | None:
    """Positive predictive value at a *stated* prevalence, via Bayes.

    Precision measured on a test set is PPV at *that set's* prevalence, which is
    rarely the prevalence where the model gets deployed. A screening clinic and a
    referral centre see very different case mixes, and PPV moves a long way
    between them while sensitivity and specificity do not.
    """
    if not (0.0 < prevalence < 1.0):
        return None
    tp = sensitivity * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    return round(tp / (tp + fp), 4) if (tp + fp) > 0 else None


def fmt(value: float | None, ci: tuple[float, float] | None) -> str:
    """'0.638 [0.56–0.71]' — the form a result should be quoted in."""
    if value is None:
        return "n/a"
    return f"{value:.3f}" + (f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else "")
