"""Privacy (image): membership-inference risk.

Signature: run(model, dataset, ctx) -> Finding

A proper membership-inference attack needs a *members* set (training images) and
a *non-members* set (held-out), then tests whether the model's confidence lets an
attacker tell them apart (AUC ~0.5 = no leakage, ->1.0 = leaky).

The showcase's tiny example set has neither split, so this metric does not fake a
number: it reports honestly that it requires the full run (free-GPU notebook,
which has the train/test split) and stays `info`.
"""
from __future__ import annotations

from typing import Any

from verifai.core.findings import Finding


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    members = ctx.get("scenario", {}).get("privacy", {}).get("members")
    if not members:
        return Finding(
            pillar="privacy", metric="membership_inference_auc", domain="image",
            value={"mia_auc": None, "status": "requires_full_run"},
            verdict="info",
            summary=("Membership inference needs training vs. holdout images. The example "
                     "set has no such split — so no number is invented here; the full run "
                     "(GPU notebook) provides the AUC."),
            details={
                "note": "Set scenario.privacy.members/non_members to enable.",
                "explain": {
                    "what": ("A model tends to be more confident on images it memorised "
                             "during training. A membership-inference attack exploits "
                             "exactly that: given one image, it guesses whether that "
                             "patient was in the training set. This metric measures how "
                             "well such an attack would work."),
                    "how": ("The result is an AUC between 0.5 and 1.0. At 0.5 the attacker "
                            "does no better than a coin flip, which is the good case. "
                            "Towards 1.0 the model reliably betrays who was in its "
                            "training data — a genuine patient-privacy problem."),
                    "limits": ("Nothing is computed here, and no number is shown, because "
                               "the attack needs images the model trained on *and* images "
                               "it never saw. The bundled example set has no such split. "
                               "Inventing a reassuring number would be worse than "
                               "reporting nothing."),
                },
            },
        )
    # (full implementation runs in the notebook where the split exists)
    return Finding(
        pillar="privacy", metric="membership_inference_auc", domain="image",
        value={"mia_auc": None, "status": "not_implemented_in_showcase"},
        verdict="info",
        summary="MIA is computed in the full GPU run.",
        details={},
    )
