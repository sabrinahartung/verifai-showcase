"""Scenario runner + tiny registry.

A scenario (YAML) declares: domain, model, dataset, sample_size, seed, and which
metrics (pillars) to run. `run_scenario` builds the pieces, runs each metric, and
collects the results into a single `Report`.

Design goals: no web server, no DB — pure functions in, `Report` out.
"""
from __future__ import annotations

import importlib
import random
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

from verifai.core.findings import Report, Finding
from verifai.core.integrity import audit_split, train_manifests_from_scenario


class SplitLeakageError(RuntimeError):
    """Raised instead of producing a flattering number from a leaking split."""

# metric id -> "module_path:function_name"
# Each metric fn has signature: fn(model, dataset, ctx: dict) -> Finding | list[Finding]
METRIC_REGISTRY: dict[str, str] = {
    "integrity.split_leakage":    "verifai.metrics.integrity.split_leakage:run",
    "performance.classification": "verifai.metrics.performance.classification:run",
    "explainability.gradcam":     "verifai.metrics.explainability.gradcam:run",
    "robustness.corruption":      "verifai.metrics.robustness.corruption:run",
    "fairness.skin_tone":         "verifai.metrics.fairness.skin_tone_ita:run",
    "privacy.mia":                "verifai.metrics.privacy.mia:run",
}


def _load(target: str) -> Callable:
    module_path, func = target.split(":")
    return getattr(importlib.import_module(module_path), func)


def _build_model(spec: dict[str, Any]):
    # spec: {domain, loader: "verifai.models.image:load", ...}
    return _load(spec["loader"])(spec)


def _build_dataset(spec: dict[str, Any]):
    return _load(spec["loader"])(spec)


def _eval_set_fingerprint(dataset) -> dict[str, Any]:
    """Identify *what was evaluated on*, precisely enough to refuse bad comparisons.

    Two runs are only comparable if they were scored on the same rows. A path is
    not enough — a manifest can be regenerated with a different seed and keep its
    name — so the file's content hash is what actually decides.
    """
    import hashlib
    meta = getattr(dataset, "meta", None) or {}
    manifest = meta.get("manifest")
    digest = None
    if manifest:
        path = Path(manifest)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return {"manifest": str(manifest) if manifest else None,
            "sha256": digest,
            "n": _safe_len(dataset)}


def _safe_len(dataset) -> int | None:
    try:
        return len(dataset)
    except TypeError:
        return None


def _enforce_split_integrity(scenario: dict[str, Any], dataset) -> None:
    """Refuse to evaluate a test set the model was trained on.

    The failure mode this exists for is silent: a contaminated split does not
    crash, it just reports a high number. Set `integrity.enforce: false` to
    downgrade this to a reported finding instead of a hard stop.
    """
    cfg = scenario.get("integrity") or {}
    if not cfg.get("enforce", True):
        return
    test_manifest = (getattr(dataset, "meta", None) or {}).get("manifest")
    train_manifests = train_manifests_from_scenario(scenario)
    if not test_manifest or not train_manifests:
        return                      # nothing declared to check against; the metric says so
    a = audit_split(test_manifest, train_manifests,
                    group_key=cfg.get("group_key", "lesion_id"),
                    id_key=cfg.get("id_key", "image_id"))
    if not a["verifiable"]:
        return                      # nothing comparable in the manifests; the metric says so
    if not a["clean"]:
        raise SplitLeakageError(
            f"{a['affected_rows']} of {a['n_test']} test images ({a['contamination']*100:.1f}%) "
            f"were seen during training: {a['shared_ids']} identical images, "
            f"{a['shared_groups']} shared lesions (e.g. {a['example_shared_groups'][:3]}). "
            f"Refusing to evaluate — the result would be a memorisation check. "
            f"Rebuild the split with scripts/build_splits.py, or set integrity.enforce: false "
            f"to report it as a finding instead."
        )


def run_scenario(scenario: dict[str, Any]) -> Report:
    seed = scenario.get("seed", 42)
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
        import torch; torch.manual_seed(seed)
    except Exception:
        pass

    model = _build_model(scenario["model"])
    dataset = _build_dataset(scenario["dataset"])
    _enforce_split_integrity(scenario, dataset)

    report = Report(
        scenario=scenario["name"],
        domain=scenario["domain"],
        model_id=scenario["model"].get("id", "unknown"),
        dataset_id=scenario["dataset"].get("id", "unknown"),
        meta={"seed": seed,
              # what was actually evaluated, not merely what the YAML asked for
              "sample_size": scenario.get("sample_size") or _safe_len(dataset),
              "device": str(getattr(model, "device", "cpu")),
              "eval_set": _eval_set_fingerprint(dataset),
              "label": scenario.get("label") or scenario["model"].get("id", scenario["name"])},
    )

    ctx = {"scenario": scenario, "seed": seed, "plot_dir": scenario.get("_plot_dir", "plots")}
    for metric_id in scenario["metrics"]:
        fn = _load(METRIC_REGISTRY[metric_id])
        result = fn(model, dataset, ctx)
        for f in (result if isinstance(result, list) else [result]):
            report.add(f)
    return report
