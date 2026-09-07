"""Scenario runner + tiny registry.

A scenario (YAML) declares: domain, model, dataset, sample_size, seed, and which
metrics (pillars) to run. `run_scenario` builds the pieces, runs each metric, and
collects the results into a single `Report`.

Design goals: no web server, no DB — pure functions in, `Report` out.
"""
from __future__ import annotations

import importlib
import random
from typing import Any, Callable

from verifai.core.findings import Report, Finding

# metric id -> "module_path:function_name"
# Each metric fn has signature: fn(model, dataset, ctx: dict) -> Finding | list[Finding]
METRIC_REGISTRY: dict[str, str] = {
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

    report = Report(
        scenario=scenario["name"],
        domain=scenario["domain"],
        model_id=scenario["model"].get("id", "unknown"),
        dataset_id=scenario["dataset"].get("id", "unknown"),
        meta={"seed": seed, "sample_size": scenario.get("sample_size"),
              "device": str(getattr(model, "device", "cpu"))},
    )

    ctx = {"scenario": scenario, "seed": seed, "plot_dir": scenario.get("_plot_dir", "plots")}
    for metric_id in scenario["metrics"]:
        fn = _load(METRIC_REGISTRY[metric_id])
        result = fn(model, dataset, ctx)
        for f in (result if isinstance(result, list) else [result]):
            report.add(f)
    return report
