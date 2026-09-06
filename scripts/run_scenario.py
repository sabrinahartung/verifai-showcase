#!/usr/bin/env python
"""CLI entry point: run one scenario, write static artifacts.

Usage:
    python scripts/run_scenario.py scenarios/skin_cancer.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

# make `verifai` importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifai.core.run import run_scenario
from verifai.export.artifacts import write_report


def main(scenario_path: str) -> None:
    scenario = yaml.safe_load(Path(scenario_path).read_text(encoding="utf-8"))
    out_dir = "showcase/artifacts"
    scenario["_plot_dir"] = str(Path(out_dir) / scenario["name"] / "plots")

    print(f"▶ running scenario: {scenario['name']} ({scenario['domain']})")
    report = run_scenario(scenario)
    card = scenario.get("card")  # optional tile metadata
    path = write_report(report, out_dir=out_dir, card=card)
    print(f"✓ {len(report.findings)} finding(s) written -> {path}")
    for f in report.findings:
        print(f"   · {f.pillar:<14} {f.metric:<22} [{f.verdict}] {f.summary[:70]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/run_scenario.py <scenario.yaml>")
        raise SystemExit(2)
    main(sys.argv[1])
