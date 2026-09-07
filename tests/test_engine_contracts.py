"""Contract tests for the pieces a second model has to plug into.

Deliberately cheap: no checkpoint download, no HF network call. They cover the
seams that broke when the engine was generalised from one hardcoded model.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from verifai.core.run import METRIC_REGISTRY, _load          # noqa: E402
from verifai.datasets import loaders                          # noqa: E402
from verifai.models.image import _resolve_module, resolve_device  # noqa: E402


# --- the registry actually resolves ------------------------------------------
@pytest.mark.parametrize("metric_id", sorted(METRIC_REGISTRY))
def test_every_registered_metric_is_importable(metric_id):
    assert callable(_load(METRIC_REGISTRY[metric_id]))


# --- layer paths, so Grad-CAM is not ResNet-only -----------------------------
def test_resolve_module_handles_index_and_dots():
    torch = pytest.importorskip("torch")
    m = torch.nn.Module()
    m.layer4 = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Conv2d(1, 2, 1))
    assert _resolve_module(m, "layer4[-1]").out_channels == 2
    assert _resolve_module(m, "layer4[0]").out_channels == 1
    with pytest.raises(ValueError):
        _resolve_module(m, "layer4[oops]")


def test_resolve_device_honours_explicit_name():
    pytest.importorskip("torch")
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


# --- a dataset must not learn its classes from a model -----------------------
def test_loader_does_not_import_the_model_package():
    src = (REPO / "verifai" / "datasets" / "loaders.py").read_text(encoding="utf-8")
    assert "verifai.models" not in src, "dataset loader must not depend on a model"


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    imgs = tmp_path / "img"
    imgs.mkdir()
    from PIL import Image
    for r in rows:
        Image.new("RGB", (8, 8)).save(imgs / r["filename"])
    man = tmp_path / "m.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return man


def test_classes_are_derived_from_the_data(tmp_path):
    pytest.importorskip("PIL")
    man = _manifest(tmp_path, [
        {"filename": "a.jpg", "label": "zebra", "lesion_id": "L1"},
        {"filename": "b.jpg", "label": "apple", "lesion_id": "L2"},
    ])
    ds = loaders.load_image_manifest({"manifest": str(man), "images_dir": str(man.parent / "img")})
    assert ds.classes == ["apple", "zebra"]          # sorted, from the labels
    assert ds.has_labels and len(ds) == 2


def test_extra_manifest_columns_survive_on_the_sample(tmp_path):
    pytest.importorskip("PIL")
    man = _manifest(tmp_path, [
        {"filename": "a.jpg", "label": "mel", "lesion_id": "HAM_1", "sex": "female"},
    ])
    ds = loaders.load_image_manifest({"manifest": str(man), "images_dir": str(man.parent / "img")})
    s = list(ds)[0]
    assert s.meta["lesion_id"] == "HAM_1" and s.meta["sex"] == "female"


def test_explicit_classes_win_over_derived(tmp_path):
    pytest.importorskip("PIL")
    man = _manifest(tmp_path, [{"filename": "a.jpg", "label": "mel"}])
    ds = loaders.load_image_manifest({"manifest": str(man), "images_dir": str(man.parent / "img"),
                                      "classes": ["nv", "mel"]})
    assert ds.classes == ["nv", "mel"]              # checkpoint order, not sorted


def test_old_loader_name_still_resolves():
    assert loaders.load_ham10000 is loaders.load_image_manifest


# --- the adapter works for an arbitrary class count / architecture -----------
def test_adapter_is_not_tied_to_seven_skin_classes():
    torch = pytest.importorskip("torch")
    tvm = pytest.importorskip("torchvision.models")
    from PIL import Image
    from verifai.models.image import ImageClassifier

    net = tvm.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, 3)
    clf = ImageClassifier(net.eval(), ["a", "b", "c"], device=torch.device("cpu"))

    probs = clf.predict_probs(Image.new("RGB", (64, 64)))
    assert set(probs) == {"a", "b", "c"}
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert clf.cam_layer is net.layer4[-1]


# --- honesty gates: a verdict must not outrun the evidence -------------------
def test_single_skin_tone_bin_never_claims_a_fairness_pass():
    """One populated ITA bin means one group — a gap of 0 there is not a pass.

    Regression: HAM10000 skews so heavily towards light skin that a full run can
    put nearly every image in one bin. The old gate accepted that and emitted a
    green 'pass' on fairness for the most skewed sample imaginable.
    """
    pytest.importorskip("numpy")
    from PIL import Image
    from verifai.metrics.fairness import skin_tone_ita as f

    class _DS:
        # 12 identical pale images -> all land in the same ITA bin
        samples = [type("S", (), {"id": f"i{i}", "label": "mel", "meta": {}})()
                   for i in range(12)]
        def __iter__(self): return iter(self.samples)
        def load(self, s): return Image.new("RGB", (64, 64), (245, 224, 210))

    class _M:
        classes = ["mel", "nv"]
        def predict_probs(self, img): return {"mel": 0.9, "nv": 0.1}

    finding = f.run(_M(), _DS(), {})
    populated = [c for c in finding.value["coverage"].values() if c > 0]
    assert len(populated) == 1, "test setup should produce exactly one populated bin"
    assert finding.verdict != "pass", "a single skin-tone bin must never read as a fairness pass"
    assert "accuracy_gap" not in finding.value, "no gap should be claimed from one group"
