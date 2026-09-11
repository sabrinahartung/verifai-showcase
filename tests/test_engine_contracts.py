"""Contract tests for the pieces a second model has to plug into.

Deliberately cheap: no checkpoint download, no HF network call. They cover the
seams that broke when the engine was generalised from one hardcoded model.
"""
from __future__ import annotations

import csv
import json
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
        def decide(self, probs): return max(probs, key=probs.get)

    finding = f.run(_M(), _DS(), {})
    populated = [c for c in finding.value["coverage"].values() if c > 0]
    assert len(populated) == 1, "test setup should produce exactly one populated bin"
    assert finding.verdict != "pass", "a single skin-tone bin must never read as a fairness pass"
    assert "accuracy_gap" not in finding.value, "no gap should be claimed from one group"


# --- split integrity: the check that guards every other number ---------------
def _split_manifest(tmp_path: Path, name: str, rows: list[tuple[str, str]]) -> Path:
    p = tmp_path / f"{name}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["filename", "image_id", "lesion_id", "label"])
        for img, les in rows:
            w.writerow([f"{img}.jpg", img, les, "mel"])
    return p


def test_audit_split_passes_a_disjoint_split(tmp_path):
    from verifai.core.integrity import audit_split
    tr = _split_manifest(tmp_path, "tr", [("i1", "L1"), ("i2", "L2")])
    te = _split_manifest(tmp_path, "te", [("i3", "L3"), ("i4", "L4")])
    a = audit_split(te, [tr])
    assert a["clean"] and a["contamination"] == 0.0


def test_audit_split_catches_a_shared_lesion_even_with_new_images(tmp_path):
    """The failure the original dataset had: distinct image_ids, same lesion."""
    from verifai.core.integrity import audit_split
    tr = _split_manifest(tmp_path, "tr", [("i1", "L1")])
    te = _split_manifest(tmp_path, "te", [("i9", "L1"), ("i8", "L2")])   # i9 is a new photo of L1
    a = audit_split(te, [tr])
    assert not a["clean"]
    assert a["shared_ids"] == 0 and a["shared_groups"] == 1
    assert a["contamination"] == 0.5


def test_audit_split_catches_identical_images(tmp_path):
    from verifai.core.integrity import audit_split
    tr = _split_manifest(tmp_path, "tr", [("i1", "L1")])
    te = _split_manifest(tmp_path, "te", [("i1", "L1")])
    a = audit_split(te, [tr])
    assert a["shared_ids"] == 1 and a["contamination"] == 1.0


def test_our_committed_manifests_are_leak_free():
    """The real split this repo ships. If this ever fails, do not publish a number."""
    from verifai.core.integrity import audit_split
    man = REPO / "data" / "manifests"
    if not (man / "ham10000_test.csv").exists():
        pytest.skip("run scripts/build_splits.py first")
    a = audit_split(man / "ham10000_test.csv",
                    [man / "ham10000_train.csv", man / "ham10000_val.csv"])
    assert a["clean"], f"committed split leaks: {a}"
    assert a["n_test"] == 1493


def test_train_manifests_are_derived_from_the_training_block():
    from verifai.core.integrity import train_manifests_from_scenario
    got = train_manifests_from_scenario(
        {"training": {"manifest_prefix": "ham10000", "manifest_dir": "data/manifests"}})
    assert got == ["data/manifests/ham10000_train.csv", "data/manifests/ham10000_val.csv"], \
        "validation counts as seen — the model was selected on it"
    assert train_manifests_from_scenario({}) == []


def test_unverifiable_split_is_never_reported_as_clean(tmp_path):
    """A manifest without identifiers answers nothing — that is not a pass.

    Regression: manifests carrying only filename+label compared as 0 shared
    lesions and 0 shared images, which scored a green 'clean split'.
    """
    from verifai.core.integrity import audit_split
    def bare(name, rows):
        p = tmp_path / f"{name}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["filename", "label"])
            w.writerows(rows)
        return p
    a = audit_split(bare("tr", [["a.jpg", "mel"]]), [bare("te", [["a.jpg", "mel"]])])
    assert a["verifiable"] is False
    assert a["clean"] is not True, "unverifiable must never read as clean"


# --- snapshots: the evidence trail that makes comparison possible -----------
def _report_with(value, pillar="performance", verdict="pass"):
    from verifai.core.findings import Report, Finding
    r = Report(scenario="s", domain="image", model_id="m", dataset_id="d")
    r.add(Finding(pillar=pillar, metric="m", domain="image", value=value, verdict=verdict))
    return r


def test_snapshot_metrics_flattens_numeric_leaves_only():
    from verifai.export.artifacts import snapshot_metrics
    m = snapshot_metrics(_report_with(
        {"accuracy": 0.8, "n": 100, "correct": True,          # bool is not a metric
         "per_class_recall": {"melanoma": 0.64, "nevi": None},
         "note": "text"}))
    assert m["performance.accuracy"] == 0.8
    assert m["performance.n"] == 100.0
    assert m["performance.per_class_recall.melanoma"] == 0.64
    assert "performance.correct" not in m, "bool must not be recorded as a metric"
    assert "performance.note" not in m
    assert "performance.per_class_recall.nevi" not in m


def test_snapshot_records_what_makes_a_run_comparable(tmp_path):
    from verifai.export.artifacts import write_snapshot
    r = _report_with({"accuracy": 0.8})
    r.add.__self__.findings.append(
        __import__("verifai.core.findings", fromlist=["Finding"]).Finding(
            pillar="integrity", metric="split_leakage", domain="image",
            value={"contamination": 0.0}, verdict="pass"))
    r.meta = {"eval_set": {"manifest": "m.csv", "sha256": "deadbeef", "n": 10},
              "label": "baseline", "device": "cpu", "seed": 42}
    path = write_snapshot(r, tmp_path)
    snap = json.loads(path.read_text(encoding="utf-8"))
    assert snap["eval_set"]["sha256"] == "deadbeef"
    assert snap["integrity"] == "pass", "the integrity verdict must travel with the snapshot"
    assert snap["label"] == "baseline"
    assert snap["metrics"]["performance.accuracy"] == 0.8


def test_eval_set_fingerprint_uses_content_not_filename(tmp_path):
    """A manifest can be regenerated with a different seed and keep its name."""
    from verifai.core.run import _eval_set_fingerprint

    class _DS:
        def __init__(self, p): self.meta = {"manifest": str(p)}
        def __len__(self): return 2

    a = tmp_path / "same_name.csv"; a.write_text("filename,label\nx.jpg,mel\n")
    first = _eval_set_fingerprint(_DS(a))["sha256"]
    a.write_text("filename,label\ny.jpg,nv\n")            # same path, different rows
    second = _eval_set_fingerprint(_DS(a))["sha256"]
    assert first and second and first != second, "different rows must not look comparable"


# --- uncertainty: a number without an interval is not a claim ---------------
def test_wilson_interval_brackets_the_estimate_and_narrows_with_n():
    from verifai.metrics._stats import wilson
    lo, hi = wilson(104, 163)                       # melanoma in the real test set
    assert lo < 104 / 163 < hi
    wide = wilson(10, 13)                           # dermatofibroma: 13 images
    narrow = wilson(776, 1009)                      # nevi: 1,009 images
    assert (wide[1] - wide[0]) > 3 * (narrow[1] - narrow[0]), \
        "a class with 13 images must not look as certain as one with 1,009"


def test_wilson_stays_inside_zero_one_at_the_extremes():
    """Where the normal approximation fails: 0/n and n/n."""
    from verifai.metrics._stats import wilson
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.0 < hi < 1.0, "0/10 is not certainty"
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.0 < lo < 1.0, "10/10 is not certainty either"
    assert wilson(5, 0) is None


def test_ppv_moves_with_prevalence_even_though_sensitivity_does_not():
    """The clinical point: precision on a test set is not PPV at deployment."""
    from verifai.metrics._stats import ppv_at_prevalence
    high = ppv_at_prevalence(0.85, 0.90, 0.10)
    low = ppv_at_prevalence(0.85, 0.90, 0.01)
    assert high > 5 * low, "PPV must fall sharply as prevalence falls"
    assert ppv_at_prevalence(0.85, 0.90, 0.0) is None


def test_per_class_metrics_from_a_known_confusion_matrix():
    from verifai.metrics.performance.classification import _per_class
    #            pred A  B
    cm = [[8, 2],      # true A: 10
          [3, 7]]      # true B: 10
    pc = _per_class(cm, ["A", "B"])
    assert pc["A"]["sensitivity"] == 0.8 and pc["A"]["support"] == 10
    assert pc["A"]["specificity"] == 0.7          # B correctly not called A: 7/10
    assert pc["A"]["ppv_test_prevalence"] == round(8 / 11, 4)
    assert pc["A"]["sensitivity_ci"][0] < 0.8 < pc["A"]["sensitivity_ci"][1]


def test_fairness_gap_is_not_claimed_when_group_intervals_overlap():
    """A large gap between two small groups is not evidence of a gap."""
    pytest.importorskip("numpy")
    from PIL import Image
    from verifai.metrics.fairness import skin_tone_ita as f

    # two bins, 10 images each, one scoring 0.6 and one 0.9 -> wide, overlapping CIs
    light = [(245, 224, 210)] * 10
    dark = [(90, 62, 48)] * 10

    class _S:
        def __init__(self, i, rgb, ok): self.id, self.rgb, self.label = f"i{i}", rgb, ("a" if ok else "b")

    samples = [_S(i, c, i < 6) for i, c in enumerate(light)] + \
              [_S(100 + i, c, i < 9) for i, c in enumerate(dark)]

    class _DS:
        def __iter__(self): return iter(samples)
        def load(self, s): return Image.new("RGB", (64, 64), s.rgb)

    class _M:
        classes = ["a", "b"]
        def predict_probs(self, img): return {"a": 0.9, "b": 0.1}   # always predicts "a"
        def decide(self, probs): return max(probs, key=probs.get)

    finding = f.run(_M(), _DS(), {})
    if "accuracy_gap" in finding.value:              # only if both bins were populated
        assert finding.value["gap_is_separated"] is False
        assert finding.verdict != "fail", \
            "an unseparated gap must not be reported as a failure"


# --- the decision rule: argmax is a choice, not a law -----------------------
def _clf(weights=None):
    torch = pytest.importorskip("torch")
    tvm = pytest.importorskip("torchvision.models")
    from verifai.models.image import ImageClassifier
    net = tvm.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, 3)
    return ImageClassifier(net.eval(), ["nevus", "melanoma", "other"],
                           device=torch.device("cpu"), decision_weights=weights)


def test_default_decision_rule_is_plain_argmax():
    clf = _clf()
    assert clf.decide({"nevus": 0.5, "melanoma": 0.3, "other": 0.2}) == "nevus"


def test_weights_let_a_rare_class_clear_a_lower_bar():
    """The whole point: melanoma wins on 0.3 vs 0.5 once missing it costs more."""
    probs = {"nevus": 0.5, "melanoma": 0.3, "other": 0.2}
    assert _clf({"melanoma": 2.0}).decide(probs) == "melanoma"   # 0.6 > 0.5
    assert _clf({"melanoma": 1.5}).decide(probs) == "nevus"      # 0.45 < 0.5, not enough
    assert _clf().decide(probs) == "nevus"                       # unweighted


def test_ranking_follows_the_same_rule_so_top_k_stays_consistent():
    probs = {"nevus": 0.5, "melanoma": 0.3, "other": 0.2}
    assert _clf().rank(probs)[0] == "nevus"
    assert _clf({"melanoma": 2.0}).rank(probs)[0] == "melanoma"


def test_unlisted_classes_keep_weight_one():
    probs = {"nevus": 0.5, "melanoma": 0.3, "other": 0.2}
    assert _clf({"other": 10.0}).decide(probs) == "other"        # 2.0 beats 0.5


# --- comparison: what is decidable from data, and what is not --------------
def test_metrics_declare_their_own_direction_rather_than_the_app_guessing():
    """'auc' is higher-better for a classifier and lower-better for membership
    inference. Only the metric can say which."""
    from verifai.export.artifacts import snapshot_directions
    from verifai.core.findings import Report, Finding
    r = Report(scenario="s", domain="image", model_id="m", dataset_id="d")
    r.add(Finding(pillar="privacy", metric="m", domain="image", value={"mia_auc": 0.6},
                  details={"better": {"mia_auc": "lower"}}))
    r.add(Finding(pillar="performance", metric="m", domain="image", value={"accuracy": 0.8},
                  details={"better": {"accuracy": "higher", "per_class.*.sensitivity": "higher"}}))
    d = snapshot_directions(r)
    assert d["privacy.mia_auc"] == "lower"
    assert d["performance.accuracy"] == "higher"
    assert d["performance.per_class.*.sensitivity"] == "higher"


# --- training variants: focal loss and balanced sampling -------------------
def test_focal_loss_reduces_to_cross_entropy_at_gamma_zero():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "scripts"))
    from train_model import FocalLoss
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.4, 0.3, 0.3]])
    target = torch.tensor([0, 0])
    ce = torch.nn.CrossEntropyLoss()(logits, target)
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, target), ce)


def test_focal_loss_shifts_weight_from_easy_examples_to_hard_ones():
    """The whole point: a confidently-correct nevus should stop contributing."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "scripts"))
    from train_model import FocalLoss
    easy = (torch.tensor([[6.0, 0.0, 0.0]]), torch.tensor([0]))    # already right
    hard = (torch.tensor([[0.4, 0.3, 0.3]]), torch.tensor([0]))    # barely right
    ce_ratio = (torch.nn.CrossEntropyLoss()(*hard) / torch.nn.CrossEntropyLoss()(*easy))
    fl = FocalLoss(gamma=2.0)
    focal_ratio = fl(*hard) / fl(*easy)
    assert focal_ratio > 10 * ce_ratio, "focal must concentrate far harder on hard cases"


def test_focal_loss_accepts_class_weights_so_it_composes_with_alpha():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "scripts"))
    from train_model import FocalLoss
    logits = torch.tensor([[0.4, 0.3, 0.3]])
    target = torch.tensor([0])
    plain = FocalLoss(gamma=2.0)(logits, target)
    weighted = FocalLoss(gamma=2.0, weight=torch.tensor([5.0, 1.0, 1.0]))(logits, target)
    assert weighted > plain, "a class weight of 5 must scale that class's loss up"


def test_lineage_never_widens_what_may_be_compared():
    """Presentation grouping must not leak into the comparability rule.

    Two configurations of one lineage that were scored on different manifests
    still have to land in different comparability groups — the content hash
    decides, not the label they share in the gallery.
    """
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    same_lineage_different_rows = [
        {"scenario": "a", "label": "a", "integrity": "pass", "created_at": "1",
         "eval_set": {"sha256": "aaa", "manifest": "test.csv", "n": 100}, "metrics": {}},
        {"scenario": "b", "label": "b", "integrity": "pass", "created_at": "2",
         "eval_set": {"sha256": "bbb", "manifest": "test.csv", "n": 100}, "metrics": {}},
    ]
    assert len(app.group_snapshots(same_lineage_different_rows)) == 2


def test_the_same_rows_at_a_different_path_stay_comparable():
    """A moved or renamed checkout must not split a comparison group.

    The hash is taken over the manifest's contents alone, so the same rows read
    from `/old/repo/test.csv` and `/new/repo/test.csv` are the same evaluation
    set. Keying on the path too would have quietly filed every run made before a
    directory rename apart from every run made after it — same rows, two groups,
    each looking like it held fewer runs than it did.
    """
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    same_rows_moved_repo = [
        {"scenario": "a", "label": "a", "integrity": "pass", "created_at": "1",
         "eval_set": {"sha256": "aaa", "manifest": "/old/repo/test.csv", "n": 100},
         "metrics": {}},
        {"scenario": "b", "label": "b", "integrity": "pass", "created_at": "2",
         "eval_set": {"sha256": "aaa", "manifest": "/new/repo/test.csv", "n": 100},
         "metrics": {}},
    ]
    assert len(app.group_snapshots(same_rows_moved_repo)) == 1


def test_unhashed_snapshots_do_not_merge_on_being_equally_unidentified():
    """No hash means nothing to compare on — those must not pool together."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    no_hash = [
        {"scenario": "a", "label": "a", "integrity": "pass", "created_at": "1",
         "eval_set": {"sha256": None, "manifest": "a.csv", "n": 7}, "metrics": {}},
        {"scenario": "b", "label": "b", "integrity": "pass", "created_at": "2",
         "eval_set": {"sha256": None, "manifest": "b.csv", "n": 9}, "metrics": {}},
    ]
    assert len(app.group_snapshots(no_hash)) == 2
