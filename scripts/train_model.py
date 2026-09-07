#!/usr/bin/env python
"""Train a model on the leak-free manifests, then hand off to evaluation.

The split manifests are the contract: this script trains on `<prefix>_train.csv`,
selects on `<prefix>_val.csv`, and never touches `<prefix>_test.csv`. Evaluation
(`run_scenario.py`) then reads only the test manifest. Because both sides read the
same versioned files, "what did the model see" stops being a matter of trust.

Reads a `training:` block from the scenario. Usage:
    python scripts/train_model.py scenarios/skin_cancer_clean.yaml
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from verifai.datasets.loaders import load_image_manifest          # noqa: E402
from verifai.models.image import build_preprocess, resolve_device  # noqa: E402


def _augment(size: int, mean, std):
    """Train-time augmentation, matching the original ML_Training_Dojo recipe."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class _TorchView:
    """Adapts our ImageDataset to what a torch DataLoader expects."""

    def __init__(self, ds, classes: list[str], tf):
        self.samples = list(ds)
        self.ds, self.tf = ds, tf
        self.index = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return self.tf(self.ds.load(s)), self.index[s.label]


def _evaluate(model, loader, device, n_classes: int):
    """Returns (accuracy, balanced accuracy) — balanced matters on this class mix."""
    import torch
    model.eval()
    correct = total = 0
    per_cls_hit = [0] * n_classes
    per_cls_tot = [0] * n_classes
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for t, p in zip(y.tolist(), pred.tolist()):
                per_cls_tot[t] += 1
                per_cls_hit[t] += int(t == p)
    recalls = [h / t for h, t in zip(per_cls_hit, per_cls_tot) if t]
    return correct / total, sum(recalls) / len(recalls)


def main(scenario_path: str) -> None:
    import torch
    from torch.utils.data import DataLoader
    import torchvision.models as tvm

    sc = yaml.safe_load(Path(scenario_path).read_text(encoding="utf-8"))
    tcfg = sc.get("training") or {}
    if not tcfg:
        sys.exit(f"{scenario_path} has no `training:` block")

    seed = int(sc.get("seed", 42))
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = resolve_device(tcfg.get("device", sc.get("model", {}).get("device", "auto")))
    epochs = int(tcfg.get("epochs", 10))
    bs = int(tcfg.get("batch_size", 32))
    lr = float(tcfg.get("lr", 1e-4))
    arch = tcfg.get("arch", sc.get("model", {}).get("arch", "resnet18"))
    size = int(tcfg.get("image_size", 224))

    # --- data: the manifests are the contract -------------------------------
    base = dict(sc["dataset"])
    base.pop("loader", None)
    man_dir, prefix = tcfg.get("manifest_dir", "data/manifests"), tcfg["manifest_prefix"]

    def _ds(split):
        spec = dict(base)
        spec["manifest"] = f"{man_dir}/{prefix}_{split}.csv"
        return load_image_manifest(spec)

    train_ds, val_ds = _ds("train"), _ds("val")
    classes = sorted({s.label for s in train_ds if s.label})
    print(f"▶ train {len(train_ds):,} images | val {len(val_ds):,} images | {len(classes)} classes")
    print(f"  device={device}  arch={arch}  epochs={epochs}  bs={bs}  lr={lr}")

    from verifai.models.image import MEAN, STD
    # persistent_workers matters on macOS: spawning 8 loader processes costs ~10s,
    # and without this they respawn every epoch, doubling a short run's wall time.
    workers = int(tcfg.get("workers", 8))
    dl_kw = dict(batch_size=bs, num_workers=workers,
                 persistent_workers=workers > 0)
    tl = DataLoader(_TorchView(train_ds, classes, _augment(size, MEAN, STD)),
                    shuffle=True, **dl_kw)
    vl = DataLoader(_TorchView(val_ds, classes, build_preprocess(size)),
                    shuffle=False, **dl_kw)

    # --- model ---------------------------------------------------------------
    net = getattr(tvm, arch)(weights="DEFAULT" if tcfg.get("pretrained", True) else None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net = net.to(device)

    # class weights: this data is 67% nevi, so plain CE would learn to say "nevi"
    counts = Counter(s.label for s in train_ds if s.label)
    w = torch.tensor([len(train_ds) / (len(classes) * counts[c]) for c in classes],
                     dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=w if tcfg.get("class_weights", True) else None)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    out_dir = Path(tcfg.get("out_dir", "artifacts_training"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{sc['name']}.pt"
    history, best = [], -1.0

    for ep in range(1, epochs + 1):
        net.train(); run = 0.0; t0 = time.perf_counter()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(net(x), y)
            loss.backward(); opt.step()
            run += loss.item() * y.size(0)
        acc, bal = _evaluate(net, vl, device, len(classes))
        history.append({"epoch": ep, "train_loss": round(run / len(train_ds), 4),
                        "val_accuracy": round(acc, 4), "val_balanced_accuracy": round(bal, 4),
                        "seconds": round(time.perf_counter() - t0, 1)})
        star = ""
        if bal > best:                      # select on balanced accuracy, not accuracy
            best = bal; torch.save(net.state_dict(), ckpt); star = "  <- saved"
        print(f"  epoch {ep:>2}/{epochs}  loss {history[-1]['train_loss']:.4f}  "
              f"val_acc {acc:.3f}  val_balanced {bal:.3f}  "
              f"({history[-1]['seconds']:.0f}s){star}")

    # --- provenance: what was trained, on exactly which rows -----------------
    meta = {
        "scenario": sc["name"], "arch": arch, "classes": classes, "seed": seed,
        "epochs": epochs, "batch_size": bs, "lr": lr, "device": str(device),
        "image_size": size, "class_weights": bool(tcfg.get("class_weights", True)),
        "best_val_balanced_accuracy": round(best, 4), "history": history,
        "manifests": {s: f"{man_dir}/{prefix}_{s}.csv" for s in ("train", "val", "test")},
        "train_images": len(train_ds), "val_images": len(val_ds),
        "note": "test manifest was not read during training",
    }
    (out_dir / f"{sc['name']}_training.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n✓ best val balanced accuracy {best:.3f}")
    print(f"  weights  -> {ckpt}")
    print(f"  metadata -> {out_dir / (sc['name'] + '_training.json')}")
    print("\nNext: point the scenario's model.weights_path at the checkpoint (or upload it\n"
          "with `huggingface-cli upload`), then:\n"
          f"  python scripts/run_scenario.py {scenario_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/train_model.py <scenario.yaml>")
        raise SystemExit(2)
    main(sys.argv[1])
