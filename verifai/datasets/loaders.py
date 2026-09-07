"""Dataset loaders (image domain).

MVP-0 uses a tiny, *versioned* manifest of real HAM10000 images that live in the
repo (`data/examples/` + `data/manifests/*.csv`). That keeps the first run:

  - REAL (real dermatoscopic images, real labels),
  - reproducible (anyone who clones the repo can rerun it),
  - small enough to run on a laptop CPU in seconds.

For a larger, statistically meaningful run, point the manifest at a bigger subset
and execute on a free GPU (see scripts/ notebook) — same code path, more rows.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Repo root = two levels up from this file (verifai/datasets/loaders.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ImageSample:
    id: str
    path: Path
    label: str | None = None
    # any extra manifest columns: lesion_id, image_id, sex, age, localization, ...
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageDataset:
    """A small, iterable set of on-disk images with optional ground-truth labels."""
    samples: list[ImageSample]
    classes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    domain: str = "image"

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[ImageSample]:
        return iter(self.samples)

    @property
    def has_labels(self) -> bool:
        return all(s.label is not None for s in self.samples) and len(self.samples) > 0

    def load(self, sample: ImageSample):
        from PIL import Image
        return Image.open(sample.path).convert("RGB")


def _resolve(path_str: str) -> Path:
    """Resolve a manifest/dir path relative to the repo root if not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_image_manifest(spec: dict[str, Any]) -> ImageDataset:
    """Load images from a manifest CSV (columns: filename,label).

    Extra columns (lesion_id, image_id, sex, age, localization, ...) are kept on
    each sample's `meta`, so metrics can use real subgroups instead of proxies.

    spec example:
      {loader: "verifai.datasets.loaders:load_image_manifest",
       id: "ham10000-examples",
       manifest: "data/manifests/ham10000_examples.csv",
       images_dir: "data/examples",     # optional; defaults next to the manifest
       classes: [...]}                  # optional; else derived from the labels

    The class list is derived from the *data*, never imported from the model —
    a dataset does not know which model will be run against it.
    """
    manifest = _resolve(spec["manifest"])
    images_dir = _resolve(spec["images_dir"]) if spec.get("images_dir") else manifest.parent
    # sensible default: repo's example folder
    if not images_dir.exists() and (REPO_ROOT / "data" / "examples").exists():
        images_dir = REPO_ROOT / "data" / "examples"

    samples: list[ImageSample] = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["filename"].strip()
            label = (row.get("label") or "").strip() or None
            meta = {k: v for k, v in row.items()
                    if k not in ("filename", "label") and v not in (None, "")}
            samples.append(ImageSample(id=Path(fname).stem, path=images_dir / fname,
                                       label=label, meta=meta))

    # keep it deterministic and optionally capped
    samples.sort(key=lambda s: s.id)
    n = spec.get("sample_size")
    if isinstance(n, int) and n > 0:
        samples = samples[:n]

    missing = [str(s.path) for s in samples if not s.path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) from the manifest are missing, e.g. {missing[0]}"
        )

    classes = list(spec.get("classes") or
                   sorted({s.label for s in samples if s.label is not None}))

    return ImageDataset(
        samples=samples,
        classes=classes,
        meta={"manifest": str(manifest), "images_dir": str(images_dir),
              "source": spec.get("source", "manifest")},
    )


# The loader used to be named after one dataset; scenarios still reference that name.
load_ham10000 = load_image_manifest
