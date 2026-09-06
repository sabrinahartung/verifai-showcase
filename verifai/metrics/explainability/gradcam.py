"""Explainability (image): Grad-CAM overlays + a deletion-faithfulness score.

Signature: run(model, dataset, ctx) -> Finding

The Grad-CAM itself is ported verbatim (semantics-wise) from
ML_Training_Dojo/streamlit_app.py: last residual block, gradient-weighted
activations, ReLU (evidence *for* the class only), no per-map normalisation.

Added here (so the dashboard can quantify, not just illustrate):
  - overlays saved as PNGs under the report's plot dir (top class per image),
  - a light *deletion faithfulness* score: mask the most-attended region and
    measure how far the class probability drops. If the map is faithful,
    hiding what it highlights should hurt the prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from verifai.core.findings import Finding

FLAT_EPS = 1e-6


def _gradcam(torch_model, x, class_idx):
    """Raw (un-normalised) [7,7] Grad-CAM tensor for `class_idx`. See module docstring."""
    layer = torch_model.layer4[-1]
    activations, gradients = {}, {}
    handles = [
        layer.register_forward_hook(lambda m, i, o: activations.update(v=o)),
        layer.register_full_backward_hook(lambda m, gi, go: gradients.update(v=go[0])),
    ]
    try:
        logits = torch_model(x)
        torch_model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    finally:
        for h in handles:
            h.remove()
    acts, grads = activations["v"][0], gradients["v"][0]  # [C,7,7]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    return (weights * acts).sum(dim=0).relu().detach()


def _overlay(img, cam, scale, alpha=0.5):
    import numpy as np
    import matplotlib
    from PIL import Image
    cam = (cam / scale).clamp(0, 1)
    img = img.convert("RGB")
    cam_img = Image.fromarray((cam.numpy() * 255).astype("uint8")).resize(img.size, Image.BICUBIC)
    heat = matplotlib.colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    return Image.blend(img, Image.fromarray((heat * 255).astype("uint8")), alpha)


def _deletion_faithfulness(model, img, cam, class_idx, frac=0.2):
    """Mask the top-`frac` most-attended pixels; return the probability drop.

    drop = p_class(original) - p_class(masked). Higher = the highlighted region
    genuinely drove the score (more faithful). Clamped to [0,1].
    """
    import numpy as np
    import torch
    from PIL import Image

    classes = model.classes
    p0 = model.predict_probs(img)[classes[class_idx]]

    cam_img = np.asarray(
        Image.fromarray((cam.numpy()).astype("float32")).resize(img.size, Image.BICUBIC),
        dtype=np.float32,
    )
    if cam_img.max() <= FLAT_EPS:
        return 0.0
    thr = np.quantile(cam_img, 1.0 - frac)
    mask = cam_img >= thr
    arr = np.asarray(img.convert("RGB")).copy()
    arr[mask] = 128  # grey out the most-attended region
    masked = Image.fromarray(arr)
    p1 = model.predict_probs(masked)[classes[class_idx]]
    return float(max(0.0, min(1.0, p0 - p1)))


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    plot_dir = Path(ctx.get("plot_dir", "plots"))
    plot_dir.mkdir(parents=True, exist_ok=True)
    classes = model.classes
    tm = model.torch_module

    rel_plots: list[str] = []
    captions: list[str] = []
    faith_scores: list[float] = []

    # cap overlays so the artifact stays light
    max_imgs = int(ctx.get("scenario", {}).get("gradcam_max_images", 7))
    for s in list(dataset)[:max_imgs]:
        img = dataset.load(s)
        probs = model.predict_probs(img)
        top = max(probs, key=probs.get)
        ci = classes.index(top)

        x = model.to_tensor(img)
        x.requires_grad_(True)
        cam = _gradcam(tm, x, ci)
        scale = max(FLAT_EPS, float(cam.max()))

        out = _overlay(img, cam, scale)
        fname = f"gradcam_{s.id}.png"
        out.save(plot_dir / fname)
        rel_plots.append(f"plots/{fname}")
        captions.append(f"{s.id}: {top} ({probs[top]*100:.0f}%)")

        faith_scores.append(_deletion_faithfulness(model, img, cam, ci))

    mean_faith = round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else None

    return Finding(
        pillar="explainability",
        metric="gradcam_faithfulness",
        domain="image",
        value={"n_overlays": len(rel_plots), "mean_deletion_faithfulness": mean_faith},
        verdict="info",
        summary=(f"Grad-CAM-Overlays für {len(rel_plots)} Beispiele; mittlere Deletion-"
                 f"Faithfulness {mean_faith} (Wahrscheinlichkeits­abfall beim Ausblenden "
                 f"der markierten Region)." if mean_faith is not None
                 else "Grad-CAM-Overlays erzeugt."),
        details={
            "target_layer": "layer4[-1]",
            "faithfulness_per_image": [round(f, 3) for f in faith_scores],
            "chart": {"kind": "images", "title": "Wohin das Modell schaut (Grad-CAM)",
                      "paths": rel_plots, "captions": captions},
            "chart2": {"kind": "gauge", "title": "Deletion-Faithfulness (0–1)",
                       "value": mean_faith or 0.0, "min": 0, "max": 1},
        },
        plots=rel_plots,
    )
