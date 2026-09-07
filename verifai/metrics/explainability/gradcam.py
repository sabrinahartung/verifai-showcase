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


def _gradcam(torch_model, layer, x, class_idx):
    """Raw (un-normalised) Grad-CAM tensor for `class_idx`. See module docstring.

    `layer` comes from the model adapter (model.cam_layer), so this works for
    any architecture rather than assuming a ResNet. Returns a CPU tensor so the
    numpy/PIL code below stays device-agnostic.
    """
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
    acts, grads = activations["v"][0], gradients["v"][0]  # [C,h,w]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    return (weights * acts).sum(dim=0).relu().detach().cpu()


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
    cam_layer = model.cam_layer

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
        cam = _gradcam(tm, cam_layer, x, ci)
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
        summary=(f"Grad-CAM overlays for {len(rel_plots)} examples; mean deletion "
                 f"faithfulness {mean_faith} (probability drop when the highlighted "
                 f"region is masked out)." if mean_faith is not None
                 else "Grad-CAM overlays generated."),
        details={
            "explain": {
                "what": ("Grad-CAM highlights the parts of the image that pushed the model "
                         "towards its answer. The faithfulness score then checks whether "
                         "those highlights are honest: the marked region is masked out and "
                         "we measure how far the model's confidence falls."),
                "how": ("In the overlays, warm colours (red/yellow) mark the regions that "
                        "drove the decision, blue marks regions that barely mattered — you "
                        "want the heat on the lesion, not on hair, rulers or the image "
                        "border. The scale below shows the average confidence drop when "
                        "that hot region is hidden: a bigger drop means the explanation "
                        "reflects what the model actually used."),
                "limits": ("A convincing heatmap is not proof of medically correct "
                           "reasoning — it shows where the model looked, not whether it "
                           "looked for the right reason. A low faithfulness score is the "
                           "clearer signal: it means the highlight is largely decorative."),
            },
            "target_layer": getattr(model, "cam_layer_path", "layer4[-1]"),
            "faithfulness_per_image": [round(f, 3) for f in faith_scores],
            "chart": {"kind": "images", "title": "Where the model looks (Grad-CAM)",
                      "paths": rel_plots, "captions": captions},
            "chart2": {
                "kind": "scale", "title": "Deletion faithfulness",
                "value": mean_faith or 0.0, "min": 0, "max": 1,
                "ticks": [0, 0.2, 0.5, 1], "tick_labels": ["0", "0.2", "0.5", "1"],
                "bands": [
                    {"to": 0.2, "label": "decorative", "color": "#F5D3CE"},
                    {"to": 0.5, "label": "partly faithful", "color": "#FAECC8"},
                    {"to": 1.0, "label": "faithful", "color": "#CDE8D5"},
                ],
                "value_label": "How far confidence falls when the highlighted region "
                               "is masked out — higher means the highlight mattered.",
            },
        },
        plots=rel_plots,
    )
