"""Image-domain model adapter: HAM10000 skin-lesion ResNet18.

Loads the already-trained model (from Hugging Face, or a local .pt) and exposes
the exact preprocessing + prediction used at training time, so every metric
sees the model behave identically to the original demo.

Mirrors ML_Training_Dojo/streamlit_app.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# --- Classes: exact sorted order used during training (index i -> CLASSES[i]) ---
CLASSES = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis-like_lesions",
    "dermatofibroma",
    "melanocytic_Nevi",
    "melanoma",
    "vascular_lesions",
]

# ImageNet normalization + 224x224, exactly as in the original training/demo.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_preprocess():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class SkinLesionModel:
    """Thin wrapper around the torch module + its class list.

    Metrics use `.torch_module` (for hooks / Grad-CAM) and the convenience
    `.predict_probs(pil_image)` / `.to_tensor(pil_image)` helpers.
    """

    def __init__(self, model, classes: list[str]):
        self.model = model
        self.classes = classes
        self._pre = build_preprocess()

    @property
    def torch_module(self):
        return self.model

    def to_tensor(self, img):
        return self._pre(img.convert("RGB")).unsqueeze(0)  # [1, 3, 224, 224]

    def predict_probs(self, img) -> dict[str, float]:
        import torch
        x = self.to_tensor(img)
        with torch.no_grad():
            probs = self.model(x).softmax(dim=1)[0]
        return {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}


def load(spec: dict[str, Any]) -> SkinLesionModel:
    """spec example:
    {loader: "verifai.models.image:load",
     id: "skin-lesion-resnet18",
     repo_id: "sabrinahartung1010/skin-lesion-resnet18",
     filename: "resnet18_ham10000_classweights.pt",
     weights_path: "/optional/local/override.pt"}   # skips the HF download if present
    """
    import torch
    from torchvision.models import resnet18

    weights_path = spec.get("weights_path")
    if weights_path and Path(weights_path).exists():
        path = weights_path
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"])

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return SkinLesionModel(model, CLASSES)
