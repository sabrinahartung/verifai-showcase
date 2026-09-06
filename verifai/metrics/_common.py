"""Small shared helpers for the image metrics.

Pure NumPy / Pillow (no torch) so they are cheap and easy to reason about:
  - ITA (Individual Typology Angle): a label-free skin-tone estimate, used for
    fairness subgroups without needing annotated Fitzpatrick types.
  - a few reproducible image corruptions for the robustness metric.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ITA bins (after Kinyanjui et al. 2019) -> coarse skin-tone groups.
# Higher ITA = lighter skin.
ITA_BINS = [
    ("hell (I–II)", 41.0, 1e9),
    ("mittel (III–IV)", 19.0, 41.0),
    ("dunkel (V–VI)", -1e9, 19.0),
]


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1], shape (...,3) -> CIE Lab (D65). Vectorised."""
    m = rgb > 0.04045
    lin = np.where(m, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    # normalise by D65 white point
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883

    def f(t):
        d = 6 / 29
        return np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return np.stack([L, a, bb], axis=-1)


def estimate_ita(img: Image.Image) -> float:
    """Estimate the Individual Typology Angle of the *skin* around the lesion.

    Uses the border region (outer frame) as a skin proxy — the lesion sits in
    the centre of dermatoscopic crops — and the robust median to resist hair,
    ruler marks and vignetting.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    by, bx = max(1, h // 6), max(1, w // 6)
    frame = np.concatenate([
        arr[:by, :, :].reshape(-1, 3), arr[-by:, :, :].reshape(-1, 3),
        arr[:, :bx, :].reshape(-1, 3), arr[:, -bx:, :].reshape(-1, 3),
    ], axis=0)
    lab = _srgb_to_lab(frame)
    L = np.median(lab[:, 0])
    b = np.median(lab[:, 2])
    if abs(b) < 1e-6:
        b = 1e-6
    return float(np.degrees(np.arctan2(L - 50.0, b)))


def ita_bin(ita: float) -> str:
    for name, lo, hi in ITA_BINS:
        if lo <= ita < hi:
            return name
    return "unbestimmt"


# --- corruptions: name -> callable(PIL.Image) -> PIL.Image (seeded outside) ---
def _noise(img, sigma=18.0, rng=None):
    rng = rng or np.random.default_rng(0)
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    a = np.clip(a + rng.normal(0, sigma, a.shape), 0, 255).astype("uint8")
    return Image.fromarray(a)


def _blur(img, radius=2.0, rng=None):
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius))


def _bright(img, factor=1.4, rng=None):
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)


def _jpeg(img, quality=25, rng=None):
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CORRUPTIONS = {
    "noise": _noise,
    "blur": _blur,
    "brightness": _bright,
    "jpeg": _jpeg,
}
