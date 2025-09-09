# pro/imageops/scnr.py
from __future__ import annotations
import numpy as np

def apply_average_neutral_scnr(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Average Neutral SCNR (green removal).
    Expects an RGB image normalized to [0, 1]. Returns float32 in [0, 1].

    amount: 0.0 → no effect, 1.0 → full SCNR
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have three channels (RGB).")
    if not (0.0 <= amount <= 1.0):
        raise ValueError("Amount parameter must be between 0.0 and 1.0.")

    img = image.astype(np.float32, copy=False)

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    # G' = min(G, 0.5*(R + B))
    G_scnr = np.minimum(G, 0.5 * (R + B))

    scnr_img = img.copy()
    scnr_img[..., 1] = G_scnr

    # Blend original and SCNR result
    out = (1.0 - amount) * img + amount * scnr_img
    return np.clip(out, 0.0, 1.0)
