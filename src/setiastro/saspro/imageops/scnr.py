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

    # G' = min(G, 0.5*(R + B)) - optimized: compute blended G directly without full array copy
    G_scnr = np.minimum(G, 0.5 * (R + B))
    
    # Blend original G and SCNR G directly: avoids allocating a full copy of the image
    G_blended = G + amount * (G_scnr - G)  # Equivalent to (1-amount)*G + amount*G_scnr
    
    # Build output array only once
    out = np.empty_like(img, dtype=np.float32)
    out[..., 0] = R
    out[..., 1] = np.clip(G_blended, 0.0, 1.0)
    out[..., 2] = B
    return out
