import numpy as np
from PIL import Image
from setiastro.saspro.file_utils import WEBP_MAX_DIM

def save_webp(path, img, *, lossless=True, quality=95, method=4, icc_profile=None):
    arr = np.asarray(img)

    h, w = arr.shape[:2]
    if max(h, w) > WEBP_MAX_DIM:
        raise ValueError(
            f"WebP supports a maximum dimension of {WEBP_MAX_DIM} px; "
            f"this image is {w}x{h}. Save as PNG or TIFF instead."
        )

    # mono -> 3-channel, same rule as the PNG path
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    save_kwargs = dict(format="WEBP", lossless=bool(lossless),
                       quality=int(quality), method=int(method))
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    Image.fromarray(arr, mode).save(path, **save_kwargs)