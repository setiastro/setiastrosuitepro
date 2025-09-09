import numpy as np
from numba import njit, prange
import cv2 
import math

@njit(parallel=True, fastmath=True)
def blend_add_numba(A, B, alpha):
    H, W, C = A.shape
    out = np.empty_like(A)
    for y in prange(H):
        for x in range(W):
            for c in range(C):
                v = A[y,x,c] + B[y,x,c] * alpha
                # clamp 0..1
                if v < 0.0: v = 0.0
                elif v > 1.0: v = 1.0
                out[y,x,c] = v
    return out

@njit(parallel=True, fastmath=True)
def blend_subtract_numba(A, B, alpha):
    H, W, C = A.shape
    out = np.empty_like(A)
    for y in prange(H):
        for x in range(W):
            for c in range(C):
                v = A[y,x,c] - B[y,x,c] * alpha
                if v < 0.0: v = 0.0
                elif v > 1.0: v = 1.0
                out[y,x,c] = v
    return out

@njit(parallel=True, fastmath=True)
def blend_multiply_numba(A, B, alpha):
    H, W, C = A.shape
    out = np.empty_like(A)
    for y in prange(H):
        for x in range(W):
            for c in range(C):
                v = (A[y,x,c] * (1-alpha)) + (A[y,x,c] * B[y,x,c] * alpha)
                if v < 0.0: v = 0.0
                elif v > 1.0: v = 1.0
                out[y,x,c] = v
    return out

@njit(parallel=True, fastmath=True)
def blend_divide_numba(A, B, alpha):
    H, W, C = A.shape
    out = np.empty_like(A)
    eps = 1e-6
    for y in prange(H):
        for x in range(W):
            for c in range(C):
                # avoid division by zero
                b = A[y,x,c] / (B[y,x,c] + eps)
                # clamp f(A,B)
                if b < 0.0: b = 0.0
                elif b > 1.0: b = 1.0
                # mix with original
                v = A[y,x,c] * (1.0 - alpha) + b * alpha
                # clamp final
                if v < 0.0: v = 0.0
                elif v > 1.0: v = 1.0
                out[y,x,c] = v
    return out

@njit(parallel=True, fastmath=True)
def blend_screen_numba(A, B, alpha):
    H, W, C = A.shape
    out = np.empty_like(A)
    for y in prange(H):
        for x in range(W):
            for c in range(C):
                # Screen: 1 - (1-A)*(1-B)
                b = 1.0 - (1.0 - A[y,x,c]) * (1.0 - B[y,x,c])
                if b < 0.0: b = 0.0
                elif b > 1.0: b = 1.0
                v = A[y,x,c] * (1.0 - alpha) + b * alpha
                if v < 0.0: v = 0.0
                elif v > 1.0: v = 1.0
                out[y,x,c] = v
    return out

@njit(parallel=True, fastmath=True)
def blend_overlay_numba(A, B, alpha):
    H, W, C = A.shape
    out = np.empty_like(A)
    for y in prange(H):
        for x in range(W):
            for c in range(C):
                a = A[y,x,c]
                b_in = B[y,x,c]
                # Overlay: if a < .5: 2*a*b, else: 1 - 2*(1-a)*(1-b)
                if a <= 0.5:
                    b = 2.0 * a * b_in
                else:
                    b = 1.0 - 2.0 * (1.0 - a) * (1.0 - b_in)
                if b < 0.0: b = 0.0
                elif b > 1.0: b = 1.0
                v = a * (1.0 - alpha) + b * alpha
                if v < 0.0: v = 0.0
                elif v > 1.0: v = 1.0
                out[y,x,c] = v
    return out

@njit(parallel=True, fastmath=True)
def blend_difference_numba(A, B, alpha):
    H, W, C = A.shape
    out = np.empty_like(A)
    for y in prange(H):
        for x in range(W):
            for c in range(C):
                # Difference: |A - B|
                b = A[y,x,c] - B[y,x,c]
                if b < 0.0: b = -b
                # clamp f(A,B) is redundant since abs() already ≥0; we cap above 1
                if b > 1.0: b = 1.0
                v = A[y,x,c] * (1.0 - alpha) + b * alpha
                if v < 0.0: v = 0.0
                elif v > 1.0: v = 1.0
                out[y,x,c] = v
    return out

@njit(parallel=True, fastmath=True)
def rescale_image_numba(image, factor):
    """
    Custom rescale function using bilinear interpolation optimized with numba.
    Supports both mono (2D) and color (3D) images.
    """
    if image.ndim == 2:
        height, width = image.shape
        new_width = int(width * factor)
        new_height = int(height * factor)
        output = np.zeros((new_height, new_width), dtype=np.float32)
        for y in prange(new_height):
            for x in prange(new_width):
                src_x = x / factor
                src_y = y / factor
                x0, y0 = int(src_x), int(src_y)
                x1 = x0 + 1 if x0 + 1 < width else width - 1
                y1 = y0 + 1 if y0 + 1 < height else height - 1
                dx = src_x - x0
                dy = src_y - y0
                output[y, x] = (image[y0, x0] * (1 - dx) * (1 - dy) +
                                image[y0, x1] * dx * (1 - dy) +
                                image[y1, x0] * (1 - dx) * dy +
                                image[y1, x1] * dx * dy)
        return output
    else:
        height, width, channels = image.shape
        new_width = int(width * factor)
        new_height = int(height * factor)
        output = np.zeros((new_height, new_width, channels), dtype=np.float32)
        for y in prange(new_height):
            for x in prange(new_width):
                src_x = x / factor
                src_y = y / factor
                x0, y0 = int(src_x), int(src_y)
                x1 = x0 + 1 if x0 + 1 < width else width - 1
                y1 = y0 + 1 if y0 + 1 < height else height - 1
                dx = src_x - x0
                dy = src_y - y0
                for c in range(channels):
                    output[y, x, c] = (image[y0, x0, c] * (1 - dx) * (1 - dy) +
                                       image[y0, x1, c] * dx * (1 - dy) +
                                       image[y1, x0, c] * (1 - dx) * dy +
                                       image[y1, x1, c] * dx * dy)
        return output

@njit(parallel=True, fastmath=True)
def bin2x2_numba(image):
    """
    Downsample the image by 2×2 via simple averaging (“integer binning”).
    Works on 2D (H×W) or 3D (H×W×C) arrays.  If dimensions aren’t even,
    the last row/column is dropped.
    """
    h, w = image.shape[:2]
    h2 = h // 2
    w2 = w // 2

    # allocate output
    if image.ndim == 2:
        out = np.empty((h2, w2), dtype=np.float32)
        for i in prange(h2):
            for j in prange(w2):
                # average 2×2 block
                s = image[2*i  , 2*j  ] \
                  + image[2*i+1, 2*j  ] \
                  + image[2*i  , 2*j+1] \
                  + image[2*i+1, 2*j+1]
                out[i, j] = s * 0.25
    else:
        c = image.shape[2]
        out = np.empty((h2, w2, c), dtype=np.float32)
        for i in prange(h2):
            for j in prange(w2):
                for k in range(c):
                    s = image[2*i  , 2*j  , k] \
                      + image[2*i+1, 2*j  , k] \
                      + image[2*i  , 2*j+1, k] \
                      + image[2*i+1, 2*j+1, k]
                    out[i, j, k] = s * 0.25

    return out

@njit(parallel=True, fastmath=True)
def flip_horizontal_numba(image):
    """
    Flips an image horizontally using Numba JIT.
    Works with both mono (2D) and color (3D) images.
    """
    if image.ndim == 2:
        height, width = image.shape
        output = np.empty((height, width), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                output[y, x] = image[y, width - x - 1]
        return output
    else:
        height, width, channels = image.shape
        output = np.empty((height, width, channels), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                for c in range(channels):
                    output[y, x, c] = image[y, width - x - 1, c]
        return output


@njit(parallel=True, fastmath=True)
def flip_vertical_numba(image):
    """
    Flips an image vertically using Numba JIT.
    Works with both mono (2D) and color (3D) images.
    """
    if image.ndim == 2:
        height, width = image.shape
        output = np.empty((height, width), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                output[y, x] = image[height - y - 1, x]
        return output
    else:
        height, width, channels = image.shape
        output = np.empty((height, width, channels), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                for c in range(channels):
                    output[y, x, c] = image[height - y - 1, x, c]
        return output


@njit(parallel=True, fastmath=True)
def rotate_90_clockwise_numba(image):
    """
    Rotates the image 90 degrees clockwise.
    Works with both mono (2D) and color (3D) images.
    """
    if image.ndim == 2:
        height, width = image.shape
        output = np.empty((width, height), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                output[x, height - 1 - y] = image[y, x]
        return output
    else:
        height, width, channels = image.shape
        output = np.empty((width, height, channels), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                for c in range(channels):
                    output[x, height - 1 - y, c] = image[y, x, c]
        return output


@njit(parallel=True, fastmath=True)
def rotate_90_counterclockwise_numba(image):
    """
    Rotates the image 90 degrees counterclockwise.
    Works with both mono (2D) and color (3D) images.
    """
    if image.ndim == 2:
        height, width = image.shape
        output = np.empty((width, height), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                output[width - 1 - x, y] = image[y, x]
        return output
    else:
        height, width, channels = image.shape
        output = np.empty((width, height, channels), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                for c in range(channels):
                    output[width - 1 - x, y, c] = image[y, x, c]
        return output


@njit(parallel=True, fastmath=True)
def invert_image_numba(image):
    """
    Inverts an image (1 - pixel value) using Numba JIT.
    Works with both mono (2D) and color (3D) images.
    """
    if image.ndim == 2:
        height, width = image.shape
        output = np.empty((height, width), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                output[y, x] = 1.0 - image[y, x]
        return output
    else:
        height, width, channels = image.shape
        output = np.empty((height, width, channels), dtype=image.dtype)
        for y in prange(height):
            for x in prange(width):
                for c in range(channels):
                    output[y, x, c] = 1.0 - image[y, x, c]
        return output



@njit(parallel=True, fastmath=True)
def apply_flat_division_numba_2d(image, master_flat, master_bias=None):
    """
    Mono version: image.shape == (H,W)
    """
    if master_bias is not None:
        master_flat = master_flat - master_bias
        image = image - master_bias

    median_flat = np.mean(master_flat)
    height, width = image.shape

    for y in prange(height):
        for x in range(width):
            image[y, x] /= (master_flat[y, x] / median_flat)

    return image


@njit(parallel=True, fastmath=True)
def apply_flat_division_numba_3d(image, master_flat, master_bias=None):
    """
    Color version: image.shape == (H,W,C)
    """
    if master_bias is not None:
        master_flat = master_flat - master_bias
        image = image - master_bias

    median_flat = np.mean(master_flat)
    height, width, channels = image.shape

    for y in prange(height):
        for x in range(width):
            for c in range(channels):
                image[y, x, c] /= (master_flat[y, x, c] / median_flat)

    return image

def apply_flat_division_numba(image, master_flat, master_bias=None):
    """
    Dispatcher that calls the correct Numba function
    depending on whether 'image' is 2D or 3D.
    """
    if image.ndim == 2:
        # Mono
        return apply_flat_division_numba_2d(image, master_flat, master_bias)
    elif image.ndim == 3:
        # Color
        return apply_flat_division_numba_3d(image, master_flat, master_bias)
    else:
        raise ValueError(f"apply_flat_division_numba: expected 2D or 3D, got shape {image.shape}")


@njit(parallel=True)
def subtract_dark_3d(frames, dark_frame):
    """
    For mono stack:
      frames.shape == (F,H,W)
      dark_frame.shape == (H,W)
    Returns the same shape (F,H,W).
    """
    num_frames, height, width = frames.shape
    result = np.empty_like(frames, dtype=np.float32)

    for i in prange(num_frames):
        # Subtract the dark frame from each 2D slice
        result[i] = frames[i] - dark_frame

    return result


@njit(parallel=True)
def subtract_dark_4d(frames, dark_frame):
    """
    For color stack:
      frames.shape == (F,H,W,C)
      dark_frame.shape == (H,W,C)
    Returns the same shape (F,H,W,C).
    """
    num_frames, height, width, channels = frames.shape
    result = np.empty_like(frames, dtype=np.float32)

    for i in prange(num_frames):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    result[i, y, x, c] = frames[i, y, x, c] - dark_frame[y, x, c]

    return result

def subtract_dark(frames, dark_frame):
    """
    Dispatcher function that calls the correct Numba function
    depending on whether 'frames' is 3D or 4D.
    """
    if frames.ndim == 3:
        # frames: (F,H,W), dark_frame: (H,W)
        return subtract_dark_3d(frames, dark_frame)
    elif frames.ndim == 4:
        # frames: (F,H,W,C), dark_frame: (H,W,C)
        return subtract_dark_4d(frames, dark_frame)
    else:
        raise ValueError(f"subtract_dark: frames must be 3D or 4D, got {frames.shape}")


import numpy as np
from numba import njit, prange

# -------------------------------
# Windsorized Sigma Clipping (Weighted, Iterative)
# -------------------------------

@njit(parallel=True, fastmath=True)
def windsorized_sigma_clip_weighted_3d_iter(stack, weights, lower=2.5, upper=2.5, iterations=2):
    """
    Iterative Weighted Windsorized Sigma Clipping for a 3D mono stack.
      stack.shape == (F,H,W)
      weights.shape can be (F,) or (F,H,W).
    Returns a tuple:
      (clipped, rejection_mask)
    where:
      clipped is a 2D image (H,W),
      rejection_mask is a boolean array of shape (F,H,W) with True indicating rejection.
    """
    num_frames, height, width = stack.shape
    clipped = np.zeros((height, width), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width), dtype=np.bool_)

    # Check weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("windsorized_sigma_clip_weighted_3d_iter: mismatch in shapes for 3D stack & weights")

    for i in prange(height):
        for j in range(width):
            pixel_values = stack[:, i, j]  # shape=(F,)
            if weights.ndim == 1:
                pixel_weights = weights[:]  # shape (F,)
            else:
                pixel_weights = weights[:, i, j]
            # Start with nonzero pixels as valid
            valid_mask = pixel_values != 0
            for _ in range(iterations):
                if np.sum(valid_mask) == 0:
                    break
                valid_vals = pixel_values[valid_mask]
                median_val = np.median(valid_vals)
                std_dev = np.std(valid_vals)
                lower_bound = median_val - lower * std_dev
                upper_bound = median_val + upper * std_dev
                valid_mask = valid_mask & (pixel_values >= lower_bound) & (pixel_values <= upper_bound)
            # Record rejections: a pixel is rejected if not valid.
            for f in range(num_frames):
                rej_mask[f, i, j] = not valid_mask[f]
            valid_vals = pixel_values[valid_mask]
            valid_w = pixel_weights[valid_mask]
            wsum = np.sum(valid_w)
            if wsum > 0:
                clipped[i, j] = np.sum(valid_vals * valid_w) / wsum
            else:
                nonzero = pixel_values[pixel_values != 0]
                if nonzero.size > 0:
                    clipped[i, j] = np.median(nonzero)
                else:
                    clipped[i, j] = 0.0
    return clipped, rej_mask


@njit(parallel=True, fastmath=True)
def windsorized_sigma_clip_weighted_4d_iter(stack, weights, lower=2.5, upper=2.5, iterations=2):
    """
    Iterative Weighted Windsorized Sigma Clipping for a 4D color stack.
      stack.shape == (F,H,W,C)
      weights.shape can be (F,) or (F,H,W,C).
    Returns a tuple:
      (clipped, rejection_mask)
    where:
      clipped is a 3D image (H,W,C),
      rejection_mask is a boolean array of shape (F,H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.zeros((height, width, channels), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width, channels), dtype=np.bool_)

    # Check weights shape
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("windsorized_sigma_clip_weighted_4d_iter: mismatch in shapes for 4D stack & weights")

    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pixel_values = stack[:, i, j, c]  # shape=(F,)
                if weights.ndim == 1:
                    pixel_weights = weights[:]
                else:
                    pixel_weights = weights[:, i, j, c]
                valid_mask = pixel_values != 0
                for _ in range(iterations):
                    if np.sum(valid_mask) == 0:
                        break
                    valid_vals = pixel_values[valid_mask]
                    median_val = np.median(valid_vals)
                    std_dev = np.std(valid_vals)
                    lower_bound = median_val - lower * std_dev
                    upper_bound = median_val + upper * std_dev
                    valid_mask = valid_mask & (pixel_values >= lower_bound) & (pixel_values <= upper_bound)
                for f in range(num_frames):
                    rej_mask[f, i, j, c] = not valid_mask[f]
                valid_vals = pixel_values[valid_mask]
                valid_w = pixel_weights[valid_mask]
                wsum = np.sum(valid_w)
                if wsum > 0:
                    clipped[i, j, c] = np.sum(valid_vals * valid_w) / wsum
                else:
                    nonzero = pixel_values[pixel_values != 0]
                    if nonzero.size > 0:
                        clipped[i, j, c] = np.median(nonzero)
                    else:
                        clipped[i, j, c] = 0.0
    return clipped, rej_mask


def windsorized_sigma_clip_weighted(stack, weights, lower=2.5, upper=2.5, iterations=2):
    """
    Dispatcher that calls the appropriate iterative Numba function.
    Now returns (clipped, rejection_mask).
    """
    if stack.ndim == 3:
        return windsorized_sigma_clip_weighted_3d_iter(stack, weights, lower, upper, iterations)
    elif stack.ndim == 4:
        return windsorized_sigma_clip_weighted_4d_iter(stack, weights, lower, upper, iterations)
    else:
        raise ValueError(f"windsorized_sigma_clip_weighted: stack must be 3D or 4D, got {stack.shape}")


# -------------------------------
# Kappa-Sigma Clipping (Weighted)
# -------------------------------

@njit(parallel=True, fastmath=True)
def kappa_sigma_clip_weighted_3d(stack, weights, kappa=2.5, iterations=3):
    """
    Kappa-Sigma Clipping for a 3D mono stack.
      stack.shape == (F,H,W)
    Returns a tuple: (clipped, rejection_mask)
    where rejection_mask is of shape (F,H,W) indicating per-frame rejections.
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    
    for i in prange(height):
        for j in range(width):
            pixel_values = stack[:, i, j].copy()
            if weights.ndim == 1:
                pixel_weights = weights[:]
            else:
                pixel_weights = weights[:, i, j].copy()
            # Initialize tracking of indices
            current_idx = np.empty(num_frames, dtype=np.int64)
            for f in range(num_frames):
                current_idx[f] = f
            current_vals = pixel_values
            current_w = pixel_weights
            current_indices = current_idx
            med = 0.0
            for _ in range(iterations):
                if current_vals.size == 0:
                    break
                med = np.median(current_vals)
                std = np.std(current_vals)
                lower_bound = med - kappa * std
                upper_bound = med + kappa * std
                valid = (current_vals != 0) & (current_vals >= lower_bound) & (current_vals <= upper_bound)
                current_vals = current_vals[valid]
                current_w = current_w[valid]
                current_indices = current_indices[valid]
            # Mark rejected: frames not in current_indices are rejected.
            for f in range(num_frames):
                # Check if f is in current_indices
                found = False
                for k in range(current_indices.size):
                    if current_indices[k] == f:
                        found = True
                        break
                if not found:
                    rej_mask[f, i, j] = True
                else:
                    rej_mask[f, i, j] = False
            if current_w.size > 0 and current_w.sum() > 0:
                clipped[i, j] = np.sum(current_vals * current_w) / current_w.sum()
            else:
                clipped[i, j] = med
    return clipped, rej_mask


@njit(parallel=True, fastmath=True)
def kappa_sigma_clip_weighted_4d(stack, weights, kappa=2.5, iterations=3):
    """
    Kappa-Sigma Clipping for a 4D color stack.
      stack.shape == (F,H,W,C)
    Returns (clipped, rejection_mask) where rejection_mask has shape (F,H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width, channels), dtype=np.bool_)
    
    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pixel_values = stack[:, i, j, c].copy()
                if weights.ndim == 1:
                    pixel_weights = weights[:]
                else:
                    pixel_weights = weights[:, i, j, c].copy()
                current_idx = np.empty(num_frames, dtype=np.int64)
                for f in range(num_frames):
                    current_idx[f] = f
                current_vals = pixel_values
                current_w = pixel_weights
                current_indices = current_idx
                med = 0.0
                for _ in range(iterations):
                    if current_vals.size == 0:
                        break
                    med = np.median(current_vals)
                    std = np.std(current_vals)
                    lower_bound = med - kappa * std
                    upper_bound = med + kappa * std
                    valid = (current_vals != 0) & (current_vals >= lower_bound) & (current_vals <= upper_bound)
                    current_vals = current_vals[valid]
                    current_w = current_w[valid]
                    current_indices = current_indices[valid]
                for f in range(num_frames):
                    found = False
                    for k in range(current_indices.size):
                        if current_indices[k] == f:
                            found = True
                            break
                    if not found:
                        rej_mask[f, i, j, c] = True
                    else:
                        rej_mask[f, i, j, c] = False
                if current_w.size > 0 and current_w.sum() > 0:
                    clipped[i, j, c] = np.sum(current_vals * current_w) / current_w.sum()
                else:
                    clipped[i, j, c] = med
    return clipped, rej_mask


def kappa_sigma_clip_weighted(stack, weights, kappa=2.5, iterations=3):
    """
    Dispatcher that returns (clipped, rejection_mask) for kappa-sigma clipping.
    """
    if stack.ndim == 3:
        return kappa_sigma_clip_weighted_3d(stack, weights, kappa, iterations)
    elif stack.ndim == 4:
        return kappa_sigma_clip_weighted_4d(stack, weights, kappa, iterations)
    else:
        raise ValueError(f"kappa_sigma_clip_weighted: stack must be 3D or 4D, got {stack.shape}")


# -------------------------------
# Trimmed Mean (Weighted)
# -------------------------------

@njit(parallel=True, fastmath=True)
def trimmed_mean_weighted_3d(stack, weights, trim_fraction=0.1):
    """
    Trimmed Mean for a 3D mono stack.
      stack.shape == (F,H,W)
    Returns (clipped, rejection_mask) where rejection_mask (F,H,W) flags frames that were trimmed.
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    
    for i in prange(height):
        for j in range(width):
            pix_all = stack[:, i, j]
            if weights.ndim == 1:
                w_all = weights[:]
            else:
                w_all = weights[:, i, j]
            # Exclude zeros and record original indices.
            valid = pix_all != 0
            pix = pix_all[valid]
            w = w_all[valid]
            orig_idx = np.empty(pix_all.shape[0], dtype=np.int64)
            count = 0
            for f in range(num_frames):
                if valid[f]:
                    orig_idx[count] = f
                    count += 1
            n = pix.size
            if n == 0:
                clipped[i, j] = 0.0
                # Mark all as rejected.
                for f in range(num_frames):
                    if not valid[f]:
                        rej_mask[f, i, j] = True
                continue
            trim = int(trim_fraction * n)
            order = np.argsort(pix)
            # Determine which indices (in the valid list) are kept.
            if n > 2 * trim:
                keep_order = order[trim:n - trim]
            else:
                keep_order = order
            # Build a mask for the valid pixels (length n) that are kept.
            keep_mask = np.zeros(n, dtype=np.bool_)
            for k in range(keep_order.size):
                keep_mask[keep_order[k]] = True
            # Map back to original frame indices.
            for idx in range(n):
                frame = orig_idx[idx]
                if not keep_mask[idx]:
                    rej_mask[frame, i, j] = True
                else:
                    rej_mask[frame, i, j] = False
            # Compute weighted average of kept values.
            sorted_pix = pix[order]
            sorted_w = w[order]
            if n > 2 * trim:
                trimmed_values = sorted_pix[trim:n - trim]
                trimmed_weights = sorted_w[trim:n - trim]
            else:
                trimmed_values = sorted_pix
                trimmed_weights = sorted_w
            wsum = trimmed_weights.sum()
            if wsum > 0:
                clipped[i, j] = np.sum(trimmed_values * trimmed_weights) / wsum
            else:
                clipped[i, j] = np.median(trimmed_values)
    return clipped, rej_mask


@njit(parallel=True, fastmath=True)
def trimmed_mean_weighted_4d(stack, weights, trim_fraction=0.1):
    """
    Trimmed Mean for a 4D color stack.
      stack.shape == (F,H,W,C)
    Returns (clipped, rejection_mask) where rejection_mask has shape (F,H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width, channels), dtype=np.bool_)
    
    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pix_all = stack[:, i, j, c]
                if weights.ndim == 1:
                    w_all = weights[:]
                else:
                    w_all = weights[:, i, j, c]
                valid = pix_all != 0
                pix = pix_all[valid]
                w = w_all[valid]
                orig_idx = np.empty(pix_all.shape[0], dtype=np.int64)
                count = 0
                for f in range(num_frames):
                    if valid[f]:
                        orig_idx[count] = f
                        count += 1
                n = pix.size
                if n == 0:
                    clipped[i, j, c] = 0.0
                    for f in range(num_frames):
                        if not valid[f]:
                            rej_mask[f, i, j, c] = True
                    continue
                trim = int(trim_fraction * n)
                order = np.argsort(pix)
                if n > 2 * trim:
                    keep_order = order[trim:n - trim]
                else:
                    keep_order = order
                keep_mask = np.zeros(n, dtype=np.bool_)
                for k in range(keep_order.size):
                    keep_mask[keep_order[k]] = True
                for idx in range(n):
                    frame = orig_idx[idx]
                    if not keep_mask[idx]:
                        rej_mask[frame, i, j, c] = True
                    else:
                        rej_mask[frame, i, j, c] = False
                sorted_pix = pix[order]
                sorted_w = w[order]
                if n > 2 * trim:
                    trimmed_values = sorted_pix[trim:n - trim]
                    trimmed_weights = sorted_w[trim:n - trim]
                else:
                    trimmed_values = sorted_pix
                    trimmed_weights = sorted_w
                wsum = trimmed_weights.sum()
                if wsum > 0:
                    clipped[i, j, c] = np.sum(trimmed_values * trimmed_weights) / wsum
                else:
                    clipped[i, j, c] = np.median(trimmed_values)
    return clipped, rej_mask


def trimmed_mean_weighted(stack, weights, trim_fraction=0.1):
    """
    Dispatcher that returns (clipped, rejection_mask) for trimmed mean.
    """
    if stack.ndim == 3:
        return trimmed_mean_weighted_3d(stack, weights, trim_fraction)
    elif stack.ndim == 4:
        return trimmed_mean_weighted_4d(stack, weights, trim_fraction)
    else:
        raise ValueError(f"trimmed_mean_weighted: stack must be 3D or 4D, got {stack.shape}")


# -------------------------------
# Extreme Studentized Deviate (ESD) Clipping (Weighted)
# -------------------------------

@njit(parallel=True, fastmath=True)
def esd_clip_weighted_3d(stack, weights, threshold=3.0):
    """
    ESD Clipping for a 3D mono stack.
      stack.shape == (F,H,W)
    Returns (clipped, rejection_mask) where rejection_mask has shape (F,H,W).
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("esd_clip_weighted_3d: mismatch in shapes for 3D stack & weights")

    for i in prange(height):
        for j in range(width):
            pix = stack[:, i, j]
            if weights.ndim == 1:
                w = weights[:]
            else:
                w = weights[:, i, j]
            valid = pix != 0
            values = pix[valid]
            wvals = w[valid]
            if values.size == 0:
                clipped[i, j] = 0.0
                for f in range(num_frames):
                    if not valid[f]:
                        rej_mask[f, i, j] = True
                continue
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                clipped[i, j] = mean_val
                for f in range(num_frames):
                    rej_mask[f, i, j] = False
                continue
            z_scores = np.abs((values - mean_val) / std_val)
            valid2 = z_scores < threshold
            # Mark rejected: for the valid entries, use valid2.
            idx = 0
            for f in range(num_frames):
                if valid[f]:
                    if not valid2[idx]:
                        rej_mask[f, i, j] = True
                    else:
                        rej_mask[f, i, j] = False
                    idx += 1
                else:
                    rej_mask[f, i, j] = True
            values = values[valid2]
            wvals = wvals[valid2]
            wsum = wvals.sum()
            if wsum > 0:
                clipped[i, j] = np.sum(values * wvals) / wsum
            else:
                clipped[i, j] = mean_val
    return clipped, rej_mask


@njit(parallel=True, fastmath=True)
def esd_clip_weighted_4d(stack, weights, threshold=3.0):
    """
    ESD Clipping for a 4D color stack.
      stack.shape == (F,H,W,C)
    Returns (clipped, rejection_mask) where rejection_mask has shape (F,H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width, channels), dtype=np.bool_)
    
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("esd_clip_weighted_4d: mismatch in shapes for 4D stack & weights")

    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pix = stack[:, i, j, c]
                if weights.ndim == 1:
                    w = weights[:]
                else:
                    w = weights[:, i, j, c]
                valid = pix != 0
                values = pix[valid]
                wvals = w[valid]
                if values.size == 0:
                    clipped[i, j, c] = 0.0
                    for f in range(num_frames):
                        if not valid[f]:
                            rej_mask[f, i, j, c] = True
                    continue
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val == 0:
                    clipped[i, j, c] = mean_val
                    for f in range(num_frames):
                        rej_mask[f, i, j, c] = False
                    continue
                z_scores = np.abs((values - mean_val) / std_val)
                valid2 = z_scores < threshold
                idx = 0
                for f in range(num_frames):
                    if valid[f]:
                        if not valid2[idx]:
                            rej_mask[f, i, j, c] = True
                        else:
                            rej_mask[f, i, j, c] = False
                        idx += 1
                    else:
                        rej_mask[f, i, j, c] = True
                values = values[valid2]
                wvals = wvals[valid2]
                wsum = wvals.sum()
                if wsum > 0:
                    clipped[i, j, c] = np.sum(values * wvals) / wsum
                else:
                    clipped[i, j, c] = mean_val
    return clipped, rej_mask


def esd_clip_weighted(stack, weights, threshold=3.0):
    """
    Dispatcher that returns (clipped, rejection_mask) for ESD clipping.
    """
    if stack.ndim == 3:
        return esd_clip_weighted_3d(stack, weights, threshold)
    elif stack.ndim == 4:
        return esd_clip_weighted_4d(stack, weights, threshold)
    else:
        raise ValueError(f"esd_clip_weighted: stack must be 3D or 4D, got {stack.shape}")


# -------------------------------
# Biweight Location (Weighted)
# -------------------------------

@njit(parallel=True, fastmath=True)
def biweight_location_weighted_3d(stack, weights, tuning_constant=6.0):
    """
    Biweight Location for a 3D mono stack.
      stack.shape == (F,H,W)
    Returns (clipped, rejection_mask) where rejection_mask has shape (F,H,W).
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("biweight_location_weighted_3d: mismatch in shapes for 3D stack & weights")
    
    for i in prange(height):
        for j in range(width):
            x = stack[:, i, j]
            if weights.ndim == 1:
                w = weights[:]
            else:
                w = weights[:, i, j]
            valid = x != 0
            x_valid = x[valid]
            w_valid = w[valid]
            # Record rejections for zeros:
            for f in range(num_frames):
                if not valid[f]:
                    rej_mask[f, i, j] = True
                else:
                    rej_mask[f, i, j] = False  # initialize as accepted; may update below
            n = x_valid.size
            if n == 0:
                clipped[i, j] = 0.0
                continue
            M = np.median(x_valid)
            mad = np.median(np.abs(x_valid - M))
            if mad == 0:
                clipped[i, j] = M
                continue
            u = (x_valid - M) / (tuning_constant * mad)
            mask = np.abs(u) < 1
            # Mark frames that were excluded by the biweight rejection:
            idx = 0
            for f in range(num_frames):
                if valid[f]:
                    if not mask[idx]:
                        rej_mask[f, i, j] = True
                    idx += 1
            x_masked = x_valid[mask]
            w_masked = w_valid[mask]
            numerator = ((x_masked - M) * (1 - u[mask]**2)**2 * w_masked).sum()
            denominator = ((1 - u[mask]**2)**2 * w_masked).sum()
            if denominator != 0:
                biweight = M + numerator / denominator
            else:
                biweight = M
            clipped[i, j] = biweight
    return clipped, rej_mask


@njit(parallel=True, fastmath=True)
def biweight_location_weighted_4d(stack, weights, tuning_constant=6.0):
    """
    Biweight Location for a 4D color stack.
      stack.shape == (F,H,W,C)
    Returns (clipped, rejection_mask) where rejection_mask has shape (F,H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width, channels), dtype=np.bool_)
    
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("biweight_location_weighted_4d: mismatch in shapes for 4D stack & weights")
    
    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                x = stack[:, i, j, c]
                if weights.ndim == 1:
                    w = weights[:]
                else:
                    w = weights[:, i, j, c]
                valid = x != 0
                x_valid = x[valid]
                w_valid = w[valid]
                for f in range(num_frames):
                    if not valid[f]:
                        rej_mask[f, i, j, c] = True
                    else:
                        rej_mask[f, i, j, c] = False
                n = x_valid.size
                if n == 0:
                    clipped[i, j, c] = 0.0
                    continue
                M = np.median(x_valid)
                mad = np.median(np.abs(x_valid - M))
                if mad == 0:
                    clipped[i, j, c] = M
                    continue
                u = (x_valid - M) / (tuning_constant * mad)
                mask = np.abs(u) < 1
                idx = 0
                for f in range(num_frames):
                    if valid[f]:
                        if not mask[idx]:
                            rej_mask[f, i, j, c] = True
                        idx += 1
                x_masked = x_valid[mask]
                w_masked = w_valid[mask]
                numerator = ((x_masked - M) * (1 - u[mask]**2)**2 * w_masked).sum()
                denominator = ((1 - u[mask]**2)**2 * w_masked).sum()
                if denominator != 0:
                    biweight = M + numerator / denominator
                else:
                    biweight = M
                clipped[i, j, c] = biweight
    return clipped, rej_mask


def biweight_location_weighted(stack, weights, tuning_constant=6.0):
    """
    Dispatcher that returns (clipped, rejection_mask) for biweight location.
    """
    if stack.ndim == 3:
        return biweight_location_weighted_3d(stack, weights, tuning_constant)
    elif stack.ndim == 4:
        return biweight_location_weighted_4d(stack, weights, tuning_constant)
    else:
        raise ValueError(f"biweight_location_weighted: stack must be 3D or 4D, got {stack.shape}")


# -------------------------------
# Modified Z-Score Clipping (Weighted)
# -------------------------------

@njit(parallel=True, fastmath=True)
def modified_zscore_clip_weighted_3d(stack, weights, threshold=3.5):
    """
    Modified Z-Score Clipping for a 3D mono stack.
      stack.shape == (F,H,W)
    Returns (clipped, rejection_mask) with rejection_mask shape (F,H,W).
    """
    num_frames, height, width = stack.shape
    clipped = np.empty((height, width), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 3 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("modified_zscore_clip_weighted_3d: mismatch in shapes for 3D stack & weights")
    
    for i in prange(height):
        for j in range(width):
            x = stack[:, i, j]
            if weights.ndim == 1:
                w = weights[:]
            else:
                w = weights[:, i, j]
            valid = x != 0
            x_valid = x[valid]
            w_valid = w[valid]
            if x_valid.size == 0:
                clipped[i, j] = 0.0
                for f in range(num_frames):
                    if not valid[f]:
                        rej_mask[f, i, j] = True
                continue
            median_val = np.median(x_valid)
            mad = np.median(np.abs(x_valid - median_val))
            if mad == 0:
                clipped[i, j] = median_val
                for f in range(num_frames):
                    rej_mask[f, i, j] = False
                continue
            modified_z = 0.6745 * (x_valid - median_val) / mad
            valid2 = np.abs(modified_z) < threshold
            idx = 0
            for f in range(num_frames):
                if valid[f]:
                    if not valid2[idx]:
                        rej_mask[f, i, j] = True
                    else:
                        rej_mask[f, i, j] = False
                    idx += 1
                else:
                    rej_mask[f, i, j] = True
            x_final = x_valid[valid2]
            w_final = w_valid[valid2]
            wsum = w_final.sum()
            if wsum > 0:
                clipped[i, j] = np.sum(x_final * w_final) / wsum
            else:
                clipped[i, j] = median_val
    return clipped, rej_mask


@njit(parallel=True, fastmath=True)
def modified_zscore_clip_weighted_4d(stack, weights, threshold=3.5):
    """
    Modified Z-Score Clipping for a 4D color stack.
      stack.shape == (F,H,W,C)
    Returns (clipped, rejection_mask) with rejection_mask shape (F,H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.empty((height, width, channels), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width, channels), dtype=np.bool_)
    
    if weights.ndim == 1 and weights.shape[0] == num_frames:
        pass
    elif weights.ndim == 4 and weights.shape == stack.shape:
        pass
    else:
        raise ValueError("modified_zscore_clip_weighted_4d: mismatch in shapes for 4D stack & weights")
    
    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                x = stack[:, i, j, c]
                if weights.ndim == 1:
                    w = weights[:]
                else:
                    w = weights[:, i, j, c]
                valid = x != 0
                x_valid = x[valid]
                w_valid = w[valid]
                if x_valid.size == 0:
                    clipped[i, j, c] = 0.0
                    for f in range(num_frames):
                        if not valid[f]:
                            rej_mask[f, i, j, c] = True
                    continue
                median_val = np.median(x_valid)
                mad = np.median(np.abs(x_valid - median_val))
                if mad == 0:
                    clipped[i, j, c] = median_val
                    for f in range(num_frames):
                        rej_mask[f, i, j, c] = False
                    continue
                modified_z = 0.6745 * (x_valid - median_val) / mad
                valid2 = np.abs(modified_z) < threshold
                idx = 0
                for f in range(num_frames):
                    if valid[f]:
                        if not valid2[idx]:
                            rej_mask[f, i, j, c] = True
                        else:
                            rej_mask[f, i, j, c] = False
                        idx += 1
                    else:
                        rej_mask[f, i, j, c] = True
                x_final = x_valid[valid2]
                w_final = w_valid[valid2]
                wsum = w_final.sum()
                if wsum > 0:
                    clipped[i, j, c] = np.sum(x_final * w_final) / wsum
                else:
                    clipped[i, j, c] = median_val
    return clipped, rej_mask


def modified_zscore_clip_weighted(stack, weights, threshold=3.5):
    """
    Dispatcher that returns (clipped, rejection_mask) for modified z-score clipping.
    """
    if stack.ndim == 3:
        return modified_zscore_clip_weighted_3d(stack, weights, threshold)
    elif stack.ndim == 4:
        return modified_zscore_clip_weighted_4d(stack, weights, threshold)
    else:
        raise ValueError(f"modified_zscore_clip_weighted: stack must be 3D or 4D, got {stack.shape}")


# -------------------------------
# Windsorized Sigma Clipping (Non-weighted)
# -------------------------------

@njit(parallel=True, fastmath=True)
def windsorized_sigma_clip_3d(stack, lower=2.5, upper=2.5):
    """
    Windsorized Sigma Clipping for a 3D mono stack (non-weighted).
      stack.shape == (F,H,W)
    Returns (clipped, rejection_mask) where rejection_mask is (F,H,W).
    """
    num_frames, height, width = stack.shape
    clipped = np.zeros((height, width), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    
    for i in prange(height):
        for j in range(width):
            pixel_values = stack[:, i, j]
            median_val = np.median(pixel_values)
            std_dev = np.std(pixel_values)
            lower_bound = median_val - lower * std_dev
            upper_bound = median_val + upper * std_dev
            valid = (pixel_values >= lower_bound) & (pixel_values <= upper_bound)
            for f in range(num_frames):
                rej_mask[f, i, j] = not valid[f]
            valid_vals = pixel_values[valid]
            if valid_vals.size > 0:
                clipped[i, j] = np.mean(valid_vals)
            else:
                clipped[i, j] = median_val
    return clipped, rej_mask


@njit(parallel=True, fastmath=True)
def windsorized_sigma_clip_4d(stack, lower=2.5, upper=2.5):
    """
    Windsorized Sigma Clipping for a 4D color stack (non-weighted).
      stack.shape == (F,H,W,C)
    Returns (clipped, rejection_mask) where rejection_mask is (F,H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    clipped = np.zeros((height, width, channels), dtype=np.float32)
    rej_mask = np.zeros((num_frames, height, width, channels), dtype=np.bool_)
    
    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                pixel_values = stack[:, i, j, c]
                median_val = np.median(pixel_values)
                std_dev = np.std(pixel_values)
                lower_bound = median_val - lower * std_dev
                upper_bound = median_val + upper * std_dev
                valid = (pixel_values >= lower_bound) & (pixel_values <= upper_bound)
                for f in range(num_frames):
                    rej_mask[f, i, j, c] = not valid[f]
                valid_vals = pixel_values[valid]
                if valid_vals.size > 0:
                    clipped[i, j, c] = np.mean(valid_vals)
                else:
                    clipped[i, j, c] = median_val
    return clipped, rej_mask


def windsorized_sigma_clip(stack, lower=2.5, upper=2.5):
    """
    Dispatcher function that calls either the 3D or 4D specialized Numba function,
    depending on 'stack.ndim'.
    """
    if stack.ndim == 3:
        return windsorized_sigma_clip_3d(stack, lower, upper)
    elif stack.ndim == 4:
        return windsorized_sigma_clip_4d(stack, lower, upper)
    else:
        raise ValueError(f"windsorized_sigma_clip: stack must be 3D or 4D, got {stack.shape}")

def max_value_stack(stack, weights=None):
    """
    Stacking by taking the maximum value along the frame axis.
    Returns (clipped, rejection_mask) for compatibility:
      - clipped: H×W (or H×W×C)
      - rejection_mask: same shape as stack, all False
    """
    clipped = np.max(stack, axis=0)
    rej_mask = np.zeros(stack.shape, dtype=bool)
    return clipped, rej_mask

@njit(parallel=True)
def subtract_dark_with_pedestal_3d(frames, dark_frame, pedestal):
    """
    For mono stack:
      frames.shape == (F,H,W)
      dark_frame.shape == (H,W)
    Adds 'pedestal' after subtracting dark_frame from each frame.
    Returns the same shape (F,H,W).
    """
    num_frames, height, width = frames.shape
    result = np.empty_like(frames, dtype=np.float32)

    # Validate dark_frame shape
    if dark_frame.ndim != 2 or dark_frame.shape != (height, width):
        raise ValueError(
            "subtract_dark_with_pedestal_3d: for 3D frames, dark_frame must be 2D (H,W)"
        )

    for i in prange(num_frames):
        for y in range(height):
            for x in range(width):
                result[i, y, x] = frames[i, y, x] - dark_frame[y, x] + pedestal

    return result

@njit(parallel=True)
def subtract_dark_with_pedestal_4d(frames, dark_frame, pedestal):
    """
    For color stack:
      frames.shape == (F,H,W,C)
      dark_frame.shape == (H,W,C)
    Adds 'pedestal' after subtracting dark_frame from each frame.
    Returns the same shape (F,H,W,C).
    """
    num_frames, height, width, channels = frames.shape
    result = np.empty_like(frames, dtype=np.float32)

    # Validate dark_frame shape
    if dark_frame.ndim != 3 or dark_frame.shape != (height, width, channels):
        raise ValueError(
            "subtract_dark_with_pedestal_4d: for 4D frames, dark_frame must be 3D (H,W,C)"
        )

    for i in prange(num_frames):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    result[i, y, x, c] = frames[i, y, x, c] - dark_frame[y, x, c] + pedestal

    return result

def subtract_dark_with_pedestal(frames, dark_frame, pedestal):
    """
    Dispatcher function that calls either the 3D or 4D specialized Numba function
    depending on 'frames.ndim'.
    """
    if frames.ndim == 3:
        return subtract_dark_with_pedestal_3d(frames, dark_frame, pedestal)
    elif frames.ndim == 4:
        return subtract_dark_with_pedestal_4d(frames, dark_frame, pedestal)
    else:
        raise ValueError(
            f"subtract_dark_with_pedestal: frames must be 3D or 4D, got {frames.shape}"
        )


@njit(parallel=True, fastmath=True)
def parallel_measure_frames(images):
    """
    Parallel processing for measuring simple stats (mean only).
    'images' is a list (or array) of N images, each of which can be:
      - 2D (H,W) for a single mono image
      - 3D (H,W,C) for a single color image
      - Possibly 3D or 4D if you're storing multi-frame stacks in 'images'
    We just compute np.mean(...) of each image, no matter how many dims.
    """
    n = len(images)
    means = np.zeros(n, dtype=np.float32)

    for i in prange(n):
        arr = images[i]
        # arr could have shape (H,W) or (H,W,C) or (F,H,W) etc.
        # np.mean works for any dimension, so no special logic needed.
        means[i] = np.float32(np.mean(arr))

    return means


@njit(fastmath=True)
def fast_mad(image):
    """ Computes the Median Absolute Deviation (MAD) as a robust noise estimator. """
    flat_image = image.ravel()  # ✅ Flatten the 2D array into 1D
    median_val = np.median(flat_image)  # Compute median
    mad = np.median(np.abs(flat_image - median_val))  # Compute MAD
    return mad * 1.4826  # ✅ Scale MAD to match standard deviation (for Gaussian noise)



@njit(fastmath=True)
def compute_snr(image):
    """ Computes the Signal-to-Noise Ratio (SNR) using fast Numba std. """
    mean_signal = np.mean(image)
    noise = compute_noise(image)
    return mean_signal / noise if noise > 0 else 0




@njit(fastmath=True)
def compute_noise(image):
    """ Estimates noise using Median Absolute Deviation (MAD). """
    return fast_mad(image)




def compute_star_count(image):
    """ Uses fast star detection instead of DAOStarFinder. """
    return fast_star_count(image)


def fast_star_count(
    image, 
    blur_size=15,      # Smaller blur preserves faint/small stars
    threshold_factor=0.8, 
    min_area=2, 
    max_area=5000
):
    """
    Estimate star count + average eccentricity by:
      1) Convert to 8-bit grayscale
      2) Blur => subtract => enhance stars
      3) Otsu's threshold * threshold_factor => final threshold
      4) Contour detection + ellipse fit => eccentricity
    Returns (star_count, avg_ecc).
    """

    # 1) Convert to grayscale if needed
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2) Normalize to 8-bit
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image_8u = (255.0 * (image - img_min) / (img_max - img_min)).astype(np.uint8)
    else:
        return 0, 0.0  # All pixels identical => no stars

    # 3) Blur + subtract => enhance
    blurred = cv2.GaussianBlur(image_8u, (blur_size, blur_size), 0)
    subtracted = cv2.absdiff(image_8u, blurred)

    # 4) Otsu's threshold on 'subtracted'
    otsu_thresh_val, _ = cv2.threshold(subtracted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Scale it down if we want to detect more/fainter stars
    final_thresh_val = int(otsu_thresh_val * threshold_factor)
    if final_thresh_val < 2:
        final_thresh_val = 2  # avoid going below 2

    # 5) Apply threshold
    _, thresh = cv2.threshold(subtracted, final_thresh_val, 255, cv2.THRESH_BINARY)

    # 6) (Optional) Morphological opening to remove single-pixel noise
    #    Adjust kernel size if you get too many/few stars
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 7) Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 8) Filter contours by area, fit ellipse => compute eccentricity
    star_count = 0
    ecc_values = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        if len(c) < 5:
            continue  # Need >=5 points to fit an ellipse

        # Fit ellipse
        ellipse = cv2.fitEllipse(c)
        (cx, cy), (major_axis, minor_axis), angle = ellipse

        # major_axis >= minor_axis
        if minor_axis > major_axis:
            major_axis, minor_axis = minor_axis, major_axis

        if major_axis > 0:
            ecc = math.sqrt(1.0 - (minor_axis**2 / major_axis**2))
        else:
            ecc = 0.0

        ecc_values.append(ecc)
        star_count += 1

    if star_count > 0:
        avg_ecc = float(np.mean(ecc_values))
    else:
        avg_ecc = 0.0

    return star_count, avg_ecc

@njit(parallel=True, fastmath=True)
def normalize_images_3d(stack, ref_median):
    """
    Normalizes each frame in a 3D mono stack (F,H,W)
    so that its median equals ref_median.

    Returns a 3D result (F,H,W).
    """
    num_frames, height, width = stack.shape
    normalized_stack = np.zeros_like(stack, dtype=np.float32)

    for i in prange(num_frames):
        # shape of one frame: (H,W)
        img = stack[i]
        img_median = np.median(img)

        # Prevent division by zero
        scale_factor = ref_median / max(img_median, 1e-6)
        # Scale the entire 2D frame
        normalized_stack[i] = img * scale_factor

    return normalized_stack

@njit(parallel=True, fastmath=True)
def normalize_images_4d(stack, ref_median):
    """
    Normalizes each frame in a 4D color stack (F,H,W,C)
    so that its median equals ref_median.

    Returns a 4D result (F,H,W,C).
    """
    num_frames, height, width, channels = stack.shape
    normalized_stack = np.zeros_like(stack, dtype=np.float32)

    for i in prange(num_frames):
        # shape of one frame: (H,W,C)
        img = stack[i]  # (H,W,C)
        # Flatten to 1D to compute median across all channels/pixels
        img_median = np.median(img.ravel())

        # Prevent division by zero
        scale_factor = ref_median / max(img_median, 1e-6)

        # Scale the entire 3D frame
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    normalized_stack[i, y, x, c] = img[y, x, c] * scale_factor

    return normalized_stack

def normalize_images(stack, ref_median):
    """
    Dispatcher that calls either the 3D or 4D specialized Numba function
    depending on 'stack.ndim'.

    - If stack.ndim == 3, we assume shape (F,H,W).
    - If stack.ndim == 4, we assume shape (F,H,W,C).
    """
    if stack.ndim == 3:
        return normalize_images_3d(stack, ref_median)
    elif stack.ndim == 4:
        return normalize_images_4d(stack, ref_median)
    else:
        raise ValueError(f"normalize_images: stack must be 3D or 4D, got shape {stack.shape}")


@njit(parallel=True, fastmath=True)
def _edge_aware_interpolate_numba(out):
    """
    For each pixel in out (shape: (H,W,3)) where out[y,x,c] == 0,
    use a simple edge-aware approach:
      1) Compute horizontal gradient = abs( left - right )
      2) Compute vertical gradient   = abs( top - bottom )
      3) Choose the direction with the smaller gradient => average neighbors
      4) If neighbors are missing or zero, fallback to a small 3x3 average

    This is simpler than AHD but usually better than naive bilinear
    for high-contrast features like star cores.
    """
    H, W, C = out.shape

    for c in range(C):
        for y in prange(H):
            for x in range(W):
                if out[y, x, c] == 0:
                    # Gather immediate neighbors
                    left = 0.0
                    right = 0.0
                    top = 0.0
                    bottom = 0.0
                    have_left = False
                    have_right = False
                    have_top = False
                    have_bottom = False

                    # Left
                    if x - 1 >= 0:
                        val = out[y, x - 1, c]
                        if val != 0:
                            left = val
                            have_left = True

                    # Right
                    if x + 1 < W:
                        val = out[y, x + 1, c]
                        if val != 0:
                            right = val
                            have_right = True

                    # Top
                    if y - 1 >= 0:
                        val = out[y - 1, x, c]
                        if val != 0:
                            top = val
                            have_top = True

                    # Bottom
                    if y + 1 < H:
                        val = out[y + 1, x, c]
                        if val != 0:
                            bottom = val
                            have_bottom = True

                    # Compute gradients
                    # If we don't have valid neighbors for that direction,
                    # set the gradient to a large number => won't be chosen
                    gh = 1e6
                    gv = 1e6

                    if have_left and have_right:
                        gh = abs(left - right)
                    if have_top and have_bottom:
                        gv = abs(top - bottom)

                    # Decide which direction to interpolate
                    if gh < gv and have_left and have_right:
                        # Horizontal interpolation
                        out[y, x, c] = 0.5 * (left + right)
                    elif gv <= gh and have_top and have_bottom:
                        # Vertical interpolation
                        out[y, x, c] = 0.5 * (top + bottom)
                    else:
                        # Fallback: average 3×3 region
                        sumv = 0.0
                        count = 0
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                yy = y + dy
                                xx = x + dx
                                if 0 <= yy < H and 0 <= xx < W:
                                    val = out[yy, xx, c]
                                    if val != 0:
                                        sumv += val
                                        count += 1
                        if count > 0:
                            out[y, x, c] = sumv / count

    return out
# === Separate Full-Resolution Demosaicing Kernels ===
# These njit functions assume the raw image is arranged in a Bayer pattern
# and that we want a full (H,W,3) output.

@njit(parallel=True, fastmath=True)
def debayer_RGGB_fullres_fast(image):
    """
    For an RGGB pattern:
      - Even rows: even cols = Red, odd cols = Green.
      - Odd rows: even cols = Green, odd cols = Blue.
    """
    H, W = image.shape
    out = np.zeros((H, W, 3), dtype=image.dtype)
    for y in prange(H):
        for x in range(W):
            if (y & 1) == 0:
                if (x & 1) == 0:
                    # Even row, even col: Red
                    out[y, x, 0] = image[y, x]
                else:
                    # Even row, odd col: Green
                    out[y, x, 1] = image[y, x]
            else:
                if (x & 1) == 0:
                    # Odd row, even col: Green
                    out[y, x, 1] = image[y, x]
                else:
                    # Odd row, odd col: Blue
                    out[y, x, 2] = image[y, x]
    _edge_aware_interpolate_numba(out)
    return out

@njit(parallel=True, fastmath=True)
def debayer_BGGR_fullres_fast(image):
    """
    For a BGGR pattern:
      - Even rows: even cols = Blue, odd cols = Green.
      - Odd rows: even cols = Green, odd cols = Red.
    """
    H, W = image.shape
    out = np.zeros((H, W, 3), dtype=image.dtype)
    for y in prange(H):
        for x in range(W):
            if (y & 1) == 0:
                if (x & 1) == 0:
                    # Even row, even col: Blue
                    out[y, x, 2] = image[y, x]
                else:
                    # Even row, odd col: Green
                    out[y, x, 1] = image[y, x]
            else:
                if (x & 1) == 0:
                    # Odd row, even col: Green
                    out[y, x, 1] = image[y, x]
                else:
                    # Odd row, odd col: Red
                    out[y, x, 0] = image[y, x]
    _edge_aware_interpolate_numba(out)
    return out

@njit(parallel=True, fastmath=True)
def debayer_GRBG_fullres_fast(image):
    """
    For a GRBG pattern:
      - Even rows: even cols = Green, odd cols = Red.
      - Odd rows: even cols = Blue, odd cols = Green.
    """
    H, W = image.shape
    out = np.zeros((H, W, 3), dtype=image.dtype)
    for y in prange(H):
        for x in range(W):
            if (y & 1) == 0:
                if (x & 1) == 0:
                    # Even row, even col: Green
                    out[y, x, 1] = image[y, x]
                else:
                    # Even row, odd col: Red
                    out[y, x, 0] = image[y, x]
            else:
                if (x & 1) == 0:
                    # Odd row, even col: Blue
                    out[y, x, 2] = image[y, x]
                else:
                    # Odd row, odd col: Green
                    out[y, x, 1] = image[y, x]
    _edge_aware_interpolate_numba(out)
    return out

@njit(parallel=True, fastmath=True)
def debayer_GBRG_fullres_fast(image):
    """
    For a GBRG pattern:
      - Even rows: even cols = Green, odd cols = Blue.
      - Odd rows: even cols = Red, odd cols = Green.
    """
    H, W = image.shape
    out = np.zeros((H, W, 3), dtype=image.dtype)
    for y in prange(H):
        for x in range(W):
            if (y & 1) == 0:
                if (x & 1) == 0:
                    # Even row, even col: Green
                    out[y, x, 1] = image[y, x]
                else:
                    # Even row, odd col: Blue
                    out[y, x, 2] = image[y, x]
            else:
                if (x & 1) == 0:
                    # Odd row, even col: Red
                    out[y, x, 0] = image[y, x]
                else:
                    # Odd row, odd col: Green
                    out[y, x, 1] = image[y, x]
    _edge_aware_interpolate_numba(out)
    return out

# === Python-Level Dispatch Function ===
# Since Numba cannot easily compare strings in nopython mode,
# we do the if/elif check here in Python and then call the appropriate njit function.

def debayer_fits_fast(image_data, bayer_pattern):
    bp = bayer_pattern.upper()
    if bp == 'RGGB':
        return debayer_RGGB_fullres_fast(image_data)
    elif bp == 'BGGR':
        return debayer_BGGR_fullres_fast(image_data)
    elif bp == 'GRBG':
        return debayer_GRBG_fullres_fast(image_data)
    elif bp == 'GBRG':
        return debayer_GBRG_fullres_fast(image_data)
    else:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")

def debayer_raw_fast(raw_image_data, bayer_pattern="RGGB"):
    # For RAW images, use the same full-resolution demosaicing logic.
    return debayer_fits_fast(raw_image_data, bayer_pattern)


@njit(parallel=True, fastmath=True)
def applyPixelMath_numba(image_array, amount):
    factor = 3 ** amount
    denom_factor = 3 ** amount - 1
    height, width, channels = image_array.shape
    output = np.empty_like(image_array, dtype=np.float32)

    for y in prange(height):
        for x in prange(width):
            for c in prange(channels):
                val = (factor * image_array[y, x, c]) / (denom_factor * image_array[y, x, c] + 1)
                output[y, x, c] = min(max(val, 0.0), 1.0)  # Equivalent to np.clip()
    
    return output

@njit(parallel=True, fastmath=True)
def adjust_saturation_numba(image_array, saturation_factor):
    height, width, channels = image_array.shape
    output = np.empty_like(image_array, dtype=np.float32)

    for y in prange(int(height)):  # Ensure y is an integer
        for x in prange(int(width)):  # Ensure x is an integer
            r, g, b = image_array[int(y), int(x)]  # Force integer indexing

            # Convert RGB to HSV manually
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            delta = max_val - min_val

            # Compute Hue (H)
            if delta == 0:
                h = 0
            elif max_val == r:
                h = (60 * ((g - b) / delta) + 360) % 360
            elif max_val == g:
                h = (60 * ((b - r) / delta) + 120) % 360
            else:
                h = (60 * ((r - g) / delta) + 240) % 360

            # Compute Saturation (S)
            s = (delta / max_val) if max_val != 0 else 0
            s *= saturation_factor  # Apply saturation adjustment
            s = min(max(s, 0.0), 1.0)  # Clip saturation

            # Convert back to RGB
            if s == 0:
                r, g, b = max_val, max_val, max_val
            else:
                c = s * max_val
                x_val = c * (1 - abs((h / 60) % 2 - 1))
                m = max_val - c

                if 0 <= h < 60:
                    r, g, b = c, x_val, 0
                elif 60 <= h < 120:
                    r, g, b = x_val, c, 0
                elif 120 <= h < 180:
                    r, g, b = 0, c, x_val
                elif 180 <= h < 240:
                    r, g, b = 0, x_val, c
                elif 240 <= h < 300:
                    r, g, b = x_val, 0, c
                else:
                    r, g, b = c, 0, x_val

                r, g, b = r + m, g + m, b + m  # Add m to shift brightness

            # ✅ Fix: Explicitly cast indices to integers
            output[int(y), int(x), 0] = r
            output[int(y), int(x), 1] = g
            output[int(y), int(x), 2] = b

    return output




@njit(parallel=True, fastmath=True)
def applySCNR_numba(image_array):
    height, width, _ = image_array.shape
    output = np.empty_like(image_array, dtype=np.float32)

    for y in prange(int(height)):
        for x in prange(int(width)):
            r, g, b = image_array[y, x]
            g = min(g, (r + b) / 2)  # Reduce green to the average of red & blue
            
            # ✅ Fix: Assign channels individually instead of a tuple
            output[int(y), int(x), 0] = r
            output[int(y), int(x), 1] = g
            output[int(y), int(x), 2] = b


    return output

# D65 reference
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883

# Matrix for RGB -> XYZ (sRGB => D65)
_M_rgb2xyz = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float32)

# Matrix for XYZ -> RGB (sRGB => D65)
_M_xyz2rgb = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
], dtype=np.float32)



@njit(parallel=True, fastmath=True)
def apply_lut_gray(image_in, lut):
    """
    Numba-accelerated application of 'lut' to a single-channel image_in in [0..1].
    'lut' is a 1D array of shape (size,) also in [0..1].
    """
    out = np.empty_like(image_in)
    height, width = image_in.shape
    size_lut = len(lut) - 1

    for y in prange(height):
        for x in range(width):
            v = image_in[y, x]
            idx = int(v * size_lut + 0.5)
            if idx < 0: idx = 0
            elif idx > size_lut: idx = size_lut
            out[y, x] = lut[idx]

    return out

@njit(parallel=True, fastmath=True)
def apply_lut_color(image_in, lut):
    """
    Numba-accelerated application of 'lut' to a 3-channel image_in in [0..1].
    'lut' is a 1D array of shape (size,) also in [0..1].
    """
    out = np.empty_like(image_in)
    height, width, channels = image_in.shape
    size_lut = len(lut) - 1

    for y in prange(height):
        for x in range(width):
            for c in range(channels):
                v = image_in[y, x, c]
                idx = int(v * size_lut + 0.5)
                if idx < 0: idx = 0
                elif idx > size_lut: idx = size_lut
                out[y, x, c] = lut[idx]

    return out

@njit(parallel=True, fastmath=True)
def apply_lut_mono_inplace(array2d, lut):
    """
    In-place LUT application on a single-channel 2D array in [0..1].
    'lut' has shape (size,) also in [0..1].
    """
    H, W = array2d.shape
    size_lut = len(lut) - 1
    for y in prange(H):
        for x in prange(W):
            v = array2d[y, x]
            idx = int(v * size_lut + 0.5)
            if idx < 0:
                idx = 0
            elif idx > size_lut:
                idx = size_lut
            array2d[y, x] = lut[idx]

@njit(parallel=True, fastmath=True)
def apply_lut_color_inplace(array3d, lut):
    """
    In-place LUT application on a 3-channel array in [0..1].
    'lut' has shape (size,) also in [0..1].
    """
    H, W, C = array3d.shape
    size_lut = len(lut) - 1
    for y in prange(H):
        for x in prange(W):
            for c in range(C):
                v = array3d[y, x, c]
                idx = int(v * size_lut + 0.5)
                if idx < 0:
                    idx = 0
                elif idx > size_lut:
                    idx = size_lut
                array3d[y, x, c] = lut[idx]

@njit(parallel=True, fastmath=True)
def rgb_to_xyz_numba(rgb):
    """
    Convert an image from sRGB to XYZ (D65).
    rgb: float32 array in [0..1], shape (H,W,3)
    returns xyz in [0..maybe >1], shape (H,W,3)
    """
    H, W, _ = rgb.shape
    out = np.empty((H, W, 3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            r = rgb[y, x, 0]
            g = rgb[y, x, 1]
            b = rgb[y, x, 2]
            # Multiply by M_rgb2xyz
            X = _M_rgb2xyz[0,0]*r + _M_rgb2xyz[0,1]*g + _M_rgb2xyz[0,2]*b
            Y = _M_rgb2xyz[1,0]*r + _M_rgb2xyz[1,1]*g + _M_rgb2xyz[1,2]*b
            Z = _M_rgb2xyz[2,0]*r + _M_rgb2xyz[2,1]*g + _M_rgb2xyz[2,2]*b
            out[y, x, 0] = X
            out[y, x, 1] = Y
            out[y, x, 2] = Z
    return out

@njit(parallel=True, fastmath=True)
def xyz_to_rgb_numba(xyz):
    """
    Convert an image from XYZ (D65) to sRGB.
    xyz: float32 array, shape (H,W,3)
    returns rgb in [0..1], shape (H,W,3)
    """
    H, W, _ = xyz.shape
    out = np.empty((H, W, 3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            X = xyz[y, x, 0]
            Y = xyz[y, x, 1]
            Z = xyz[y, x, 2]
            # Multiply by M_xyz2rgb
            r = _M_xyz2rgb[0,0]*X + _M_xyz2rgb[0,1]*Y + _M_xyz2rgb[0,2]*Z
            g = _M_xyz2rgb[1,0]*X + _M_xyz2rgb[1,1]*Y + _M_xyz2rgb[1,2]*Z
            b = _M_xyz2rgb[2,0]*X + _M_xyz2rgb[2,1]*Y + _M_xyz2rgb[2,2]*Z
            # Clip to [0..1]
            if r < 0: r = 0
            elif r > 1: r = 1
            if g < 0: g = 0
            elif g > 1: g = 1
            if b < 0: b = 0
            elif b > 1: b = 1
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b
    return out

@njit
def f_lab_numba(t):
    delta = 6/29
    out = np.empty_like(t, dtype=np.float32)
    for i in range(t.size):
        val = t.flat[i]
        if val > delta**3:
            out.flat[i] = val**(1/3)
        else:
            out.flat[i] = val/(3*delta*delta) + (4/29)
    return out

@njit(parallel=True, fastmath=True)
def xyz_to_lab_numba(xyz):
    """
    xyz => shape(H,W,3), in D65. 
    returns lab in shape(H,W,3): L in [0..100], a,b in ~[-128..127].
    """
    H, W, _ = xyz.shape
    out = np.empty((H,W,3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            X = xyz[y, x, 0] / _Xn
            Y = xyz[y, x, 1] / _Yn
            Z = xyz[y, x, 2] / _Zn
            fx = (X)**(1/3) if X > (6/29)**3 else X/(3*(6/29)**2) + 4/29
            fy = (Y)**(1/3) if Y > (6/29)**3 else Y/(3*(6/29)**2) + 4/29
            fz = (Z)**(1/3) if Z > (6/29)**3 else Z/(3*(6/29)**2) + 4/29
            L = 116*fy - 16
            a = 500*(fx - fy)
            b = 200*(fy - fz)
            out[y, x, 0] = L
            out[y, x, 1] = a
            out[y, x, 2] = b
    return out

@njit(parallel=True, fastmath=True)
def lab_to_xyz_numba(lab):
    """
    lab => shape(H,W,3): L in [0..100], a,b in ~[-128..127].
    returns xyz shape(H,W,3).
    """
    H, W, _ = lab.shape
    out = np.empty((H,W,3), dtype=np.float32)
    delta = 6/29
    for y in prange(H):
        for x in prange(W):
            L = lab[y, x, 0]
            a = lab[y, x, 1]
            b = lab[y, x, 2]
            fy = (L+16)/116
            fx = fy + a/500
            fz = fy - b/200

            if fx > delta:
                xr = fx**3
            else:
                xr = 3*delta*delta*(fx - 4/29)
            if fy > delta:
                yr = fy**3
            else:
                yr = 3*delta*delta*(fy - 4/29)
            if fz > delta:
                zr = fz**3
            else:
                zr = 3*delta*delta*(fz - 4/29)

            X = _Xn * xr
            Y = _Yn * yr
            Z = _Zn * zr
            out[y, x, 0] = X
            out[y, x, 1] = Y
            out[y, x, 2] = Z
    return out

@njit(parallel=True, fastmath=True)
def rgb_to_hsv_numba(rgb):
    H, W, _ = rgb.shape
    out = np.empty((H,W,3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            r = rgb[y,x,0]
            g = rgb[y,x,1]
            b = rgb[y,x,2]
            cmax = max(r,g,b)
            cmin = min(r,g,b)
            delta = cmax - cmin
            # Hue
            h = 0.0
            if delta != 0.0:
                if cmax == r:
                    h = 60*(((g-b)/delta) % 6)
                elif cmax == g:
                    h = 60*(((b-r)/delta) + 2)
                else:
                    h = 60*(((r-g)/delta) + 4)
            # Saturation
            s = 0.0
            if cmax > 0.0:
                s = delta / cmax
            v = cmax
            out[y,x,0] = h
            out[y,x,1] = s
            out[y,x,2] = v
    return out

@njit(parallel=True, fastmath=True)
def hsv_to_rgb_numba(hsv):
    H, W, _ = hsv.shape
    out = np.empty((H,W,3), dtype=np.float32)
    for y in prange(H):
        for x in prange(W):
            h = hsv[y,x,0]
            s = hsv[y,x,1]
            v = hsv[y,x,2]
            c = v*s
            hh = (h/60.0) % 6
            x_ = c*(1 - abs(hh % 2 - 1))
            m = v - c
            r = 0.0
            g = 0.0
            b = 0.0
            if 0 <= hh < 1:
                r,g,b = c,x_,0
            elif 1 <= hh < 2:
                r,g,b = x_,c,0
            elif 2 <= hh < 3:
                r,g,b = 0,c,x_
            elif 3 <= hh < 4:
                r,g,b = 0,x_,c
            elif 4 <= hh < 5:
                r,g,b = x_,0,c
            else:
                r,g,b = c,0,x_
            out[y,x,0] = (r + m)
            out[y,x,1] = (g + m)
            out[y,x,2] = (b + m)
    return out

@njit(parallel=True, fastmath=True)
def _cosmetic_correction_numba_fixed(corrected, H, W, C, hot_sigma, cold_sigma):
    """
    Optimized Numba-compiled local outlier correction.
    - Computes median and standard deviation from 8 surrounding pixels (excluding center).
    - If the center pixel is greater than (median + hot_sigma * std_dev), it is replaced with the median.
    - If the center pixel is less than (median - cold_sigma * std_dev), it is replaced with the median.
    - Edge pixels are skipped (avoiding padding artifacts).
    """
    local_vals = np.empty(9, dtype=np.float32)  # Holds 8 surrounding pixels

    # Process pixels in parallel, skipping edges
    for y in prange(1, H - 1):  # Skip first and last rows
        for x in range(1, W - 1):  # Skip first and last columns
            # If the image is grayscale, set C=1 and handle accordingly
            for c_i in prange(C if corrected.ndim == 3 else 1):
                k = 0
                for dy in range(-1, 2):  # -1, 0, +1
                    for dx in range(-1, 2):  # -1, 0, +1
                        if corrected.ndim == 3:  # Color image
                            local_vals[k] = corrected[y + dy, x + dx, c_i]
                        else:  # Grayscale image
                            local_vals[k] = corrected[y + dy, x + dx]
                        k += 1

                # Compute median
                M = np.median(local_vals)

                # Compute MAD manually
                abs_devs = np.abs(local_vals - M)
                MAD = np.median(abs_devs)

                # Convert MAD to an approximation of standard deviation
                sigma_mad = 1.4826 * MAD  

                # Get center pixel
                if corrected.ndim == 3:
                    T = corrected[y, x, c_i]
                else:
                    T = corrected[y, x]

                threshold_high = M + (hot_sigma * sigma_mad)
                threshold_low = M - (cold_sigma * sigma_mad)

                # **Apply correction ONLY if center pixel is an outlier**
                if T > threshold_high or T < threshold_low:
                    if corrected.ndim == 3:
                        corrected[y, x, c_i] = M  # Replace center pixel in color image
                    else:
                        corrected[y, x] = M  # Replace center pixel in grayscale image

def bulk_cosmetic_correction_bayer(image, hot_sigma=5.0, cold_sigma=5.0):
    """
    Perform cosmetic correction on a single-channel Bayer mosaic.
    Assumes a default Bayer pattern "RGGB":
      - Red: even rows, even columns
      - Green1: even rows, odd columns
      - Green2: odd rows, even columns
      - Blue: odd rows, odd columns
    Applies cosmetic correction separately on each channel and reassembles them.
    """
    H, W = image.shape
    # Create a copy to hold the corrected image.
    corrected = image.astype(np.float32).copy()
    
    # For each channel, extract the subarray and apply the standard correction.
    # We use your existing bulk_cosmetic_correction_numba function, which accepts a 2D array.
    # Red channel (even rows, even cols)
    red = corrected[0:H:2, 0:W:2]
    red_corrected = bulk_cosmetic_correction_numba(red, hot_sigma, cold_sigma)
    corrected[0:H:2, 0:W:2] = red_corrected

    # Blue channel (odd rows, odd cols)
    blue = corrected[1:H:2, 1:W:2]
    blue_corrected = bulk_cosmetic_correction_numba(blue, hot_sigma, cold_sigma)
    corrected[1:H:2, 1:W:2] = blue_corrected

    # Green channel: two sets:
    # Green1 (even rows, odd cols)
    green1 = corrected[0:H:2, 1:W:2]
    green1_corrected = bulk_cosmetic_correction_numba(green1, hot_sigma, cold_sigma)
    corrected[0:H:2, 1:W:2] = green1_corrected

    # Green2 (odd rows, even cols)
    green2 = corrected[1:H:2, 0:W:2]
    green2_corrected = bulk_cosmetic_correction_numba(green2, hot_sigma, cold_sigma)
    corrected[1:H:2, 0:W:2] = green2_corrected

    return corrected

def bulk_cosmetic_correction_numba(image, hot_sigma=3.0, cold_sigma=3.0, window_size=3):
    """
    Optimized local outlier correction using Numba.
    - Identifies hot and cold outliers based on local neighborhood statistics.
    - Uses median and standard deviation from surrounding pixels to detect and replace outliers.
    - Applies separate hot_sigma and cold_sigma thresholds.
    - Skips edge pixels to avoid padding artifacts.
    """

    was_gray = False

    if image.ndim == 2:  # Convert grayscale to 3D
        H, W = image.shape
        C = 1
        was_gray = True
        image = image[:, :, np.newaxis]  # Explicitly add a color channel dimension

    else:
        H, W, C = image.shape

    # Copy the image for modification
    corrected = image.astype(np.float32).copy()

    # Apply fast correction (no padding, edges skipped)
    _cosmetic_correction_numba_fixed(corrected, H, W, C, hot_sigma, cold_sigma)

    if was_gray:
        corrected = corrected[:, :, 0]  # Convert back to 2D if originally grayscale

    return corrected

def evaluate_polynomial(H: int, W: int, coeffs: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluates the polynomial function over the entire image domain.
    """
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    A_full = build_poly_terms(xx.ravel(), yy.ravel(), degree)
    return (A_full @ coeffs).reshape(H, W)



@njit(parallel=True, fastmath=True)
def numba_mono_final_formula(rescaled, median_rescaled, target_median):
    """
    Applies the final formula *after* we already have the rescaled values.
    
    rescaled[y,x] = (original[y,x] - black_point) / (1 - black_point)
    median_rescaled = median(rescaled)
    
    out_val = ((median_rescaled - 1) * target_median * r) /
              ( median_rescaled*(target_median + r -1) - target_median*r )
    """
    H, W = rescaled.shape
    out = np.empty_like(rescaled)

    for y in prange(H):
        for x in range(W):
            r = rescaled[y, x]
            numer = (median_rescaled - 1.0) * target_median * r
            denom = median_rescaled * (target_median + r - 1.0) - target_median * r
            if np.abs(denom) < 1e-12:
                denom = 1e-12
            out[y, x] = numer / denom

    return out

@njit(parallel=True, fastmath=True)
def numba_color_final_formula_linked(rescaled, median_rescaled, target_median):
    """
    Linked color transform: we use one median_rescaled for all channels.
    rescaled: (H,W,3), already = (image - black_point)/(1 - black_point)
    median_rescaled = median of *all* pixels in rescaled
    """
    H, W, C = rescaled.shape
    out = np.empty_like(rescaled)

    for y in prange(H):
        for x in range(W):
            for c in range(C):
                r = rescaled[y, x, c]
                numer = (median_rescaled - 1.0) * target_median * r
                denom = median_rescaled * (target_median + r - 1.0) - target_median * r
                if np.abs(denom) < 1e-12:
                    denom = 1e-12
                out[y, x, c] = numer / denom

    return out

@njit(parallel=True, fastmath=True)
def numba_color_final_formula_unlinked(rescaled, medians_rescaled, target_median):
    """
    Unlinked color transform: a separate median_rescaled per channel.
    rescaled: (H,W,3), where each channel is already (val - black_point[c]) / (1 - black_point[c])
    medians_rescaled: shape (3,) with median of each channel in the rescaled array.
    """
    H, W, C = rescaled.shape
    out = np.empty_like(rescaled)

    for y in prange(H):
        for x in range(W):
            for c in range(C):
                r = rescaled[y, x, c]
                med = medians_rescaled[c]
                numer = (med - 1.0) * target_median * r
                denom = med * (target_median + r - 1.0) - target_median * r
                if np.abs(denom) < 1e-12:
                    denom = 1e-12
                out[y, x, c] = numer / denom

    return out


def build_poly_terms(x_array: np.ndarray, y_array: np.ndarray, degree: int) -> np.ndarray:
    """
    Precomputes polynomial basis terms efficiently using NumPy, supporting up to degree 6.
    """
    ones = np.ones_like(x_array, dtype=np.float32)

    if degree == 1:
        return np.column_stack((ones, x_array, y_array))

    elif degree == 2:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2))

    elif degree == 3:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2, 
                                x_array**3, x_array**2 * y_array, x_array * y_array**2, y_array**3))

    elif degree == 4:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2, 
                                x_array**3, x_array**2 * y_array, x_array * y_array**2, y_array**3,
                                x_array**4, x_array**3 * y_array, x_array**2 * y_array**2, x_array * y_array**3, y_array**4))

    elif degree == 5:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2, 
                                x_array**3, x_array**2 * y_array, x_array * y_array**2, y_array**3,
                                x_array**4, x_array**3 * y_array, x_array**2 * y_array**2, x_array * y_array**3, y_array**4,
                                x_array**5, x_array**4 * y_array, x_array**3 * y_array**2, x_array**2 * y_array**3, x_array * y_array**4, y_array**5))

    elif degree == 6:
        return np.column_stack((ones, x_array, y_array, 
                                x_array**2, x_array * y_array, y_array**2, 
                                x_array**3, x_array**2 * y_array, x_array * y_array**2, y_array**3,
                                x_array**4, x_array**3 * y_array, x_array**2 * y_array**2, x_array * y_array**3, y_array**4,
                                x_array**5, x_array**4 * y_array, x_array**3 * y_array**2, x_array**2 * y_array**3, x_array * y_array**4, y_array**5,
                                x_array**6, x_array**5 * y_array, x_array**4 * y_array**2, x_array**3 * y_array**3, x_array**2 * y_array**4, x_array * y_array**5, y_array**6))

    else:
        raise ValueError(f"Unsupported polynomial degree={degree}. Max supported is 6.")




def generate_sample_points(image: np.ndarray, num_points: int = 100) -> np.ndarray:
    """
    Generates sample points uniformly across the image.

    - Places points in a uniform grid (no randomization).
    - Avoids border pixels.
    - Skips any points with value 0.000 or above 0.85.

    Returns:
        np.ndarray: Array of shape (N, 2) containing (x, y) coordinates of sample points.
    """
    H, W = image.shape[:2]
    points = []

    # Create a uniform grid (avoiding the border)
    grid_size = int(np.sqrt(num_points))  # Roughly equal spacing
    x_vals = np.linspace(10, W - 10, grid_size, dtype=int)  # Avoids border
    y_vals = np.linspace(10, H - 10, grid_size, dtype=int)

    for y in y_vals:
        for x in x_vals:
            # Skip values that are too dark (0.000) or too bright (> 0.85)
            if np.any(image[int(y), int(x)] == 0.000) or np.any(image[int(y), int(x)] > 0.85):
                continue  # Skip this pixel

            points.append((int(x), int(y)))

            if len(points) >= num_points:
                return np.array(points, dtype=np.int32)  # Return only valid points

    return np.array(points, dtype=np.int32)  # Return all collected points

@njit(parallel=True, fastmath=True)
def numba_unstretch(image: np.ndarray, stretch_original_medians: np.ndarray, stretch_original_mins: np.ndarray) -> np.ndarray:
    """
    Numba-optimized function to undo the unlinked stretch.
    Restores each channel separately.
    """
    H, W, C = image.shape
    out = np.empty_like(image, dtype=np.float32)

    for c in prange(C):  # Parallelize per channel
        cmed_stretched = np.median(image[..., c])
        orig_med = stretch_original_medians[c]
        orig_min = stretch_original_mins[c]

        if cmed_stretched != 0 and orig_med != 0:
            for y in prange(H):
                for x in range(W):
                    r = image[y, x, c]
                    numerator = (cmed_stretched - 1) * orig_med * r
                    denominator = cmed_stretched * (orig_med + r - 1) - orig_med * r
                    if denominator == 0:
                        denominator = 1e-6  # Avoid division by zero
                    out[y, x, c] = numerator / denominator

            # Restore the original black point
            out[..., c] += orig_min

    return np.clip(out, 0, 1)  # Clip to valid range


@njit(fastmath=True)
def drizzle_deposit_numba_naive(
    img_data,       # shape (H, W), mono
    transform,      # shape (2, 3), e.g. [[a,b,tx],[c,d,ty]]
    drizzle_buffer, # shape (outH, outW)
    coverage_buffer,# shape (outH, outW)
    drizzle_factor: float,
    frame_weight: float
):
    """
    Naive deposit: each input pixel is mapped to exactly one output pixel,
    ignoring drop_shrink. 2D single-channel version (mono).
    """
    h, w = img_data.shape
    out_h, out_w = drizzle_buffer.shape

    # Build a 3×3 matrix M
    # transform is 2×3, so we expand to 3×3 for the standard [x, y, 1] approach
    M = np.zeros((3, 3), dtype=np.float32)
    M[0, 0] = transform[0, 0]  # a
    M[0, 1] = transform[0, 1]  # b
    M[0, 2] = transform[0, 2]  # tx
    M[1, 0] = transform[1, 0]  # c
    M[1, 1] = transform[1, 1]  # d
    M[1, 2] = transform[1, 2]  # ty
    M[2, 2] = 1.0

    # We'll reuse a small input vector for each pixel
    in_coords = np.zeros(3, dtype=np.float32)
    in_coords[2] = 1.0

    for y in range(h):
        for x in range(w):
            val = img_data[y, x]
            if val == 0:
                continue

            # Fill the input vector
            in_coords[0] = x
            in_coords[1] = y

            # Multiply
            out_coords = M @ in_coords
            X = out_coords[0]
            Y = out_coords[1]

            # Multiply by drizzle_factor
            Xo = int(X * drizzle_factor)
            Yo = int(Y * drizzle_factor)

            if 0 <= Xo < out_w and 0 <= Yo < out_h:
                drizzle_buffer[Yo, Xo] += val * frame_weight
                coverage_buffer[Yo, Xo] += frame_weight

    return drizzle_buffer, coverage_buffer


@njit(fastmath=True)
def drizzle_deposit_numba_footprint(
    img_data,       # shape (H, W), mono
    transform,      # shape (2, 3)
    drizzle_buffer, # shape (outH, outW)
    coverage_buffer,# shape (outH, outW)
    drizzle_factor: float,
    drop_shrink: float,
    frame_weight: float
):
    """
    Distributes each input pixel over a bounding box of width=drop_shrink
    in the drizzle (out) plane. (Mono 2D version)
    """
    h, w = img_data.shape
    out_h, out_w = drizzle_buffer.shape

    # Build a 3×3 matrix M
    M = np.zeros((3, 3), dtype=np.float32)
    M[0, 0] = transform[0, 0]  # a
    M[0, 1] = transform[0, 1]  # b
    M[0, 2] = transform[0, 2]  # tx
    M[1, 0] = transform[1, 0]  # c
    M[1, 1] = transform[1, 1]  # d
    M[1, 2] = transform[1, 2]  # ty
    M[2, 2] = 1.0

    in_coords = np.zeros(3, dtype=np.float32)
    in_coords[2] = 1.0

    footprint_radius = drop_shrink * 0.5

    for y in range(h):
        for x in range(w):
            val = img_data[y, x]
            if val == 0:
                continue

            # Transform to output coords
            in_coords[0] = x
            in_coords[1] = y
            out_coords = M @ in_coords
            X = out_coords[0]
            Y = out_coords[1]

            # Upsample
            Xo = X * drizzle_factor
            Yo = Y * drizzle_factor

            # bounding box
            min_x = int(np.floor(Xo - footprint_radius))
            max_x = int(np.floor(Xo + footprint_radius))
            min_y = int(np.floor(Yo - footprint_radius))
            max_y = int(np.floor(Yo + footprint_radius))

            # clip
            if max_x < 0 or min_x >= out_w or max_y < 0 or min_y >= out_h:
                continue
            if min_x < 0:
                min_x = 0
            if max_x >= out_w:
                max_x = out_w - 1
            if min_y < 0:
                min_y = 0
            if max_y >= out_h:
                max_y = out_h - 1

            width_foot = (max_x - min_x + 1)
            height_foot = (max_y - min_y + 1)
            area_pixels = width_foot * height_foot
            if area_pixels <= 0:
                continue

            deposit_val = (val * frame_weight) / area_pixels
            coverage_fraction = frame_weight / area_pixels

            for oy in range(min_y, max_y+1):
                for ox in range(min_x, max_x+1):
                    drizzle_buffer[oy, ox] += deposit_val
                    coverage_buffer[oy, ox] += coverage_fraction

    return drizzle_buffer, coverage_buffer


@njit(parallel=True)
def finalize_drizzle_2d(drizzle_buffer, coverage_buffer, final_out):
    """
    parallel-friendly final step: final_out = drizzle_buffer / coverage_buffer,
    with coverage < 1e-8 => 0
    """
    out_h, out_w = drizzle_buffer.shape
    for y in prange(out_h):
        for x in range(out_w):
            cov = coverage_buffer[y, x]
            if cov < 1e-8:
                final_out[y, x] = 0.0
            else:
                final_out[y, x] = drizzle_buffer[y, x] / cov
    return final_out

@njit(fastmath=True)
def drizzle_deposit_color_naive(
    img_data,         # shape (H,W,C)
    transform,        # shape (2,3)
    drizzle_buffer,   # shape (outH,outW,C)
    coverage_buffer,  # shape (outH,outW,C)
    drizzle_factor: float,
    drop_shrink: float,  # unused here
    frame_weight: float
):
    """
    Naive color deposit:
    Each input pixel is mapped to exactly one output pixel (ignores drop_shrink).
    """
    H, W, channels = img_data.shape
    outH, outW, outC = drizzle_buffer.shape

    # Build 3×3 matrix M
    M = np.zeros((3, 3), dtype=np.float32)
    M[0, 0] = transform[0, 0]
    M[0, 1] = transform[0, 1]
    M[0, 2] = transform[0, 2]
    M[1, 0] = transform[1, 0]
    M[1, 1] = transform[1, 1]
    M[1, 2] = transform[1, 2]
    M[2, 2] = 1.0

    in_coords = np.zeros(3, dtype=np.float32)
    in_coords[2] = 1.0

    for y in range(H):
        for x in range(W):
            # 1) Transform
            in_coords[0] = x
            in_coords[1] = y
            out_coords = M @ in_coords
            X = out_coords[0]
            Y = out_coords[1]

            # 2) Upsample
            Xo = int(X * drizzle_factor)
            Yo = int(Y * drizzle_factor)

            # 3) Check bounds
            if 0 <= Xo < outW and 0 <= Yo < outH:
                # 4) For each channel
                for cidx in range(channels):
                    val = img_data[y, x, cidx]
                    if val != 0:
                        drizzle_buffer[Yo, Xo, cidx] += val * frame_weight
                        coverage_buffer[Yo, Xo, cidx] += frame_weight

    return drizzle_buffer, coverage_buffer
@njit(fastmath=True)
def drizzle_deposit_color_footprint(
    img_data,         # shape (H,W,C)
    transform,        # shape (2,3)
    drizzle_buffer,   # shape (outH,outW,C)
    coverage_buffer,  # shape (outH,outW,C)
    drizzle_factor: float,
    drop_shrink: float,
    frame_weight: float
):
    """
    Color version with a bounding-box footprint of width=drop_shrink
    for distributing flux in the output plane.
    """
    H, W, channels = img_data.shape
    outH, outW, outC = drizzle_buffer.shape

    # Build 3×3 matrix
    M = np.zeros((3, 3), dtype=np.float32)
    M[0, 0] = transform[0, 0]
    M[0, 1] = transform[0, 1]
    M[0, 2] = transform[0, 2]
    M[1, 0] = transform[1, 0]
    M[1, 1] = transform[1, 1]
    M[1, 2] = transform[1, 2]
    M[2, 2] = 1.0

    in_coords = np.zeros(3, dtype=np.float32)
    in_coords[2] = 1.0

    footprint_radius = drop_shrink * 0.5

    for y in range(H):
        for x in range(W):
            # Transform once per pixel
            in_coords[0] = x
            in_coords[1] = y
            out_coords = M @ in_coords
            X = out_coords[0]
            Y = out_coords[1]

            # Upsample
            Xo = X * drizzle_factor
            Yo = Y * drizzle_factor

            # bounding box
            min_x = int(np.floor(Xo - footprint_radius))
            max_x = int(np.floor(Xo + footprint_radius))
            min_y = int(np.floor(Yo - footprint_radius))
            max_y = int(np.floor(Yo + footprint_radius))

            if max_x < 0 or min_x >= outW or max_y < 0 or min_y >= outH:
                continue
            if min_x < 0:
                min_x = 0
            if max_x >= outW:
                max_x = outW - 1
            if min_y < 0:
                min_y = 0
            if max_y >= outH:
                max_y = outH - 1

            width_foot = (max_x - min_x + 1)
            height_foot = (max_y - min_y + 1)
            area_pixels = width_foot * height_foot
            if area_pixels <= 0:
                continue

            for cidx in range(channels):
                val = img_data[y, x, cidx]
                if val == 0:
                    continue

                deposit_val = (val * frame_weight) / area_pixels
                coverage_fraction = frame_weight / area_pixels

                for oy in range(min_y, max_y + 1):
                    for ox in range(min_x, max_x + 1):
                        drizzle_buffer[oy, ox, cidx] += deposit_val
                        coverage_buffer[oy, ox, cidx] += coverage_fraction

    return drizzle_buffer, coverage_buffer


@njit
def finalize_drizzle_3d(drizzle_buffer, coverage_buffer, final_out):
    """
    final_out[y,x,c] = drizzle_buffer[y,x,c] / coverage_buffer[y,x,c]
    if coverage < 1e-8 => 0
    """
    outH, outW, channels = drizzle_buffer.shape
    for y in range(outH):
        for x in range(outW):
            for cidx in range(channels):
                cov = coverage_buffer[y, x, cidx]
                if cov < 1e-8:
                    final_out[y, x, cidx] = 0.0
                else:
                    final_out[y, x, cidx] = drizzle_buffer[y, x, cidx] / cov
    return final_out



@njit
def piecewise_linear(val, xvals, yvals):
    """
    Performs piecewise linear interpolation:
    Given a scalar 'val', and arrays xvals, yvals (each of length N),
    finds i s.t. xvals[i] <= val < xvals[i+1],
    then returns the linear interpolation between yvals[i], yvals[i+1].
    If val < xvals[0], returns yvals[0].
    If val > xvals[-1], returns yvals[-1].
    """
    if val <= xvals[0]:
        return yvals[0]
    for i in range(len(xvals)-1):
        if val < xvals[i+1]:
            # Perform a linear interpolation in interval [xvals[i], xvals[i+1]]
            dx = xvals[i+1] - xvals[i]
            dy = yvals[i+1] - yvals[i]
            ratio = (val - xvals[i]) / dx
            return yvals[i] + ratio * dy
    return yvals[-1]

@njit(parallel=True, fastmath=True)
def apply_curves_numba(image, xvals, yvals):
    """
    Numba-accelerated routine to apply piecewise linear interpolation 
    to each pixel in 'image'.
    - image can be (H,W) or (H,W,3).
    - xvals, yvals are the curve arrays in ascending order.
    Returns the adjusted image as float32.
    """
    if image.ndim == 2:
        H, W = image.shape
        out = np.empty((H, W), dtype=np.float32)
        for y in prange(H):
            for x in range(W):
                val = image[y, x]
                out[y, x] = piecewise_linear(val, xvals, yvals)
        return out
    elif image.ndim == 3:
        H, W, C = image.shape
        out = np.empty((H, W, C), dtype=np.float32)
        for y in prange(H):
            for x in range(W):
                for c in range(C):
                    val = image[y, x, c]
                    out[y, x, c] = piecewise_linear(val, xvals, yvals)
        return out
    else:
        # Unexpected shape
        return image  # Fallback

def fast_star_detect(image, 
                     blur_size=9, 
                     threshold_factor=0.7, 
                     min_area=1, 
                     max_area=5000):
    """
    Finds star positions via contour detection + ellipse fitting.
    Returns Nx2 array of (x, y) star coordinates in the same coordinate system as 'image'.
    """

    # 1) Convert to grayscale if needed
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2) Normalize to 8-bit [0..255]
    img_min, img_max = image.min(), image.max()
    if img_max <= img_min:
        return np.empty((0,2), dtype=np.float32)  # All pixels same => no stars
    image_8u = (255.0 * (image - img_min) / (img_max - img_min)).astype(np.uint8)

    # 3) Blur => subtract => highlight stars
    blurred = cv2.GaussianBlur(image_8u, (blur_size, blur_size), 0)
    subtracted = cv2.absdiff(image_8u, blurred)

    # 4) Otsu's threshold => scaled by threshold_factor
    otsu_thresh, _ = cv2.threshold(subtracted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    final_thresh_val = max(2, int(otsu_thresh * threshold_factor))

    _, thresh = cv2.threshold(subtracted, final_thresh_val, 255, cv2.THRESH_BINARY)

    # 5) (Optional) morphological opening to remove single-pixel noise
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 6) Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7) Filter by area, fit ellipse => use ellipse center as star position
    star_positions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        if len(c) < 5:
            # Need >=5 points to fit an ellipse
            continue

        ellipse = cv2.fitEllipse(c)
        (cx, cy), (major_axis, minor_axis), angle = ellipse
        # You could check eccentricity, etc. if you want to filter out weird shapes
        star_positions.append((cx, cy))

    if len(star_positions) == 0:
        return np.empty((0,2), dtype=np.float32)
    else:
        return np.array(star_positions, dtype=np.float32)