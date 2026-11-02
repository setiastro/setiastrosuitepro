"""Bounded, device-resident projective drizzle accumulation."""
from __future__ import annotations

import math

import numpy as np


class DrizzleAccumulator:
    """Keep numerator/coverage resident across frames; download only the result.

    Source pixels are processed in bounded blocks. Sparse CFA planes keep
    independent channel coverage, and clipped footprints are renormalized in
    the same way as the CPU projective deposit kernels.
    """

    _verified = set()

    @classmethod
    def create(cls, shape, dtype, scale, pixfrac, kernel, sigma, log):
        from .torch_rejection import _get_torch, _device, _is_directml_privateuse

        torch = _get_torch()
        dev = _device()
        if dev is None or dev.type == "cpu":
            return None
        if _is_directml_privateuse(dev):
            # torch-directml 0.2.5 can overwrite duplicate scatter_add indices
            # instead of summing them. This corrupts overlapping drizzle drops.
            log("DirectML lacks reliable scatter-add for overlapping drops; drizzle uses CPU.")
            return None
        if dev.type != "cuda" and np.dtype(dtype) == np.float64:
            log("64-bit drizzle accumulation uses CPU on this backend.")
            return None
        key = (str(dev), np.dtype(dtype).str, int(kernel))
        if key not in cls._verified:
            cls._verify(torch, dev, dtype, kernel)
            cls._verified.add(key)
        if dev.type == "cuda":
            free, _ = torch.cuda.mem_get_info(dev)
            required = int(np.prod(shape)) * np.dtype(dtype).itemsize * 2
            if required + 128 * 1024**2 > free * 0.8:
                log("Drizzle canvas exceeds GPU memory budget; using CPU.")
                return None
        return cls(torch, dev, shape, dtype, scale, pixfrac, kernel, sigma)

    @classmethod
    def _verify(cls, torch, dev, dtype, kernel):
        """Probe the actual backend before trusting a resident accumulator."""
        from .legacy.numba_utils import drizzle_deposit_color_kernel_proj

        image = np.arange(1, 37, dtype=np.float32).reshape(3, 4, 3) / 36
        image[::2, :, 1] = 0
        transform = np.array([[1, .02, -.2], [0, 1, .3], [.001, 0, 1]], np.float32)
        probe = cls(torch, dev, (6, 8, 3), dtype, 2., .9, kernel, .45)
        try:
            probe.add(image, transform, .7)
            num = np.zeros((6, 8, 3), dtype)
            cov = np.zeros_like(num)
            num, cov = drizzle_deposit_color_kernel_proj(
                image, transform, num, cov, 2., .9, .7, kernel, .45)
            np.testing.assert_allclose(probe.numerator.cpu().numpy(), num, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(probe.coverage.cpu().numpy(), cov, atol=2e-5, rtol=2e-5)
        finally:
            probe.close()

    def __init__(self, torch, dev, shape, dtype, scale, pixfrac, kernel, sigma):
        self.torch = torch
        self.dev = dev
        self.shape = tuple(shape)
        self.scale = float(scale)
        self.kernel = int(kernel)
        radius = 0.5 * float(pixfrac) * self.scale
        self.sigma = max(float(sigma) * self.scale if kernel == 2 else radius, 1e-6)
        self.rmax = max(0, int(math.ceil(3 * self.sigma if kernel == 2 else radius)))
        if (2 * self.rmax + 1) ** 2 > 4096:
            raise ValueError("Drizzle footprint exceeds GPU scratch budget")
        dt = torch.float64 if np.dtype(dtype) == np.float64 else torch.float32
        self.numerator = torch.zeros(shape, dtype=dt, device=dev)
        self.coverage = torch.zeros_like(self.numerator)

    def add(self, image, transform, weight):
        torch = self.torch
        image = np.asarray(image, dtype=np.float32)
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim == 3 else 1
        expected_c = self.shape[2] if len(self.shape) == 3 else 1
        if c != expected_c:
            raise ValueError("Drizzle source channels do not match the output canvas")
        # Match the CPU's float32 transform products, then do footprint math
        # in double on CUDA/CPU. MPS/DirectML only support float32 coordinates.
        coord_dtype = torch.float64 if self.dev.type in ("cuda", "cpu") else torch.float32
        m = torch.as_tensor(np.asarray(transform, np.float32), device=self.dev)
        flat = image.reshape(-1, c)
        side = 2 * self.rmax + 1
        # Limit footprint scratch independently of image dimensions/pixfrac.
        block = max(64, min(32768, 1_048_576 // (side * side)))
        offset = torch.arange(-self.rmax, self.rmax + 1, device=self.dev)
        oy, ox = torch.meshgrid(offset, offset, indexing="ij")
        ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
        out_h, out_w = self.shape[:2]
        numerator = self.numerator.reshape(-1, c)
        coverage = self.coverage.reshape(-1, c)
        with torch.no_grad():
            for start in range(0, h * w, block):
                stop = min(start + block, h * w)
                values = torch.as_tensor(np.ascontiguousarray(flat[start:stop]), device=self.dev)
                index = torch.arange(start, stop, device=self.dev)
                x = (index % w).float()
                y = (index // w).float()
                px = m[0, 0] * x + m[0, 1] * y + m[0, 2]
                py = m[1, 0] * x + m[1, 1] * y + m[1, 2]
                ww = m[2, 0] * x + m[2, 1] * y + m[2, 2]
                ww = torch.where(ww == 0, 1e-12, ww).to(coord_dtype)
                xx = px.to(coord_dtype) / ww * self.scale
                yy = py.to(coord_dtype) / ww * self.scale
                if self.rmax == 0:
                    tx, ty = xx.long()[:, None], yy.long()[:, None]
                    weights = torch.ones_like(tx, dtype=coord_dtype)
                else:
                    tx = xx.floor().long()[:, None] + ox
                    ty = yy.floor().long()[:, None] + oy
                    dx = tx + 0.5 - xx[:, None]
                    dy = ty + 0.5 - yy[:, None]
                    if self.kernel == 0:
                        overlap_x = (torch.minimum(tx + 1, xx[:, None] + self.sigma)
                                     - torch.maximum(tx, xx[:, None] - self.sigma)).clamp_min(0)
                        overlap_y = (torch.minimum(ty + 1, yy[:, None] + self.sigma)
                                     - torch.maximum(ty, yy[:, None] - self.sigma)).clamp_min(0)
                        weights = overlap_x * overlap_y
                    elif self.kernel == 1:
                        weights = ((dx * dx + dy * dy) <= self.sigma**2).to(coord_dtype)
                    else:
                        z = (dx * dx + dy * dy) / (2 * self.sigma**2)
                        weights = torch.where(z <= 9, (-z).exp(), 0.0)
                    # The CPU stores each kernel weight in float32 scratch,
                    # but sums its unrounded value for normalization.
                inside = (tx >= 0) & (tx < out_w) & (ty >= 0) & (ty < out_h)
                weights = torch.where(inside, weights, 0.0)
                total = weights.sum(dim=1, keepdim=True)
                normalized = weights.float().to(coord_dtype) / total.clamp_min(1e-12)
                target = (ty.clamp(0, out_h - 1) * out_w + tx.clamp(0, out_w - 1)).long()
                # Circle drops can miss all cell centers. CPU then deposits at
                # int(Xo),int(Yo), provided the footprint intersects the canvas.
                fallback = total[:, 0] <= 1e-12
                fx, fy = xx.long(), yy.long()
                intersects = ((xx + self.rmax).floor() >= 0) & ((xx - self.rmax).floor() < out_w)
                intersects &= ((yy + self.rmax).floor() >= 0) & ((yy - self.rmax).floor() < out_h)
                fallback &= intersects & (fx >= 0) & (fx < out_w) & (fy >= 0) & (fy < out_h)
                normalized[:, 0] = torch.where(fallback, 1.0, normalized[:, 0])
                target[:, 0] = torch.where(fallback, fy * out_w + fx, target[:, 0])
                for channel in range(c):
                    present = values[:, channel] != 0.0
                    cov = (normalized * float(weight) * present[:, None]).to(coverage.dtype)
                    val = torch.where(normalized > 0,
                        normalized * (values[:, channel].to(coord_dtype) * float(weight))[:, None],
                        0.0).to(numerator.dtype)
                    numerator[:, channel].scatter_add_(0, target.reshape(-1), val.reshape(-1))
                    coverage[:, channel].scatter_add_(0, target.reshape(-1), cov.reshape(-1))

    def finish(self):
        result = self.torch.where(self.coverage >= 1e-8,
                                  self.numerator / self.coverage.clamp_min(1e-8), 0.0)
        return result.float().cpu().numpy()

    def close(self):
        self.numerator = None
        self.coverage = None
