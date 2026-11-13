"""Bounded, device-resident projective drizzle accumulation."""
from __future__ import annotations

import math

import numpy as np


def _lb_clip(torch, p, q, t0, t1, ok):
    """One Liang-Barsky half-plane step: keep the part of the segment with
    p * t <= q."""
    zero = p == 0
    ok = ok & (~zero | (q >= 0))
    safe = torch.where(zero, torch.ones_like(p), p)
    r = q / safe
    t0 = torch.where((~zero) & (p < 0), torch.maximum(t0, r), t0)
    t1 = torch.where((~zero) & (p > 0), torch.minimum(t1, r), t1)
    return t0, t1, ok


def _seg_term(torch, ax, ay, dx, dy, t0, t1, ok):
    """Shoelace contribution of the surviving piece of one segment."""
    px = ax + t0 * dx
    py = ay + t0 * dy
    rx = ax + t1 * dx
    ry = ay + t1 * dy
    term = px * ry - rx * py
    return torch.where(ok & (t1 > t0), term, torch.zeros_like(term))


def _quad_cell_area(torch, qx, qy, x0, y0):
    """Exact area shared by a CCW convex quad and the unit cell at (x0, y0).

    qx/qy are lists of four tensors broadcastable against the cell grid.
    Green's theorem, same as _quad_cell_overlap in numba_utils: the boundary
    of the intersection is the quad's edges inside the cell plus the cell's
    edges inside the quad, and each piece is one straight segment.
    """
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    acc = torch.zeros_like(x0)

    for i in range(4):
        j = (i + 1) & 3
        ax = qx[i]
        ay = qy[i]
        dx = qx[j] - ax
        dy = qy[j] - ay
        t0 = torch.zeros_like(x0)
        t1 = torch.ones_like(x0)
        ok = torch.ones_like(x0, dtype=torch.bool)
        t0, t1, ok = _lb_clip(torch, -dx, ax - x0, t0, t1, ok)
        t0, t1, ok = _lb_clip(torch, dx, x1 - ax, t0, t1, ok)
        t0, t1, ok = _lb_clip(torch, -dy, ay - y0, t0, t1, ok)
        t0, t1, ok = _lb_clip(torch, dy, y1 - ay, t0, t1, ok)
        acc = acc + _seg_term(torch, ax, ay, dx, dy, t0, t1, ok)

    for ax, ay, bx, by in ((x0, y0, x1, y0), (x1, y0, x1, y1),
                           (x1, y1, x0, y1), (x0, y1, x0, y0)):
        dx = bx - ax
        dy = by - ay
        t0 = torch.zeros_like(x0)
        t1 = torch.ones_like(x0)
        ok = torch.ones_like(x0, dtype=torch.bool)
        for i in range(4):
            j = (i + 1) & 3
            ex = qx[j] - qx[i]
            ey = qy[j] - qy[i]
            fa = ex * (ay - qy[i]) - ey * (ax - qx[i])
            fd = ex * dy - ey * dx
            t0, t1, ok = _lb_clip(torch, -fd, fa, t0, t1, ok)
        acc = acc + _seg_term(torch, ax, ay, dx, dy, t0, t1, ok)

    return torch.clamp(0.5 * acc, min=0.0)



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
        self.pixfrac = float(pixfrac)
        self.radius = radius
        self.sigma = max(float(sigma) * self.scale if kernel == 2 else radius, 1e-6)
        if kernel == 2:
            self.rmax = max(0, int(math.ceil(3 * self.sigma)))
        elif kernel == 3:
            # worst case for a pure rotation; add() recomputes it per frame
            # from the transform's own linear part
            self.rmax = int(math.ceil(radius * 1.4143)) + 1
        else:
            self.rmax = max(0, int(math.ceil(radius)))
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
        tm = np.asarray(transform, np.float32)
        m = torch.as_tensor(tm, device=self.dev)
        flat = image.reshape(-1, c)
        rmax = self.rmax
        if self.kernel == 3:
            # the exact drop turns with the frame, so size the footprint from
            # the transform, exactly as the numba kernel does
            _ex = abs(float(tm[0, 0])) + abs(float(tm[0, 1]))
            _ey = abs(float(tm[1, 0])) + abs(float(tm[1, 1]))
            rmax = int(math.ceil(self.radius * (_ex if _ex > _ey else _ey))) + 1
            if (2 * rmax + 1) ** 2 > 4096:
                raise ValueError("Drizzle footprint exceeds GPU scratch budget")
        side = 2 * rmax + 1
        # Limit footprint scratch independently of image dimensions/pixfrac.
        budget = 262_144 if self.kernel == 3 else 1_048_576
        block = max(64, min(32768, budget // (side * side)))
        offset = torch.arange(-rmax, rmax + 1, device=self.dev)
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
                if rmax == 0:
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
                    elif self.kernel == 3:
                        # exact drop: the quad image of the shrunk input pixel
                        hh = 0.5 * self.pixfrac
                        qxs = []
                        qys = []
                        for ddx, ddy in ((-hh, -hh), (hh, -hh),
                                         (hh, hh), (-hh, hh)):
                            ux = x + ddx
                            uy = y + ddy
                            cpx = m[0, 0] * ux + m[0, 1] * uy + m[0, 2]
                            cpy = m[1, 0] * ux + m[1, 1] * uy + m[1, 2]
                            cww = m[2, 0] * ux + m[2, 1] * uy + m[2, 2]
                            cww = torch.where(cww == 0, 1e-12, cww).to(coord_dtype)
                            qxs.append(
                                (cpx.to(coord_dtype) / cww * self.scale)[:, None])
                            qys.append(
                                (cpy.to(coord_dtype) / cww * self.scale)[:, None])
                        sgn = torch.zeros_like(qxs[0])
                        for i4 in range(4):
                            j4 = (i4 + 1) & 3
                            sgn = sgn + qxs[i4] * qys[j4] - qxs[j4] * qys[i4]
                        flip = sgn < 0
                        n1x = torch.where(flip, qxs[3], qxs[1])
                        n1y = torch.where(flip, qys[3], qys[1])
                        n3x = torch.where(flip, qxs[1], qxs[3])
                        n3y = torch.where(flip, qys[1], qys[3])
                        qxs[1] = n1x
                        qys[1] = n1y
                        qxs[3] = n3x
                        qys[3] = n3y
                        weights = _quad_cell_area(torch, qxs, qys,
                                                  tx.to(coord_dtype),
                                                  ty.to(coord_dtype))
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
                intersects = ((xx + rmax).floor() >= 0) & ((xx - rmax).floor() < out_w)
                intersects &= ((yy + rmax).floor() >= 0) & ((yy - rmax).floor() < out_h)
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
