"""Bounded, device-resident projective drizzle accumulation."""
from __future__ import annotations

import math

import numpy as np


def _exact_drop_window(tm, h, w, half, scale):
    """Rows and columns of output cells one exact drop can touch.

    The window is anchored per pixel at floor(min corner) -- the CPU kernel's
    own per-pixel bbox -- so it only has to be as wide as the drop: a quad of
    width W spans at most ceil(W) + 1 cells. That is tighter than a window
    centred on the drop (2 * ceil(W / 2) + 1), e.g. 4x4 instead of 5x5 for a
    rotated drop at 2x / drop shrink 1.0.

    Widths come from the real transform at a grid of sample pixels, because a
    homography stretches drops across the field. The slack covers the float32
    corner arithmetic the CPU uses and _exact_drop_weights reproduces.
    """
    M = np.asarray(tm, np.float64)
    gx, gy = np.meshgrid(np.linspace(0.0, max(w - 1, 0), 9),
                         np.linspace(0.0, max(h - 1, 0), 9))
    gx = gx.reshape(-1, 1) + np.array([-half, half, half, -half])
    gy = gy.reshape(-1, 1) + np.array([-half, -half, half, half])
    nx = M[0, 0] * gx + M[0, 1] * gy + M[0, 2]
    ny = M[1, 0] * gx + M[1, 1] * gy + M[1, 2]
    nw = M[2, 0] * gx + M[2, 1] * gy + M[2, 2]
    nw = np.where(nw == 0, 1e-12, nw)
    qx = nx / nw * scale
    qy = ny / nw * scale
    width_x = float((qx.max(1) - qx.min(1)).max())
    width_y = float((qy.max(1) - qy.min(1)).max())
    mag = max(float(np.abs(nx).max()), float(np.abs(ny).max()), float(max(h, w)))
    slack = 32.0 * 2.0**-24 * scale * mag / max(float(np.abs(nw).min()), 1e-12)
    rows = int(math.ceil(width_y * (1 + 1e-6) + slack)) + 1
    cols = int(math.ceil(width_x * (1 + 1e-6) + slack)) + 1
    return rows, cols


def _exact_drop_corners(torch, x, y, k):
    """Corners of each pixel's shrunk footprint in output coords, (B, 4).

    Built exactly as _drop_quad_proj does: np.float32(x + dx), float32
    projective numerators, then the divide in coordinate precision. Adjacent
    drops therefore share bit-identical corners and tile without gaps.
    """
    cd = k["dtype"]
    if cd == torch.float64:
        ux = (x.to(cd)[:, None] + k["cdx"]).float()
        uy = (y.to(cd)[:, None] + k["cdy"]).float()
    else:
        ux = x[:, None] + k["cdx"]
        uy = y[:, None] + k["cdy"]
    m = k["m"]
    cpx = ux * m[0][0] + uy * m[0][1] + m[0][2]
    cpy = ux * m[1][0] + uy * m[1][1] + m[1][2]
    cww = ux * m[2][0] + uy * m[2][1] + m[2][2]
    cww = torch.where(cww == 0, 1e-12, cww).to(cd)
    return cpx.to(cd) / cww * k["scale"], cpy.to(cd) / cww * k["scale"]


def _exact_drop_weights(torch, qx, qy, k):
    """Area each quad (B, 4) shares with every cell of its window.

    The integral is the one _quad_cell_overlap evaluates on the CPU: the
    contour integral of clamp(x, x0, x1) dy over the parts of the quad's
    edges inside the cell's row. Clamping is continuous, so an edge lying
    along a cell line contributes zero instead of being counted twice.

    It is arranged for a GPU that is limited by kernel launches rather than
    arithmetic (~35 us per launch on Windows, whatever the tensor size):

    * the four edges ride one tensor axis, so each step is one launch, not
      four;
    * the row clip depends only on the row and the column crossings only on
      the column, so both are computed on (B, 4, n) and combined once on
      (B, 4, rows, cols);
    * coordinates are local to the window, so the cell lines are the
      integers 0..n. The x0 * (t1 - t0) part of each term sums to zero
      around a closed contour, so it is left out rather than added at the
      magnitude of the canvas coordinate and cancelled.

    Returns weights (B, rows*cols), tx/ty (B, rows*cols) int64, and a (B,)
    mask of drops the CPU would deposit somewhere: non-degenerate, with a
    bbox that reaches the canvas.
    """
    bx = qx.amin(1, keepdim=True).floor()
    by = qy.amin(1, keepdim=True).floor()
    lx = qx - bx
    ly = qy - by
    nx = lx.roll(-1, 1)
    ny = ly.roll(-1, 1)
    ex = nx - lx
    ey = ny - ly
    # twice the signed area; a clockwise quad integrates to minus its area
    turn = (lx * ny - nx * ly).sum(1)

    # Walk every edge with x increasing. The integral of clamp(x) along an
    # edge does not depend on direction; ey keeps the edge's own sign.
    back = ex < 0
    sx = torch.where(back, nx, lx)
    sy = torch.where(back, ny, ly)
    adx = ex.abs()
    sdy = torch.where(back, -ey, ey)
    big = k["big"]
    # a vertical edge takes the dx -> 0+ limit; a horizontal one is
    # multiplied by ey == 0 below, so its finite stand-in never matters
    inv_x = torch.where(adx > 1.0 / big, adx.reciprocal(), big)
    inv_y = torch.where(sdy.abs() > 1.0 / big, sdy.reciprocal(), big)

    g = (k["ry"] - sy[..., None]) * inv_y[..., None]       # (B, 4, rows + 1)
    t0 = torch.minimum(g[..., :-1], g[..., 1:]).clamp_(min=0.0)
    t1 = torch.maximum(g[..., :-1], g[..., 1:]).clamp_(max=1.0)
    t1 = torch.maximum(t1, t0)[..., None]                   # empty if it misses the row
    t0 = t0[..., None]
    s = (k["cx"] - sx[..., None]) * inv_x[..., None]        # (B, 4, cols + 1)
    xi = (sx[..., None] - k["cx"][..., :-1])[..., None, :]  # start, from each column's left line

    lo = torch.clamp(s[..., None, :-1], t0, t1)             # (B, 4, rows, cols)
    hi = torch.clamp(s[..., None, 1:], t0, t1)
    # clamp(x, x0, x1) - x0 along the edge: 0 until lo, linear to hi, 1 after
    part = torch.addcmul(xi, lo + hi, (0.5 * adx)[..., None, None])
    part = torch.addcmul(t1 - hi, hi - lo, part)
    area = (part * ey[..., None, None]).sum(1)
    weights = (area.reshape(area.shape[0], -1) * turn.sign()[:, None]).clamp_(min=0.0)

    tx = bx.long() + k["col"]
    ty = by.long() + k["row"]
    reach = ((qx.amax(1) >= 0) & (bx[:, 0] < k["out_w"])
             & (qy.amax(1) >= 0) & (by[:, 0] < k["out_h"]) & (turn != 0))
    return weights, tx, ty, reach


def _scratch_cells(torch, dev):
    """Footprint cells per block that fit the device comfortably.

    Bigger blocks mean fewer launches, which is what costs time here. On
    CUDA the budget follows free memory (after the canvas is allocated);
    elsewhere it stays at a fixed, modest size.
    """
    if dev.type == "cuda":
        try:
            free, _ = torch.cuda.mem_get_info(dev)
            # blocks this process's allocator has cached but not handed out
            free += torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        except Exception:
            free = 0
        scratch = min(1024**3, max(64 * 1024**2, int(free * 0.25)))
        # measured peak 80-125 bytes per footprint cell across kernels 0-3
        return scratch // 128
    return 2_097_152


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
            # unused by add(), which sizes the window from each frame's
            # transform; kept as the worst case of a pure 45 deg rotation
            self.rmax = int(math.ceil(radius * 1.4143))
        else:
            self.rmax = max(0, int(math.ceil(radius)))
        if (2 * self.rmax + 1) ** 2 > 4096:
            raise ValueError("Drizzle footprint exceeds GPU scratch budget")
        dt = torch.float64 if np.dtype(dtype) == np.float64 else torch.float32
        self.numerator = torch.zeros(shape, dtype=dt, device=dev)
        self.coverage = torch.zeros_like(self.numerator)
        self.cell_budget = _scratch_cells(torch, dev)

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
        out_h, out_w = self.shape[:2]
        rmax = self.rmax
        if self.kernel == 3:
            # the exact drop turns and stretches with the frame, so its
            # window comes from this frame's transform
            half = max(0.5 * self.pixfrac, 1e-4)  # as the CPU: never a point
            rows, cols = _exact_drop_window(tm, h, w, half, self.scale)
            if rows * cols > 4096:
                raise ValueError("Drizzle footprint exceeds GPU scratch budget")
            cdt = torch.float64 if coord_dtype == torch.float64 else torch.float32
            k3 = {
                "dtype": coord_dtype,
                "m": [[float(v) for v in row] for row in tm],
                "scale": self.scale,
                "cdx": torch.tensor([[-half, half, half, -half]], dtype=cdt, device=self.dev),
                "cdy": torch.tensor([[-half, -half, half, half]], dtype=cdt, device=self.dev),
                "ry": torch.arange(rows + 1, dtype=coord_dtype, device=self.dev).view(1, 1, -1),
                "cx": torch.arange(cols + 1, dtype=coord_dtype, device=self.dev).view(1, 1, -1),
                "row": torch.arange(rows, device=self.dev).repeat_interleave(cols).view(1, -1),
                "col": torch.arange(cols, device=self.dev).repeat(rows).view(1, -1),
                "big": 1e300 if coord_dtype == torch.float64 else 1e30,
                "out_w": out_w,
                "out_h": out_h,
            }
            # the per-edge grid (B, 4, rows, cols) is the widest scratch
            cells = 4 * rows * cols
        else:
            side = 2 * rmax + 1
            cells = side * side
            offset = torch.arange(-rmax, rmax + 1, device=self.dev)
            oy, ox = torch.meshgrid(offset, offset, indexing="ij")
            ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
        # Launch count, not bandwidth, is what limits this stage, so blocks
        # are as large as the scratch budget allows.
        block = max(64, min(1 << 20, self.cell_budget // cells))
        numerator = self.numerator.reshape(-1, c)
        coverage = self.coverage.reshape(-1, c)
        with torch.no_grad():
            # One upload per frame. A per-block upload from pageable memory
            # blocks the host until the GPU drains, so launching the next
            # block could never overlap executing this one.
            flat = torch.as_tensor(np.ascontiguousarray(image.reshape(-1, c)),
                                   device=self.dev)
            for start in range(0, h * w, block):
                stop = min(start + block, h * w)
                values = flat[start:stop]
                index = torch.arange(start, stop, device=self.dev)
                x = (index % w).float()
                y = (index // w).float()
                px = m[0, 0] * x + m[0, 1] * y + m[0, 2]
                py = m[1, 0] * x + m[1, 1] * y + m[1, 2]
                ww = m[2, 0] * x + m[2, 1] * y + m[2, 2]
                ww = torch.where(ww == 0, 1e-12, ww).to(coord_dtype)
                xx = px.to(coord_dtype) / ww * self.scale
                yy = py.to(coord_dtype) / ww * self.scale
                if self.kernel == 3:
                    # exact drop: the quad image of the shrunk input pixel
                    qx, qy = _exact_drop_corners(torch, x, y, k3)
                    weights, tx, ty, intersects = _exact_drop_weights(torch, qx, qy, k3)
                elif rmax == 0:
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
                if self.kernel != 3:
                    # (the exact drop reports its own bbox test, like the CPU's)
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
