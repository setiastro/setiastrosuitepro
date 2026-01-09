# src/setiastro/saspro/imageops/narrowband_normalization.py
from __future__ import annotations
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
import traceback

ProgressCB = Optional[Callable[[int, str], None]]

# ---------------- params ----------------

@dataclass(frozen=True, slots=True)
class NBNParams:
    scenario: str            # "HOO"/"SHO"/"HSO"/"HOS"
    mode: int                # 0 linear, 1 non-linear
    lightness: int           # HOO: 0..3, others: 0..4
    blackpoint: float        # 0..1
    hlrecover: float         # >= 0.25
    hlreduct: float          # >= 0.25
    brightness: float        # >= 0.25
    blendmode: int = 0       # HOO only: 0/1/2
    hablend: float = 0.6     # HOO only: 0..1
    oiiiboost: float = 1.0   # HOO OIII boost
    siiboost: float = 1.0    # SHO/HSO/HOS
    oiiiboost2: float = 1.0  # SHO/HSO/HOS
    scnr: bool = False       # SHO/HSO/HOS


class MissingChannelsError(ValueError):
    pass


__all__ = ["NBNParams", "MissingChannelsError", "normalize_narrowband"]


# ---------------- PixelMath primitives ----------------

_EPS = 1e-12


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _inv01(x: np.ndarray) -> np.ndarray:
    """PixelMath '~' complement for normalized images."""
    return 1.0 - x


def _rescale(x: np.ndarray, lo: float | np.ndarray, hi: float | np.ndarray) -> np.ndarray:
    """Map x from [lo, hi] -> [0, 1] (clipped)."""
    loa = np.asarray(lo, dtype=np.float32)
    hia = np.asarray(hi, dtype=np.float32)
    denom = np.maximum(hia - loa, _EPS)
    return _clip01((x - loa) / denom)


def _mtf(m: float | np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    PixInsight Midtones Transfer Function.
    For m in (0,1): m is the midtone (pivot) value.
    """
    m = np.asarray(m, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    m = np.clip(m, _EPS, 1.0 - _EPS)
    x = _clip01(x)

    # y = (m - 1) * x / ((2*m - 1)*x - m)
    num = (m - 1.0) * x
    den = (2.0 * m - 1.0) * x - m

    # IMPORTANT: never allow 0 denominator (np.sign(0) == 0). Use ±EPS.
    safe_den = np.where(
        np.abs(den) < _EPS,
        np.where(den >= 0.0, _EPS, -_EPS).astype(np.float32),
        den,
    )
    return _clip01(num / safe_den)


def _adev(x: np.ndarray) -> float:
    """Approx absolute deviation. PixelMath adev()."""
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def _stats_min_med_mean(chs: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match PI per-channel stats behavior: compute min/median/mean for each channel.
    Returns float32 vectors of shape (C,).
    """
    c = len(chs)
    mins = np.empty((c,), dtype=np.float32)
    meds = np.empty((c,), dtype=np.float32)
    means = np.empty((c,), dtype=np.float32)
    for i, ch in enumerate(chs):
        mins[i] = float(np.nanmin(ch))
        meds[i] = float(np.nanmedian(ch))
        means[i] = float(np.nanmean(ch))
    return mins, meds, means


def _stats_adev_vec(chs: Tuple[np.ndarray, ...]) -> np.ndarray:
    v = np.empty((len(chs),), dtype=np.float32)
    for i, ch in enumerate(chs):
        v[i] = float(_adev(ch))
    return v

def _default_workers() -> int:
    # Don’t go crazy; too many workers can reduce perf due to memory bandwidth.
    n = os.cpu_count() or 4
    return max(1, min(32, n))


def _run_tiles_parallel(
    tiles: list[tuple[int, int, int, int]],
    worker_fn,  # callable(y0,y1,x0,x1,ti)
    *,
    progress_cb: ProgressCB,
    p0: int,
    p1: int,
    label: str,
    max_workers: Optional[int] = None,
) -> None:
    """
    Run per-tile worker_fn in parallel. worker_fn must write into shared output using non-overlapping slices.
    """
    ntiles = len(tiles)
    if ntiles == 0:
        return

    workers = int(max_workers or _default_workers())
    workers = max(1, min(workers, ntiles))

    # Progress bookkeeping
    done = 0
    lock = threading.Lock()
    last_emit = {"p": -1}

    def _on_done():
        nonlocal done
        with lock:
            done += 1
            # Throttle: emit only when percent changes by >=1
            if progress_cb:
                p = _map_progress(done, ntiles, p0, p1)
                if p != last_emit["p"]:
                    last_emit["p"] = p
                    progress_cb(p, f"{label} {done}/{ntiles}")

    if progress_cb:
        progress_cb(p0, f"{label} 0/{ntiles} (workers={workers})")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for ti, (y0, y1, x0, x1) in enumerate(tiles):
            futs.append(ex.submit(worker_fn, y0, y1, x0, x1, ti))

        # Drain futures; propagate exceptions
        for f in as_completed(futs):
            f.result()
            _on_done()

    if progress_cb:
        progress_cb(p1, f"{label} {ntiles}/{ntiles}")


def _iter_tiles(h: int, w: int, tile: int = 1024):
    """Yield (y0,y1,x0,x1) tiles covering an HxW image."""
    for y0 in range(0, h, tile):
        y1 = min(y0 + tile, h)
        for x0 in range(0, w, tile):
            x1 = min(x0 + tile, w)
            yield y0, y1, x0, x1


def _map_progress(i: int, n: int, p0: int, p1: int) -> int:
    """Map tile index i in [0..n] to integer percent in [p0..p1]."""
    if n <= 0:
        return int(p1)
    return int(p0 + (p1 - p0) * (i / n))

def _finish_tiled(out: np.ndarray, params: NBNParams, progress_cb: ProgressCB) -> np.ndarray:
    h, w, _ = out.shape
    tiles = list(_iter_tiles(h, w, tile=1024))

    if progress_cb:
        progress_cb(80, f"Finishing tiles 0/{len(tiles)}")

    def _worker(y0, y1, x0, x1, ti):
        out[y0:y1, x0:x1, :] = _apply_hl_reduction_and_brightness_and_recover(
            out[y0:y1, x0:x1, :], params
        )

    _run_tiles_parallel(tiles, _worker, progress_cb=progress_cb, p0=80, p1=98, label="Finishing tiles")
    return out

# ---------------- Color space helpers (as in script) ----------------

def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    return np.where(u > 0.04045, ((u + 0.055) / 1.055) ** 2.4, u / 12.92)


def _linear_to_srgb(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    # Gamma encoding is undefined for negatives; clamp them.
    u = np.clip(u, 0.0, 1.0)
    u = np.where(np.isfinite(u), u, 0.0)
    u = np.maximum(u, 0.0)
    return np.where(u > 0.0031308, 1.055 * (u ** (1.0 / 2.4)) - 0.055, 12.92 * u)


def _rgb_to_xyz_pi(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Matches coefficients in the PixelMath script (PI's D65-ish matrix used there)
    r1 = _srgb_to_linear(_clip01(r))
    g1 = _srgb_to_linear(_clip01(g))
    b1 = _srgb_to_linear(_clip01(b))

    X = (r1 * 0.4360747) + (g1 * 0.3850649) + (b1 * 0.1430804)
    Y = (r1 * 0.2225045) + (g1 * 0.7168786) + (b1 * 0.0606169)
    Z = (r1 * 0.0139322) + (g1 * 0.0971045) + (b1 * 0.7141733)
    return X, Y, Z


def _xyz_to_lab_pi(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # PixelMath uses the 0.008856 threshold and the affine segment
    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, t ** (1.0 / 3.0), (7.787 * t) + (16.0 / 116.0))

    X1 = f(X)
    Y1 = f(Y)
    Z1 = f(Z)

    L = 116.0 * Y1 - 16.0
    a = 500.0 * (X1 - Y1)
    b = 200.0 * (Y1 - Z1)
    return L, a, b


def _xyz_to_rgb_pi(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Inverse matrix from script
    R2 = (X * 3.1338561) + (Y * -1.6168667) + (Z * -0.4906146)
    G2 = (X * -0.9787684) + (Y * 1.9161415) + (Z * 0.0334540)
    B2 = (X * 0.0719453) + (Y * -0.2289914) + (Z * 1.4052427)

    R3 = _linear_to_srgb(R2)
    G3 = _linear_to_srgb(G2)
    B3 = _linear_to_srgb(B2)
    return _clip01(R3), _clip01(G3), _clip01(B3)


def _ciel_lightness_from_rgb(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    X, Y, Z = _rgb_to_xyz_pi(r, g, b)
    L, _, _ = _xyz_to_lab_pi(X, Y, Z)
    return L / 100.0  # normalized-ish 0..1


def _lab_lightness_replace(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    Y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the script's Lab lightness replacement path:
      - Convert RGB -> XYZ -> Lab
      - Replace the Y-like term using caller-supplied Y2 (already in the script's (..+0.16)/1.16 space)
      - Rebuild XYZ using a/b and Y2 exactly like the script (no extra normalization)
      - Convert XYZ -> RGB
    """
    X, Y, Z = _rgb_to_xyz_pi(R, G, B)
    L, a, b = _xyz_to_lab_pi(X, Y, Z)

    # Script rebuild:
    X2 = (a / 500.0) + Y2
    Z2 = Y2 - (b / 200.0)

    def finv(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, t ** 3, (t - 16.0 / 116.0) / 7.787)

    X3 = finv(X2)
    Y3 = finv(Y2)
    Z3 = finv(Z2)
    return _xyz_to_rgb_pi(X3, Y3, Z3)


# ---------------- Common finishing steps ----------------

def _apply_hl_reduction_and_brightness_and_recover(E10: np.ndarray, params: NBNParams) -> np.ndarray:
    hlr = max(float(params.hlreduct), 0.25)      # HLReduction (0.5..2.0 typical)
    br = max(float(params.brightness), 0.25)     # Brightness  (0.5..2.0 typical)
    hrec = max(float(params.hlrecover), 0.25)    # HLRecover   (0.5..2.0 typical)

    # E11 = (mtf(~(1/HLReduction*.5),E10)*E10) + (E10*~E10);
    # NOTE: 1/HLReduction*.5 means (1/HLReduction)*0.5
    m_hlr = 1.0 - (0.5 / hlr)  # ~(0.5/hlr)
    m_hlr = float(np.clip(m_hlr, _EPS, 1.0 - _EPS))
    E11 = (_mtf(m_hlr, E10) * E10) + (E10 * _inv01(E10))

    # E12 = mtf((1/Brightness*.5),E11);
    m_b = float(np.clip(0.5 / br, _EPS, 1.0 - _EPS))
    E12 = _mtf(m_b, E11)

    # E13 = rescale(E12,0,HLRecover);
    E13 = _rescale(E12, 0.0, hrec)
    return _clip01(E13)


# ---------------- Shared “core normalize” building blocks ----------------

def _compute_M_E0(chs: Tuple[np.ndarray, ...], blackpoint: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements:
      M = min($T) + Blackpoint*(med($T)-min($T))
      E0 = adev($T)/1.2533 + mean($T) - M
    on a per-channel basis.
    """
    mins, meds, means = _stats_min_med_mean(chs)
    M = mins + float(blackpoint) * (meds - mins)
    adevs = _stats_adev_vec(chs)
    E0 = (adevs / 1.2533) + means - M
    return M.astype(np.float32), E0.astype(np.float32)


def _pm_norm_channel(
    Tref: np.ndarray,
    Ta: float,
    Tb: float,
    Mref: float,
    E0: np.ndarray,
    boost: float,
) -> np.ndarray:
    """
    PixelMath pattern used repeatedly:
      A = E0 / ~Mref
      E = (A[ref]*(1-A[other])/(A[ref]-2*A[ref]*A[other] + A[other])) / boost
      E2 = rescale(Tref, Mref, 1)
      E3 = ~(~mtf(E, E2) * ~min(Tref, Mref))
    Caller supplies indices/values already extracted as scalars Ta/Tb etc.
    """
    invM = max(float(_inv01(np.asarray(Mref, dtype=np.float32))), _EPS)
    A = E0 / invM

    denom = (Ta - 2.0 * Ta * Tb + Tb)
    E = (Ta * (1.0 - Tb)) / max(float(denom), _EPS)
    E = E / max(float(boost), _EPS)

    E2 = _rescale(Tref, float(Mref), 1.0)
    min_T_M = np.minimum(Tref, float(Mref))
    E3 = _inv01(_inv01(_mtf(E, E2)) * _inv01(min_T_M))
    return _clip01(E3)


# ---------------- Scenario cores ----------------

def _normalize_hoo(ha: np.ndarray, oiii: np.ndarray, params: NBNParams, progress_cb: ProgressCB) -> np.ndarray:
    T0 = ha
    T1 = oiii
    T2 = oiii

    if progress_cb:
        progress_cb(12, "Computing global stats")
    M, E0 = _compute_M_E0((T0, T1, T2), params.blackpoint)

    # --- scalar E1 for OIII normalize ---
    invM1 = max(float(_inv01(M[1])), _EPS)
    A0 = E0 / invM1
    Ta = float(A0[1])
    Tb = float(A0[0])
    denom = (Ta - 2.0 * Ta * Tb + Tb)
    E1 = (Ta * (1.0 - Tb)) / max(float(denom), _EPS)
    E1 = E1 / max(float(params.oiiiboost), _EPS)

    hb = float(np.clip(params.hablend, 0.0, 1.0))
    inv_hb = 1.0 - hb

    # Prealloc output
    h, w = T0.shape
    out = np.empty((h, w, 3), dtype=np.float32)

    tiles = list(_iter_tiles(h, w, tile=1024))

    if progress_cb:
        progress_cb(18, "Normalizing channels (tiled)")

    def _tile_worker(y0, y1, x0, x1, ti):
        t0 = T0[y0:y1, x0:x1]
        t1 = T1[y0:y1, x0:x1]

        # E3 for OIII
        E2 = _rescale(t1, float(M[1]), 1.0)
        min_t1_m1 = np.minimum(t1, float(M[1]))
        E3 = _inv01(_inv01(_mtf(E1, E2)) * _inv01(min_t1_m1))
        E3 = _clip01(E3)

        # Blend E4
        if params.blendmode == 0:
            E4 = (t0 * hb) + (E3 * inv_hb)
        elif params.blendmode == 1:
            E4 = (E3 * hb) + (t1 * inv_hb)
        else:
            E4 = (t0 * hb) + (t1 * inv_hb)

        R = t0
        G = E4
        B = E3

        if params.mode == 0:
            out[y0:y1, x0:x1, 0] = R
            out[y0:y1, x0:x1, 1] = G
            out[y0:y1, x0:x1, 2] = B
        else:
            if params.lightness == 0:
                X, Y, Z = _rgb_to_xyz_pi(R, G, B)
                L, _, _ = _xyz_to_lab_pi(X, Y, Z)
                Y2 = (L + 16.0) / 116.0
            elif params.lightness == 1:
                rgbT = np.stack([t0, t1, t1], axis=-1)
                ciel = _ciel_lightness_from_rgb(rgbT)
                Y2 = (ciel + 0.16) / 1.16
            elif params.lightness == 2:
                Y2 = (t0 + 0.16) / 1.16
            else:
                Y2 = (t1 + 0.16) / 1.16

            r3, g3, b3 = _lab_lightness_replace(R, G, B, Y2.astype(np.float32))
            out[y0:y1, x0:x1, 0] = r3
            out[y0:y1, x0:x1, 1] = g3
            out[y0:y1, x0:x1, 2] = b3

    _run_tiles_parallel(
        tiles,
        _tile_worker,
        progress_cb=progress_cb,
        p0=18,
        p1=75,
        label="Processing tiles",
    )


    if progress_cb:
        progress_cb(80, "Finishing (HL reduction / brightness / recover)")

    # Finish as a single pass (still vectorized), or tile if you want even more granular updates
    out = _finish_tiled(out, params, progress_cb)
    return out

def _normalize_sho(ha: np.ndarray, oiii: np.ndarray, sii: np.ndarray, params: NBNParams, progress_cb: ProgressCB) -> np.ndarray:
    # $T[0]=SII (R), $T[1]=Ha (G), $T[2]=OIII (B)
    T0 = sii
    T1 = ha
    T2 = oiii

    if progress_cb:
        progress_cb(12, "Computing global stats")
    M, E0 = _compute_M_E0((T0, T1, T2), params.blackpoint)

    # --- scalar params for SII normalize ---
    invM0 = max(float(_inv01(M[0])), _EPS)
    A = E0 / invM0
    Ta_sii = float(A[0])
    Tb_sii = float(A[1])
    denom = (Ta_sii - 2.0 * Ta_sii * Tb_sii + Tb_sii)
    E1_sii = (Ta_sii * (1.0 - Tb_sii)) / max(float(denom), _EPS)
    E1_sii = E1_sii / max(float(params.siiboost), _EPS)

    # --- scalar params for OIII normalize ---
    invM2 = max(float(_inv01(M[2])), _EPS)
    A = E0 / invM2
    Ta_oiii = float(A[2])
    Tb_oiii = float(A[1])
    denom = (Ta_oiii - 2.0 * Ta_oiii * Tb_oiii + Tb_oiii)
    E1_oiii = (Ta_oiii * (1.0 - Tb_oiii)) / max(float(denom), _EPS)
    E1_oiii = E1_oiii / max(float(params.oiiiboost2), _EPS)

    h, w = T0.shape
    out = np.empty((h, w, 3), dtype=np.float32)

    tiles = list(_iter_tiles(h, w, tile=1024))

    if progress_cb:
        progress_cb(18, "Normalizing channels (tiled)")

    def _tile_worker(y0, y1, x0, x1, ti):
        t0 = T0[y0:y1, x0:x1]  # SII
        t1 = T1[y0:y1, x0:x1]  # Ha
        t2 = T2[y0:y1, x0:x1]  # OIII

        # SII -> E3
        E2 = _rescale(t0, float(M[0]), 1.0)
        min_t0_m0 = np.minimum(t0, float(M[0]))
        E3 = _inv01(_inv01(_mtf(E1_sii, E2)) * _inv01(min_t0_m0))
        E3 = _clip01(E3)

        # OIII -> E6
        E5 = _rescale(t2, float(M[2]), 1.0)
        min_t2_m2 = np.minimum(t2, float(M[2]))
        E6 = _inv01(_inv01(_mtf(E1_oiii, E5)) * _inv01(min_t2_m2))
        E6 = _clip01(E6)

        R = E3
        if not params.scnr:
            G = t1
        else:
            G = np.minimum((R + E6) * 0.5, t1)
        B = E6

        if params.mode == 0:
            out[y0:y1, x0:x1, 0] = R
            out[y0:y1, x0:x1, 1] = G
            out[y0:y1, x0:x1, 2] = B
        else:
            if params.lightness == 0:
                X, Y, Z = _rgb_to_xyz_pi(R, G, B)
                L, _, _ = _xyz_to_lab_pi(X, Y, Z)
                Y2 = (L + 16.0) / 116.0
            elif params.lightness == 1:
                rgbT = np.stack([t0, t1, t2], axis=-1)
                ciel = _ciel_lightness_from_rgb(rgbT)
                Y2 = (ciel + 0.16) / 1.16
            elif params.lightness == 2:
                Y2 = (t1 + 0.16) / 1.16  # Ha
            elif params.lightness == 3:
                Y2 = (t0 + 0.16) / 1.16  # SII
            else:
                Y2 = (t2 + 0.16) / 1.16  # OIII

            r3, g3, b3 = _lab_lightness_replace(R, G, B, Y2.astype(np.float32))
            out[y0:y1, x0:x1, 0] = r3
            out[y0:y1, x0:x1, 1] = g3
            out[y0:y1, x0:x1, 2] = b3

    _run_tiles_parallel(
        tiles,
        _tile_worker,
        progress_cb=progress_cb,
        p0=18,
        p1=75,
        label="Processing tiles",
    )


    if progress_cb:
        progress_cb(80, "Finishing (HL reduction / brightness / recover)")
    out = _finish_tiled(out, params, progress_cb)
    return out

def _normalize_hso(ha: np.ndarray, oiii: np.ndarray, sii: np.ndarray, params: NBNParams, progress_cb: ProgressCB) -> np.ndarray:
    # $T[0]=Ha, $T[1]=SII, $T[2]=OIII
    T0 = ha
    T1 = sii
    T2 = oiii

    if progress_cb:
        progress_cb(12, "Computing global stats")
    M, E0 = _compute_M_E0((T0, T1, T2), params.blackpoint)

    # scalar for SII normalize (uses M[1], A[1] vs A[0])
    invM1 = max(float(_inv01(M[1])), _EPS)
    A = E0 / invM1
    Ta_sii = float(A[1])
    Tb_sii = float(A[0])
    denom = (Ta_sii - 2.0 * Ta_sii * Tb_sii + Tb_sii)
    E1_sii = (Ta_sii * (1.0 - Tb_sii)) / max(float(denom), _EPS)
    E1_sii = E1_sii / max(float(params.siiboost), _EPS)

    # scalar for OIII normalize (uses M[2], A[2] vs A[0])
    invM2 = max(float(_inv01(M[2])), _EPS)
    A = E0 / invM2
    Ta_oiii = float(A[2])
    Tb_oiii = float(A[0])
    denom = (Ta_oiii - 2.0 * Ta_oiii * Tb_oiii + Tb_oiii)
    E1_oiii = (Ta_oiii * (1.0 - Tb_oiii)) / max(float(denom), _EPS)
    E1_oiii = E1_oiii / max(float(params.oiiiboost2), _EPS)

    h, w = T0.shape
    out = np.empty((h, w, 3), dtype=np.float32)
    tiles = list(_iter_tiles(h, w, tile=1024))

    if progress_cb:
        progress_cb(18, "Normalizing channels (tiled)")

    def _tile_worker(y0, y1, x0, x1, ti):
        t0 = T0[y0:y1, x0:x1]  # Ha
        t1 = T1[y0:y1, x0:x1]  # SII
        t2 = T2[y0:y1, x0:x1]  # OIII

        # SII -> E3 (HSO uses T1 and M[1])
        E2 = _rescale(t1, float(M[1]), 1.0)
        min_t1_m1 = np.minimum(t1, float(M[1]))
        E3 = _inv01(_inv01(_mtf(E1_sii, E2)) * _inv01(min_t1_m1))
        E3 = _clip01(E3)
        # OIII -> E6
        E5 = _rescale(t2, float(M[2]), 1.0)
        min_t2_m2 = np.minimum(t2, float(M[2]))
        E6 = _inv01(_inv01(_mtf(E1_oiii, E5)) * _inv01(min_t2_m2))
        E6 = _clip01(E6)

        R = t0
        if not params.scnr:
            G = E3
        else:
            G = np.minimum((R + E6) * 0.5, E3)
        B = E6

        if params.mode == 0:
            out[y0:y1, x0:x1, 0] = R
            out[y0:y1, x0:x1, 1] = G
            out[y0:y1, x0:x1, 2] = B
        else:
            if params.lightness == 0:
                X, Y, Z = _rgb_to_xyz_pi(R, G, B)
                L, _, _ = _xyz_to_lab_pi(X, Y, Z)
                Y2 = (L + 16.0) / 116.0
            elif params.lightness == 1:
                rgbT = np.stack([t0, t1, t2], axis=-1)
                ciel = _ciel_lightness_from_rgb(rgbT)
                Y2 = (ciel + 0.16) / 1.16
            elif params.lightness == 2:
                Y2 = (t1 + 0.16) / 1.16  # Ha
            elif params.lightness == 3:
                Y2 = (t0 + 0.16) / 1.16  # SII
            else:
                Y2 = (t2 + 0.16) / 1.16  # OIII

            r3, g3, b3 = _lab_lightness_replace(R, G, B, Y2.astype(np.float32))
            out[y0:y1, x0:x1, 0] = r3
            out[y0:y1, x0:x1, 1] = g3
            out[y0:y1, x0:x1, 2] = b3

    _run_tiles_parallel(
        tiles,
        _tile_worker,
        progress_cb=progress_cb,
        p0=18,
        p1=75,
        label="Processing tiles",
    )


    if progress_cb:
        progress_cb(80, "Finishing (HL reduction / brightness / recover)")
    out = _finish_tiled(out, params, progress_cb)
    return out


def _normalize_hos(ha: np.ndarray, oiii: np.ndarray, sii: np.ndarray, params: NBNParams, progress_cb: ProgressCB) -> np.ndarray:
    # $T[0]=Ha, $T[1]=OIII, $T[2]=SII
    T0 = ha
    T1 = oiii
    T2 = sii

    if progress_cb:
        progress_cb(12, "Computing global stats")
    M, E0 = _compute_M_E0((T0, T1, T2), params.blackpoint)

    # scalar for OIII normalize (uses M[1], A[1] vs A[0])
    invM1 = max(float(_inv01(M[1])), _EPS)
    A = E0 / invM1
    Ta_oiii = float(A[1])
    Tb_oiii = float(A[0])
    denom = (Ta_oiii - 2.0 * Ta_oiii * Tb_oiii + Tb_oiii)
    E1_oiii = (Ta_oiii * (1.0 - Tb_oiii)) / max(float(denom), _EPS)
    E1_oiii = E1_oiii / max(float(params.oiiiboost2), _EPS)

    # scalar for SII normalize (uses M[2], A[2] vs A[0])
    invM2 = max(float(_inv01(M[2])), _EPS)
    A = E0 / invM2
    Ta_sii = float(A[2])
    Tb_sii = float(A[0])
    denom = (Ta_sii - 2.0 * Ta_sii * Tb_sii + Tb_sii)
    E1_sii = (Ta_sii * (1.0 - Tb_sii)) / max(float(denom), _EPS)
    E1_sii = E1_sii / max(float(params.siiboost), _EPS)

    h, w = T0.shape
    out = np.empty((h, w, 3), dtype=np.float32)

    tiles = list(_iter_tiles(h, w, tile=1024))

    if progress_cb:
        progress_cb(18, "Normalizing channels (tiled)")

    def _tile_worker(y0, y1, x0, x1, ti):
        t0 = T0[y0:y1, x0:x1]  # Ha
        t1 = T1[y0:y1, x0:x1]  # OIII
        t2 = T2[y0:y1, x0:x1]  # SII

        # OIII -> E3 (uses t1 and M[1])
        E2 = _rescale(t1, float(M[1]), 1.0)
        min_t1_m1 = np.minimum(t1, float(M[1]))
        E3 = _inv01(_inv01(_mtf(E1_oiii, E2)) * _inv01(min_t1_m1))
        E3 = _clip01(E3)

        # SII -> E6 (uses t2 and M[2])
        E5 = _rescale(t2, float(M[2]), 1.0)
        min_t2_m2 = np.minimum(t2, float(M[2]))
        E6 = _inv01(_inv01(_mtf(E1_sii, E5)) * _inv01(min_t2_m2))
        E6 = _clip01(E6)

        R = t0
        if not params.scnr:
            G = E3
        else:
            G = np.minimum((R + E6) * 0.5, E3)
        B = E6

        if params.mode == 0:
            out[y0:y1, x0:x1, 0] = R
            out[y0:y1, x0:x1, 1] = G
            out[y0:y1, x0:x1, 2] = B
        else:
            if params.lightness == 0:
                X, Y, Z = _rgb_to_xyz_pi(R, G, B)
                L, _, _ = _xyz_to_lab_pi(X, Y, Z)
                Y2 = (L + 16.0) / 116.0
            elif params.lightness == 1:
                rgbT = np.stack([t0, t1, t2], axis=-1)
                ciel = _ciel_lightness_from_rgb(rgbT)
                Y2 = (ciel + 0.16) / 1.16
            elif params.lightness == 2:
                Y2 = (t1 + 0.16) / 1.16  # Ha
            elif params.lightness == 3:
                Y2 = (t0 + 0.16) / 1.16  # SII
            else:
                Y2 = (t2 + 0.16) / 1.16  # OIII

            r3, g3, b3 = _lab_lightness_replace(R, G, B, Y2.astype(np.float32))
            out[y0:y1, x0:x1, 0] = r3
            out[y0:y1, x0:x1, 1] = g3
            out[y0:y1, x0:x1, 2] = b3

    _run_tiles_parallel(
        tiles,
        _tile_worker,
        progress_cb=progress_cb,
        p0=18,
        p1=75,
        label="Processing tiles",
    )


    if progress_cb:
        progress_cb(80, "Finishing (HL reduction / brightness / recover)")
    out = _finish_tiled(out, params, progress_cb)
    return out


def normalize_narrowband(
    ha: np.ndarray | None,
    oiii: np.ndarray | None,
    sii: np.ndarray | None,
    params: NBNParams,
    *,
    progress_cb: ProgressCB = None,
) -> np.ndarray:
    """
    Entry point used by the UI/worker. Dispatches to the correct scenario core.

    Inputs are expected to be mono float32 arrays in [0..1] (or at least clipped-ish).
    Returns RGB float32 [0..1].
    """
    scen = (params.scenario or "").split()[0].strip().upper()

    # small helper so we can always give sane progress ranges
    def cb(p: int, msg: str = ""):
        if progress_cb:
            progress_cb(int(max(0, min(100, p))), msg)

    cb(0, f"Starting {scen}")

    # Validate requirements
    if scen == "HOO":
        if ha is None or oiii is None:
            raise MissingChannelsError("HOO requires Ha and OIII.")
        # sii ignored for HOO
        cb(5, "Dispatching HOO")
        out = _normalize_hoo(
            ha.astype(np.float32, copy=False),
            oiii.astype(np.float32, copy=False),
            params,
            cb,
        )
        cb(100, "Done")
        return _clip01(out).astype(np.float32, copy=False)

    if scen in ("SHO", "HSO", "HOS"):
        missing = []
        if ha is None: missing.append("Ha")
        if oiii is None: missing.append("OIII")
        if sii is None: missing.append("SII")
        if missing:
            raise MissingChannelsError(f"{scen} requires " + ", ".join(missing) + ".")

        ha = ha.astype(np.float32, copy=False)
        oiii = oiii.astype(np.float32, copy=False)
        sii = sii.astype(np.float32, copy=False)

        cb(5, f"Dispatching {scen}")

        if scen == "SHO":
            out = _normalize_sho(ha, oiii, sii, params, cb)
        elif scen == "HSO":
            out = _normalize_hso(ha, oiii, sii, params, cb)
        else:  # "HOS"
            out = _normalize_hos(ha, oiii, sii, params, cb)

        cb(100, "Done")
        return _clip01(out).astype(np.float32, copy=False)

    # Unknown scenario
    raise ValueError(f"Unknown narrowband normalization scenario: {params.scenario!r}")
