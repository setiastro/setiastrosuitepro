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
    lightness: int           # HOO: 0..3, others: 0..4  (V1 legacy)
    blackpoint: float        # 0..1  (V1 legacy)
    hlrecover: float         # >= 0.25  (V1 legacy)
    hlreduct: float          # >= 0.25  (V1 legacy)
    brightness: float        # >= 0.25  (V1 legacy)
    blendmode: int = 0       # HOO only: 0/1/2  (V1 legacy)
    hablend: float = 0.6     # HOO only: 0..1  (V1 legacy)
    oiiiboost: float = 1.0   # HOO OIII boost  (V1 legacy)
    siiboost: float = 1.0    # SHO/HSO/HOS  (V1 legacy)
    oiiiboost2: float = 1.0  # SHO/HSO/HOS  (V1 legacy)
    scnr: bool = False       # SHO/HSO/HOS  (V1 legacy)

    # # === SASpro Narrowband Normalization V2 (Bill Blanshan) ===
    # V2 (Bill Blanshan v2.23) fields.  use_v2=True routes to the new
    # algorithm; existing callers get V2 with these defaults which match
    # Bill's script defaults on a linear stretched image.
    use_v2: bool = True
    bgn: bool = True                     # background neutralisation
    background_noise: float = 1.0        # MAD units above sky
    red_boost: float = 0.5               # boost midtone; 0.5 = neutral
    green_boost: float = 0.5
    blue_boost: float = 0.5
    v2_blend_mode: str = "Mode 1"        # OSC HOO synthetic green blend
    v2_blend_amount: float = 0.5
    luminance_hold: str = "Off"          # "Off"|"Preserve"|"Red"|"Green"|"Blue"
    osc_hoo: bool = False                # HOO only: use synthetic green
    show_background: bool = False        # diagnostic: paint sky as white


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


# # === SASpro Narrowband Normalization V2 (Bill Blanshan) ===
# ---------------------------------------------------------------------------
# V2 core.  Ported from Bill Blanshan's NarrowbandNormalizationV2 v2.23
# (lib/Methods.js, 2026, GPLv3).  All credit for the algorithm to Bill.
# ---------------------------------------------------------------------------

# Rec.709 luma coefficients (Bill uses these in shared/lib/Luminance.js)
_V2_LUMA_R, _V2_LUMA_G, _V2_LUMA_B = 0.2126, 0.7152, 0.0722

# The fillet that removes the hard edge at the sky.  See Bill's Methods.js.
_V2_NBN_FILLET = 0.50
_V2_NBN_FILLET_SHIFT = 0.23


def _v2_mtf(m: float, x: np.ndarray) -> np.ndarray:
    """Element-wise MTF.  mtf(m, m) = 0.5, mtf(0.5, x) = x."""
    if m == 0.5:
        return x
    num = (m - 1.0) * x
    den = (2.0 * m - 1.0) * x - m
    return (num / den).astype(np.float32, copy=False)


def _v2_channel_stats(ch: np.ndarray) -> dict:
    """Per-channel: mean, median, MAD, avg-abs-dev, 5th-percentile."""
    a = np.asarray(ch, dtype=np.float32).ravel()
    if a.size == 0:
        return dict(mean=0.0, median=0.0, mad=0.0, adev=0.0, background=0.0)
    mean = float(a.mean())
    median = float(np.median(a))
    dev = np.abs(a - median)
    return dict(
        mean=mean,
        median=median,
        mad=float(np.median(dev)),
        adev=float(dev.mean()),
        background=float(np.percentile(a, 5.0)),
    )


def _v2_hold_luminance(rgb_stacked: np.ndarray,
                       L_new: np.ndarray) -> np.ndarray:
    """Scale RGB per-pixel so Rec.709 luma equals L_new.  Preserves hue,
    desaturates instead of clipping on overshoot."""
    R = rgb_stacked[..., 0]; G = rgb_stacked[..., 1]; B = rgb_stacked[..., 2]
    L_old = _V2_LUMA_R * R + _V2_LUMA_G * G + _V2_LUMA_B * B
    k = np.where(L_old > 1e-6, L_new / (L_old + 1e-12), 1.0).astype(np.float32)
    out = rgb_stacked * k[..., None]
    mx = out.max(axis=-1)
    over = np.maximum(mx - 1.0, 0.0)
    denom = np.maximum(1.0 - over, 1e-6)
    out = np.where(over[..., None] > 0,
                   (out - over[..., None]) / denom[..., None],
                   out)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _v2_pack_rgb(scenario: str,
                 ha: np.ndarray | None,
                 oiii: np.ndarray | None,
                 sii: np.ndarray | None) -> np.ndarray:
    """Pack Ha/OIII/SII into an (H,W,3) RGB stack per the palette."""
    scen = (scenario or "").split()[0].strip().upper()
    if scen == "HOO":
        if ha is None or oiii is None:
            raise MissingChannelsError("HOO requires Ha and OIII.")
        return np.stack([ha, oiii, oiii], axis=-1).astype(np.float32,
                                                          copy=False)
    if scen not in ("SHO", "HSO", "HOS"):
        raise ValueError(f"Unknown palette scenario: {scenario!r}")
    missing = []
    if ha is None: missing.append("Ha")
    if oiii is None: missing.append("OIII")
    if sii is None: missing.append("SII")
    if missing:
        raise MissingChannelsError(
            f"{scen} requires " + ", ".join(missing) + ".")
    if scen == "SHO":
        return np.stack([sii, ha, oiii], axis=-1).astype(np.float32,
                                                         copy=False)
    if scen == "HSO":
        return np.stack([ha, sii, oiii], axis=-1).astype(np.float32,
                                                         copy=False)
    return np.stack([ha, oiii, sii], axis=-1).astype(np.float32,
                                                     copy=False)  # HOS


def _normalize_v2(rgb: np.ndarray, params: NBNParams,
                  progress_cb: ProgressCB) -> np.ndarray:
    """Bill Blanshan's V2 (v2.23) algorithm on an RGB stack."""
    def cb(p: int, msg: str = ""):
        if progress_cb:
            progress_cb(int(max(0, min(100, p))), msg)

    cb(10, "V2: measuring channel statistics")

    linear = (int(params.mode) == 0)  # SASpro's mode: 0=linear, 1=nonlinear
    osc = bool(params.osc_hoo)
    bgn = bool(params.bgn)
    show_background = bool(params.show_background)

    st = [_v2_channel_stats(rgb[..., c]) for c in range(3)]

    M = np.array([st[c]["background"] for c in range(3)], dtype=np.float32)
    M0 = float(M.mean())
    M3 = np.array([params.background_noise * st[c]["mad"] for c in range(3)],
                  dtype=np.float32)

    # N0 = (adev/1.2533 + mean - M) / (1 - M),  N1 = max(N0)
    N0 = np.zeros(3, dtype=np.float32)
    for c in range(3):
        denom = 1.0 - M[c]
        if denom > 1e-10:
            N0[c] = (st[c]["adev"] / 1.2533 + st[c]["mean"] - M[c]) / denom
    N1 = float(N0.max())

    # Inner midtones + linear gain
    innerM = np.zeros(3, dtype=np.float32)
    gain = np.ones(3, dtype=np.float32)
    for c in range(3):
        innerM[c] = _v2_mtf(N1, N0[c])
        if N0[c] > 1e-10:
            gain[c] = N1 / N0[c]

    # Fillet radius per channel
    filletK = np.zeros(3, dtype=np.float32)
    for c in range(3):
        m = float(innerM[c])
        if m > 1e-6:
            filletK[c] = max(0.0, _V2_NBN_FILLET * (1.0 - 2.0 * m) / m)

    # base / lo / fillet width
    base = np.zeros(3, dtype=np.float32)
    lo = np.zeros(3, dtype=np.float32)
    fillet = np.zeros(3, dtype=np.float32)
    for c in range(3):
        base[c] = M0 if bgn else M[c]
        lo[c] = (base[c] + (params.background_noise
                            - _V2_NBN_FILLET_SHIFT * filletK[c])
                 * st[c]["mad"])
        span = 1.0 - lo[c]
        if filletK[c] > 0 and span > 1e-10:
            fillet[c] = filletK[c] * st[c]["mad"] / span

    boost_vals = np.array([params.red_boost, params.green_boost,
                           params.blue_boost], dtype=np.float32).clip(
                               0.001, 0.999)
    boostM = 1.0 - boost_vals
    boostDiv = boostM * 2.0

    bgnFactor = np.ones(3, dtype=np.float32)
    if bgn:
        for c in range(3):
            d = 1.0 - M[c]
            if d > 1e-10:
                bgnFactor[c] = (1.0 - M0) / d

    # show_background diagnostic
    if show_background:
        eps = 1e-6
        out = np.zeros_like(rgb)
        for c in range(3):
            out[..., c] = np.where(rgb[..., c] <= lo[c] + eps, 1.0, 0.0)
        cb(100, "V2: background map")
        return out.astype(np.float32, copy=False)

    cb(40, "V2: applying normalization")

    out = np.zeros_like(rgb)
    for c in range(3):
        src = rgb[..., c]
        M2 = 1.0 - bgnFactor[c] * (1.0 - src) if bgn else src
        pedestal = base[c] + M3[c]
        M4 = np.minimum(M2, pedestal)

        span = 1.0 - lo[c]
        if span > 1e-10:
            N2 = np.clip((M2 - lo[c]) / span, 0.0, 1.0)
        else:
            N2 = np.zeros_like(M2)
        if fillet[c] > 0:
            N2 = N2 * (1.0 - np.exp(-N2 / fillet[c]))

        if linear:
            N3 = gain[c] * N2
        else:
            N3 = _v2_mtf(float(innerM[c]), N2)

        if linear:
            v = N3 / boostDiv[c] + M4
        else:
            v = 1.0 - (1.0 - _v2_mtf(float(boostM[c]), N3)) * (1.0 - M4)

        out[..., c] = np.maximum(v, M2)

    # OSC HOO synthetic green
    if osc:
        M2_G = (1.0 - bgnFactor[1] * (1.0 - rgb[..., 1])) if bgn else rgb[..., 1]
        bm = str(params.v2_blend_mode)
        if bm == "Mode 1":
            first, second = out[..., 0], out[..., 1]
        elif bm == "Mode 2":
            first, second = out[..., 0], M2_G
        else:  # Mode 3
            first, second = out[..., 1], M2_G
        ba = float(np.clip(params.v2_blend_amount, 0.0, 1.0))
        out[..., 1] = first * ba + second * (1.0 - ba)

    # Luminance hold
    lm = str(params.luminance_hold)
    if lm != "Off":
        if lm == "Preserve":
            if bgn:
                M2_all = np.stack(
                    [1.0 - bgnFactor[c] * (1.0 - rgb[..., c])
                     for c in range(3)], axis=-1)
            else:
                M2_all = rgb
            L_new = (_V2_LUMA_R * M2_all[..., 0]
                     + _V2_LUMA_G * M2_all[..., 1]
                     + _V2_LUMA_B * M2_all[..., 2])
        elif lm == "Red":
            L_new = out[..., 0]
        elif lm == "Green":
            L_new = out[..., 1]
        elif lm == "Blue":
            L_new = out[..., 2]
        else:
            L_new = None
        if L_new is not None:
            out = _v2_hold_luminance(out, L_new)

    cb(95, "V2: finalizing")
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


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

    # # === SASpro Narrowband Normalization V2 (Bill Blanshan) ===: route to V2 by default
    if getattr(params, "use_v2", True):
        cb(0, f"V2 {scen}")
        rgb = _v2_pack_rgb(scen, ha, oiii, sii)
        out = _normalize_v2(rgb, params, cb)
        cb(100, "Done")
        return out

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
