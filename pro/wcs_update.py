# pro/wcs_update.py
from __future__ import annotations
import numpy as np

def _get_header_from_meta(meta: dict):
    return (meta.get("original_header") or
            meta.get("fits_header") or
            meta.get("header"))

def _has_sip(header) -> bool:
    try:
        return any(k in header for k in ("A_ORDER","B_ORDER","AP_ORDER","BP_ORDER"))
    except Exception:
        return False

def _strip_wcs_keys(hdr):
    wcs_prefixes = (
        "CTYPE","CUNIT","CDELT","CRPIX","CRVAL","PC","CD",
        "PV","PS","LONPOLE","LATPOLE","PROJP","RADESYS","EQUINOX",
        "A_","B_","AP_","BP_",
        "WCSAXES"
    )
    keys = list(hdr.keys())
    for k in keys:
        up = str(k).upper()
        if any(up.startswith(p) for p in wcs_prefixes):
            del hdr[k]

def _pixscale_rot_from_wcs(w):
    """
    Return (scale_x, scale_y) arcsec/pixel and rotation angle (deg).
    Works for CD or PC+CDELT. Assumes celestial 2D.
    """
    import numpy as _np
    # build CD matrix
    if w.wcs.has_cd():
        CD = _np.array(w.wcs.cd)
    else:
        # CDELT + PC
        CDELT = _np.array(w.wcs.cdelt)
        PC = _np.array(w.wcs.pc) if w.wcs.pc is not None else _np.eye(2)
        CD = PC @ _np.diag(CDELT)
    # scales are sqrt of column norms; convert deg/pix -> arcsec/pix
    sx = float(np.hypot(CD[0,0], CD[1,0])) * 3600.0
    sy = float(np.hypot(CD[0,1], CD[1,1])) * 3600.0
    # rotation is atan2 of -CD10, CD00 (TAN convention; aligns with "east-left" images)
    theta = float(np.degrees(np.arctan2(-CD[1,0], CD[0,0])))
    return sx, sy, theta

def update_wcs_after_crop(metadata: dict, M_src_to_dst: np.ndarray, out_w: int, out_h: int) -> dict:
    """
    Refit a WCS (TAN or TAN-SIP) after crop given the src->dst homography.
    If metadata contains 'debug_wcs_crop': True, prints detailed diagnostics
    and places a summary in metadata['__wcs_debug__'].
    """
    debug = True

    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.wcs.utils import fit_wcs_from_points
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except Exception:
        if debug:
            print("[WCS-CROP] astropy not available; skipping WCS update.")
        return metadata

    hdr0 = _get_header_from_meta(metadata)
    if hdr0 is None:
        if debug:
            print("[WCS-CROP] No header found in metadata; skipping.")
        return metadata

    # Normalize to fits.Header
    if not isinstance(hdr0, fits.Header):
        try:
            tmp = fits.Header()
            for k, v in dict(hdr0).items():
                try:
                    tmp[k] = v
                except Exception:
                    pass
            hdr0 = tmp
        except Exception:
            if debug:
                print("[WCS-CROP] Could not coerce header to fits.Header; skipping.")
            return metadata

    # Build old WCS
    try:
        w_old = WCS(hdr0, relax=True)
    except Exception as e:
        if debug:
            print(f"[WCS-CROP] WCS() failed: {e}; skipping.")
        return metadata

    # Grab some "before" stats
    try:
        old_crval = (float(w_old.wcs.crval[0]), float(w_old.wcs.crval[1]))
        old_crpix = (float(w_old.wcs.crpix[0]), float(w_old.wcs.crpix[1]))
    except Exception:
        old_crval = (np.nan, np.nan)
        old_crpix = (np.nan, np.nan)
    try:
        old_sx, old_sy, old_rot = _pixscale_rot_from_wcs(w_old)
    except Exception:
        old_sx = old_sy = old_rot = float("nan")

    # dst->src inverse homography
    try:
        M_dst_to_src = np.linalg.inv(M_src_to_dst)
    except Exception as e:
        if debug:
            print(f"[WCS-CROP] inv(M) failed: {e}")
        return metadata

    # Sample grid across output (unchanged)
    nx = min(25, max(5, out_w // max(1, out_w // 25)))
    ny = min(25, max(5, out_h // max(1, out_h // 25)))
    xs = np.linspace(0.5, out_w - 0.5, nx)
    ys = np.linspace(0.5, out_h - 0.5, ny)
    Xn, Yn = np.meshgrid(xs, ys)       # shapes (ny, nx)
    ones = np.ones_like(Xn)

    # NEW->OLD via inverse homography (unchanged)
    Xo_h = (M_dst_to_src[0,0]*Xn + M_dst_to_src[0,1]*Yn + M_dst_to_src[0,2]*ones)
    Yo_h = (M_dst_to_src[1,0]*Xn + M_dst_to_src[1,1]*Yn + M_dst_to_src[1,2]*ones)
    Wo_h = (M_dst_to_src[2,0]*Xn + M_dst_to_src[2,1]*Yn + M_dst_to_src[2,2]*ones)
    Xo = Xo_h / Wo_h
    Yo = Yo_h / Wo_h

    # Old WCS → sky coords (unchanged)
    try:
        sky = w_old.pixel_to_world(Xo, Yo)  # SkyCoord with shape (ny, nx) or similar
        from astropy.coordinates import SkyCoord
        if not isinstance(sky, SkyCoord):
            sky = SkyCoord(sky.ra, sky.dec)
    except Exception:
        radec = w_old.wcs_pix2world(np.column_stack([Xo.ravel(), Yo.ravel()]), 0)
        sky = SkyCoord(radec[:,0].reshape(Xo.shape), radec[:,1].reshape(Yo.shape), unit="deg")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # NEW: flatten to 1-D for fitting
    x_new = Xn.ravel()
    y_new = Yn.ravel()
    # SkyCoord supports reshape/ravel; make it 1-D of length N
    sky_flat = sky.reshape(x_new.shape)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # SIP degree choice (unchanged)
    use_sip = _has_sip(hdr0)
    sip_degree = None
    if use_sip:
        try:
            sip_degree = int(hdr0.get("A_ORDER", hdr0.get("AP_ORDER", 3)))
        except Exception:
            sip_degree = 3

    # Fit NEW WCS (pass 1-D data)
    try:
        w_new = fit_wcs_from_points(
            (x_new, y_new), sky_flat,
            sip_degree=sip_degree,
            projection='TAN',
            proj_point='center'
        )
        w_new.array_shape = (out_h, out_w)
    except Exception as e:
        if debug:
            print(f"[WCS-CROP] fit_wcs_from_points failed: {e}")
        return metadata

    # Residuals (evaluate on the same 1-D sample, then summarize)
    try:
        sky_fit = w_new.pixel_to_world(x_new, y_new)
        sep = sky_flat.separation(sky_fit).arcsecond
        rms_arcsec = float(np.sqrt(np.mean(sep**2)))
        p50 = float(np.percentile(sep, 50))
        p95 = float(np.percentile(sep, 95))
    except Exception:
        rms_arcsec = p50 = p95 = float("nan")

    # After stats
    try:
        new_crval = (float(w_new.wcs.crval[0]), float(w_new.wcs.crval[1]))
        new_crpix = (float(w_new.wcs.crpix[0]), float(w_new.wcs.crpix[1]))
    except Exception:
        new_crval = (np.nan, np.nan)
        new_crpix = (np.nan, np.nan)
    try:
        new_sx, new_sy, new_rot = _pixscale_rot_from_wcs(w_new)
    except Exception:
        new_sx = new_sy = new_rot = float("nan")

    # Build new header
    new_hdr = hdr0.copy()
    _strip_wcs_keys(new_hdr)
    wcards = w_new.to_header(relax=True)
    for k, v in wcards.items():
        try:
            new_hdr[k] = v
        except Exception:
            pass
    new_hdr["NAXIS"]  = 2
    new_hdr["NAXIS1"] = int(out_w)
    new_hdr["NAXIS2"] = int(out_h)

    # Optional: print debug
    if debug:
        print("[WCS-CROP] === BEFORE ===")
        print(f"  CRVAL  (deg): {old_crval}")
        print(f"  CRPIX  (pix): {old_crpix}")
        print(f"  SCALE  (as/px): ({old_sx:.3f}, {old_sy:.3f})  ROT(deg): {old_rot:.3f}")
        print("[WCS-CROP] === AFTER  ===")
        print(f"  CRVAL  (deg): {new_crval}")
        print(f"  CRPIX  (pix): {new_crpix}   (image is {out_w}x{out_h})")
        print(f"  SCALE  (as/px): ({new_sx:.3f}, {new_sy:.3f})  ROT(deg): {new_rot:.3f}")
        print(f"  SIP degree: {sip_degree if use_sip else 'None (pure TAN)'}")
        print(f"  Fit residuals (arcsec): RMS={rms_arcsec:.3f}  p50={p50:.3f}  p95={p95:.3f}")

    # Stash a structured summary for the UI
    debug_summary = {
        "before": {
            "crval_deg": old_crval,
            "crpix_pix": old_crpix,
            "scale_as_per_pix": (old_sx, old_sy),
            "rot_deg": old_rot,
        },
        "after": {
            "crval_deg": new_crval,
            "crpix_pix": new_crpix,
            "scale_as_per_pix": (new_sx, new_sy),
            "rot_deg": new_rot,
            "sip_degree": (sip_degree if use_sip else None),
            "size": (int(out_w), int(out_h)),
        },
        "fit": {
            "rms_arcsec": rms_arcsec,
            "p50_arcsec": p50,
            "p95_arcsec": p95,
            "grid": (int(nx), int(ny)),
        }
    }

    out_meta = dict(metadata)
    out_meta["original_header"] = new_hdr
    try:
        out_meta["wcs"] = w_new
    except Exception:
        pass
    out_meta["__wcs_debug__"] = debug_summary
    return out_meta
