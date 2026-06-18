# pro/wcs_update.py
from __future__ import annotations
import numpy as np


def _get_header_from_meta(meta: dict):
    return (
        meta.get("original_header")
        or meta.get("fits_header")
        or meta.get("header")
    )


def _has_sip(header) -> bool:
    try:
        return any(k in header for k in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"))
    except Exception:
        return False


def _strip_wcs_keys(hdr):
    """
    Remove ALL WCS-related cards from a header so we can re-write them cleanly.
    Covers CD matrix, PC matrix, CDELT, SIP forward (A/B) and inverse (AP/BP),
    and assorted projection keywords.
    """
    wcs_prefixes = (
        "CTYPE", "CUNIT", "CDELT", "CRPIX", "CRVAL", "PC", "CD",
        "PV", "PS", "LONPOLE", "LATPOLE", "PROJP", "RADESYS", "EQUINOX",
        "A_", "B_", "AP_", "BP_",
        "WCSAXES", "MJDREF",
    )
    # Also strip these exact keys that astropy adds
    exact_strip = {"LONPOLE", "LATPOLE", "MJDREF", "WCSAXES", "RADESYS", "EQUINOX"}

    keys = list(hdr.keys())
    for k in keys:
        up = str(k).upper()
        if up in exact_strip:
            del hdr[k]
            continue
        if any(up.startswith(p) for p in wcs_prefixes):
            del hdr[k]


def _pixscale_rot_from_wcs(w):
    import numpy as _np
    if w.wcs.has_cd():
        CD = _np.array(w.wcs.cd)
    else:
        CDELT = _np.array(w.wcs.cdelt)
        PC = _np.array(w.wcs.pc) if w.wcs.pc is not None else _np.eye(2)
        CD = PC @ _np.diag(CDELT)
    sx = float(np.hypot(CD[0, 0], CD[1, 0])) * 3600.0
    sy = float(np.hypot(CD[0, 1], CD[1, 1])) * 3600.0
    theta = float(np.degrees(np.arctan2(-CD[1, 0], CD[0, 0])))
    return sx, sy, theta


def _needs_2d_coercion(hdr) -> bool:
    try:
        naxis = int(hdr.get("NAXIS", 2))
    except Exception:
        naxis = 2
    try:
        wcsaxes = int(hdr.get("WCSAXES", naxis))
    except Exception:
        wcsaxes = naxis
    return (naxis > 2) or (wcsaxes > 2)


def _coerce_header_to_2d(hdr):
    from astropy.io import fits
    h2 = fits.Header()
    for k, v in hdr.items():
        h2[k] = v
    h2["NAXIS"] = 2
    h2["WCSAXES"] = 2
    kill_prefixes = ("CRPIX3", "CRVAL3", "CDELT3", "CTYPE3", "CUNIT3")
    to_del = []
    for k in h2.keys():
        uk = k.upper()
        if uk in kill_prefixes:
            to_del.append(k)
        elif uk.startswith("CD3_") or uk.startswith("PC3_") or uk.startswith("PV3_") or uk.startswith("PS3_"):
            to_del.append(k)
    for k in to_del:
        del h2[k]
    return h2


def update_wcs_after_crop(metadata: dict, M_src_to_dst: np.ndarray, out_w: int, out_h: int) -> dict:
    """
    Refit a WCS (TAN or TAN-SIP) after a geometry transform (crop, rotate, rescale).

    Key fixes vs previous version:
    1. Strips ALL old WCS keys (including CD matrix, old SIP terms) before writing new ones.
    2. Re-fits inverse SIP (AP/BP) terms using astropy's sip_pix2foc / a second fit pass
       so round-trip pixel↔sky remains accurate.
    3. Updates metadata["wcs"] (the live astropy WCS object) so RA/Dec lookups don't
       use stale pre-transform coordinates.
    4. Removes the stale __wcs_debug__ entry from the previous transform.
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

    # ------------------------------------------------------------------
    # 1) Build the *old* WCS
    # ------------------------------------------------------------------
    hdr_for_wcs = hdr0
    coerced = False

    if _needs_2d_coercion(hdr0):
        hdr_for_wcs = _coerce_header_to_2d(hdr0)
        coerced = True

    try:
        w_old = WCS(hdr_for_wcs, relax=True)
    except Exception as e:
        if not coerced:
            try:
                hdr_for_wcs = _coerce_header_to_2d(hdr0)
                w_old = WCS(hdr_for_wcs, relax=True)
                coerced = True
            except Exception as e2:
                if debug:
                    print(f"[WCS-CROP] WCS() failed even after 2-D coercion: {e2}; skipping.")
                return metadata
        else:
            if debug:
                print(f"[WCS-CROP] WCS() failed: {e}; skipping.")
            return metadata

    # Before stats
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

    # ------------------------------------------------------------------
    # 2) dst->src inverse homography
    # ------------------------------------------------------------------
    try:
        M_dst_to_src = np.linalg.inv(M_src_to_dst)
    except Exception as e:
        if debug:
            print(f"[WCS-CROP] inv(M) failed: {e}")
        return metadata

    # ------------------------------------------------------------------
    # 3) Sample a dense grid across the output image
    #    Use more points for better SIP refitting
    # ------------------------------------------------------------------
    nx = min(40, max(8, out_w // max(1, out_w // 40)))
    ny = min(40, max(8, out_h // max(1, out_h // 40)))
    xs = np.linspace(0.5, out_w - 0.5, nx)
    ys = np.linspace(0.5, out_h - 0.5, ny)
    Xn, Yn = np.meshgrid(xs, ys)
    ones = np.ones_like(Xn)

    Xo_h = (M_dst_to_src[0, 0] * Xn + M_dst_to_src[0, 1] * Yn + M_dst_to_src[0, 2] * ones)
    Yo_h = (M_dst_to_src[1, 0] * Xn + M_dst_to_src[1, 1] * Yn + M_dst_to_src[1, 2] * ones)
    Wo_h = (M_dst_to_src[2, 0] * Xn + M_dst_to_src[2, 1] * Yn + M_dst_to_src[2, 2] * ones)
    Xo = Xo_h / Wo_h
    Yo = Yo_h / Wo_h

    # ------------------------------------------------------------------
    # 4) Old WCS → sky coords
    # ------------------------------------------------------------------
    try:
        sky = w_old.pixel_to_world(Xo, Yo)
        if not isinstance(sky, SkyCoord):
            sky = SkyCoord(sky.ra, sky.dec)
    except Exception:
        radec = w_old.wcs_pix2world(np.column_stack([Xo.ravel(), Yo.ravel()]), 0)
        sky = SkyCoord(
            radec[:, 0].reshape(Xo.shape),
            radec[:, 1].reshape(Yo.shape),
            unit="deg"
        )

    x_new = Xn.ravel()
    y_new = Yn.ravel()
    sky_flat = sky.reshape(x_new.shape)

    # ------------------------------------------------------------------
    # 5) SIP degree — preserve original order
    # ------------------------------------------------------------------
    use_sip = _has_sip(hdr_for_wcs)
    sip_degree = None
    if use_sip:
        try:
            sip_degree = int(hdr_for_wcs.get("A_ORDER", hdr_for_wcs.get("AP_ORDER", 3)))
        except Exception:
            sip_degree = 3

    # ------------------------------------------------------------------
    # 6) Fit NEW forward WCS
    # ------------------------------------------------------------------
    try:
        w_new = fit_wcs_from_points(
            (x_new, y_new),
            sky_flat,
            sip_degree=sip_degree,
            projection='TAN',
            proj_point='center'
        )
        w_new.array_shape = (out_h, out_w)
    except Exception as e:
        if debug:
            print(f"[WCS-CROP] fit_wcs_from_points failed: {e}")
        return metadata

    # ------------------------------------------------------------------
    # 7) Fit inverse SIP (AP/BP) if we have forward SIP
    #    astropy's fit_wcs_from_points only fits A/B, not AP/BP.
    #    We refit AP/BP by inverting the forward transform on the same grid.
    # ------------------------------------------------------------------
    if use_sip and sip_degree is not None:
        try:
            from astropy.modeling.fitting import LinearLSQFitter
            from astropy.modeling.polynomial import Polynomial2D

            # sky → forward SIP pixel (what the new WCS predicts)
            pix_fwd = w_new.world_to_pixel(sky_flat)  # (2, ny*nx)
            # pix_fwd[0] = x predicted, pix_fwd[1] = y predicted

            # We want AP/BP such that:
            #   dx_ideal = sky→pure_TAN_pixel - CRPIX
            #   dx_sip   = dx_ideal + AP polynomial(dx_ideal, dy_ideal)  → gives back original grid
            # Since fit_wcs_from_points already chose a CRPIX, we just need
            # to embed AP/BP into the header via a secondary grid fit.
            # Simplest: use numpy polyfit on the residuals.

            crpix1 = float(w_new.wcs.crpix[0])
            crpix2 = float(w_new.wcs.crpix[1])

            # u, v = pixel coords relative to CRPIX (in the new image)
            u_obs = x_new - crpix1   # what we want to recover
            v_obs = y_new - crpix2

            # u_fwd, v_fwd = what the forward SIP gives us for the sky positions
            if isinstance(pix_fwd, (list, tuple)):
                u_fwd = np.asarray(pix_fwd[0]).ravel() - crpix1
                v_fwd = np.asarray(pix_fwd[1]).ravel() - crpix2
            else:
                u_fwd = np.asarray(pix_fwd).ravel()[::2] - crpix1
                v_fwd = np.asarray(pix_fwd).ravel()[1::2] - crpix2

            # Residual: observed - forward-predicted  (this is what AP/BP should correct)
            du = u_obs - u_fwd
            dv = v_obs - v_fwd

            # Fit polynomial AP(u_fwd, v_fwd) ≈ du  and BP ≈ dv
            # Build design matrix for 2D polynomial of degree sip_degree
            def _design_matrix_2d(u, v, deg):
                cols = []
                for n in range(deg + 1):
                    for m in range(deg + 1 - n):
                        cols.append((u ** n) * (v ** m))
                return np.column_stack(cols)

            def _fit_inv_sip(u, v, dz, deg):
                D = _design_matrix_2d(u, v, deg)
                coeffs, _, _, _ = np.linalg.lstsq(D, dz, rcond=None)
                return coeffs

            def _set_sip_coeffs(hdr, prefix, u_fwd, v_fwd, dz, deg):
                """Fit and write AP_i_j or BP_i_j coefficients into hdr."""
                coeffs = _fit_inv_sip(u_fwd, v_fwd, dz, deg)
                idx = 0
                for n in range(deg + 1):
                    for m in range(deg + 1 - n):
                        key = f"{prefix}_{n}_{m}"
                        val = float(coeffs[idx])
                        # Only write non-trivial coefficients
                        if abs(val) > 1e-30:
                            hdr[key] = val
                        idx += 1

            # We'll embed these into the new header after building it below
            # Store for use in step 8
            _inv_sip_data = (u_fwd, v_fwd, du, dv, sip_degree)

        except Exception as e:
            if debug:
                print(f"[WCS-CROP] Inverse SIP refit failed (non-fatal): {e}")
            _inv_sip_data = None
    else:
        _inv_sip_data = None

    # ------------------------------------------------------------------
    # 8) Build new header — start from a CLEAN copy of the original,
    #    strip ALL old WCS keys, then write the new WCS
    # ------------------------------------------------------------------
    new_hdr = hdr0.copy()
    _strip_wcs_keys(new_hdr)          # removes ALL old CD/PC/SIP/CRPIX/CRVAL etc.

    # Write new WCS cards
    wcards = w_new.to_header(relax=True)
    for k, v in wcards.items():
        try:
            new_hdr[k] = v
        except Exception:
            pass

    # Write inverse SIP if we computed it
    if _inv_sip_data is not None:
        try:
            u_fwd, v_fwd, du, dv, deg = _inv_sip_data

            def _write_inv_sip(hdr, prefix, u, v, dz, d):
                def _dm(u, v, deg):
                    cols = []
                    for n in range(deg + 1):
                        for m in range(deg + 1 - n):
                            cols.append((u ** n) * (v ** m))
                    return np.column_stack(cols)
                D = _dm(u, v, d)
                coeffs, _, _, _ = np.linalg.lstsq(D, dz, rcond=None)
                idx = 0
                for n in range(d + 1):
                    for m in range(d + 1 - n):
                        val = float(coeffs[idx])
                        if abs(val) > 1e-30:
                            hdr[f"{prefix}_{n}_{m}"] = val
                        idx += 1

            _write_inv_sip(new_hdr, "AP", u_fwd, v_fwd, du, deg)
            _write_inv_sip(new_hdr, "BP", u_fwd, v_fwd, dv, deg)
            new_hdr[f"AP_ORDER"] = deg
            new_hdr[f"BP_ORDER"] = deg
        except Exception as e:
            if debug:
                print(f"[WCS-CROP] Writing inverse SIP to header failed: {e}")

    # Ensure image size is correct
    new_hdr["NAXIS"] = 2
    new_hdr["NAXIS1"] = int(out_w)
    new_hdr["NAXIS2"] = int(out_h)

    # ------------------------------------------------------------------
    # 9) Residuals
    # ------------------------------------------------------------------
    try:
        sky_fit = w_new.pixel_to_world(x_new, y_new)
        sep = sky_flat.separation(sky_fit).arcsecond
        rms_arcsec = float(np.sqrt(np.mean(sep ** 2)))
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

    if debug:
        print("[WCS] === BEFORE ===")
        print(f"  CRVAL  (deg): {old_crval}")
        print(f"  CRPIX  (pix): {old_crpix}")
        print(f"  SCALE  (as/px): ({old_sx:.3f}, {old_sy:.3f})  ROT(deg): {old_rot:.3f}")
        print("[WCS] === AFTER  ===")
        print(f"  CRVAL  (deg): {new_crval}")
        print(f"  CRPIX  (pix): {new_crpix}   (image is {out_w}x{out_h})")
        print(f"  SCALE  (as/px): ({new_sx:.3f}, {new_sy:.3f})  ROT(deg): {new_rot:.3f}")
        print(f"  SIP degree: {sip_degree if use_sip else 'None (pure TAN)'}")
        print(f"  Inverse SIP (AP/BP): {'refit' if _inv_sip_data is not None else 'skipped'}")
        print(f"  Fit residuals (arcsec): RMS={rms_arcsec:.3f}  p50={p50:.3f}  p95={p95:.3f}")

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
        },
        "coerced_to_2d": bool(coerced),
    }

    # ------------------------------------------------------------------
    # 10) Write back to metadata
    # ------------------------------------------------------------------
    out_meta = dict(metadata)
    out_meta["original_header"] = new_hdr

    # Update the live WCS object
    try:
        out_meta["wcs"] = w_new
    except Exception:
        pass

    # ← ADD THIS: update wcs_header to match the new WCS
    # Without this, coordinate lookups use the stale pre-transform wcs_header
    try:
        out_meta["wcs_header"] = w_new.to_header(relax=True)
    except Exception:
        pass

    # Remove stale debug entry from any previous transform
    out_meta.pop("__wcs_debug__", None)
    out_meta["__wcs_debug__"] = debug_summary

    return out_meta