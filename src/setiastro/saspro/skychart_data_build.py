#saspro/skychart_data_build.py
# Copyright (C) Franklin Marek / SetiAstro. Part of Seti Astro Suite Pro (GPLv3).
"""
Builds the bundled sky-chart tables in setiastro/data/skychart/ from their
original sources. Developer tool: SASpro never runs this; it ships the output.

    python -m setiastro.saspro.skychart_data_build SOURCE_DIR [OUT_DIR]

SOURCE_DIR holds the files the original Sky Chart Generator script (Jean
d'Hooghvorst) downloads; their URLs are listed in SOURCES below. Each table is
written in the compact form skychart_engine reads:

  stars_hyg.npz        HYG 4.1 bright stars (from hygdata_v41.csv, 34 MB -> 2.4 MB)
  wds_doubles.npz      Washington Double Star primaries (from wds.tsv)
  ngc_outlines.npz     OpenNGC nebula outlines (from the OpenNGC repository zip,
                       or an outlines cache already built from it)
  halpha_map.npz       Finkbeiner 2003 H-alpha, HEALPix nside 512 NESTED galactic,
  dust_map.npz         SFD 1998 E(B-V), same layout. Both log10-quantised to
                       uint8 (worst-case error 2.5%, ~0.01 dex) -- the chart only
                       contours them, at 80 R and 0.32 dex.
  milkyway_gaia.npy    Integrated Gaia DR3 G-band starlight, HEALPix nside 128
                       NESTED ICRS (float32). Built by J. d'Hooghvorst from SASpro's
                       local Gaia library; the chart's contour levels are tuned to
                       this file, so it is carried as-is rather than rebuilt.
  NGC.csv, addendum.csv                        OpenNGC, verbatim
  constellations*.json, mw.json                d3-celestial, verbatim
  stellarium_modern_st.json                    Stellarium modern_st, verbatim
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

SOURCES = {
    "hygdata_v41.csv": "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv",
    "NGC.csv": "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/NGC.csv",
    "addendum.csv": "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/addendum.csv",
    "constellations.json": "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.json",
    "constellations.lines.json": "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.lines.json",
    "constellations.bounds.json": "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.bounds.json",
    "mw.json": "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/mw.json",
    "stellarium_st.json": "https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/modern_st/index.json",
    "OpenNGC-master.zip": "https://github.com/mattiaverga/OpenNGC/archive/refs/heads/master.zip",
    "lambda_halpha_fwhm06_0512.fits": "https://lambda.gsfc.nasa.gov/data/foregrounds/halpha/lambda_halpha_fwhm06_0512.fits",
    "lambda_sfd_ebv.fits": "https://lambda.gsfc.nasa.gov/data/foregrounds/SFD/lambda_sfd_ebv.fits",
    "wds.tsv": ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=B/wds/wds&-out=WDS&-out=Comp"
                "&-out=sep2&-out=mag1&-out=mag2&-out=RAJ2000&-out=DEJ2000&mag1=%3C13&-out.max=unlimited"),
}

NSIDE_MAPS = 512


def _f(s, default=float("nan")):
    try:
        s = s.strip()
        return float(s) if s else default
    except Exception:
        return default


def _sex2deg(s: str) -> float:
    s = s.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    parts = [float(x) for x in re.split(r"[:\s]+", s.lstrip("+-")) if x]
    while len(parts) < 3:
        parts.append(0.0)
    return sign * (parts[0] + parts[1] / 60.0 + parts[2] / 3600.0)


def build_stars(src: Path, out: Path):
    cols = {k: [] for k in ("ra", "dec", "mag", "ci", "name", "bayer", "flam", "con",
                            "var", "dbl", "hip", "hd", "varname")}
    with open(src / "hygdata_v41.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = _f(row["mag"])
            if not (m > -5.0):                  # drops the Sun and bad rows
                continue
            cols["ra"].append(_f(row["ra"]) * 15.0)
            cols["dec"].append(_f(row["dec"]))
            cols["mag"].append(m)
            cols["ci"].append(_f(row["ci"]))
            cols["name"].append(row["proper"].strip())
            cols["bayer"].append(row["bayer"].strip())
            cols["flam"].append(row["flam"].strip())
            cols["con"].append(row["con"].strip())
            # every catalogued variable (GCVS designation), whatever its amplitude
            cols["var"].append(bool(row["var"].strip()))
            cols["dbl"].append(bool(row.get("base", "").strip()))
            cols["hip"].append(int(_f(row.get("hip", ""), 0.0)))
            cols["hd"].append(int(_f(row.get("hd", ""), 0.0)))
            cols["varname"].append(row["var"].strip())
    np.savez_compressed(
        out / "stars_hyg.npz",
        ra=np.array(cols["ra"], np.float64), dec=np.array(cols["dec"], np.float64),
        mag=np.array(cols["mag"], np.float32), ci=np.array(cols["ci"], np.float32),
        name=np.array(cols["name"], "U32"), bayer=np.array(cols["bayer"], "U12"),
        flam=np.array(cols["flam"], "U6"), con=np.array(cols["con"], "U4"),
        var=np.array(cols["var"], bool), dbl=np.array(cols["dbl"], bool),
        hip=np.array(cols["hip"], np.int64), hd=np.array(cols["hd"], np.int64),
        varname=np.array(cols["varname"], "U12"))
    return len(cols["ra"])


def build_wds(src: Path, out: Path):
    """Primaries of real doubles: companion 0.5-45" away and at most 2 mag fainter
    (or brighter than mag 7). Faint, distant optical companions do not count."""
    ra, dec, mag = [], [], []
    for ln in open(src / "wds.tsv", encoding="utf-8", errors="replace"):
        if ln.startswith("#") or not ln[:1].isdigit():
            continue
        p = ln.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        try:
            sep, m1, m2 = float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        if not (0.5 <= sep <= 45.0 and m2 <= max(m1 + 2.0, 7.0)):
            continue
        ra.append(_sex2deg(p[5]) * 15.0)
        dec.append(_sex2deg(p[6]))
        mag.append(m1)
    np.savez_compressed(out / "wds_doubles.npz", ra=np.array(ra, np.float64),
                        dec=np.array(dec, np.float64), mag=np.array(mag, np.float32))
    return len(ra)


def build_outlines(src: Path, out: Path):
    names, levels, starts, pts = [], [], [0], []
    zp = src / "OpenNGC-master.zip"
    if zp.exists():
        def flush(ring, m):
            if len(ring) >= 4:
                names.append(m.group(1))
                levels.append(int(m.group(2) or 2))
                pts.extend(ring)
                starts.append(len(pts))

        with zipfile.ZipFile(zp) as z:
            for n in sorted(z.namelist()):
                m = re.search(r"/outlines/objects/([^/]+?)(?:_lv(\d))?\.txt$", n)
                if not m:
                    continue
                ring = []
                for line in z.read(n).decode("utf-8", "replace").splitlines()[1:]:
                    fl = line.split("\t")
                    if len(fl) < 4:
                        continue
                    try:
                        ring.append((float(fl[2]), float(fl[3])))
                    except ValueError:
                        continue
                    if fl[1].strip() == "*":
                        flush(ring, m)
                        ring = []
                flush(ring, m)
        names = np.array(names, "U24")
        levels = np.array(levels, np.int8)
        starts = np.array(starts, np.int64)
        pts = np.array(pts, np.float64)
    else:
        # The original script deletes the zip once it has its cache; that cache
        # holds exactly these four arrays, built by the same code as above.
        z = np.load(src / "outlines_cache.npz", allow_pickle=False)
        names, levels, starts, pts = z["names"], z["levels"], z["starts"], z["pts"]
    np.savez_compressed(out / "ngc_outlines.npz", names=names, levels=levels.astype(np.int8),
                        starts=starts.astype(np.int64), pts=np.asarray(pts, np.float32))
    return len(names)


def _quantise_log_u8(a: np.ndarray):
    a = np.asarray(a, np.float32)
    pos = a[a > 0]
    lo = float(np.log10(pos.min()))
    hi = float(np.log10(a.max()))
    q = np.round((np.log10(np.maximum(a, 10 ** lo)) - lo) / (hi - lo) * 255.0)
    return np.clip(q, 0, 255).astype(np.uint8), lo, hi


def build_sky_map(src: Path, out: Path, fits_name: str, npy_name: str, out_name: str):
    if (src / npy_name).exists():
        arr = np.load(src / npy_name).astype(np.float32)
    else:
        from astropy.io import fits
        with fits.open(src / fits_name, memmap=False) as h:
            hdr = h[1].header
            if hdr.get("ORDERING", "NESTED") != "NESTED":
                raise ValueError(f"{fits_name}: expected NESTED ordering")
            if str(hdr.get("COORDSYS", "G")).upper()[:1] not in ("G", ""):
                raise ValueError(f"{fits_name}: expected galactic coordinates")
            col = h[1].data.columns.names[0]
            arr = np.asarray(h[1].data[col], np.float32).ravel()
    if len(arr) != 12 * NSIDE_MAPS * NSIDE_MAPS:
        raise ValueError(f"{out_name}: unexpected HEALPix length {len(arr)}")
    q, lo, hi = _quantise_log_u8(arr)
    np.savez_compressed(out / out_name, q=q, log_lo=np.float64(lo), log_hi=np.float64(hi),
                        nside=np.int64(NSIDE_MAPS))
    return lo, hi


def build(src: Path, out: Path):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print("stars:", build_stars(src, out))
    print("doubles:", build_wds(src, out))
    print("outlines:", build_outlines(src, out))
    print("H-alpha log range:", build_sky_map(src, out, "lambda_halpha_fwhm06_0512.fits",
                                              "halpha_map.npy", "halpha_map.npz"))
    print("dust log range:", build_sky_map(src, out, "lambda_sfd_ebv.fits",
                                           "dust_map.npy", "dust_map.npz"))
    mw = np.load(src / "milkyway_gaia.npy")
    np.save(out / "milkyway_gaia.npy", np.asarray(mw, np.float32))
    for name in ("NGC.csv", "addendum.csv", "constellations.json", "constellations.lines.json",
                 "constellations.bounds.json", "mw.json"):
        shutil.copyfile(src / name, out / name)
    shutil.copyfile(src / "stellarium_st.json", out / "stellarium_modern_st.json")
    # sanity: every JSON parses
    for name in ("constellations.json", "constellations.lines.json", "constellations.bounds.json",
                 "mw.json", "stellarium_modern_st.json"):
        json.load(open(out / name, encoding="utf-8"))
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"wrote {out}  ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    here = Path(__file__).resolve().parent.parent / "data" / "skychart"
    build(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else here)
