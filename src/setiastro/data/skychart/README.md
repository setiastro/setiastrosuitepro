# Sky chart data

Tables read by `setiastro/saspro/skychart_engine.py` (Finder Chart's drawn chart
and its overlay on survey images). Rebuild them from the original sources with:

    python -m setiastro.saspro.skychart_data_build SOURCE_DIR

The engine is based on the Sky Chart Generator by Jean d'Hooghvorst, contributed
to SASpro.

| File | Source | Licence |
|---|---|---|
| `stars_hyg.npz` | HYG Database v4.1, David Nash (astronexus/HYG-Database) | CC BY-SA 4.0 |
| `NGC.csv`, `addendum.csv` | OpenNGC, Mattia Verga (mattiaverga/OpenNGC), verbatim | CC BY-SA 4.0 |
| `ngc_outlines.npz` | OpenNGC nebula outlines (same repository) | CC BY-SA 4.0 |
| `constellations.json`, `constellations.lines.json`, `constellations.bounds.json`, `mw.json` | d3-celestial, Olaf Frohn (ofrohn/d3-celestial), verbatim | BSD-3-Clause |
| `stellarium_modern_st.json` | Stellarium `modern_st` sky culture (Sky & Telescope figures), verbatim | GPL-2.0-or-later |
| `wds_doubles.npz` | Washington Double Star Catalog (USNO), via VizieR B/wds | US Government work |
| `halpha_map.npz` | Finkbeiner 2003 full-sky H-alpha map, NASA LAMBDA (`lambda_halpha_fwhm06_0512.fits`) | NASA data |
| `dust_map.npz` | Schlegel, Finkbeiner & Davis 1998 E(B-V) dust map, NASA LAMBDA (`lambda_sfd_ebv.fits`) | NASA data |
| `dobashi_dark.npz` | Dobashi et al. 2011 (PASJ 63, S1) 2MASS extinction A_J, reduced to dark-cloud excess by J. d'Hooghvorst | Published data |
| `rcw.tsv` | RCW H-alpha regions (Rodgers, Campbell & Whiteoak 1960), VizieR VII/216, verbatim | VizieR / CDS |
| `darkneb.tsv` | Dark nebulae merged from 21 catalogs (Dutra & Bica 2002), VizieR J/A+A/383/631, verbatim | VizieR / CDS |
| `milkyway_gaia.npy` | Integrated G-band starlight of Gaia DR3 (ESA/Gaia/DPAC), built by J. d'Hooghvorst | Gaia data: CC BY-SA 3.0 IGO |

Encodings:

- `stars_hyg.npz`: every HYG star brighter than mag -5 excluded (the Sun), with
  name, Bayer, Flamsteed, constellation, B-V, HIP/HD, GCVS variable flag and name.
- `wds_doubles.npz`: primaries of pairs 0.5-45" apart with the companion at most
  2 mag fainter (or brighter than mag 7).
- `halpha_map.npz`, `dust_map.npz`: HEALPix nside 512, NESTED, galactic;
  `value = 10 ** (log_lo + q / 255 * (log_hi - log_lo))`. Worst-case error 2.5%
  (about 0.01 dex); the chart only contours these maps, at 80 R and 0.32 dex.
- `dobashi_dark.npz`: `aj` (uint8, galactic grid at 2', |b| <= 70) and `meta`
  `(b0, db, l0, dl, step, margin, bmax)`; `excess = aj * step - margin` (mag of A_J
  above the local ~1 deg background and twice the map noise). Prepared offline, not
  rebuilt by `skychart_data_build.py`. Where it covers the sky it replaces the SFD
  dust map for the dark nebulae.
- `rcw.tsv`, `darkneb.tsv`: names only (labels for the shapes the H-alpha and dust
  maps draw), never drawn as layers.
- `milkyway_gaia.npy`: HEALPix nside 128, NESTED, ICRS, float32. The Milky Way
  contour levels in the engine are tuned to this file's scale.
