"""Register and Integrate: per-frame star count / eccentricity used for weights.

The Register path measures every calibrated light with
``stacking_measure_worker.measure_file`` (2x2 superpixel preview ->
``compute_star_count_fast_preview`` for StarCount + Ecc, ``_measure_fwhm_halfres``
for FWHM) and scores it with ``StackingSuiteDialog._score_frame_terms``.

These tests feed it synthetic Seestar-like frames: 2-D GRBG CFA mosaics,
1920x1080, with the sky level and per-CFA-site noise sampled from real
calibrated Seestar S50 subs (fdcd58_Light_Pan2_10.0s_LP_20260928-023955_c.fit), round
Gaussian stars of FWHM 4 raw px (what sep measures on the real subs) and ~60
detectable stars (the real subs show 40-60 at 5 sigma). A star count that
measures stars must:
  * report far fewer stars on a frame that has none,
  * not change when the frame is shifted by a pixel or two,
  * not change between two exposures of the same sky that differ only in noise,
  * and the eccentricity must tell elongated stars from round ones.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from setiastro.saspro import stacking_measure_worker as measure_worker  # noqa: E402

H, W = 1920, 1080
# GRBG sites: (0,0)=G (0,1)=R (1,0)=B (1,1)=G. Sky and MAD-sigma per site, from
# a real calibrated sub; COLOR is a plausible relative star response.
SKY = {(0, 0): 0.00940, (0, 1): 0.00810, (1, 0): 0.00865, (1, 1): 0.00935}
SIG = {(0, 0): 0.00081, (0, 1): 0.00054, (1, 0): 0.00066, (1, 1): 0.00077}
COLOR = {(0, 0): 1.00, (0, 1): 0.80, (1, 0): 0.60, (1, 1): 1.00}
FWHM = 4.0


def _star_field(seed, n_bright=60, n_faint=100):
    """Star positions + raw-grid total fluxes. Bright stars peak ~8-1000 sigma
    on the superpixel preview; faint ones ~1-4 sigma (sky texture)."""
    rng = np.random.default_rng(seed)
    n = n_bright + n_faint
    xs = rng.uniform(20, W - 20, n)
    ys = rng.uniform(20, H - 20, n)
    fb = np.exp(rng.uniform(np.log(0.04), np.log(6.0), n_bright))
    ff = np.exp(rng.uniform(np.log(0.005), np.log(0.02), n_faint))
    return xs, ys, np.concatenate([fb, ff])


def _render(stars, *, noise_seed=0, stars_on=True, q=1.0, theta=0.0):
    """2-D GRBG CFA frame (H, W) float32. q = minor/major axis ratio of the PSF."""
    xs, ys, flux = stars
    sig = FWHM / 2.3548
    sa, sb = sig / np.sqrt(q), sig * np.sqrt(q)
    ct, st = np.cos(theta), np.sin(theta)
    img = np.zeros((H, W), np.float64)
    if stars_on:
        r = int(np.ceil(5 * sa)) + 1
        for x, y, f in zip(xs, ys, flux):
            x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
            y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx, dy = xx - x, yy - y
            u, v = dx * ct + dy * st, -dx * st + dy * ct
            img[y0:y1, x0:x1] += (f / (2 * np.pi * sa * sb)
                                  * np.exp(-0.5 * ((u / sa) ** 2 + (v / sb) ** 2)))
    rng = np.random.default_rng(1000 + noise_seed)
    out = np.empty((H, W), np.float32)
    for (oy, ox), sky in SKY.items():
        blk = img[oy::2, ox::2] * COLOR[(oy, ox)] + sky
        out[oy::2, ox::2] = blk + rng.normal(0.0, SIG[(oy, ox)], blk.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _bayer_after_crop(crop):
    """BAYERPAT of a GRBG mosaic after dropping `crop` rows and columns."""
    lab = np.array([["G", "R"], ["B", "G"]])
    return "".join(lab[(crop + i) % 2, (crop + j) % 2] for i in (0, 1) for j in (0, 1))


class _Frames:
    """Writes frames as calibrated-light FITS and measures them with measure_file."""

    def __init__(self, root):
        self.root = root
        self.n = 0

    def measure(self, cfa, bayer="GRBG"):
        self.n += 1
        fp = os.path.join(self.root, f"synth_{self.n:03d}_c.fit")
        hdr = fits.Header()
        hdr["BAYERPAT"] = bayer
        hdr["XBINNING"] = 1
        hdr["YBINNING"] = 1
        hdr["EXPTIME"] = 10.0
        fits.PrimaryHDU(np.ascontiguousarray(cfa), header=hdr).writeto(fp)
        status, _fp, payload = measure_worker.measure_file(fp, 1, 1)
        if status != "ok":
            raise AssertionError(f"measure_file failed on synthetic frame: {status} {payload}")
        mean, med, count, ecc, fwhm, noise = payload
        return dict(mean=mean, med=med, count=int(count), ecc=float(ecc),
                    fwhm=float(fwhm), noise=float(noise))


class StarCountStabilityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="saspro-starcount-")
        cls.frames = _Frames(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _m(self, cfa, bayer="GRBG"):
        return self.frames.measure(cfa, bayer)

    def test_starless_frame_counts_far_fewer_stars_than_star_field(self):
        # Same sky and noise model, with and without the stars (e.g. a sub that
        # clouded over completely). A correct detector finds ~0 stars in pure
        # noise; 25% of the star-field count leaves room for a low-sigma fallback.
        stars = _star_field(seed=1)
        rows = []
        for seed in (0, 1, 2):
            with_stars = self._m(_render(stars, noise_seed=seed))["count"]
            starless = self._m(_render(stars, noise_seed=seed, stars_on=False))["count"]
            rows.append((seed, with_stars, starless))
        for seed, with_stars, starless in rows:
            self.assertGreater(with_stars, 0, rows)
            self.assertLessEqual(starless, 0.25 * with_stars,
                                 f"noise seed {seed}: starless frame counted {starless} stars, "
                                 f"star field {with_stars}  (seed, with_stars, starless)={rows}")

    def test_star_count_is_stable_under_one_pixel_crop(self):
        # Same frame, same noise, cropped by 0..3 px. Only the sampling phase of
        # the PSFs moves, so only stars right at threshold can flip. sep at 5
        # sigma changes by <=4% here, and by 0.94-1.05x (p10-p90) between the
        # real subs and their 1-px crops; 15% leaves ~3x margin.
        rows = []
        for scene in (1, 2, 3):
            cfa = _render(_star_field(seed=scene))
            counts = [self._m(cfa[k:, k:], _bayer_after_crop(k))["count"] for k in range(4)]
            rows.append((scene, counts))
        for scene, counts in rows:
            self.assertGreater(min(counts), 0, rows)
            self.assertLessEqual(
                max(counts) / min(counts), 1.15,
                f"scene {scene}: counts for crops 0,1,2,3 px = {counts}; all={rows}")

    def test_star_count_is_repeatable_across_noise_realizations(self):
        # Six exposures of the identical sky (same stars, same transparency and
        # seeing), differing only in the noise. sep at 5 sigma varies 43-45 and
        # 49-50 on these scenes; 20% max/min leaves ~4x margin.
        rows = []
        for scene in (1, 2):
            stars = _star_field(seed=scene)
            counts = [self._m(_render(stars, noise_seed=k))["count"] for k in range(6)]
            rows.append((scene, counts))
        for scene, counts in rows:
            self.assertGreater(min(counts), 0, rows)
            self.assertLessEqual(
                max(counts) / min(counts), 1.20,
                f"scene {scene}: counts over 6 noise draws = {counts}; all={rows}")

    def test_eccentricity_separates_elongated_from_round_stars(self):
        # Round stars (true e=0) vs stars stretched to b/a=0.5 (true e=0.87),
        # e.g. trailing. sep measures 0.24 vs 0.74 here (median); any shape
        # measurement made on the stars should separate them by well over 0.2.
        stars = _star_field(seed=1)
        round_ecc = self._m(_render(stars))["ecc"]
        long_ecc = self._m(_render(stars, q=0.5, theta=0.5))["ecc"]
        self.assertGreaterEqual(long_ecc - round_ecc, 0.2,
                                f"Ecc round={round_ecc:.3f} elongated={long_ecc:.3f}")


class StarlessFrameWeightTests(unittest.TestCase):
    """End to end: the Balanced weight the Register path gives a frame
    with no stars, relative to a frame of the same sky with stars."""

    @classmethod
    def setUpClass(cls):
        from setiastro.saspro.stacking_suite import StackingSuiteDialog
        cls.dlg_cls = StackingSuiteDialog
        cls.tmp = tempfile.mkdtemp(prefix="saspro-starweight-")
        cls.frames = _Frames(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _weights(self, measured):
        # Mirrors the Register path (stacking_suite.py, "weighting mode +
        # population normalization" and the raw_scores loop): population
        # medians of positive FWHM / noise, bg floored at 1e-3, Balanced mode.
        exps = self.dlg_cls._weight_mode_exponents("Balanced")
        fw = [m["fwhm"] for m in measured.values() if m["fwhm"] > 1e-6]
        nz = [m["noise"] for m in measured.values() if m["noise"] > 1e-9]
        fwhm_ref = float(np.median(fw)) if fw else 0.0
        noise_ref = float(np.median(nz)) if nz else 0.0
        out = {}
        for k, m in measured.items():
            w, _ = self.dlg_cls._score_frame_terms(
                count=float(m["count"]), ecc=m["ecc"], bg=max(m["med"], 1e-3),
                fwhm=m["fwhm"], noise=m["noise"] or None,
                fwhm_ref=fwhm_ref, noise_ref=noise_ref, exps=exps)
            out[k] = w
        return out

    def test_starless_frame_weighs_far_less_than_star_field(self):
        # Mean-normalisation and the 0.1 floor that follow preserve the ratio
        # (down to the floor), so the raw-score ratio is what integration sees.
        stars = _star_field(seed=1)
        rows = []
        for seed in (0, 1, 2):
            measured = {
                "stars": self.frames.measure(_render(stars, noise_seed=seed)),
                "starless": self.frames.measure(_render(stars, noise_seed=seed, stars_on=False)),
            }
            w = self._weights(measured)
            rows.append((seed, round(w["stars"], 3), round(w["starless"], 3),
                         measured["stars"]["count"], measured["starless"]["count"]))
        for seed, w_stars, w_starless, _c1, _c0 in rows:
            self.assertGreater(w_stars, 0.0, rows)
            self.assertLessEqual(
                w_starless, 0.25 * w_stars,
                f"noise seed {seed}: weight starless={w_starless} vs stars={w_stars}; "
                f"(seed, w_stars, w_starless, count_stars, count_starless)={rows}")


if __name__ == "__main__":
    unittest.main()
