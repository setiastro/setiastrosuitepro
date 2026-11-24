"""Linear seed calibration from SASpro's Gaia XP library and spectral curves."""
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
from scipy.optimize import minimize
import sep

from .blot_stack import stars
from .gaia_database import GaiaBulkLibrary, WL_GRID, _cie_cmf
from .spectral_curves import read_curve


@dataclass(frozen=True)
class SpectralCalibration:
    filters: tuple   # three (FITS library path, curve name) pairs
    sensor: tuple
    white: tuple
    catalog: Path
    common_filter: tuple | None = None

    def response(self):
        # XP supplies spectral energy density per nm. CCD/CMOS counts photons.
        wl = WL_GRID.astype(float)
        def throughput(curve):
            return np.interp(wl, *read_curve(*curve), left=0., right=0.)
        common = wl * throughput(self.sensor)
        if self.common_filter:
            common *= throughput(self.common_filter)
        bands = np.array([common * throughput(f) for f in self.filters])
        with fits.open(self.white[0], memmap=False) as hdus:
            sed = hdus[self.white[1]].data
            wave = np.asarray(sed["WAVELENGTH"], dtype=float)
            if wave.max() > 2000:  # Built-in Pickles SEDs are in Angstrom.
                wave /= 10.
            white = np.interp(wl, wave, sed["FLUX"], left=0., right=0.)
        flux = bands @ white
        if not np.all(np.isfinite(flux) & (flux > 0)):
            raise ValueError("The selected passbands and white spectrum must overlap Gaia XP.")
        bands /= flux[:, None]
        _, target, _ = _cie_cmf(wl)
        target /= target @ white
        norm = np.linalg.norm(target)
        solution = minimize(lambda w: np.sum(((w @ bands-target)/norm)**2),
                            [1/3]*3, bounds=[(0, 1)]*3,
                            constraints=[dict(type="eq", fun=lambda w: w.sum()-1.)])
        if not solution.success:
            raise ValueError("Could not fit luminance to the selected passbands.")
        return wl, bands, solution.x

    @staticmethod
    def ratios(image, apertures, coverage):
        x, y, radius = np.asarray(apertures).T
        fluxes = []
        for plane, valid in zip(image, coverage):
            flux, _, flags = sep.sum_circle(np.ascontiguousarray(plane, dtype=np.float32), x, y, radius,
                                            bkgann=(1.5*radius, 2.5*radius),
                                            mask=~valid, subpix=5)
            flux[(flags != 0) | (flux <= 0)] = np.nan
            fluxes.append(flux)
        return (np.asarray(fluxes)[[0, 2]] / fluxes[1]).T

    def fit(self, image, coverage, header, check):
        wcs = WCS(header, naxis=2).celestial
        wl, bands, luminance = self.response()
        detected = stars(image[1], coverage.all(axis=0))
        if len(detected) < 3:
            raise ValueError("SPCC requires at least three measurable stars.")
        detected = detected[detected["flag"] == 0]
        detected = detected[np.argsort(detected["flux"])[::-1][:1000]]
        sky = wcs.pixel_to_world_values(detected["x"], detected["y"])
        coords = list(zip(*sky))
        radius = np.clip(2 * np.mean(proj_plane_pixel_scales(wcs)) * 3600, 1., 6.)
        apertures, expected = [], []
        used = set()
        with closing(GaiaBulkLibrary(self.catalog)) as library:
            matches = library.find_nearest_batch(coords, radius_arcsec=radius)
            for i, (sid, _) in sorted(matches.items(), key=lambda item: item[1][1]):
                check()
                if sid in used:
                    continue
                spectrum = library.get_spectrum(sid)
                if spectrum is None:
                    continue
                photons = bands @ np.interp(wl, spectrum.wavelengths, spectrum.flux, left=0., right=0.)
                if not np.all(np.isfinite(photons) & (photons > 0)):
                    continue
                used.add(sid)
                s = detected[i]
                apertures.append((s["x"], s["y"], np.clip(3.*s["a"], 2.5, 18.)))
                expected.append(photons[[0, 2]] / photons[1])
        if len(apertures) < 3:
            raise ValueError("Fewer than three Gaia XP matches. Check the reference WCS and installed XP sky coverage.")
        apertures = np.asarray(apertures)
        ratios = self.ratios(image, apertures, coverage)
        candidates = np.log(np.asarray(expected) / ratios)
        good = np.isfinite(candidates).all(axis=1)
        if good.any():
            center = np.median(candidates[good], axis=0)
            mad = np.median(np.abs(candidates[good] - center), axis=0)
            good &= (np.abs(candidates - center) <= np.maximum(4.5*mad, .02)).all(axis=1)
        if good.sum() < 3:
            raise ValueError("SPCC requires three photometrically consistent stars in all three channels.")
        gains = np.ones(3)
        gains[[0, 2]] = np.exp(np.median(candidates[good], axis=0))
        common = coverage.all(axis=0)
        background = np.median(image[:, common], axis=1)
        offsets = background[1] - gains * background
        for key, curve in zip(("R", "G", "B", "sensor", "white"), (*self.filters, self.sensor, self.white)):
            header.add_history(f"SPCC {key}: {curve[1]}")
        if self.common_filter:
            header.add_history(f"SPCC common filter: {self.common_filter[1]}")
        return gains, offsets, luminance, apertures[good]
