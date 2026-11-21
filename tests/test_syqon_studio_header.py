# Copyright (c) Franklin Marek / SetiAstro. GPLv3.
"""The temp FITS SASpro hands SyQon Studio carries the document's full header
and plate solution, cleaned the way the FITS writer cleans it."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from setiastro.saspro.syqon_studio import _SyQonStudioCLIThread, studio_input_headers

H, W = 48, 64


class _Doc:
    def __init__(self, image, metadata):
        self.image = image
        self.metadata = metadata


def _solved_siril_doc(color=False):
    rng = np.random.default_rng(0)
    img = (0.05 + 0.01 * rng.standard_normal((H, W))).astype(np.float32)
    img[2:8, 3:20] = 0.7                       # marks the top rows
    if color:
        img = np.stack([img, img * 0.9, img * 0.8], axis=-1).astype(np.float32)
    w = WCS(naxis=2)
    w.wcs.crpix = [W / 2, H / 2]
    w.wcs.crval = [83.82, -5.39]
    w.wcs.cdelt = [-0.0004, 0.0004]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_hdr = w.to_header(relax=True)
    orig = fits.Header()
    orig["SIMPLE"] = True
    orig["BITPIX"] = 16
    orig["NAXIS"] = 2
    orig["NAXIS1"] = 9999                      # stale geometry
    orig["NAXIS2"] = 9999
    orig["BZERO"] = 32768                      # 16-bit source scaling
    orig["BSCALE"] = 1
    orig["OBJECT"] = "M42"
    orig["EXPTIME"] = 180.0
    orig["FOCALLEN"] = 530.0
    orig["DATAMIN"] = 0
    orig["DATAMAX"] = 65535                    # Siril scale, not 0..1
    orig["CBLACK"] = 120
    orig["CWHITE"] = 4000
    orig["ROWORDER"] = "TOP-DOWN"
    orig.update(wcs_hdr)
    return _Doc(img, {"original_header": orig, "wcs_header": wcs_hdr}), w


@pytest.mark.parametrize("color", [False, True])
def test_temp_fits_carries_full_header_and_solution(tmp_path, color):
    doc, w = _solved_siril_doc(color)
    hdr, wcs_hdr = studio_input_headers(doc)
    path = str(tmp_path / "in.fits")
    _SyQonStudioCLIThread._write_fits(path, doc.image, header=hdr, wcs_header=wcs_hdr)

    with fits.open(path) as hdul:
        h = hdul[0].header
        data = np.asarray(hdul[0].data)

    # Pixels exactly as the document holds them; colour as (3, H, W)
    got = data if data.ndim == 2 else np.transpose(data, (1, 2, 0))
    assert np.array_equal(got, doc.image)

    # Structural cards rebuilt for this array, not copied from the source
    assert h["BITPIX"] == -32
    assert (h["NAXIS1"], h["NAXIS2"]) == (W, H)
    assert h["NAXIS"] == (3 if color else 2)
    assert h.get("BZERO", 0) == 0

    # Metadata and plate solution present; the WCS describes these pixels
    assert h["OBJECT"] == "M42" and h["EXPTIME"] == 180.0 and h["FOCALLEN"] == 530.0
    sky = WCS(h).celestial.pixel_to_world_values(0, 0)
    assert np.allclose(sky, w.pixel_to_world_values(0, 0))
    assert h["ROWORDER"] == "TOP-DOWN"

    # Level cards follow the 0..1 data, not Siril's 0..65535
    assert h["DATAMAX"] == pytest.approx(float(doc.image.max()))
    assert h["CWHITE"] < 1.0


def test_document_headers_are_copied_not_shared():
    doc, _ = _solved_siril_doc()
    hdr, wcs_hdr = studio_input_headers(doc)
    hdr["OBJECT"] = "changed"
    wcs_hdr["CRVAL1"] = 0.0
    assert doc.metadata["original_header"]["OBJECT"] == "M42"
    assert doc.metadata["wcs_header"]["CRVAL1"] == 83.82


def test_no_header_still_writes_pixels(tmp_path):
    img = np.linspace(0, 1, H * W, dtype=np.float32).reshape(H, W)
    hdr, wcs_hdr = studio_input_headers(_Doc(img, {}))
    assert hdr is None and wcs_hdr is None
    path = str(tmp_path / "in.fits")
    _SyQonStudioCLIThread._write_fits(path, img, header=hdr, wcs_header=wcs_hdr)
    assert np.array_equal(fits.getdata(path), img)
