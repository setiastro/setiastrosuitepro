"""A new TAN solution must not keep the file's previous SIP terms."""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from setiastro.saspro.mosaic_master import MosaicMasterDialog


def test_writing_a_tan_solution_removes_the_old_sip(tmp_path):
    path = tmp_path / "panel.fits"
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["CRPIX1"] = 4.0
    header["CRPIX2"] = 4.0
    header["CRVAL1"] = 1.0
    header["CRVAL2"] = 2.0
    header["CD1_1"] = -0.001
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.001
    header["A_ORDER"] = 2
    header["B_ORDER"] = 2
    header["A_2_0"] = 1.0e-5
    header["OBJECT"] = "M42"
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32), header=header).writeto(path)

    solved = fits.Header()
    solved["CTYPE1"] = "RA---TAN"
    solved["CTYPE2"] = "DEC--TAN"
    solved["CRPIX1"] = 4.5
    solved["CRPIX2"] = 4.5
    solved["CRVAL1"] = 10.0
    solved["CRVAL2"] = 20.0
    solved["CD1_1"] = -0.002
    solved["CD1_2"] = 0.0
    solved["CD2_1"] = 0.0
    solved["CD2_2"] = 0.002
    solved["WCSAXES"] = 2
    solved["RADESYS"] = "ICRS"

    MosaicMasterDialog._write_wcs_cards(MosaicMasterDialog, str(path), solved)

    with fits.open(path) as hdul:
        written = hdul[0].header
        assert written["OBJECT"] == "M42"
        assert written["CTYPE1"] == "RA---TAN"
        assert float(written["CRVAL1"]) == 10.0
        assert "A_ORDER" not in written
        assert "A_2_0" not in written
        assert "B_ORDER" not in written
        assert np.allclose(hdul[0].data, 0.0)
