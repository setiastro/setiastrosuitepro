from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from setiastro.saspro.doc_manager import DocManager, ImageDocument


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _two_hdu_fits(path):
    primary = fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32))
    extra = fits.ImageHDU(np.full((8, 8), 0.5, dtype=np.float32), name="WEIGHT")
    fits.HDUList([primary, extra]).writeto(path, overwrite=True)


def test_open_path_loads_auxiliary_image_hdus_by_default(qapp, tmp_path):
    path = str(tmp_path / "pair.fits")
    _two_hdu_fits(path)

    dm = DocManager()
    dm.open_path(path)
    images = [d for d in dm._docs if isinstance(d, ImageDocument)]

    assert len(images) == 2
    assert images[0].image.shape[:2] == (8, 8)


def test_open_path_can_skip_auxiliary_image_hdus(qapp, tmp_path):
    path = str(tmp_path / "pair.fits")
    _two_hdu_fits(path)

    dm = DocManager()
    dm.open_path(path, open_auxiliary_images=False)
    images = [d for d in dm._docs if isinstance(d, ImageDocument)]

    assert len(images) == 1
