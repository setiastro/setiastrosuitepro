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


def _settings_ini(tmp_path):
    from PyQt6.QtCore import QSettings
    ini = tmp_path / "prefs.ini"
    return QSettings(str(ini), QSettings.Format.IniFormat)


def test_settings_dialog_open_all_hdus_defaults_checked(qapp, tmp_path):
    from PyQt6.QtWidgets import QWidget
    from setiastro.saspro.ops.settings import SettingsDialog

    parent = QWidget()
    dlg = SettingsDialog(parent, _settings_ini(tmp_path))
    assert dlg.chk_open_all_hdus.isChecked()
    assert "Open All detected HDUs" in dlg.chk_open_all_hdus.text()
    dlg.close()


def test_settings_dialog_persists_open_all_hdus(qapp, tmp_path):
    from PyQt6.QtWidgets import QWidget
    from setiastro.saspro.ops.settings import SettingsDialog

    settings = _settings_ini(tmp_path)
    parent = QWidget()
    dlg = SettingsDialog(parent, settings)
    dlg.chk_open_all_hdus.setChecked(False)
    dlg._save_and_accept()
    assert settings.value("files/open_auxiliary_images", True, type=bool) is False

    dlg2 = SettingsDialog(parent, settings)
    assert dlg2.chk_open_all_hdus.isChecked() is False
    dlg2.chk_open_all_hdus.setChecked(True)
    dlg2._save_and_accept()
    assert settings.value("files/open_auxiliary_images", False, type=bool) is True
