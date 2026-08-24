from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from setiastro.saspro.color_space_manager import (
    COLOR_SPACES,
    DEFAULT_COLOR_SPACE,
    VIEWPORT_MODE_UNMANAGED,
    get_icc_key_from_color_space_key,
    get_icc_profile_bytes,
    get_key_from_icc_key,
    get_qimage_color_space,
    get_viewport_color_mode_from_settings,
    get_working_color_space_from_settings,
    normalize_color_space_key,
    set_viewport_color_mode_to_settings,
    set_working_color_space_to_settings,
    tag_qimage_with_color_space,
    tag_qimage_with_working_color_space,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_aliases_normalize_to_canonical_keys():
    assert normalize_color_space_key("Display P3") == "DisplayP3"
    assert normalize_color_space_key("P3") == "DisplayP3"
    assert normalize_color_space_key("Adobe RGB (1998)") == "AdobeRGB"
    assert normalize_color_space_key("AdobeRGB1998") == "AdobeRGB"
    assert normalize_color_space_key("ProPhoto") == "ProPhotoRGB"
    assert normalize_color_space_key("ProPhoto RGB") == "ProPhotoRGB"
    assert normalize_color_space_key("sRGB") == "sRGB"


def test_icc_key_mapping_accepts_display_and_export_names():
    assert get_icc_key_from_color_space_key("Display P3") == "P3"
    assert get_icc_key_from_color_space_key("Adobe RGB (1998)") == "AdobeRGB"
    assert get_icc_key_from_color_space_key("ProPhoto") == "ProPhotoRGB"
    assert get_icc_key_from_color_space_key("sRGB") == "sRGB"
    assert get_key_from_icc_key("P3") == "DisplayP3"
    assert get_key_from_icc_key("ProPhotoRGB") == "ProPhotoRGB"


def test_srgb_icc_profile_bytes_are_available():
    data = get_icc_profile_bytes("sRGB")
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_settings_round_trip_uses_canonical_key():
    settings = QSettings("SetiAstroTest", "SASproColorTest")
    settings.clear()

    assert get_working_color_space_from_settings(settings) == DEFAULT_COLOR_SPACE
    assert set_working_color_space_to_settings("ProPhoto", settings) is True
    assert get_working_color_space_from_settings(settings) == "ProPhotoRGB"

    settings.clear()


def test_qimage_can_be_tagged_with_srgb(qapp):
    img = QImage(8, 8, QImage.Format.Format_RGB888)
    img.fill(Qt.GlobalColor.red)

    assert tag_qimage_with_color_space(img, "sRGB") is True
    assert get_qimage_color_space(img) == "sRGB"


def test_viewport_mode_defaults_to_unmanaged_and_can_convert_when_target_enabled(qapp):
    settings = QSettings("SetiAstroTest", "SASproViewportModeTest")
    settings.clear()

    assert get_viewport_color_mode_from_settings(settings) == VIEWPORT_MODE_UNMANAGED

    img = QImage(8, 8, QImage.Format.Format_RGB888)
    img.fill(Qt.GlobalColor.red)
    tag_qimage_with_working_color_space(img, settings)
    assert not img.colorSpace().isValid()

    assert set_working_color_space_to_settings("DisplayP3", settings) is True
    assert set_viewport_color_mode_to_settings("AdobeRGB", settings) is True
    assert get_viewport_color_mode_from_settings(settings) == "AdobeRGB"

    tagged_img = QImage(8, 8, QImage.Format.Format_RGB888)
    tagged_img.fill(Qt.GlobalColor.red)
    tag_qimage_with_working_color_space(tagged_img, settings)
    assert tagged_img.colorSpace().isValid()

    settings.clear()


def test_export_dialog_defaults_to_working_color_space(qapp):
    from setiastro.saspro.save_options import ExportDialog

    settings = QSettings("SetiAstroTest", "SASproExportTest")
    settings.clear()
    settings.setValue("color_management/working_space", "AdobeRGB")

    dialog = ExportDialog(None, "png", "8-bit", settings=settings)
    try:
        assert dialog.combo_color_space.count() == len(COLOR_SPACES)
        assert dialog.combo_color_space.currentData() == "AdobeRGB"
    finally:
        dialog.deleteLater()
        settings.clear()