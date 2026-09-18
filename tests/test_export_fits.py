from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from setiastro.saspro.doc_manager import DocManager, ImageDocument
from setiastro.saspro.export_fits import export_document_as_fits, export_fits_path


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _doc(qapp, *, file_path="", title="untitled", image=None, extra_meta=None):
    del qapp  # required so Qt is up before QObject docs
    arr = image if image is not None else np.full((6, 8), 0.25, dtype=np.float32)
    meta = {
        "file_path": file_path,
        "display_name": title,
        "bit_depth": "32-bit floating point",
        "is_mono": True,
    }
    if extra_meta:
        meta.update(extra_meta)
    return ImageDocument(arr, metadata=meta)


def test_export_fits_path_uses_source_stem(qapp, tmp_path):
    doc = _doc(qapp, file_path=str(tmp_path / "M31 Ha.tif"), title="ignored")
    out = export_fits_path(doc, str(tmp_path / "out"))
    assert os.path.basename(out) == "M31_Ha.fits"


def test_export_fits_path_strips_hdu_suffix(qapp, tmp_path):
    src = str(tmp_path / "pair.fits") + "::HDU 2"
    doc = _doc(qapp, file_path=src, title="pair")
    out = export_fits_path(doc, str(tmp_path / "out"))
    assert os.path.basename(out) == "pair.fits"


def test_export_fits_path_falls_back_to_display_name(qapp, tmp_path):
    doc = _doc(qapp, file_path="", title="NGC 7000")
    out = export_fits_path(doc, str(tmp_path / "out"))
    assert os.path.basename(out) == "NGC_7000.fits"


def test_export_skips_when_out_dir_missing(qapp):
    doc = _doc(qapp, title="no_folder")
    result = export_document_as_fits(DocManager(), doc, {"out_dir": "", "overwrite": True})
    assert result.ok is False
    assert result.skipped is True
    assert "folder" in result.reason.lower()


def test_export_writes_fits_without_changing_doc_path(qapp, tmp_path):
    src = str(tmp_path / "source.xisf")
    out_dir = tmp_path / "export"
    doc = _doc(qapp, file_path=src, title="source", image=np.full((6, 8), 0.4, dtype=np.float32))
    dm = DocManager()

    result = export_document_as_fits(dm, doc, {"out_dir": str(out_dir), "overwrite": True})

    assert result.ok is True
    assert result.path == str(out_dir / "source.fits")
    assert os.path.isfile(result.path)
    data = fits.getdata(result.path)
    assert data.shape == (6, 8)
    assert np.allclose(data, 0.4, atol=1e-5)
    assert doc.metadata.get("file_path") == src


def test_export_skips_existing_when_overwrite_false(qapp, tmp_path):
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    dest = out_dir / "keep.fits"
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32)).writeto(dest)

    doc = _doc(qapp, title="keep", image=np.full((6, 8), 0.9, dtype=np.float32))
    result = export_document_as_fits(
        DocManager(), doc, {"out_dir": str(out_dir), "overwrite": False}
    )

    assert result.ok is False
    assert result.skipped is True
    data = fits.getdata(dest)
    assert data.shape == (4, 4)
    assert np.allclose(data, 1.0)


def test_export_overwrites_when_requested(qapp, tmp_path):
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    dest = out_dir / "keep.fits"
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32)).writeto(dest)

    doc = _doc(qapp, title="keep", image=np.full((6, 8), 0.9, dtype=np.float32))
    result = export_document_as_fits(
        DocManager(), doc, {"out_dir": str(out_dir), "overwrite": True}
    )

    assert result.ok is True
    data = fits.getdata(result.path)
    assert data.shape == (6, 8)
    assert np.allclose(data, 0.9, atol=1e-5)


def test_export_fits_command_id_aliases():
    from setiastro.saspro.command_ids import normalize_command_id
    assert normalize_command_id("Export FITS") == "export_fits"
    assert normalize_command_id("save_fits") == "export_fits"


def test_export_fits_command_spec_is_registered():
    from setiastro.saspro.ops.commands import get_spec
    spec = get_spec("export_fits")
    assert spec is not None
    assert spec.headless_method == "_apply_export_fits_to_doc"
    assert spec.ui_method == "_open_export_fits"


def test_export_fits_preset_dialog_roundtrip(qapp):
    from setiastro.saspro.export_fits import ExportFitsPresetDialog
    dlg = ExportFitsPresetDialog(None, initial={"out_dir": "/tmp/fits_out", "overwrite": False})
    assert dlg.result_dict() == {"out_dir": "/tmp/fits_out", "overwrite": False}


def test_export_writes_live_fits_header_cards(qapp, tmp_path):
    hdr = fits.Header()
    hdr["OBJECT"] = "M31"
    hdr["FILTER"] = "Ha"
    hdr["CRVAL1"] = 10.6847
    hdr["CRVAL2"] = 41.2690
    doc = _doc(
        qapp,
        file_path=str(tmp_path / "m31.xisf"),
        title="m31",
        extra_meta={"original_header": hdr, "fits_header": hdr},
    )
    result = export_document_as_fits(
        DocManager(), doc, {"out_dir": str(tmp_path / "export"), "overwrite": True}
    )
    assert result.ok is True
    saved = fits.getheader(result.path)
    assert saved["OBJECT"] == "M31"
    assert saved["FILTER"] == "Ha"
    assert saved["CRVAL1"] == pytest.approx(10.6847)
    assert saved["CRVAL2"] == pytest.approx(41.2690)
    assert doc.metadata.get("file_path") == str(tmp_path / "m31.xisf")
    assert doc.metadata.get("original_header") is hdr


def test_export_rebuilds_header_from_snapshot(qapp, tmp_path):
    doc = _doc(
        qapp,
        file_path=str(tmp_path / "snap.fit"),
        title="snap",
        extra_meta={
            "__header_snapshot__": {
                "format": "fits-cards",
                "cards": [
                    ["OBJECT", "NGC7000", "target"],
                    ["FILTER", "OIII", ""],
                    ["CRVAL1", 314.0, ""],
                    ["CRVAL2", 44.3, ""],
                ],
            }
        },
    )
    result = export_document_as_fits(
        DocManager(), doc, {"out_dir": str(tmp_path / "export"), "overwrite": True}
    )
    assert result.ok is True
    saved = fits.getheader(result.path)
    assert saved["OBJECT"] == "NGC7000"
    assert saved["FILTER"] == "OIII"
    assert saved["CRVAL1"] == pytest.approx(314.0)
    assert "original_header" not in doc.metadata
    assert doc.metadata["__header_snapshot__"]["cards"][0][1] == "NGC7000"


def test_export_dialog_closes_after_successful_apply(qapp, tmp_path):
    from PyQt6.QtWidgets import QWidget
    from setiastro.saspro.export_fits import ExportFitsDialog

    class _Settings:
        def value(self, key, default="", type=str):
            return default

        def setValue(self, key, value):
            pass

    class _Main(QWidget):
        def __init__(self, doc, dm):
            super().__init__()
            self.docman = dm
            self.settings = _Settings()
            self._doc = doc
            self.logs = []

        def _active_doc(self):
            return self._doc

        def _log(self, msg):
            self.logs.append(msg)

    hdr = fits.Header()
    hdr["OBJECT"] = "M42"
    doc = _doc(qapp, title="orion", extra_meta={"original_header": hdr})
    mw = _Main(doc, DocManager())
    dlg = ExportFitsDialog(mw, {"out_dir": str(tmp_path), "overwrite": True})
    dlg.show()
    assert dlg.isVisible()
    dlg._apply()
    assert not dlg.isVisible()
    assert os.path.isfile(tmp_path / "orion.fits")
    assert fits.getheader(tmp_path / "orion.fits")["OBJECT"] == "M42"
