# =============================================================================
# Seti Astro Suite Pro - Community Sky Atlas Submission Dialog
# Copyright (c) 2026 Franklin Marek | www.setiastro.com
# =============================================================================

import io
import json

import numpy as np
import requests
from PIL import Image as PILImage

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QCheckBox, QFrame, QMessageBox,
    QProgressBar, QPlainTextEdit,
)

ATLAS_ENDPOINT = "https://saspro-atlas-server.onrender.com/submit"

FILTER_OPTIONS = [
    "LRGB",
    "RGB",
    "Broadband",
    "Broadband+Narrowband",
    "Luminance",
    "Ha",
    "OIII",
    "SII",
    "Hβ",
    "SHO",
    "HOO",
    "HOS",
    "HSS",
    "Foraxx",
    "Other",
    "Unknown",
]


# ---------------------------------------------------------------------------
# Background upload worker
# ---------------------------------------------------------------------------
class _UploadWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, image_data: np.ndarray, wcs, filter_name: str,
                 object_name: str, description: str, link: str):
        super().__init__()
        self._image_data  = image_data
        self._wcs         = wcs
        self._filter_name = filter_name
        self._object_name = object_name
        self._description = description
        self._link        = link

    def run(self):
        try:
            self.progress.emit("Preparing thumbnail…")
            jpeg_bytes = self._make_thumbnail()

            self.progress.emit("Extracting WCS footprint…")
            meta = self._extract_wcs_meta()

            self.progress.emit("Uploading to atlas…")
            url = self._upload(jpeg_bytes, meta)

            self.finished.emit(url)

        except requests.exceptions.Timeout:
            self.error.emit(
                "Upload timed out. The server may be waking up — please try again in a moment."
            )
        except requests.exceptions.ConnectionError:
            self.error.emit("Could not reach the atlas server. Check your internet connection.")
        except Exception as exc:
            self.error.emit(str(exc))

    def _make_thumbnail(self) -> bytes:
        arr = self._image_data
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)

        arr8 = np.clip(arr * 255, 0, 255).astype(np.uint8)

        if arr8.ndim == 2:
            pil = PILImage.fromarray(arr8, mode="L").convert("RGB")
        elif arr8.shape[2] == 1:
            pil = PILImage.fromarray(arr8[:, :, 0], mode="L").convert("RGB")
        else:
            pil = PILImage.fromarray(arr8, mode="RGB")

        w, h = pil.size
        scale = 1024 / max(w, h)
        if scale < 1.0:
            pil = pil.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue()

    def _extract_wcs_meta(self) -> dict:
        from astropy.wcs import WCS as AstropyWCS
        import math

        wcs: AstropyWCS = self._wcs

        # Drop to 2D if WCS has extra axes from a 3D FITS header
        if wcs.naxis > 2:
            wcs = wcs.dropaxis(2)

        shape = getattr(wcs, "array_shape", None) or getattr(wcs, "pixel_shape", None)
        if shape is None:
            h, w = self._image_data.shape[:2]
        else:
            if len(shape) == 2:
                h, w = shape[0], shape[1]
            else:
                h, w = shape[-2], shape[-1]

        cx, cy = w / 2.0, h / 2.0
        sky_center = wcs.pixel_to_world(cx, cy)
        ra_center  = float(sky_center.ra.deg)
        dec_center = float(sky_center.dec.deg)

        corners_px = [(0, 0), (w, 0), (w, h), (0, h)]
        c_ra, c_dec = [], []
        for px, py in corners_px:
            sky = wcs.pixel_to_world(px, py)
            c_ra.append(round(float(sky.ra.deg), 6))
            c_dec.append(round(float(sky.dec.deg), 6))

        sky_left  = wcs.pixel_to_world(0,  cy)
        sky_right = wcs.pixel_to_world(w,  cy)
        sky_top   = wcs.pixel_to_world(cx, 0)
        sky_bot   = wcs.pixel_to_world(cx, h)
        fov_w = float(sky_left.separation(sky_right).deg)
        fov_h = float(sky_top.separation(sky_bot).deg)

        try:
            sky_north = wcs.pixel_to_world(cx, cy + 10)
            dra  = sky_north.ra.deg  - ra_center
            ddec = sky_north.dec.deg - dec_center
            pa   = round(math.degrees(math.atan2(-dra, ddec)) % 360, 2)
        except Exception:
            pa = 0.0

        return {
            "ra_center":   round(ra_center,  6),
            "dec_center":  round(dec_center, 6),
            "fov_w_deg":   round(fov_w, 4),
            "fov_h_deg":   round(fov_h, 4),
            "pa_deg":      pa,
            "corners_ra":  json.dumps(c_ra),
            "corners_dec": json.dumps(c_dec),
        }

    def _upload(self, jpeg_bytes: bytes, meta: dict) -> str:
        data = {
            "ra_center":   str(meta["ra_center"]),
            "dec_center":  str(meta["dec_center"]),
            "fov_w_deg":   str(meta["fov_w_deg"]),
            "fov_h_deg":   str(meta["fov_h_deg"]),
            "pa_deg":      str(meta["pa_deg"]),
            "corners_ra":  meta["corners_ra"],
            "corners_dec": meta["corners_dec"],
            "filter_name": self._filter_name,
            "object_name": self._object_name,
            "description": self._description,
            "link":        self._link,
        }
        files = {"thumbnail": ("thumbnail.jpg", jpeg_bytes, "image/jpeg")}
        resp = requests.post(ATLAS_ENDPOINT, data=data, files=files, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        return result.get("thumbnail_url", "")


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------
class AtlasDialog(QDialog):
    def __init__(self, doc, settings=None, parent=None):
        super().__init__(parent)
        self._doc      = doc
        self._settings = settings
        self._worker   = None

        self.setWindowTitle(self.tr("Share to Seti Astro Community Atlas"))
        self.setMinimumWidth(500)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        title = QLabel(self.tr("Seti Astro Community Sky Atlas"))
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(self.tr("Share your plate-solved image with the SASpro community."))
        subtitle.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(subtitle)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        layout.addWidget(line)

        # Filter + object row
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel(self.tr("Filter:")))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(FILTER_OPTIONS)
        self._filter_combo.setFixedWidth(160)
        filter_row.addWidget(self._filter_combo)
        filter_row.addSpacing(16)
        filter_row.addWidget(QLabel(self.tr("Object:")))
        self._object_edit = QLineEdit()
        self._object_edit.setPlaceholderText("e.g. M33  (optional)")
        filter_row.addWidget(self._object_edit, 1)
        layout.addLayout(filter_row)

        # Description row
        desc_row = QHBoxLayout()
        desc_label = QLabel(self.tr("Description:"))
        desc_label.setFixedWidth(80)
        desc_row.addWidget(desc_label)
        self._desc_edit = QPlainTextEdit()
        self._desc_edit.setPlaceholderText(
            "Optional — imaging notes, equipment, location, etc. (max 300 chars)"
        )
        self._desc_edit.setFixedHeight(60)
        self._desc_edit.setMaximumWidth(9999)
        self._desc_char_label = QLabel("0/300")
        self._desc_char_label.setStyleSheet("color: #666; font-size: 10px;")
        self._desc_char_label.setFixedWidth(40)
        self._desc_edit.textChanged.connect(self._on_desc_changed)
        desc_row.addWidget(self._desc_edit, 1)
        desc_row.addWidget(self._desc_char_label)
        layout.addLayout(desc_row)

        # Link row
        link_row = QHBoxLayout()
        link_label = QLabel(self.tr("Link:"))
        link_label.setFixedWidth(80)
        link_row.addWidget(link_label)
        self._link_edit = QLineEdit()
        self._link_edit.setPlaceholderText(
            "Optional — Astrobin, website, YouTube, etc. (https://...)"
        )
        link_row.addWidget(self._link_edit, 1)
        layout.addLayout(link_row)

        # Privacy info panel
        info_frame = QFrame()
        info_frame.setStyleSheet(
            "QFrame { background: #1e2a1e; border: 1px solid #3a5a3a; border-radius: 6px; }"
        )
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(12, 10, 12, 10)

        info_header = QHBoxLayout()
        privacy_label = QLabel(self.tr("What gets uploaded?"))
        privacy_label.setStyleSheet("font-weight: bold; color: #8bc88b;")
        info_header.addWidget(privacy_label)
        info_header.addStretch()

        self._info_toggle = QPushButton(self.tr("▼ Details"))
        self._info_toggle.setFlat(True)
        self._info_toggle.setStyleSheet("color: #8bc88b; font-size: 11px;")
        self._info_toggle.setCheckable(True)
        self._info_toggle.setChecked(False)
        self._info_toggle.toggled.connect(self._toggle_info)
        info_header.addWidget(self._info_toggle)
        info_layout.addLayout(info_header)

        self._info_detail = QLabel(self.tr(
            "• A downsampled thumbnail (max 1024px) of your image\n"
            "• The WCS sky coordinates (RA/Dec center, field of view, corner positions)\n"
            "• Your selected filter, optional object name, description, and link\n\n"
            "No personal information is collected — not your name, email, IP address, "
            "or location. By sharing, you grant Seti Astro a non-exclusive license to "
            "display this thumbnail in the community atlas. You retain full copyright."
        ))
        self._info_detail.setWordWrap(True)
        self._info_detail.setStyleSheet("color: #ccc; font-size: 11px;")
        self._info_detail.setVisible(False)
        info_layout.addWidget(self._info_detail)
        layout.addWidget(info_frame)

        # Consent
        self._consent_check = QCheckBox(
            self.tr("I agree to share this image under the terms above")
        )
        layout.addWidget(self._consent_check)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # Buttons
        btn_row = QHBoxLayout()
        self._atlas_btn = QPushButton(self.tr("View Atlas"))
        self._atlas_btn.setFlat(True)
        self._atlas_btn.setStyleSheet("color: #7a9fff; font-size: 11px;")
        self._atlas_btn.clicked.connect(self._open_atlas_page)
        btn_row.addWidget(self._atlas_btn)
        btn_row.addStretch()
        self._cancel_btn = QPushButton(self.tr("Cancel"))
        self._cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(self._cancel_btn)
        self._share_btn = QPushButton(self.tr("Share Image"))
        self._share_btn.setDefault(True)
        self._share_btn.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; padding: 6px 18px; border-radius: 4px; }"
            "QPushButton:hover { background: #388e3c; }"
            "QPushButton:disabled { background: #444; color: #888; }"
        )
        self._share_btn.clicked.connect(self._on_share)
        btn_row.addWidget(self._share_btn)
        layout.addLayout(btn_row)

    def _on_desc_changed(self):
        text = self._desc_edit.toPlainText()
        if len(text) > 300:
            cursor = self._desc_edit.textCursor()
            self._desc_edit.setPlainText(text[:300])
            self._desc_edit.setTextCursor(cursor)
        count = min(len(self._desc_edit.toPlainText()), 300)
        self._desc_char_label.setText(f"{count}/300")
        color = "#cc6666" if count >= 290 else "#666"
        self._desc_char_label.setStyleSheet(f"color: {color}; font-size: 10px;")

    def _open_atlas_page(self):
        import webbrowser
        webbrowser.open("https://f005.backblazeb2.com/file/setiastro-atlas/atlas.html")

    def _toggle_info(self, checked: bool):
        self._info_detail.setVisible(checked)
        self._info_toggle.setText(self.tr("▲ Hide") if checked else self.tr("▼ Details"))
        self.adjustSize()

    def _on_share(self):
        if not self._consent_check.isChecked():
            QMessageBox.information(
                self,
                self.tr("Consent Required"),
                self.tr("Please check the agreement box before sharing.")
            )
            return

        image_data = getattr(self._doc, "image", None)
        wcs        = self._doc.metadata.get("wcs") if hasattr(self._doc, "metadata") else None

        if image_data is None:
            QMessageBox.warning(self, self.tr("No Image"),
                                self.tr("No image data found in the active document."))
            return
        if wcs is None:
            QMessageBox.warning(self, self.tr("No WCS"),
                                self.tr("No WCS solution found. Plate solve the image first."))
            return

        filter_name = self._filter_combo.currentText()
        object_name = self._object_edit.text().strip()
        description = self._desc_edit.toPlainText().strip()
        link        = self._link_edit.text().strip()

        # Basic URL validation
        if link and not link.startswith(("http://", "https://")):
            link = "https://" + link

        self._share_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._status_label.setVisible(True)
        self._status_label.setText(self.tr("Starting upload…"))

        self._worker = _UploadWorker(image_data, wcs, filter_name,
                                     object_name, description, link)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg: str):
        self._status_label.setText(msg)

    def _on_finished(self, thumbnail_url: str):
        self._progress_bar.setVisible(False)
        self._status_label.setVisible(False)
        self._share_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)

        msg = QMessageBox(self)
        msg.setWindowTitle(self.tr("Shared Successfully"))
        msg.setText(self.tr(
            "Your image has been added to the Seti Astro Community Sky Atlas.\n\n"
            "Thank you for contributing!\n\n"
            "It may take a minute or two to appear — the atlas updates in the background."
        ))
        msg.setIcon(QMessageBox.Icon.Information)
        view_btn = msg.addButton(self.tr("View Atlas"), QMessageBox.ButtonRole.AcceptRole)
        msg.addButton(self.tr("Close"), QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        if msg.clickedButton() == view_btn:
            import webbrowser
            webbrowser.open("https://f005.backblazeb2.com/file/setiastro-atlas/atlas.html")

        self.close()

    def _on_error(self, msg: str):
        self._progress_bar.setVisible(False)
        self._status_label.setVisible(False)
        self._share_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)
        QMessageBox.critical(
            self,
            self.tr("Upload Failed"),
            self.tr(f"Could not share image:\n\n{msg}")
        )