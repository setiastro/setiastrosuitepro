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

    def __init__(self, image_data, wcs, filter_name, object_name, description, link, doc=None):
        super().__init__()
        self._image_data  = image_data
        self._wcs         = wcs
        self._doc         = doc   # fallback for header-based WCS
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
        from astropy.io import fits
        import math

        # --- Resolve WCS object from whatever is available in the doc ------
        wcs = self._wcs  # may be None, or a proper AstropyWCS

        if not isinstance(wcs, AstropyWCS):
            doc_meta = getattr(self._doc, "metadata", {}) or {}

            # 1) Try meta["wcs"] directly (astropy WCS object)
            candidate = doc_meta.get("wcs")
            if isinstance(candidate, AstropyWCS):
                wcs = candidate

            # 2) Try wcs_header (fits.Header or serialized string)
            if wcs is None:
                wcs_hdr = doc_meta.get("wcs_header")
                if wcs_hdr is not None:
                    try:
                        if isinstance(wcs_hdr, str) and wcs_hdr.strip():
                            wcs_hdr = fits.Header.fromstring(wcs_hdr)
                        if isinstance(wcs_hdr, fits.Header):
                            wcs = AstropyWCS(wcs_hdr)
                    except Exception:
                        wcs = None

            # 3) Try original_header / fits_header
            if wcs is None:
                for key in ("original_header", "fits_header"):
                    hdr = doc_meta.get(key)
                    if isinstance(hdr, str) and hdr.strip():
                        try:
                            hdr = fits.Header.fromstring(hdr)
                        except Exception:
                            hdr = None
                    if isinstance(hdr, fits.Header):
                        try:
                            wcs = AstropyWCS(hdr)
                            break
                        except Exception:
                            wcs = None

            # 4) Try image_meta["WCS"] dict (stored by Copy Astrometry)
            if wcs is None:
                wcs_dict = (doc_meta.get("image_meta") or {}).get("WCS")
                if wcs_dict and isinstance(wcs_dict, dict):
                    try:
                        hdr = fits.Header()
                        for k, v in wcs_dict.items():
                            try:
                                hdr[str(k)] = v.item() if hasattr(v, "item") else v
                            except Exception:
                                pass
                        wcs = AstropyWCS(hdr)
                    except Exception:
                        wcs = None

        if wcs is None:
            raise ValueError("No WCS solution found. Plate solve this image first.")

        # Drop to 2D if WCS has extra axes (e.g. from a 3D FITS header)
        if wcs.naxis > 2:
            wcs = wcs.dropaxis(2)

        # --- Image shape ---------------------------------------------------
        shape = getattr(wcs, "array_shape", None) or getattr(wcs, "pixel_shape", None)
        if shape is None:
            h, w = self._image_data.shape[:2]
        else:
            if len(shape) == 2:
                h, w = shape[0], shape[1]
            else:
                h, w = shape[-2], shape[-1]

        # --- Center RA/Dec -------------------------------------------------
        cx, cy = w / 2.0, h / 2.0
        sky_center = wcs.pixel_to_world(cx, cy)
        ra_center  = float(sky_center.ra.deg)
        dec_center = float(sky_center.dec.deg)

        # --- Four corners --------------------------------------------------
        corners_px = [(0, 0), (w, 0), (w, h), (0, h)]
        c_ra, c_dec = [], []
        for px, py in corners_px:
            sky = wcs.pixel_to_world(px, py)
            c_ra.append(round(float(sky.ra.deg), 6))
            c_dec.append(round(float(sky.dec.deg), 6))

        # --- FOV -----------------------------------------------------------
        sky_left  = wcs.pixel_to_world(0,  cy)
        sky_right = wcs.pixel_to_world(w,  cy)
        sky_top   = wcs.pixel_to_world(cx, 0)
        sky_bot   = wcs.pixel_to_world(cx, h)
        fov_w = float(sky_left.separation(sky_right).deg)
        fov_h = float(sky_top.separation(sky_bot).deg)

        # --- Position angle (fallback only) --------------------------------
        try:
            sky_north = wcs.pixel_to_world(cx, cy + 10)
            dra  = sky_north.ra.deg  - ra_center
            ddec = sky_north.dec.deg - dec_center
            pa   = round(math.degrees(math.atan2(-dra, ddec)) % 360, 2)
        except Exception:
            pa = 0.0

        # --- CD matrix scaled to thumbnail pixel resolution ----------------
        # The WCS CD matrix is in deg/pixel at ORIGINAL image resolution.
        # The thumbnail is 1024px on the long side, so we must scale it.
        THUMB_LONG = 1024
        if w >= h:
            thumb_w = THUMB_LONG
            thumb_h = max(1, round(THUMB_LONG * h / w))
        else:
            thumb_h = THUMB_LONG
            thumb_w = max(1, round(THUMB_LONG * w / h))

        # Scale factor: original pixels per thumbnail pixel
        scale_x = w / thumb_w   # original pixels per thumbnail pixel in X
        scale_y = h / thumb_h   # original pixels per thumbnail pixel in Y

        try:
            wcs_out = wcs.to_header(relax=True)

            if "CD1_1" in wcs_out:
                # Native CD matrix
                cd1_1 = float(wcs_out["CD1_1"])
                cd1_2 = float(wcs_out["CD1_2"])
                cd2_1 = float(wcs_out["CD2_1"])
                cd2_2 = float(wcs_out["CD2_2"])
            elif "PC1_1" in wcs_out:
                # PC matrix + CDELT — convert to CD
                cdelt1 = float(wcs_out.get("CDELT1", 1.0))
                cdelt2 = float(wcs_out.get("CDELT2", 1.0))
                cd1_1 = float(wcs_out["PC1_1"]) * cdelt1
                cd1_2 = float(wcs_out["PC1_2"]) * cdelt2
                cd2_1 = float(wcs_out["PC2_1"]) * cdelt1
                cd2_2 = float(wcs_out["PC2_2"]) * cdelt2
            else:
                raise ValueError("No CD or PC matrix in WCS header")

            # Scale from original to thumbnail pixel resolution
            # CD element units are deg/original_pixel
            # → deg/thumb_pixel = deg/original_pixel * original_pixels/thumb_pixel
            cd1_1_thumb = cd1_1 * scale_x
            cd1_2_thumb = cd1_2 * scale_y
            cd2_1_thumb = cd2_1 * scale_x
            cd2_2_thumb = cd2_2 * scale_y

        except Exception:
            # Fallback: reconstruct from pa and fov at thumbnail resolution
            pa_rad = pa * math.pi / 180.0
            cd1_1_thumb = -(fov_w / thumb_w) * math.cos(pa_rad)
            cd1_2_thumb = -(fov_h / thumb_h) * math.sin(pa_rad)
            cd2_1_thumb =  (fov_w / thumb_w) * math.sin(pa_rad)
            cd2_2_thumb =  (fov_h / thumb_h) * math.cos(pa_rad)

        return {
            "ra_center":   round(ra_center,  6),
            "dec_center":  round(dec_center, 6),
            "fov_w_deg":   round(fov_w, 4),
            "fov_h_deg":   round(fov_h, 4),
            "pa_deg":      pa,
            "cd1_1":       cd1_1_thumb,
            "cd1_2":       cd1_2_thumb,
            "cd2_1":       cd2_1_thumb,
            "cd2_2":       cd2_2_thumb,
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
            "cd1_1": str(meta["cd1_1"]),
            "cd1_2": str(meta["cd1_2"]),
            "cd2_1": str(meta["cd2_1"]),
            "cd2_2": str(meta["cd2_2"]),            
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
        # Refresh share state if the document changes (e.g. after plate solve or copy astrometry)
        if self._doc is not None and hasattr(self._doc, "changed"):
            self._doc.changed.connect(self._update_share_state)
            
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

        self._no_wcs_label = QLabel("")
        self._no_wcs_label.setStyleSheet("color: #cc8844; font-size: 11px;")
        self._no_wcs_label.setWordWrap(True)
        self._no_wcs_label.setVisible(False)
        layout.addWidget(self._no_wcs_label)
        # Buttons
        btn_row = QHBoxLayout()
        self._atlas_btn = QPushButton(self.tr("View Atlas"))
        self._atlas_btn.setStyleSheet(
            "QPushButton { background: #1a2a4a; color: #7a9fff; padding: 6px 18px; "
            "border-radius: 4px; border: 1px solid #3355aa; }"
            "QPushButton:hover { background: #223366; border-color: #5577cc; }"
        )
        self._atlas_btn.clicked.connect(self._open_atlas_page)
        btn_row.addWidget(self._atlas_btn)
        self._gallery_btn = QPushButton(self.tr("View Gallery"))
        self._gallery_btn.setStyleSheet(
            "QPushButton { background: #1a2a4a; color: #7a9fff; padding: 6px 18px; "
            "border-radius: 4px; border: 1px solid #3355aa; }"
            "QPushButton:hover { background: #223366; border-color: #5577cc; }"
        )
        self._gallery_btn.clicked.connect(self._open_gallery_page)
        btn_row.addWidget(self._gallery_btn)
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
        self._update_share_state()

    def _open_gallery_page(self):
        import webbrowser
        webbrowser.open("https://f005.backblazeb2.com/file/setiastro-atlas/gallery.html")

    def _update_share_state(self):
        image_data = getattr(self._doc, "image", None) if self._doc else None
        meta = self._doc.metadata if (self._doc and hasattr(self._doc, "metadata")) else {}

        has_wcs = bool(
            meta.get("wcs")
            or meta.get("wcs_header")
            or (meta.get("image_meta") or {}).get("WCS")
            or meta.get("HasAstrometricSolution")
        )

        if image_data is None or not has_wcs:
            self._share_btn.setEnabled(False)
            reason = ("Open an image in SASpro to submit to the atlas."
                    if (self._doc is None or image_data is None)
                    else "Plate solve this image first to enable submission.")
            self._no_wcs_label.setText(self.tr(reason))
            self._no_wcs_label.setVisible(True)
        else:
            self._share_btn.setEnabled(True)
            self._no_wcs_label.setVisible(False)

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
        meta = self._doc.metadata if hasattr(self._doc, "metadata") else {}
        wcs = (meta.get("wcs")
            or meta.get("image_meta", {}).get("WCS"))
        # if wcs is not AstropyWCS, pass None and let _extract_wcs_meta build it from headers
        if not hasattr(wcs, "pixel_to_world"):
            wcs = None
        filter_name = self._filter_combo.currentText()
        object_name = self._object_edit.text().strip()
        description = self._desc_edit.toPlainText().strip()
        link        = self._link_edit.text().strip()

        if link and not link.startswith(("http://", "https://")):
            link = "https://" + link

        self._share_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._status_label.setVisible(True)
        self._status_label.setText(self.tr("Starting upload…"))

        self._worker = _UploadWorker(image_data, wcs, filter_name,
                                    object_name, description, link, doc=self._doc)
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