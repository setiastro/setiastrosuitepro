from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColorSpace, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from setiastro.saspro.color_space_manager import (
    COLOR_SPACES,
    DEFAULT_RENDERING_INTENT,
    RENDERING_INTENT_PERCEPTUAL,
    RENDERING_INTENT_RELATIVE,
    convert_qimage_with_lcms,
    get_color_space_from_key,
    get_profile_search_directories,
    get_softproof_black_point_compensation_from_settings,
    get_softproof_rendering_intent_from_settings,
    get_viewport_color_mode_from_settings,
    get_working_color_space_from_settings,
    set_softproof_black_point_compensation_to_settings,
    set_softproof_rendering_intent_to_settings,
)


def _icc_search_dirs() -> list[Path]:
    dirs = [Path(p) for p in get_profile_search_directories()]
    system = platform.system()
    if system == "Windows":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        dirs.append(Path(root) / "System32" / "spool" / "drivers" / "color")
    elif system == "Darwin":
        dirs.extend([Path("/System/Library/ColorSync/Profiles"), Path("/Library/ColorSync/Profiles")])
    else:
        dirs.extend([Path("/usr/share/color/icc"), Path("/usr/local/share/color/icc")])

    unique: list[Path] = []
    seen = set()
    for d in dirs:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _find_icc_files() -> list[Path]:
    files: list[Path] = []
    seen = set()
    for directory in _icc_search_dirs():
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            candidates = list(directory.glob("*.icc")) + list(directory.glob("*.icm"))
        except Exception:
            continue
        for path in candidates:
            key = str(path).lower()
            if key not in seen:
                seen.add(key)
                files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def _qimage_to_array(img: QImage) -> np.ndarray:
    converted = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = converted.width(), converted.height()
    ptr = converted.bits()
    ptr.setsize(h * converted.bytesPerLine())
    raw = np.frombuffer(ptr, dtype=np.uint8).reshape((h, converted.bytesPerLine()))
    return raw[:, : w * 3].reshape((h, w, 3)).copy()


def _array_to_qimage(rgb: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _copy_preview_array(source: np.ndarray, max_dim: int = 1400) -> np.ndarray:
    arr = np.asarray(source, dtype=np.float32).copy()
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3].copy()

    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        step = int(np.ceil(longest / max_dim))
        arr = arr[::step, ::step].copy()
    return np.clip(arr, 0.0, 1.0)


class SoftProofDialog(QDialog):
    def __init__(self, image_array: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Soft Proof"))
        self.resize(1000, 760)
        self._source = _copy_preview_array(image_array)
        self._icc_files = _find_icc_files()

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        self.combo_display = QComboBox(self)
        for key, info in COLOR_SPACES.items():
            self.combo_display.addItem(self._color_space_label(key), key)
        current_display = get_viewport_color_mode_from_settings()
        if current_display not in COLOR_SPACES:
            current_display = get_working_color_space_from_settings()
        idx = self.combo_display.findData(current_display)
        if idx >= 0:
            self.combo_display.setCurrentIndex(idx)
        form.addRow(self.tr("Display profile:"), self.combo_display)

        proof_row = QHBoxLayout()
        self.combo_proof = QComboBox(self)
        for path in self._icc_files:
            self.combo_proof.addItem(path.name, str(path))
        proof_row.addWidget(self.combo_proof, 1)
        self.btn_browse = QPushButton(self.tr("Browse…"), self)
        self.btn_browse.clicked.connect(self._browse_profile)
        proof_row.addWidget(self.btn_browse)
        proof_widget = QWidget(self)
        proof_widget.setLayout(proof_row)
        form.addRow(self.tr("Proof profile:"), proof_widget)

        self.combo_intent = QComboBox(self)
        self.combo_intent.addItem(self.tr("Relative Colorimetric"), RENDERING_INTENT_RELATIVE)
        self.combo_intent.addItem(self.tr("Perceptual"), RENDERING_INTENT_PERCEPTUAL)
        intent_idx = self.combo_intent.findData(get_softproof_rendering_intent_from_settings())
        if intent_idx < 0:
            intent_idx = self.combo_intent.findData(DEFAULT_RENDERING_INTENT)
        if intent_idx >= 0:
            self.combo_intent.setCurrentIndex(intent_idx)
        form.addRow(self.tr("Rendering intent:"), self.combo_intent)

        self.chk_bpc = QCheckBox(self.tr("Black-point compensation"), self)
        self.chk_bpc.setChecked(get_softproof_black_point_compensation_from_settings())
        form.addRow("", self.chk_bpc)

        controls = QHBoxLayout()
        self.chk_softproof = QCheckBox(self.tr("Soft proof view"), self)
        self.chk_softproof.setChecked(True)
        self.chk_display_gamut = QCheckBox(self.tr("Show display out-of-gamut areas"), self)
        self.chk_proof_gamut = QCheckBox(self.tr("Show proof out-of-gamut areas"), self)
        controls.addWidget(self.chk_softproof)
        controls.addWidget(self.chk_display_gamut)
        controls.addWidget(self.chk_proof_gamut)
        controls.addStretch(1)
        root.addLayout(controls)

        self.lbl_status = QLabel(self)
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.preview = QLabel(self)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.preview)
        root.addWidget(self.scroll, 1)

        for widget in (self.combo_display, self.combo_proof, self.combo_intent):
            widget.currentIndexChanged.connect(self._render)
        for widget in (self.chk_softproof, self.chk_display_gamut, self.chk_proof_gamut, self.chk_bpc):
            widget.toggled.connect(self._render)

        self._render()

    def _color_space_label(self, key: str) -> str:
        if key == "DisplayP3":
            return self.tr("Display P3")
        if key == "AdobeRGB":
            return self.tr("Adobe RGB (1998)")
        if key == "ProPhotoRGB":
            return self.tr("ProPhoto RGB")
        if key == "sRGB":
            return self.tr("sRGB")
        return str(key)

    def _browse_profile(self):
        start = str(_icc_search_dirs()[0]) if _icc_search_dirs() else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select ICC Profile"),
            start,
            self.tr("ICC profiles (*.icc *.icm);;All files (*)"),
        )
        if not path:
            return
        idx = self.combo_proof.findData(path)
        if idx < 0:
            self.combo_proof.addItem(Path(path).name, path)
            idx = self.combo_proof.count() - 1
        self.combo_proof.setCurrentIndex(idx)

    def _qcolorspace_from_profile(self, path: str) -> QColorSpace | None:
        try:
            with open(path, "rb") as handle:
                cs = QColorSpace.fromIccProfile(handle.read())
            if cs.isValid():
                return cs
        except Exception:
            return None
        return None

    def _source_qimage(self) -> QImage:
        rgb = (np.clip(self._source, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        img = _array_to_qimage(rgb)
        source = get_color_space_from_key(get_working_color_space_from_settings())
        if source is not None:
            img.setColorSpace(source)
        return img

    def _converted_copy(self, target: QColorSpace | None) -> QImage:
        img = self._source_qimage()
        if target is None:
            return img
        return convert_qimage_with_lcms(
            img,
            get_working_color_space_from_settings(),
            target,
            rendering_intent=self.combo_intent.currentData(),
            black_point_compensation=self.chk_bpc.isChecked(),
        )

    def _gamut_mask(self, target: QColorSpace | None) -> np.ndarray:
        if target is None:
            return np.zeros(self._source.shape[:2], dtype=bool)
        source = self._source_qimage()
        source_cs = source.colorSpace()
        if not source_cs.isValid():
            return np.zeros(self._source.shape[:2], dtype=bool)
        roundtrip = source.copy()
        try:
            roundtrip = convert_qimage_with_lcms(
                roundtrip,
                get_working_color_space_from_settings(),
                target,
                rendering_intent=self.combo_intent.currentData(),
                black_point_compensation=self.chk_bpc.isChecked(),
            )
            roundtrip = convert_qimage_with_lcms(
                roundtrip,
                target,
                source_cs,
                rendering_intent=self.combo_intent.currentData(),
                black_point_compensation=self.chk_bpc.isChecked(),
            )
        except Exception:
            return np.zeros(self._source.shape[:2], dtype=bool)
        before = _qimage_to_array(source).astype(np.int16)
        after = _qimage_to_array(roundtrip).astype(np.int16)
        return np.max(np.abs(before - after), axis=2) > 3

    def _render(self):
        display_key = self.combo_display.currentData()
        display_cs = get_color_space_from_key(display_key)
        proof_path = self.combo_proof.currentData()
        proof_cs = self._qcolorspace_from_profile(proof_path) if proof_path else None

        if self.chk_softproof.isChecked() and proof_cs is not None:
            img = convert_qimage_with_lcms(
                self._source_qimage(),
                get_working_color_space_from_settings(),
                proof_cs,
                rendering_intent=self.combo_intent.currentData(),
                black_point_compensation=self.chk_bpc.isChecked(),
            )
            if display_cs is not None:
                img = convert_qimage_with_lcms(
                    img,
                    proof_cs,
                    display_cs,
                    rendering_intent=self.combo_intent.currentData(),
                    black_point_compensation=self.chk_bpc.isChecked(),
                )
        else:
            img = self._converted_copy(display_cs)

        set_softproof_rendering_intent_to_settings(self.combo_intent.currentData())
        set_softproof_black_point_compensation_to_settings(self.chk_bpc.isChecked())

        rgb = _qimage_to_array(img)
        if self.chk_display_gamut.isChecked():
            rgb[self._gamut_mask(display_cs)] = np.array([255, 0, 0], dtype=np.uint8)
        if self.chk_proof_gamut.isChecked() and proof_cs is not None:
            rgb[self._gamut_mask(proof_cs)] = np.array([0, 80, 255], dtype=np.uint8)

        qimg = _array_to_qimage(rgb)
        pm = QPixmap.fromImage(qimg)
        if not pm.isNull():
            max_size = QSize(max(200, self.scroll.viewport().width() - 16), max(200, self.scroll.viewport().height() - 16))
            self.preview.setPixmap(pm.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        if proof_path:
            self.lbl_status.setText(self.tr("Proof profile: {profile}").format(profile=Path(str(proof_path)).name))
        else:
            self.lbl_status.setText(self.tr("No proof profile selected."))