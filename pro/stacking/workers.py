# Stacking module - imports from functions.py for shared utilities
from __future__ import annotations
import os
import sys
import math
import time
import numpy as np
import cv2
cv2.setNumThreads(0)

from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal, QObject, pyqtSlot, QThread, QEvent, QPoint, QSize, QEventLoop, QCoreApplication, QRectF, QPointF, QMetaObject
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction, QIntValidator, QDoubleValidator, QFontMetrics, QTextCursor, QPalette, QPainter, QPen, QTransform, QColor, QBrush, QCursor
from PyQt6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeWidget, QHeaderView, QTreeWidgetItem, QProgressBar, QProgressDialog,
                             QFormLayout, QDialogButtonBox, QToolBar, QToolButton, QFileDialog, QTabWidget, QAbstractItemView, QSpinBox, QDoubleSpinBox, QGroupBox, QRadioButton,
                             QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication, QScrollArea, QTextEdit, QMenu, QPlainTextEdit, QGraphicsEllipseItem,
                             QMessageBox, QSlider, QCheckBox, QInputDialog, QComboBox)

from astropy.io import fits
from datetime import datetime

# Import shared utilities from functions.py
from .functions import (
    _asarray, _WINDOWS_RESERVED, _FITS_EXTS,
    get_valid_header, LRUDict, load_image, save_image,
    _torch_ok, _gpu_algo_supported, _torch_reduce_tile,
    windsorized_sigma_clip_weighted, kappa_sigma_clip_weighted,
    debayer_raw_fast, drizzle_deposit_numba_kernel_mono, drizzle_deposit_color_kernel,
    finalize_drizzle_2d, finalize_drizzle_3d,
    bulk_cosmetic_correction_numba, drizzle_deposit_numba_naive, drizzle_deposit_color_naive,
    bulk_cosmetic_correction_bayer,
    compute_star_count_fast_preview, siril_style_autostretch,
)
from .functions import *


class AfterAlignWorker(QObject):
    progress = pyqtSignal(str)                 # emits status lines
    finished = pyqtSignal(bool, str)           # (success, message)
    need_comet_review = pyqtSignal(list, dict, object)  # (files, initial_xy, responder)

    def __init__(self, dialog, *,
                 light_files,
                 frame_weights,
                 transforms_dict,
                 drizzle_dict,
                 autocrop_enabled,
                 autocrop_pct, ui_owner=None):
        super().__init__()
        self.dialog = dialog                    # we will call pure methods on it
        self.light_files = light_files
        self.frame_weights = frame_weights
        self.transforms_dict = transforms_dict
        self.drizzle_dict = drizzle_dict
        self.autocrop_enabled = autocrop_enabled
        self.autocrop_pct = autocrop_pct
        self.ui_owner         = ui_owner  

    @pyqtSlot()
    def run(self):
        dlg = self.dialog  # the StackingSuiteDialog you passed in

        try:
            result = dlg.stack_images_mixed_drizzle(
                grouped_files=self.light_files,
                frame_weights=self.frame_weights,
                transforms_dict=self.transforms_dict,
                drizzle_dict=self.drizzle_dict,
                autocrop_enabled=self.autocrop_enabled,
                autocrop_pct=self.autocrop_pct,
                status_cb=self.progress.emit,   # stream status back to UI
            )
            summary = "\n".join(result["summary_lines"])
            self.finished.emit(True, f"Post-alignment complete.\n\n{summary}")
        except Exception as e:
            self.finished.emit(False, f"Post-alignment failed: {e}")

