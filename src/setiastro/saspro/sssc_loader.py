# saspro/sssc_loader.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
#
# Splash screen + background loader for the SSSC dialog.
# Mirrors the wimi_loader.py pattern exactly.

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QApplication
from PyQt6.QtGui import QPixmap, QFont


class SSSCSplash(QWidget):
    """Lightweight splash shown while SSSC imports and initialises."""

    def __init__(self, parent=None, sssc_path: str = ""):
        super().__init__(parent, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(460, 230)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        # Icon
        logo_label = QLabel()
        if sssc_path:
            pm = QPixmap(sssc_path).scaled(
                64, 64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(pm)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        # Title
        title = QLabel("Spectrophotometric Standard Star Calibration")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        subtitle = QLabel("SSSC — Empirical system response from Gaia XP spectra")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(subtitle)

        self._status = QLabel("Initialising…")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self._status)

        bar = QProgressBar()
        bar.setRange(0, 0)       # indeterminate / marquee
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        layout.addWidget(bar)

        self.setStyleSheet("""
            SSSCSplash {
                background: #1e1e2e;
                border: 1px solid #44475a;
                border-radius: 8px;
            }
            QLabel { color: #cdd6f4; }
            QProgressBar {
                background: #313244;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: #89b4fa;
                border-radius: 3px;
            }
        """)

        # Centre on parent or screen
        if parent:
            geo = parent.geometry()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )
        else:
            from PyQt6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen().geometry()
            self.move(
                screen.center().x() - self.width() // 2,
                screen.center().y() - self.height() // 2,
            )

    def set_status(self, msg: str):
        self._status.setText(msg)
        QApplication.processEvents()


class SSSCLoader(QThread):
    """
    Does the heavy imports off the GUI thread.
    Emits ready() on success, error(msg) on failure.
    progress(str) emitted for status updates.
    """
    progress = pyqtSignal(str)
    ready    = pyqtSignal()
    error    = pyqtSignal(str)

    def run(self):
        try:
            self.progress.emit("Loading astronomy modules…")
            import numpy          # noqa
            import scipy          # noqa
            import sep            # noqa
            from astropy.io import fits    # noqa
            from astropy.wcs import WCS   # noqa
            from astroquery.simbad import Simbad  # noqa

            self.progress.emit("Loading Gaia modules…")
            from setiastro.saspro.gaia_downloader import GaiaDownloader  # noqa
            from setiastro.saspro.gaia_database import get_library        # noqa

            self.progress.emit("Loading SSSC…")
            from setiastro.saspro.sssc import SSSCDialog  # noqa — warms the cache

            self.progress.emit("Almost ready…")
            self.ready.emit()

        except Exception as exc:
            self.error.emit(str(exc))