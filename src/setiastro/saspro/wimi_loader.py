#saspro.wimi_loader.py

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QApplication
from PyQt6.QtGui import QPixmap, QFont

class WIMISplash(QWidget):
    """Lightweight splash shown while WIMI loads."""

    def __init__(self, parent=None, wimi_path: str = ""):
        super().__init__(parent, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(420, 220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Logo / title row
        logo_label = QLabel()
        if wimi_path:
            pm = QPixmap(wimi_path).scaled(
                64, 64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(pm)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        title = QLabel("What's In My Image")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        self._status = QLabel("Initialising…")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self._status)

        bar = QProgressBar()
        bar.setRange(0, 0)          # indeterminate / marquee
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        layout.addWidget(bar)

        self.setStyleSheet("""
            WIMISplash {
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


class WIMILoader(QThread):
    """
    Does the heavy import + WIMIDialog construction off the GUI thread.
    Emits ready(dlg) on success, error(msg) on failure.
    progress(str) can be emitted for status updates.
    """

    progress = pyqtSignal(str)
    ready    = pyqtSignal(object)   # passes the constructed WIMIDialog
    error    = pyqtSignal(str)

    def __init__(self, parent_widget, wimi_path, wrench_path, settings, doc_manager):
        super().__init__(parent_widget)
        self._parent_widget = parent_widget
        self._wimi_path     = wimi_path
        self._wrench_path   = wrench_path
        self._settings      = settings
        self._doc_manager   = doc_manager

    def run(self):
        try:
            self.progress.emit("Loading astronomy modules…")
            from setiastro.saspro.wimi import WIMIDialog   # noqa – just warm the cache
            self.progress.emit("Almost ready…")
            self.ready.emit(None)
        except Exception as exc:
            self.error.emit(str(exc))