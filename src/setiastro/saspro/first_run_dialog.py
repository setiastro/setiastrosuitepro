# ============================================================================
# setiastro/saspro/first_run_dialog.py
#
# First-time user welcome dialog for Seti Astro Suite Pro.
# Shows once after the main window is fully built, after the splash fades out.
# Suppressed permanently via QSettings when "Don't show this again" is checked.
#
# Wiring (gui_entry.py _kick_updates_after_splash):
#
#   from setiastro.saspro.first_run_dialog import maybe_show_first_run_dialog
#   QTimer.singleShot(200, lambda: maybe_show_first_run_dialog(win))
#
# ============================================================================

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFrame, QScrollArea, QWidget, QSizePolicy,
)
from PyQt6.QtCore import Qt, QSettings, QUrl, QSize
from PyQt6.QtGui import QDesktopServices, QPixmap, QIcon


# ── Settings key ─────────────────────────────────────────────────────────────
_SETTINGS_KEY = "ui/first_run_dialog_shown_v1"


def _is_first_run() -> bool:
    return not QSettings().value(_SETTINGS_KEY, False, type=bool)


def _mark_shown():
    s = QSettings()
    s.setValue(_SETTINGS_KEY, True)
    s.sync()


def maybe_show_first_run_dialog(main_window) -> None:
    """Call this after the splash is gone. Shows the dialog only on first run."""
    if not _is_first_run():
        return
    dlg = FirstRunDialog(main_window)
    dlg.exec()


# ── Dialog ───────────────────────────────────────────────────────────────────

class FirstRunDialog(QDialog):

    _STYLESHEET = """
    /* ── Dialog shell ──────────────────────────────────────────────────── */
    FirstRunDialog {
        background-color: #13131f;
    }

    /* ── Header band ────────────────────────────────────────────────────── */
    #header_band {
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:1,
            stop:0 #0e0e20,
            stop:1 #1c1c38
        );
        border-bottom: 2px solid #2e2e5a;
    }

    #lbl_title {
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
    }

    #lbl_subtitle {
        color: #8888bb;
        font-size: 11px;
        letter-spacing: 1.8px;
    }

    /* ── Scroll / content ───────────────────────────────────────────────── */
    #scroll_area, #scroll_content {
        background-color: #13131f;
        border: none;
    }
    QScrollBar:vertical {
        background: #1a1a2e;
        width: 8px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #3a3a6a;
        border-radius: 4px;
        min-height: 20px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    /* ── Section cards ──────────────────────────────────────────────────── */
    #card {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
    }

    #card_title {
        color: #7788cc;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2.5px;
    }

    /* ── Step number badge ──────────────────────────────────────────────── */
    #step_bullet {
        background-color: #252550;
        color: #8899ee;
        font-size: 12px;
        font-weight: 700;
        border-radius: 12px;
        min-width: 24px;
        max-width: 24px;
        min-height: 24px;
        max-height: 24px;
        qproperty-alignment: AlignCenter;
    }

    /* ── Step title ─────────────────────────────────────────────────────── */
    #step_text {
        color: #dde0ff;
        font-size: 13px;
        font-weight: 600;
    }

    /* ── Step note / description ────────────────────────────────────────── */
    #step_note {
        color: #aaaacc;
        font-size: 12px;
    }

    /* ── Link / action buttons ──────────────────────────────────────────── */
    #link_btn {
        background-color: #1e1e40;
        color: #88aaff;
        border: 1px solid #3a3a70;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 600;
        text-align: left;
    }
    #link_btn:hover {
        background-color: #28285a;
        color: #aaccff;
        border-color: #5566aa;
    }
    #link_btn:pressed {
        background-color: #333370;
    }

    /* ── Primary button (Open Preferences) ─────────────────────────────── */
    #primary_btn {
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 #3344bb,
            stop:1 #5566dd
        );
        color: #ffffff;
        border: none;
        border-radius: 7px;
        padding: 9px 22px;
        font-size: 13px;
        font-weight: 700;
    }
    #primary_btn:hover {
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 #4455cc,
            stop:1 #6677ee
        );
    }
    #primary_btn:pressed {
        background: #2233aa;
    }

    /* ── Footer ─────────────────────────────────────────────────────────── */
    #footer_strip {
        background-color: #0e0e1c;
        border-top: 1px solid #222240;
    }

    /* ── Never-show checkbox ────────────────────────────────────────────── */
    #chk_never {
        color: #6666aa;
        font-size: 11px;
        spacing: 7px;
    }
    #chk_never::indicator {
        width: 15px;
        height: 15px;
        border: 1px solid #3a3a66;
        border-radius: 3px;
        background: #13131f;
    }
    #chk_never::indicator:checked {
        background: #4455bb;
        border-color: #6677cc;
    }

    /* ── Close button ───────────────────────────────────────────────────── */
    #close_btn {
        background-color: transparent;
        color: #666699;
        border: 1px solid #2a2a4a;
        border-radius: 6px;
        padding: 9px 20px;
        font-size: 12px;
    }
    #close_btn:hover {
        background-color: #1a1a30;
        color: #9999cc;
        border-color: #4444aa;
    }

    /* ── Horizontal rule ────────────────────────────────────────────────── */
    #h_sep {
        background-color: #252545;
        max-height: 1px;
        border: none;
    }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FirstRunDialog")
        self.setWindowTitle("Welcome to Seti Astro Suite Pro")
        self.setModal(True)
        self.setMinimumWidth(680)
        self.setMaximumWidth(820)
        self.setMinimumHeight(520)
        self.setStyleSheet(self._STYLESHEET)

        flags = self.windowFlags()
        flags &= ~Qt.WindowType.WindowContextHelpButtonHint
        self.setWindowFlags(flags)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────────
        root.addWidget(self._build_header(parent))

        # ── Scrollable content ───────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setObjectName("scroll_area")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("scroll_content")
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(20, 18, 20, 18)
        content_lay.setSpacing(14)

        self._add_card(
            content_lay,
            section_label="THINGS TO DO FIRST",
            steps=[
                {
                    "text": "Install Python 3.12",
                    "note": "Python 3.13 and 3.14 are NOT supported. Hardware acceleration requires Python 3.12 specifically.",
                    "link": ("⬇  Download Python 3.12", "https://www.python.org/downloads/release/python-3128/"),
                },
                {
                    "text": "Open Preferences to configure SASpro",
                    "note": "Set the file paths for GraXpert, ASTAP, Starnet++, and your Astrometry.net API key so all tools work correctly.",
                    "prefs_btn": True,
                },
                {
                    "text": "Install Hardware Acceleration",
                    "note": "This is not optional — hardware acceleration powers a large portion of SASpro, "
                        "including Cosmic Clarity, Star Removal, Denoising, Stacking, Drizzle Integration, "
                        "Multi-Scale Decomposition, and many other core tools. Without it, much of the "
                        "application will not function correctly. Find this under "
                        "Preferences → Acceleration → Install/Repair.",
                },
                {
                    "text": "Download the latest AI Models",
                    "note": "AI models are required for Cosmic Clarity. Find this under Preferences → AI Models → Download/Update Models.",
                },
                {
                    "text": "Choose a Theme",
                    "note": "Pick from Dark, Gray, Light, System, or Custom under Preferences → Theme.",
                },
            ],
        )

        self._add_card(
            content_lay,
            section_label="LEARNING RESOURCES",
            steps=[
                {
                    "text": "New to astrophotography processing?",
                    "note": "Watch the beginner's walkthrough to get comfortable with the core tools and workflow.",
                    "link": ("▶  Watch the Beginner Tutorial on YouTube", "https://youtu.be/zBUOZl2OgaY"),
                },
                {
                    "text": "Prefer reading over video?",
                    "note": "The SASpro Wiki has detailed documentation on a lot of tools in the suite.",
                    "link": ("📖  Open the SASpro Wiki", "https://github.com/setiastro/setiastrosuitepro/wiki"),
                },
                {
                    "text": "Join the community",
                    "note": "Get help from other users, share your images, and stay up to date with new features.",
                    "link": ("💬  Join the Discord Server", "https://discord.gg/vvYH82C82f"),
                },
            ],
        )

        content_lay.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # ── Footer ───────────────────────────────────────────────────────────
        footer = QWidget()
        footer.setObjectName("footer_strip")
        footer_lay = QHBoxLayout(footer)
        footer_lay.setContentsMargins(20, 12, 20, 12)
        footer_lay.setSpacing(10)

        self.chk_never = QCheckBox("Don't show this again")
        self.chk_never.setObjectName("chk_never")

        btn_prefs = QPushButton("⚙  Open Preferences")
        btn_prefs.setObjectName("primary_btn")
        btn_prefs.setFixedHeight(38)
        btn_prefs.clicked.connect(self._open_preferences)

        btn_close = QPushButton("Close")
        btn_close.setObjectName("close_btn")
        btn_close.setFixedHeight(38)
        btn_close.clicked.connect(self._on_close)

        footer_lay.addWidget(self.chk_never)
        footer_lay.addStretch(1)
        footer_lay.addWidget(btn_prefs)
        footer_lay.addWidget(btn_close)

        root.addWidget(footer)

    # ── Header builder ────────────────────────────────────────────────────────

    def _build_header(self, parent) -> QWidget:
        header = QWidget()
        header.setObjectName("header_band")
        header.setFixedHeight(88)

        lay = QHBoxLayout(header)
        lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(16)

        # Logo
        logo_pix = self._load_logo(parent)
        if logo_pix and not logo_pix.isNull():
            lbl_logo = QLabel()
            lbl_logo.setPixmap(logo_pix)
            lbl_logo.setFixedSize(58, 58)
            lbl_logo.setScaledContents(True)
            lbl_logo.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
            lay.addWidget(lbl_logo)

        # Title + subtitle
        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        text_col.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        lbl_title = QLabel("Welcome to Seti Astro Suite Pro")
        lbl_title.setObjectName("lbl_title")

        lbl_sub = QLabel("FIRST TIME SETUP  ·  LET'S GET YOU STARTED")
        lbl_sub.setObjectName("lbl_subtitle")

        text_col.addWidget(lbl_title)
        text_col.addWidget(lbl_sub)
        lay.addLayout(text_col, 1)

        return header

    def _load_logo(self, parent) -> QPixmap | None:
        """Try several sources for the app logo pixmap, 58×58."""
        # 1. Parent window's app_icon attribute
        try:
            icon = getattr(parent, "app_icon", None)
            if icon and not icon.isNull():
                pm = icon.pixmap(QSize(58, 58))
                if not pm.isNull():
                    return pm
        except Exception:
            pass

        # 2. Resources module — prefer PNG sibling for clean rendering
        try:
            import os
            from setiastro.saspro.resources import icon_path
            if icon_path:
                root, _ext = os.path.splitext(icon_path)
                png_try = root + ".png"
                path = png_try if os.path.exists(png_try) else icon_path
                pm = QPixmap(path)
                if not pm.isNull():
                    return pm.scaled(
                        58, 58,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
        except Exception:
            pass

        # 3. QApplication window icon fallback
        try:
            from PyQt6.QtWidgets import QApplication
            icon = QApplication.windowIcon()
            if icon and not icon.isNull():
                pm = icon.pixmap(QSize(58, 58))
                if not pm.isNull():
                    return pm
        except Exception:
            pass

        return None

    # ── Card builder ──────────────────────────────────────────────────────────

    def _add_card(self, parent_layout: QVBoxLayout, section_label: str, steps: list):
        card = QFrame()
        card.setObjectName("card")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(18, 14, 18, 16)
        card_lay.setSpacing(0)

        lbl_sec = QLabel(section_label)
        lbl_sec.setObjectName("card_title")
        card_lay.addWidget(lbl_sec)
        card_lay.addSpacing(8)

        sep = QFrame()
        sep.setObjectName("h_sep")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        card_lay.addWidget(sep)
        card_lay.addSpacing(12)

        for i, step in enumerate(steps):
            card_lay.addLayout(self._build_step(step, i + 1))
            if i < len(steps) - 1:
                card_lay.addSpacing(12)
                divider = QFrame()
                divider.setObjectName("h_sep")
                divider.setFrameShape(QFrame.Shape.HLine)
                divider.setFixedHeight(1)
                card_lay.addWidget(divider)
                card_lay.addSpacing(12)

        parent_layout.addWidget(card)

    def _build_step(self, step: dict, number: int) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)
        row.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Numbered badge, pinned to top of its column
        bullet = QLabel(str(number))
        bullet.setObjectName("step_bullet")
        bullet.setFixedSize(24, 24)
        bullet.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bullet_wrap = QVBoxLayout()
        bullet_wrap.setContentsMargins(0, 1, 0, 0)
        bullet_wrap.setSpacing(0)
        bullet_wrap.addWidget(bullet)
        bullet_wrap.addStretch(1)
        row.addLayout(bullet_wrap)

        # Text + note + optional button
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(5)

        lbl_text = QLabel(step["text"])
        lbl_text.setObjectName("step_text")
        lbl_text.setWordWrap(True)
        lbl_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        col.addWidget(lbl_text)

        if step.get("note"):
            lbl_note = QLabel(step["note"])
            lbl_note.setObjectName("step_note")
            lbl_note.setWordWrap(True)
            lbl_note.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            col.addWidget(lbl_note)

        if step.get("link"):
            label_text, url = step["link"]
            btn = QPushButton(label_text)
            btn.setObjectName("link_btn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
            col.addSpacing(3)
            col.addWidget(btn)

        if step.get("prefs_btn"):
            btn = QPushButton("⚙  Open Preferences")
            btn.setObjectName("link_btn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(self._open_preferences)
            col.addSpacing(3)
            col.addWidget(btn)

        row.addLayout(col, 1)
        return row

    # ── Actions ───────────────────────────────────────────────────────────────

    def _open_preferences(self):
        parent = self.parent()
        if parent and hasattr(parent, "_open_settings"):
            try:
                parent._open_settings()
            except Exception:
                pass

    def _on_close(self):
        if self.chk_never.isChecked():
            _mark_shown()
        self.accept()

    def closeEvent(self, event):
        if self.chk_never.isChecked():
            _mark_shown()
        super().closeEvent(event)