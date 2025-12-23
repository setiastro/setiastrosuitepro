from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QFormLayout, QPushButton
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QIcon

class StatisticsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("App Statistics"))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self.resize(300, 200)

        # Settings to read stats
        self.settings = QSettings("SetiAstro", "SetiAstroSuitePro")

        layout = QVBoxLayout(self)

        form_layout = QFormLayout()
        
        # Time Spent
        total_seconds = self.settings.value("stats/total_time_seconds", 0, type=float)
        days = int(total_seconds // 86400)
        hours = int((total_seconds % 86400) // 3600)
        minutes = int((total_seconds % 3600) // 60)
        
        time_str = f"{days} {self.tr('Days')}, {hours} {self.tr('Hours')}, {minutes} {self.tr('Minutes')}"
        if days == 0:
             time_str = f"{hours} {self.tr('Hours')}, {minutes} {self.tr('Minutes')}"
        
        self.lbl_time = QLabel(time_str)
        form_layout.addRow(self.tr("Time Spent:"), self.lbl_time)

        # Images Opened
        images_count = self.settings.value("stats/opened_images_count", 0, type=int)
        self.lbl_images = QLabel(str(images_count))
        form_layout.addRow(self.tr("Images Opened:"), self.lbl_images)

        # Tools Opened
        tools_count = self.settings.value("stats/opened_tools_count", 0, type=int)
        self.lbl_tools = QLabel(str(tools_count))
        form_layout.addRow(self.tr("Tools Opened:"), self.lbl_tools)

        layout.addLayout(form_layout)

        # Close button
        btn_close = QPushButton(self.tr("Close"))
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)
