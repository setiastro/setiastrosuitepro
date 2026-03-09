from PyQt6.QtWidgets import QMainWindow, QWidget

class DockHostWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seti Astro Suite Pro Panels")
        self.resize(900, 700)

        center = QWidget(self)
        self.setCentralWidget(center)