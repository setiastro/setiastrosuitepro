from PyQt6.QtCore import QSettings, QCoreApplication
import sys

app = QCoreApplication(sys.argv)
settings = QSettings("SetiAstro", "SetiAstroSuitePro")
lang = settings.value("ui/language", "en")
print(f"Current language in settings: {lang}")
