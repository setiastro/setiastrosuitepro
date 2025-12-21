# pro/log_bus.py
from PyQt6.QtCore import QObject, pyqtSignal

class LogBus(QObject):
    posted = pyqtSignal(str)  # emitted from any thread; connect with QueuedConnection
