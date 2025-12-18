# pro/config.py
import os
import sys
from PyQt6.QtCore import QStandardPaths

class Config:
    """Central configuration for Seti Astro Suite Pro."""
    
    # GitHub Repos
    GITHUB_ABERRATION_REPO = "riccardoalberghi/abberation_models"
    
    # Paths
    @staticmethod
    def get_app_data_dir() -> str:
        """Returns the application data directory."""
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        if not base:
            base = os.path.expanduser("~/.local/share/SetiAstro")
        return base

    @staticmethod
    def get_models_dir() -> str:
        """Returns the directory for storing AI models."""
        return os.path.join(Config.get_app_data_dir(), "Models")

    @staticmethod
    def get_aberration_models_dir() -> str:
        """Returns the directory for Aberration AI models."""
        return os.path.join(Config.get_models_dir(), "aberration_ai")

    @staticmethod
    def get_graxpert_default_path() -> str | None:
        """Returns the default GraXpert executable path based on OS."""
        if sys.platform == "darwin":
            return "/Applications/GraXpert.app/Contents/MacOS/GraXpert"
        elif sys.platform == "win32":
            return "GraXpert.exe"
        return None
