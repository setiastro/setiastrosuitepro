# -*- coding: utf-8 -*-
"""
Internationalization (i18n) module for Seti Astro Suite Pro.

Handles loading and managing translations using PyQt6's QTranslator.
"""
from __future__ import annotations

import os
from typing import Optional, Dict, List

from PyQt6.QtCore import QCoreApplication, QTranslator, QLocale, QSettings

# Module-level translator instance (kept alive for the app lifetime)
_translator: Optional[QTranslator] = None
_current_language: str = "en"

# Available languages with display names
AVAILABLE_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "it": "Italiano",
    "fr": "Français",
    "es": "Español",
    "zh": "简体中文",
    "de": "Deutsch",
    "pt": "Português",
    "ja": "日本語",
    "hi": "हिन्दी",
    "sw": "Kiswahili",
    "ua": "Українська",
    "ru": "Русский",
    "ar": "العربية",
}


def get_translations_dir() -> str:
    """Get the path to the translations directory."""
    # Source / installed package location
    module_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.join(module_dir, "translations")

    # PyInstaller frozen builds
    if hasattr(os.sys, "_MEIPASS"):
        # New bundle layout (preferred)
        frozen_internal = os.path.join(os.sys._MEIPASS, "_internal", "translations")
        if os.path.exists(frozen_internal):
            return frozen_internal

        # Legacy bundle layout fallback
        frozen_legacy = os.path.join(os.sys._MEIPASS, "translations")
        if os.path.exists(frozen_legacy):
            return frozen_legacy

    return pkg_dir



def get_available_languages() -> Dict[str, str]:
    """
    Get available languages.
    
    Returns:
        Dict mapping language codes to display names (e.g., {"en": "English"})
    """
    return AVAILABLE_LANGUAGES.copy()


def get_current_language() -> str:
    """Get the currently loaded language code."""
    return _current_language


def get_saved_language() -> str:
    """
    Get the language saved in settings.
    
    Returns:
        Language code from settings, defaults to "en"
    """
    settings = QSettings("SetiAstro", "SetiAstroSuitePro")
    return settings.value("ui/language", "en", type=str) or "en"


def save_language(lang_code: str) -> None:
    """
    Save the language preference to settings.
    
    Args:
        lang_code: Language code (e.g., "it", "fr", "es")
    """
    settings = QSettings("SetiAstro", "SetiAstroSuitePro")
    settings.setValue("ui/language", lang_code)
    settings.sync()


def load_language(lang_code: str = None, app: QCoreApplication = None) -> bool:
    """
    Load a translation for the specified language.
    
    Must be called BEFORE creating any widgets for full effect.
    
    Args:
        lang_code: Language code (e.g., "it", "fr", "es"). 
                   If None, reads from settings.
        app: QApplication instance. If None, uses QCoreApplication.instance()
    
    Returns:
        True if translation was loaded successfully, False otherwise
    """
    global _translator, _current_language
    
    if app is None:
        app = QCoreApplication.instance()
    
    if app is None:
        return False
    
    # Get language from settings if not specified
    if lang_code is None:
        lang_code = get_saved_language()
    
    # English is the base language, no translation needed
    if lang_code == "en":
        # Remove any existing translator
        if _translator is not None:
            app.removeTranslator(_translator)
            _translator = None
        _current_language = "en"
        return True
    
    # Find translation file
    translations_dir = get_translations_dir()
    qm_file = os.path.join(translations_dir, f"saspro_{lang_code}.qm")
    
    if not os.path.exists(qm_file):
        # Translation file doesn't exist yet
        _current_language = lang_code
        return False
    
    # Remove old translator if any
    if _translator is not None:
        app.removeTranslator(_translator)
    
    # Load new translator
    _translator = QTranslator()
    if _translator.load(qm_file):
        app.installTranslator(_translator)
        _current_language = lang_code
        return True
    else:
        _translator = None
        return False


def tr(text: str, context: str = "Global") -> str:
    """
    Translate a string outside of a QObject context.
    
    For use in module-level code or functions that aren't methods of QObject.
    
    Args:
        text: Text to translate
        context: Context for disambiguation (default: "Global")
    
    Returns:
        Translated string
    """
    return QCoreApplication.translate(context, text)
