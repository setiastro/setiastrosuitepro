# -*- coding: utf-8 -*-
"""
Common UI strings translation module.

This module provides centralized translation for common button labels, 
dialog titles, and messages used throughout the application.
"""
from PyQt6.QtCore import QCoreApplication

def _tr(text: str, context: str = "Common") -> str:
    """Translate common UI strings."""
    return QCoreApplication.translate(context, text)

# ============================================================================
# Common Button Labels
# ============================================================================
class Buttons:
    @staticmethod
    def apply(): return _tr("Apply")
    @staticmethod
    def cancel(): return _tr("Cancel")
    @staticmethod
    def close(): return _tr("Close")
    @staticmethod
    def ok(): return _tr("OK")
    @staticmethod
    def yes(): return _tr("Yes")
    @staticmethod
    def no(): return _tr("No")
    @staticmethod
    def save(): return _tr("Save")
    @staticmethod
    def save_as(): return _tr("Save As...")
    @staticmethod
    def open(): return _tr("Open...")
    @staticmethod
    def browse(): return _tr("Browse...")
    @staticmethod
    def reset(): return _tr("Reset")
    @staticmethod
    def clear(): return _tr("Clear")
    @staticmethod
    def delete(): return _tr("Delete")
    @staticmethod
    def add(): return _tr("Add")
    @staticmethod
    def remove(): return _tr("Remove")
    @staticmethod
    def preview(): return _tr("Preview")
    @staticmethod
    def apply_to_document(): return _tr("Apply to Document")
    @staticmethod
    def zoom_in(): return _tr("Zoom In")
    @staticmethod
    def zoom_out(): return _tr("Zoom Out")
    @staticmethod
    def fit_to_preview(): return _tr("Fit to Preview")
    @staticmethod
    def show_original(): return _tr("Show Original")
    @staticmethod
    def run(): return _tr("Run")
    @staticmethod
    def stop(): return _tr("Stop")
    @staticmethod
    def export(): return _tr("Export")
    @staticmethod
    def import_(): return _tr("Import")

# ============================================================================
# Common Dialog Titles
# ============================================================================
class Titles:
    @staticmethod
    def error(): return _tr("Error")
    @staticmethod
    def warning(): return _tr("Warning")
    @staticmethod
    def info(): return _tr("Information")
    @staticmethod
    def confirm(): return _tr("Confirm")
    @staticmethod
    def success(): return _tr("Success")
    @staticmethod
    def processing(): return _tr("Processing...")
    @staticmethod
    def please_wait(): return _tr("Please wait")

# ============================================================================
# Common Messages
# ============================================================================
class Messages:
    @staticmethod
    def no_image(): return _tr("No image loaded.")
    @staticmethod
    def no_active_image(): return _tr("No active image.")
    @staticmethod
    def load_image_first(): return _tr("Load an image first.")
    @staticmethod
    def processing_failed(): return _tr("Processing failed.")
    @staticmethod
    def operation_complete(): return _tr("Operation complete.")
    @staticmethod
    def are_you_sure(): return _tr("Are you sure?")
    @staticmethod
    def unsaved_changes(): return _tr("You have unsaved changes.")
    @staticmethod
    def exit_confirm(): return _tr("Do you really want to exit?")
