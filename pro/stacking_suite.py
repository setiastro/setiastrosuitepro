# pro/stacking_suite.py
"""
Compatibility wrapper for the refactored Stacking Suite.

The actual implementation is now in pro/stacking/ package.
This file exists for backward compatibility with existing imports.
"""

from pro.stacking import StackingSuiteDialog

__all__ = ["StackingSuiteDialog"]