#!/usr/bin/env python3
"""
Export requirements.txt from Poetry's pyproject.toml and poetry.lock.

This script generates requirements.txt for backward compatibility with users
who prefer pip install -r requirements.txt over Poetry.

Usage:
    poetry run python ops/export_requirements.py
    # or
    python ops/export_requirements.py
"""

import subprocess
import sys
from pathlib import Path


def export_requirements():
    """Export requirements.txt from Poetry."""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"

    print("📦 Exporting requirements.txt from Poetry...")

    # Check if poetry is available
    try:
        result = subprocess.run(
            ["poetry", "--version"], capture_output=True, text=True, check=True
        )
        print(f"✓ Found {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Poetry not found. Please install Poetry first.")
        print("   Visit: https://python-poetry.org/docs/#installation")
        sys.exit(1)

    # Check if poetry.lock exists
    lock_file = project_root / "poetry.lock"
    if not lock_file.exists():
        print(
            "⚠️  Warning: poetry.lock not found. Run 'poetry lock' first to generate it."
        )
        print(
            "   Generating requirements.txt from pyproject.toml (may be less precise)..."
        )

    # Export requirements.txt
    # --without-hashes: Don't include hashes (simpler for users)
    # --without dev: Exclude development dependencies
    # -o: Output file
    cmd = [
        "poetry",
        "export",
        "-f",
        "requirements.txt",
        "--without-hashes",
        "--without",
        "dev",
        "-o",
        str(requirements_file),
    ]

    try:
        result = subprocess.run(
            cmd, cwd=project_root, capture_output=True, text=True, check=True
        )

        print(f"✅ Successfully exported requirements.txt")
        print(f"   Location: {requirements_file}")

        # Show file size
        if requirements_file.exists():
            size = requirements_file.stat().st_size
            line_count = len(requirements_file.read_text().splitlines())
            print(f"   Size: {size} bytes, {line_count} lines")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"❌ Error exporting requirements.txt:")
        print(f"   Command: {' '.join(cmd)}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        if e.stdout:
            print(f"   Output: {e.stdout}")

        # Check if export plugin is needed
        error_msg = (e.stderr or "").lower()
        if (
            "export" in error_msg
            or "unknown command" in error_msg
            or "not found" in error_msg
        ):
            print("\n💡 Tip: Poetry's export command may require the export plugin.")
            print("   Install it with: poetry self add poetry-plugin-export")
            print("   Or use Poetry 1.2+ which includes export by default.")
            print("   For Poetry 1.1.x, you may need: pip install poetry-plugin-export")

        return 1


if __name__ == "__main__":
    sys.exit(export_requirements())
