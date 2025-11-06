#!/bin/bash

# SetiAstroSuitePro DMG Creation Script
set -e  # Exit on any error

PROJECT_NAME="SetiAstroSuitePro"
APP_NAME="SetiAstroSuitePro.app"

# Architecture detection and naming
ARCH=$(uname -m)
case "$ARCH" in
    "arm64")
        ARCH_SUFFIX="AppleSilicon"
        ARCH_DISPLAY="Apple Silicon"
        ;;
    "x86_64")
        ARCH_SUFFIX="Intel"
        ARCH_DISPLAY="Intel"
        ;;
    *)
        ARCH_SUFFIX="_${ARCH}"
        ARCH_DISPLAY="$ARCH"
        ;;
esac

DMG_NAME="${PROJECT_NAME}_${ARCH_SUFFIX}"
echo "🚀 Creating DMG for ${PROJECT_NAME} (${ARCH_DISPLAY})"

echo "🔧 Generating build info..."
python3 -c "
import os
from datetime import datetime
from pathlib import Path

gen_dir = Path('pro/_generated')
gen_dir.mkdir(exist_ok=True)

# __init__.py
(gen_dir / '__init__.py').write_text('# Generated\\n')

# build_info.py
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
content = f'BUILD_TIMESTAMP = \"{timestamp}\"\\n'
(gen_dir / 'build_info.py').write_text(content)
print(f'Build timestamp: {timestamp}')
"

# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib pre-warm (font cache + backend import) BEFORE build
# This avoids first-run stalls on user machines.
# Also optionally call your helper if present (e.g., scripts/matplotlib.py).
# ──────────────────────────────────────────────────────────────────────────────
echo "🎨 Pre-warming Matplotlib cache (headless)…"
python3 - <<'PY'
import os, sys
# Force non-GUI backend to avoid any Qt issues during cache build
import matplotlib
matplotlib.use("Agg", force=True)

# Trigger font manager cache build
from matplotlib import font_manager
# Newer MPL uses this; for older, this still refreshes the cache files.
font_manager.get_font_names()  # touches cache
try:
    # Force rebuild path if needed
    font_manager._load_fontmanager(try_read_cache=False)
except Exception:
    pass

# Touch pyplot once to ensure everything is importable
from matplotlib import pyplot as plt
plt.figure(); plt.plot([0,1],[0,1]); plt.close()

print("Matplotlib cache pre-warmed ✅")
PY

# If you keep a helper to bundle/patch MPL at build-time, call it here:
# e.g. scripts/matplotlib.py (optional)
if [ -f "scripts/matplotlib.py" ]; then
  echo "🧩 Running project Matplotlib helper…"
  python3 scripts/matplotlib.py || echo "ℹ️ matplotlib.py helper returned non-zero; continuing."
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/
rm -f "${DMG_NAME}.dmg"

# Build the app
echo "🔨 Building application..."
pyinstaller --clean --noconfirm setiastrosuitepro_mac.spec

# Verify the app was built
if [ ! -d "dist/${APP_NAME}" ]; then  # -d for directory, not -f for file
    echo "❌ Build failed - app bundle not found!"
    exit 1
fi

echo "✅ Build successful!"

# (Optional) sanity: ensure mpl-data exists in bundle (PyInstaller usually includes it)
# If you want to be extra safe, uncomment this block to copy mpl-data explicitly.
#: <<'MPL_COPY'
#echo "🔎 Ensuring mpl-data is present in app bundle…"
#MPL_DATA=$(python3 -c "import matplotlib, sys; print(matplotlib.get_data_path())" 2>/dev/null || true)
#if [ -n "$MPL_DATA" ] && [ -d "$MPL_DATA" ]; then
#  APP_RESOURCES="dist/${APP_NAME}/Contents/Resources"
#  mkdir -p "${APP_RESOURCES}/mpl-data"
#  rsync -a --delete "${MPL_DATA}/" "${APP_RESOURCES}/mpl-data/"
#  echo "📦 mpl-data copied into app Resources."
#else
#  echo "ℹ️ Could not resolve matplotlib data path; skipping explicit copy."
#fi
#MPL_COPY

# Create DMG staging directory
STAGING_DIR="dmg_staging"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# Copy the app to staging
echo "📦 Preparing DMG contents..."
cp -R "dist/${APP_NAME}" "${STAGING_DIR}/"

# Create an alias to Applications folder
ln -s /Applications "${STAGING_DIR}/Applications"

# Copy additional files if they exist
[ -f "README.md" ] && cp "README.md" "${STAGING_DIR}/"
[ -f "LICENSE" ]  && cp "LICENSE"  "${STAGING_DIR}/"

# Create DMG using hdiutil
echo "💿 Creating DMG..."
hdiutil create -volname "${PROJECT_NAME}" \
    -srcfolder "${STAGING_DIR}" \
    -ov -format UDZO \
    -imagekey zlib-level=9 \
    "${DMG_NAME}.dmg"

# Clean up staging
rm -rf "${STAGING_DIR}"

echo "✅ DMG created successfully: ${DMG_NAME}.dmg"

# Show DMG info
echo "📊 DMG Information:"
ls -lh "${DMG_NAME}.dmg"
hdiutil imageinfo "${DMG_NAME}.dmg" | grep -E "(Format|Size|Compressed|Checksum)"

echo "🎉 Done! Your DMG is ready for distribution."
