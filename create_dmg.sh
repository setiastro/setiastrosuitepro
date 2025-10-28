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
echo "🚀 Creating DMG for ${PROJECT_NAME}"

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

# Create DMG staging directory
STAGING_DIR="dmg_staging"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# Copy the app to staging
echo "📦 Preparing DMG contents..."
cp -R "dist/${APP_NAME}" "${STAGING_DIR}/"  # -R for recursive copy
#cp "dist/${APP_NAME}" "${STAGING_DIR}/"

# Create an alias to Applications folder
ln -s /Applications "${STAGING_DIR}/Applications"

# Copy additional files if they exist
if [ -f "README.md" ]; then
    cp "README.md" "${STAGING_DIR}/"
fi

if [ -f "LICENSE" ]; then
    cp "LICENSE" "${STAGING_DIR}/"
fi

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