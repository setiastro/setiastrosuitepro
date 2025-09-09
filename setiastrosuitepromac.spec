# -*- mode: python ; coding: utf-8 -*-
import sys, os
sys.setrecursionlimit(sys.getrecursionlimit() * 10)
from pathlib import Path

# ---------------------------------------------------------------------
# Project root (works whether __file__ is defined or not)
# ---------------------------------------------------------------------
try:
    SPEC_DIR = Path(__file__).resolve().parent
except NameError:
    SPEC_DIR = Path.cwd()

PROJECT_ROOT = SPEC_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# PyInstaller helpers
# ---------------------------------------------------------------------
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

def try_collect_data(pkg, **kwargs):
    try:
        return collect_data_files(pkg, **kwargs)
    except Exception as e:
        print(f"[spec] WARN: skipping data for '{pkg}': {e}")
        return []

def try_collect_submodules(pkg):
    try:
        return collect_submodules(pkg)
    except Exception as e:
        print(f"[spec] WARN: skipping submodules for '{pkg}': {e}")
        return []

def try_collect_binaries(pkg):
    try:
        return collect_dynamic_libs(pkg)
    except Exception as e:
        print(f"[spec] WARN: skipping binaries for '{pkg}': {e}")
        return []

def try_collect_all(pkg):
    try:
        return collect_all(pkg)
    except Exception as e:
        print(f"[spec] WARN: skipping collect_all('{pkg}'): {e}")
        return ([], [], [])

def maybe_add_file(relpath, dest='.'):
    """Return (src, dest) if file/dir exists (relative to project root), else None."""
    p = PROJECT_ROOT / relpath
    if p.exists():
        return (str(p), dest)
    else:
        print(f"[spec] WARN missing asset: {p}")
        return None

# ---------------------------------------------------------------------
# 3rd-party package collections (guarded so build doesn't crash)
# ---------------------------------------------------------------------
# Kaleido (Plotly static image engine) – optional
kaleido_datas, kaleido_bins, kaleido_hidden = try_collect_all('kaleido')

# photutils
photutils_datas = try_collect_data('photutils')
photutils_bins  = try_collect_binaries('photutils')
photutils_mods  = try_collect_submodules('photutils')

# astropy, astroalign, skimage, cv2, sep_pjw, imagecodecs
astropy_mods     = try_collect_submodules('astropy')
astroalign_mods  = try_collect_submodules('astroalign')
skimage_mods     = try_collect_submodules('skimage')
cv2_mods         = try_collect_submodules('cv2')
cv2_bins         = try_collect_binaries('cv2')
sep_pjw_mods     = try_collect_submodules('sep_pjw')          # some builds alias sep to sep_pjw
sep_pjw_bins     = try_collect_binaries('sep_pjw')
imagecodecs_mods = try_collect_submodules('imagecodecs')
imagecodecs_bins = try_collect_binaries('imagecodecs')

# Dask data (templates, etc.) – optional
dask_datas       = try_collect_data('dask', include_py_files=False)

# sklearn array_api compat – optional (present on some scikit-learn builds)
sklearn_api_mods = try_collect_submodules('sklearn.externals.array_api_compat.numpy')

# certifi (SSL root certs)
certifi_datas    = try_collect_data('certifi')

# astroquery bits (CITATION + simbad data) – optional
astroquery_datas = try_collect_data('astroquery', includes=['CITATION', 'simbad/data/*'])

# Optional: lightkurve style (kept from SASv2)
lightkurve_datas = try_collect_data('lightkurve', includes=['data/lightkurve.mplstyle'])

# ---------------------------------------------------------------------
# Your own packages (force-include submodules for any lazy/dynamic imports)
# ---------------------------------------------------------------------
pro_mods       = try_collect_submodules('pro')
ops_mods       = try_collect_submodules('ops')
legacy_mods    = try_collect_submodules('legacy')
imageops_mods  = try_collect_submodules('imageops')

# ---------------------------------------------------------------------
# Hidden imports & binaries
# ---------------------------------------------------------------------
hiddenimports = [
    # common extras your v2 spec depended on
    'lz4.block',
    'zstandard',
    'base64',
    'ast',
    'cv2',
    'astropy.io.fits',
    'astropy.wcs',
    'skimage.transform',
    'skimage.feature',
    'scipy.spatial',
    'astroalign',
    'sep',
    'sep_pjw',
    'sep_pjw._version',
    '_version',

    # 3rd-party modules harvested above
    *photutils_mods,
    *astropy_mods,
    *astroalign_mods,
    *skimage_mods,
    *cv2_mods,
    *sep_pjw_mods,
    *imagecodecs_mods,
    *kaleido_hidden,
    *sklearn_api_mods,

    # your code packages
    *pro_mods,
    *ops_mods,
    *legacy_mods,
    *imageops_mods,
]

binaries = []
binaries += photutils_bins
binaries += cv2_bins
binaries += sep_pjw_bins
binaries += kaleido_bins
binaries += imagecodecs_bins

# ---------------------------------------------------------------------
# Data files (icons, CSV, FITS, folders)
# ---------------------------------------------------------------------
ICON_FILES = [
    'astrosuitepro.png',
    'astrosuitepro.ico',      # ok to include; mac prefers .icns below, this is harmless
    'green.png', 'neutral.png', 'whitebalance.png', 'morpho.png', 'clahe.png',
    'starnet.png', 'staradd.png',
    'LExtract.png', 'LInsert.png',
    'slot0.png','slot1.png','slot2.png','slot3.png','slot4.png','slot5.png','slot6.png','slot7.png','slot8.png','slot9.png',
    'rgbcombo.png','rgbextract.png','copyslot.png',
    'graxpert.png','cropicon.png','openfile.png','abeicon.png',
    'undoicon.png','redoicon.png','blaster.png','hdr.png',
    'invert.png','fliphorizontal.png','flipvertical.png',
    'rotateclockwise.png','rotatecounterclockwise.png',
    'maskcreate.png','maskapply.png','maskremove.png',
    'pixelmath.png','histogram.png','mosaic.png','rescale.png',
    'staralign.png','platesolve.png','psf.png','supernova.png',
    'starregistration.png','stacking.png','pedestal.png',
    'starspike.png','aperture.png','jwstpupil.png',
    'pen.png','livestacking.png','HRDiagram.png','convo.png',
    'spcc.png','SASP_data.fits','exoicon.png','gridicon.png',
    'dse.png','astrobin_filters.csv','isophote.png',
    'statstretch.png','starstretch.png','curves.png','disk.png',
    'uhs.png','blink.png','ppp.png','nbtorgb.png','freqsep.png',
    'contsub.png','halo.png','cosmic.png','cosmicsat.png',
    'imagecombine.png','wrench_icon.png','eye.png','nuke.png',
    'hubble.png','collage.png','annotated.png','colorwheel.png',
    'font.png','cvs.png','spinner.gif','wims.png',
    'linearfit.png','debayer.png',
    # icon with ambiguous name – try common extensions too
    'wimi_icon_256x256', 'wimi_icon_256x256.png', 'wimi_icon_256x256.ico',
]

datas = []

# imgs/ folder as a whole → keep under 'imgs' in dist
imgs_pair = maybe_add_file('imgs', dest='imgs')
if imgs_pair: datas.append(imgs_pair)

# individual assets at project root
for rel in ICON_FILES:
    pair = maybe_add_file(rel, dest='.')
    if pair: datas.append(pair)

# stand-alone helpers that you previously carried as data
for rel in ['xisf.py', 'numba_utils.py', 'wimi.py', 'wims.py']:
    pair = maybe_add_file(rel, dest='.')
    if pair: datas.append(pair)

# package data (3rd party)
datas += dask_datas
datas += photutils_datas
datas += kaleido_datas
datas += certifi_datas
datas += astroquery_datas
datas += try_collect_data('astropy', includes=['CITATION'])
datas += lightkurve_datas

# If a top-level _version.py exists in site-packages, include it (back-compat)
import importlib.util
try:
    spec = importlib.util.find_spec('_version')
    if spec and spec.origin and os.path.exists(spec.origin):
        datas.append((spec.origin, '.'))
except Exception:
    pass

# ---------------------------------------------------------------------
# Runtime hooks — build the list BEFORE Analysis, include only files
# ---------------------------------------------------------------------
RUNTIME_HOOKS = []
for fname in ('minimize_console.py', 'disable_zfpy.py', 'fix_importlib_metadata.py'):
    p = PROJECT_ROOT / fname
    if p.is_file():
        RUNTIME_HOOKS.append(str(p))

# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------
a = Analysis(
    ['setiastrosuitepro.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=RUNTIME_HOOKS,  # important: pass here; don't mutate later
    excludes=['torch', 'torchvision', 'PyQt5', 'PySide6', 'shiboken6', 'zfpy', 'numcodecs.zfpy'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# ------------------------- ONEFILE build (like SASv2) -------------------------
# Older-style onefile spec (works on current PyInstaller too):
# Pack everything into a single executable. No COLLECT, no BUNDLE.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,       # include binaries directly
    a.datas,          # include data directly
    [],               # (legacy slot; keep empty list)
    name='SetiAstroSuitePro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,        # UPX is usually unavailable on macOS; keep False
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,    # windowed
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / 'astrosuitepro.icns') if (PROJECT_ROOT / 'astrosuitepro.icns').exists() else None,
    onefile=True,     # <<— single-file like your SASv2 build
)
