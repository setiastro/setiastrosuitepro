# pro/resources.py
"""
Centralized resource paths for Seti Astro Suite Pro.

This module provides a single source of truth for all icon and resource paths,
handling both PyInstaller frozen builds and development environments.

Usage:
    from setiastro.saspro.resources import Icons, Resources
    
    icon = QIcon(Icons.WRENCH)
    data_path = Resources.SASP_DATA
"""
from __future__ import annotations
import os
import sys
from functools import lru_cache
from pathlib import Path


def _get_base_path() -> str:
    """Get base path for resources (PyInstaller, installed package, or development)."""
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    
    # First, check if we're running from source (setiastrosuitepro.py exists)
    # This takes priority when running the script directly from source
    try:
        # Check if we can find it via __main__ module (most reliable)
        if '__main__' in sys.modules:
            main_module = sys.modules['__main__']
            if hasattr(main_module, '__file__') and main_module.__file__:
                main_file = main_module.__file__
                main_dir = os.path.dirname(os.path.abspath(main_file))
                
                # Case 1: Running setiastrosuitepro.py directly
                if os.path.basename(main_file) == 'setiastrosuitepro.py':
                    images_dir = os.path.join(main_dir, 'images')
                    if os.path.exists(images_dir):
                        return main_dir
                
                # Case 2: Running as module (python -m setiastro.saspro)
                # __main__.py is at src/setiastro/saspro/__main__.py
                # Need to go up 4 levels to reach project root
                if os.path.basename(main_file) == '__main__.py':
                    # Walk up from __main__.py to find project root with images/
                    search_dir = main_dir
                    for _ in range(6):  # Don't go too far up
                        images_dir = os.path.join(search_dir, 'images')
                        if os.path.exists(images_dir):
                            return search_dir
                        parent = os.path.dirname(search_dir)
                        if parent == search_dir:  # Reached filesystem root
                            break
                        search_dir = parent
        
        # Check current working directory for setiastrosuitepro.py
        # This handles the case where user runs: python setiastrosuitepro.py
        cwd = os.getcwd()
        main_script = os.path.join(cwd, 'setiastrosuitepro.py')
        if os.path.exists(main_script):
            images_dir = os.path.join(cwd, 'images')
            if os.path.exists(images_dir):
                return cwd
        
        # Also check parent directories (in case we're in a subdirectory)
        # Walk up from current file location looking for images/ directory
        current_file = os.path.abspath(__file__)
        search_dir = os.path.dirname(current_file)
        for _ in range(6):  # Don't go too far up
            images_dir = os.path.join(search_dir, 'images')
            if os.path.exists(images_dir):
                return search_dir
            parent = os.path.dirname(search_dir)
            if parent == search_dir:  # Reached filesystem root
                break
            search_dir = parent
    except Exception:
        pass
    
    # Development: resources are in package directory (src/setiastro/images/)
    # File is at: src/setiastro/saspro/resources.py
    # Check if images/ exists in the setiastro package directory
    current_file = os.path.abspath(__file__)
    # Go up from resources.py -> saspro -> setiastro (package directory)
    package_dir = os.path.dirname(os.path.dirname(current_file))
    images_dir = os.path.join(package_dir, 'images')
    if os.path.exists(images_dir):
        return package_dir
    
    # Fallback: try project root (for backward compatibility)
    # Go up from resources.py -> saspro -> setiastro -> src -> project root
    base = os.path.dirname(os.path.dirname(package_dir))
    images_dir = os.path.join(base, 'images')
    if os.path.exists(images_dir):
        return base
    
    # Fallback: try going up one more level (in case structure is different)
    base = os.path.dirname(base)
    if os.path.exists(os.path.join(base, 'images')):
        return base
    
    # Check if we're in an installed package (last resort)
    # When installed via pip, the package is in site-packages
    try:
        import setiastro
        package_dir = os.path.dirname(os.path.abspath(setiastro.__file__))
        # Check if images/ exists at package root level (for pip-installed packages)
        # The images/ directory should be at the same level as setiastro/ in site-packages
        package_parent = os.path.dirname(package_dir)
        images_dir = os.path.join(package_parent, 'images')
        if os.path.exists(images_dir):
            return package_parent
        
        # Also check if images/ is inside the package directory
        images_in_package = os.path.join(package_dir, 'images')
        if os.path.exists(images_in_package):
            return package_dir
    except (ImportError, AttributeError):
        pass
    
    # Last resort: return the calculated path anyway
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))


def _resource_path(filename: str) -> str:
    base = _get_base_path()
    fn = filename

    is_img = fn.lower().endswith(('.png','.ico','.gif','.icns','.svg','.jpg','.jpeg','.bmp'))
    if is_img:
        candidates = [
            os.path.join(base, 'images', fn),
            os.path.join(base, 'setiastro', 'images', fn),
            os.path.join(base, 'setiastro', 'saspro', 'images', fn),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

    # data / other files
    candidates = [
        os.path.join(base, fn),
        os.path.join(base, 'setiastro', fn),
        os.path.join(base, 'setiastro', 'saspro', fn),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    return os.path.join(base, fn)


class Icons:
    """
    Centralized icon paths.
    
    All paths are computed lazily and cached.
    Access via class attributes: Icons.WRENCH, Icons.HISTOGRAM, etc.
    """
    
    # Application
    APP = property(lambda self: _resource_path('astrosuitepro.png'))
    APP_ICO = property(lambda self: _resource_path('astrosuitepro.ico'))
    BACKGROUND = property(lambda self: _resource_path('background.png')) 
    
    # Processing tools
    GREEN = property(lambda self: _resource_path('green.png'))
    NEUTRAL = property(lambda self: _resource_path('neutral.png'))
    WHITE_BALANCE = property(lambda self: _resource_path('whitebalance.png'))
    TEXTURE_CLARITY = property(lambda self: _resource_path('TextureClarity.svg'))
    MORPHOLOGY = property(lambda self: _resource_path('morpho.png'))
    CLAHE = property(lambda self: _resource_path('clahe.png'))
    HDR = property(lambda self: _resource_path('hdr.png'))
    INVERT = property(lambda self: _resource_path('invert.png'))
    
    # Star operations
    STARNET = property(lambda self: _resource_path('starnet.png'))
    STAR_ADD = property(lambda self: _resource_path('staradd.png'))
    STAR_ALIGN = property(lambda self: _resource_path('staralign.png'))
    STAR_REGISTRATION = property(lambda self: _resource_path('starregistration.png'))
    STAR_SPIKE = property(lambda self: _resource_path('starspike.png'))
    ASTRO_SPIKE = property(lambda self: _resource_path('Astro_Spikes.png'))
    STAR_STRETCH = property(lambda self: _resource_path('starstretch.png'))
    
    # Luminance
    L_EXTRACT = property(lambda self: _resource_path('LExtract.png'))
    L_INSERT = property(lambda self: _resource_path('LInsert.png'))
    
    # Slots (0-9)
    SLOT_0 = property(lambda self: _resource_path('slot0.png'))
    SLOT_1 = property(lambda self: _resource_path('slot1.png'))
    SLOT_2 = property(lambda self: _resource_path('slot2.png'))
    SLOT_3 = property(lambda self: _resource_path('slot3.png'))
    SLOT_4 = property(lambda self: _resource_path('slot4.png'))
    SLOT_5 = property(lambda self: _resource_path('slot5.png'))
    SLOT_6 = property(lambda self: _resource_path('slot6.png'))
    SLOT_7 = property(lambda self: _resource_path('slot7.png'))
    SLOT_8 = property(lambda self: _resource_path('slot8.png'))
    SLOT_9 = property(lambda self: _resource_path('slot9.png'))
    
    # RGB operations
    RGB_COMBO = property(lambda self: _resource_path('rgbcombo.png'))
    RGB_EXTRACT = property(lambda self: _resource_path('rgbextract.png'))
    RGB_ALIGN = property(lambda self: _resource_path('rgbalign.png'))
    COPY_SLOT = property(lambda self: _resource_path('copyslot.png'))
    
    # External tools
    GRAXPERT = property(lambda self: _resource_path('graxpert.png'))
    COSMIC = property(lambda self: _resource_path('cosmic.png'))
    COSMIC_SAT = property(lambda self: _resource_path('cosmicsat.png'))
    
    # Image operations
    CROP = property(lambda self: _resource_path('cropicon.png'))
    OPEN_FILE = property(lambda self: _resource_path('openfile.png'))
    ABE = property(lambda self: _resource_path('abeicon.png'))
    BLASTER = property(lambda self: _resource_path('blaster.png'))
    CLONESTAMP = property(lambda self: _resource_path('clonestamp.png'))
    
    # Undo/Redo
    UNDO = property(lambda self: _resource_path('undoicon.png'))
    REDO = property(lambda self: _resource_path('redoicon.png'))
    
    # Transformations
    FLIP_HORIZONTAL = property(lambda self: _resource_path('fliphorizontal.png'))
    FLIP_VERTICAL = property(lambda self: _resource_path('flipvertical.png'))
    ROTATE_CW = property(lambda self: _resource_path('rotateclockwise.png'))
    ROTATE_CCW = property(lambda self: _resource_path('rotatecounterclockwise.png'))
    ROTATE_180 = property(lambda self: _resource_path('rotate180.png'))
    ROTATE_ANY      = property(lambda self: _resource_path('rotatearbitrary.png'))
    RESCALE = property(lambda self: _resource_path('rescale.png'))
    
    # Masks
    MASK_CREATE = property(lambda self: _resource_path('maskcreate.png'))
    MASK_APPLY = property(lambda self: _resource_path('maskapply.png'))
    MASK_REMOVE = property(lambda self: _resource_path('maskremove.png'))
    
    # Analysis
    PIXEL_MATH = property(lambda self: _resource_path('pixelmath.png'))
    HISTOGRAM = property(lambda self: _resource_path('histogram.png'))
    MOSAIC = property(lambda self: _resource_path('mosaic.png'))
    PLATE_SOLVE = property(lambda self: _resource_path('platesolve.png'))
    PSF = property(lambda self: _resource_path('psf.png'))
    ISOPHOTE = property(lambda self: _resource_path('isophote.png'))
    
    # Stacking
    STACKING = property(lambda self: _resource_path('stacking.png'))
    LIVE_STACKING = property(lambda self: _resource_path('livestacking.png'))
    IMAGE_COMBINE = property(lambda self: _resource_path('imagecombine.png'))
    PLANETARY_STACKER = property(lambda self: _resource_path('planetarystacker.png'))
    PLANET_PROJECTION = property(lambda self: _resource_path('3dplanet.png'))   
    
    # Moon phase (WIMS)
    MOON_NEW            = property(lambda self: _resource_path('new_moon.png'))
    MOON_WAXING_CRES_1  = property(lambda self: _resource_path('waxing_crescent_1.png'))
    MOON_WAXING_CRES_2  = property(lambda self: _resource_path('waxing_crescent_2.png'))
    MOON_WAXING_CRES_3  = property(lambda self: _resource_path('waxing_crescent_3.png'))
    MOON_WAXING_CRES_4  = property(lambda self: _resource_path('waxing_crescent_4.png'))
    MOON_WAXING_CRES_5  = property(lambda self: _resource_path('waxing_crescent_5.png'))

    MOON_FIRST_QUARTER  = property(lambda self: _resource_path('first_quarter.png'))

    MOON_WAXING_GIB_1   = property(lambda self: _resource_path('waxing_gibbous_1.png'))
    MOON_WAXING_GIB_2   = property(lambda self: _resource_path('waxing_gibbous_2.png'))
    MOON_WAXING_GIB_3   = property(lambda self: _resource_path('waxing_gibbous_3.png'))
    MOON_WAXING_GIB_4   = property(lambda self: _resource_path('waxing_gibbous_4.png'))
    MOON_WAXING_GIB_5   = property(lambda self: _resource_path('waxing_gibbous_5.png'))

    MOON_FULL           = property(lambda self: _resource_path('full_moon.png'))

    MOON_WANING_GIB_1   = property(lambda self: _resource_path('waning_gibbous_1.png'))
    MOON_WANING_GIB_2   = property(lambda self: _resource_path('waning_gibbous_2.png'))
    MOON_WANING_GIB_3   = property(lambda self: _resource_path('waning_gibbous_3.png'))
    MOON_WANING_GIB_4   = property(lambda self: _resource_path('waning_gibbous_4.png'))
    MOON_WANING_GIB_5   = property(lambda self: _resource_path('waning_gibbous_5.png'))

    MOON_LAST_QUARTER   = property(lambda self: _resource_path('last_quarter.png'))

    MOON_WANING_CRES_1  = property(lambda self: _resource_path('waning_crescent_1.png'))
    MOON_WANING_CRES_2  = property(lambda self: _resource_path('waning_crescent_2.png'))
    MOON_WANING_CRES_3  = property(lambda self: _resource_path('waning_crescent_3.png'))
    MOON_WANING_CRES_4  = property(lambda self: _resource_path('waning_crescent_4.png'))
    MOON_WANING_CRES_5  = property(lambda self: _resource_path('waning_crescent_5.png'))


    # Special features
    SUPERNOVA = property(lambda self: _resource_path('supernova.png'))
    PEDESTAL = property(lambda self: _resource_path('pedestal.png'))
    APERTURE = property(lambda self: _resource_path('aperture.png'))
    JWST_PUPIL = property(lambda self: _resource_path('jwstpupil.png'))
    SIGNATURE = property(lambda self: _resource_path('pen.png'))
    HR_DIAGRAM = property(lambda self: _resource_path('HRDiagram.png'))
    EXOPLANET = property(lambda self: _resource_path('exoicon.png'))
    
    # Deconvolution & filters
    CONVO = property(lambda self: _resource_path('convo.png'))
    FREQ_SEP = property(lambda self: _resource_path('freqsep.png'))
    MULTISCALE_DECOMP = property(lambda self: _resource_path('multiscale_decomp.png'))
    CONT_SUB = property(lambda self: _resource_path('contsub.png'))
    HALO = property(lambda self: _resource_path('halo.png'))
    ABERRATION = property(lambda self: _resource_path('aberration.png'))
    
    # Color
    SPCC = property(lambda self: _resource_path('spcc.png'))
    MAGNITUDE = property(lambda self: _resource_path('magnitude.png'))
    SNR_TOOL = property(lambda self: _resource_path('snr.png'))
    DSE = property(lambda self: _resource_path('dse.png'))
    COLOR_WHEEL = property(lambda self: _resource_path('colorwheel.png'))
    SELECTIVE_COLOR = property(lambda self: _resource_path('selectivecolor.png'))
    NB_TO_RGB = property(lambda self: _resource_path('nbtorgb.png'))
    NARROWBANDNORMALIZATION = property(lambda self: _resource_path('narrowbandnormalization.png'))
    
    # Stretching
    STAT_STRETCH = property(lambda self: _resource_path('statstretch.png'))
    CURVES = property(lambda self: _resource_path('curves.png'))
    UHS = property(lambda self: _resource_path('uhs.png'))
    
    # UI elements
    WRENCH = property(lambda self: _resource_path('wrench_icon.png'))
    EYE = property(lambda self: _resource_path('eye.png'))
    DISK = property(lambda self: _resource_path('disk.png'))
    NUKE = property(lambda self: _resource_path('nuke.png'))
    GRID = property(lambda self: _resource_path('gridicon.png'))
    FONT = property(lambda self: _resource_path('font.png'))
    CSV = property(lambda self: _resource_path('cvs.png'))
    PPP = property(lambda self: _resource_path('ppp.png'))
    SCRIPT = property(lambda self: _resource_path('script.png'))
    ACV = property(lambda self: _resource_path('acv_icon.png'))
    
    # Blink & comparison
    BLINK = property(lambda self: _resource_path('blink.png'))
    HUBBLE = property(lambda self: _resource_path('hubble.png'))
    COLLAGE = property(lambda self: _resource_path('collage.png'))
    ANNOTATED = property(lambda self: _resource_path('annotated.png'))
    
    # WIMS/WIMI
    WIMS = property(lambda self: _resource_path('wims.png'))
    WIMI = property(lambda self: _resource_path('wimi_icon_256x256.png'))
    
    # Other
    LINEAR_FIT = property(lambda self: _resource_path('linearfit.png'))
    DEBAYER = property(lambda self: _resource_path('debayer.png'))
    FUNCTION_BUNDLES = property(lambda self: _resource_path('functionbundle.png'))
    VIEW_BUNDLES = property(lambda self: _resource_path('viewbundle.png'))
    FINDER_CHART = property(lambda self: _resource_path('finderchart.png'))

# Singleton instances for easy access
_icons_instance = None
_resources_instance = None


def get_icons() -> Icons:
    """Get the Icons singleton instance."""
    global _icons_instance
    if _icons_instance is None:
        _icons_instance = Icons()
    return _icons_instance


def get_resources() -> Resources:
    """Get the Resources singleton instance."""
    global _resources_instance
    if _resources_instance is None:
        _resources_instance = Resources()
    return _resources_instance


# Convenience functions for direct path access
@lru_cache(maxsize=128)
def get_icon_path(name: str) -> str:
    """
    Get icon path by name.
    
    Args:
        name: Icon filename (with or without extension)
        
    Returns:
        Full path to icon file
        
    Example:
        path = get_icon_path('wrench_icon.png')
        path = get_icon_path('histogram')  # .png added automatically
    """
    if not name.endswith(('.png', '.ico', '.gif', '.svg')):
        name = f"{name}.png"
    return _resource_path(name)


@lru_cache(maxsize=32)
def get_data_path(name: str) -> str:
    """
    Get data file path by name.
    
    Args:
        name: Data filename
        
    Returns:
        Full path to data file
    """
    return _resource_path(name)

# ---------------- Legacy compatibility (LAZY) ----------------
# These names match the original module-level variables used in older code.

_LEGACY_ICON_MAP = {
    'icon_path': 'astrosuitepro.png',
    'windowslogo_path': 'astrosuitepro.ico',
    'green_path': 'green.png',
    'neutral_path': 'neutral.png',
    'whitebalance_path': 'whitebalance.png',
    'texture_clarity_path': 'TextureClarity.svg',
    'morpho_path': 'morpho.png',
    'clahe_path': 'clahe.png',
    'starnet_path': 'starnet.png',
    'staradd_path': 'staradd.png',
    'LExtract_path': 'LExtract.png',
    'LInsert_path': 'LInsert.png',
    'slot0_path': 'slot0.png',
    'slot1_path': 'slot1.png',
    'slot2_path': 'slot2.png',
    'slot3_path': 'slot3.png',
    'slot4_path': 'slot4.png',
    'slot5_path': 'slot5.png',
    'slot6_path': 'slot6.png',
    'slot7_path': 'slot7.png',
    'slot8_path': 'slot8.png',
    'slot9_path': 'slot9.png',
    'acv_icon_path': 'acv_icon.png',

    'moon_new_path': 'new_moon.png',
    'moon_waxing_crescent_1_path': 'waxing_crescent_1.png',
    'moon_waxing_crescent_2_path': 'waxing_crescent_2.png',
    'moon_waxing_crescent_3_path': 'waxing_crescent_3.png',
    'moon_waxing_crescent_4_path': 'waxing_crescent_4.png',
    'moon_waxing_crescent_5_path': 'waxing_crescent_5.png',
    'moon_first_quarter_path': 'first_quarter.png',
    'moon_waxing_gibbous_1_path': 'waxing_gibbous_1.png',
    'moon_waxing_gibbous_2_path': 'waxing_gibbous_2.png',
    'moon_waxing_gibbous_3_path': 'waxing_gibbous_3.png',
    'moon_waxing_gibbous_4_path': 'waxing_gibbous_4.png',
    'moon_waxing_gibbous_5_path': 'waxing_gibbous_5.png',
    'moon_full_path': 'full_moon.png',
    'moon_waning_gibbous_1_path': 'waning_gibbous_1.png',
    'moon_waning_gibbous_2_path': 'waning_gibbous_2.png',
    'moon_waning_gibbous_3_path': 'waning_gibbous_3.png',
    'moon_waning_gibbous_4_path': 'waning_gibbous_4.png',
    'moon_waning_gibbous_5_path': 'waning_gibbous_5.png',
    'moon_last_quarter_path': 'last_quarter.png',
    'moon_waning_crescent_1_path': 'waning_crescent_1.png',
    'moon_waning_crescent_2_path': 'waning_crescent_2.png',
    'moon_waning_crescent_3_path': 'waning_crescent_3.png',
    'moon_waning_crescent_4_path': 'waning_crescent_4.png',
    'moon_waning_crescent_5_path': 'waning_crescent_5.png',

    'rgbcombo_path': 'rgbcombo.png',
    'rgbextract_path': 'rgbextract.png',
    'copyslot_path': 'copyslot.png',
    'graxperticon_path': 'graxpert.png',
    'cropicon_path': 'cropicon.png',
    'openfile_path': 'openfile.png',
    'abeicon_path': 'abeicon.png',
    'undoicon_path': 'undoicon.png',
    'redoicon_path': 'redoicon.png',
    'blastericon_path': 'blaster.png',
    'clonestampicon_path': 'clonestamp.png',
    'hdr_path': 'hdr.png',
    'invert_path': 'invert.png',
    'fliphorizontal_path': 'fliphorizontal.png',
    'flipvertical_path': 'flipvertical.png',
    'rotateclockwise_path': 'rotateclockwise.png',
    'rotatecounterclockwise_path': 'rotatecounterclockwise.png',
    'rotate180_path': 'rotate180.png',
    'rotatearbitrary_path': 'rotatearbitrary.png',
    'maskcreate_path': 'maskcreate.png',
    'maskapply_path': 'maskapply.png',
    'maskremove_path': 'maskremove.png',
    'pixelmath_path': 'pixelmath.png',
    'histogram_path': 'histogram.png',
    'mosaic_path': 'mosaic.png',
    'rescale_path': 'rescale.png',
    'staralign_path': 'staralign.png',
    'mask_path': 'maskapply.png',
    'platesolve_path': 'platesolve.png',
    'psf_path': 'psf.png',
    'supernova_path': 'supernova.png',
    'starregistration_path': 'starregistration.png',
    'stacking_path': 'stacking.png',
    'pedestal_icon_path': 'pedestal.png',
    'starspike_path': 'starspike.png',
    'astrospike_path': 'Astro_Spikes.png',
    'aperture_path': 'aperture.png',
    'jwstpupil_path': 'jwstpupil.png',
    'signature_icon_path': 'pen.png',
    'livestacking_path': 'livestacking.png',
    'hrdiagram_path': 'HRDiagram.png',
    'convoicon_path': 'convo.png',
    'spcc_icon_path': 'spcc.png',

    'exoicon_path': 'exoicon.png',
    'peeker_icon': 'gridicon.png',
    'dse_icon_path': 'dse.png',
    'isophote_path': 'isophote.png',
    'statstretch_path': 'statstretch.png',
    'starstretch_path': 'starstretch.png',
    'curves_path': 'curves.png',
    'disk_path': 'disk.png',
    'uhs_path': 'uhs.png',
    'blink_path': 'blink.png',
    'ppp_path': 'ppp.png',
    'nbtorgb_path': 'nbtorgb.png',
    'freqsep_path': 'freqsep.png',
    'multiscale_decomp_path': 'multiscale_decomp.png',
    'contsub_path': 'contsub.png',
    'halo_path': 'halo.png',
    'cosmic_path': 'cosmic.png',
    'satellite_path': 'cosmicsat.png',
    'imagecombine_path': 'imagecombine.png',
    'wrench_path': 'wrench_icon.png',
    'eye_icon_path': 'eye.png',
    'disk_icon_path': 'disk.png',
    'nuke_path': 'nuke.png',
    'hubble_path': 'hubble.png',
    'collage_path': 'collage.png',
    'annotated_path': 'annotated.png',
    'colorwheel_path': 'colorwheel.png',
    'narrowbandnormalization_path': 'narrowbandnormalization.png',
    'font_path': 'font.png',
    'csv_icon_path': 'cvs.png',
    'wims_path': 'wims.png',
    'wimi_path': 'wimi_icon_256x256.png',
    'linearfit_path': 'linearfit.png',
    'debayer_path': 'debayer.png',
    'aberration_path': 'aberration.png',
    'functionbundles_path': 'functionbundle.png',
    'planetarystacker_path': 'planetarystacker.png',
    'viewbundles_path': 'viewbundle.png',
    'selectivecolor_path': 'selectivecolor.png',
    'rgbalign_path': 'rgbalign.png',
    'background_path': 'background.png',
    'script_icon_path': 'script.png',
    'planetprojection_path': '3dplanet.png',
    'finderchart_path': 'finderchart.png',
    'magnitude_path': 'magnitude.png',
    'snr_path': 'snr.png',
}

_LEGACY_DATA_MAP = {
    'sasp_data_path': 'data/SASP_data.fits',
    'astrobin_filters_csv_path': 'data/catalogs/astrobin_filters.csv',
    'spinner_path': 'spinner.gif',
}

def __getattr__(name: str):
    # --- legacy paths (lazy) ---
    if name in _LEGACY_ICON_MAP:
        return get_icon_path(_LEGACY_ICON_MAP[name])
    if name in _LEGACY_DATA_MAP:
        return get_data_path(_LEGACY_DATA_MAP[name])

    # --- special exports ---
    if name == 'background_startup_path':
        return _resource_path('Background_startup.jpg')
    if name == 'resource_monitor_qml':
        return _resource_path(os.path.join("qml", "ResourceMonitor.qml"))

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return (
        list(globals().keys())
        + list(_LEGACY_ICON_MAP.keys())
        + list(_LEGACY_DATA_MAP.keys())
        + ['background_startup_path', 'resource_monitor_qml']
    )


class Resources:
    """
    Centralized data resource paths.
    """
    SASP_DATA = property(lambda self: _resource_path('data/SASP_data.fits'))
    ASTROBIN_FILTERS_CSV = property(lambda self: _resource_path('data/catalogs/astrobin_filters.csv'))
    SPINNER_GIF = property(lambda self: _resource_path('spinner.gif'))

    # --- Models root ---
    MODELS_DIR = property(lambda self: get_models_dir())

    # --- Cosmic Clarity Sharpen ---
    CC_STELLAR_SHARP_PTH  = property(lambda self: model_path('deep_sharp_stellar_cnn_AI3_5s.pth'))
    CC_STELLAR_SHARP_ONNX = property(lambda self: model_path('deep_sharp_stellar_cnn_AI3_5s.onnx'))

    CC_NS1_PTH  = property(lambda self: model_path('deep_nonstellar_sharp_cnn_radius_1AI3_5s.pth'))
    CC_NS1_ONNX = property(lambda self: model_path('deep_nonstellar_sharp_cnn_radius_1AI3_5s.onnx'))

    CC_NS2_PTH  = property(lambda self: model_path('deep_nonstellar_sharp_cnn_radius_2AI3_5s.pth'))
    CC_NS2_ONNX = property(lambda self: model_path('deep_nonstellar_sharp_cnn_radius_2AI3_5s.onnx'))
    CC_NS4_PTH  = property(lambda self: model_path('deep_nonstellar_sharp_cnn_radius_4AI3_5s.pth'))
    CC_NS4_ONNX = property(lambda self: model_path('deep_nonstellar_sharp_cnn_radius_4AI3_5s.onnx'))

    CC_NS8_PTH  = property(lambda self: model_path('deep_nonstellar_sharp_cnn_radius_8AI3_5s.pth'))
    CC_NS8_ONNX = property(lambda self: model_path('deep_nonstellar_sharp_cnn_radius_8AI3_5s.onnx'))

    CC_S_PTH = property(lambda self: model_path('deep_sharp_stellar_AI4.pth'))
    CC_STELLAR_NAF_ONNX = property(lambda self: model_path('deep_sharp_stellar_AI4.onnx'))
    CC_NS_PTH = property(lambda self: model_path('deep_nonstellar_sharp_conditional_psf_AI4.pth'))
    CC_NS_COND_NAF_ONNX = property(lambda self: model_path('deep_nonstellar_sharp_conditional_psf_AI4.onnx'))

    # --- Cosmic Clarity Denoise (NAFNet AI4) ---
    CC_DENOISE_MONO_PTH  = property(lambda self: model_path('deep_denoise_mono_AI4.pth'))
    CC_DENOISE_MONO_ONNX = property(lambda self: model_path('deep_denoise_mono_AI4.onnx'))

    CC_DENOISE_COLOR_PTH  = property(lambda self: model_path('deep_denoise_color_AI4.pth'))
    CC_DENOISE_COLOR_ONNX = property(lambda self: model_path('deep_denoise_color_AI4.onnx'))
    CC_DENOISE_PTH  = property(lambda self: model_path('deep_denoise_cnn_AI3_6.pth'))
    CC_DENOISE_ONNX = property(lambda self: model_path('deep_denoise_cnn_AI3_6.onnx'))

    # --- Super Resolution ---
    CC_SUPERRES_2X_PTH  = property(lambda self: model_path('superres_2x.pth'))
    CC_SUPERRES_2X_ONNX = property(lambda self: model_path('superres_2x.onnx'))
    CC_SUPERRES_3X_PTH  = property(lambda self: model_path('superres_3x.pth'))
    CC_SUPERRES_3X_ONNX = property(lambda self: model_path('superres_3x.onnx'))

    CC_SUPERRES_4X_PTH  = property(lambda self: model_path('superres_4x.pth'))
    CC_SUPERRES_4X_ONNX = property(lambda self: model_path('superres_4x.onnx'))

    # --- Dark Star (Star Removal) ---
    CC_DARKSTAR_MONO_PTH   = property(lambda self: model_path('darkstar_v2.1.pth'))
    CC_DARKSTAR_MONO_ONNX  = property(lambda self: model_path('darkstar_v2.1.onnx'))
    CC_DARKSTAR_COLOR_PTH  = property(lambda self: model_path('darkstar_v2.1c.pth'))
    CC_DARKSTAR_COLOR_ONNX = property(lambda self: model_path('darkstar_v2.1c.onnx'))

    # --- Cosmic Clarity Satellite Removal ---
    CC_SAT_DETECT1_PTH  = property(lambda self: model_path('satellite_trail_detector_AI3.5.pth'))
    CC_SAT_DETECT1_ONNX = property(lambda self: model_path('satellite_trail_detector_AI3.5.onnx'))
    CC_SAT_DETECT2_PTH  = property(lambda self: model_path('satellite_trail_detector_mobilenetv2.5.pth'))
    CC_SAT_DETECT2_ONNX = property(lambda self: model_path('satellite_trail_detector_mobilenetv2.5.onnx'))

    CC_SAT_REMOVE_PTH   = property(lambda self: model_path('satelliteremovalAI3.5.pth'))
    CC_SAT_REMOVE_ONNX  = property(lambda self: model_path('satelliteremovalAI3.5.onnx'))

@lru_cache(maxsize=8)
def get_models_dir() -> str:
    """
    Models are NOT packaged resources. They must be installed via the model manager.
    This returns the user models root and never falls back to _internal/data/models.
    """
    from setiastro.saspro.model_manager import models_root
    p = Path(models_root())
    # Ensure dir exists (models_root should already do this, but harmless)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _assert_not_internal_models_path(p: str):
    s = str(p).lower().replace("/", "\\")
    if "\\_internal\\" in s and "\\data\\models\\" in s:
        raise RuntimeError(f"Legacy internal model path detected: {p}")

def model_path(filename: str) -> str:
    from setiastro.saspro.model_manager import require_model
    p = str(require_model(filename))
    _assert_not_internal_models_path(p)
    return p


# QML helper
resource_monitor_qml = _resource_path(os.path.join("qml", "ResourceMonitor.qml"))

# Export list for `from setiastro.saspro.resources import *`
__all__ = [
    'Icons', 'Resources',
    'get_icons', 'get_resources',
    'get_icon_path', 'get_data_path',
    'resource_monitor_qml',
    'background_startup_path',
] + list(_LEGACY_ICON_MAP.keys()) + list(_LEGACY_DATA_MAP.keys())