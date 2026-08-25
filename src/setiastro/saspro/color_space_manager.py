"""Central color-space and ICC profile handling for SASpro."""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColorSpace, QImage

logger = logging.getLogger(__name__)


SETTINGS_KEY = "color_management/working_space"
DEFAULT_COLOR_SPACE = "DisplayP3"
VIEWPORT_MODE_SETTINGS_KEY = "color_management/viewport_mode"
VIEWPORT_MODE_UNMANAGED = "unmanaged"
VIEWPORT_MODE_TAGGED = "tagged"
DEFAULT_VIEWPORT_MODE = VIEWPORT_MODE_UNMANAGED
VIEWPORT_COLOR_MODES = (VIEWPORT_MODE_UNMANAGED, VIEWPORT_MODE_TAGGED)
RENDERING_INTENT_RELATIVE = "relative_colorimetric"
RENDERING_INTENT_PERCEPTUAL = "perceptual"
DEFAULT_RENDERING_INTENT = RENDERING_INTENT_RELATIVE
SOFTPROOF_INTENT_SETTINGS_KEY = "color_management/softproof_rendering_intent"
SOFTPROOF_BPC_SETTINGS_KEY = "color_management/softproof_black_point_compensation"


@dataclass(frozen=True)
class ColorSpaceInfo:
    key: str
    name: str
    description: str
    icc_key: str
    qt_names: tuple[str, ...]
    profile_names: tuple[str, ...]


COLOR_SPACES: dict[str, ColorSpaceInfo] = {
    "DisplayP3": ColorSpaceInfo(
        key="DisplayP3",
        name="Display P3",
        description="Wide-gamut color space with D65 white point and P3 primaries.",
        icc_key="P3",
        qt_names=("DisplayP3",),
        profile_names=(
            "DisplayP3.icc",
            "Display P3.icc",
            "DisplayP3.icm",
            "Display P3.icm",
        ),
    ),
    "AdobeRGB": ColorSpaceInfo(
        key="AdobeRGB",
        name="Adobe RGB (1998)",
        description="Wide-gamut photography color space using Adobe RGB (1998) primaries.",
        icc_key="AdobeRGB",
        qt_names=("AdobeRgb", "AdobeRGB"),
        profile_names=(
            "AdobeRGB.icc",
            "AdobeRGB1998.icc",
            "Adobe RGB (1998).icc",
            "AdobeRGB.icm",
            "AdobeRGB1998.icm",
            "Adobe RGB (1998).icm",
        ),
    ),
    "ProPhotoRGB": ColorSpaceInfo(
        key="ProPhotoRGB",
        name="ProPhoto RGB",
        description="Very wide-gamut color space for professional color workflows.",
        icc_key="ProPhotoRGB",
        qt_names=("ProPhotoRgb", "ProPhotoRGB"),
        profile_names=(
            "ProPhotoRGB.icc",
            "ProPhoto RGB.icc",
            "ProPhoto.icc",
            "ROMM RGB.icc",
            "ProPhotoRGB.icm",
            "ProPhoto RGB.icm",
            "ProPhoto.icm",
            "ROMM RGB.icm",
        ),
    ),
    "sRGB": ColorSpaceInfo(
        key="sRGB",
        name="sRGB",
        description="Standard RGB color space for web sharing and broad compatibility.",
        icc_key="sRGB",
        qt_names=("SRgb", "SRGB", "Srgb"),
        profile_names=("sRGB.icc", "sRGB.icm", "srgb.icc", "srgb.icm"),
    ),
}


_ALIASES = {
    "p3": "DisplayP3",
    "displayp3": "DisplayP3",
    "display p3": "DisplayP3",
    "display-p3": "DisplayP3",
    "display_p3": "DisplayP3",
    "adobergb": "AdobeRGB",
    "adobe-rgb": "AdobeRGB",
    "adobe_rgb": "AdobeRGB",
    "adobergb1998": "AdobeRGB",
    "adobe rgb (1998)": "AdobeRGB",
    "prophoto": "ProPhotoRGB",
    "prophotorgb": "ProPhotoRGB",
    "prophoto rgb": "ProPhotoRGB",
    "prophoto-rgb": "ProPhotoRGB",
    "prophoto_rgb": "ProPhotoRGB",
    "romm rgb": "ProPhotoRGB",
    "rommrgb": "ProPhotoRGB",
    "srgb": "sRGB",
    "s rgb": "sRGB",
}


def _clean_key(key: object) -> str:
    return str(key or "").strip()


def normalize_color_space_key(key: object) -> str:
    """Return a supported canonical key, defaulting to Display P3."""
    raw = _clean_key(key)
    if raw in COLOR_SPACES:
        return raw
    collapsed = " ".join(raw.replace("_", " ").replace("-", " ").split()).lower()
    return _ALIASES.get(collapsed, DEFAULT_COLOR_SPACE)


def get_color_space_names() -> list[str]:
    return list(COLOR_SPACES.keys())


def get_color_space_display_names() -> list[str]:
    return [info.name for info in COLOR_SPACES.values()]


def get_color_space_options() -> list[ColorSpaceInfo]:
    return list(COLOR_SPACES.values())


def get_description_from_key(key: object) -> str:
    return COLOR_SPACES[normalize_color_space_key(key)].description


def get_icc_key_from_color_space_key(key: object) -> str:
    return COLOR_SPACES[normalize_color_space_key(key)].icc_key


def get_key_from_icc_key(icc_key: object) -> str:
    raw = normalize_color_space_key(icc_key)
    if raw in COLOR_SPACES:
        return raw
    text = _clean_key(icc_key).lower()
    for info in COLOR_SPACES.values():
        if info.icc_key.lower() == text:
            return info.key
    return DEFAULT_COLOR_SPACE


def _profile_directories() -> tuple[Path, ...]:
    package_dir = Path(__file__).resolve().parent
    dirs = [
        package_dir / "icc_profiles",
        package_dir / "legacy" / "resources",
    ]
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        dirs.append(Path(system_root) / "System32" / "spool" / "drivers" / "color")
    elif sys_profile_dir := os.environ.get("COLOR_SYNC_PROFILES"):
        dirs.append(Path(sys_profile_dir))
    dirs.extend(
        Path(p) for p in (
            "/System/Library/ColorSync/Profiles",
            "/Library/ColorSync/Profiles",
            "/usr/share/color/icc",
            "/usr/local/share/color/icc",
        )
    )
    seen = set()
    out = []
    for directory in dirs:
        key = str(directory).lower()
        if key not in seen:
            seen.add(key)
            out.append(directory)
    return tuple(out)


def get_profile_search_directories() -> tuple[str, ...]:
    return tuple(str(path) for path in _profile_directories())


@functools.lru_cache(maxsize=64)
def _read_profile_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except Exception:
        return None


@functools.lru_cache(maxsize=16)
def _find_profile_path(key: str) -> Optional[str]:
    info = COLOR_SPACES[normalize_color_space_key(key)]
    wanted = {name.lower(): name for name in info.profile_names}
    for directory in _profile_directories():
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            entries = {child.name.lower(): child for child in directory.iterdir() if child.is_file()}
        except Exception:
            continue
        for lowered in wanted:
            found = entries.get(lowered)
            if found is not None:
                return str(found)
    return None


def get_icc_profile_path(key: object) -> Optional[str]:
    return _find_profile_path(normalize_color_space_key(key))


@functools.lru_cache(maxsize=1)
def _srgb_profile_bytes() -> Optional[bytes]:
    profile_path = get_icc_profile_path("sRGB")
    if profile_path:
        data = _read_profile_bytes(profile_path)
        if data:
            return data
    try:
        from PIL import ImageCms

        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception as exc:
        logger.warning("Could not build sRGB ICC profile: %s", exc)
        return None


def get_icc_profile_bytes(key: object, *, fallback_to_srgb: bool = True) -> Optional[bytes]:
    canonical = normalize_color_space_key(key)
    if canonical == "sRGB":
        return _srgb_profile_bytes()

    profile_path = get_icc_profile_path(canonical)
    if profile_path:
        data = _read_profile_bytes(profile_path)
        if data:
            return data

    if fallback_to_srgb:
        info = COLOR_SPACES[canonical]
        logger.warning("ICC profile for %s was not found; falling back to sRGB.", info.name)
        return _srgb_profile_bytes()
    return None


def is_profile_available(key: object) -> bool:
    canonical = normalize_color_space_key(key)
    if canonical == "sRGB":
        return _srgb_profile_bytes() is not None
    return get_icc_profile_path(canonical) is not None


def profile_presence_report() -> dict[str, dict[str, object]]:
    return {
        key: {
            "name": info.name,
            "available": is_profile_available(key),
            "path": get_icc_profile_path(key),
        }
        for key, info in COLOR_SPACES.items()
    }


def missing_profile_message(key: object) -> str:
    info = COLOR_SPACES[normalize_color_space_key(key)]
    names = ", ".join(info.profile_names[:4])
    dirs = "\n".join(f"- {path}" for path in get_profile_search_directories())
    return (
        f"{info.name} ICC profile was not found.\n\n"
        f"Expected names include: {names}\n\n"
        f"SASpro checks these locations:\n{dirs}\n\n"
        "Display and export tagging will fall back to sRGB until the profile is installed."
    )


def _qcolor_space_from_named(info: ColorSpaceInfo) -> Optional[QColorSpace]:
    enum = getattr(QColorSpace, "NamedColorSpace", None)
    if enum is None:
        return None
    for name in info.qt_names:
        named_value = getattr(enum, name, None)
        if named_value is None:
            continue
        try:
            color_space = QColorSpace(named_value)
            if color_space.isValid():
                return color_space
        except Exception:
            continue
    return None


def _qcolor_space_from_icc(info: ColorSpaceInfo) -> Optional[QColorSpace]:
    if not hasattr(QColorSpace, "fromIccProfile"):
        return None
    data = get_icc_profile_bytes(info.key, fallback_to_srgb=False)
    if not data:
        return None
    try:
        color_space = QColorSpace.fromIccProfile(data)
        if color_space.isValid():
            return color_space
    except Exception:
        return None
    return None


@functools.lru_cache(maxsize=8)
def _cached_qcolor_space(key: str) -> Optional[QColorSpace]:
    info = COLOR_SPACES[normalize_color_space_key(key)]
    return _qcolor_space_from_named(info) or _qcolor_space_from_icc(info)


def get_color_space_from_key(key: object) -> Optional[QColorSpace]:
    return _cached_qcolor_space(normalize_color_space_key(key))


def get_color_space(qt_name) -> Optional[QColorSpace]:
    try:
        color_space = QColorSpace(qt_name)
        if color_space.isValid():
            return color_space
    except Exception:
        return None
    return None


def get_default_color_space() -> Optional[QColorSpace]:
    return get_color_space_from_key(DEFAULT_COLOR_SPACE)


def _settings_or_default(settings: Optional[QSettings]) -> QSettings:
    return settings if settings is not None else QSettings()


def get_working_color_space_from_settings(settings: Optional[QSettings] = None) -> str:
    value = _settings_or_default(settings).value(SETTINGS_KEY, DEFAULT_COLOR_SPACE, type=str)
    return normalize_color_space_key(value)


def get_working_color_space_key(settings: Optional[QSettings] = None) -> str:
    return get_working_color_space_from_settings(settings)


def set_working_color_space_to_settings(key: object, settings: Optional[QSettings] = None) -> bool:
    canonical = normalize_color_space_key(key)
    if canonical not in COLOR_SPACES:
        return False
    target = _settings_or_default(settings)
    target.setValue(SETTINGS_KEY, canonical)
    target.sync()
    return True


def normalize_viewport_color_mode(mode: object) -> str:
    raw_value = _clean_key(mode)
    if raw_value in COLOR_SPACES:
        return raw_value
    collapsed = raw_value.strip().lower().replace("-", "_").replace(" ", "_")
    if collapsed in ("legacy", "unmanaged", "unmanaged_legacy", "unmanaged_(legacy)"):
        return VIEWPORT_MODE_UNMANAGED
    normalized_color_key = normalize_color_space_key(raw_value)
    if normalized_color_key in COLOR_SPACES and raw_value:
        return normalized_color_key
    if collapsed in ("tagged", "working", "working_profile", "qt", "qt_tagged"):
        return VIEWPORT_MODE_TAGGED
    return DEFAULT_VIEWPORT_MODE


def get_viewport_color_mode_from_settings(settings: Optional[QSettings] = None) -> str:
    value = _settings_or_default(settings).value(
        VIEWPORT_MODE_SETTINGS_KEY,
        DEFAULT_VIEWPORT_MODE,
        type=str,
    )
    return normalize_viewport_color_mode(value)


def set_viewport_color_mode_to_settings(mode: object, settings: Optional[QSettings] = None) -> bool:
    canonical = normalize_viewport_color_mode(mode)
    target = _settings_or_default(settings)
    target.setValue(VIEWPORT_MODE_SETTINGS_KEY, canonical)
    target.sync()
    return True


def normalize_rendering_intent(intent: object) -> str:
    raw = str(intent or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in ("perceptual", "perceptive"):
        return RENDERING_INTENT_PERCEPTUAL
    if raw in ("relative", "relative_colorimetric", "relative_colourimetric", "colorimetric", "colourimetric"):
        return RENDERING_INTENT_RELATIVE
    return DEFAULT_RENDERING_INTENT


def get_softproof_rendering_intent_from_settings(settings: Optional[QSettings] = None) -> str:
    value = _settings_or_default(settings).value(
        SOFTPROOF_INTENT_SETTINGS_KEY,
        DEFAULT_RENDERING_INTENT,
        type=str,
    )
    return normalize_rendering_intent(value)


def set_softproof_rendering_intent_to_settings(intent: object, settings: Optional[QSettings] = None) -> bool:
    canonical = normalize_rendering_intent(intent)
    target = _settings_or_default(settings)
    target.setValue(SOFTPROOF_INTENT_SETTINGS_KEY, canonical)
    target.sync()
    return True


def get_softproof_black_point_compensation_from_settings(settings: Optional[QSettings] = None) -> bool:
    return _settings_or_default(settings).value(SOFTPROOF_BPC_SETTINGS_KEY, True, type=bool)


def set_softproof_black_point_compensation_to_settings(enabled: bool, settings: Optional[QSettings] = None) -> bool:
    target = _settings_or_default(settings)
    target.setValue(SOFTPROOF_BPC_SETTINGS_KEY, bool(enabled))
    target.sync()
    return True


def qimage_should_be_tagged_for_viewport(settings: Optional[QSettings] = None) -> bool:
    return get_viewport_color_mode_from_settings(settings) != VIEWPORT_MODE_UNMANAGED


def prepare_qimage_for_viewport(img: QImage, settings: Optional[QSettings] = None) -> QImage:
    if img is None:
        return img
    try:
        if img.isNull():
            return img
    except Exception:
        return img

    viewport_mode = get_viewport_color_mode_from_settings(settings)
    if viewport_mode == VIEWPORT_MODE_UNMANAGED:
        return img

    try:
        img = img.copy()
    except Exception:
        pass

    source_key = get_working_color_space_from_settings(settings)
    source_space = get_color_space_from_key(source_key)
    if source_space is None:
        return img

    try:
        img.setColorSpace(source_space)
    except Exception as exc:
        logger.warning("Failed to set source color space on viewport QImage: %s", exc)
        return img

    if viewport_mode == VIEWPORT_MODE_TAGGED:
        return img

    target_space = get_color_space_from_key(viewport_mode)
    if target_space is None:
        return img

    try:
        if source_space != target_space:
            img.convertToColorSpace(target_space)
        else:
            img.setColorSpace(target_space)
    except Exception as exc:
        logger.warning("Failed to convert viewport QImage color space: %s", exc)
    return img


def _qimage_to_rgb_array(img: QImage) -> np.ndarray:
    converted = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = converted.width(), converted.height()
    ptr = converted.bits()
    ptr.setsize(h * converted.bytesPerLine())
    raw = np.frombuffer(ptr, dtype=np.uint8).reshape((h, converted.bytesPerLine()))
    return raw[:, : w * 3].reshape((h, w, 3)).copy()


def _rgb_array_to_qimage(rgb: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _imagecms_intent(intent: object):
    from PIL import ImageCms

    canonical = normalize_rendering_intent(intent)
    enum = getattr(ImageCms, "Intent", None)
    if enum is not None:
        if canonical == RENDERING_INTENT_PERCEPTUAL:
            return enum.PERCEPTUAL
        return enum.RELATIVE_COLORIMETRIC
    if canonical == RENDERING_INTENT_PERCEPTUAL:
        return getattr(ImageCms, "INTENT_PERCEPTUAL", 0)
    return getattr(ImageCms, "INTENT_RELATIVE_COLORIMETRIC", 1)


def _imagecms_flags(black_point_compensation: bool):
    if not black_point_compensation:
        return 0
    from PIL import ImageCms

    flags = getattr(ImageCms, "Flags", None)
    if flags is not None:
        return flags.BLACKPOINTCOMPENSATION
    return getattr(ImageCms, "FLAGS_BLACKPOINTCOMPENSATION", 0)


def _imagecms_profile_from_key(key: object):
    from PIL import ImageCms

    data = get_icc_profile_bytes(key, fallback_to_srgb=True)
    if data:
        return ImageCms.ImageCmsProfile(BytesIO(data))
    if normalize_color_space_key(key) == "sRGB":
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    return None


def _imagecms_profile_from_qcolorspace(color_space: QColorSpace | None):
    from PIL import ImageCms

    if color_space is None or not color_space.isValid():
        return None
    try:
        data = bytes(color_space.iccProfile())
    except Exception:
        data = b""
    if not data:
        return None
    try:
        return ImageCms.ImageCmsProfile(BytesIO(data))
    except Exception:
        return None


def convert_qimage_with_lcms(
    img: QImage,
    source_key: object,
    target_profile,
    *,
    rendering_intent: object = DEFAULT_RENDERING_INTENT,
    black_point_compensation: bool = True,
) -> QImage:
    if img is None:
        return img
    try:
        if img.isNull():
            return img
    except Exception:
        return img

    try:
        from PIL import Image, ImageCms

        if isinstance(source_key, QColorSpace):
            source_profile = _imagecms_profile_from_qcolorspace(source_key)
        else:
            source_profile = _imagecms_profile_from_key(source_key)
        if source_profile is None:
            return img.copy()
        if isinstance(target_profile, QColorSpace):
            target = _imagecms_profile_from_qcolorspace(target_profile)
        elif isinstance(target_profile, (str, Path)) and Path(str(target_profile)).exists():
            target = ImageCms.ImageCmsProfile(str(target_profile))
        else:
            target = _imagecms_profile_from_key(target_profile)
        if target is None:
            return img.copy()

        rgb = _qimage_to_rgb_array(img)
        pil_img = Image.fromarray(rgb, "RGB")
        converted = ImageCms.profileToProfile(
            pil_img,
            source_profile,
            target,
            outputMode="RGB",
            renderingIntent=_imagecms_intent(rendering_intent),
            flags=_imagecms_flags(black_point_compensation),
        )
        qimg = _rgb_array_to_qimage(np.asarray(converted, dtype=np.uint8))
        qtarget = get_color_space_from_key(target_profile) if not isinstance(target_profile, QColorSpace) else target_profile
        if qtarget is not None and qtarget.isValid():
            qimg.setColorSpace(qtarget)
        return qimg
    except Exception as exc:
        logger.warning("LittleCMS QImage conversion failed: %s", exc)
        return img.copy()


def convert_float_rgb_with_lcms(
    arr: np.ndarray,
    source_key: object,
    target_key: object,
    *,
    rendering_intent: object = DEFAULT_RENDERING_INTENT,
    black_point_compensation: bool = True,
) -> np.ndarray:
    rgb8 = np.ascontiguousarray((np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    img = _rgb_array_to_qimage(rgb8)
    converted = convert_qimage_with_lcms(
        img,
        source_key,
        target_key,
        rendering_intent=rendering_intent,
        black_point_compensation=black_point_compensation,
    )
    return _qimage_to_rgb_array(converted).astype(np.float32) / 255.0


def tag_qimage_with_color_space(img: QImage, color_space_key: object) -> bool:
    if img is None:
        return False
    try:
        if img.isNull():
            return False
    except Exception:
        return False

    color_space = get_color_space_from_key(color_space_key)
    if color_space is None:
        return False
    try:
        img.setColorSpace(color_space)
        return True
    except Exception as exc:
        logger.warning("Failed to set color space on QImage: %s", exc)
        return False


def tag_qimage_with_working_color_space(img: QImage, settings: Optional[QSettings] = None) -> QImage:
    return prepare_qimage_for_viewport(img, settings)


def get_qimage_color_space(img: QImage) -> Optional[str]:
    try:
        color_space = img.colorSpace()
        if not color_space.isValid():
            return None
        for key in COLOR_SPACES:
            known = get_color_space_from_key(key)
            if known is not None and color_space == known:
                return key
        description = color_space.description().lower()
        for key, info in COLOR_SPACES.items():
            if info.name.lower() in description or key.lower() in description:
                return key
    except Exception:
        return None
    return None


class ColorSpaceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._settings = QSettings()
        self._initialized = True

    def get_working_space_key(self) -> str:
        return get_working_color_space_from_settings(self._settings)

    def set_working_space_key(self, key: object) -> bool:
        return set_working_color_space_to_settings(key, self._settings)

    def get_viewport_color_mode(self) -> str:
        return get_viewport_color_mode_from_settings(self._settings)

    def set_viewport_color_mode(self, mode: object) -> bool:
        return set_viewport_color_mode_to_settings(mode, self._settings)

    def get_working_space_qcolor_space(self) -> Optional[QColorSpace]:
        return get_color_space_from_key(self.get_working_space_key())

    def get_icc_export_key(self) -> str:
        return get_icc_key_from_color_space_key(self.get_working_space_key())

    def tag_qimage(self, img: QImage) -> bool:
        return tag_qimage_with_color_space(img, self.get_working_space_key())