# ops/prime_matplotlib_cache.py
import os
import sys
from pathlib import Path

def resolve_mpl_cfg():
    """
    Decide where to put the Matplotlib cache:
    - If running from a frozen app:
        * macOS:   <App>.app/Contents/Resources/mpl_config
        * Windows: <dist>\setiastrosuitepro\mpl_config (next to exe)
        * Linux:   <dist>/setiastrosuitepro/mpl_config (next to exe)
    - If running from repo (not frozen), try common post-PyInstaller outputs:
        * macOS:   dist/SetiAstroSuitePro.app/Contents/Resources/mpl_config
        * Win/Linux one-folder: dist/setiastrosuitepro/mpl_config
    - If env MPLCACHE_TARGET is provided, use that exactly.
    """
    # Explicit override wins
    env_target = os.environ.get("MPLCACHE_TARGET")
    if env_target:
        return Path(env_target)

    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        if sys.platform == "darwin":
            # .../SetiAstroSuitePro.app/Contents/MacOS/SetiAstroSuitePro
            contents = exe.parent.parent  # MacOS -> Contents
            return contents / "Resources" / "mpl_config"
        else:
            # Windows/Linux one-folder: put next to the exe folder
            return exe.parent / "mpl_config"

    # Not frozen: we're in the repo
    repo_root = Path(__file__).resolve().parents[1]

    mac_app_cfg = repo_root / "dist" / "SetiAstroSuitePro.app" / "Contents" / "Resources" / "mpl_config"
    win_folder_cfg = repo_root / "dist" / "setiastrosuitepro" / "mpl_config"
    # Fallback: capitalized folder name some specs produce
    alt_folder_cfg = repo_root / "dist" / "SetiAstroSuitePro" / "mpl_config"

    for p in (mac_app_cfg, win_folder_cfg, alt_folder_cfg):
        if p.parents[1].exists():  # parent of mpl_config dir exists (the bundle/folder)
            return p

    # Last resort: create under dist
    return repo_root / "dist" / "mpl_config"

mpl_cfg = resolve_mpl_cfg()
mpl_cfg.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cfg)

# Use a non-GUI backend while priming
import matplotlib
matplotlib.use("Agg", force=True)

from matplotlib import font_manager

# Build/refresh the cache using public APIs across MPL versions
try:
    fm = font_manager.get_font_manager()  # MPL >= 3.6
except AttributeError:
    fm = font_manager.FontManager()       # older MPL

# Resolve a common font; on newer MPL this can rebuild if missing
try:
    font_manager.findfont("DejaVu Sans", rebuild_if_missing=True)
except TypeError:
    _ = font_manager.FontManager()  # older MPL fallback

print("Matplotlib font cache primed into:", mpl_cfg)
