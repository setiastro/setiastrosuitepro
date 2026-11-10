from __future__ import annotations

import os

import pytest

from setiastro.saspro import icon_packs, resources


@pytest.fixture
def packs(tmp_path, monkeypatch):
    """Point pack discovery and the saved selection at a temp folder."""
    user, bundled = tmp_path / "user", tmp_path / "bundled"
    monkeypatch.setattr(icon_packs, "user_packs_dir", lambda: user)
    monkeypatch.setattr(icon_packs, "_bundled_packs_dir", lambda: bundled)

    def select(name):
        monkeypatch.setattr(icon_packs, "get_saved_icon_pack", lambda: name)
        icon_packs._active_pack_index.cache_clear()

    select(icon_packs.DEFAULT_PACK)
    yield user, bundled, select
    icon_packs._active_pack_index.cache_clear()


def _add(root, relpath):
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


def _stock(name):
    return os.path.join(resources._get_base_path(), "images", name)


def test_default_uses_stock_icons(packs):
    user, _, _ = packs
    _add(user, "Neon/hdr.png")
    assert resources._resource_path("hdr.png") == _stock("hdr.png")


def test_pack_icon_found_in_category_subfolder(packs):
    user, _, select = packs
    icon = _add(user, "Neon/Tools/Deep/hdr.png")
    select("Neon")
    assert resources._resource_path("hdr.png") == icon


def test_partial_pack_falls_back_to_stock(packs):
    user, _, select = packs
    _add(user, "Neon/hdr.png")
    select("Neon")
    assert resources._resource_path("clahe.png") == _stock("clahe.png")


def test_missing_selected_pack_falls_back_to_stock(packs):
    _, _, select = packs
    select("Uninstalled")
    assert resources._resource_path("hdr.png") == _stock("hdr.png")


def test_png_and_svg_stand_in_for_each_other(packs):
    user, _, select = packs
    svg = _add(user, "Neon/hdr.svg")
    png = _add(user, "Neon/TextureClarity.png")
    select("Neon")
    assert resources._resource_path("hdr.png") == svg
    assert resources._resource_path("TextureClarity.svg") == png


def test_exact_extension_preferred_over_stand_in(packs):
    user, _, select = packs
    _add(user, "Neon/hdr.svg")
    png = _add(user, "Neon/hdr.png")
    select("Neon")
    assert resources._resource_path("hdr.png") == png


def test_only_png_and_svg_are_replaced(packs):
    user, _, select = packs
    _add(user, "Neon/astrosuitepro.png")
    _add(user, "Neon/astrosuitepro.ico")
    select("Neon")
    assert resources._resource_path("astrosuitepro.ico") == _stock("astrosuitepro.ico")


def test_user_pack_overrides_bundled_pack_of_same_name(packs):
    user, bundled, _ = packs
    _add(bundled, "Neon/hdr.png")
    _add(bundled, "Bundled Only/hdr.png")
    _add(user, "Neon/hdr.png")
    found = icon_packs.available_icon_packs()
    assert list(found) == ["Bundled Only", "Neon"]
    assert found["Neon"] == user / "Neon"


def test_read_failure_falls_back_to_stock(packs, monkeypatch):
    def boom():
        raise OSError("unreadable settings")

    monkeypatch.setattr(icon_packs, "get_saved_icon_pack", boom)
    icon_packs._active_pack_index.cache_clear()
    assert resources._resource_path("hdr.png") == _stock("hdr.png")


def _aurora():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "ops" / "build_icon_pack.py"
    spec = importlib.util.spec_from_file_location("build_icon_pack", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Path(resources._get_base_path()) / "images" / "icon_packs" / "Aurora"


def test_aurora_icons_all_replace_a_real_stock_icon():
    module, _ = _aurora()
    missing = [
        name for name in module.ICONS
        if not any(os.path.exists(_stock(name + ext)) for ext in (".png", ".svg"))
    ]
    assert missing == []


def test_aurora_bundled_files_match_generator():
    module, folder = _aurora()
    built = {p.relative_to(folder).as_posix() for p in folder.rglob("*.png")}
    assert built == {f"{cat}/{name}.png" for name, (cat, _) in module.ICONS.items()}
