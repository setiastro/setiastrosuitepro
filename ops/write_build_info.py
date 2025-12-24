# ops/write_build_info.py
import datetime
import pathlib
import sys
import textwrap

def _read_poetry_version(pyproject_path: pathlib.Path) -> str | None:
    try:
        import tomllib  # py311+
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except Exception:
        try:
            import tomli  # py310
            data = tomli.loads(pyproject_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    v = (
        data.get("tool", {})
            .get("poetry", {})
            .get("version", None)
    )
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None

def main():
    UTC = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    pyproject = repo_root / "pyproject.toml"
    version = _read_poetry_version(pyproject) or "0.0.0"

    out_dir = repo_root / "src" / "setiastro" / "saspro" / "_generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    content = textwrap.dedent(f"""\
    # Auto-generated at build time. Do not edit.
    BUILD_TIMESTAMP = "{UTC}"
    APP_VERSION = "{version}"
    """)
    (out_dir / "build_info.py").write_text(content, encoding="utf-8")
    print("Wrote", out_dir / "build_info.py", "APP_VERSION =", version, "BUILD_TIMESTAMP =", UTC)
    return 0

if __name__ == "__main__":
    sys.exit(main())
