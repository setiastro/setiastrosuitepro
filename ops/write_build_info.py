# ops/write_build_info.py
import os
import datetime
import textwrap
import pathlib
import sys


def main():
    """Entry point for the write-build-info script."""
    UTC = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    
    # Update path to use the new package structure
    script_dir = pathlib.Path(__file__).parent.parent
    out_dir = script_dir / "src" / "setiastro" / "saspro" / "_generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    content = textwrap.dedent(f"""\
    # Auto-generated at build time. Do not edit.
    BUILD_TIMESTAMP = "{UTC}"
""")
    
    (out_dir / "build_info.py").write_text(content, encoding="utf-8")
    print("Wrote", out_dir / "build_info.py", "with BUILD_TIMESTAMP =", UTC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
