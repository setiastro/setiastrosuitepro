# ops/write_build_info.py
import os
import datetime
import textwrap
import pathlib

UTC = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

out_dir = pathlib.Path("pro/_generated")
out_dir.mkdir(parents=True, exist_ok=True)

content = textwrap.dedent(f"""\
    # Auto-generated at build time. Do not edit.
    BUILD_TIMESTAMP = "{UTC}"
""")

(out_dir / "build_info.py").write_text(content, encoding="utf-8")
print("Wrote pro/_generated/build_info.py with BUILD_TIMESTAMP =", UTC)
