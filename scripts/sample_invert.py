# Sample SASpro script
# Put scripts in this folder; they appear in Scripts menu.
# Required entrypoint:
#   def run(ctx):
#       ...

SCRIPT_NAME = "Invert Image (Sample)"
SCRIPT_GROUP = "Samples"

import numpy as np

def run(ctx):
    img = ctx.get_image()
    if img is None:
        ctx.log("No active image.")
        return

    ctx.log(f"Inverting image... shape={img.shape}, dtype={img.dtype}")

    f = img.astype(np.float32)
    mx = float(np.nanmax(f)) if f.size else 1.0
    if mx > 1.0:
        f = f / mx
    f = np.clip(f, 0.0, 1.0)

    out = 1.0 - f
    ctx.set_image(out, step_name="Invert via Script")
    ctx.log("Done.")
