# SASpro user script
SCRIPT_NAME = "IO Test"
SCRIPT_GROUP = "Samples"

import numpy as np

def run(ctx):
    #base = ctx.get_image()
    img, hdr, bit, mono = ctx.load_image(r"C:\Users\Gaming\Downloads\swisstransfer_bad3f320-976a-4d60-b009-55681ac1f136\Frank\Master Lights\masterLight_BIN-1_9576x6388_EXPOSURE-180.00s_FILTER-L_mono_autocrop_-0.1 mm.xisf")

    ctx.open_new_document(img, metadata={"file_path":"test.xisf"}, name="test from disk")