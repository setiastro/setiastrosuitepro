#
###############################################################################################
# This Gemini generated script for Seti Astro by Fred Danowski is based on the script by Cyril Richard from Franklin Marek SAS Siril script
###############################################################################################
#
import numpy as np

def run(ctx):
    """
    NB to RGB Composer.
    Looks for open documents named 'Ha.fits', 'OIII.fits', and 'SII.fits'.
    Combines them into an RGB image.
    """
    
    # 1. Find the documents by title
    views = ctx.list_image_views()
    docs = {title: doc for title, doc in views}
    
    # --- FIX APPLIED HERE: Added '.fits' extension to match view names in the log. ---
    ha_doc = docs.get('Ha.fits')
    oiii_doc = docs.get('OIII.fits')
    sii_doc = docs.get('SII.fits')
    # ---------------------------------------------------------------------------------

    if not ha_doc or not oiii_doc:
        ctx.log("Error: Could not find views named 'Ha.fits' and 'OIII.fits'.")
        ctx.log("Please ensure your open views are named 'Ha.fits', 'OIII.fits', (and optionally 'SII.fits').")
        ctx.log(f"Open views found: {list(docs.keys())}")
        return

    ctx.log("Found narrowband source images.")

    # 2. Get Data
    # Assuming float32 [0,1]
    ha = ha_doc.image
    oiii = oiii_doc.image
    sii = sii_doc.image if sii_doc else None

    # Ensure dimensions match
    if ha.shape != oiii.shape:
        ctx.log("Error: Ha and OIII dimensions do not match.")
        return

    # Handle Mono inputs (H, W) vs (H, W, 1)
    if ha.ndim == 3: ha = ha[:,:,0]
    if oiii.ndim == 3: oiii = oiii[:,:,0]
    if sii is not None and sii.ndim == 3: sii = sii[:,:,0]

    # 3. Apply Composition Logic (SHO Style)
    # Defaulting to SHO mapping
    # R = SII (or Ha if SII missing)
    # G = Ha
    # B = OIII
    
    h, w = ha.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    if sii is not None:
        ctx.log("Composing SHO Palette...")
        out[:,:,0] = sii  # R
        out[:,:,1] = ha   # G
        out[:,:,2] = oiii # B
    else:
        ctx.log("Composing HOO Palette (No SII found)...")
        # HOO mapping
        out[:,:,0] = ha   # R
        out[:,:,1] = oiii # G
        out[:,:,2] = oiii # B

    # 4. Optional: Apply SCNR-style Green reduction (common in NB)
    # Simple neutral average SCNR
    # If Green > max(Red, Blue), clamp Green to max(Red, Blue)
    # Uncomment below to enable:
    # max_rb = np.maximum(out[:,:,0], out[:,:,2])
    # mask = out[:,:,1] > max_rb
    # out[mask, 1] = max_rb[mask]

    # 5. Output
    out = np.clip(out, 0.0, 1.0)
    name = "SHO_Composition" if sii is not None else "HOO_Composition"
    ctx.open_new_document(out, name=name)