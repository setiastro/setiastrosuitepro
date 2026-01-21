##############################################################################################
# This Gemini generated script for Seti Astro by Fred Danowski is based on the Carlo Mollicone - AstroBOH Siril script
##############################################################################################


import numpy as np

def run(ctx):
    """
    Hubble Palette Simulator.
    Takes active RGB image (OSC), extracts R,G,B and remixes them
    into a synthetic SHO palette.
    """
    img = ctx.get_image()
    if img is None:
        ctx.log("No active image found.")
        return

    if img.ndim != 3 or img.shape[2] != 3:
        ctx.log("Active image must be Color (RGB).")
        return

    ctx.log("Processing Hubble Palette (Classic Formula)...")

    # 1. Extract Channels (Normalized 0-1)
    # SASpro images are usually HxWx3. 
    # R=0, G=1, B=2
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    # 2. Define Mapping (Classic Formula from source)
    # Ha is Red channel
    HA = R
    
    # OIII is average of Green and Blue
    OIII = (G + B) * 0.5
    
    # Synthetic SII is average of Ha and OIII
    S2 = (HA + OIII) * 0.5

    # 3. Create SHO mapping
    # Sii -> Red
    # Ha  -> Green
    # Oiii -> Blue
    
    # Initialize output array
    h, w, c = img.shape
    sho_img = np.zeros((h, w, 3), dtype=np.float32)
    
    sho_img[:,:,0] = S2    # R channel gets S2
    sho_img[:,:,1] = HA    # G channel gets Ha
    sho_img[:,:,2] = OIII  # B channel gets OIII

    # 4. Normalize result
    sho_img = np.clip(sho_img, 0.0, 1.0)

    # 5. Open as new document
    ctx.open_new_document(sho_img, name="Hubble_Simulation_SHO")
    ctx.log("Created new document: Hubble_Simulation_SHO")