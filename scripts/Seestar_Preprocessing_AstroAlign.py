# This Gemini generated script by Fred Danowski is based on the Cyril Richard Siril script
#
# SeeStar_Preprocessing v1.3
#
########### PREPROCESSING SCRIPT ###########
#
# Script for Seestar Deep Sky images where only 
# lights are needed.
#
# Please, REMOVE all jpg files from the
# directory.
#
# If you find that too many images are discarded
# before stacking, you can increase the value after
# -filter-round= in the seqapplyreg command, line 47
# Before making this change, you must make a copy of
# this script, place it in another folder, and enter the path
# to that folder under Scripts Storage Directories in the
# Get Scripts sections. If you don't do this, and modify
# the original script, it will be overwritten the next time Siril is started.
#
# Needs 1 set of RAW images in the working
# directory, within 1 directory:
#   lights/
#
############################################

import os
import glob
import numpy as np
import sys
import traceback
from astropy.io import fits
from astropy.stats import sigma_clip
import astroalign # <-- Key: Used for aligning non-WCS images via star-matching

def run(ctx):
    """
    Seestar Preprocessing Pipeline (1x AstroAlign Stack)
    
    Uses astroalign for robust 1x alignment of frames that are missing WCS headers.
    
    *** IMPORTANT: SET YOUR BASE_DIR PATH BELOW TO YOUR DATA FOLDER ***
    """
    
    # --- Configuration ---
    
    # !!! CHANGE THIS LINE: Set the absolute path to the directory 
    # !!! that CONTAINS your 'lights/' subfolder.
    BASE_DIR = "D:/Seti_Astro_Working" 
    
    CLIP_SIGMA = 3.0
    
    ctx.log("--- Starting Seestar 1x AstroAlign Stack Pipeline ---")
    
    try:
        lights_dir = os.path.join(BASE_DIR, "lights")
        
        # 1. Load Data
        files = sorted(glob.glob(os.path.join(lights_dir, "*.fit*")))
        if not files:
            ctx.log(f"CRITICAL ERROR: No .fit/.fits files found in: {lights_dir}")
            return
            
        input_images = []
        ctx.log(f"Found {len(files)} files. Starting load and preparation...")
        
        for i, f in enumerate(files):
            try:
                with fits.open(f) as hdul:
                    light_data = hdul[0].data.astype(np.float32)
                    # Keep 1-channel Bayer data
                    if light_data.ndim == 3: light_data = light_data[:,:,0] 
                    if light_data.max() > 1.0: light_data /= light_data.max()
                    
                    input_images.append(light_data)
            except Exception as e:
                ctx.log(f"File Read Error for {os.path.basename(f)}: {e}")
                continue

        if len(input_images) < 2:
            ctx.log("CRITICAL ERROR: Stack failed. Less than 2 valid images found.")
            return
            
        # 2. Perform Alignment (Astroalign) and Stacking
        
        # Set the first image as the reference frame
        reference_image = input_images[0]
        aligned_images = [reference_image]
        
        ctx.log(f"Starting AstroAlign Registration of {len(input_images)} images...")
        
        # Align all other images to the reference image
        for i in range(1, len(input_images)):
            target_image = input_images[i]
            ctx.log(f"Aligning image {i+1} of {len(input_images)}...")
            
            # Astroalign aligns the target to the reference (1x resolution)
            aligned_img, _ = astroalign.register(target_image, reference_image)
            aligned_images.append(aligned_img)

        # 3. Median Stack
        stack_arr = np.array(aligned_images)
        
        # Sigma clip and Median stack (np.ma.median ignores NaNs and clipped values)
        final_clipped = sigma_clip(stack_arr, sigma=CLIP_SIGMA, axis=0)
        final_image_mono = np.ma.median(final_clipped, axis=0).data
        
        # Fill any remaining NaNs (edges/corners) with 0.0
        final_image_mono = np.nan_to_num(final_image_mono)
        
        # 4. Load into SASpro
        final_image_mono = np.clip(final_image_mono, 0.0, 1.0)
        
        name = f"Seestar_1xAstroAlign_Mono_{len(input_images)}subs"
        ctx.open_new_document(final_image_mono, name=name)
        
        # 5. Debayer the final image
        ctx.log("Final Debayering (RGB conversion)...")
        ctx.run_command('debayer') 
        ctx.log("Final image successfully debayered to RGB.")
            
        ctx.log("1x AstroAlign Stack Pipeline Finished SUCCESSFULLY.")

    except Exception as e:
        ctx.log(f"--- SCRIPT FAILED ---")
        ctx.log(f"An unexpected error occurred: {type(e).__name__}: {str(e)}")
        ctx.log("Full Traceback:")
        ctx.log(traceback.format_exc())
        ctx.log("-------------------")