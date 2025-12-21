############################################
#This Gemini generated script by Fred Danowski is based on Cyril Richard Siril script
# Preprocessing v1.4
#
########### PREPROCESSING SCRIPT ###########
#
# Script for color camera preprocessing
#
# Needs 4 sets of RAW images in the working
# directory, within 4 directories:
#   biases/
#   flats/
#   darks/
#   lights/
# Saves masters to ./masters/
#
############################################

import os
import glob
import numpy as np
import sys
import traceback
from astropy.io import fits
from astropy.stats import sigma_clip
import astroalign # Used for robust 1x alignment via star-matching

def run(ctx):
    """
    OSC Preprocessing Pipeline (1x AstroAlign Stack)
    
    This script performs full calibration (Bias/Dark/Flat) and then stacks 
    1-channel Bayer data at 1x resolution using Astroalign for alignment.
    
    *** IMPORTANT: SET YOUR BASE_DIR PATH BELOW TO YOUR DATA FOLDER ***
    """
    
    # --- Configuration ---
    
    # !!! CHANGE THIS LINE: Set the absolute path to the directory 
    # !!! that CONTAINS your subfolders (biases/, darks/, flats/, lights/).
    BASE_DIR = "D:/Seti_Astro_Working" 
    
    CLIP_SIGMA = 3.0
    
    ctx.log("--- Starting OSC 1x AstroAlign Stack Pipeline ---")
    
    try:
        # Helper function to process single-channel master frames
        def process_master(folder_name, stack_method='median', norm=False):
            dir_path = os.path.join(BASE_DIR, folder_name)
            files = sorted(glob.glob(os.path.join(dir_path, "*.fit*")))
            if not files: return None
            
            stack_list = []
            for f in files:
                try:
                    with fits.open(f) as hdul:
                        data = hdul[0].data.astype(np.float32)
                        # Keep 1-channel Bayer data
                        if data.ndim == 3: data = data[:,:,0] 
                        if data.max() > 1.0: data /= data.max()
                        stack_list.append(data)
                except Exception: continue

            if not stack_list: return None
            stack_arr = np.array(stack_list)
            clipped_data = sigma_clip(stack_arr, sigma=CLIP_SIGMA, axis=0)
            
            if stack_method == 'median':
                master_frame = np.ma.median(clipped_data, axis=0).data
            else:
                master_frame = np.ma.mean(clipped_data, axis=0).data

            if norm:
                mean_val = np.mean(master_frame)
                if mean_val > 0: master_frame /= mean_val
            return master_frame

        # 1. Master Frame Creation (Bias, Dark, Flat)
        ctx.log("Creating Master Calibration Frames...")
        MB = process_master("biases", stack_method='median', norm=False)
        MD = process_master("darks", stack_method='median', norm=False)
        
        flat_files = sorted(glob.glob(os.path.join(BASE_DIR, "flats", "*.fit*")))
        calibrated_flats = []
        for f in flat_files:
            try:
                with fits.open(f) as hdul:
                    flat_data = hdul[0].data.astype(np.float32)
                    if flat_data.ndim == 3: flat_data = flat_data[:,:,0]
                    if flat_data.max() > 1.0: flat_data /= flat_data.max()
                    cal_flat = flat_data.copy()
                    if MB is not None: cal_flat -= MB
                    mean_val = np.mean(cal_flat)
                    if mean_val > 0: cal_flat /= mean_val
                    calibrated_flats.append(cal_flat)
            except Exception: continue
            
        MF = None
        if calibrated_flats:
            flat_stack = np.array(calibrated_flats)
            MF_clipped = sigma_clip(flat_stack, sigma=CLIP_SIGMA, axis=0)
            MF = np.ma.median(MF_clipped, axis=0).data
            
        # --- Light Calibration and 1x AstroAlign Stacking ---
        lights_dir = os.path.join(BASE_DIR, "lights")
        light_files = sorted(glob.glob(os.path.join(lights_dir, "*.fit*")))
            
        if not light_files:
            ctx.log(f"Error: No light frames found in: {lights_dir}")
            return
            
        calibrated_lights = []
        
        # 2. Calibrate Light Frames
        ctx.log(f"Calibrating {len(light_files)} Light frames...")
        for f in light_files:
            try:
                with fits.open(f) as hdul:
                    light_data = hdul[0].data.astype(np.float32)
                    if light_data.ndim == 3: light_data = light_data[:,:,0]
                    if light_data.max() > 1.0: light_data /= light_data.max()

                    CL = light_data.copy()
                    if MD is not None: CL -= MD
                    if MF is not None: CL /= MF
                    
                    calibrated_lights.append(CL)
            except Exception: continue

        if len(calibrated_lights) < 2:
            ctx.log("Stacking failed: Less than 2 valid calibrated lights found.")
            return
            
        # 3. Perform Alignment (Astroalign) and Stacking
        
        # Set the first calibrated image as the reference frame
        reference_image = calibrated_lights[0]
        aligned_images = [reference_image]
        
        ctx.log(f"Starting AstroAlign Registration of {len(calibrated_lights)} images...")
        
        # Align all other images to the reference image
        for i in range(1, len(calibrated_lights)):
            target_image = calibrated_lights[i]
            ctx.log(f"Aligning image {i+1} of {len(calibrated_lights)}...")
            
            # Astroalign aligns the target to the reference (1x resolution)
            aligned_img, _ = astroalign.register(target_image, reference_image)
            aligned_images.append(aligned_img)
        
        # 4. Median Stack
        stack_arr = np.array(aligned_images)
        
        # Sigma clip and Median stack (np.ma.median ignores NaNs and clipped values)
        final_clipped = sigma_clip(stack_arr, sigma=CLIP_SIGMA, axis=0)
        final_image_mono = np.ma.median(final_clipped, axis=0).data
        
        # Fill any remaining NaNs (edges/corners) with 0.0
        final_image_mono = np.nan_to_num(final_image_mono)
        
        # 5. Load into SASpro
        final_image_mono = np.clip(final_image_mono, 0.0, 1.0)
        
        name = f"OSC_1xAstroAlign_Mono_{len(calibrated_lights)}subs"
        ctx.open_new_document(final_image_mono, name=name)
        
        # 6. Debayer the final image
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