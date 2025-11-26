############################################
# This Gemini generated script by Fred Danowski is based on Cyril Richard Siril script
# Mono_Preprocessing v1.4
#
########### PREPROCESSING SCRIPT ###########
#
# Script for mono camera preprocessing
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
from astropy.io import fits
from astropy.stats import sigma_clip
import astroalign as aa

def run(ctx):
    """
    Mono Preprocessing Pipeline for SASpro.
    Creates Master Bias, Dark, and Flat. Calibrates, aligns, and stacks lights.
    
    Requires subfolders: biases/, darks/, flats/, lights/
    """
    
    # --- Configuration ---
    BASE_DIR = "D:/Seti_Astro_Working"
    CLIP_SIGMA = 3.0  # Sigma for rejection during stacking (3 sigma)
    
    ctx.log("--- Starting Mono Preprocessing Pipeline ---")

    # Helper function to load, stack (median/mean), and save master frames
    def process_master(folder_name, stack_method='median', norm=False):
        dir_path = os.path.join(BASE_DIR, folder_name)
        files = sorted(glob.glob(os.path.join(dir_path, "*.fit*")))
        
        if not files:
            ctx.log(f"Warning: No files found in '{folder_name}/'. Skipping master creation.")
            return None, None

        ctx.log(f"Processing {len(files)} files for Master {folder_name}...")
        stack_list = []

        for f in files:
            try:
                with fits.open(f) as hdul:
                    # Assume data is in the first HDU
                    data = hdul[0].data.astype(np.float32)
                    # Normalize large data range if necessary (e.g., 16-bit to 0-1)
                    if data.max() > 1.0: data /= data.max()
                    stack_list.append(data)
            except Exception as e:
                ctx.log(f"Error loading {os.path.basename(f)}: {e}")
                continue

        if not stack_list: return None, None
        
        stack_arr = np.array(stack_list)
        
        # Sigma Clipping: clip 3-sigma outliers from the time dimension
        clipped_data = sigma_clip(stack_arr, sigma=CLIP_SIGMA, axis=0)
        
        if stack_method == 'median':
            master_frame = np.ma.median(clipped_data, axis=0).data
        else: # mean
            master_frame = np.ma.mean(clipped_data, axis=0).data

        if norm:
            # Normalization (needed for flats)
            mean_val = np.mean(master_frame)
            if mean_val > 0:
                master_frame /= mean_val

        ctx.log(f"Master {folder_name} created. Shape: {master_frame.shape}")
        return master_frame, stack_arr[0].shape

    # 1. Master Bias
    MB, _ = process_master("biases", stack_method='median', norm=False)

    # 2. Master Dark
    MD, light_shape = process_master("darks", stack_method='median', norm=False)

    # 3. Master Flat (Requires Bias subtraction first)
    flat_dir = os.path.join(BASE_DIR, "flats")
    flat_files = sorted(glob.glob(os.path.join(flat_dir, "*.fit*")))
    calibrated_flats = []
    
    ctx.log(f"Calibrating {len(flat_files)} Flat frames...")
    
    for f in flat_files:
        try:
            with fits.open(f) as hdul:
                flat_data = hdul[0].data.astype(np.float32)
                if flat_data.max() > 1.0: flat_data /= flat_data.max()
                
                # Apply: (Flat - MB) / Mean(Flat - MB)
                if MB is not None:
                    cal_flat = flat_data - MB
                else:
                    cal_flat = flat_data
                    
                mean_val = np.mean(cal_flat)
                if mean_val > 0:
                    cal_flat /= mean_val
                    
                calibrated_flats.append(cal_flat)
        except Exception as e:
            ctx.log(f"Error calibrating flat {os.path.basename(f)}: {e}")
            continue

    if not calibrated_flats:
        MF = None
    else:
        # Stack Calibrated Flats
        flat_stack = np.array(calibrated_flats)
        MF_clipped = sigma_clip(flat_stack, sigma=CLIP_SIGMA, axis=0)
        MF = np.ma.median(MF_clipped, axis=0).data
        ctx.log("Master Flat (MF) created.")
        
    # --- Light Calibration, Alignment, and Stacking ---
    lights_dir = os.path.join(BASE_DIR, "lights")
    light_files = sorted(glob.glob(os.path.join(lights_dir, "*.fit*")))
    aligned_lights = []

    if not light_files:
        ctx.log("Error: No light frames found to process.")
        return
        
    # Use the first light frame as alignment reference
    ref_data = None
    
    ctx.log(f"Calibrating and Aligning {len(light_files)} Light frames...")
    
    for i, f in enumerate(light_files):
        try:
            with fits.open(f) as hdul:
                light_data = hdul[0].data.astype(np.float32)
                if light_data.max() > 1.0: light_data /= light_data.max()

                # 4. Calibrate Light (CL = (Light - MD) / MF)
                CL = light_data.copy()
                if MD is not None:
                    CL -= MD
                
                # Flat field correction (division)
                if MF is not None:
                    CL /= MF

                # Set reference frame if not set
                if ref_data is None:
                    ref_data = CL
                    aligned_lights.append(CL)
                    ctx.log("Reference image set.")
                    continue
                
                # Align (astroalign requires 2D input for star detection)
                aligned_img, _ = aa.register(CL, ref_data)
                aligned_lights.append(aligned_img)

        except Exception as e:
            ctx.log(f"Error processing {os.path.basename(f)}: {e}")
            continue

    # 5. Final Stack
    if not aligned_lights:
        ctx.log("Stacking failed: No lights were successfully calibrated/aligned.")
        return

    stack_arr = np.array(aligned_lights)
    ctx.log(f"Stacking {len(stack_arr)} aligned images...")
    
    # Sigma clip and Mean/Median stack
    final_clipped = sigma_clip(stack_arr, sigma=CLIP_SIGMA, axis=0)
    final_image = np.ma.median(final_clipped, axis=0).data
    
    # 6. Load into SASpro
    final_image = np.clip(final_image, 0.0, 1.0)
    name = f"Mono_Stacked_{len(aligned_lights)}subs"
    ctx.open_new_document(final_image, name=name)
    ctx.log("Pipeline complete. Stacked image loaded.")