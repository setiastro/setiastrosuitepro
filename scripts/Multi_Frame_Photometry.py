#################################################################
# This Gemini generated script for Seti Astro by Fred Danowski
# Multi-Frame Variable Star Flux Tracker (Aperture Photometry)
#
# Extracts time-series flux data for a target star 
# normalized against a reference star from a sequence of images 
# in a designated directory.
#
#################################################################

import numpy as np
import traceback
import sys
import os

# Placeholder/Hypothetical function for loading FITS data from a path.
# IMPORTANT: This must be replaced with the actual function provided by 
# the Seti Astro context (ctx) or by using an external library like astropy.
# We are assuming 'ctx' can load raw data when provided the file path.
def load_image_data(ctx, file_path):
    """
    ATTENTION: This is a placeholder function! 
    You must replace the code inside this function with the correct 
    Seti Astro API call or a FITS reader (e.g., astropy.io.fits)
    to load the raw image array from the file_path.
    """
    try:
        # Example of a hypothetical API call that might work:
        return ctx.load_image_data_from_path(file_path)
    except Exception as e:
        ctx.log(f"ERROR: Failed to load FITS data from path: {file_path}. {e}")
        return None

def run(ctx):
    """
    Performs differential aperture photometry on all FITS images in a directory.
    Outputs a light curve (time series of magnitude changes) to the console.
    """
    
    SCRIPT_NAME = "Multi-Frame Variable Star Flux Tracker"
    ctx.log(f"--- Starting {SCRIPT_NAME} ---")

    # !!! CONFIGURATION SECTION: EDIT THESE VALUES !!!
    # -----------------------------------------------
    # Directory Path: The folder containing your sequence of FITS images.
    # NOTE: You MUST set this to a valid path on your system.
    IMAGE_DIRECTORY_PATH = "D:/Seti_Astro_Working/Variable_Star_Sequence/" # <--- SET PATH HERE

    # TARGET STAR PARAMETERS (Y, X) [row, column]:
    TARGET_Y = 500      
    TARGET_X = 500      
    
    # REFERENCE STAR PARAMETERS (Y, X) [row, column]:
    REFERENCE_Y = 150   
    REFERENCE_X = 800   
    
    # APERTURE SETTINGS (Apply to both stars):
    APERTURE_RADIUS = 5     
    BACKGROUND_INNER = 10   
    BACKGROUND_OUTER = 15   
    # -----------------------------------------------

    def calculate_flux(data, y_center, x_center, aperture_r, bg_inner_r, bg_outer_r):
        """Calculates the net flux for a single star."""
        H, W = data.shape
        Y, X = np.ogrid[:H, :W]
        R = np.sqrt((Y - y_center)**2 + (X - x_center)**2)
        
        aperture_mask = R < aperture_r
        bg_mask = (R >= bg_inner_r) & (R < bg_outer_r)
        
        bg_pixels = np.sum(bg_mask)
        if bg_pixels == 0:
            return 0.0, 0.0
        
        bg_sum = np.sum(data * bg_mask)
        bg_per_pixel = bg_sum / bg_pixels
        
        total_flux_sum = np.sum(data * aperture_mask)
        aperture_pixels = np.sum(aperture_mask)
        
        net_flux = total_flux_sum - (aperture_pixels * bg_per_pixel)
        
        return net_flux, bg_per_pixel

    try:
        # 1. Directory and File Check
        if not os.path.isdir(IMAGE_DIRECTORY_PATH):
            ctx.log(f"CRITICAL ERROR: Directory not found: {IMAGE_DIRECTORY_PATH}")
            return

        # Filter for FITS files and sort them chronologically (by filename)
        all_files = sorted([f for f in os.listdir(IMAGE_DIRECTORY_PATH) if f.lower().endswith(('.fit', '.fits'))])
        
        if not all_files:
            ctx.log(f"CRITICAL ERROR: No FITS files found in directory: {IMAGE_DIRECTORY_PATH}")
            return

        ctx.log(f"Found {len(all_files)} FITS images to process in: {IMAGE_DIRECTORY_PATH}")
        
        results_list = []
        
        # 2. Process each image in the sequence
        for filename in all_files:
            file_path = os.path.join(IMAGE_DIRECTORY_PATH, filename)
            
            # Load image data using the custom function/placeholder
            image_data = load_image_data(ctx, file_path) 
            
            if image_data is None:
                ctx.log(f"SKIPPING: Failed to load data for {filename}. Check the 'load_image_data' function.")
                continue
                
            # Convert to Monochromatic Luminance if RGB (same as single-frame script)
            if image_data.ndim == 3:
                data = 0.299 * image_data[:, :, 0] + 0.587 * image_data[:, :, 1] + 0.114 * image_data[:, :, 2]
            else:
                data = image_data.copy()

            # Ensure data is float32 for consistent calculations
            data = data.astype(np.float32)

            # 3. Perform Photometry
            
            # Calculate flux for the target star
            target_flux, _ = calculate_flux(data, TARGET_Y, TARGET_X, 
                                            APERTURE_RADIUS, BACKGROUND_INNER, BACKGROUND_OUTER)
            
            # Calculate flux for the reference star
            reference_flux, _ = calculate_flux(data, REFERENCE_Y, REFERENCE_X, 
                                                APERTURE_RADIUS, BACKGROUND_INNER, BACKGROUND_OUTER)

            # 4. Calculate Normalized Flux and Magnitude (Relative)
            normalized_flux = 0.0
            relative_magnitude = 99.99
            
            if reference_flux > 0 and target_flux > 0:
                normalized_flux = target_flux / reference_flux
                relative_magnitude = -2.5 * np.log10(normalized_flux)
            elif reference_flux <= 0:
                ctx.log(f"WARNING: Ref star flux <= 0 in {filename}. Skipping magnitude calculation.")
            
            # Store results
            results_list.append({
                'Filename': filename,
                'Target_Flux': target_flux,
                'Reference_Flux': reference_flux,
                'Normalized_Flux': normalized_flux,
                'Relative_Magnitude': relative_magnitude
            })
            ctx.log(f"Processed: {filename}. Mag: {relative_magnitude:.4f}")


        # 5. Report Final Results (Light Curve)
        
        ctx.log("=========================================================================")
        ctx.log(f"   {SCRIPT_NAME} FINAL LIGHT CURVE RESULTS   ")
        ctx.log("=========================================================================")
        ctx.log(f"Target Center: ({TARGET_Y}, {TARGET_X}) | Reference Center: ({REFERENCE_Y}, {REFERENCE_X})")
        ctx.log(f"Aperture: {APERTURE_RADIUS} px | Background: {BACKGROUND_INNER}-{BACKGROUND_OUTER} px")
        ctx.log("-------------------------------------------------------------------------")
        
        # Output Header
        ctx.log("CSV_Output_Start:")
        ctx.log("Filename,Target_Flux,Reference_Flux,Normalized_Flux,Relative_Magnitude")
        
        # Output Data Rows
        for res in results_list:
            ctx.log(f"{res['Filename']},{res['Target_Flux']:.4f},{res['Reference_Flux']:.4f},{res['Normalized_Flux']:.4f},{res['Relative_Magnitude']:.4f}")
            
        ctx.log("CSV_Output_End")

        ctx.log("-------------------------------------------------------------------------")
        ctx.log(f"--- {SCRIPT_NAME} Finished Processing {len(results_list)} Frames ---")

    except Exception as e:
        ctx.log(f"--- {SCRIPT_NAME} FAILED ---")
        ctx.log(f"Error: {type(e).__name__}: {str(e)}")
        ctx.log(traceback.format_exc())
        ctx.log("------------------------------------------")