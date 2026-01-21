#################################################################
# This Gemini generated script for Seti Astro by Fred Danowski
# Variable Star Flux Tracker (Aperture Photometry)
#
# Extracts time-series flux data for a target star 
# normalized against a reference star from a single image.
#
#################################################################

import numpy as np
import traceback
import sys
import os

def run(ctx):
    """
    Performs aperture photometry on a target star and a reference star.
    The script assumes the image is a single FITS frame and will output 
    the raw and normalized flux (light curve data) to the console.
    
    *** IMPORTANT: This script currently only supports single-frame analysis. 
    For multi-frame analysis, the script would need to loop through a directory 
    of images, which requires manual setup of file paths. ***
    """
    
    SCRIPT_NAME = "Variable Star Flux Tracker"
    ctx.log(f"--- Starting {SCRIPT_NAME} ---")

    # !!! CONFIGURATION SECTION: EDIT THESE VALUES !!!
    # NOTE: Coordinates are (Y, X) [row, column] based on image array indexing (0,0 is top-left)
    # -----------------------------------------------
    # TARGET STAR PARAMETERS:
    TARGET_Y = 500      # Y-coordinate (row) of the target star center
    TARGET_X = 500      # X-coordinate (column) of the target star center
    
    # REFERENCE STAR PARAMETERS:
    REFERENCE_Y = 150   # Y-coordinate (row) of the reference star center
    REFERENCE_X = 800   # X-coordinate (column) of the reference star center
    
    # APERTURE SETTINGS (Apply to both stars):
    APERTURE_RADIUS = 5     # Radius of the inner circle to sum the star's light
    BACKGROUND_INNER = 10   # Inner radius for the background annulus
    BACKGROUND_OUTER = 15   # Outer radius for the background annulus
    # -----------------------------------------------

    def calculate_flux(data, y_center, x_center, aperture_r, bg_inner_r, bg_outer_r):
        """Calculates the net flux for a single star."""
        H, W = data.shape
        
        # 1. Create coordinate grid centered at the star
        Y, X = np.ogrid[:H, :W]
        R = np.sqrt((Y - y_center)**2 + (X - x_center)**2)
        
        # 2. Define masks
        aperture_mask = R < aperture_r
        bg_mask = (R >= bg_inner_r) & (R < bg_outer_r)
        
        # 3. Calculate Background (BG) per pixel
        bg_pixels = np.sum(bg_mask)
        if bg_pixels == 0:
            return 0.0, 0.0 # Cannot calculate background, return flux and background per pixel
        
        bg_sum = np.sum(data * bg_mask)
        bg_per_pixel = bg_sum / bg_pixels
        
        # 4. Calculate Total Flux within Aperture
        total_flux_sum = np.sum(data * aperture_mask)
        aperture_pixels = np.sum(aperture_mask)
        
        # 5. Calculate Net Flux (Subtract background contribution)
        net_flux = total_flux_sum - (aperture_pixels * bg_per_pixel)
        
        return net_flux, bg_per_pixel


    try:
        # 1. Get the active image data
        # FIX: Changed 'get_current_image_data' to 'get_image_for(view_name_or_uid)'
        try:
            # Attempt to get the name of the active document from the context
            active_name = ctx.current_document_name 
            ctx.log(f"Attempting to fetch data for active image: {active_name}")
        except AttributeError:
            # Fallback to the known image name from previous logs
            active_name = "New_M27_ star recomposition result.fit" 
            ctx.log(f"Using hardcoded image name as fallback: {active_name}")
            
        image_data = ctx.get_image_for(active_name)
        
        if image_data is None:
            ctx.log(f"CRITICAL ERROR: Failed to get image data for '{active_name}'.")
            ctx.log("Please ensure an image is loaded and selected before running.")
            return

        # Ensure data is converted to single channel (luminance) if it's RGB
        if image_data.ndim == 3:
            ctx.log("Input is RGB. Converting to Monochromatic Luminance for photometry.")
            # Simple luminance conversion: 0.299*R + 0.587*G + 0.114*B
            data = 0.299 * image_data[:, :, 0] + 0.587 * image_data[:, :, 1] + 0.114 * image_data[:, :, 2]
        else:
            data = image_data.copy()

        # 2. Perform Photometry
        
        # Calculate flux for the target star
        target_flux, target_bg = calculate_flux(data, TARGET_Y, TARGET_X, 
                                                APERTURE_RADIUS, BACKGROUND_INNER, BACKGROUND_OUTER)
        
        # Calculate flux for the reference star
        reference_flux, reference_bg = calculate_flux(data, REFERENCE_Y, REFERENCE_X, 
                                                      APERTURE_RADIUS, BACKGROUND_INNER, BACKGROUND_OUTER)

        # 3. Calculate Normalized Flux and Magnitude (Relative)
        if reference_flux <= 0:
            ctx.log("ERROR: Reference star flux is non-positive. Check coordinates or image quality.")
            normalized_flux = 0.0
        else:
            normalized_flux = target_flux / reference_flux
        
        # Relative Magnitude Difference (dm = -2.5 * log10(F_target / F_ref))
        if normalized_flux > 0:
            relative_magnitude = -2.5 * np.log10(normalized_flux)
        else:
            relative_magnitude = 99.99 # Placeholder for invalid magnitude

        
        # 4. Report Results to Console (CSV Format for easy export)
        
        ctx.log("=========================================================================")
        ctx.log(f"   {SCRIPT_NAME} RESULTS (Single Frame Analysis)   ")
        ctx.log("=========================================================================")
        ctx.log("Target Star Configuration:")
        ctx.log(f"  Center (Y, X): ({TARGET_Y}, {TARGET_X}) | Aperture Radius: {APERTURE_RADIUS} pixels")
        ctx.log(f"  Background Annulus: {BACKGROUND_INNER}-{BACKGROUND_OUTER} pixels")
        ctx.log("-------------------------------------------------------------------------")
        
        # Output Header
        ctx.log("CSV_Output_Start:")
        ctx.log("Target Name,Ref Name,Aperture Radius,Target Flux,Reference Flux,Normalized Flux (Target/Ref),Relative Magnitude")
        
        # Output Data Row
        ctx.log(f"TARGET_STAR,REF_STAR,{APERTURE_RADIUS},{target_flux:.4f},{reference_flux:.4f},{normalized_flux:.4f},{relative_magnitude:.4f}")
        ctx.log("CSV_Output_End")

        ctx.log("-------------------------------------------------------------------------")
        ctx.log("NOTE: Copy the CSV data above to a spreadsheet for light curve plotting.")
        ctx.log(f"--- {SCRIPT_NAME} Finished Successfully ---")

    except Exception as e:
        ctx.log(f"--- {SCRIPT_NAME} FAILED ---")
        ctx.log(f"Error: {type(e).__name__}: {str(e)}")
        ctx.log(traceback.format_exc())
        ctx.log("------------------------------------------")