################################################################## 
# This Gemini generated script for Seti Astro by Fred Danowski
# Spectral Line Analyzer (FWHM)
#
# Analyzes a spectral profile and calculates the Full Width at Half Maximum (FWHM).
#
##################################################################

import numpy as np
import traceback
import sys
import os

# Placeholder for a simple Gaussian fit function
def gaussian(x, a, mu, sigma, offset):
    """Gaussian function model."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset

def run(ctx):
    """
    Analyzes a designated spectral line in the active image.
    The script assumes the spectral line is centered vertically and requires 
    a rectangular selection or manually defined area for analysis.
    
    The results are printed to the Script Console/Log.
    """
    
    SCRIPT_NAME = "Spectral Line Analyzer"
    ctx.log(f"--- Starting {SCRIPT_NAME} ---")

    # !!! CONFIGURATION SECTION: EDIT THESE VALUES !!!
    # -----------------------------------------------
    # PROFILE_Y_START and PROFILE_Y_END: The vertical span (rows) to average for the spectral profile.
    # Adjust these values based on where your spectrum is located vertically in the image.
    PROFILE_Y_START = 100 
    PROFILE_Y_END = 200 
    
    # PEAK_FINDER_SMOOTHING: Number of pixels to smooth the profile before peak detection (e.g., 5).
    PEAK_FINDER_SMOOTHING = 5
    # -----------------------------------------------

    try:
        # 1. Get the active image data
        # FIX: Changed 'get_current_image_data' to 'get_image_for(view_name_or_uid)' to resolve errors.
        try:
            # Attempt to get the name of the active document from the context
            active_name = ctx.current_document_name 
            ctx.log(f"Attempting to fetch data for active image: {active_name}")
        except AttributeError:
            # Fallback if 'ctx.current_document_name' doesn't exist
            # Note: This fallback relies on the image "New_M27_ star recomposition result.fit" being loaded.
            active_name = "New_M27_ star recomposition result.fit" 
            ctx.log(f"Using hardcoded image name as fallback: {active_name}")
            
        image_data = ctx.get_image_for(active_name)
        
        if image_data is None:
            ctx.log(f"CRITICAL ERROR: Failed to get image data for '{active_name}'.")
            ctx.log("Please ensure an image is loaded and selected before running.")
            return

        # Convert to single channel (luminance) if it's RGB
        if image_data.ndim == 3:
            ctx.log("Input is RGB. Converting to Monochromatic Luminance for analysis.")
            # Simple luminance conversion: 0.299*R + 0.587*G + 0.114*B
            luminance = 0.299 * image_data[:, :, 0] + 0.587 * image_data[:, :, 1] + 0.114 * image_data[:, :, 2]
        else:
            luminance = image_data

        data = luminance.astype(np.float32)
        H, W = data.shape
        
        # Check configuration bounds
        if PROFILE_Y_START < 0 or PROFILE_Y_END > H or PROFILE_Y_START >= PROFILE_Y_END:
            ctx.log(f"ERROR: Invalid PROFILE_Y_START ({PROFILE_Y_START}) or PROFILE_Y_END ({PROFILE_Y_END}) for image height {H}.")
            return

        # 2. Extract Spectral Profile
        # Average pixel values horizontally (along the x-axis) over the specified vertical range
        ctx.log(f"Extracting spectral profile by averaging rows {PROFILE_Y_START} to {PROFILE_Y_END}.")
        spectral_area = data[PROFILE_Y_START:PROFILE_Y_END, :]
        profile = np.mean(spectral_area, axis=0) # Average along the vertical axis
        x_values = np.arange(W)
        
        # 3. Smooth the profile (optional but helps for peak finding)
        if PEAK_FINDER_SMOOTHING > 1:
            window = np.ones(PEAK_FINDER_SMOOTHING) / PEAK_FINDER_SMOOTHING
            profile_smoothed = np.convolve(profile, window, mode='same')
        else:
            profile_smoothed = profile
            
        # 4. Find Peak (Line Center)
        peak_index = np.argmax(profile_smoothed)
        peak_intensity = profile_smoothed[peak_index]
        median_bg = np.median(profile_smoothed)
        net_peak_intensity = peak_intensity - median_bg
        
        if net_peak_intensity <= 0:
            ctx.log("Analysis failed: Could not find a line peak significantly above the background.")
            return

        # 5. Calculate Full Width at Half Maximum (FWHM)
        # Half-Max intensity level relative to the background
        half_max_level = median_bg + (net_peak_intensity / 2.0)
        
        # Search outwards from the peak index to find the points where the profile crosses the Half-Max level
        
        # Search Left (Blue/Short Wavelength) Side
        idx_L = peak_index
        while idx_L >= 0 and profile_smoothed[idx_L] > half_max_level:
            idx_L -= 1
        
        # Search Right (Red/Long Wavelength) Side
        idx_R = peak_index
        while idx_R < W and profile_smoothed[idx_R] > half_max_level:
            idx_R += 1

        # Calculate FWHM in pixels
        fwhm_pixels = idx_R - idx_L
        
        ctx.log("=========================================")
        ctx.log(f"   {SCRIPT_NAME} RESULTS   ")
        ctx.log("=========================================")
        ctx.log(f"Profile Extracted (Y={PROFILE_Y_START} to {PROFILE_Y_END})")
        ctx.log(f"Line Center (Pixel): {peak_index}")
        ctx.log(f"Peak Intensity (Net): {net_peak_intensity:.2f}")
        ctx.log("-----------------------------------------")
        ctx.log(f"Calculated FWHM: {fwhm_pixels} pixels")
        ctx.log("--- Lower FWHM indicates a sharper spectral line ---")
        ctx.log("=========================================")
        
        ctx.log(f"--- {SCRIPT_NAME} Finished Successfully ---")

    except Exception as e:
        ctx.log(f"--- {SCRIPT_NAME} FAILED ---")
        ctx.log(f"Error: {type(e).__name__}: {str(e)}")
        ctx.log(traceback.format_exc())
        ctx.log("------------------------------------------")