################################################################## 
# This Gemini generated script for Seti Astro by Fred Danowski
# Star Field Quality Analyzer (Half-Flux Radius)
#
# Provides objective metrics (HFR) for image focus and seeing.
#
##################################################################

import numpy as np
import traceback
import sys
import os

def run(ctx):
    """
    Analyzes the sharpness of stars in the active image by calculating the 
    Half-Flux Radius (HFR) for a sample of the brightest stars.
    HFR is a robust metric for focus and seeing quality.
    
    The results are printed to the Script Console/Log.
    """
    
    SCRIPT_NAME = "Star Field Quality Analyzer"
    ctx.log(f"--- Starting {SCRIPT_NAME} ---")

    # !!! CONFIGURATION SECTION: EDIT THESE VALUES !!!
    # -----------------------------------------------
    # THRESHOLD_FACTOR: Determines how bright a peak must be relative to the median 
    # to be considered a star. (e.g., 20 means 20x brighter than the median background).
    THRESHOLD_FACTOR = 5.0
    
    # BOX_SIZE: The size of the square cutout (postage stamp) around each star peak.
    BOX_SIZE = 25 
    
    # MAX_STARS: Maximum number of stars to analyze (speeds up processing).
    MAX_STARS = 100
    # -----------------------------------------------

    try:
        # 1. Get the active image data
        # FIX: The method requires the 'view_name_or_uid'. We assume the active
        # document name can be retrieved via ctx.current_document_name or similar.
        # Since 'ctx.current_document_name' is a common API pattern for the active view, we will try it.
        try:
            active_name = ctx.current_document_name
            ctx.log(f"Attempting to fetch data for active image: {active_name}")
        except AttributeError:
            # Fallback if the above property doesn't exist
            active_name = "New_M27_ star recomposition result.fit"
            ctx.log(f"Using hardcoded image name as fallback: {active_name}")

        image_data = ctx.get_image_for(active_name)
        
        if image_data is None:
            ctx.log(f"CRITICAL ERROR: Failed to get image data for '{active_name}'.")
            ctx.log("Please ensure the image is loaded and selected.")
            return

        # Convert to single channel (luminance) if it's RGB
        if image_data.ndim == 3:
            # Simple luminance conversion: 0.299*R + 0.587*G + 0.114*B
            ctx.log("Input is RGB. Converting to Monochromatic Luminance for analysis.")
            luminance = 0.299 * image_data[:, :, 0] + 0.587 * image_data[:, :, 1] + 0.114 * image_data[:, :, 2]
        else:
            luminance = image_data

        # Ensure data is standardized and float32
        data = luminance.astype(np.float32)
        H, W = data.shape
        
        # 2. Peak Detection (Finding Star Centers)
        median_bg = np.median(data)
        ctx.log(f"Background Median Value: {median_bg:.4f}")
        
        # Create a mask for areas brighter than the threshold
        threshold = median_bg * THRESHOLD_FACTOR
        bright_mask = data > threshold
        
        # Simple centroid finder on the brightest pixels
        y_coords, x_coords = np.where(bright_mask)
        
        # Group close centroids to find the true peak
        # This is a simplified approach to avoid complex grouping/filtering
        potential_peaks = []
        for y, x in zip(y_coords, x_coords):
            # Check if this pixel is the local maximum within a 5x5 area
            y_start, y_end = max(0, y-2), min(H, y+3)
            x_start, x_end = max(0, x-2), min(W, x+3)
            if data[y, x] == np.max(data[y_start:y_end, x_start:x_end]):
                # Add to potential peaks if not too close to the edge
                if BOX_SIZE // 2 < x < W - BOX_SIZE // 2 and BOX_SIZE // 2 < y < H - BOX_SIZE // 2:
                    potential_peaks.append((y, x, data[y, x]))

        # Sort by brightness and take the top N unique stars
        potential_peaks.sort(key=lambda x: x[2], reverse=True)
        unique_stars = []
        
        # Filter out closely clustered peaks
        star_radius = BOX_SIZE // 2
        for peak in potential_peaks:
            is_new = True
            for unique in unique_stars: 
                # Calculate distance between peaks
                dist = np.sqrt((peak[0] - unique[0])**2 + (peak[1] - unique[1])**2)
                if dist < star_radius: # If peaks are too close, skip
                    is_new = False
                    break
            if is_new and len(unique_stars) < MAX_STARS:
                unique_stars.append(peak)
        
        ctx.log(f"Found {len(unique_stars)} suitable stars for analysis.")

        if not unique_stars:
            ctx.log("Analysis failed: No stars found above the threshold. Try lowering THRESHOLD_FACTOR.")
            return
            
        # 3. HFR (Half-Flux Radius) Calculation
        hfr_results = []
        box_half = BOX_SIZE // 2
        
        for y_peak, x_peak, peak_value in unique_stars:
            # Extract postage stamp (sub-image)
            postage = data[y_peak-box_half:y_peak+box_half+1, x_peak-box_half:x_peak+box_half+1]
            
            # The background level is assumed to be the minimum in the postage stamp
            # (Crude, but avoids using advanced background modeling)
            local_bg = np.min(postage) 
            
            # Star's net flux (relative to local background)
            star_net_flux = np.sum(postage - local_bg)
            
            # Half-Flux Threshold
            half_flux_level = star_net_flux / 2.0
            
            # Find the radius containing half the star's net flux (HFR)
            current_flux = 0.0
            radius = 0
            max_radius = np.ceil(np.sqrt(2) * box_half) # Diagonal distance to corner
            
            # Iterate outwards from the center until half the flux is captured
            for r in np.arange(1.0, max_radius, 0.1): # check radius in small steps
                r_int = int(np.ceil(r))
                # Create a circular mask to sum pixels within the radius 'r'
                Y, X = np.ogrid[-box_half:box_half+1, -box_half:box_half+1]
                mask = X*X + Y*Y <= r*r
                
                # Sum the flux inside the current radius
                current_flux = np.sum((postage - local_bg) * mask)
                
                if current_flux >= half_flux_level:
                    hfr_results.append(r)
                    break
                
        # 4. Report Results
        avg_hfr = np.mean(hfr_results)
        min_hfr = np.min(hfr_results)
        max_hfr = np.max(hfr_results)
        
        ctx.log("=========================================")
        ctx.log(f"   {SCRIPT_NAME} RESULTS   ")
        ctx.log("=========================================")
        ctx.log(f"Stars Analyzed: {len(hfr_results)}")
        ctx.log(f"Average Half-Flux Radius (HFR): {avg_hfr:.2f} pixels")
        ctx.log("-----------------------------------------")
        ctx.log(f"Best Focus (Min HFR): {min_hfr:.2f} pixels")
        ctx.log(f"Worst Focus (Max HFR): {max_hfr:.2f} pixels")
        ctx.log("--- Lower HFR is better focus/seeing ---")
        ctx.log("=========================================")
        
        ctx.log(f"--- {SCRIPT_NAME} Finished Successfully ---")

    except Exception as e:
        ctx.log(f"--- {SCRIPT_NAME} FAILED ---")
        ctx.log(f"Error: {type(e).__name__}: {str(e)}")
        ctx.log(traceback.format_exc())
        ctx.log("------------------------------------------")