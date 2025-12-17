"""
Runtime hook for astroquery to prevent network downloads during import.
"""
import os
import sys
import tempfile
import json
from pathlib import Path

def setup_astroquery_offline():
    """Configure astroquery and astropy for offline operation."""
    
    # Disable astropy data downloads
    os.environ['ASTROPY_DOWNLOAD_CACHE_TIMEOUT'] = '0'
    os.environ['ASTROPY_DOWNLOAD_BLOCK_SIZE'] = '0'
    os.environ['ASTROQUERY_DOWNLOAD_CACHE_TIMEOUT'] = '0'
    
    # Set up cache directories
    if hasattr(sys, '_MEIPASS'):
        # Running in PyInstaller bundle
        cache_dir = os.path.join(tempfile.gettempdir(), 'astropy_cache')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['XDG_CACHE_HOME'] = cache_dir
        
        # Create the missing query_criteria_fields.json file
        try:
            # Create minimal data that astroquery.simbad needs
            query_criteria = {
                "basic": [
                    "MAIN_ID", "RA", "DEC", "COO_ERR_MAJA", "COO_ERR_MINA", 
                    "COO_ERR_ANGLE", "COO_QUAL", "COO_WAVELENGTH", "COO_BIBCODE"
                ],
                "ids": ["ID"],
                "biblio": ["BIBCODELIST"],
                "measurements": ["FLUX", "FLUXDATA"],
                "note": "Minimal data for offline PyInstaller bundle"
            }
            
            # Create in multiple possible locations
            locations = [
                os.path.join(tempfile.gettempdir(), 'astropy_data'),
                os.path.join(cache_dir, 'astropy', 'data'),
                os.path.join(cache_dir, 'downloads'),
            ]
            
            for loc in locations:
                os.makedirs(loc, exist_ok=True)
                query_file = os.path.join(loc, 'query_criteria_fields.json')
                with open(query_file, 'w') as f:
                    json.dump(query_criteria, f, indent=2)
                print(f"Created astroquery data file: {query_file}")
                
        except Exception as e:
            print(f"Warning: Could not create astroquery data files: {e}")

# Monkey patch astropy.utils.data.download_file BEFORE any imports
def patched_download_file(*args, **kwargs):
    """Patched download_file that returns a local fallback instead of downloading."""
    
    # If it's asking for query_criteria_fields.json, return our local version
    if len(args) > 0 and 'query_criteria_fields.json' in str(args[0]):
        fallback_locations = [
            os.path.join(tempfile.gettempdir(), 'astropy_data', 'query_criteria_fields.json'),
            os.path.join(tempfile.gettempdir(), 'astropy_cache', 'astropy', 'data', 'query_criteria_fields.json'),
        ]
        
        for fallback_file in fallback_locations:
            if os.path.exists(fallback_file):
                print(f"Using local astroquery data: {fallback_file}")
                return fallback_file
    
    # For other files, create a dummy file to prevent crashes
    dummy_file = os.path.join(tempfile.gettempdir(), 'astropy_dummy.json')
    if not os.path.exists(dummy_file):
        with open(dummy_file, 'w') as f:
            json.dump({"note": "Dummy file for offline operation"}, f)
    
    return dummy_file

# Apply patches before any astropy/astroquery imports
try:
    setup_astroquery_offline()
    
    # Patch the download function early
    import astropy.utils.data
    if not hasattr(astropy.utils.data, '_original_download_file'):
        astropy.utils.data._original_download_file = astropy.utils.data.download_file
        astropy.utils.data.download_file = patched_download_file
        print("✓ Patched astropy.utils.data.download_file for offline operation")
        
except Exception as e:
    print(f"Warning: Could not fully patch astropy for offline operation: {e}")
    # Continue anyway - the app might still work

print("✓ astroquery runtime hook applied")