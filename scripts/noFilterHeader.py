from __future__ import annotations
# SASpro user script
SCRIPT_NAME = "Header No Filter"
SCRIPT_GROUP = "User"

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QFileDialog
from astropy.io import fits
import os
from pathlib import Path

"""
This script adds a FILTER with name 'NoF' to the header of a batch of FITS files.
It is designed for SasPro users using OSC cameras without filters.
It annoyed me that the default use of the Stacking Suite would append "Unknown" to all 
the stacking suite lists.  NoF is slightly cleaner, IMHO, and expresses the true
situation succinctly.

A top-level directory below which all your fits files are stored may be specified in your environment
with variable name ASTRO_DIR. If this is specified, the file dialog will start here.  Otherwise, it 
will start in your home directory.

User selects a file, a group of files.  No recursion is done.

This script could easily be modified to perform other header mods.

Steve Cohen
"""



class FITSBatchModify(QWidget): 
    def __init__(self, ctx, parent, keyword, value, comment, files=None, modifyExisting=True):
        super().__init__(parent)
        self.ctx = ctx
        self.keyword=keyword
        self.value=value
        self.comment=comment
        self.modifyExisting=modifyExisting

    def pick_files(self):
        astrodir = os.getenv('ASTRO_DIR') 
        if not astrodir:
            astrodir = str(Path.home())
        print(f'looking for FITS files starting in {astrodir}') 
        files, _ = QFileDialog.getOpenFileNames(self, "Select FITS files", astrodir , "FITS files (*.fits *.fit *.fts *.fz)")#
        #print(files)
        print(f'{len(files)} files will be processed.')
        if not files:
            return
        self.files = files


    def run(self):
        n_ok = 0
        n_err = 0
        self.pick_files()
        if self.files:
            for fp in self.files:
                try:
                    with fits.open(fp, mode='update', memmap=False) as hdul:
                        targets = range(len(hdul)) 
                        for i in targets:
                            hdr = hdul[i].header
                            hdr[self.keyword] = (self.value, self.comment)
                        hdul.flush()
                        n_ok += 1
                except Exception as e:
                    print(f"[Batch FITS] Error on {fp}: {e}")
                    n_err += 1
            print(f"{self.value} added as {self.keyword} to {n_ok} files with {n_err} errors")
        else:
            print("No files selected")

def run(ctx):
    """
    SASpro entry point.
    """
    try:
        # modify to use different parameter names, values, comments.
        m = FITSBatchModify(ctx, ctx.app, "FILTER", "NoF", "No header")
        print("FITSBatchModify initialized")
        m.run()
        if m.files:
            print('run complete')
    except Exception as e:
        print(f'exception:{e}')



