"""
FITS_Sorter.py (SAS Pro) - Version 1.0
--------------------------------------
Sort FITS files into subdirectories based on header keywords
(filter, object, panel, exposure time, and custom keywords).

Creates organized folder structure within the source data directory.
"""

from __future__ import annotations
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QGroupBox, QGridLayout, QCheckBox, QLineEdit,
    QFileDialog, QTextEdit, QProgressBar, QComboBox
)
from PyQt6.QtCore import Qt

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

SCRIPT_NAME = "FITS Sorter"

# --- Header keyword variants for each property ---
OBJECT_KEYS = ["OBJECT", "OBJNAME", "TARGET", "OBJCTNAME"]
FILTER_KEYS = ["FILTER", "FILTER1", "FILTNAM1", "FILTER-1"]
EXPTIME_KEYS = ["EXPTIME", "EXPOSURE", "EXP_TIME", "EXPTM"]


def get_header_value(header, keys, default="Unknown"):
    """
    Look through a list of possible header keys and return the first
    that exists. Return 'default' if none are present.
    """
    for key in keys:
        if key in header:
            value = header[key]
            if isinstance(value, str):
                value = value.strip().replace(" ", "_").replace("/", "-")
            return str(value)
    return default


def get_custom_header_value(header, key, default="Unknown"):
    """Get a specific header keyword value."""
    if key in header:
        value = header[key]
        if isinstance(value, str):
            value = value.strip().replace(" ", "_").replace("/", "-")
        return str(value)
    return default


def extract_header_info(hdu, use_object=True, use_filter=True,
                        use_exptime=True, custom_keys=None):
    """
    Extract selected keywords from a FITS HDU header.
    Returns a dict of {keyword_name: value}.
    """
    header = hdu.header
    info = {}

    if use_object:
        info['object'] = get_header_value(header, OBJECT_KEYS, default="UnknownObject")

    if use_filter:
        info['filter'] = get_header_value(header, FILTER_KEYS, default="UnknownFilter")

    if use_exptime:
        exptime_str = get_header_value(header, EXPTIME_KEYS, default="UnknownExp")
        try:
            exptime_val = float(exptime_str)
            info['exptime'] = f"{int(round(exptime_val))}s"
        except (ValueError, TypeError):
            info['exptime'] = exptime_str

    if custom_keys:
        for key in custom_keys:
            key = key.strip().upper()
            if key:
                info[key.lower()] = get_custom_header_value(header, key, default=f"Unknown{key}")

    return info


def build_subdir_path(info, template_order):
    """Build subdirectory path from extracted info based on template order."""
    parts = []
    for key in template_order:
        if key in info:
            parts.append(info[key])
    return "/".join(parts) if parts else "Unsorted"


class FITSSorterDialog(QDialog):
    """Main dialog for FITS file sorting."""

    def __init__(self, ctx):
        super().__init__(parent=ctx.app if ctx else None)
        self.ctx = ctx
        self.setWindowTitle(SCRIPT_NAME)
        self.resize(700, 600)
        self._source_dir = None
        self._build_ui()

    def _build_ui(self):
        """Build the dialog UI."""
        root = QVBoxLayout(self)

        # === Source Directory Selection ===
        dir_box = QGroupBox("Source Directory")
        dir_layout = QHBoxLayout(dir_box)

        self.lbl_source = QLabel("No directory selected")
        self.lbl_source.setStyleSheet("color: gray;")
        dir_layout.addWidget(self.lbl_source, 1)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_directory)
        dir_layout.addWidget(btn_browse)

        root.addWidget(dir_box)

        # === Sorting Options ===
        options_box = QGroupBox("Sorting Keywords")
        options_layout = QGridLayout(options_box)

        # Checkboxes for standard keywords
        self.chk_object = QCheckBox("Object Name")
        self.chk_object.setChecked(True)
        self.chk_object.setToolTip("Sort by OBJECT, OBJNAME, TARGET keywords")
        options_layout.addWidget(self.chk_object, 0, 0)

        self.chk_filter = QCheckBox("Filter")
        self.chk_filter.setChecked(True)
        self.chk_filter.setToolTip("Sort by FILTER, FILTER1, FILTNAM1 keywords")
        options_layout.addWidget(self.chk_filter, 0, 1)

        self.chk_exptime = QCheckBox("Exposure Time")
        self.chk_exptime.setChecked(True)
        self.chk_exptime.setToolTip("Sort by EXPTIME, EXPOSURE keywords")
        options_layout.addWidget(self.chk_exptime, 1, 0)

        # Custom keywords input
        options_layout.addWidget(QLabel("Custom Keywords:"), 2, 0)
        self.txt_custom_keys = QLineEdit()
        self.txt_custom_keys.setPlaceholderText("e.g., GAIN, BINNING, CCD-TEMP (comma-separated)")
        self.txt_custom_keys.setToolTip(
            "Enter additional FITS header keywords to sort by.\n"
            "Separate multiple keywords with commas."
        )
        options_layout.addWidget(self.txt_custom_keys, 2, 1)

        root.addWidget(options_box)

        # === Folder Structure Order ===
        order_box = QGroupBox("Folder Structure Order")
        order_layout = QVBoxLayout(order_box)

        order_layout.addWidget(QLabel("Drag to reorder (top = outermost folder):"))

        # Simple combo-based ordering
        self.order_widgets = []
        order_grid = QGridLayout()

        labels = ["1st level:", "2nd level:", "3rd level:", "4th level:"]
        defaults = ["object", "filter", "exptime", "custom"]

        for i, (label, default) in enumerate(zip(labels, defaults)):
            order_grid.addWidget(QLabel(label), i, 0)
            combo = QComboBox()
            combo.addItems(["object", "filter", "exptime", "custom", "(skip)"])
            combo.setCurrentText(default if i < 3 else "(skip)")
            self.order_widgets.append(combo)
            order_grid.addWidget(combo, i, 1)

        order_layout.addLayout(order_grid)
        root.addWidget(order_box)

        # === Operation Mode ===
        mode_box = QGroupBox("Operation")
        mode_layout = QGridLayout(mode_box)

        self.chk_move = QCheckBox("Move files (instead of copy)")
        self.chk_move.setToolTip("If checked, files will be MOVED. Otherwise they are COPIED.")
        mode_layout.addWidget(self.chk_move, 0, 0)

        self.chk_recursive = QCheckBox("Include subdirectories")
        self.chk_recursive.setChecked(True)
        self.chk_recursive.setToolTip("Search for FITS files in subdirectories too")
        mode_layout.addWidget(self.chk_recursive, 0, 1)

        self.chk_collect_errors = QCheckBox("Collect errored files to _errors folder")
        self.chk_collect_errors.setChecked(True)
        self.chk_collect_errors.setToolTip(
            "If checked, files that fail to read will be copied/moved\n"
            "to a '_errors' subfolder for later review."
        )
        mode_layout.addWidget(self.chk_collect_errors, 1, 0, 1, 2)

        root.addWidget(mode_box)

        # === Output Log ===
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(150)
        log_layout.addWidget(self.txt_log)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        log_layout.addWidget(self.progress)

        root.addWidget(log_box)

        # === Buttons ===
        btn_layout = QHBoxLayout()

        btn_preview = QPushButton("Preview (Dry Run)")
        btn_preview.setToolTip("Show what would happen without moving/copying files")
        btn_preview.clicked.connect(self._run_preview)
        btn_layout.addWidget(btn_preview)

        btn_execute = QPushButton("Execute Sort")
        btn_execute.setToolTip("Actually sort the files")
        btn_execute.clicked.connect(self._run_execute)
        btn_layout.addWidget(btn_execute)

        btn_layout.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        root.addLayout(btn_layout)

    def _browse_directory(self):
        """Open directory browser."""
        start_dir = str(self._source_dir) if self._source_dir else str(Path.home())
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select FITS Source Directory", start_dir
        )
        if dir_path:
            self._source_dir = Path(dir_path)
            self.lbl_source.setText(str(self._source_dir))
            self.lbl_source.setStyleSheet("color: black;")
            self._log(f"Selected directory: {self._source_dir}")

            # Count FITS files
            pattern = "**/*.fits" if self.chk_recursive.isChecked() else "*.fits"
            fits_files = list(self._source_dir.glob(pattern))
            fits_files += list(self._source_dir.glob(pattern.replace(".fits", ".fit")))
            fits_files += list(self._source_dir.glob(pattern.replace(".fits", ".FITS")))
            fits_files += list(self._source_dir.glob(pattern.replace(".fits", ".FIT")))
            self._log(f"Found {len(fits_files)} FITS files")

    def _log(self, message):
        """Add message to log."""
        self.txt_log.append(message)

    def _get_template_order(self):
        """Get the folder structure order from combo boxes."""
        order = []
        for combo in self.order_widgets:
            val = combo.currentText()
            if val != "(skip)" and val not in order:
                order.append(val)
        return order

    def _get_custom_keys(self):
        """Parse custom keywords from input."""
        text = self.txt_custom_keys.text().strip()
        if not text:
            return []
        return [k.strip().upper() for k in text.split(",") if k.strip()]

    def _collect_fits_files(self):
        """Collect all FITS files from source directory."""
        if not self._source_dir:
            return []

        pattern_base = "**/" if self.chk_recursive.isChecked() else ""
        extensions = ["*.fits", "*.fit", "*.FITS", "*.FIT", "*.fts", "*.FTS"]

        files = []
        for ext in extensions:
            files.extend(self._source_dir.glob(pattern_base + ext))

        return list(set(files))  # Remove duplicates

    def _process_files(self, dry_run=True):
        """Process all FITS files."""
        if not HAS_ASTROPY:
            QMessageBox.critical(
                self, "Error",
                "astropy is required but not installed.\n\nInstall with: pip install astropy"
            )
            return

        if not self._source_dir:
            QMessageBox.warning(self, "Warning", "Please select a source directory first.")
            return

        fits_files = self._collect_fits_files()
        if not fits_files:
            QMessageBox.information(self, "Info", "No FITS files found in the selected directory.")
            return

        # Get options
        use_object = self.chk_object.isChecked()
        use_filter = self.chk_filter.isChecked()
        use_exptime = self.chk_exptime.isChecked()
        custom_keys = self._get_custom_keys()
        template_order = self._get_template_order()
        move_files = self.chk_move.isChecked()
        collect_errors = self.chk_collect_errors.isChecked()

        # Validate at least one sorting option
        if not any([use_object, use_filter, use_exptime, custom_keys]):
            QMessageBox.warning(self, "Warning", "Please select at least one sorting keyword.")
            return

        self.txt_log.clear()
        action = "MOVE" if move_files else "COPY"
        mode = "DRY RUN - " if dry_run else ""
        self._log(f"{mode}Processing {len(fits_files)} files ({action})...")
        self._log(f"Folder order: {' / '.join(template_order)}")
        self._log("-" * 50)

        self.progress.setVisible(True)
        self.progress.setMaximum(len(fits_files))
        self.progress.setValue(0)

        success_count = 0
        error_count = 0
        error_files = []

        for i, fpath in enumerate(fits_files):
            self.progress.setValue(i + 1)

            try:
                with fits.open(fpath, memmap=False, ignore_missing_simple=True) as hdul:
                    info = extract_header_info(
                        hdul[0],
                        use_object=use_object,
                        use_filter=use_filter,
                        use_exptime=use_exptime,
                        custom_keys=custom_keys
                    )
            except Exception as e:
                # Get file size for diagnostics
                try:
                    file_size = fpath.stat().st_size
                    size_str = f"{file_size / 1024 / 1024:.2f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.1f} KB"
                except:
                    size_str = "unknown size"

                error_type = type(e).__name__
                error_msg = str(e)

                # Provide more helpful error descriptions
                if "truncated" in error_msg.lower() or "unexpected end" in error_msg.lower():
                    hint = "(file appears truncated/incomplete)"
                elif "SIMPLE" in error_msg:
                    hint = "(missing FITS header)"
                elif "BITPIX" in error_msg:
                    hint = "(invalid/missing BITPIX keyword)"
                elif "permission" in error_msg.lower():
                    hint = "(permission denied)"
                elif "decode" in error_msg.lower() or "encoding" in error_msg.lower():
                    hint = "(encoding/binary read error)"
                else:
                    hint = ""

                self._log(f"[ERROR] {fpath.name} ({size_str})")
                self._log(f"        Type: {error_type} {hint}")
                self._log(f"        Details: {error_msg}")
                self._log(f"        Path: {fpath}")

                # Collect errored file to _errors folder
                if collect_errors:
                    errors_dir = fpath.parent / "_errors"
                    error_dest = errors_dir / fpath.name

                    # Handle filename conflicts
                    if error_dest.exists():
                        stem = fpath.stem
                        suffix = fpath.suffix
                        counter = 1
                        while error_dest.exists():
                            error_dest = errors_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                    if dry_run:
                        self._log(f"        -> Would {action.lower()} to: _errors/{error_dest.name}")
                    else:
                        try:
                            errors_dir.mkdir(parents=True, exist_ok=True)
                            if move_files:
                                shutil.move(str(fpath), str(error_dest))
                            else:
                                shutil.copy2(str(fpath), str(error_dest))
                            self._log(f"        -> Collected to: _errors/{error_dest.name}")
                        except Exception as copy_err:
                            self._log(f"        -> Failed to collect: {copy_err}")

                error_count += 1
                error_files.append(fpath.name)
                continue

            # Build subdirectory path
            subdir = build_subdir_path(info, template_order)

            # Create destination in same parent as source file
            dest_dir = fpath.parent / subdir
            dest_path = dest_dir / fpath.name

            # Skip if file is already in correct location
            if dest_path == fpath:
                self._log(f"[SKIP] {fpath.name} already in correct location")
                continue

            # Handle filename conflicts
            if dest_path.exists():
                stem = fpath.stem
                suffix = fpath.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            rel_dest = dest_path.relative_to(self._source_dir) if dest_path.is_relative_to(self._source_dir) else dest_path
            self._log(f"[{action}] {fpath.name} -> {rel_dest}")

            if not dry_run:
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    if move_files:
                        shutil.move(str(fpath), str(dest_path))
                    else:
                        shutil.copy2(str(fpath), str(dest_path))
                    success_count += 1
                except Exception as e:
                    self._log(f"[ERROR] Failed to {action.lower()} {fpath.name}: {e}")
                    error_count += 1
            else:
                success_count += 1

        self._log("-" * 50)
        self._log(f"Complete: {success_count} files processed, {error_count} errors")

        # List error files summary if any
        if error_files:
            self._log("")
            self._log(f"=== FILES WITH ERRORS ({len(error_files)}) ===")
            for ef in error_files:
                self._log(f"  - {ef}")
            self._log("")
            self._log("These files may be corrupted, truncated, or not valid FITS format.")

        self.progress.setVisible(False)

        if not dry_run:
            error_info = f"\n\n{error_count} files had errors (see log for details)" if error_count > 0 else ""
            QMessageBox.information(
                self, "Complete",
                f"Sorting complete!\n\n{success_count} files {action.lower()}d{error_info}"
            )

    def _run_preview(self):
        """Run in dry-run mode."""
        self._process_files(dry_run=True)

    def _run_execute(self):
        """Run actual sort."""
        if not self._source_dir:
            QMessageBox.warning(self, "Warning", "Please select a source directory first.")
            return

        action = "moved" if self.chk_move.isChecked() else "copied"
        reply = QMessageBox.question(
            self, "Confirm",
            f"This will sort FITS files in:\n{self._source_dir}\n\n"
            f"Files will be {action} into subdirectories.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._process_files(dry_run=False)


def run(ctx):
    """Entry point for SAS Pro."""
    dlg = FITSSorterDialog(ctx)
    dlg.exec()
