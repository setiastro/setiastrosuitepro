#!/usr/bin/env python3
"""
NINA_ImageMetaData - user script for SetiAstroSuitePro to process NINA Image Metadata and filter Lights accordingly.

"""

SCRIPT_NAME = "NINA_ImageMetaData - Display NINA Image MetaData for Lights"
SCRIPT_GROUP = "Blink"
VERSION = "0.0.3"

import sys
from typing import Optional
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# Try Qt bindings first (do not create a new QApplication if one already exists)
def _import_qt():
    for mod in ("PySide6", "PyQt6", "PyQt5", "PySide2"):
        try:
            pkg = __import__(mod)
            # map to common names
            if mod.startswith("PySide"):
                from importlib import import_module

                QtWidgets = import_module(f"{mod}.QtWidgets")
                return QtWidgets
            else:
                from importlib import import_module

                QtWidgets = import_module(f"{mod}.QtWidgets")
                return QtWidgets
        except Exception:
            continue
    return None


def select_text_file() -> Optional[str]:
    """Select a .txt file using Qt file dialog when possible, otherwise fall back.

    This avoids creating a second AppKit application (which causes the macOS
    crash you observed when mixing tkinter with a running Qt application).
    """
    QtWidgets = _import_qt()
    if QtWidgets is not None:
        try:
            QApplication = QtWidgets.QApplication
            QFileDialog = QtWidgets.QFileDialog
            app = QApplication.instance()
            created_app = False
            if app is None:
                # Create a temporary QApplication for CLI/testing only
                app = QApplication(sys.argv)
                created_app = True

            # Use the native dialog by default when running inside an app.
            # Creating non-native dialogs can trigger unexpected code paths
            # on macOS; avoid forcing `DontUseNativeDialog`.
            path, _ = QFileDialog.getOpenFileName(
                None,
                "Select a metadata file",
                "",
                "JSON files (*.json);;CSV files (*.csv);;All files (*)",
            )

            if created_app:
                # Clean up the temporary app
                try:
                    app.quit()
                except Exception:
                    pass

            return path or None
        except Exception:
            # If a QApplication already exists (we're running inside the app),
            # do NOT fall back to tkinter — that will create a second AppKit
            # instance and crash on macOS. Instead, surface the error so the
            # caller can log it and continue.
            try:
                QApplication = QtWidgets.QApplication
                if QApplication.instance() is not None:
                    raise RuntimeError("Qt file dialog failed while running inside Qt application")
            except Exception:
                # If we couldn't check instance, be conservative and avoid tkinter.
                raise RuntimeError("Qt file dialog failed and fallback is unsafe in-app")

    # Last resort: tkinter (only for CLI/testing, where no QApplication exists)
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        raise RuntimeError("No GUI backend available (Qt bindings or tkinter missing)")

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    path = filedialog.askopenfilename(
        title="Select a metadata file",
        filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*")],
        defaultextension=".json",
    )
    root.destroy()
    return path or None


def run(ctx=None):
    """Entry point expected by SetiAstroSuitePro scripts.

    If `ctx` provides `log`, messages are sent there; otherwise printed.
    """
    try:
        path = select_text_file()
    except Exception as e:
        msg = f"Error opening file dialog: {e}"
        if ctx and hasattr(ctx, "log"):
            ctx.log(msg)
        else:
            print(msg, file=sys.stderr)
        return None

    if not path:
        if ctx and hasattr(ctx, "log"):
            ctx.log("No file selected.")
        else:
            print("No file selected.")
        return None

    if ctx and hasattr(ctx, "log"):
        ctx.log(f"Selected file: {path}")
    else:
        print(path)

    print(f"[NINA_ImageMetaData] launching UI for: {path}")

    # Launch the Qt UI to display and plot the metadata. When running inside
    # the app, this will reuse the existing QApplication; from CLI it will
    # create a temporary one.
    try:
        win = launch_qt_ui(path, ctx=ctx)
        return win
    except Exception as e:
        msg = f"Failed to launch UI: {e}"
        if ctx and hasattr(ctx, "log"):
            ctx.log(msg)
        else:
            print(msg, file=sys.stderr)
        return None


# NOTE: `run()` is called from the module main block at the end of the file
# so that `launch_qt_ui` and other helpers are defined before invocation.


### JSON parsing + pure-Qt plotting UI
import json
import os
import numpy as np
from datetime import datetime
import math


def _import_qt_bindings():
    # Return tuple (QtWidgets, binding_name, matplotlib_backend_module)
    for mod in ("PySide6", "PyQt6", "PyQt5", "PySide2"):
        try:
            pkg = __import__(mod)
            from importlib import import_module
            QtWidgets = import_module(f"{mod}.QtWidgets")
            return QtWidgets, mod
        except Exception:
            continue
    return None, None



def parse_image_metadata_json(path: str):
    """Parse NINA ImageMetaData.json and return a list of dict entries.

    This function is tolerant: it will accept a top-level list, or a dict
    containing a list under common keys like 'frames', 'images', or 'ImageMetaData'.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Return a mapping of group_name -> list[dict].
    groups = {}
    if isinstance(raw, list):
        groups['All'] = raw
    elif isinstance(raw, dict):
        # If the dict itself maps group names to lists, collect non-empty lists
        found_lists = {k: v for k, v in raw.items() if isinstance(v, list)}
        # Prefer obvious container keys if present
        for key in ("frames", "images", "ImageMetaData", "imageMetaData", "data"):
            v = raw.get(key)
            if isinstance(v, list):
                groups['All'] = v
                break
        else:
            if found_lists:
                # include only non-empty lists
                for k, v in found_lists.items():
                    if v:
                        groups[str(k)] = v
                if not groups:
                    raise ValueError("JSON contains only empty groups")
            else:
                # Last resort: find a nested list of dicts
                entries = None
                for v in raw.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        entries = v
                        break
                if entries is None:
                    # Maybe the JSON uses a dict-of-dicts (index-keyed). Convert to list.
                    dict_of_dicts = {k: v for k, v in raw.items() if isinstance(v, dict)}
                    if dict_of_dicts:
                        entries = list(dict_of_dicts.values())
                    else:
                        raise ValueError("Could not locate list of image metadata entries in JSON")
                groups['All'] = entries
    else:
        raise ValueError("Unsupported JSON root type for ImageMetaData")

    # Normalize: ensure each group's entries are lists of dicts
    norm = {}
    for g, lst in groups.items():
        parsed = [e for e in lst if isinstance(e, dict)]
        if parsed:
            norm[g] = parsed
    if not norm:
        raise ValueError("No valid metadata entries found in JSON")
    return norm

# Stub for CSV parsing
def parse_image_metadata_csv(path: str):
    """Parse NINA ImageMetaData.csv and return a list of dict entries (stub)."""
    import csv
    groups = {}
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        groups['All'] = [row for row in reader]
    return groups


def extract_series(entries, key: str, base_dir: str = None):
    """Extract x (index) and y numeric values for given key, and labels.

    Returns (xs, ys, labels) where labels are filenames (or empty string)
    corresponding to each entry when available. If `base_dir` is provided,
    entries whose filename is missing from that directory are skipped.
    """
    xs = []
    ys = []
    labels = []
    filename_keys = ("FileName", "fileName", "filename", "File", "FilePath", "FileNameWithPath", "Path", "image", "Image")
    for i, e in enumerate(entries):
        v = e.get(key)
        if v is None:
            yval = float('nan')
        else:
            if isinstance(v, (int, float)):
                yval = float(v)
            else:
                try:
                    yval = float(str(v))
                except Exception:
                    yval = float('nan')

        # determine filename label (basename)
        fname = ""
        for fk in filename_keys:
            fv = e.get(fk)
            if fv:
                try:
                    fname = os.path.basename(str(fv))
                except Exception:
                    fname = str(fv)
                break

        if base_dir and fname:
            try:
                candidate = os.path.join(base_dir, fname)
                if not os.path.exists(candidate):
                    continue
            except Exception:
                pass

        xs.append(i)
        ys.append(yval)
        labels.append(fname)

    return xs, ys, labels


def _available_numeric_keys(entries):
    def _is_numeric(v):
        try:
            if v is None:
                return False
            if isinstance(v, (int, float)):
                return math.isfinite(float(v))
            s = str(v).strip()
            if s == "":
                return False
            if s.lower() in ("none", "null", "n/a", "na"):
                return False
            val = float(s)
            return math.isfinite(val)
        except Exception:
            return False

    keys = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        for k, v in e.items():
            if _is_numeric(v):
                keys.add(k)

    return sorted(keys)


def launch_qt_ui(path: str = None, ctx=None):
    QtWidgets, binding = _import_qt_bindings()
    print(f"[NINA_ImageMetaData] Qt binding: {binding}")
    if QtWidgets is None:
        raise RuntimeError("No Qt bindings available (PySide6/PyQt6/PyQt5/PySide2)")

    # Try to import the appropriate matplotlib Qt canvas for the detected binding
    use_matplotlib = True
    FigureCanvas = None
    try:
        if binding in ("PySide6", "PyQt6"):
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        else:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        use_matplotlib = False

    class MainWindow(QtWidgets.QMainWindow):
        def _numpy_stat_stretch(self, img, target_median=0.35, gamma=0.8):
            import numpy as np
            arr = np.array(img, dtype=np.float32)
            med = np.median(arr)
            if med == 0:
                med = 1e-6
            scale = target_median / med
            stretched = arr * scale
            # Clip to [0, 1] for display
            stretched = np.clip(stretched, 0, 1)
            # Apply gamma correction for more aggressive stretch
            stretched = np.power(stretched, gamma)
            return stretched

        def _populate_keys_for_group(self, group_name: str):
            self.list_widget.clear()
            entries = self.entries.get(group_name, [])
            keys = _available_numeric_keys(entries)
            for k in keys:
                self.list_widget.addItem(k)
            if keys:
                self.list_widget.setCurrentRow(0)

        def move_files_above_threshold(self):
            import shutil
            items = self.list_widget.selectedItems()
            if not items:
                self.log("No key selected.")
                return
            key = items[0].text()
            group = getattr(self, '_current_group', None)
            entries = self.entries.get(group) if group else None
            if entries is None:
                self.log("No entries loaded.")
                return
            thr_val = self.threshold_spin.value()
            base_dir = getattr(self, '_json_dir', None)
            xs, ys, labels = extract_series(entries, key, base_dir=base_dir)
            is_detected_stars = key.strip().lower() == "detectedstars"
            def is_flagged(y):
                if not isinstance(y, (int, float)) or (isinstance(y, float) and (y != y)):
                    return False
                if is_detected_stars:
                    return y < thr_val
                else:
                    return y > thr_val
            files_to_move = []
            for y, label in zip(ys, labels):
                try:
                    if not (label and base_dir):
                        continue
                    if is_flagged(y):
                        src = os.path.join(base_dir, label)
                        if os.path.exists(src):
                            files_to_move.append(src)
                except Exception:
                    continue
            if not files_to_move:
                self.log("No flagged files to move.")
                return
            QtWidgets = _import_qt()
            QFileDialog = QtWidgets.QFileDialog
            dest_dir = QFileDialog.getExistingDirectory(self, "Select destination directory for flagged files")
            if not dest_dir:
                self.log("No destination directory selected.")
                return
            moved = 0
            for src in files_to_move:
                try:
                    dst = os.path.join(dest_dir, os.path.basename(src))
                    shutil.move(src, dst)
                    moved += 1
                except Exception as e:
                    self.log(f"Failed to move {src}: {e}")
            self.log(f"Moved {moved} flagged files to {dest_dir}.")
            # Redraw the plot, then recalculate and reset the threshold
            self.on_selection()
            # Recalculate threshold after plot redraw (using updated data)
            entries = self.entries.get(group) if group else None
            xs, ys, labels = extract_series(entries, key, base_dir=base_dir)
            try:
                yarr = np.array([y for y in ys if isinstance(y, (int, float)) and not (isinstance(y, float) and (y != y))], dtype=float)
                if yarr.size == 0:
                    default_thr = 0.0
                else:
                    mx = np.nanmax(yarr)
                    mn = np.nanmin(yarr)
                    yrange = mx - mn if not np.isclose(mx, mn) else abs(mx) if mx != 0 else 1.0
                    margin = max(1e-6, abs(yrange) * 0.02, abs(mx) * 0.01)
                    if is_detected_stars:
                        default_thr = mn - margin
                    else:
                        default_thr = mx + margin
                self._thr_values[(group, key)] = default_thr
            except Exception:
                self._thr_values[(group, key)] = 0.0
            self.on_selection()

        def __init__(self, ctx=None):
            super().__init__()
            self.ctx = ctx
            self.setWindowTitle("NINA ImageMetaData Viewer v" + VERSION)
            self.resize(800, 600)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)


            # Top controls (fixed height)
            top_controls = QtWidgets.QWidget()
            top_controls_layout = QtWidgets.QHBoxLayout(top_controls)
            top_controls_layout.setContentsMargins(0, 0, 0, 0)
            self.open_btn = QtWidgets.QPushButton("Open Metadata File")
            self.open_btn.clicked.connect(self.open_file)
            top_controls_layout.addWidget(self.open_btn)

            # ...existing code...
            self.path_label = QtWidgets.QLabel("")
            top_controls_layout.addWidget(self.path_label)
            # Set size policy to fixed vertically, expanding horizontally (cross-Qt compatibility)
            try:
                # Qt6 style
                expanding = QtWidgets.QSizePolicy.Policy.Expanding
                fixed = QtWidgets.QSizePolicy.Policy.Fixed
            except AttributeError:
                # Qt5 style
                expanding = 7  # QSizePolicy.Expanding
                fixed = 0      # QSizePolicy.Fixed
            top_controls.setSizePolicy(expanding, fixed)
            layout.addWidget(top_controls)

            # Splitter: left column contains key list, right plot
            splitter = QtWidgets.QSplitter()
            try:
                splitter.setSizePolicy(expanding, expanding)
            except Exception:
                pass

            # Left panel: numeric-key list only (group selection removed)
            class LeftPanelWidget(QtWidgets.QWidget):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._sync_button_width_callback = None
                def set_sync_button_width_callback(self, cb):
                    self._sync_button_width_callback = cb
                def resizeEvent(self, event):
                    if self._sync_button_width_callback:
                        self._sync_button_width_callback()
                    super().resizeEvent(event)

            left_panel = LeftPanelWidget()
            left_layout = QtWidgets.QVBoxLayout(left_panel)
            self.list_widget = QtWidgets.QListWidget()
            self.list_widget.itemSelectionChanged.connect(self.on_selection)
            left_layout.addWidget(self.list_widget)

            # Threshold controls: enable checkbox + numeric spinbox
            thr_row = QtWidgets.QWidget()
            thr_layout = QtWidgets.QHBoxLayout(thr_row)
            thr_layout.setContentsMargins(0, 0, 0, 0)
            thr_label = QtWidgets.QLabel("Threshold:")
            thr_layout.addWidget(thr_label)
            self.threshold_spin = QtWidgets.QDoubleSpinBox()
            self.threshold_spin.setRange(-1e12, 1e12)
            self.threshold_spin.setDecimals(6)
            self.threshold_spin.setValue(0.0)
            thr_layout.addWidget(self.threshold_spin)
            left_layout.addWidget(thr_row)

            # Add Move Files Above Threshold button below threshold controls
            self.move_files_btn = QtWidgets.QPushButton("Move Flagged Files")
            self.move_files_btn.clicked.connect(self.move_files_above_threshold)
            left_layout.addWidget(self.move_files_btn)

            # always-enabled threshold: spinbox triggers a replott
            try:
                self.threshold_spin.valueChanged.connect(self.on_selection)
            except Exception:
                pass

            splitter.addWidget(left_panel)

            # Set Open JSON button width to match left_panel width after splitter is shown
            def sync_button_width():
                left_width = left_panel.width()
                self.open_btn.setMinimumWidth(left_width)
                self.open_btn.setMaximumWidth(left_width)
            splitter.splitterMoved.connect(lambda pos, index: sync_button_width())
            left_panel.set_sync_button_width_callback(sync_button_width)
            # Initial sync after UI is shown
            QtWidgets.QApplication.instance().processEvents()
            sync_button_width()

            if use_matplotlib and FigureCanvas is not None:
                from matplotlib.figure import Figure

                self.fig = Figure(figsize=(5, 4))
                self.canvas = FigureCanvas(self.fig)
                splitter.addWidget(self.canvas)
            else:
                self.canvas = QtWidgets.QLabel("matplotlib not available")
                self.canvas.setAlignment(QtWidgets.Qt.AlignCenter if hasattr(QtWidgets, 'Qt') else 0)
                splitter.addWidget(self.canvas)

            splitter.setStretchFactor(1, 1)
            layout.addWidget(splitter)


            self.entries = {}
            # per-(group,key) threshold values
            self._thr_values = {}
            self._current_key_id = None
            self._thr_line = None
            self._current_sc = None

            if path:
                self.load_path(path)

        def log(self, msg):
            # Only log to SetiAstroSuitePro's ctx if available, else print
            if hasattr(self, 'ctx') and hasattr(self.ctx, 'log'):
                self.ctx.log(msg)
            else:
                print(msg)

        def open_file(self):
            try:
                p = select_text_file()
            except Exception as e:
                self.log(f"Error opening dialog: {e}")
                return
            if not p:
                return
            self.load_path(p)

        def load_path(self, p: str):
            self.path_label.setText(p)
            try:
                ext = os.path.splitext(p)[1].lower()
                if ext == '.json':
                    self.entries = parse_image_metadata_json(p)
                elif ext == '.csv':
                    self.entries = parse_image_metadata_csv(p)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
            except Exception as e:
                self.log(f"Failed to parse metadata: {e}")
                return
            # Default to the first group (if any)
            groups = list(self.entries.keys())
            if groups:
                # remember JSON directory for existence checks
                try:
                    self._json_dir = os.path.dirname(p) or None
                except Exception:
                    self._json_dir = None
                self._current_group = groups[0]
                self._populate_keys_for_group(groups[0])

        def on_selection(self):
            items = self.list_widget.selectedItems()
            if not items:
                return
            key = items[0].text()
            group = getattr(self, '_current_group', None)
            entries = self.entries.get(group) if group else None
            if entries is None:
                entries = []
            xs, ys, labels = extract_series(entries, key, base_dir=getattr(self, '_json_dir', None))
            is_detected_stars = key.strip().lower() == "detectedstars"
            # Always recalculate and reset the threshold to default before plotting
            try:
                yarr = np.array([y for y in ys if isinstance(y, (int, float)) and not (isinstance(y, float) and (y != y))], dtype=float)
                if yarr.size == 0:
                    default_thr = 0.0
                else:
                    mx = np.nanmax(yarr)
                    mn = np.nanmin(yarr)
                    yrange = mx - mn if not np.isclose(mx, mn) else abs(mx) if mx != 0 else 1.0
                    margin = max(1e-6, abs(yrange) * 0.02, abs(mx) * 0.01)
                    if is_detected_stars:
                        default_thr = mn - margin
                    else:
                        default_thr = mx + margin
                self._thr_values[(group, key)] = default_thr
            except Exception:
                self._thr_values[(group, key)] = 0.0

            if use_matplotlib and FigureCanvas is not None:
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                key_id = (group, key)
                try:
                    yarr = np.array(ys, dtype=float)
                    if yarr.size == 0 or np.all(np.isnan(yarr)):
                        default_thr = 0.0
                    else:
                        mx = np.nanmax(yarr)
                        mn = np.nanmin(yarr)
                        yrange = mx - mn if not math.isclose(mx, mn) else abs(mx) if mx != 0 else 1.0
                        margin = max(1e-6, abs(yrange) * 0.02, abs(mx) * 0.01)
                        if is_detected_stars:
                            default_thr = mn - margin
                        else:
                            default_thr = mx + margin
                except Exception:
                    default_thr = 0.0
                thr_val = self._thr_values.get(key_id, default_thr)
                try:
                    if not np.isfinite(float(thr_val)):
                        thr_val = default_thr
                except Exception:
                    thr_val = default_thr
                try:
                    self.threshold_spin.blockSignals(True)
                    self.threshold_spin.setValue(float(thr_val))
                finally:
                    try:
                        self.threshold_spin.blockSignals(False)
                    except Exception:
                        pass
                def _compute_colors(val):
                    cols = []
                    for y in ys:
                        try:
                            if not math.isnan(y):
                                if is_detected_stars:
                                    if y < val:
                                        cols.append('red')
                                    else:
                                        cols.append('C0')
                                else:
                                    if y > val:
                                        cols.append('red')
                                    else:
                                        cols.append('C0')
                            else:
                                cols.append('C0')
                        except Exception:
                            cols.append('C0')
                    return cols
                colors = _compute_colors(thr_val)
                sc = ax.scatter(xs, ys, c=colors, picker=10, s=40, edgecolors='k', linewidths=0.5, alpha=0.9)
                # draw threshold line but preserve existing y-limits so autoscale isn't affected
                try:
                    ylims = ax.get_ylim()
                    thr_line = ax.axhline(float(thr_val), color='red', linestyle='--', linewidth=1.25, alpha=0.9, zorder=5)
                    # ensure the threshold line does not generate pick_event spam
                    try:
                        thr_line.set_picker(False)
                        thr_line.set_zorder(5)
                        thr_line.set_visible(True)
                    except Exception:
                        pass
                    ax.set_ylim(ylims)
                except Exception:
                    thr_line = None

                # remember this threshold for the key
                try:
                    self._thr_values[key_id] = float(thr_val)
                except Exception:
                    pass

                # disconnect any previous threshold handlers
                try:
                    for attr in ('_thr_press_cid', '_thr_motion_cid', '_thr_release_cid'):
                        cid = getattr(self, attr, None)
                        if cid is not None:
                            try:
                                self.canvas.mpl_disconnect(cid)
                            except Exception:
                                pass
                except Exception:
                    pass

                # add mouse handlers to allow dragging the threshold line
                self._thr_dragging = False

                def _thr_press(event):
                    try:
                        if event.inaxes != ax or thr_line is None:
                            return
                        # compute y position of the line in pixel coords
                        try:
                            ypix_line = ax.transData.transform((0.0, thr_line.get_ydata()[0]))[1]
                        except Exception:
                            return
                        # if click is close to the line (8 pixels), start dragging
                        if abs(ypix_line - event.y) <= 8 and event.button == 1:
                            self._thr_dragging = True
                    except Exception:
                        pass

                def _thr_motion(event):
                    try:
                        if not getattr(self, '_thr_dragging', False) or event.inaxes != ax or thr_line is None:
                            return
                        try:
                            inv = ax.transData.inverted()
                            _, ydata = inv.transform((event.x, event.y))
                        except Exception:
                            return
                        try:
                            thr_line.set_ydata([ydata, ydata])
                        except Exception:
                            pass
                        try:
                            cols = _compute_colors(ydata)
                            sc.set_color(cols)
                        except Exception:
                            pass
                        try:
                            self.threshold_spin.blockSignals(True)
                            self.threshold_spin.setValue(float(ydata))
                        finally:
                            try:
                                self.threshold_spin.blockSignals(False)
                            except Exception:
                                pass
                        try:
                            self._thr_values[key_id] = float(ydata)
                        except Exception:
                            pass
                        self.canvas.draw_idle()
                    except Exception:
                        pass

                def _thr_release(event):
                    try:
                        if getattr(self, '_thr_dragging', False):
                            self._thr_dragging = False
                            # final store of spinbox value
                            try:
                                val = float(self.threshold_spin.value())
                                self._thr_values[key_id] = val
                            except Exception:
                                pass
                    except Exception:
                        pass

                try:
                    self._thr_press_cid = self.canvas.mpl_connect('button_press_event', _thr_press)
                    self._thr_motion_cid = self.canvas.mpl_connect('motion_notify_event', _thr_motion)
                    self._thr_release_cid = self.canvas.mpl_connect('button_release_event', _thr_release)
                except Exception:
                    self._thr_press_cid = self._thr_motion_cid = self._thr_release_cid = None

                ax.set_title(key)
                ax.set_xlabel('index')
                ax.set_ylabel(key)
                # Force axes and figure background to opaque white
                ax.set_facecolor((1,1,1,1))
                self.fig.set_facecolor((1,1,1,1))

                # create annotation bound to this axes (recreate after clearing figure)
                self._annot = ax.annotate('', xy=(0,0), xytext=(0.5,1.0), textcoords='axes fraction',
                                      bbox=dict(boxstyle='square,pad=0.5', facecolor=(1,1,1,1), edgecolor=(0,0,0,1), linewidth=1.5, alpha=1.0), arrowprops=dict(arrowstyle='->'), ha='center', alpha=1.0)
                # Add a manual Rectangle patch for opaque background
                import matplotlib.patches as mpatches
                import matplotlib.transforms as mtransforms
                self._annot_bg_rect = mpatches.Rectangle((0,0), 1, 1, color=(1,1,1,1), zorder=99, linewidth=0, transform=ax.figure.transFigure)
                self._annot_bg_rect.set_visible(False)
                # Add the rectangle patch directly to the axes so it is above the plot but below the annotation
                ax.add_patch(self._annot_bg_rect)
                self._annot_bg_rect.set_zorder(101)  # annotation text box is 100, so 101 ensures it's just behind
                # Set annotation zorder higher so it is always above the rectangle
                self._annot.set_zorder(102)
                patch = self._annot.get_bbox_patch()
                if patch is not None:
                    patch.set_facecolor((1,1,1,1))
                    patch.set_edgecolor((0,0,0,1))
                    patch.set_alpha(1.0)
                    patch.set_zorder(100)
                    patch.set_linewidth(1.5)
                    # Debug: print patch alpha to verify
                    print('[DEBUG] patch alpha after set:', patch.get_alpha())
                # Remove forced annotation visible for debugging

                # disconnect previous mpl cids if present
                try:
                    if hasattr(self, '_mpl_cid') and self._mpl_cid is not None:
                        self.canvas.mpl_disconnect(self._mpl_cid)
                except Exception:
                    pass
                try:
                    if hasattr(self, '_mpl_pick_cid') and self._mpl_pick_cid is not None:
                        self.canvas.mpl_disconnect(self._mpl_pick_cid)
                except Exception:
                    pass

                def _hover(event):
                    try:
                        if event.inaxes != ax:
                            if self._annot.get_visible():
                                self._annot.set_visible(False)
                                self._annot_bg_rect.set_visible(False)
                                self.canvas.draw_idle()
                            return

                        # transform data points to pixel coordinates
                        try:
                            pix = ax.transData.transform(np.column_stack((xs, ys)))
                        except Exception:
                            return

                        xpix, ypix = event.x, event.y
                        d = np.hypot(pix[:, 0] - xpix, pix[:, 1] - ypix)
                        idx = int(np.argmin(d))
                        # threshold in pixels
                        if d[idx] <= 10:
                            x = xs[idx]
                            y = ys[idx]
                            label = labels[idx] if idx < len(labels) else ''
                            # Center annotation in axes, arrow points to datapoint
                            txt = f"{label}\n{key}: {y:.3f}" if label else f"{key}: {y:.3f}"
                            self._annot.set_text(txt)
                            renderer = self.canvas.figure.canvas.get_renderer()
                            ax_bbox = ax.get_window_extent()
                            # Center of axes in display (pixel) coordinates
                            center_x = ax_bbox.x0 + ax_bbox.width / 2
                            center_y = ax_bbox.y0 + ax_bbox.height / 2
                            # Convert center_x, center_y to data coordinates
                            inv = ax.transData.inverted()
                            center_data_x, center_data_y = inv.transform((center_x, center_y))
                            self._annot.xy = (x, y)
                            self._annot.set_visible(True)
                            # Update and show the manual background rectangle to match only the annotation text box (not the arrow)
                            bbox_patch = self._annot.get_bbox_patch()
                            if bbox_patch is not None:
                                bbox = bbox_patch.get_window_extent(renderer=renderer)
                                fig = ax.figure
                                # Convert display (pixel) to figure fraction
                                l, b = fig.transFigure.inverted().transform((bbox.x0, bbox.y0))
                                r, t = fig.transFigure.inverted().transform((bbox.x1, bbox.y1))
                                self._annot_bg_rect.set_bounds(l, b, r-l, t-b)
                                self._annot_bg_rect.set_visible(True)
                            self.canvas.draw_idle()
                        else:
                            if self._annot.get_visible():
                                self._annot.set_visible(False)
                                self._annot_bg_rect.set_visible(False)
                                self.canvas.draw_idle()
                    except Exception as _e:
                        self.log(f"hover handler error: {_e}")

                self._mpl_cid = self.canvas.mpl_connect('motion_notify_event', _hover)

                def _on_pick(event):
                    try:
                        # Only handle pick events for the scatter artist
                        if event.artist is sc and event.ind is not None and len(event.ind) > 0:
                            idx = event.ind[0]  # index of the clicked data point
                            x = xs[idx]
                            y = ys[idx]
                            label = labels[idx] if idx < len(labels) else ''
                            self._annot.xy = (x, y)
                            txt = f"{label}\n{key}: {y:.3f}" if label else f"{key}: {y:.3f}"
                            self._annot.set_text(txt)
                            self._annot.set_visible(True)
                            self.canvas.draw_idle()

                            # Attempt to open the file using ctx.load_image if available
                            if hasattr(self, 'ctx') and hasattr(self.ctx, 'load_image'):
                                file_path = None
                                if label:
                                    base_dir = getattr(self, '_json_dir', None)
                                    if base_dir:
                                        file_path = os.path.join(base_dir, label)
                                    else:
                                        file_path = label
                                if file_path:
                                    try:
                                        img, original_header, bit_depth, is_mono = self.ctx.load_image(file_path)
                                        self.log(f"Loaded image: {file_path}")
                                        self._show_image_window(img, file_path)
                                    except Exception as e:
                                        self.log(f"Failed to load image: {file_path} ({e})")
                            else:
                                self.log("ctx.load_image not available in this context.")
                    except Exception as _e:
                        self.log(f"pick handler error: {_e}")

                self._mpl_pick_cid = self.canvas.mpl_connect('pick_event', _on_pick)
                self.canvas.draw()
            else:
                self.canvas.setText(f"Plot unavailable. Showing first values for {key}:\n" + ", ".join(str(x) for x in ys[:50]))

        def _show_image_window(self, img, file_path):
            import os
            filename = os.path.basename(file_path)
            stretched_img = img
            used_numpy_stretch = False
            if hasattr(self, 'ctx') and hasattr(self.ctx, 'open_image') and hasattr(self.ctx, 'run_command'):
                try:
                    doc = self.ctx.open_image(file_path)
                    try:
                        self.ctx.run_command('stat_stretch', {'target_median': 0.25, 'linked': True}, doc)
                    except TypeError:
                        self.ctx.run_command('stat_stretch', {'target_median': 0.25, 'linked': True})
                    if hasattr(doc, 'get_image_data'):
                        stretched_img = doc.get_image_data()
                        self.log(f"Statistical stretch applied to document: {filename}")
                    else:
                        self.log(f"Statistical stretch applied to document: {filename} (no direct data access)")
                except Exception as e:
                    self.log(f"Statistical stretch failed: {e} (using numpy fallback)")
                    stretched_img = self._numpy_stat_stretch(img)
                    used_numpy_stretch = True
            else:
                stretched_img = self._numpy_stat_stretch(img)
                used_numpy_stretch = True
            if used_numpy_stretch:
                self.log(f"Numpy statistical stretch applied for display: {filename}")
            QtWidgets = _import_qt()
            class ImageWindow(QtWidgets.QMainWindow):
                def __init__(self, img, filename):
                    super().__init__()
                    self.setWindowTitle(f"Image: {filename}")
                    self.resize(800, 600)
                    app = QtWidgets.QApplication.instance()
                    is_dark = False
                    if app is not None:
                        palette = app.palette()
                        try:
                            from PyQt5 import QtGui as _QtGui
                        except ImportError:
                            try:
                                from PySide2 import QtGui as _QtGui
                            except ImportError:
                                try:
                                    from PyQt6 import QtGui as _QtGui
                                except ImportError:
                                    from PySide6 import QtGui as _QtGui
                        window_role = getattr(_QtGui.QPalette, 'Window', None)
                        if window_role is None:
                            window_role = getattr(_QtGui.QPalette.ColorRole, 'Window', 10)
                        window_color = palette.color(window_role)
                        is_dark = (window_color.value() < 128)
                    from matplotlib.figure import Figure
                    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                    fig = Figure(facecolor='black' if is_dark else 'white')
                    self.canvas = FigureCanvas(fig)
                    self.setCentralWidget(self.canvas)
                    ax = fig.add_subplot(111, facecolor='black' if is_dark else 'white')
                    import numpy as np
                    arr = np.array(img) if not isinstance(img, np.ndarray) else img
                    if arr.ndim == 2:
                        ax.imshow(arr, cmap='gray', origin='upper')
                    else:
                        ax.imshow(arr, origin='upper')
                    ax.set_title("")
                    ax.axis('off')
                    fig.tight_layout()
                    self.canvas.setStyleSheet(f"background-color: {'#222' if is_dark else '#fff'};")
            if not hasattr(self, '_image_windows'):
                self._image_windows = []
            win = ImageWindow(stretched_img, filename)
            win.show()
            self._image_windows.append(win)

    app = QtWidgets.QApplication.instance()
    created = False
    if app is None:
        print("[NINA_ImageMetaData] No existing QApplication, creating new one")
        app = QtWidgets.QApplication(sys.argv)
        created = True
    else:
        print("[NINA_ImageMetaData] Reusing existing QApplication")
    
    # Create and show the main window
    win = MainWindow(ctx=ctx)
    win.show()

    try:
        # Try to bring the window to the front (helpful on macOS)
        try:
            win.raise_()
            win.activateWindow()
        except Exception:
            pass
        print("[NINA_ImageMetaData] Window shown; entering Qt event loop")
    except Exception:
        pass

    if created:
        app.exec()
    return win

if __name__ == "__main__":
    run()
