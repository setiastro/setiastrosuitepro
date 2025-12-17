#pro.fitsmodifier.py
from __future__ import annotations
import os
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
try:
    from astropy.io.fits.verify import VerifyError
except Exception:
    # Fallback for older Astropy – same pattern as in legacy.image_manager
    class VerifyError(Exception):
        pass
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout
)
from PyQt6.QtCore import QSettings
from setiastro.saspro.legacy.image_manager import (
    load_image as legacy_load_image,
    save_image as legacy_save_image,
    _drop_invalid_cards,    # ← new
)

class FITSModifier(QDialog):
    def __init__(self, file_path: Optional[str], header,
                 image_manager=None,
                 doc_manager=None, active_document=None,   # <— rename param
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("FITS Header Editor")
        self.resize(800, 600)

        self._doc_manager = doc_manager
        self._active_document = active_document
        self.image_manager = None  # stop using old ImageManager path

        self.file_path = file_path if (file_path and os.path.isfile(file_path)) else None

        self.hdul = None
        self.current_hdu_index = 0
        self._fallback_header = header

        self._populating = False
        self._dirty = False

        # UI
        top = QHBoxLayout()
        self.path_label = QLabel(self.file_path or "(no file)")
        self.open_btn = QPushButton("Open FITS…")
        self.reload_btn = QPushButton("Reload")
        self.hdu_combo = QComboBox()
        self.save_btn = QPushButton("Save")
        self.saveas_btn = QPushButton("Save a Copy As…")
        # self.apply_to_slot_btn = QPushButton("Apply to Slot Metadata")  # optional

        top.addWidget(QLabel("File:"))
        top.addWidget(self.path_label, 1)
        top.addWidget(QLabel("HDU:"))
        top.addWidget(self.hdu_combo)
        top.addWidget(self.open_btn)
        top.addWidget(self.reload_btn)
        #top.addWidget(self.save_btn)
        top.addWidget(self.saveas_btn)
        # top.addWidget(self.apply_to_slot_btn)

        batch = QHBoxLayout()
        self.batch_btn = QPushButton("Batch Modify...")
        batch.addStretch()
        batch.addWidget(self.batch_btn)
        batch.addStretch()

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Keyword", "Value", "Comment"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setEditTriggers(QTreeWidget.EditTrigger.DoubleClicked | QTreeWidget.EditTrigger.SelectedClicked)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setStyleSheet("""
        QTreeWidget::item:selected:active { background-color: #1E90FF; color: white; }
        QTreeWidget::item:selected:!active { background-color: #5AA7FF; color: white; }
        QTreeWidget::item:hover { background-color: rgba(30,144,255,0.18); }
        QTreeWidget::item { padding: 2px 6px; }
        """)

        bottom = QHBoxLayout()
        self.add_key_edit = QLineEdit(); self.add_key_edit.setPlaceholderText("KEYWORD")
        self.add_val_edit = QLineEdit(); self.add_val_edit.setPlaceholderText("Value")
        self.add_com_edit = QLineEdit(); self.add_com_edit.setPlaceholderText("Comment (optional)")
        self.add_btn = QPushButton("Add/Update")
        self.del_btn = QPushButton("Delete Selected")
        self.all_hdus_chk = QCheckBox("Apply add/update/delete to all HDUs")
        bottom.addWidget(self.add_key_edit)
        bottom.addWidget(self.add_val_edit)
        bottom.addWidget(self.add_com_edit)
        bottom.addWidget(self.all_hdus_chk)
        bottom.addWidget(self.add_btn)
        bottom.addWidget(self.del_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(batch)
        layout.addWidget(self.tree, 1)
        layout.addLayout(bottom)

        # Signals
        self.open_btn.clicked.connect(self._choose_file)
        self.reload_btn.clicked.connect(self._reload)
        self.hdu_combo.currentIndexChanged.connect(self._on_hdu_changed)
        self.save_btn.clicked.connect(self._save_in_place)
        self.saveas_btn.clicked.connect(self._save_as_copy)
        # self.apply_to_slot_btn.clicked.connect(self._apply_to_slot_metadata)
        self.add_btn.clicked.connect(self._add_or_update_keyword)
        self.del_btn.clicked.connect(self._delete_selected)

        # Initial content
        if self.file_path:
            ok = self._load_file(self.file_path)
            if not ok and header is not None:
                self._init_from_header(header)
        elif header is not None:
            self._init_from_header(header)
        else:
            self._init_from_header(fits.Header())

        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(self._on_row_selected)
        self.batch_btn.clicked.connect(self._open_batch_modifier)

    # ---- helpers ----
    def _get_active_doc(self):
        """
        Prefer the explicitly passed active_document; fall back to doc_manager.
        This avoids accidentally selecting the last opened document when MDI focus
        detection is flaky.
        """
        # 1) if caller passed a doc, use it
        if getattr(self, "_active_document", None) is not None:
            return self._active_document

        # 2) otherwise ask the doc manager
        try:
            if self._doc_manager and hasattr(self._doc_manager, "get_active_document"):
                d = self._doc_manager.get_active_document()
                if d is not None:
                    return d
        except Exception:
            pass

        return None

    def _update_multi_hdu_ui(self):
        n = len(self.hdul) if self.hdul else 0
        self.all_hdus_chk.setVisible(n > 1)

    def _on_row_selected(self, curr, prev):
        if not curr:
            return
        self.add_key_edit.setText(curr.text(0))
        self.add_val_edit.setText(curr.text(1))
        self.add_com_edit.setText(curr.text(2))

    def _selected_row_triplet(self) -> Tuple[str, str, str]:
        it = self.tree.currentItem()
        if not it:
            return "", "", ""
        return (it.text(0).strip(), it.text(1), it.text(2))

    def _open_batch_modifier(self):
        key, val, com = self._selected_row_triplet()
        dlg = BatchFITSHeaderDialog(parent=self,
                                    preset_keyword=key,
                                    preset_value=val,
                                    preset_comment=com)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _init_from_header(self, header):
        phdu = fits.PrimaryHDU()
        if isinstance(header, fits.Header):
            clean = _drop_invalid_cards(header)
            phdu.header = clean.copy()
        elif isinstance(header, dict):
            for k, v in header.items():
                try:
                    phdu.header[k] = v
                except Exception:
                    pass
        self.hdul = fits.HDUList([phdu])
        self._refresh_hdu_combo()
        self._populate_tree_from_header(phdu.header)

    def _apply_to_slot_metadata(self):
        try:
            hdr = self.hdul[self.current_hdu_index].header.copy()
            hdr = _drop_invalid_cards(hdr)
            doc = self._get_active_doc()
            if doc is not None and hasattr(doc, "metadata"):
                doc.metadata["original_header"] = hdr
                if hasattr(doc, "changed"):
                    doc.changed.emit()
        except Exception as e:
            print(f"[FITSModifier] _apply_to_slot_metadata error: {e}")


    def _set_dirty(self, dirty=True):
        self._dirty = dirty
        self.setWindowTitle("FITS Header Editor" + (" *" if dirty else ""))

    def _sync_tree_to_header(self):
        if not self.hdul:
            return
        hdr = self.hdul[self.current_hdu_index].header
        self._collect_tree_into_header(hdr)

    def _choose_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", self._last_dir(), "FITS files (*.fits *.fit *.fts *.fz)")
        if not fn:
            return
        self._load_file(fn)

    def _load_file(self, path) -> bool:
        try:
            if self.hdul is not None:
                self.hdul.close()
        except Exception:
            pass
        try:
            self.hdul = fits.open(path, mode='update', memmap=False)
        except Exception as e:
            QMessageBox.warning(self, "Invalid FITS",
                                f"This file does not appear to be a valid FITS:\n\n{path}\n\n{e}\n\n"
                                "Tip: Choose a FITS file via 'Open FITS…' or edit an in-memory header.")
            self.hdul = None
            self.file_path = None
            self.path_label.setText("(no file)")
            self.hdu_combo.clear()
            self._update_multi_hdu_ui()
            return False

        # Sanitize all HDU headers to drop invalid cards (e.g. bad TELESCOP)
        for hdu in self.hdul:
            try:
                if isinstance(hdu.header, fits.Header):
                    hdu.header = _drop_invalid_cards(hdu.header)
            except Exception as e:
                print(f"[FITSModifier] Header sanitize failed for HDU: {e}")

        self.file_path = path
        self.path_label.setText(path)
        self._save_last_dir(os.path.dirname(path))
        self._refresh_hdu_combo()

        # Use the sanitized header for the current HDU
        hdr = self.hdul[self.current_hdu_index].header
        self._populate_tree_from_header(hdr)
        return True


    def _reload(self):
        if not self.hdul and not self.file_path:
            return
        if self.file_path:
            self._load_file(self.file_path)
        else:
            self._populate_tree_from_header(self.hdul[0].header)

    def _refresh_hdu_combo(self):
        self.hdu_combo.blockSignals(True)
        self.hdu_combo.clear()
        for i, hdu in enumerate(self.hdul):
            name = getattr(hdu, 'name', 'UNKNOWN')
            self.hdu_combo.addItem(f"{i}: {name}")
        self.hdu_combo.setCurrentIndex(0)
        self.current_hdu_index = 0
        self.hdu_combo.blockSignals(False)
        self._update_multi_hdu_ui()

    def _on_hdu_changed(self, idx):
        self.current_hdu_index = int(idx)
        hdr = self.hdul[self.current_hdu_index].header
        hdr = _drop_invalid_cards(hdr)
        self._populate_tree_from_header(hdr)

    def _populate_tree_from_header(self, header: fits.Header):
        self._populating = True
        try:
            self.tree.blockSignals(True)
            self.tree.clear()
            for card in header.cards:
                key = card.keyword
                if key in ("HISTORY", "COMMENT"):
                    val = ""
                    com = ""
                else:
                    try:
                        val = self._val_to_str(card.value)
                        com = card.comment or ""
                    except VerifyError as e:
                        print(f"[FITSModifier] Skipping invalid card {key!r}: {e}")
                        continue  # Don't add this card to the tree
                it = QTreeWidgetItem([key, val, com])
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
                self.tree.addTopLevelItem(it)
            self.tree.resizeColumnToContents(0)
        finally:
            self.tree.blockSignals(False)
            self._populating = False
            self._set_dirty(False)



    def _collect_tree_into_header(self, header: fits.Header):
        new_header = fits.Header()
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            key = (it.text(0) or "").strip()
            val_txt = it.text(1)
            com = it.text(2)
            if not key:
                continue
            if key in ("HISTORY", "COMMENT"):
                if key == "HISTORY" and val_txt:
                    new_header.add_history(val_txt)
                elif key == "COMMENT" and val_txt:
                    new_header.add_comment(val_txt)
                else:
                    if key == "COMMENT" and not val_txt and com:
                        new_header.add_comment(com)
                continue
            try:
                val = self._parse_val(val_txt)
                new_header[key] = (val, com if com else None)
            except Exception:
                new_header[key] = (val_txt, com if com else None)

        header.clear()
        header.update(new_header)




    def _on_item_changed(self, item, column):
        if self._populating:
            return
        self._sync_tree_to_header()
        self._set_dirty(True)

    def _edited_primary_header(self) -> fits.Header:
        """Header we’ll write back out (primary HDU for single-image saves)."""
        if self.hdul is not None and len(self.hdul) > 0:
            return self.hdul[0].header.copy()
        if isinstance(self._fallback_header, fits.Header):
            return self._fallback_header.copy()
        return fits.Header()

    def _active_doc(self):
        return self._doc_manager.get_active_document() if self._doc_manager else None

    # 1) Change _write_to_path to optionally NOT touch the active doc
    def _write_to_path(self, out_path: str, *, update_doc_metadata: bool = True) -> bool:
        """
        Save the active document’s pixels with the edited header to `out_path`
        using legacy.save_image.

        If update_doc_metadata=False, this is a pure 'Save Copy' and we leave the
        active document + dialog state untouched (no open, no rename).
        """
        doc = self._active_doc()
        if doc is None or doc.image is None:
            QMessageBox.warning(self, "No Image", "No active image/document to save.")
            return False

        edited_hdr = self._edited_primary_header()

        # pick format from the extension (fallback to doc’s current)
        ext = os.path.splitext(out_path)[1].lower().lstrip(".")
        if not ext:
            ext = doc.metadata.get("original_format", "fits")
            out_path = out_path + f".{ext}"

        bit_depth = doc.metadata.get("bit_depth")
        is_mono   = doc.metadata.get("is_mono", (doc.image.ndim == 2))
        image_meta = doc.metadata.get("image_meta")
        file_meta  = doc.metadata.get("file_meta")

        try:
            legacy_save_image(
                img_array=doc.image,
                filename=out_path,
                original_format=ext,
                bit_depth=bit_depth,
                original_header=edited_hdr,
                is_mono=is_mono,
                image_meta=image_meta,
                file_meta=file_meta,
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save:\n{e}")
            return False

        if update_doc_metadata:
            # Only in true “Save” (not Save As copy)
            self.file_path = out_path
            self.path_label.setText(out_path)
            doc.metadata["file_path"] = out_path
            doc.metadata["original_format"] = ext
            doc.metadata["original_header"] = edited_hdr
            doc.changed.emit()
            self._set_dirty(False)

        return True


    def _save_in_place(self):
        # If the editor was opened on a file, use that.
        # Else use the active document’s file_path if present,
        # otherwise fall back to Save As…
        target = self.file_path
        if not target:
            doc = self._get_active_doc() 
            target = (doc.metadata.get("file_path") if doc else None)
            if not target:
                return self._save_as_copy()
        # apply any in-tree edits first
        self._sync_tree_to_header()
        self._write_to_path(target)


    def _save_as_copy(self):
        self._sync_tree_to_header()
        last = self._settings().value("fits_modifier/last_dir", "", type=str) or ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As",
            last,
            "FITS (*.fits *.fit);;TIFF (*.tif *.tiff);;PNG (*.png);;JPEG (*.jpg *.jpeg);;XISF (*.xisf)"
        )
        if not path:
            return
        ok = self._write_to_path(path, update_doc_metadata=False)  # <<— key change
        if ok:
            self._save_last_dir(os.path.dirname(path))
            # Optional: toast confirmation only
            QMessageBox.information(self, "Saved Copy", f"Saved a copy to:\n{path}")


    def _save_to_path(self, path: str) -> bool:
        """Write image+updated header to 'path' using legacy save if available; else fall back to astropy."""
        try:
            hdr = self.hdul[self.current_hdu_index].header.copy()
        except Exception:
            hdr = fits.Header()

        img, bit_depth, is_mono, src = self._grab_image_for_save(self.file_path)

        # Prefer legacy save_image if available and we have image data
        if legacy_save_image and img is not None:
            try:
                if bit_depth is not None and is_mono is not None:
                    legacy_save_image(path, img, hdr, bit_depth=bit_depth, is_mono=is_mono)
                else:
                    # permissive fallback if legacy signature is simpler
                    legacy_save_image(path, img, hdr)
                return True
            except TypeError:
                # Try positional signature: (path, img, hdr, bit_depth, is_mono)
                try:
                    legacy_save_image(path, img, hdr, bit_depth, is_mono)
                    return True
                except Exception:
                    pass
            except Exception:
                pass

        # Fall back: write a FITS with astropy
        try:
            phdu = fits.PrimaryHDU(data=img, header=hdr)
            hdul = fits.HDUList([phdu])
            hdul.writeto(path, overwrite=True)
            return True
        except Exception:
            return False

    def _grab_image_for_save(self, fallback_path: Optional[str]):
        doc = self._get_active_doc()   # <— new
        img = getattr(doc, "image", None) if doc is not None else None
        bit_depth = (doc.metadata.get("bit_depth") if (doc and hasattr(doc, "metadata")) else None)
        is_mono = (doc.metadata.get("is_mono") if (doc and hasattr(doc, "metadata")) else None)
        src = ((doc.metadata.get("file_path") if (doc and hasattr(doc, "metadata")) else None) 
            or fallback_path)

        if img is None and src and legacy_load_image:
            try:
                # common signature from your WIMI usage
                img, orig_hdr, bit_depth, is_mono = legacy_load_image(src)
            except TypeError:
                # try alternate signatures
                try:
                    img, orig_hdr = legacy_load_image(src)
                except Exception:
                    pass
            except Exception:
                pass

        # As a last resort, use data from opened HDUList if present
        if img is None and self.hdul is not None:
            try:
                img = self.hdul[0].data
            except Exception:
                pass

        return img, bit_depth, is_mono, src

    def _add_or_update_keyword(self):
        key = self.add_key_edit.text().strip()
        if not key:
            return
        val = self.add_val_edit.text()
        com = self.add_com_edit.text()
        try:
            parsed_val = self._parse_val(val)
        except Exception:
            parsed_val = val  # leave as string if parsing fails

        targets = range(len(self.hdul)) if self.all_hdus_chk.isChecked() and self.hdul else [self.current_hdu_index]
        for idx in targets:
            hdr = self.hdul[idx].header
            hdr[key] = (parsed_val, com if com else None)

        if not self.all_hdus_chk.isChecked():
            self._populate_tree_from_header(self.hdul[self.current_hdu_index].header)

    def _delete_selected(self):
        items = self.tree.selectedItems()
        if not items:
            return
        targets = range(len(self.hdul)) if self.all_hdus_chk.isChecked() and self.hdul else [self.current_hdu_index]
        for it in items:
            key = it.text(0).strip()
            for idx in targets:
                hdr = self.hdul[idx].header
                if key in ("HISTORY", "COMMENT"):
                    rebuilt = fits.Header()
                    for c in hdr.cards:
                        if c.keyword == key:
                            if (key == "HISTORY" and c.value == it.text(1)) or \
                               (key == "COMMENT" and (c.value == it.text(1) or c.comment == it.text(2))):
                                continue
                        rebuilt.append(c)
                    hdr.clear(); hdr.update(rebuilt)
                else:
                    if key in hdr:
                        del hdr[key]
        self._populate_tree_from_header(self.hdul[self.current_hdu_index].header)

    # ---- value parsing helpers ----
    def _parse_val(self, s: str):
        if s is None:
            return ""
        t = s.strip()
        if t.lower() in ("true", "t"): return True
        if t.lower() in ("false", "f"): return False
        if t.lower() in ("nan",): return np.nan
        try:
            if t.startswith("0x"):
                return int(t, 16)
            return int(t)
        except ValueError:
            pass
        try:
            return float(t)
        except ValueError:
            pass
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            return t[1:-1]
        return t

    def _val_to_str(self, v):
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return "nan"
        return str(v)

    def closeEvent(self, e):
        try:
            if self.hdul is not None:
                self.hdul.close()
        except Exception:
            pass
        super().closeEvent(e)

    # ---- QSettings helpers ----
    def _settings(self):
        return self.parent().settings if (self.parent() and hasattr(self.parent(), "settings")) else QSettings()
    def _last_dir(self):
        return self._settings().value("fits_modifier/last_dir", "", type=str) or ""
    def _save_last_dir(self, d):
        self._settings().setValue("fits_modifier/last_dir", d)


class BatchFITSHeaderDialog(QDialog):
    def __init__(self, parent=None, preset_keyword: str = "", preset_value: str = "", preset_comment: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Batch Modify FITS Headers")
        self.resize(520, 220)

        v = QVBoxLayout(self)

        row1 = QHBoxLayout()
        self.files_edit = QLineEdit(); self.files_edit.setPlaceholderText("No files selected")
        self.pick_btn = QPushButton("Choose FITS Files…")
        row1.addWidget(self.files_edit, 1); row1.addWidget(self.pick_btn)

        row2 = QHBoxLayout()
        self.key_edit = QLineEdit(); self.key_edit.setPlaceholderText("KEYWORD")
        self.val_edit = QLineEdit(); self.val_edit.setPlaceholderText("Value (leave blank for delete)")
        self.com_edit = QLineEdit(); self.com_edit.setPlaceholderText("Comment (optional)")
        row2.addWidget(self.key_edit); row2.addWidget(self.val_edit); row2.addWidget(self.com_edit)

        row3 = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Add/Update", "Delete"])
        self.all_hdus_chk = QCheckBox("Apply to all HDUs")
        self.add_if_missing_chk = QCheckBox("Add if missing (for Add/Update)")
        self.add_if_missing_chk.setChecked(True)
        row3.addWidget(self.mode_combo)
        row3.addWidget(self.all_hdus_chk)
        row3.addWidget(self.add_if_missing_chk)
        row3.addStretch()

        row4 = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.close_btn = QPushButton("Close")
        row4.addStretch(); row4.addWidget(self.run_btn); row4.addWidget(self.close_btn)

        v.addLayout(row1)
        v.addLayout(row2)
        v.addLayout(row3)
        v.addLayout(row4)

        if preset_keyword:
            self.key_edit.setText(preset_keyword)
        if preset_value:
            self.val_edit.setText(preset_value)
        if preset_comment:
            self.com_edit.setText(preset_comment)

        self.pick_btn.clicked.connect(self._pick_files)
        self.run_btn.clicked.connect(self._run)
        self.close_btn.clicked.connect(self.close)

        self.files = []

    def _settings(self):
        return self.parent().settings if (self.parent() and hasattr(self.parent(), "settings")) else QSettings()

    def _pick_files(self):
        last = self._settings().value("fits_modifier/batch_dir", "", type=str) or ""
        files, _ = QFileDialog.getOpenFileNames(self, "Select FITS files", last, "FITS files (*.fits *.fit *.fts *.fz)")
        if not files:
            return
        self.files = files
        self.files_edit.setText(f"{len(files)} files selected")
        self._settings().setValue("fits_modifier/batch_dir", os.path.dirname(files[0]))

    def _parse_val(self, s: str):
        t = (s or "").strip()
        if t == "": return ""
        if t.lower() in ("true", "t"): return True
        if t.lower() in ("false", "f"): return False
        if t.lower() in ("nan",): return np.nan
        try:
            if t.startswith("0x"):
                return int(t, 16)
            return int(t)
        except ValueError:
            pass
        try:
            return float(t)
        except ValueError:
            pass
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            return t[1:-1]
        return t

    def _run(self):
        if not self.files:
            QMessageBox.warning(self, "No files", "Please choose one or more FITS files.")
            return
        key = self.key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "Missing keyword", "Please enter a FITS keyword.")
            return

        mode = self.mode_combo.currentText()
        apply_all_hdus = self.all_hdus_chk.isChecked()
        add_if_missing = self.add_if_missing_chk.isChecked()
        com = self.com_edit.text().strip()
        value_txt = self.val_edit.text()

        n_ok, n_err = 0, 0
        for fp in self.files:
            try:
                with fits.open(fp, mode='update', memmap=False) as hdul:
                    targets = range(len(hdul)) if apply_all_hdus else [0]
                    if mode == "Delete":
                        for i in targets:
                            hdr = hdul[i].header
                            if key in ("HISTORY", "COMMENT"):
                                rebuilt = fits.Header()
                                for c in hdr.cards:
                                    if c.keyword == key:
                                        if value_txt and str(c.value) == value_txt:
                                            continue
                                        if (not value_txt) and (not com):
                                            continue
                                        if com and (c.comment == com):
                                            continue
                                    rebuilt.append(c)
                                hdr.clear(); hdr.update(rebuilt)
                            else:
                                if key in hdr:
                                    del hdr[key]
                        hdul.flush()
                    else:
                        try:
                            val = self._parse_val(value_txt)
                        except Exception:
                            val = value_txt
                        for i in targets:
                            hdr = hdul[i].header
                            if key in hdr or add_if_missing:
                                hdr[key] = (val, com if com else None)
                        hdul.flush()
                n_ok += 1
            except Exception as e:
                print(f"[Batch FITS] Error on {fp}: {e}")
                n_err += 1

        QMessageBox.information(self, "Batch Complete", f"Updated {n_ok} file(s); {n_err} error(s).")
