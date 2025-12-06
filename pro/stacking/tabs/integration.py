# Tab module - imports from parent packages
from __future__ import annotations
import os
import sys
import platform
import math
import time
import numpy as np
import cv2
cv2.setNumThreads(0)

from PyQt6.QtCore import Qt, QObject, QThread, QTimer, pyqtSlot, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QFileDialog, QAbstractItemView,
    QProgressDialog, QApplication, QMessageBox, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QToolButton, QLineEdit, QSlider,
    QInputDialog, QMenu
)
from astropy.io import fits
from datetime import datetime

# Import shared utilities from project
from legacy.image_manager import load_image, save_image
from legacy.numba_utils import debayer_raw_fast


class IntegrationTab(QObject):
    """Extracted Integration tab functionality."""
    
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    def stack_images_mixed_drizzle(
        self,
        grouped_files,           # { group_key: [aligned _n_r.fit paths] }
        frame_weights,           # { file_path: weight }
        transforms_dict,         # { normalized_path -> aligned_path } (kept for compatibility)
        drizzle_dict,            # { group_key: {drizzle_enabled, scale_factor, drop_shrink} }
        *,
        autocrop_enabled: bool,
        autocrop_pct: float,
        status_cb=None
    ):
        """
        Runs normal integration (to get rejection coords), saves masters,
        optionally runs comet-mode stacks and drizzle. Designed to run in a worker thread.

        Returns:
            {
            "summary_lines": [str, ...],
            "autocrop_outputs": [(group_key, out_path_crop), ...]
            }
        """
        log = status_cb or (lambda *_: None)
        comet_mode = bool(getattr(self, "comet_cb", None) and self.main.comet_cb.isChecked())
        if not hasattr(self, "orig_by_aligned") or not isinstance(getattr(self, "orig_by_aligned"), dict):
            self.main.orig_by_aligned = {}
        # Comet-mode defaults (surface later if desired)
        COMET_ALGO = "Comet High-Clip Percentile"          # or "Comet Lower-Trim (30%)"
        STARS_ALGO = "Comet High-Clip Percentile"          # star-aligned stack: median best suppresses moving comet

        n_groups = len(grouped_files)
        n_frames = sum(len(v) for v in grouped_files.values())
        log(f"📁 Post-align: {n_groups} group(s), {n_frames} aligned frame(s).")
        QApplication.processEvents()

        # Precompute a single global crop rect if enabled (pure computation, no UI).
        global_rect = None
        if autocrop_enabled:
            log("✂️ Auto Crop Enabled. Calculating bounding box…")

            # --- FAST PATH: use transforms (no image I/O) ---------------------------
            try:
                # Prefer model-aware (drizzle) xforms if present; else affine fallback
                xforms = dict(getattr(self, "drizzle_xforms", {}) or {})
                if not xforms:
                    xforms = dict(getattr(self, "valid_matrices", {}) or {})

                # Geometry of the reference (H, W) captured at alignment time
                ref_hw = tuple(getattr(self, "ref_shape_for_drizzle", ()))
                if xforms and len(ref_hw) == 2:
                    global_rect = self.main._rect_from_transforms_fast(
                        xforms,
                        src_hw=ref_hw,
                        coverage_pct=float(autocrop_pct),
                        allow_homography=True,   # supports 3x3 too
                        min_side=16
                    )
                    if global_rect:
                        x0, y0, x1, y1 = map(int, global_rect)
                        log(f"✂️ Transform crop (global) → [{x0}:{x1}]×[{y0}:{y1}] ({x1-x0}×{y1-y0})")
                    else:
                        log("✂️ Transform crop produced no stable global rect; falling back to mask-based.")
                else:
                    log("✂️ No transforms/geometry available for fast global crop; falling back to mask-based.")
            except Exception as e:
                log(f"⚠️ Transform-based global crop failed ({e}); falling back to mask-based.")
                global_rect = None

            # --- SLOW FALLBACK: your existing mask-based method ---------------------
            if global_rect is None:
                try:
                    global_rect = self.main._compute_common_autocrop_rect(
                        grouped_files, autocrop_pct, status_cb=log
                    )
                    if global_rect:
                        x0, y0, x1, y1 = map(int, global_rect)
                        log(f"✂️ Mask-based crop (global) → [{x0}:{x1}]×[{y0}:{y1}] ({x1-x0}×{y1-y0})")
                    else:
                        log("✂️ Global crop disabled; will fall back to per-group.")
                except Exception as e:
                    global_rect = None
                    log(f"⚠️ Global crop (mask-based) failed: {e}")
        QApplication.processEvents()

        group_integration_data = {}
        summary_lines = []
        autocrop_outputs = []

        for gi, (group_key, file_list) in enumerate(grouped_files.items(), 1):
            t_g = perf_counter()
            log(f"🔹 [{gi}/{n_groups}] Integrating '{group_key}' with {len(file_list)} file(s)…")
            QApplication.processEvents()

            # ---- STARS (reference-aligned) integration ----
            # Force a comet-safe reducer for the star-aligned stack only when comet_mode is on.
            integrated_image, rejection_map, ref_header = self.normal_integration_with_rejection(
                group_key, file_list, frame_weights,
                status_cb=log,
                algo_override=(STARS_ALGO if comet_mode else None)   # << correct: stars use STARS_ALGO in comet mode
            )
            log(f"   ↳ Integration done in {perf_counter() - t_g:.1f}s.")
            QApplication.processEvents()
            if integrated_image is None:
                continue

            if ref_header is None:
                ref_header = fits.Header()

            # --- Save the non-cropped STAR master (MEF w/ rejection layers if present) ---
            hdr_orig = ref_header.copy()
            hdr_orig["IMAGETYP"] = "MASTER STACK"
            hdr_orig["BITPIX"]   = -32
            hdr_orig["STACKED"]  = (True, "Stacked using normal_integration_with_rejection")
            hdr_orig["CREATOR"]  = "SetiAstroSuite"
            hdr_orig["DATE-OBS"] = datetime.utcnow().isoformat()

            is_mono_orig = (integrated_image.ndim == 2)
            if is_mono_orig:
                hdr_orig["NAXIS"]  = 2
                hdr_orig["NAXIS1"] = integrated_image.shape[1]
                hdr_orig["NAXIS2"] = integrated_image.shape[0]
                if "NAXIS3" in hdr_orig:
                    del hdr_orig["NAXIS3"]
            else:
                hdr_orig["NAXIS"]  = 3
                hdr_orig["NAXIS1"] = integrated_image.shape[1]
                hdr_orig["NAXIS2"] = integrated_image.shape[0]
                hdr_orig["NAXIS3"] = integrated_image.shape[2]

            n_frames_group = len(file_list)
            H, W = integrated_image.shape[:2]
            display_group = self.main._label_with_dims(group_key, W, H)
            base = f"MasterLight_{display_group}_{n_frames_group}stacked"
            base = self.main._normalize_master_stem(base)
            out_path_orig = self.main._build_out(self.main.stacking_directory, base, "fit")

            # Try to attach rejection maps that were accumulated during integration
            maps = getattr(self, "_rej_maps", {}).get(group_key)
            save_layers = self.main.settings.value("stacking/save_rejection_layers", True, type=bool)

            if maps and save_layers:
                try:
                    _save_master_with_rejection_layers(
                        integrated_image,
                        hdr_orig,
                        out_path_orig,
                        rej_any = maps.get("any"),
                        rej_frac= maps.get("frac"),
                    )
                    log(f"✅ Saved integrated image (with rejection layers) for '{group_key}': {out_path_orig}")
                except Exception as e:
                    log(f"⚠️ MEF save failed ({e}); falling back to single-HDU save.")
                    save_image(
                        img_array=integrated_image,
                        filename=out_path_orig,
                        original_format="fit",
                        bit_depth="32-bit floating point",
                        original_header=hdr_orig,
                        is_mono=is_mono_orig
                    )
                    log(f"✅ Saved integrated image (single-HDU) for '{group_key}': {out_path_orig}")
            else:
                # No maps available or feature disabled → single-HDU save
                save_image(
                    img_array=integrated_image,
                    filename=out_path_orig,
                    original_format="fit",
                    bit_depth="32-bit floating point",
                    original_header=hdr_orig,
                    is_mono=is_mono_orig
                )
                log(f"✅ Saved integrated image (original) for '{group_key}': {out_path_orig}")

            # ---- Decide the group’s fixed crop rect (used for ALL outputs in this group) ----
            group_rect = None
            if autocrop_enabled:
                if global_rect is not None:
                    group_rect = tuple(global_rect)
                else:
                    # FAST per-group rect from transforms (no image I/O)
                    try:
                        # Build subset of transforms for just this group's aligned paths
                        xforms_g = {}
                        # orig_by_aligned maps registered path -> original/normalized path
                        oba = getattr(self, "orig_by_aligned", {}) or {}
                        has_model_xforms = bool(getattr(self, "drizzle_xforms", None))

                        for ap in file_list:
                            apn = os.path.normpath(ap)
                            M = None
                            # Prefer model-aware transform keyed by original path
                            if has_model_xforms:
                                op = oba.get(apn)
                                if op:
                                    M = getattr(self, "drizzle_xforms", {}).get(os.path.normpath(op))
                            # Fallback: affine keyed by aligned path
                            if M is None:
                                M = getattr(self, "matrix_by_aligned", {}).get(apn)
                            if M is not None:
                                xforms_g[apn] = np.asarray(M)

                        ref_hw = tuple(getattr(self, "ref_shape_for_drizzle", ()))
                        if xforms_g and len(ref_hw) == 2:
                            group_rect = self.main._rect_from_transforms_fast(
                                xforms_g,
                                src_hw=ref_hw,
                                coverage_pct=float(autocrop_pct),
                                allow_homography=True,
                                min_side=16
                            )

                        if group_rect is None:
                            # Fallback to existing mask-based per-group method
                            group_rect = self.main._compute_common_autocrop_rect(
                                {group_key: file_list}, autocrop_pct, status_cb=log
                            )
                    except Exception as e:
                        log(f"⚠️ Per-group transform crop failed for '{group_key}': {e}")
                        group_rect = None

                if group_rect:
                    x1, y1, x2, y2 = map(int, group_rect)
                    log(f"✂️ Using fixed crop rect for '{group_key}': ({x1},{y1})–({x2},{y2})")
                else:
                    log("✂️ No stable rect found for this group; per-image fallback will be used.")

            # --- Optional: auto-cropped STAR copy (uses group_rect if available, else global/per-image logic) ---
            if autocrop_enabled:
                cropped_img, hdr_crop = self.main._apply_autocrop(
                    integrated_image,
                    file_list,
                    ref_header.copy(),
                    scale=1.0,
                    rect_override=group_rect if group_rect is not None else global_rect
                )
                is_mono_crop = (cropped_img.ndim == 2)
                Hc, Wc = (cropped_img.shape[:2] if cropped_img.ndim >= 2 else (H, W))
                display_group_crop = self.main._label_with_dims(group_key, Wc, Hc)
                base_crop = f"MasterLight_{display_group_crop}_{n_frames_group}stacked_autocrop"
                base_crop = self.main._normalize_master_stem(base_crop)
                out_path_crop = self.main._build_out(self.main.stacking_directory, base_crop, "fit")

                save_image(
                    img_array=cropped_img,
                    filename=out_path_crop,
                    original_format="fit",
                    bit_depth="32-bit floating point",
                    original_header=hdr_crop,
                    is_mono=is_mono_crop
                )
                log(f"✂️ Saved auto-cropped image for '{group_key}': {out_path_crop}")
                autocrop_outputs.append((group_key, out_path_crop))

            # ---- Optional: COMET mode ----
            if comet_mode:
                log("🌠 Comet mode enabled for this group")

                # registered, time-sorted
                sorted_files = sorted(file_list, key=CS.time_key)

                # Build seeds in the *registered* space
                seeds = {}
                reg_path = None  # ensure defined for logging checks below
                if hasattr(self, "_comet_seed") and self.main._comet_seed:
                    seed_src_path = os.path.normpath(self.main._comet_seed.get("path", ""))
                    seed_xy       = tuple(self.main._comet_seed.get("xy", (0.0, 0.0)))

                    # 1) try exact mapping: original/normalized -> registered path
                    if transforms_dict:
                        reg_path = transforms_dict.get(seed_src_path)

                    # 2) fuzzy fallback by basename prefix (handles _n -> _n_r)
                    if not reg_path:
                        bn = os.path.basename(os.path.splitext(seed_src_path)[0])
                        for p in sorted_files:
                            if os.path.basename(p).startswith(bn):
                                reg_path = p
                                break

                    # 3) transform XY into registered frame
                    if reg_path:
                        M = self.main.matrix_by_aligned.get(os.path.normpath(reg_path))
                        if M is not None and np.asarray(M).shape == (2,3):
                            x, y = seed_xy
                            a,b,tx = M[0]; c,d,ty = M[1]
                            seeds[os.path.normpath(reg_path)] = (
                                float(a*x + b*y + tx),
                                float(c*x + d*y + ty)
                            )
                            log(f"  ◦ using user seed on {os.path.basename(reg_path)}")
                        else:
                            log("  ⚠️ user seed: no affine for that registered file")

                # 4) Last resort: if no seed mapped to any of the files, drop the reference-frame seed
                if not any(fp in seeds for fp in sorted_files):
                    if getattr(self, "_comet_ref_xy", None):
                        seeds[sorted_files[0]] = tuple(map(float, self.main._comet_ref_xy))
                        log("  ◦ seeding first registered frame with _comet_ref_xy")

                # Sanity log if we actually have a reg_path and seed
                if reg_path and (os.path.normpath(reg_path) in seeds):
                    sx, sy = seeds[os.path.normpath(reg_path)]
                    log(f"  ◦ seed xy={sx:.1f},{sy:.1f} within {W}×{H}? "
                        f"{'OK' if (0<=sx<W and 0<=sy<H) else 'OUT-OF-BOUNDS'}")

                # 1) Measure comet centers (auto baseline)
                log("🟢 Measuring comet centers (template match)…")
                comet_xy = CS.measure_comet_positions(sorted_files, seeds=seeds, status_cb=log)

                # 2) Offer preview (GUI) via worker signal
                ui_target = None
                try:
                    ui_target = self.main._find_ui_target() if hasattr(self, "_find_ui_target") else None
                    if ui_target is None:
                        # inline helper (same as your earlier version)
                        def _find_ui_target() -> QWidget | None:
                            ow = getattr(self, "ui_owner", None)
                            if isinstance(ow, QWidget) and hasattr(ow, "show_comet_preview"):
                                return ow
                            par = self.main.parent()
                            if isinstance(par, QWidget) and hasattr(par, "show_comet_preview"):
                                return par
                            aw = QApplication.activeWindow()
                            if isinstance(aw, QWidget) and hasattr(aw, "show_comet_preview"):
                                return aw
                            for w in QApplication.topLevelWidgets():
                                if hasattr(w, "show_comet_preview"):
                                    return w
                            return None
                        ui_target = _find_ui_target()
                except Exception:
                    ui_target = None

                if ui_target is not None:
                    try:
                        responder = _Responder()
                        loop = QEventLoop()
                        result_box = {"res": None}

                        def _store_and_quit(res):
                            result_box["res"] = res
                            loop.quit()

                        responder.finished.connect(_store_and_quit)

                        emitter = getattr(self, "post_worker", None)
                        if emitter is None:
                            log("  ⚠️ comet preview skipped: no worker emitter present")
                        else:
                            emitter.need_comet_review.emit(sorted_files, comet_xy, responder)
                            loop.exec()  # block this worker thread until GUI responds

                            edited = result_box["res"]
                            if isinstance(edited, dict) and edited:
                                comet_xy = edited
                                log(f"  ◦ user confirmed/edited {len(comet_xy)} centroids")
                            else:
                                log("  ◦ user cancelled or no edits — using auto centroids")
                    except Exception as e:
                        log(f"  ⚠️ comet preview skipped: {e!r}")
                else:
                    log("  ⚠️ comet preview unavailable (no UI target)")

                # 3) Comet-aligned integration
                usable = [fp for fp in sorted_files if fp in comet_xy]
                if len(usable) < 2:
                    log("⚠️ Not enough frames with valid comet centroids; skipping comet stack.")
                else:
                    log("🟠 Comet-aligned integration…")
                    comet_only, comet_rej_map, ref_header_c = self.main.integrate_comet_aligned(
                        group_key=f"{group_key}",
                        file_list=usable,
                        comet_xy=comet_xy,
                        frame_weights=frame_weights,
                        status_cb=log,
                        algo_override=COMET_ALGO  # << comet-friendly reducer
                    )

                    # Save CometOnly
                    Hc, Wc = comet_only.shape[:2]
                    display_group_c = self.main._label_with_dims(group_key, Wc, Hc)
                    comet_path = self.main._build_out(
                        self.main.stacking_directory,
                        f"MasterCometOnly_{display_group_c}_{len(usable)}stacked",
                        "fit"
                    )
                    save_image(
                        comet_only, comet_path, "fit", "32-bit floating point",
                        original_header=(ref_header_c or ref_header),
                        is_mono=(comet_only.ndim==2)
                    )
                    log(f"✅ Saved CometOnly → {comet_path}")

                    # --- Crop CometOnly identically (if requested) ---
                    if autocrop_enabled and (group_rect is not None or global_rect is not None):
                        comet_only_crop, hdr_c_crop = self.main._apply_autocrop(
                            comet_only,
                            file_list,  # ok to reuse; rect is forced
                            (ref_header_c or ref_header).copy(),
                            scale=1.0,
                            rect_override=group_rect if group_rect is not None else global_rect
                        )
                        Hcc, Wcc = comet_only_crop.shape[:2]
                        display_group_cc = self.main._label_with_dims(group_key, Wcc, Hcc)
                        comet_path_crop = self.main._build_out(
                            self.main.stacking_directory,
                            f"MasterCometOnly_{display_group_cc}_{len(usable)}stacked_autocrop",
                            "fit"
                        )
                        save_image(
                            comet_only_crop, comet_path_crop, "fit", "32-bit floating point",
                            original_header=hdr_c_crop,
                            is_mono=(comet_only_crop.ndim==2)
                        )
                        log(f"✂️ Saved CometOnly (auto-cropped) → {comet_path_crop}")

                    # Optional blend
                    if getattr(self, "comet_blend_cb", None) and self.main.comet_blend_cb.isChecked():
                        mix = float(self.main.comet_mix.value())

                        log(f"🟡 Blending Stars+Comet (screen after 5% stretch; mix={mix:.2f})…")
                        stars_img, comet_img = _match_channels(integrated_image, comet_only)

                        # Screen blend after identical display-stretch on both images
                        blend = CS.blend_screen_stretched(
                            comet_only=comet_img,
                            stars_only=stars_img,
                            stretch_pct=0.05,
                            mix=mix
                        )

                        is_mono_blend = (blend.ndim == 2) or (blend.ndim == 3 and blend.shape[2] == 1)
                        blend_path = self.main._build_out(
                            self.main.stacking_directory,
                            f"MasterCometBlend_{display_group_c}_{len(usable)}stacked",
                            "fit"
                        )
                        save_image(blend, blend_path, "fit", "32-bit floating point",
                                ref_header, is_mono=is_mono_blend)
                        log(f"✅ Saved CometBlend → {blend_path}")

                        # --- Crop CometBlend identically (if requested) ---
                        if autocrop_enabled and (group_rect is not None or global_rect is not None):
                            blend_crop, hdr_b_crop = self.main._apply_autocrop(
                                blend,
                                file_list,
                                ref_header.copy(),
                                scale=1.0,
                                rect_override=group_rect if group_rect is not None else global_rect
                            )
                            Hb, Wb = blend_crop.shape[:2]
                            display_group_bc = self.main._label_with_dims(group_key, Wb, Hb)
                            blend_path_crop = self.main._build_out(
                                self.main.stacking_directory,
                                f"MasterCometBlend_{display_group_bc}_{len(usable)}stacked_autocrop",
                                "fit"
                            )
                            save_image(
                                blend_crop, blend_path_crop, "fit", "32-bit floating point",
                                original_header=hdr_b_crop,
                                is_mono=(blend_crop.ndim == 2 or (blend_crop.ndim == 3 and blend_crop.shape[2] == 1))
                            )
                            log(f"✂️ Saved CometBlend (auto-cropped) → {blend_path_crop}")

            # ---- Drizzle bookkeeping for this group ----
            dconf = drizzle_dict.get(group_key, {})
            if dconf.get("drizzle_enabled", False):
                sasr_path = os.path.join(self.main.stacking_directory, f"{group_key}_rejections.sasr")
                self.main.save_rejection_map_sasr(rejection_map, sasr_path)
                log(f"✅ Saved rejection map to {sasr_path}")
                group_integration_data[group_key] = {
                    "integrated_image": integrated_image,
                    "rejection_map": rejection_map,
                    "n_frames": n_frames_group,
                    "drizzled": True
                }
            else:
                group_integration_data[group_key] = {
                    "integrated_image": integrated_image,
                    "rejection_map": None,
                    "n_frames": n_frames_group,
                    "drizzled": False
                }
                log(f"ℹ️ Skipping rejection map save for '{group_key}' (drizzle disabled).")

        QApplication.processEvents()
        # Build ORIGINALS list for each group (needed for true drizzle)
        originals_by_group = {}
        oba = getattr(self, "orig_by_aligned", {}) or {}
        for group, reg_list in grouped_files.items():
            orig_list = []
            for rp in reg_list:
                op = oba.get(os.path.normpath(rp))
                if op:
                    orig_list.append(op)
            originals_by_group[group] = orig_list
        # ---- Drizzle pass (only for groups with drizzle enabled) ----
        for group_key, file_list in grouped_files.items():
            dconf = drizzle_dict.get(group_key)
            if not (dconf and dconf.get("drizzle_enabled", False)):
                log(f"✅ Group '{group_key}' not set for drizzle. Integrated image already saved.")
                continue

            scale_factor = self.main._get_drizzle_scale()
            drop_shrink  = self.main._get_drizzle_pixfrac()

            # Optional: also read kernel for logging/branching
            kernel = (self.main.settings.value("stacking/drizzle_kernel", "square", type=str) or "square").lower()
            status_cb(f"Drizzle cfg → scale={scale_factor}×, pixfrac={drop_shrink:.3f}, kernel={kernel}")
            rejections_for_group = group_integration_data[group_key]["rejection_map"]
            n_frames_group = group_integration_data[group_key]["n_frames"]

            log(f"📐 Drizzle for '{group_key}' at {scale_factor}× (drop={drop_shrink}) using {n_frames_group} frame(s).")

            self.drizzle_stack_one_group(
                group_key=group_key,
                file_list=file_list,                          # registered (for headers/labels)
                original_list=originals_by_group.get(group_key, []),  # <-- NEW
                transforms_dict=transforms_dict,
                frame_weights=frame_weights,
                scale_factor=scale_factor,
                drop_shrink=drop_shrink,
                rejection_map=rejections_for_group,
                autocrop_enabled=autocrop_enabled,
                rect_override=global_rect,
                status_cb=log
            )

        # Build summary lines
        for group_key, info in group_integration_data.items():
            n_frames_group = info["n_frames"]
            drizzled = info["drizzled"]
            summary_lines.append(f"• {group_key}: {n_frames_group} stacked{' + drizzle' if drizzled else ''}")

        if autocrop_outputs:
            summary_lines.append("")
            summary_lines.append("Auto-cropped files saved:")
            for g, p in autocrop_outputs:
                summary_lines.append(f"  • {g} → {p}")

        return {
            "summary_lines": summary_lines,
            "autocrop_outputs": autocrop_outputs
        }




    def drizzle_stack_one_group(
        self,
        *,
        group_key,
        file_list,           # registered _n_r.fit (keep for headers/metadata)
        original_list,       # NEW: originals (normalized), used as pixel sources
        transforms_dict,
        frame_weights,
        scale_factor,
        drop_shrink,
        rejection_map,
        autocrop_enabled,
        rect_override,
        status_cb
    ):
        # Load per-frame transforms (SASD v2)
        log = status_cb or (lambda *_: None)        
        sasd_path = os.path.join(self.main.stacking_directory, "alignment_transforms.sasd")
        ref_H, ref_W, xforms = self.main._load_sasd_v2(sasd_path)
        if not (ref_H and ref_W):
            # Fallback to in-memory shape captured at alignment time
            rs = getattr(self, "ref_shape_for_drizzle", None)
            if isinstance(rs, tuple) and len(rs) == 2 and all(int(v) > 0 for v in rs):
                ref_H, ref_W = int(rs[0]), int(rs[1])
                status_cb(f"ℹ️ Using in-memory REF_SHAPE fallback: {ref_H}×{ref_W}")
            else:
                status_cb("⚠️ Missing REF_SHAPE in SASD; cannot drizzle.")
                return

        log(f"✅ SASD v2: loaded {len(xforms)} transform(s).")
        # Debug (first few):
        try:
            sample_need = [os.path.basename(p) for p in original_list[:5]]
            sample_have = [os.path.basename(p) for p in list(xforms.keys())[:5]]
            log(f"   originals needed (sample): {sample_need}")
            log(f"   sasd FILEs (sample):       {sample_have}")
        except Exception:
            pass

        canvas_H, canvas_W = int(ref_H * scale_factor), int(ref_W * scale_factor)        


        # --- kernel config from settings ---
        kernel_name = self.main.settings.value("stacking/drizzle_kernel", "square", type=str).lower()
        gauss_sigma = self.main.settings.value(
            "stacking/drizzle_gauss_sigma", float(drop_shrink) * 0.5, type=float
        )
        if kernel_name.startswith("gauss"):
            _kcode = 2
        elif kernel_name.startswith("circ"):
            _kcode = 1
        else:
            _kcode = 0  # square

        total_rej = sum(len(v) for v in (rejection_map or {}).values())
        log(f"🔭 Drizzle stacking for group '{group_key}' with {total_rej} total rejected pixels.")

        if len(file_list) < 2:
            log(f"⚠️ Group '{group_key}' does not have enough frames to drizzle.")
            return

        # --- establish geometry + is_mono before choosing depositor ---
        first_file = file_list[0]
        first_img, hdr, _, _ = load_image(first_file)
        if first_img is None:
            log(f"⚠️ Could not load {first_file} to determine drizzle shape!")
            return

        if first_img.ndim == 2:
            is_mono = True
            h, w = first_img.shape
            c = 1
        else:
            is_mono = False
            h, w, c = first_img.shape

        # --- choose depositor ONCE (and log it) ---
        if _kcode == 0 and drop_shrink >= 0.99:
            # square + pixfrac≈1 → naive “one-to-one” deposit
            deposit_func = drizzle_deposit_numba_naive if is_mono else drizzle_deposit_color_naive
            kinf = "naive (square, pixfrac≈1)"
        else:
            # Any other case → kernelized path (square/circular/gaussian)
            deposit_func = drizzle_deposit_numba_kernel_mono if is_mono else drizzle_deposit_color_kernel
            kinf = ["square", "circular", "gaussian"][_kcode]
        log(f"Using {kinf} kernel drizzle ({'mono' if is_mono else 'color'}).")

        # --- allocate buffers ---
        out_h = int(canvas_H)
        out_w = int(canvas_W)
        drizzle_buffer  = np.zeros((out_h, out_w) if is_mono else (out_h, out_w, c), dtype=self.main._dtype())
        coverage_buffer = np.zeros_like(drizzle_buffer, dtype=self.main._dtype())
        finalize_func   = finalize_drizzle_2d if is_mono else finalize_drizzle_3d

        def _invert_2x3(A23: np.ndarray) -> np.ndarray:
            A23 = np.asarray(A23, np.float32).reshape(2, 3)
            A33 = np.eye(3, dtype=np.float32); A33[:2] = A23
            return np.linalg.inv(A33)  # 3x3

        def _apply_H_point(H: np.ndarray, x: float, y: float) -> tuple[int, int]:
            v = H @ np.array([x, y, 1.0], dtype=np.float32)
            if abs(v[2]) < 1e-8:
                return (int(round(v[0])), int(round(v[1])))
            return (int(round(v[0] / v[2])), int(round(v[1] / v[2])))


        # --- main loop ---
        # Map original (normalized) → aligned path (for weights & rejection map lookups)
        orig_to_aligned = {}
        for op in original_list:
            ap = transforms_dict.get(os.path.normpath(op))
            if ap:
                orig_to_aligned[os.path.normpath(op)] = os.path.normpath(ap)

        for orig_file in original_list:
            orig_key = os.path.normpath(orig_file)
            aligned_file = orig_to_aligned.get(orig_key)

            weight = frame_weights.get(aligned_file, frame_weights.get(orig_key, 1.0))

            kind, X = xforms.get(orig_key, (None, None))
            log(f"🧭 Drizzle uses {kind or '-'} for {os.path.basename(orig_key)}")
            if kind is None:
                log(f"⚠️ No usable transform for {os.path.basename(orig_file)} – skipping")
                continue

            # --- choose pixel source + mapping ---
            pixels_are_registered = False
            img_data = None

            if isinstance(kind, str) and (kind.startswith("poly") or kind in ("tps","thin_plate_spline")):
                # Already warped to reference during registration
                pixel_path = aligned_file
                if not pixel_path:
                    log(f"⚠️ {kind} frame has no aligned counterpart – skipping {os.path.basename(orig_file)}")
                    continue
                H_canvas = np.eye(3, dtype=np.float32)
                pixels_are_registered = True

            elif kind == "affine" and X is not None:
                pixel_path = orig_file
                H_canvas = np.eye(3, dtype=np.float32)
                H_canvas[:2] = np.asarray(X, np.float32).reshape(2, 3)

            elif kind == "homography" and X is not None:
                # Pre-warp originals -> reference, then identity deposit
                pixel_path = orig_file
                raw_img, _, _, _ = load_image(pixel_path)
                if raw_img is None:
                    log(f"⚠️ Failed to read {os.path.basename(pixel_path)} – skipping")
                    continue
                H = np.asarray(X, np.float32).reshape(3, 3)
                if raw_img.ndim == 2:
                    img_data = cv2.warpPerspective(
                        raw_img, H, (ref_W, ref_H),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )
                else:
                    img_data = np.stack([
                        cv2.warpPerspective(
                            raw_img[..., ch], H, (ref_W, ref_H),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0
                        ) for ch in range(raw_img.shape[2])
                    ], axis=2)
                H_canvas = np.eye(3, dtype=np.float32)
                pixels_are_registered = True

            else:
                log(f"⚠️ Unsupported transform '{kind}' – skipping {os.path.basename(orig_file)}")
                continue

            # read pixels if not produced above
            if img_data is None:
                img_data, _, _, _ = load_image(pixel_path)
                if img_data is None:
                    log(f"⚠️ Failed to read {os.path.basename(pixel_path)} – skipping")
                    continue

            # --- debug bbox once ---
            if orig_file is original_list[0]:
                x0, y0 = 0, 0; x1, y1 = ref_W-1, ref_H-1
                p0 = H_canvas @ np.array([x0, y0, 1], np.float32); p0 /= max(p0[2], 1e-8)
                p1 = H_canvas @ np.array([x1, y1, 1], np.float32); p1 /= max(p1[2], 1e-8)
                log(f"   bbox(ref)→reg: ({p0[0]:.1f},{p0[1]:.1f}) to ({p1[0]:.1f},{p1[1]:.1f}); "
                    f"canvas {int(ref_W*scale_factor)}×{int(ref_H*scale_factor)} @ {scale_factor}×")

            # --- apply per-file rejections ---
            if rejection_map and aligned_file in rejection_map:
                coords_for_this_file = rejection_map.get(aligned_file, [])
                if coords_for_this_file:
                    dilate_px = int(self.main.settings.value("stacking/reject_dilate_px", 0, type=int))
                    dilate_shape = self.main.settings.value("stacking/reject_dilate_shape", "square", type=str).lower()
                    offsets = [(0, 0)]
                    if dilate_px > 0:
                        r = dilate_px
                        offsets = [(dx, dy) for dx in range(-r, r+1) for dy in range(-r, r+1)]
                        if dilate_shape.startswith("dia"):
                            offsets = [(dx, dy) for (dx,dy) in offsets if (abs(dx)+abs(dy) <= r)]

                    Hraw, Wraw = img_data.shape[0], img_data.shape[1]

                    if pixels_are_registered:
                        # Directly zero registered pixels
                        for (x_r, y_r) in coords_for_this_file:
                            for (ox, oy) in offsets:
                                xr, yr = x_r + ox, y_r + oy
                                if 0 <= xr < Wraw and 0 <= yr < Hraw:
                                    img_data[yr, xr] = 0.0
                    else:
                        # Back-project via inverse affine
                        Hinv = np.linalg.inv(H_canvas)
                        for (x_r, y_r) in coords_for_this_file:
                            for (ox, oy) in offsets:
                                xr, yr = x_r + ox, y_r + oy
                                v = Hinv @ np.array([xr, yr, 1.0], np.float32)
                                x_raw = int(round(v[0] / max(v[2], 1e-8)))
                                y_raw = int(round(v[1] / max(v[2], 1e-8)))
                                if 0 <= x_raw < Wraw and 0 <= y_raw < Hraw:
                                    img_data[y_raw, x_raw] = 0.0

            # --- deposit (identity for registered pixels) ---
            if deposit_func is drizzle_deposit_numba_naive:
                drizzle_buffer, coverage_buffer = deposit_func(
                    img_data, H_canvas[:2], drizzle_buffer, coverage_buffer, scale_factor, weight
                )
            elif deposit_func is drizzle_deposit_color_naive:
                drizzle_buffer, coverage_buffer = deposit_func(
                    img_data, H_canvas[:2], drizzle_buffer, coverage_buffer, scale_factor, drop_shrink, weight
                )
            else:
                A23 = H_canvas[:2, :]
                drizzle_buffer, coverage_buffer = deposit_func(
                    img_data, A23, drizzle_buffer, coverage_buffer,
                    scale_factor, drop_shrink, weight, _kcode, float(gauss_sigma)
                )


        # --- finalize, save, optional autocrop ---
        final_drizzle = np.zeros_like(drizzle_buffer, dtype=np.float32)
        final_drizzle = finalize_func(drizzle_buffer, coverage_buffer, final_drizzle)

        # Save original drizzle (single-HDU; no rejection layers here)
        Hd, Wd = final_drizzle.shape[:2] if final_drizzle.ndim >= 2 else (0, 0)
        display_group_driz = self.main._label_with_dims(group_key, Wd, Hd)
        base_stem = f"MasterLight_{display_group_driz}_{len(file_list)}stacked_drizzle"
        base_stem = self.main._normalize_master_stem(base_stem) 
        out_path_orig = self.main._build_out(self.main.stacking_directory, base_stem, "fit")

        hdr_orig = hdr.copy() if hdr is not None else fits.Header()
        hdr_orig["IMAGETYP"]   = "MASTER STACK - DRIZZLE"
        hdr_orig["DRIZFACTOR"] = (float(scale_factor), "Drizzle scale factor")
        hdr_orig["DROPFRAC"]   = (float(drop_shrink),  "Drizzle drop shrink/pixfrac")
        hdr_orig["CREATOR"]    = "SetiAstroSuite"
        hdr_orig["DATE-OBS"]   = datetime.utcnow().isoformat()

        if final_drizzle.ndim == 2:
            hdr_orig["NAXIS"]  = 2
            hdr_orig["NAXIS1"] = final_drizzle.shape[1]
            hdr_orig["NAXIS2"] = final_drizzle.shape[0]
            if "NAXIS3" in hdr_orig:
                del hdr_orig["NAXIS3"]
        else:
            hdr_orig["NAXIS"]  = 3
            hdr_orig["NAXIS1"] = final_drizzle.shape[1]
            hdr_orig["NAXIS2"] = final_drizzle.shape[0]
            hdr_orig["NAXIS3"] = final_drizzle.shape[2]

        is_mono_driz = (final_drizzle.ndim == 2)

        save_image(
            img_array=final_drizzle,
            filename=out_path_orig,
            original_format="fit",
            bit_depth="32-bit floating point",
            original_header=hdr_orig,
            is_mono=is_mono_driz
        )
        log(f"✅ Drizzle (original) saved: {out_path_orig}")

        # Optional auto-crop (respects global rect if provided)
        if autocrop_enabled:
            cropped_drizzle, hdr_crop = self.main._apply_autocrop(
                final_drizzle,
                file_list,
                hdr.copy() if hdr is not None else fits.Header(),
                scale=float(scale_factor),
                rect_override=rect_override
            )
            is_mono_crop = (cropped_drizzle.ndim == 2)
            display_group_driz_crop = self.main._label_with_dims(group_key, cropped_drizzle.shape[1], cropped_drizzle.shape[0])
            base_crop = f"MasterLight_{display_group_driz_crop}_{len(file_list)}stacked_drizzle_autocrop"
            base_crop = self.main._normalize_master_stem(base_crop) 
            out_path_crop = self.main._build_out(self.main.stacking_directory, base_crop, "fit")

            save_image(
                img_array=cropped_drizzle,
                filename=out_path_crop,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=hdr_crop,
                is_mono=is_mono_crop
            )
            if not hasattr(self, "_autocrop_outputs"):
                self.main._autocrop_outputs = []
            self.main._autocrop_outputs.append((group_key, out_path_crop))
            log(f"✂️ Drizzle (auto-cropped) saved: {out_path_crop}")


    def normal_integration_with_rejection(
        self,
        group_key,
        file_list,
        frame_weights,
        status_cb=None,
        *,
        algo_override: str | None = None
    ):
        log = status_cb or (lambda *_: None)
        log(f"Starting integration for group '{group_key}' with {len(file_list)} files.")
        if not file_list:
            return None, {}, None

        # --- reference frame (unchanged) ---
        ref_file = file_list[0]
        ref_data, ref_header, _, _ = load_image(ref_file)
        if ref_data is None:
            log(f"⚠️ Could not load reference '{ref_file}' for group '{group_key}'.")
            return None, {}, None
        if ref_header is None:
            ref_header = fits.Header()

        is_color = (ref_data.ndim == 3 and ref_data.shape[2] == 3)
        height, width = ref_data.shape[:2]
        channels = 3 if is_color else 1
        N = len(file_list)

        algo = (algo_override or self.main.rejection_algorithm)
        use_gpu = bool(self.main._hw_accel_enabled()) and _torch_ok() and _gpu_algo_supported(algo)

        log(f"📊 Stacking group '{group_key}' with {algo}{' [GPU]' if use_gpu else ''}")

        # --- keep all FITSes open (memmap) once for the whole group ---
        sources = []
        try:
            for p in file_list:
                sources.append(_MMImage(p))   # << was _MMFits
        except Exception as e:
            for s in sources:
                try: s.close()
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            log(f"⚠️ Failed to open images (memmap): {e}")
            return None, {}, None

        DTYPE = self.main._dtype()
        # Use smart_zeros for large arrays - will use memmap if > 500MB
        integrated_image, integrated_memmap_path = smart_zeros((height, width, channels), dtype=DTYPE)
        per_file_rejections = {f: [] for f in file_list}

        # --- chunk size ---
        pref_h = self.main.chunk_height
        pref_w = self.main.chunk_width
        try:
            chunk_h, chunk_w = compute_safe_chunk(height, width, N, channels, DTYPE, pref_h, pref_w)
            log(f"🔧 Using chunk size {chunk_h}×{chunk_w} for {DTYPE}")
        except MemoryError as e:
            for s in sources: s.close()
            log(f"⚠️ {e}")
            return None, {}, None

        # --- reusable C-order tile buffers (avoid copies before GPU) ---
        # Use pinned memory only if we’ll actually ship tiles to GPU.
        def _mk_buf():
            buf = np.empty((N, chunk_h, chunk_w, channels), dtype=np.float32, order='C')
            if use_gpu:
                # Mark as pinned (page-locked) so torch can H->D quickly if _torch_reduce_tile uses it.
                try:
                    import torch
                    # torch can wrap numpy pinned via from_numpy(...).pin_memory() ONLY on tensors.
                    # We'll keep numpy here; _torch_reduce_tile will move to pinned tensors internally.
                    # So no-op here to avoid extra copies.
                except Exception:
                    pass
            return buf

        buf0 = _mk_buf()
        buf1 = _mk_buf()

        # --- weights once per group ---
        weights_array = np.array([frame_weights.get(p, 1.0) for p in file_list], dtype=np.float32)

        n_rows  = math.ceil(height / chunk_h)
        n_cols  = math.ceil(width  / chunk_w)
        total_tiles = n_rows * n_cols

        # --- group-level rejection maps (RAM-light) ---
        rej_any   = np.zeros((height, width), dtype=np.bool_)
        rej_count = np.zeros((height, width), dtype=np.uint16)

        # --------- helper: read a tile into a provided buffer (blocking) ----------
        def _read_tile_into(buf, y0, y1, x0, x1):
            th = y1 - y0
            tw = x1 - x0
            # slice view (C-order)
            ts = buf[:N, :th, :tw, :channels]
            # sequential, low-overhead sliced reads (OS prefetch + memmap)
            for i, src in enumerate(sources):
                sub = src.read_tile(y0, y1, x0, x1)  # float32, (th,tw) or (th,tw,3)
                if sub.ndim == 2:
                    if channels == 3:
                        sub = sub[:, :, None].repeat(3, axis=2)
                    else:
                        sub = sub[:, :, None]
                ts[i, :, :, :] = sub
            return th, tw  # actual extents for edge tiles

        # Prefetcher (single background worker is enough; IO is the bottleneck)
        from concurrent.futures import ThreadPoolExecutor
        tp = ThreadPoolExecutor(max_workers=1)

        # Precompute tile grid
        tiles = []
        for y0 in range(0, height, chunk_h):
            y1 = min(y0 + chunk_h, height)
            for x0 in range(0, width, chunk_w):
                x1 = min(x0 + chunk_w, width)
                tiles.append((y0, y1, x0, x1))

        # Prime first read
        tile_idx = 0
        y0, y1, x0, x1 = tiles[0]
        fut = tp.submit(_read_tile_into, buf0, y0, y1, x0, x1)
        use_buf0 = True

        # Torch inference guard (if available)
        _ctx = _safe_torch_inference_ctx() if use_gpu else contextlib.nullcontext
        with _ctx():
            for tile_idx, (y0, y1, x0, x1) in enumerate(tiles, start=1):
                t0 = time.perf_counter()

                # Wait for current tile to be ready
                th, tw = fut.result()
                ts = (buf0 if use_buf0 else buf1)[:N, :th, :tw, :channels]

                # Kick off prefetch for the NEXT tile (if any) into the other buffer
                if tile_idx < total_tiles:
                    ny0, ny1, nx0, nx1 = tiles[tile_idx]
                    fut = tp.submit(_read_tile_into, (buf1 if use_buf0 else buf0), ny0, ny1, nx0, nx1)

                # --- rejection/integration for this tile ---
                log(f"Integrating tile {tile_idx}/{total_tiles} "
                    f"[y:{y0}:{y1} x:{x0}:{x1} size={th}×{tw}] "
                    f"mode={'GPU' if use_gpu else 'CPU'}…")

                if use_gpu:
                    print(f"Using GPU for tile {tile_idx} with algo {algo}")
                    tile_result, tile_rej_map = _torch_reduce_tile(
                        ts,                         # NumPy view, C-contiguous
                        weights_array,              # (N,)
                        algo_name=algo,
                        kappa=float(self.main.kappa),
                        iterations=int(self.main.iterations),
                        sigma_low=float(self.main.sigma_low),
                        sigma_high=float(self.main.sigma_high),
                        trim_fraction=float(self.main.trim_fraction),
                        esd_threshold=float(self.main.esd_threshold),
                        biweight_constant=float(self.main.biweight_constant),
                        modz_threshold=float(self.main.modz_threshold),
                        comet_hclip_k=float(self.main.settings.value("stacking/comet_hclip_k", 1.30, type=float)),
                        comet_hclip_p=float(self.main.settings.value("stacking/comet_hclip_p", 25.0, type=float)),
                    )
                    # _torch_reduce_tile should already return NumPy; if it returns tensors, convert here.
                    if hasattr(tile_result, "detach"):
                        tile_result = tile_result.detach().cpu().numpy()
                    if hasattr(tile_rej_map, "detach"):
                        tile_rej_map = tile_rej_map.detach().cpu().numpy()
                else:
                    # CPU path (NumPy/Numba)
                    if algo in ("Comet Median", "Simple Median (No Rejection)"):
                        tile_result  = np.median(ts, axis=0)
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Comet High-Clip Percentile":
                        k = self.main.settings.value("stacking/comet_hclip_k", 1.30, type=float)
                        p = self.main.settings.value("stacking/comet_hclip_p", 25.0, type=float)
                        tile_result  = _high_clip_percentile(ts, k=float(k), p=float(p))
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Comet Lower-Trim (30%)":
                        tile_result  = _lower_trimmed_mean(ts, trim_hi_frac=0.30)
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Comet Percentile (40th)":
                        tile_result  = _percentile40(ts)
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Simple Average (No Rejection)":
                        tile_result  = np.average(ts, axis=0, weights=weights_array)
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Weighted Windsorized Sigma Clipping":
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                            ts, weights_array, lower=self.main.sigma_low, upper=self.main.sigma_high
                        )
                    elif algo == "Kappa-Sigma Clipping":
                        tile_result, tile_rej_map = kappa_sigma_clip_weighted(
                            ts, weights_array, kappa=self.main.kappa, iterations=self.main.iterations
                        )
                    elif algo == "Trimmed Mean":
                        tile_result, tile_rej_map = trimmed_mean_weighted(
                            ts, weights_array, trim_fraction=self.main.trim_fraction
                        )
                    elif algo == "Extreme Studentized Deviate (ESD)":
                        tile_result, tile_rej_map = esd_clip_weighted(
                            ts, weights_array, threshold=self.main.esd_threshold
                        )
                    elif algo == "Biweight Estimator":
                        tile_result, tile_rej_map = biweight_location_weighted(
                            ts, weights_array, tuning_constant=self.main.biweight_constant
                        )
                    elif algo == "Modified Z-Score Clipping":
                        tile_result, tile_rej_map = modified_zscore_clip_weighted(
                            ts, weights_array, threshold=self.main.modz_threshold
                        )
                    elif algo == "Max Value":
                        tile_result, tile_rej_map = max_value_stack(ts, weights_array)
                    else:
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                            ts, weights_array, lower=self.main.sigma_low, upper=self.main.sigma_high
                        )

                # write back
                integrated_image[y0:y1, x0:x1, :] = tile_result

                # --- rejection bookkeeping ---
                trm = tile_rej_map
                if trm.ndim == 4:
                    trm = np.any(trm, axis=-1)  # collapse color → (N, th, tw)

                rej_any[y0:y1, x0:x1]  |= np.any(trm, axis=0)
                rej_count[y0:y1, x0:x1] += trm.sum(axis=0).astype(np.uint16)

                # per-file coords (existing behavior)
                for i, fpath in enumerate(file_list):
                    ys, xs = np.where(trm[i])
                    if ys.size:
                        per_file_rejections[fpath].extend(zip(x0 + xs, y0 + ys))

                # perf log
                dt = time.perf_counter() - t0
                # simple “work” metric: pixels processed (× frames × channels)
                work_px = th * tw * N * channels
                mpx_s = (work_px / 1e6) / dt if dt > 0 else float("inf")
                log(f"  ↳ tile {tile_idx} done in {dt:.3f}s  (~{mpx_s:.1f} MPx/s)")

                # flip buffer
                use_buf0 = not use_buf0

        # close mmapped FITSes and prefetch pool
        tp.shutdown(wait=True)
        for s in sources:
            s.close()

        if channels == 1:
            integrated_image = integrated_image[..., 0]

        # stash group-level maps
        if not hasattr(self, "_rej_maps"):
            self.main._rej_maps = {}
        rej_frac = (rej_count.astype(np.float32) / float(max(1, N)))  # [0..1]
        self.main._rej_maps[group_key] = {"any": rej_any, "frac": rej_frac, "count": rej_count, "n": N}

        log(f"Integration complete for group '{group_key}'.")
        
        # If we used memmap, convert to regular array and cleanup
        if integrated_memmap_path is not None:
            integrated_image = np.array(integrated_image)  # Copy to regular array
            try:
                cleanup_memmap(None, integrated_memmap_path)
            except Exception:
                pass
        
        try:
            _free_torch_memory()
        except Exception:
            pass  # Ignore torch cleanup errors
        return integrated_image, per_file_rejections, ref_header



    def integrate_registered_images(self):
        """
        Integrate frames that are already aligned (and typically normalized).
        We only do fast measurements for weights; no re-normalization, no re-alignment.
        """
        if getattr(self, "_registration_busy", False):
            self.main.update_status("⏸ Another job is running; ignoring extra click.")
            return
        self.main._set_registration_busy(True)

        try:
            self.main.update_status("🔄 Integrating Previously Registered Images…")

            # 1) Pull files from the tree
            self.main.extract_light_files_from_tree()
            if not self.main.light_files:
                self.main.update_status("⚠️ No registered images found!")
                self.main._set_registration_busy(False)
                return

            # Flatten
            all_files = [p for lst in self.main.light_files.values() for p in lst]
            if not all_files:
                self.main.update_status("⚠️ No frames found in the registration tree!")
                self.main._set_registration_busy(False)
                return

            # Prefer already-normalized/registered frames
            def _looks_norm_reg(p: str) -> bool:
                bn = os.path.basename(p).lower()
                if (
                    bn.endswith("_n.fit") or bn.endswith("_n.fits")
                    or bn.endswith("_n_r.fit") or bn.endswith("_n_r.fits")
                    or ("aligned_images" in os.path.normpath(p).lower())
                    or bn.endswith(".xisf")  # ← treat XISF as already prepared by PI
                ):
                    return True
                # header fallback
                try:
                    hdr = _get_header_fast(p)  # now supports XISF
                    if hdr and (hdr.get("DEBAYERED") is not None or hdr.get("SAS_RSMP") or hdr.get("SAS_NORM") or hdr.get("NORMALIZ")):
                        return True
                except Exception:
                    pass
                return False

            cand = [p for p in all_files if _looks_norm_reg(p)]
            if not cand:
                # fall back to everything, but we still won't normalize here
                cand = all_files[:]

            self.main.update_status(f"📊 Found {len(cand)} aligned/normalized frames. Measuring in parallel previews…")

            # 2) Chunked preview measurement (mean + star count/ecc)
            self.main.frame_weights = {}
            mean_values = {}
            star_counts = {}
            measured_frames = []

            max_workers = os.cpu_count() or 4
            chunk_size = max_workers

            def chunk_list(lst, size):
                for i in range(0, len(lst), size):
                    yield lst[i:i+size]

            chunks = list(chunk_list(cand, chunk_size))
            total_chunks = len(chunks)

            # For already registered images, we don’t need to rescale to a target bin.
            # We can just make small previews directly from the FITS (debayer-aware superpixel).
            from concurrent.futures import ThreadPoolExecutor, as_completed

            for idx, chunk in enumerate(chunks, 1):
                self.main.update_status(f"📦 Measuring chunk {idx}/{total_chunks} ({len(chunk)} frames)")
                QApplication.processEvents()

                # Load tiny previews in parallel
                previews = []
                paths_ok = []

                def _preview_job(fp: str):
                    return _quick_preview_from_path(fp, target_xbin=1, target_ybin=1)

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(_preview_job, fp): fp for fp in chunk}
                    for fut in as_completed(futs):
                        fp = futs[fut]
                        try:
                            p = fut.result()
                            if p is None:
                                continue
                            previews.append(p)
                            paths_ok.append(fp)
                        except Exception as e:
                            self.main.update_status(f"⚠️ Preview error for {fp}: {e}")
                        QApplication.processEvents()

                if not previews:
                    self.main.update_status("⚠️ No valid previews in this chunk.")
                    continue

                # Crop all previews in this chunk to a common min size (cheap)
                min_h = min(im.shape[0] for im in previews)
                min_w = min(im.shape[1] for im in previews)
                if any((im.shape[0] != min_h or im.shape[1] != min_w) for im in previews):
                    previews = [_center_crop_2d(im, min_h, min_w) for im in previews]

                # Means (vectorized)
                means = np.array([float(np.mean(ci)) for ci in previews], dtype=np.float32)

                # Star count + ecc on small, further-downsampled previews
                def _star_job(i_fp):
                    i, fp = i_fp
                    p = previews[i]
                    # normalize preview to [0..] by subtracting local min for robustness
                    pmin = float(np.nanmin(p))
                    # fast count on tiny image
                    c, ecc = compute_star_count_fast_preview(p - pmin)
                    med = float(np.median(p - pmin))
                    return fp, float(means[i]), med, c, ecc

                star_workers = min(max_workers, 8)
                with ThreadPoolExecutor(max_workers=star_workers) as ex:
                    for fp, mean_v, med, c, ecc in ex.map(_star_job, enumerate(paths_ok)):
                        mean_values[fp] = mean_v
                        star_counts[fp] = {"count": int(c), "eccentricity": float(ecc)}
                        measured_frames.append(fp)

                del previews

            if not measured_frames:
                self.main.update_status("⚠️ No frames could be measured!")
                return

            self.main.update_status(f"✅ All chunks complete! Measured {len(measured_frames)} frames total.")
            QApplication.processEvents()

            # 3) Weights — keep your current logic (fast & good)
            self.main.update_status("⚖️ Computing frame weights…")
            dbg = ["\n📊 **Frame Weights Debug Log:**"]
            max_w = 0.0
            for fp in measured_frames:
                c   = star_counts[fp]["count"]
                ecc = star_counts[fp]["eccentricity"]
                m   = mean_values[fp]
                # same weighting you had during registration measurement
                c = max(c, 1)
                m = max(m, 1e-6)
                raw_w = (c * min(1.0, max(1.0 - ecc, 0.0))) / m
                self.main.frame_weights[fp] = raw_w
                max_w = max(max_w, raw_w)
                dbg.append(f"📂 {os.path.basename(fp)} → StarCount={c}, Ecc={ecc:.4f}, Mean={m:.4f}, Weight={raw_w:.4f}")

            if max_w > 0:
                for k in self.main.frame_weights:
                    self.main.frame_weights[k] /= max_w

            self.main.update_status("\n".join(dbg))
            self.main.update_status("✅ Frame weights computed!")
            QApplication.processEvents()

            # 4) Choose reference (optional for visual/log purposes)
            if getattr(self, "reference_frame", None):
                self.main.update_status(f"📌 Using user-specified reference: {self.main.reference_frame}")
            else:
                self.main.reference_frame = max(self.main.frame_weights, key=self.main.frame_weights.get)
                self.main.update_status(f"📌 Auto-selected reference: {self.main.reference_frame}")

            # 5) Clear transforms; not needed for already aligned frames
            self.main.valid_transforms = {}

            # 6) Hand off to your unified pipeline (this will stack without re-alignment)
            aligned_light_files = {g: lst for g, lst in self.main.light_files.items() if lst}
            self.main._run_mfdeconv_then_continue(aligned_light_files)
            return

        except Exception:
            self.main._set_registration_busy(False)
            raise

