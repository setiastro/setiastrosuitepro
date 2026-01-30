# pro/gui/mixins/toolbar_mixin.py
"""
Toolbar and action management mixin for AstroSuiteProMainWindow.
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QDesktopServices
from PyQt6.QtWidgets import QMenu, QToolButton

from PyQt6.QtCore import QElapsedTimer

import sys
import os

if TYPE_CHECKING:
    pass

# Import icon paths - these are needed at runtime
from setiastro.saspro.resources import (
    icon_path, green_path, neutral_path, whitebalance_path, texture_clarity_path,
    morpho_path, clahe_path, starnet_path, staradd_path, LExtract_path,
    LInsert_path, rgbcombo_path, rgbextract_path, graxperticon_path,
    cropicon_path, openfile_path, abeicon_path, undoicon_path, redoicon_path,
    blastericon_path, hdr_path, invert_path, fliphorizontal_path,
    flipvertical_path, rotateclockwise_path, rotatecounterclockwise_path,rotatearbitrary_path,
    rotate180_path, maskcreate_path, maskapply_path, maskremove_path,
    pixelmath_path, histogram_path, mosaic_path, rescale_path, staralign_path,
    platesolve_path, psf_path, supernova_path, starregistration_path,
    stacking_path, pedestal_icon_path, starspike_path, astrospike_path,
    signature_icon_path, livestacking_path, convoicon_path, spcc_icon_path,
    exoicon_path, peeker_icon, dse_icon_path, isophote_path, statstretch_path,
    starstretch_path, curves_path, disk_path, uhs_path, blink_path, ppp_path, narrowbandnormalization_path,
    nbtorgb_path, freqsep_path, multiscale_decomp_path, contsub_path, halo_path, cosmic_path,
    satellite_path, imagecombine_path, wims_path, wimi_path, linearfit_path,
    debayer_path, aberration_path, functionbundles_path, viewbundles_path, planetarystacker_path,
    selectivecolor_path, rgbalign_path, planetprojection_path, clonestampicon_path, finderchart_path,magnitude_path,
)

# Import shortcuts module
from setiastro.saspro.shortcuts import DraggableToolBar, ShortcutManager


class ToolbarMixin:
    """
    Mixin for toolbar and action management.
    
    Provides methods for creating and managing toolbars and actions.
    """
    
    # Placeholder methods for tool openers (implemented in main window)
    
    
    def _sync_link_action_state(self):
        """Synchronize the link views action state."""
        if not hasattr(self, "_link_views_enabled"):
            return
        
        if hasattr(self, "action_link_views"):
            self.action_link_views.setChecked(self._link_views_enabled)
    
    def _find_action_by_cid(self, command_id: str) -> QAction | None:
        """
        Find an action by its command ID.
        
        Args:
            command_id: The command identifier string
            
        Returns:
            The QAction if found, None otherwise
        """
        for action in self.findChildren(QAction):
            if getattr(action, "command_id", None) == command_id:
                return action
        return None

    def _init_toolbar(self):
        # View toolbar (Undo / Redo / Display-Stretch)
        tb = DraggableToolBar(self.tr("View"), self)
        tb.setObjectName("View")
        tb.setSettingsKey("Toolbar/View")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        tb.addAction(self.act_open)
        tb.addAction(self.act_save)
        tb.addSeparator()
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)
        tb.addSeparator()

        # Put Display-Stretch on the bar first so we can attach a menu to its button
        tb.addAction(self.act_autostretch)
        tb.addAction(self.act_zoom_out)
        tb.addAction(self.act_zoom_in)
        tb.addAction(self.act_zoom_1_1)
        tb.addAction(self.act_zoom_fit)

        # Style the autostretch button + add menu
        btn = tb.widgetForAction(self.act_autostretch)
        if isinstance(btn, QToolButton):
            menu = QMenu(btn)
            menu.addAction(self.act_stretch_linked)
            menu.addAction(self.act_hardstretch)

            # NEW: advanced controls + presets
            menu.addSeparator()
            menu.addAction(self.act_display_target)
            menu.addAction(self.act_display_sigma)

            presets = QMenu("Presets", menu)
            a_norm = presets.addAction("Normal (target 0.30, Ïƒ 5)")
            a_midy = presets.addAction("Mid (target 0.40, Ïƒ 3)")
            a_hard = presets.addAction("Hard (target 0.50, Ïƒ 2)")
            menu.addMenu(presets)
            menu.addSeparator()
            menu.addAction(self.act_bake_display_stretch)

            # push numbers to the active view and (optionally) turn on autostretch
            def _apply_preset(t, s, also_enable=True):
                self.settings.setValue("display/target", float(t))
                self.settings.setValue("display/sigma", float(s))
                sw = self.mdi.activeSubWindow()
                if not sw:
                    return
                view = sw.widget()
                if hasattr(view, "set_autostretch_target"):
                    view.set_autostretch_target(float(t))
                if hasattr(view, "set_autostretch_sigma"):
                    view.set_autostretch_sigma(float(s))
                if also_enable and not getattr(view, "autostretch_enabled", False):
                    if hasattr(view, "set_autostretch"):
                        view.set_autostretch(True)
                    self._sync_autostretch_action(True)

            a_norm.triggered.connect(lambda: _apply_preset(0.30, 5.0))
            a_midy.triggered.connect(lambda: _apply_preset(0.40, 3.0))
            a_hard.triggered.connect(lambda: _apply_preset(0.50, 2.0))

            btn.setMenu(menu)
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

            btn.setStyleSheet("""
                QToolButton { color: #dcdcdc; }
                QToolButton:checked { color: #DAA520; font-weight: 600; }
            """)

        btn_fit = tb.widgetForAction(self.act_zoom_fit)
        if isinstance(btn_fit, QToolButton):
            fit_menu = QMenu(btn_fit)

            # Use the existing action created in _create_actions()
            fit_menu.addAction(self.act_auto_fit_resize)

            # (Optional) make sure it reflects current flag at startup
            self.act_auto_fit_resize.blockSignals(True)
            try:
                self.act_auto_fit_resize.setChecked(bool(self._auto_fit_on_resize))
            finally:
                self.act_auto_fit_resize.blockSignals(False)

            btn_fit.setMenu(fit_menu)
            btn_fit.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

            btn_fit.setStyleSheet("""
                QToolButton { color: #dcdcdc; }
                QToolButton:checked { color: #DAA520; font-weight: 600; }
            """)


        # Make sure the visual state matches the flag at startup
        self._restore_toolbar_order(tb, "Toolbar/View")
        self._restore_toolbar_memberships()   # (or keep your existing placement, but the bind must be AFTER it)
        self._bind_view_toolbar_menus(tb)
        self._sync_fit_auto_visual()
        # Apply hidden state immediately after order restore (prevents flash)
        try:
            tb.apply_hidden_state()
        except Exception:
            pass

        # Functions toolbar
        tb_fn = DraggableToolBar(self.tr("Functions"), self)
        tb_fn.setObjectName("Functions")
        tb_fn.setSettingsKey("Toolbar/Functions")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_fn)

        tb_fn.addAction(self.act_crop)
        tb_fn.addAction(self.act_histogram)
        tb_fn.addAction(self.act_pedestal)
        tb_fn.addAction(self.act_linear_fit)
        tb_fn.addAction(self.act_stat_stretch)
        tb_fn.addAction(self.act_star_stretch)
        tb_fn.addAction(self.act_curves)
        tb_fn.addAction(self.act_ghs)
        tb_fn.addAction(self.act_abe)
        tb_fn.addAction(self.act_graxpert)
        tb_fn.addAction(self.act_remove_stars)
        tb_fn.addAction(self.act_add_stars)
        tb_fn.addAction(self.act_background_neutral)
        tb_fn.addAction(self.act_white_balance)
        tb_fn.addAction(self.act_sfcc)
        tb_fn.addAction(self.act_remove_green)
        tb_fn.addAction(self.act_convo)
        tb_fn.addAction(self.act_extract_luma)
        tb_fn.addAction(self.act_recombine_luma)
        tb_fn.addAction(self.act_rgb_extract)
        tb_fn.addAction(self.act_rgb_combine)
        tb_fn.addAction(self.act_blemish)
        tb_fn.addAction(self.act_clone_stamp) 
        tb_fn.addAction(self.act_wavescale_hdr)
        tb_fn.addAction(self.act_wavescale_de)
        tb_fn.addAction(self.act_clahe)
        tb_fn.addAction(self.act_texture_clarity)
        tb_fn.addAction(self.act_morphology)
        tb_fn.addAction(self.act_pixelmath)
        tb_fn.addAction(self.act_signature)
        tb_fn.addAction(self.act_halobgon)

        self._restore_toolbar_order(tb_fn, "Toolbar/Functions")
        try:
            tb_fn.apply_hidden_state()
        except Exception:
            pass

        tbCosmic = DraggableToolBar(self.tr("Cosmic Clarity"), self)
        tbCosmic.setObjectName("Cosmic Clarity")
        tbCosmic.setSettingsKey("Toolbar/Cosmic")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tbCosmic)

        tbCosmic.addAction(self.actAberrationAI)
        tbCosmic.addAction(self.actCosmicUI)
        tbCosmic.addAction(self.actCosmicSat)

        self._restore_toolbar_order(tbCosmic, "Toolbar/Cosmic")
        try:
            tbCosmic.apply_hidden_state()
        except Exception:
            pass

        tb_tl = DraggableToolBar(self.tr("Tools"), self)
        tb_tl.setObjectName("Tools")
        tb_tl.setSettingsKey("Toolbar/Tools")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_tl)

        tb_tl.addAction(self.act_blink)  # Tools start here; Blink shows with QIcon(blink_path)
        tb_tl.addAction(self.act_ppp)    # Perfect Palette Picker
        tb_tl.addAction(self.act_nbtorgb)
        tb_tl.addAction(self.act_narrowband_normalization)
        
        tb_tl.addAction(self.act_selective_color)
        tb_tl.addAction(self.act_freqsep)
        tb_tl.addAction(self.act_multiscale_decomp)
        tb_tl.addAction(self.act_contsub)
        tb_tl.addAction(self.act_image_combine)
        tb_tl.addAction(self.act_magnitude)

        self._restore_toolbar_order(tb_tl, "Toolbar/Tools")
        try:
            tb_tl.apply_hidden_state()
        except Exception:
            pass

        tb_geom = DraggableToolBar(self.tr("Geometry"), self)
        tb_geom.setObjectName("Geometry")
        tb_geom.setSettingsKey("Toolbar/Geometry")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_geom)

        tb_geom.addAction(self.act_geom_invert)
        tb_geom.addSeparator()
        tb_geom.addAction(self.act_geom_flip_h)
        tb_geom.addAction(self.act_geom_flip_v)
        tb_geom.addSeparator()
        tb_geom.addAction(self.act_geom_rot_cw)
        tb_geom.addAction(self.act_geom_rot_ccw)
        tb_geom.addAction(self.act_geom_rot_180)
        tb_geom.addAction(self.act_geom_rot_any)
        tb_geom.addSeparator()
        tb_geom.addAction(self.act_geom_rescale)
        tb_geom.addSeparator()
        tb_geom.addAction(self.act_debayer)

        self._restore_toolbar_order(tb_geom, "Toolbar/Geometry")
        try:
            tb_geom.apply_hidden_state()
        except Exception:
            pass

        tb_star = DraggableToolBar(self.tr("Star Stuff"), self)
        tb_star.setObjectName("Star Stuff")
        tb_star.setSettingsKey("Toolbar/StarStuff")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_star)

        tb_star.addAction(self.act_image_peeker)
        tb_star.addAction(self.act_psf_viewer)
        tb_star.addAction(self.act_stacking_suite)
        tb_star.addAction(self.act_live_stacking)
        tb_star.addAction(self.act_planetary_stacker)
        tb_star.addAction(self.act_planet_projection)
        tb_star.addAction(self.act_plate_solve)
        
        tb_star.addAction(self.act_star_align)
        tb_star.addAction(self.act_star_register)
        tb_star.addAction(self.act_rgb_align)
        tb_star.addAction(self.act_mosaic_master)
        tb_star.addAction(self.act_supernova_hunter)
        tb_star.addAction(self.act_star_spikes)
        tb_star.addAction(self.act_astrospike)
        tb_star.addAction(self.act_exo_detector)
        tb_star.addAction(self.act_isophote)

        self._restore_toolbar_order(tb_star, "Toolbar/StarStuff")
        try:
            tb_star.apply_hidden_state()
        except Exception:
            pass

        tb_msk = DraggableToolBar(self.tr("Masks"), self)
        tb_msk.setObjectName("Masks")
        tb_msk.setSettingsKey("Toolbar/Masks")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_msk)

        tb_msk.addAction(self.act_create_mask)
        tb_msk.addAction(self.act_apply_mask)
        tb_msk.addAction(self.act_remove_mask)

        self._restore_toolbar_order(tb_msk, "Toolbar/Masks")
        try:
            tb_msk.apply_hidden_state()
        except Exception:
            pass

        tb_wim = DraggableToolBar(self.tr("What's In My..."), self)
        tb_wim.setObjectName("What's In My...")
        tb_wim.setSettingsKey("Toolbar/WhatsInMy")
        tb_wim.addAction(self.act_finder_chart)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_wim)

        tb_wim.addAction(self.act_whats_in_my_sky)
        tb_wim.addAction(self.act_wimi)

        self._restore_toolbar_order(tb_wim, "Toolbar/WhatsInMy")
        try:
            tb_wim.apply_hidden_state()
        except Exception:
            pass

        tb_bundle = DraggableToolBar(self.tr("Bundles"), self)
        tb_bundle.setObjectName("Bundles")
        tb_bundle.setSettingsKey("Toolbar/Bundles")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_bundle)

        tb_bundle.addAction(self.act_view_bundles)
        tb_bundle.addAction(self.act_function_bundles)

        self._restore_toolbar_order(tb_bundle, "Toolbar/Bundles")
        try:
            tb_bundle.apply_hidden_state()
        except Exception:
            pass


        tb_hidden = DraggableToolBar(self.tr("Hidden"), self)
        tb_hidden.setObjectName("Hidden")
        tb_hidden.setSettingsKey("Toolbar/Hidden")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_hidden)
        #tb_hidden.addAction(self.act_narrowband_normalization)
        tb_hidden.setVisible(False)  # <- always hidden

        # This can move actions between toolbars, so do it after each toolbar has its base order restored.
        self._restore_toolbar_memberships()

        # Migrate legacy per-toolbar Hidden lists into the new Hidden toolbar
        self._migrate_old_hidden_sets_to_hidden_toolbar()

        # (Optional) Re-apply per-toolbar order after migration (since we removed actions)
        for _tb in self.findChildren(DraggableToolBar):
            key = getattr(_tb, "_settings_key", None)
            if key:
                self._restore_toolbar_order(_tb, str(key))

        # Re-apply hidden state AFTER memberships (actions may have moved toolbars).
        # This also guarantees correctness even if any toolbar was rebuilt/adjusted internally.
        for _tb in self.findChildren(DraggableToolBar):
            try:
                _tb.apply_hidden_state()
            except Exception:
                pass

        # Rebind ALL dropdowns after reorder/membership moves
        self._rebind_view_dropdowns()
        self._rebind_extract_luma_dropdown()

    def _toolbar_containing_action(self, action: QAction):
        from setiastro.saspro.shortcuts import DraggableToolBar
        for tb in self.findChildren(DraggableToolBar):
            if action in tb.actions():
                return tb
        return None

    def _rebind_extract_luma_dropdown(self):
        from PyQt6.QtWidgets import QMenu, QToolButton
        from setiastro.saspro.luminancerecombine import LUMA_PROFILES

        tb = self._toolbar_containing_action(self.act_extract_luma)
        if not tb:
            return

        btn = tb.widgetForAction(self.act_extract_luma)
        if not isinstance(btn, QToolButton):
            return

        menu = QMenu(btn)

        # Ensure cache exists (created in _create_actions(), but be defensive)
        if not hasattr(self, "_luma_sensor_actions"):
            self._luma_sensor_actions = {}  # key -> QAction

        cur_method = str(getattr(self, "luma_method", "rec709"))

        # ============================================================
        # PATCH SITE #1 (Standard menu)  <-- THIS IS WHERE "C" GOES
        # Find your old block:
        #
        #   # ---- Standard (use your QActionGroup, keep it simple) ----
        #   if getattr(self, "_luma_group", None) is not None:
        #       std_menu = QMenu(self.tr("Standard"), menu)
        #       std_menu.addActions(self._luma_group.actions())
        #       menu.addMenu(std_menu)
        #
        # Replace it with the block below.
        # ============================================================
        if getattr(self, "_luma_group", None) is not None:
            # Sync standard checked states from self.luma_method
            for a in self._luma_group.actions():
                data = str(a.data() or "")
                if data.startswith("sensor:"):
                    continue  # standards only in Standard menu
                a.blockSignals(True)
                try:
                    a.setChecked(data == cur_method)
                finally:
                    a.blockSignals(False)

            # Add ONLY standard actions to the Standard menu (exclude sensors)
            std_menu = QMenu(self.tr("Standard"), menu)
            for a in self._luma_group.actions():
                data = str(a.data() or "")
                if data.startswith("sensor:"):
                    continue
                std_menu.addAction(a)
            menu.addMenu(std_menu)

        # ---- Sensors (nested by category path) ----
        sensors_root = QMenu(self.tr("Sensors"), menu)

        # Build tree of submenus keyed by "Sensors/<path>"
        submenu_cache: dict[str, QMenu] = {}

        def get_or_make_path(root_menu: QMenu, path: str) -> QMenu:
            parts = [p for p in path.split("/") if p.strip()]
            cur = root_menu
            acc = ""
            for part in parts:
                acc = f"{acc}/{part}" if acc else part
                if acc not in submenu_cache:
                    sm = QMenu(self.tr(part), cur)
                    cur.addMenu(sm)
                    submenu_cache[acc] = sm
                cur = submenu_cache[acc]
            return cur

        any_sensor = False
        for key, prof in LUMA_PROFILES.items():
            if not str(key).startswith("sensor:"):
                continue

            cat = str(prof.get("category", "Sensors/Other"))
            group_path = cat.split("Sensors/", 1)[1] if "Sensors/" in cat else cat
            parent_menu = get_or_make_path(sensors_root, group_path)

            display_name = key.split("sensor:", 1)[1].strip()
            desc = str(prof.get("description", display_name))
            info = str(prof.get("info", "")).strip()

            # ============================================================
            # PATCH SITE #2 (Sensors become mutually exclusive with standards)
            # - Cache the QAction so we don't pile up connections/actions
            # - Add it to the SAME QActionGroup (exclusive) as standards
            # - Do NOT use _pick_sensor; rely on QActionGroup.triggered
            # ============================================================
            act = self._luma_sensor_actions.get(key)
            if act is None:
                act = parent_menu.addAction(self.tr(display_name))
                act.setCheckable(True)
                act.setData(key)  # IMPORTANT: enables group.triggered to set luma_method

                # Put sensors into the SAME exclusive group so selecting one
                # deselects everything else (standards and other sensors).
                if getattr(self, "_luma_group", None) is not None:
                    self._luma_group.addAction(act)

                self._luma_sensor_actions[key] = act
            else:
                # Reuse action, but re-add it to the correct submenu location
                parent_menu.addAction(act)
                act.setText(self.tr(display_name))

            # Update UI info each bind (safe if translations change)
            if info:
                act.setStatusTip(info)
                act.setToolTip(f"{desc}\n{info}")
            else:
                act.setToolTip(desc)

            # Checked state reflects current selection
            act.blockSignals(True)
            try:
                act.setChecked(cur_method == str(key))
            finally:
                act.blockSignals(False)

            any_sensor = True

        if any_sensor:
            menu.addMenu(sensors_root)

        btn.setMenu(menu)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        btn.setStyleSheet("""
            QToolButton { color: #dcdcdc; }
            QToolButton:pressed, QToolButton:checked { color: #DAA520; font-weight: 600; }
        """)


    def _rebind_view_dropdowns(self):
        """
        Rebind dropdown menus for Display-Stretch + Fit buttons
        on whatever toolbar those actions currently live in.
        Call this AFTER all restore/reorder/membership moves.
        """
        # ---- Display-Stretch dropdown ----
        tb = self._toolbar_containing_action(self.act_autostretch)
        if tb:
            btn = tb.widgetForAction(self.act_autostretch)
            if isinstance(btn, QToolButton):
                menu = QMenu(btn)
                menu.addAction(self.act_stretch_linked)
                menu.addAction(self.act_hardstretch)

                menu.addSeparator()
                menu.addAction(self.act_display_target)
                menu.addAction(self.act_display_sigma)

                presets = QMenu(self.tr("Presets"), menu)
                a_norm = presets.addAction(self.tr("Normal (target 0.30, σ 5)"))
                a_midy = presets.addAction(self.tr("Mid (target 0.40, σ 3)"))
                a_hard = presets.addAction(self.tr("Hard (target 0.50, σ 2)"))
                menu.addMenu(presets)

                menu.addSeparator()
                menu.addAction(self.act_bake_display_stretch)

                def _apply_preset(t, s, also_enable=True):
                    self.settings.setValue("display/target", float(t))
                    self.settings.setValue("display/sigma", float(s))
                    sw = self.mdi.activeSubWindow()
                    if not sw:
                        return
                    view = sw.widget()
                    if hasattr(view, "set_autostretch_target"):
                        view.set_autostretch_target(float(t))
                    if hasattr(view, "set_autostretch_sigma"):
                        view.set_autostretch_sigma(float(s))
                    if also_enable and not getattr(view, "autostretch_enabled", False):
                        if hasattr(view, "set_autostretch"):
                            view.set_autostretch(True)
                        self._sync_autostretch_action(True)

                a_norm.triggered.connect(lambda: _apply_preset(0.30, 5.0))
                a_midy.triggered.connect(lambda: _apply_preset(0.40, 3.0))
                a_hard.triggered.connect(lambda: _apply_preset(0.50, 2.0))

                btn.setMenu(menu)
                btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

        # ---- Fit dropdown ----
        tb_fit = self._toolbar_containing_action(self.act_zoom_fit)
        if tb_fit:
            btn_fit = tb_fit.widgetForAction(self.act_zoom_fit)
            if isinstance(btn_fit, QToolButton):
                fit_menu = QMenu(btn_fit)
                fit_menu.addAction(self.act_auto_fit_resize)  # use the real action
                btn_fit.setMenu(fit_menu)
                btn_fit.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        tb = self._toolbar_containing_action(self.act_autostretch)
        if tb:
            btn = tb.widgetForAction(self.act_autostretch)
            if isinstance(btn, QToolButton):
                # ... build menu ...
                btn.setMenu(menu)
                btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

                # IMPORTANT: re-apply style after action moves / rebind
                self._style_toggle_toolbutton(btn)

    def _bind_view_toolbar_menus(self, tb: DraggableToolBar):
        # --- Display-Stretch menu ---
        btn = tb.widgetForAction(self.act_autostretch)
        if isinstance(btn, QToolButton):
            menu = QMenu(btn)
            menu.addAction(self.act_stretch_linked)
            menu.addAction(self.act_hardstretch)

            menu.addSeparator()
            menu.addAction(self.act_display_target)
            menu.addAction(self.act_display_sigma)

            presets = QMenu(self.tr("Presets"), menu)
            a_norm = presets.addAction(self.tr("Normal (target 0.30, σ 5)"))
            a_midy = presets.addAction(self.tr("Mid (target 0.40, σ 3)"))
            a_hard = presets.addAction(self.tr("Hard (target 0.50, σ 2)"))
            menu.addMenu(presets)
            menu.addSeparator()
            menu.addAction(self.act_bake_display_stretch)

            def _apply_preset(t, s, also_enable=True):
                self.settings.setValue("display/target", float(t))
                self.settings.setValue("display/sigma", float(s))
                sw = self.mdi.activeSubWindow()
                if not sw:
                    return
                view = sw.widget()
                if hasattr(view, "set_autostretch_target"):
                    view.set_autostretch_target(float(t))
                if hasattr(view, "set_autostretch_sigma"):
                    view.set_autostretch_sigma(float(s))
                if also_enable and not getattr(view, "autostretch_enabled", False):
                    if hasattr(view, "set_autostretch"):
                        view.set_autostretch(True)
                    self._sync_autostretch_action(True)

            a_norm.triggered.connect(lambda: _apply_preset(0.30, 5.0))
            a_midy.triggered.connect(lambda: _apply_preset(0.40, 3.0))
            a_hard.triggered.connect(lambda: _apply_preset(0.50, 2.0))

            btn.setMenu(menu)
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

        # --- Fit menu (Auto-fit checkbox) ---
        btn_fit = tb.widgetForAction(self.act_zoom_fit)
        if isinstance(btn_fit, QToolButton):
            fit_menu = QMenu(btn_fit)

            # IMPORTANT: use your existing action (don’t create a new one)
            fit_menu.addAction(self.act_auto_fit_resize)

            btn_fit.setMenu(fit_menu)
            btn_fit.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

    def _linux_force_text_action(self, act: QAction, text: str) -> None:
        """On Linux, show text-only for this action (no theme icon)."""
        if not sys.platform.startswith("linux"):
            return
        act.setIcon(QIcon())   # remove whatever theme icon got assigned
        act.setText(text)      # show the glyph/text

    def _create_actions(self):
        # File actions
        self.act_open = QAction(QIcon(openfile_path), self.tr("Open..."), self)
        self.act_open.setIconVisibleInMenu(True)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.setStatusTip(self.tr("Open image(s)"))
        self.act_open.triggered.connect(self.open_files)


        self.act_project_new  = QAction(self.tr("New Project"), self)
        self.act_project_save = QAction(self.tr("Save Project..."), self)
        self.act_project_load = QAction(self.tr("Load Project..."), self)

        self.act_project_new.setStatusTip(self.tr("Close all views and clear shortcuts"))
        self.act_project_save.setStatusTip(self.tr("Save all views, histories, and shortcuts to a .sas file"))
        self.act_project_load.setStatusTip(self.tr("Load a .sas project (views, histories, shortcuts)"))

        self.act_project_new.triggered.connect(self._new_project)
        self.act_project_save.triggered.connect(self._save_project)
        self.act_project_load.triggered.connect(self._load_project)

        self.act_clear_views = QAction(self.tr("Clear All Views"), self)
        self.act_clear_views.setStatusTip(self.tr("Close all views and documents, keep desktop shortcuts"))
        # optional shortcut (pick anything you like or omit)
        # self.act_clear_views.setShortcut(QKeySequence("Ctrl+Shift+W"))
        self.act_clear_views.triggered.connect(self._clear_views_keep_shortcuts)

        self.act_save = QAction(QIcon(disk_path), self.tr("Save As..."), self)
        self.act_save.setIconVisibleInMenu(True)
        self.act_save.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.act_save.setStatusTip(self.tr("Save the active image"))
        self.act_save.triggered.connect(self.save_active)

        self.act_exit = QAction(self.tr("&Exit"), self)
        self.act_exit.setShortcut(QKeySequence.StandardKey.Quit)  # Cmd+Q / Ctrl+Q
        # Make it appear under the app menu on macOS automatically:
        self.act_exit.setMenuRole(QAction.MenuRole.QuitRole)
        self.act_exit.triggered.connect(self._on_exit)

        self.act_cascade = QAction(self.tr("Cascade Views"), self)
        self.act_cascade.setStatusTip(self.tr("Cascade all subwindows"))
        self.act_cascade.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.act_cascade.triggered.connect(self._cascade_views)

        self.act_tile = QAction(self.tr("Tile Views"), self)
        self.act_tile.setStatusTip(self.tr("Tile all subwindows"))
        self.act_tile.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self.act_tile.triggered.connect(self._tile_views)

        self.act_tile_vert = QAction(self.tr("Tile Vertically"), self)
        self.act_tile_vert.setStatusTip(self.tr("Split the workspace into equal vertical columns"))
        self.act_tile_vert.triggered.connect(lambda: self._tile_views_direction("v"))

        self.act_tile_horiz = QAction(self.tr("Tile Horizontally"), self)
        self.act_tile_horiz.setStatusTip(self.tr("Split the workspace into equal horizontal rows"))
        self.act_tile_horiz.triggered.connect(lambda: self._tile_views_direction("h"))

        self.act_tile_grid = QAction(self.tr("Smart Grid"), self)
        self.act_tile_grid.setStatusTip(self.tr("Arrange subwindows in a near-square grid"))
        self.act_tile_grid.triggered.connect(self._tile_views_grid)

        self.act_link_group = QAction(self.tr("Link Pan/Zoom"), self)
        self.act_link_group.setCheckable(True)  # checked when in any group
        self.act_link_group.triggered.connect(self._cycle_group_for_active)  # << add

        self.act_undo = QAction(QIcon(undoicon_path), self.tr("Undo"), self)
        self.act_redo = QAction(QIcon(redoicon_path), self.tr("Redo"), self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)               # Ctrl+Z
        self.act_redo.setShortcuts([QKeySequence.StandardKey.Redo, "Ctrl+Y"])  # Shift+Ctrl+Z / Ctrl+Y
        self.act_undo.setIconVisibleInMenu(True)
        self.act_redo.setIconVisibleInMenu(True)
        self.act_undo.triggered.connect(self._undo_active)
        self.act_redo.triggered.connect(self._redo_active)

        # View-ish action (toolbar toggle)
        self.act_autostretch = QAction(self.tr("Display-Stretch"), self, checkable=True)
        self.act_autostretch.setStatusTip(self.tr("Toggle display auto-stretch for the active window"))
        self.act_autostretch.setShortcut(QKeySequence("A"))  # optional: mirror the view shortcut
        self.act_autostretch.toggled.connect(self._toggle_autostretch)

        self.act_hardstretch = QAction(self.tr("Hard-Display-Stretch"), self, checkable=True)
        self.addAction(self.act_hardstretch)
        self.act_hardstretch.setShortcut(QKeySequence("H"))
        self.act_hardstretch.setStatusTip(self.tr("Toggle hard profile for Display-Stretch (H)"))

        # use toggled(bool), not triggered()
        self.act_hardstretch.toggled.connect(self._set_hard_autostretch_from_action)

        # NEW: Linked/Unlinked toggle (global default via QSettings, per-view runtime)
        self.act_stretch_linked = QAction(self.tr("Link RGB channels"), self, checkable=True)
        self.act_stretch_linked.setStatusTip(self.tr("Apply the same stretch to all RGB channels"))
        self.act_stretch_linked.setShortcut(QKeySequence("Ctrl+Shift+L"))
        self.act_stretch_linked.setChecked(
            self.settings.value("display/stretch_linked", False, type=bool)
        )
        self.act_stretch_linked.toggled.connect(self._set_linked_stretch_from_action)

        self.act_display_target = QAction(self.tr("Set Target Median..."), self)
        self.act_display_target.setStatusTip(self.tr("Set the target median for Display-Stretch (e.g., 0.30)"))
        self.act_display_target.triggered.connect(self._edit_display_target)

        self.act_display_sigma = QAction(self.tr("Set Sigma..."), self)
        self.act_display_sigma.setStatusTip(self.tr("Set the sigma for Display-Stretch (e.g., 5.0)"))
        self.act_display_sigma.triggered.connect(self._edit_display_sigma)

        # Defaults if not already present
        if self.settings.value("display/target", None) is None:
            self.settings.setValue("display/target", 0.30)
        if self.settings.value("display/sigma", None) is None:
            self.settings.setValue("display/sigma", 5.0)

        self.act_bake_display_stretch = QAction(self.tr("Make Display-Stretch Permanent"), self)
        self.act_bake_display_stretch.setStatusTip(
            self.tr("Apply the current Display-Stretch to the image and add an undo step")
        )
        # choose any shortcut you like; avoid Ctrl+A etc
        self.act_bake_display_stretch.setShortcut(QKeySequence("Shift+A"))
        self.act_bake_display_stretch.triggered.connect(self._bake_display_stretch)

        # --- Zoom controls ---
        self.act_zoom_out = QAction(QIcon.fromTheme("zoom-out"), self.tr("Zoom Out"), self)
        self.act_zoom_out.setStatusTip(self.tr("Zoom out"))
        self.act_zoom_out.setShortcuts([QKeySequence("Ctrl+-")])
        self.act_zoom_out.triggered.connect(lambda: self._zoom_step_active(-1))
        self._linux_force_text_action(self.act_zoom_out, "−")   # true minus

        self.act_zoom_in = QAction(QIcon.fromTheme("zoom-in"), self.tr("Zoom In"), self)
        self.act_zoom_in.setStatusTip(self.tr("Zoom in"))
        self.act_zoom_in.setShortcuts([QKeySequence("Ctrl++"), QKeySequence("Ctrl+=")])
        self.act_zoom_in.triggered.connect(lambda: self._zoom_step_active(+1))
        self._linux_force_text_action(self.act_zoom_in, "+")

        self.act_zoom_1_1 = QAction(QIcon.fromTheme("zoom-original"), self.tr("1:1"), self)
        self.act_zoom_1_1.setStatusTip(self.tr("Zoom to 100% (pixel-for-pixel)"))
        self.act_zoom_1_1.setShortcut(QKeySequence("Ctrl+1"))
        self.act_zoom_1_1.triggered.connect(self._zoom_active_1_1)
        self._linux_force_text_action(self.act_zoom_1_1, "1:1")

        self.act_zoom_fit = QAction(QIcon.fromTheme("zoom-fit-best"), self.tr("Fit"), self)
        self.act_zoom_fit.setStatusTip(self.tr("Fit image to current window"))
        self.act_zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        self.act_zoom_fit.triggered.connect(self._zoom_active_fit)
        self.act_zoom_fit.setCheckable(True)
        self._linux_force_text_action(self.act_zoom_fit, "Fit")

        self.act_auto_fit_resize = QAction(self.tr("Auto-fit on Resize"), self)
        self.act_auto_fit_resize.setCheckable(True)

        auto_on = self.settings.value("view/auto_fit_on_resize", False, type=bool)
        self._auto_fit_on_resize = bool(auto_on)
        self.act_auto_fit_resize.setChecked(self._auto_fit_on_resize)

        self.act_auto_fit_resize.toggled.connect(self._toggle_auto_fit_on_resize)

        # View state copy/paste (optional quick commands)
        self._copied_view_state = None
        self.act_copy_view = QAction(self.tr("Copy View (zoom/pan)"), self)
        self.act_paste_view = QAction(self.tr("Paste View"), self)
        self.act_copy_view.setShortcut("Ctrl+Shift+C")
        self.act_paste_view.setShortcut("Ctrl+Shift+V")
        self.act_copy_view.triggered.connect(self._copy_active_view)
        self.act_paste_view.triggered.connect(self._paste_active_view)
        # --- Edit: Mono -> RGB (triplicate channels) ---
        self.act_mono_to_rgb = QAction(self.tr("Convert Mono to RGB"), self)
        self.act_mono_to_rgb.setStatusTip(self.tr("Convert a mono image to RGB by duplicating the channel"))
        self.act_mono_to_rgb.triggered.connect(self._convert_mono_to_rgb_active)
        self.act_swap_rb = QAction(self.tr("Swap R and B Channels"), self)
        self.act_swap_rb.setStatusTip(self.tr("Swap red and blue channels on the active RGB image"))
        self.act_swap_rb.triggered.connect(self._swap_rb_active)
        # Functions
        self.act_crop = QAction(QIcon(cropicon_path), self.tr("Crop..."), self)
        self.act_crop.setStatusTip(self.tr("Crop / rotate with handles"))
        self.act_crop.setIconVisibleInMenu(True)
        self.act_crop.triggered.connect(self._open_crop_dialog)

        self.act_histogram = QAction(QIcon(histogram_path), self.tr("Histogram..."), self)
        self.act_histogram.setStatusTip(self.tr("View histogram and basic stats for the active image"))
        self.act_histogram.setIconVisibleInMenu(True)
        self.act_histogram.triggered.connect(self._open_histogram)

        self.act_stat_stretch = QAction(QIcon(statstretch_path), self.tr("Statistical Stretch..."), self)
        self.act_stat_stretch.setStatusTip(self.tr("Stretch the image using median/SD statistics"))
        self.act_stat_stretch.setIconVisibleInMenu(True)
        self.act_stat_stretch.triggered.connect(self._open_statistical_stretch)

        self.act_star_stretch = QAction(QIcon(starstretch_path), self.tr("Star Stretch..."), self)
        self.act_star_stretch.setStatusTip(self.tr("Arcsinh star stretch with optional SCNR and color boost"))
        self.act_star_stretch.setIconVisibleInMenu(True)
        self.act_star_stretch.triggered.connect(self._open_star_stretch)

        self.act_curves = QAction(QIcon(curves_path), self.tr("Curves Editor..."), self)
        self.act_curves.setStatusTip(self.tr("Open the Curves Editor for the active image"))
        self.act_curves.setIconVisibleInMenu(True)
        self.act_curves.triggered.connect(self._open_curves_editor)

        self.act_ghs = QAction(QIcon(uhs_path), self.tr("Hyperbolic Stretch..."), self)
        self.act_ghs.setStatusTip(self.tr("Generalized hyperbolic stretch (α/beta/gamma, LP/HP, pivot)"))
        self.act_ghs.setIconVisibleInMenu(True)
        self.act_ghs.triggered.connect(self._open_hyperbolic)

        self.act_abe = QAction(QIcon(abeicon_path), self.tr("ABE..."), self)
        self.act_abe.setStatusTip(self.tr("Automatic Background Extraction"))
        self.act_abe.setIconVisibleInMenu(True)
        self.act_abe.triggered.connect(self._open_abe_tool)

        self.act_graxpert = QAction(QIcon(graxperticon_path), self.tr("Remove Gradient (GraXpert)..."), self)
        self.act_graxpert.setIconVisibleInMenu(True)
        self.act_graxpert.setStatusTip(self.tr("Run GraXpert background extraction on the active image"))
        self.act_graxpert.triggered.connect(self._open_graxpert)

        self.act_remove_stars = QAction(QIcon(starnet_path), self.tr("Remove Stars..."), self)
        self.act_remove_stars.setIconVisibleInMenu(True)
        self.act_remove_stars.setStatusTip(self.tr("Run star removal on the active image"))
        self.act_remove_stars.triggered.connect(lambda: self._remove_stars())

        self.act_add_stars = QAction(QIcon(staradd_path), self.tr("Add Stars..."), self)
        self.act_add_stars.setStatusTip(self.tr("Blend a starless view with a stars-only view"))
        self.act_add_stars.setIconVisibleInMenu(True)
        self.act_add_stars.triggered.connect(lambda: self._add_stars())

        self.act_pedestal = QAction(QIcon(pedestal_icon_path), self.tr("Remove Pedestal"), self)
        self.act_pedestal.setToolTip(self.tr("Subtract per-channel minimum.\nClick: active view\nAlt+Drag: drop onto a view"))
        self.act_pedestal.setShortcut("Ctrl+P")
        self.act_pedestal.triggered.connect(self._on_remove_pedestal)

        self.act_linear_fit = QAction(QIcon(linearfit_path), self.tr("Linear Fit..."), self)
        self.act_linear_fit.setIconVisibleInMenu(True)
        self.act_linear_fit.setStatusTip(self.tr("Match image levels using Linear Fit"))
        # optional shortcut; change if you already use it elsewhere
        self.act_linear_fit.setShortcut("Ctrl+L")
        self.act_linear_fit.triggered.connect(self._open_linear_fit)

        self.act_remove_green = QAction(QIcon(green_path), self.tr("Remove Green (SCNR)..."), self)
        self.act_remove_green.setToolTip(self.tr("SCNR-style green channel removal."))
        self.act_remove_green.setIconVisibleInMenu(True)
        self.act_remove_green.triggered.connect(self._open_remove_green)

        # Texture and Clarity
        self.act_texture_clarity = QAction(QIcon(texture_clarity_path), self.tr("Texture and Clarity..."), self)
        self.act_texture_clarity.setToolTip(self.tr("Enhance texture and clarity using Unsharp Masking"))
        self.act_texture_clarity.setIconVisibleInMenu(True)
        self.act_texture_clarity.triggered.connect(self._open_texture_clarity)

        self.act_background_neutral = QAction(QIcon(neutral_path), self.tr("Background Neutralization..."), self)
        self.act_background_neutral.setStatusTip(self.tr("Neutralize background color balance using a sampled region"))
        self.act_background_neutral.setIconVisibleInMenu(True)
        self.act_background_neutral.triggered.connect(self._open_background_neutral)

        self.act_white_balance = QAction(QIcon(whitebalance_path), self.tr("White Balance..."), self)
        self.act_white_balance.setStatusTip(self.tr("Apply white balance (Star-Based, Manual, or Auto)"))
        self.act_white_balance.triggered.connect(self._open_white_balance)

        self.act_sfcc = QAction(QIcon(spcc_icon_path), self.tr("Spectral Flux Color Calibration..."), self)
        self.act_sfcc.setObjectName("sfcc")
        self.act_sfcc.setToolTip(self.tr("Open SFCC (Pickles + Filters + Sensor QE)"))
        self.act_sfcc.triggered.connect(self.SFCC_show)

        self.act_convo = QAction(QIcon(convoicon_path), self.tr("Convolution / Deconvolution..."), self)
        self.act_convo.setObjectName("convo_deconvo")
        self.act_convo.setToolTip(self.tr("Open Convolution / Deconvolution"))
        self.act_convo.triggered.connect(self.show_convo_deconvo)

        self.act_multiscale_decomp = QAction(QIcon(multiscale_decomp_path), self.tr("Multiscale Decomposition..."), self)
        self.act_multiscale_decomp.setStatusTip(self.tr("Multiscale detail/residual decomposition with per-layer controls"))
        self.act_multiscale_decomp.setIconVisibleInMenu(True)
        self.act_multiscale_decomp.triggered.connect(self._open_multiscale_decomp)




        # --- Extract Luminance main action ---
        self.act_extract_luma = QAction(QIcon(LExtract_path), self.tr("Extract Luminance"), self)
        self.act_extract_luma.setStatusTip(self.tr("Create a new mono document using the selected luminance method"))
        self.act_extract_luma.setIconVisibleInMenu(True)
        self.act_extract_luma.triggered.connect(lambda: self._extract_luminance(doc=None))

        # --- Luminance method actions (checkable group) ---
        self.luma_method = getattr(self, "luma_method", "rec709")  # default
        self._luma_group = QActionGroup(self)
        self._luma_group.setExclusive(True)
        self._luma_sensor_actions = {}  # key -> QAction

        def _mk(method_key, text):
            act = QAction(text, self, checkable=True)
            act.setData(method_key)
            self._luma_group.addAction(act)
            return act

        self.act_luma_rec709  = _mk("rec709",  "Broadband RGB (Rec.709)")
        self.act_luma_max     = _mk("max",     "Narrowband mappings (Max)")
        self.act_luma_snr     = _mk("snr",     "Unequal Noise (SNR)")
        self.act_luma_rec601  = _mk("rec601",  "Rec.601")
        self.act_luma_rec2020 = _mk("rec2020", "Rec.2020")

        # restore selection
        for a in self._luma_group.actions():
            a.setChecked(a.data() == self.luma_method)

        # update method when user picks from the menu
        def _on_luma_pick(act):
            key = act.data()
            if key is None:
                return
            self.luma_method = str(key)
            try:
                self.settings.setValue("ui/luminance_method", self.luma_method)
            except Exception:
                pass

        self._luma_group.triggered.connect(_on_luma_pick)

        self.act_recombine_luma = QAction(QIcon(LInsert_path), self.tr("Recombine Luminance..."), self)
        self.act_recombine_luma.setStatusTip(self.tr("Replace the active image's luminance from another view"))
        self.act_recombine_luma.setIconVisibleInMenu(True)
        self.act_recombine_luma.triggered.connect(lambda: self._recombine_luminance_ui(target_doc=None))

        self.act_rgb_extract = QAction(QIcon(rgbextract_path), self.tr("RGB Extract"), self)
        self.act_rgb_extract.setIconVisibleInMenu(True)
        self.act_rgb_extract.setStatusTip(self.tr("Extract R/G/B as three mono documents"))
        self.act_rgb_extract.triggered.connect(self._rgb_extract_active)

        self.act_rgb_combine = QAction(QIcon(rgbcombo_path), self.tr("RGB Combination..."), self)
        self.act_rgb_combine.setIconVisibleInMenu(True)
        self.act_rgb_combine.setStatusTip(self.tr("Combine three mono images into RGB"))
        self.act_rgb_combine.triggered.connect(self._open_rgb_combination)

        self.act_blemish = QAction(QIcon(blastericon_path), self.tr("Blemish Blaster..."), self)
        self.act_blemish.setIconVisibleInMenu(True)
        self.act_blemish.setStatusTip(self.tr("Interactive blemish removal on the active view"))
        self.act_blemish.triggered.connect(self._open_blemish_blaster)

        self.act_clone_stamp = QAction(QIcon(clonestampicon_path), self.tr("Clone Stamp..."), self)
        self.act_clone_stamp.setIconVisibleInMenu(True)
        self.act_clone_stamp.setStatusTip(self.tr("Interactive clone stamp on the active view"))
        self.act_clone_stamp.triggered.connect(self._open_clone_stamp)

        self.act_wavescale_hdr = QAction(QIcon(hdr_path), self.tr("WaveScale HDR..."), self)
        self.act_wavescale_hdr.setStatusTip(self.tr("Wave-scale HDR with luminance-masked starlet"))
        self.act_wavescale_hdr.setIconVisibleInMenu(True)
        self.act_wavescale_hdr.triggered.connect(self._open_wavescale_hdr)

        self.act_wavescale_de = QAction(QIcon(dse_icon_path), self.tr("WaveScale Dark Enhancer..."), self)
        self.act_wavescale_de.setStatusTip(self.tr("Enhance faint/dark structures with wavelet-guided masking"))
        self.act_wavescale_de.setIconVisibleInMenu(True)
        self.act_wavescale_de.triggered.connect(self._open_wavescale_dark_enhance)

        self.act_clahe = QAction(QIcon(clahe_path), self.tr("CLAHE..."), self)
        self.act_clahe.setStatusTip(self.tr("Contrast Limited Adaptive Histogram Equalization"))
        self.act_clahe.setIconVisibleInMenu(True)
        self.act_clahe.triggered.connect(self._open_clahe)

        self.act_morphology = QAction(QIcon(morpho_path), self.tr("Morphological Operations..."), self)
        self.act_morphology.setStatusTip(self.tr("Erosion, dilation, opening, and closing."))
        self.act_morphology.setIconVisibleInMenu(True)
        self.act_morphology.triggered.connect(self._open_morphology)

        self.act_pixelmath = QAction(QIcon(pixelmath_path), self.tr("Pixel Math..."), self)
        self.act_pixelmath.setStatusTip(self.tr("Evaluate expressions using open view names"))
        self.act_pixelmath.setIconVisibleInMenu(True)
        self.act_pixelmath.triggered.connect(self._open_pixel_math)

        self.act_signature = QAction(QIcon(signature_icon_path), self.tr("Signature / Insert..."), self)
        self.act_signature.setIconVisibleInMenu(True)
        self.act_signature.setStatusTip(self.tr("Add signatures/overlays and bake them into the active image"))
        self.act_signature.triggered.connect(self._open_signature_insert)

        self.act_halobgon = QAction(QIcon(halo_path), self.tr("Halo-B-Gon..."), self)
        self.act_halobgon.setIconVisibleInMenu(True)
        self.act_halobgon.setStatusTip(self.tr("Remove those pesky halos around your stars"))
        self.act_halobgon.triggered.connect(self._open_halo_b_gon)

        self.act_image_combine = QAction(QIcon(imagecombine_path), self.tr("Image Combine..."), self)
        self.act_image_combine.setIconVisibleInMenu(True)
        self.act_image_combine.setStatusTip(self.tr("Blend two open images (replace A or create new)"))
        self.act_image_combine.triggered.connect(self._open_image_combine)

        # --- Geometry ---
        self.act_geom_invert = QAction(QIcon(invert_path), self.tr("Invert"), self)
        self.act_geom_invert.setIconVisibleInMenu(True)
        self.act_geom_invert.setStatusTip(self.tr("Invert image colors"))
        self.act_geom_invert.triggered.connect(self._exec_geom_invert)

        self.act_geom_flip_h = QAction(QIcon(fliphorizontal_path), self.tr("Flip Horizontal"), self)
        self.act_geom_flip_h.setIconVisibleInMenu(True)
        self.act_geom_flip_h.setStatusTip(self.tr("Flip image left<->right"))
        self.act_geom_flip_h.triggered.connect(self._exec_geom_flip_h)

        self.act_geom_flip_v = QAction(QIcon(flipvertical_path), self.tr("Flip Vertical"), self)
        self.act_geom_flip_v.setIconVisibleInMenu(True)
        self.act_geom_flip_v.setStatusTip(self.tr("Flip image top<->bottom"))
        self.act_geom_flip_v.triggered.connect(self._exec_geom_flip_v)

        self.act_geom_rot_cw = QAction(QIcon(rotateclockwise_path), self.tr("Rotate 90° Clockwise"), self)
        self.act_geom_rot_cw.setIconVisibleInMenu(True)
        self.act_geom_rot_cw.setStatusTip(self.tr("Rotate image 90° clockwise"))
        self.act_geom_rot_cw.triggered.connect(self._exec_geom_rot_cw)

        self.act_geom_rot_ccw = QAction(QIcon(rotatecounterclockwise_path), self.tr("Rotate 90° Counterclockwise"), self)
        self.act_geom_rot_ccw.setIconVisibleInMenu(True)
        self.act_geom_rot_ccw.setStatusTip(self.tr("Rotate image 90° counterclockwise"))
        self.act_geom_rot_ccw.triggered.connect(self._exec_geom_rot_ccw)

        self.act_geom_rot_180 = QAction(QIcon(rotate180_path), self.tr("Rotate 180°"), self)
        self.act_geom_rot_180.setIconVisibleInMenu(True)
        self.act_geom_rot_180.setStatusTip(self.tr("Rotate image 180°"))
        self.act_geom_rot_180.triggered.connect(self._exec_geom_rot_180)

        self.act_geom_rot_any = QAction(QIcon(rotatearbitrary_path), self.tr("Rotate..."), self)
        self.act_geom_rot_any.setIconVisibleInMenu(True)
        self.act_geom_rot_any.setStatusTip(self.tr("Rotate image by an arbitrary angle (degrees)"))
        self.act_geom_rot_any.triggered.connect(self._exec_geom_rot_any)


        self.act_geom_rescale = QAction(QIcon(rescale_path), self.tr("Rescale..."), self)
        self.act_geom_rescale.setIconVisibleInMenu(True)
        self.act_geom_rescale.setStatusTip(self.tr("Rescale image by a factor"))
        self.act_geom_rescale.triggered.connect(self._exec_geom_rescale)

        self.act_debayer = QAction(QIcon(debayer_path), self.tr("Debayer..."), self)
        self.act_debayer.setObjectName("debayer")
        self.act_debayer.setProperty("command_id", "debayer")
        self.act_debayer.setStatusTip(self.tr("Demosaic a Bayer-mosaic mono image to RGB"))
        self.act_debayer.triggered.connect(self._open_debayer)

        # (Optional example shortcuts; uncomment if you want)
        self.act_geom_invert.setShortcut("Ctrl+I")
        
        # self.act_geom_flip_h.setShortcut("H")
        # self.act_geom_flip_v.setShortcut("V")
        # self.act_geom_rot_cw.setShortcut("]")
        # self.act_geom_rot_ccw.setShortcut("[")
        # self.act_geom_rescale.setShortcut("Ctrl+R")


        # actions (use your actual icon paths if you have them)
        try:
            cosmic_icon = QIcon(cosmic_path)  # define cosmic_path like your other icons (same pattern as halo_path)
        except Exception:
            cosmic_icon = QIcon()

        try:
            sat_icon = QIcon(satellite_path)  # optional icon for satellite
        except Exception:
            sat_icon = QIcon()

        self.actCosmicUI  = QAction(cosmic_icon, self.tr("Cosmic Clarity UI..."), self)
        self.actCosmicSat = QAction(sat_icon, self.tr("Cosmic Clarity Satellite..."), self)

        self.actCosmicUI.triggered.connect(self._open_cosmic_clarity_ui)
        self.actCosmicSat.triggered.connect(self._open_cosmic_clarity_satellite)


        ab_icon = QIcon(aberration_path)  # falls back if file missing

        self.actAberrationAI = QAction(ab_icon, self.tr("Aberration Correction (AI)..."), self)
        self.actAberrationAI.triggered.connect(self._open_aberration_ai)



        #Tools
        self.act_blink = QAction(QIcon(blink_path), self.tr("Blink Comparator..."), self)
        self.act_blink.setStatusTip(self.tr("Compare a stack of images by blinking"))
        self.act_blink.triggered.connect(self._open_blink_tool)        

        self.act_ppp = QAction(QIcon(ppp_path), self.tr("Perfect Palette Picker..."), self)
        self.act_ppp.setStatusTip(self.tr("Pick the perfect palette for your image"))
        self.act_ppp.triggered.connect(self._open_ppp_tool) 

        self.act_narrowband_normalization = QAction(QIcon(narrowbandnormalization_path), self.tr("Narrowband Normalization..."), self)
        self.act_narrowband_normalization.setStatusTip(
            self.tr("Normalize HOO/SHO/HSO/HOS (PixelMath port by Bill Blanshan and Mike Cranfield)")
        )

        self.act_narrowband_normalization.triggered.connect(self._open_narrowband_normalization_tool)

        self.act_nbtorgb = QAction(QIcon(nbtorgb_path), self.tr("NB->RGB Stars..."), self)
        self.act_nbtorgb.setStatusTip(self.tr("Combine narrowband to RGB with optional OSC stars"))
        self.act_nbtorgb.setIconVisibleInMenu(True)
        self.act_nbtorgb.triggered.connect(self._open_nbtorgb_tool)

        self.act_selective_color = QAction(QIcon(selectivecolor_path), self.tr("Selective Color Correction..."), self)
        self.act_selective_color.setStatusTip(self.tr("Adjust specific hue ranges with CMY/RGB controls"))
        self.act_selective_color.triggered.connect(self._open_selective_color_tool)

        self.act_magnitude = QAction(QIcon(magnitude_path), self.tr("Magnitude / Surface Brightness..."), self)
        self.act_magnitude.setStatusTip(self.tr("Measure magnitude and mag/arcsec² from the active linear view"))
        self.act_magnitude.setIconVisibleInMenu(True)
        self.act_magnitude.triggered.connect(self._open_magnitude_tool)

        # NEW: Frequency Separation
        self.act_freqsep = QAction(QIcon(freqsep_path), self.tr("Frequency Separation..."), self)
        self.act_freqsep.setStatusTip(self.tr("Split into LF/HF and enhance HF (scale, wavelet, denoise)"))
        self.act_freqsep.setIconVisibleInMenu(True)
        self.act_freqsep.triggered.connect(self._open_freqsep_tool)

        self.act_contsub = QAction(QIcon(contsub_path), self.tr("Continuum Subtract..."), self)
        self.act_contsub.setStatusTip(self.tr("Continuum Subtract (NB - scaled broadband)"))
        self.act_contsub.setIconVisibleInMenu(True)
        self.act_contsub.triggered.connect(self._open_contsub_tool)

        # History
        self.act_history_explorer = QAction(self.tr("History Explorer..."), self)
        self.act_history_explorer.setStatusTip(self.tr("Inspect and restore from the slot's history"))
        self.act_history_explorer.triggered.connect(self._open_history_explorer)


        #STAR STUFF
        self.act_image_peeker = QAction(QIcon(peeker_icon), self.tr("Image Peeker..."), self)
        self.act_image_peeker.setIconVisibleInMenu(True)
        self.act_image_peeker.setStatusTip(self.tr("Image Inspector and Focal Plane Analysis"))
        self.act_image_peeker.triggered.connect(self._open_image_peeker)

        self.act_psf_viewer = QAction(QIcon(psf_path), self.tr("PSF Viewer..."), self)
        self.act_psf_viewer.setIconVisibleInMenu(True)
        self.act_psf_viewer.setStatusTip(self.tr("Inspect star PSF/HFR and flux histograms (SEP)"))
        self.act_psf_viewer.triggered.connect(self._open_psf_viewer)        

        self.act_stacking_suite = QAction(QIcon(stacking_path), self.tr("Stacking Suite..."), self)
        self.act_stacking_suite.setIconVisibleInMenu(True)
        self.act_stacking_suite.setStatusTip(self.tr("Stacking! Darks, Flats, Lights, Calibration, Drizzle, and more!!"))
        self.act_stacking_suite.triggered.connect(self._open_stacking_suite)

        self.act_live_stacking = QAction(QIcon(livestacking_path), self.tr("Live Stacking..."), self)
        self.act_live_stacking.setIconVisibleInMenu(True)
        self.act_live_stacking.setStatusTip(self.tr("Live monitor and stack incoming frames"))
        self.act_live_stacking.triggered.connect(self._open_live_stacking)

        self.act_planetary_stacker = QAction(QIcon(planetarystacker_path), self.tr("Planetary Stacker..."), self)
        self.act_planetary_stacker.setIconVisibleInMenu(True)
        self.act_planetary_stacker.setStatusTip(self.tr("Stack SER videos (planetary/solar/lunar)"))
        self.act_planetary_stacker.triggered.connect(self._open_planetary_stacker)

        self.act_planet_projection = QAction(QIcon(planetprojection_path), self.tr("3D Projection..."), self)
        self.act_planet_projection.setIconVisibleInMenu(True)
        self.act_planet_projection.setStatusTip(self.tr("View your planets with stereographic projection"))
        self.act_planet_projection.triggered.connect(self._open_planet_projection)

        self.act_plate_solve = QAction(QIcon(platesolve_path), self.tr("Plate Solver..."), self)
        self.act_plate_solve.setIconVisibleInMenu(True)
        self.act_plate_solve.setStatusTip(self.tr("Solve WCS/SIP for the active image or a file"))
        self.act_plate_solve.triggered.connect(self._open_plate_solver)

        self.act_star_align = QAction(QIcon(staralign_path), self.tr("Stellar Alignment..."), self)
        self.act_star_align.setIconVisibleInMenu(True)
        self.act_star_align.setStatusTip(self.tr("Align images via astroalign / triangles"))
        self.act_star_align.triggered.connect(self._open_stellar_alignment)

        self.act_star_register = QAction(QIcon(starregistration_path), self.tr("Stellar Register..."), self)
        self.act_star_register.setIconVisibleInMenu(True)
        self.act_star_register.setStatusTip(self.tr("Batch-align frames to a reference"))
        self.act_star_register.triggered.connect(self._open_stellar_registration)

        self.act_mosaic_master = QAction(QIcon(mosaic_path), self.tr("Mosaic Master..."), self)
        self.act_mosaic_master.setIconVisibleInMenu(True)
        self.act_mosaic_master.setStatusTip(self.tr("Build mosaics from overlapping frames"))
        self.act_mosaic_master.triggered.connect(self._open_mosaic_master)

        self.act_supernova_hunter = QAction(QIcon(supernova_path), self.tr("Supernova / Asteroid Hunter..."), self)
        self.act_supernova_hunter.setIconVisibleInMenu(True)
        self.act_supernova_hunter.setStatusTip(self.tr("Find transients/anomalies across frames"))
        self.act_supernova_hunter.triggered.connect(self._open_supernova_hunter)

        self.act_star_spikes = QAction(QIcon(starspike_path), self.tr("Diffraction Spikes..."), self)
        self.act_star_spikes.setIconVisibleInMenu(True)
        self.act_star_spikes.setStatusTip(self.tr("Add diffraction spikes to detected stars"))
        self.act_star_spikes.triggered.connect(self._open_star_spikes)

        self.act_astrospike = QAction(QIcon(astrospike_path), self.tr("AstroSpike..."), self)
        self.act_astrospike.setIconVisibleInMenu(True)
        self.act_astrospike.setStatusTip(self.tr("Advanced diffraction spikes with halos, flares and rainbow effects"))
        self.act_astrospike.triggered.connect(self._open_astrospike)

        self.act_exo_detector = QAction(QIcon(exoicon_path), self.tr("Exoplanet Detector..."), self)
        self.act_exo_detector.setIconVisibleInMenu(True)
        self.act_exo_detector.setStatusTip(self.tr("Detect exoplanet transits from time-series subs"))
        self.act_exo_detector.triggered.connect(self._open_exo_detector)

        self.act_isophote = QAction(QIcon(isophote_path), self.tr("GLIMR -- Isophote Modeler..."), self)
        self.act_isophote.setIconVisibleInMenu(True)
        self.act_isophote.setStatusTip(self.tr("Fit galaxy isophotes and reveal residuals"))
        self.act_isophote.triggered.connect(self._open_isophote)

        self.act_rgb_align = QAction(QIcon(rgbalign_path), self.tr("RGB Align..."), self)
        self.act_rgb_align.setIconVisibleInMenu(True)
        self.act_rgb_align.setStatusTip(self.tr("Align R and B channels to G using astroalign (affine/homography/poly)"))
        self.act_rgb_align.triggered.connect(self._open_rgb_align)

        self.act_finder_chart = QAction(QIcon(finderchart_path), self.tr("Finder Chart..."), self)
        self.act_finder_chart.setIconVisibleInMenu(True)
        self.act_finder_chart.setStatusTip(self.tr("Show a finder chart for the active plate-solved image"))
        self.act_finder_chart.triggered.connect(self._open_finder_chart)

        self.act_whats_in_my_sky = QAction(QIcon(wims_path), self.tr("What's In My Sky..."), self)
        self.act_whats_in_my_sky.setIconVisibleInMenu(True)
        self.act_whats_in_my_sky.setStatusTip(self.tr("Plan targets by altitude, transit time, and lunar separation"))
        self.act_whats_in_my_sky.triggered.connect(self._open_whats_in_my_sky)

        self.act_wimi = QAction(QIcon(wimi_path), self.tr("What's In My Image..."), self)
        self.act_wimi.setIconVisibleInMenu(True)
        self.act_wimi.setStatusTip(self.tr("Identify objects in a plate-solved frame"))
        self.act_wimi.triggered.connect(self._open_wimi)

        # --- Scripts actions ---
        self.act_open_scripts_folder = QAction(self.tr("Open Scripts Folder..."), self)
        self.act_open_scripts_folder.setStatusTip(self.tr("Open the SASpro user scripts folder"))
        self.act_open_scripts_folder.triggered.connect(self._open_scripts_folder)

        self.act_reload_scripts = QAction(self.tr("Reload Scripts"), self)
        self.act_reload_scripts.setStatusTip(self.tr("Rescan the scripts folder and reload .py files"))
        self.act_reload_scripts.triggered.connect(self._reload_scripts)

        self.act_create_sample_script = QAction(self.tr("Create Sample Scripts..."), self)
        self.act_create_sample_script.setStatusTip(self.tr("Write a ready-to-edit sample script into the scripts folder"))
        self.act_create_sample_script.triggered.connect(self._create_sample_script)

        self.act_script_editor = QAction(self.tr("Script Editor..."), self)
        self.act_script_editor.setStatusTip(self.tr("Open the built-in script editor"))
        self.act_script_editor.triggered.connect(self._show_script_editor)

        self.act_open_user_scripts_github = QAction(self.tr("Open User Scripts (GoogleDrive)..."), self)
        self.act_open_user_scripts_github.triggered.connect(self._open_user_scripts_github)

        self.act_open_scripts_discord = QAction(self.tr("Open Scripts Forum (Discord)..."), self)
        self.act_open_scripts_discord.triggered.connect(self._open_scripts_discord_forum)

        # --- FITS Header Modifier action ---
        self.act_fits_modifier = QAction(self.tr("FITS Header Modifier..."), self)
        # self.act_fits_modifier.setIcon(QIcon(path_to_icon))  # (optional) icon goes here later
        self.act_fits_modifier.setIconVisibleInMenu(True)
        self.act_fits_modifier.setStatusTip(self.tr("View/Edit FITS headers"))
        self.act_fits_modifier.triggered.connect(self._open_fits_modifier)

        self.act_fits_batch_modifier = QAction(self.tr("FITS Header Batch Modifier..."), self)
        # self.act_fits_modifier.setIcon(QIcon(path_to_icon))  # (optional) icon goes here later
        self.act_fits_batch_modifier.setIconVisibleInMenu(True)
        self.act_fits_batch_modifier.setStatusTip(self.tr("Batch Modify FITS Headers"))
        self.act_fits_batch_modifier.triggered.connect(self._open_fits_batch_modifier)

        self.act_batch_renamer = QAction(self.tr("Batch Rename from FITS..."), self)
        # self.act_batch_renamer.setIcon(QIcon(batch_renamer_icon_path))  # (optional icon)
        self.act_batch_renamer.triggered.connect(self._open_batch_renamer)

        self.act_astrobin_exporter = QAction(self.tr("AstroBin Exporter..."), self)
        # self.act_astrobin_exporter.setIcon(QIcon(astrobin_icon_path))  # optional icon
        self.act_astrobin_exporter.triggered.connect(self._open_astrobin_exporter)

        self.act_batch_convert = QAction(self.tr("Batch Converter..."), self)
        # self.act_batch_convert.setIcon(QIcon("path/to/icon.svg"))  # optional later
        self.act_batch_convert.triggered.connect(self._open_batch_convert)

        self.act_copy_astrometry = QAction(self.tr("Copy Astrometric Solution..."), self)
        self.act_copy_astrometry.triggered.connect(self._open_copy_astrometry)

        self.act_acv_exporter = QAction(self.tr("Astro Catalogue Viewer Exporter..."), self)
        self.act_acv_exporter.setIconVisibleInMenu(True)
        self.act_acv_exporter.setStatusTip(self.tr("Export current view into Astro Catalogue Viewer folders"))
        self.act_acv_exporter.triggered.connect(self._open_acv_exporter)



        # Create Mask
        self.act_create_mask = QAction(QIcon(maskcreate_path), self.tr("Create Mask..."), self)
        self.act_create_mask.setIconVisibleInMenu(True)
        self.act_create_mask.setStatusTip(self.tr("Create a mask from the active image"))
        self.act_create_mask.triggered.connect(self._action_create_mask)

        # --- Masks ---
        self.act_apply_mask = QAction(QIcon(maskapply_path), self.tr("Apply Mask"), self)
        self.act_apply_mask.setStatusTip(self.tr("Apply a mask document to the active image"))
        self.act_apply_mask.triggered.connect(self._apply_mask_menu)

        self.act_remove_mask = QAction(QIcon(maskremove_path), self.tr("Remove Active Mask"), self)
        self.act_remove_mask.setStatusTip(self.tr("Remove the active mask from the active image"))
        self.act_remove_mask.triggered.connect(self._remove_mask_menu)

        self.act_show_mask = QAction(self.tr("Show Mask Overlay"), self)
        self.act_hide_mask = QAction(self.tr("Hide Mask Overlay"), self)
        self.act_show_mask.triggered.connect(self._show_mask_overlay)
        self.act_hide_mask.triggered.connect(self._hide_mask_overlay)

        self.act_invert_mask = QAction(self.tr("Invert Mask"), self)
        self.act_invert_mask.triggered.connect(self._invert_mask)
        self.act_invert_mask.setShortcut("Ctrl+Shift+I")

        self.act_check_updates = QAction(self.tr("Check for Updates..."), self)
        self.act_check_updates.triggered.connect(self.check_for_updates_now)

        self.act_docs = QAction(self.tr("Documentation..."), self)
        self.act_docs.setStatusTip(self.tr("Open the Seti Astro Suite Pro online documentation"))
        self.act_docs.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/setiastro/setiastrosuitepro/wiki"))
        )

        # Qt6-safe shortcut for Help/Docs (F1)
        try:
            # Qt6 enum lives under StandardKey
            self.act_docs.setShortcut(QKeySequence(QKeySequence.StandardKey.HelpContents))
        except Exception:
            # Fallback works everywhere
            self.act_docs.setShortcut(QKeySequence("F1"))

        self.act_view_bundles = QAction(QIcon(viewbundles_path), self.tr("View Bundles..."), self)
        self.act_view_bundles.setStatusTip(self.tr("Create bundles of views; drop shortcuts to apply to all"))
        self.act_view_bundles.triggered.connect(self._open_view_bundles)

        self.act_function_bundles = QAction(QIcon(functionbundles_path), self.tr("Function Bundles..."), self)
        self.act_function_bundles.setStatusTip(self.tr("Create and run bundles of functions/shortcuts"))
        self.act_function_bundles.triggered.connect(self._open_function_bundles)

        # give each action a stable id and register
        def reg(cid, act):
            act.setProperty("command_id", cid)
            act.setObjectName(cid)  # also becomes default if we ever need it
            self.shortcuts.register_action(cid, act)

        # create manager once MDI exists
        if not hasattr(self, "shortcuts"):
            # self.mdi is your QMdiArea used elsewhere (stat_stretch uses it)
            self.shortcuts = ShortcutManager(self.mdi, self)

        # register whatever you want draggable/launchable
        reg("open",           self.act_open)
        reg("save_as",        self.act_save)
        reg("undo",           self.act_undo)
        reg("redo",           self.act_redo)
        reg("autostretch",    self.act_autostretch)
        reg("zoom_1_1",       self.act_zoom_1_1)
        reg("crop",           self.act_crop)
        reg("histogram",      self.act_histogram)
        reg("stat_stretch",   self.act_stat_stretch)
        reg("star_stretch",   self.act_star_stretch)
        reg("curves",         self.act_curves)
        reg("ghs",            self.act_ghs)
        reg("blink",          self.act_blink)
        reg("ppp",            self.act_ppp)
        reg("nbtorgb",       self.act_nbtorgb)
        reg("freqsep",       self.act_freqsep)
        reg("selective_color", self.act_selective_color)
        reg("narrowband_normalization", self.act_narrowband_normalization)
        reg("contsub",      self.act_contsub)
        reg("abe",          self.act_abe)
        reg("create_mask", self.act_create_mask)
        reg("graxpert", self.act_graxpert)
        reg("remove_stars", self.act_remove_stars)
        reg("add_stars", self.act_add_stars)
        reg("pedestal",       self.act_pedestal)
        reg("remove_green",   self.act_remove_green)
        reg("texture_clarity", self.act_texture_clarity)
        reg("background_neutral", self.act_background_neutral)
        reg("white_balance", self.act_white_balance)
        reg("sfcc",    self.act_sfcc)
        reg("convo", self.act_convo)
        reg("extract_luminance", self.act_extract_luma)
        reg("recombine_luminance", self.act_recombine_luma)
        reg("rgb_extract", self.act_rgb_extract)
        reg("rgb_combine", self.act_rgb_combine)
        reg("blemish_blaster", self.act_blemish)
        reg("clone_stamp", self.act_clone_stamp)
        reg("wavescale_hdr", self.act_wavescale_hdr)
        reg("wavescale_dark_enhance", self.act_wavescale_de)
        reg("clahe", self.act_clahe)
        reg("morphology", self.act_morphology)
        reg("pixel_math", self.act_pixelmath)
        reg("signature_insert", self.act_signature) 
        reg("halo_b_gon", self.act_halobgon)
        reg("planetary_stacker", self.act_planetary_stacker)
        reg("multiscale_decomp", self.act_multiscale_decomp)        
        reg("geom_invert",                 self.act_geom_invert)
        reg("geom_flip_horizontal",        self.act_geom_flip_h)
        reg("geom_flip_vertical",          self.act_geom_flip_v)
        reg("geom_rotate_clockwise",       self.act_geom_rot_cw)
        reg("geom_rotate_counterclockwise",self.act_geom_rot_ccw)
        reg("geom_rotate_180",             self.act_geom_rot_180) 
        reg("geom_rotate_any", self.act_geom_rot_any)
        reg("geom_rescale",                self.act_geom_rescale)        
        reg("project_new",  self.act_project_new)
        reg("project_save", self.act_project_save)
        reg("project_load", self.act_project_load)     
        reg("image_combine", self.act_image_combine)   
        reg("psf_viewer", self.act_psf_viewer)
        reg("plate_solve", self.act_plate_solve)
        reg("magnitude_tool", self.act_magnitude)
        reg("star_align", self.act_star_align)
        reg("star_register", self.act_star_register)
        reg("mosaic_master", self.act_mosaic_master)
        reg("image_peeker", self.act_image_peeker)
        reg("live_stacking", self.act_live_stacking)
        reg("stacking_suite", self.act_stacking_suite)
        reg("planet_projection", self.act_planet_projection)
        reg("supernova_hunter", self.act_supernova_hunter)
        reg("star_spikes", self.act_star_spikes)
        reg("astrospike", self.act_astrospike)
        reg("exo_detector", self.act_exo_detector)
        reg("isophote", self.act_isophote) 
        reg("finder_chart", self.act_finder_chart)
        reg("rgb_align", self.act_rgb_align) 
        reg("whats_in_my_sky", self.act_whats_in_my_sky)
        reg("whats_in_my_image", self.act_wimi)
        reg("linear_fit", self.act_linear_fit)
        reg("debayer", self.act_debayer)
        reg("cosmicclarity", self.actCosmicUI)
        reg("cosmicclaritysat", self.actCosmicSat)
        reg("aberrationai", self.actAberrationAI)
        reg("view_bundles", self.act_view_bundles)
        reg("function_bundles", self.act_function_bundles)

    def _reset_all_toolbars_to_factory(self):
        """
        Clears all persisted toolbar membership/order/hidden state so the UI
        returns to the factory layout defined in code.
        """
        if not hasattr(self, "settings"):
            return

        # 1) Clear global toolbar persistence
        keys_to_clear = [
            "Toolbar/Assignments",
            "Toolbar/HiddenPrev",
            "Toolbar/HiddenMigrationDone",

            # Per-toolbar order lists
            "Toolbar/View",
            "Toolbar/Functions",
            "Toolbar/Cosmic",
            "Toolbar/Tools",
            "Toolbar/Geometry",
            "Toolbar/StarStuff",
            "Toolbar/Masks",
            "Toolbar/WhatsInMy",
            "Toolbar/Bundles",
            "Toolbar/Hidden",
        ]

        for k in keys_to_clear:
            try:
                self.settings.remove(k)
            except Exception:
                pass

        # 2) Clear legacy hidden lists (old system) if they exist
        # (These are the ones like "Toolbar/Tools/Hidden")
        try:
            # brute-force remove known ones
            legacy_hidden = [
                "Toolbar/View/Hidden",
                "Toolbar/Functions/Hidden",
                "Toolbar/Cosmic/Hidden",
                "Toolbar/Tools/Hidden",
                "Toolbar/Geometry/Hidden",
                "Toolbar/StarStuff/Hidden",
                "Toolbar/Masks/Hidden",
                "Toolbar/WhatsInMy/Hidden",
                "Toolbar/Bundles/Hidden",
            ]
            for k in legacy_hidden:
                self.settings.remove(k)
        except Exception:
            pass

        try:
            self.settings.sync()
        except Exception:
            pass

        # 3) Rebuild toolbars from code defaults
        # Safest approach: remove existing toolbars and rebuild.
        from setiastro.saspro.shortcuts import DraggableToolBar
        for tb in self.findChildren(DraggableToolBar):
            try:
                self.removeToolBar(tb)
                tb.deleteLater()
            except Exception:
                pass

        # Build fresh
        self._init_toolbar()

        # Optional: ensure Hidden stays hidden
        tb_hidden = self._hidden_toolbar()
        if tb_hidden:
            tb_hidden.setVisible(False)


    def _restore_toolbar_order(self, tb, settings_key: str):
        """
        Restore toolbar action order from QSettings, using command_id/objectName.
        Unknown actions and separators keep their relative order at the end.
        """
        if not hasattr(self, "settings"):
            return

        order = self.settings.value(settings_key, None)
        if not order:
            return

        # QSettings may return QVariantList or str; normalize to Python list[str]
        if isinstance(order, str):
            # if you ever decide to JSON-encode, you could json.loads here
            order = [order]
        try:
            order_list = list(order)
        except Exception:
            return

        actions = list(tb.actions())

        def _cid(act):
            return act.property("command_id") or act.objectName() or ""

        rank = {str(cid): i for i, cid in enumerate(order_list)}
        big = len(rank) + len(actions) + 10

        indexed = list(enumerate(actions))
        indexed.sort(
            key=lambda pair: (
                rank.get(str(_cid(pair[1])), big),
                pair[0],
            )
        )

        tb.clear()
        for _, act in indexed:
            tb.addAction(act)

    def _restore_toolbar_memberships(self):
        """
        Restore which toolbar each action belongs to, based on Toolbar/Assignments.

        We:
          - Read JSON {command_id: settings_key}.
          - Collect all DraggableToolBar instances and their settings keys.
          - Collect all QActions by command_id/objectName.
          - Move each assigned action to its target toolbar.
          - Re-apply per-toolbar ordering via _restore_toolbar_order.
        """
        if not hasattr(self, "settings"):
            return

        try:
            raw = self.settings.value("Toolbar/Assignments", "", type=str) or ""
        except Exception:
            return

        try:
            mapping = json.loads(raw) if raw else {}
        except Exception:
            return

        if not mapping:
            return

        # Gather all DraggableToolBar instances
        from setiastro.saspro.shortcuts import DraggableToolBar
        toolbars: list[DraggableToolBar] = [
            tb for tb in self.findChildren(DraggableToolBar)
        ]

        tb_by_key: dict[str, DraggableToolBar] = {}
        for tb in toolbars:
            key = getattr(tb, "_settings_key", None)
            if key:
                tb_by_key[str(key)] = tb

        if not tb_by_key:
            return

        # Map command_id → QAction
        from PyQt6.QtGui import QAction
        acts_by_id: dict[str, QAction] = {}
        for act in self.findChildren(QAction):
            cid = act.property("command_id") or act.objectName()
            if cid:
                acts_by_id[str(cid)] = act

        # Move actions to their assigned toolbars
        for cid, key in mapping.items():
            act = acts_by_id.get(str(cid))
            tb  = tb_by_key.get(str(key))
            if not act or not tb:
                continue

            # Remove from any toolbar that currently contains it
            for t in toolbars:
                if act in t.actions():
                    t.removeAction(act)
            # Add to the desired toolbar
            tb.addAction(act)

        # Re-apply per-toolbar order now that memberships are correct
        for tb in toolbars:
            key = getattr(tb, "_settings_key", None)
            if key:
                self._restore_toolbar_order(tb, str(key))

    def _hidden_toolbar(self):
        from setiastro.saspro.shortcuts import DraggableToolBar
        for tb in self.findChildren(DraggableToolBar):
            if getattr(tb, "_settings_key", None) == "Toolbar/Hidden" or tb.objectName() == "Hidden":
                return tb
        return None

    def _toolbar_by_settings_key(self, key: str):
        from setiastro.saspro.shortcuts import DraggableToolBar
        for tb in self.findChildren(DraggableToolBar):
            if getattr(tb, "_settings_key", None) == key:
                return tb
        return None

    def _action_cid(self, act):
        return str(act.property("command_id") or act.objectName() or "")

    def _hide_action_to_hidden_toolbar(self, act):
        """
        Move action to the Hidden toolbar, remembering where it came from.
        """
        tb_hidden = self._hidden_toolbar()
        if not tb_hidden:
            return

        cid = self._action_cid(act)
        if not cid:
            return

        # Find current toolbar (if any) and remember it
        cur_tb = self._toolbar_containing_action(act)
        if cur_tb and getattr(cur_tb, "_settings_key", None) and cur_tb is not tb_hidden:
            prev_key = str(cur_tb._settings_key)

            # Persist "previous toolbar" so unhide can restore
            try:
                raw = self.settings.value("Toolbar/HiddenPrev", "", type=str) or ""
            except Exception:
                raw = ""
            try:
                prev = json.loads(raw) if raw else {}
            except Exception:
                prev = {}
            prev[cid] = prev_key
            self.settings.setValue("Toolbar/HiddenPrev", json.dumps(prev))

        # Remove from all toolbars, add to hidden
        from setiastro.saspro.shortcuts import DraggableToolBar
        for t in self.findChildren(DraggableToolBar):
            if act in t.actions():
                t.removeAction(act)

        tb_hidden.addAction(act)

        # Update assignment mapping so restore works on next launch
        tb_hidden._update_assignment_for_action(act)

        # Persist ordering (optional but nice)
        try:
            tb_hidden._persist_order()
        except Exception:
            pass
        if cur_tb:
            try:
                cur_tb._persist_order()
            except Exception:
                pass

    def _unhide_action_from_hidden_toolbar(self, act):
        """
        Move action out of Hidden toolbar back to its remembered toolbar.
        Falls back to Toolbar/View if missing.
        """
        tb_hidden = self._hidden_toolbar()
        if not tb_hidden:
            return

        cid = self._action_cid(act)
        if not cid:
            return

        # Load prev mapping
        try:
            raw = self.settings.value("Toolbar/HiddenPrev", "", type=str) or ""
        except Exception:
            raw = ""
        try:
            prev = json.loads(raw) if raw else {}
        except Exception:
            prev = {}

        target_key = prev.get(cid) or "Toolbar/View"
        tb_target = self._toolbar_by_settings_key(target_key) or self._toolbar_by_settings_key("Toolbar/View")
        if not tb_target:
            return

        tb_hidden.removeAction(act)
        tb_target.addAction(act)

        # Update assignment mapping
        tb_target._update_assignment_for_action(act)

        # Cleanup prev mapping (optional)
        if cid in prev:
            prev.pop(cid, None)
            self.settings.setValue("Toolbar/HiddenPrev", json.dumps(prev))

        try:
            tb_target._persist_order()
            tb_hidden._persist_order()
        except Exception:
            pass

    def _migrate_old_hidden_sets_to_hidden_toolbar(self):
        """
        One-time migration:
        Old system: per-toolbar QSettings lists at "<ToolbarKey>/Hidden" storing action IDs
        New system: move those actions into the dedicated Hidden toolbar (Toolbar/Hidden)
                    and persist membership via Toolbar/Assignments.
        """
        if not hasattr(self, "settings"):
            return

        # Run once
        if self.settings.value("Toolbar/HiddenMigrationDone", False, type=bool):
            return

        tb_hidden = self._hidden_toolbar()
        if not tb_hidden:
            return

        from setiastro.saspro.shortcuts import DraggableToolBar

        # If Hidden toolbar already has actions, assume user is already on new system
        # but we still can migrate any remaining old keys.
        # (We won't early-return.)

        for tb in self.findChildren(DraggableToolBar):
            k = getattr(tb, "_settings_key", None)
            if not k or str(k) == "Toolbar/Hidden":
                continue

            # Old storage key
            old_key = f"{k}/Hidden"

            # Read old hidden list (Qt backend may return list OR str)
            try:
                raw = self.settings.value(old_key, [], type=list)
            except Exception:
                raw = self.settings.value(old_key, [])  # fallback

            if not raw:
                continue

            if isinstance(raw, str):
                raw_list = [raw]
            else:
                try:
                    raw_list = list(raw)
                except Exception:
                    raw_list = []

            hidden_ids = {str(x) for x in raw_list if str(x).strip()}
            if not hidden_ids:
                # Clear junk so we don’t keep trying
                self.settings.setValue(old_key, [])
                continue

            # Move matching actions into Hidden toolbar
            for act in list(tb.actions()):
                if act is None or act.isSeparator():
                    continue
                cid = str(act.property("command_id") or act.objectName() or "")
                if cid and cid in hidden_ids:
                    self._hide_action_to_hidden_toolbar(act)

            # Clear old storage so you don't keep re-migrating
            self.settings.setValue(old_key, [])

        # Mark migration complete
        self.settings.setValue("Toolbar/HiddenMigrationDone", True)


    def update_undo_redo_action_labels(self):
        if not hasattr(self, "act_undo"):  # not built yet
            return

        # Always compute against the history root
        doc = self._active_history_doc()

        if doc:
            try:
                can_u = bool(doc.can_undo()) if hasattr(doc, "can_undo") else False
            except Exception:
                can_u = False
            try:
                can_r = bool(doc.can_redo()) if hasattr(doc, "can_redo") else False
            except Exception:
                can_r = False

            undo_name = None
            redo_name = None
            try:
                undo_name = doc.last_undo_name() if hasattr(doc, "last_undo_name") else None
            except Exception:
                pass
            try:
                redo_name = doc.last_redo_name() if hasattr(doc, "last_redo_name") else None
            except Exception:
                pass

            self.act_undo.setText(f"Undo {undo_name}" if (can_u and undo_name) else "Undo")
            self.act_redo.setText(f"Redo {redo_name}" if (can_r and redo_name) else "Redo")

            self.act_undo.setToolTip("Nothing to undo" if not can_u else (f"Undo: {undo_name}" if undo_name else "Undo last action"))
            self.act_redo.setToolTip("Nothing to redo" if not can_r else (f"Redo: {redo_name}" if redo_name else "Redo last action"))

            self.act_undo.setStatusTip(self.act_undo.toolTip())
            self.act_redo.setStatusTip(self.act_redo.toolTip())

            self.act_undo.setEnabled(can_u)
            self.act_redo.setEnabled(can_r)
        else:
            # No active doc
            for a, tip in ((self.act_undo, "Nothing to undo"),
                           (self.act_redo, "Nothing to redo")):
                # Normalize label to plain "Undo"/"Redo"
                base = "Undo" if "Undo" in a.text() else ("Redo" if "Redo" in a.text() else a.text())
                a.setText(base)
                a.setToolTip(tip)
                a.setStatusTip(tip)
                a.setEnabled(False)


    def _sync_link_action_state(self):
        g = self._current_group_of_active()
        self.act_link_group.blockSignals(True)
        try:
            self.act_link_group.setChecked(bool(g))
            self.act_link_group.setText(f"Link Pan/Zoom{'' if not g else f' ({g})'}")
            try:
                if getattr(self, "_link_btn", None):
                    self._link_btn.setText(self.act_link_group.text())
            except Exception:
                pass
        finally:
            self.act_link_group.blockSignals(False)

    def _undo_active(self):
        doc = self._active_history_doc()
        if doc and getattr(doc, "can_undo", lambda: False)():
            # Ensure the correct view is active so Qt routes shortcut focus correctly
            sw = self._subwindow_for_history_doc(doc)
            if sw is not None:
                try:
                    self.mdi.setActiveSubWindow(sw)
                except Exception:
                    pass
            name = doc.undo()
            if name:
                self._log(f"Undo: {name}")
        # Defer label refresh to end of event loop (lets views repaint first)
        QTimer.singleShot(0, self.update_undo_redo_action_labels)

    def _redo_active(self):
        doc = self._active_history_doc()
        if doc and getattr(doc, "can_redo", lambda: False)():
            sw = self._subwindow_for_history_doc(doc)
            if sw is not None:
                try:
                    self.mdi.setActiveSubWindow(sw)
                except Exception:
                    pass
            name = doc.redo()
            if name:
                self._log(f"Redo: {name}")
        QTimer.singleShot(0, self.update_undo_redo_action_labels)

    def _refresh_mask_action_states(self):

        active_doc = self._active_doc()

        can_apply = bool(active_doc and self._list_candidate_mask_sources(exclude_doc=active_doc))
        can_remove = bool(active_doc and getattr(active_doc, "active_mask_id", None))

        if hasattr(self, "act_apply_mask"):
            self.act_apply_mask.setEnabled(can_apply)
        if hasattr(self, "act_remove_mask"):
            self.act_remove_mask.setEnabled(can_remove)

        # NEW: enable/disable Invert
        if hasattr(self, "act_invert_mask"):
            self.act_invert_mask.setEnabled(can_remove)

        vw = self._active_view()
        overlay_on = bool(getattr(vw, "show_mask_overlay", False)) if vw else False
        has_mask   = bool(active_doc and getattr(active_doc, "active_mask_id", None))

        if hasattr(self, "act_show_mask"):
            self.act_show_mask.setEnabled(has_mask and not overlay_on)
        if hasattr(self, "act_hide_mask"):
            self.act_hide_mask.setEnabled(has_mask and overlay_on)

    def _style_toggle_toolbutton(self, btn: QToolButton):
        # Make sure the action visually shows "on" state
        btn.setCheckable(True)  # safe even if already
        btn.setStyleSheet("""
            QToolButton { color: #dcdcdc; }
            QToolButton:checked { color: #DAA520; font-weight: 600; }
        """)
