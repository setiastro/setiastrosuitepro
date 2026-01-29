# pro/gui/mixins/menu_mixin.py
"""
Menu management mixin for AstroSuiteProMainWindow.
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenu, QToolButton, QWidgetAction
from PyQt6.QtCore import Qt

if TYPE_CHECKING:
    pass


class MenuMixin:
    """
    Mixin for menu management.
    
    Provides methods for creating and managing menus in the main window.
    """
    
    def _init_menubar(self):
        """Initialize the main menu bar."""
        # This method will be implemented as part of the main window
        # For now, this is a placeholder showing the mixin pattern
        pass

    def _show_statistics(self):
        from setiastro.saspro.gui.statistics_dialog import StatisticsDialog
        dlg = StatisticsDialog(self)
        dlg.exec()

    def _hook_tool_stats(self, menus):
        if not hasattr(self, "_on_tool_triggered"):
            return
        
        seen = set()
        for menu in menus:
            for action in self._iter_menu_actions(menu):
                if action in seen: continue
                seen.add(action)
                if action.isSeparator(): continue
                
                try:
                    action.triggered.connect(self._on_tool_triggered)
                except Exception:
                    pass

    
    def _rebuild_recent_menus(self):
        """Rebuild the recent files and projects menus."""
        if not hasattr(self, "_recent_image_paths") or not hasattr(self, "_recent_project_paths"):
            return
        
        # Recent images
        if hasattr(self, "_recent_images_menu"):
            self._recent_images_menu.clear()
            for path in self._recent_image_paths[:10]:  # Show top 10
                action = self._recent_images_menu.addAction(path)
                action.triggered.connect(lambda checked=False, p=path: self._open_recent_image(p))
        
        # Recent projects
        if hasattr(self, "_recent_projects_menu"):
            self._recent_projects_menu.clear()
            for path in self._recent_project_paths[:10]:
                action = self._recent_projects_menu.addAction(path)
                action.triggered.connect(lambda checked=False, p=path: self._open_recent_project(p))
    
    def _strip_menu_text(self, text: str) -> str:
        """Remove decorations and mnemonics from menu text."""
        # Remove ampersands (mnemonics)
        text = text.replace("&", "")
        # Could add more stripping logic here
        return text
    
    def _iter_menu_actions(self, menu: QMenu):
        """
        Iterate over all actions in a menu recursively.
        
        Args:
            menu: The menu to iterate
            
        Yields:
            QAction objects
        """
        for action in menu.actions():
            if action.menu():
                yield from self._iter_menu_actions(action.menu())
            else:
                yield action
# Extracted MENU methods

    def _init_menubar(self):
        mb = self.menuBar()

        # File
        m_file = mb.addMenu(self.tr("&File"))
        m_file.addAction(self.act_open)
        m_file.addSeparator()
        m_file.addAction(self.act_save)
        m_file.addSeparator()
        m_file.addAction(self.act_clear_views) 
        m_file.addSeparator()
        m_file.addAction(self.act_project_new)
        m_file.addAction(self.act_project_save)
        m_file.addAction(self.act_project_load)
        # --- Recent submenus ----------------------------------------
        m_file.addSeparator()
        self.m_recent_images_menu = m_file.addMenu(self.tr("Open Recent Images"))
        self.m_recent_projects_menu = m_file.addMenu(self.tr("Open Recent Projects"))

        m_file.addSeparator()
        m_file.addAction(self.act_exit)

        # Populate from QSettings
        self._rebuild_recent_menus()

        # Edit (with icons)
        m_edit = mb.addMenu(self.tr("&Edit"))
        m_edit.addAction(self.act_undo)
        m_edit.addAction(self.act_redo)
        m_edit.addSeparator()
        m_edit.addAction(self.act_mono_to_rgb)
        m_edit.addAction(self.act_swap_rb)  


        # Functions
        m_fn = mb.addMenu(self.tr("&Functions"))

        m_fn.addAction(self.act_abe)
        m_fn.addAction(self.act_add_stars)
        m_fn.addAction(self.act_background_neutral)
        m_fn.addAction(self.act_blemish)
        m_fn.addAction(self.act_clahe)
        m_fn.addAction(self.act_clone_stamp) 
        m_fn.addAction(self.act_convo)
        m_fn.addAction(self.act_crop)
        m_fn.addAction(self.act_curves)
        m_fn.addAction(self.act_extract_luma)
        m_fn.addAction(self.act_graxpert)
        m_fn.addAction(self.act_ghs)
        m_fn.addAction(self.act_halobgon)
        m_fn.addAction(self.act_histogram)
        m_fn.addAction(self.act_linear_fit)
        m_fn.addAction(self.act_morphology)
        m_fn.addAction(self.act_pedestal)
        m_fn.addAction(self.act_pixelmath)
        m_fn.addAction(self.act_recombine_luma)
        m_fn.addAction(self.act_remove_green)
        m_fn.addAction(self.act_remove_stars)
        m_fn.addAction(self.act_rgb_combine)
        m_fn.addAction(self.act_rgb_extract)
        m_fn.addAction(self.act_sfcc)
        m_fn.addAction(self.act_signature)
        m_fn.addAction(self.act_star_stretch)
        m_fn.addAction(self.act_stat_stretch)
        m_fn.addAction(self.act_texture_clarity)
        m_fn.addAction(self.act_wavescale_de)
        m_fn.addAction(self.act_wavescale_hdr)
        m_fn.addAction(self.act_white_balance)


        mCosmic = mb.addMenu(self.tr("&Smart Tools"))
        mCosmic.addAction(self.actAberrationAI)
        mCosmic.addAction(self.actCosmicUI)
        mCosmic.addAction(self.actCosmicSat)
        mCosmic.addAction(self.act_graxpert)
        mCosmic.addAction(self.act_remove_stars)

        m_tools = mb.addMenu(self.tr("&Tools"))

        m_tools.addAction(self.act_blink)
        m_tools.addAction(self.act_contsub)
        m_tools.addAction(self.act_freqsep)
        m_tools.addAction(self.act_image_combine)
        m_tools.addAction(self.act_multiscale_decomp)
        m_tools.addAction(self.act_narrowband_normalization)
        m_tools.addAction(self.act_nbtorgb)
        m_tools.addAction(self.act_ppp)
        
        m_tools.addAction(self.act_selective_color)
        m_tools.addSeparator()
        m_tools.addAction(self.act_view_bundles) 
        m_tools.addAction(self.act_function_bundles)

        m_geom = mb.addMenu(self.tr("&Geometry"))
        m_geom.addAction(self.act_geom_invert)
        m_geom.addSeparator()
        m_geom.addAction(self.act_geom_flip_h)
        m_geom.addAction(self.act_geom_flip_v)
        m_geom.addSeparator()
        m_geom.addAction(self.act_geom_rot_cw)
        m_geom.addAction(self.act_geom_rot_ccw)
        m_geom.addAction(self.act_geom_rot_180)   
        m_geom.addAction(self.act_geom_rot_any) 
        m_geom.addSeparator()
        m_geom.addAction(self.act_geom_rescale)
        m_geom.addSeparator()
        m_geom.addAction(self.act_debayer)

        m_star = mb.addMenu(self.tr("&Star Stuff"))

        m_star.addAction(self.act_astrospike)
        m_star.addAction(self.act_exo_detector)

        m_star.addAction(self.act_image_peeker)
        m_star.addAction(self.act_isophote)
        m_star.addAction(self.act_live_stacking)
        m_star.addAction(self.act_mosaic_master)
        m_star.addAction(self.act_planet_projection)
        m_star.addAction(self.act_planetary_stacker)
        m_star.addAction(self.act_plate_solve)
        m_star.addAction(self.act_psf_viewer)
        m_star.addAction(self.act_rgb_align)
        m_star.addAction(self.act_star_align)
        m_star.addAction(self.act_star_register)
        m_star.addAction(self.act_star_spikes)
        m_star.addAction(self.act_stacking_suite)
        m_star.addAction(self.act_supernova_hunter)

        m_masks = mb.addMenu(self.tr("&Masks"))
        m_masks.addAction(self.act_create_mask)
        m_masks.addSeparator()
        m_masks.addAction(self.act_apply_mask)
        m_masks.addAction(self.act_remove_mask)
        m_masks.addSeparator()
        m_masks.addAction(self.act_show_mask)
        m_masks.addAction(self.act_hide_mask)
        m_masks.addSeparator()
        m_masks.addAction(self.act_invert_mask)

        m_wim = mb.addMenu(self.tr("&What's In My..."))
        m_wim.addAction(self.act_whats_in_my_sky)
        m_wim.addAction(self.act_wimi)
        m_wim.addAction(self.act_finder_chart) 

        m_scripts = mb.addMenu(self.tr("&Scripts"))
        self.menu_scripts = m_scripts
        self.scriptman.rebuild_menu(m_scripts)


        m_header = mb.addMenu(self.tr("&Header Mods && Misc"))
        m_header.addAction(self.act_acv_exporter)        
        m_header.addAction(self.act_astrobin_exporter)
        m_header.addAction(self.act_batch_convert)
        m_header.addAction(self.act_batch_renamer)
        m_header.addAction(self.act_copy_astrometry)
        m_header.addAction(self.act_fits_batch_modifier)
        m_header.addAction(self.act_fits_modifier)


        m_hist = mb.addMenu(self.tr("&History"))
        m_hist.addAction(self.act_history_explorer)

        m_short = mb.addMenu(self.tr("&Shortcuts"))

        act_cheats = QAction(self.tr("Keyboard Shortcut Cheat Sheet..."), self)
        act_cheats.triggered.connect(self._show_cheat_sheet)
        m_short.addAction(act_cheats)

        # act_save_sc = QAction("Save Shortcuts Now", self, triggered=self.shortcuts.save_shortcuts)
        # Keep it if you like, but add explicit export/import:
        act_export_sc = QAction(self.tr("Export Shortcuts..."), self, triggered=self._export_shortcuts_dialog)
        act_import_sc = QAction(self.tr("Import Shortcuts..."), self, triggered=self._import_shortcuts_dialog)
        act_clear_sc  = QAction(self.tr("Clear All Shortcuts"), self, triggered=self.shortcuts.clear)

        m_short.addAction(act_export_sc)
        m_short.addAction(act_import_sc)
        m_short.addSeparator()
        # m_short.addAction(act_save_sc)   # optional: keep
        m_short.addAction(act_clear_sc)

        m_view = mb.addMenu(self.tr("&View"))
        m_view.addAction(self.act_cascade)
        m_view.addAction(self.act_tile)
        m_view.addAction(self.act_tile_vert)
        m_view.addAction(self.act_tile_horiz)
        m_view.addAction(self.act_tile_grid)        
        m_view.addSeparator()

        # NEW: Minimize All Views
        self.act_minimize_all_views = QAction(self.tr("Minimize All Views"), self)
        self.act_minimize_all_views.setShortcut(QKeySequence("Ctrl+Shift+M"))
        self.act_minimize_all_views.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_minimize_all_views.triggered.connect(self._minimize_all_views)
        m_view.addAction(self.act_minimize_all_views)

        m_view.addSeparator()

        # a button that shows current group & opens a drop-down
        self._link_btn = QToolButton(self)
        self._link_btn.setDefaultAction(self.act_link_group)  # text/checked state mirrors the action
        self._link_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

        link_menu = QMenu(self._link_btn)
        a_none = link_menu.addAction(self.tr("None"))
        a_A = link_menu.addAction(self.tr("Group A"))
        a_B = link_menu.addAction(self.tr("Group B"))
        a_C = link_menu.addAction(self.tr("Group C"))
        a_D = link_menu.addAction(self.tr("Group D"))
        self._link_btn.setMenu(link_menu)

        a_none.setCheckable(True)
        a_A.setCheckable(True)
        a_B.setCheckable(True)
        a_C.setCheckable(True)
        a_D.setCheckable(True)

        def _sync_menu_checks():
            g = self._current_group_of_active()
            a_none.setChecked(g is None)
            a_A.setChecked(g == "A")
            a_B.setChecked(g == "B")
            a_C.setChecked(g == "C")
            a_D.setChecked(g == "D")

        link_menu.aboutToShow.connect(_sync_menu_checks)

        # hook the menu choices to your helpers
        a_none.triggered.connect(lambda: self._set_group_for_active(None))
        a_A.triggered.connect(lambda: self._set_group_for_active("A"))
        a_B.triggered.connect(lambda: self._set_group_for_active("B"))
        a_C.triggered.connect(lambda: self._set_group_for_active("C"))
        a_D.triggered.connect(lambda: self._set_group_for_active("D"))

        # wrap it so it can live inside the menu
        wa = QWidgetAction(self)
        wa.setDefaultWidget(self._link_btn)
        m_view.addAction(wa)

        # first-time sync of label/checked state
        self._sync_link_action_state()

        m_settings = mb.addMenu(self.tr("&Settings"))
        m_settings.addAction(self.tr("Preferences..."), self._open_settings)
        m_settings.addSeparator()
        m_settings.addAction(self.tr("Benchmark..."), self._open_benchmark)

        m_about = mb.addMenu(self.tr("&About"))
        m_about.addAction(self.act_docs)  
        m_about.addSeparator()
        m_about.addAction(self.tr("About..."), self._about)
        m_about.addAction(self.act_check_updates)


        m_about.addSeparator()
        m_about.addAction(self.tr("Statistics..."), self._show_statistics)

        # Connect tool stats
        self._hook_tool_stats([m_fn, m_tools, mCosmic, m_geom, m_star, m_masks, m_header, m_scripts])

        # initialize enabled state + names
        self.update_undo_redo_action_labels()

    def _init_tools_menu(self):
        tools = self.menuBar().addMenu("Tools")
        act_batch = QAction("Batch Convert...", self)
        act_batch.triggered.connect(self._open_batch_convert)
        tools.addAction(act_batch)



    #------------Tools-----------------

    def _rebuild_recent_menus(self):
        """Rebuild both 'Open Recent' submenus."""
        # Menus might not exist yet if called very early
        if not hasattr(self, "m_recent_images_menu") or not hasattr(self, "m_recent_projects_menu"):
            return

        # ---- Images ----------------------------------------
        self.m_recent_images_menu.clear()
        if not self._recent_image_paths:
            act = self.m_recent_images_menu.addAction(self.tr("No recent images"))
            act.setEnabled(False)
        else:
            for path in self._recent_image_paths:
                label = os.path.basename(path) or path
                act = self.m_recent_images_menu.addAction(label)
                act.setToolTip(path)
                act.triggered.connect(
                    lambda checked=False, p=path: self._open_recent_image(p)
                )
            self.m_recent_images_menu.addSeparator()
            clear_act = self.m_recent_images_menu.addAction(self.tr("Clear List"))
            clear_act.triggered.connect(self._clear_recent_images)

        # ---- Projects ----------------------------------------
        self.m_recent_projects_menu.clear()
        if not self._recent_project_paths:
            act = self.m_recent_projects_menu.addAction(self.tr("No recent projects"))
            act.setEnabled(False)
        else:
            for path in self._recent_project_paths:
                label = os.path.basename(path) or path
                act = self.m_recent_projects_menu.addAction(label)
                act.setToolTip(path)
                act.triggered.connect(
                    lambda checked=False, p=path: self._open_recent_project(p)
                )
            self.m_recent_projects_menu.addSeparator()
            clear_act = self.m_recent_projects_menu.addAction(self.tr("Clear List"))
            clear_act.triggered.connect(self._clear_recent_projects)

    def _iter_menu_actions(self, menu: QMenu):
        """Depth-first iterator over all actions inside a QMenu tree."""
        for act in menu.actions():
            yield act
            sub = act.menu()
            if sub is not None:
                yield from self._iter_menu_actions(sub)

    def _minimize_all_views(self):
        mdi = getattr(self, "mdi", None)
        if mdi is None:
            return

        try:
            for sw in mdi.subWindowList():
                try:
                    if not sw.isVisible():
                        continue
                    # Minimize each MDI child
                    sw.showMinimized()
                except Exception:
                    pass
        except Exception:
            pass
