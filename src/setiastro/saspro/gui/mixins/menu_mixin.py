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
        m_file.addAction(self.act_create_table)
        m_file.addSeparator()
        m_file.addAction(self.act_save)  # still works as top-level "Save As..."
        m_save_as = m_file.addMenu(self.tr("Save As Format"))
        m_save_as.addAction(self.act_save_fits)
        m_save_as.addAction(self.act_save_xisf)
        m_save_as.addAction(self.act_save_tiff)
        m_save_as.addAction(self.act_save_png)
        m_save_as.addAction(self.act_save_jpeg)
        m_file.addAction(self.act_export_fits_bundle)
        m_file.addAction(self.act_checkpoint_save) 
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
        m_edit.addAction(self.act_copy)
        m_edit.addAction(self.act_paste)
        m_edit.addSeparator()        
        m_edit.addAction(self.act_mono_to_rgb)
        m_edit.addAction(self.act_rgb_to_mono)
        m_edit.addAction(self.act_swap_rb)  

        m_display = mb.addMenu(self.tr("&Display"))
        m_display.addAction(self.act_autostretch)
        m_display.addAction(self.act_hardstretch)
        m_display.addAction(self.act_autostretch_continuous)
        m_display.addAction(self.act_stretch_linked)
        m_display.addAction(self.act_display_target)
        m_display.addAction(self.act_display_sigma)
        m_display.addAction(self.act_bake_display_stretch)
        m_display.addSeparator()
        m_display.addAction(self.act_zoom_in)
        m_display.addAction(self.act_zoom_out)
        m_display.addAction(self.act_zoom_1_1)
        m_display.addAction(self.act_zoom_fit)
        m_display.addAction(self.act_auto_fit_resize)

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
        m_fn.addAction(self.act_hist_transform)
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
        mCosmic.addAction(self.actSyQonTools) 
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
        m_tools.addAction(self.act_selective_lum) 
        m_tools.addAction(self.act_magnitude)
        m_tools.addAction(self.act_snr)
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
        m_geom.addAction(self.act_geom_resize_canvas)
        m_geom.addSeparator()
        m_geom.addAction(self.act_debayer)

        m_star = mb.addMenu(self.tr("&Star Stuff"))

        m_star.addAction(self.act_astrospike)
        m_star.addAction(self.act_exo_detector)
        m_star.addAction(self.act_dither_analysis)
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
        act_icon_sheet = QAction(self.tr("Icon Cheat Sheet..."), self)
        act_icon_sheet.triggered.connect(self._show_icon_cheat_sheet)
        m_short.addAction(act_icon_sheet)
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

        m_view.addSeparator()
        m_view.addAction(self.act_open_panel_host)
        m_view.addAction(self.act_send_panels_to_host)
        m_view.addAction(self.act_return_panels_to_main)
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

        m_workflow = mb.addMenu(self.tr("&Workflows"))
        m_workflow.addAction(self.act_workflows)

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
        m_about.addAction(self.act_welcome)

        m_about.addSeparator()
        m_about.addAction(self.tr("Statistics..."), self._show_statistics)
        m_about.addSeparator()
        m_about.addAction(self.act_bored)
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

    def _show_icon_cheat_sheet(self):
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QScrollArea,
            QWidget, QGridLayout, QLabel, QPushButton, QSizePolicy, QFileDialog
        )
        from PyQt6.QtCore import Qt, QSize
        from PyQt6.QtGui import QPixmap
        if hasattr(self, "_icon_cheat_sheet_dlg") and self._icon_cheat_sheet_dlg is not None:
            try:
                self._icon_cheat_sheet_dlg.raise_()
                self._icon_cheat_sheet_dlg.activateWindow()
                return
            except Exception:
                self._icon_cheat_sheet_dlg = None
        dlg = QDialog(self)
        dlg.setWindowTitle("Icon Cheat Sheet")
        dlg.setMinimumSize(900, 650)
        dlg.setStyleSheet("""
            QDialog { background: #1a1a2e; color: #eaeaea; }
            QLineEdit {
                background: #16213e; color: #eaeaea;
                border: 1px solid #0f3460; border-radius: 4px;
                padding: 6px 10px; font-size: 12px;
            }
            QScrollArea { border: none; background: #1a1a2e; }
            QPushButton {
                background: #0f3460; color: #eaeaea;
                border-radius: 4px; padding: 6px 16px; font-size: 11px;
            }
            QPushButton:hover { background: #e94560; }
        """)

        root = QVBoxLayout(dlg)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        search_row = QHBoxLayout()
        search = QLineEdit()
        search.setPlaceholderText("Search by name, tooltip, or category…")
        search.setFixedHeight(32)
        search_row.addWidget(search)

        btn_pdf = QPushButton("Export PDF…")
        btn_pdf.setFixedHeight(32)
        search_row.addWidget(btn_pdf)
        root.addLayout(search_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setStyleSheet("background: #1a1a2e;")
        grid = QGridLayout(container)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(4)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        ICON_SIZE = 24
        CARD_W = 260

        def _card_style():
            return (
                "QWidget { background: #16213e; border: 1px solid #0f3460; "
                "border-radius: 6px; }"
            )

        MENU_GROUPS = [
            ("File", [
                self.act_open, self.act_save, self.act_checkpoint_save,
                self.act_project_new, self.act_project_save, self.act_project_load,
            ]),
            ("Edit", [
                self.act_undo, self.act_redo, self.act_copy, self.act_paste,
                self.act_mono_to_rgb, self.act_rgb_to_mono, self.act_swap_rb,
            ]),
            ("Display", [
                self.act_autostretch, self.act_hardstretch, self.act_bake_display_stretch,
                self.act_zoom_in, self.act_zoom_out, self.act_zoom_1_1, self.act_zoom_fit,
            ]),
            ("Functions", [
                self.act_abe, self.act_graxpert, self.act_background_neutral,
                self.act_stat_stretch, self.act_star_stretch, self.act_ghs,
                self.act_curves, self.act_hist_transform, self.act_histogram,
                self.act_white_balance, self.act_sfcc, self.act_remove_green,
                self.act_linear_fit, self.act_remove_stars, self.act_add_stars,
                self.act_halobgon, self.act_convo, self.act_clahe,
                self.act_texture_clarity, self.act_wavescale_hdr, self.act_wavescale_de,
                self.act_morphology, self.act_extract_luma, self.act_recombine_luma,
                self.act_rgb_extract, self.act_rgb_combine, self.act_pedestal,
                self.act_blemish, self.act_clone_stamp, self.act_crop,
                self.act_pixelmath, self.act_signature, self.act_image_combine,
            ]),
            ("Smart Tools", [
                self.actAberrationAI, self.actCosmicUI, self.actCosmicSat,
                self.actSyQonTools,
            ]),
            ("Tools", [
                self.act_blink, self.act_ppp, self.act_nbtorgb,
                self.act_narrowband_normalization, self.act_selective_color,
                self.act_selective_lum, self.act_freqsep, self.act_multiscale_decomp,
                self.act_contsub, self.act_magnitude, self.act_snr,
                self.act_view_bundles, self.act_function_bundles,
            ]),
            ("Geometry", [
                self.act_geom_invert, self.act_geom_flip_h, self.act_geom_flip_v,
                self.act_geom_rot_cw, self.act_geom_rot_ccw, self.act_geom_rot_180,
                self.act_geom_rot_any, self.act_geom_rescale,
                self.act_geom_resize_canvas, self.act_debayer,
            ]),
            ("Star Stuff", [
                self.act_stacking_suite, self.act_live_stacking, self.act_planetary_stacker,
                self.act_star_align, self.act_star_register, self.act_rgb_align,
                self.act_mosaic_master, self.act_plate_solve, self.act_dither_analysis,
                self.act_image_peeker, self.act_psf_viewer, self.act_exo_detector,
                self.act_isophote, self.act_supernova_hunter,
                self.act_planet_projection, self.act_star_spikes, self.act_astrospike,
            ]),
            ("Masks", [
                self.act_create_mask, self.act_apply_mask, self.act_remove_mask,
                self.act_show_mask, self.act_hide_mask, self.act_invert_mask,
            ]),
            ("What's In My…", [
                self.act_whats_in_my_sky, self.act_wimi, self.act_finder_chart,
            ]),
            ("Header & Misc", [
                self.act_fits_modifier, self.act_fits_batch_modifier,
                self.act_batch_renamer, self.act_batch_convert,
                self.act_astrobin_exporter, self.act_acv_exporter,
                self.act_copy_astrometry,
            ]),
        ]

        def _entry(act):
            if act is None:
                return None
            title = act.text().replace("&", "").strip()
            if not title:
                return None
            icon = act.icon()
            pm = None
            if icon and not icon.isNull():
                pm = icon.pixmap(QSize(ICON_SIZE, ICON_SIZE))
                if pm.isNull():
                    pm = None
            tip = act.statusTip() or act.toolTip() or ""
            sc  = act.shortcut().toString() if act.shortcut() else ""
            return (pm, title, tip, sc)

        seen = set()
        entries = []  # (group_name, pm, title, tip, sc)
        for group_name, acts in MENU_GROUPS:
            group_entries = []
            for act in acts:
                e = _entry(act)
                if e is None:
                    continue
                pm, title, tip, sc = e
                if title in seen:
                    continue
                seen.add(title)
                group_entries.append((group_name, pm, title, tip, sc))
            group_entries.sort(key=lambda e: e[2].lower())  # sort by title within group
            entries.extend(group_entries)

        card_widgets = []

        def _build_cards(filter_text=""):
            ft = filter_text.lower().strip()
            while grid.count():
                item = grid.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)
            card_widgets.clear()

            col_count = 3
            row = col = 0
            current_group = None

            for group_name, pm, title, tip, sc in entries:
                if ft and ft not in title.lower() and ft not in tip.lower() and ft not in group_name.lower():
                    continue

                # Group header — spans all columns
                if group_name != current_group:
                    current_group = group_name
                    if col > 0:
                        col = 0
                        row += 1
                    hdr = QLabel(group_name.upper())
                    hdr.setStyleSheet(
                        "font-size:10px;font-weight:700;color:#e94560;"
                        "background:transparent;border:none;"
                        "padding:8px 4px 4px 4px;letter-spacing:2px;"
                    )
                    grid.addWidget(hdr, row, 0, 1, col_count)
                    row += 1

                card = QWidget()
                card.setFixedWidth(CARD_W)
                card.setStyleSheet(_card_style())
                card.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

                cl = QHBoxLayout(card)
                cl.setContentsMargins(8, 6, 8, 6)
                cl.setSpacing(8)

                ico_lbl = QLabel()
                ico_lbl.setFixedSize(ICON_SIZE, ICON_SIZE)
                ico_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                if pm:
                    ico_lbl.setPixmap(pm)
                else:
                    ico_lbl.setText("·")
                    ico_lbl.setStyleSheet("color:#555;font-size:18px;")
                cl.addWidget(ico_lbl)

                text_col = QVBoxLayout()
                text_col.setSpacing(1)
                text_col.setContentsMargins(0, 0, 0, 0)

                title_lbl = QLabel(title)
                title_lbl.setStyleSheet(
                    "font-size:11px;font-weight:600;color:#eaeaea;"
                    "border:none;background:transparent;"
                )
                title_lbl.setWordWrap(False)
                text_col.addWidget(title_lbl)

                if tip:
                    tip_lbl = QLabel(tip)
                    tip_lbl.setStyleSheet(
                        "font-size:10px;color:#888;border:none;background:transparent;"
                    )
                    tip_lbl.setWordWrap(True)
                    text_col.addWidget(tip_lbl)

                if sc:
                    sc_lbl = QLabel(sc)
                    sc_lbl.setStyleSheet(
                        "font-size:9px;color:#e94560;border:none;background:transparent;"
                    )
                    text_col.addWidget(sc_lbl)

                cl.addLayout(text_col, 1)
                grid.addWidget(card, row, col)
                card_widgets.append(card)

                col += 1
                if col >= col_count:
                    col = 0
                    row += 1

        _build_cards()
        search.textChanged.connect(_build_cards)

        def _export_pdf():
            path, _ = QFileDialog.getSaveFileName(
                dlg, "Export Icon Cheat Sheet", "saspro_icons.pdf",
                "PDF Files (*.pdf)"
            )
            if not path:
                return
            try:
                from PyQt6.QtPrintSupport import QPrinter
                from PyQt6.QtGui import QPainter, QFont, QColor
                from PyQt6.QtCore import QRectF

                printer = QPrinter(QPrinter.PrinterMode.HighResolution)
                printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
                printer.setOutputFileName(path)
                from PyQt6.QtGui import QPageSize
                printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))

                painter = QPainter(printer)
                dpi = printer.resolution()
                page_rect = printer.pageRect(QPrinter.Unit.DevicePixel)
                pw = page_rect.width()
                ph = page_rect.height()

                margin    = dpi * 0.4
                col_count = 3
                cell_w    = (pw - margin * 2) / col_count
                cell_h    = dpi * 0.45
                icon_px   = int(dpi * 0.22)
                x_base    = margin
                y         = margin

                title_font   = QFont("Segoe UI", 8, QFont.Weight.Bold)
                tip_font     = QFont("Segoe UI", 6)
                sc_font      = QFont("Segoe UI", 6)
                hdr_font     = QFont("Segoe UI", 14, QFont.Weight.Bold)
                grp_font     = QFont("Segoe UI", 7, QFont.Weight.Bold)

                # Document title
                painter.setFont(hdr_font)
                painter.setPen(QColor("#e94560"))
                painter.drawText(
                    QRectF(x_base, y, pw - margin * 2, dpi * 0.35),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    "Seti Astro Suite Pro — Icon Cheat Sheet"
                )
                y += dpi * 0.45

                col = 0
                current_group = None
                ft = search.text().lower().strip()

                for group_name, pm, title, tip, sc in entries:
                    if ft and ft not in title.lower() and ft not in tip.lower() and ft not in group_name.lower():
                        continue

                    # Group header in PDF
                    if group_name != current_group:
                        current_group = group_name
                        if col > 0:
                            col = 0
                            y += cell_h + dpi * 0.04

                        if y + dpi * 0.25 > ph - margin:
                            printer.newPage()
                            y = margin

                        painter.setFont(grp_font)
                        painter.setPen(QColor("#e94560"))
                        painter.drawText(
                            QRectF(x_base, y, pw - margin * 2, dpi * 0.22),
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                            group_name.upper()
                        )
                        y += dpi * 0.25

                    if y + cell_h > ph - margin:
                        printer.newPage()
                        y = margin
                        # Reprint group header at top of new page
                        painter.setFont(grp_font)
                        painter.setPen(QColor("#e94560"))
                        painter.drawText(
                            QRectF(x_base, y, pw - margin * 2, dpi * 0.22),
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                            f"{group_name.upper()} (continued)"
                        )
                        y += dpi * 0.25

                    x = x_base + col * cell_w
                    cell_margin = dpi * 0.05

                    if pm:
                        scaled_pm = pm.scaled(
                            icon_px, icon_px,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                        painter.drawPixmap(
                            int(x + cell_margin),
                            int(y + (cell_h - icon_px) / 2),
                            scaled_pm
                        )

                    tx     = x + cell_margin + icon_px + dpi * 0.06
                    text_w = cell_w - cell_margin - icon_px - dpi * 0.1

                    painter.setFont(title_font)
                    painter.setPen(QColor("#111111"))
                    painter.drawText(
                        QRectF(tx, y + cell_margin, text_w, cell_h * 0.4),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        title
                    )

                    if tip:
                        painter.setFont(tip_font)
                        painter.setPen(QColor("#444444"))
                        painter.drawText(
                            QRectF(tx, y + cell_h * 0.42, text_w, cell_h * 0.35),
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop |
                            Qt.TextFlag.TextWordWrap,
                            tip[:80]
                        )

                    if sc:
                        painter.setFont(sc_font)
                        painter.setPen(QColor("#cc3355"))
                        painter.drawText(
                            QRectF(tx, y + cell_h * 0.78, text_w, cell_h * 0.2),
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                            sc
                        )

                    col += 1
                    if col >= col_count:
                        col = 0
                        y += cell_h + dpi * 0.04

                painter.end()
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(dlg, "Export PDF", f"Saved to:\n{path}")

            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(dlg, "Export PDF", f"PDF export failed:\n{e}")

        if not hasattr(self, "_icon_cheat_sheet_dlg"):
            self._icon_cheat_sheet_dlg = None
        self._icon_cheat_sheet_dlg = dlg
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.show()
        dlg.raise_()