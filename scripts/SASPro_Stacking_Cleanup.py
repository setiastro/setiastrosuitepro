
def run(ctx):
    ctx.log("Welcome to SASPro Stacking Cleanup")
    show_results([], ctx=ctx)


def _import_qt():
    for mod in ("PySide6", "PyQt6", "PyQt5", "PySide2"):
        try:
            pkg = __import__(mod)
            from importlib import import_module
            QtWidgets = import_module(f"{mod}.QtWidgets")
            return QtWidgets
        except Exception:
            continue
    return None

# SetiAstroSuitePro script metadata
SCRIPT_NAME = "SASPro Stacking Cleanup"
SCRIPT_GROUP = "Stacking"
VERSION = "1.0.0"


import os

TARGET_FOLDERS = ['Aligned_Images', 'Calibrated', 'Normalized_Images']


def scan_folders(root_folder, ctx=None):
    parent_totals = {}
    for dirpath, dirnames, filenames in os.walk(root_folder):
        total_size = 0
        found = False
        for target in TARGET_FOLDERS:
            if target in dirnames:
                found = True
                subfolder_path = os.path.join(dirpath, target)
                for subdirpath, _, subfilenames in os.walk(subfolder_path):
                    for fname in subfilenames:
                        fpath = os.path.join(subdirpath, fname)
                        if os.path.isfile(fpath):
                            total_size += os.path.getsize(fpath)
        if found:
            parent_name = os.path.basename(dirpath)
            parent_totals[parent_name] = parent_totals.get(parent_name, 0) + total_size
    summary = [(parent, total) for parent, total in parent_totals.items()]
    if ctx:
        ctx.log(f"Scan complete. Found {len(summary)} parent folders.")
    return summary


def scan_folders_detailed(root_folder, ctx=None):
    """Return list of (parent_name, parent_path, total_size, children)
    where children is list of (subfolder_name, subfolder_path, size_bytes).
    """
    parent_list = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Skip if this directory is itself one of the target folders
        current_dir_name = os.path.basename(dirpath)
        if current_dir_name in TARGET_FOLDERS:
            continue
        
        children = []
        parent_total = 0
        for target in TARGET_FOLDERS:
            if target in dirnames:
                subfolder_path = os.path.join(dirpath, target)
                subtotal = 0
                for subdirpath, _, subfilenames in os.walk(subfolder_path):
                    for fname in subfilenames:
                        fpath = os.path.join(subdirpath, fname)
                        if os.path.isfile(fpath):
                            subtotal += os.path.getsize(fpath)
                if subtotal > 0:
                    children.append((target, subfolder_path, subtotal))
                    parent_total += subtotal
        
        # Scan for .sasd files in the parent folder itself
        sasd_total = 0
        sasd_files = []
        for fname in filenames:
            if fname.lower().endswith('.sasd'):
                fpath = os.path.join(dirpath, fname)
                if os.path.isfile(fpath):
                    sasd_total += os.path.getsize(fpath)
                    sasd_files.append(fpath)
        if sasd_total > 0:
            # Store comma-separated list of .sasd file paths
            children.append(("Misc stacking files (.sasd)", ",".join(sasd_files), sasd_total))
            parent_total += sasd_total
        
        if children:
            parent_name = os.path.basename(dirpath) or dirpath
            parent_list.append((parent_name, dirpath, parent_total, children))
    if ctx:
        ctx.log(f"Detailed scan complete. Found {len(parent_list)} parent folders.")
    return parent_list


def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def show_results(results, ctx=None):
    QtWidgets = _import_qt()
    if QtWidgets is None:
        if ctx:
            ctx.log("No Qt binding available. Showing results in log.")
            if not results:
                ctx.log("No target folders found.")
                return
            ctx.log("Scan results:")
            header = f"{'Parent Folder':30} | {'Total Size':12}"
            ctx.log(header)
            ctx.log("-" * len(header))
            for parent_name, total_size in results:
                line = f"{parent_name:30} | {format_size(total_size):12}"
                ctx.log(line)
        return

    parent = None
    if ctx and hasattr(ctx, "main_window"):
        try:
            parent = ctx.main_window()
        except Exception:
            parent = None

    QtCore = None
    QtFlag = None
    QtCheck = None
    try:
        from importlib import import_module
        QtCore = import_module(QtWidgets.__name__.replace(".QtWidgets", ".QtCore"))
        # Qt6: enums are in QtCore.Qt.ItemFlag and QtCore.Qt.CheckState
        if hasattr(QtCore.Qt, "ItemFlag"):
            QtFlag = QtCore.Qt.ItemFlag
        else:
            QtFlag = QtCore.Qt
        if hasattr(QtCore.Qt, "CheckState"):
            QtCheck = QtCore.Qt.CheckState
        else:
            QtCheck = QtCore.Qt
    except Exception as e:
        if ctx:
            ctx.log(f"QtCore import error: {e}")

    class ResultsDialog(QtWidgets.QDialog):
        def __init__(self, ctx=None, parent=None):
            super().__init__(parent)
            self.ctx = ctx
            self.setWindowTitle("SASPro Stacking Cleanup " + (f" v{VERSION}" if VERSION else ""))
            self.resize(700, 500)
            self.folder = ""
            self.layout = QtWidgets.QVBoxLayout(self)

            # Folder path display and browse button
            folder_row = QtWidgets.QHBoxLayout()
            self.folder_label = QtWidgets.QLabel("")
            folder_row.addWidget(QtWidgets.QLabel("Root Processing Folder:"))
            folder_row.addWidget(self.folder_label)
            self.browse_btn = QtWidgets.QPushButton("Browse...")
            self.browse_btn.clicked.connect(self.browse_folder)
            folder_row.addWidget(self.browse_btn)
            self.layout.addLayout(folder_row)

            # Tree view
            try:
                self.table = QtWidgets.QTreeWidget(self)
            except Exception:
                self.table = getattr(QtWidgets, 'QTreeWidget')(self)
            try:
                self.table.setColumnCount(4)
                self.table.setHeaderLabels(["Stacking Folder", "Size", "Total Size", "Selected Size"])
            except Exception:
                try:
                    self.table.headerItem().setText(0, "Stacking Folder")
                    self.table.headerItem().setText(1, "Size")
                    self.table.headerItem().setText(2, "Total Size")
                    self.table.headerItem().setText(3, "Selected Size")
                except Exception:
                    pass
            try:
                self.table.itemChanged.connect(self.on_item_changed)
            except Exception:
                pass
            try:
                self.table.itemClicked.connect(self.on_item_clicked)
            except Exception:
                pass
            self.table.setRootIsDecorated(True)
            self.table.setColumnWidth(0, 300)
            self.layout.addWidget(self.table)
            
            # Flag to prevent cascading updates
            self._updating_parent = False

            # Toggle checkboxes button
            self.toggle_btn = QtWidgets.QPushButton("Check All")
            self.toggle_btn.clicked.connect(self.toggle_all_checkboxes)
            self.layout.addWidget(self.toggle_btn)
            self.all_checked = False  # Track toggle state

            # Action buttons
            btn_row = QtWidgets.QHBoxLayout()
            self.delete_btn = QtWidgets.QPushButton("Delete Selected")
            self.delete_btn.clicked.connect(self.delete_selected_items)
            btn_row.addWidget(self.delete_btn)
            btn_row.addStretch(1)
            self.close_btn = QtWidgets.QPushButton("Close")
            self.close_btn.clicked.connect(self.accept)
            btn_row.addWidget(self.close_btn)
            self.layout.addLayout(btn_row)

            # Tree is empty until user selects folder
            try:
                self.table.clear()
            except Exception:
                try:
                    self.table.clearContents()
                except Exception:
                    pass

            self.summary_label = QtWidgets.QLabel("Total selected size: 0 B")
            self.layout.addWidget(self.summary_label)

        def _get_user_role(self, offset=0):
            """Get UserRole constant with fallback to numeric value"""
            try:
                return QtCore.Qt.UserRole + offset
            except (AttributeError, TypeError):
                return 256 + offset  # Qt.UserRole = 256

        def _get_item_check_state(self, item):
            """Get check state from item with fallbacks"""
            if not QtCheck:
                return False
            try:
                return item.data(0, QtCore.Qt.CheckStateRole) == QtCheck.Checked
            except Exception:
                try:
                    return item.checkState(0) == QtCheck.Checked
                except Exception:
                    try:
                        return item.checkState() == QtCheck.Checked
                    except Exception:
                        return False

        def _set_item_check_state(self, item, state):
            """Set check state on item with fallbacks"""
            try:
                item.setData(0, QtCore.Qt.CheckStateRole, state)
            except Exception:
                try:
                    item.setCheckState(0, state)
                except Exception:
                    try:
                        item.setCheckState(state)
                    except Exception:
                        pass

        def _get_item_size_bytes(self, item):
            """Get size in bytes from item data or text"""
            try:
                val = item.data(0, self._get_user_role())
                if val is not None:
                    return int(val)
            except Exception:
                pass
            try:
                val = item.data(self._get_user_role())
                if val is not None:
                    return int(val)
            except Exception:
                pass
            # Parse from text as fallback - check column 1 (Selected Size for parent, size for children)
            try:
                txt = (item.text(1) or "").strip()
                parts = txt.split()
                if len(parts) >= 2:
                    num = float(parts[0].replace(',', ''))
                    unit = parts[1].upper()
                    mul = {
                        'B': 1, 'KB': 1024, 'MB': 1024**2,
                        'GB': 1024**3, 'TB': 1024**4, 'PB': 1024**5
                    }.get(unit, 1)
                    return int(num * mul)
                elif parts:
                    return int(float(parts[0]))
            except Exception:
                pass
            return 0

        def browse_folder(self):
            # Use static method for better cross-platform compatibility (especially Win11)
            if hasattr(QtWidgets.QFileDialog, 'ShowDirsOnly'):
                folder = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    "Select Root Processing Folder",
                    self.folder or "",
                    QtWidgets.QFileDialog.ShowDirsOnly
                )
            else:
                folder = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    "Select Root Processing Folder",
                    self.folder or ""
                )
            if folder:
                self.folder = folder
                self.folder_label.setText(self.folder)
                self.update_table(self.folder)

        def update_table(self, folder):
            if self.ctx:
                self.ctx.log(f"Scanning folder: {folder}")
            
            # Show status to user
            self.summary_label.setText("Scanning folder...")
            QtWidgets.QApplication.processEvents()  # Force UI update
            
            try:
                detailed = scan_folders_detailed(folder, ctx=self.ctx)
            except Exception as e:
                if self.ctx:
                    self.ctx.log(f"Error scanning folder: {e}")
                QtWidgets.QMessageBox.warning(self, "Scan Error", f"Error scanning folder:\n{str(e)}")
                self.summary_label.setText("Error during scan")
                return
            
            # Sort by total size, largest first
            detailed.sort(key=lambda x: x[2], reverse=True)
            try:
                self.table.blockSignals(True)
            except Exception:
                pass
            try:
                self.table.clear()
            except Exception:
                try:
                    self.table.clearContents()
                except Exception:
                    pass
            for parent_name, parent_path, total_size, children in detailed:
                try:
                    pitem = QtWidgets.QTreeWidgetItem([str(parent_name), "", format_size(total_size), "0 B"])
                except Exception:
                    pitem = getattr(QtWidgets, 'QTreeWidgetItem')([str(parent_name), "", format_size(total_size), "0 B"])
                # make parent checkable
                try:
                    if QtCore and QtFlag and QtCheck:
                        pitem.setFlags(pitem.flags() | QtFlag.ItemIsUserCheckable | QtFlag.ItemIsEnabled)
                    else:
                        pitem.setFlags(pitem.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                except Exception:
                    pass
                # set initial checked state using CheckStateRole when possible
                try:
                    pitem.setData(0, QtCore.Qt.CheckStateRole, QtCheck.Unchecked)
                except Exception:
                    try:
                        pitem.setCheckState(0, QtCheck.Unchecked)
                    except Exception:
                        try:
                            pitem.setCheckState(QtCheck.Unchecked)
                        except Exception:
                            pass
                try:
                    pitem.setData(0, self._get_user_role(), int(total_size))
                except Exception:
                    pass
                try:
                    pitem.setData(0, self._get_user_role(1), parent_path)
                except Exception:
                    pass
                for cname, cpath, csize in children:
                    try:
                        citem = QtWidgets.QTreeWidgetItem([str(cname), format_size(csize), "", ""])
                    except Exception:
                        citem = getattr(QtWidgets, 'QTreeWidgetItem')([str(cname), format_size(csize), "", ""])
                    try:
                        citem.setData(0, self._get_user_role(), int(csize))
                    except Exception:
                        pass
                    try:
                        citem.setData(0, self._get_user_role(1), cpath)
                    except Exception:
                        pass
                    try:
                        if QtCore and QtFlag and QtCheck:
                            citem.setFlags(citem.flags() | QtFlag.ItemIsUserCheckable | QtFlag.ItemIsEnabled)
                        else:
                            citem.setFlags(citem.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    except Exception:
                        pass
                    try:
                        citem.setData(0, QtCore.Qt.CheckStateRole, QtCheck.Unchecked)
                    except Exception:
                        try:
                            citem.setCheckState(0, QtCheck.Unchecked)
                        except Exception:
                            try:
                                citem.setCheckState(QtCheck.Unchecked)
                            except Exception:
                                pass
                    try:
                        pitem.addChild(citem)
                    except Exception:
                        # couldn't add child; skip this child
                        pass
                try:
                    self.table.addTopLevelItem(pitem)
                except Exception:
                    try:
                        self.table.insertTopLevelItem(0, pitem)
                    except Exception:
                        pass
            try:
                self.table.blockSignals(False)
            except Exception:
                pass
            try:
                self.table.expandAll()
            except Exception:
                pass
            
            # Show message if no results found
            if not detailed:
                if self.ctx:
                    self.ctx.log(f"No target folders found in {folder}")
                QtWidgets.QMessageBox.information(
                    self, 
                    "No Results", 
                    f"No stacking folders found.\n\nSearched for folders named:\n{', '.join(TARGET_FOLDERS)}\n\nand .sasd files in:\n{folder}"
                )
                self.summary_label.setText("No target folders found")
                self.all_checked = False
                self.toggle_btn.setText("Check All")
            else:
                self.all_checked = False
                self.toggle_btn.setText("Check All")
                self.update_summary()

        def on_item_clicked(self, item, column):
            try:
                if item is None:
                    return
                if item.parent() is None:
                    # Parent item - toggle checkbox
                    if QtCheck:
                        current_state = self._get_item_check_state(item)
                        new_state = QtCheck.Unchecked if current_state else QtCheck.Checked
                        self._set_item_check_state(item, new_state)
                else:
                    # Child item - toggle checkbox
                    if QtCheck:
                        current_state = self._get_item_check_state(item)
                        new_state = QtCheck.Unchecked if current_state else QtCheck.Checked
                        self._set_item_check_state(item, new_state)
            except Exception:
                pass

        def on_item_changed(self, item):
            try:
                if item is None:
                    return
                # Skip if we're updating parent from child change
                if self._updating_parent:
                    return
                parent = None
                try:
                    parent = item.parent()
                except Exception:
                    parent = None
                if parent is None:
                    # parent toggled -> apply to children and update displayed selected size
                    try:
                        state = item.data(0, QtCore.Qt.CheckStateRole)
                    except Exception:
                        try:
                            state = item.checkState(0)
                        except Exception:
                            state = item.checkState()
                    
                    # Apply state to all children
                    self.table.blockSignals(True)
                    for ci in range(item.childCount()):
                        child = item.child(ci)
                        if child:
                            self._set_item_check_state(child, state)
                    self.table.blockSignals(False)
                    
                    # Update parent's displayed size based on checked children
                    sel_total = sum(
                        self._get_item_size_bytes(item.child(ci))
                        for ci in range(item.childCount())
                        if item.child(ci) and self._get_item_check_state(item.child(ci))
                    )
                    item.setText(3, format_size(sel_total))
                    self.update_summary()
                else:
                    # child toggled -> update parent checkstate and selected size
                    try:
                        self._updating_parent = True
                        p = parent
                        
                        # Count checked children and sum their sizes
                        checked_count = 0
                        sel_total = 0
                        for ci in range(p.childCount()):
                            ch = p.child(ci)
                            if ch and self._get_item_check_state(ch):
                                checked_count += 1
                                sel_total += self._get_item_size_bytes(ch)
                        
                        # Update parent check state (logical OR)
                        self.table.blockSignals(True)
                        new_state = QtCheck.Checked if checked_count > 0 else QtCheck.Unchecked
                        self._set_item_check_state(p, new_state)
                        self.table.blockSignals(False)
                        
                        # Update parent's displayed size
                        p.setText(3, format_size(sel_total))
                    finally:
                        self._updating_parent = False
                    self.update_summary()
            except Exception:
                pass

        def update_summary(self):
            total_selected = 0
            total_all_parents = 0
            
            for i in range(self.table.topLevelItemCount()):
                parent_item = self.table.topLevelItem(i)
                if not parent_item:
                    continue
                
                # Sum all children for this parent (regardless of selection)
                parent_total = 0
                for ci in range(parent_item.childCount()):
                    child = parent_item.child(ci)
                    if child:
                        child_size = self._get_item_size_bytes(child)
                        parent_total += child_size
                        if self._get_item_check_state(child):
                            total_selected += child_size
                
                total_all_parents += parent_total
            
            # Get disk usage info and calculate percentage
            disk_info = ""
            if self.folder:
                try:
                    import shutil
                    usage = shutil.disk_usage(self.folder)
                    percentage = (total_all_parents / usage.total * 100) if usage.total > 0 else 0
                    disk_info = f" | Volume size: {format_size(usage.total)} | Free space: {format_size(usage.free)} | Stacking files: {format_size(total_all_parents)} ({percentage:.1f}% of volume)"
                except Exception:
                    pass
            
            self.summary_label.setText(f"Selected size: {format_size(total_selected)}{disk_info}")
        def delete_selected_items(self):
            """Delete all checked files/folders"""
            import shutil
            
            # Collect items to delete and track parent folders
            items_to_delete = []
            parents_with_selections = []
            
            for i in range(self.table.topLevelItemCount()):
                parent_item = self.table.topLevelItem(i)
                if not parent_item:
                    continue
                
                # Check if parent is checked
                parent_checked = self._get_item_check_state(parent_item)
                parent_has_selected_children = False
                
                for ci in range(parent_item.childCount()):
                    child = parent_item.child(ci)
                    if not child:
                        continue
                    
                    # Only delete children that are actually checked
                    child_checked = self._get_item_check_state(child)
                    
                    if child_checked:
                        parent_has_selected_children = True
                        try:
                            path = child.data(0, self._get_user_role(1))
                            if path:
                                items_to_delete.append((child.text(0), path))
                        except Exception:
                            pass
                
                # Track parent if it has selected children
                if parent_has_selected_children:
                    parents_with_selections.append(parent_item.text(0))
            
            if not items_to_delete:
                QtWidgets.QMessageBox.information(self, "No Selection", "No items are selected for deletion.")
                return
            
            # Confirm deletion - show parent folders instead of individual items
            msg = f"Are you sure you want to delete selected items from {len(parents_with_selections)} parent folder(s)?\n\n"
            msg += "Parent folders with selected items:\n"
            msg += "\n".join([f"- {name}" for name in parents_with_selections[:10]])
            if len(parents_with_selections) > 10:
                msg += f"\n... and {len(parents_with_selections) - 10} more"
            
            # Get button constants with fallbacks
            try:
                yes_btn = QtWidgets.QMessageBox.StandardButton.Yes
                no_btn = QtWidgets.QMessageBox.StandardButton.No
            except AttributeError:
                try:
                    yes_btn = QtWidgets.QMessageBox.Yes
                    no_btn = QtWidgets.QMessageBox.No
                except AttributeError:
                    yes_btn = 0x00004000  # QMessageBox.Yes value
                    no_btn = 0x00010000   # QMessageBox.No value
            
            reply = QtWidgets.QMessageBox.question(
                self, "Confirm Deletion", msg,
                yes_btn | no_btn,
                no_btn
            )
            
            if reply != yes_btn:
                return
            
            # Delete items
            deleted_count = 0
            errors = []
            for name, path in items_to_delete:
                try:
                    if name == "Misc stacking files (.sasd)":
                        # path is comma-separated list of files
                        for fpath in path.split(','):
                            if os.path.isfile(fpath):
                                os.remove(fpath)
                                deleted_count += 1
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        deleted_count += 1
                    elif os.path.isfile(path):
                        os.remove(path)
                        deleted_count += 1
                except Exception as e:
                    errors.append(f"{name}: {str(e)}")
            
            # Show results
            if errors:
                error_msg = f"Deleted {deleted_count} item(s) with {len(errors)} error(s):\n\n"
                error_msg += "\n".join(errors[:5])
                if len(errors) > 5:
                    error_msg += f"\n... and {len(errors) - 5} more errors"
                QtWidgets.QMessageBox.warning(self, "Deletion Complete with Errors", error_msg)
            else:
                QtWidgets.QMessageBox.information(self, "Deletion Complete", f"Successfully deleted {deleted_count} item(s).")
            
            # Refresh the table
            if self.folder:
                self.update_table(self.folder)
        def toggle_all_checkboxes(self):
            if not QtCheck:
                return
            new_state = QtCheck.Unchecked if self.all_checked else QtCheck.Checked
            for i in range(self.table.topLevelItemCount()):
                item = self.table.topLevelItem(i)
                if item:
                    self._set_item_check_state(item, new_state)
            self.all_checked = not self.all_checked
            self.toggle_btn.setText("Select All" if not self.all_checked else "Unselect All")

    dlg = ResultsDialog(ctx=ctx, parent=parent)
    dlg.exec()
    if ctx:
        ctx.log("Results dialog displayed.")

# end of script
