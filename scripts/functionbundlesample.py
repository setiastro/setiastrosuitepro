# run_function_bundle.py
#
# Lets you pick a Function Bundle (from the Function Bundles dialog)
# and run it on the active view using the 'function_bundle' command.
from __future__ import annotations
SCRIPT_NAME  = "Run Function Bundle…"
SCRIPT_GROUP = "Function Bundles"



import json

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QDialogButtonBox, QMessageBox
)


def run(ctx):
    mw = ctx.main_window()  # this is ctx.app under the hood

    # --- load bundles from the same QSettings key used by FunctionBundleDialog ---
    s = QSettings()
    raw = s.value("functionbundles/v1", "[]", type=str)
    try:
        bundles = json.loads(raw)
    except Exception:
        bundles = []

    if not isinstance(bundles, list) or not bundles:
        QMessageBox.information(mw, "Run Bundle", "No Function Bundles found.")
        return

    # --- pick which bundle to run ---
    dlg = QDialog(mw)
    dlg.setWindowTitle("Run Bundle")
    v = QVBoxLayout(dlg)
    v.addWidget(QLabel("Select a Function Bundle to run on the active view:"))

    lb = QListWidget()
    lb.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

    # Use UserRole to store the bundle name
    for b in bundles:
        if not isinstance(b, dict):
            continue
        name = (b.get("name") or "Function Bundle").strip()
        steps = b.get("steps") or []
        item = QListWidgetItem(f"{name}  ({len(steps)} steps)")
        item.setData(Qt.ItemDataRole.UserRole, name)
        lb.addItem(item)
    v.addWidget(lb)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok |
        QDialogButtonBox.StandardButton.Cancel
    )
    v.addWidget(buttons)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    item = lb.currentItem()
    if item is None:
        return

    bundle_name = item.data(Qt.ItemDataRole.UserRole)
    if not bundle_name:
        QMessageBox.warning(mw, "Run Bundle", "Invalid bundle selection.")
        return

    # --- Tell the command which bundle to run ---
    # This feeds straight into pro/function_bundle.run_function_bundle_command(ctx, preset)
    # which then synthesizes a 'function_bundle' drop payload and calls
    # app._handle_command_drop(...) for you.
    cfg = {
        "bundle_name": bundle_name,  # also accepts "name": bundle_name
        "inherit_target": True,      # let child steps reuse the same target_sw
        # Optional:
        # "targets": "all_open"    # to fan out to all open views, if you want later
    }

    try:
        ctx.run_command("function_bundle", cfg)
    except Exception as e:
        QMessageBox.critical(
            mw,
            "Run Bundle",
            f"Failed to run bundle '{bundle_name}':\n\n{e}"
        )
