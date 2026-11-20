# Command discovery review and QA

## Scope

This contribution adds native macOS Help search and a fuzzy command popup. The document filter is unchanged. The existing Command Search dock retains its matching and Ctrl+Shift+P behavior.

## Review findings addressed

- The search menu action was owned only by a menu. The shortcut action is now registered directly on its main window and reused when menus are rebuilt. A separate keypad-division binding handles its distinct Qt modifier. Auto-repeat cannot toggle the popup repeatedly.
- Popup input previously allowed Qt's default maximum length and retained raw queries in the match cache. Popup and proxy queries are now limited to 256 characters, with normalized matching cached separately. Punctuation-only queries produce no results.
- Commands were collected only at startup. Opening the popup now refreshes enabled commands and descriptions, including dynamically added scripts. Items are inserted in one batch.
- Storing a QAction directly in item data left a dangling QObject pointer if a dynamic command was destroyed. Reading that QVariant could crash before Python checked the action. An opaque Python target retains the wrapper, and execution checks whether the Qt object still exists and remains enabled.
- Repeated Enter events could invoke callbacks on an already hidden popup. Execution now requires a visible popup, a valid source index and a non-repeat Enter event.
- Popup geometry is clamped to the available screen, and long result labels do not add horizontal scrolling.

## Automated checks

Run the focused tests without the image-processing dependencies:

```sh
python -m pip install pytest PyQt6==6.11.0 PyQt6-Qt6==6.11.2 PyQt6-sip==13.12.0
python -m pytest -q tests/test_command_search.py tests/test_command_palette.py tests/test_native_menus.py
```

Set `PYTHONPATH=src`. On a headless machine, set `QT_QPA_PLATFORM=offscreen`. Native Windows keyboard tests require `QT_QPA_PLATFORM=windows`; they skip on other backends.

The workflow `.github/workflows/command-discovery.yml` runs these checks on Windows, Linux and macOS, using the Qt versions pinned by the repository. Windows additionally injects native virtual-key events under US and Polish (Programmers) layouts. It checks focus before injecting keys and restores the previous layout afterward. This exercises Windows key translation, which synthetic QTest key sequences bypass.

Covered cases include:

| Area | Checks |
| --- | --- |
| Matching | Exact names, descriptions, typo ranking, word order, case, decomposed accents, non-Latin scripts, whitespace, punctuation and emoji |
| Large input | Pasting one million characters, bounded cache input, clearing and searching again |
| Results | Empty results, changing models, source-index mapping, long titles, keyboard navigation and modal tool opening |
| Execution | Disabled/destroyed actions, stale indices, repeated Enter, hidden popup |
| Shortcuts | Main image/MDI, text fields, spinboxes, menu rebuilds, ordinary slash text, keypad division and popup closing |
| Geometry | A 320×180 available screen, long labels |
| macOS menus | Native-menu policy, Help/About/Preferences roles, link-group action wiring |

Local full-window smoke checks also exercise live action refresh, disabled and destroyed commands, Ctrl+Shift+P with the dock hidden, menu rebuilding, fuzzy ADBE lookup and Enter opening Curves on a synthetic FITS image.

## Manual release checks

1. On macOS, confirm native Help search locates a nested menu command.
2. On Windows with Polish (Programmers), open an image and test Ctrl+/ from the image, a parameter field and Command Search. Repeat using keypad division if available. Confirm About → Search Commands opens the same popup.
3. In the popup, test `automatic`, `automtic`, Enter, Escape, repeat shortcut, Up/Down, PageUp/PageDown, paste and clear.
4. Confirm plain `/` remains ordinary text, and Ctrl+Shift+P retains the existing dock/popup behavior.
5. Open a modal tool through Enter and confirm the popup is hidden before the tool opens.
6. Change UI language and repeat shortcut and Help-menu checks.

The initial Windows report confirmed Ctrl+Shift+P but not Ctrl+/. The exact failing focus/key path was not recorded. The window-owned action and keypad binding address concrete shortcut-registration cases; they must not be described as proof that the original hardware report is resolved without a matching retest.

## Local review results

On macOS with Python 3.13.7 and Qt 6.12.0, the focused suite passes 40 tests; two native Windows layout tests skip as expected. The earlier full-suite audit recorded 235 passed, 36 failed, 17 skipped and one collection error. The failing/error test IDs are identical to the previously recorded main baseline; no additional failing IDs appeared. The collection error is the existing missing HDR `exposure_order_warning` import.

Full-window smoke tests pass live action refresh, disabled and destroyed action execution guards, popup shortcut opening/closing and menu rebuilding. Native keyboard translation on Windows is verified separately by the workflow; local offscreen results alone do not establish that a physical Windows shortcut works.

## Active-view regression

The popup stores a reference to the same QAction used by the menu and toolbar; it does not capture an image document. A regression test opens two MDI views, activates A → B → A, searches and executes the command each time, and checks that keyboard focus entering the popup does not change the selected view. Direct QAction activation is checked against the same document context.

A separate full-window smoke check opens the actual Curves and ADBE dialogs for two synthetic images. It verifies the exact ROI-aware document proxy of the active view and confirms that both source images remain unchanged. Document selection remains the responsibility of the existing tool handlers, just as for menu/toolbar activation.
