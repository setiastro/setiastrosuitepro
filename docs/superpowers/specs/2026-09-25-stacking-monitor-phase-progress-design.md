# Stacking Execution Monitor — live phase progress

## Problem

During long phases (Measurements, Normalization, etc.) the Stacking Execution
Monitor stays stuck on the phase-start Note (e.g. “Phase: Normalization
starting…”) while the Stacking Log already shows granular progress. Users need
the same progress in the monitor Note column.

## Root cause

Progress is driven entirely by log-message classification in
`stacking_monitor.py`. Messages that do not match `_RULES`, or that match
`_SUPPRESS` first, never update the open RUNNING row — even when they appear
in the Stacking Log via `LogBus`.

## Goals

- While a phase row is RUNNING, its Note updates with live N/M (or chunk)
  progress already visible in the Stacking Log.
- Cover: Measurements, Normalization, Calibration, Registration, Integration.
- Preserve existing start/complete lifecycle and Satellite Trails / Star-Trail
  patterns that already work.
- No new monitor API; stay on message classification.

## Non-goals

- Master Dark / Master Flat tile progress (dedicated QProgressDialog already).
- Dual-band per-frame NB chatter (intentionally noisy).
- Removing the spurious generic “Complete” catch-all row for unrelated ✅ lines.
- Changing progress cadence in emitters except Registration (needs a LogBus emit).

## Design

### Classifier (`stacking_monitor.py`)

| Phase | Action |
|-------|--------|
| Measurements | Add rule `📦 Measured \d+/\d+ frames` → Measurements RUNNING |
| Normalization | Add rule `🌀 Normalizing chunk \d+/\d+` → Normalization RUNNING; remove matching suppress |
| Calibration | Remove `📷 Progress:` from `_SUPPRESS` (rule already exists) |
| Integration | Add rules for `🔧 Tile \d+/\d+` and `🔧 [LowRAM] Tile \d+/\d+`; narrow/remove suppress `tile \d+/\d+` so those match (keep comet / other noisy tile lines suppressed) |
| Registration | Add rule for throttled align progress message (below) |

Existing RUNNING-row update path (`_on_message` when `op in self._open`) refreshes Note.

### Registration emit (`stacking_suite.py`)

`_flush_align_progress` today only updates `QProgressDialog`. Also call
`update_status` with a throttled message (~20 updates max, same idea as
Measurements `report_every`), e.g. `📐 Aligning stars… (N/M)`, so the monitor
can classify it. Remove `Aligning stars… (\d` / bare `Aligning stars` from
`_SUPPRESS` only as needed so the new message is not dropped.

### Tests

Extend `tests/test_stacking_monitor_satellite.py` (or a sibling test module)
with classify + dialog Note-update cases for each phase above.

## Success criteria

- Measurements Note shows `Measured N/M frames` during the phase.
- Normalization Note shows `Normalizing chunk N/M …` (not stuck on starting).
- Calibration Note shows `Progress: group — N/M frames`.
- Registration Note shows `Aligning stars… (N/M)`.
- Integration Note shows `Tile N/M [group] …` (or LowRAM variant).
- Existing satellite / calibration group tests still pass.
