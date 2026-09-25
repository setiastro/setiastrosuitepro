# Stacking Monitor Phase Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show live N/M (or chunk) progress in the Stacking Execution Monitor Note for Measurements, Normalization, Calibration, Registration, and Integration.

**Architecture:** Keep LogBus + `_classify` / `_SUPPRESS` as the sole progress channel. Unsuppress or add rules for messages already emitted; add one throttled `update_status` from alignment flush for Registration.

**Tech Stack:** Python, PyQt6, unittest (`tests/test_stacking_monitor_*.py`)

## Global Constraints

- Do not invent a separate monitor progress API.
- Do not change Master Dark/Flat or dual-band NB chatter.
- Match existing Satellite Trails RUNNING-note update pattern.
- Throttle Registration LogBus posts (~20 updates max).
- Do not commit unless the user asks.

---

### Task 1: Classifier rules + suppress fixes

**Files:**
- Modify: `src/setiastro/saspro/stacking_monitor.py` (`_RULES` Measurements/Normalization/Registration/Integration; `_SUPPRESS`)
- Test: `tests/test_stacking_monitor_phase_progress.py` (create)

**Interfaces:**
- Consumes: existing `_classify(msg) -> Optional[tuple[str, str, str]]`, `StackingMonitorDialog._on_message`
- Produces: classified RUNNING for progress strings listed in the design spec

- [ ] **Step 1: Write failing tests**

Create `tests/test_stacking_monitor_phase_progress.py` with classify assertions and one dialog note-update test per phase (Measurements, Normalization, Calibration Progress, Integration Tile, Registration Aligning).

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest tests/test_stacking_monitor_phase_progress.py -v`

- [ ] **Step 3: Implement rules / suppress**

In `stacking_monitor.py`:
- After Measurements start/complete rules, add `_r(r"📦 Measured \d+/\d+ frames", "Measurements", _ST_RUNNING)`
- After Normalization start/complete, add `_r(r"🌀 Normalizing chunk \d+/\d+", "Normalization", _ST_RUNNING)`
- After Registration start/complete, add `_r(r"(?:📐\s*)?Aligning stars… \(\d+/\d+\)", "Registration", _ST_RUNNING)`
- After Integration stacking-group rule, add:
  - `_r(r"🔧 \[LowRAM\] Tile \d+/\d+", "Integration", _ST_RUNNING)`
  - `_r(r"🔧 Tile \d+/\d+", "Integration", _ST_RUNNING)`
- From `_SUPPRESS`, remove: `📷 Progress:`, `🌀 Normalizing chunk`, `tile \d+/\d+`, `Aligning stars… \(\d`, and the bare `Aligning stars` entry (keep `Image Registration` / `🔄 Image Registration` if still needed for other noise).

- [ ] **Step 4: Run tests — expect PASS**

Run: `python -m pytest tests/test_stacking_monitor_phase_progress.py tests/test_stacking_monitor_satellite.py -v`

---

### Task 2: Registration LogBus emit

**Files:**
- Modify: `src/setiastro/saspro/stacking_suite.py` (`_flush_align_progress`, ~6780)

**Interfaces:**
- Consumes: `self._align_prog_pending`, `self.update_status`
- Produces: throttled messages matching Task 1 Registration rule

- [ ] **Step 1: Add throttled status post in `_flush_align_progress`**

After updating the QProgressDialog, when `total > 0`:
- `report_every = max(1, total // 20)`
- Post when `done in (1, total)` or `done % report_every == 0`
- Deduplicate with `self._align_prog_status_last`
- Message: `self.tr("📐 Aligning stars… ({0}/{1})").format(done, total)`

- [ ] **Step 2: Smoke-check classify on that exact string**

Run a one-liner or assert in the phase-progress test that `_classify("📐 Aligning stars… (42/400)")` returns Registration RUNNING.
