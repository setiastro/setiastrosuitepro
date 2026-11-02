# GPU_OPT stacking changes

The branch reduces host copies and GPU round-trips without changing the
stacking controls or the default 16-row band height.

- Normal integration reads FITS/XISF directly into reusable float32 buffers.
  Integer normalization and Astropy BSCALE/BZERO behavior are preserved.
- Retired satellite sidecar masks no longer allocate/clear/upload empty masks.
- Per-group GPU sessions upload frame weights once. CUDA uses two bounded
  pinned input slots and a separate upload stream when host/device memory
  budgets permit. An already-read next band can upload during current-band
  reduction. Other backends retain single-tile staging and weight caching.
- CUDA output copies queue together and synchronize once per tile. Rejection
  summaries are computed before downloads. When drizzle/MFDeconv needs full
  per-frame masks, channels are collapsed on the GPU before transfer.
- Winsorized clipping retains its original median during iteration: a clamp
  centered on that median cannot change it. CPU/GPU both omit the redundant
  recalculation. Scale estimation, convergence, thresholds and weights remain.
- DirectML uses device-side selection for NaN medians/quantiles, including
  lower-middle medians, interpolated quantiles and all-NaN results. Finite
  replacement avoids DirectML's `nan_to_num` CPU fallback. DirectML uses
  `no_grad` because inference-mode tensor indexing can fail on that backend.
- Simple Average omits its unused median fallback for finite positive weights.
  Other weights retain the original median fallback.
- Low RAM mode reuses its I/O executor and caches GPU weights, without adding
  pinned look-ahead buffers to its memory budget.
- GPU drizzle keeps numerator and per-channel coverage resident across frames,
  processing bounded source blocks. Square/circle/Gaussian footprints,
  projective transforms and sparse CFA coverage are checked against CPU kernels
  before enabling a backend. A runtime failure restarts the entire group on
  CPU so partially deposited frames cannot corrupt the result.

DirectML drizzle deliberately remains on CPU. On the installed torch-directml
0.2.5 runtime, scatter-add with indices `[0, 0, 2, 3]` and values `[1, 2, 3, 4]`
produced 2 at index 0 instead of 3. The alternative `index_add` uses a CPU
fallback. Non-CUDA backends also retain CPU drizzle for 64-bit accumulation.
CUDA/MPS drizzle still needs validation on those actual devices. GPU atomic
accumulation is checked with tolerances rather than requiring bitwise equality.

## Timing

Normal integration logs read-future wait, host upload staging, reduction and
download/wait totals. CUDA reduction uses event timing; DirectML's combined
reduction/download time includes the host synchronization. These are overlapping
stages and must not be added to estimate total runtime.

`benchmark_stacking_gpu.py` measures a deterministic synthetic reducer, warms
up each implementation and includes result downloads. It does not install
packages. Example from a Python environment with the SAS Pro runtime available:

```powershell
python ops/benchmark_stacking_gpu.py --backend directml --baseline-ref dev --frames 600 --rows 16 --width 1080 --repeat 3
```

On the RX 7900 XTX / PyTorch 2.4.1 / torch-directml 0.2.5 test runtime, the
600-frame RGB Winsorized example measured median 0.7331 s before and 0.4655 s
after (1.57x). Output matched the CPU reference within 2e-5 and rejection maps
matched exactly. This is a synthetic reducer result, not an end-to-end stacking
speedup. Disk reads, calibration, alignment and drizzle are excluded; the
CUDA pipeline is not exercised by a DirectML benchmark.

## Checks

`tests/test_stacking_gpu_optimizations.py` covers device selection, quantile
semantics, redundant-median removal, cached/prepared tiles, rejection-map
equivalence and all drizzle kernels in mono/RGB and float32/float64. Set
`SASPRO_TEST_DIRECTML=1` to include the actual DirectML hardware test. The CUDA
slot-reuse test skips without CUDA hardware.

`tests/test_stacking_integration_memory.py` also checks direct reader parity,
the low-RAM reader, disabled masks, existing memory-error recovery and GPU
fallback. Real-file stacking and CUDA/MPS hardware performance remain separate
validation steps.
