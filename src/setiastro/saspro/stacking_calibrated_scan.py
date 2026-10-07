# src/setiastro/saspro/stacking_calibrated_scan.py
"""
Header scan behind the Stacking Suite's Image Integration tree.

Business logic only, no widgets: list the calibrated frames, read each
frame's header once, cache the result by path, mtime and size, and do the
reads on a few daemon threads. StackingSuiteDialog turns the results into
tree rows.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Iterable, NamedTuple, Optional

from astropy.io import fits

# Extensions the Image Integration tab accepts.
CALIBRATED_EXTS = frozenset({".fit", ".fits", ".ftz", ".fz", ".tiff", ".tif", ".xisf"})

# Header reads are dominated by file-open latency (disk, antivirus), so a few
# threads overlap it well even though header parsing holds the GIL.
DEFAULT_SCAN_WORKERS = 4


class ScanEntry(NamedTuple):
    """A frame path plus the stat fields that validate its cache entry."""
    path: str
    mtime_ns: Optional[int]
    size: Optional[int]


class CalibratedMeta(NamedTuple):
    """Grouping metadata for one frame of the Image Integration tree."""
    filt: str
    exp: float
    size: str
    gain: Optional[float]
    obj: str
    usable: bool = True  # False: the frame is left out of the tree


def stat_entry(path: str) -> ScanEntry:
    """Return a ScanEntry for path, with stat fields None if it can't be stat'ed."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return ScanEntry(path, None, None)
    return ScanEntry(path, st.st_mtime_ns, st.st_size)


def list_folder_entries(folder: str, exts: Iterable[str] = CALIBRATED_EXTS) -> list[ScanEntry]:
    """
    Files directly inside folder whose extension is in exts, in listing order.

    os.scandir returns the file type and stat data with the listing on
    Windows, so this costs no extra system call per file there.
    """
    entries = []
    if not os.path.isdir(folder):
        return entries
    with os.scandir(folder) as it:
        for de in it:
            if os.path.splitext(de.name)[1].lower() not in exts:
                continue
            try:
                if not de.is_file():
                    continue
            except OSError:
                continue
            try:
                st = de.stat()
            except OSError:
                entries.append(ScanEntry(de.path, None, None))
                continue
            entries.append(ScanEntry(de.path, st.st_mtime_ns, st.st_size))
    return entries


def _declares_2d(value) -> bool:
    try:
        return value is not None and int(value) >= 2
    except (TypeError, ValueError):
        return False


def read_fits_headers(path: str):
    """
    Return (primary_header, science_header) from a single open of a FITS file.

    primary_header is what fits.getheader(path, ext=0) returns. science_header
    belongs to the first HDU declaring two or more axes (NAXIS or ZNAXIS),
    falling back to the primary one; that is the HDU get_valid_header picks.
    science_header is None if the extension HDUs can't be parsed. Pixel data
    is never read.
    """
    with fits.open(path, mode="readonly", memmap=False, lazy_load_hdus=True) as hdul:
        primary = hdul[0].header
        try:
            science = primary
            for hdu in hdul:
                hdr = hdu.header
                if _declares_2d(hdr.get("NAXIS")) or _declares_2d(hdr.get("ZNAXIS")):
                    science = hdr
                    break
        except Exception:
            science = None
    return primary, science


class CalibratedMetaCache:
    """Thread-safe CalibratedMeta store; an entry is valid while mtime and size match."""

    def __init__(self):
        self._items: dict[str, tuple[int, int, CalibratedMeta]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def lookup(self, entry: ScanEntry) -> Optional[CalibratedMeta]:
        if entry.mtime_ns is None:
            return None
        key = self._key(entry.path)
        with self._lock:
            hit = self._items.get(key)
        if hit is None or hit[0] != entry.mtime_ns or hit[1] != entry.size:
            return None
        return hit[2]

    def store(self, entry: ScanEntry, meta: CalibratedMeta) -> None:
        if entry.mtime_ns is None:
            return
        key = self._key(entry.path)
        with self._lock:
            self._items[key] = (entry.mtime_ns, entry.size, meta)


def read_calibrated_metas(
    entries: Iterable[ScanEntry],
    cache: CalibratedMetaCache,
    probe: Callable[[str], CalibratedMeta],
    *,
    workers: int = DEFAULT_SCAN_WORKERS,
    stop: Optional[threading.Event] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Optional[dict[str, CalibratedMeta]]:
    """
    Return {path: meta} for entries, calling probe(path) only on cache misses.

    Misses are read on up to `workers` daemon threads, so an app exit never
    waits on them. on_progress(done, total) runs on those threads about 20
    times over the misses. A path whose probe raised is left out of the
    result. Returns None if `stop` gets set.
    """
    metas: dict[str, CalibratedMeta] = {}
    misses = []
    for entry in entries:
        meta = cache.lookup(entry)
        if meta is None:
            misses.append(entry)
        else:
            metas[entry.path] = meta

    total = len(misses)
    if total:
        lock = threading.Lock()
        pending = iter(misses)
        step = max(1, -(-total // 20))  # ceil: at most 20 reports
        done = 0

        def _drain():
            nonlocal done
            while stop is None or not stop.is_set():
                with lock:
                    entry = next(pending, None)
                if entry is None:
                    return
                try:
                    meta = probe(entry.path)
                except Exception:
                    meta = None
                if meta is not None:
                    cache.store(entry, meta)
                with lock:
                    if meta is not None:
                        metas[entry.path] = meta
                    done += 1
                    count = done
                if on_progress is not None and (count % step == 0 or count == total):
                    on_progress(count, total)

        n_threads = max(1, min(workers, total))
        if n_threads == 1:
            _drain()
        else:
            threads = [
                threading.Thread(target=_drain, name=f"saspro-calibrated-scan-{i}", daemon=True)
                for i in range(n_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

    if stop is not None and stop.is_set():
        return None
    return metas


def start_background_read(
    entries: Iterable[ScanEntry],
    cache: CalibratedMetaCache,
    probe: Callable[[str], CalibratedMeta],
    *,
    stop: threading.Event,
    workers: int = DEFAULT_SCAN_WORKERS,
    on_progress: Optional[Callable[[int, int], None]] = None,
    on_done: Optional[Callable[[], None]] = None,
) -> threading.Thread:
    """
    Fill cache for entries on a daemon thread (see read_calibrated_metas).

    on_done() runs on that thread once the reads finish, unless `stop` was set.
    """
    entries = list(entries)

    def _run():
        metas = read_calibrated_metas(
            entries, cache, probe, workers=workers, stop=stop, on_progress=on_progress
        )
        if metas is not None and on_done is not None:
            on_done()

    thread = threading.Thread(target=_run, name="saspro-calibrated-scan", daemon=True)
    thread.start()
    return thread
