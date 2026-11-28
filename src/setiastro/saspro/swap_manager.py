import os, shutil, tempfile, uuid, atexit, threading
from collections import OrderedDict
import numpy as np

import queue
import threading

class _SwapIOWorker:
    def __init__(self, get_path_fn, on_done_fn=None):
        self._get_path = get_path_fn
        self._on_done = on_done_fn
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="SwapIOWorker", daemon=True)
        self._t.start()

    def submit_save(self, swap_id: str, arr: np.ndarray):
        # make a safe copy for background write
        safe = np.ascontiguousarray(arr)  # forces contiguous copy if needed
        self._q.put(("save", swap_id, safe))

    def submit_delete(self, swap_id: str):
        self._q.put(("delete", swap_id, None))

    def shutdown(self):
        self._stop.set()
        self._q.put(("stop", None, None))

    def _run(self):
        while not self._stop.is_set():
            kind, swap_id, payload = self._q.get()
            if kind == "stop":
                break
            try:
                if kind == "save":
                    path = self._get_path(swap_id)
                    np.save(path, payload, allow_pickle=False)
                    if self._on_done:
                        self._on_done("save", swap_id, True, None)
                elif kind == "delete":
                    path = self._get_path(swap_id)
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
                    if self._on_done:
                        self._on_done("delete", swap_id, True, None)
            except Exception as e:
                if self._on_done:
                    self._on_done(kind, swap_id, False, e)
            finally:
                self._q.task_done()

import os, shutil, tempfile, uuid, atexit, threading, queue, time
from collections import OrderedDict
import numpy as np


# QSettings key for the user-chosen undo/history swap root.
SWAP_WORKDIR_SETTINGS_KEY = "paths/swap_workdir"


def default_swap_workdir() -> str:
    """Historical default: <tempdir>/SetiAstroSuitePro_Swap (often /tmp)."""
    return os.path.join(tempfile.gettempdir(), "SetiAstroSuitePro_Swap")


def resolve_swap_workdir() -> str:
    """Return the swap root: Preferences override if set, else the default."""
    custom = ""
    try:
        from PyQt6.QtCore import QSettings
        custom = str(
            QSettings().value(SWAP_WORKDIR_SETTINGS_KEY, "", type=str) or ""
        ).strip()
    except Exception:
        custom = ""
    root = os.path.expanduser(custom) if custom else default_swap_workdir()
    try:
        os.makedirs(root, exist_ok=True)
    except Exception:
        root = default_swap_workdir()
        os.makedirs(root, exist_ok=True)
    return root


# Each session dir names its owning process, so a second SASpro starting up
# never deletes the undo history of one that is still running.
_OWNER_FILE = "owner.pid"
# A dir without an owner file came from a build that did not write one; that
# build may still be running, so it must sit idle this long before removal.
_UNOWNED_MIN_IDLE_S = 3 * 24 * 3600


def _process_start_time(pid: int):
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _write_session_owner(session_dir: str) -> None:
    try:
        pid = os.getpid()
        st = _process_start_time(pid)
        with open(os.path.join(session_dir, _OWNER_FILE), "w") as f:
            f.write(f"{pid} {'' if st is None else repr(st)}")
    except Exception:
        pass


def _newest_mtime(path: str) -> float:
    newest = os.path.getmtime(path)
    with os.scandir(path) as it:
        for e in it:
            try:
                newest = max(newest, e.stat().st_mtime)
            except OSError:
                pass
    return newest


def _session_in_use(path: str) -> bool:
    """True when a running process owns *path*, or when that cannot be ruled out."""
    owner = os.path.join(path, _OWNER_FILE)
    if not os.path.isfile(owner):
        try:
            return (time.time() - _newest_mtime(path)) < _UNOWNED_MIN_IDLE_S
        except Exception:
            return True
    try:
        with open(owner) as f:
            parts = f.read().split()
        pid = int(parts[0])
        started = float(parts[1]) if len(parts) > 1 else None
    except Exception:
        return True
    try:
        import psutil
    except Exception:
        return True
    if not psutil.pid_exists(pid):
        return False
    now_started = _process_start_time(pid)
    if started is None or now_started is None:
        return True
    # Same PID but a different start time: the PID was reused, owner is gone.
    return abs(now_started - started) < 2.0


def _cleanup_orphaned_swap_sessions(root: str, keep_dir: str | None = None) -> None:
    """
    Best-effort removal of leftover session UUID dirs under *root*.

    OOM / hard kills skip atexit cleanup; those dirs otherwise sit forever
    (and on tmpfs, forever in RAM). Dirs owned by a running SASpro (another
    window, another install) are left alone.
    """
    try:
        if not root or not os.path.isdir(root):
            return
        keep = os.path.abspath(keep_dir) if keep_dir else None
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            if keep and os.path.abspath(path) == keep:
                continue
            # Session dirs are uuid4 hex (32) or uuid4 str (36 with dashes)
            compact = name.replace("-", "")
            if len(compact) != 32 or any(c not in "0123456789abcdef" for c in compact.lower()):
                continue
            if _session_in_use(path):
                continue
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


# Undo images are written in slices, and the writer checks for shutdown between
# slices, so closing SASpro never waits for a whole image to reach the disk.
_WRITE_CHUNK_BYTES = 64 * 1024 * 1024


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _save_npy_interruptible(path: str, arr: np.ndarray, stop) -> bool:
    """
    np.save(path, arr) in slices. Returns False, and leaves no file, when
    *stop* is set part way. The bytes match np.save's (same .npy 1.0 header).
    """
    import io
    arr = np.ascontiguousarray(arr)
    fmt = np.lib.format
    try:
        hdr = io.BytesIO()
        fmt.write_array_header_1_0(hdr, fmt.header_data_from_array_1_0(arr))
    except Exception:
        # A dtype the 1.0 header cannot describe: write it in one piece.
        np.save(path, arr, allow_pickle=False)
        return True
    data = memoryview(arr.reshape(-1).view(np.uint8))
    try:
        with open(path, "wb") as f:
            f.write(hdr.getvalue())
            for i in range(0, len(data), _WRITE_CHUNK_BYTES):
                if stop.is_set():
                    break
                f.write(data[i:i + _WRITE_CHUNK_BYTES])
            else:
                return True
    except BaseException:
        _remove_quietly(path)
        raise
    _remove_quietly(path)
    return False


class SwapManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        *,
        cache_bytes: int = 5_000_000_000,
        max_hot_states: int = 5,              # best-effort "keep last N hot" if you pin them
        write_immediately: bool = True,       # enqueue disk write right away (still non-blocking)
    ):
        if self._initialized:
            return
        self._initialized = True

        self.swap_root = resolve_swap_workdir()
        _cleanup_orphaned_swap_sessions(self.swap_root)
        self.temp_dir = os.path.join(self.swap_root, str(uuid.uuid4()))
        os.makedirs(self.temp_dir, exist_ok=True)
        _write_session_owner(self.temp_dir)
        atexit.register(self.cleanup_all)

        # LRU of in-RAM states: swap_id -> ndarray
        self._cache = OrderedDict()
        self._cache_bytes = int(cache_bytes)
        self._cache_used = 0
        self._cache_lock = threading.Lock()

        # pin counts: swap_id -> int
        self._pins = {}
        self._pins_lock = threading.Lock()

        # bookkeeping
        self._on_disk = set()         # swap_ids we believe have a file
        self._pending_write = set()   # swap_ids currently queued/writing
        self._tombstone = set()       # swap_ids deleted (ignore late writer)

        self._max_hot_states = int(max_hot_states)
        self._write_immediately = bool(write_immediately)

        # background I/O worker
        self._ioq = queue.Queue()
        self._stop = threading.Event()
        self._io_thread = threading.Thread(target=self._io_loop, name="SwapIO", daemon=True)
        self._io_thread.start()

    # ---------------- paths ----------------

    def get_swap_path(self, swap_id: str) -> str:
        return os.path.join(self.temp_dir, f"{swap_id}.npy")

    def _arr_nbytes(self, a: np.ndarray) -> int:
        try:
            return int(a.nbytes)
        except Exception:
            return 0

    # ---------------- pinning ----------------

    def pin_state(self, swap_id: str):
        if not swap_id:
            return
        with self._pins_lock:
            self._pins[swap_id] = self._pins.get(swap_id, 0) + 1

    def unpin_state(self, swap_id: str):
        if not swap_id:
            return
        with self._pins_lock:
            n = self._pins.get(swap_id, 0)
            if n <= 1:
                self._pins.pop(swap_id, None)
            else:
                self._pins[swap_id] = n - 1

    def _is_pinned(self, swap_id: str) -> bool:
        with self._pins_lock:
            return self._pins.get(swap_id, 0) > 0

    # ---------------- RAM cache ----------------

    def _cache_put(self, swap_id: str, arr: np.ndarray):
        if arr is None:
            return
        n = self._arr_nbytes(arr)
        if n <= 0:
            return

        with self._cache_lock:
            # refresh if exists
            old = self._cache.pop(swap_id, None)
            if old is not None:
                self._cache_used -= self._arr_nbytes(old)

            self._cache[swap_id] = arr
            self._cache_used += n

            self._evict_if_needed_locked()

    def _cache_get(self, swap_id: str):
        with self._cache_lock:
            arr = self._cache.pop(swap_id, None)
            if arr is None:
                return None
            self._cache[swap_id] = arr  # MRU
            return arr

    def _evict_if_needed_locked(self):
        """
        Evict LRU until under budget, BUT skip pinned items and skip items
        currently pending write (we want to keep those in RAM until the write lands).
        """
        # quick exit
        if self._cache_used <= self._cache_bytes:
            return

        # We may need multiple passes because pinned items might be at front.
        # We'll iterate through LRU order and evict the first evictable items.
        while self._cache_used > self._cache_bytes and self._cache:
            evicted_any = False

            for k in list(self._cache.keys()):  # LRU -> MRU
                if self._is_pinned(k):
                    continue
                if k in self._pending_write:
                    continue

                v = self._cache.pop(k, None)
                if v is None:
                    continue
                self._cache_used -= self._arr_nbytes(v)
                evicted_any = True

                # If we aren't sure it's on disk yet, ensure it gets written.
                if (k not in self._on_disk) and (k not in self._tombstone):
                    self._enqueue_write(k, v)

                if self._cache_used <= self._cache_bytes:
                    return

            if not evicted_any:
                # Everything is pinned or pending; can't evict further safely.
                return

    # ---------------- async I/O ----------------

    def _enqueue_write(self, swap_id: str, arr: np.ndarray):
        if not swap_id or arr is None:
            return
        if swap_id in self._tombstone:
            return
        if swap_id in self._pending_write:
            return

        # IMPORTANT: writer must not see a mutable view that changes later.
        safe = np.ascontiguousarray(arr)
        self._pending_write.add(swap_id)
        self._ioq.put(("write", swap_id, safe))

    def _enqueue_delete(self, swap_id: str):
        if not swap_id:
            return
        self._ioq.put(("delete", swap_id, None))

    def _io_loop(self):
        while not self._stop.is_set():
            kind, swap_id, payload = self._ioq.get()
            try:
                if kind == "stop":
                    return

                if kind == "write":
                    try:
                        if swap_id in self._tombstone:
                            continue
                        path = self.get_swap_path(swap_id)
                        if _save_npy_interruptible(path, payload, self._stop):
                            self._on_disk.add(swap_id)
                    finally:
                        self._pending_write.discard(swap_id)

                elif kind == "delete":
                    # even if a write races, tombstone prevents future loads
                    self._tombstone.add(swap_id)
                    self._pending_write.discard(swap_id)
                    self._on_disk.discard(swap_id)
                    path = self.get_swap_path(swap_id)
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass

            except Exception as e:
                # Keep it quiet in production; or print in debug builds.
                print(f"[SwapManager] I/O worker error ({kind} {swap_id}): {e}")
            finally:
                self._ioq.task_done()

    # ---------------- public API ----------------

    def save_state(self, image: np.ndarray) -> str | None:
        """
        Fast path:
        - store in RAM cache immediately
        - enqueue write in background (non-blocking) unless configured otherwise
        """
        if image is None:
            return None
        swap_id = uuid.uuid4().hex
        self._tombstone.discard(swap_id)

        try:
            arr = np.asarray(image)
            # Make sure the cached version is contiguous & stable
            arr = np.ascontiguousarray(arr)

            # Put in RAM first (instant)
            self._cache_put(swap_id, arr)

            # Schedule disk write (non-blocking)
            if self._write_immediately:
                self._enqueue_write(swap_id, arr)

            return swap_id
        except Exception as e:
            print(f"[SwapManager] Failed to save state {swap_id}: {e}")
            return None

    def load_state(self, swap_id: str, *, cache: bool = True) -> np.ndarray | None:
        if not swap_id:
            return None
        if swap_id in self._tombstone:
            return None

        # 1) RAM hot
        hot = self._cache_get(swap_id)
        if hot is not None:
            return hot

        # 2) If a write is pending, we *could* wait briefly (optional),
        # but usually undo/redo will be hot due to pinning.
        # We'll just fall through to disk if present.
        path = self.get_swap_path(swap_id)
        if not os.path.exists(path):
            return None

        try:
            # mmap read is fast; materialize to a writable copy only when caching.
            mm = np.load(path, mmap_mode="r", allow_pickle=False)
            if not cache:
                # Read-only view — caller must not mutate. Avoids a RAM copy
                # plus a 5GB-cache insert (critical for project save).
                return mm
            arr = np.array(mm, copy=True)
            self._cache_put(swap_id, arr)
            return arr
        except Exception as e:
            print(f"[SwapManager] Failed to load state {swap_id}: {e}")
            return None

    def delete_state(self, swap_id: str):
        if not swap_id:
            return
        # Remove from RAM immediately
        with self._cache_lock:
            old = self._cache.pop(swap_id, None)
            if old is not None:
                self._cache_used -= self._arr_nbytes(old)

        # Remove pin if any (best-effort)
        with self._pins_lock:
            self._pins.pop(swap_id, None)

        # Enqueue disk delete (non-blocking)
        self._enqueue_delete(swap_id)

    def cleanup_all(self):
        # Stop the writer and wait for it. Windows cannot delete a file that is
        # still open, so removing the folder under a running write used to leave
        # the whole image behind. Queued writes are skipped (the loop checks
        # _stop before taking the next one) and a running one stops at its next
        # slice, so this wait is short.
        try:
            self._stop.set()
            self._ioq.put(("stop", None, None))
        except Exception:
            pass
        try:
            t = getattr(self, "_io_thread", None)
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=10.0)
        except Exception:
            pass

        # clear RAM
        try:
            with self._cache_lock:
                self._cache.clear()
                self._cache_used = 0
        except Exception:
            pass

        try:
            self._remove_session_dir()
        except Exception as e:
            print(f"[SwapManager] Cleanup failed: {e}")

    def _remove_session_dir(self):
        """Delete this session's folder, owner file last: a folder that cannot
        be emptied keeps its owner, so the next start removes it straight away
        (an ownerless one waits _UNOWNED_MIN_IDLE_S)."""
        d = self.temp_dir
        if not d or not os.path.isdir(d):
            return
        stuck = False
        for name in os.listdir(d):
            if name == _OWNER_FILE:
                continue
            p = os.path.join(d, name)
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            except OSError:
                stuck = True
        if stuck:
            print(f"[SwapManager] Could not empty {d}; the next start removes it.")
            return
        _remove_quietly(os.path.join(d, _OWNER_FILE))
        try:
            os.rmdir(d)
        except OSError:
            shutil.rmtree(d, ignore_errors=True)


# Global instance
_swap_mgr = SwapManager(
    cache_bytes=5_000_000_000,
    max_hot_states=5,
    write_immediately=True,
)

def get_swap_manager():
    return _swap_mgr
