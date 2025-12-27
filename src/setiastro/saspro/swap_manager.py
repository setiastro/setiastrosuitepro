import os, shutil, tempfile, uuid, atexit, threading
from collections import OrderedDict
import numpy as np

class SwapManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, *, cache_bytes: int = 1_000_000_000):
        if self._initialized:
            return
        self._initialized = True

        self.temp_dir = os.path.join(
            tempfile.gettempdir(), "SetiAstroSuitePro_Swap", str(uuid.uuid4())
        )
        os.makedirs(self.temp_dir, exist_ok=True)
        atexit.register(self.cleanup_all)

        # LRU of in-RAM states: swap_id -> ndarray
        self._cache = OrderedDict()
        self._cache_bytes = int(cache_bytes)
        self._cache_used = 0
        self._cache_lock = threading.Lock()

    def get_swap_path(self, swap_id: str) -> str:
        # store as .npy (fast + supports mmap)
        return os.path.join(self.temp_dir, f"{swap_id}.npy")

    def _arr_nbytes(self, a: np.ndarray) -> int:
        try:
            return int(a.nbytes)
        except Exception:
            return 0

    def _cache_put(self, swap_id: str, arr: np.ndarray):
        if arr is None:
            return
        n = self._arr_nbytes(arr)
        if n <= 0:
            return

        with self._cache_lock:
            # If already present, refresh
            old = self._cache.pop(swap_id, None)
            if old is not None:
                self._cache_used -= self._arr_nbytes(old)

            self._cache[swap_id] = arr
            self._cache_used += n

            # Evict LRU until under budget
            while self._cache_used > self._cache_bytes and self._cache:
                k, v = self._cache.popitem(last=False)
                self._cache_used -= self._arr_nbytes(v)

    def _cache_get(self, swap_id: str):
        with self._cache_lock:
            arr = self._cache.pop(swap_id, None)
            if arr is None:
                return None
            # move to MRU
            self._cache[swap_id] = arr
            return arr

    def save_state(self, image: np.ndarray) -> str | None:
        swap_id = uuid.uuid4().hex
        path = self.get_swap_path(swap_id)
        try:
            # Write fast .npy
            np.save(path, image, allow_pickle=False)
            # Optionally keep it hot in RAM too (depends how you use it)
            self._cache_put(swap_id, image)
            return swap_id
        except Exception as e:
            print(f"[SwapManager] Failed to save state {swap_id}: {e}")
            return None

    def load_state(self, swap_id: str) -> np.ndarray | None:
        #print("[SwapManager] LOAD", swap_id)
        # First: try RAM
        hot = self._cache_get(swap_id)
        if hot is not None:
            return hot

        path = self.get_swap_path(swap_id)
        if not os.path.exists(path):
            print(f"[SwapManager] Swap file not found: {path}")
            return None

        try:
            # mmap_mode="r" is extremely fast; convert to real ndarray only if needed
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            # If your pipeline needs a writable array, materialize:
            # arr = np.array(arr, copy=True)
            # Cache the loaded array (mmap object still OK to cache; you can decide)
            self._cache_put(swap_id, np.array(arr, copy=False))
            return np.array(arr, copy=False)
        except Exception as e:
            print(f"[SwapManager] Failed to load state {swap_id}: {e}")
            return None

    def delete_state(self, swap_id: str):
        with self._cache_lock:
            old = self._cache.pop(swap_id, None)
            if old is not None:
                self._cache_used -= self._arr_nbytes(old)

        path = self.get_swap_path(swap_id)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[SwapManager] Failed to delete state {swap_id}: {e}")

    def cleanup_all(self):
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"[SwapManager] Cleanup failed: {e}")

# Global instance
_swap_mgr = SwapManager()

def get_swap_manager():
    return _swap_mgr
