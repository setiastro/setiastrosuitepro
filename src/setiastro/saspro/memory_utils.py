# pro/memory_utils.py
"""
Memory management utilities for Seti Astro Suite Pro.

Provides:
- Memory-mapped array creation for large datasets
- Reusable buffer pools to reduce allocation overhead
- Lazy image loading with preview-first strategy
"""
from __future__ import annotations
import os
import tempfile
import hashlib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import threading
import weakref


# ============================================================================
# COPY-ON-WRITE ARRAY WRAPPER
# ============================================================================

class CopyOnWriteArray:
    """
    A wrapper that defers copying a numpy array until it's actually modified.
    
    This is used to optimize duplicate_document: instead of copying the
    full image immediately, we share the source array and only copy when
    the duplicate is about to be modified (via apply_edit).
    
    Usage:
        cow = CopyOnWriteArray(source_array)
        arr = cow.get_array()  # Returns view of source (no copy)
        arr = cow.get_writable()  # Forces copy if not already copied
    """
    
    __slots__ = ('_source', '_copy', '_lock', '_copied')
    
    def __init__(self, source: np.ndarray):
        """
        Initialize with source array (no copy made yet).
        
        Args:
            source: The source numpy array to potentially copy later
        """
        self._source = source
        self._copy: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._copied = False
    
    def get_array(self) -> np.ndarray:
        """
        Get the array for read-only access.
        
        Returns the copy if one was made, otherwise the source.
        This does NOT make a copy.
        """
        if self._copied:
            return self._copy
        return self._source
    
    def get_writable(self) -> np.ndarray:
        """
        Get a writable copy of the array.
        
        Forces a copy if one hasn't been made yet.
        Thread-safe.
        """
        if self._copied:
            return self._copy
        
        with self._lock:
            # Double-check after acquiring lock
            if self._copied:
                return self._copy
            
            # Make the copy now
            if self._source is not None:
                self._copy = self._source.copy()
            else:
                self._copy = None
            self._copied = True
            self._source = None  # Release reference to source
            return self._copy
    
    @property
    def is_copied(self) -> bool:
        """Check if a copy has been made."""
        return self._copied
    
    @property
    def shape(self) -> tuple:
        """Get shape of the underlying array."""
        arr = self.get_array()
        return arr.shape if arr is not None else ()
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        arr = self.get_array()
        return arr.ndim if arr is not None else 0
    
    @property
    def dtype(self):
        """Get dtype of the underlying array."""
        arr = self.get_array()
        return arr.dtype if arr is not None else None
    
    def __array__(self, dtype=None):
        """Support numpy array conversion."""
        arr = self.get_array()
        if dtype is None:
            return arr
        return arr.astype(dtype)


# ============================================================================
# LRU DICTIONARY FOR BOUNDED CACHES
# ============================================================================

from collections import OrderedDict

class LRUDict(OrderedDict):
    """
    Simple LRU cache based on OrderedDict.
    When maxsize is exceeded, oldest items are evicted.
    Thread-safe for basic operations.
    """
    __slots__ = ('maxsize', '_lock')
    
    def __init__(self, maxsize: int = 500):
        super().__init__()
        self.maxsize = maxsize
        self._lock = threading.RLock()
    
    def __getitem__(self, key):
        with self._lock:
            # Move to end on access (most recently used)
            self.move_to_end(key)
            return super().__getitem__(key)
    
    def get(self, key, default=None):
        with self._lock:
            if key in self:
                self.move_to_end(key)
                return super().__getitem__(key)
            return default
    
    def __setitem__(self, key, value):
        with self._lock:
            if key in self:
                self.move_to_end(key)
            super().__setitem__(key, value)
            # Evict oldest if over limit
            while len(self) > self.maxsize:
                self.popitem(last=False)  # Remove oldest


# ============================================================================
# MEMORY-MAPPED ARRAY UTILITIES
# ============================================================================

_TEMP_DIR: Optional[Path] = None
_MEMMAP_FILES: weakref.WeakSet = weakref.WeakSet()


def get_temp_dir() -> Path:
    """Get or create the temporary directory for memory-mapped files."""
    global _TEMP_DIR
    if _TEMP_DIR is None or not _TEMP_DIR.exists():
        _TEMP_DIR = Path(tempfile.mkdtemp(prefix="sasp_memmap_"))
    return _TEMP_DIR


def create_memmap_array(
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    mode: str = 'w+',
    prefix: str = "array_"
) -> Tuple[np.memmap, Path]:
    """
    Create a memory-mapped array backed by a temporary file.
    
    Args:
        shape: Shape of the array
        dtype: Data type (default float32)
        mode: File mode ('w+' for read/write, 'r+' for existing)
        prefix: Prefix for temp file name
        
    Returns:
        Tuple of (memmap array, path to backing file)
    """
    temp_dir = get_temp_dir()
    temp_file = tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=".npy",
        dir=temp_dir,
        delete=False
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    mm = np.memmap(str(temp_path), dtype=dtype, mode=mode, shape=shape)
    return mm, temp_path


def cleanup_memmap(mm: np.memmap, path: Path) -> None:
    """
    Properly cleanup a memory-mapped array and its backing file.
    
    Args:
        mm: The memmap array to cleanup
        path: Path to the backing file
    """
    try:
        del mm
    except Exception:
        pass
    
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def should_use_memmap(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> bool:
    """
    Determine if memmap should be used based on array size.
    
    Uses memmap for arrays larger than 500MB to reduce RAM usage.
    
    Args:
        shape: Shape of the array
        dtype: Data type
        
    Returns:
        True if memmap should be used
    """
    itemsize = np.dtype(dtype).itemsize
    size_bytes = int(np.prod(shape)) * itemsize
    threshold = 500 * 1024 * 1024  # 500 MB
    return size_bytes > threshold


def smart_zeros(
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    force_memmap: bool = False
) -> Tuple[np.ndarray, Optional[Path]]:
    """
    Create a zeros array, using memmap for large arrays.
    
    Args:
        shape: Shape of the array
        dtype: Data type
        force_memmap: Force use of memmap regardless of size
        
    Returns:
        Tuple of (array, optional path to memmap file)
    """
    if force_memmap or should_use_memmap(shape, dtype):
        mm, path = create_memmap_array(shape, dtype, 'w+', "zeros_")
        mm[:] = 0
        return mm, path
    else:
        return np.zeros(shape, dtype=dtype), None


def smart_empty(
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    force_memmap: bool = False
) -> Tuple[np.ndarray, Optional[Path]]:
    """
    Create an empty array, using memmap for large arrays.
    
    Args:
        shape: Shape of the array
        dtype: Data type
        force_memmap: Force use of memmap regardless of size
        
    Returns:
        Tuple of (array, optional path to memmap file)
    """
    if force_memmap or should_use_memmap(shape, dtype):
        mm, path = create_memmap_array(shape, dtype, 'w+', "empty_")
        return mm, path
    else:
        return np.empty(shape, dtype=dtype), None


# ============================================================================
# BUFFER POOL FOR REUSABLE MEMORY
# ============================================================================

class BufferPool:
    """
    A pool of reusable numpy buffers to reduce allocation overhead.
    
    Thread-safe buffer management for frequently allocated arrays.
    """
    
    def __init__(self, max_buffers: int = 8):
        """
        Initialize buffer pool.
        
        Args:
            max_buffers: Maximum number of buffers to keep per shape/dtype
        """
        self._pools: Dict[Tuple, list] = {}
        self._lock = threading.Lock()
        self._max_buffers = max_buffers
    
    def _key(self, shape: Tuple[int, ...], dtype: np.dtype) -> Tuple:
        """Create a hashable key for shape/dtype combination."""
        return (shape, str(dtype))
    
    def get(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Get a buffer from the pool or create a new one.
        
        The buffer contents are NOT zeroed - caller should initialize if needed.
        
        Args:
            shape: Desired shape
            dtype: Desired dtype
            
        Returns:
            A numpy array of the requested shape/dtype
        """
        key = self._key(shape, dtype)
        
        with self._lock:
            pool = self._pools.get(key, [])
            if pool:
                return pool.pop()
        
        # Create new buffer
        return np.empty(shape, dtype=dtype)
    
    def get_zeros(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Get a zeroed buffer from the pool.
        
        Args:
            shape: Desired shape
            dtype: Desired dtype
            
        Returns:
            A zeroed numpy array of the requested shape/dtype
        """
        buf = self.get(shape, dtype)
        buf.fill(0)
        return buf
    
    def release(self, buf: np.ndarray) -> None:
        """
        Return a buffer to the pool for reuse.
        
        Args:
            buf: Buffer to return
        """
        if buf is None:
            return
            
        key = self._key(buf.shape, buf.dtype)
        
        with self._lock:
            if key not in self._pools:
                self._pools[key] = []
            
            if len(self._pools[key]) < self._max_buffers:
                self._pools[key].append(buf)
    
    def clear(self) -> None:
        """Clear all buffers from the pool."""
        with self._lock:
            self._pools.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get statistics about pool usage."""
        with self._lock:
            return {
                "num_shapes": len(self._pools),
                "total_buffers": sum(len(p) for p in self._pools.values()),
                "shapes": list(self._pools.keys())
            }


# Global buffer pool instance
_global_pool: Optional[BufferPool] = None


def get_buffer_pool() -> BufferPool:
    """Get the global buffer pool instance."""
    global _global_pool
    if _global_pool is None:
        _global_pool = BufferPool(max_buffers=8)
    return _global_pool


# ============================================================================
# THUMBNAIL/PREVIEW CACHE
# ============================================================================

class ThumbnailCache:
    """
    Disk-based cache for image thumbnails/previews.
    
    Speeds up repeated loading of the same images by caching
    downscaled preview versions.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size_mb: int = 500):
        """
        Initialize thumbnail cache.
        
        Args:
            cache_dir: Directory for cache files (default: temp dir)
            max_size_mb: Maximum cache size in MB
        """
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "sasp_thumb_cache"
        
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._memory_cache: Dict[str, np.ndarray] = {}  # LRU in-memory cache
        self._max_memory_items = 50
    
    def _get_cache_key(self, path: str, target_size: Tuple[int, int]) -> str:
        """Generate a unique cache key for a file and target size."""
        # Include file path, mtime, and target size in hash
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0
        
        key_str = f"{path}|{mtime}|{target_size[0]}x{target_size[1]}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a key."""
        return self._cache_dir / f"{key}.npy"
    
    def get(self, path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Get a cached thumbnail if available.
        
        Args:
            path: Original image path
            target_size: Target thumbnail size (width, height)
            
        Returns:
            Cached thumbnail array or None
        """
        key = self._get_cache_key(path, target_size)
        
        # Check in-memory cache first
        with self._lock:
            if key in self._memory_cache:
                return self._memory_cache[key].copy()
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                thumb = np.load(str(cache_path))
                # Add to memory cache
                with self._lock:
                    self._memory_cache[key] = thumb
                    self._trim_memory_cache()
                return thumb.copy()
            except Exception:
                # Corrupted cache file, remove it
                try:
                    cache_path.unlink()
                except Exception:
                    pass
        
        return None
    
    def put(self, path: str, target_size: Tuple[int, int], thumb: np.ndarray) -> None:
        """
        Store a thumbnail in the cache.
        
        Args:
            path: Original image path
            target_size: Target thumbnail size
            thumb: Thumbnail array to cache
        """
        key = self._get_cache_key(path, target_size)
        
        # Store in memory cache
        with self._lock:
            self._memory_cache[key] = thumb.copy()
            self._trim_memory_cache()
        
        # Store on disk
        cache_path = self._get_cache_path(key)
        try:
            np.save(str(cache_path), thumb)
        except Exception:
            pass
        
        # Trim disk cache if needed
        self._trim_disk_cache()
    
    def _trim_memory_cache(self) -> None:
        """Trim memory cache to max size (caller must hold lock)."""
        while len(self._memory_cache) > self._max_memory_items:
            # Remove oldest item (first key)
            oldest = next(iter(self._memory_cache))
            del self._memory_cache[oldest]
    
    def _trim_disk_cache(self) -> None:
        """Trim disk cache to max size."""
        try:
            files = list(self._cache_dir.glob("*.npy"))
            total_size = sum(f.stat().st_size for f in files)
            
            if total_size > self._max_size:
                # Sort by access time, oldest first
                files.sort(key=lambda f: f.stat().st_atime)
                
                while total_size > self._max_size * 0.8 and files:  # Trim to 80%
                    oldest = files.pop(0)
                    try:
                        size = oldest.stat().st_size
                        oldest.unlink()
                        total_size -= size
                    except Exception:
                        pass
        except Exception:
            pass
    
    def clear(self) -> None:
        """Clear all cached thumbnails."""
        with self._lock:
            self._memory_cache.clear()
        
        try:
            for f in self._cache_dir.glob("*.npy"):
                try:
                    f.unlink()
                except Exception:
                    pass
        except Exception:
            pass


# Global thumbnail cache instance
_thumb_cache: Optional[ThumbnailCache] = None


def get_thumbnail_cache() -> ThumbnailCache:
    """Get the global thumbnail cache instance."""
    global _thumb_cache
    if _thumb_cache is None:
        _thumb_cache = ThumbnailCache()
    return _thumb_cache


# ============================================================================
# LAZY IMAGE LOADER
# ============================================================================

class LazyImage:
    """
    Lazy image loader that loads full resolution on demand.
    
    Initially loads only a preview/thumbnail, deferring full
    resolution loading until actually needed.
    """
    
    def __init__(
        self,
        path: str,
        preview_size: Tuple[int, int] = (512, 512),
        load_preview_fn: Optional[callable] = None,
        load_full_fn: Optional[callable] = None
    ):
        """
        Initialize lazy image.
        
        Args:
            path: Path to the image file
            preview_size: Size for preview image
            load_preview_fn: Function to load preview (path, size) -> array
            load_full_fn: Function to load full image (path) -> array
        """
        self.path = path
        self.preview_size = preview_size
        self._preview: Optional[np.ndarray] = None
        self._full: Optional[np.ndarray] = None
        self._load_preview_fn = load_preview_fn
        self._load_full_fn = load_full_fn
        self._lock = threading.Lock()
    
    @property
    def preview(self) -> Optional[np.ndarray]:
        """Get preview image, loading if necessary."""
        if self._preview is None and self._load_preview_fn is not None:
            with self._lock:
                if self._preview is None:
                    # Check cache first
                    cache = get_thumbnail_cache()
                    cached = cache.get(self.path, self.preview_size)
                    if cached is not None:
                        self._preview = cached
                    else:
                        self._preview = self._load_preview_fn(self.path, self.preview_size)
                        if self._preview is not None:
                            cache.put(self.path, self.preview_size, self._preview)
        return self._preview
    
    @property
    def full(self) -> Optional[np.ndarray]:
        """Get full resolution image, loading if necessary."""
        if self._full is None and self._load_full_fn is not None:
            with self._lock:
                if self._full is None:
                    self._full = self._load_full_fn(self.path)
        return self._full
    
    @property
    def is_full_loaded(self) -> bool:
        """Check if full resolution image is loaded."""
        return self._full is not None
    
    def unload_full(self) -> None:
        """Unload full resolution to free memory."""
        with self._lock:
            self._full = None
    
    def unload_all(self) -> None:
        """Unload all images to free memory."""
        with self._lock:
            self._preview = None
            self._full = None


# ============================================================================
# CLEANUP UTILITIES
# ============================================================================

def cleanup_temp_files() -> None:
    """Cleanup all temporary memory-mapped files."""
    global _TEMP_DIR
    if _TEMP_DIR is not None and _TEMP_DIR.exists():
        try:
            import shutil
            shutil.rmtree(_TEMP_DIR, ignore_errors=True)
        except Exception:
            pass
        _TEMP_DIR = None


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0
