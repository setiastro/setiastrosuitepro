"""
Parallelization utilities for Seti Astro Suite Pro.

This module provides standardized parallelization helpers to ensure consistent
behavior across the codebase. It addresses the optimization recommendation #4
from OTTIMIZZAZIONI_PROPOSTE.md.

Key features:
- Centralized worker count calculation
- Consistent ThreadPoolExecutor/ProcessPoolExecutor usage
- Memory-aware parallelization
- Task-type specific worker allocation

Usage:
    from pro.parallel_utils import (
        get_optimal_workers,
        get_io_workers,
        get_cpu_workers,
        run_parallel,
        run_in_thread_pool,
        run_in_process_pool,
    )
    
    # For CPU-bound tasks
    workers = get_cpu_workers()
    
    # For I/O-bound tasks
    workers = get_io_workers()
    
    # Run parallel tasks
    results = run_in_thread_pool(process_image, images, max_workers=workers)
"""

from __future__ import annotations

import os
import sys
import gc
import logging
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    Future,
    as_completed,
)
from typing import (
    Callable,
    TypeVar,
    Iterable,
    Iterator,
    Optional,
    Any,
    List,
    Tuple,
    Union,
)
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# ---------------------------------------------------------------------------
# Constants and Defaults
# ---------------------------------------------------------------------------

# Maximum workers to prevent over-subscription
MAX_WORKERS_CAP = 32

# Default I/O worker multiplier (I/O-bound can use more threads)
IO_WORKER_MULTIPLIER = 2

# Minimum workers
MIN_WORKERS = 1

# Memory threshold for reducing workers (in bytes, ~1GB)
MEMORY_THRESHOLD_LOW = 1 * 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Worker Count Calculation
# ---------------------------------------------------------------------------

def get_cpu_count() -> int:
    """
    Get the number of available CPU cores.
    
    Falls back to a conservative value if detection fails.
    
    Returns:
        Number of CPU cores (at least 1)
    """
    try:
        count = os.cpu_count()
        return max(1, count) if count else 4
    except Exception:
        return 4


def get_available_memory() -> int:
    """
    Get available system memory in bytes.
    
    Returns:
        Available memory in bytes, or 0 if unable to determine
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return 0
    except Exception:
        return 0


def is_memory_constrained(threshold: int = MEMORY_THRESHOLD_LOW) -> bool:
    """
    Check if the system is running low on memory.
    
    Args:
        threshold: Memory threshold in bytes
        
    Returns:
        True if available memory is below threshold
    """
    available = get_available_memory()
    if available == 0:
        return False  # Can't determine, assume OK
    return available < threshold


def get_optimal_workers(
    task_count: Optional[int] = None,
    max_cap: int = MAX_WORKERS_CAP,
    min_workers: int = MIN_WORKERS,
    memory_aware: bool = True,
    io_bound: bool = False,
) -> int:
    """
    Calculate the optimal number of workers for parallel execution.
    
    This function considers:
    - Available CPU cores
    - Task count (no point having more workers than tasks)
    - System memory constraints
    - Task type (I/O-bound vs CPU-bound)
    
    Args:
        task_count: Number of tasks to process (None for default)
        max_cap: Maximum workers cap
        min_workers: Minimum workers to use
        memory_aware: Reduce workers if memory is low
        io_bound: True for I/O-bound tasks (allows more workers)
        
    Returns:
        Optimal number of workers
        
    Example:
        >>> workers = get_optimal_workers(task_count=100, io_bound=True)
        >>> with ThreadPoolExecutor(max_workers=workers) as pool:
        ...     results = list(pool.map(process, items))
    """
    cpu_count = get_cpu_count()
    
    # Start with CPU count
    if io_bound:
        # I/O-bound tasks can use more threads (they spend time waiting)
        workers = min(cpu_count * IO_WORKER_MULTIPLIER, max_cap)
    else:
        # CPU-bound tasks should not exceed core count
        workers = min(cpu_count, max_cap)
    
    # Don't exceed task count
    if task_count is not None and task_count > 0:
        workers = min(workers, task_count)
    
    # Reduce if memory constrained
    if memory_aware and is_memory_constrained():
        workers = max(min_workers, workers // 2)
        logger.debug("Memory constrained, reduced workers to %d", workers)
    
    # Ensure within bounds
    return max(min_workers, min(workers, max_cap))


def get_cpu_workers(task_count: Optional[int] = None) -> int:
    """
    Get optimal worker count for CPU-bound tasks.
    
    Args:
        task_count: Number of tasks to process
        
    Returns:
        Optimal worker count
    """
    return get_optimal_workers(
        task_count=task_count,
        max_cap=MAX_WORKERS_CAP,
        io_bound=False,
    )


def get_io_workers(task_count: Optional[int] = None) -> int:
    """
    Get optimal worker count for I/O-bound tasks.
    
    Args:
        task_count: Number of tasks to process
        
    Returns:
        Optimal worker count (typically higher than CPU workers)
    """
    return get_optimal_workers(
        task_count=task_count,
        max_cap=MAX_WORKERS_CAP * 2,  # Allow more for I/O
        io_bound=True,
    )


def get_image_processing_workers(
    image_count: int,
    image_size_mb: float = 100.0,
) -> int:
    """
    Get optimal worker count for image processing tasks.
    
    Takes into account typical memory usage per image.
    
    Args:
        image_count: Number of images to process
        image_size_mb: Estimated size per image in MB
        
    Returns:
        Optimal worker count
    """
    cpu_count = get_cpu_count()
    
    # Estimate memory needed per worker
    available_mb = get_available_memory() / (1024 * 1024)
    if available_mb > 0:
        # Reserve 2GB for system, rest for workers
        usable_mb = max(0, available_mb - 2048)
        memory_limited_workers = max(1, int(usable_mb / image_size_mb))
    else:
        memory_limited_workers = cpu_count
    
    # Use the more restrictive limit
    workers = min(cpu_count, memory_limited_workers, image_count)
    
    return max(1, workers)


# ---------------------------------------------------------------------------
# Parallel Execution Helpers
# ---------------------------------------------------------------------------

def run_in_thread_pool(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[R]:
    """
    Execute a function across items using ThreadPoolExecutor.
    
    Best for I/O-bound tasks (file operations, network requests).
    
    Args:
        func: Function to apply to each item
        items: Items to process
        max_workers: Max parallel workers (auto-calculated if None)
        progress_callback: Optional callback(completed, total)
        
    Returns:
        List of results in original order
        
    Example:
        >>> def load_image(path):
        ...     return Image.open(path)
        >>> images = run_in_thread_pool(load_image, file_paths, max_workers=8)
    """
    items_list = list(items)
    total = len(items_list)
    
    if total == 0:
        return []
    
    if max_workers is None:
        max_workers = get_io_workers(total)
    
    results: List[Optional[R]] = [None] * total
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with their indices
        future_to_idx = {
            executor.submit(func, item): idx
            for idx, item in enumerate(items_list)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("Task %d failed: %s", idx, e)
                raise
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
    
    return results  # type: ignore


def run_in_process_pool(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[R]:
    """
    Execute a function across items using ProcessPoolExecutor.
    
    Best for CPU-bound tasks that benefit from true parallelism.
    Note: func and items must be picklable.
    
    Args:
        func: Function to apply to each item (must be picklable)
        items: Items to process (must be picklable)
        max_workers: Max parallel workers (auto-calculated if None)
        progress_callback: Optional callback(completed, total)
        
    Returns:
        List of results in original order
        
    Example:
        >>> def heavy_compute(data):
        ...     return np.fft.fft2(data)
        >>> results = run_in_process_pool(heavy_compute, data_arrays)
    """
    items_list = list(items)
    total = len(items_list)
    
    if total == 0:
        return []
    
    if max_workers is None:
        max_workers = get_cpu_workers(total)
    
    results: List[Optional[R]] = [None] * total
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(func, item): idx
            for idx, item in enumerate(items_list)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("Task %d failed: %s", idx, e)
                raise
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
    
    return results  # type: ignore


@contextmanager
def thread_pool(max_workers: Optional[int] = None, io_bound: bool = True):
    """
    Context manager for ThreadPoolExecutor with automatic worker count.
    
    Args:
        max_workers: Max workers (auto-calculated if None)
        io_bound: Whether tasks are I/O-bound
        
    Yields:
        ThreadPoolExecutor instance
        
    Example:
        >>> with thread_pool(io_bound=True) as pool:
        ...     futures = [pool.submit(load_file, f) for f in files]
        ...     results = [f.result() for f in futures]
    """
    if max_workers is None:
        max_workers = get_io_workers() if io_bound else get_cpu_workers()
    
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


@contextmanager
def process_pool(max_workers: Optional[int] = None):
    """
    Context manager for ProcessPoolExecutor with automatic worker count.
    
    Args:
        max_workers: Max workers (auto-calculated if None)
        
    Yields:
        ProcessPoolExecutor instance
        
    Example:
        >>> with process_pool() as pool:
        ...     futures = [pool.submit(compute, data) for data in datasets]
        ...     results = [f.result() for f in futures]
    """
    if max_workers is None:
        max_workers = get_cpu_workers()
    
    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def batch_items(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """
    Split items into batches of specified size.
    
    Args:
        items: List of items to batch
        batch_size: Maximum items per batch
        
    Yields:
        Batches of items
        
    Example:
        >>> for batch in batch_items(images, batch_size=10):
        ...     process_batch(batch)
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def run_in_batches(
    func: Callable[[List[T]], List[R]],
    items: List[T],
    batch_size: int,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[R]:
    """
    Process items in batches using parallel execution.
    
    Useful for reducing overhead when processing many small items.
    
    Args:
        func: Function that processes a batch and returns list of results
        items: All items to process
        batch_size: Items per batch
        max_workers: Max parallel workers
        progress_callback: Optional callback(completed_items, total_items)
        
    Returns:
        Flattened list of all results
    """
    batches = list(batch_items(items, batch_size))
    total_items = len(items)
    completed_items = 0
    all_results: List[R] = []
    
    if max_workers is None:
        max_workers = get_cpu_workers(len(batches))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(func, batch): batch
            for batch in batches
        }
        
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                logger.error("Batch processing failed: %s", e)
                raise
            
            completed_items += len(batch)
            if progress_callback:
                progress_callback(completed_items, total_items)
    
    return all_results


# ---------------------------------------------------------------------------
# Cleanup and Memory Management
# ---------------------------------------------------------------------------

def cleanup_after_parallel(force_gc: bool = True) -> None:
    """
    Clean up after parallel processing.
    
    Should be called after large parallel operations to free memory.
    
    Args:
        force_gc: Whether to force garbage collection
    """
    if force_gc:
        gc.collect()


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    'MAX_WORKERS_CAP',
    'IO_WORKER_MULTIPLIER',
    'MIN_WORKERS',
    
    # Worker calculation
    'get_cpu_count',
    'get_available_memory',
    'is_memory_constrained',
    'get_optimal_workers',
    'get_cpu_workers',
    'get_io_workers',
    'get_image_processing_workers',
    
    # Execution helpers
    'run_in_thread_pool',
    'run_in_process_pool',
    'thread_pool',
    'process_pool',
    
    # Batch processing
    'batch_items',
    'run_in_batches',
    
    # Cleanup
    'cleanup_after_parallel',
]
