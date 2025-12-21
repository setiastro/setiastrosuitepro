# pro/logging_config.py
"""
Centralized logging configuration for Seti Astro Suite Pro.

Provides:
- Structured logging with consistent formatting
- Performance logging decorators
- Context managers for timed operations
- Log rotation and file output
"""
from __future__ import annotations

import logging
import sys
import time
import functools
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar
from contextlib import contextmanager

# For type hints
F = TypeVar('F', bound=Callable[..., Any])


# ============================================================================
# Custom Formatters
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    """
    
    COLORS = {
        logging.DEBUG: '\033[36m',      # Cyan
        logging.INFO: '\033[32m',       # Green
        logging.WARNING: '\033[33m',    # Yellow
        logging.ERROR: '\033[31m',      # Red
        logging.CRITICAL: '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """
    Structured formatter for file output.
    Includes timestamp, level, module, function, and message.
    """
    
    def format(self, record):
        # Add extra context if available
        extra = ""
        if hasattr(record, 'duration_ms'):
            extra += f" [duration={record.duration_ms:.2f}ms]"
        if hasattr(record, 'memory_mb'):
            extra += f" [memory={record.memory_mb:.1f}MB]"
        if hasattr(record, 'context'):
            extra += f" [context={record.context}]"
        
        record.extra = extra
        return super().format(record)


# ============================================================================
# Logger Setup
# ============================================================================

def setup_logging(
    app_name: str = "SetiAstroSuite",
    log_dir: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        app_name: Application name for the logger
        log_dir: Directory for log files (default: ~/.setiastrosuite/logs)
        console_level: Logging level for console output
        file_level: Logging level for file output
        max_file_size_mb: Maximum size of each log file
        backup_count: Number of backup files to keep
        enable_colors: Whether to use colored console output
        
    Returns:
        Configured logger
    """
    # Create or get the main logger
    logger = logging.getLogger(app_name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)  # Capture all, filter at handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    if enable_colors and sys.stdout.isatty():
        console_format = ColoredFormatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir is not None:
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Use RotatingFileHandler if available
            try:
                from logging.handlers import RotatingFileHandler
                
                log_file = log_dir / f"{app_name.lower()}.log"
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_file_size_mb * 1024 * 1024,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
            except ImportError:
                # Fallback to regular FileHandler
                log_file = log_dir / f"{app_name.lower()}_{datetime.now():%Y%m%d}.log"
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
            
            file_handler.setLevel(file_level)
            file_format = StructuredFormatter(
                '%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s%(extra)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    return logger


def get_logger(name: str = "SetiAstroSuite") -> logging.Logger:
    """
    Get a logger with the given name.
    
    If the logger doesn't exist, it will be created as a child of the main logger.
    
    Args:
        name: Logger name (e.g., "SetiAstroSuite.stacking")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# Performance Logging Decorators
# ============================================================================

def log_performance(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    log_args: bool = False,
    log_result: bool = False
) -> Callable[[F], F]:
    """
    Decorator to log function execution time.
    
    Usage:
        @log_performance()
        def my_function(x, y):
            ...
        
        @log_performance(logger=my_logger, level=logging.INFO)
        def important_function():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger()
            
            # Log start
            msg_parts = [f"Starting {func.__qualname__}"]
            if log_args:
                msg_parts.append(f"args={args}, kwargs={kwargs}")
            _logger.log(level, " ".join(msg_parts))
            
            # Execute and time
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Log completion
                msg = f"Completed {func.__qualname__} in {duration_ms:.2f}ms"
                if log_result:
                    msg += f" result={result}"
                _logger.log(level, msg)
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                _logger.error(
                    f"Failed {func.__qualname__} after {duration_ms:.2f}ms: {type(e).__name__}: {e}"
                )
                raise
        
        return wrapper  # type: ignore
    
    return decorator


def log_slow_operations(
    threshold_ms: float = 1000,
    logger: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator to log operations that exceed a time threshold.
    
    Usage:
        @log_slow_operations(threshold_ms=500)
        def possibly_slow_function():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            if duration_ms >= threshold_ms:
                _logger = logger or get_logger()
                _logger.warning(
                    f"Slow operation: {func.__qualname__} took {duration_ms:.2f}ms "
                    f"(threshold: {threshold_ms}ms)"
                )
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator


# ============================================================================
# Context Managers
# ============================================================================

@contextmanager
def log_timing(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG
):
    """
    Context manager for logging operation timing.
    
    Usage:
        with log_timing("Image processing"):
            process_image(data)
    """
    _logger = logger or get_logger()
    _logger.log(level, f"Starting: {operation_name}")
    
    start_time = time.perf_counter()
    try:
        yield
        duration_ms = (time.perf_counter() - start_time) * 1000
        _logger.log(level, f"Completed: {operation_name} in {duration_ms:.2f}ms")
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        _logger.error(f"Failed: {operation_name} after {duration_ms:.2f}ms: {e}")
        raise


@contextmanager
def log_memory(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG
):
    """
    Context manager for logging memory usage changes.
    
    Usage:
        with log_memory("Loading large dataset"):
            load_dataset(path)
    """
    try:
        import psutil
        process = psutil.Process()
        has_psutil = True
    except ImportError:
        has_psutil = False
    
    _logger = logger or get_logger()
    
    if not has_psutil:
        yield
        return
    
    mem_before = process.memory_info().rss / (1024 * 1024)
    _logger.log(level, f"Starting: {operation_name} (memory: {mem_before:.1f}MB)")
    
    try:
        yield
        mem_after = process.memory_info().rss / (1024 * 1024)
        delta = mem_after - mem_before
        sign = "+" if delta >= 0 else ""
        _logger.log(
            level,
            f"Completed: {operation_name} (memory: {mem_after:.1f}MB, {sign}{delta:.1f}MB)"
        )
    except Exception as e:
        mem_after = process.memory_info().rss / (1024 * 1024)
        _logger.error(f"Failed: {operation_name} (memory: {mem_after:.1f}MB): {e}")
        raise


# ============================================================================
# Progress Logger
# ============================================================================

class ProgressLogger:
    """
    Logger for tracking progress of long operations.
    
    Usage:
        with ProgressLogger("Processing files", total=100) as progress:
            for i, file in enumerate(files):
                process(file)
                progress.update(i + 1)
    """
    
    def __init__(
        self,
        operation_name: str,
        total: int,
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10,  # Log every N percent
        level: int = logging.INFO
    ):
        self.operation_name = operation_name
        self.total = total
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        self.level = level
        self.current = 0
        self.last_logged_percent = -1
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.log(self.level, f"Starting: {self.operation_name} (0/{self.total})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            self.logger.log(
                self.level,
                f"Completed: {self.operation_name} ({self.total}/{self.total}) "
                f"in {duration:.2f}s"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation_name} at {self.current}/{self.total} "
                f"after {duration:.2f}s: {exc_val}"
            )
        return False
    
    def update(self, current: int, message: str = ""):
        """Update progress and log if interval reached."""
        self.current = current
        
        if self.total <= 0:
            return
        
        percent = int((current / self.total) * 100)
        
        # Log at intervals
        if percent >= self.last_logged_percent + self.log_interval:
            self.last_logged_percent = (percent // self.log_interval) * self.log_interval
            
            elapsed = time.perf_counter() - self.start_time if self.start_time else 0
            eta = (elapsed / current * (self.total - current)) if current > 0 else 0
            
            msg = f"Progress: {self.operation_name} {percent}% ({current}/{self.total})"
            if message:
                msg += f" - {message}"
            msg += f" [elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s]"
            
            self.logger.log(self.level, msg)


# ============================================================================
# Convenience Exports
# ============================================================================

# Pre-configured logger for common use
_main_logger: Optional[logging.Logger] = None


def init_app_logging(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Initialize application logging.
    
    Call this once at application startup.
    
    Args:
        log_dir: Directory for log files (optional)
        
    Returns:
        Main application logger
    """
    global _main_logger
    
    if log_dir is None:
        # Default to user's home directory
        log_dir = Path.home() / ".setiastrosuite" / "logs"
    
    _main_logger = setup_logging(
        app_name="SetiAstroSuite",
        log_dir=log_dir,
        console_level=logging.INFO,
        file_level=logging.DEBUG
    )
    
    return _main_logger


def app_logger() -> logging.Logger:
    """Get the main application logger."""
    global _main_logger
    if _main_logger is None:
        _main_logger = setup_logging(app_name="SetiAstroSuite")
    return _main_logger
