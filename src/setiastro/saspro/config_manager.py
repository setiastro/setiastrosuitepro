# pro/config_manager.py
"""
Centralized configuration manager for Seti Astro Suite Pro.

Provides type-safe access to application settings with validation,
defaults, and change notifications.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, TypeVar, Generic, Callable, List
from PyQt6.QtCore import QSettings, QObject, pyqtSignal
import json
import os

T = TypeVar('T')


class ConfigValue(Generic[T]):
    """
    Descriptor for a typed configuration value.
    
    Usage:
        class MyConfig(ConfigManager):
            my_setting = ConfigValue("my_setting", default=10, type_=int)
    """
    def __init__(
        self,
        key: str,
        default: T,
        type_: type = str,
        validator: Callable[[T], bool] | None = None,
        description: str = ""
    ):
        self.key = key
        self.default = default
        self.type_ = type_
        self.validator = validator
        self.description = description

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None) -> T:
        if obj is None:
            return self  # type: ignore
        return obj.get(self.key, self.default, self.type_)

    def __set__(self, obj, value: T):
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value for {self.key}: {value}")
        obj.set(self.key, value)


class ConfigManager(QObject):
    """
    Centralized configuration manager with type safety.
    
    Wraps QSettings with:
    - Type-safe get/set
    - Default values
    - Change notifications
    - Validation
    
    Usage:
        config = ConfigManager.instance()
        config.set("my_key", 42)
        value = config.get("my_key", default=0, type_=int)
        
    Or with descriptors:
        class AppConfig(ConfigManager):
            chunk_height = ConfigValue("stacking/chunk_height", default=512, type_=int)
        
        config = AppConfig.instance()
        print(config.chunk_height)
        config.chunk_height = 1024
    """
    
    # Singleton instance
    _instance: Optional['ConfigManager'] = None
    
    # Signal emitted when any setting changes
    settingChanged = pyqtSignal(str, object)  # key, new_value
    
    def __init__(self, organization: str = "SetiAstro", application: str = "SetiAstroSuitePro"):
        super().__init__()
        self._settings = QSettings(organization, application)
        self._cache: Dict[str, Any] = {}
        self._listeners: Dict[str, List[Callable]] = {}
    
    @classmethod
    def instance(cls) -> 'ConfigManager':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton (mainly for testing)."""
        cls._instance = None
    
    def get(self, key: str, default: T = None, type_: type = str) -> T:
        """
        Get a configuration value with type conversion.
        
        Args:
            key: The setting key (can use "/" for groups, e.g., "stacking/chunk_size")
            default: Default value if key doesn't exist
            type_: Expected type (int, float, bool, str, list, dict)
            
        Returns:
            The value converted to the specified type
        """
        # Check cache first
        cache_key = f"{key}:{type_.__name__}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        value = self._settings.value(key, default)
        
        # Convert to requested type
        if value is None:
            return default
        
        try:
            if type_ == bool:
                # QSettings stores bools as strings sometimes
                if isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes')
                else:
                    value = bool(value)
            elif type_ == int:
                value = int(value)
            elif type_ == float:
                value = float(value)
            elif type_ == list:
                if isinstance(value, str):
                    value = json.loads(value) if value else []
                elif not isinstance(value, list):
                    value = list(value) if value else []
            elif type_ == dict:
                if isinstance(value, str):
                    value = json.loads(value) if value else {}
                elif not isinstance(value, dict):
                    value = {}
            else:
                value = type_(value)
        except (ValueError, TypeError, json.JSONDecodeError):
            value = default
        
        # Cache the result
        self._cache[cache_key] = value
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The setting key
            value: The value to store
        """
        # Invalidate cache for this key
        for cache_key in list(self._cache.keys()):
            if cache_key.startswith(f"{key}:"):
                del self._cache[cache_key]
        
        # Convert complex types to JSON for storage
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        
        self._settings.setValue(key, value)
        
        # Emit signal
        self.settingChanged.emit(key, value)
        
        # Notify listeners
        if key in self._listeners:
            for callback in self._listeners[key]:
                try:
                    callback(value)
                except Exception:
                    pass
    
    def remove(self, key: str) -> None:
        """Remove a setting."""
        # Invalidate cache
        for cache_key in list(self._cache.keys()):
            if cache_key.startswith(f"{key}:"):
                del self._cache[cache_key]
        
        self._settings.remove(key)
    
    def contains(self, key: str) -> bool:
        """Check if a setting exists."""
        return self._settings.contains(key)
    
    def all_keys(self) -> List[str]:
        """Get all setting keys."""
        return self._settings.allKeys()
    
    def clear(self) -> None:
        """Clear all settings."""
        self._cache.clear()
        self._settings.clear()
    
    def sync(self) -> None:
        """Force sync to disk."""
        self._settings.sync()
    
    def add_listener(self, key: str, callback: Callable[[Any], None]) -> None:
        """
        Add a listener for changes to a specific key.
        
        Args:
            key: The setting key to watch
            callback: Function to call when value changes
        """
        if key not in self._listeners:
            self._listeners[key] = []
        self._listeners[key].append(callback)
    
    def remove_listener(self, key: str, callback: Callable[[Any], None]) -> None:
        """Remove a listener."""
        if key in self._listeners and callback in self._listeners[key]:
            self._listeners[key].remove(callback)
    
    def begin_group(self, prefix: str) -> None:
        """Begin a settings group (for hierarchical settings)."""
        self._settings.beginGroup(prefix)
    
    def end_group(self) -> None:
        """End the current settings group."""
        self._settings.endGroup()
    
    def get_group(self, prefix: str) -> Dict[str, Any]:
        """
        Get all settings in a group as a dictionary.
        
        Args:
            prefix: The group prefix
            
        Returns:
            Dictionary of key -> value for all settings in the group
        """
        result = {}
        self._settings.beginGroup(prefix)
        for key in self._settings.childKeys():
            result[key] = self._settings.value(key)
        self._settings.endGroup()
        return result
    
    def set_group(self, prefix: str, values: Dict[str, Any]) -> None:
        """
        Set multiple settings in a group.
        
        Args:
            prefix: The group prefix
            values: Dictionary of key -> value to set
        """
        self._settings.beginGroup(prefix)
        for key, value in values.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            self._settings.setValue(key, value)
        self._settings.endGroup()
        self._cache.clear()  # Invalidate all cache


# Convenience function
def get_config() -> ConfigManager:
    """Get the global configuration manager."""
    return ConfigManager.instance()


# ============================================================================
# Application-specific configuration class with typed properties
# ============================================================================

class AppConfig(ConfigManager):
    """
    Application configuration with typed properties.
    
    Usage:
        config = AppConfig.instance()
        print(config.chunk_height)
        config.chunk_height = 1024
    """
    
    # Stacking settings
    chunk_height = ConfigValue("stacking/chunk_height", default=512, type_=int)
    chunk_width = ConfigValue("stacking/chunk_width", default=512, type_=int)
    use_gpu = ConfigValue("stacking/use_gpu", default=True, type_=bool)
    rejection_algorithm = ConfigValue("stacking/rejection_algorithm", default="sigma_clip", type_=str)
    
    # UI settings
    theme = ConfigValue("ui/theme", default="dark", type_=str)
    show_toolbar = ConfigValue("ui/show_toolbar", default=True, type_=bool)
    recent_files_max = ConfigValue("ui/recent_files_max", default=10, type_=int)
    
    # Performance settings
    max_threads = ConfigValue("performance/max_threads", default=0, type_=int)  # 0 = auto
    memory_limit_mb = ConfigValue("performance/memory_limit_mb", default=0, type_=int)  # 0 = no limit
    use_memmap = ConfigValue("performance/use_memmap", default=True, type_=bool)
    
    # Paths
    last_open_dir = ConfigValue("paths/last_open_dir", default="", type_=str)
    last_save_dir = ConfigValue("paths/last_save_dir", default="", type_=str)
    output_dir = ConfigValue("paths/output_dir", default="", type_=str)


def get_app_config() -> AppConfig:
    """Get the application configuration with typed properties."""
    if AppConfig._instance is None:
        AppConfig._instance = AppConfig()
    return AppConfig._instance  # type: ignore
