import os
import shutil
import tempfile
import uuid
import pickle
import atexit
import threading
import numpy as np

class SwapManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SwapManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Create a unique temp directory for this session
        self.temp_dir = os.path.join(tempfile.gettempdir(), "SetiAstroSuitePro_Swap", str(uuid.uuid4()))
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)

    def get_swap_path(self, swap_id: str) -> str:
        return os.path.join(self.temp_dir, f"{swap_id}.swap")

    def save_state(self, image: np.ndarray) -> str:
        """
        Save the image array to a swap file.
        Returns the unique swap_id.
        """
        swap_id = uuid.uuid4().hex
        path = self.get_swap_path(swap_id)
        
        # We only save the image data to disk. Metadata is kept in RAM by the caller.
        # Using pickle for simplicity and robustness with numpy arrays.
        # For pure numpy arrays, np.save might be slightly faster, but pickle is more flexible if we change what we store.
        # Let's stick to pickle for now as per plan.
        try:
            with open(path, "wb") as f:
                pickle.dump(image, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[SwapManager] Failed to save state {swap_id}: {e}")
            return None
            
        return swap_id

    def load_state(self, swap_id: str) -> np.ndarray | None:
        """
        Load the image array from the swap file.
        """
        path = self.get_swap_path(swap_id)
        if not os.path.exists(path):
            print(f"[SwapManager] Swap file not found: {path}")
            return None
            
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[SwapManager] Failed to load state {swap_id}: {e}")
            return None

    def delete_state(self, swap_id: str):
        """
        Delete a specific swap file.
        """
        path = self.get_swap_path(swap_id)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[SwapManager] Failed to delete state {swap_id}: {e}")

    def cleanup_all(self):
        """
        Delete the entire temporary directory for this session.
        """
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                # print(f"[SwapManager] Cleaned up {self.temp_dir}")
        except Exception as e:
            print(f"[SwapManager] Cleanup failed: {e}")

# Global instance
_swap_mgr = SwapManager()

def get_swap_manager():
    return _swap_mgr
