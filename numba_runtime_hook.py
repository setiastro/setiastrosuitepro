import sys
import types

if "numba.core.types.old_scalars" not in sys.modules:
    dummy = types.ModuleType("numba.core.types.old_scalars")
    dummy.Boolean = bool    # Dummy definition for Boolean
    dummy.Integer = int     # Dummy definition for Integer
    sys.modules["numba.core.types.old_scalars"] = dummy
