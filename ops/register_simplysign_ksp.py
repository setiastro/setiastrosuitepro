# ops/register_simplysign_ksp.py
"""
Register Certum's "SimplySign KSP" with Windows CNG, the step that
SimplySignKSP.dll's own DllRegisterServer is supposed to do but crashes in
(regsvr32 access violation 0xc0000005, seen in the Application log on
2026-11-26, both at install time and when run by hand).

This makes the same two standard calls Microsoft's KSP sample uses:
    BCryptRegisterProvider("SimplySign KSP", UM image SimplySignKSP.dll,
                           interface NCRYPT_KEY_STORAGE_INTERFACE / KEY_STORAGE)
    BCryptAddContextFunctionProvider(CRYPT_LOCAL, Default, KEY_STORAGE, ...)

Run from an ADMINISTRATOR prompt:
    python ops/register_simplysign_ksp.py            register + verify
    python ops/register_simplysign_ksp.py --remove   undo
"""
import ctypes
import sys
from ctypes import wintypes as wt

NAME = "SimplySign KSP"
IMAGE = "SimplySignKSP.dll"          # resolved from System32, where the installer put it
NCRYPT_KEY_STORAGE_INTERFACE = 0x00010001
KEY_STORAGE = "KEY_STORAGE"          # NCRYPT_KEY_STORAGE_ALGORITHM
CRYPT_LOCAL = 0x00000001
CRYPT_OVERWRITE = 0x00000001
CRYPT_PRIORITY_BOTTOM = 0xFFFFFFFF


class CRYPT_INTERFACE_REG(ctypes.Structure):
    _fields_ = [("dwInterface", wt.ULONG), ("dwFlags", wt.ULONG),
                ("cFunctions", wt.ULONG), ("rgpszFunctions", ctypes.POINTER(wt.LPWSTR))]


class CRYPT_IMAGE_REG(ctypes.Structure):
    _fields_ = [("pszImage", wt.LPWSTR), ("cInterfaces", wt.ULONG),
                ("rgpInterfaces", ctypes.POINTER(ctypes.POINTER(CRYPT_INTERFACE_REG)))]


class CRYPT_PROVIDER_REG(ctypes.Structure):
    _fields_ = [("cAliases", wt.ULONG), ("rgpszAliases", ctypes.POINTER(wt.LPWSTR)),
                ("pUM", ctypes.POINTER(CRYPT_IMAGE_REG)), ("pKM", ctypes.POINTER(CRYPT_IMAGE_REG))]


bcrypt = ctypes.WinDLL("bcrypt.dll")
ncrypt = ctypes.WinDLL("ncrypt.dll")
for fn in (bcrypt.BCryptRegisterProvider, bcrypt.BCryptAddContextFunctionProvider,
           bcrypt.BCryptRemoveContextFunctionProvider, bcrypt.BCryptUnregisterProvider,
           ncrypt.NCryptOpenStorageProvider, ncrypt.NCryptFreeObject):
    fn.restype = ctypes.c_long


def hx(status):
    return "0x%08X" % (status & 0xFFFFFFFF)


def register():
    funcs = (wt.LPWSTR * 1)(KEY_STORAGE)
    iface = CRYPT_INTERFACE_REG(NCRYPT_KEY_STORAGE_INTERFACE, CRYPT_LOCAL, 1, funcs)
    ifaces = (ctypes.POINTER(CRYPT_INTERFACE_REG) * 1)(ctypes.pointer(iface))
    image = CRYPT_IMAGE_REG(IMAGE, 1, ifaces)
    reg = CRYPT_PROVIDER_REG(0, None, ctypes.pointer(image), None)

    st = bcrypt.BCryptRegisterProvider(NAME, CRYPT_OVERWRITE, ctypes.byref(reg))
    print("BCryptRegisterProvider           ", hx(st))
    if st != 0:
        return False
    st = bcrypt.BCryptAddContextFunctionProvider(CRYPT_LOCAL, None, NCRYPT_KEY_STORAGE_INTERFACE,
                                                 KEY_STORAGE, NAME, CRYPT_PRIORITY_BOTTOM)
    # 0xC0000035 = already listed, which is fine on a re-run
    print("BCryptAddContextFunctionProvider ", hx(st))
    return st in (0, -1073741771)


def remove():
    st = bcrypt.BCryptRemoveContextFunctionProvider(CRYPT_LOCAL, None, NCRYPT_KEY_STORAGE_INTERFACE,
                                                    KEY_STORAGE, NAME)
    print("BCryptRemoveContextFunctionProvider", hx(st))
    st = bcrypt.BCryptUnregisterProvider(NAME)
    print("BCryptUnregisterProvider           ", hx(st))


def verify():
    h = ctypes.c_void_p()
    st = ncrypt.NCryptOpenStorageProvider(ctypes.byref(h), NAME, 0)
    print("NCryptOpenStorageProvider        ", hx(st))
    if st == 0:
        ncrypt.NCryptFreeObject(h)
        return True
    return False


def main():
    if not ctypes.windll.shell32.IsUserAnAdmin():
        sys.exit("Run this from an ADMINISTRATOR prompt (it writes HKLM crypto settings).")
    if "--remove" in sys.argv:
        remove()
        return
    if register() and verify():
        print("\nOK: '%s' is registered and opens. Try signing now." % NAME)
    else:
        sys.exit("\nFAILED - send the lines above to Claude.")


if __name__ == "__main__":
    main()
