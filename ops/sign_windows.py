# ops/sign_windows.py
r"""
Authenticode-sign the Windows build with the Certum code-signing certificate.

    python ops\sign_windows.py dist\SetiAstroSuitePro
        Signs every .exe/.dll/.pyd under the folder that has no signature yet.
        Files their vendor already signed (Qt, Microsoft, PSF, Intel) keep
        that signature untouched.

    python ops\sign_windows.py --file <path>
        Signs one file unconditionally. This is the command Inno Setup runs
        (SignTool=saspro in saspro_setup.iss) for the setup exe, the temporary
        self-copies it starts from, and the uninstaller. build_installer.cmd
        hands it to ISCC with /S.

Why: Windows 11 Smart App Control blocks unsigned exe/dll/pyd files and the
installer's unsigned temp copy outright (error 4551), with no user override.

The private key lives on Certum's SimplySign cloud card, so SimplySign Desktop
must be logged in (email + 6-digit code from the phone app) for the whole run.
"No certificates were found that met all the given criteria" means it is not.

The certificate is picked by subject name, so next year's renewal (same name)
works without editing this file. Every signature is RFC 3161 timestamped;
without a timestamp a signature stops validating when the certificate expires.
"""
import argparse
import glob
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CERT_SUBJECT = "Open Source Developer Franklin Marek"
TIMESTAMP_URL = "http://time.certum.pl"
SIGN_EXTS = (".exe", ".dll", ".pyd")

# Each signature is a round trip to Certum's cloud (~0.9 s) plus one to the
# timestamp server (~0.4 s), so signtool processes run in parallel. Measured
# 2026-11-26: 1.35 s/file in series, 0.41 s/file with 8 at once, every file valid.
JOBS = int(os.environ.get("SASPRO_SIGN_JOBS", "8"))
BATCH_FILES = 10        # files per signtool call; a failed batch is re-signed whole
MAX_CMDLINE = 30000     # CreateProcess caps a command line at 32767 chars
ATTEMPTS = 3

FAIL_HINT = """\
[sign] Signing FAILED.
[sign]   "No certificates were found" -> SimplySign Desktop is not logged in.
[sign]       Tray icon -> Connect to SimplySign, email + code from the phone app.
[sign]       Still failing while logged in (e.g. after reinstalling SimplySign)?
[sign]       Its installer fails to register the KSP; from an ADMIN prompt run
[sign]       python ops\\register_simplysign_ksp.py
[sign]   Timestamp errors -> time.certum.pl did not answer; just run it again.
[sign]   Re-running is safe: files that are already signed are skipped."""


def log(msg):
    print(msg, flush=True)


def find_signtool():
    env = os.environ.get("SIGNTOOL")
    if env and os.path.isfile(env):
        return env
    kits = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                        "Windows Kits", "10", "bin")
    found = glob.glob(os.path.join(kits, "10.*", "x64", "signtool.exe"))
    if found:
        def sdk_version(p):
            name = os.path.basename(os.path.dirname(os.path.dirname(p)))
            return tuple(int(x) for x in name.split(".") if x.isdigit())
        return max(found, key=sdk_version)
    on_path = shutil.which("signtool")
    if on_path:
        return on_path
    sys.exit("[sign] signtool.exe not found. Install the Windows SDK "
             "'Signing Tools for Desktop Apps', or set SIGNTOOL=<path>.")


def has_signature(path):
    """True if the PE file carries an embedded Authenticode signature,
    False if it does not, None if the file is not a PE image at all."""
    with open(path, "rb") as f:
        dos = f.read(64)
        if len(dos) < 64 or dos[:2] != b"MZ":
            return None
        f.seek(struct.unpack_from("<I", dos, 0x3C)[0])
        hdr = f.read(24 + 240)
    if len(hdr) < 26 or hdr[:4] != b"PE\0\0":
        return None
    # Optional header starts 24 bytes in; its data directories start 96 (PE32)
    # or 112 (PE32+) bytes later. Entry 4 is the certificate table.
    magic = struct.unpack_from("<H", hdr, 24)[0]
    cert_dir = 24 + (112 if magic == 0x20B else 96) + 4 * 8
    if len(hdr) < cert_dir + 8:
        return None
    _offset, size = struct.unpack_from("<II", hdr, cert_dir)
    return size > 0


def batches(files):
    batch, length = [], 0
    for p in files:
        if batch and (len(batch) >= BATCH_FILES or length + len(p) + 3 > MAX_CMDLINE):
            yield batch
            batch, length = [], 0
        batch.append(p)
        length += len(p) + 3
    if batch:
        yield batch


def sign_args(files):
    return ["sign", "/a", "/n", CERT_SUBJECT, "/fd", "sha256",
            "/tr", TIMESTAMP_URL, "/td", "sha256", *files]


def verify_args(files):
    # /tw turns a missing timestamp into a warning (exit 2), which counts as a failure.
    return ["verify", "/pa", "/tw", *files]


def run_signtool(signtool, args):
    """Run signtool with retries. Signing again replaces our own signature,
    so a batch that died halfway (e.g. a timestamp timeout) is simply redone."""
    out = ""
    for attempt in range(1, ATTEMPTS + 1):
        r = subprocess.run([signtool, *args], capture_output=True, text=True)
        if r.returncode == 0:
            return True
        out = (r.stdout + r.stderr).strip()
        if "No certificates were found" in out:
            break   # SimplySign not logged in: retrying cannot help
        if attempt < ATTEMPTS:
            log(f"[sign]   signtool {args[0]} failed (attempt {attempt}/{ATTEMPTS}), retrying...")
            time.sleep(5 * attempt)
    log(out)
    return False


def run_parallel(signtool, files, make_args, progress=None):
    """Run signtool over the files in batches, JOBS processes at once.
    Stops handing out batches at the first one that fails for good."""
    chunks = list(batches(files))
    lock = threading.Lock()
    state = {"done": 0, "last": 0.0}
    t0 = time.perf_counter()

    def work(chunk):
        ok = run_signtool(signtool, make_args(chunk))
        if ok and progress:
            with lock:
                state["done"] += len(chunk)
                now = time.perf_counter() - t0
                if now - state["last"] >= 5 or state["done"] == len(files):
                    state["last"] = now
                    log(f"[sign] {progress}: signed {state['done']}/{len(files)}  ({now:.0f} s)")
        return ok

    with ThreadPoolExecutor(max_workers=max(1, min(JOBS, len(chunks)))) as pool:
        futures = [pool.submit(work, c) for c in chunks]
        for fut in as_completed(futures):
            if not fut.result():
                for other in futures:
                    other.cancel()
                return False
    return True


def sign_files(signtool, files, label):
    t0 = time.perf_counter()
    if not run_parallel(signtool, files, sign_args, progress=label):
        log(FAIL_HINT)
        return False
    if not run_parallel(signtool, files, verify_args):
        log("[sign] Verification FAILED for the files above (bad chain or no timestamp).")
        return False
    log(f"[sign] {label}: {len(files)} file(s) signed, timestamped and verified "
        f"in {time.perf_counter() - t0:.0f} s")
    return True


def sign_tree(signtool, root):
    if not os.path.isdir(root):
        sys.exit(f"[sign] folder not found: {root}")
    todo, vendor, not_pe = [], 0, []
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            if not name.lower().endswith(SIGN_EXTS):
                continue
            path = os.path.join(dirpath, name)
            state = has_signature(path)
            if state is None:
                not_pe.append(path)
            elif state:
                vendor += 1
            else:
                todo.append(path)
    todo.sort()
    for p in not_pe:
        log(f"[sign] skipping, not a PE file: {p}")
    log(f"[sign] {root}: {len(todo)} to sign, {vendor} already signed (kept as is)")
    if not todo:
        return True
    if not sign_files(signtool, todo, os.path.basename(os.path.normpath(root))):
        return False
    left = [p for p in todo if not has_signature(p)]
    if left:
        log(f"[sign] {len(left)} file(s) still unsigned, e.g. {left[0]}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="sign every unsigned binary under this folder")
    ap.add_argument("--file", help="sign this one file unconditionally (Inno Setup SignTool)")
    a = ap.parse_args()
    if bool(a.folder) == bool(a.file):
        ap.error("give either a folder or --file <path>")

    signtool = find_signtool()
    if a.file:
        ok = sign_files(signtool, [os.path.abspath(a.file)], os.path.basename(a.file))
    else:
        ok = sign_tree(signtool, a.folder)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
