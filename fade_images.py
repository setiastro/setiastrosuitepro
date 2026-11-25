# fade_images.py
# SetiAstro - Franklin Marek / www.setiastro.com
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Standalone: crossfade a list of still images into a looping MP4 (or GIF).

    python fade_images.py                      run with the settings below
    python fade_images.py a.jpg b.jpg c.jpg -o out.mp4
    python fade_images.py --fade 2 --hold 1 --pause 4 --cycles 1

Sequence per cycle:

    img1 hold -> fade -> img2 hold -> fade -> img3 PAUSE -> fade -> (img1 ...)

The images are never resized, cropped or padded. Every input must already have
the same width and height, and the finished file is probed to confirm it still
has that width and height.

Needs Pillow, numpy, and ffmpeg (on PATH, or at C:\\ffmpeg\\bin).
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

# ----------------------------------------------------------------- settings --
FOLDER = Path(r"T:\SGP iOptron Imaging Newt\SASpro Output NGC660")
IMAGES = [
    FOLDER / "lrgb_bigzoom.jpg",
    FOLDER / "ha_contsub_bigzoom.jpg",
    FOLDER / "lrgbha_bigzoom.jpg",
]
OUTPUT = FOLDER / "NGC660_fade.mp4"

HOLD = 1.0      # seconds each image stays up before fading (all but the last)
FADE = 1.5      # seconds per crossfade
PAUSE = 3.0     # seconds the LAST image stays up before the cycle repeats
CYCLES = 3      # times the cycle is repeated inside an MP4 (a GIF loops forever)
CRF = 20        # MP4 quality: lower = better and bigger
# ------------------------------------------------------------------------------

Image.MAX_IMAGE_PIXELS = None
THREADS = max(1, min(16, os.cpu_count() or 1))


def find_tool(name: str) -> str:
    exe = shutil.which(name)
    if exe:
        return exe
    fallback = Path(r"C:\ffmpeg\bin") / f"{name}.exe"
    if fallback.is_file():
        return str(fallback)
    sys.exit(f"{name} not found. Install ffmpeg or add it to PATH.")


def load_srgb(path: Path) -> np.ndarray:
    """Load as 8-bit RGB, converting a tagged profile (e.g. Display P3) to sRGB.

    Video and GIF players mostly assume sRGB/BT.709, so P3 numbers passed
    straight through would show up undersaturated.
    """
    with Image.open(path) as im:
        im.load()
        icc = im.info.get("icc_profile")
        if icc:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            space = src.profile.xcolor_space.strip()
            # A mono JPEG carrying an RGB profile can't go through the CMM, and
            # doesn't need to: neutral grey is the same in P3 and sRGB.
            if (im.mode, space) in (("RGB", "RGB"), ("L", "GRAY")):
                im = ImageCms.profileToProfile(
                    im, src, ImageCms.createProfile("sRGB"),
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB")
        return np.array(im.convert("RGB"))


def build_schedule(n: int, fps: float, hold: float, fade: float, pause: float,
                   loop_fade: bool) -> list[tuple[int, int, float, int]]:
    """One cycle as (image_a, image_b, mix, repeat_count) entries."""
    out = []
    steps = round(fade * fps)
    for i in range(n):
        last = i == n - 1
        out.append((i, i, 0.0, max(1, round((pause if last else hold) * fps))))
        if last and not loop_fade:
            break
        for k in range(1, steps + 1):
            t = k / (steps + 1)
            out.append((i, (i + 1) % n, t * t * (3.0 - 2.0 * t), 1))  # ease in/out
    return out


class Blender:
    """Crossfades in float32, rounding once to the depth being encoded."""

    def __init__(self, shape, deep: bool, pool: ThreadPoolExecutor):
        self.scale = 257.0 if deep else 1.0
        self.out = np.empty(shape, np.uint16 if deep else np.uint8)
        edges = np.linspace(0, shape[0], THREADS + 1).astype(int)
        self.strips = list(zip(edges[:-1], edges[1:]))
        self.pool = pool

    def __call__(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        def work(strip):
            s = slice(*strip)
            f = np.multiply(a[s], self.scale * (1.0 - t), dtype=np.float32)
            if t > 0.0:
                f += np.multiply(b[s], self.scale * t, dtype=np.float32)
            f += 0.5
            np.copyto(self.out[s], f, casting="unsafe")

        list(self.pool.map(work, self.strips))
        return self.out


def encoder_args(out: Path, w: int, h: int, codec: str, bits: int, crf: int,
                 cycle_frames: int) -> tuple[bool, list[str]]:
    """Returns (pipe 16-bit RGB?, ffmpeg output args)."""
    if out.suffix.lower() == ".gif":
        return False, ["-vf", "split[a][b];[a]palettegen=stats_mode=single[p];"
                              "[b][p]paletteuse=new=1:dither=sierra2_4a",
                       "-loop", "0"]

    if codec == "auto":
        # H.264 plays everywhere, but ordinary (4:2:0) H.264 can't store an odd
        # width or height and most decoders stop at 4096. AV1 has neither limit.
        fits = w % 2 == 0 and h % 2 == 0 and max(w, h) <= 4096
        codec = "h264" if fits else "av1"

    if codec == "av1":
        # 10-bit is marginally smoother in faint gradients, but Chrome and Edge
        # drop about half the frames at 36 MP. 8-bit plays cleanly.
        pix, deep = ("yuv420p10le", True) if bits == 10 else ("yuv420p", False)
        enc = ["-c:v", "libsvtav1", "-preset", "6", "-crf", str(crf)]
    else:
        pix, deep = ("yuv420p" if w % 2 == 0 and h % 2 == 0 else "yuv444p"), False
        enc = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]

    vf = (f"scale=out_color_matrix=bt709:out_range=tv:"
          f"flags=lanczos+accurate_rnd+full_chroma_int,format={pix},"
          f"setparams=range=tv:colorspace=bt709:color_primaries=bt709:"
          f"color_trc=iec61966-2-1")
    # One keyframe per cycle: a mid-hold keyframe shows up as a quality "pop".
    return deep, ["-vf", vf, *enc, "-g", str(cycle_frames), "-movflags", "+faststart"]


def probe(path: Path) -> tuple[int, int, str, int]:
    r = subprocess.run(
        [find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries",
         "stream=width,height,codec_name,nb_read_packets",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True)
    d = dict(line.split("=", 1) for line in r.stdout.split() if "=" in line)
    return (int(d["width"]), int(d["height"]), d["codec_name"],
            int(d["nb_read_packets"]))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Crossfade still images into a looping MP4 or GIF, "
                    "without changing their dimensions.")
    ap.add_argument("images", nargs="*", type=Path, help="images, in order")
    ap.add_argument("-o", "--output", type=Path, help=".mp4 or .gif")
    ap.add_argument("--hold", type=float, default=HOLD)
    ap.add_argument("--fade", type=float, default=FADE)
    ap.add_argument("--pause", type=float, default=PAUSE)
    ap.add_argument("--cycles", type=int, default=CYCLES, help="MP4 only")
    ap.add_argument("--fps", type=float, help="default 30 for MP4, 15 for GIF")
    ap.add_argument("--crf", type=int, default=CRF)
    ap.add_argument("--codec", choices=("auto", "av1", "h264"), default="auto")
    ap.add_argument("--bits", type=int, choices=(8, 10), default=8, help="AV1 only")
    ap.add_argument("--no-loop-fade", action="store_true",
                    help="cut from the last image back to the first, no fade")
    args = ap.parse_args()

    paths = args.images or IMAGES
    if len(paths) < 2:
        ap.error("need at least two images")
    # Images given without -o: write beside them, not over the default output.
    out = args.output or (paths[0].with_name(paths[0].stem + "_fade.mp4")
                          if args.images else OUTPUT)
    kind = out.suffix.lower()
    if kind not in (".mp4", ".gif"):
        ap.error("output must end in .mp4 or .gif")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        ap.error("not found: " + ", ".join(missing))
    fps = args.fps or (15.0 if kind == ".gif" else 30.0)

    frames = [load_srgb(p) for p in paths]
    for p, f in zip(paths, frames):
        print(f"  {p.name}: {f.shape[1]} x {f.shape[0]}")
    if len({f.shape for f in frames}) != 1:
        sys.exit("The images are not all the same size. Nothing is resized here, "
                 "so they have to match first.")
    h, w = frames[0].shape[:2]

    sched = build_schedule(len(frames), fps, args.hold, args.fade, args.pause,
                           not args.no_loop_fade)
    total = sum(s[3] for s in sched)
    deep, enc = encoder_args(out, w, h, args.codec, args.bits, args.crf, total)
    cycles = max(1, args.cycles) if kind == ".mp4" else 1
    print(f"  {total} frames per cycle at {fps:g} fps = {total / fps:.1f} s"
          + (f", x{cycles} cycles" if cycles > 1 else ""))

    ffmpeg = find_tool("ffmpeg")
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    part = out.with_name(out.stem + ".part" + out.suffix)
    done = False
    try:
        proc = subprocess.Popen(
            [*base, "-f", "rawvideo", "-pix_fmt", "rgb48le" if deep else "rgb24",
             "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", *enc, str(part)],
            stdin=subprocess.PIPE, env={**os.environ, "SVT_LOG": "1"})
        sent = 0
        try:
            with ThreadPoolExecutor(THREADS) as pool:
                blend = Blender(frames[0].shape, deep, pool)
                for a, b, t, repeat in sched:
                    buf = memoryview(blend(frames[a], frames[b], t)).cast("B")
                    for _ in range(repeat):
                        proc.stdin.write(buf)
                        sent += 1
                        print(f"\r  encoding {sent}/{total}", end="", flush=True)
            proc.stdin.close()
        except OSError:
            pass                        # ffmpeg quit early; its exit code says why
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        print("\r  finishing...            ", flush=True)
        if proc.wait() != 0 or sent != total:
            sys.exit(f"ffmpeg failed (exit {proc.returncode}, {sent}/{total} frames)")

        if cycles > 1:                  # repeat by stream copy: no re-encode
            subprocess.run(
                [*base, "-stream_loop", str(cycles - 1), "-i", str(part),
                 "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
        else:
            os.replace(part, out)
        done = True
    finally:
        if part.exists():
            part.unlink()
        if not done:
            print("\n  no output written")

    pw, ph, codec, n = probe(out)
    if (pw, ph) != (w, h):
        sys.exit(f"DIMENSIONS CHANGED: {w}x{h} in, {pw}x{ph} out. Do not use {out}")
    print(f"  {out}\n  {codec}, {pw} x {ph} (unchanged), {n} frames, "
          f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
