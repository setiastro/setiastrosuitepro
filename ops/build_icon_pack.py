#!/usr/bin/env python3
"""
Render the "Aurora" icon pack: one tile style, category colours, bold glyphs.

    python ops/build_icon_pack.py [output_dir]

All artwork is defined below as SVG on a 256x256 canvas with a 52-204 live area.
Strokes are at least 16 units so they survive being shown at 24 px. Colour means
something: the tile is the category, red/green/blue are image channels, and
yellow is the one accent. See setiastro/saspro/icon_packs.py for how packs load.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

W, Y, R, G, B, C = "#FFFFFF", "#FFD54A", "#FF5A5F", "#4ADE80", "#60A5FA", "#5EEAD4"
DIM = "#FFFFFF88"
S = 18

# folder: (tile top, tile bottom)
PALETTE = {
    "Display": ("#2AA6A0", "#14706F"),
    "Edit": ("#6B7A90", "#465264"),
    "File": ("#6B7A90", "#465264"),
    "Functions": ("#4F63E0", "#2F3BA8"),
    "Geometry": ("#2F8FD8", "#1B5FA3"),
    "Masks": ("#9A52D8", "#6A2FA5"),
    "Smart_Tools": ("#D0479C", "#94276C"),
    "Special": ("#E08A2C", "#AE5C14"),
    "Star_Stuff": ("#3E4FA8", "#232C70"),
    "Tools": ("#2FA36B", "#17704A"),
    "UI_Elements": ("#6B7A90", "#465264"),
    "Whats_In_My": ("#E08A2C", "#AE5C14"),
}
TILE = "#000000"  # set per icon: for cut-outs and badge outlines


# ---- primitives ----
def line(x1, y1, x2, y2, c=W, w=S):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" '
            f'stroke-width="{w}" stroke-linecap="round"/>')


def path(d, c=W, w=S):
    return (f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}" '
            'stroke-linecap="round" stroke-linejoin="round"/>')


def fill(d, c=W, rule="nonzero"):
    return f'<path d="{d}" fill="{c}" fill-rule="{rule}"/>'


def ring(x, y, r, c=W, w=S):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="{c}" stroke-width="{w}"/>'


def dot(x, y, r, c=W):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'


def box(x, y, w, h, c=W, sw=S, rx=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" '
            f'stroke="{c}" stroke-width="{sw}" stroke-linejoin="round"/>')


def block(x, y, w, h, c=W, rx=6):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{c}"/>'


def oval(x, y, rx, ry, c=W, w=S, rot=0):
    return (f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="none" stroke="{c}" '
            f'stroke-width="{w}" transform="rotate({rot} {x} {y})"/>')


def spin(deg, x, y, *parts):
    return f'<g transform="rotate({deg} {x} {y})">{"".join(parts)}</g>'


def star(x, y, r, c=W, inner=0.45):
    pts = []
    for i in range(10):
        rad = r if i % 2 == 0 else r * inner
        a = math.radians(-90 + i * 36)
        pts.append(f"{x + rad * math.cos(a):.1f},{y + rad * math.sin(a):.1f}")
    return (f'<polygon points="{" ".join(pts)}" fill="{c}" stroke="{c}" stroke-width="7" '
            'stroke-linejoin="round"/>')


def spark(x, y, r, c=W):
    return fill(f"M{x},{y-r} Q{x},{y} {x+r},{y} Q{x},{y} {x},{y+r} "
                f"Q{x},{y} {x-r},{y} Q{x},{y} {x},{y-r}Z", c)


def arrow(x1, y1, x2, y2, c=W, w=S, head=32):
    a = math.atan2(y2 - y1, x2 - x1)
    out = line(x1, y1, x2, y2, c, w)
    for d in (2.55, -2.55):
        out += line(x2, y2, x2 + head * math.cos(a + d), y2 + head * math.sin(a + d), c, w)
    return out


def arc(x, y, r, a0, a1, c=W, w=S, head=30):
    """Circular arc from angle a0 to a1 (degrees, clockwise from east), optional arrowhead."""
    p0 = (x + r * math.cos(math.radians(a0)), y + r * math.sin(math.radians(a0)))
    p1 = (x + r * math.cos(math.radians(a1)), y + r * math.sin(math.radians(a1)))
    sweep = 1 if a1 > a0 else 0
    large = 1 if abs(a1 - a0) > 180 else 0
    out = path(f"M{p0[0]:.1f},{p0[1]:.1f} A{r},{r} 0 {large} {sweep} {p1[0]:.1f},{p1[1]:.1f}", c, w)
    if head:
        t = math.radians(a1 + (90 if sweep else -90))
        for d in (2.5, -2.5):
            out += line(p1[0], p1[1], p1[0] + head * math.cos(t + d), p1[1] + head * math.sin(t + d), c, w)
    return out


def hump_d(x0, x1, base, h):
    wd = x1 - x0
    return (f"M{x0},{base} C{x0 + .32 * wd},{base} {x0 + .32 * wd},{base - h} {x0 + .5 * wd},{base - h} "
            f"C{x0 + .68 * wd},{base - h} {x0 + .68 * wd},{base} {x1},{base}")


def hump(x0, x1, base, h, c=W, solid=True):
    return fill(hump_d(x0, x1, base, h) + "Z", c) if solid else path(hump_d(x0, x1, base, h), c)


def badge(kind, x=182, y=182, r=38, c=Y):
    ink = "#222834"
    out = dot(x, y, r + 8, TILE) + dot(x, y, r, c)
    k = r * 0.52
    if kind in ("plus", "minus"):
        out += line(x - k, y, x + k, y, ink, 13)
    if kind == "plus":
        out += line(x, y - k, x, y + k, ink, 13)
    if kind == "x":
        out += line(x - k, y - k, x + k, y + k, ink, 13) + line(x - k, y + k, x + k, y - k, ink, 13)
    if kind == "check":
        out += path(f"M{x-k},{y+1} L{x-k*.2},{y+k*.75} L{x+k},{y-k*.7}", ink, 13)
    if kind == "spark":
        out += spark(x, y, r * .78, ink)
    return out


def axes():
    return path("M58,52 V198 H204", DIM, 12)


def layers(top=70, c=W):
    out = ""
    for i, col in enumerate(("#FFFFFF77", "#FFFFFFAA", c)):
        y = top + (2 - i) * 40
        out += path(f"M128,{y-34} L204,{y} L128,{y+34} L52,{y}Z", TILE, 10)
        out += fill(f"M128,{y-34} L204,{y} L128,{y+34} L52,{y}Z", col)
    return out


def window_grid(n, x=56, y=56, size=144, gap=12, colours=None):
    cell = (size - gap * (n - 1)) / n
    out = ""
    for i in range(n * n):
        col = colours[i] if colours else W
        out += block(x + (i % n) * (cell + gap), y + (i // n) * (cell + gap), cell, cell, col, 8)
    return out


def fan(colours):
    out = ""
    n = len(colours)
    for i, col in enumerate(colours):
        deg = -62 + i * (124 / (n - 1))
        out += spin(deg, 128, 196, block(112, 62, 32, 134, col, 8))
    return out + dot(128, 188, 10, TILE)


def mask(x=112, y=112, k=1.0):
    def p(a, b):
        return f"{x + a * k:.1f},{y + b * k:.1f}"
    outer = (f"M{p(-66,-34)} C{p(-40,-46)} {p(-14,-40)} {p(0,-24)} C{p(14,-40)} {p(40,-46)} {p(66,-34)} "
             f"C{p(72,4)} {p(52,36)} {p(24,36)} C{p(12,36)} {p(6,24)} {p(0,14)} "
             f"C{p(-6,24)} {p(-12,36)} {p(-24,36)} C{p(-52,36)} {p(-72,4)} {p(-66,-34)}Z")
    eye = (f"M{p(-46,-16)} C{p(-36,-24)} {p(-22,-20)} {p(-16,-10)} C{p(-24,0)} {p(-40,0)} {p(-46,-16)}Z "
           f"M{p(46,-16)} C{p(36,-24)} {p(22,-20)} {p(16,-10)} C{p(24,0)} {p(40,0)} {p(46,-16)}Z")
    return fill(outer + " " + eye, W, "evenodd")


def rainbow(y, h=36, x=52, wd=152):
    colours = ["#8B5CF6", "#60A5FA", "#4ADE80", "#FFD54A", "#FB923C", "#FF5A5F"]
    seg = wd / len(colours)
    return "".join(block(x + i * seg, y, seg + 1, h, c, 0) for i, c in enumerate(colours))


def gradient_box(x=52, y=60, wd=152, h=136):
    return ('<linearGradient id="gb" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFFFFF"/>'
            '<stop offset="1" stop-color="#FFFFFF" stop-opacity="0.08"/></linearGradient>'
            f'<rect x="{x}" y="{y}" width="{wd}" height="{h}" rx="12" fill="url(#gb)"/>')


def tri_points(r=62, cx=128, cy=132):
    return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in (-90, 30, 150)]


def eye(x=128, y=128, k=1.0, c=W):
    wd, h = 80 * k, 50 * k
    return (path(f"M{x-wd},{y} Q{x},{y-h*2} {x+wd},{y} Q{x},{y+h*2} {x-wd},{y}Z", c)
            + dot(x, y, 24 * k, c))


def pin(x, y, k=1.0, c=W):
    return fill(f"M{x},{y+76*k} C{x-30*k},{y+30*k} {x-52*k},{y+10*k} {x-52*k},{y-22*k} "
                f"A{52*k},{52*k} 0 1 1 {x+52*k},{y-22*k} C{x+52*k},{y+10*k} {x+30*k},{y+30*k} {x},{y+76*k}Z", c)


def cross(x, y, r, c=W, w=S):
    return line(x - r, y - r, x + r, y + r, c, w) + line(x - r, y + r, x + r, y - r, c, w)


def plus(x, y, r, c=W, w=S):
    return line(x - r, y, x + r, y, c, w) + line(x, y - r, x, y + r, c, w)


def letter_L(x=70, y=54, h=148, c=W):
    return path(f"M{x},{y} V{y+h} H{x+74}", c, 24)


def planet_bands():
    return dot(128, 128, 76, W) + "".join(
        path(f"M{128-math.sqrt(max(76**2-(yy-128)**2,0)):.1f},{yy} "
             f"H{128+math.sqrt(max(76**2-(yy-128)**2,0)):.1f}", TILE, h)
        for yy, h in ((88, 9), (116, 15), (150, 9), (174, 8)))


# ---- every icon: filename -> (folder, draw function) ----
ICONS = {
    # Display
    "histogram": ("Display", lambda: block(52, 138, 26, 62) + block(84, 98, 26, 102) + block(116, 56, 26, 144, Y)
                  + block(148, 110, 26, 90) + block(180, 158, 26, 42)),
    "histogram_transform": ("Display", lambda: hump(52, 204, 156, 100)
                            + fill("M60,172 L80,204 H40Z", "#222834") + fill("M128,172 L148,204 H108Z", DIM)
                            + fill("M196,172 L216,204 H176Z", W)),
    # Edit
    "undoicon": ("Edit", lambda: path("M84,92 H154 A44,44 0 0 1 154,180 H100") + path("M112,56 L74,92 L112,128")),
    "redoicon": ("Edit", lambda: path("M172,92 H102 A44,44 0 0 0 102,180 H156") + path("M144,56 L182,92 L144,128")),
    "redo": ("Edit", lambda: path("M172,92 H102 A44,44 0 0 0 102,180 H156") + path("M144,56 L182,92 L144,128")),
    # File and other toolbar basics
    "openfile": ("File", lambda: path("M52,190 V70 H104 L122,90 H192 V112")
                 + fill("M72,118 H220 L192,196 H46Z")),
    "collage": ("File", lambda: box(54, 56, 148, 144) + line(128, 56, 128, 200) + line(54, 128, 202, 128)
                + spark(92, 92, 26, Y) + dot(166, 164, 15)),
    "psf": ("File", lambda: ring(128, 128, 76, DIM, 12) + ring(128, 128, 46, W, 12) + dot(128, 128, 21, Y)),
    "TextureClarity": ("File", lambda: "".join(dot(66 + i * 31, 66 + j * 31, 8 + (i + j) * .9, W if i + j > 2 else DIM)
                                               for i in range(5) for j in range(5))),
    # Functions
    "LExtract": ("Functions", lambda: letter_L(68, 74, 128) + arrow(116, 142, 198, 60, Y)),
    "LInsert": ("Functions", lambda: letter_L(68, 74, 128) + arrow(198, 60, 122, 136, Y)),
    "abeicon": ("Functions", lambda: gradient_box() + path("M60,172 C100,150 144,104 196,78", Y, 14)),
    "blaster": ("Functions", lambda: fill("M48,100 H150 L174,120 L150,140 H104 L92,194 H62 L68,140 H48Z")
                + line(190, 120, 212, 120, Y, 16) + line(186, 86, 204, 72, Y, 14) + line(186, 154, 204, 168, Y, 14)),
    "clahe": ("Functions", lambda: window_grid(3, colours=["#FFFFFF", "#FFFFFF44", "#FFFFFFAA", "#FFFFFF66", "#FFFFFF",
                                                           "#FFFFFF44", "#FFFFFFAA", "#FFFFFF33", "#FFFFFF"])),
    "clonestamp": ("Functions", lambda: dot(128, 72, 28) + block(114, 80, 28, 58) + fill("M76,148 Q76,130 96,130 H160 Q180,130 180,148 V166 H76Z")
                   + line(62, 196, 194, 196, Y, 18)),
    "contadd": ("Functions", lambda: hump(40, 190, 198, 136) + badge("plus", 182, 84, 34)),
    "contsub": ("Functions", lambda: hump(40, 190, 198, 136) + badge("minus", 182, 84, 34)),
    "convo": ("Functions", lambda: "".join(spin(a, 128, 92, line(128, 50, 128, 134, Y, 20)) for a in (0, 60, 120))
              + hump(52, 204, 206, 48)),
    "cosmeticcorrection": ("Functions", lambda: block(50, 96, 64, 64, R, 10) + arrow(126, 128, 170, 128, W, 16, 24)
                           + box(186 - 4, 104, 34, 48, W, 12, 6)),
    "cropicon": ("Functions", lambda: path("M86,48 V170 H208") + path("M48,86 H170 V208") + line(94, 162, 162, 94, Y, 14)),
    "curves": ("Functions", lambda: axes() + path("M64,190 C118,190 118,74 196,62", W) + block(90, 142, 30, 30, Y, 7) + block(144, 72, 30, 30, Y, 7)),
    "freqsep": ("Functions", lambda: path("M50,86 " + " ".join(f"L{50+i*13},{86 + (26 if i % 2 else -26)}" for i in range(1, 13)), W, 12)
                + path("M50,170 C86,120 110,120 128,170 S172,220 206,170", Y)),
    "fx": ("Functions", lambda: line(62, 194, 152, 104, W, 24) + spark(174, 82, 40, Y) + spark(92, 84, 22) + spark(182, 164, 18)),
    "green": ("Functions", lambda: dot(128, 128, 78, G) + cross(128, 128, 34, "#123524", 24)),
    "halo": ("Functions", lambda: ring(128, 128, 72, W, 12) + spark(128, 128, 38) + line(70, 186, 186, 70, Y)),
    "hdr": ("Functions", lambda: fill("M128,86 A42,42 0 0 0 128,170Z") + ring(128, 128, 42, W, 14)
            + "".join(spin(a, 128, 128, line(128, 54, 128, 66, Y, 16)) for a in range(0, 360, 45))),
    "invert": ("Functions", lambda: box(56, 56, 144, 144, W, 14, 10) + fill("M56,56 H200 L56,200Z")
               + fill("M102,102 A37,37 0 0 1 154,154Z", W) + fill("M102,102 A37,37 0 0 0 154,154Z", TILE)),
    "morpho": ("Functions", lambda: dot(128, 128, 30, Y) + arrow(128, 84, 128, 42, W, 16, 22) + arrow(128, 172, 128, 214, W, 16, 22)
               + arrow(84, 128, 42, 128, W, 16, 22) + arrow(172, 128, 214, 128, W, 16, 22)),
    "multiscale_decomp": ("Functions", lambda: box(54, 54, 148, 148, W, 14) + line(128, 54, 128, 202, W, 14) + line(54, 128, 202, 128, W, 14)
                          + block(61, 61, 60, 60, Y, 2) + line(91, 61, 91, 121, TILE, 8) + line(61, 91, 121, 91, TILE, 8)),
    "narrowbandnormalization": ("Functions", lambda: hump(30, 130, 196, 102, R) + hump(78, 178, 196, 102, G) + hump(126, 226, 196, 102, B) + line(42, 66, 214, 66, W, 14)),
    "nbextract": ("Functions", lambda: path("M104,62 L164,172 H44Z", W, 16) + line(32, 136, 88, 118, W, 14)
                  + line(124, 112, 218, 74, R, 14) + line(132, 124, 222, 118, G, 14) + line(142, 138, 214, 162, B, 14)),
    "nbtorgb": ("Functions", lambda: line(56, 64, 56, 192, W, 16) + line(84, 64, 84, 192, W, 16) + arrow(106, 128, 150, 128, Y, 16, 22)
                + dot(192, 86, 25, R) + dot(172, 164, 25, G) + dot(216, 164, 25, B)),
    "neutral": ("Functions", lambda: ring(112, 142, 58, W, 14) + fill("M112,84 A58,58 0 0 0 112,200Z", DIM)
                + line(142, 112, 196, 58, Y, 24) + line(166, 56, 198, 88, Y, 14)),
    "pedestal": ("Functions", lambda: block(52, 138, 50, 62, DIM, 4) + block(103, 94, 50, 106, W, 4) + block(154, 158, 50, 42, DIM, 4) + star(128, 62, 24, Y)),
    "pen": ("Functions", lambda: fill("M202,50 C150,54 96,96 84,162 L94,172 C160,160 200,106 202,50Z") + line(84, 172, 58, 198, W, 14)
            + path("M100,204 Q140,186 206,204", Y, 12)),
    "pixelmath": ("Functions", lambda: plus(86, 86, 26, W, 18) + line(144, 86, 196, 86, Y, 18) + cross(86, 170, 22, Y, 18)
                  + line(144, 170, 196, 170, W, 18) + dot(170, 146, 9) + dot(170, 194, 9)),
    "rgbalign": ("Functions", lambda: ring(104, 110, 46, R, 15) + ring(150, 110, 46, G, 15) + ring(128, 150, 46, B, 15)),
    "rgbcombo": ("Functions", lambda: "".join(line(x, y, 128, 132, DIM, 14) for x, y in tri_points(68)) + "".join(dot(x, y, 28, c) for (x, y), c in zip(tri_points(68), (R, G, B))) + dot(128, 132, 28)),
    "rgbextract": ("Functions", lambda: "".join(arrow(128, 132, x + (128 - x) * .34, y + (132 - y) * .34, W, 13, 18) for x, y in tri_points(68)) + "".join(dot(x, y, 25, c) for (x, y), c in zip(tri_points(72), (R, G, B))) + dot(128, 132, 21)),
    "satchroma": ("Functions", lambda: "".join(arc(128, 128, 62, a + 4, a + 56, c, 30, head=0) for a, c in zip(range(0, 360, 60), ("#FB923C", "#FFD54A", "#4ADE80", "#60A5FA", "#8B5CF6", "#FF5A5F"))) + dot(128, 128, 18)),
    "spcc": ("Functions", lambda: star(128, 92, 44) + rainbow(158)),
    "starstretch": ("Functions", lambda: axes() + hump(56, 138, 198, 118) + star(168, 94, 38, Y)),
    "statstretch": ("Functions", lambda: axes() + hump(56, 138, 198, 118) + arrow(132, 164, 200, 70, Y)),
    "uhs": ("Functions", lambda: line(128, 52, 128, 204, DIM, 10) + path("M56,192 C112,192 116,180 128,128 S152,64 200,64") + block(114, 114, 28, 28, Y, 7)),
    "whitebalance": ("Functions", lambda: path("M42,88 L60,172 L84,114 L108,172 L126,88", W, 18) + path("M152,88 V172 H180 A21,21 0 0 0 180,130 H152 M180,130 A21,21 0 0 0 180,88 H152", Y, 18)),
    # Geometry
    "debayer": ("Geometry", lambda: window_grid(2, 58, 58, 140, 8, [R, G, G, B])),
    "fliphorizontal": ("Geometry", lambda: line(128, 48, 128, 208, Y, 13) + fill("M108,70 V186 L44,186Z") + path("M148,70 V186 H212Z", W, 14)),
    "flipvertical": ("Geometry", lambda: line(48, 128, 208, 128, Y, 13) + fill("M70,108 H186 L186,44Z") + path("M70,148 H186 V212Z", W, 14)),
    "rescale": ("Geometry", lambda: box(54, 124, 78, 78, W, 16) + arrow(116, 140, 196, 60, Y)),
    "resizecanvas": ("Geometry", lambda: box(44, 44, 168, 168, DIM, 12) + block(100, 100, 56, 56, Y) + arrow(128, 88, 128, 62, W, 13, 17) + arrow(128, 168, 128, 194, W, 13, 17) + arrow(88, 128, 62, 128, W, 13, 17) + arrow(168, 128, 194, 128, W, 13, 17)),
    "rotate180": ("Geometry", lambda: box(100, 100, 56, 56, Y, 15) + arc(128, 128, 76, 180, 360, W) + arc(128, 128, 76, 0, 180, DIM, 12, head=0)),
    "rotateclockwise": ("Geometry", lambda: box(62, 116, 80, 80, Y, 15) + arc(128, 150, 88, -135, -28)),
    "rotatecounterclockwise": ("Geometry", lambda: box(114, 116, 80, 80, Y, 15) + arc(128, 150, 88, -45, -152)),
    "rotatearbitrary": ("Geometry", lambda: spin(24, 128, 138, box(94, 104, 68, 68, Y, 15)) + arc(128, 138, 78, -158, -12)),
    # Masks
    "maskcreate": ("Masks", lambda: mask(112, 112) + badge("spark")),
    "maskapply": ("Masks", lambda: mask(112, 112) + badge("check", c=G)),
    "maskremove": ("Masks", lambda: mask(112, 112) + badge("minus", c="#FF7B80")),
    # Smart tools
    "aberration": ("Smart_Tools", lambda: path("M128,48 C164,92 164,164 128,208 C92,164 92,92 128,48Z", W, 15)
                   + line(30, 92, 110, 106, R, 12) + line(30, 128, 110, 128, G, 12) + line(30, 164, 110, 150, B, 12)
                   + line(146, 110, 212, 128, R, 12) + line(150, 128, 212, 128, G, 12) + line(146, 146, 212, 128, B, 12)),
    "cosmic": ("Smart_Tools", lambda: spark(128, 128, 78) + oval(128, 128, 86, 30, Y, 11, -24)),
    "cosmicsat": ("Smart_Tools", lambda: spin(-40, 108, 104, block(34, 84, 46, 40, W, 3), block(88, 72, 40, 64, Y, 7), block(136, 84, 46, 40, W, 3))
                  + badge("x", 180, 180, 36, "#FF7B80")),
    "graxpert": ("Smart_Tools", lambda: gradient_box(50, 62, 130, 130) + badge("spark", 180, 178, 40)),
    "starnet": ("Smart_Tools", lambda: star(112, 110, 70) + badge("minus", 180, 180, 36)),
    # Special
    "wimi_icon_256x256": ("Special", lambda: box(50, 56, 130, 112, W, 15) + spark(96, 106, 26) + ring(148, 148, 34, Y, 15) + line(174, 174, 206, 206, Y, 20)),
    "wims": ("Special", lambda: pin(128, 104) + fill("M138,70 A32,32 0 1 0 150,120 A26,26 0 1 1 138,70Z", TILE)),
    # Star stuff
    "3dplanet": ("Star_Stuff", lambda: path("M36,108 L58,76 M220,108 L198,76", W, 14) + block(38, 108, 80, 56, R, 12) + block(138, 108, 80, 56, C, 12) + path("M118,122 H138", W, 14)),
    "Astro_Spikes": ("Star_Stuff", lambda: spark(106, 118, 72) + spark(180, 76, 34, Y) + spark(176, 182, 26, C)),
    "atlas": ("Star_Stuff", lambda: ring(128, 128, 76, W, 13) + oval(128, 128, 32, 76, W, 11) + line(54, 128, 202, 128, W, 11) + spark(182, 72, 28, Y)),
    "exoicon": ("Star_Stuff", lambda: dot(128, 80, 40, Y) + dot(150, 88, 15, "#1B2150") + path("M42,160 H98 Q112,204 128,204 Q144,204 158,160 H214", W, 15)),
    "flythrough": ("Star_Stuff", lambda: oval(128, 128, 84, 72, DIM, 10) + oval(128, 128, 54, 44, W, 11) + oval(128, 128, 26, 20, W, 11) + spark(128, 128, 18, Y)),
    "gaia": ("Star_Stuff", lambda: "".join(fill(f"M52,{y-40} A76,24 0 0 1 204,{y-40} V{y} A76,24 0 0 1 52,{y}Z", c) + oval(128, y - 40, 76, 24, TILE, 7) for y, c in ((200, "#FFFFFF77"), (156, "#FFFFFFAA"), (112, W))) + star(128, 72, 19, Y)),
    "planetarystacker": ("Star_Stuff", lambda: planet_bands() + dot(160, 133, 16, Y)),
    "platesolve": ("Star_Stuff", lambda: path("M58,178 L100,114 L162,136 L194,72", DIM, 10) + dot(58, 178, 14) + dot(162, 136, 14) + dot(194, 72, 14) + dot(100, 114, 13, Y) + ring(100, 114, 36, Y, 12)),
    "staradd": ("Star_Stuff", lambda: star(112, 110, 70) + badge("plus", 180, 180, 36)),
    "staralign": ("Star_Stuff", lambda: star(78, 78, 36, DIM) + star(176, 172, 44, W) + arrow(104, 104, 148, 146, Y, 15, 22)),
    "starregistration": ("Star_Stuff", lambda: box(58, 48, 112, 158, W, 15) + line(84, 86, 142, 86, W, 12) + line(84, 118, 124, 118, W, 12) + star(166, 166, 44, Y)),
    "starspike": ("Star_Stuff", lambda: line(36, 36, 220, 220, W, 7) + line(36, 220, 220, 36, W, 7) + spark(128, 128, 62, W) + dot(128, 128, 22, Y)),
    # Tools
    "blink": ("Tools", lambda: eye(128, 128) + line(128, 34, 128, 54, Y, 14) + line(62, 52, 76, 70, Y, 14) + line(194, 52, 180, 70, Y, 14)),
    "ditheranalysis": ("Tools", lambda: ring(128, 128, 76, DIM, 10) + ring(128, 128, 38, DIM, 10) + line(52, 128, 204, 128, DIM, 8) + line(128, 52, 128, 204, DIM, 8)
                       + dot(106, 94, 13, Y) + dot(160, 112, 13) + dot(98, 158, 13) + dot(146, 164, 13, Y) + dot(174, 150, 11)),
    "dse": ("Tools", lambda: fill("M150,52 A58,58 0 1 0 202,132 A46,46 0 1 1 150,52Z") + path("M50,196 Q88,166 128,196 T206,196", Y)),
    "imagecombine": ("Tools", lambda: box(52, 52, 104, 104, W, 15) + box(100, 100, 104, 104, W, 15) + block(108, 108, 40, 40, Y, 3)),
    "isophote": ("Tools", lambda: oval(128, 128, 82, 58, DIM, 11, -24) + oval(128, 128, 54, 36, W, 11, -24) + oval(128, 128, 26, 15, W, 11, -24) + dot(128, 128, 9, Y)),
    "linearfit": ("Tools", lambda: axes() + line(70, 184, 200, 66, Y, 16) + dot(88, 138, 12) + dot(126, 160, 12) + dot(120, 106, 12) + dot(170, 122, 12) + dot(160, 68, 12)),
    "livestacking": ("Tools", lambda: layers(66) + dot(186, 66, 22, "#FF7B80")),
    "magnitude": ("Tools", lambda: arc(128, 158, 82, 180, 360, W, 18, head=0) + line(128, 158, 174, 96, Y, 16) + dot(128, 158, 18) + line(46, 196, 210, 196, DIM, 12)),
    "mosaic": ("Tools", lambda: fill("M58,88 H104 A24,24 0 1 1 152,88 H198 V132 A24,24 0 1 0 198,180 V206 H58Z")),
    "selectivecolor": ("Tools", lambda: fan(["#8B5CF6", "#60A5FA", "#4ADE80", "#FFD54A", "#FF5A5F"])),
    "selectiveluma": ("Tools", lambda: fan(["#FFFFFF33", "#FFFFFF66", "#FFFFFF99", "#FFFFFFCC", "#FFFFFF"])),
    "slap": ("Tools", lambda: dot(128, 128, 52) + fill("M128,76 A52,52 0 0 1 128,180Z", "#1E5D42") + oval(128, 128, 96, 26, Y, 12, -18)),
    "snr": ("Tools", lambda: path("M44,178 L62,166 L76,186 L92,170 L106,180 L128,62 L148,178 L164,166 L178,186 L194,168 L212,180", W, 13) + dot(128, 62, 15, Y)),
    "sssc": ("Tools", lambda: star(128, 90, 38) + ring(128, 90, 56, Y, 11) + rainbow(164, 36) + line(86, 164, 86, 200, "#1B2130", 8) + line(150, 164, 150, 200, "#1B2130", 8)),
    "stacking": ("Tools", lambda: layers(70)),
    "surfacemosaic": ("Tools", lambda: dot(128, 128, 78) + dot(100, 100, 13, "#17704A99") + dot(160, 112, 18, "#17704A99") + dot(118, 164, 16, "#17704A99") + line(128, 50, 128, 206, TILE, 9) + line(50, 128, 206, 128, TILE, 9)),
    "unwarp": ("Tools", lambda: "".join(path(f"M{x},52 Q{x + (128 - x) * .5},128 {x},204", W, 11) for x in (56, 128, 200)) + "".join(path(f"M52,{y} Q128,{y + (128 - y) * .5} 204,{y}", W, 11) for y in (56, 128, 200))),
    "upscale": ("Tools", lambda: block(52, 172, 32, 32) + block(92, 132, 32, 72, DIM) + block(132, 132, 32, 32, DIM) + block(132, 172, 72, 32, DIM) + arrow(108, 108, 200, 60, Y, 16, 26)),
    # UI
    "benchmark": ("UI_Elements", lambda: "".join(line(x, 40, x, 216, W, 11) for x in (92, 128, 164)) + "".join(line(40, y, 216, y, W, 11) for y in (92, 128, 164))
                  + block(62, 62, 132, 132, "#384254", 14) + fill("M138,76 L100,136 H128 L118,180 L158,116 H130Z", Y)),
    "copyslot": ("UI_Elements", lambda: box(52, 52, 104, 118, DIM, 15) + block(96, 92, 108, 118, W, 10) + line(120, 128, 180, 128, "#384254", 11) + line(120, 164, 180, 164, "#384254", 11)),
    "disk": ("UI_Elements", lambda: fill("M52,60 Q52,52 60,52 H176 L204,80 V196 Q204,204 196,204 H60 Q52,204 52,196Z") + block(88, 52, 72, 46, "#384254", 3) + block(136, 58, 14, 32, W, 2) + block(82, 136, 92, 68, Y, 3)),
    "functionbundle": ("UI_Elements", lambda: path("M92,52 Q68,52 68,76 V110 Q68,128 46,128 Q68,128 68,146 V180 Q68,204 92,204") + path("M164,52 Q188,52 188,76 V110 Q188,128 210,128 Q188,128 188,146 V180 Q188,204 164,204") + dot(100, 128, 12, Y) + dot(128, 128, 12, Y) + dot(156, 128, 12, Y)),
    "gridicon": ("UI_Elements", lambda: box(52, 52, 152, 152, W, 14) + line(102, 52, 102, 204, W, 12) + line(154, 52, 154, 204, W, 12) + line(52, 102, 204, 102, W, 12) + line(52, 154, 204, 154, W, 12)),
    "ppp": ("UI_Elements", lambda: fill("M128,52 C68,52 40,92 40,132 C40,178 82,204 124,204 C146,204 148,186 140,174 C132,160 144,148 160,148 H186 C204,148 216,134 216,114 C216,76 176,52 128,52Z")
            + dot(86, 108, 15, R) + dot(120, 80, 15, Y) + dot(162, 88, 15, G) + dot(90, 156, 15, B)),
    "viewbundle": ("UI_Elements", lambda: box(44, 48, 112, 88, DIM, 13) + box(72, 80, 112, 88, DIM, 13) + block(100, 112, 112, 92, W, 9) + line(100, 136, 212, 136, "#384254", 9)),
    # What's In My
    "finderchart": ("Whats_In_My", lambda: ring(128, 128, 74, W, 14) + "".join(spin(a, 128, 128, line(128, 36, 128, 54, Y, 14)) for a in (0, 90, 180, 270))
                    + spark(102, 106, 26) + spark(158, 150, 18) + dot(154, 94, 8) + dot(98, 162, 8)),
    "supernova": ("Whats_In_My", lambda: "".join(spin(a, 128, 128, line(128, 72 if i % 2 else 42, 128, 88, W, 13)) for i, a in enumerate(range(0, 360, 30))) + dot(128, 128, 30, Y)),
}


def _split_alpha(svg: str) -> str:
    """QtSvg misreads #RRGGBBAA, so spell the alpha out as an opacity attribute."""
    def expand(m):
        return f'{m[1]}="#{m[2]}" {m[1]}-opacity="{int(m[3], 16) / 255:.3f}"'
    return re.sub(r'(fill|stroke)="#([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})"', expand, svg)


def icon_svg(name: str) -> str:
    global TILE
    folder, draw = ICONS[name]
    top, bottom = PALETTE[folder]
    TILE = bottom
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">'
        f'<defs><linearGradient id="t" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{top}"/>'
        f'<stop offset="1" stop-color="{bottom}"/></linearGradient></defs>'
        '<rect x="6" y="6" width="244" height="244" rx="52" fill="url(#t)" stroke="#FFFFFF" stroke-opacity="0.2" stroke-width="4"/>'
        f"{_split_alpha(draw())}</svg>"
    )


def build(out: Path) -> int:
    from PyQt6.QtCore import QByteArray
    from PyQt6.QtGui import QGuiApplication, QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer

    _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    for name, (folder, _) in ICONS.items():
        renderer = QSvgRenderer(QByteArray(icon_svg(name).encode()))
        if not renderer.isValid():
            raise ValueError(f"invalid SVG for {name}")
        image = QImage(256, 256, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        target = out / folder / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not image.save(str(target)):
            raise OSError(f"could not write {target}")
    return len(ICONS)


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "src/setiastro/images/icon_packs/Aurora"
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    print(f"Wrote {build(target_dir)} icons to {target_dir}")
