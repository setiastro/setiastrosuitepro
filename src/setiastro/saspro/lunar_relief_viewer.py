# src/setiastro/saspro/lunar_relief_viewer.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Interactive 3D Moon explorer: one self-contained HTML file (WebGL, no internet).

The user's image is the texture and the vertices are the surface points its
pixels see (lunar_relief.relief_grid), so the picture lies on real terrain.
Optionally the rest of the globe -- LOLA LDEM_16 heights and Clementine 750 nm
albedo, lit by the same Sun -- so the image sits on a whole Moon.

Scene frame ("world"): km, x east / y north / z up at the image's centre point
on the 1737.4 km sphere; the Moon's centre is at (0, 0, -R).
"""
from __future__ import annotations

import base64
import html as _html
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from setiastro.saspro import lunar_relief as LR

R = LR.R_MOON
_MAX_TEX = 8192
_ALBEDO_TILES = ("clementine750_16ppd_e.png", "clementine750_16ppd_w.png")   # lon 0..180, -180..0


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _mtf(x, m):
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def _mtf_for(x_med, target):
    """Midtones balance that maps x_med to target."""
    return x_med * (target - 1.0) / (2.0 * x_med * target - target - x_med)


def display_stretch(samples):
    """(white point, midtones) that make a dark linear lunar image readable.
    midtones 0 = leave the tones alone (the image is already display-ready)."""
    v = np.asarray(samples, np.float64)
    v = v[np.isfinite(v)]
    if v.size < 16:
        return 1.0, 0.0
    hi = float(np.percentile(v, 99.5))
    lit = v[v > max(0.004, 0.08 * hi)]
    if lit.size < 16:
        return 1.0, 0.0
    w = float(np.clip(np.percentile(lit, 99.9), 0.02, 1.0))
    if w > 0.9:
        w = 1.0
    xm = float(np.median(lit)) / w
    m = _mtf_for(xm, 0.38) if 1e-4 < xm < 0.2 else 0.0
    return w, float(m)


def load_albedo(data_dir=None) -> Optional[np.ndarray]:
    """Clementine 750 nm albedo, (2880, 5760) uint8, row 0 = north, lon 0..360 east;
    None when the Lunar Solver has not downloaded it."""
    d = Path(data_dir) if data_dir else LR.LolaDEM.default_dir()
    tiles = [cv2.imread(str(d / n), cv2.IMREAD_GRAYSCALE) for n in _ALBEDO_TILES]
    if any(t is None for t in tiles):
        return None
    return _fill_gaps(np.ascontiguousarray(np.concatenate(tiles, axis=1)))


def _fill_gaps(a8: np.ndarray) -> np.ndarray:
    """Fill the mosaic's no-data pixels (orbit-strip gaps and the south-polar cap:
    0, with a dim fringe) from their surroundings (push-pull), so they do not
    render as black holes."""
    miss = (a8 < 10).astype(np.uint8)
    if not miss.any():
        return a8
    w0 = (cv2.dilate(miss, np.ones((7, 7), np.uint8)) == 0).astype(np.float32)

    def pushpull(v, w):
        if w.min() > 0:
            return v
        if w.max() == 0:
            return np.zeros_like(v)
        h, wd = v.shape
        if min(h, wd) <= 2:
            return np.where(w > 0, v, float((v * w).sum() / w.sum())).astype(np.float32)
        hs, ws = (h + 1) // 2, (wd + 1) // 2
        vw = cv2.resize(v * w, (ws, hs), interpolation=cv2.INTER_AREA)
        ww = cv2.resize(w, (ws, hs), interpolation=cv2.INTER_AREA)
        coarse = pushpull(np.where(ww > 0, vw / np.maximum(ww, 1e-12), 0.0).astype(np.float32),
                          (ww > 0).astype(np.float32))
        up = cv2.resize(coarse, (wd, h), interpolation=cv2.INTER_LINEAR)
        return np.where(w > 0, v, up).astype(np.float32)

    out = pushpull(a8.astype(np.float32), w0)
    return np.clip(out + 0.5, 1, 255).astype(np.uint8)


def _sample_nearest(img, uv):
    H, W = img.shape[:2]
    x = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, W - 1)
    y = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, H - 1)
    return img[y, x]


def _albedo_at(alb, lat, lon):
    """Nearest albedo (0..1) at body lat/lon (deg)."""
    H, W = alb.shape
    ppd = W / 360.0
    x = np.floor((np.mod(lon, 360.0)) * ppd).astype(np.int64) % W
    y = np.clip(np.floor((90.0 - lat) * ppd).astype(np.int64), 0, H - 1)
    return alb[y, x].astype(np.float32) / 255.0


def _lommel_seeliger(nrm, S, O_hat):
    ci = nrm @ S
    ce = np.maximum(nrm @ O_hat, 0.2)
    return np.where(ci > 0, ci / np.maximum(ci + ce, 1e-6), 0.0)


def _jpeg(u8, quality=92) -> str:
    if u8.ndim == 3:
        u8 = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("could not encode the viewer texture")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _texture(img, grid, w, m):
    """Display-ready JPEG of the image area the mesh covers + its pixel rectangle."""
    H, W = img.shape[:2]
    uvv = grid.uv[grid.valid]
    tx0 = max(0, int(math.floor(uvv[:, 0].min())) - 1)
    ty0 = max(0, int(math.floor(uvv[:, 1].min())) - 1)
    tx1 = min(W, int(math.ceil(uvv[:, 0].max())) + 2)
    ty1 = min(H, int(math.ceil(uvv[:, 1].max())) + 2)
    t = img[ty0:ty1, tx0:tx1]
    if t.ndim == 3:
        t = t[..., :3]
        if t.shape[2] == 1 or (np.array_equal(t[::7, ::7, 0], t[::7, ::7, 1])
                               and np.array_equal(t[::7, ::7, 0], t[::7, ::7, 2])):
            t = t[..., 0]
    x = np.clip(t.astype(np.float32) / float(w), 0.0, 1.0)
    if m > 0:
        x = _mtf(x, m)
    u8 = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    s = min(1.0, _MAX_TEX / float(max(u8.shape[:2])))
    if s < 1.0:
        u8 = cv2.resize(u8, (max(1, int(round(u8.shape[1] * s))), max(1, int(round(u8.shape[0] * s)))),
                        interpolation=cv2.INTER_AREA)
    return _jpeg(u8), [tx0, ty0, tx1 - tx0, ty1 - ty0]


def _whole_disc(camera, shape) -> bool:
    H, W = shape[:2]
    rd = float(camera.disk_radius_px)
    return (camera.cx - rd > -0.05 * rd and camera.cx + rd < W + 0.05 * rd
            and camera.cy - rd > -0.05 * rd and camera.cy + rd < H + 0.05 * rd)


def _footprint_sd(grid, camera, Xg):
    """Signed distance (image px, + inside) from globe points Xg (...,3) to the
    edge of the area the image mesh covers; far negative where not seen."""
    valid = grid.valid.astype(np.uint8)
    pad = np.pad(valid, 1)                     # the grid's own border is an edge too
    din = cv2.distanceTransform(pad, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - pad, cv2.DIST_L2, 5)
    sd = (din - dout).astype(np.float32)       # grid cells, on the padded grid
    shp = Xg.shape[:-1]
    uv, vis = camera.project(Xg.reshape(-1, 3))
    gx = (uv[:, 0] - grid.x0) / grid.step + 1.0
    gy = (uv[:, 1] - grid.y0) / grid.step + 1.0
    PH, PW = sd.shape
    cx, cy = np.clip(gx, 0, PW - 1), np.clip(gy, 0, PH - 1)
    out = LR._bilinear(sd, cx, cy) - np.hypot(gx - cx, gy - cy)
    out = np.where(vis & np.isfinite(out), out * grid.step, -1.0e4)
    return out.reshape(shp).astype(np.float32)


def _globe_payload(grid, camera, img_mono, C, M, dem16, albedo, w, ppd=2):
    geom = camera.geom
    hk = dem16.data.astype(np.float32) * 0.0005                 # LDEM DN = 0.5 m
    f = int(round(dem16.ppd / ppd))
    gh0, gw0 = hk.shape[0] // f, hk.shape[1] // f
    small = hk[:gh0 * f, :gw0 * f].reshape(gh0, f, gw0, f).mean(axis=(1, 3))
    rows, cols = 180 * ppd + 1, 360 * ppd + 1
    lat = 90.0 - np.arange(rows) / ppd
    lon = np.arange(cols) / ppd
    LO, LA = np.meshgrid(lon, lat)
    padded = np.concatenate([small[:, -1:], small, small[:, :1]], axis=1)
    hn = LR._bilinear(padded, LO * ppd - 0.5 + 1.0, np.clip((90.0 - LA) * ppd - 0.5, 0, gh0 - 1))
    dirs = LR.sph_to_body(LA, LO, 1.0)
    Xg = dirs * (R + hn)[..., None]
    sd = _footprint_sd(grid, camera, Xg)

    # brightness: fit the image's linear level to albedo x Lommel-Seeliger with the
    # solved Sun, so the rest of the Moon continues the picture at the same exposure
    O_hat = LR._unit(geom.O)
    S = np.asarray(geom.S, np.float64)
    Xv = grid.X[grid.valid]
    uvv = grid.uv[grid.valid]
    if len(Xv) > 60000:
        sel = np.linspace(0, len(Xv) - 1, 60000).astype(np.int64)
        Xv, uvv = Xv[sel], uvv[sel]
    nrm = LR._unit(Xv)
    la, lo = LR.body_to_latlon(Xv)
    a = _albedo_at(albedo, la, lo) if albedo is not None else np.full(len(Xv), 0.45, np.float32)
    model = a * _lommel_seeliger(nrm, S, O_hat)
    v = _sample_nearest(img_mono, uvv).astype(np.float64)
    ok = (nrm @ S > 0.2) & (a > 0.03) & (v > 0) & (model > 1e-4)
    gain = float(np.median(v[ok] / model[ok])) if ok.sum() > 200 else float(w) / 0.25

    if albedo is not None:
        alb8 = cv2.resize(albedo, (360 * 8, 180 * 8), interpolation=cv2.INTER_AREA)
    else:
        alb8 = np.full((8, 16), 115, np.uint8)
    h0, h1 = float(hn.min()), float(hn.max())
    hs = max(h1 - h0, 1e-6) / 65535.0
    qh = np.clip(np.round((hn - h0) / hs), 0, 65535).astype("<u2")
    qs = np.clip(np.round(sd * 4.0), -32767, 32767).astype("<i2")
    meta = dict(ppd=ppd, rows=rows, cols=cols, h0=h0, hs=hs, sd_scale=0.25,
                cut_px=float(1.5 * grid.step), gain=gain, has_albedo=albedo is not None)
    return meta, _b64(qh), _b64(qs), _jpeg(alb8, 90)


def viewer_html(grid, image01, camera, *, title: str, subtitle: str = "", exaggeration: float = 5.0,
                dem16=None, albedo=None, notes=()) -> str:
    """The explorer page.  image01: float [0,1] mono or RGB (linear or stretched);
    dem16 (LolaDEM.global16()) adds the rest of the Moon, albedo (load_albedo()) its
    surface brightness."""
    img = np.asarray(image01, np.float32)
    mono = img if img.ndim == 2 else img[..., :3].mean(axis=2)
    geom = camera.geom
    Xv = grid.X[grid.valid]
    C = LR._unit(Xv.mean(axis=0)) * R
    e, n, u = LR._local_basis(C)
    M = np.stack([e, n, u])

    # tile: every grid vertex (invalid ones flagged), quantised to 16 bits
    gh, gw = grid.valid.shape
    X = np.where(grid.valid[..., None], grid.X, C)
    h = np.linalg.norm(X, axis=-1) - R
    P = (X - C) @ M.T
    Pv, hv = P[grid.valid], h[grid.valid]
    p0 = Pv.min(axis=0)
    ps = np.maximum(Pv.max(axis=0) - p0, 1e-6) / 65535.0
    qp = np.clip(np.round((P - p0) / ps), 0, 65535).astype("<u2")
    h0 = float(hv.min())
    hs = max(float(hv.max()) - h0, 1e-6) / 65534.0
    qh = np.clip(np.round((h - h0) / hs), 0, 65534).astype("<u2")
    qh[~grid.valid] = 65535

    samples = _sample_nearest(mono, grid.uv[grid.valid][:: max(1, int(grid.valid.sum()) // 200000)])
    w, m = display_stretch(samples)
    tex_b64, rect = _texture(img, grid, w, m)

    # views
    uvv = grid.uv[grid.valid]
    cen = uvv.mean(axis=0)
    i_c = int(np.argmin(np.sum((uvv - cen) ** 2, axis=1)))
    target = P[grid.valid][i_c]
    d0 = camera.rays(cen[None].astype(np.float64))[0]
    d1 = camera.rays((cen - [0.0, 10.0])[None].astype(np.float64))[0]
    whole = _whole_disc(camera, img.shape)
    ext = R if whole else float(np.percentile(np.linalg.norm(Pv - target, axis=1), 98))
    km_px = float(camera.km_per_px())
    view = dict(earth=(M @ LR._unit(geom.O)).tolist(), up=(M @ LR._unit(d1 - d0)).tolist(),
                north=(M @ np.array([0.0, 0.0, 1.0])).tolist(), sun=(M @ np.asarray(geom.S)).tolist(),
                target=[float(v) for v in target], ext=max(ext, 5.0 * km_px * grid.step))

    globe = None
    globe_parts = ("", "", "")
    if dem16 is not None:
        globe, gh_b64, gs_b64, alb_b64 = _globe_payload(grid, camera, mono, C, M, dem16, albedo, w)
        globe_parts = (gh_b64, gs_b64, alb_b64)

    meta = dict(
        R=R, title=title, sub=subtitle, exag=float(exaggeration), whole=bool(whole),
        notes=[str(s) for s in notes if s], M=M.tolist(), source=grid.source,
        tile=dict(gh=gh, gw=gw, p0=p0.tolist(), ps=ps.tolist(), h0=h0, hs=hs,
                  reg=[grid.x0, grid.y0, grid.step], rect=rect, spacing_km=km_px * grid.step),
        stretch=dict(w=w, m=m), view=view, globe=globe)
    meta_json = json.dumps(meta).replace("</", "<\\/")
    page = _TEMPLATE
    # payloads first (base64 never holds a placeholder), the user's title last
    parts = {"__TILE_P__": _b64(qp), "__TILE_H__": _b64(qh), "__TEX__": tex_b64,
             "__GLOBE_H__": globe_parts[0], "__GLOBE_SD__": globe_parts[1], "__ALB__": globe_parts[2]}
    for k, v in parts.items():
        page = page.replace(k, v)
    page = page.replace("__META__", meta_json)
    return page.replace("__TITLE__", _html.escape(title))


def display_image(image01, grid) -> np.ndarray:
    """The image with the viewer's display stretch (for mesh export colours)."""
    img = np.asarray(image01, np.float32)
    mono = img if img.ndim == 2 else img[..., :3].mean(axis=2)
    w, m = display_stretch(_sample_nearest(mono, grid.uv[grid.valid][:: max(1, int(grid.valid.sum()) // 200000)]))
    x = np.clip(img / float(w), 0.0, 1.0)
    return _mtf(x, m).astype(np.float32) if m > 0 else x


# ---------------------------------------------------------------------------
# The page.  Plain WebGL (1 or 2) so it opens offline in any current browser.
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ - Lunar Relief</title>
<style>
html,body{margin:0;height:100%;overflow:hidden;background:#000;color:#d8dce2;
  font:13px/1.4 "Segoe UI",system-ui,-apple-system,sans-serif}
canvas{display:block;width:100%;height:100%;outline:none;touch-action:none;cursor:grab}
canvas.drag{cursor:grabbing}
.ov{position:absolute;text-shadow:0 0 4px #000,0 0 2px #000}
#title{left:14px;top:10px;pointer-events:none;max-width:60%}
#title b{font-size:16px;color:#fff;font-weight:600}
#title span{color:#aab}
#notes{color:#e8c27a;margin-top:4px;font-size:12px}
#hud{left:14px;bottom:10px;pointer-events:none;color:#c8ccd4}
#hud .hint{color:#8a8f99;font-size:12px}
#panel{position:absolute;right:12px;top:12px;width:262px;background:rgba(18,20,24,.88);
  border:1px solid #2e333b;border-radius:8px;padding:8px 12px 10px;user-select:none}
#panel h3{margin:0;font-size:13px;font-weight:600;display:flex;justify-content:space-between;align-items:center}
#panel.min .body{display:none}
.body{margin-top:6px}
.row{margin:8px 0 0}
.row label{display:flex;justify-content:space-between;color:#b9bec7}
.row output{color:#fff}
input[type=range]{width:100%;margin:2px 0 0;accent-color:#7fa7ff}
.ck{display:flex;gap:6px;align-items:flex-start;margin-top:8px;color:#b9bec7}
.ck input{margin:2px 0 0}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
button{background:#2a2f38;color:#e4e7ec;border:1px solid #3b414c;border-radius:5px;padding:5px 6px;
  font:inherit;cursor:pointer}
button:hover{background:#353b46}
#min{padding:0 7px;line-height:18px}
#help{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.6)}
#help.show{display:flex}
#help .box{background:#15181d;border:1px solid #39404b;border-radius:10px;padding:16px 22px;max-width:560px}
#help h2{margin:0 0 8px;font-size:16px}
#help table{border-collapse:collapse}
#help td{padding:3px 10px 3px 0;vertical-align:top}
#help td:first-child{white-space:nowrap;color:#fff}
#msg{position:absolute;left:50%;top:45%;transform:translate(-50%,-50%);font-size:15px;color:#ccd;
  background:rgba(0,0,0,.7);padding:12px 18px;border-radius:8px;text-align:center;max-width:70%}
</style></head>
<body>
<canvas id="c" tabindex="0"></canvas>
<div id="title" class="ov"><b>__TITLE__</b><br><span id="sub"></span><div id="notes"></div></div>
<div id="hud" class="ov"><div id="hudc">&nbsp;</div><div id="huda">&nbsp;</div>
  <div class="hint">Left-drag rotate &middot; Right-drag move &middot; Wheel zoom &middot; Middle-drag look around &middot; WASD fly &middot; Double-click fly there &middot; H help</div></div>
<div id="panel"><h3><span>Lunar Relief</span><button id="min" title="Hide / show">&ndash;</button></h3>
<div class="body">
  <div class="btns">
    <button data-v="earth" title="As your telescope saw it (1)">From Earth</button>
    <button data-v="tilt" title="Oblique view (2)">Tilted</button>
    <button data-v="top" title="Straight down, north up (3)">Top down</button>
    <button data-v="globe" title="The whole Moon (4)">Whole Moon</button>
  </div>
  <div class="row"><label>Height exaggeration<output id="ok"></output></label>
    <input type="range" id="k" min="1" max="20" step="0.5"></div>
  <div class="row"><label>Brightness<output id="ob"></output></label>
    <input type="range" id="b" min="-2" max="2" step="0.05" value="0"></div>
  <div class="row"><label>Gamma<output id="og"></output></label>
    <input type="range" id="g" min="0.4" max="2.5" step="0.05" value="1"></div>
  <div class="row"><label>Light up shadows<output id="of"></output></label>
    <input type="range" id="f" min="0" max="0.4" step="0.01" value="0"></div>
  <div id="globeRows">
    <label class="ck"><input type="checkbox" id="gl" checked><span>Rest of the Moon<br>
      <small style="color:#8a8f99">NASA LOLA + Clementine, lit by the same Sun</small></span></label>
    <div class="row"><label>Rest of the Moon brightness<output id="oa"></output></label>
      <input type="range" id="a" min="0" max="2" step="0.05" value="1"></div>
  </div>
  <div class="btns"><button id="shot">Save picture</button><button id="helpb">Controls (H)</button></div>
</div></div>
<div id="help"><div class="box"><h2>Controls</h2><table>
<tr><td>Left-drag</td><td>Rotate around the spot under the cursor (around the whole Moon when far out)</td></tr>
<tr><td>Right-drag, Shift+drag</td><td>Grab the surface and move across it</td></tr>
<tr><td>Wheel</td><td>Zoom toward the cursor</td></tr>
<tr><td>Middle-drag, Ctrl+drag</td><td>Look around from where you are</td></tr>
<tr><td>W A S D / arrows</td><td>Fly over the surface (Q / E down / up, hold Shift for faster)</td></tr>
<tr><td>Double-click</td><td>Fly to that spot</td></tr>
<tr><td>1 2 3 4</td><td>From Earth, Tilted, Top down, Whole Moon</td></tr>
<tr><td>H, Esc</td><td>Show / hide this help</td></tr>
</table><p style="color:#8a8f99;margin:10px 0 0">Horizontal distances are true; heights are exaggerated by
the slider. Your image is laid on the terrain exactly where each pixel saw it.</p></div></div>
<div id="msg">Loading&hellip;</div>
<script id="meta" type="application/json">__META__</script>
<script id="b_tp" type="application/octet-stream">__TILE_P__</script>
<script id="b_th" type="application/octet-stream">__TILE_H__</script>
<script id="b_gh" type="application/octet-stream">__GLOBE_H__</script>
<script id="b_gs" type="application/octet-stream">__GLOBE_SD__</script>
<img id="tex" alt="" hidden src="data:image/jpeg;base64,__TEX__">
<img id="alb" alt="" hidden src="data:image/jpeg;base64,__ALB__">
<script>
"use strict";
(function () {
const $ = (id) => document.getElementById(id);
const msg = (t) => { const m = $("msg"); m.innerHTML = t; m.style.display = t ? "block" : "none"; };
const META = JSON.parse($("meta").textContent);
const R = META.R, MC = [0, 0, -R], VW = META.view, G = META.globe;
const MR = META.M;                     // rows e, n, u: world = M (X - C)

// ---- small vector kit ----------------------------------------------------
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
function rot(v, k, ang) {             // Rodrigues, k unit
  const c = Math.cos(ang), s = Math.sin(ang), d = dot(k, v) * (1 - c), x = cross(k, v);
  return [v[0] * c + x[0] * s + k[0] * d, v[1] * c + x[1] * s + k[1] * d, v[2] * c + x[2] * s + k[2] * d];
}
const radial = (p) => norm(sub(p, MC));
function toBody(d) { return [MR[0][0] * d[0] + MR[1][0] * d[1] + MR[2][0] * d[2],
                             MR[0][1] * d[0] + MR[1][1] * d[1] + MR[2][1] * d[2],
                             MR[0][2] * d[0] + MR[1][2] * d[1] + MR[2][2] * d[2]]; }
function fromBody(d) { return [dot(MR[0], d), dot(MR[1], d), dot(MR[2], d)]; }
function latlon(p) { const b = toBody(radial(p));
  return [Math.asin(clamp(b[2], -1, 1)) * 180 / Math.PI, Math.atan2(b[1], b[0]) * 180 / Math.PI]; }
function bytes(id) {
  const el = $(id); const s = atob(el.textContent.trim()); el.textContent = "";
  const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return u.buffer;
}

// ---- WebGL ----------------------------------------------------------------
const canvas = $("c");
let gl = canvas.getContext("webgl2", { antialias: true });
const GL2 = !!gl;
if (!gl) {
  gl = canvas.getContext("webgl", { antialias: true }) || canvas.getContext("experimental-webgl");
  if (gl && !gl.getExtension("OES_element_index_uint")) gl = null;
}
if (!gl) { msg("This browser cannot show 3D (WebGL is off or not supported).<br>Try Chrome, Edge or Firefox with hardware acceleration on."); return; }
const ANISO = gl.getExtension("EXT_texture_filter_anisotropic") || gl.getExtension("WEBKIT_EXT_texture_filter_anisotropic");

const VS = `
attribute vec3 aP; attribute float aH; attribute vec2 aT; attribute vec3 aN; attribute float aS;
uniform mat4 uMVP; uniform vec3 uMc; uniform float uR, uK, uInset;
varying vec2 vT; varying vec3 vN; varying vec3 vW; varying float vS; varying vec3 vR;
void main() {
  vec3 r = normalize(aP - uMc);
  vec3 p = uMc + r * (uR + uK * aH - uInset);
  vW = p; vT = aT; vN = aN; vS = aS; vR = r;
  gl_Position = uMVP * vec4(p, 1.0);
}`;
const FS = `
precision highp float;
uniform sampler2D uTex; uniform int uMode;
uniform float uGain, uGamma, uFill, uCut, uAlbG, uSw, uSm, uBack;
uniform vec3 uSun, uEarth, uHead;
varying vec2 vT; varying vec3 vN; varying vec3 vW; varying float vS; varying vec3 vR;
float mtf(float x, float m) { return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m); }
void main() {
  if (uMode == 1 && vS > uCut) discard;
  vec3 n = normalize(vN);
  vec3 c;
  if (uMode == 1) {
    float a = texture2D(uTex, vT).r;
    float ci = dot(n, uSun);
    float ce = max(dot(normalize(vR), uEarth), 0.2);
    float ls = ci > 0.0 ? ci / (ci + ce) : 0.0;
    float x = clamp(uAlbG * a * ls / uSw, 0.0, 1.0);
    if (uSm > 0.0) x = mtf(x, uSm);
    c = vec3(x * uBack);
  } else {
    c = texture2D(uTex, vT).rgb * (1.0 - 0.55 * clamp(vS, 0.0, 1.0));   // vS = 1 on the edge walls
  }
  c = pow(max(c * uGain, vec3(0.0)), vec3(uGamma));
  c += uFill * (0.3 + 0.7 * max(dot(n, uHead), 0.0));
  gl_FragColor = vec4(min(c, vec3(1.0)), 1.0);
}`;
const FSP = `
precision highp float;
uniform int uMode; uniform float uCut, uFar; uniform vec3 uEye;
varying vec2 vT; varying vec3 vN; varying vec3 vW; varying float vS; varying vec3 vR;
void main() {
  if (uMode == 1 && vS > uCut) discard;
  float d = floor(clamp(length(vW - uEye) / uFar, 0.0, 1.0) * 16777215.0);
  float r = floor(d / 65536.0); d -= r * 65536.0;
  float g = floor(d / 256.0); float b = d - g * 256.0;
  gl_FragColor = vec4(r / 255.0, g / 255.0, b / 255.0, uMode == 1 ? 0.5 : 1.0);   // alpha: which surface
}`;
function shader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(fs) {
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, VS)); gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
  ["aP", "aH", "aT", "aN", "aS"].forEach((n, i) => gl.bindAttribLocation(p, i, n));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
  return { p, u };
}
function buffer(data, target) {
  const b = gl.createBuffer(); gl.bindBuffer(target || gl.ARRAY_BUFFER, b);
  gl.bufferData(target || gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); return b;
}
function texture(img) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  let src = img;
  const mx = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  if (img.naturalWidth > mx || img.naturalHeight > mx) {
    const s = mx / Math.max(img.naturalWidth, img.naturalHeight), cv = document.createElement("canvas");
    cv.width = Math.floor(img.naturalWidth * s); cv.height = Math.floor(img.naturalHeight * s);
    cv.getContext("2d").drawImage(img, 0, 0, cv.width, cv.height); src = cv;
  }
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, src);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  if (GL2) {
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    if (ANISO) gl.texParameterf(gl.TEXTURE_2D, ANISO.TEXTURE_MAX_ANISOTROPY_EXT,
                                Math.min(8, gl.getParameter(ANISO.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
  } else {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  return t;
}

// ---- meshes -----------------------------------------------------------------
function gridIndex(gh, gw, ok) {
  const out = new Uint32Array((gh - 1) * (gw - 1) * 6); let n = 0;
  for (let j = 0; j < gh - 1; j++) for (let i = 0; i < gw - 1; i++) {
    const a = j * gw + i, b = a + 1, c = a + gw, d = c + 1;
    if (ok && !(ok[a] && ok[b] && ok[c] && ok[d])) continue;
    out[n++] = a; out[n++] = c; out[n++] = b; out[n++] = b; out[n++] = c; out[n++] = d;
  }
  return out.subarray(0, n);
}
// normals of the exaggerated surface (central differences on the grid)
function gridNormals(P, H, gh, gw, ok, k, inset, wrap, out) {
  const nv = gh * gw, E = new Float32Array(nv * 3);
  for (let v = 0; v < nv; v++) {
    const x = P[3 * v] - MC[0], y = P[3 * v + 1] - MC[1], z = P[3 * v + 2] - MC[2];
    const s = (R + k * H[v] - inset) / (Math.hypot(x, y, z) || 1);
    E[3 * v] = MC[0] + x * s; E[3 * v + 1] = MC[1] + y * s; E[3 * v + 2] = MC[2] + z * s;
  }
  for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
    const v = j * gw + i;
    let il = i - 1, ir = i + 1;
    if (wrap) { if (il < 0) il = gw - 2; if (ir >= gw) ir = 1; }
    let l = il >= 0 ? j * gw + il : v, r = ir < gw ? j * gw + ir : v;
    let u = j > 0 ? v - gw : v, d = j < gh - 1 ? v + gw : v;
    if (ok) { if (!ok[l]) l = v; if (!ok[r]) r = v; if (!ok[u]) u = v; if (!ok[d]) d = v; }
    const ax = E[3 * r] - E[3 * l], ay = E[3 * r + 1] - E[3 * l + 1], az = E[3 * r + 2] - E[3 * l + 2];
    const bx = E[3 * d] - E[3 * u], by = E[3 * d + 1] - E[3 * u + 1], bz = E[3 * d + 2] - E[3 * u + 2];
    let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const rx = E[3 * v] - MC[0], ry = E[3 * v + 1] - MC[1], rz = E[3 * v + 2] - MC[2];
    if (nx * rx + ny * ry + nz * rz < 0) { nx = -nx; ny = -ny; nz = -nz; }
    let L = Math.hypot(nx, ny, nz);
    if (!(L > 0)) { nx = rx; ny = ry; nz = rz; L = Math.hypot(nx, ny, nz) || 1; }
    out[3 * v] = nx / L; out[3 * v + 1] = ny / L; out[3 * v + 2] = nz / L;
  }
}

const T = META.tile, TGH = T.gh, TGW = T.gw, TNV = TGH * TGW;
const SKIRT = 4.0;          // km (true height) the walls along the image's edges hang down
let tile, globe = null, tileClamp, globeClamp, progMain, progPick, texTile, texAlb;
let tP, tH, tT, tN, tOK, skOrig;
let gP, gH, gT, gS, gN, GR, GC;

function buildTile() {
  const qp = new Uint16Array(bytes("b_tp")), qh = new Uint16Array(bytes("b_th"));
  tOK = new Uint8Array(TNV);
  for (let v = 0; v < TNV; v++) if (qh[v] !== 65535) tOK[v] = 1;
  // skirts: a wall hanging from every outer edge, so the gap between the image's
  // edge and the (lower, coarser) rest of the Moon never shows through to space
  const qok = (j, i) => j >= 0 && i >= 0 && j < TGH - 1 && i < TGW - 1 && tOK[j * TGW + i] &&
      tOK[j * TGW + i + 1] && tOK[(j + 1) * TGW + i] && tOK[(j + 1) * TGW + i + 1];
  const edges = [];
  for (let j = 0; j < TGH - 1; j++) for (let i = 0; i < TGW - 1; i++) {
    if (!qok(j, i)) continue;
    const a = j * TGW + i, b = a + 1, c = a + TGW, d = c + 1;
    if (!qok(j - 1, i)) edges.push(a, b);
    if (!qok(j + 1, i)) edges.push(c, d);
    if (!qok(j, i - 1)) edges.push(a, c);
    if (!qok(j, i + 1)) edges.push(b, d);
  }
  const sk = new Map();
  for (const v of edges) if (!sk.has(v)) sk.set(v, TNV + sk.size);
  const NV = TNV + sk.size;
  tP = new Float32Array(NV * 3); tH = new Float32Array(NV); tT = new Float32Array(NV * 2); tN = new Float32Array(NV * 3);
  for (let v = 0; v < TNV; v++) {
    for (let k = 0; k < 3; k++) tP[3 * v + k] = T.p0[k] + qp[3 * v + k] * T.ps[k];
    if (tOK[v]) tH[v] = T.h0 + qh[v] * T.hs;
  }
  const [x0, y0, st] = T.reg, [rx, ry, rw, rh] = T.rect;
  for (let j = 0; j < TGH; j++) for (let i = 0; i < TGW; i++) {
    const v = j * TGW + i;
    tT[2 * v] = (x0 + i * st - rx + 0.5) / rw; tT[2 * v + 1] = (y0 + j * st - ry + 0.5) / rh;
  }
  skOrig = new Uint32Array(sk.size);
  for (const [v, s] of sk) {
    skOrig[s - TNV] = v;
    tP[3 * s] = tP[3 * v]; tP[3 * s + 1] = tP[3 * v + 1]; tP[3 * s + 2] = tP[3 * v + 2];
    tH[s] = tH[v] - SKIRT; tT[2 * s] = tT[2 * v]; tT[2 * s + 1] = tT[2 * v + 1];
  }
  const base = gridIndex(TGH, TGW, tOK), idx = new Uint32Array(base.length + edges.length * 3);
  idx.set(base);
  let n = base.length;
  for (let e = 0; e < edges.length; e += 2) {
    const a = edges[e], b = edges[e + 1], sa = sk.get(a), sb = sk.get(b);
    idx[n++] = a; idx[n++] = b; idx[n++] = sa; idx[n++] = b; idx[n++] = sb; idx[n++] = sa;
  }
  const wall = new Float32Array(NV); wall.fill(1, TNV);
  tile = { bP: buffer(tP), bH: buffer(tH), bT: buffer(tT), bN: buffer(tN), bS: buffer(wall),
           bI: buffer(idx, gl.ELEMENT_ARRAY_BUFFER), n: idx.length };
  // ground-clearance map: highest point per cell of the sphere's tangent plane
  let xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
  const sx = new Float32Array(TNV), sy = new Float32Array(TNV);
  for (let v = 0; v < TNV; v++) {
    if (!tOK[v]) continue;
    const d = radial([tP[3 * v], tP[3 * v + 1], tP[3 * v + 2]]);
    sx[v] = MC[0] + R * d[0]; sy[v] = MC[1] + R * d[1];
    xmin = Math.min(xmin, sx[v]); xmax = Math.max(xmax, sx[v]); ymin = Math.min(ymin, sy[v]); ymax = Math.max(ymax, sy[v]);
  }
  let b = Math.max(2 * T.spacing_km, 0.05);
  b = Math.max(b, (xmax - xmin) / 2048, (ymax - ymin) / 2048);
  const nx = Math.ceil((xmax - xmin) / b) + 1, ny = Math.ceil((ymax - ymin) / b) + 1;
  const hm = new Float32Array(nx * ny).fill(-1e9);
  for (let v = 0; v < TNV; v++) {
    if (!tOK[v]) continue;
    const c = Math.floor((sx[v] - xmin) / b) + nx * Math.floor((sy[v] - ymin) / b);
    if (tH[v] > hm[c]) hm[c] = tH[v];
  }
  const hd = Float32Array.from(hm);              // grow one cell so gaps between rows fill
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let m = hm[j * nx + i];
    for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
      const jj = j + dj, ii = i + di;
      if (jj >= 0 && jj < ny && ii >= 0 && ii < nx) m = Math.max(m, hm[jj * nx + ii]);
    }
    hd[j * nx + i] = m;
  }
  tileClamp = (d) => {                            // d: unit radial of the camera
    if (d[2] < 0.05) return null;
    const x = MC[0] + R * d[0], y = MC[1] + R * d[1];
    const i = Math.floor((x - xmin) / b), j = Math.floor((y - ymin) / b);
    if (i < 0 || j < 0 || i >= nx || j >= ny) return null;
    const h = hd[j * nx + i]; return h > -1e8 ? h : null;
  };
}

function buildGlobe() {
  GR = G.rows; GC = G.cols;
  const nv = GR * GC, qh = new Uint16Array(bytes("b_gh")), qs = new Int16Array(bytes("b_gs"));
  gP = new Float32Array(nv * 3); gH = new Float32Array(nv); gT = new Float32Array(nv * 2);
  gS = new Float32Array(nv); gN = new Float32Array(nv * 3);
  for (let r = 0; r < GR; r++) {
    const lat = (90 - r / G.ppd) * Math.PI / 180, cl = Math.cos(lat), sl = Math.sin(lat);
    for (let c = 0; c < GC; c++) {
      const v = r * GC + c, lon = (c / G.ppd) * Math.PI / 180;
      const w = fromBody([cl * Math.cos(lon), cl * Math.sin(lon), sl]);
      gP[3 * v] = MC[0] + R * w[0]; gP[3 * v + 1] = MC[1] + R * w[1]; gP[3 * v + 2] = MC[2] + R * w[2];
      gH[v] = G.h0 + qh[v] * G.hs; gS[v] = qs[v] * G.sd_scale;
      gT[2 * v] = c / (GC - 1); gT[2 * v + 1] = r / (GR - 1);
    }
  }
  const idx = gridIndex(GR, GC, null);
  globe = { bP: buffer(gP), bH: buffer(gH), bT: buffer(gT), bN: buffer(gN), bS: buffer(gS),
            bI: buffer(idx, gl.ELEMENT_ARRAY_BUFFER), n: idx.length };
  const hm = new Float32Array(nv);               // max over the 4 nodes of each cell
  for (let r = 0; r < GR; r++) for (let c = 0; c < GC; c++) {
    const v = r * GC + c, r1 = Math.min(r + 1, GR - 1), c1 = Math.min(c + 1, GC - 1);
    hm[v] = Math.max(gH[v], gH[r * GC + c1], gH[r1 * GC + c], gH[r1 * GC + c1]);
  }
  globeClamp = (d) => {
    const b = toBody(d), lat = Math.asin(clamp(b[2], -1, 1)) * 180 / Math.PI;
    let lon = Math.atan2(b[1], b[0]) * 180 / Math.PI; if (lon < 0) lon += 360;
    const r = clamp(Math.floor((90 - lat) * G.ppd), 0, GR - 1), c = clamp(Math.floor(lon * G.ppd), 0, GC - 1);
    return hm[r * GC + c];
  };
}

// ---- state ------------------------------------------------------------------------
const S = { k: META.exag, gain: 1, gamma: 1, fill: 0, globe: !!G, back: 1 };
const cam = { P: [0, 0, 100], F: [0, 0, -1], U: [0, 1, 0], fov: 45 * Math.PI / 180 };
let W = 1, H = 1, dpr = 1, pivot = null, raf = 0, anim = null, last = 0;
const keys = new Set();
const inset = () => 0.3 + 0.2 * S.k;
const right = () => norm(cross(cam.F, cam.U));
function level(hint) {
  let u = sub(hint, mul(cam.F, dot(hint, cam.F)));
  if (len(u) < 1e-6) u = sub(cam.U, mul(cam.F, dot(cam.U, cam.F)));
  cam.U = norm(u);
}
function groundR(p) {                    // radius of the drawn ground under p (null = none)
  const d = radial(p); let r = null;
  const ht = tileClamp(d); if (ht !== null) r = R + S.k * ht;
  if (S.globe && globeClamp) { const g = R + S.k * globeClamp(d) - inset(); r = r === null ? g : Math.max(r, g); }
  return r;
}
function keepAbove() {
  const g = groundR(cam.P); if (g === null) return;
  const lo = g + 0.05, dc = len(sub(cam.P, MC));
  if (dc < lo) cam.P = add(MC, mul(radial(cam.P), lo));
}
function altitude() {
  const g = groundR(cam.P), dc = len(sub(cam.P, MC));
  return dc - (g === null ? R : g);
}
function turn(pt, axis, ang) {           // rigid turn of the camera about a point
  cam.P = add(pt, rot(sub(cam.P, pt), axis, ang)); cam.F = norm(rot(cam.F, axis, ang)); cam.U = norm(rot(cam.U, axis, ang));
}

// ---- matrices ------------------------------------------------------------------
function frameMats() {
  const f = cam.F, r = right(), u = cross(r, f), p = cam.P;
  const view = [r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0,
                -dot(r, p), -dot(u, p), dot(f, p), 1];
  const dc = len(sub(p, MC));
  const near = clamp(altitude() * 0.1, 0.001, 2000), far = dc + 1.1 * R + 50;
  const t = 1 / Math.tan(cam.fov / 2), a = W / H;
  const proj = [t / a, 0, 0, 0, 0, t, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0];
  const m = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let rr = 0; rr < 4; rr++) {
    let s = 0; for (let k = 0; k < 4; k++) s += proj[k * 4 + rr] * view[c * 4 + k]; m[c * 4 + rr] = s;
  }
  return { mvp: m, far };
}
function rayDir(cx, cy) {
  const w = canvas.clientWidth, h = canvas.clientHeight, t = Math.tan(cam.fov / 2);
  const r = right(), u = cross(r, cam.F);
  return norm(add(cam.F, add(mul(r, (2 * cx / w - 1) * t * w / h), mul(u, (1 - 2 * cy / h) * t))));
}
function sphereHit(o, d, rad) {
  const oc = sub(o, MC), b = dot(oc, d), c = dot(oc, oc) - rad * rad, disc = b * b - c;
  if (disc < 0) return null;
  const t = -b - Math.sqrt(disc); return t > 0 ? add(o, mul(d, t)) : null;
}

// ---- drawing ------------------------------------------------------------------
function bindMesh(m) {
  const at = (loc, b, n) => { gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0); };
  at(0, m.bP, 3); at(1, m.bH, 1); at(2, m.bT, 2); at(3, m.bN, 3);
  if (m.bS) at(4, m.bS, 1); else { gl.disableVertexAttribArray(4); gl.vertexAttrib1f(4, -1e6); }
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.bI);
}
function common(pr, mats) {
  const u = pr.u;
  gl.uniformMatrix4fv(u.uMVP, false, mats.mvp);
  gl.uniform3fv(u.uMc, MC); gl.uniform1f(u.uR, R); gl.uniform1f(u.uK, S.k);
  if (u.uCut) gl.uniform1f(u.uCut, G ? G.cut_px : 0);
}
function drawMeshes(pr, mats, pick) {
  const u = pr.u;
  gl.uniform1i(u.uMode, 0); gl.uniform1f(u.uInset, 0);
  if (!pick) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texTile); gl.uniform1i(u.uTex, 0); }
  bindMesh(tile); gl.drawElements(gl.TRIANGLES, tile.n, gl.UNSIGNED_INT, 0);
  if (globe && S.globe) {
    gl.uniform1i(u.uMode, 1); gl.uniform1f(u.uInset, inset());
    if (!pick) gl.bindTexture(gl.TEXTURE_2D, texAlb);
    bindMesh(globe); gl.drawElements(gl.TRIANGLES, globe.n, gl.UNSIGNED_INT, 0);
  }
}
function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (w !== W || h !== H) {
    W = w; H = h; canvas.width = W; canvas.height = H;
    if (pickFbo) gl.deleteFramebuffer(pickFbo);
    pickFbo = null;
  }
}
function render() {
  resize();
  const mats = frameMats();
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, W, H); gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
  const pr = progMain, u = pr.u; gl.useProgram(pr.p); common(pr, mats);
  gl.uniform1f(u.uGain, S.gain); gl.uniform1f(u.uGamma, S.gamma); gl.uniform1f(u.uFill, S.fill);
  gl.uniform1f(u.uSw, META.stretch.w); gl.uniform1f(u.uSm, META.stretch.m);
  gl.uniform1f(u.uAlbG, G ? G.gain : 1); gl.uniform1f(u.uBack, S.back);
  gl.uniform3fv(u.uSun, norm(VW.sun)); gl.uniform3fv(u.uEarth, norm(VW.earth));
  gl.uniform3fv(u.uHead, norm(add(mul(cam.F, -1), add(mul(cam.U, 0.6), mul(right(), -0.4)))));
  drawMeshes(pr, mats, false);
  pickStale = true;
  hudAlt();
}
function req() { if (!raf) raf = requestAnimationFrame(tick); }

// ---- picking (distance along the cursor ray, encoded in RGB) ----------------------
let pickFbo = null, pickTex, pickDepth, pickStale = true, pickFar = 1;
const px = new Uint8Array(4);
function pickTarget() {
  if (pickFbo) return;
  if (pickTex) { gl.deleteTexture(pickTex); gl.deleteRenderbuffer(pickDepth); }
  pickStale = true;
  pickFbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, pickFbo);
  pickTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, pickTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pickTex, 0);
  pickDepth = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, pickDepth);
  gl.renderbufferStorage(gl.RENDERBUFFER, GL2 ? gl.DEPTH_COMPONENT24 : gl.DEPTH_COMPONENT16, W, H);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, pickDepth);
}
let pickSig = "";
function pick(cx, cy) {
  resize(); pickTarget();
  gl.bindFramebuffer(gl.FRAMEBUFFER, pickFbo);
  // the pick image must be of this exact camera (moves render on the next frame)
  const sig = [cam.P, cam.F, cam.U, W, H, S.k, S.globe].join(",");
  if (pickStale || sig !== pickSig) {
    pickSig = sig;
    const mats = frameMats(); pickFar = mats.far;
    gl.viewport(0, 0, W, H); gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
    const pr = progPick; gl.useProgram(pr.p); common(pr, mats);
    gl.uniform1f(pr.u.uFar, pickFar); gl.uniform3fv(pr.u.uEye, cam.P);
    drawMeshes(pr, mats, true);
    pickStale = false;
  }
  const ix = clamp(Math.floor(cx * dpr), 0, W - 1), iy = clamp(Math.floor(cy * dpr), 0, H - 1);
  gl.readPixels(ix, H - 1 - iy, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (px[3] === 0) return null;
  const d = (px[0] * 65536 + px[1] * 256 + px[2]) / 16777215 * pickFar;
  // the distance belongs to that pixel's CENTRE ray (half a pixel is ~0.6 km at 1300 km)
  const q = add(cam.P, mul(rayDir((ix + 0.5) / dpr, (iy + 0.5) / dpr), d));
  q.globe = px[3] < 192;                 // hit the rest of the Moon (drawn inset), not the image
  return q;
}

// ---- views ---------------------------------------------------------------------------
const fovT = () => Math.tan(cam.fov / 2);
function viewPose(name) {
  const E = norm(VW.earth), T0 = VW.target, up0 = radial(T0);
  const big = META.whole || VW.ext > 0.6 * R;     // a disc or most of one: frame the globe
  let P, F, Uh;
  if (name === "globe" || (name === "earth" && big)) {
    const d = (name === "globe" ? 1.25 : 1.06) * R / Math.sin(cam.fov / 2);
    P = add(MC, mul(E, d)); F = mul(E, -1); Uh = VW.up;
  } else if (name === "earth") {
    P = add(T0, mul(E, VW.ext * 1.1 / fovT())); F = mul(E, -1); Uh = VW.up;
  } else if (name === "top") {
    P = add(T0, mul(up0, Math.min(VW.ext, R) * 1.1 / fovT())); F = mul(up0, -1); Uh = VW.north;
  } else if (big) {                         // tilted disc: turn the globe 35 deg
    // (from the image's bottom edge, like the close-up view)
    let dn = sub(VW.up, mul(E, dot(VW.up, E)));
    dn = len(dn) > 1e-6 ? norm(dn) : [0, 0, 1];
    const a = 35 * Math.PI / 180, dir = norm(add(mul(E, Math.cos(a)), mul(dn, -Math.sin(a))));
    P = add(MC, mul(dir, 1.1 * R / Math.sin(cam.fov / 2))); F = mul(dir, -1); Uh = VW.up;
  } else {                                  // tilted: from the image's bottom edge, 32 deg up
    let hz = sub(VW.up, mul(up0, dot(VW.up, up0)));
    hz = len(hz) > 1e-6 ? norm(hz) : norm(sub(VW.north, mul(up0, dot(VW.north, up0))));
    const el = 32 * Math.PI / 180, d = VW.ext / fovT();
    P = add(T0, mul(add(mul(up0, Math.sin(el)), mul(hz, -Math.cos(el))), d));
    F = norm(sub(T0, P)); Uh = up0;
  }
  let U = sub(Uh, mul(F, dot(Uh, F)));
  if (len(U) < 1e-6) U = Math.abs(F[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  return { P, F: norm(F), U: norm(sub(U, mul(F, dot(U, F)))) };
}
function setPose(p) { cam.P = p.P; cam.F = p.F; cam.U = p.U; }
function slerp(a, b, s) {
  const c = clamp(dot(a, b), -1, 1), om = Math.acos(c);
  if (om < 1e-5) return norm(add(mul(a, 1 - s), mul(b, s)));
  let ax = cross(a, b);
  if (len(ax) < 1e-9) ax = Math.abs(a[0]) < 0.9 ? cross(a, [1, 0, 0]) : cross(a, [0, 1, 0]);
  return norm(rot(a, norm(ax), om * s));
}
function flyTo(p, ms) {
  anim = { a: { P: cam.P, F: cam.F, U: cam.U }, b: p, t0: performance.now(), ms: ms || 900 }; req();
}
function stepAnim(now) {
  const s0 = clamp((now - anim.t0) / anim.ms, 0, 1), s = s0 * s0 * (3 - 2 * s0), a = anim.a, b = anim.b;
  const ra = sub(a.P, MC), rb = sub(b.P, MC), la = len(ra), lb = len(rb);
  cam.P = add(MC, mul(slerp(norm(ra), norm(rb), s), la * Math.pow(lb / la, s)));
  cam.F = slerp(a.F, b.F, s);
  const u = slerp(a.U, b.U, s);
  cam.U = norm(sub(u, mul(cam.F, dot(u, cam.F))));
  if (s0 >= 1) anim = null;
}

// ---- input ------------------------------------------------------------------------
let drag = null;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  canvas.focus(); anim = null;
  let mode = "orbit";
  if (e.button === 2 || (e.button === 0 && e.shiftKey)) mode = "pan";
  if (e.button === 1 || (e.button === 0 && (e.ctrlKey || e.altKey || e.metaKey))) mode = "look";
  e.preventDefault();
  const q = pick(e.offsetX, e.offsetY);
  drag = { mode, x: e.offsetX, y: e.offsetY, id: e.pointerId };
  if (mode === "orbit") {
    const far = len(sub(cam.P, MC)) > 2.2 * R;
    if (far || (!q && !pivot)) drag.globe = true;
    else { drag.globe = false; drag.pt = q || pivot; pivot = drag.pt; }
  } else if (mode === "pan") {
    const g = q || sphereHit(cam.P, rayDir(e.offsetX, e.offsetY), R);
    if (g) { drag.g = g; drag.rg = len(sub(g, MC)); }
  }
  try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* synthetic / lost pointer */ }
  canvas.classList.add("drag");
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) { hoverAt(e.offsetX, e.offsetY); return; }
  const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y; drag.x = e.offsetX; drag.y = e.offsetY;
  if (!dx && !dy) return;
  const a = 0.0055;
  if (drag.mode === "orbit") {
    if (drag.globe) {
      turn(MC, cam.U, -dx * a); turn(MC, right(), -dy * a);
    } else {
      const T0 = drag.pt, A = radial(T0);
      turn(T0, A, -dx * a);
      const r = right(), v = norm(sub(cam.P, T0));
      const el = Math.asin(clamp(dot(v, A), -1, 1)), sg = Math.sign(dot(cross(r, v), A)) || 1;
      const el2 = clamp(el + dy * a, -45 * Math.PI / 180, 89.5 * Math.PI / 180);
      turn(T0, r, sg * (el2 - el));
    }
  } else if (drag.mode === "pan") {
    if (drag.g) {
      const h = sphereHit(cam.P, rayDir(e.offsetX, e.offsetY), drag.rg);
      if (h) {
        const a1 = norm(sub(h, MC)), b1 = norm(sub(drag.g, MC)), ax = cross(a1, b1), s = len(ax);
        if (s > 1e-12) turn(MC, mul(ax, 1 / s), Math.min(Math.atan2(s, dot(a1, b1)), 0.25));
      }
    } else {
      const d = len(sub(cam.P, MC)) * 2 * fovT() / canvas.clientHeight;
      cam.P = add(cam.P, add(mul(right(), -dx * d), mul(cam.U, dy * d)));
    }
  } else {
    const up = radial(cam.P);
    turn(cam.P, up, -dx * a * 0.6);
    const r = right(), el = Math.asin(clamp(dot(cam.F, up), -1, 1));
    const el2 = clamp(el - dy * a * 0.6, -89 * Math.PI / 180, 89 * Math.PI / 180);
    turn(cam.P, r, el2 - el);
    level(up);
  }
  keepAbove(); req();
});
const endDrag = (e) => { if (drag && e.pointerId === drag.id) { drag = null; canvas.classList.remove("drag"); } };
canvas.addEventListener("pointerup", endDrag); canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault(); anim = null;
  let steps = e.deltaY / (e.deltaMode === 1 ? 3 : e.deltaMode === 2 ? 0.3 : 100);
  steps = clamp(steps, -4, 4);
  const k = Math.pow(0.82, -steps), dir = rayDir(e.offsetX, e.offsetY);
  // toward the surface under the cursor; over empty sky, toward the middle of the view
  let q = pick(e.offsetX, e.offsetY) || sphereHit(cam.P, dir, R) ||
          pick(canvas.clientWidth / 2, canvas.clientHeight / 2) || sphereHit(cam.P, cam.F, R) || MC;
  const P2 = add(q, mul(sub(cam.P, q), k));
  if (k < 1 && len(sub(P2, q)) < 0.01) return;
  if (len(sub(P2, MC)) > 40 * R) return;
  cam.P = P2; keepAbove(); req(); hoverAt(e.offsetX, e.offsetY);
}, { passive: false });
canvas.addEventListener("dblclick", (e) => {
  const q = pick(e.offsetX, e.offsetY); if (!q) return;
  const F = norm(sub(q, cam.P)), d = Math.max(len(sub(q, cam.P)) * 0.3, 0.3);
  const P = sub(q, mul(F, d)), up = radial(q);
  let U = sub(up, mul(F, dot(up, F))); U = len(U) > 1e-6 ? norm(U) : cam.U;
  pivot = q; flyTo({ P, F, U }, 800);
});
window.addEventListener("keydown", (e) => {
  if (e.target && e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (k === "h" || k === "?") { $("help").classList.toggle("show"); return; }
  if (k === "escape") { $("help").classList.remove("show"); return; }
  if ("1234".includes(k) && k.length === 1) { flyTo(viewPose(["earth", "tilt", "top", "globe"][+k - 1])); return; }
  if (["w", "a", "s", "d", "q", "e", "arrowup", "arrowdown", "arrowleft", "arrowright", "pageup", "pagedown"].includes(k)) {
    keys.add(k); e.preventDefault(); anim = null; last = 0; req();
  }
  if (e.key === "Shift") keys.add("shift");
});
window.addEventListener("keyup", (e) => { keys.delete(e.key.toLowerCase()); if (e.key === "Shift") keys.delete("shift"); });
window.addEventListener("blur", () => keys.clear());
function flyKeys(dt) {
  const up = radial(cam.P);
  const v = Math.max(altitude(), 0.3) * 0.9 * (keys.has("shift") ? 4 : 1) * dt;
  let f = sub(cam.F, mul(up, dot(cam.F, up)));
  f = len(f) > 1e-6 ? norm(f) : norm(sub(cam.U, mul(up, dot(cam.U, up))));
  const r = norm(cross(f, up));
  let m = [0, 0, 0];
  if (keys.has("w") || keys.has("arrowup")) m = add(m, f);
  if (keys.has("s") || keys.has("arrowdown")) m = sub(m, f);
  if (keys.has("d") || keys.has("arrowright")) m = add(m, r);
  if (keys.has("a") || keys.has("arrowleft")) m = sub(m, r);
  if (len(m) > 0) {
    const dir = norm(m), dc = len(sub(cam.P, MC));
    turn(MC, norm(cross(up, dir)), v / dc);
  }
  if (keys.has("e") || keys.has("pageup")) cam.P = add(cam.P, mul(up, v));
  if (keys.has("q") || keys.has("pagedown")) cam.P = sub(cam.P, mul(up, v));
  keepAbove();
}
function tick(now) {
  raf = 0;
  const dt = last ? Math.min((now - last) / 1000, 0.1) : 0.016; last = now;
  let more = false;
  if (anim) { stepAnim(now); more = !!anim; }
  if (keys.size && [...keys].some((k) => k !== "shift")) { flyKeys(dt); more = true; } else last = 0;
  render();
  if (more) req();
}

// ---- HUD ---------------------------------------------------------------------------
const fmtLL = (la, lo) => `${Math.abs(la).toFixed(2)}°${la >= 0 ? "N" : "S"}  ${Math.abs(lo).toFixed(2)}°${lo >= 0 ? "E" : "W"}`;
const fmtKm = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)) + " km";
let hoverT = 0;
function hoverAt(x, y) {
  const now = performance.now(); if (now - hoverT < 60) return; hoverT = now;
  const q = pick(x, y);
  if (!q) { $("hudc").innerHTML = "&nbsp;"; return; }
  const [la, lo] = latlon(q), h = (len(sub(q, MC)) - R + (q.globe ? inset() : 0)) / S.k;
  $("hudc").textContent = `Cursor  ${fmtLL(la, lo)}   height ${h >= 0 ? "+" : ""}${h.toFixed(2)} km` +
      (q.globe ? `   (rest of the Moon, coarse ${G.ppd} px/deg terrain)` : "");
}
function hudAlt() {
  const [la, lo] = latlon(cam.P);
  $("huda").textContent = `Camera  ${fmtKm(altitude())} above the ground over ${fmtLL(la, lo)}   (heights ×${S.k})`;
}

// ---- panel ----------------------------------------------------------------------------
function wire(id, out, fmt, set) {
  const el = $(id), o = $(out);
  const f = () => { set(parseFloat(el.value)); o.textContent = fmt(parseFloat(el.value)); req(); };
  el.addEventListener("input", f); f();
}
let normT = 0;
function refreshNormals() {
  gridNormals(tP, tH, TGH, TGW, tOK, S.k, 0, false, tN);
  for (let s = 0; s < skOrig.length; s++) {
    const v = skOrig[s], w = TNV + s;
    tN[3 * w] = tN[3 * v]; tN[3 * w + 1] = tN[3 * v + 1]; tN[3 * w + 2] = tN[3 * v + 2];
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, tile.bN); gl.bufferData(gl.ARRAY_BUFFER, tN, gl.STATIC_DRAW);
  if (globe && !globe.lit) {
    // the rest of the Moon is lit by the real Sun on its TRUE slopes, like the
    // photograph (exaggerated slopes would light peaks far past the terminator)
    gridNormals(gP, gH, GR, GC, null, 1, 0, true, gN);
    gl.bindBuffer(gl.ARRAY_BUFFER, globe.bN); gl.bufferData(gl.ARRAY_BUFFER, gN, gl.STATIC_DRAW);
    globe.lit = true;
  }
  req();
}
function setupPanel() {
  $("k").value = S.k;
  wire("k", "ok", (v) => "×" + v, (v) => {
    S.k = v; clearTimeout(normT); normT = setTimeout(() => { refreshNormals(); keepAbove(); }, 120); });
  wire("b", "ob", (v) => "×" + Math.pow(2, v).toFixed(2), (v) => { S.gain = Math.pow(2, v); });
  wire("g", "og", (v) => v.toFixed(2), (v) => { S.gamma = v; });
  wire("f", "of", (v) => Math.round(v * 250) + "%", (v) => { S.fill = v; });
  if (G) {
    wire("a", "oa", (v) => Math.round(v * 100) + "%", (v) => { S.back = v; });
    $("gl").addEventListener("change", () => { S.globe = $("gl").checked; keepAbove(); req(); });
  } else $("globeRows").style.display = "none";
  document.querySelectorAll("[data-v]").forEach((b) => b.addEventListener("click", () => flyTo(viewPose(b.dataset.v))));
  $("helpb").addEventListener("click", () => $("help").classList.toggle("show"));
  $("help").addEventListener("click", () => $("help").classList.remove("show"));
  $("min").addEventListener("click", () => { $("panel").classList.toggle("min"); $("min").innerHTML = $("panel").classList.contains("min") ? "+" : "&ndash;"; });
  $("shot").addEventListener("click", () => {
    render();
    canvas.toBlob((b) => { const a = document.createElement("a"); a.href = URL.createObjectURL(b);
      a.download = (META.title || "lunar_relief").replace(/[^\w.-]+/g, "_") + "_3d.png"; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000); }, "image/png");
  });
  $("sub").textContent = META.sub || "";
  $("notes").textContent = (META.notes || []).join("  ");
}
function whenLoaded(img) { return img.complete && img.naturalWidth ? Promise.resolve() : new Promise((ok, bad) => { img.onload = ok; img.onerror = bad; }); }

// ---- start ----------------------------------------------------------------------------
setTimeout(async () => {
  try {
    progMain = program(FS); progPick = program(FSP);
    buildTile();
    if (G) buildGlobe();
    await whenLoaded($("tex")); texTile = texture($("tex"));
    if (G) { await whenLoaded($("alb")); texAlb = texture($("alb")); }
    refreshNormals(); setupPanel();
    new ResizeObserver(() => req()).observe(canvas);
    setPose(viewPose("earth")); keepAbove(); render(); msg("");
    setTimeout(() => flyTo(viewPose("tilt"), 1600), 500);
  } catch (err) {
    msg("Could not start the 3D view: " + (err && err.message ? err.message : err));
    console.error(err);
  }
}, 30);
})();
</script>
</body></html>
"""
