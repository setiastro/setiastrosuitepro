#!/usr/bin/env python3
# ======================================================
#  quad_core.py — geometric core for a native blind plate solver
#
#  Reference implementation of the pieces that are easy to get subtly
#  wrong and expensive to debug later:
#
#    1. The 4-star geometric hash ("quad code") in the AB frame, after
#       Lang, Hogg, Mierle, Blanton & Roweis (2010), AJ 139, 1782.
#    2. Canonicalisation under the A<->B and C<->D relabelings.
#    3. Mirror (parity) handling — the code is invariant to translation,
#       rotation and scale but NOT to reflection, so the solver queries
#       both the code and its mirror, and the hit tells you the parity.
#    4. A memmap-friendly uniform-bin hash over code space, with the
#       2-bins-per-axis neighbourhood proof baked into the query.
#    5. Bayesian verification by log-odds, after Lang et al. §3 — the
#       thing that makes a whole-sky search trustworthy, where an RMS
#       gate is not.
#
#  Everything here is pure numpy + scipy.spatial. No healpy, no SQLite,
#  no C extension — so it survives a frozen build.
#
#  Run this file directly for an end-to-end self test: a synthetic sky,
#  a random sub-field with unknown rotation / scale / parity / noise /
#  distractors, solved blind.
# ======================================================

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Quad codes
# ─────────────────────────────────────────────────────────────────────────────

# The 6 pairs of a 4-set, and for each pair the complementary two indices.
_PAIRS = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64)
_OTHER = np.array([(2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)], dtype=np.int64)

# In the AB frame, A sits at (0,0) and B at (1,1). C and D are required to lie
# inside the circle having AB as its diameter: centre (0.5,0.5), radius 1/sqrt2.
# So code coordinates live in [0.5 - 1/sqrt2, 0.5 + 1/sqrt2] ~ [-0.207, 1.207].
CIRCLE_C = 0.5
CIRCLE_R2 = 0.5
CODE_LO, CODE_HI = -0.25, 1.25          # binning range, with a little slack


def _canon(code, idx=None):
    """
    Canonicalise a code under the 4 relabelings of (A,B) x (C,D).

    Conventions, which the builder and the solver MUST share:
        (1) x_C <= x_D                 -- fixes the C/D label
        (2) x_C + x_D <= 1             -- fixes the A/B label

    Swapping A and B maps (x,y) -> (1-x, 1-y) for both C and D.
    """
    code = np.array(code, dtype=np.float64, copy=True)
    idx = None if idx is None else np.array(idx, dtype=np.int64, copy=True)

    s = code[:, 0] > code[:, 2]
    if s.any():
        code[s] = code[s][:, [2, 3, 0, 1]]
        if idx is not None:
            idx[s] = idx[s][:, [0, 1, 3, 2]]

    f = (code[:, 0] + code[:, 2]) > 1.0
    if f.any():
        c = 1.0 - code[f]
        # after the A/B swap x_C >= x_D, so undo (1) by re-swapping C,D
        code[f] = c[:, [2, 3, 0, 1]]
        if idx is not None:
            idx[f] = idx[f][:, [1, 0, 3, 2]]

    return code if idx is None else (code, idx)


def quad_codes(xy, quads):
    """
    Codes for a set of 4-star quads.

    xy    : (N,2) positions — pixels for the image side, tangent-plane
            degrees for the catalogue side. Any similarity transform
            between the two leaves the code unchanged.
    quads : (M,4) indices into xy.

    Returns (code, idx, ok, ab_len):
        code   (M,4) float64, canonical (xC,yC,xD,yD)
        idx    (M,4) the quad's stars reordered to canonical (A,B,C,D)
        ok     (M,)  bool — C and D actually inside the AB circle
        ab_len (M,)  |AB| in input units; the quad's scale
    """
    q = np.asarray(quads, dtype=np.int64)
    if q.size == 0:
        return (np.zeros((0, 4)), np.zeros((0, 4), np.int64),
                np.zeros(0, bool), np.zeros(0))

    P = np.asarray(xy, dtype=np.float64)[q]                       # (M,4,2)
    d2 = np.stack([((P[:, i] - P[:, j]) ** 2).sum(1) for i, j in _PAIRS], axis=1)
    k = np.argmax(d2, axis=1)                                     # widest pair -> A,B
    order = np.column_stack([_PAIRS[k, 0], _PAIRS[k, 1], _OTHER[k, 0], _OTHER[k, 1]])

    r = np.arange(len(q))[:, None]
    Q = P[r, order]                                               # (M,4,2) = A,B,C,D
    idx = q[r, order]

    A, B = Q[:, 0], Q[:, 1]
    v = B - A
    L2 = (v * v).sum(1)
    L2 = np.where(L2 > 0, L2, 1.0)

    # M = [[a,-b],[b,a]] is the scaled rotation with M(B-A) = (1,1).
    a = (v[:, 0] + v[:, 1]) / L2
    b = (v[:, 0] - v[:, 1]) / L2

    dC, dD = Q[:, 2] - A, Q[:, 3] - A
    xC = a * dC[:, 0] - b * dC[:, 1]
    yC = b * dC[:, 0] + a * dC[:, 1]
    xD = a * dD[:, 0] - b * dD[:, 1]
    yD = b * dD[:, 0] + a * dD[:, 1]

    ok = (((xC - CIRCLE_C) ** 2 + (yC - CIRCLE_C) ** 2) <= CIRCLE_R2) & \
         (((xD - CIRCLE_C) ** 2 + (yD - CIRCLE_C) ** 2) <= CIRCLE_R2)

    code, idx = _canon(np.column_stack([xC, yC, xD, yD]), idx)
    return code, idx, ok, np.sqrt((v * v).sum(1))


def mirror_codes(code, idx=None):
    """
    The code the same four stars would produce if the field were mirrored.

    Reflection about the AB line maps (x,y) -> (y,x) in the AB frame, so the
    mirrored code is just the axis swap, re-canonicalised. A hit on the direct
    code means the image->sky map preserves orientation; a hit on the mirror
    means it reverses it. That is where parity comes from, for free.
    """
    m = np.asarray(code, dtype=np.float64)[:, [1, 0, 3, 2]]
    return _canon(m, idx)


# ─────────────────────────────────────────────────────────────────────────────
#  Code-space hash
# ─────────────────────────────────────────────────────────────────────────────
#
#  Uniform bins of width 2*tol per axis, key = mixed-radix pack of the 4 bin
#  indices, quads sorted by key, lookup by np.searchsorted on a memmapped
#  array. No tree to build, no load time, vectorises over every query quad at
#  once.
#
#  Why 2 bins per axis is sufficient: with bin width w = 2*tol, a query at
#  offset o in [0,w) has its tol-neighbourhood [p-tol, p+tol] contained in
#  bin k and bin k-1 when o < tol, or bin k and k+1 when o >= tol. So 2^4 = 16
#  candidate cells cover the whole tolerance box exactly.


def nbins_for_tol(tol):
    return max(4, int(np.ceil((CODE_HI - CODE_LO) / (2.0 * tol))))


def pack_keys(code, nbins):
    t = (np.asarray(code) - CODE_LO) / (CODE_HI - CODE_LO) * nbins
    b = np.clip(np.floor(t).astype(np.int64), 0, nbins - 1)
    k = b[:, 0]
    for d in range(1, 4):
        k = k * nbins + b[:, d]
    return k


def probe_keys(code, nbins):
    """(M,16) candidate keys covering the tolerance box of each query code."""
    t = (np.asarray(code) - CODE_LO) / (CODE_HI - CODE_LO) * nbins
    b0 = np.floor(t).astype(np.int64)
    frac = t - b0
    b1 = np.where(frac < 0.5, b0 - 1, b0 + 1)
    b0 = np.clip(b0, 0, nbins - 1)
    b1 = np.clip(b1, 0, nbins - 1)

    out = np.empty((len(t), 16), dtype=np.int64)
    for m in range(16):
        k = (b1 if (m >> 0) & 1 else b0)[:, 0]
        for d in range(1, 4):
            sel = (b1 if (m >> d) & 1 else b0)[:, d]
            k = k * nbins + sel
        out[:, m] = k
    return out


class CodeHash:
    """Sorted key array + the codes themselves. Stands in for the mmapped file."""

    def __init__(self, code, tol):
        self.tol = float(tol)
        self.nbins = nbins_for_tol(tol)
        keys = pack_keys(code, self.nbins)
        o = np.argsort(keys, kind="stable")
        self.order = o
        self.keys = keys[o]
        self.code = np.asarray(code, dtype=np.float64)[o]

    def query(self, qcode):
        """
        For each query code, candidate quad ids within tol (Chebyshev in code
        space). Returns (qi, ci) index pairs into (qcode, self.code).
        """
        if len(qcode) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        pk = probe_keys(qcode, self.nbins)                      # (M,16)
        flat = pk.ravel()
        lo = np.searchsorted(self.keys, flat, "left")
        hi = np.searchsorted(self.keys, flat, "right")
        n = hi - lo
        tot = int(n.sum())
        if tot == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)

        qi = np.repeat(np.repeat(np.arange(len(qcode)), 16), n)
        # expand [lo,hi) ranges without a python loop
        starts = np.repeat(lo, n)
        offs = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
        ci = starts + offs

        # de-duplicate (a cell can be probed twice when clipped at the edge)
        pair = qi * (len(self.keys) + 1) + ci
        _, u = np.unique(pair, return_index=True)
        qi, ci = qi[u], ci[u]

        d = np.abs(self.code[ci] - np.asarray(qcode)[qi]).max(axis=1)
        keep = d <= self.tol
        return qi[keep], ci[keep]


# ─────────────────────────────────────────────────────────────────────────────
#  Quad generation (shared by builder and solver — they must agree)
# ─────────────────────────────────────────────────────────────────────────────

def make_quads(xy, order, dmin, dmax, max_b_per_a=4):
    """
    For each star A (brightest first), each B with dmin <= |AB| <= dmax, take
    the two brightest stars inside the circle with AB as diameter.

    `order` is an index array into xy sorted brightest-first; a star's rank in
    `order` IS its brightness rank, which is what makes the builder and the
    solver pick the same C and D when they see the same stars.
    """
    from scipy.spatial import cKDTree

    order = np.asarray(order, dtype=np.int64)
    pos = np.asarray(xy, dtype=np.float64)[order]
    if len(pos) < 4:
        return np.zeros((0, 4), np.int64)

    tree = cKDTree(pos)
    out = []
    for ai in range(len(pos)):
        A = pos[ai]
        cand = np.array(sorted(tree.query_ball_point(A, dmax)), dtype=np.int64)
        cand = cand[cand > ai]                      # bi > ai kills duplicates
        if cand.size == 0:
            continue
        dd = np.hypot(*(pos[cand] - A).T)
        cand = cand[dd >= dmin]
        nb = 0
        for bi in cand:                             # brightest B first
            ctr = (A + pos[bi]) * 0.5
            rad = float(np.hypot(*(pos[bi] - A))) * 0.5
            ins = sorted(tree.query_ball_point(ctr, rad))
            ins = [c for c in ins if c != ai and c != bi]
            if len(ins) < 2:
                continue
            out.append((order[ai], order[bi], order[ins[0]], order[ins[1]]))
            nb += 1
            if nb >= max_b_per_a:
                break
    return (np.array(out, dtype=np.int64) if out
            else np.zeros((0, 4), np.int64))


# ─────────────────────────────────────────────────────────────────────────────
#  Verification
# ─────────────────────────────────────────────────────────────────────────────

def log_odds(field_xy, cat_xy, sigma, field_area, eps=0.25):
    """
    Log-odds that a candidate alignment is real, vs the null "these detections
    are uniform noise". After Lang et al. (2010) §3.

    Per field star, odds contribution:
        (1-eps) * (A / (N_cat * 2*pi*sigma^2)) * exp(-d^2 / 2sigma^2)  +  eps

    The leading factor is "area per catalogue star divided by the area of a
    sigma disc" — how surprising it is to land that close by chance. eps is the
    prior that a detection is a distractor with no catalogue counterpart.

    Accept at log_odds > ln(1e9) ~ 20.7. That threshold is what lets you search
    10^8 quads whole-sky without ever returning a confident wrong answer; an
    RMS gate cannot do that, because six cherry-picked pairs always fit.
    """
    from scipy.spatial import cKDTree

    field_xy = np.asarray(field_xy, dtype=np.float64)
    cat_xy = np.asarray(cat_xy, dtype=np.float64)
    if len(cat_xy) == 0 or len(field_xy) == 0:
        return -np.inf, 0

    d, _ = cKDTree(cat_xy).query(field_xy, k=1, workers=-1)
    gain = (1.0 - eps) * field_area / (len(cat_xy) * 2.0 * np.pi * sigma ** 2)
    terms = np.log(gain * np.exp(-0.5 * (d / sigma) ** 2) + eps)
    return float(terms.sum()), int((d < 3.0 * sigma).sum())


# ─────────────────────────────────────────────────────────────────────────────
#  Similarity transform from a matched quad
# ─────────────────────────────────────────────────────────────────────────────

def similarity_from_pair(img_A, img_B, cat_A, cat_B, flipped):
    """
    The unique similarity taking image coords -> catalogue coords given the
    A and B correspondence and the parity. Complex arithmetic keeps it to
    two lines and no sign-convention bugs.
    """
    zi_a = complex(*img_A); zi_b = complex(*img_B)
    zc_a = complex(*cat_A); zc_b = complex(*cat_B)
    if flipped:
        m = (zc_b - zc_a) / np.conj(zi_b - zi_a)
        t = zc_a - m * np.conj(zi_a)
    else:
        m = (zc_b - zc_a) / (zi_b - zi_a)
        t = zc_a - m * zi_a
    return m, t


def project_catalog(cat_xy, m, t, flipped):
    """Catalogue coords -> image pixels under the inverse of the similarity."""
    z = cat_xy[:, 0] + 1j * cat_xy[:, 1]
    w = (z - t) / m
    if flipped:
        w = np.conj(w)
    return np.column_stack([w.real, w.imag])


# ─────────────────────────────────────────────────────────────────────────────
#  Self test
# ─────────────────────────────────────────────────────────────────────────────

def make_sky(rng, sky_deg=12.0, ncat=6000):
    """A flat patch of 'sky'. Magnitudes only matter for their ordering."""
    return (rng.uniform(0.0, sky_deg, size=(ncat, 2)),
            rng.uniform(7.0, 15.0, ncat))


def build_index(cat, mag, cell=0.45, qlo=0.30, qhi=0.60, per_cell=4, tol=0.008):
    """
    Stand-in for the offline builder: keep the brightest `per_cell` stars in
    each cell (this is what caps index density and keeps quads uniform on the
    sky), make quads at one scale, hash the codes.
    """
    ckey = (cat // cell).astype(np.int64)
    ckey = ckey[:, 0] * 100000 + ckey[:, 1]
    keep = []
    for c in np.unique(ckey):
        w = np.where(ckey == c)[0]
        keep.extend(w[np.argsort(mag[w])][:per_cell])
    keep = np.array(sorted(keep))

    quads = make_quads(cat, keep[np.argsort(mag[keep])], qlo, qhi, max_b_per_a=4)
    code, idx, ok, _ = quad_codes(cat, quads)
    code, idx = code[ok], idx[ok]

    h = CodeHash(code, tol)
    return h, idx[h.order], len(keep)


def make_image(rng, cat, mag, W=1600, H=1100, fov_w=1.6, flip=None,
               sigma=0.6, keep_frac=0.85, junk_frac=0.25):
    """Render a random sub-field with unknown rotation, parity, noise, junk."""
    px_per_deg = W / fov_w
    cx, cy = rng.uniform(2.0, cat.max() - 2.0, 2)
    theta = rng.uniform(0, 2 * np.pi)
    if flip is None:
        flip = bool(rng.integers(0, 2))

    z = px_per_deg * np.exp(1j * theta) * ((cat[:, 0] - cx) + 1j * (cat[:, 1] - cy))
    if flip:
        z = np.conj(z)
    px = np.column_stack([z.real + W / 2, z.imag + H / 2])

    sel = np.where((px[:, 0] >= 0) & (px[:, 0] < W) &
                   (px[:, 1] >= 0) & (px[:, 1] < H))[0]
    sel = sel[rng.random(len(sel)) < keep_frac]
    xy = px[sel] + rng.normal(0, sigma, (len(sel), 2))
    mg = mag[sel]

    njunk = int(junk_frac * max(len(sel), 1))       # spurious detections, faint
    xy = np.vstack([xy, np.column_stack([rng.uniform(0, W, njunk),
                                         rng.uniform(0, H, njunk)])])
    mg = np.concatenate([mg, rng.uniform(14.5, 16.0, njunk)])

    o = np.argsort(mg)
    truth = dict(cx=cx, cy=cy, theta=theta, flip=flip, W=W, H=H,
                 sigma=sigma, njunk=njunk,
                 scale_arcsec=3600.0 / px_per_deg)
    return xy[o], truth


def solve_blind(det_xy, index, iidx, cat, truth, ndet=60, sigma=0.6,
                thresh=np.log(1e9), collect=False, span=(0.12, 0.55)):
    """
    The whole runtime loop, minus the sphere: field quads -> both parities ->
    code lookup -> similarity from the A/B pair -> log-odds verification.

    `span` is the |AB| window for field quads, as a fraction of the image
    diagonal. With the pixel scale known this is set so that |AB| lands in the
    layer's angular band; blind, it must be swept, because a quad window that
    is too narrow silently throws away every quad the index could have matched.
    """
    W, H = truth["W"], truth["H"]
    diag = float(np.hypot(W, H))
    n = min(ndet, len(det_xy))

    fq = make_quads(det_xy[:n], np.arange(n), span[0] * diag, span[1] * diag,
                    max_b_per_a=4)
    fcode, fidx, fok, _ = quad_codes(det_xy[:n], fq)
    fcode, fidx = fcode[fok], fidx[fok]
    mcode, midx = mirror_codes(fcode, fidx)

    best, hits, verified, odds = None, 0, 0, []
    for flipped, qc, qidx in ((False, fcode, fidx), (True, mcode, midx)):
        qi, ci = index.query(qc)
        hits += len(qi)
        for a, b in zip(qi, ci):
            m, t = similarity_from_pair(det_xy[qidx[a, 0]], det_xy[qidx[a, 1]],
                                        cat[iidx[b, 0]], cat[iidx[b, 1]], flipped)
            if not np.isfinite(abs(m)) or abs(m) == 0:
                continue
            proj = project_catalog(cat, m, t, flipped)
            inf = ((proj[:, 0] > -20) & (proj[:, 0] < W + 20) &
                   (proj[:, 1] > -20) & (proj[:, 1] < H + 20))
            if inf.sum() < 10:
                continue
            verified += 1
            lo, nm = log_odds(det_xy, proj[inf], sigma * 1.6, W * H)
            if collect:
                odds.append((lo, m, t, flipped))
            if lo > thresh and (best is None or lo > best[0]):
                best = (lo, nm, m, t, flipped, int(inf.sum()))

    return dict(best=best, n_field_quads=len(fcode), code_hits=hits,
                verified=verified, odds=odds)


def _report(res, truth, det_xy, verbose=True):
    if res["best"] is None:
        return False, None
    lo, nm, m, t, flipped, ncat = res["best"]
    scale = abs(m) * 3600.0                          # |m| is deg per pixel
    ctr = complex(truth["W"] / 2, truth["H"] / 2)
    sky = m * (np.conj(ctr) if flipped else ctr) + t

    d_centre = np.hypot(sky.real - truth["cx"], sky.imag - truth["cy"]) * 3600
    ok = (flipped == truth["flip"]
          and abs(scale - truth["scale_arcsec"]) / truth["scale_arcsec"] < 0.01
          and d_centre < 30)
    if verbose:
        print(f"  solved   log-odds {lo:>9,.1f} | {nm}/{len(det_xy)} stars matched "
              f"| parity {'flip' if flipped else 'norm'} "
              f"(truth {'flip' if truth['flip'] else 'norm'}) "
              f"| scale {scale:.4f}\" (truth {truth['scale_arcsec']:.4f}) "
              f"| centre off {d_centre:.2f}\"  -> {'PASS' if ok else 'FAIL'}")
    return ok, lo


def _selftest(ntrials=12):
    import time

    print("═" * 78)
    print(" A. recovery — alternating parity, random centre / rotation / noise")
    print("═" * 78)

    passes, worst_false = 0, -np.inf
    t_lookup = []
    for s in range(ntrials):
        rng = np.random.default_rng(s)
        cat, mag = make_sky(rng)
        index, iidx, nkeep = build_index(cat, mag)
        det_xy, truth = make_image(rng, cat, mag, flip=bool(s % 2))

        t0 = time.perf_counter()
        res = solve_blind(det_xy, index, iidx, cat, truth, collect=True)
        t_lookup.append(time.perf_counter() - t0)

        if s == 0:
            print(f"  index: {nkeep:,} stars, {len(index.code):,} quads, "
                  f"{index.nbins}^4 bins, tol={index.tol}")
            print(f"  query: {res['n_field_quads']} field quads x 2 parities\n")

        ok, _ = _report(res, truth, det_xy)
        passes += ok

        # how close did any WRONG candidate get to the accept threshold?
        if res["best"] is not None:
            good_m = res["best"][2]
            for lo, m, t, f in res["odds"]:
                if abs(abs(m) - abs(good_m)) / abs(good_m) > 0.02:
                    worst_false = max(worst_false, lo)

    print(f"\n  {passes}/{ntrials} solved   "
          f"median solve {1000*np.median(t_lookup):.0f} ms")
    print(f"  highest log-odds reached by any WRONG candidate: {worst_false:.1f} "
          f"(accept threshold {np.log(1e9):.1f})")

    print()
    print("═" * 78)
    print(" B. true negative — image of a sky the index has never seen")
    print("═" * 78)
    rng = np.random.default_rng(999)
    cat_a, mag_a = make_sky(rng)
    index, iidx, _ = build_index(cat_a, mag_a)
    cat_b, mag_b = make_sky(np.random.default_rng(1234))
    det_xy, truth = make_image(np.random.default_rng(4321), cat_b, mag_b)
    res = solve_blind(det_xy, index, iidx, cat_a, truth, collect=True)
    top = max((o[0] for o in res["odds"]), default=-np.inf)
    print(f"  {res['code_hits']} code hits, {res['verified']} verified, "
          f"best log-odds {top:.1f}")
    neg_ok = res["best"] is None
    print(f"  {'PASS' if neg_ok else 'FAIL'}: "
          f"{'correctly refused to solve' if neg_ok else 'RETURNED A FALSE SOLUTION'}")

    print()
    print("═" * 78)
    print(" C. degradation — how much the field can be ruined before it fails")
    print("═" * 78)
    for label, kw in (("clean                 ", {}),
                      ("40% of stars missing  ", dict(keep_frac=0.60)),
                      ("100% spurious padding ", dict(junk_frac=1.00)),
                      ("2.0 px centroid noise ", dict(sigma=2.0)),
                      ("60% missing + 100% junk", dict(keep_frac=0.60,
                                                       junk_frac=1.00))):
        n_ok = 0
        for s in range(6):
            rng = np.random.default_rng(100 + s)
            cat, mag = make_sky(rng)
            index, iidx, _ = build_index(cat, mag)
            det_xy, truth = make_image(rng, cat, mag, flip=bool(s % 2), **kw)
            res = solve_blind(det_xy, index, iidx, cat, truth,
                              sigma=kw.get("sigma", 0.6))
            ok, _ = _report(res, truth, det_xy, verbose=False)
            n_ok += ok
        print(f"  {label}  {n_ok}/6")

    return passes == ntrials and neg_ok


if __name__ == "__main__":
    import sys
    ok = _selftest(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOMETHING FAILED'}")
    sys.exit(0 if ok else 1)