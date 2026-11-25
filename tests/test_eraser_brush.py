# tests/test_eraser_brush.py
"""Eraser brush (Clone Stamp + Blemish Blaster): fill dabs, the preview value of a
fill, channel medians, stroke undo tiles, and both dialogs driven offscreen."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _sky(h=160, w=220, seed=1, mono=False):
    rng = np.random.default_rng(seed)
    img = np.array([0.020, 0.030, 0.025], np.float32) + rng.normal(0, 0.004, (h, w, 3)).astype(np.float32)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(25):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        img += (0.8 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0))[..., None].astype(np.float32)
    img = np.clip(img, 0, 1).astype(np.float32)
    return img[..., 1].copy() if mono else img


# ── pure functions ─────────────────────────────────────────────────────────

def test_fill_dab_core_exact_edges_blend_once_and_border_clip():
    from setiastro.saspro.eraser_brush import fill_dab_inplace
    from setiastro.saspro.clone_stamp import _circle_mask
    img = _sky()
    ref = img.copy()
    mask = _circle_mask(10, 0.5)
    col = np.array([0.3, 0.6, 0.9], np.float32)
    fill_dab_inplace(img, 50, 60, col, mask, 10, 1.0)
    assert np.array_equal(img[60, 50], col)                     # a = 1 -> exactly the fill
    a = mask * 1.0
    want = (1 - a[..., None]) * ref[50:71, 40:61] + a[..., None] * col
    np.testing.assert_allclose(img[50:71, 40:61], want, rtol=0, atol=1e-7)
    out = np.ones(img.shape[:2], bool); out[50:71, 40:61] = False
    assert np.array_equal(img[out], ref[out])
    img2 = ref.copy()
    fill_dab_inplace(img2, 2, 3, np.zeros(3, np.float32), _circle_mask(8, 0.0), 8, 0.5)
    np.testing.assert_allclose(img2[3, 2], 0.5 * ref[3, 2], rtol=0, atol=1e-7)   # half, once
    fill_dab_inplace(img2, -50, -50, col, mask, 10, 1.0)        # fully off-image: no-op
    fill_dab_inplace(img2, 5, 5, col, mask, 10, 0.0)            # opacity 0: no-op


@pytest.mark.parametrize("linked", [True, False])
def test_display_fill_matches_the_stretch(linked):
    from setiastro.saspro.eraser_brush import display_fill
    from setiastro.saspro.imageops.stretch import stretch_color_image
    img = _sky(seed=4)
    disp = stretch_color_image(img, target_median=0.25, linked=linked, normalize=False, apply_curves=False)
    assert np.array_equal(display_fill(img, disp, np.zeros(3, np.float32)), np.zeros(3, np.float32))
    for col in ([0.02, 0.03, 0.025], [0.05, 0.01, 0.2], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]):
        col = np.array(col, np.float32)
        # f of the fill: stretch an image with the fill in one pixel (the stats barely move)
        probe = img.copy(); probe[0, 0] = col
        f = stretch_color_image(probe, target_median=0.25, linked=linked, normalize=False, apply_curves=False)[0, 0]
        np.testing.assert_allclose(display_fill(img, disp, col, linked=linked), f, atol=0.02)


def test_display_fill_is_np_interp_on_the_samples():
    from setiastro.saspro.eraser_brush import display_fill
    rng = np.random.default_rng(7)
    lin = rng.random((90, 110, 3), dtype=np.float32) * 0.6 + 0.1
    disp = np.sqrt(lin)                                          # any non-decreasing map
    for v in (0.0, 0.05, 0.1234, 0.33, 0.69, 0.95, 1.0):
        col = np.full(3, v, np.float32)
        got = display_fill(lin, disp, col)
        for c in range(3):
            x = lin[..., c].ravel(); k = np.argsort(x, kind="stable")
            xs = np.concatenate(([0.0], x[k], [1.0])); ys = np.concatenate(([0.0], disp[..., c].ravel()[k], [1.0]))
            assert got[c] == pytest.approx(np.interp(v, xs, ys), abs=1e-6)
    assert display_fill(lin, lin, np.full(3, 0.4, np.float32)).tolist() == pytest.approx([0.4] * 3)


def test_channel_medians():
    from setiastro.saspro.eraser_brush import channel_medians
    img = _sky(seed=5)
    np.testing.assert_array_equal(channel_medians(img, False),
                                  np.array([np.median(img[..., c]) for c in range(3)], np.float32))
    m = _sky(seed=5, mono=True)
    img3 = np.repeat(m[..., None], 3, axis=2)
    assert channel_medians(img3, True).tolist() == [np.float32(np.median(m))] * 3
    img[3, 3, 1] = np.nan
    med = channel_medians(img, False)
    assert np.all(np.isfinite(med))


def test_stroke_tiles_restore_the_stroke():
    from setiastro.saspro.eraser_brush import StrokeTiles, fill_dab_inplace
    from setiastro.saspro.clone_stamp import _circle_mask
    img = _sky(h=300, w=260)
    ref = img.copy()
    t = StrokeTiles(img)
    mask = _circle_mask(30, 0.3)
    for x in range(20, 250, 7):
        t.touch(x, 150 + (x % 40), 30)
        fill_dab_inplace(img, x, 150 + (x % 40), np.zeros(3, np.float32), mask, 30, 1.0)
    assert len(t.patches) == len(t.seen)
    for y0, x0, pix in t.patches:
        img[y0:y0 + pix.shape[0], x0:x0 + pix.shape[1]] = pix
    assert np.array_equal(img, ref)


# ── the dialogs ────────────────────────────────────────────────────────────

@pytest.fixture
def qapp(tmp_path):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    old = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)       # keep the eraser keys out of the registry
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    opened = []
    yield app, opened
    for d in opened:            # never close(): close releases buffers; just drop them
        d.hide()
        d.deleteLater()
    app.processEvents()
    QSettings.setDefaultFormat(old)


class _Doc:
    def __init__(self, img):
        self.image = img
        self.set = None

    def set_image(self, img, step_name=""):
        self.set = (img, step_name)


def _open(qa, dlg):
    app, opened = qa
    opened.append(dlg)
    dlg.show()
    app.processEvents(); app.processEvents()      # let the first-fit timers run
    return dlg


def _stroke(dlg, pts):
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    vp = dlg.view.viewport()

    def ev(kind, pt, buttons):
        p = dlg.view.mapFromScene(QPointF(*pt))
        btn = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton
        dlg.eventFilter(vp, QMouseEvent(kind, QPointF(p), QPointF(vp.mapToGlobal(p)), btn, buttons,
                                        Qt.KeyboardModifier.NoModifier))
    L = Qt.MouseButton.LeftButton
    ev(QEvent.Type.MouseButtonPress, pts[0], L)
    for p in pts[1:]:
        ev(QEvent.Type.MouseMove, p, L)
    ev(QEvent.Type.MouseButtonRelease, pts[-1], Qt.MouseButton.NoButton)


def test_clone_stamp_opacity_is_applied_once(qapp):
    from PyQt6.QtCore import QPointF
    from setiastro.saspro.clone_stamp import CloneStampDialogPro
    img = np.zeros((120, 200, 3), np.float32); img[:, 100:] = 1.0
    cs = _open(qapp, CloneStampDialogPro(None, _Doc(img)))
    cs.s_radius.setValue(8); cs.s_feather.setValue(0); cs.s_opacity.setValue(50)
    cs._set_source_at(QPointF(150, 60)); cs._start_paint_at(QPointF(40, 60)); cs._end_paint()
    assert cs._image[60, 40, 0] == pytest.approx(0.5, abs=1e-6)     # was 0.75: every dab blended twice


def test_clone_stamp_eraser(qapp):
    from PyQt6.QtGui import QColor
    from setiastro.saspro.clone_stamp import CloneStampDialogPro
    src = _sky()
    doc = _Doc(src.copy())
    cs = _open(qapp, CloneStampDialogPro(None, doc))
    cs.rb_eraser.setChecked(True)
    assert cs.eraser_fill.isVisible() and not cs.btn_clear_src.isVisible()
    cs.s_radius.setValue(12); cs.s_feather.setValue(0); cs.s_opacity.setValue(100)
    ref = cs._image.copy()
    _stroke(cs, [(60, 60), (90, 62), (120, 70)])                    # no source needed
    on_path = [(60, 60), (75, 61), (90, 62), (105, 66), (120, 70)]
    assert all(np.all(cs._image[y - 3:y + 4, x - 3:x + 4] == 0.0) for x, y in on_path)
    assert len(cs._undo) == 1
    cs._undo_step()
    assert np.array_equal(cs._image, ref)
    cs._redo_step()
    cs.eraser_fill.rb_median.setChecked(True)
    _stroke(cs, [(170, 120)])
    assert np.array_equal(cs._image[120, 170], cs.eraser_fill._median)
    cs.eraser_fill._color = QColor(255, 0, 0)
    cs.eraser_fill.rb_color.setChecked(True)
    _stroke(cs, [(30, 130)])
    assert cs._image[130, 30].tolist() == [1.0, 0.0, 0.0]
    cs._commit_to_doc()
    out, step = doc.set
    assert step == "Clone Stamp" and all(np.all(out[y - 3:y + 4, x - 3:x + 4] == 0.0) for x, y in on_path)


def test_clone_stamp_eraser_mono_color_is_gray(qapp):
    from PyQt6.QtGui import QColor
    from setiastro.saspro.clone_stamp import CloneStampDialogPro
    doc = _Doc(_sky(mono=True))
    cs = _open(qapp, CloneStampDialogPro(None, doc))
    cs.rb_eraser.setChecked(True)
    cs.eraser_fill._color = QColor(255, 0, 0)
    cs.eraser_fill.rb_color.setChecked(True)
    cs.s_feather.setValue(0)
    _stroke(cs, [(80, 80)])
    cs._commit_to_doc()
    out, _ = doc.set
    assert out.ndim == 2 and out[80, 80] == pytest.approx(1 / 3, abs=1e-6)


def test_blemish_blaster_eraser(qapp):
    from setiastro.saspro.blemish_blaster import BlemishBlasterDialogPro
    doc = _Doc(_sky(seed=2))
    bb = _open(qapp, BlemishBlasterDialogPro(None, doc))
    bb.rb_eraser.setChecked(True)
    assert bb._mode == 3 and bb.eraser_fill.isVisible()
    bb.eraser_fill.rb_black.setChecked(True)
    bb.s_radius.setValue(10); bb.s_feather.setValue(50); bb.s_opacity.setValue(100)
    ref = bb._image.copy()
    _stroke(bb, [(100, 80), (120, 85), (140, 80)])
    assert np.all(bb._image[78:83, 98:103] == 0.0)
    assert np.all(bb._display[80, 100] == 0.0)                     # the stretched preview shows it black
    assert len(bb._undo) == 1 and isinstance(bb._undo[0], list)
    stroked = bb._image.copy()
    # a heal's single-patch entry on top (as _on_worker_done records it)
    healed = bb._image.copy(); healed[10:20, 10:20] = 0.5
    bb._on_worker_done(healed, (10, 10, bb._image[10:20, 10:20].copy()))
    bb._undo_step()
    assert np.array_equal(bb._image, stroked)
    bb._undo_step()
    assert np.array_equal(bb._image, ref)
    bb._redo_step(); bb._redo_step()
    assert np.array_equal(bb._image[10:20, 10:20], healed[10:20, 10:20])
    assert np.all(bb._image[78:83, 98:103] == 0.0)
    bb._commit_to_doc()
    out, step = doc.set
    assert step == "Blemish Blaster" and np.all(out[78:83, 98:103] == 0.0)
