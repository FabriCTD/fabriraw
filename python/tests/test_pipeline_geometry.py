"""Motor Pro — geometry, optics, zoom regions, preview options."""

from __future__ import annotations

import numpy as np
import pytest

from fraw.processor import FRAWProcessor, to_uint8
from test_processor import flat, make_file

LIN = {"gamma_mode": "linear"}


def grid_proc(h: int = 64, w: int = 96, orientation: int = 1) -> FRAWProcessor:
    yy, xx = np.mgrid[0:h, 0:w]
    cam = np.stack([0.1 + 0.8 * xx / w, 0.1 + 0.8 * yy / h, np.full((h, w), 0.3)], axis=-1)
    return FRAWProcessor(make_file(cam, orientation=orientation))


def test_rotate90_and_flips_exact() -> None:
    p = grid_proc()
    base = p.develop(LIN, demosaic="bilinear")
    np.testing.assert_array_equal(p.develop({**LIN, "rotate_90": "90"}, demosaic="bilinear"), np.rot90(base, -1))
    np.testing.assert_array_equal(p.develop({**LIN, "rotate_90": "270"}, demosaic="bilinear"), np.rot90(base, 1))
    np.testing.assert_array_equal(p.develop({**LIN, "rotate_90": "180"}, demosaic="bilinear"), base[::-1, ::-1])
    np.testing.assert_array_equal(p.develop({**LIN, "flip_h": True}, demosaic="bilinear"), base[:, ::-1])
    np.testing.assert_array_equal(p.develop({**LIN, "flip_v": True}, demosaic="bilinear"), base[::-1])
    assert p.output_size({"rotate_90": "90"}) == (64, 96)


def test_crop_exact_size_and_aspect() -> None:
    p = grid_proc()
    prm = {**LIN, "crop_left": 0.25, "crop_right": 0.75, "crop_top": 0.25, "crop_bottom": 0.75}
    out = p.develop(prm, demosaic="bilinear")
    assert out.shape == (32, 48, 3)
    full = p.develop(LIN, demosaic="bilinear")
    np.testing.assert_array_equal(out, full[16:48, 24:72])
    assert p.output_size(prm) == (48, 32)
    sq = p.develop({**prm, "crop_aspect": "1:1"}, demosaic="bilinear")
    assert sq.shape[0] == sq.shape[1] == 32
    uncropped = p.preview(prm, apply_crop=False)
    assert uncropped.shape == p.preview(LIN).shape
    assert p.preview(prm).shape == (16, 24, 3)


def test_straighten_fills_frame_without_black() -> None:
    p = FRAWProcessor(make_file(flat(64, 96, (0.5, 0.5, 0.5))))
    for deg in (-12.0, 5.0, 30.0):
        out = p.develop({**LIN, "rotate_deg": deg}, demosaic="bilinear")
        assert out.shape == (64, 96, 3)
        assert out.min() > 0.45, deg  # no empty corners (auto zoom)
    g = grid_proc()
    a = g.develop({**LIN, "rotate_deg": 10.0}, demosaic="bilinear")
    b = g.develop(LIN, demosaic="bilinear")
    assert np.abs(a - b).mean() > 0.01
    # rotation direction: positive = clockwise → top-left corner moves right/down: brighter G (y) at top-left
    assert a[2, 2, 1] > b[2, 2, 1]


def test_perspective_distortion_ca_vignette() -> None:
    g = grid_proc()
    b = g.develop(LIN, demosaic="bilinear")
    v = g.develop({**LIN, "perspective_vertical": 0.6}, demosaic="bilinear")
    # widened top: horizontal gradient (R) is flatter in the top rows than in the bottom rows
    top, bot = np.diff(v[2, :, 0]).mean(), np.diff(v[-3, :, 0]).mean()
    assert top < bot and v.min() > 0.05
    s = g.develop({**LIN, "perspective_scale": 1.5}, demosaic="bilinear")
    assert np.diff(s[32, :, 0]).mean() < 0.75 * np.diff(b[32, :, 0]).mean()  # zoomed in: less scene per pixel
    d = g.develop({**LIN, "lens_distortion": 1.0}, demosaic="bilinear")
    assert np.abs(d - b)[:8, :8].mean() > 0.02 and d.min() > 0.05
    ca = g.develop({**LIN, "ca_red_cyan": 1.0}, demosaic="bilinear")
    assert np.abs(ca[..., 0] - b[..., 0])[:, :8].mean() > np.abs(ca[..., 0] - b[..., 0])[:, 44:52].mean()
    np.testing.assert_allclose(ca[30:34, 46:50, 1], b[30:34, 46:50, 1], atol=1e-2)
    pf = FRAWProcessor(make_file(flat(64, 96, (0.2, 0.2, 0.2))))
    lv = pf.develop({**LIN, "lens_vignette": 1.0}, demosaic="bilinear")
    assert lv[0, 0, 1] > 1.8 * 0.2 and abs(lv[32, 48, 1] - 0.2) < 0.01  # +1 EV corners


def test_orientation_then_geometry() -> None:
    p = grid_proc(orientation=6)
    assert p.oriented_size() == (64, 96)
    out = p.develop({**LIN, "crop_left": 0.5}, demosaic="bilinear")
    assert out.shape == (96, 32, 3)
    full = p.develop(LIN, demosaic="bilinear")
    np.testing.assert_array_equal(out, full[:, 32:])


@pytest.mark.parametrize("prm", [{}, {"rotate_deg": 4.0, "crop_left": 0.125, "crop_bottom": 0.75},
                                 {"rotate_90": "90", "flip_h": True, "crop_top": 0.25},
                                 {"clarity": 0.5, "shadows": 0.5, "sharpen_amount": 0.8, "vignette_amount": -0.5}])
def test_render_region_matches_develop(prm) -> None:
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:96, 0:128]
    cam = np.stack([0.3 + 0.2 * np.sin(xx / 7.0), 0.3 + 0.2 * np.cos(yy / 5.0), 0.2 + 0.1 * np.sin((xx + yy) / 9.0)], -1)
    p = FRAWProcessor(make_file(np.clip(cam + rng.normal(0, 0.003, cam.shape), 0, 1), orientation=8))
    full = to_uint8(p.develop(prm, demosaic="bilinear"))
    h, w = full.shape[:2]
    rect = (0.25, 0.25, 0.75, 0.75)
    reg = p.render_region(prm, rect, w // 2, h // 2, demosaic="bilinear")
    ref = full[h // 4 : h // 4 + h // 2, w // 4 : w // 4 + w // 2]
    assert reg.shape == ref.shape
    assert np.abs(reg.astype(int) - ref.astype(int)).mean() < 2.0
    small = p.render_region(prm, (0, 0, 1, 1), w // 4, h // 4)  # superpixel source
    assert small.shape == (h // 4, w // 4, 3)


def test_pixel_values_histogram_clipping() -> None:
    cam = np.zeros((16, 32, 3))
    cam[:, 16:] = 1.0
    p = FRAWProcessor(make_file(cam))
    img = p.preview({})
    assert p.pixel_values(0.1, 0.5, {}) == (0, 0, 0)
    assert p.pixel_values(0.9, 0.5, {}) == (255, 255, 255)
    h = FRAWProcessor.histogram(img)
    assert set(h) >= {"r", "g", "b", "l", "clip_shadows", "clip_highlights"}
    assert h["r"].sum() == img.shape[0] * img.shape[1] and len(h["l"]) == 256
    assert h["clip_shadows"] == pytest.approx(0.5) and h["clip_highlights"] == pytest.approx(0.5)
    sm, hm = p.clipping_mask(img)
    assert sm.shape == img.shape[:2] and sm[:, :4].all() and hm[:, -4:].all() and not (sm & hm).any()
