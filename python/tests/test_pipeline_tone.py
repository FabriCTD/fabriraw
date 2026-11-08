"""Motor Pro — basic / tone curve groups: direction of every tonal control."""

from __future__ import annotations

import numpy as np

from fraw.pipeline.tone import monotone_spline, parametric_curve
from fraw.processor import FRAWProcessor
from test_processor import flat, make_file

LIN = {"gamma_mode": "linear"}


def ramp_proc(n: int = 64) -> FRAWProcessor:
    r = np.linspace(0.005, 0.95, n)
    cam = np.broadcast_to(r[None, :, None], (8, n, 3)).copy()
    return FRAWProcessor(make_file(cam))


def test_monotone_spline_fritsch_carlson() -> None:
    x = np.linspace(0, 1, 2001)
    pts = [[0, 0], [0.2, 0.1], [0.5, 0.7], [0.8, 0.75], [1, 1]]
    y = monotone_spline(pts, x)
    assert np.all(np.diff(y) >= -1e-12)  # monotone data → monotone curve (no overshoot)
    for px, py in pts:
        assert abs(monotone_spline(pts, np.array([px]))[0] - py) < 1e-9
    flat_seg = monotone_spline([[0, 0], [0.3, 0.5], [0.6, 0.5], [1, 1]], np.linspace(0.3, 0.6, 50))
    np.testing.assert_allclose(flat_seg, 0.5, atol=1e-12)  # flat segment stays flat
    np.testing.assert_allclose(monotone_spline([[0.2, 0.3], [0.8, 0.9]], np.array([0.0, 1.0])), [0.3, 0.9])


def test_parametric_curve_regions() -> None:
    x = np.linspace(0, 1, 1001)
    for i, name in enumerate(("shadows", "darks", "lights", "highlights")):
        args = [0.0] * 4
        args[i] = 1.0
        y = parametric_curve(x, *args)
        c = (0.125, 0.375, 0.625, 0.875)[i]
        k = int(c * 1000)
        assert y[k] > x[k] + 0.08, name  # its region goes up
        assert y[0] == 0 and abs(y[-1] - 1) < 1e-12 and np.all(np.diff(y) >= 0)
        far = int(((c + 0.5) % 1.0) * 1000)
        assert abs(y[far] - x[far]) < 0.03


def test_curve_master_lifts_midtones() -> None:
    p = FRAWProcessor(make_file(flat(8, 8, (0.18, 0.18, 0.18))))
    base = float(p.develop({}, half_size=True).mean())  # ~0.46 encoded
    out = p.develop({"curve_master": [[0, 0], [base, 0.7], [1, 1]]}, half_size=True)
    np.testing.assert_allclose(out, 0.7, atol=2e-3)
    red = p.develop({"curve_red": [[0, 0], [0.5, 0.8], [1, 1]]}, half_size=True)
    assert red[..., 0].mean() > base + 0.1 and abs(red[..., 1].mean() - base) < 2e-3


def test_local_shadows_highlights_whites_blacks() -> None:
    p = ramp_proc()
    base = p.develop(LIN, half_size=True)[..., 1].mean(axis=0)
    sh = p.develop({**LIN, "shadows": 1}, half_size=True)[..., 1].mean(axis=0)
    hl = p.develop({**LIN, "highlights": -1}, half_size=True)[..., 1].mean(axis=0)
    assert sh[0] > 1.8 * base[0] and sh[-1] < 1.02 * base[-1]
    assert hl[-1] < 0.65 * base[-1] and abs(hl[0] / base[0] - 1) < 1e-3
    wh = p.develop({**LIN, "whites": 1}, half_size=True)[..., 1].mean(axis=0)
    bl = p.develop({**LIN, "blacks": -1}, half_size=True)[..., 1].mean(axis=0)
    assert wh[-1] > base[-1] * 1.05 and abs(wh[0] / base[0] - 1) < 0.02
    assert bl[0] < base[0] * 0.8 and abs(bl[-1] / base[-1] - 1) < 0.01
    lift = p.develop({**LIN, "blacks": 1}, half_size=True)[..., 1].mean(axis=0)
    assert lift[0] > base[0] + 0.002


def test_local_tone_keeps_local_contrast() -> None:
    """Highlights −1 compresses a bright region but keeps its texture (no flat wash)."""
    rng = np.random.default_rng(1)
    n = 192
    cam = np.full((n, n, 3), 0.05)
    tex = 0.7 + 0.06 * np.sin(np.arange(n) * np.pi / 2)[None, :]  # 4 px period
    cam[:, n // 2 :] = tex[:, n // 2 :, None]
    cam += rng.normal(0, 0.002, cam.shape)
    p = FRAWProcessor(make_file(np.clip(cam, 0, 1)))
    kw = {"demosaic": "bilinear"}
    a = p.develop(LIN, **kw)[..., 1]
    b = p.develop({**LIN, "highlights": -1}, **kw)[..., 1]
    ra, rb = a[:, 130:170], b[:, 130:170]
    assert rb.mean() < 0.75 * ra.mean()  # brightness compressed
    # relative local contrast kept (a per-pixel global curve would flatten it)
    assert rb.std() / rb.mean() > 0.9 * (ra.std() / ra.mean())
    np.testing.assert_allclose(b[:, :70], a[:, :70], rtol=0.02)  # dark side untouched


def test_clarity_texture_dehaze_contrast() -> None:
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:96, 0:96]
    big = 0.18 * (1 + 0.5 * np.sin(xx / 9.0))
    fine = 0.02 * np.sign(np.sin(xx * 1.3 + yy * 0.7))
    cam = np.repeat((big + fine)[..., None], 3, axis=2) + rng.normal(0, 0.001, (96, 96, 3))
    p = FRAWProcessor(make_file(np.clip(cam, 0, 1)))
    base = p.develop({}, demosaic="bilinear")[..., 1]
    for key in ("clarity", "texture"):
        up = p.develop({key: 1}, demosaic="bilinear")[..., 1]
        dn = p.develop({key: -1}, demosaic="bilinear")[..., 1]
        hp = lambda im: np.abs(np.diff(im, axis=1)).mean()  # noqa: E731
        assert hp(up) > hp(base) > hp(dn), key
    haze = np.clip(0.6 * cam + 0.35, 0, 1)  # low-contrast veil
    ph = FRAWProcessor(make_file(haze))
    b0 = ph.develop({}, demosaic="bilinear")
    d1 = ph.develop({"dehaze": 1}, demosaic="bilinear")
    assert d1.std() > 1.3 * b0.std() and d1.mean() < b0.mean()
    assert ph.develop({"dehaze": -1}, demosaic="bilinear").std() < b0.std()
    c1 = p.develop({"contrast": 1}, demosaic="bilinear")
    assert c1.std() > base.std()


def test_exposure_saturation_vibrance() -> None:
    cam = np.zeros((16, 48, 3))
    cam[:, :16] = (0.3, 0.12, 0.1)   # moderately saturated red
    cam[:, 16:32] = (0.2, 0.18, 0.17)  # almost neutral
    cam[:, 32:] = (0.5, 0.02, 0.02)  # very saturated
    p = FRAWProcessor(make_file(cam))

    def sat(img: np.ndarray, sl: slice) -> float:
        v = img[4:-4, sl].reshape(-1, 3)
        return float(((v.max(1) - v.min(1)) / v.max(1)).mean())

    base = p.develop({}, demosaic="bilinear")
    vib = p.develop({"vibrance": 1}, demosaic="bilinear")
    gain_low = sat(vib, slice(20, 28)) - sat(base, slice(20, 28))
    gain_high = sat(vib, slice(36, 44)) - sat(base, slice(36, 44))
    assert gain_low > 0.02 and gain_low > 1.5 * max(gain_high, 0.0)  # protects saturated colours
    s0 = p.develop({"saturation": 0}, demosaic="bilinear")
    assert sat(s0, slice(4, 12)) < 0.01
    ev = p.develop({"exposure_ev": 1}, demosaic="bilinear")
    assert ev.mean() > base.mean()
