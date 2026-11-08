"""Motor Pro — HSL, colour grading, calibration, white balance (Kelvin, auto, eyedropper), highlights."""

from __future__ import annotations

import numpy as np
import pytest

from fraw.format import Color
from fraw.pipeline import color as pcolor
from fraw.pipeline.wb import kelvin_to_multipliers, kelvin_to_xy, multipliers_to_kelvin
from fraw.processor import FRAWProcessor
from test_processor import flat, make_file

PATCHES = {"red": (0.5, 0.08, 0.06), "blue": (0.05, 0.1, 0.5), "grey": (0.18, 0.18, 0.18),
           "green": (0.08, 0.4, 0.06), "dark": (0.02, 0.02, 0.02), "white": (0.8, 0.8, 0.8)}


def patch_proc() -> FRAWProcessor:
    cam = np.zeros((24, 16 * len(PATCHES), 3))
    for i, c in enumerate(PATCHES.values()):
        cam[:, i * 16 : (i + 1) * 16] = c
    return FRAWProcessor(make_file(cam))


def means(img: np.ndarray) -> dict[str, np.ndarray]:
    return {k: img[6:-6, i * 16 + 4 : (i + 1) * 16 - 4].reshape(-1, 3).mean(0) for i, k in enumerate(PATCHES)}


def dev(p: FRAWProcessor, prm: dict) -> dict[str, np.ndarray]:
    return means(p.develop(prm, demosaic="bilinear"))


def test_hsl_blue_luminance_only_blues() -> None:
    p = patch_proc()
    b, a = dev(p, {}), dev(p, {"hsl_blue_l": -1})
    assert a["blue"].mean() < b["blue"].mean() - 0.05
    for k in ("red", "grey", "green", "white"):
        np.testing.assert_allclose(a[k], b[k], atol=2 / 255)


def test_hsl_sat_and_hue_bands() -> None:
    p = patch_proc()
    b = dev(p, {})
    s = dev(p, {"hsl_red_s": -1})
    sat = lambda v: (v.max() - v.min()) / v.max()  # noqa: E731
    assert sat(s["red"]) < 0.5 * sat(b["red"])
    np.testing.assert_allclose(s["green"], b["green"], atol=1 / 255)
    h = dev(p, {"hsl_green_h": 1})["green"]
    assert h[2] > b["green"][2] + 0.03  # green turns towards aqua (+B)
    np.testing.assert_allclose(dev(p, {"hsl_green_h": 1})["grey"], b["grey"], atol=1 / 255)


def test_color_grading_zones() -> None:
    p = patch_proc()
    b = dev(p, {})
    sh = dev(p, {"grade_shadows_hue": 225, "grade_shadows_sat": 1})
    hl = dev(p, {"grade_highlights_hue": 30, "grade_highlights_sat": 1})
    assert sh["dark"][2] - sh["dark"][0] > b["dark"][2] - b["dark"][0] + 0.01  # dark tinted blue
    assert np.abs(sh["white"] - b["white"]).max() < 2 / 255
    assert hl["white"][0] - hl["white"][2] > 0.05  # whites tinted orange
    assert np.abs(hl["dark"] - b["dark"]).max() < 2 / 255
    lum = dev(p, {"grade_global_lum": 1})
    assert all(lum[k].mean() > b[k].mean() for k in PATCHES)
    wide = dev(p, {"grade_shadows_hue": 225, "grade_shadows_sat": 1, "grade_balance": -1})
    assert wide["grey"][2] - wide["grey"][0] > sh["grey"][2] - sh["grey"][0]  # balance −: bigger shadow zone


def test_calibration_matrix_and_effect() -> None:
    m = pcolor.calibration_matrix(0.7, -0.4, 0.2, 0.5, -0.6, 0.3)
    np.testing.assert_allclose(m @ np.ones(3), np.ones(3), atol=1e-9)  # white preserved
    np.testing.assert_array_equal(pcolor.calibration_matrix(0, 0, 0, 0, 0, 0), np.eye(3))
    p = patch_proc()
    b, a = dev(p, {}), dev(p, {"cal_red_hue": 1})
    assert np.abs(a["red"] - b["red"]).max() > 0.05
    np.testing.assert_allclose(a["grey"], b["grey"], atol=1 / 255)
    s = dev(p, {"cal_blue_sat": -1})["blue"]
    sat = lambda v: (v.max() - v.min()) / v.max()  # noqa: E731
    assert sat(s) < sat(b["blue"])
    t = dev(p, {"cal_shadows_tint": 1})["dark"]
    assert t[1] < b["dark"][1] - 1 / 255  # magenta in the shadows (less green)


# ------------------------------------------------------------------ Kelvin
COLORS = [
    Color(cam_xyz=(0.6722, -0.0635, -0.0963, -0.4287, 1.2460, 0.2028, -0.0908, 0.2162, 0.5668), flags=2),  # 5D-like
    Color(daylight_wb=(2.1, 1.0, 1.4, 1.0), rgb_cam=(1.6, -0.4, -0.2, -0.2, 1.5, -0.3, 0, -0.5, 1.5), flags=1),
    Color(),
]


@pytest.mark.parametrize("ci", range(len(COLORS)))
def test_kelvin_round_trip(ci: int) -> None:
    c = COLORS[ci]
    for k in (2500, 3200, 4500, 5500, 6500, 8000, 10000, 12000):
        for tint in (-40.0, 0.0, 35.0):
            mul = kelvin_to_multipliers(k, tint, c)
            assert mul[1] == 1.0 and mul[0] > 0 and mul[2] > 0
            k2, t2 = multipliers_to_kelvin(mul, c)
            assert abs(k2 - k) <= 50, (k, tint, k2)
            assert abs(t2 - tint) <= 1.0
            assert multipliers_to_kelvin(c, mul) == (k2, t2)  # either argument order


def test_kelvin_direction_and_locus() -> None:
    x, y = kelvin_to_xy(6504)
    assert abs(x - 0.3135) < 0.003 and abs(y - 0.3237) < 0.004  # Planckian 6500 K
    c = COLORS[0]
    warm, cool = kelvin_to_multipliers(3000, 0, c), kelvin_to_multipliers(9000, 0, c)
    assert warm[2] / warm[0] > cool[2] / cool[0]  # tungsten needs more blue gain
    green = kelvin_to_multipliers(5500, 80, c)  # greener illuminant → larger R,B gains vs G
    neut = kelvin_to_multipliers(5500, 0, c)
    assert green[0] > neut[0] and green[2] > neut[2]


def test_kelvin_mode_and_as_shot_kelvin() -> None:
    c = COLORS[0]
    mul = kelvin_to_multipliers(4200, 10, c)
    f = make_file(flat(8, 8, (0.2, 0.2, 0.2)), wb=(mul[0], 1.0, mul[2], 1.0))
    f.color.cam_xyz, f.color.flags = c.cam_xyz, 2
    p = FRAWProcessor(f)
    k, t = p.as_shot_kelvin()
    assert abs(k - 4200) < 50 and abs(t - 10) < 1
    a = p.develop({"wb_mode": "kelvin", "wb_kelvin": k, "wb_kelvin_tint": t}, half_size=True)
    b = p.develop({}, half_size=True)
    np.testing.assert_allclose(a, b, atol=2e-3)


def test_auto_wb_and_eyedropper_neutralize() -> None:
    rng = np.random.default_rng(0)
    cam = np.clip(rng.uniform(0.05, 0.6, (64, 64, 1)) * np.ones(3) + rng.normal(0, 0.02, (64, 64, 3)), 0, 1)
    cam[:8] = 1.0  # saturated sky must be ignored
    f = make_file(cam, wb=(2.0, 1.0, 1.6, 1.0))
    f.color.as_shot_wb = (1.0, 1.0, 1.0, 1.0)  # as-shot WB wrong
    p = FRAWProcessor(f)
    r = p.auto_wb()
    assert r["wb_mode"] == "custom"
    np.testing.assert_allclose(r["wb_custom"], [2.0, 1.0, 1.6], rtol=0.08)
    out = p.develop(r, half_size=True)[8:]
    m = out.reshape(-1, 3).mean(0)
    assert np.ptp(m) < 0.03
    grey = make_file(flat(32, 32, (0.3, 0.3, 0.3)), wb=(1.7, 1.0, 1.4, 1.0))
    grey.color.as_shot_wb = (1.0, 1.0, 1.0, 1.0)
    pg = FRAWProcessor(grey)
    for prm in ({}, {"rotate_90": "90", "crop_left": 0.2}):
        mul = pg.sample_wb(0.5, 0.5, 3, params=prm)
        np.testing.assert_allclose(mul, [1.7, 1.0, 1.4], rtol=1e-3)


# --------------------------------------------------------------- highlights
def test_highlight_reconstruct_recovers_clipped_channel() -> None:
    n = 96
    yy, xx = np.mgrid[0:n, 0:n]
    tex = 0.5 + 0.1 * np.sin(xx / 6.0) * np.cos(yy / 8.0)
    true = np.stack([tex * 1.9, tex, tex * 0.8], axis=-1)  # R exceeds the sensor range where tex > 0.526
    cam = np.clip(true, 0, 1)
    p = FRAWProcessor(make_file(cam))
    lin = {"gamma_mode": "linear", "exposure_ev": -1.0}
    clip = p.develop({**lin, "highlight_mode": "clip"}, half_size=True)
    rec = p.develop({**lin, "highlight_mode": "reconstruct"}, half_size=True)
    blend = p.develop({**lin, "highlight_mode": "blend"}, half_size=True)
    tr = 0.5 * true[..., 0].reshape(n // 2, 2, n // 2, 2).mean(axis=(1, 3))  # value at the superpixel centre
    sat = true[::2, ::2, 0] > 1.0
    assert sat.sum() > 50
    err_clip = np.abs(clip[..., 0] - tr)[sat].mean()
    err_rec = np.abs(rec[..., 0] - tr)[sat].mean()
    assert err_rec < 0.5 * err_clip
    assert np.corrcoef(rec[..., 0][sat], tr[sat])[0, 1] > 0.9  # texture recovered
    assert clip[..., 0][sat].std() < 1e-6  # flat in clip mode
    low = true[::2, ::2, 0] < 0.98
    np.testing.assert_allclose(rec[low], clip[low], atol=1e-6)  # unclipped pixels untouched
    # blend: fades towards neutral near saturation (R/G ratio lower than clip's)
    assert (blend[..., 0] / np.maximum(blend[..., 1], 1e-6))[sat].mean() <= (clip[..., 0] / clip[..., 1])[sat].mean()
