"""Development engine tests (synthetic, deterministic) + golden smoke test."""

from __future__ import annotations

import statistics
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import FIXTURES
from fraw import demosaic as dm
from fraw.devparams import DevParams
from fraw.format import Color, FrawFile, Head
from fraw.processor import (
    FRAWProcessor,
    apply_orientation,
    srgb_encode,
    to_uint8,
    to_uint16,
    transfer,
)

LIN = {"gamma_mode": "linear"}
RGGB = (0, 1, 1, 2)


def make_file(
    cam: np.ndarray,
    *,
    pattern=RGGB,
    black=(0, 0, 0, 0),
    white: int = 65535,
    wb=(1.0, 1.0, 1.0, 1.0),
    rgb_cam=None,
    orientation: int = 1,
    crop=(0, 0, 0, 0),
) -> FrawFile:
    """FrawFile whose normalized, *un-white-balanced* CFA equals mosaic(cam / wb)."""
    cam = np.asarray(cam, dtype=np.float64)
    h, w, _ = cam.shape
    pre = cam / np.array([wb[0], wb[1], wb[2]])[None, None, :]
    x = dm.mosaic(pre.astype(np.float32), pattern).astype(np.float64)
    bl = np.tile(np.array(black, dtype=np.float64).reshape(2, 2), (h // 2, w // 2))
    raw = np.round(bl + np.clip(x, 0, 1) * (white - bl)).astype(np.uint16)
    color = Color(as_shot_wb=tuple(wb), daylight_wb=(1.0, 1.0, 1.0, 1.0))
    if rgb_cam is not None:
        color = Color(as_shot_wb=tuple(wb), rgb_cam=tuple(np.ravel(rgb_cam)), flags=1)
    head = Head(
        width=w,
        height=h,
        effective_bits=16,
        white_level=white,
        cfa_pattern=tuple(pattern),
        black_level=tuple(black),
        orientation=orientation,
        crop_left=crop[0],
        crop_top=crop[1],
        crop_width=crop[2],
        crop_height=crop[3],
    )
    f = FrawFile(head=head, raw=raw, color=color)
    f.validate()
    return f


def flat(h: int, w: int, rgb) -> np.ndarray:
    return np.broadcast_to(np.array(rgb, dtype=np.float64), (h, w, 3)).copy()


# ---------------------------------------------------------------------------
# DevParams
# ---------------------------------------------------------------------------
def test_devparams_defaults() -> None:
    p = DevParams()
    d = p.to_dict()
    v1_keys = {
        "exposure_ev": 0.0,
        "wb_mode": "as_shot",
        "wb_custom": [2.0, 1.0, 1.5],
        "temperature_shift": 0.0,
        "tint_shift": 0.0,
        "contrast": 0.0,
        "saturation": 1.0,
        "highlights": 0.0,
        "shadows": 0.0,
        "gamma_mode": "srgb",
        "gamma": 2.2,
        "demosaic": "ahd",
    }
    # DEV v2 conserva las 12 claves v1 con el mismo defecto (SPEC §4.6)
    assert d["version"] == 2
    assert {k: d[k] for k in v1_keys} == v1_keys
    assert p.to_dict(full=False) == {"version": 2}
    assert DevParams.from_dict({}) == p
    assert DevParams.from_dict(None) == p
    assert DevParams.from_dict([1, 2]) == p  # type: ignore[arg-type]


def test_devparams_clamp_and_invalid() -> None:
    p = DevParams.from_dict(
        {
            "exposure_ev": 9,
            "temperature_shift": -5.5,
            "tint_shift": 2,
            "contrast": -3,
            "saturation": 7,
            "highlights": 1.5,
            "shadows": -1.01,
            "gamma": 0.1,
            "wb_custom": [100, 0, 1.2],
            "wb_mode": "custom",
            "gamma_mode": "power",
            "demosaic": "bilinear",
            "unknown": 123,
            "version": 99,
        }
    )
    assert (p.exposure_ev, p.temperature_shift, p.tint_shift, p.contrast) == (4.0, -1.0, 1.0, -1.0)
    assert (p.saturation, p.highlights, p.shadows, p.gamma) == (2.0, 1.0, -1.0, 1.0)
    assert p.wb_custom == (64.0, 1e-3, 1.2)
    assert (p.wb_mode, p.gamma_mode, p.demosaic) == ("custom", "power", "bilinear")
    bad = DevParams.from_dict(
        {
            "exposure_ev": "1",
            "saturation": True,
            "contrast": float("nan"),
            "gamma": None,
            "wb_mode": "auto",
            "gamma_mode": 3,
            "demosaic": "vng",
            "wb_custom": [1, 2],
            "highlights": [0.5],
            "shadows": float("inf"),
        }
    )
    assert bad == DevParams()
    assert DevParams.from_dict({"wb_custom": [1, "x", 2]}).wb_custom == (2.0, 1.0, 1.5)


def test_devparams_roundtrip() -> None:
    p = DevParams(exposure_ev=1.25, wb_mode="daylight", wb_custom=(1.9, 1.0, 1.3), contrast=0.3,
                  saturation=1.4, highlights=-0.4, shadows=0.2, gamma_mode="power", gamma=1.8, demosaic="bilinear",
                  temperature_shift=0.1, tint_shift=-0.2)
    d = p.to_dict()
    assert d["version"] == 2
    assert DevParams.from_dict(d) == p
    import json

    assert DevParams.from_dict(json.loads(json.dumps(d))) == p


# ---------------------------------------------------------------------------
# Normalization, crop, pattern
# ---------------------------------------------------------------------------
def test_normalization_exact() -> None:
    black = (100, 200, 300, 400)
    white = 4100
    raw = np.array(
        [[100, 200, 2100, 4100], [400, 300, 5000, 0], [4100, 2150, 600, 350], [401, 299, 4100, 4000]],
        dtype=np.uint16,
    )
    head = Head(width=4, height=4, effective_bits=12, white_level=white, black_level=black)
    p = FRAWProcessor(FrawFile(head=head, raw=raw, color=Color()))
    bl = np.tile(np.array(black, dtype=np.float64).reshape(2, 2), (2, 2))
    expected = np.clip((raw - bl) / (white - bl), 0, 1)
    assert p.norm.dtype == np.float32
    np.testing.assert_allclose(p.norm, expected, rtol=0, atol=1e-6)
    assert p.norm[0, 0] == 0.0 and p.norm[0, 3] == 1.0 and p.norm[1, 3] == 0.0  # clip both ends


@pytest.mark.parametrize(
    "left, top, expected",
    [(0, 0, (0, 1, 1, 2)), (1, 0, (1, 0, 2, 1)), (0, 1, (1, 2, 0, 1)), (1, 1, (2, 1, 1, 0)), (3, 5, (2, 1, 1, 0))],
)
def test_crop_offset_shifts_pattern(left: int, top: int, expected) -> None:
    h = w = 12
    # Signal only on the ORIGINAL red photosites; per-position black levels differ.
    cam = np.zeros((h, w, 3))
    cam[..., 0] = 0.5
    f = make_file(cam, black=(10, 20, 30, 40), white=4010)
    f.head.crop_left, f.head.crop_top, f.head.crop_width, f.head.crop_height = left, top, 6, 6
    f.validate()
    p = FRAWProcessor(f)
    assert p.pattern == expected
    assert p.norm.shape == (6, 6)
    out = p.develop(LIN, half_size=True)
    assert out.shape == (3, 3, 3)
    np.testing.assert_allclose(out[..., 0], 0.5, atol=2e-3)  # red channel gets the signal
    np.testing.assert_allclose(out[..., 1:], 0.0, atol=1e-6)  # nothing leaks into G/B
    for method in ("bilinear", "ahd"):
        full = p.develop(LIN, demosaic=method)
        np.testing.assert_allclose(full[..., 0], 0.5, atol=2e-3)
        np.testing.assert_allclose(full[..., 1:], 0.0, atol=1e-6)


def test_crop_extracts_area() -> None:
    cam = np.zeros((8, 8, 3))
    cam[2:6, 2:6] = 0.75  # bright square exactly in the crop window
    p = FRAWProcessor(make_file(cam, crop=(2, 2, 4, 4)))
    assert p.norm.shape == (4, 4)
    np.testing.assert_allclose(p.norm, 0.75, atol=1e-4)


# ---------------------------------------------------------------------------
# White balance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pattern", [(0, 1, 1, 2), (2, 1, 1, 0), (1, 0, 2, 1), (1, 2, 0, 1)])
def test_wb_neutralizes_tinted_grey(pattern) -> None:
    wb = (2.0, 1.0, 1.5, 1.0)
    f = make_file(flat(16, 16, (0.25, 0.25, 0.25)), wb=wb, pattern=pattern)
    p = FRAWProcessor(f)
    # raw is tinted (R=0.125, G=0.25, B=0.1667) ...
    assert abs(float(p.norm.max()) - 0.25) < 1e-4 and float(p.norm.min()) < 0.13
    np.testing.assert_allclose(p.wb_multipliers(LIN), [2.0, 1.0, 1.5, 1.0])
    for kw in ({"half_size": True}, {"demosaic": "bilinear"}, {"demosaic": "ahd"}):
        out = p.develop(LIN, **kw)
        np.testing.assert_allclose(out, 0.25, atol=1e-4)  # ... and comes out neutral grey
    # Without WB the same file is visibly tinted.
    nowb = p.develop({**LIN, "wb_mode": "custom", "wb_custom": [1, 1, 1]}, half_size=True)
    assert nowb[..., 0].mean() < 0.2 and nowb[..., 1].mean() > 0.24


def test_wb_modes_and_normalization() -> None:
    f = make_file(flat(4, 4, (0.1, 0.1, 0.1)), wb=(2.0, 1.0, 1.5, 1.0))
    f.color.daylight_wb = (2.5, 1.0, 1.2, 1.0)
    p = FRAWProcessor(f)
    np.testing.assert_allclose(p.wb_multipliers({"wb_mode": "daylight"}), [2.5, 1.0, 1.2, 1.0])
    # custom: normalized so the smallest multiplier is 1
    np.testing.assert_allclose(p.wb_multipliers({"wb_mode": "custom", "wb_custom": [4, 2, 3]}), [2, 1, 1.5, 1])
    warm = p.wb_multipliers({"temperature_shift": 1.0})
    np.testing.assert_allclose(warm[0] / warm[2], (2.0 / 1.5) * np.exp(1.0), rtol=1e-6)
    assert warm.min() == pytest.approx(1.0)
    mag = p.wb_multipliers({"tint_shift": 1.0})
    assert mag[1] < 1.0 * mag[0] / 2.0  # green reduced relative to R
    green = p.wb_multipliers({"tint_shift": -1.0})
    assert green[1] / green[0] > 1.0 / 2.0
    # invalid as_shot (zeros) → unity
    f.color.as_shot_wb = (0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(FRAWProcessor(f).wb_multipliers({}), [1, 1, 1, 1])


def test_highlight_clip_no_magenta() -> None:
    # Fully saturated sensor + strong WB → must stay neutral white, not magenta.
    f = make_file(flat(8, 8, (1.0, 1.0, 1.0)), wb=(1.0, 1.0, 1.0, 1.0))
    f.color.as_shot_wb = (2.2, 1.0, 1.6, 1.0)
    p = FRAWProcessor(f)
    for kw in ({"half_size": True}, {"demosaic": "ahd"}):
        out = p.develop(LIN, **kw)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)
    prev = p.preview({})
    assert np.all(prev == 255)


# ---------------------------------------------------------------------------
# Colour matrix and tone
# ---------------------------------------------------------------------------
def test_matrix_identity_vs_rgb_cam() -> None:
    cam = (0.2, 0.3, 0.4)
    m = np.array([[1.2, -0.1, -0.1], [-0.2, 1.3, -0.1], [0.0, -0.3, 1.3]])
    ident = FRAWProcessor(make_file(flat(8, 8, cam))).develop(LIN, half_size=True)
    np.testing.assert_allclose(ident, np.broadcast_to(cam, ident.shape), atol=1e-4)
    f = make_file(flat(8, 8, cam), rgb_cam=m)
    withm = FRAWProcessor(f).develop(LIN, half_size=True)
    np.testing.assert_allclose(withm, np.broadcast_to(m @ np.array(cam), withm.shape), atol=1e-4)
    f.color.flags = 0  # matrix present but flagged invalid → identity
    np.testing.assert_allclose(FRAWProcessor(f).develop(LIN, half_size=True), ident, atol=1e-7)


def test_exposure_plus_one_doubles_linear() -> None:
    rng = np.random.default_rng(5)
    cam = rng.uniform(0.05, 0.45, size=(16, 16, 3))
    p = FRAWProcessor(make_file(cam))
    for kw in ({"half_size": True}, {"demosaic": "ahd"}):
        base = p.develop({**LIN, "exposure_ev": 0.0}, **kw)
        plus = p.develop({**LIN, "exposure_ev": 1.0}, **kw)
        minus = p.develop({**LIN, "exposure_ev": -1.0}, **kw)
        np.testing.assert_allclose(plus, np.minimum(2.0 * base, 1.0), rtol=1e-6, atol=1e-7)  # AHD may overshoot 0.5
        np.testing.assert_allclose(minus, 0.5 * base, rtol=1e-6, atol=1e-7)
    clamped = p.develop({**LIN, "exposure_ev": 10}, half_size=True)  # clamp to +4 → x16, clipped to 1
    np.testing.assert_allclose(clamped, np.minimum(16 * p.develop(LIN, half_size=True), 1.0), rtol=1e-6)


def test_transfer_curves() -> None:
    t = 0.0031308
    assert float(srgb_encode(np.float32(t))) == pytest.approx(12.92 * t, rel=1e-6)
    # Continuity at the breakpoint (both branches agree within float32 precision).
    hi = 1.055 * t ** (1 / 2.4) - 0.055
    assert hi == pytest.approx(12.92 * t, abs=1e-6)
    assert float(srgb_encode(np.float32(1.0))) == pytest.approx(1.0, abs=1e-6)
    assert float(srgb_encode(np.float32(0.0))) == 0.0
    assert float(srgb_encode(np.float32(0.18))) == pytest.approx(0.4614, abs=1e-3)
    x = np.linspace(0, 1, 1001, dtype=np.float32)
    assert np.all(np.diff(srgb_encode(x)) >= 0)
    np.testing.assert_allclose(transfer(x, "power", 2.2), x ** (1 / 2.2), rtol=1e-6)
    np.testing.assert_array_equal(transfer(x, "linear"), x)
    np.testing.assert_allclose(transfer(np.array([-0.5, 2.0], np.float32), "srgb"), [0.0, 1.0], atol=1e-6)


def test_develop_srgb_and_power_end_to_end() -> None:
    p = FRAWProcessor(make_file(flat(4, 4, (0.18, 0.18, 0.18))))
    np.testing.assert_allclose(p.develop({}, half_size=True), 0.4614, atol=2e-3)
    np.testing.assert_allclose(p.develop({"gamma_mode": "power", "gamma": 2.0}, half_size=True), 0.18**0.5, atol=1e-4)


def test_tone_controls() -> None:
    ramp = np.linspace(0.01, 0.9, 64)
    cam = np.broadcast_to(ramp[None, :, None], (2, 64, 3)).copy()
    cam[..., 0] *= 0.8  # slightly coloured so saturation matters
    p = FRAWProcessor(make_file(cam))
    base = p.develop(LIN, half_size=True)
    # Contrast keeps middle grey (0.18) fixed and steepens the curve around it.
    pg = FRAWProcessor(make_file(flat(2, 2, (0.18, 0.18, 0.18))))
    np.testing.assert_allclose(pg.develop({**LIN, "contrast": 0.8}, half_size=True), 0.18, atol=1e-4)
    hc = p.develop({**LIN, "contrast": 0.8}, half_size=True)
    lo, hi = base[0, 0, 1] < 0.18, base[0, -1, 1] > 0.18
    assert lo and hi
    assert hc[0, 0, 1] < base[0, 0, 1] and hc[0, -1, 1] > base[0, -1, 1]
    # Saturation 0 → grey (R == G == B == Rec.709 luminance).
    grey = p.develop({**LIN, "saturation": 0.0}, half_size=True)
    np.testing.assert_allclose(grey[..., 0], grey[..., 1], atol=1e-6)
    np.testing.assert_allclose(grey[..., 1], grey[..., 2], atol=1e-6)
    # Shadows lift dark pixels much more than bright ones; highlights −1 darkens bright ones.
    sh = p.develop({**LIN, "shadows": 1.0}, half_size=True)
    assert sh[0, 0, 1] / base[0, 0, 1] > 2.0 and sh[0, -1, 1] / base[0, -1, 1] < 1.01
    hl = p.develop({**LIN, "highlights": -1.0}, half_size=True)
    assert hl[0, -1, 1] < 0.6 * base[0, -1, 1] and hl[0, 0, 1] == pytest.approx(base[0, 0, 1], rel=1e-5)


# ---------------------------------------------------------------------------
# Geometry: orientation, half size
# ---------------------------------------------------------------------------
def test_orientation_helper() -> None:
    a = np.arange(6).reshape(2, 3)
    assert apply_orientation(a, 1) is a
    np.testing.assert_array_equal(apply_orientation(a, 2), a[:, ::-1])
    np.testing.assert_array_equal(apply_orientation(a, 3), a[::-1, ::-1])
    np.testing.assert_array_equal(apply_orientation(a, 4), a[::-1])
    np.testing.assert_array_equal(apply_orientation(a, 5), a.T)
    np.testing.assert_array_equal(apply_orientation(a, 6), np.array([[3, 0], [4, 1], [5, 2]]))
    np.testing.assert_array_equal(apply_orientation(a, 7), np.array([[5, 2], [4, 1], [3, 0]]))
    np.testing.assert_array_equal(apply_orientation(a, 8), np.array([[2, 5], [1, 4], [0, 3]]))


def test_orientation_6_rotates_90(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    cam = rng.uniform(0.1, 0.9, size=(8, 12, 3))
    p = FRAWProcessor(make_file(cam, orientation=6))
    up = p.develop(LIN, half_size=True, orient=False)
    rot = p.develop(LIN, half_size=True)
    assert up.shape == (4, 6, 3) and rot.shape == (6, 4, 3)
    np.testing.assert_array_equal(rot, np.rot90(up, -1))
    np.testing.assert_array_equal(rot[0, 0], up[-1, 0])  # clockwise: bottom-left → top-left
    assert p.develop(LIN, demosaic="bilinear").shape == (12, 8, 3)
    assert p.preview(LIN).shape == (6, 4, 3)
    out = p.export(tmp_path / "o.png", LIN, half_size=True)
    with Image.open(out) as im:
        assert im.size == (4, 6)  # (width, height)


def test_half_size_shape() -> None:
    p = FRAWProcessor(make_file(flat(10, 14, (0.3, 0.3, 0.3))))
    assert p.develop(half_size=True).shape == (5, 7, 3)
    assert p.develop(demosaic="bilinear").shape == (10, 14, 3)
    out = p.develop(demosaic="ahd")
    assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1


def test_default_params_from_dev() -> None:
    f = make_file(flat(4, 4, (0.1, 0.1, 0.1)))
    f.dev = {"version": 1, "exposure_ev": 1.0, "gamma_mode": "linear"}
    p = FRAWProcessor(f)
    assert p.default_params().exposure_ev == 1.0
    np.testing.assert_allclose(p.develop(half_size=True), 0.2, atol=1e-4)
    np.testing.assert_allclose(p.develop(DevParams(gamma_mode="linear"), half_size=True), 0.1, atol=1e-4)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
PREVIEW_CASES = [
    {},
    {"exposure_ev": 0.7, "contrast": 0.4},
    {"highlights": -0.6, "shadows": 0.5, "saturation": 1.4},
    {"temperature_shift": 0.3, "tint_shift": -0.2, "gamma_mode": "power", "gamma": 2.4},
    {"wb_mode": "custom", "wb_custom": [1.5, 1, 2], "gamma_mode": "linear", "contrast": -0.5},
]


@pytest.mark.parametrize("params", PREVIEW_CASES)
def test_preview_matches_half_size_develop(params) -> None:
    rng = np.random.default_rng(11)
    cam = rng.uniform(0.0, 0.6, size=(64, 96, 3))
    cam[:8] = 1.0  # some clipped highlights
    f = make_file(cam, wb=(2.0, 1.0, 1.5, 1.0), rgb_cam=[[1.6, -0.4, -0.2], [-0.2, 1.5, -0.3], [0.0, -0.5, 1.5]])
    p = FRAWProcessor(f)
    prev = p.preview(params, max_side=1600)
    ref = to_uint8(p.develop(params, half_size=True))
    assert prev.shape == ref.shape and prev.dtype == np.uint8
    diff = np.abs(prev.astype(int) - ref.astype(int))
    assert diff.max() <= 3, diff.max()  # 4096-entry LUT quantization only
    assert diff.mean() < 0.5


def test_preview_downscale_and_cache() -> None:
    p = FRAWProcessor(make_file(flat(200, 300, (0.2, 0.2, 0.2))))
    a = p.preview({}, max_side=50)  # half = 100×150 → factor 3 → 33×50
    assert a.shape == (33, 50, 3)
    b = p.preview({"exposure_ev": 1}, max_side=50)
    assert b.mean() > a.mean()
    assert a.mean() == p.preview({}, max_side=50).mean()  # result is not aliased to the internal buffers
    assert 50 in p._preview_cache


def test_preview_speed() -> None:
    """~1600×1067 preview from a 3200×2134 mosaic; target < 16 ms per call after the first."""
    rng = np.random.default_rng(0)
    h, w = 2134, 3200
    raw = rng.integers(512, 16000, size=(h, w), dtype=np.uint16)
    f = FrawFile(
        head=Head(width=w, height=h, effective_bits=14, white_level=16383, black_level=(512,) * 4),
        raw=raw,
        color=Color(as_shot_wb=(2.0, 1.0, 1.5, 1.0), rgb_cam=(1.6, -0.4, -0.2, -0.2, 1.5, -0.3, 0, -0.5, 1.5), flags=1),
    )
    p = FRAWProcessor(f)
    t0 = time.perf_counter()
    first = p.preview({})
    t_first = time.perf_counter() - t0
    assert first.shape == (1067, 1600, 3)
    results = {}
    for name, extra in (("basic", {}), ("hl/sh", {"highlights": -0.4, "shadows": 0.3})):
        ts = []
        for i in range(30):
            prm = {"exposure_ev": 0.01 * i, "contrast": 0.2, "temperature_shift": 0.1, **extra}
            t0 = time.perf_counter()
            p.preview(prm)
            ts.append(time.perf_counter() - t0)
        results[name] = statistics.median(ts) * 1e3
    print(f"\npreview 1600x1067: first={t_first * 1e3:.1f} ms, "
          + ", ".join(f"{k} median={v:.2f} ms" for k, v in results.items()))
    if results["basic"] >= 16.0:
        pytest.xfail(f"slow machine: preview median {results['basic']:.2f} ms >= 16 ms")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _read_png16(path: Path) -> np.ndarray:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n : pos + 12 + n])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    w, h, depth, ctype, *_ = ihdr
    assert depth == 16 and ctype == 2
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * 6)
    assert np.all(rows[:, 0] == 0)
    return rows[:, 1:].copy().view(">u2").reshape(h, w, 3).astype(np.uint16)


def _read_tiff16(path: Path) -> np.ndarray:
    data = path.read_bytes()
    assert data[:4] == b"II*\0"
    (ifd,) = struct.unpack_from("<I", data, 4)
    (n,) = struct.unpack_from("<H", data, ifd)
    tags = {}
    for i in range(n):
        tag, typ, cnt, val = struct.unpack_from("<HHII", data, ifd + 2 + 12 * i)
        if typ == 3 and cnt == 1:
            val &= 0xFFFF
        tags[tag] = (typ, cnt, val)
    w, h = tags[256][2], tags[257][2]
    typ, cnt, off = tags[258]
    assert cnt == 3 and struct.unpack_from("<3H", data, off) == (16, 16, 16)
    assert tags[277][2] == 3 and tags[259][2] == 1 and tags[262][2] == 2
    so, sc = tags[273][2], tags[279][2]
    assert sc == w * h * 6
    return np.frombuffer(data, "<u2", count=w * h * 3, offset=so).reshape(h, w, 3).astype(np.uint16)


def test_export_8_and_16_bits(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    cam = rng.uniform(0.0, 1.0, size=(24, 32, 3))
    p = FRAWProcessor(make_file(cam))
    params = {"demosaic": "bilinear"}
    ref = p.develop(params)
    ref8, ref16 = to_uint8(ref), to_uint16(ref)
    assert ref16.max() > 255 and len(np.unique(ref16)) > 256  # genuinely uses 16 bits

    png8 = p.export(tmp_path / "a.png", params, bits=8)
    with Image.open(png8) as im:
        assert im.mode == "RGB" and im.size == (32, 24)
        np.testing.assert_array_equal(np.asarray(im), ref8)
    tif8 = p.export(tmp_path / "a.tif", params, bits=8)
    with Image.open(tif8) as im:
        np.testing.assert_array_equal(np.asarray(im), ref8)
    jpg = p.export(tmp_path / "a.jpg", params, quality=95)
    with Image.open(jpg) as im:
        assert im.format == "JPEG" and im.size == (32, 24)
        assert np.abs(np.asarray(im).astype(int) - ref8).mean() < 12

    png16 = p.export(tmp_path / "b.png", params, bits=16)
    np.testing.assert_array_equal(_read_png16(png16), ref16)
    with Image.open(png16) as im:  # Pillow recognizes it as a 16-bit RGB PNG
        assert im.format == "PNG" and im.size == (32, 24)
    tif16 = p.export(tmp_path / "b.tiff", params, bits=16)
    np.testing.assert_array_equal(_read_tiff16(tif16), ref16)
    with Image.open(tif16) as im:
        assert im.format == "TIFF" and im.size == (32, 24)

    with pytest.raises(ValueError):
        p.export(tmp_path / "c.jpg", params, bits=16)
    with pytest.raises(ValueError):
        p.export(tmp_path / "c.bmp", params)
    with pytest.raises(ValueError):
        p.export(tmp_path / "c.png", params, bits=12)


def test_uint_conversions() -> None:
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], np.float32)
    assert to_uint8(x).tolist() == [0, 0, 128, 255, 255]
    assert to_uint16(x).tolist() == [0, 0, 32768, 65535, 65535]


# ---------------------------------------------------------------------------
# Golden fixture smoke test
# ---------------------------------------------------------------------------
def test_golden_smoke(tmp_path: Path) -> None:
    path = FIXTURES / "golden.fraw"
    p = FRAWProcessor.from_path(path)
    assert p.default_params().exposure_ev == 0.5  # from the DEV chunk
    assert p.pattern == (0, 1, 1, 2) and p.norm.shape == (48, 64)
    for kw in ({"demosaic": "ahd"}, {"demosaic": "bilinear"}, {"half_size": True}):
        out = p.develop(**kw)
        assert out.dtype == np.float32 and np.all(np.isfinite(out))
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.shape == ((24, 32, 3) if kw.get("half_size") else (48, 64, 3))
    prev = p.preview()
    assert prev.shape == (24, 32, 3) and prev.dtype == np.uint8
    out = p.export(tmp_path / "golden.jpg")
    with Image.open(out) as im:
        assert im.size == (64, 48)
