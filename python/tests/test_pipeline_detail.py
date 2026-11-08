"""Motor Pro — detail (sharpen, NR), effects (vignette, grain), auto tone, export (ICC/EXIF/resize)."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fraw.processor import FRAWProcessor, to_uint8
from test_processor import flat, make_file


def edge_proc() -> FRAWProcessor:
    cam = np.full((64, 64, 3), 0.1)
    cam[:, 32:] = 0.5
    yy, xx = np.mgrid[0:64, 0:64]
    cam[40:, :] = (0.18 + 0.03 * np.sin(xx[40:] * 1.7) * np.cos(yy[40:] * 1.3))[..., None]
    return FRAWProcessor(make_file(cam))


def test_sharpen_increases_acutance() -> None:
    p = edge_proc()
    base = p.develop({}, demosaic="bilinear")[..., 1]
    up = p.develop({"sharpen_amount": 1.5}, demosaic="bilinear")[..., 1]
    mid = p.develop({"sharpen_amount": 0.6}, demosaic="bilinear")[..., 1]
    zero = p.develop({"sharpen_amount": 0.0}, demosaic="bilinear")[..., 1]
    grad = lambda im: np.abs(np.diff(im[5:35], axis=1))[:, 28:36].max(axis=1).mean()  # noqa: E731
    np.testing.assert_array_equal(base, zero)  # defecto = 0 = sin enfoque (igual que DEV v1)
    assert grad(up) > 1.1 * grad(mid) > 1.1 * grad(base)
    # masking spares the low-contrast texture, keeps the edge
    mask = p.develop({"sharpen_amount": 1.5, "sharpen_masking": 1.0}, demosaic="bilinear")[..., 1]
    tex = lambda im: im[44:60, 4:60].std()  # noqa: E731
    assert tex(mask) < tex(up) and grad(mask) > 1.1 * grad(base)


def test_color_nr_reduces_chroma_variance_and_luma_nr() -> None:
    rng = np.random.default_rng(0)
    cam = np.clip(0.25 + rng.normal(0, 0.03, (64, 64, 3)), 0, 1)
    p = FRAWProcessor(make_file(cam))
    chroma = lambda im: (im[..., 0] - im[..., 1]).std() + (im[..., 2] - im[..., 1]).std()  # noqa: E731
    base = p.develop({}, demosaic="bilinear")
    cnr = p.develop({"nr_color": 1.0}, demosaic="bilinear")
    assert chroma(cnr) < 0.6 * chroma(base)
    assert abs(cnr.mean() - base.mean()) < 0.005
    lnr = p.develop({"nr_luminance": 1.0}, demosaic="bilinear")
    lum = lambda im: (im @ np.array([0.2126, 0.7152, 0.0722])).std()  # noqa: E731
    assert lum(lnr) < 0.8 * lum(base)
    detail = p.develop({"nr_luminance": 1.0, "nr_luminance_detail": 1.0}, demosaic="bilinear")
    assert lum(lnr) < lum(detail) < lum(base)


def test_vignette_corners_not_center() -> None:
    p = FRAWProcessor(make_file(flat(64, 96, (0.18, 0.18, 0.18))))
    base = p.develop({}, demosaic="bilinear")
    dark = p.develop({"vignette_amount": -1}, demosaic="bilinear")
    light = p.develop({"vignette_amount": 1}, demosaic="bilinear")
    c = (slice(28, 36), slice(44, 52))
    np.testing.assert_allclose(dark[c], base[c], atol=1e-3)
    assert dark[:4, :4].mean() < 0.6 * base[:4, :4].mean()
    assert light[:4, :4].mean() > base[:4, :4].mean() + 0.2
    # applies to the cropped frame (post-crop vignette)
    cr = p.develop({"vignette_amount": -1, "crop_left": 0.5}, demosaic="bilinear")
    assert cr[32, 0].mean() < cr[32, 24].mean()
    prot = FRAWProcessor(make_file(flat(64, 96, (0.9, 0.9, 0.9))))
    a = prot.develop({"vignette_amount": -1}, demosaic="bilinear")[:4, :4].mean()
    b = prot.develop({"vignette_amount": -1, "vignette_highlights": 1}, demosaic="bilinear")[:4, :4].mean()
    assert b > a + 0.1


def test_grain_deterministic_and_luminance_weighted() -> None:
    f = make_file(flat(64, 64, (0.18, 0.18, 0.18)))
    a = FRAWProcessor(f).develop({"grain_amount": 1}, demosaic="bilinear")
    b = FRAWProcessor(f).develop({"grain_amount": 1}, demosaic="bilinear")
    np.testing.assert_array_equal(a, b)  # same image → same seed → same grain
    assert a.std() > 0.01
    np.testing.assert_allclose(a[..., 0], a[..., 1], atol=1e-6)  # monochromatic
    g = make_file(flat(64, 64, (0.18, 0.18, 0.18)))
    g.raw = g.raw.copy()
    g.raw[0, 0] += 1  # different image → different seed
    c = FRAWProcessor(g).develop({"grain_amount": 1}, demosaic="bilinear")
    assert np.abs(c[8:, 8:] - a[8:, 8:]).mean() > 1e-3
    big = FRAWProcessor(f).develop({"grain_amount": 1, "grain_size": 1}, demosaic="bilinear")
    hf = lambda im: np.abs(np.diff(im[..., 1], axis=1)).mean() / im[..., 1].std()  # noqa: E731
    assert hf(big) < hf(a)  # larger grain → lower frequency
    pv = FRAWProcessor(f)
    np.testing.assert_array_equal(pv.preview({"grain_amount": 0.5}), pv.preview({"grain_amount": 0.5}))


@pytest.mark.parametrize("ev", [-3.0, -1.5, 0.0, 1.5])
def test_auto_tone_targets_median(ev: float) -> None:
    rng = np.random.default_rng(3)
    scene = np.exp(rng.normal(np.log(0.18), 0.7, (96, 96)))  # log-normal scene, median 0.18
    cam = np.clip(scene * 2.0**ev, 0, 1)[..., None] * np.ones(3)
    p = FRAWProcessor(make_file(cam))
    r = p.auto_tone({})
    assert set(r) >= {"exposure_ev", "contrast", "highlights", "shadows", "whites", "blacks"}
    out = p.develop({**r, "gamma_mode": "linear", "contrast": 0.0}, half_size=True)[..., 1]
    med = float(np.median(out))
    assert 0.12 < med < 0.26, (ev, r, med)
    clipped_input = float(np.mean(cam[::2, ::2, 1] >= 0.999))
    assert float(np.mean(out >= 0.999)) <= max(0.005, clipped_input) + 1e-9, r


def test_export_icc_exif_resize_sharpen(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    f = make_file(rng.uniform(0.0, 1.0, size=(48, 64, 3)))
    f.meta = {"make": "Canon", "model": "EOS R5", "lens": "RF24-70mm", "iso": 400, "shutter_s": 0.004,
              "aperture": 2.8, "focal_mm": 50.0, "datetime": "2026-11-07T19:09:00", "artist": "Fabri",
              "copyright": "(c) Fabri"}
    p = FRAWProcessor(f)
    jpg = p.export(tmp_path / "a.jpg", {"demosaic": "bilinear"}, long_edge=32, output_sharpen="standard")
    with Image.open(jpg) as im:
        assert im.size == (32, 24)
        assert im.info.get("icc_profile")
        ex = im.getexif()
        assert ex[271] == "Canon" and ex[272] == "EOS R5" and ex[305] == "CTD Photo Studio"
        assert ex[315] == "Fabri" and ex[33432] == "(c) Fabri"
        sub = ex.get_ifd(0x8769)
        assert sub[34855] == 400 and float(sub[33437]) == pytest.approx(2.8) and sub[42036] == "RF24-70mm"
        assert float(sub[33434]) == pytest.approx(0.004) and float(sub[37386]) == pytest.approx(50.0)
        assert sub[36867] == "2026:11:07 19:09:00"
    plain = p.export(tmp_path / "b.jpg", {"demosaic": "bilinear"}, embed_icc=False, metadata=False)
    with Image.open(plain) as im:
        assert not im.info.get("icc_profile") and 271 not in im.getexif()
    png = p.export(tmp_path / "c.png", {"demosaic": "bilinear"})
    with Image.open(png) as im:
        assert im.info.get("icc_profile")
        np.testing.assert_array_equal(np.asarray(im), to_uint8(p.develop({"demosaic": "bilinear"})))
    t16 = p.export(tmp_path / "d.tif", {"demosaic": "bilinear"}, bits=16)
    data = t16.read_bytes()
    (ifd,) = struct.unpack_from("<I", data, 4)
    (n,) = struct.unpack_from("<H", data, ifd)
    tags = {struct.unpack_from("<H", data, ifd + 2 + 12 * i)[0]: struct.unpack_from("<HHII", data, ifd + 2 + 12 * i)
            for i in range(n)}
    _, typ, cnt, off = tags[34675]
    from fraw.pipeline.icc import srgb_icc

    assert typ == 7 and data[off : off + cnt] == srgb_icc()
    with Image.open(t16) as im:
        assert im.size == (64, 48)
    p16 = p.export(tmp_path / "e.png", {"demosaic": "bilinear"}, bits=16)
    assert b"iCCP" in p16.read_bytes()[:200]
    with Image.open(io.BytesIO(p16.read_bytes())) as im:
        assert im.size == (64, 48)
    # output sharpening raises local contrast after the resize
    s0 = p.export(tmp_path / "f.png", {"demosaic": "bilinear"}, long_edge=40)
    s1 = p.export(tmp_path / "g.png", {"demosaic": "bilinear"}, long_edge=40, output_sharpen="high")
    with Image.open(s0) as a, Image.open(s1) as b:
        ga = np.abs(np.diff(np.asarray(a).astype(float), axis=1)).mean()
        gb = np.abs(np.diff(np.asarray(b).astype(float), axis=1)).mean()
    assert gb > ga
    with pytest.raises(ValueError):
        p.export(tmp_path / "h.jpg", {}, output_sharpen="extreme")
