"""Demosaic tests (synthetic, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest

from fraw import demosaic as dm

PATTERNS = {
    "RGGB": (0, 1, 1, 2),
    "BGGR": (2, 1, 1, 0),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
}
METHODS = ("bilinear", "ahd")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def scene(size: int = 192) -> tuple[np.ndarray, dict[str, tuple[slice, slice]]]:
    """RGB test scene: smooth ramp (top), sharp diagonal/straight edges (middle), flat colour (bottom)."""
    n = size // 3
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    img = np.zeros((size, size, 3))
    # Smooth zone: low-frequency colour ramps.
    img[..., 0] = 0.2 + 0.6 * x / size
    img[..., 1] = 0.3 + 0.4 * (y % n) / n
    img[..., 2] = 0.5 + 0.3 * np.sin(2 * np.pi * x / 64.0)
    # Edge zone: hard diagonal stripes (both diagonals) + a vertical and a horizontal edge.
    zy = slice(n, 2 * n)
    yy, xx = y[zy], x[zy]
    light = np.array([0.85, 0.75, 0.6])
    dark = np.array([0.08, 0.12, 0.18])
    third = size / 3
    wave = np.where(
        xx < third,
        np.sin(2 * np.pi * (xx + yy) / 14.0),  # diagonal stripes
        np.where(xx < 2 * third, np.sin(2 * np.pi * xx / 13.0), np.sin(2 * np.pi * yy / 13.0)),  # vertical | horizontal
    )
    img[zy] = np.where((wave > 0)[..., None], light, dark)
    # Flat coloured patch.
    img[2 * n :] = (0.6, 0.3, 0.2)
    m = 4  # ignore a 4-px margin around each zone
    zones = {
        "smooth": (slice(m, n - m), slice(m, size - m)),
        "edges": (slice(n + m, 2 * n - m), slice(m, size - m)),
        "flat": (slice(2 * n + m, size - m), slice(m, size - m)),
    }
    return img.astype(np.float32), zones


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", PATTERNS)
def test_flat_grey_is_flat(method: str, name: str) -> None:
    cfa = np.full((32, 40), 0.4, dtype=np.float32)
    out = dm.demosaic(cfa, PATTERNS[name], method)
    assert out.shape == (32, 40, 3) and out.dtype == np.float32
    assert np.max(np.abs(out - 0.4)) < 1e-4


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", PATTERNS)
def test_known_pixels_exact(method: str, name: str) -> None:
    pat = PATTERNS[name]
    rng = np.random.default_rng(7)
    cfa = rng.random((30, 26), dtype=np.float32)
    out = dm.demosaic(cfa, pat, method)
    for i in range(4):
        sub = out[i >> 1 :: 2, i & 1 :: 2, pat[i]]
        assert np.array_equal(sub, cfa[i >> 1 :: 2, i & 1 :: 2])


@pytest.mark.parametrize("method", METHODS)
def test_odd_sizes_and_tiny(method: str) -> None:
    rng = np.random.default_rng(1)
    for shape in ((2, 2), (3, 5), (17, 9)):
        cfa = rng.random(shape, dtype=np.float32)
        out = dm.demosaic(cfa, (1, 2, 0, 1), method)
        assert out.shape == shape + (3,)
        assert np.all(np.isfinite(out))


def test_bayer_offset() -> None:
    assert dm.bayer_offset((0, 1, 1, 2)) == (0, 0)
    assert dm.bayer_offset((1, 0, 2, 1)) == (0, 1)
    assert dm.bayer_offset((1, 2, 0, 1)) == (1, 0)
    assert dm.bayer_offset((2, 1, 1, 0)) == (1, 1)
    assert dm.bayer_offset((0, 2, 1, 1)) is None  # not Bayer (greens on a row)
    assert dm.bayer_offset((0, 0, 1, 2)) is None


def test_non_bayer_falls_back_to_bilinear() -> None:
    rng = np.random.default_rng(3)
    cfa = rng.random((16, 16), dtype=np.float32)
    np.testing.assert_array_equal(dm.ahd(cfa, (0, 2, 1, 1)), dm.bilinear(cfa, (0, 2, 1, 1)))


def test_ahd_strips_are_seamless(monkeypatch: pytest.MonkeyPatch) -> None:
    img, _ = scene(96)
    big = np.tile(img, (6, 1, 1))  # 576 rows → several strips
    for pat in PATTERNS.values():
        cfa = dm.mosaic(big, pat)
        multi = dm.ahd(cfa, pat)
        monkeypatch.setattr(dm, "_STRIP", 10_000)
        single = dm.ahd(cfa, pat, threads=1)
        monkeypatch.setattr(dm, "_STRIP", 256)
        np.testing.assert_array_equal(multi, single)


@pytest.mark.parametrize("name", PATTERNS)
def test_quality_ahd_beats_bilinear(name: str) -> None:
    pat = PATTERNS[name]
    img, zones = scene()
    cfa = dm.mosaic(img, pat)
    res = {m: dm.demosaic(cfa, pat, m) for m in METHODS}
    p = {(m, z): psnr(res[m][zones[z]], img[zones[z]]) for m in METHODS for z in zones}
    print(f"\n{name}: " + ", ".join(f"{m}/{z}={v:.2f}dB" for (m, z), v in p.items()))
    assert p[("bilinear", "smooth")] > 30.0
    assert p[("ahd", "smooth")] > 30.0
    assert p[("ahd", "edges")] > p[("bilinear", "edges")] + 5.0
    assert p[("bilinear", "flat")] > 60.0 and p[("ahd", "flat")] > 60.0

    # Zipper / false-colour artefacts in the edge zone: fraction of pixels whose
    # chroma (R−G, B−G) error exceeds 0.1, and the mean |Δ| of the chroma error
    # between horizontally adjacent pixels (alternating "zipper" pattern).
    ez = zones["edges"]
    stats = {}
    for m in METHODS:
        err = res[m][ez].astype(np.float64) - img[ez]
        cerr = np.stack([err[..., 0] - err[..., 1], err[..., 2] - err[..., 1]], -1)
        false_color = float(np.mean(np.max(np.abs(cerr), -1) > 0.1))
        zipper = float(np.mean(np.abs(np.diff(cerr, axis=1))))
        stats[m] = (false_color, zipper)
    print(f"{name}: false-colour px bilinear={stats['bilinear'][0]:.3%} ahd={stats['ahd'][0]:.3%}; "
          f"zipper bilinear={stats['bilinear'][1]:.4f} ahd={stats['ahd'][1]:.4f}")
    assert stats["ahd"][0] < stats["bilinear"][0]
    assert stats["ahd"][1] < stats["bilinear"][1]


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        dm.bilinear(np.zeros((4, 4, 1)), (0, 1, 1, 2))
    with pytest.raises(ValueError):
        dm.ahd(np.zeros((4, 4)), (0, 0, 0, 0))
    with pytest.raises(ValueError):
        dm.demosaic(np.zeros((4, 4)), (0, 1, 1, 2), "vng")
