"""Tests for fraw.ingest (camera RAW -> .fraw).

Unit tests use a fake object mimicking the rawpy.RawPy interface and always run
(rawpy is only needed for the few tests marked with importorskip).
Integration tests run on real RAW files found in tests/fixtures_ingest/raw/ and
in $FRAW_TEST_RAW_DIR; they are skipped if there are none or rawpy is missing.
"""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import datetime as dt
from dataclasses import replace

import numpy as np
import pytest

from fraw import FrawFile, FrawSet
from fraw import ingest as ing
from fraw.ingest import (
    SUPPORTED_EXTENSIONS,
    black_levels_for_pattern,
    effective_bits_for,
    even_geometry,
    flip_to_orientation,
    from_rawpy,
    is_valid_matrix,
    map_cfa_pattern,
    normalize_wb,
    pattern_full_from_visible,
    rgb_cam_from_cam_xyz,
)


HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "fixtures_ingest" / "raw"

RGGB_VIS = [[0, 1], [3, 2]]  # LibRaw indices, cdesc "RGBG": R G / G2 B
SONY_XYZ = [[0.5991, -0.1456, -0.0455], [-0.4764, 1.2135, 0.2980], [-0.0707, 0.1425, 0.6701]]


# ---------------------------------------------------------------------------
# Fake rawpy.RawPy
# ---------------------------------------------------------------------------
class FakeRaw:
    """Minimal stand-in for rawpy.RawPy with LibRaw's coordinate semantics:
    the CFA ``vis_pattern`` is relative to the VISIBLE area and ``raw_color``
    takes FULL-image coordinates (like rawpy 0.28)."""

    def __init__(
        self,
        *,
        h=10,
        w=12,
        top=0,
        left=0,
        vis_h=None,
        vis_w=None,
        vis_pattern=RGGB_VIS,
        desc=b"RGBG",
        black=(64, 64, 64, 64),
        white=4095,
        flip=0,
        cam_wb=(2048.0, 1024.0, 1536.0, 1024.0),
        day_wb=(2.2, 0.95, 1.1, 0.0),
        xyz=SONY_XYZ,
        cm=None,
        num_colors=3,
        raw_type="Flat",
        pattern_override=None,
        thumb=None,
        other=None,
        lens_model="FE 50mm F1.8",
        image=None,
    ):
        rng = np.random.default_rng(1234)
        self.raw_image = (
            image if image is not None else rng.integers(0, white + 1, size=(h, w), dtype=np.uint16)
        )
        h, w = self.raw_image.shape[:2]
        self._top, self._left = top, left
        self._vis = vis_pattern
        self.sizes = SimpleNamespace(
            raw_height=h,
            raw_width=w,
            top_margin=top,
            left_margin=left,
            height=vis_h if vis_h is not None else h - top,
            width=vis_w if vis_w is not None else w - left,
            flip=flip,
        )
        self.color_desc = desc
        self.black_level_per_channel = list(black)
        self.white_level = white
        self.camera_whitebalance = list(cam_wb)
        self.daylight_whitebalance = list(day_wb)
        m = np.zeros((4, 3), np.float32)
        m[:3] = np.asarray(xyz, np.float32)
        self.rgb_xyz_matrix = m
        c = np.zeros((3, 4), np.float32)
        if cm is not None:
            c[:, :3] = cm
        self.color_matrix = c
        self.num_colors = num_colors
        self.raw_type = SimpleNamespace(name=raw_type)
        self._pattern_override = pattern_override
        self._thumb = thumb
        self.other = other or SimpleNamespace(
            iso_speed=200.0,
            shutter_speed=0.0062500000931322575,
            aperture=2.799999952316284,
            focal_length=50.0,
            timestamp=dt.datetime(2026, 11, 7, 19, 9, 0),
            shot_order=0,
            artist="",
        )
        self.lens = SimpleNamespace(model=lens_model, make="", min_focal=0.0, max_focal=0.0)
        self.postprocess_calls = []

    def raw_color(self, row, col):
        return self._vis[(row - self._top) % 2][(col - self._left) % 2]

    @property
    def raw_pattern(self):
        if self._pattern_override is not None:
            return np.asarray(self._pattern_override, np.uint8)
        return np.array([[self.raw_color(y, x) for x in range(2)] for y in range(2)], np.uint8)

    def extract_thumb(self):
        if self._thumb is None:
            rawpy = pytest.importorskip("rawpy")
            raise rawpy.LibRawNoThumbnailError()
        kind, data = self._thumb
        return SimpleNamespace(format=SimpleNamespace(name=kind), data=data)

    def postprocess(self, **kw):
        self.postprocess_calls.append(kw)
        h, w = self.raw_image.shape
        return np.full((h // 2, w // 2, 3), 128, np.uint8)


def jpeg_bytes(w, h, color=(200, 100, 50), exif=None):
    from PIL import Image

    buf = io.BytesIO()
    im = Image.new("RGB", (w, h), color)
    kw = {}
    if exif:
        ex = Image.Exif()
        for k, v in exif.items():
            ex[k] = v
        kw["exif"] = ex.tobytes()
    im.save(buf, format="JPEG", quality=90, **kw)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestPattern:
    def test_even_margins_unchanged(self):
        assert pattern_full_from_visible(RGGB_VIS, 0, 0) == RGGB_VIS
        assert pattern_full_from_visible(RGGB_VIS, 36, 264) == RGGB_VIS

    def test_odd_top_swaps_rows(self):
        assert pattern_full_from_visible(RGGB_VIS, 1, 0) == [[3, 2], [0, 1]]
        assert pattern_full_from_visible(RGGB_VIS, 13, 2) == [[3, 2], [0, 1]]

    def test_odd_left_swaps_cols(self):
        assert pattern_full_from_visible(RGGB_VIS, 0, 1) == [[1, 0], [2, 3]]

    def test_both_odd(self):
        assert pattern_full_from_visible(RGGB_VIS, 3, 5) == [[2, 3], [1, 0]]

    def test_shift_is_consistent_with_visible_semantics(self):
        # For every pixel of the full mosaic, the full-origin pattern must give the
        # colour LibRaw assigns to that pixel in visible coordinates.
        for top in range(3):
            for left in range(3):
                full = pattern_full_from_visible(RGGB_VIS, top, left)
                for y in range(6):
                    for x in range(6):
                        assert full[y % 2][x % 2] == RGGB_VIS[(y - top) % 2][(x - left) % 2]

    @pytest.mark.parametrize(
        "idx, expected",
        [
            ([[0, 1], [3, 2]], (0, 1, 1, 2)),  # RGGB
            ([[2, 3], [1, 0]], (2, 1, 1, 0)),  # BGGR
            ([[1, 0], [2, 3]], (1, 0, 2, 1)),  # GRBG
            ([[3, 2], [0, 1]], (1, 2, 0, 1)),  # GBRG
            ([[0, 1], [1, 2]], (0, 1, 1, 2)),  # no separate G2 index
        ],
    )
    def test_map_cfa(self, idx, expected):
        assert map_cfa_pattern(idx, b"RGBG") == expected

    def test_map_cfa_str_and_nul(self):
        assert map_cfa_pattern([[0, 1], [3, 2]], "RGBG\x00") == (0, 1, 1, 2)

    @pytest.mark.parametrize(
        "idx, desc",
        [
            ([[0, 1], [3, 2]], b"GMCY"),  # CMYG sensor
            ([[0, 1], [3, 2]], b"RGBE"),  # emerald 4-colour
            ([[0, 0], [1, 2]], b"RGBG"),  # two reds: not Bayer
            ([[0, 1], [5, 2]], b"RGBG"),  # index out of range
        ],
    )
    def test_map_cfa_rejects(self, idx, desc):
        with pytest.raises(ValueError):
            map_cfa_pattern(idx, desc)

    def test_black_per_position(self):
        assert black_levels_for_pattern([[0, 1], [3, 2]], [500, 510, 520, 530]) == (500, 510, 530, 520)
        assert black_levels_for_pattern([[2, 3], [1, 0]], [500, 510, 520, 530]) == (520, 530, 510, 500)


@pytest.mark.parametrize("flip, orient", [(0, 1), (3, 3), (5, 8), (6, 6), (1, 2), (2, 4), (4, 5), (7, 7), (99, 1), (-1, 1)])
def test_flip_to_orientation(flip, orient):
    assert flip_to_orientation(flip) == orient


@pytest.mark.parametrize(
    "white, bits",
    [(4095, 12), (16383, 14), (15360, 14), (1023, 10), (255, 10), (0, 10), (65535, 16), (4096, 13), (16000, 14)],
)
def test_effective_bits(white, bits):
    assert effective_bits_for(white) == bits


class TestWB:
    def test_normalize_by_g(self):
        assert normalize_wb([2048, 1024, 1536, 1024]) == (2.0, 1.0, 1.5, 1.0)

    def test_g2_zero_uses_g(self):
        assert normalize_wb([2.2, 1.1, 1.65, 0.0]) == pytest.approx((2.0, 1.0, 1.5, 1.0))

    def test_g2_distinct_kept(self):
        assert normalize_wb([2.0, 1.0, 1.5, 1.02]) == pytest.approx((2.0, 1.0, 1.5, 1.02))

    @pytest.mark.parametrize("bad", [None, [0, 0, 0, 0], [1, 0, 1, 1], [float("nan"), 1, 1, 1], [1, 1], [-1, 1, 1, 1]])
    def test_invalid(self, bad):
        assert normalize_wb(bad) is None


class TestMatrices:
    def test_validity(self):
        assert is_valid_matrix(np.eye(3))
        assert not is_valid_matrix(np.zeros((3, 3)))
        assert not is_valid_matrix([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert not is_valid_matrix([[1, 1, 1], [1, 1, 1], [0, 0, 1]])  # singular
        assert not is_valid_matrix(np.eye(4))
        assert not is_valid_matrix([[2, 0, 0], [0, 1, 0], [0, 0, 1]], rows_sum_to_one=True)
        assert is_valid_matrix([[2, -1, 0], [0, 1, 0], [0, 0, 1]], rows_sum_to_one=True)

    def test_srgb_camera_gives_identity(self):
        # A camera whose native space is sRGB: cam_xyz = inverse(xyz_rgb).
        cam_xyz = np.linalg.inv(ing._XYZ_RGB)
        np.testing.assert_allclose(rgb_cam_from_cam_xyz(cam_xyz), np.eye(3), atol=1e-9)

    def test_real_matrix_rows_sum_to_one(self):
        m = rgb_cam_from_cam_xyz(SONY_XYZ)
        assert m is not None
        np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-9)
        assert m[0, 0] > 1 and m[1, 1] > 1 and m[2, 2] > 1  # typical saturation boost

    def test_unusable(self):
        assert rgb_cam_from_cam_xyz(np.zeros((3, 3))) is None
        assert rgb_cam_from_cam_xyz(np.zeros((4, 3))) is None


class TestGeometry:
    def test_even_full(self):
        assert even_geometry(6048, 4024, 0, 0, 6048, 4024) == (6048, 4024, (0, 0, 0, 0))

    def test_crop(self):
        assert even_geometry(6288, 4056, 264, 36, 6024, 4020) == (6288, 4056, (264, 36, 6024, 4020))

    def test_odd_trim_and_clip(self):
        # 101x51 -> 100x50, the visible area touching the dropped column/row is clipped.
        assert even_geometry(101, 51, 1, 1, 100, 50) == (100, 50, (1, 1, 99, 49))
        assert even_geometry(101, 51, 0, 0, 101, 51) == (100, 50, (0, 0, 0, 0))

    def test_outside(self):
        with pytest.raises(ValueError):
            even_geometry(10, 10, 12, 0, 4, 4)
        with pytest.raises(ValueError):
            even_geometry(1, 10, 0, 0, 1, 10)


# ---------------------------------------------------------------------------
# from_rawpy with the fake
# ---------------------------------------------------------------------------
class TestFromFake:
    def test_basic(self):
        f = from_rawpy(FakeRaw(), None, thumbnail=False)
        assert f.head.cfa_pattern == (0, 1, 1, 2)
        assert f.head.black_level == (64, 64, 64, 64)
        assert f.head.white_level == 4095 and f.head.effective_bits == 12
        assert (f.head.crop_width, f.head.crop_height) == (0, 0)
        assert f.color.as_shot_wb == (2.0, 1.0, 1.5, 1.0)
        assert f.color.illuminant == 21 and f.color.flags == 3
        assert f.thumbnail is None
        assert f.meta["software"] == "CTD Photo Studio 0.2.0"

    @pytest.mark.parametrize("top, left", [(0, 0), (1, 0), (0, 1), (1, 1), (3, 5), (2, 7)])
    def test_odd_margins_cfa_matches_raw_color(self, top, left):
        fake = FakeRaw(h=20, w=24, top=top, left=left, vis_h=14, vis_w=16, black=(500, 510, 520, 530))
        f = from_rawpy(fake, None, thumbnail=False)
        desc = "RGBG"
        for y in range(4):
            for x in range(4):
                want = {"R": 0, "G": 1, "B": 2}[desc[fake.raw_color(y, x)]]
                assert f.head.cfa_color_at(y, x) == want, (y, x)
                assert f.head.black_level[f.head.cfa_index_at(y, x)] == [500, 510, 520, 530][fake.raw_color(y, x)]
        assert (f.head.crop_left, f.head.crop_top, f.head.crop_width, f.head.crop_height) == (left, top, 16, 14)

    def test_odd_dimensions_trimmed(self):
        fake = FakeRaw(h=11, w=13, top=1, left=1)
        f = from_rawpy(fake, None, thumbnail=False)
        assert (f.head.width, f.head.height) == (12, 10)
        assert np.array_equal(f.raw, fake.raw_image[:10, :12])
        # visible area 12x10 starting at (1,1) is clipped to the trimmed mosaic
        assert (f.head.crop_left, f.head.crop_top, f.head.crop_width, f.head.crop_height) == (1, 1, 11, 9)
        assert f.head.cfa_pattern == (2, 1, 1, 0)  # RGGB visible at (1,1) -> BGGR at full (0,0)

    def test_mosaic_is_a_copy(self):
        fake = FakeRaw()
        f = from_rawpy(fake, None, thumbnail=False)
        fake.raw_image[:] = 0
        assert f.raw.any()

    def test_orientation(self):
        assert from_rawpy(FakeRaw(flip=6), None, thumbnail=False).head.orientation == 6
        assert from_rawpy(FakeRaw(flip=5), None, thumbnail=False).head.orientation == 8

    def test_roundtrip_byte_identical(self, tmp_path):
        fake = FakeRaw(h=8, w=10, top=1, left=2, flip=3)
        f = from_rawpy(fake, None, thumbnail=False)
        p = tmp_path / "x.fraw"
        f.write(p)
        g = FrawFile.read(p)
        assert g.raw.tobytes() == np.ascontiguousarray(fake.raw_image).tobytes()
        assert replace(g.head, compression=0) == f.head and g.meta == f.meta
        assert g.color.flags == f.color.flags

    def test_color_fallbacks(self):
        c = from_rawpy(FakeRaw(xyz=np.zeros((3, 3)), cm=None), None, thumbnail=False).color
        assert c.flags == 0 and c.rgb_cam == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert c.cam_xyz == (0.0,) * 9
        cm = [[1.6, -0.5, -0.1], [-0.2, 1.5, -0.3], [0.0, -0.4, 1.4]]
        c = from_rawpy(FakeRaw(xyz=np.zeros((3, 3)), cm=cm), None, thumbnail=False).color
        assert c.flags == 1
        np.testing.assert_allclose(c.rgb_cam, np.asarray(cm, np.float32).reshape(-1), rtol=1e-6)
        c = from_rawpy(FakeRaw(xyz=[[np.nan] * 3] * 3), None, thumbnail=False).color
        assert c.flags == 0

    def test_wb_fallback_to_daylight(self):
        c = from_rawpy(FakeRaw(cam_wb=(0, 0, 0, 0), day_wb=(2.2, 1.1, 1.65, 0)), None, thumbnail=False).color
        assert c.as_shot_wb == pytest.approx((2.0, 1.0, 1.5, 1.0))
        assert c.daylight_wb == pytest.approx((2.0, 1.0, 1.5, 1.0))
        c = from_rawpy(FakeRaw(cam_wb=(0, 0, 0, 0), day_wb=(0, 0, 0, 0)), None, thumbnail=False).color
        assert c.as_shot_wb == (1.0, 1.0, 1.0, 1.0)

    @pytest.mark.parametrize(
        "kw, msg",
        [
            ({"pattern_override": np.zeros((6, 6))}, "X-Trans"),
            ({"num_colors": 4}, "num_colors"),
            ({"raw_type": "Stack"}, "flat"),
            ({"image": np.zeros((4, 4, 3), np.uint16)}, "2-D"),
            ({"desc": b"GMCY"}, "non-RGB"),
        ],
    )
    def test_rejects_non_bayer(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            from_rawpy(FakeRaw(**kw), None, thumbnail=False)

    def test_rejects_black_ge_white(self):
        with pytest.raises(ValueError, match="black level"):
            from_rawpy(FakeRaw(black=(4095, 0, 0, 0), white=4095), None, thumbnail=False)

    def test_meta(self):
        f = from_rawpy(FakeRaw(), None, thumbnail=False)
        m = f.meta
        assert m["iso"] == 200 and isinstance(m["iso"], int)
        assert m["shutter_s"] == 0.00625 and m["aperture"] == 2.8 and m["focal_mm"] == 50.0
        assert m["datetime"] == "2026-11-07T19:09:00"
        assert m["lens"] == "FE 50mm F1.8"
        assert "make" not in m and "artist" not in m and "source_file" not in m

    def test_meta_omits_missing(self):
        other = SimpleNamespace(
            iso_speed=0.0, shutter_speed=0.0, aperture=0.0, focal_length=0.0,
            timestamp=dt.datetime.fromtimestamp(0), shot_order=0, artist="Fabri",
        )
        m = from_rawpy(FakeRaw(other=other, lens_model=""), None, thumbnail=False).meta
        assert set(m) == {"artist", "software"}


class TestThumbnail:
    @pytest.fixture(autouse=True)
    def _pil(self):
        pytest.importorskip("PIL")

    def test_small_jpeg_verbatim(self):
        data = jpeg_bytes(160, 120)
        f = from_rawpy(FakeRaw(thumb=("JPEG", data)), None)
        t = f.thumbnail
        assert (t.format, t.width, t.height) == (0, 160, 120) and t.data == data

    def test_big_jpeg_resized(self):
        from PIL import Image

        f = from_rawpy(FakeRaw(thumb=("JPEG", jpeg_bytes(1600, 1000))), None, thumb_max=512)
        t = f.thumbnail
        assert t.format == 0 and (t.width, t.height) == (512, 320)
        with Image.open(io.BytesIO(t.data)) as im:
            assert im.format == "JPEG" and im.size == (512, 320)

    def test_small_bitmap_rgb8(self):
        arr = np.arange(30 * 20 * 3, dtype=np.uint32).astype(np.uint8).reshape(20, 30, 3)
        t = from_rawpy(FakeRaw(thumb=("BITMAP", arr)), None).thumbnail
        assert (t.format, t.width, t.height) == (1, 30, 20) and t.data == arr.tobytes()

    def test_big_bitmap_to_jpeg(self):
        arr = np.zeros((600, 900, 3), np.uint8)
        t = from_rawpy(FakeRaw(thumb=("BITMAP", arr)), None, thumb_max=300).thumbnail
        assert t.format == 0 and (t.width, t.height) == (300, 200)

    def test_no_thumb_generates(self):
        pytest.importorskip("rawpy")
        fake = FakeRaw(h=40, w=60)
        t = from_rawpy(fake, None, thumb_max=16).thumbnail
        assert t.format == 0 and max(t.width, t.height) == 16
        (kw,) = fake.postprocess_calls
        assert kw["half_size"] is True and kw["use_camera_wb"] is True and kw["user_flip"] == 0

    def test_corrupt_jpeg_regenerates(self):
        pytest.importorskip("rawpy")
        fake = FakeRaw(h=40, w=60, thumb=("JPEG", b"\xff\xd8garbage"))
        t = from_rawpy(fake, None).thumbnail
        assert t.format == 0 and len(fake.postprocess_calls) == 1

    def test_make_model_from_preview_exif(self):
        data = jpeg_bytes(64, 48, exif={0x010F: "Sony", 0x0110: "ILCE-6000"})
        m = from_rawpy(FakeRaw(thumb=("JPEG", data)), None).meta
        assert m["make"] == "Sony" and m["model"] == "ILCE-6000"

    def test_thumb_max_validation(self):
        with pytest.raises(ValueError):
            from_rawpy(FakeRaw(), None, thumb_max=0)


# ---------------------------------------------------------------------------
# Make/Model from file headers
# ---------------------------------------------------------------------------
def _tiff(bo: str, make: bytes, model: bytes) -> bytes:
    hdr = (b"II" if bo == "<" else b"MM") + struct.pack(bo + "HI", 42, 8)
    n = 3
    data_off = 8 + 2 + 12 * n + 4
    make_z, model_z = make + b"\0", model + b"\0"
    entries = struct.pack(bo + "HHI", 0x0100, 4, 1) + struct.pack(bo + "I", 6000)
    entries += struct.pack(bo + "HHII", 0x010F, 2, len(make_z), data_off)
    if len(model_z) <= 4:
        entries += struct.pack(bo + "HHI", 0x0110, 2, len(model_z)) + model_z.ljust(4, b"\0")
    else:
        entries += struct.pack(bo + "HHII", 0x0110, 2, len(model_z), data_off + len(make_z))
    return hdr + struct.pack(bo + "H", n) + entries + b"\0\0\0\0" + make_z + model_z


class TestMakeModel:
    def test_tiff_le(self, tmp_path):
        p = tmp_path / "a.NEF"
        p.write_bytes(_tiff("<", b"NIKON CORPORATION", b"NIKON D3200"))
        assert ing._read_make_model_from_file(str(p)) == {"make": "NIKON CORPORATION", "model": "NIKON D3200"}

    def test_tiff_be_inline_value(self, tmp_path):
        p = tmp_path / "a.PEF"
        p.write_bytes(_tiff(">", b"PENTAX", b"K-1"))  # "K-1\0" fits inline
        assert ing._read_make_model_from_file(str(p)) == {"make": "PENTAX", "model": "K-1"}

    def test_cr3_cmt1(self, tmp_path):
        tiff = _tiff("<", b"Canon", b"Canon EOS M50")
        box = struct.pack(">I", 8 + len(tiff)) + b"CMT1" + tiff
        p = tmp_path / "a.CR3"
        p.write_bytes(struct.pack(">I", 24) + b"ftypcrx " + b"\0" * 12 + box)
        assert ing._read_make_model_from_file(str(p)) == {"make": "Canon", "model": "Canon EOS M50"}

    def test_garbage(self, tmp_path):
        p = tmp_path / "a.ARW"
        p.write_bytes(b"II*\x00\xff\xff\xff\x7f")
        assert ing._read_make_model_from_file(str(p)) == {}
        p.write_bytes(b"FUJIFILMCCD-RAW ")
        assert ing._read_make_model_from_file(str(p)) == {}


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------
def test_supported_extensions():
    # Sólo para filtros de diálogos: la aceptación real es por contenido (fraw.sniff).
    assert {".cr3", ".cr2", ".nef", ".nrw", ".arw", ".sr2", ".dng", ".raf", ".orf", ".rw2", ".pef"} <= SUPPORTED_EXTENSIONS


def test_rejects_processed_image_by_content(tmp_path):
    # Un JPEG se rechaza por su contenido, se llame como se llame.
    for name in ("photo.jpg", "photo.NEF"):
        p = tmp_path / name
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)
        with pytest.raises(ing.NotRawError, match="JPEG"):
            ing.ingest(p)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ing.ingest(tmp_path / "nope.NeF")  # extension check is case-insensitive


def test_import_error_message(tmp_path, monkeypatch):
    p = tmp_path / "x.dng"
    p.write_bytes(b"II*\x00")
    monkeypatch.setitem(sys.modules, "rawpy", None)
    with pytest.raises(ImportError, match=r"pip install fabriraw\[ingest\]"):
        ing.ingest(p)


def test_import_is_lazy():
    code = "import sys, fraw.ingest; print('rawpy' in sys.modules, 'PIL' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_burst_requires_sources(tmp_path):
    with pytest.raises(ValueError):
        ing.ingest_burst([], tmp_path / "b.fset")


# ---------------------------------------------------------------------------
# Integration with real camera RAWs
# ---------------------------------------------------------------------------
def _real_raws() -> list[Path]:
    dirs = [RAW_DIR]
    if os.environ.get("FRAW_TEST_RAW_DIR"):
        dirs.append(Path(os.environ["FRAW_TEST_RAW_DIR"]))
    out: list[Path] = []
    for d in dirs:
        if d.is_dir():
            out += sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return out


REAL = _real_raws()
_REAL_PARAMS = [pytest.param(p, id=p.name) for p in REAL] or [
    pytest.param(None, marks=pytest.mark.skip(reason=f"no RAW files in {RAW_DIR} or $FRAW_TEST_RAW_DIR"))
]
_cache: dict[Path, FrawFile] = {}


@pytest.fixture
def rawpy_mod():
    return pytest.importorskip("rawpy")


def _ingested(p: Path) -> FrawFile:
    if p not in _cache:
        _cache[p] = ing.ingest(p)
    return _cache[p]


@pytest.mark.parametrize("src", _REAL_PARAMS)
class TestRealRaw:
    @pytest.fixture(autouse=True)
    def _needs_rawpy(self):
        pytest.importorskip("rawpy")

    def test_roundtrip_byte_identical(self, src, rawpy_mod, tmp_path):
        dst = tmp_path / "out.fraw"
        f = ing.ingest_to_file(src, dst)
        g = FrawFile.read(dst)
        with rawpy_mod.imread(str(src)) as r:
            full = r.raw_image
            ref = np.ascontiguousarray(full[: full.shape[0] & ~1, : full.shape[1] & ~1])
            s = r.sizes
            assert g.raw.dtype == np.uint16 and g.raw.shape == ref.shape
            assert g.raw.tobytes() == ref.tobytes()
            assert replace(g.head, compression=0) == f.head and g.color == f.color and g.meta == f.meta
            assert g.thumbnail == f.thumbnail
            # geometry & levels
            assert (g.head.width, g.head.height) == (ref.shape[1], ref.shape[0])
            crop = (g.head.crop_left, g.head.crop_top, g.head.crop_width, g.head.crop_height)
            if crop != (0, 0, 0, 0):
                assert crop[:2] == (s.left_margin, s.top_margin)
                assert crop[2] == min(s.width, g.head.width - s.left_margin)
                assert crop[3] == min(s.height, g.head.height - s.top_margin)
            assert g.head.white_level == r.white_level
            assert g.head.orientation == flip_to_orientation(s.flip)
            bl = r.black_level_per_channel
            for y in range(2):
                for x in range(2):
                    assert g.head.black_level[y * 2 + x] == bl[r.raw_color(y, x)]

    def test_cfa_matches_raw_colors(self, src, rawpy_mod):
        f = _ingested(src)
        with rawpy_mod.imread(str(src)) as r:
            desc = r.color_desc.decode()
            colors = r.raw_colors[:6, :6]
            for y in range(6):
                for x in range(6):
                    assert f.head.cfa_color_at(y, x) == {"R": 0, "G": 1, "B": 2}[desc[colors[y, x]]]

    def test_cfa_matches_libraw_render(self, src, rawpy_mod):
        """Independent check: LibRaw's half-size camera-RGB render takes each output
        channel straight from the mosaic sites of that colour, so the mosaic plane
        at each 2x2 position must correlate best with the channel our CFA claims."""
        f = _ingested(src)
        h = f.head
        with rawpy_mod.imread(str(src)) as r:
            out = r.postprocess(
                half_size=True, output_color=rawpy_mod.ColorSpace.raw, gamma=(1, 1), no_auto_bright=True,
                user_wb=[1, 1, 1, 1], output_bps=16, user_flip=0,
            ).astype(np.float64)
            top, left = r.sizes.top_margin, r.sizes.left_margin
        oh, ow = out.shape[:2]
        for dy in range(2):
            for dx in range(2):
                plane = f.raw[top + dy : top + dy + 2 * oh : 2, left + dx : left + dx + 2 * ow : 2].astype(np.float64)
                ph, pw = plane.shape
                a = plane[::3, ::3].ravel()
                corr = [np.corrcoef(a, out[:ph:3, :pw:3, c].ravel())[0, 1] for c in range(3)]
                claimed = h.cfa_color_at(top + dy, left + dx)
                assert int(np.argmax(corr)) == claimed, (dy, dx, corr)
                assert corr[claimed] > 0.99

    def test_green_sites_agree(self, src):
        # The two G sites of each 2x2 block see (nearly) the same signal.
        f = _ingested(src)
        h = f.head
        planes = {i: f.raw[i >> 1 :: 2, i & 1 :: 2].astype(np.float64)[::4, ::4].ravel() for i in range(4)}
        g = [i for i in range(4) if h.cfa_pattern[i] == 1]
        assert len(g) == 2
        assert np.corrcoef(planes[g[0]], planes[g[1]])[0, 1] > 0.98

    def test_rgb_cam_matches_libraw(self, src, rawpy_mod):
        """Our rgb_cam (from rgb_xyz_matrix) equals the matrix LibRaw applies for sRGB output."""
        f = _ingested(src)
        assert f.color.flags & 3 == 3
        kw = dict(half_size=True, gamma=(1, 1), no_auto_bright=True, user_wb=[1, 1, 1, 1], output_bps=16,
                  user_flip=0, highlight_mode=rawpy_mod.HighlightMode.Ignore)
        with rawpy_mod.imread(str(src)) as r:
            a = r.postprocess(output_color=rawpy_mod.ColorSpace.raw, **kw).reshape(-1, 3).astype(np.float64)
        with rawpy_mod.imread(str(src)) as r:
            b = r.postprocess(output_color=rawpy_mod.ColorSpace.sRGB, **kw).reshape(-1, 3).astype(np.float64)
        m = (a.max(1) < 30000) & (a.min(1) > 300) & (b.min(1) > 50) & (b.max(1) < 60000)
        if m.sum() < 1000:
            pytest.skip("not enough unclipped pixels to fit the matrix")
        fit, *_ = np.linalg.lstsq(a[m], b[m], rcond=None)
        np.testing.assert_allclose(fit.T, np.asarray(f.color.rgb_cam).reshape(3, 3), atol=0.01)

    def test_color_and_meta(self, src):
        f = _ingested(src)
        c = f.color
        assert c.as_shot_wb[1] == 1.0 and c.daylight_wb[1] == 1.0
        assert all(v > 0 for v in c.as_shot_wb + c.daylight_wb)
        assert c.illuminant == 21
        np.testing.assert_allclose(np.asarray(c.rgb_cam).reshape(3, 3).sum(axis=1), 1.0, atol=1e-4)
        m = f.meta
        assert m["source_file"] == src.name and m["source_format"] == src.suffix[1:].upper()
        assert m["software"] == "CTD Photo Studio 0.2.0"
        assert m.get("make") and m.get("model")

    def test_thumbnail_decodable(self, src):
        from PIL import Image

        t = _ingested(src).thumbnail
        assert t is not None and max(t.width, t.height) <= 512
        if t.format == 0:
            with Image.open(io.BytesIO(t.data)) as im:
                im.load()
                assert im.size == (t.width, t.height)
        else:
            assert len(t.data) == t.width * t.height * 3


def test_real_burst(tmp_path):
    if not REAL:
        pytest.skip(f"no RAW files in {RAW_DIR} or $FRAW_TEST_RAW_DIR")
    pytest.importorskip("rawpy")
    srcs = (REAL * 2)[:2]
    dst = tmp_path / "burst.fset"
    fs = ing.ingest_burst(srcs, dst, burst_id="test-burst", thumbnail=False)
    back = FrawSet.read(dst)
    assert back.burst_id == "test-burst" and back.created == fs.created
    assert len(back.frames) == 2
    for fr, src in zip(back.frames, srcs):
        assert fr.meta["source_file"] == src.name
        assert fr.thumbnail is None
    assert np.array_equal(back.frames[0].raw, fs.frames[0].raw)
