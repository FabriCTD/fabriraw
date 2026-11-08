"""FabriRAW Python implementation tests (mirror of core/tests)."""

from __future__ import annotations

import hashlib
import json
import random
import re
import struct
import subprocess
import zlib

import numpy as np
import pytest

import fraw
from fraw import ErrorCode, FrawError, FrawFile, FrawSet, update_dev
from fraw.format import (
    COLR_FMT,
    HEAD_FMT,
    MAGIC,
    SET_MAGIC,
    THMB_FMT,
    Color,
    ExtraChunk,
    Head,
    Thumbnail,
    _crc,
    _frame,
    _json_parse,
    make_golden,
)

from conftest import (
    FIXTURES,
    chunk_bytes,
    erase_chunk,
    find_chunk,
    fix_crc,
    get_payload,
    insert_before,
    locate_chunks,
    make_small,
    move_before,
    patch_head,
    replace_payload,
)


def small(z: bool = False) -> bytes:
    return make_small().to_bytes(z)


def expect(code: ErrorCode, data) -> None:
    with pytest.raises(FrawError) as ei:
        FrawFile.from_bytes(data)
    assert ei.value.code is code, str(ei.value)


u16 = lambda v: struct.pack("<H", v)  # noqa: E731
u32 = lambda v: struct.pack("<I", v)  # noqa: E731


# ------------------------------------------------------------------ package / layout
def test_package_exports():
    assert fraw.__version__ == "0.2.0"
    assert fraw.MAGIC == b"FRAW\r\n\x1a\n" == MAGIC
    assert SET_MAGIC == b"FSET\r\n\x1a\n"
    for name in ("FrawFile", "FrawSet", "Head", "Color", "Thumbnail", "FrawError", "ErrorCode", "update_dev"):
        assert hasattr(fraw, name)
    assert [e.name for e in ErrorCode] == [
        "BadMagic", "Truncated", "CorruptChunk", "UnsupportedVersion", "UnsupportedCriticalChunk",
        "MissingChunk", "DuplicateChunk", "BadOrder", "InvalidHeader", "SizeMismatch",
    ]  # fmt: skip


def test_struct_sizes():
    assert struct.calcsize(HEAD_FMT) == 64
    assert struct.calcsize(COLR_FMT) == 112
    assert struct.calcsize(THMB_FMT) == 16


def test_head_pack_offsets():
    h = Head(
        width=0x0A0B0C0E, height=0x11121314, effective_bits=14, white_level=0xFFFF, cfa_pattern=(2, 1, 1, 0),
        black_level=(0x100, 0x101, 0x102, 0x103), version_major=0x0102, version_minor=0x0304, compression=1,
        orientation=6, crop_left=1, crop_top=2, crop_width=3, crop_height=4,
    )  # fmt: skip
    b = h.pack()
    assert len(b) == 64
    assert struct.unpack_from("<HHII", b, 0) == (0x0102, 0x0304, 0x0A0B0C0E, 0x11121314)
    assert struct.unpack_from("<HH", b, 12) == (16, 14)
    assert b[16:24] == bytes([2, 1, 1, 0, 1, 6, 0, 0])
    assert struct.unpack_from("<4I", b, 24) == (0x100, 0x101, 0x102, 0x103)
    assert struct.unpack_from("<6I", b, 40) == (0xFFFF, 1, 2, 3, 4, 0)
    assert Head.unpack(b) == h


def test_color_and_thumb_pack():
    c = Color(
        as_shot_wb=(1, 2, 3, 4), daylight_wb=(10, 11, 12, 13), rgb_cam=tuple(range(20, 29)),
        cam_xyz=tuple(range(40, 49)), illuminant=21, flags=3,
    )  # fmt: skip
    b = c.pack()
    assert len(b) == 112
    assert struct.unpack_from("<f", b, 0)[0] == 1.0
    assert struct.unpack_from("<f", b, 16)[0] == 10.0
    assert struct.unpack_from("<f", b, 32)[0] == 20.0
    assert struct.unpack_from("<f", b, 68)[0] == 40.0
    assert struct.unpack_from("<II", b, 104) == (21, 3)
    t = Thumbnail(format=1, width=8, height=6, data=b"\x00" * 144).pack()
    assert t[:16] == bytes([1, 0, 0, 0]) + u32(8) + u32(6) + u32(144)


def test_cfa_helpers():
    h = make_small().head
    assert [Head.cfa_index_at(y, x) for y, x in ((0, 0), (0, 1), (1, 0), (3, 5))] == [0, 1, 2, 3]
    assert [int(h.cfa_color_at(y, x)) for y, x in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2))] == [0, 1, 1, 2, 0]


# ------------------------------------------------------------------ crc
def test_crc_vector():
    assert zlib.crc32(b"123456789") == 0xCBF43926
    framed = _frame(b"ABCD", 0, b"\x01\x02\x03")
    assert struct.unpack_from("<I", framed, 19)[0] == zlib.crc32(b"ABCD\x01\x02\x03")
    # flags are not covered by the CRC
    assert _frame(b"ABCD", 0xFFFFFFFE, b"\x01\x02\x03")[19:] == framed[19:]
    assert _crc(b"END ", b"") == zlib.crc32(b"END ")


# ------------------------------------------------------------------ json
BAD_JSON = [
    b"", b"[]", b"1", b"{", b'{"a":}', b'{"a":01}', b'{"a":1.}', b'{"a":NaN}', b'{"a":Infinity}',
    b'{"a":1e400}', b'{"a":"\\ud800"}', b'{"a":"\\udc00"}', b'{"a":"\x01"}', b'{"a":1}x',
    b"\xef\xbb\xbf{}", b'{"a":"\xc0\xaf"}', b'{"a":"\xed\xa0\x80"}', b"{'a':1}", b'{"a":tru}', b'{"a":"\\x"}',
]  # fmt: skip


@pytest.mark.parametrize("bad", BAD_JSON)
def test_json_rejects(bad):
    b = bytearray(small())
    replace_payload(b, b"META", bad)
    expect(ErrorCode.CorruptChunk, b)


def test_json_accepts_and_limits():
    for ok in (
        b'{"a":1,"b":[true,false,null,-0.5e3,"x\\u00e9\\ud83d\\ude00"],"c":{}}',
        b" \t\r\n{ } \n",
        '{"k":"café"}'.encode(),
        b'{"tiny":1e-400}',
    ):
        assert isinstance(_json_parse(ok), dict)
    deep_ok = b'{"a":' + b"[" * 255 + b"]" * 255 + b"}"
    deep_bad = b'{"a":' + b"[" * 256 + b"]" * 256 + b"}"
    _json_parse(deep_ok)
    with pytest.raises(ValueError):
        _json_parse(deep_bad)
    _json_parse(b'{"a":' + b"7" * 4300 + b"}")
    with pytest.raises(ValueError):
        _json_parse(b'{"a":' + b"7" * 4301 + b"}")


# ------------------------------------------------------------------ roundtrip
@pytest.mark.parametrize("z", [False, True])
def test_roundtrip(z):
    f = make_small()
    b = f.to_bytes(z)
    assert b.startswith(MAGIC)
    g = FrawFile.from_bytes(b)
    assert g.head.compression == (1 if z else 0)
    assert g.raw.dtype == np.uint16 and g.raw.shape == (4, 4)
    assert np.array_equal(g.raw, f.raw)
    assert g.meta == f.meta and g.dev == f.dev
    assert g.color.pack() == f.color.pack()
    assert g.thumbnail == f.thumbnail
    assert g.to_bytes(z) == b
    assert f.head.compression == 0  # to_bytes does not mutate


def test_canonical_order_and_flags():
    b = small()
    locs = locate_chunks(b)
    assert [c for c, _, _ in locs] == [b"HEAD", b"META", b"COLR", b"THMB", b"DATA", b"DEV ", b"END "]
    assert [struct.unpack_from("<I", b, off + 4)[0] for _, off, _ in locs] == [1, 0, 1, 0, 1, 0, 1]
    assert locs[-1][2] == 0 and locs[-1][1] + 20 == len(b)


def test_large_file_io(tmp_path):
    f = make_small(False, False)
    f.head = Head(width=1024, height=768, effective_bits=12, white_level=4095, black_level=(64,) * 4)
    f.raw = np.random.default_rng(1).integers(64, 4064, size=(768, 1024), dtype=np.uint16)
    p = tmp_path / "large.fraw"
    f.write(p)
    g = FrawFile.read(p)
    assert np.array_equal(g.raw, f.raw) and g.thumbnail is None and g.dev is None


def test_meta_optional():
    b = bytearray(make_small(False, False).to_bytes(False))
    erase_chunk(b, b"META")
    assert FrawFile.from_bytes(b).meta == {}


def test_any_order_before_data():
    b = move_before(small(), b"COLR", b"META")
    assert locate_chunks(b)[1][0] == b"COLR"
    b = move_before(bytes(b), b"THMB", b"COLR")
    g = FrawFile.from_bytes(b)
    assert g.to_bytes(False) == small()


def test_minor_version_and_reserved_ignored():
    b = bytearray(small())
    patch_head(b, 2, u16(7))
    patch_head(b, 23, b"\xaa")
    patch_head(b, 60, u32(0x12345678))
    off, _ = find_chunk(b, b"THMB")
    b[off + 17] = 0x55
    fix_crc(b, off)
    off, _ = find_chunk(b, b"META")
    struct.pack_into("<I", b, off + 4, 0xFFFFFFF0)
    assert FrawFile.from_bytes(b).head.version_minor == 7


def test_writer_validates():
    f = make_small()
    f.raw = f.raw[:, :3]
    with pytest.raises(FrawError) as ei:
        f.to_bytes()
    assert ei.value.code is ErrorCode.SizeMismatch
    f = make_small()
    f.head.white_level = 10
    with pytest.raises(FrawError) as ei:
        f.to_bytes()
    assert ei.value.code is ErrorCode.InvalidHeader
    f = make_small()
    f.raw = f.raw.astype(np.int32)
    with pytest.raises(FrawError) as ei:
        f.to_bytes()
    assert ei.value.code is ErrorCode.InvalidHeader
    f = make_small()
    f.meta = {"x": float("nan")}
    with pytest.raises(FrawError) as ei:
        f.to_bytes()
    assert ei.value.code is ErrorCode.CorruptChunk


# ------------------------------------------------------------------ unknown chunks
def test_unknown_noncritical_preserved():
    b = bytearray(small())
    insert_before(b, b"META", _frame(b"xTRA", 0, b"\x09\x08\x07"))
    insert_before(b, b"DEV ", _frame(b"zAFT", 2, b"\x01"))
    g = FrawFile.from_bytes(b)
    assert g.extra_chunks == [ExtraChunk(b"xTRA", 0, b"\x09\x08\x07", False), ExtraChunk(b"zAFT", 2, b"\x01", True)]
    out = g.to_bytes(False)
    ids = [c for c, _, _ in locate_chunks(out)]
    assert ids == [b"HEAD", b"META", b"COLR", b"THMB", b"xTRA", b"DATA", b"zAFT", b"DEV ", b"END "]
    assert len(FrawFile.from_bytes(out).extra_chunks) == 2


def test_unknown_critical_rejected():
    b = bytearray(small())
    insert_before(b, b"DATA", _frame(b"xCRT", 1, b"\x01\x02"))
    expect(ErrorCode.UnsupportedCriticalChunk, b)
    b = bytearray(small(True))
    insert_before(b, b"END ", _frame(b"yCRT", 1, b""))
    expect(ErrorCode.UnsupportedCriticalChunk, b)
    f = make_small()
    f.extra_chunks.append(ExtraChunk(b"qqqq", 1, b""))
    with pytest.raises(FrawError) as ei:
        f.to_bytes()
    assert ei.value.code is ErrorCode.UnsupportedCriticalChunk


# ------------------------------------------------------------------ each ErrorCode
def test_bad_magic():
    b = bytearray(small())
    b[0] = ord("X")
    expect(ErrorCode.BadMagic, b)
    b = bytearray(small())
    b[4] = 0x0A
    expect(ErrorCode.BadMagic, b)
    expect(ErrorCode.BadMagic, FrawSet([make_small()]).to_bytes(False))
    expect(ErrorCode.BadMagic, b"nope")


def test_truncated():
    b = small()
    expect(ErrorCode.Truncated, b[:-1])
    expect(ErrorCode.Truncated, MAGIC)
    expect(ErrorCode.Truncated, b"")
    c = bytearray(b)
    off, _ = find_chunk(c, b"DATA")
    struct.pack_into("<Q", c, off + 8, 0xFFFFFFFFFFFFFFFF)
    expect(ErrorCode.Truncated, c)
    d = bytearray(b)
    off, _ = find_chunk(d, b"META")
    struct.pack_into("<Q", d, off + 8, 0x7FFFFFFFFFFFFFF0)
    expect(ErrorCode.Truncated, d)
    end_off, _ = find_chunk(b, b"END ")
    expect(ErrorCode.Truncated, b[:end_off])


def test_corrupt_chunk():
    b = bytearray(small())
    off, _ = find_chunk(b, b"DATA")
    b[off + 16 + 3] ^= 1
    expect(ErrorCode.CorruptChunk, b)
    b = bytearray(small())
    b[8] = ord("h")
    expect(ErrorCode.CorruptChunk, b)
    b = bytearray(small())
    b[-1] ^= 0xFF
    expect(ErrorCode.CorruptChunk, b)
    for cid, payload in ((b"META", b"{x"), (b"DEV ", b"[]"), (b"META", b"\xef\xbb\xbf{}")):
        b = bytearray(small())
        replace_payload(b, cid, payload)
        expect(ErrorCode.CorruptChunk, b)
    z = get_payload(small(True), b"DATA")
    for bad in (z[:-1] + bytes([z[-1] ^ 0x5A]), z[:-3], z + b"\x00"):
        b = bytearray(small(True))
        replace_payload(b, b"DATA", bad)
        expect(ErrorCode.CorruptChunk, b)
    for cid, flags in ((b"COLR", 0), (b"META", 1)):
        b = bytearray(small())
        off, _ = find_chunk(b, cid)
        struct.pack_into("<I", b, off + 4, flags)
        expect(ErrorCode.CorruptChunk, b)


def test_unsupported_version():
    for major in (2, 0):
        b = bytearray(small())
        patch_head(b, 0, u16(major))
        expect(ErrorCode.UnsupportedVersion, b)


def test_unsupported_critical_chunk():
    b = bytearray(small())
    insert_before(b, b"COLR", _frame(b"ABCD", 0x81, b""))
    expect(ErrorCode.UnsupportedCriticalChunk, b)


def test_missing_chunk():
    for cid in (b"COLR", b"DATA"):
        b = bytearray(make_small(True, False).to_bytes(False))
        erase_chunk(b, cid)
        expect(ErrorCode.MissingChunk, b)
    expect(ErrorCode.MissingChunk, MAGIC + _frame(b"END ", 1, b""))


def test_duplicate_chunk():
    for cid in (b"META", b"COLR", b"THMB", b"DEV "):
        b = bytearray(small())
        insert_before(b, cid, chunk_bytes(b, cid))
        expect(ErrorCode.DuplicateChunk, b)
    b = bytearray(small())
    insert_before(b, b"DATA", chunk_bytes(b, b"HEAD"))
    expect(ErrorCode.DuplicateChunk, b)


def test_bad_order():
    for cid, before in (
        (b"META", b"END "), (b"THMB", b"DEV "), (b"COLR", b"END "), (b"DEV ", b"DATA"), (b"META", b"HEAD"),
    ):  # fmt: skip
        expect(ErrorCode.BadOrder, move_before(small(), cid, before))
    b = bytearray(small())
    b[8:8] = _frame(b"zzzz", 0, b"\x01")
    expect(ErrorCode.BadOrder, b)
    expect(ErrorCode.BadOrder, small() + b"\x00")
    expect(ErrorCode.BadOrder, small() + _frame(b"META", 0, b"{}"))


@pytest.mark.parametrize(
    "off,value",
    [
        (4, u32(5)), (8, u32(0)), (12, u16(12)), (14, u16(9)), (14, u16(17)), (16, bytes([0, 1, 1, 3])),
        (16, bytes([0, 1, 1, 1])), (20, b"\x02"), (21, b"\x00"), (21, b"\x09"), (22, b"\x01"), (24, u32(4095)),
        (40, u32(70000)), (44, u32(1)),
    ],
)  # fmt: skip
def test_invalid_header(off, value):
    b = bytearray(small())
    patch_head(b, off, value)
    expect(ErrorCode.InvalidHeader, b)


def test_invalid_header_crop_and_thumb():
    b = bytearray(small())
    patch_head(b, 44, u32(2))
    patch_head(b, 52, u32(3))
    patch_head(b, 56, u32(2))
    expect(ErrorCode.InvalidHeader, b)
    b = bytearray(small())
    patch_head(b, 44, u32(2))
    patch_head(b, 52, u32(2))
    patch_head(b, 56, u32(4))
    assert FrawFile.from_bytes(b).head.crop_width == 2
    b = bytearray(small())
    tp = bytearray(get_payload(b, b"THMB"))
    tp[0] = 7
    replace_payload(b, b"THMB", bytes(tp))
    expect(ErrorCode.InvalidHeader, b)


def test_size_mismatch():
    def with_payload(cid, payload, z=False):
        b = bytearray(small(z))
        replace_payload(b, cid, payload)
        return b

    expect(ErrorCode.SizeMismatch, with_payload(b"HEAD", get_payload(small(), b"HEAD")[:-1]))
    expect(ErrorCode.SizeMismatch, with_payload(b"COLR", get_payload(small(), b"COLR") + b"\x00"))
    expect(ErrorCode.SizeMismatch, with_payload(b"DATA", get_payload(small(), b"DATA")[:-2]))
    for n in (30, 34):
        expect(ErrorCode.SizeMismatch, with_payload(b"DATA", zlib.compress(b"\x07" * n, 6), z=True))
    # zlib bomb guard: 32768x32768 claimed, tiny stream
    b = bytearray(small(True))
    patch_head(b, 4, u32(32768))
    patch_head(b, 8, u32(32768))
    replace_payload(b, b"DATA", zlib.compress(b"\x00" * 4096, 9))
    expect(ErrorCode.SizeMismatch, b)
    # width*height*2 overflows u64
    b = bytearray(small())
    patch_head(b, 4, u32(0xFFFFFFFE))
    patch_head(b, 8, u32(0xFFFFFFFE))
    expect(ErrorCode.SizeMismatch, b)
    tp = get_payload(small(), b"THMB")
    expect(ErrorCode.SizeMismatch, with_payload(b"THMB", tp + b"\x01"))
    expect(ErrorCode.SizeMismatch, with_payload(b"THMB", b"\x00" * 10))
    tp3 = bytearray(tp)
    struct.pack_into("<I", tp3, 4, 3)
    expect(ErrorCode.SizeMismatch, with_payload(b"THMB", bytes(tp3)))
    expect(ErrorCode.SizeMismatch, with_payload(b"END ", b"\x00"))


# ------------------------------------------------------------------ truncation + fuzz
@pytest.mark.parametrize("z", [False, True])
def test_truncation_every_prefix(z):
    full = small(z)
    FrawFile.from_bytes(full)
    for n in range(len(full)):
        with pytest.raises(FrawError):
            FrawFile.from_bytes(full[:n])


@pytest.mark.parametrize("reseal", [False, True])
@pytest.mark.parametrize("z", [False, True])
def test_fuzz_random_flips(z, reseal):
    """Random bit flips (fixed seed): always FrawError or a valid read, never another exception."""
    rng = random.Random(0xF4A3 + z * 2 + reseal)
    base = small(z)
    errors = 0
    for _ in range(1500):
        b = bytearray(base)
        for _ in range(rng.randint(1, 4)):
            b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
        if reseal:
            for _cid, off, length in locate_chunks(b):
                if off + 20 + length <= len(b):
                    fix_crc(b, off)
        try:
            FrawFile.from_bytes(bytes(b))
        except FrawError:
            errors += 1
    assert errors > 0


# ------------------------------------------------------------------ update_dev
def test_update_dev_rewrites_only_tail(tmp_path):
    p = tmp_path / "u.fraw"
    make_small().write(p, compress=True)
    before = p.read_bytes()
    dev_off, _ = find_chunk(before, b"DEV ")

    new_dev = {"version": 1, "exposure_ev": 2.0, "wb_mode": "daylight", "note": "más larga"}
    update_dev(p, new_dev)
    after = p.read_bytes()
    assert after[:dev_off] == before[:dev_off]
    assert FrawFile.read(p).dev == new_dev

    update_dev(p, {"version": 1})
    after2 = p.read_bytes()
    assert after2[:dev_off] == before[:dev_off]
    assert len(after2) == dev_off + 20 + len(b'{"version":1}') + 20

    update_dev(p, None)
    assert FrawFile.read(p).dev is None
    update_dev(p, {"version": 1, "contrast": 0.3})
    after3 = p.read_bytes()
    assert after3[:dev_off] == before[:dev_off]
    assert FrawFile.read(p).dev == {"version": 1, "contrast": 0.3}

    with pytest.raises(FrawError) as ei:
        update_dev(p, {"bad": float("inf")})
    assert ei.value.code is ErrorCode.CorruptChunk
    assert p.read_bytes() == after3


def test_update_dev_preserves_unknown_after_dev(tmp_path):
    b = bytearray(small())
    insert_before(b, b"END ", _frame(b"zTAL", 0, b"tail"))
    p = tmp_path / "x.fraw"
    p.write_bytes(bytes(b))
    dev_off, _ = find_chunk(b, b"DEV ")
    update_dev(p, {"version": 1, "saturation": 1.5})
    out = p.read_bytes()
    assert out[:dev_off] == bytes(b[:dev_off])
    g = FrawFile.read(p)
    assert g.dev == {"version": 1, "saturation": 1.5}
    assert g.extra_chunks == [ExtraChunk(b"zTAL", 0, b"tail", True)]


# ------------------------------------------------------------------ fset
def make_set() -> FrawSet:
    f2 = make_small(False, False)
    f2.raw[0, 0] = 4000
    return FrawSet(frames=[make_small(), f2, make_golden()], burst_id='burst-"0001"\n', created="2026-11-07T19:09:00")


@pytest.mark.parametrize("z", [False, True])
def test_fset_roundtrip(z, tmp_path):
    s = make_set()
    b = s.to_bytes(z)
    assert b.startswith(SET_MAGIC)
    d = FrawSet.from_bytes(b)
    assert len(d.frames) == 3 and d.burst_id == s.burst_id and d.created == s.created
    for a, c in zip(s.frames, d.frames):
        assert np.array_equal(a.raw, c.raw) and a.meta == c.meta
    assert d.to_bytes(z) == b
    p = tmp_path / "s.fset"
    s.write(p, z)
    assert len(FrawSet.read(p).frames) == 3
    assert s.seth_json() == (
        '{"version":1,"frame_count":3,"burst_id":"burst-\\"0001\\"\\n","created":"2026-11-07T19:09:00"}'
    )


def seth_file(seth: bytes, frames: int) -> bytes:
    parts = [SET_MAGIC, _frame(b"SETH", 1, seth)]
    parts += [_frame(b"FRAM", 1, small()) for _ in range(frames)]
    parts.append(_frame(b"END ", 1, b""))
    return b"".join(parts)


def expect_set(code: ErrorCode, data: bytes) -> None:
    with pytest.raises(FrawError) as ei:
        FrawSet.from_bytes(data)
    assert ei.value.code is code, str(ei.value)


def test_fset_errors():
    seth = b'{"version":1,"frame_count":1,"burst_id":"b","created":""}'
    seth2 = b'{"version":1,"frame_count":2,"burst_id":"b","created":""}'
    assert len(FrawSet.from_bytes(seth_file(seth2, 2)).frames) == 2
    expect_set(ErrorCode.SizeMismatch, seth_file(b'{"version":1,"frame_count":3,"burst_id":"b","created":""}', 2))
    expect_set(ErrorCode.UnsupportedVersion, seth_file(b'{"version":2,"frame_count":1,"burst_id":"b","created":""}', 1))
    expect_set(ErrorCode.InvalidHeader, seth_file(b'{"version":1,"frame_count":1,"created":""}', 1))
    expect_set(ErrorCode.InvalidHeader, seth_file(b'{"version":1,"frame_count":1.0,"burst_id":"b","created":""}', 1))
    expect_set(ErrorCode.InvalidHeader, seth_file(b'{"version":1,"frame_count":true,"burst_id":"b","created":""}', 1))
    expect_set(ErrorCode.InvalidHeader, seth_file(b'{"version":1,"frame_count":0,"burst_id":"b","created":""}', 0))
    expect_set(ErrorCode.CorruptChunk, seth_file(b"{nope", 1))
    expect_set(ErrorCode.BadMagic, small())
    b = bytearray(make_set().to_bytes(False))
    off, _ = find_chunk(b, b"FRAM")
    b[off + 16 + 9] ^= 0x10
    expect_set(ErrorCode.CorruptChunk, bytes(b))
    expect_set(ErrorCode.BadOrder, SET_MAGIC + _frame(b"FRAM", 1, small()) + _frame(b"SETH", 1, seth) + _frame(b"END ", 1, b""))  # noqa: E501
    dup = bytearray(seth_file(seth, 1))
    insert_before(dup, b"END ", _frame(b"SETH", 1, seth))
    expect_set(ErrorCode.DuplicateChunk, bytes(dup))
    expect_set(ErrorCode.MissingChunk, SET_MAGIC + _frame(b"END ", 1, b""))
    full = seth_file(seth, 1)
    for n in range(len(full)):
        with pytest.raises(FrawError):
            FrawSet.from_bytes(full[:n])
    with pytest.raises(FrawError) as ei:
        FrawSet().to_bytes()
    assert ei.value.code is ErrorCode.InvalidHeader


# ------------------------------------------------------------------ golden fixtures
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_golden_content(f: FrawFile) -> None:
    h = f.head
    assert (h.width, h.height, h.effective_bits, h.white_level) == (64, 48, 14, 16383)
    assert h.cfa_pattern == (0, 1, 1, 2) and h.black_level == (512,) * 4
    assert list(f.meta.items()) == [("make", "FabriCTD"), ("model", "Golden Test"), ("iso", 100)]
    assert f.dev == {"version": 1, "exposure_ev": 0.5}
    assert f.color.pack() == make_golden().color.pack()
    assert f.color.illuminant == 21 and f.color.flags == 1
    assert f.thumbnail == Thumbnail(1, 8, 6, bytes((255, 128, 0)) * 48)
    y, x = np.mgrid[0:48, 0:64]
    assert np.array_equal(f.raw, (512 + (x * 97 + y * 131) % 15000).astype(np.uint16))


def test_golden_generation():
    g = make_golden()
    check_golden_content(g)
    b = g.to_bytes(False)
    check_golden_content(FrawFile.from_bytes(b))
    check_golden_content(FrawFile.from_bytes(g.to_bytes(True)))
    meta_payload = get_payload(b, b"META")
    assert meta_payload == b'{"make":"FabriCTD","model":"Golden Test","iso":100}'
    assert get_payload(b, b"DEV ") == b'{"version":1,"exposure_ev":0.5}'


def test_golden_fixture_committed_matches():
    committed = (FIXTURES / "golden.fraw").read_bytes()
    assert committed == make_golden().to_bytes(False)
    check_golden_content(FrawFile.from_bytes(committed))
    z = FrawFile.read(FIXTURES / "golden_z.fraw")
    assert z.head.compression == 1
    check_golden_content(z)
    assert z.to_bytes(False) == committed


def test_golden_sha_in_readme():
    readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
    sha = _sha((FIXTURES / "golden.fraw").read_bytes())
    assert re.search(rf"\b{sha}\b", readme), "tests/fixtures/README.md must list the golden.fraw SHA-256"


def test_cli_make_fixture(tmp_path):
    import sys

    out = tmp_path / "py.fraw"
    r = subprocess.run(
        [sys.executable, "-m", "fraw.format", "--make-fixture", str(out)], capture_output=True, text=True, check=True
    )
    assert "RuntimeWarning" not in r.stderr
    assert out.read_bytes() == make_golden().to_bytes(False)


# ------------------------------------------------------------------ cross-implementation (needs core/build)
def test_cross_cpp_fixture_sha_equals_python(fraw_info, tmp_path):
    out = tmp_path / "cpp.fraw"
    subprocess.run([str(fraw_info), "--make-fixture", str(out)], check=True, capture_output=True)
    cpp = out.read_bytes()
    py = make_golden().to_bytes(False)
    assert _sha(cpp) == _sha(py) == _sha((FIXTURES / "golden.fraw").read_bytes())


def test_cross_cpp_zlib_fixture_decodes_same(fraw_info, tmp_path):
    out = tmp_path / "cpp_z.fraw"
    subprocess.run([str(fraw_info), "--make-fixture", str(out), "--zlib"], check=True, capture_output=True)
    f = FrawFile.read(out)
    assert f.head.compression == 1
    check_golden_content(f)
    # compressed bytes may differ between zlib builds; only decoded content is required to match
    assert f.to_bytes(False) == make_golden().to_bytes(False)


def test_cross_cpp_reads_python_files(fraw_info, tmp_path):
    files = {
        "py.fraw": make_golden().to_bytes(False),
        "py_z.fraw": make_golden().to_bytes(True),
        "py.fset": make_set().to_bytes(True),
    }
    for name, data in files.items():
        p = tmp_path / name
        p.write_bytes(data)
        r = subprocess.run([str(fraw_info), str(p)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert r.stdout.rstrip().endswith("OK")
        if name != "py.fset":
            assert '{"make":"FabriCTD","model":"Golden Test","iso":100}' in r.stdout
    # and rejects a corrupt one with exit code 1
    bad = bytearray(files["py.fraw"])
    bad[100] ^= 0xFF
    p = tmp_path / "bad.fraw"
    p.write_bytes(bytes(bad))
    r = subprocess.run([str(fraw_info), str(p)], capture_output=True, text=True)
    assert r.returncode == 1 and "CorruptChunk" in r.stdout


def test_json_dump_matches_spec_format():
    # compact separators, no ASCII escaping: UTF-8 bytes on disk
    f = make_small()
    f.meta = {"artist": "Fabricio Ñandú", "iso": 100}
    b = f.to_bytes(False)
    assert get_payload(b, b"META") == json.dumps(f.meta, separators=(",", ":"), ensure_ascii=False).encode()
    assert "Ñandú".encode() in get_payload(b, b"META")
