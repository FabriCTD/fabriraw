"""Shared helpers for the FabriRAW Python tests."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from fraw.format import (
    CHUNK_HEADER_FMT,
    Color,
    FrawFile,
    Head,
    Thumbnail,
    _crc,
    _frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
CORE_BUILD = REPO_ROOT / "core" / "build"


def find_fraw_info() -> Path | None:
    exe = "fraw-info.exe" if sys.platform == "win32" else "fraw-info"
    for sub in ("", "Release", "RelWithDebInfo", "Debug", "MinSizeRel"):
        p = CORE_BUILD / sub / exe
        if p.is_file():
            return p
    return None


@pytest.fixture
def fraw_info() -> Path:
    p = find_fraw_info()
    if p is None:
        pytest.skip(f"fraw-info binary not found under {CORE_BUILD} (build core/ with CMake to enable cross tests)")
    return p


def make_small(with_thumb: bool = True, with_dev: bool = True) -> FrawFile:
    """4x4 file mirroring core/tests/harness.cpp make_small()."""
    head = Head(width=4, height=4, effective_bits=12, white_level=4095, black_level=(64, 64, 64, 64))
    raw = (100 + np.arange(16, dtype=np.int64) * 37).astype(np.uint16).reshape(4, 4)
    return FrawFile(
        head=head,
        raw=raw,
        meta={"make": "Test", "iso": 200},
        color=Color(as_shot_wb=(2.0, 1.0, 1.0, 1.0), flags=1),
        thumbnail=Thumbnail(format=1, width=2, height=1, data=bytes([1, 2, 3, 4, 5, 6])) if with_thumb else None,
        dev={"version": 1, "exposure_ev": -1.25} if with_dev else None,
    )


# ---- byte-level manipulation helpers (mirror core/tests/harness.cpp) ----
def locate_chunks(b: bytes | bytearray, magic_size: int = 8) -> list[tuple[bytes, int, int]]:
    """[(id, header_offset, payload_length), ...] — tolerant of garbage."""
    out = []
    pos = magic_size
    while pos + 16 <= len(b):
        cid, _flags, length = struct.unpack_from(CHUNK_HEADER_FMT, b, pos)
        out.append((cid, pos, length))
        if length > len(b) - pos - 16:
            break
        pos += 20 + length
    return out


def find_chunk(b: bytes | bytearray, cid: bytes) -> tuple[int, int]:
    for c, off, length in locate_chunks(b):
        if c == cid:
            return off, length
    raise KeyError(cid)


def fix_crc(b: bytearray, offset: int) -> None:
    (length,) = struct.unpack_from("<Q", b, offset + 8)
    cid = bytes(b[offset : offset + 4])
    struct.pack_into("<I", b, offset + 16 + length, _crc(cid, bytes(b[offset + 16 : offset + 16 + length])))


def chunk_bytes(b: bytes | bytearray, cid: bytes) -> bytes:
    off, length = find_chunk(b, cid)
    return bytes(b[off : off + 20 + length])


def get_payload(b: bytes | bytearray, cid: bytes) -> bytes:
    off, length = find_chunk(b, cid)
    return bytes(b[off + 16 : off + 16 + length])


def erase_chunk(b: bytearray, cid: bytes) -> None:
    off, length = find_chunk(b, cid)
    del b[off : off + 20 + length]


def insert_before(b: bytearray, cid: bytes, data: bytes) -> None:
    off, _ = find_chunk(b, cid)
    b[off:off] = data


def replace_payload(b: bytearray, cid: bytes, payload: bytes) -> None:
    off, length = find_chunk(b, cid)
    (flags,) = struct.unpack_from("<I", b, off + 4)
    b[off : off + 20 + length] = _frame(cid, flags, payload)


def move_before(b: bytes, cid: bytes, before: bytes) -> bytearray:
    out = bytearray(b)
    data = chunk_bytes(out, cid)
    erase_chunk(out, cid)
    insert_before(out, before, data)
    return out


def patch_head(b: bytearray, field_off: int, value: bytes) -> None:
    off, _ = find_chunk(b, b"HEAD")
    b[off + 16 + field_off : off + 16 + field_off + len(value)] = value
    fix_crc(b, off)
