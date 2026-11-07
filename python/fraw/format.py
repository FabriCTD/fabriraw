"""FabriRAW (.fraw / .frw) reader/writer — pure Python + numpy.

Exact mirror of the C++ core (``core/``). Normative reference: ``docs/SPEC.md``.

Every multi-byte value is little-endian on disk; all binary payloads are
described by ``struct`` formats with the ``<`` prefix (no padding) and their
sizes are asserted with ``struct.calcsize``.

Run ``python -m fraw.format --make-fixture out.fraw [--zlib]`` to generate the
deterministic golden fixture (byte-identical to ``fraw-info --make-fixture``).
"""

from __future__ import annotations

import enum
import json
import math
import os
import struct
import sys
import zlib
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np

__all__ = [
    "MAGIC",
    "SET_MAGIC",
    "CHUNK_HEADER_FMT",
    "HEAD_FMT",
    "COLR_FMT",
    "THMB_FMT",
    "FLAG_CRITICAL",
    "ErrorCode",
    "FrawError",
    "CfaColor",
    "Compression",
    "ThumbFormat",
    "Head",
    "Color",
    "Thumbnail",
    "ExtraChunk",
    "FrawFile",
    "update_dev",
    "make_golden",
]

# ---------------------------------------------------------------------------
# Constants (SPEC §3, §3.1, §4, §6)
# ---------------------------------------------------------------------------
MAGIC = b"FRAW\r\n\x1a\n"
SET_MAGIC = b"FSET\r\n\x1a\n"
assert len(MAGIC) == 8 and len(SET_MAGIC) == 8

CHUNK_HEADER_FMT = "<4sIQ"  # id[4], flags u32, length u64
HEAD_FMT = "<HHIIHH4B4B4I6I"  # see Head.pack for the field order
COLR_FMT = "<4f4f9f9fII"
THMB_FMT = "<B3xIII"
CRC_FMT = "<I"

CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FMT)
HEAD_SIZE = struct.calcsize(HEAD_FMT)
COLR_SIZE = struct.calcsize(COLR_FMT)
THMB_HEADER_SIZE = struct.calcsize(THMB_FMT)
CRC_SIZE = struct.calcsize(CRC_FMT)
assert CHUNK_HEADER_SIZE == 16
assert HEAD_SIZE == 64
assert COLR_SIZE == 112
assert THMB_HEADER_SIZE == 16
assert CRC_SIZE == 4
FRAMING_SIZE = CHUNK_HEADER_SIZE + CRC_SIZE  # 20

FLAG_CRITICAL = 0x00000001

ID_HEAD = b"HEAD"
ID_META = b"META"
ID_COLR = b"COLR"
ID_THMB = b"THMB"
ID_DATA = b"DATA"
ID_DEV = b"DEV "
ID_END = b"END "
ID_SETH = b"SETH"
ID_FRAM = b"FRAM"

KNOWN_IDS = (ID_HEAD, ID_META, ID_COLR, ID_THMB, ID_DATA, ID_DEV, ID_END)
CRITICAL_IDS = (ID_HEAD, ID_COLR, ID_DATA, ID_END)

VERSION_MAJOR = 1
VERSION_MINOR = 0

ZLIB_LEVEL = 6
# Deflate cannot expand more than 1032:1 (zlib-bomb guard, same as C++).
ZLIB_MAX_RATIO = 1032

JSON_MAX_DEPTH = 256
JSON_MAX_INT_DIGITS = 4300

_U64_MAX = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Errors (SPEC §7) — names identical to the C++ fraw::ErrorCode
# ---------------------------------------------------------------------------
class ErrorCode(enum.Enum):
    BadMagic = "BadMagic"
    Truncated = "Truncated"
    CorruptChunk = "CorruptChunk"
    UnsupportedVersion = "UnsupportedVersion"
    UnsupportedCriticalChunk = "UnsupportedCriticalChunk"
    MissingChunk = "MissingChunk"
    DuplicateChunk = "DuplicateChunk"
    BadOrder = "BadOrder"
    InvalidHeader = "InvalidHeader"
    SizeMismatch = "SizeMismatch"


class FrawError(Exception):
    """Format error. ``code`` is an :class:`ErrorCode`."""

    def __init__(self, code: ErrorCode, msg: str):
        super().__init__(f"{code.value}: {msg}")
        self.code = code
        self.msg = msg


class CfaColor(enum.IntEnum):
    R = 0
    G = 1
    B = 2


class Compression(enum.IntEnum):
    NONE = 0
    ZLIB = 1


class ThumbFormat(enum.IntEnum):
    JPEG = 0
    RGB8 = 1


# ---------------------------------------------------------------------------
# Strict JSON (mirrors core/src/json.cpp)
# ---------------------------------------------------------------------------
def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name}")


def _parse_float(s: str) -> float:
    v = float(s)
    if not math.isfinite(v):
        raise ValueError("number overflows binary64")
    return v


def _parse_int(s: str) -> int:
    if len(s.lstrip("-")) > JSON_MAX_INT_DIGITS:
        raise ValueError("integer has too many digits")
    return int(s)


def _check_json_tree(obj: Any) -> None:
    """Depth (root container = 1) <= JSON_MAX_DEPTH and all strings valid UTF-8."""
    stack: list[tuple[Any, int]] = [(obj, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, (dict, list)):
            if depth > JSON_MAX_DEPTH:
                raise ValueError("JSON nesting too deep")
            if isinstance(node, dict):
                for k, v in node.items():
                    k.encode("utf-8")  # raises on lone surrogates
                    stack.append((v, depth + 1))
            else:
                for v in node:
                    stack.append((v, depth + 1))
        elif isinstance(node, str):
            node.encode("utf-8")


def _json_parse(data: bytes | memoryview) -> Any:
    """Parses a strict JSON document. Raises ValueError on any problem."""
    text = bytes(data).decode("utf-8")  # strict: overlongs / surrogates rejected
    try:
        obj = json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            parse_int=_parse_int,
        )
    except RecursionError as e:
        raise ValueError("JSON nesting too deep") from e
    _check_json_tree(obj)
    return obj


def _json_object_from_payload(data: bytes | memoryview, what: str) -> dict:
    try:
        obj = _json_parse(data)
    except (ValueError, UnicodeError) as e:  # JSONDecodeError is a ValueError
        raise FrawError(ErrorCode.CorruptChunk, f"{what} is not a valid JSON object: {e}") from None
    if type(obj) is not dict:
        raise FrawError(ErrorCode.CorruptChunk, f"{what} is not a valid JSON object: root must be an object")
    return obj


def _dump_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _json_object_to_payload(obj: Any, what: str) -> bytes:
    if not isinstance(obj, dict):
        raise FrawError(ErrorCode.CorruptChunk, f"{what} must be a dict")
    try:
        payload = _dump_json(obj).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as e:
        raise FrawError(ErrorCode.CorruptChunk, f"{what} is not serializable as JSON: {e}") from None
    _json_object_from_payload(payload, what)  # same validation as the reader
    return payload


def _json_quote(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Payload dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Head:
    """HEAD chunk, 64 bytes (SPEC §4.1). Reserved fields are not stored."""

    width: int
    height: int
    effective_bits: int
    white_level: int
    cfa_pattern: tuple[int, int, int, int] = (0, 1, 1, 2)
    black_level: tuple[int, int, int, int] = (0, 0, 0, 0)
    version_major: int = VERSION_MAJOR
    version_minor: int = VERSION_MINOR
    bits_per_sample: int = 16
    compression: int = 0
    orientation: int = 1
    sample_layout: int = 0
    crop_left: int = 0
    crop_top: int = 0
    crop_width: int = 0
    crop_height: int = 0

    def pack(self) -> bytes:
        try:
            return struct.pack(
                HEAD_FMT,
                self.version_major,
                self.version_minor,
                self.width,
                self.height,
                self.bits_per_sample,
                self.effective_bits,
                *self.cfa_pattern,
                self.compression,
                self.orientation,
                self.sample_layout,
                0,  # reserved0
                *self.black_level,
                self.white_level,
                self.crop_left,
                self.crop_top,
                self.crop_width,
                self.crop_height,
                0,  # reserved1
            )
        except struct.error as e:
            raise FrawError(ErrorCode.InvalidHeader, f"HEAD field out of range: {e}") from None

    @classmethod
    def unpack(cls, payload: bytes | memoryview) -> "Head":
        if len(payload) != HEAD_SIZE:
            raise FrawError(ErrorCode.SizeMismatch, "HEAD payload must be 64 bytes")
        v = struct.unpack(HEAD_FMT, payload)
        return cls(
            version_major=v[0],
            version_minor=v[1],
            width=v[2],
            height=v[3],
            bits_per_sample=v[4],
            effective_bits=v[5],
            cfa_pattern=tuple(v[6:10]),
            compression=v[10],
            orientation=v[11],
            sample_layout=v[12],
            # v[13] = reserved0 (ignored)
            black_level=tuple(v[14:18]),
            white_level=v[18],
            crop_left=v[19],
            crop_top=v[20],
            crop_width=v[21],
            crop_height=v[22],
            # v[23] = reserved1 (ignored)
        )

    def validate(self) -> None:
        """Raises UnsupportedVersion / InvalidHeader (same order as C++ validate_head)."""

        def bad(msg: str) -> None:
            raise FrawError(ErrorCode.InvalidHeader, msg)

        if self.version_major != VERSION_MAJOR:
            raise FrawError(ErrorCode.UnsupportedVersion, f"unsupported major version {self.version_major}")
        if len(self.cfa_pattern) != 4 or len(self.black_level) != 4:
            bad("cfa_pattern and black_level need 4 entries")
        if self.width < 2 or self.height < 2:
            bad("width and height must be >= 2")
        if (self.width & 1) or (self.height & 1):
            bad("width and height must be even")
        if self.bits_per_sample != 16:
            bad("bits_per_sample must be 16")
        if not 10 <= self.effective_bits <= 16:
            bad("effective_bits must be in [10, 16]")
        if any(v not in (0, 1, 2) for v in self.cfa_pattern):
            bad("cfa_pattern values must be 0, 1 or 2")
        if not {0, 1, 2} <= set(self.cfa_pattern):
            bad("cfa_pattern must contain R, G and B")
        if self.compression not in (0, 1):
            bad("unknown compression")
        if not 1 <= self.orientation <= 8:
            bad("orientation must be in [1, 8]")
        if self.sample_layout != 0:
            bad("unsupported sample_layout")
        if not max(self.black_level) < self.white_level:
            bad("black_level must be < white_level")
        if self.white_level > 65535:
            bad("white_level must be <= 65535")
        if self.crop_left or self.crop_top or self.crop_width or self.crop_height:
            if self.crop_width == 0 or self.crop_height == 0:
                bad("crop area must be non-empty")
            if self.crop_left + self.crop_width > self.width:
                bad("crop exceeds image width")
            if self.crop_top + self.crop_height > self.height:
                bad("crop exceeds image height")
        # Field ranges for packing (u16/u32/u8) — rejected before struct.pack.
        for name, value, bits in (
            ("version_minor", self.version_minor, 16),
            ("width", self.width, 32),
            ("height", self.height, 32),
            ("crop_left", self.crop_left, 32),
            ("crop_top", self.crop_top, 32),
            ("crop_width", self.crop_width, 32),
            ("crop_height", self.crop_height, 32),
        ):
            if not 0 <= value < (1 << bits):
                bad(f"{name} out of range")
        if any(not 0 <= b < (1 << 32) for b in self.black_level):
            bad("black_level out of range")

    @staticmethod
    def cfa_index_at(y: int, x: int) -> int:
        """Index into cfa_pattern / black_level for pixel (y, x)."""
        return ((y & 1) << 1) | (x & 1)

    def cfa_color_at(self, y: int, x: int) -> CfaColor:
        return CfaColor(self.cfa_pattern[self.cfa_index_at(y, x)])


@dataclass
class Color:
    """COLR chunk, 112 bytes (SPEC §4.3). Floats are stored as f32."""

    as_shot_wb: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    daylight_wb: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    rgb_cam: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    cam_xyz: tuple[float, ...] = (0.0,) * 9
    illuminant: int = 0
    flags: int = 0

    def pack(self) -> bytes:
        if len(self.as_shot_wb) != 4 or len(self.daylight_wb) != 4 or len(self.rgb_cam) != 9 or len(self.cam_xyz) != 9:
            raise FrawError(ErrorCode.SizeMismatch, "COLR arrays must have 4, 4, 9 and 9 entries")
        try:
            return struct.pack(
                COLR_FMT, *self.as_shot_wb, *self.daylight_wb, *self.rgb_cam, *self.cam_xyz, self.illuminant, self.flags
            )
        except (struct.error, OverflowError) as e:
            raise FrawError(ErrorCode.InvalidHeader, f"COLR field out of range: {e}") from None

    @classmethod
    def unpack(cls, payload: bytes | memoryview) -> "Color":
        if len(payload) != COLR_SIZE:
            raise FrawError(ErrorCode.SizeMismatch, "COLR payload must be 112 bytes")
        v = struct.unpack(COLR_FMT, payload)
        return cls(
            as_shot_wb=tuple(v[0:4]),
            daylight_wb=tuple(v[4:8]),
            rgb_cam=tuple(v[8:17]),
            cam_xyz=tuple(v[17:26]),
            illuminant=v[26],
            flags=v[27],
        )


@dataclass
class Thumbnail:
    """THMB chunk (SPEC §4.4). ``data_length`` is always ``len(data)``."""

    format: int
    width: int
    height: int
    data: bytes

    def validate(self) -> None:
        if len(self.data) > 0xFFFFFFFF:
            raise FrawError(ErrorCode.SizeMismatch, "thumbnail data too large")
        _validate_thumb(self.format, self.width, self.height, len(self.data), len(self.data))

    def pack(self) -> bytes:
        try:
            header = struct.pack(THMB_FMT, self.format, self.width, self.height, len(self.data))
        except struct.error as e:
            raise FrawError(ErrorCode.InvalidHeader, f"THMB field out of range: {e}") from None
        return header + bytes(self.data)


def _validate_thumb(fmt: int, width: int, height: int, data_length: int, data_size: int) -> None:
    if fmt > 1:
        raise FrawError(ErrorCode.InvalidHeader, "unknown thumbnail format")
    if width == 0 or height == 0:
        raise FrawError(ErrorCode.InvalidHeader, "thumbnail dimensions must be >= 1")
    if data_length != data_size:
        raise FrawError(ErrorCode.SizeMismatch, "THMB data_length mismatch")
    if fmt == ThumbFormat.RGB8 and width * height * 3 != data_size:
        raise FrawError(ErrorCode.SizeMismatch, "RGB8 thumbnail data_length != width*height*3")


@dataclass
class ExtraChunk:
    """Unknown, non-CRITICAL chunk preserved verbatim."""

    id: bytes
    flags: int
    payload: bytes
    after_data: bool = False  # False: between HEAD and DATA; True: after DATA


# ---------------------------------------------------------------------------
# Chunk framing (mirrors core/src/chunk.cpp)
# ---------------------------------------------------------------------------
@dataclass
class _Chunk:
    id: bytes
    flags: int
    offset: int
    payload: memoryview

    @property
    def critical(self) -> bool:
        return bool(self.flags & FLAG_CRITICAL)

    @property
    def total_size(self) -> int:
        return FRAMING_SIZE + len(self.payload)


def _crc(chunk_id: bytes, payload: bytes | memoryview) -> int:
    return zlib.crc32(payload, zlib.crc32(chunk_id)) & 0xFFFFFFFF


def _printable(chunk_id: bytes) -> bool:
    return len(chunk_id) == 4 and all(0x20 <= c <= 0x7E for c in chunk_id)


def _read_chunks(data: bytes | bytearray | memoryview, magic: bytes) -> list[_Chunk]:
    """Strict framing reader. Same checks, same order as C++ read_chunks()."""
    mv = memoryview(data).cast("B")
    n = len(mv)
    cmp = min(n, len(magic))
    if bytes(mv[:cmp]) != magic[:cmp]:
        raise FrawError(ErrorCode.BadMagic, "bad magic")
    if n < len(magic):
        raise FrawError(ErrorCode.Truncated, "file shorter than magic")
    pos = len(magic)
    out: list[_Chunk] = []
    while True:
        remaining = n - pos
        if remaining < CHUNK_HEADER_SIZE:
            raise FrawError(
                ErrorCode.Truncated, "end of data before END chunk" if remaining == 0 else "truncated chunk header"
            )
        cid, flags, length = struct.unpack_from(CHUNK_HEADER_FMT, mv, pos)
        avail = remaining - CHUNK_HEADER_SIZE
        if length > avail or avail - length < CRC_SIZE:
            raise FrawError(ErrorCode.Truncated, f"chunk {cid!r} extends past end of data")
        payload_off = pos + CHUNK_HEADER_SIZE
        payload = mv[payload_off : payload_off + length]
        (stored,) = struct.unpack_from(CRC_FMT, mv, payload_off + length)
        out.append(_Chunk(cid, flags, pos, payload))
        if stored != _crc(cid, payload):
            raise FrawError(ErrorCode.CorruptChunk, f"CRC mismatch in chunk {cid!r}")
        if not _printable(cid):
            raise FrawError(ErrorCode.CorruptChunk, "non-printable chunk id")
        pos = payload_off + length + CRC_SIZE
        if cid == ID_END:
            if length != 0:
                raise FrawError(ErrorCode.SizeMismatch, "END chunk with non-zero length")
            if pos != n:
                raise FrawError(ErrorCode.BadOrder, "data after END chunk")
            return out


def _frame(chunk_id: bytes, flags: int, payload: bytes | memoryview) -> bytes:
    return b"".join(
        (
            struct.pack(CHUNK_HEADER_FMT, chunk_id, flags, len(payload)),
            bytes(payload),
            struct.pack(CRC_FMT, _crc(chunk_id, payload)),
        )
    )


def _canonical_flags(chunk_id: bytes) -> int:
    return FLAG_CRITICAL if chunk_id in CRITICAL_IDS else 0


# ---------------------------------------------------------------------------
# DATA codec
# ---------------------------------------------------------------------------
def _raw_byte_count(head: Head) -> int:
    px = head.width * head.height
    if px > _U64_MAX // 2:
        raise FrawError(ErrorCode.SizeMismatch, "image too large")
    if px * 2 > sys.maxsize:
        raise FrawError(ErrorCode.SizeMismatch, "image too large for this platform")
    return px * 2


def _inflate_exact(payload: memoryview, expected: int) -> bytes:
    d = zlib.decompressobj()
    try:
        out = d.decompress(payload, expected + 1)
    except zlib.error as e:
        raise FrawError(ErrorCode.CorruptChunk, f"invalid zlib stream: {e}") from None
    if len(out) > expected:
        raise FrawError(ErrorCode.SizeMismatch, "zlib stream decompresses to more bytes than width*height*2")
    if not d.eof:
        raise FrawError(ErrorCode.CorruptChunk, "truncated zlib stream")
    if d.unused_data:
        raise FrawError(ErrorCode.CorruptChunk, "trailing bytes after zlib stream")
    if len(out) != expected:
        raise FrawError(ErrorCode.SizeMismatch, "zlib stream decompresses to fewer bytes than width*height*2")
    return out


def _decode_raw(head: Head, payload: memoryview) -> np.ndarray:
    expected = _raw_byte_count(head)
    if head.compression == Compression.ZLIB:
        if expected // ZLIB_MAX_RATIO > len(payload):
            raise FrawError(ErrorCode.SizeMismatch, "DATA zlib stream too small for width*height*2 (bomb guard)")
        buf: bytes | memoryview = _inflate_exact(payload, expected)
    elif len(payload) != expected:
        raise FrawError(ErrorCode.SizeMismatch, "DATA length != width*height*2")
    else:
        buf = payload
    arr = np.frombuffer(buf, dtype="<u2").astype(np.uint16, copy=True)
    return arr.reshape(head.height, head.width)


# ---------------------------------------------------------------------------
# FrawFile
# ---------------------------------------------------------------------------
@dataclass
class FrawFile:
    head: Head
    raw: np.ndarray  # uint16, shape (height, width)
    meta: dict = field(default_factory=dict)
    color: Color = field(default_factory=Color)
    thumbnail: Thumbnail | None = None
    dev: dict | None = None
    extra_chunks: list[ExtraChunk] = field(default_factory=list)

    # ---------------------------------------------------------------- read
    @classmethod
    def read(cls, path: str | os.PathLike) -> "FrawFile":
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "FrawFile":
        chunks = _read_chunks(data, MAGIC)
        head: Head | None = None
        meta: dict = {}
        color: Color | None = None
        thumb: Thumbnail | None = None
        raw: np.ndarray | None = None
        dev: dict | None = None
        extra: list[ExtraChunk] = []
        phase = 0  # 0 expect HEAD, 1 before DATA, 2 after DATA
        seen: set[bytes] = set()

        for ch in chunks:
            name = ch.id
            if name not in KNOWN_IDS:
                if ch.critical:
                    raise FrawError(ErrorCode.UnsupportedCriticalChunk, f"unknown critical chunk {name!r}")
                if phase == 0:
                    raise FrawError(ErrorCode.BadOrder, "HEAD must be the first chunk")
                extra.append(ExtraChunk(name, ch.flags, bytes(ch.payload), phase == 2))
                continue
            if ch.critical != (name in CRITICAL_IDS):
                raise FrawError(ErrorCode.CorruptChunk, f"wrong CRITICAL flag on chunk {name!r}")
            if name == ID_END:
                if phase == 0:
                    raise FrawError(ErrorCode.MissingChunk, "missing HEAD chunk")
                if ID_COLR not in seen:
                    raise FrawError(ErrorCode.MissingChunk, "missing COLR chunk")
                if ID_DATA not in seen:
                    raise FrawError(ErrorCode.MissingChunk, "missing DATA chunk")
                break
            if name in seen:
                raise FrawError(ErrorCode.DuplicateChunk, f"duplicate chunk {name!r}")
            seen.add(name)
            if phase == 0 and name != ID_HEAD:
                raise FrawError(ErrorCode.BadOrder, "HEAD must be the first chunk")

            if name == ID_HEAD:
                head = Head.unpack(ch.payload)
                head.validate()
                phase = 1
            elif name in (ID_META, ID_COLR, ID_THMB):
                if phase == 2:
                    raise FrawError(ErrorCode.BadOrder, f"chunk {name!r} after DATA")
                if name == ID_META:
                    meta = _json_object_from_payload(ch.payload, "META")
                elif name == ID_COLR:
                    color = Color.unpack(ch.payload)
                else:
                    if len(ch.payload) < THMB_HEADER_SIZE:
                        raise FrawError(ErrorCode.SizeMismatch, "THMB payload shorter than 16 bytes")
                    fmt, tw, th, tlen = struct.unpack_from(THMB_FMT, ch.payload, 0)
                    _validate_thumb(fmt, tw, th, tlen, len(ch.payload) - THMB_HEADER_SIZE)
                    thumb = Thumbnail(fmt, tw, th, bytes(ch.payload[THMB_HEADER_SIZE:]))
            elif name == ID_DATA:
                assert head is not None
                raw = _decode_raw(head, ch.payload)
                phase = 2
            else:  # DEV
                if phase != 2:
                    raise FrawError(ErrorCode.BadOrder, "DEV must come after DATA")
                dev = _json_object_from_payload(ch.payload, "DEV")

        assert head is not None and color is not None and raw is not None
        return cls(head=head, raw=raw, meta=meta, color=color, thumbnail=thumb, dev=dev, extra_chunks=extra)

    # ---------------------------------------------------------------- write
    def validate(self) -> None:
        self.head.validate()
        _raw_byte_count(self.head)
        raw = self.raw
        if not isinstance(raw, np.ndarray) or raw.dtype.kind != "u" or raw.dtype.itemsize != 2:
            raise FrawError(ErrorCode.InvalidHeader, "raw must be a uint16 numpy array")
        if raw.shape != (self.head.height, self.head.width):
            raise FrawError(ErrorCode.SizeMismatch, "raw size != width*height")
        _json_object_to_payload(self.meta, "META")
        if self.dev is not None:
            _json_object_to_payload(self.dev, "DEV")
        if self.thumbnail is not None:
            self.thumbnail.validate()
        for x in self.extra_chunks:
            if not isinstance(x.id, (bytes, bytearray)) or not _printable(bytes(x.id)):
                raise FrawError(ErrorCode.CorruptChunk, "non-printable unknown chunk id")
            if bytes(x.id) in KNOWN_IDS:
                raise FrawError(ErrorCode.DuplicateChunk, "unknown chunk uses a reserved id")
            if x.flags & FLAG_CRITICAL:
                raise FrawError(ErrorCode.UnsupportedCriticalChunk, "cannot write unknown CRITICAL chunk")

    def to_bytes(self, compress: bool = True) -> bytes:
        self.validate()
        head = replace(self.head, compression=int(Compression.ZLIB if compress else Compression.NONE))
        parts = [MAGIC]
        parts.append(_frame(ID_HEAD, _canonical_flags(ID_HEAD), head.pack()))
        parts.append(_frame(ID_META, _canonical_flags(ID_META), _json_object_to_payload(self.meta, "META")))
        parts.append(_frame(ID_COLR, _canonical_flags(ID_COLR), self.color.pack()))
        if self.thumbnail is not None:
            parts.append(_frame(ID_THMB, _canonical_flags(ID_THMB), self.thumbnail.pack()))
        for x in self.extra_chunks:
            if not x.after_data:
                parts.append(_frame(bytes(x.id), x.flags, x.payload))
        raw = np.ascontiguousarray(self.raw, dtype="<u2").tobytes()
        if compress:
            raw = zlib.compress(raw, ZLIB_LEVEL)
        parts.append(_frame(ID_DATA, _canonical_flags(ID_DATA), raw))
        for x in self.extra_chunks:
            if x.after_data:
                parts.append(_frame(bytes(x.id), x.flags, x.payload))
        if self.dev is not None:
            parts.append(_frame(ID_DEV, _canonical_flags(ID_DEV), _json_object_to_payload(self.dev, "DEV")))
        parts.append(_frame(ID_END, _canonical_flags(ID_END), b""))
        return b"".join(parts)

    def write(self, path: str | os.PathLike, compress: bool = True) -> None:
        data = self.to_bytes(compress)
        with open(path, "wb") as f:
            f.write(data)


def update_dev(path: str | os.PathLike, dev: dict | None) -> None:
    """Non-destructive edit: rewrites ONLY the tail (DEV + END) in place.

    Every byte before the old ``DEV `` chunk (or ``END `` if there was none)
    is left untouched; DATA is never rewritten. ``dev=None`` removes DEV.
    Unknown chunks that followed the old DEV are preserved verbatim.
    """
    dev_payload = _json_object_to_payload(dev, "DEV") if dev is not None else None
    with open(path, "rb") as f:
        data = f.read()
    FrawFile.from_bytes(data)  # full validation of the existing file
    chunks = _read_chunks(data, MAGIC)
    first_after = next((i for i, c in enumerate(chunks) if c.id in (ID_DEV, ID_END)), None)
    if first_after is None:  # unreachable: _read_chunks guarantees END
        raise FrawError(ErrorCode.MissingChunk, "missing END chunk")
    cut = chunks[first_after].offset
    tail = [
        bytes(data[c.offset : c.offset + c.total_size])
        for c in chunks[first_after:]
        if c.id not in (ID_DEV, ID_END)
    ]
    if dev_payload is not None:
        tail.append(_frame(ID_DEV, _canonical_flags(ID_DEV), dev_payload))
    tail.append(_frame(ID_END, _canonical_flags(ID_END), b""))
    blob = b"".join(tail)
    with open(path, "r+b") as f:
        f.seek(cut)
        f.write(blob)
        f.truncate()


# ---------------------------------------------------------------------------
# Golden fixture (must match core/src/golden.cpp byte for byte)
# ---------------------------------------------------------------------------
GOLDEN_WIDTH = 64
GOLDEN_HEIGHT = 48


def golden_raw() -> np.ndarray:
    y, x = np.mgrid[0:GOLDEN_HEIGHT, 0:GOLDEN_WIDTH].astype(np.int64)
    return (512 + (x * 97 + y * 131) % 15000).astype(np.uint16)


def make_golden() -> FrawFile:
    head = Head(
        width=GOLDEN_WIDTH,
        height=GOLDEN_HEIGHT,
        effective_bits=14,
        white_level=16383,
        cfa_pattern=(0, 1, 1, 2),
        black_level=(512, 512, 512, 512),
    )
    color = Color(
        as_shot_wb=(2.0, 1.0, 1.5, 1.0),
        daylight_wb=(2.1, 1.0, 1.4, 1.0),
        rgb_cam=(1.6, -0.4, -0.2, -0.2, 1.5, -0.3, 0.0, -0.5, 1.5),
        cam_xyz=(0.0,) * 9,
        illuminant=21,
        flags=1,
    )
    thumb = Thumbnail(format=ThumbFormat.RGB8, width=8, height=6, data=bytes((255, 128, 0)) * (8 * 6))
    return FrawFile(
        head=head,
        raw=golden_raw(),
        meta={"make": "FabriCTD", "model": "Golden Test", "iso": 100},
        color=color,
        thumbnail=thumb,
        dev={"version": 1, "exposure_ev": 0.5},
    )


def _main(argv: Iterable[str]) -> int:
    args = list(argv)
    if len(args) in (2, 3) and args[0] == "--make-fixture" and (len(args) == 2 or args[2] == "--zlib"):
        zl = len(args) == 3
        make_golden().write(args[1], compress=zl)
        print(f"wrote {args[1]} ({'zlib' if zl else 'uncompressed'})")
        return 0
    print("usage: python -m fraw.format --make-fixture <out.fraw> [--zlib]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
