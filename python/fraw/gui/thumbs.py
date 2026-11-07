"""Fast THMB/META/HEAD peek for the library panel (no DATA decoding, no Qt).

Walks the chunk headers with ``seek`` so a 30 MB .fraw costs a few small
reads. Only the chunks it returns are CRC-checked; full validation is done when
the file is actually opened.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass

from ..format import CHUNK_HEADER_FMT, MAGIC, SET_MAGIC, Head, _json_parse

__all__ = ["Peek", "peek_file"]

_HDR = struct.calcsize(CHUNK_HEADER_FMT)
_MAX_SMALL = 64 << 20  # sanity limit for THMB / META payloads


@dataclass
class Peek:
    thumb_format: int | None = None  # 0 JPEG, 1 RGB8
    thumb_size: tuple[int, int] = (0, 0)
    thumb_data: bytes = b""
    meta: dict | None = None
    width: int = 0
    height: int = 0
    orientation: int = 1
    frames: int = 1  # .fset frame count (1 for .fraw)


def _walk(f, base: int, end: int, out: Peek) -> None:
    """Parses a .fraw starting at ``base`` (its magic) up to DATA."""
    f.seek(base)
    if f.read(8) != MAGIC:
        raise ValueError("no es un .fraw")
    pos = base + 8
    while pos + _HDR <= end:
        f.seek(pos)
        cid, _flags, length = struct.unpack(CHUNK_HEADER_FMT, f.read(_HDR))
        payload_pos = pos + _HDR
        if cid in (b"HEAD", b"THMB", b"META") and length <= _MAX_SMALL:
            payload = f.read(length)
            crc_raw = f.read(4)
            if len(payload) != length or len(crc_raw) != 4:
                raise ValueError("truncado")
            (crc,) = struct.unpack("<I", crc_raw)
            if crc != zlib.crc32(payload, zlib.crc32(cid)) & 0xFFFFFFFF:
                raise ValueError(f"CRC incorrecto en {cid!r}")
            if cid == b"HEAD":
                h = Head.unpack(payload)
                out.width, out.height, out.orientation = h.width, h.height, h.orientation
                if any((h.crop_width, h.crop_height)):
                    out.width, out.height = h.crop_width, h.crop_height
            elif cid == b"META":
                try:
                    m = _json_parse(payload)
                    out.meta = m if isinstance(m, dict) else None
                except (ValueError, UnicodeError):
                    out.meta = None
            elif len(payload) >= 16:
                fmt, tw, th, tl = struct.unpack_from("<B3xIII", payload, 0)
                if fmt in (0, 1) and tl == len(payload) - 16:
                    out.thumb_format, out.thumb_size, out.thumb_data = fmt, (tw, th), payload[16:]
        if cid in (b"DATA", b"END "):
            return
        pos = payload_pos + length + 4


def peek_file(path: str | os.PathLike) -> Peek:
    """Thumbnail + META + geometry of a .fraw/.frw or the first frame of a .fset."""
    out = Peek()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic == MAGIC:
            _walk(f, 0, size, out)
            return out
        if magic != SET_MAGIC:
            raise ValueError("no es un archivo FabriRAW")
        pos = 8
        first_frame: int | None = None
        count = 0
        while pos + _HDR <= size:
            f.seek(pos)
            cid, _flags, length = struct.unpack(CHUNK_HEADER_FMT, f.read(_HDR))
            if cid == b"FRAM":
                count += 1
                if first_frame is None:
                    first_frame = pos + _HDR
            if cid == b"END ":
                break
            pos += _HDR + length + 4
        out.frames = max(1, count)
        if first_frame is not None:
            _walk(f, first_frame, size, out)
    return out
