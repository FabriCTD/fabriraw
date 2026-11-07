"""FabriRAW burst container (.fset), SPEC §6 — mirror of core/src/fset.cpp."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .format import (
    FLAG_CRITICAL,
    ID_END,
    ID_FRAM,
    ID_SETH,
    SET_MAGIC,
    ErrorCode,
    FrawError,
    FrawFile,
    _frame,
    _json_parse,
    _json_quote,
    _read_chunks,
)

__all__ = ["FrawSet"]

_SET_KNOWN = (ID_SETH, ID_FRAM, ID_END)


def _uint_value(v: object, key: str) -> int:
    if type(v) is not int or v < 0:
        raise FrawError(ErrorCode.InvalidHeader, f"SETH '{key}' must be a non-negative integer")
    if v >= 10**18:
        raise FrawError(ErrorCode.InvalidHeader, f"SETH '{key}' too large")
    return v


@dataclass
class FrawSet:
    frames: list[FrawFile] = field(default_factory=list)
    burst_id: str = ""
    created: str = ""

    def seth_json(self) -> str:
        return (
            '{"version":1,"frame_count":%d,"burst_id":%s,"created":%s}'
            % (len(self.frames), _json_quote(self.burst_id), _json_quote(self.created))
        )

    @classmethod
    def read(cls, path: str | os.PathLike) -> "FrawSet":
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "FrawSet":
        chunks = _read_chunks(data, SET_MAGIC)
        out = cls()
        seen_seth = False
        frame_count = 0
        for i, ch in enumerate(chunks):
            if ch.id not in _SET_KNOWN:
                if ch.critical:
                    raise FrawError(ErrorCode.UnsupportedCriticalChunk, f"unknown critical chunk {ch.id!r}")
                if not seen_seth:
                    raise FrawError(ErrorCode.BadOrder, "SETH must be the first chunk")
                continue  # non-critical unknown chunks are ignored in .fset
            if not ch.critical:
                raise FrawError(ErrorCode.CorruptChunk, f"wrong CRITICAL flag on chunk {ch.id!r}")
            if ch.id == ID_END:
                if not seen_seth:
                    raise FrawError(ErrorCode.MissingChunk, "missing SETH chunk")
                break
            if ch.id == ID_SETH:
                if seen_seth:
                    raise FrawError(ErrorCode.DuplicateChunk, "duplicate SETH chunk")
                if i != 0:
                    raise FrawError(ErrorCode.BadOrder, "SETH must be the first chunk")
                seen_seth = True
                try:
                    obj = _json_parse(ch.payload)
                except (ValueError, UnicodeError) as e:
                    raise FrawError(ErrorCode.CorruptChunk, f"SETH is not valid JSON: {e}") from None
                if type(obj) is not dict:
                    raise FrawError(ErrorCode.CorruptChunk, "SETH root must be an object")
                if not all(k in obj for k in ("version", "frame_count", "burst_id", "created")):
                    raise FrawError(ErrorCode.InvalidHeader, "SETH missing required keys")
                if _uint_value(obj["version"], "version") != 1:
                    raise FrawError(ErrorCode.UnsupportedVersion, "unsupported SETH version")
                frame_count = _uint_value(obj["frame_count"], "frame_count")
                if frame_count < 1:
                    raise FrawError(ErrorCode.InvalidHeader, "frame_count must be >= 1")
                if type(obj["burst_id"]) is not str or type(obj["created"]) is not str:
                    raise FrawError(ErrorCode.InvalidHeader, "SETH burst_id/created must be strings")
                out.burst_id = obj["burst_id"]
                out.created = obj["created"]
                continue
            # FRAM
            if not seen_seth:
                raise FrawError(ErrorCode.BadOrder, "FRAM before SETH")
            out.frames.append(FrawFile.from_bytes(ch.payload))
        if len(out.frames) != frame_count:
            raise FrawError(ErrorCode.SizeMismatch, "frame_count != number of FRAM chunks")
        return out

    def to_bytes(self, compress: bool = True) -> bytes:
        if not self.frames:
            raise FrawError(ErrorCode.InvalidHeader, "a .fset needs at least one frame")
        try:
            seth = self.seth_json().encode("utf-8")
        except UnicodeError as e:
            raise FrawError(ErrorCode.InvalidHeader, f"invalid SETH strings: {e}") from None
        parts = [SET_MAGIC, _frame(ID_SETH, FLAG_CRITICAL, seth)]
        for f in self.frames:
            parts.append(_frame(ID_FRAM, FLAG_CRITICAL, f.to_bytes(compress)))
        parts.append(_frame(ID_END, FLAG_CRITICAL, b""))
        return b"".join(parts)

    def write(self, path: str | os.PathLike, compress: bool = True) -> None:
        data = self.to_bytes(compress)
        with open(path, "wb") as f:
            f.write(data)
