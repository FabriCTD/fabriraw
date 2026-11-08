"""Detección por contenido: se aceptan RAW con datos de sensor aunque tengan otra
extensión, y se rechazan imágenes ya procesadas aunque se llamen .dng/.nef."""

from __future__ import annotations

import io
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES
from fraw.sniff import NotRawError, sniff, sniff_bytes

RAW_DIR = Path(__file__).resolve().parent / "fixtures_ingest" / "raw"
EXTRA_DIR = Path(__file__).resolve().parent / "fixtures_ingest" / "raw_extra"
GOLDEN = FIXTURES / "golden.fraw"


# ------------------------------------------------------------ helpers
def _tiff(entries: list[tuple[int, int, int, int]], *, little: bool = True, extra_ifd=None) -> bytes:
    """TIFF mínimo con un IFD0 (tag, type, count, value)."""
    e = "<" if little else ">"
    hdr = (b"II" if little else b"MM") + struct.pack(e + "HI", 42, 8)
    ifd = struct.pack(e + "H", len(entries))
    for tag, typ, cnt, val in sorted(entries):
        if typ == 3 and cnt == 1:
            ifd += struct.pack(e + "HHIHH", tag, typ, cnt, val, 0)
        else:
            ifd += struct.pack(e + "HHII", tag, typ, cnt, val)
    ifd += struct.pack(e + "I", 0)
    return hdr + ifd + b"\0" * 64


def _pil_bytes(fmt: str, **kw) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((np.arange(48 * 32 * 3) % 251).astype("uint8").reshape(32, 48, 3)).save(buf, fmt, **kw)
    return buf.getvalue()


# ------------------------------------------------------------ sniff_bytes
@pytest.mark.parametrize(
    "head,kind,label",
    [
        (b"FRAW\r\n\x1a\n" + b"\0" * 8, "fraw", "FabriRAW"),
        (b"FSET\r\n\x1a\n" + b"\0" * 8, "fset", "FabriRAW ráfaga"),
        (b"\0\0\0\x18ftypcrx \0\0\0\x01crx isom", "raw", "Canon CR3"),
        (b"\0\0\0\x18ftypheic\0\0\0\0mif1heic", "processed", "HEIC"),
        (b"\0\0\0\x1cftypavif\0\0\0\0avifmif1", "processed", "AVIF"),
        (b"FUJIFILMCCD-RAW 0201", "raw", "Fujifilm RAF"),
        (b"IIRO\x08\0\0\0", "raw", "Olympus ORF"),
        (b"IIU\0\x08\0\0\0", "raw", "Panasonic RW2"),
        (b"II*\0\x10\0\0\0CR\x02\0", "raw", "Canon CR2"),
        (b"\xff\xd8\xff\xe1\0\0Exif", "processed", "JPEG"),
        (b"\x89PNG\r\n\x1a\n\0\0", "processed", "PNG"),
        (b"RIFF\0\0\0\0WEBPVP8 ", "processed", "WebP"),
        (b"GIF89a\0\0", "processed", "GIF"),
        (b"8BPS\0\x01", "processed", "PSD"),
        (b"%PDF-1.7", "processed", "PDF"),
        (b"", "empty", "vacío"),
        (b"\x13\x37whatever", "maybe_raw", "desconocido"),
    ],
)
def test_sniff_magic(head, kind, label):
    d = sniff_bytes(head)
    assert (d.kind, d.label) == (kind, label)
    if kind == "processed":
        assert d.reason and "sensor" in d.reason or "PDF" in d.label


def test_tiff_dng_bayer_is_raw():
    d = sniff_bytes(_tiff([(50706, 1, 4, 0x00000401), (262, 3, 1, 32803), (254, 4, 1, 0)]))
    assert (d.kind, d.label) == ("raw", "DNG")


def test_tiff_dng_linear_is_processed():
    d = sniff_bytes(_tiff([(50706, 1, 4, 0x00000401), (262, 3, 1, 34892), (254, 4, 1, 0)]))
    assert d.kind == "processed" and "lineal" in d.label.lower()


def test_tiff_with_cfa_is_raw_even_without_dng_tag():
    assert sniff_bytes(_tiff([(262, 3, 1, 32803)])).kind == "raw"


def test_plain_rgb_tiff_is_processed():
    d = sniff_bytes(_tiff([(254, 4, 1, 0), (262, 3, 1, 2), (277, 3, 1, 3)]))
    assert d.kind == "processed" and d.label == "TIFF"


def test_big_endian_tiff_dng():
    d = sniff_bytes(_tiff([(50706, 1, 4, 0x01040000), (262, 3, 1, 32803)], little=False))
    assert d.kind == "raw"


def test_truncated_or_garbage_tiff_never_crashes():
    base = _tiff([(50706, 1, 4, 1), (262, 3, 1, 32803)])
    rng = np.random.default_rng(7)
    for n in range(len(base)):
        sniff_bytes(base[:n])  # no debe lanzar
    for _ in range(300):
        b = bytearray(base)
        for i in rng.integers(0, len(b), 6):
            b[int(i)] = int(rng.integers(0, 256))
        sniff_bytes(bytes(b))


def test_real_pillow_images_are_processed(tmp_path):
    for fmt, ext in (("JPEG", ".jpg"), ("PNG", ".png"), ("TIFF", ".tif"), ("BMP", ".bmp"), ("WEBP", ".webp")):
        p = tmp_path / ("foto" + ext)
        p.write_bytes(_pil_bytes(fmt))
        assert sniff(p).kind == "processed", fmt
        # renombrarlo a extensión RAW no lo vuelve RAW
        q = tmp_path / ("foto_" + fmt + ".dng")
        shutil.copyfile(p, q)
        assert sniff(q).kind == "processed", fmt


def test_fraw_detected_regardless_of_extension(tmp_path):
    p = tmp_path / "golden.dat"
    shutil.copyfile(GOLDEN, p)
    assert sniff(p).kind == "fraw"


def test_missing_file_is_empty(tmp_path):
    d = sniff(tmp_path / "nope.raw")
    assert d.kind == "empty" and d.reason


# ------------------------------------------------------------ ingest (rechazos sin rawpy real)
def test_ingest_rejects_processed_before_libraw(tmp_path):
    from fraw.ingest import ingest

    p = tmp_path / "foto.NEF"
    p.write_bytes(_pil_bytes("JPEG"))
    with pytest.raises(NotRawError, match="JPEG"):
        ingest(p)
    with pytest.raises(NotRawError, match="FabriRAW"):
        ingest(GOLDEN)
    assert issubclass(NotRawError, ValueError)


# ------------------------------------------------------------ integración con RAW reales
def _need(name: str) -> Path:
    pytest.importorskip("rawpy")
    p = RAW_DIR / name if (RAW_DIR / name).is_file() else EXTRA_DIR / name
    if not p.is_file():
        pytest.skip(f"falta {name} en fixtures_ingest/raw (no se versionan; ver raw.pixls.us)")
    return p


def test_dng_renamed_to_tiff_is_accepted(tmp_path):
    from fraw.ingest import ingest

    src = _need("iPhoneSE_bayer.dng")
    renamed = tmp_path / "descargado del navegador.tiff"
    shutil.copyfile(src, renamed)
    assert sniff(renamed).kind == "raw"
    a = ingest(renamed, thumbnail=False)
    b = ingest(src, thumbnail=False)
    assert a.head == b.head
    assert np.array_equal(a.raw, b.raw)


def test_nef_with_unknown_extension_is_accepted(tmp_path):
    from fraw.ingest import ingest

    src = _need("DSC_0059.NEF")
    renamed = tmp_path / "foto.bin"
    shutil.copyfile(src, renamed)
    f = ingest(renamed, thumbnail=False)
    assert (f.head.width, f.head.height) == (6080, 4012)


def test_linear_rgb_dng_is_rejected():
    from fraw.ingest import ingest

    with pytest.raises(NotRawError, match="Bayer"):
        ingest(_need("5D3_linear_rgb.dng"), thumbnail=False)


def test_cli_batch_skips_processed_and_reports(tmp_path, capsys):
    from fraw.cli import main

    src = _need("iPhoneSE_bayer.dng")
    shutil.copyfile(src, tmp_path / "a.tiff")
    (tmp_path / "b.jpg").write_bytes(_pil_bytes("JPEG"))
    rc = main(["ingest", str(tmp_path / "*.*"), "--outdir", str(tmp_path / "out"), "--no-thumb"])
    err = capsys.readouterr().err
    assert rc == 1
    assert (tmp_path / "out" / "a.fraw").is_file()
    assert not (tmp_path / "out" / "b.fraw").exists()
    assert "omitido" in err and "JPEG" in err


def test_cli_single_processed_file_fails_with_reason(tmp_path, capsys):
    from fraw.cli import main

    p = tmp_path / "x.tif"
    p.write_bytes(_pil_bytes("TIFF"))
    assert main(["info", str(p)]) == 2
    assert "procesada" in capsys.readouterr().err
