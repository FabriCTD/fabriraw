"""ctd-raw CLI tests (golden fixtures always; real camera RAWs when present)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import FIXTURES, find_chunk
from fraw import __version__
from fraw.cli import main
from fraw.format import FrawFile
from fraw.fset import FrawSet

GOLDEN = FIXTURES / "golden.fraw"
RAW_DIR = Path(__file__).resolve().parent / "fixtures_ingest" / "raw"
RAWS = sorted(RAW_DIR.glob("*.*")) if RAW_DIR.is_dir() else []


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    p = tmp_path / "g.fraw"
    shutil.copyfile(GOLDEN, p)
    return p


def _rawpy_available() -> bool:
    try:
        import rawpy  # noqa: F401
    except ImportError:
        return False
    return True


needs_raw = pytest.mark.skipif(not RAWS or not _rawpy_available(), reason="sin RAW de cámara de prueba o sin rawpy")


# --------------------------------------------------------------- basics
def test_version(capsys):
    code, out, _ = run(capsys, "--version")
    assert code == 0 and __version__ in out


def test_no_args_is_usage_error(capsys):
    code, _, err = run(capsys)
    assert code == 2 and "usage" in err.lower()


def test_python_dash_m(tmp_path):
    r = subprocess.run([sys.executable, "-m", "fraw", "--version"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and __version__ in r.stdout


# ----------------------------------------------------------------- info
def test_info_text(capsys, golden):
    code, out, _ = run(capsys, "info", golden)
    assert code == 0
    for s in ("HEAD", "META", "COLR", "THMB", "DATA", "DEV", "END", "64x48", "RGGB", "Golden Test"):
        assert s in out


def test_info_json(capsys, golden):
    code, out, _ = run(capsys, "info", golden, "--json")
    assert code == 0
    d = json.loads(out)
    assert d["type"] == "fraw"
    assert [c["id"] for c in d["chunks"]] == ["HEAD", "META", "COLR", "THMB", "DATA", "DEV ", "END "]
    assert d["head"]["width"] == 64 and d["head"]["cfa_pattern"] == "RGGB"
    assert d["meta"]["model"] == "Golden Test"
    assert d["dev"] == {"version": 1, "exposure_ev": 0.5}
    assert d["dev_effective"]["exposure_ev"] == 0.5


def test_info_fset(capsys, tmp_path):
    f = FrawFile.read(GOLDEN)
    s = FrawSet(frames=[f, f], burst_id="b1", created="2026-11-07T00:00:00Z")
    p = tmp_path / "b.fset"
    s.write(p)
    code, out, _ = run(capsys, "info", p, "--json")
    assert code == 0
    d = json.loads(out)
    assert d["type"] == "fset" and d["seth"]["frame_count"] == 2 and len(d["frames"]) == 2
    code, out, _ = run(capsys, "info", p)
    assert code == 0 and "Frame 1" in out


def test_info_corrupt_exit2(capsys, tmp_path, golden):
    b = bytearray(golden.read_bytes())
    off, _ = find_chunk(b, b"DATA")
    b[off + 20] ^= 0xFF  # payload byte → CRC mismatch
    bad = tmp_path / "bad.fraw"
    bad.write_bytes(bytes(b))
    code, _, err = run(capsys, "info", bad)
    assert code == 2 and "CorruptChunk" in err


def test_missing_file_exit2(capsys, tmp_path):
    code, _, err = run(capsys, "info", tmp_path / "nope.fraw")
    assert code == 2 and "no existe" in err


# --------------------------------------------------------------- verify
def test_verify(capsys, tmp_path, golden):
    code, out, _ = run(capsys, "verify", golden, FIXTURES / "golden_z.fraw")
    assert code == 0 and out.count("OK") == 2
    bad = tmp_path / "trunc.fraw"
    bad.write_bytes(golden.read_bytes()[:200])
    code, out, _ = run(capsys, "verify", golden, bad)
    assert code == 1 and "ERROR" in out and "Truncated" in out


def test_verify_glob(capsys, tmp_path, golden):
    shutil.copyfile(GOLDEN, tmp_path / "h.fraw")
    code, out, _ = run(capsys, "verify", str(tmp_path / "*.fraw"))
    assert code == 0 and "2/2" in out


def test_verify_glob_no_match(capsys, tmp_path):
    code, _, err = run(capsys, "verify", str(tmp_path / "*.fraw"))
    assert code == 2 and "coincide" in err


# -------------------------------------------------------------- develop
@pytest.mark.parametrize("ext,bits", [(".jpg", 8), (".png", 8), (".png", 16), (".tif", 16), (".tif", 8)])
def test_develop_formats(capsys, tmp_path, golden, ext, bits):
    out = tmp_path / f"o{ext}"
    code, _, err = run(capsys, "develop", golden, "-o", out, "--bits", str(bits), "--ev", "0.3", "--wb", "daylight")
    assert code == 0, err
    with Image.open(out) as im:
        im.load()
        assert im.size == (64, 48)


def test_develop_overrides_change_output(capsys, tmp_path, golden):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    assert run(capsys, "develop", golden, "-o", a)[0] == 0
    assert run(capsys, "develop", golden, "-o", b, "--ev", "-2", "--wb", "2.0,1.0,1.2", "--gamma", "power:2.2",
               "--contrast", "0.5", "--saturation", "0.5", "--demosaic", "bilinear")[0] == 0
    ia = np.asarray(Image.open(a), dtype=np.int32)
    ib = np.asarray(Image.open(b), dtype=np.int32)
    assert ib.mean() < ia.mean() - 20


def test_develop_half(capsys, tmp_path, golden):
    out = tmp_path / "h.jpg"
    assert run(capsys, "develop", golden, "-o", out, "--half")[0] == 0
    assert Image.open(out).size == (32, 24)


def test_develop_bad_args(capsys, tmp_path, golden):
    assert run(capsys, "develop", golden, "-o", tmp_path / "x.bmp")[0] == 2
    assert run(capsys, "develop", golden, "-o", tmp_path / "x.jpg", "--bits", "16")[0] == 2
    assert run(capsys, "develop", golden, "-o", tmp_path / "x.jpg", "--wb", "violeta")[0] == 2
    assert run(capsys, "develop", golden, "-o", tmp_path / "x.jpg", "--gamma", "cubo")[0] == 2


def test_develop_save_dev(capsys, tmp_path, golden):
    before = golden.read_bytes()
    off, length = find_chunk(before, b"DATA")
    code, _, _ = run(capsys, "develop", golden, "-o", tmp_path / "o.jpg", "--ev", "1.25", "--save-dev")
    assert code == 0
    after = golden.read_bytes()
    assert after[: off + 20 + length] == before[: off + 20 + length]  # DATA and all before untouched
    assert FrawFile.read(golden).dev["exposure_ev"] == 1.25


# ------------------------------------------------------------------ dev
def test_dev_set_show_reset(capsys, golden):
    before = golden.read_bytes()
    off, length = find_chunk(before, b"DATA")
    code, _, err = run(capsys, "dev", golden, "--set", "exposure_ev=0.75", "wb_mode=custom", "wb_custom=2.1,1,1.3",
                       "gamma_mode=\"power\"", "contrast=9")
    assert code == 0, err
    after = golden.read_bytes()
    assert after[: off + 20 + length] == before[: off + 20 + length]
    f = FrawFile.read(golden)
    assert f.dev["exposure_ev"] == 0.75
    assert f.dev["wb_mode"] == "custom" and f.dev["wb_custom"] == [2.1, 1.0, 1.3]
    assert f.dev["gamma_mode"] == "power"
    assert f.dev["contrast"] == 1.0  # clamped
    assert np.array_equal(f.raw, FrawFile.read(GOLDEN).raw)
    code, out, _ = run(capsys, "dev", golden, "--show")
    assert code == 0 and "0.75" in out
    code, _, _ = run(capsys, "dev", golden, "--reset")
    assert code == 0 and FrawFile.read(golden).dev is None
    code, out, _ = run(capsys, "dev", golden)
    assert code == 0 and "no tiene" in out


def test_dev_errors(capsys, golden, tmp_path):
    assert run(capsys, "dev", golden, "--set", "nope=1")[0] == 2
    assert run(capsys, "dev", golden, "--set", "exposure_ev")[0] == 2
    p = tmp_path / "b.fset"
    FrawSet(frames=[FrawFile.read(GOLDEN)], burst_id="x", created="y").write(p)
    assert run(capsys, "dev", p, "--set", "exposure_ev=1")[0] == 2


# --------------------------------------------------------- extract-thumb
def test_extract_thumb(capsys, tmp_path, golden):
    out = tmp_path / "t.png"
    code, _, _ = run(capsys, "extract-thumb", golden, "-o", out)
    assert code == 0
    im = Image.open(out)
    assert im.size == (8, 6) and im.getpixel((0, 0)) == (255, 128, 0)


def test_extract_thumb_missing(capsys, tmp_path):
    f = FrawFile.read(GOLDEN)
    f.thumbnail = None
    p = tmp_path / "nt.fraw"
    f.write(p)
    assert run(capsys, "extract-thumb", p, "-o", tmp_path / "t.jpg")[0] == 2


# ------------------------------------------------------- real camera RAW
@needs_raw
def test_ingest_develop_real_raw(capsys, tmp_path):
    src = next((p for p in RAWS if p.suffix.lower() == ".arw"), RAWS[0])
    code, out, err = run(capsys, "ingest", src, "--outdir", tmp_path)
    assert code == 0, err
    fr = tmp_path / (src.stem + ".fraw")
    assert fr.is_file()
    assert run(capsys, "verify", fr)[0] == 0
    code, out, _ = run(capsys, "info", fr, "--json")
    meta = json.loads(out)["meta"]
    assert meta.get("make") and meta.get("iso")
    jpg = tmp_path / "o.jpg"
    code, _, err = run(capsys, "develop", fr, "-o", jpg, "--half", "--ev", "0.3")
    assert code == 0, err
    with Image.open(jpg) as im:
        im.load()
        assert im.format == "JPEG" and min(im.size) > 1000
    thumb = tmp_path / "t.jpg"
    assert run(capsys, "extract-thumb", fr, "-o", thumb)[0] == 0
    assert Image.open(thumb).format == "JPEG"


@needs_raw
def test_develop_camera_raw_directly(capsys, tmp_path):
    src = RAWS[0]
    out = tmp_path / "d.jpg"
    code, _, err = run(capsys, "develop", src, "-o", out, "--half")
    assert code == 0, err
    assert Image.open(out).format == "JPEG"
    # --save-dev makes no sense on a camera RAW
    assert run(capsys, "develop", src, "-o", out, "--half", "--save-dev")[0] == 2


@needs_raw
def test_ingest_multi_and_burst(capsys, tmp_path):
    srcs = RAWS[:2]
    code, out, err = run(capsys, "ingest", *srcs, "--outdir", tmp_path, "--no-thumb")
    assert code == 0, err
    for s in srcs:
        assert FrawFile.read(tmp_path / (s.stem + ".fraw")).thumbnail is None
    assert run(capsys, "ingest", *srcs, "-o", tmp_path / "x.fraw")[0] == 2  # -o with several inputs
    fset = tmp_path / "b.fset"
    code, _, err = run(capsys, "burst", *srcs, "-o", fset, "--no-thumb")
    assert code == 0, err
    assert len(FrawSet.read(fset).frames) == len(srcs)
    code, _, _ = run(capsys, "develop", fset, "-o", tmp_path / "f.jpg", "--frame", "1", "--half")
    assert code == 0
    assert run(capsys, "develop", fset, "-o", tmp_path / "f.jpg", "--frame", "9", "--half")[0] == 2


def test_ingest_unsupported_extension(capsys, tmp_path):
    p = tmp_path / "x.bmp"
    p.write_bytes(b"BM")
    assert run(capsys, "ingest", p)[0] == 2
