"""Motor Pro — ``ctd-raw develop`` options: --set, --auto-tone, --auto-wb, --long-edge, --output-sharpen."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import FIXTURES
from fraw.cli import main
from fraw.processor import FRAWProcessor, to_uint8


def run(capsys, *argv) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def golden(tmp_path: Path) -> Path:
    p = tmp_path / "g.fraw"
    shutil.copyfile(FIXTURES / "golden.fraw", p)
    return p


def test_develop_set_any_key(capsys, tmp_path, golden) -> None:
    out = tmp_path / "s.png"
    code, _, err = run(capsys, "develop", golden, "-o", out, "--set", "clarity=0.4", "--set",
                       "curve_master=[[0,0],[0.5,0.7],[1,1]]", "--set", "crop_right=0.5", "--set", "wb_mode=daylight",
                       "--demosaic", "bilinear")
    assert code == 0, err
    ref = FRAWProcessor.from_path(golden).develop(
        {"exposure_ev": 0.5, "clarity": 0.4, "curve_master": [[0, 0], [0.5, 0.7], [1, 1]], "crop_right": 0.5,
         "wb_mode": "daylight", "demosaic": "bilinear"})
    with Image.open(out) as im:
        assert im.size == (32, 48)
        np.testing.assert_array_equal(np.asarray(im), to_uint8(ref))
    assert run(capsys, "develop", golden, "-o", out, "--set", "nope=1")[0] == 2
    assert run(capsys, "develop", golden, "-o", out, "--set", "clarity")[0] == 2


def test_develop_auto_resize_sharpen(capsys, tmp_path, golden) -> None:
    out = tmp_path / "a.jpg"
    code, _, err = run(capsys, "develop", golden, "-o", out, "--auto-tone", "--auto-wb", "--long-edge", "40",
                       "--output-sharpen", "standard")
    assert code == 0, err
    with Image.open(out) as im:
        assert im.size == (40, 30) and im.info.get("icc_profile")
    assert run(capsys, "develop", golden, "-o", out, "--output-sharpen", "bad")[0] == 2
