"""Motor Pro — measured preview timings at 1600 px (informative: xfail on slow/loaded machines).

Budgets: defaults ≤ 16 ms; draft with everything on ≤ 33 ms; full with everything on ≤ 120 ms.
Set ``FRAW_PERF_24MP=1`` to also time a 24 MP full-quality export.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

from fraw.format import Color, FrawFile, Head
from fraw.processor import FRAWProcessor

ALL_ON = dict(
    highlight_mode="reconstruct", highlights=-0.4, shadows=0.3, whites=0.2, blacks=-0.1, clarity=0.3, texture=0.2,
    dehaze=0.2, vibrance=0.3, saturation=1.1, contrast=0.2, curve_p_lights=0.2,
    curve_master=[[0, 0], [0.5, 0.6], [1, 1]], curve_red=[[0, 0], [0.4, 0.45], [1, 1]], hsl_blue_l=-0.3,
    hsl_orange_s=0.2, hsl_green_h=0.3, grade_shadows_hue=200, grade_shadows_sat=0.3, grade_highlights_hue=40,
    grade_highlights_sat=0.2, grade_global_lum=0.05, sharpen_amount=0.6, nr_luminance=0.3, nr_color=0.5,
    rotate_deg=2.0, perspective_vertical=0.2, lens_distortion=0.2, lens_vignette=0.3, ca_red_cyan=0.3, defringe=0.4,
    crop_left=0.05, crop_right=0.95, vignette_amount=-0.4, grain_amount=0.3, cal_red_hue=0.2, cal_shadows_tint=0.1,
)


def synthetic(h: int, w: int) -> FrawFile:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.5 + 0.4 * np.sin(xx / 37.0) * np.cos(yy / 53.0)
    raw = (512 + np.clip(base * 0.6 + rng.normal(0, 0.02, (h, w)), 0, 1) * 15000).astype(np.uint16)
    return FrawFile(
        head=Head(width=w, height=h, effective_bits=14, white_level=16383, black_level=(512,) * 4), raw=raw,
        color=Color(as_shot_wb=(2.0, 1.0, 1.5, 1.0), rgb_cam=(1.6, -0.4, -0.2, -0.2, 1.5, -0.3, 0, -0.5, 1.5), flags=1),
    )


def _median_ms(fn, n: int) -> float:
    fn(0)
    ts = []
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        fn(i)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def test_preview_budgets() -> None:
    p = FRAWProcessor(synthetic(2134, 3200))
    res = {
        "defaults": _median_ms(lambda i: p.preview({"exposure_ev": 0.01 * i}), 20),
        "draft_all": _median_ms(lambda i: p.preview({**ALL_ON, "exposure_ev": 0.01 * i}, draft=True), 12),
        "full_all": _median_ms(lambda i: p.preview({**ALL_ON, "exposure_ev": 0.01 * i}), 6),
        "full_all_grain_only": _median_ms(lambda i: p.preview({**ALL_ON, "grain_amount": 0.3 + 0.01 * i}), 6),
        "full_all_grading_only": _median_ms(lambda i: p.preview({**ALL_ON, "grade_global_lum": 0.01 * i}), 6),
    }
    print("\npreview 1600 px: " + ", ".join(f"{k}={v:.1f} ms" for k, v in res.items()))
    over = [k for k, lim in (("defaults", 16), ("draft_all", 33), ("full_all", 120)) if res[k] > lim]
    if over:
        pytest.xfail("over budget on this machine: " + ", ".join(f"{k} {res[k]:.1f} ms" for k in over))


@pytest.mark.skipif(not os.environ.get("FRAW_PERF_24MP"), reason="set FRAW_PERF_24MP=1 to time a 24 MP export")
def test_export_24mp(tmp_path: Path) -> None:
    p = FRAWProcessor(synthetic(4000, 6000))
    for name, prm in (("defaults", {}), ("all_on", ALL_ON)):
        t0 = time.perf_counter()
        p.export(tmp_path / f"{name}.jpg", prm)
        print(f"\nexport 24 MP JPEG ({name}, AHD): {time.perf_counter() - t0:.2f} s")
