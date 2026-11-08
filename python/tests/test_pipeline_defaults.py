"""Motor Pro: defaults reproduce the DEV v1 engine; every SCHEMA setting has an effect."""

from __future__ import annotations

import numpy as np
import pytest

import legacy_processor_v1 as legacy
from conftest import FIXTURES
from fraw.devparams import SCHEMA, SPECS, DevParams
from fraw.format import FrawFile
from fraw.processor import STAGE_KEYS, FRAWProcessor, to_uint8
from test_processor import make_file

RGB_CAM = [[1.6, -0.4, -0.2], [-0.2, 1.5, -0.3], [0.0, -0.5, 1.5]]


def scene(h: int = 64, w: int = 96, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.25 + 0.2 * np.sin(xx / 5.0) * np.cos(yy / 7.0)
    cam = np.stack([base * 1.1, base, base * 0.8], axis=-1) + rng.normal(0, 0.02, (h, w, 3))
    cam[: h // 8] = 1.0  # clipped band
    cam[h // 2 :, : w // 4] = (0.05, 0.1, 0.5)  # blue block
    cam[h // 2 :, w // 4 : w // 2] = (0.5, 0.08, 0.06)  # red block
    return np.clip(cam, 0, 1)


def files() -> list[FrawFile]:
    out = [make_file(scene(), wb=(2.0, 1.0, 1.5, 1.0), rgb_cam=RGB_CAM)]
    out.append(make_file(scene(48, 80, 5), wb=(1.8, 1.0, 1.3, 1.0), orientation=6, pattern=(1, 0, 2, 1)))
    out.append(make_file(scene(40, 60, 7), orientation=3, pattern=(2, 1, 1, 0), rgb_cam=RGB_CAM))
    out.append(FrawFile.read(FIXTURES / "golden.fraw"))
    return out


@pytest.mark.parametrize("idx", range(4))
def test_defaults_identical_to_v1_engine(idx: int) -> None:
    f = files()[idx]
    new, old = FRAWProcessor(f), legacy.FRAWProcessor(f)
    for prm in (DevParams(), DevParams(exposure_ev=0.6, contrast=0.3, saturation=1.3, temperature_shift=0.2),
                DevParams(gamma_mode="power", gamma=2.4, wb_mode="daylight", contrast=-0.4)):
        for kw in ({"half_size": True}, {"demosaic": "bilinear"}, {"demosaic": "ahd"}):
            a, b = new.develop(prm, **kw), old.develop(prm, **kw)
            np.testing.assert_array_equal(a, b)  # bit-exact
        for side in (1600, 40, 17):
            np.testing.assert_array_equal(new.preview(prm, side), old.preview(prm, side))


def test_general_path_matches_fast_path() -> None:
    """Forcing the stage pipeline (a no-op crop) gives the same pixels as the v1 fast path."""
    p = FRAWProcessor(files()[0])
    for prm in ({}, {"exposure_ev": -0.4, "contrast": 0.5, "gamma_mode": "power", "gamma": 1.8}):
        dp = DevParams.from_dict(prm)
        fast = p.preview(prm)
        np.testing.assert_array_equal(fast, p._preview_locked(dp, 1600))  # v1 fast path
        gen = p._preview_general(dp, 1600, p._plan(dp, True), True, False)  # stage pipeline, forced
        assert p._stages
        np.testing.assert_array_equal(fast, gen)
        # develop through the stage pipeline vs. the exact v1 maths
        a = p.develop({**prm, "sharpen_amount": 1e-9}, demosaic="bilinear")
        b = p.develop(prm, demosaic="bilinear")
        assert np.abs(a - b).max() < 1.0 / 255


def test_schema_fully_covered_by_stages() -> None:
    covered = set().union(*STAGE_KEYS.values())
    assert covered == set(SPECS), set(SPECS) ^ covered
    assert len(SCHEMA) == 107


# key -> (value, context needed for the key to matter)
ALT = {
    "wb_mode": ("daylight", {}),
    "wb_kelvin": (3500.0, {"wb_mode": "kelvin"}),
    "wb_kelvin_tint": (60.0, {"wb_mode": "kelvin"}),
    "wb_custom": ((1.2, 1.0, 2.2), {"wb_mode": "custom"}),
    "gamma": (1.6, {"gamma_mode": "power"}),
    "gamma_mode": ("linear", {}),
    "highlight_mode": ("reconstruct", {"exposure_ev": -1.0}),
    "demosaic": ("bilinear", {}),
    "curve_master": ([[0, 0], [0.5, 0.7], [1, 1]], {}),
    "curve_red": ([[0, 0], [0.5, 0.7], [1, 1]], {}),
    "curve_green": ([[0, 0], [0.5, 0.7], [1, 1]], {}),
    "curve_blue": ([[0, 0], [0.5, 0.7], [1, 1]], {}),
    "rotate_90": ("90", {}),
    "flip_h": (True, {}),
    "flip_v": (True, {}),
    "crop_aspect": ("1:1", {}),
    "crop_left": (0.2, {}),
    "crop_top": (0.2, {}),
    "crop_right": (0.8, {}),
    "crop_bottom": (0.8, {}),
    "perspective_scale": (1.3, {}),
    "sharpen_radius": (2.5, {"sharpen_amount": 1.0}),
    "sharpen_detail": (1.0, {"sharpen_amount": 1.0}),
    "sharpen_masking": (1.0, {"sharpen_amount": 1.0}),
    "nr_luminance_detail": (0.0, {"nr_luminance": 1.0}),
    "nr_color_detail": (0.0, {"nr_color": 1.0}),
    "vignette_midpoint": (0.1, {"vignette_amount": -1.0}),
    "vignette_roundness": (-1.0, {"vignette_amount": -1.0}),
    "vignette_feather": (0.0, {"vignette_amount": -1.0}),
    "vignette_highlights": (1.0, {"vignette_amount": -1.0}),
    "grain_size": (1.0, {"grain_amount": 1.0}),
    "grain_roughness": (0.0, {"grain_amount": 1.0}),
    "grade_blending": (1.0, {"grade_shadows_sat": 1.0, "grade_shadows_hue": 200.0}),
    "grade_balance": (1.0, {"grade_shadows_sat": 1.0, "grade_shadows_hue": 200.0}),
    "defringe": (1.0, {}),
}
for _z in ("shadows", "midtones", "highlights", "global"):
    ALT[f"grade_{_z}_hue"] = (120.0, {f"grade_{_z}_sat": 1.0})


@pytest.mark.parametrize("key", [s.key for s in SCHEMA])
def test_every_setting_has_an_effect(key: str) -> None:
    spec = SPECS[key]
    cam = scene(48, 64)
    yy, xx = np.mgrid[0:48, 0:64]
    cam[20:28, 20:44] = np.where(((xx[20:28, 20:44] // 2) % 2)[..., None] == 0, 0.9, 0.05)  # sharp edges
    cam[20:28, 20:44, 2] *= 0.3  # coloured edge (fringe-like)
    p = FRAWProcessor(make_file(cam, wb=(2.0, 1.0, 1.5, 1.0), rgb_cam=RGB_CAM))
    if key in ALT:
        val, ctx = ALT[key]
    elif spec.kind == "float":
        d = float(spec.default)
        val, ctx = (spec.hi if d < spec.hi - 1e-9 and (spec.hi - d) >= (d - spec.lo) else spec.lo), {}
        if key == "defringe":
            val = 1.0
    else:
        pytest.fail(f"no alternative value for {key}")
    base = DevParams.from_dict(ctx)
    alt = DevParams.from_dict({**ctx, key: val})
    assert not alt.is_default(key)
    kw = {} if key == "demosaic" else {"demosaic": "ahd"}
    a, b = p.develop(base, **kw), p.develop(alt, **kw)
    if a.shape != b.shape:
        return
    diff = np.abs(to_uint8(a).astype(int) - to_uint8(b).astype(int))
    assert diff.max() >= 1, f"{key} = {val!r} has no visible effect"
