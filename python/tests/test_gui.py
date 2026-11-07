"""CTD Photo Studio GUI tests (headless: QT_QPA_PLATFORM=offscreen)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import FIXTURES, find_chunk  # noqa: E402
from fraw.format import Color, FrawFile, Head  # noqa: E402
from fraw.fset import FrawSet  # noqa: E402
from fraw.gui import render  # noqa: E402
from fraw.gui.app import MainWindow, create_app, resource_path  # noqa: E402
from fraw.gui.canvas import qimage_from_rgb8  # noqa: E402
from fraw.gui.thumbs import peek_file  # noqa: E402
from fraw.gui.winassoc import EXTENSIONS, PROGID, registry_plan  # noqa: E402
from fraw.processor import FRAWProcessor, apply_orientation  # noqa: E402

GOLDEN = FIXTURES / "golden.fraw"


@pytest.fixture(scope="module")
def app():
    return create_app(["test"])


@pytest.fixture
def win(app, tmp_path):
    s = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow(settings=s)
    w.resize(1280, 800)
    w.show()
    yield w
    w.dirty = False
    w.close()
    app.processEvents()


@pytest.fixture
def golden(tmp_path) -> Path:
    p = tmp_path / "g.fraw"
    shutil.copyfile(GOLDEN, p)
    return p


def pump(app, cond, timeout=10.0) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.001)
    return False


def qimage_bytes(w) -> bytes:
    img = w.canvas.current_qimage()
    assert img is not None and not img.isNull()
    return bytes(img.constBits())[: img.sizeInBytes()]


# ----------------------------------------------------------------- basic
def test_resources_present():
    assert resource_path("theme.qss") is not None
    assert resource_path("icon.png") is not None


def test_qimage_zero_copy():
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    img = qimage_from_rgb8(arr)
    arr[0, 0] = (255, 0, 0)  # visible through the QImage → it wraps the same memory
    assert img.pixelColor(0, 0).red() == 255 and (img.width(), img.height()) == (20, 10)


def test_load_golden_and_sliders(app, win, golden):
    win.open_path(golden, sync=True)
    assert win.proc is not None and win.kind == "fraw"
    assert win.params.exposure_ev == 0.5  # DEV of the fixture
    assert not win.dirty
    assert win.canvas.has_image() and win.canvas.image_size == (64, 48)
    assert win.meta_labels["camera"].text() == "FabriCTD Golden Test"
    before = qimage_bytes(win)

    # Move a slider like the user would → asynchronous worker path.
    n0 = win.render_count
    win.sliders["exposure_ev"].setValue(-1.5)
    assert win.dirty
    assert pump(app, lambda: win.render_count > n0)
    after = qimage_bytes(win)
    assert after != before

    # Other controls, forced synchronous processing.
    win.set_param("contrast", 0.6)
    win.set_param("saturation", 0.2)
    win.set_param("wb_mode", "daylight")
    img = win.render_now()
    assert img is not None and qimage_bytes(win) != after
    assert win.histogram._h is not None and win.histogram._h.shape == (3, 256)


def test_slider_latency(app, win, tmp_path, capsys):
    """Slider → new image on the canvas (coalescing timer + worker + QImage)."""
    win.open_path(GOLDEN, sync=True)
    times = []
    for i in range(15):
        n0 = win.render_count
        t0 = time.perf_counter()
        win.sliders["exposure_ev"].setValue(-2 + 0.25 * i)
        assert pump(app, lambda: win.render_count > n0)
        times.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(times)
    with capsys.disabled():
        print(f"\n[gui] golden 64x48: slider->imagen mediana {med:.1f} ms (throttle 16 ms)")
    assert med < 500


def test_save_dev_roundtrip(app, win, golden):
    win.open_path(golden, sync=True)
    raw_before = golden.read_bytes()
    off, length = find_chunk(raw_before, b"DATA")
    win.sliders["exposure_ev"].setValue(1.75)
    win.sliders["shadows"].setValue(0.4)
    win.set_param("gamma_mode", "linear")
    assert win.dirty and win.btn_save.isEnabled()
    assert win.save_dev()
    assert not win.dirty
    raw_after = golden.read_bytes()
    assert raw_after[: off + 20 + length] == raw_before[: off + 20 + length]  # DATA untouched
    f = FrawFile.read(golden)
    assert f.dev["exposure_ev"] == 1.75 and f.dev["shadows"] == 0.4 and f.dev["gamma_mode"] == "linear"
    # Reopen → controls reflect the saved DEV
    win.open_path(golden, sync=True)
    assert win.sliders["exposure_ev"].value() == 1.75
    assert win.gamma_combo.currentData() == "linear"
    assert not win.dirty


def test_double_click_reset(app, win):
    win.open_path(GOLDEN, sync=True)
    row = win.sliders["contrast"]
    row.setValue(0.8)
    assert win.params.contrast == 0.8
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QMouseEvent

    ev = QMouseEvent(QEvent.Type.MouseButtonDblClick, QPointF(5, 5), QPointF(5, 5),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    row.slider.mouseDoubleClickEvent(ev)
    assert win.params.contrast == 0.0


def test_before_after(app, win):
    win.open_path(GOLDEN, sync=True)
    win.set_param("exposure_ev", 2.0)
    win.render_now()
    after = qimage_bytes(win)
    win.show_before(True)
    assert qimage_bytes(win) != after
    win.show_before(False)
    assert qimage_bytes(win) == after


def test_eyedropper_sets_custom_wb(app, win):
    win.open_path(GOLDEN, sync=True)
    win.btn_picker.setChecked(True)
    assert win.canvas.picker
    win.canvas.picked.emit(20.0, 20.0)
    assert win.params.wb_mode == "custom"
    r, g, b = win.params.wb_custom
    assert g == 1.0 and r > 0 and b > 0
    assert win.wb_combo.currentData() == "custom"
    assert not win.btn_picker.isChecked()
    # The picked patch is neutral after WB: camera means × multipliers are equal.
    cam = render.patch_cam_rgb(win.proc, 20, 20, 5)
    assert np.allclose(cam * np.array([r, g, b]), cam[1], rtol=1e-6)


def test_zoom_and_detail_tile(app, win, tmp_path):
    # 4000x3000 synthetic file: the preview (1600) is magnified at 100 % → detail tile.
    h, w = 3000, 4000
    y, x = np.mgrid[0:h, 0:w]
    raw = (600 + (x * 7 + y * 3) % 12000).astype(np.uint16)
    f = FrawFile(head=Head(width=w, height=h, effective_bits=14, white_level=16383, black_level=(512,) * 4),
                 raw=raw, meta={}, color=Color(as_shot_wb=(2.0, 1.0, 1.5, 1.0), flags=0))
    p = tmp_path / "big.fraw"
    f.write(p)
    win.open_path(p, sync=True)
    assert win.canvas.fit_mode
    win.canvas.zoom_100()
    assert abs(win.canvas.zoom - 1.0) < 1e-9
    assert win.needs_detail()
    # The tile must end up covering the visible area (it is re-requested after view changes).
    assert pump(app, lambda: win.canvas.has_tile() and win.canvas._tile_rect.contains(win.canvas.visible_image_rect()),
                timeout=30)
    win.canvas.fit()
    assert not win.needs_detail()


def test_fset_strip(app, win, tmp_path):
    f = FrawFile.read(GOLDEN)
    f2 = FrawFile.read(GOLDEN)
    f2.dev = {"version": 1, "exposure_ev": -2.0}
    p = tmp_path / "b.fset"
    FrawSet(frames=[f, f2], burst_id="b", created="c").write(p)
    win.open_path(p, sync=True)
    assert win.kind == "fset" and win.strip.isVisible() and win.strip.count() == 2
    assert not win.btn_save.isEnabled()
    win.strip.setCurrentRow(1)
    assert pump(app, lambda: win.frame_index == 1 and win.params.exposure_ev == -2.0)


def test_library_thumbnails(app, win, tmp_path):
    shutil.copyfile(GOLDEN, tmp_path / "a.fraw")
    shutil.copyfile(GOLDEN, tmp_path / "b.frw")
    win.set_library_folder(tmp_path)
    assert win.library.count() == 2
    assert pump(app, lambda: all(not win.library.item(i).icon().isNull() for i in range(2)))


def test_export_sync(app, win, tmp_path):
    from PIL import Image

    win.open_path(GOLDEN, sync=True)
    out = tmp_path / "e.png"
    win.export_to(str(out), bits=16, demosaic="bilinear", sync=True)
    with Image.open(out) as im:
        assert im.size == (64, 48)


# ------------------------------------------------------------ pure helpers
@pytest.mark.parametrize("o", range(1, 9))
def test_orientation_mapping(o):
    h, w = 6, 10
    s = np.arange(h * w).reshape(h, w)
    oi = apply_orientation(s, o)
    for yo in range(oi.shape[0]):
        for xo in range(oi.shape[1]):
            ys, xs = render.oriented_to_stored(yo, xo, w, h, o)
            assert oi[yo, xo] == s[ys, xs]
            assert render.stored_to_oriented(ys, xs, w, h, o) == (yo, xo)
    assert render.oriented_size(w, h, o) == (oi.shape[1], oi.shape[0])


@pytest.mark.parametrize("o", [1, 3, 6, 8])
def test_develop_region_matches_full(o):
    f = FrawFile.read(GOLDEN)
    f.head.orientation = o
    full = FRAWProcessor(f).develop(None, demosaic="bilinear")
    from fraw.processor import to_uint8

    full8 = to_uint8(full)
    oh, ow = full8.shape[:2]
    rect = (ow * 0.25, oh * 0.25, ow * 0.75, oh * 0.75)
    params = {"demosaic": "bilinear", "exposure_ev": 0.5}
    arr, (x0, y0, x1, y1) = render.develop_region(f, params, rect, margin=0)
    assert arr.shape == (y1 - y0, x1 - x0, 3)
    # Interior pixels (away from the region border, where interpolation differs) match.
    inner = arr[2:-2, 2:-2].astype(int)
    ref = full8[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2].astype(int)
    assert np.abs(inner - ref).max() <= 1


def test_neutral_wb():
    assert render.neutral_wb_from_patch(np.array([0.2, 0.4, 0.1])) == pytest.approx((2.0, 1.0, 4.0))
    assert render.neutral_wb_from_patch(np.array([0.0, 0.4, 0.1])) is None
    assert render.neutral_wb_from_patch(np.array([1.0, 1.0, 1.0])) is None


def test_peek_file(tmp_path):
    pk = peek_file(GOLDEN)
    assert pk.thumb_format == 1 and pk.thumb_size == (8, 6) and pk.meta["iso"] == 100
    assert (pk.width, pk.height) == (64, 48)
    f = FrawFile.read(GOLDEN)
    p = tmp_path / "s.fset"
    FrawSet(frames=[f, f, f], burst_id="x", created="y").write(p)
    pk = peek_file(p)
    assert pk.frames == 3 and pk.thumb_size == (8, 6)


def test_winassoc_plan_is_hkcu_only():
    plan = registry_plan(r"C:\Apps\CTD Photo Studio\CTDPhotoStudio.exe")
    assert all(k.startswith("Software\\Classes\\") for k, _, _ in plan)
    keys = {k for k, _, _ in plan}
    for ext in EXTENSIONS:
        assert f"Software\\Classes\\{ext}" in keys
    cmd = [d for k, n, d in plan if k.endswith("shell\\open\\command")][0]
    assert cmd == '"C:\\Apps\\CTD Photo Studio\\CTDPhotoStudio.exe" "%1"'
    assert PROGID in {d for _, _, d in plan}


def test_preview_fps_real_raw(app, win, capsys, tmp_path):
    """Optional: real 24 MP RAW → measured slider→image latency (skip if absent)."""
    raw_dir = Path(__file__).resolve().parent / "fixtures_ingest" / "raw"
    srcs = sorted(raw_dir.glob("*.ARW")) if raw_dir.is_dir() else []
    if not srcs:
        pytest.skip("sin RAW de prueba")
    pytest.importorskip("rawpy")
    from fraw.ingest import ingest_to_file

    p = tmp_path / "r.fraw"
    ingest_to_file(srcs[0], p, thumbnail=False)
    win.open_path(p, sync=True)
    times = []
    for i in range(20):
        n0 = win.render_count
        t0 = time.perf_counter()
        win.sliders["exposure_ev"].setValue(-1 + 0.1 * i)
        assert pump(app, lambda: win.render_count > n0, timeout=20)
        times.append((time.perf_counter() - t0) * 1000)
    # Pure processing time (no 16 ms coalescing timer)
    proc_ms = []
    for i in range(10):
        t0 = time.perf_counter()
        win.proc.preview({"exposure_ev": 0.05 * i, "shadows": 0.3}, 1600)
        proc_ms.append((time.perf_counter() - t0) * 1000)
    med, pm = statistics.median(times), statistics.median(proc_ms)
    with capsys.disabled():
        print(f"\n[gui] {srcs[0].name} {win.image_size()}: slider->imagen mediana {med:.1f} ms "
              f"(~{1000 / med:.0f} actualizaciones/s); preview() puro mediana {pm:.1f} ms")
    assert med < 1000
