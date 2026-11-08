"""Interfaz de revelado profesional: controles generados del SCHEMA, curvas, ruedas,
historial, presets, copiar/pegar, recorte, META, exportación por lotes y latencias.
Headless (QT_QPA_PLATFORM=offscreen)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from conftest import FIXTURES  # noqa: E402
from fraw.devparams import CURVE_MAX_POINTS, GROUPS, SCHEMA, SPECS, DevParams, normalize_curve  # noqa: E402
from fraw.format import FrawFile  # noqa: E402
from fraw.gui import croptool, curves, library, presets  # noqa: E402
from fraw.gui.app import MainWindow, create_app  # noqa: E402
from fraw.gui.batch import ExportJob, ExportOptions, export_batch, render_name  # noqa: E402
from fraw.gui.colorwheel import hue_sat_from_xy, xy_from_hue_sat  # noqa: E402
from fraw.gui.controls import BoolRow, EnumControl, ParamRow, format_value, parse_value  # noqa: E402
from fraw.gui.curves import CurveEditor  # noqa: E402
from fraw.gui.history import History  # noqa: E402
from fraw.gui.thumbs import peek_file  # noqa: E402

GOLDEN = FIXTURES / "golden.fraw"
RAW_DIR = Path(__file__).resolve().parent / "fixtures_ingest" / "raw"


@pytest.fixture(scope="module")
def app():
    return create_app(["test"])


@pytest.fixture
def win(app, tmp_path):
    s = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow(settings=s, data_dir=tmp_path / "data")
    w.resize(1400, 900)
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


# ------------------------------------------------------------ controles
def test_one_control_per_visible_spec(win):
    keys = {s.key for s in SCHEMA if not s.hidden}
    assert set(win.panel.controls) == keys
    for s in SCHEMA:
        if s.hidden:
            continue
        w = win.panel.controls[s.key]
        if s.kind in ("float", "int") and not s.key.startswith("grade_") or s.key == "grade_blending":
            assert isinstance(w, ParamRow), s.key
        elif s.kind == "bool":
            assert isinstance(w, BoolRow)
        elif s.kind == "enum":
            assert isinstance(w, EnumControl)
        elif s.kind == "curve":
            assert isinstance(w, CurveEditor)
    assert [sec.key for sec in win.panel.sections.values()] == [g.key for g in GROUPS]


def test_format_and_parse_scale():
    c = SPECS["contrast"]
    assert format_value(c, 0.35) == "+35" and parse_value(c, "35") == pytest.approx(0.35)
    assert parse_value(c, "500") == 1.0  # recortado
    e = SPECS["exposure_ev"]
    assert format_value(e, -1.5) == "-1,50 EV" and parse_value(e, "-1,5 EV") == -1.5
    k = SPECS["wb_kelvin"]
    assert parse_value(k, "6500") == 6500.0 and format_value(k, 6500) == "6500 K"
    assert parse_value(c, "abc") is None


def test_typing_value_applies_with_scale(app, win, golden):
    win.open_path(golden, sync=True)
    row = win.panel.rows["contrast"]
    row.field.setFocus()
    row.field.setText("35")
    QTest.keyClick(row.field, Qt.Key.Key_Return)
    assert win.params.contrast == pytest.approx(0.35)
    assert row.slider.value() == pytest.approx(0.35)
    sat = win.panel.rows["saturation"]
    sat.field.setText("150")
    sat.field.apply_text()
    assert win.params.saturation == pytest.approx(1.5)
    hue = win.panel.rows["hsl_orange_h"]
    hue.field.setText("-20")
    hue.field.apply_text()
    assert win.params.hsl_orange_h == pytest.approx(-0.2)
    assert win.dirty


def test_double_click_label_and_slider_reset(app, win):
    win.open_path(GOLDEN, sync=True)
    row = win.panel.rows["shadows"]
    row.setValue(0.6)
    assert win.params.shadows == pytest.approx(0.6)
    QTest.mouseDClick(row.name, Qt.MouseButton.LeftButton)
    assert win.params.shadows == 0.0
    row.setValue(-0.3)
    QTest.mouseDClick(row.slider, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    assert win.params.shadows == 0.0


def test_wheel_and_keys_fine_coarse(app, win):
    win.open_path(GOLDEN, sync=True)
    sl = win.panel.rows["clarity"].slider
    sl.setFocus()
    QTest.keyClick(sl, Qt.Key.Key_Right)
    assert win.params.clarity == pytest.approx(0.01)
    QTest.keyClick(sl, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert win.params.clarity == pytest.approx(0.11)
    QTest.keyClick(sl, Qt.Key.Key_Left, Qt.KeyboardModifier.ControlModifier)
    assert win.params.clarity == pytest.approx(0.109)


def test_group_reset_and_indicator(app, win):
    win.open_path(GOLDEN, sync=True)
    win.set_param("contrast", 0.4)
    win.set_param("hsl_red_s", 0.5)
    assert win.panel.sections["basic"].dot.isVisibleTo(win.panel)
    win.reset_group("basic")
    assert win.params.group_is_default("basic")
    assert win.params.hsl_red_s == 0.5
    assert not win.panel.sections["basic"].dot.isVisibleTo(win.panel)
    assert win.panel.sections["hsl"].dot.isVisibleTo(win.panel)


def test_bypass_does_not_touch_params(app, win, golden):
    win.open_path(golden, sync=True)
    win.set_param("exposure_ev", 2.0)
    with_fx = win.render_now().copy()
    win.panel.sections["basic"].switch.click()  # ocultar efecto
    assert "basic" in win.bypass
    without = win.render_now().copy()
    assert win.params.exposure_ev == 2.0
    assert not np.array_equal(with_fx, without)
    ref = win.engine.preview(DevParams(), 1600)
    assert np.array_equal(without, ref)  # golden: Básico en defecto
    assert win.save_dev()
    assert FrawFile.read(golden).dev["exposure_ev"] == 2.0
    win.set_bypass("basic", False)
    assert np.array_equal(win.render_now(), with_fx)


def test_search_filter(app, win):
    win.panel.filter("satur")
    keys = set(win.panel.matching_keys())
    assert "saturation" in keys and "hsl_red_s" in keys and "grade_shadows_sat" in keys
    assert "exposure_ev" not in keys
    win.panel.filter("")
    assert "exposure_ev" in set(win.panel.matching_keys())


# --------------------------------------------------------------- curvas
def test_curve_pure_helpers():
    pts = [[0.0, 0.0], [1.0, 1.0]]
    pts, i = curves.add_point(pts, 0.5, 0.6)
    assert i == 1 and pts[1] == [0.5, 0.6]
    pts = curves.move_point(pts, 1, 1.5, 0.7)  # x no puede pasar al vecino
    assert pts[1][0] == pytest.approx(1.0 - curves.MIN_DX)
    pts = curves.move_point(pts, 1, 0.4, 0.7)
    for k in range(12):
        pts, _ = curves.add_point(pts, 0.05 + 0.07 * k, 0.3)
    assert len(pts) <= CURVE_MAX_POINTS
    pts2, j = curves.add_point(pts, 0.999, 0.5)
    assert len(pts) < CURVE_MAX_POINTS or j == -1
    xs = [p[0] for p in pts]
    assert all(b > a for a, b in zip(xs, xs[1:]))
    assert normalize_curve(pts) is not None
    pts = curves.delete_point(pts, 2)
    assert curves.delete_point([[0, 0], [1, 1]], 0) == [[0.0, 0.0], [1.0, 1.0]]
    y = curves.eval_curve([[0, 0], [0.5, 0.7], [1, 1]], np.linspace(0, 1, 50))
    assert np.all(np.diff(y) >= -1e-12) and y[0] == 0 and y[-1] == 1


def test_curve_editor_mouse(app, win, golden):
    win.open_path(golden, sync=True)
    win.panel.sections["tone_curve"].set_expanded(True)
    ed = win.panel.curve
    cv = ed.canvas
    cv.resize(260, 260)
    a = cv.area()
    p = cv.to_px(0.25, 0.4).toPoint()
    QTest.mouseClick(cv, Qt.MouseButton.LeftButton, pos=p)  # agrega
    assert len(win.params.curve_master) == 3
    sel = cv.selected
    # mover con los campos Entrada/Salida
    ed.out_spin.setValue(51)
    assert win.params.curve_master[sel][1] == pytest.approx(0.2)
    # arrastrar
    q = cv.to_px(*win.params.curve_master[sel]).toPoint()
    QTest.mousePress(cv, Qt.MouseButton.LeftButton, pos=q)
    QTest.mouseMove(cv, cv.to_px(0.3, 0.1).toPoint())
    QTest.mouseRelease(cv, Qt.MouseButton.LeftButton, pos=cv.to_px(0.3, 0.1).toPoint())
    x, y = win.params.curve_master[sel]
    assert x == pytest.approx(0.3, abs=0.01) and y == pytest.approx(0.1, abs=0.01)
    # canal rojo independiente
    ed.set_channel("curve_red")
    QTest.mouseClick(cv, Qt.MouseButton.LeftButton, pos=cv.to_px(0.6, 0.8).toPoint())
    assert len(win.params.curve_red) == 3 and len(win.params.curve_master) == 3
    # borrar con doble click
    QTest.mouseDClick(cv, Qt.MouseButton.LeftButton, pos=cv.to_px(*win.params.curve_red[1]).toPoint())
    assert len(win.params.curve_red) == 2
    # borrar arrastrando afuera
    ed.set_channel("curve_master")
    q = cv.to_px(*win.params.curve_master[1]).toPoint()
    QTest.mousePress(cv, Qt.MouseButton.LeftButton, pos=q)
    QTest.mouseMove(cv, QPoint(int(a.right() + 60), q.y()))
    QTest.mouseRelease(cv, Qt.MouseButton.LeftButton, pos=QPoint(int(a.right() + 60), q.y()))
    assert len(win.params.curve_master) == 2
    for k in ("curve_master", "curve_red"):
        assert normalize_curve(getattr(win.params, k)) is not None
    assert win.history.labels()[-1] in ("Curva RGB", "Curva rojo")


def test_curve_max_points_ui(app):
    ed = CurveEditor()
    ed.resize(300, 340)
    ed.show()
    cv = ed.canvas
    for i in range(30):
        QTest.mouseClick(cv, Qt.MouseButton.LeftButton, pos=cv.to_px(0.02 + i * 0.032, 0.5).toPoint())
    assert len(ed.curves["curve_master"]) == CURVE_MAX_POINTS
    xs = [p[0] for p in ed.curves["curve_master"]]
    assert all(b > a for a, b in zip(xs, xs[1:]))


# --------------------------------------------------------------- ruedas
def test_color_wheel(app, win, golden):
    h, s = hue_sat_from_xy(*xy_from_hue_sat(120.0, 0.5, 50.0), 50.0)
    assert h == pytest.approx(120.0) and s == pytest.approx(0.5)
    win.open_path(golden, sync=True)
    win.panel.sections["grading"].set_expanded(True)
    wb = win.panel.wheels["shadows"]
    wheel = wb.wheel
    wheel.resize(130, 130)
    c = wheel.center()
    dx, dy = xy_from_hue_sat(200.0, 0.6, wheel.radius())
    QTest.mouseClick(wheel, Qt.MouseButton.LeftButton, pos=QPointF(c.x() + dx, c.y() + dy).toPoint())
    assert win.params.grade_shadows_hue == pytest.approx(200.0, abs=2)
    assert win.params.grade_shadows_sat == pytest.approx(0.6, abs=0.03)
    wb.f_sat.setText("25")
    wb.f_sat.apply_text()
    assert win.params.grade_shadows_sat == pytest.approx(0.25)
    assert wheel.sat == pytest.approx(0.25)
    QTest.mouseDClick(wheel, Qt.MouseButton.LeftButton, pos=c.toPoint())
    assert win.params.grade_shadows_sat == 0.0


# -------------------------------------------------------------- historial
def test_history_merges_drags():
    h = History()
    h.reset({"a": 0})
    for v in (1, 2, 3):
        h.record("A", {"a": v}, "a", final=False)
    h.record("A", {"a": 4}, "a", final=True)
    h.record("A", {"a": 5}, "a", final=True)  # nuevo arrastre → nuevo paso
    assert h.labels() == ["Abierto", "A", "A"] and h.current.state == {"a": 5}
    assert h.undo() == {"a": 4} and h.undo() == {"a": 0} and h.undo() is None
    assert h.redo() == {"a": 4}
    h.record("B", {"a": 9}, "b")
    assert not h.can_redo() and h.labels() == ["Abierto", "A", "B"]


def test_undo_redo_window(app, win, golden):
    win.open_path(golden, sync=True)
    sl = win.panel.rows["exposure_ev"].slider
    for v in (0.6, 0.8, 1.0, 1.2):  # un arrastre continuo
        sl.setValue(v, final=False)
    sl.changed.emit(sl.value(), True)
    win.set_param("contrast", 0.3)
    assert win.history.labels() == ["Abierto", "Exposición +1,20", "Contraste +30"]
    win.act_undo.trigger()
    assert win.params.contrast == 0.0 and win.params.exposure_ev == pytest.approx(1.2)
    assert win.panel.rows["contrast"].value() == 0.0
    win.act_undo.trigger()
    assert win.params.exposure_ev == pytest.approx(0.5)
    win.act_redo.trigger()
    win.act_redo.trigger()
    assert win.params.contrast == pytest.approx(0.3)
    assert QSettings  # noqa: B018
    assert {s.toString() for s in win.act_redo.shortcuts()} == {"Ctrl+Shift+Z", "Ctrl+Y"}
    win.goto_history(0)
    assert win.params.exposure_ev == pytest.approx(0.5) and win.params.contrast == 0.0
    win.new_snapshot("Base")
    win.set_param("vibrance", 0.5)
    win.restore_snapshot(0)
    assert win.params.vibrance == 0.0 and win.history.labels()[-1] == "Instantánea «Base»"


# ---------------------------------------------------------------- presets
def test_builtin_presets_valid():
    ps = presets.load_builtin()
    assert len(ps) >= 12
    names = {p.name for p in ps}
    for n in ("Natural", "Vívido", "Retrato suave", "Paisaje", "B&N alto contraste", "B&N suave", "Película cálida",
              "Mate", "Atardecer", "Noche urbana", "Teal & orange", "Sólo corregir lente"):
        assert n in names
    for p in ps:
        raw = json.loads(p.path.read_text(encoding="utf-8"))["settings"]
        assert raw and all(k in SPECS for k in raw), p.name
        parsed = DevParams.from_dict(raw).to_dict()
        for k, v in raw.items():  # from_dict no descarta ni recorta nada
            if isinstance(v, list):
                assert [list(map(float, q)) for q in v] == parsed[k], (p.name, k)
            else:
                assert parsed[k] == pytest.approx(v), (p.name, k)


def test_apply_preset_only_touches_its_keys(app, win, golden):
    win.open_path(golden, sync=True)
    win.set_param("lens_distortion", 0.4)
    before = win.params.to_dict()
    mate = next(p for p in win.all_presets() if p.name == "Mate")
    win.apply_preset(mate)
    after = win.params.to_dict()
    for k in SPECS:
        if k in mate.settings:
            assert after[k] == (mate.settings[k] if not isinstance(after[k], float) else pytest.approx(mate.settings[k]))
        else:
            assert after[k] == before[k], k
    assert win.history.labels()[-1] == "Preset «Mate»"


def test_fpreset_roundtrip(app, win, golden, tmp_path):
    win.open_path(golden, sync=True)
    win.set_param("vibrance", 0.42)
    win.set_param("hsl_blue_s", -0.3)
    pr = win.create_preset("Mío", ["basic", "hsl"], "Viajes")
    assert pr.path.suffix == ".fpreset" and pr.path.parent == win.user_preset_dir()
    data = json.loads(pr.path.read_text(encoding="utf-8"))
    assert data["fraw_preset"] == 1 and data["name"] == "Mío" and data["group"] == "Viajes"
    assert set(data["settings"]) == set(presets.keys_for_groups(["basic", "hsl"]))
    assert any(p.name == "Mío" for p in win.user_presets)
    out = tmp_path / "exp.fpreset"
    presets.save_preset(presets.Preset(pr.name, pr.group, pr.settings), path=out)
    back = presets.load_preset(out)
    assert back.settings == pr.settings
    (win.user_preset_dir() / "Mío.fpreset").unlink()
    win.reload_user_presets()
    imp = win.import_preset(out)
    assert imp is not None and any(p.name == "Mío" for p in win.user_presets)
    win.reset_all()
    win.apply_preset(imp)
    assert win.params.vibrance == pytest.approx(0.42) and win.params.hsl_blue_s == pytest.approx(-0.3)


def test_copy_paste_selective(app, win, golden):
    win.open_path(golden, sync=True)
    win.set_param("hsl_green_l", 0.25)
    win.set_param("contrast", 0.5)
    clip = win.copy_settings(["hsl"])
    assert set(clip) == set(presets.keys_for_groups(["hsl"]))
    win.reset_all()
    win.set_param("clarity", 0.2)
    win.paste_settings()
    assert win.params.hsl_green_l == pytest.approx(0.25)
    assert win.params.contrast == 0.0 and win.params.clarity == pytest.approx(0.2)


def test_sync_to_selection(app, win, tmp_path):
    a, b = tmp_path / "a.fraw", tmp_path / "b.fraw"
    shutil.copyfile(GOLDEN, a)
    shutil.copyfile(GOLDEN, b)
    win.open_path(a, sync=True)
    win.set_param("dehaze", 0.3)
    win.set_param("sharpen_amount", 0.9)
    win.sync_to([b], ["basic"], sync=True)
    dev = DevParams.from_dict(FrawFile.read(b).dev)
    assert dev.dehaze == pytest.approx(0.3) and dev.sharpen_amount == pytest.approx(0.0)  # detail no se sincroniza: queda en defecto


# ----------------------------------------------------------------- recorte
@pytest.mark.parametrize("handle", croptool.HANDLES)
def test_crop_aspect_locked(handle):
    W, H = 6000, 4000
    a = croptool.aspect_value("4:5", W, H)
    r = croptool.fit_aspect((0.0, 0.0, 1.0, 1.0), a, W, H)
    assert croptool.rect_aspect(r, W, H) == pytest.approx(0.8)
    r2 = croptool.drag_handle(r, handle, 0.1, 0.95, a, W, H)
    assert croptool.rect_aspect(r2, W, H) == pytest.approx(0.8, rel=1e-6)
    assert all(-1e-9 <= v <= 1 + 1e-9 for v in r2) and r2[0] < r2[2] and r2[1] < r2[3]
    assert croptool.rotate_rect_90(croptool.rotate_rect_90(r2, True), False) == pytest.approx(r2)


def test_crop_tool_in_window(app, win, golden):
    win.open_path(golden, sync=True)
    win.set_param("crop_aspect", "1:1")
    p = win.params
    w, h = win.image_size(apply_crop=False)
    assert (p.crop_right - p.crop_left) * w == pytest.approx((p.crop_bottom - p.crop_top) * h, abs=1e-6)
    win.reset_group("geometry")
    win.panel.btn_crop.click()  # R
    assert win.crop_mode and win.canvas.tool == "crop"
    assert win.canvas.image_size == (64, 48)  # sin recortar
    win.canvas.cropChanged.emit((0.1, 0.2, 0.6, 0.9))
    win.canvas.straighten.emit(3.0, False)
    win.canvas.straighten.emit(5.0, True)
    assert win.params.rotate_deg == pytest.approx(5.0)
    win.canvas.cropAccepted.emit()
    assert not win.crop_mode and win.history.labels()[-1] == "Recorte"
    assert (win.params.crop_left, win.params.crop_top, win.params.crop_right, win.params.crop_bottom) == \
        pytest.approx((0.1, 0.2, 0.6, 0.9))
    win.render_now()
    assert win.canvas.image_size == (32, 34)
    # Esc cancela
    win.set_crop_tool(True)
    win.canvas.cropChanged.emit((0.0, 0.0, 0.5, 0.5))
    win.canvas.cropCancelled.emit()
    assert win.params.crop_right == pytest.approx(0.6)
    # girar 90° transforma el recorte
    win._panel_action("rotate_right")
    assert win.params.rotate_90 == "90"
    assert (win.params.crop_left, win.params.crop_top) == pytest.approx((0.1, 0.1))


# ------------------------------------------------------- vista y herramientas
def test_compare_clipping_target_and_readout(app, win, golden):
    win.open_path(golden, sync=True)
    win.set_param("exposure_ev", 3.0)
    win.set_compare("split")
    win.render_now()
    assert win.canvas.compare == "split" and win._before_img is not None
    win.set_compare("side")
    assert win.canvas.view_width() == win.canvas.width() / 2
    win.set_compare("off")
    win.set_clipping(False, True)
    win.render_now()
    assert win.canvas._overlay is not None
    assert win.histogram.clip_hl > 0
    win._on_hover(10, 10)
    assert win.rgb_label.text().startswith("R ")
    # ajuste dirigido HSL: arrastrar hacia arriba sube la banda bajo el cursor
    win.panel.hsl_tabs.setCurrentIndex(1)  # Saturación
    win.panel.btn_target.click()
    assert win.canvas.tool == "target"
    before = win.params.to_dict()
    win.canvas.targetDrag.emit(5.0, 5.0, 40.0, False)
    win.canvas.targetDrag.emit(5.0, 5.0, 80.0, True)
    changed = [k for k in SPECS if k.startswith("hsl_") and win.params.to_dict()[k] != before[k]]
    assert changed and all(k.endswith("_s") for k in changed)
    assert win.history.labels()[-1].startswith("Ajuste dirigido")


def test_auto_tone_and_wb(app, win, golden):
    win.open_path(golden, sync=True)
    win.auto_tone()
    assert win.history.labels()[-1] == "Auto tono"
    win.auto_wb()
    assert win.history.labels()[-1] == "Auto balance de blancos"


def test_save_dev_and_reread(app, win, golden):
    win.open_path(golden, sync=True)
    win.apply_updates({"curve_master": [[0, 0], [0.3, 0.2], [1, 1]]}, "Curva RGB", "curve_master", True)
    win.set_param("hsl_purple_l", -0.4)
    win.set_param("grain_amount", 0.3)
    win.set_param("flip_h", True)
    expected = win.params.to_dict()
    assert win.save_dev() and not win.dirty
    dev = FrawFile.read(golden).dev
    assert dev["version"] == 2 and "contrast" not in dev  # sólo lo que difiere del defecto
    assert DevParams.from_dict(dev).to_dict() == expected
    win.open_path(golden, sync=True)
    assert win.params.to_dict() == expected and not win.dirty


# ---------------------------------------------------- biblioteca / META
def test_rating_label_meta_persist(app, win, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    for n in ("a", "b", "c"):
        shutil.copyfile(GOLDEN, lib / f"{n}.fraw")
    raw0 = FrawFile.read(lib / "a.fraw").raw.copy()
    win.set_library_folder(lib)
    assert win.library.count() == 3
    win.library.item(0).setSelected(True)
    win.library.item(2).setSelected(True)
    win.rate_selection(4)
    win.label_selection("green")
    for n in ("a", "c"):
        m = peek_file(lib / f"{n}.fraw").meta
        assert m["rating"] == 4 and m["label"] == "green" and m["make"] == "FabriCTD"
        f = FrawFile.read(lib / f"{n}.fraw")  # sigue siendo válido
        assert np.array_equal(f.raw, raw0) and f.dev == {"version": 1, "exposure_ev": 0.5}
    assert "rating" not in (peek_file(lib / "b.fraw").meta or {})
    win.rate_selection(0)
    assert "rating" not in peek_file(lib / "a.fraw").meta
    assert library.stars(3) == "★★★☆☆"
    win.set_library_folder(lib)
    assert pump(app, lambda: win.library.item(0).data(int(Qt.ItemDataRole.UserRole) + 2) == "green")


def test_update_meta_inserts_when_missing(tmp_path):
    from conftest import make_small

    f = make_small()
    f.meta = {}
    p = tmp_path / "s.fraw"
    f.write(p)
    library.set_rating(p, 5)
    g = FrawFile.read(p)
    assert g.meta == {"rating": 5} and np.array_equal(g.raw, f.raw) and g.dev == f.dev


# ------------------------------------------------------------- exportación
def test_export_batch(tmp_path):
    from PIL import Image

    srcs = []
    for n in ("uno", "dos"):
        p = tmp_path / f"{n}.fraw"
        shutil.copyfile(GOLDEN, p)
        srcs.append(p)
    out_dir = tmp_path / "out"
    opts = ExportOptions(fmt="JPEG", quality=80, long_edge=32, template="{nombre}_{fecha}_{n}", folder=str(out_dir),
                         output_sharpen="screen")
    seen = []
    out = export_batch([ExportJob(p) for p in srcs], opts, lambda i, t: seen.append(i))
    assert len(out) == 2 and all(p.exists() for p in out)
    assert out[0].name.startswith("uno_") and out[0].name.endswith("_001.jpg")
    with Image.open(out[0]) as im:
        assert max(im.size) == 32
    opts16 = ExportOptions(fmt="TIFF", bits=16, folder=str(out_dir))
    out2 = export_batch([ExportJob(srcs[0])], opts16)
    assert out2[0].suffix == ".tif" and out2[0].stat().st_size > 64 * 48 * 6
    # cancelación
    assert export_batch([ExportJob(p) for p in srcs], opts, cancelled=lambda: True) == []
    assert render_name("{nombre}-{n}", Path("x.fraw"), 7, {}, 12) == "x-007"


def test_export_selection_from_window(app, win, tmp_path):
    a = tmp_path / "a.fraw"
    b = tmp_path / "b.fraw"
    shutil.copyfile(GOLDEN, a)
    shutil.copyfile(GOLDEN, b)
    win.open_path(a, sync=True)
    win.set_param("exposure_ev", -1.0)
    opts = ExportOptions(fmt="PNG", folder=str(tmp_path / "exp"), template="{nombre}")
    win.export_selection([a, b], opts, sync=True)
    assert sorted(p.name for p in win.last_batch) == ["a.png", "b.png"]


def test_shortcuts_and_layout(app, win):
    rows = win.shortcut_rows
    keys = {r[2] for r in rows}
    assert {"Ctrl+Z", "R", "J", "\\", "Ctrl+Shift+C", "Ctrl+Shift+V"} <= keys
    menus = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert menus == ["Archivo", "Editar", "Ver", "Revelado", "Herramientas", "Ayuda"]
    win.dock_presets.hide()
    win.reset_layout()
    assert win.dock_presets.isVisible()


# ---------------------------------------------------------------- latencia
def test_latency_real_raw_draft_vs_final(app, win, tmp_path, capsys):
    srcs = sorted(RAW_DIR.glob("DSC01542.ARW")) if RAW_DIR.is_dir() else []
    if not srcs:
        pytest.skip("sin RAW de prueba")
    pytest.importorskip("rawpy")
    from fraw.ingest import ingest_to_file

    p = tmp_path / "r.fraw"
    ingest_to_file(srcs[0], p, thumbnail=False)
    win.open_path(p, sync=True)
    win.set_param("clarity", 0.2)
    win.set_param("hsl_orange_s", 0.2)
    win.render_now()
    sl = win.panel.rows["exposure_ev"].slider
    draft, final = [], []
    for i in range(12):
        n0 = win.render_count
        t0 = time.perf_counter()
        sl.setValue(-1 + 0.1 * i, final=False)  # arrastre → borrador
        assert pump(app, lambda: win.render_count > n0, timeout=30)
        draft.append((time.perf_counter() - t0) * 1000)
    for i in range(6):
        n0 = win.render_count
        t0 = time.perf_counter()
        sl.setValue(0.2 + 0.1 * i, final=True)  # soltar → final
        assert pump(app, lambda: win.render_count > n0, timeout=30)
        final.append((time.perf_counter() - t0) * 1000)
    md, mf = statistics.median(draft), statistics.median(final)
    pd = statistics.median(win.render_ms["draft"][-12:])
    pf = statistics.median(win.render_ms["final"][-6:])
    with capsys.disabled():
        print(f"\n[gui-pro] {srcs[0].name} {win.engine.base_size()}: control->imagen borrador mediana {md:.1f} ms "
              f"(preview {pd:.1f} ms) · final mediana {mf:.1f} ms (preview {pf:.1f} ms) · "
              f"motor sin: {', '.join(win.engine.missing()) or '—'}")
    assert md < 2000 and mf < 5000
