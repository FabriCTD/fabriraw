"""Panel de revelado generado desde :data:`fraw.devparams.SCHEMA` / :data:`GROUPS`.

Cada categoría es una sección plegable (:class:`GroupSection`) con: título,
indicador de cambios, interruptor "efecto activo" (bypass temporal: la vista
previa usa el grupo en su defecto sin perder los valores) y botón restablecer.
Dentro, subsecciones por ``ParamSpec.section``. Editores especiales: balance de
blancos (modo + Kelvin estimado + cuentagotas + Auto), curva de tonos, HSL con
pestañas y herramienta de ajuste dirigido, ruedas de gradación, herramientas de
geometría. Arriba, un buscador filtra ajustes por nombre.

Toda edición sale por ``edited(updates: dict, label: str, merge_key: str, final: bool)``.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..devparams import CURVE_KEYS, GROUPS, SCHEMA, SPECS, DevParams
from .colorwheel import ColorWheel
from .controls import (
    BoolRow,
    EnumControl,
    ParamRow,
    ParamSlider,
    ToggleSwitch,
    ValueField,
    display_name,
    fold,
    format_value,
    history_label,
    make_control,
)
from .curves import CurveEditor

__all__ = ["DevelopPanel", "GroupSection", "WheelBlock", "GRADE_ZONES"]

GRADE_ZONES = (("shadows", "Sombras"), ("midtones", "Medios tonos"), ("highlights", "Iluminaciones"), ("global", "Global"))
HSL_TABS = (("h", "Tono"), ("s", "Saturación"), ("l", "Luminancia"), ("all", "Todo"))
DEFAULT_EXPANDED = {"basic", "tone_curve"}


def _tool(text: str, tip: str, checkable: bool = False, obj: str = "") -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    if obj:
        b.setObjectName(obj)
    return b


class SubSection(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = title
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 2)
        lay.setSpacing(1)
        self.header = QLabel(title.upper())
        self.header.setObjectName("sectionTitle")
        lay.addWidget(self.header)
        self.body = QVBoxLayout()
        self.body.setSpacing(1)
        lay.addLayout(self.body)
        self.items: list[tuple[QWidget, str]] = []

    def add(self, w: QWidget, hay: str) -> None:
        self.body.addWidget(w)
        self.items.append((w, fold(hay)))


class GroupSection(QFrame):
    resetRequested = Signal(str)
    enabledToggled = Signal(str, bool)

    def __init__(self, key: str, label: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.key, self.label = key, label
        self.setObjectName("groupSection")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QFrame()
        head.setObjectName("groupHeader")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(8, 5, 6, 5)
        hl.setSpacing(6)
        self.btn = QPushButton()
        self.btn.setObjectName("groupTitle")
        self.btn.setCheckable(True)
        self.btn.setFlat(True)
        self.btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.dot = QLabel("●")
        self.dot.setObjectName("changedDot")
        self.dot.setToolTip("Este grupo tiene ajustes modificados")
        self.dot.setVisible(False)
        self.switch = ToggleSwitch()
        self.switch.setChecked(True)
        self.switch.setToolTip("Ver / ocultar el efecto de este grupo (no cambia los valores)")
        self.btn_reset = _tool("↺", f"Restablecer «{label}»", obj="groupReset")
        hl.addWidget(self.btn, 1)
        hl.addWidget(self.dot)
        hl.addWidget(self.switch)
        hl.addWidget(self.btn_reset)
        lay.addWidget(head)
        self.body = QWidget()
        self.body.setObjectName("groupBody")
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(10, 2, 10, 10)
        self.body_lay.setSpacing(2)
        lay.addWidget(self.body)
        self.subs: dict[str, SubSection] = {}
        self.btn.toggled.connect(self.set_expanded)
        self.btn_reset.clicked.connect(lambda: self.resetRequested.emit(self.key))
        self.switch.toggled.connect(lambda on: self.enabledToggled.emit(self.key, bool(on)))
        self.set_expanded(False)

    def set_expanded(self, on: bool) -> None:
        self.btn.blockSignals(True)
        self.btn.setChecked(on)
        self.btn.blockSignals(False)
        self.btn.setText(("▾  " if on else "▸  ") + self.label)
        self.body.setVisible(on)

    def expanded(self) -> bool:
        return self.btn.isChecked()

    def sub(self, title: str) -> SubSection:
        s = self.subs.get(title)
        if s is None:
            s = SubSection(title)
            self.subs[title] = s
            self.body_lay.addWidget(s)
        return s

    def set_changed(self, changed: bool) -> None:
        self.dot.setVisible(changed)

    def set_bypassed(self, bypassed: bool) -> None:
        self.switch.blockSignals(True)
        self.switch.setChecked(not bypassed)
        self.switch.blockSignals(False)
        self.switch.update()
        self.btn.setProperty("bypassed", bypassed)
        self.btn.style().unpolish(self.btn)
        self.btn.style().polish(self.btn)


class WheelBlock(QWidget):
    """Rueda + Tono/Saturación numéricos + Luminancia para una zona de gradación."""

    edited = Signal(dict, str, str, bool)

    def __init__(self, zone: str, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.zone, self.title = zone, title
        self.kh, self.ks, self.kl = f"grade_{zone}_hue", f"grade_{zone}_sat", f"grade_{zone}_lum"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 6)
        lay.setSpacing(3)
        t = QLabel(title)
        t.setObjectName("wheelTitle")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        self.wheel = ColorWheel(size=118)
        lay.addWidget(self.wheel)
        row = QHBoxLayout()
        row.setSpacing(3)
        self.f_hue = ValueField(SPECS[self.kh])
        self.f_sat = ValueField(SPECS[self.ks])
        for f in (self.f_hue, self.f_sat):
            f.setFixedWidth(52)
        lh, ls = QLabel("T"), QLabel("S")
        lh.setObjectName("muted")
        ls.setObjectName("muted")
        lh.setToolTip("Tono")
        ls.setToolTip("Saturación")
        row.addWidget(lh)
        row.addWidget(self.f_hue)
        row.addWidget(ls)
        row.addWidget(self.f_sat)
        lay.addLayout(row)
        lrow = QHBoxLayout()
        lrow.setSpacing(3)
        ll = QLabel("L")
        ll.setObjectName("muted")
        ll.setToolTip("Luminancia")
        self.lum = ParamSlider(SPECS[self.kl], compact=True)
        self.f_lum = ValueField(SPECS[self.kl])
        self.f_lum.setFixedWidth(44)
        lrow.addWidget(ll)
        lrow.addWidget(self.lum, 1)
        lrow.addWidget(self.f_lum)
        lay.addLayout(lrow)
        self.wheel.changed.connect(self._wheel)
        self.f_hue.committed.connect(lambda v, fin: self._emit({self.kh: v}, fin))
        self.f_sat.committed.connect(lambda v, fin: self._emit({self.ks: v}, fin))
        self.lum.changed.connect(lambda v, fin: (self.f_lum.show_value(v), self._emit({self.kl: v}, fin, "lum")))
        self.f_lum.committed.connect(lambda v, fin: (self.lum.set_silently(v), self._emit({self.kl: v}, fin, "lum")))

    def _wheel(self, hue: float, sat: float, final: bool) -> None:
        self.f_hue.show_value(hue)
        self.f_sat.show_value(sat)
        self._emit({self.kh: hue, self.ks: sat}, final)

    def _emit(self, upd: dict, final: bool, what: str = "color") -> None:
        if what == "lum":
            lbl = f"{self.title}: luminancia {format_value(SPECS[self.kl], upd[self.kl], unit=False)}"
        else:
            h = upd.get(self.kh, self.wheel.hue)
            s = upd.get(self.ks, self.wheel.sat)
            if self.kh in upd or self.ks in upd:
                self.wheel.set_value(h, s)
            lbl = f"{self.title}: color {h:.0f}° / {s * 100:.0f}"
        self.edited.emit(upd, f"Gradación · {lbl}", f"wheel_{self.zone}_{what}", final)

    def sync(self, p: DevParams) -> None:
        h, s, lum = getattr(p, self.kh), getattr(p, self.ks), getattr(p, self.kl)
        self.wheel.set_value(h, s)
        self.f_hue.show_value(h)
        self.f_sat.show_value(s)
        self.lum.set_silently(lum)
        self.f_lum.show_value(lum)

    # compatibilidad con el resto de controles
    def set_silently(self, _v: Any) -> None:  # pragma: no cover - sync() hace el trabajo
        pass


class DevelopPanel(QWidget):
    edited = Signal(dict, str, str, bool)  # updates, etiqueta de historial, clave de fusión, final
    groupReset = Signal(str)
    bypassChanged = Signal(str, bool)
    eyedropperToggled = Signal(bool)
    targetToggled = Signal(bool)
    cropToolToggled = Signal(bool)
    action = Signal(str)  # auto_tone | auto_wb | rotate_left | rotate_right | flip_h | flip_v

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.params = DevParams()
        self.as_shot: tuple[float, float] | None = None
        self.controls: dict[str, QWidget] = {}
        self.rows: dict[str, ParamRow] = {}
        self.sections: dict[str, GroupSection] = {}
        self._expanded_before_filter: dict[str, bool] | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        top = QHBoxLayout()
        top.setContentsMargins(8, 6, 8, 0)
        self.search = QLineEdit()
        self.search.setObjectName("search")
        self.search.setPlaceholderText("Buscar ajuste…  (Ctrl+F)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter)
        top.addWidget(self.search, 1)
        self.btn_expand = _tool("▾▾", "Expandir / plegar todo")
        self.btn_expand.clicked.connect(self._toggle_all)
        top.addWidget(self.btn_expand)
        outer.addLayout(top)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("panelInner")
        self.inner_lay = QVBoxLayout(inner)
        self.inner_lay.setContentsMargins(6, 0, 6, 6)
        self.inner_lay.setSpacing(6)
        self.scroll.setWidget(inner)
        outer.addWidget(self.scroll, 1)
        self.empty_label = QLabel("Ningún ajuste coincide con la búsqueda")
        self.empty_label.setObjectName("muted")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.hide()
        self.inner_lay.addWidget(self.empty_label)
        self._build()
        self.inner_lay.addStretch(1)

    # ================================================================ build
    def _build(self) -> None:
        for g in GROUPS:
            sec = GroupSection(g.key, g.label)
            sec.resetRequested.connect(self.groupReset)
            sec.enabledToggled.connect(lambda key, on: self.bypassChanged.emit(key, not on))
            sec.set_expanded(g.key in DEFAULT_EXPANDED)
            self.sections[g.key] = sec
            self.inner_lay.addWidget(sec)
            builder = getattr(self, f"_build_{g.key}", None)
            if builder is not None:
                builder(sec)
            self._build_generic(sec, [s for s in SCHEMA if s.group == g.key])
        self._hsl_tab(0)

    def _register(self, w: QWidget, key: str) -> None:
        self.controls[key] = w
        if isinstance(w, ParamRow):
            self.rows[key] = w
        if hasattr(w, "changed"):
            w.changed.connect(self._single)  # type: ignore[attr-defined]

    def _build_generic(self, sec: GroupSection, specs) -> None:
        for s in specs:
            if s.hidden or s.key in self.controls:
                continue
            w = make_control(s)
            if w is None:
                continue
            self._register(w, s.key)
            sec.sub(s.section or sec.label).add(w, f"{s.label} {display_name(s)} {s.section} {sec.label} {s.key}")

    def _single(self, key: str, value: Any, final: bool) -> None:
        spec = SPECS[key]
        upd = {key: value}
        if key in ("wb_kelvin", "wb_kelvin_tint") and self.params.wb_mode != "kelvin":
            upd["wb_mode"] = "kelvin"
            other = "wb_kelvin_tint" if key == "wb_kelvin" else "wb_kelvin"
            upd[other] = self.controls[other].value()  # type: ignore[attr-defined]
        if key == "wb_mode" and value == "kelvin" and self.params.wb_mode != "kelvin" and self.as_shot:
            k, t = self.as_shot
            upd["wb_kelvin"] = float(min(SPECS["wb_kelvin"].hi, max(SPECS["wb_kelvin"].lo, k)))  # type: ignore[type-var]
            upd["wb_kelvin_tint"] = float(min(150.0, max(-150.0, t)))
        self.edited.emit(upd, history_label(spec, value), key, final)

    # ---------------------------------------------------------- special: basic
    def _build_basic(self, sec: GroupSection) -> None:
        sub = sec.sub("Balance de blancos")
        w = make_control(SPECS["wb_mode"])
        self._register(w, "wb_mode")
        sub.add(w, "balance de blancos modo wb")
        tools = QWidget()
        tl = QHBoxLayout(tools)
        tl.setContentsMargins(128, 0, 0, 2)
        tl.setSpacing(4)
        self.btn_picker = _tool("Cuentagotas", "Click en una zona neutra (gris/blanca) de la foto (W)", True)
        self.btn_auto_wb = _tool("Auto", "Balance de blancos automático")
        self.btn_picker.toggled.connect(self.eyedropperToggled)
        self.btn_auto_wb.clicked.connect(lambda: self.action.emit("auto_wb"))
        tl.addWidget(self.btn_picker)
        tl.addWidget(self.btn_auto_wb)
        tl.addStretch(1)
        sub.add(tools, "cuentagotas auto balance blancos")
        self.wb_info = QLabel("")
        self.wb_info.setObjectName("hint")
        sub.add(self.wb_info, "balance multiplicadores kelvin")
        for k in ("wb_kelvin", "wb_kelvin_tint", "temperature_shift", "tint_shift"):
            w = make_control(SPECS[k])
            self._register(w, k)
            sub.add(w, f"{SPECS[k].label} balance blancos temperatura matiz")
        tsub = sec.sub("Tono")
        tools2 = QWidget()
        t2 = QHBoxLayout(tools2)
        t2.setContentsMargins(128, 0, 0, 2)
        self.btn_auto_tone = _tool("Auto tono", "Ajusta exposición, contraste, luces y sombras automáticamente (Ctrl+U)")
        self.btn_auto_tone.clicked.connect(lambda: self.action.emit("auto_tone"))
        t2.addWidget(self.btn_auto_tone)
        t2.addStretch(1)
        tsub.add(tools2, "auto tono automatico exposicion")

    # ----------------------------------------------------- special: tone curve
    def _build_tone_curve(self, sec: GroupSection) -> None:
        sub = sec.sub("Curva de puntos")
        self.curve = CurveEditor()
        self.curve.changed.connect(self._curve_changed)
        for k in CURVE_KEYS:
            self.controls[k] = self.curve
        sub.add(self.curve, "curva de puntos tonos rgb rojo verde azul curve")

    def _curve_changed(self, key: str, pts, final: bool) -> None:
        self.edited.emit({key: [list(p) for p in pts]}, SPECS[key].label, key, final)

    # ------------------------------------------------------------ special: HSL
    def _build_hsl(self, sec: GroupSection) -> None:
        head = QWidget()
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 4, 0, 4)
        hl.setSpacing(6)
        self.hsl_tabs = QTabBar()
        self.hsl_tabs.setObjectName("chanTabs")
        self.hsl_tabs.setDrawBase(False)
        self.hsl_tabs.setExpanding(False)
        for _k, lbl in HSL_TABS:
            self.hsl_tabs.addTab(lbl)
        self.btn_target = _tool("◎", "Ajuste dirigido: arrastrá verticalmente sobre la foto para cambiar\n"
                                "el color que está bajo el cursor (T)", True, "targetTool")
        self.btn_target.toggled.connect(self.targetToggled)
        hl.addWidget(self.hsl_tabs, 1)
        hl.addWidget(self.btn_target)
        sec.body_lay.addWidget(head)
        self.hsl_tabs.currentChanged.connect(self._hsl_tab)
        for _c, title in HSL_TABS[:3]:
            sec.sub(title)
        self._hsl_sec = sec

    def _hsl_tab(self, i: int) -> None:
        comp = HSL_TABS[i][0]
        for c, title in HSL_TABS[:3]:
            sub = self._hsl_sec.subs.get(title)
            if sub is not None:
                sub.setVisible(comp == "all" or comp == c)
                sub.header.setVisible(comp == "all")

    def hsl_component(self) -> str:
        c = HSL_TABS[self.hsl_tabs.currentIndex()][0]
        return "s" if c == "all" else c

    # -------------------------------------------------------- special: grading
    def _build_grading(self, sec: GroupSection) -> None:
        box = QWidget()
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        self.wheels: dict[str, WheelBlock] = {}
        for i, (zone, title) in enumerate(GRADE_ZONES):
            wb = WheelBlock(zone, title)
            wb.edited.connect(self.edited)
            self.wheels[zone] = wb
            grid.addWidget(wb, i // 2, i % 2)
            for k in (wb.kh, wb.ks, wb.kl):
                self.controls[k] = wb
        sec.sub("Ruedas de color").add(box, "rueda gradacion color sombras medios tonos iluminaciones global tono saturacion luminancia")

    # ------------------------------------------------------- special: geometry
    def _build_geometry(self, sec: GroupSection) -> None:
        tools = QWidget()
        tl = QHBoxLayout(tools)
        tl.setContentsMargins(0, 4, 0, 4)
        tl.setSpacing(4)
        self.btn_crop = _tool("Recortar", "Herramienta de recorte y enderezado (R)\nEnter aplica · Esc cancela", True, "cropTool")
        self.btn_crop.toggled.connect(self.cropToolToggled)
        tl.addWidget(self.btn_crop)
        for text, tip, act in (("⟲ 90°", "Girar 90° a la izquierda (Ctrl+[)", "rotate_left"),
                               ("⟳ 90°", "Girar 90° a la derecha (Ctrl+])", "rotate_right"),
                               ("⇆", "Voltear horizontal", "flip_h"), ("⇅", "Voltear vertical", "flip_v")):
            b = _tool(text, tip)
            b.clicked.connect(lambda _c=False, a=act: self.action.emit(a))
            tl.addWidget(b)
        tl.addStretch(1)
        sec.sub("Herramientas").add(tools, "recortar recorte girar voltear enderezar rotar crop")

    # ================================================================ state
    def sync(self, params: DevParams, bypass: set[str] | None = None) -> None:
        """Refleja ``params`` en todos los controles sin emitir señales."""
        self.params = DevParams.coerce(params)
        p = self.params
        for key, w in self.controls.items():
            if isinstance(w, (ParamRow, BoolRow, EnumControl)):
                w.set_silently(getattr(p, key))
        self.curve.set_curves({k: [list(pt) for pt in getattr(p, k)] for k in CURVE_KEYS})
        for wb in self.wheels.values():
            wb.sync(p)
        if p.wb_mode != "kelvin" and self.as_shot:
            self.rows["wb_kelvin"].set_silently(min(SPECS["wb_kelvin"].hi, max(SPECS["wb_kelvin"].lo, self.as_shot[0])))  # type: ignore[type-var]
            self.rows["wb_kelvin_tint"].set_silently(min(150.0, max(-150.0, self.as_shot[1])))
            for k in ("wb_kelvin", "wb_kelvin_tint"):  # es un estimado, no un ajuste modificado
                self.rows[k].mark_changed(False)
        if p.wb_mode == "custom":
            r, g, b = p.wb_custom
            self.wb_info.setText(f"Multiplicadores R {r:.3f} · G {g:.3f} · B {b:.3f}")
        elif p.wb_mode != "kelvin" and self.as_shot:
            self.wb_info.setText(f"Estimado de la toma: {self.as_shot[0]:.0f} K, matiz {self.as_shot[1]:+.0f}")
        else:
            self.wb_info.setText("")
        self.wb_info.setVisible(bool(self.wb_info.text()))
        self.rows["gamma"].setEnabled(self.isEnabled() and p.gamma_mode == "power")
        for g, sec in self.sections.items():
            sec.set_changed(not p.group_is_default(g))
            if bypass is not None:
                sec.set_bypassed(g in bypass)

    def set_histogram(self, hist: dict | None) -> None:
        self.curve.set_histogram(hist)

    def expanded_groups(self) -> list[str]:
        return [g for g, s in self.sections.items() if s.expanded()]

    def set_expanded_groups(self, groups) -> None:
        gs = set(groups)
        for g, s in self.sections.items():
            s.set_expanded(g in gs)

    def _toggle_all(self) -> None:
        any_closed = any(not s.expanded() for s in self.sections.values())
        for s in self.sections.values():
            s.set_expanded(any_closed)

    def reveal(self, group: str) -> None:
        sec = self.sections.get(group)
        if sec is not None:
            sec.set_expanded(True)
            self.scroll.ensureWidgetVisible(sec, 0, 0)

    # ================================================================ filter
    def filter(self, text: str) -> None:
        q = fold(text.strip())
        if q and self._expanded_before_filter is None:
            self._expanded_before_filter = {g: s.expanded() for g, s in self.sections.items()}
        any_match = False
        for g, sec in self.sections.items():
            group_hit = bool(q) and q in fold(sec.label)
            sec_any = False
            for sub in sec.subs.values():
                sub_any = False
                for w, hay in sub.items:
                    ok = not q or group_hit or q in hay or q in fold(sub.title)
                    w.setVisible(ok)
                    sub_any |= ok
                sub.setVisible(sub_any or not q)
                sec_any |= sub_any
            if not q:
                sec.setVisible(True)
                continue
            sec.setVisible(sec_any)
            if sec_any:
                sec.set_expanded(True)
                any_match = True
        if not q:
            self._hsl_tab(self.hsl_tabs.currentIndex())
            if self._expanded_before_filter is not None:
                for g, on in self._expanded_before_filter.items():
                    self.sections[g].set_expanded(on)
                self._expanded_before_filter = None
        else:
            for _c, title in HSL_TABS[:3]:  # en búsqueda se ven las tres
                sub = self._hsl_sec.subs.get(title)
                if sub is not None:
                    sub.header.setVisible(True)
        self.empty_label.setVisible(bool(q) and not any_match)

    def matching_keys(self) -> list[str]:
        """Claves cuyo control queda visible con el filtro actual (para tests)."""
        out = []
        for sec in self.sections.values():
            if not sec.isVisibleTo(self):
                continue
            for sub in sec.subs.values():
                for w, _hay in sub.items:
                    if not w.isHidden() and not sub.isHidden():
                        out += [k for k, c in self.controls.items() if c is w or w.isAncestorOf(c)]
        return out

