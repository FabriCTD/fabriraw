"""CTD Photo Studio — PySide6 GUI for FabriRAW (``ctd-studio``).

Rendering model
---------------
* Every slider change updates ``self.params`` and is throttled to at most one
  submission per 16 ms frame (leading edge + QTimer for the trailing edge). The
  latest state is submitted to a :class:`LatestWorker` (QThread) that always
  processes only the most recent request (``FRAWProcessor.preview``, cached
  half-size superpixel + LUTs, ~10–15 ms for 24 MP). The result (uint8 array)
  is wrapped by a QImage without copying and painted by :class:`ImageCanvas`.
* When the preview is visibly magnified (zoom ≥ 50 % of the real resolution),
  220 ms after the last change a second worker develops the *visible region*
  at full resolution with the chosen demosaic and the canvas draws it on top.
* Saving (Ctrl+S) only rewrites the DEV tail of the file via ``update_dev``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QByteArray, QPointF, QRectF, QSettings, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..devparams import DevParams
from ..format import MAGIC, SET_MAGIC, FrawFile, update_dev
from ..processor import FRAWProcessor
from . import winassoc
from .canvas import ImageCanvas
from .render import develop_region, neutral_wb_from_patch, oriented_size, patch_cam_rgb
from .widgets import Histogram, SliderRow, histogram_rgb
from .workers import LatestWorker, TaskThread, ThumbLoader, orientation_transform

__all__ = ["MainWindow", "main", "APP_NAME", "PREVIEW_SIDE"]

APP_NAME = "CTD Photo Studio"
ORG_NAME = "FabriCTD"
PREVIEW_SIDE = 1600
COALESCE_MS = 16
DETAIL_DELAY_MS = 220
DETAIL_MAX_PIXELS = 8_000_000
FRAW_EXTS = (".fraw", ".frw", ".fset")
GUI_DIR = Path(__file__).resolve().parent
SITE_URL = "https://fraw.fabrictd.com"
REPO_URL = "https://github.com/FabriCTD/fabriraw"

# key, label, lo, hi, step, fmt
SLIDERS: list[tuple[str, str, float, float, float, str]] = [
    ("exposure_ev", "Exposición (EV)", -4.0, 4.0, 0.01, "{:+.2f}"),
    ("temperature_shift", "Temperatura", -1.0, 1.0, 0.01, "{:+.2f}"),
    ("tint_shift", "Tinte", -1.0, 1.0, 0.01, "{:+.2f}"),
    ("contrast", "Contraste", -1.0, 1.0, 0.01, "{:+.2f}"),
    ("highlights", "Altas luces", -1.0, 1.0, 0.01, "{:+.2f}"),
    ("shadows", "Sombras", -1.0, 1.0, 0.01, "{:+.2f}"),
    ("saturation", "Saturación", 0.0, 2.0, 0.01, "{:.2f}"),
]
WB_MODES = [("as_shot", "Como se tomó"), ("daylight", "Luz día"), ("custom", "Personalizado")]
GAMMA_MODES = [("srgb", "sRGB"), ("power", "Potencia"), ("linear", "Lineal")]
DEMOSAICS = [("ahd", "AHD (calidad)"), ("bilinear", "Bilineal (rápido)")]


def _camera_exts() -> frozenset[str]:
    from ..ingest import SUPPORTED_EXTENSIONS  # noqa: PLC0415 (no rawpy import here)

    return SUPPORTED_EXTENSIONS


def resource_path(name: str) -> Path | None:
    """Locates a bundled resource (dev tree or PyInstaller onedir)."""
    cands = [GUI_DIR / name]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        cands += [exe_dir / name, Path(getattr(sys, "_MEIPASS", exe_dir)) / name]
    cands.append(GUI_DIR.parents[2] / name)  # repo root (THIRD_PARTY_NOTICES.md, LICENSE)
    for c in cands:
        if c.is_file():
            return c
    return None


def app_icon() -> QIcon:
    p = resource_path("icon.png")
    return QIcon(str(p)) if p else QIcon()


def _fmt_shutter(v: Any) -> str:
    try:
        s = float(v)
    except (TypeError, ValueError):
        return "—"
    if s <= 0:
        return "—"
    return f"1/{round(1 / s)} s" if s < 1 else f"{s:g} s"


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------
class ExportDialog(QDialog):
    def __init__(self, demosaic: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Exportar")
        lay = QFormLayout(self)
        self.fmt = QComboBox()
        self.fmt.addItems(["JPEG", "PNG", "TIFF"])
        self.bits = QComboBox()
        self.bits.addItems(["8 bits", "16 bits"])
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(95)
        self.demosaic = QComboBox()
        for k, label in DEMOSAICS:
            self.demosaic.addItem(label, k)
        self.demosaic.setCurrentIndex(max(0, [k for k, _ in DEMOSAICS].index(demosaic)))
        self.half = QCheckBox("Mitad de resolución (rápido)")
        lay.addRow("Formato", self.fmt)
        lay.addRow("Profundidad", self.bits)
        lay.addRow("Calidad JPEG", self.quality)
        lay.addRow("Demosaico", self.demosaic)
        lay.addRow("", self.half)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Exportar…")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self.fmt.currentIndexChanged.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        jpeg = self.fmt.currentIndex() == 0
        if jpeg:
            self.bits.setCurrentIndex(0)
        self.bits.setEnabled(not jpeg)
        self.quality.setEnabled(jpeg)

    def options(self) -> dict:
        return {
            "ext": [".jpg", ".png", ".tif"][self.fmt.currentIndex()],
            "bits": 16 if self.bits.currentIndex() == 1 else 8,
            "quality": self.quality.value(),
            "demosaic": self.demosaic.currentData(),
            "half_size": self.half.isChecked(),
        }


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Acerca de {APP_NAME}")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(72, 72))
        top.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)
        text = QLabel(
            f"<h2 style='margin:0'>{APP_NAME}</h2>"
            f"<p style='color:#A6A3AD'>Versión {__version__} · formato FabriRAW 1.0 (.fraw / .frw / .fset)</p>"
            "<p><b>Hecho 100% con Claude (Anthropic) a partir de un prompt de FabriCTD.</b></p>"
            f"<p>Sitio: <a style='color:#F0B232' href='{SITE_URL}'>{SITE_URL}</a><br>"
            f"Código: <a style='color:#F0B232' href='{REPO_URL}'>{REPO_URL}</a></p>"
            "<p style='color:#A6A3AD'>Licencia MIT. Usa PySide6/Qt (LGPL-3.0), LibRaw vía rawpy, "
            "NumPy, SciPy y Pillow. Ver «Licencias» para los avisos completos.</p>"
        )
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.RichText)
        text.setOpenExternalLinks(True)
        top.addWidget(text, 1)
        lay.addLayout(top)
        row = QHBoxLayout()
        lic = QPushButton("Licencias…")
        lic.clicked.connect(lambda: LicensesDialog(self).exec())
        ok = QPushButton("Cerrar")
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        row.addWidget(lic)
        row.addStretch(1)
        row.addWidget(ok)
        lay.addLayout(row)


class LicensesDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Licencias de terceros")
        self.resize(760, 620)
        lay = QVBoxLayout(self)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        parts = []
        for name in ("THIRD_PARTY_NOTICES.md", "LICENSE"):
            p = resource_path(name)
            if p:
                parts.append(p.read_text(encoding="utf-8", errors="replace"))
        if parts:
            tb.setMarkdown(parts[0] + ("\n\n---\n\n## Licencia de CTD Photo Studio / FabriRAW\n\n```\n" + parts[1] + "\n```" if len(parts) > 1 else ""))
        else:
            tb.setPlainText(f"No se encontró THIRD_PARTY_NOTICES.md. Ver {REPO_URL}")
        lay.addWidget(tb)
        b = QPushButton("Cerrar")
        b.clicked.connect(self.accept)
        lay.addWidget(b, 0, Qt.AlignmentFlag.AlignRight)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    imageUpdated = Signal()

    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.settings = settings or QSettings(ORG_NAME, APP_NAME)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)

        # document state
        self.path: Path | None = None
        self.kind: str | None = None  # "fraw" | "fset"
        self.frames: list[FrawFile] = []
        self.frame_index = 0
        self.ffile: FrawFile | None = None
        self.proc: FRAWProcessor | None = None
        self.params = DevParams()
        self.saved_params: dict = DevParams().to_dict()
        self.dirty = False
        self.showing_before = False
        self._before_img: np.ndarray | None = None
        self._current_img: np.ndarray | None = None
        self._gen = 0
        self._tasks: set[TaskThread] = set()
        self._thumb_loader: ThumbLoader | None = None
        self.last_render_ms = 0.0
        self._req_t0: float | None = None
        self._last_submit = 0.0
        self.render_count = 0
        self.library_dir: Path | None = None
        self._loading: str | None = None

        self.render_worker = LatestWorker(self._render_job, "render")
        self.render_worker.result.connect(self._on_render)
        self.render_worker.failed.connect(self._on_worker_failed)
        self.detail_worker = LatestWorker(self._detail_job, "detail")
        self.detail_worker.result.connect(self._on_detail)
        self.detail_worker.failed.connect(self._on_worker_failed)

        self._coalesce = QTimer(self)
        self._coalesce.setSingleShot(True)
        self._coalesce.setInterval(COALESCE_MS)
        self._coalesce.timeout.connect(self._submit_render)
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._submit_detail)

        self._build_ui()
        self._build_menu()
        self._restore_settings()
        self._update_actions()

    # ================================================================== UI
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- top bar
        bar = QFrame()
        bar.setObjectName("topbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 14, 8)
        bl.setSpacing(8)
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(30, 30))
        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        self.file_label = QLabel("")
        self.file_label.setObjectName("appSubtitle")
        self.dirty_label = QLabel("")
        self.dirty_label.setContentsMargins(8, 1, 8, 1)
        bl.addWidget(logo)
        bl.addWidget(title)
        bl.addSpacing(10)
        bl.addWidget(self.file_label)
        bl.addWidget(self.dirty_label)
        bl.addStretch(1)
        self.btn_open = QPushButton("Abrir")
        self.btn_import = QPushButton("Importar RAW")
        self.btn_export = QPushButton("Exportar")
        self.btn_save = QPushButton("Guardar revelado")
        self.btn_save.setObjectName("primary")
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_import.clicked.connect(self.import_dialog)
        self.btn_export.clicked.connect(self.export_dialog)
        self.btn_save.clicked.connect(self.save_dev)
        self.btn_open.setToolTip("Abrir .fraw / .frw / .fset (Ctrl+O)")
        self.btn_import.setToolTip("Convertir RAW de cámara a .fraw (Ctrl+I)")
        self.btn_export.setToolTip("Exportar a JPEG / PNG / TIFF (Ctrl+E)")
        self.btn_save.setToolTip("Guardar el revelado no destructivo en el archivo (Ctrl+S)")
        for b in (self.btn_open, self.btn_import, self.btn_export, self.btn_save):
            bl.addWidget(b)
        root.addWidget(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.splitter = split

        # --- left: library
        left = QFrame()
        left.setObjectName("panel")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        hdr = QHBoxLayout()
        lt = QLabel("BIBLIOTECA")
        lt.setObjectName("panelTitle")
        self.btn_folder = QToolButton()
        self.btn_folder.setText("Carpeta…")
        self.btn_folder.clicked.connect(self.open_folder_dialog)
        hdr.addWidget(lt)
        hdr.addStretch(1)
        hdr.addWidget(self.btn_folder)
        ll.addLayout(hdr)
        self.folder_label = QLabel("Sin carpeta")
        self.folder_label.setObjectName("muted")
        self.folder_label.setWordWrap(True)
        ll.addWidget(self.folder_label)
        self.library = QListWidget()
        self.library.setViewMode(QListView.ViewMode.IconMode)
        self.library.setFlow(QListView.Flow.TopToBottom)
        self.library.setWrapping(False)
        self.library.setMovement(QListView.Movement.Static)
        self.library.setResizeMode(QListView.ResizeMode.Adjust)
        self.library.setIconSize(QSize(168, 112))
        self.library.setGridSize(QSize(190, 158))
        self.library.setSpacing(4)
        self.library.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.library.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.library.setWordWrap(True)
        self.library.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.library.itemActivated.connect(self._library_activated)
        self.library.itemClicked.connect(self._library_activated)
        ll.addWidget(self.library, 1)
        left.setMinimumWidth(222)
        split.addWidget(left)

        # --- center: canvas + zoom bar + frame strip
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        self.canvas = ImageCanvas()
        self.canvas.zoomChanged.connect(self._on_zoom)
        self.canvas.viewChanged.connect(self._schedule_detail)
        self.canvas.picked.connect(self._on_pick)
        cl.addWidget(self.canvas, 1)
        zb = QHBoxLayout()
        zb.setContentsMargins(8, 0, 8, 6)
        self.btn_fit = QToolButton()
        self.btn_fit.setText("Ajustar")
        self.btn_fit.clicked.connect(self.canvas.fit)
        self.btn_100 = QToolButton()
        self.btn_100.setText("100%")
        self.btn_100.clicked.connect(self.canvas.zoom_100)
        self.zoom_label = QLabel("—")
        self.zoom_label.setObjectName("muted")
        self.btn_before = QPushButton("Antes / Después")
        self.btn_before.setToolTip("Mantener presionado para ver el original (sin ajustes)")
        self.btn_before.pressed.connect(lambda: self.show_before(True))
        self.btn_before.released.connect(lambda: self.show_before(False))
        self.render_label = QLabel("")
        self.render_label.setObjectName("muted")
        zb.addWidget(self.btn_fit)
        zb.addWidget(self.btn_100)
        zb.addWidget(self.zoom_label)
        zb.addStretch(1)
        zb.addWidget(self.render_label)
        zb.addSpacing(12)
        zb.addWidget(self.btn_before)
        cl.addLayout(zb)
        self.strip = QListWidget()
        self.strip.setFlow(QListView.Flow.LeftToRight)
        self.strip.setViewMode(QListView.ViewMode.IconMode)
        self.strip.setIconSize(QSize(96, 64))
        self.strip.setFixedHeight(112)
        self.strip.setWrapping(False)
        self.strip.setMovement(QListView.Movement.Static)
        self.strip.currentRowChanged.connect(self._strip_changed)
        self.strip.hide()
        cl.addWidget(self.strip)
        split.addWidget(center)

        # --- right: histogram, metadata, development
        right = QFrame()
        right.setObjectName("panel")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 10, 10, 10)
        t = QLabel("HISTOGRAMA")
        t.setObjectName("panelTitle")
        rl.addWidget(t)
        self.histogram = Histogram()
        rl.addWidget(self.histogram)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 6, 4, 0)
        t = QLabel("METADATOS")
        t.setObjectName("panelTitle")
        il.addWidget(t)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        self.meta_labels: dict[str, QLabel] = {}
        for i, (k, label) in enumerate(
            [("camera", "Cámara"), ("lens", "Lente"), ("iso", "ISO"), ("shutter", "Velocidad"),
             ("aperture", "Diafragma"), ("focal", "Focal"), ("size", "Tamaño"), ("date", "Fecha")]
        ):
            a = QLabel(label)
            a.setObjectName("muted")
            b = QLabel("—")
            b.setObjectName("metaValue")
            b.setWordWrap(True)
            b.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(a, i, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(b, i, 1)
            self.meta_labels[k] = b
        grid.setColumnStretch(1, 1)
        il.addLayout(grid)
        il.addSpacing(10)

        t = QLabel("REVELADO")
        t.setObjectName("panelTitle")
        il.addWidget(t)
        wb_row = QHBoxLayout()
        wl = QLabel("Balance de blancos")
        self.wb_combo = QComboBox()
        for k, label in WB_MODES:
            self.wb_combo.addItem(label, k)
        self.wb_combo.currentIndexChanged.connect(self._wb_changed)
        self.btn_picker = QToolButton()
        self.btn_picker.setText("Cuentagotas")
        self.btn_picker.setCheckable(True)
        self.btn_picker.setToolTip("Click en una zona neutra (gris/blanca) de la imagen")
        self.btn_picker.toggled.connect(self.canvas.set_picker)
        wb_row.addWidget(self.wb_combo, 1)
        wb_row.addWidget(self.btn_picker)
        il.addWidget(wl)
        il.addLayout(wb_row)
        self.wb_custom_label = QLabel("")
        self.wb_custom_label.setObjectName("muted")
        il.addWidget(self.wb_custom_label)

        self.sliders: dict[str, SliderRow] = {}
        defaults = DevParams()
        for key, label, lo, hi, step, fmt in SLIDERS:
            row = SliderRow(label, lo, hi, getattr(defaults, key), step, fmt)
            row.valueChanged.connect(lambda v, k=key: self._param_changed(k, v))
            self.sliders[key] = row
            il.addWidget(row)

        g = QFormLayout()
        g.setContentsMargins(0, 6, 0, 0)
        self.gamma_combo = QComboBox()
        for k, label in GAMMA_MODES:
            self.gamma_combo.addItem(label, k)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(1.0, 4.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(2.2)
        gr = QHBoxLayout()
        gr.addWidget(self.gamma_combo, 1)
        gr.addWidget(self.gamma_spin)
        self.gamma_combo.currentIndexChanged.connect(self._gamma_changed)
        self.gamma_spin.valueChanged.connect(self._gamma_changed)
        g.addRow("Curva (gamma)", gr)
        self.demosaic_combo = QComboBox()
        for k, label in DEMOSAICS:
            self.demosaic_combo.addItem(label, k)
        self.demosaic_combo.currentIndexChanged.connect(self._demosaic_changed)
        self.demosaic_combo.setToolTip("Demosaico para exportar y para el detalle al 100%")
        g.addRow("Demosaico", self.demosaic_combo)
        il.addLayout(g)
        self.btn_reset_all = QPushButton("Restablecer todo")
        self.btn_reset_all.clicked.connect(self.reset_all)
        il.addWidget(self.btn_reset_all)
        il.addStretch(1)
        scroll.setWidget(inner)
        rl.addWidget(scroll, 1)
        right.setMinimumWidth(300)
        split.addWidget(right)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([230, 880, 330])
        wrap = QWidget()
        wl2 = QHBoxLayout(wrap)
        wl2.setContentsMargins(8, 8, 8, 8)
        wl2.addWidget(split)
        root.addWidget(wrap, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Listo")
        self.resize(1440, 900)

    def _build_menu(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&Archivo")
        self.act_open = self._act(m, "Abrir…", self.open_dialog, QKeySequence.StandardKey.Open)
        self._act(m, "Abrir carpeta…", self.open_folder_dialog, "Ctrl+Shift+O")
        self._act(m, "Importar RAW…", self.import_dialog, "Ctrl+I")
        m.addSeparator()
        self.act_save = self._act(m, "Guardar revelado", self.save_dev, QKeySequence.StandardKey.Save)
        self.act_export = self._act(m, "Exportar…", self.export_dialog, "Ctrl+E")
        m.addSeparator()
        self._act(m, "Salir", self.close, QKeySequence.StandardKey.Quit)
        m = mb.addMenu("&Ver")
        self._act(m, "Ajustar a la ventana", self.canvas.fit, "Ctrl+0")
        self._act(m, "100%", self.canvas.zoom_100, "Ctrl+1")
        self._act(m, "Acercar", lambda: self.canvas.zoom_at(1.25, QPointF(self.canvas.width() / 2, self.canvas.height() / 2)), QKeySequence.StandardKey.ZoomIn)
        self._act(m, "Alejar", lambda: self.canvas.zoom_at(0.8, QPointF(self.canvas.width() / 2, self.canvas.height() / 2)), QKeySequence.StandardKey.ZoomOut)
        m = mb.addMenu("&Herramientas")
        self.act_assoc = self._act(m, "Asociar archivos .fraw/.frw/.fset con CTD Photo Studio", self.associate)
        self.act_unassoc = self._act(m, "Quitar asociación", self.unassociate)
        if not winassoc.available():
            for a in (self.act_assoc, self.act_unassoc):
                a.setEnabled(False)
                a.setToolTip("Disponible sólo en el .exe portable de Windows")
        m = mb.addMenu("A&yuda")
        self._act(m, "Acerca de CTD Photo Studio", lambda: AboutDialog(self).exec())
        self._act(m, "Licencias de terceros", lambda: LicensesDialog(self).exec())
        self._act(m, "Sitio web", lambda: QDesktopServices.openUrl(QUrl(SITE_URL)))

    def _act(self, menu, text: str, slot: Callable, shortcut=None) -> QAction:
        a = QAction(text, self)
        if shortcut is not None:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    # ============================================================ settings
    def _restore_settings(self) -> None:
        g = self.settings.value("geometry")
        if isinstance(g, (QByteArray, bytes)):
            self.restoreGeometry(g)
        s = self.settings.value("splitter")
        if isinstance(s, (QByteArray, bytes)):
            self.splitter.restoreState(s)
        last = self.settings.value("last_folder", "")
        if last and os.path.isdir(str(last)):
            self.set_library_folder(Path(str(last)))

    def _save_settings(self) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("splitter", self.splitter.saveState())
        if self.library_dir:
            self.settings.setValue("last_folder", str(self.library_dir))
        self.settings.sync()

    def _last_dir(self) -> str:
        return str(self.settings.value("last_folder", "") or (self.path.parent if self.path else Path.home()))

    # ============================================================== tasks
    def run_task(self, fn: Callable, on_done: Callable[[Any], None], *, sync: bool = False,
                 progress: QProgressDialog | None = None, on_fail: Callable[[str], None] | None = None) -> TaskThread | None:
        """Runs ``fn(report, cancelled)`` in a QThread (or inline when ``sync``)."""
        fail = on_fail or (lambda msg: QMessageBox.critical(self, APP_NAME, msg))
        if sync:
            try:
                r = fn(lambda i, t="": None, lambda: False)
            except Exception as e:  # noqa: BLE001
                from .workers import _describe  # noqa: PLC0415

                fail(_describe(e))
                return None
            on_done(r)
            return None
        t = TaskThread(fn)
        self._tasks.add(t)

        def finished() -> None:
            self._tasks.discard(t)
            if progress is not None:
                progress.close()
            t.deleteLater()

        t.done.connect(on_done)
        t.failed.connect(fail)
        t.finished.connect(finished)
        if progress is not None:
            t.progress.connect(lambda i, txt: (progress.setValue(i), progress.setLabelText(txt)))
            progress.canceled.connect(t.cancel)
            progress.show()
        t.start()
        return t

    def _busy(self, text: str, maximum: int = 0) -> QProgressDialog:
        d = QProgressDialog(text, "Cancelar", 0, maximum, self)
        d.setWindowTitle(APP_NAME)
        d.setWindowModality(Qt.WindowModality.WindowModal)
        d.setMinimumDuration(0)
        d.setAutoClose(False)
        d.setAutoReset(False)
        return d

    # ============================================================= loading
    def confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(
            self,
            APP_NAME,
            f"Hay cambios de revelado sin guardar en {self.path.name if self.path else 'el archivo'}.\n¿Guardarlos?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if r == QMessageBox.StandardButton.Save:
            return self.save_dev()
        return r == QMessageBox.StandardButton.Discard

    def open_path(self, path: str | os.PathLike, *, sync: bool = False) -> None:
        p = Path(path)
        ext = p.suffix.lower()
        if ext in _camera_exts():
            self.import_files([str(p)], sync=sync)
            return
        if not self.confirm_discard():
            return
        self.statusBar().showMessage(f"Abriendo {p.name}…")
        self._loading = str(p)

        def work(_report, _cancelled):
            with open(p, "rb") as fh:
                magic = fh.read(8)
            if magic == SET_MAGIC:
                from ..fset import FrawSet  # noqa: PLC0415

                frames = FrawSet.read(p).frames
                kind = "fset"
            elif magic == MAGIC:
                frames = [FrawFile.read(p)]
                kind = "fraw"
            else:
                raise ValueError(f"{p.name} no es un archivo FabriRAW")
            proc = FRAWProcessor(frames[0])
            params = DevParams.from_dict(frames[0].dev)
            first = proc.preview(params, PREVIEW_SIDE)  # warms the cache off the UI thread
            return {"path": p, "kind": kind, "frames": frames, "proc": proc, "params": params, "img": first}

        def failed(msg: str) -> None:
            self._loading = None
            self.statusBar().showMessage("Error al abrir")
            QMessageBox.critical(self, APP_NAME, msg)

        self.run_task(work, self._on_loaded, sync=sync, on_fail=failed)

    def _on_loaded(self, r: dict) -> None:
        self._loading = None
        self.path, self.kind, self.frames = r["path"], r["kind"], r["frames"]
        self.frame_index = 0
        self._set_document(r["frames"][0], r["proc"], r["params"], r["img"])
        self._fill_strip()
        if self.library_dir is None or self.library_dir != self.path.parent:
            self.set_library_folder(self.path.parent)
        self._select_library(self.path)
        self.statusBar().showMessage(f"Abierto {self.path}")

    def _set_document(self, f: FrawFile, proc: FRAWProcessor, params: DevParams, img: np.ndarray | None) -> None:
        self._gen += 1
        self.ffile, self.proc = f, proc
        self.params = params
        self.saved_params = params.to_dict()
        self._before_img = None
        self._set_dirty(False)
        self._sync_controls()
        self._fill_meta()
        self.canvas.clear_tile()
        if img is not None:
            self._show_array(img)
            self.histogram.set_data(histogram_rgb(img))
        else:
            self.request_render()
        self._update_actions()

    def image_size(self) -> tuple[int, int]:
        if self.proc is None:
            return (0, 0)
        h, w = self.proc.norm.shape
        return oriented_size(w, h, self.proc.orientation)

    def _show_array(self, arr: np.ndarray) -> None:
        self._current_img = arr
        if not self.showing_before:
            self.canvas.set_image(arr, self.image_size())
        self.imageUpdated.emit()

    def _fill_meta(self) -> None:
        m = (self.ffile.meta if self.ffile else {}) or {}
        cam = " ".join(str(m.get(k, "")).strip() for k in ("make", "model")).strip()
        w, h = self.image_size()
        vals = {
            "camera": cam or "—",
            "lens": str(m.get("lens") or "—"),
            "iso": str(m.get("iso")) if m.get("iso") is not None else "—",
            "shutter": _fmt_shutter(m.get("shutter_s")),
            "aperture": f"f/{float(m['aperture']):g}" if isinstance(m.get("aperture"), (int, float)) else "—",
            "focal": f"{float(m['focal_mm']):g} mm" if isinstance(m.get("focal_mm"), (int, float)) else "—",
            "size": f"{w} × {h} px" if w else "—",
            "date": str(m.get("datetime") or "—").replace("T", " "),
        }
        for k, v in vals.items():
            self.meta_labels[k].setText(v)

    def _fill_strip(self) -> None:
        self.strip.blockSignals(True)
        self.strip.clear()
        if self.kind == "fset":
            for i, f in enumerate(self.frames):
                it = QListWidgetItem(f"Frame {i + 1}")
                img = None
                if f.thumbnail is not None:
                    from .workers import _decode_thumb  # noqa: PLC0415

                    t = f.thumbnail
                    img = _decode_thumb(t.format, (t.width, t.height), t.data)
                    tr = orientation_transform(f.head.orientation)
                    if img is not None and tr is not None:
                        img = img.transformed(tr)
                if img is not None:
                    it.setIcon(QIcon(QPixmap.fromImage(img)))
                self.strip.addItem(it)
            self.strip.setCurrentRow(0)
            self.strip.show()
        else:
            self.strip.hide()
        self.strip.blockSignals(False)

    def _strip_changed(self, row: int) -> None:
        if self.kind != "fset" or row < 0 or row == self.frame_index or row >= len(self.frames):
            return
        f = self.frames[row]
        self.frame_index = row

        def work(_r, _c):
            proc = FRAWProcessor(f)
            params = DevParams.from_dict(f.dev)
            return (proc, params, proc.preview(params, PREVIEW_SIDE))

        self.run_task(work, lambda r: self._set_document(f, *r))

    # ============================================================ library
    def set_library_folder(self, folder: Path) -> None:
        self.library_dir = folder
        self.folder_label.setText(folder.name or str(folder))
        self.folder_label.setToolTip(str(folder))
        self.settings.setValue("last_folder", str(folder))
        if self._thumb_loader is not None:
            self._thumb_loader.cancel()
            self._thumb_loader.wait(2000)
        self.library.clear()
        exts = set(FRAW_EXTS) | set(_camera_exts())
        try:
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
        except OSError:
            files = []
        self._library_items: dict[str, QListWidgetItem] = {}
        for p in files:
            raw = p.suffix.lower() not in FRAW_EXTS
            it = QListWidgetItem(p.name + ("\nRAW · importar" if raw else ""))
            it.setData(Qt.ItemDataRole.UserRole, str(p))
            it.setToolTip(str(p))
            self.library.addItem(it)
            self._library_items[str(p)] = it
        if not files:
            self.library.addItem(QListWidgetItem("(sin archivos .fraw ni RAW)"))
        loader = ThumbLoader([str(p) for p in files])
        loader.loaded.connect(self._on_thumb)
        self._thumb_loader = loader
        loader.start()

    def _on_thumb(self, path: str, img: QImage | None, info: dict) -> None:
        it = getattr(self, "_library_items", {}).get(path)
        if it is None:
            return
        if img is not None and not img.isNull():
            it.setIcon(QIcon(QPixmap.fromImage(img)))
        meta = (info or {}).get("meta") or {}
        cam = " ".join(str(meta.get(k, "")) for k in ("make", "model")).strip()
        extra = []
        if cam:
            extra.append(cam)
        if (info or {}).get("frames", 1) > 1:
            extra.append(f"ráfaga · {info['frames']} frames")
        if (info or {}).get("error"):
            extra.append("⚠ " + str(info["error"])[:60])
        if extra:
            it.setText(Path(path).name + "\n" + " · ".join(extra))

    def _select_library(self, path: Path) -> None:
        it = getattr(self, "_library_items", {}).get(str(path))
        if it is not None:
            self.library.blockSignals(True)
            self.library.setCurrentItem(it)
            self.library.blockSignals(False)

    def _library_activated(self, it: QListWidgetItem) -> None:
        p = it.data(Qt.ItemDataRole.UserRole)
        if p and (self.path is None or Path(p) != self.path) and p != self._loading:
            self.open_path(p)

    def open_folder_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Abrir carpeta", self._last_dir())
        if d:
            self.set_library_folder(Path(d))

    # ============================================================== dialogs
    def open_dialog(self) -> None:
        cams = " ".join(f"*{e}" for e in sorted(_camera_exts()))
        fn, _ = QFileDialog.getOpenFileName(
            self, "Abrir", self._last_dir(),
            f"FabriRAW (*.fraw *.frw *.fset);;RAW de cámara ({cams});;Todos (*.*)",
        )
        if fn:
            self.open_path(fn)

    def import_dialog(self) -> None:
        cams = " ".join(f"*{e}" for e in sorted(_camera_exts()))
        fns, _ = QFileDialog.getOpenFileNames(self, "Importar RAW de cámara", self._last_dir(), f"RAW de cámara ({cams})")
        if fns:
            self.import_files(fns)

    def import_files(self, srcs: list[str], *, outdir: str | None = None, sync: bool = False) -> None:
        if outdir is None and not sync:
            box = QMessageBox(self)
            box.setWindowTitle("Importar RAW")
            box.setText(f"Convertir {len(srcs)} archivo(s) a .fraw.\n¿Dónde guardar los .fraw?")
            b_next = box.addButton("Junto al original", QMessageBox.ButtonRole.AcceptRole)
            b_pick = box.addButton("Elegir carpeta…", QMessageBox.ButtonRole.ActionRole)
            box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is b_pick:
                outdir = QFileDialog.getExistingDirectory(self, "Carpeta destino", self._last_dir())
                if not outdir:
                    return
            elif box.clickedButton() is not b_next:
                return

        def dst_for(src: str) -> str:
            name = Path(src).with_suffix(".fraw").name
            return os.path.join(outdir, name) if outdir else str(Path(src).with_suffix(".fraw"))

        def work(report, cancelled):
            from ..ingest import ingest_to_file  # noqa: PLC0415

            done = []
            for i, s in enumerate(srcs):
                if cancelled():
                    break
                report(i, f"Convirtiendo {Path(s).name} ({i + 1}/{len(srcs)})…")
                d = dst_for(s)
                ingest_to_file(s, d)
                done.append(d)
            report(len(srcs), "Listo")
            return done

        prog = None if sync else self._busy("Importando…", len(srcs))

        def finished(done: list[str]) -> None:
            if not done:
                return
            self.statusBar().showMessage(f"Importados {len(done)} archivo(s)")
            first = Path(done[0])
            self.set_library_folder(first.parent)
            self.open_path(first, sync=sync)

        self.run_task(work, finished, sync=sync, progress=prog)

    def export_dialog(self) -> None:
        if self.proc is None:
            return
        dlg = ExportDialog(self.params.demosaic, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        o = dlg.options()
        base = self.path.stem if self.path else "imagen"
        if self.kind == "fset":
            base += f"_frame{self.frame_index + 1}"
        filters = {".jpg": "JPEG (*.jpg)", ".png": "PNG (*.png)", ".tif": "TIFF (*.tif)"}
        fn, _ = QFileDialog.getSaveFileName(self, "Exportar", str(Path(self._last_dir()) / (base + o["ext"])), filters[o["ext"]])
        if fn:
            if Path(fn).suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                fn += o["ext"]
            self.export_to(fn, **{k: v for k, v in o.items() if k != "ext"})

    def export_to(self, fn: str, bits: int = 8, quality: int = 95, demosaic: str | None = None,
                  half_size: bool = False, *, sync: bool = False) -> None:
        proc, params = self.proc, DevParams.coerce(self.params)
        assert proc is not None

        def work(report, _c):
            t0 = time.perf_counter()
            report(0, f"Revelando {Path(fn).name}…")
            proc.export(fn, params, bits=bits, quality=quality, demosaic=demosaic, half_size=half_size)
            return time.perf_counter() - t0

        prog = None if sync else self._busy(f"Exportando {Path(fn).name}…")
        if prog is not None:
            prog.setCancelButton(None)
        self.run_task(work, lambda dt: self.statusBar().showMessage(f"Exportado {fn} en {dt:.1f} s"), sync=sync, progress=prog)

    # ============================================================== params
    def _sync_controls(self) -> None:
        p = self.params
        for key, row in self.sliders.items():
            row.set_silently(getattr(p, key))
        for combo, val, items in (
            (self.wb_combo, p.wb_mode, WB_MODES),
            (self.gamma_combo, p.gamma_mode, GAMMA_MODES),
            (self.demosaic_combo, p.demosaic, DEMOSAICS),
        ):
            combo.blockSignals(True)
            combo.setCurrentIndex([k for k, _ in items].index(val))
            combo.blockSignals(False)
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(p.gamma)
        self.gamma_spin.blockSignals(False)
        self.gamma_spin.setEnabled(p.gamma_mode == "power")
        self._update_wb_label()

    def _update_wb_label(self) -> None:
        if self.params.wb_mode == "custom":
            r, g, b = self.params.wb_custom
            self.wb_custom_label.setText(f"R {r:.3f} · G {g:.3f} · B {b:.3f}")
        else:
            self.wb_custom_label.setText("")

    def _param_changed(self, key: str, value: Any) -> None:
        setattr(self.params, key, value)
        self._after_param_change()

    def _wb_changed(self, _i: int) -> None:
        self.params.wb_mode = self.wb_combo.currentData()
        self._update_wb_label()
        self._after_param_change()

    def _gamma_changed(self, *_a) -> None:
        self.params.gamma_mode = self.gamma_combo.currentData()
        self.params.gamma = float(self.gamma_spin.value())
        self.gamma_spin.setEnabled(self.params.gamma_mode == "power")
        self._after_param_change()

    def _demosaic_changed(self, _i: int) -> None:
        self.params.demosaic = self.demosaic_combo.currentData()
        self._after_param_change(preview=False)

    def _after_param_change(self, preview: bool = True) -> None:
        self._gen += 1
        self._set_dirty(DevParams.coerce(self.params).to_dict() != self.saved_params)
        self.canvas.clear_tile()
        if preview:
            self.request_render()
        self._schedule_detail()

    def set_param(self, key: str, value: Any) -> None:
        """Programmatic change (tests / scripting): moves the control like the user would."""
        if key in self.sliders:
            self.sliders[key].setValue(value)
        else:
            setattr(self.params, key, value)
            self._sync_controls()
            self._after_param_change()

    def reset_all(self) -> None:
        self.params = DevParams()
        self._sync_controls()
        self._after_param_change()

    def _on_pick(self, x: float, y: float) -> None:
        if self.proc is None:
            return
        cam = patch_cam_rgb(self.proc, x, y, 5)
        wb = neutral_wb_from_patch(cam)
        if wb is None:
            self.statusBar().showMessage("Cuentagotas: zona demasiado oscura o saturada, probá otra")
            return
        self.params.wb_mode = "custom"
        self.params.wb_custom = wb
        self.params.temperature_shift = 0.0
        self.params.tint_shift = 0.0
        self._sync_controls()
        self.btn_picker.setChecked(False)
        self.statusBar().showMessage(f"Balance personalizado: R {wb[0]:.3f} · G 1 · B {wb[2]:.3f}")
        self._after_param_change()

    def _set_dirty(self, d: bool) -> None:
        self.dirty = d
        if self.path is None:
            self.dirty_label.setText("")
            self.dirty_label.setObjectName("")
        elif d:
            self.dirty_label.setText("● Sin guardar")
            self.dirty_label.setObjectName("dirtyBadge")
        else:
            self.dirty_label.setText("✓ Guardado")
            self.dirty_label.setObjectName("savedBadge")
        self.dirty_label.style().unpolish(self.dirty_label)
        self.dirty_label.style().polish(self.dirty_label)
        self.dirty_label.adjustSize()
        self.dirty_label.updateGeometry()
        name = self.path.name if self.path else ""
        if self.kind == "fset" and self.path:
            name += f" — frame {self.frame_index + 1}/{len(self.frames)}"
        self.file_label.setText(name)
        self.setWindowTitle(f"{'• ' if d else ''}{name + ' — ' if name else ''}{APP_NAME}")
        self.setWindowModified(d)
        self._update_actions()

    def _update_actions(self) -> None:
        has = self.proc is not None
        can_save = has and self.kind == "fraw"
        for w in (self.btn_export, self.act_export) if hasattr(self, "act_export") else (self.btn_export,):
            w.setEnabled(has)
        self.btn_save.setEnabled(can_save and self.dirty)
        if hasattr(self, "act_save"):
            self.act_save.setEnabled(can_save)
        self.btn_save.setToolTip(
            "Guardar el revelado no destructivo en el archivo (Ctrl+S)" if self.kind != "fset"
            else "En ráfagas .fset el revelado no se guarda en el archivo (exportá el frame)"
        )
        for w in [self.btn_before, self.btn_picker, self.btn_reset_all, self.btn_fit, self.btn_100,
                  self.wb_combo, self.gamma_combo, self.demosaic_combo, *self.sliders.values()]:
            w.setEnabled(has)

    def save_dev(self) -> bool:
        if self.proc is None or self.path is None:
            return False
        if self.kind != "fraw":
            QMessageBox.information(self, APP_NAME, "Las ráfagas .fset no guardan revelado; exportá el frame.")
            return False
        d = DevParams.coerce(self.params).to_dict()
        try:
            update_dev(self.path, d)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"No se pudo guardar el revelado:\n{e}")
            return False
        self.saved_params = d
        if self.ffile is not None:
            self.ffile.dev = d
        self._set_dirty(False)
        self.statusBar().showMessage(f"Revelado guardado en {self.path.name} (sólo DEV; DATA intacto)")
        return True

    # ============================================================ rendering
    def request_render(self) -> None:
        """Leading-edge throttle: submits at once if the last submission is ≥ 16 ms old,
        otherwise arms the 16 ms timer (so at most one submission per frame). The
        worker itself also drops superseded requests (latest wins)."""
        if self.proc is None:
            return
        now = time.perf_counter()
        if self._req_t0 is None:
            self._req_t0 = now
        if self._coalesce.isActive():
            return
        since = (now - self._last_submit) * 1000.0
        if since >= COALESCE_MS:
            self._submit_render()
        else:
            self._coalesce.start(max(1, int(COALESCE_MS - since)))

    def _submit_render(self) -> None:
        if self.proc is None:
            return
        self._last_submit = time.perf_counter()
        self.render_worker.submit((self._gen, self.proc, DevParams.coerce(self.params), self._req_t0))
        self._req_t0 = None

    def _render_job(self, job):
        gen, proc, params, t0 = job
        ts = time.perf_counter()
        img = proc.preview(params, PREVIEW_SIDE)
        hist = histogram_rgb(img)
        return gen, proc, img, hist, (time.perf_counter() - ts) * 1000.0, t0

    def _on_render(self, r) -> None:
        gen, proc, img, hist, ms, t0 = r
        if proc is not self.proc:
            return
        self.render_count += 1
        self._show_array(img)
        self.histogram.set_data(hist)
        if t0 is not None:
            self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        self.render_label.setText(f"preview {ms:.0f} ms")

    def render_now(self) -> np.ndarray | None:
        """Synchronous render on the calling thread (tests / scripting)."""
        if self.proc is None:
            return None
        self._coalesce.stop()
        r = self._render_job((self._gen, self.proc, DevParams.coerce(self.params), time.perf_counter()))
        self._on_render(r)
        return r[2]

    def _on_worker_failed(self, msg: str) -> None:
        self.statusBar().showMessage("Error al procesar: " + msg.strip().splitlines()[-1])

    def _on_zoom(self, z: float) -> None:
        self.zoom_label.setText(f"{z * 100:.0f}%")

    def needs_detail(self) -> bool:
        if self.proc is None or self._current_img is None or self.showing_before:
            return False
        w, _ = self.image_size()
        magnification = self.canvas.zoom * w / max(1, self._current_img.shape[1])
        return magnification > 1.15 and self.canvas.zoom >= 0.5

    def _schedule_detail(self) -> None:
        if self.needs_detail():
            self._detail_timer.start()
        else:
            self._detail_timer.stop()

    def _submit_detail(self) -> None:
        if not self.needs_detail() or self.ffile is None:
            return
        r = self.canvas.visible_image_rect()
        if r.isEmpty() or r.width() * r.height() > DETAIL_MAX_PIXELS:
            return
        if self.canvas.has_tile() and self.canvas._tile_rect.contains(r):  # already covered
            return
        self.render_label.setText("detalle 100%…")
        self.detail_worker.submit((self._gen, self.proc, self.ffile, DevParams.coerce(self.params),
                                   (r.left(), r.top(), r.right(), r.bottom())))

    def _detail_job(self, job):
        gen, proc, f, params, rect = job
        if gen != self._gen:
            return None
        t0 = time.perf_counter()
        arr, out = develop_region(f, params, rect)
        return gen, proc, arr, out, (time.perf_counter() - t0) * 1000.0

    def _on_detail(self, r) -> None:
        gen, proc, arr, out, ms = r
        if proc is not self.proc or gen != self._gen or self.showing_before:
            return
        x0, y0, x1, y1 = out
        self.canvas.set_tile(arr, QRectF(x0, y0, x1 - x0, y1 - y0))
        self.render_label.setText(f"detalle {arr.shape[1]}×{arr.shape[0]} en {ms:.0f} ms")

    def show_before(self, on: bool) -> None:
        if self.proc is None:
            return
        self.showing_before = on
        if on:
            if self._before_img is None:
                self._before_img = self.proc.preview(DevParams(), PREVIEW_SIDE)
            self.canvas.clear_tile()
            self.canvas.set_image(self._before_img, self.image_size())
            self.render_label.setText("ANTES (sin ajustes)")
        else:
            if self._current_img is not None:
                self.canvas.set_image(self._current_img, self.image_size())
            self.render_label.setText("")
            self._schedule_detail()

    # ======================================================= association
    def associate(self) -> None:
        if not winassoc.available():
            return
        try:
            winassoc.register(sys.executable)
        except OSError as e:
            QMessageBox.critical(self, APP_NAME, f"No se pudo asociar: {e}")
            return
        QMessageBox.information(self, APP_NAME, "Listo: .fraw, .frw y .fset se abren con CTD Photo Studio (sólo tu usuario).")

    def unassociate(self) -> None:
        if not winassoc.available():
            return
        try:
            winassoc.unregister()
        except OSError as e:
            QMessageBox.critical(self, APP_NAME, f"No se pudo quitar la asociación: {e}")
            return
        QMessageBox.information(self, APP_NAME, "Asociación quitada.")

    # ============================================================= events
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        raws = [f for f in files if Path(f).suffix.lower() in _camera_exts()]
        fr = [f for f in files if Path(f).suffix.lower() in FRAW_EXTS]
        if fr:
            self.open_path(fr[0])
        elif raws:
            self.import_files(raws)

    def closeEvent(self, e):  # noqa: N802
        if not self.confirm_discard():
            e.ignore()
            return
        self._save_settings()
        self.shutdown()
        e.accept()

    def shutdown(self) -> None:
        if self._thumb_loader is not None:
            self._thumb_loader.cancel()
            self._thumb_loader.wait(3000)
        for w in (self.render_worker, self.detail_worker):
            w.stop()
        for t in list(self._tasks):
            t.cancel()
            t.wait(30000)


# ---------------------------------------------------------------------------
def load_theme(app: QApplication) -> None:
    p = resource_path("theme.qss")
    if p:
        app.setStyleSheet(p.read_text(encoding="utf-8"))


def create_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    app.setStyle("Fusion")
    load_theme(app)
    return app


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FabriCTD.CTDPhotoStudio")
        except Exception:  # noqa: BLE001 - cosmetic (taskbar icon grouping)
            pass
    app = create_app([sys.argv[0], *args])
    w = MainWindow()
    w.show()
    files = [a for a in args if not a.startswith("-") and os.path.isfile(a)]
    if files:
        QTimer.singleShot(0, lambda: w.open_path(files[0]))
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
