"""CTD Photo Studio — interfaz de revelado profesional para FabriRAW (``ctd-studio``).

Modelo de render
----------------
* Toda edición entra por :meth:`MainWindow.apply_updates` (panel, curvas, ruedas,
  presets, deshacer…). Mientras dura una interacción continua (arrastre, rueda)
  se pide ``preview(draft=True)``; al soltar, el render final (``draft=False``).
  Las peticiones se limitan a una cada 16 ms (flanco anterior + temporizador) y
  un :class:`LatestWorker` procesa siempre sólo la más reciente.
* Los grupos con el efecto oculto (bypass) se previsualizan en su defecto sin
  tocar ``self.params`` (lo que se guarda).
* Con zoom ≥ 50 % y la vista previa ampliada, 220 ms después del último cambio
  se pide al motor el área visible a resolución real (``render_region``).
* Guardar (Ctrl+S) sólo reescribe la cola DEV del archivo (``update_dev``).
"""

from __future__ import annotations

import colorsys
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QByteArray, QPointF, QRectF, QSettings, QSize, QStandardPaths, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..devparams import GROUPS, HSL_BAND_HUES, HSL_BAND_LABELS, SPECS, DevParams
from ..format import MAGIC, SET_MAGIC, FrawFile, update_dev
from ..processor import FRAWProcessor
from . import croptool, library, presets, winassoc
from .batch import ExportJob, ExportOptions, export_batch
from .canvas import ImageCanvas, clip_overlay
from .controls import history_label
from .develop_panel import DevelopPanel
from .dialogs import AboutDialog, ExportDialog, GroupsDialog, LicensesDialog, NewPresetDialog, ShortcutsDialog
from .engine import GEOMETRY_KEYS, Engine, before_params, effective_params
from .history import History
from .render import develop_region
from .widgets import Histogram
from .workers import LatestWorker, TaskThread, ThumbLoader, orientation_transform

__all__ = ["MainWindow", "main", "APP_NAME", "PREVIEW_SIDE", "create_app", "resource_path"]

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
RATING_ROLE = int(Qt.ItemDataRole.UserRole) + 1
LABEL_ROLE = int(Qt.ItemDataRole.UserRole) + 2
PRESET_ROLE = int(Qt.ItemDataRole.UserRole) + 3

#: Extensiones de imágenes procesadas que ni se intentan detectar al listar carpetas.
_SKIP_LIST_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".avif", ".webp", ".gif", ".bmp", ".psd",
    ".xmp", ".pp3", ".json", ".txt", ".pdf", ".mp4", ".mov", ".avi", ".thm", ".ini", ".db",
    ".fpreset", ".tif", ".tiff", ".zip", ".rar", ".7z", ".gz", ".exe", ".msi", ".iso", ".dll", ".doc", ".docx",
    ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".mp3", ".wav", ".flac", ".mkv", ".webm", ".html", ".htm", ".svg",
    ".py", ".js", ".css", ".csv", ".xml", ".ico", ".lnk", ".url", ".torrent", ".apk", ".dmg", ".epub", ".md",
})
MAX_SNIFF = 200  # archivos con extensión desconocida que se inspeccionan por contenido al listar


def _sniff(path: str | os.PathLike):
    from ..sniff import sniff  # noqa: PLC0415

    return sniff(path)


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
class LibraryDelegate(QStyledItemDelegate):
    """Dibuja la marca de color y las estrellas sobre cada miniatura."""

    def paint(self, painter, option, index):  # noqa: D401
        super().paint(painter, option, index)
        rating = int(index.data(RATING_ROLE) or 0)
        label = index.data(LABEL_ROLE) or ""
        r = option.rect
        painter.save()
        if label in library.LABEL_COLORS:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(library.LABEL_COLORS[label]))
            painter.drawRoundedRect(QRectF(r.left() + 5, r.top() + 5, 5, r.height() - 30), 2, 2)
        if rating:
            painter.setPen(QColor("#F0B232"))
            f = painter.font()
            f.setPointSizeF(8)
            painter.setFont(f)
            painter.drawText(QRectF(r.left(), r.top() + 4, r.width() - 8, 14),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, "★" * rating)
        painter.restore()


def _dock(title: str, name: str, widget: QWidget) -> QDockWidget:
    d = QDockWidget(title)
    d.setObjectName(name)
    d.setWidget(widget)
    d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable
                  | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
    return d


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    imageUpdated = Signal()

    def __init__(self, settings: QSettings | None = None, data_dir: str | os.PathLike | None = None):
        super().__init__()
        self.settings = settings or QSettings(ORG_NAME, APP_NAME)
        self.data_dir = Path(data_dir) if data_dir else Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation) or Path.home() / ".ctd-studio")
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.setDockNestingEnabled(True)

        # documento
        self.path: Path | None = None
        self.kind: str | None = None  # "fraw" | "fset"
        self.frames: list[FrawFile] = []
        self.frame_index = 0
        self.ffile: FrawFile | None = None
        self.proc: FRAWProcessor | None = None
        self.engine: Engine | None = None
        self.params = DevParams()
        self.saved_params: dict = DevParams().to_dict()
        self.dirty = False
        self.bypass: set[str] = set()
        self.history = History()
        self.clipboard: dict | None = None
        self.last_export: dict = {}
        # vista
        self.showing_before = False
        self.compare = "off"
        self.clip_sh = False
        self.clip_hl = False
        self.crop_mode = False
        self._crop_backup: dict | None = None
        self._straighten_base: float | None = None
        self._target_base: dict | None = None
        self._before_img: np.ndarray | None = None
        self._before_key: tuple | None = None
        self._current_img: np.ndarray | None = None
        self._gen = 0
        self._tasks: set[TaskThread] = set()
        self._thumb_loader: ThumbLoader | None = None
        self.last_render_ms = 0.0
        self._req_t0: float | None = None
        self._req_draft = False
        self._last_submit = 0.0
        self.render_count = 0
        self.latency: dict[str, list[float]] = {"draft": [], "final": []}
        self.render_ms: dict[str, list[float]] = {"draft": [], "final": []}
        self.library_dir: Path | None = None
        self._library_items: dict[str, QListWidgetItem] = {}
        self._library_fraw: dict[str, bool] = {}
        self._loading: str | None = None
        self.last_import_skipped: list[str] = []
        self.detail_error = ""
        self.builtin_presets = presets.load_builtin()
        self.user_presets: list[presets.Preset] = []

        self.render_worker = LatestWorker(self._render_job, "render")
        self.render_worker.result.connect(self._on_render)
        self.render_worker.failed.connect(self._on_worker_failed)
        self.detail_worker = LatestWorker(self._detail_job, "detail")
        self.detail_worker.result.connect(self._on_detail)
        self.detail_worker.failed.connect(self._on_worker_failed)
        self.preset_worker = LatestWorker(self._preset_preview_job, "preset")
        self.preset_worker.result.connect(self._on_preset_preview)

        self._coalesce = QTimer(self)
        self._coalesce.setSingleShot(True)
        self._coalesce.setInterval(COALESCE_MS)
        self._coalesce.timeout.connect(self._submit_render)
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._submit_detail)

        self._build_ui()
        self._build_docks()
        self._build_menu()
        self._default_state = self.saveState()
        self._restore_settings()
        self.reload_user_presets()
        self.history.reset(self.params.to_dict())
        self._refresh_history()
        self._update_actions()

    # ================================================================== UI
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        bar = QFrame()
        bar.setObjectName("topbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 7, 14, 7)
        bl.setSpacing(8)
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(28, 28))
        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        self.file_label = QLabel("")
        self.file_label.setObjectName("appSubtitle")
        self.file_label.setMaximumWidth(420)
        self.dirty_label = QLabel("")
        self.dirty_label.setContentsMargins(8, 1, 8, 1)
        bl.addWidget(logo)
        bl.addWidget(title)
        bl.addSpacing(10)
        bl.addWidget(self.file_label)
        bl.addWidget(self.dirty_label)
        bl.addStretch(1)
        self.btn_undo = QToolButton()
        self.btn_undo.setText("↶")
        self.btn_undo.setToolTip("Deshacer (Ctrl+Z)")
        self.btn_redo = QToolButton()
        self.btn_redo.setText("↷")
        self.btn_redo.setToolTip("Rehacer (Ctrl+Shift+Z)")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_undo.setObjectName("iconBtn")
        self.btn_redo.setObjectName("iconBtn")
        bl.addWidget(self.btn_undo)
        bl.addWidget(self.btn_redo)
        bl.addSpacing(6)
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

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(6, 6, 6, 0)
        cl.setSpacing(4)
        self.canvas = ImageCanvas()
        self.canvas.zoomChanged.connect(self._on_zoom)
        self.canvas.viewChanged.connect(self._schedule_detail)
        self.canvas.picked.connect(self._on_pick)
        self.canvas.hovered.connect(self._on_hover)
        self.canvas.cropChanged.connect(self._on_crop_rect)
        self.canvas.straighten.connect(self._on_straighten)
        self.canvas.cropAccepted.connect(self.accept_crop)
        self.canvas.cropCancelled.connect(self.cancel_crop)
        self.canvas.targetDrag.connect(self._on_target_drag)
        cl.addWidget(self.canvas, 1)

        zb = QHBoxLayout()
        zb.setContentsMargins(4, 0, 4, 4)
        zb.setSpacing(4)

        def tb(text: str, tip: str, slot: Callable | None = None, checkable: bool = False) -> QToolButton:
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            if slot is not None:
                (b.toggled if checkable else b.clicked).connect(slot)
            zb.addWidget(b)
            return b

        self.btn_fit = tb("Ajustar", "Ajustar a la ventana (Ctrl+0)", lambda: self.canvas.fit())
        self.btn_100 = tb("100%", "Tamaño real (Ctrl+1)", lambda: self.canvas.zoom_100())
        self.btn_200 = tb("200%", "Zoom 200% (Ctrl+2)", lambda: self.canvas.zoom_to(2.0))
        self.zoom_label = QLabel("—")
        self.zoom_label.setObjectName("muted")
        self.zoom_label.setMinimumWidth(44)
        zb.addWidget(self.zoom_label)
        zb.addStretch(1)
        self.render_label = QLabel("")
        self.render_label.setObjectName("muted")
        self.render_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        zb.addWidget(self.render_label)
        zb.addSpacing(10)
        self.btn_before = tb("Antes", "Alternar Antes / Después (\\)", self.show_before, True)
        self.btn_split = tb("Dividida", "Vista dividida Antes | Después con cortina arrastrable (Y)",
                            lambda on: self.set_compare("split" if on else "off"), True)
        self.btn_side = tb("Lado a lado", "Antes y Después lado a lado (Shift+Y)",
                           lambda on: self.set_compare("side" if on else "off"), True)
        zb.addSpacing(6)
        self.btn_clip = tb("Recortes", "Aviso de luces (rojo) y sombras (azul) recortadas (J)",
                           lambda on: self.set_clipping(on, on), True)
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
        root.addWidget(center, 1)
        self.setCentralWidget(central)

        self.rgb_label = QLabel("")
        self.rgb_label.setObjectName("rgbReadout")
        self.rgb_label.setMinimumWidth(440)
        self.statusBar().addPermanentWidget(self.rgb_label)
        self.statusBar().showMessage("Listo")
        self.resize(1560, 960)

    def _build_docks(self) -> None:
        # ---- revelado
        self.panel = DevelopPanel()
        self.panel.edited.connect(self.apply_updates)
        self.panel.groupReset.connect(self.reset_group)
        self.panel.bypassChanged.connect(self.set_bypass)
        self.panel.eyedropperToggled.connect(self._toggle_picker)
        self.panel.targetToggled.connect(self._toggle_target)
        self.panel.cropToolToggled.connect(self.set_crop_tool)
        self.panel.action.connect(self._panel_action)
        self.sliders = self.panel.rows  # compatibilidad: clave → ParamRow
        self.wb_combo = self.panel.controls["wb_mode"]
        self.gamma_combo = self.panel.controls["gamma_mode"]
        self.demosaic_combo = self.panel.controls["demosaic"]
        self.btn_picker = self.panel.btn_picker
        self.panel.setMinimumWidth(384)
        self.dock_develop = _dock("Revelado", "dock_develop", self.panel)

        # ---- histograma
        hw = QWidget()
        hl = QVBoxLayout(hw)
        hl.setContentsMargins(8, 4, 8, 6)
        self.histogram = Histogram()
        self.histogram.setMinimumHeight(132)
        self.histogram.clipToggled.connect(self._hist_clip)
        hl.addWidget(self.histogram)
        self.shot_label = QLabel("")
        self.shot_label.setObjectName("shotInfo")
        self.shot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl.addWidget(self.shot_label)
        self.dock_hist = _dock("Histograma", "dock_hist", hw)

        # ---- metadatos
        mw = QWidget()
        grid = QGridLayout(mw)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        self.meta_labels: dict[str, QLabel] = {}
        for i, (k, label) in enumerate(
            [("camera", "Cámara"), ("lens", "Lente"), ("iso", "ISO"), ("shutter", "Velocidad"),
             ("aperture", "Diafragma"), ("focal", "Focal"), ("size", "Tamaño"), ("date", "Fecha"),
             ("rating", "Calificación"), ("label", "Marca")]
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
        grid.setRowStretch(20, 1)
        self.dock_meta = _dock("Metadatos", "dock_meta", mw)

        # ---- historial + instantáneas
        hwid = QWidget()
        hv = QVBoxLayout(hwid)
        hv.setContentsMargins(8, 6, 8, 8)
        hv.setSpacing(4)
        row = QHBoxLayout()
        t = QLabel("PASOS")
        t.setObjectName("panelTitle")
        row.addWidget(t)
        row.addStretch(1)
        hv.addLayout(row)
        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")
        self.history_list.itemClicked.connect(lambda it: self.goto_history(self.history_list.row(it)))
        hv.addWidget(self.history_list, 3)
        row = QHBoxLayout()
        t = QLabel("INSTANTÁNEAS")
        t.setObjectName("panelTitle")
        row.addWidget(t)
        row.addStretch(1)
        b_snap = QToolButton()
        b_snap.setText("＋")
        b_snap.setToolTip("Nueva instantánea (Ctrl+N)")
        b_snap.clicked.connect(lambda: self.new_snapshot())
        b_del = QToolButton()
        b_del.setText("−")
        b_del.setToolTip("Eliminar instantánea")
        b_del.clicked.connect(self._delete_snapshot)
        row.addWidget(b_snap)
        row.addWidget(b_del)
        hv.addLayout(row)
        self.snap_list = QListWidget()
        self.snap_list.itemClicked.connect(lambda it: self.restore_snapshot(self.snap_list.row(it)))
        hv.addWidget(self.snap_list, 1)
        self.dock_history = _dock("Historial", "dock_history", hwid)

        # ---- presets
        pw = QWidget()
        pv = QVBoxLayout(pw)
        pv.setContentsMargins(8, 6, 8, 8)
        pv.setSpacing(4)
        prow = QHBoxLayout()
        for text, tip, slot in (("＋", "Crear preset con los ajustes actuales (Ctrl+Shift+N)", self.new_preset_dialog),
                                ("Importar", "Importar archivos .fpreset", self.import_preset_dialog),
                                ("Exportar", "Exportar el preset seleccionado a .fpreset", self.export_preset_dialog),
                                ("−", "Eliminar el preset de usuario seleccionado", self.delete_selected_preset)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            prow.addWidget(b)
        prow.addStretch(1)
        pv.addLayout(prow)
        self.preset_tree = QTreeWidget()
        self.preset_tree.setHeaderHidden(True)
        self.preset_tree.setMouseTracking(True)
        self.preset_tree.itemClicked.connect(self._preset_clicked)
        self.preset_tree.itemEntered.connect(self._preset_hovered)
        pv.addWidget(self.preset_tree, 1)
        self.preset_preview = QLabel("Pasá el mouse sobre un preset para previsualizarlo")
        self.preset_preview.setObjectName("presetPreview")
        self.preset_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preset_preview.setMinimumHeight(150)
        self.preset_preview.setWordWrap(True)
        pv.addWidget(self.preset_preview)
        self.dock_presets = _dock("Ajustes preestablecidos", "dock_presets", pw)

        # ---- biblioteca (tira inferior)
        lw = QWidget()
        lv = QVBoxLayout(lw)
        lv.setContentsMargins(8, 4, 8, 4)
        lv.setSpacing(2)
        hdr = QHBoxLayout()
        self.folder_label = QLabel("Sin carpeta")
        self.folder_label.setObjectName("muted")
        self.folder_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.btn_folder = QToolButton()
        self.btn_folder.setText("Carpeta…")
        self.btn_folder.clicked.connect(self.open_folder_dialog)
        self.sel_label = QLabel("")
        self.sel_label.setObjectName("muted")
        hdr.addWidget(self.btn_folder)
        hdr.addWidget(self.folder_label, 1)
        hdr.addWidget(self.sel_label)
        lv.addLayout(hdr)
        self.library = QListWidget()
        self.library.setObjectName("filmstrip")
        self.library.setViewMode(QListView.ViewMode.IconMode)
        self.library.setFlow(QListView.Flow.LeftToRight)
        self.library.setWrapping(False)
        self.library.setMovement(QListView.Movement.Static)
        self.library.setResizeMode(QListView.ResizeMode.Adjust)
        self.library.setIconSize(QSize(132, 88))
        self.library.setGridSize(QSize(156, 128))
        self.library.setSpacing(2)
        self.library.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.library.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.library.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.library.setWordWrap(True)
        self.library.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.library.setItemDelegate(LibraryDelegate(self.library))
        self.library.itemActivated.connect(self._library_activated)
        self.library.itemClicked.connect(self._library_clicked)
        self.library.itemSelectionChanged.connect(self._library_selection)
        lv.addWidget(self.library, 1)
        self.dock_library = _dock("Biblioteca", "dock_library", lw)
        self.dock_library.dockLocationChanged.connect(self._library_dock_moved)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_hist)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_develop)
        self.splitDockWidget(self.dock_hist, self.dock_develop, Qt.Orientation.Vertical)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_presets)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_history)
        self.splitDockWidget(self.dock_presets, self.dock_history, Qt.Orientation.Vertical)
        self.tabifyDockWidget(self.dock_history, self.dock_meta)
        self.dock_history.raise_()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_library)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.resizeDocks([self.dock_develop, self.dock_presets], [396, 250], Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_hist, self.dock_develop], [190, 700], Qt.Orientation.Vertical)
        self.resizeDocks([self.dock_library], [160], Qt.Orientation.Vertical)

    # ---------------------------------------------------------------- menu
    def _build_menu(self) -> None:
        mb = self.menuBar()
        self.shortcut_rows: list[tuple[str, str, str]] = []
        m = mb.addMenu("&Archivo")
        self._cat = "Archivo"
        self.act_open = self._act(m, "Abrir…", self.open_dialog, QKeySequence.StandardKey.Open)
        self._act(m, "Abrir carpeta…", self.open_folder_dialog, "Ctrl+Shift+O")
        self._act(m, "Importar RAW…", self.import_dialog, "Ctrl+I")
        m.addSeparator()
        self.act_save = self._act(m, "Guardar revelado", self.save_dev, QKeySequence.StandardKey.Save)
        self.act_autosave = self._act(m, "Autoguardar al cambiar de foto", self._toggle_autosave)
        self.act_autosave.setCheckable(True)
        m.addSeparator()
        self.act_export = self._act(m, "Exportar…", self.export_dialog, "Ctrl+E")
        self.act_export_again = self._act(m, "Exportar con los ajustes anteriores", self.export_again, "Ctrl+Shift+E")
        m.addSeparator()
        self._act(m, "Importar preset…", self.import_preset_dialog)
        self._act(m, "Exportar preset…", self.export_preset_dialog)
        m.addSeparator()
        self._act(m, "Salir", self.close, QKeySequence.StandardKey.Quit)

        m = mb.addMenu("&Editar")
        self._cat = "Editar"
        self.act_undo = self._act(m, "Deshacer", self.undo, "Ctrl+Z")
        self.act_redo = self._act(m, "Rehacer", self.redo, "Ctrl+Shift+Z")
        self.act_redo.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
        self.shortcut_rows[-1] = ("Editar", "Rehacer", "Ctrl+Shift+Z · Ctrl+Y")
        m.addSeparator()
        self.act_copy = self._act(m, "Copiar ajustes…", self.copy_settings_dialog, "Ctrl+Shift+C")
        self.act_paste = self._act(m, "Pegar ajustes", self.paste_settings, "Ctrl+Shift+V")
        self.act_sync = self._act(m, "Sincronizar ajustes con la selección…", self.sync_dialog, "Ctrl+Alt+S")
        m.addSeparator()
        rm = m.addMenu("Calificación")
        self._cat = "Biblioteca"
        for n in range(6):
            self._act(rm, f"{library.stars(n)}  ({n})", lambda _c=False, n=n: self.rate_selection(n), str(n))
        lm = m.addMenu("Marca de color")
        for (val, lbl), key in zip(library.LABELS, ("", "6", "7", "8", "9", "")):
            self._act(lm, lbl, lambda _c=False, v=val: self.label_selection(v), key or None)

        m = mb.addMenu("&Ver")
        self._cat = "Ver"
        self._act(m, "Ajustar a la ventana", lambda: self.canvas.fit(), "Ctrl+0")
        self._act(m, "100%", lambda: self.canvas.zoom_100(), "Ctrl+1")
        self._act(m, "200%", lambda: self.canvas.zoom_to(2.0), "Ctrl+2")
        self._act(m, "Acercar", lambda: self.canvas.zoom_at(1.25, QPointF(self.canvas.view_width() / 2, self.canvas.height() / 2)), QKeySequence.StandardKey.ZoomIn)
        self._act(m, "Alejar", lambda: self.canvas.zoom_at(0.8, QPointF(self.canvas.view_width() / 2, self.canvas.height() / 2)), QKeySequence.StandardKey.ZoomOut)
        m.addSeparator()
        self.act_before = self._act(m, "Antes / Después: alternar", lambda: self.btn_before.toggle(), "\\")
        self.act_split = self._act(m, "Antes / Después: vista dividida", lambda: self.btn_split.toggle(), "Y")
        self.act_side = self._act(m, "Antes / Después: lado a lado", lambda: self.btn_side.toggle(), "Shift+Y")
        m.addSeparator()
        self.act_clip = self._act(m, "Aviso de recorte de luces y sombras", lambda: self.btn_clip.toggle(), "J")
        m.addSeparator()
        pm = m.addMenu("Paneles")
        for d in (self.dock_develop, self.dock_hist, self.dock_presets, self.dock_history, self.dock_meta, self.dock_library):
            pm.addAction(d.toggleViewAction())
        self._act(m, "Restablecer diseño", self.reset_layout)
        self._act(m, "Pantalla completa", self._toggle_fullscreen, "F11")

        m = mb.addMenu("&Revelado")
        self._cat = "Revelado"
        self._act(m, "Auto tono", self.auto_tone, "Ctrl+U")
        self._act(m, "Auto balance de blancos", self.auto_wb, "Ctrl+Shift+U")
        self._act(m, "Cuentagotas de balance de blancos", lambda: self.btn_picker.toggle(), "W")
        m.addSeparator()
        self._act(m, "Girar 90° a la izquierda", lambda: self._panel_action("rotate_left"), "Ctrl+[")
        self._act(m, "Girar 90° a la derecha", lambda: self._panel_action("rotate_right"), "Ctrl+]")
        self._act(m, "Voltear horizontal", lambda: self._panel_action("flip_h"))
        self._act(m, "Voltear vertical", lambda: self._panel_action("flip_v"))
        m.addSeparator()
        self._act(m, "Nueva instantánea…", lambda: self.new_snapshot(), "Ctrl+N")
        self._act(m, "Nuevo preset con los ajustes actuales…", self.new_preset_dialog, "Ctrl+Shift+N")
        m.addSeparator()
        self._act(m, "Buscar ajuste", self._focus_search, "Ctrl+F")
        self._act(m, "Restablecer todo", self.reset_all, "Ctrl+Shift+R")

        m = mb.addMenu("&Herramientas")
        self._cat = "Herramientas"
        self.act_crop = self._act(m, "Recortar y enderezar", lambda: self.panel.btn_crop.toggle(), "R")
        self._act(m, "Ajuste dirigido (HSL)", lambda: self.panel.btn_target.toggle(), "T")
        m.addSeparator()
        self.act_assoc = self._act(m, "Asociar archivos .fraw/.frw/.fset con CTD Photo Studio", self.associate)
        self.act_unassoc = self._act(m, "Quitar asociación", self.unassociate)
        if not winassoc.available():
            for a in (self.act_assoc, self.act_unassoc):
                a.setEnabled(False)
                a.setToolTip("Disponible sólo en el .exe portable de Windows")

        m = mb.addMenu("A&yuda")
        self._cat = "Ayuda"
        self._act(m, "Atajos de teclado…", self.show_shortcuts, "Ctrl+/")
        m.addSeparator()
        self._act(m, "Acerca de CTD Photo Studio", self.show_about)
        self._act(m, "Licencias de terceros", self.show_licenses)
        self._act(m, "Sitio web", lambda: QDesktopServices.openUrl(QUrl(SITE_URL)))
        # atajos fuera de menús (documentados)
        self.shortcut_rows += [
            ("Recorte", "Aplicar / cancelar recorte", "Enter / Esc"),
            ("Recorte", "Enderezar", "Arrastrar fuera del rectángulo"),
            ("Controles", "Ajuste fino / grueso / muy fino", "Rueda · Shift+rueda · Ctrl+rueda"),
            ("Controles", "Paso con teclado", "← → (Shift ×10, Ctrl ÷10)"),
            ("Controles", "Restablecer un ajuste", "Doble click en la etiqueta o el deslizador"),
            ("Controles", "Escribir un valor", "Click en el número, escribir y Enter"),
            ("Vista", "Zoom centrado en el cursor", "Rueda sobre la foto"),
            ("Vista", "Mover la foto", "Arrastrar"),
            ("Curva", "Agregar / borrar punto", "Click / doble click o arrastrar afuera"),
        ]

    def _act(self, menu, text: str, slot: Callable, shortcut=None) -> QAction:
        a = QAction(text, self)
        if shortcut is not None:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        a.triggered.connect(slot)
        menu.addAction(a)
        if shortcut is not None:
            self.shortcut_rows.append((self._cat, text, QKeySequence(shortcut).toString(QKeySequence.SequenceFormat.NativeText)))
        return a

    # ============================================================ settings
    def _restore_settings(self) -> None:
        g = self.settings.value("geometry")
        if isinstance(g, (QByteArray, bytes)):
            self.restoreGeometry(g)
        s = self.settings.value("dock_state")
        if isinstance(s, (QByteArray, bytes)):
            self.restoreState(s)
        exp = self.settings.value("expanded_groups")
        if isinstance(exp, str) and exp:
            self.panel.set_expanded_groups(exp.split(","))
        self.act_autosave.setChecked(str(self.settings.value("autosave", "false")).lower() == "true")
        try:
            self.last_export = json.loads(str(self.settings.value("last_export", "{}")))
        except (TypeError, ValueError):
            self.last_export = {}
        last = self.settings.value("last_folder", "")
        if last and os.path.isdir(str(last)):
            # diferido: una carpeta grande no debe demorar la aparición de la ventana
            QTimer.singleShot(0, lambda: self.library_dir is None and self._loading is None and self.path is None
                              and self.set_library_folder(Path(str(last))))

    def _save_settings(self) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("dock_state", self.saveState())
        self.settings.setValue("expanded_groups", ",".join(self.panel.expanded_groups()))
        self.settings.setValue("autosave", "true" if self.act_autosave.isChecked() else "false")
        self.settings.setValue("last_export", json.dumps(self.last_export))
        if self.library_dir:
            self.settings.setValue("last_folder", str(self.library_dir))
        self.settings.sync()

    def reset_layout(self) -> None:
        self.restoreState(self._default_state)
        for d in (self.dock_develop, self.dock_hist, self.dock_presets, self.dock_history, self.dock_meta, self.dock_library):
            d.setFloating(False)
            d.show()
        self.dock_history.raise_()

    def _toggle_autosave(self, on: bool) -> None:
        self.settings.setValue("autosave", "true" if on else "false")

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

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
        if self.act_autosave.isChecked() and self.kind == "fraw":
            return self.save_dev()
        r = QMessageBox.question(
            self, APP_NAME,
            f"Hay cambios de revelado sin guardar en {self.path.name if self.path else 'el archivo'}.\n¿Guardarlos?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if r == QMessageBox.StandardButton.Save:
            return self.save_dev()
        return r == QMessageBox.StandardButton.Discard

    def open_path(self, path: str | os.PathLike, *, sync: bool = False) -> None:
        p = Path(path)
        det = _sniff(p)  # por contenido, no por extensión
        if det.may_be_raw:
            self.import_files([str(p)], sync=sync)
            return
        if not det.is_fabriraw:
            QMessageBox.warning(self, APP_NAME, f"No se puede abrir {p.name}:\n{det.reason}")
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
            eng = Engine(proc)
            first = eng.preview(params, PREVIEW_SIDE)  # calienta la caché fuera del hilo de la UI
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
        self.statusBar().showMessage(f"Abierto {self.path.name}")

    def _set_document(self, f: FrawFile, proc: FRAWProcessor, params: DevParams, img: np.ndarray | None) -> None:
        self._gen += 1
        if self.crop_mode:
            self.crop_mode = False
            self.canvas.set_tool("none")
            self._set_checked(self.panel.btn_crop, False)
        self.ffile, self.proc = f, proc
        self.engine = Engine(proc)
        self.params = DevParams.coerce(params)
        self.saved_params = self.params.to_dict()
        self.bypass = set()
        self._before_img = None
        self._before_key = None
        self.panel.as_shot = self.engine.as_shot_kelvin()
        self.history.reset(self.params.to_dict(), "Abierto")
        self._refresh_history()
        self._set_dirty(False)
        self._sync_controls()
        self._fill_meta()
        self.canvas.clear_tile()
        if img is not None:
            self._show_array(img)
            hist = self.engine.histogram(img)
            self.histogram.set_data(hist)
            self.panel.set_histogram(hist)
            if self.clip_sh or self.clip_hl or self.compare != "off":
                self.request_render(False)
        else:
            self.request_render(False)
        self._update_actions()

    def image_size(self, apply_crop: bool | None = None) -> tuple[int, int]:
        if self.engine is None:
            return (0, 0)
        if apply_crop is None:
            apply_crop = not self.crop_mode
        return self.engine.output_size(effective_params(self.params, self.bypass), apply_crop)

    def _show_array(self, arr: np.ndarray, size: tuple[int, int] | None = None) -> None:
        self._current_img = arr
        if not self.showing_before:
            self.canvas.set_image(arr, size or self.image_size())
        self.imageUpdated.emit()

    def _fill_meta(self) -> None:
        m = (self.ffile.meta if self.ffile else {}) or {}
        cam = " ".join(str(m.get(k, "")).strip() for k in ("make", "model")).strip()
        w, h = self.engine.base_size() if self.engine else (0, 0)
        lab = str(m.get("label") or "")
        vals = {
            "camera": cam or "—",
            "lens": str(m.get("lens") or "—"),
            "iso": str(m.get("iso")) if m.get("iso") is not None else "—",
            "shutter": _fmt_shutter(m.get("shutter_s")),
            "aperture": f"f/{float(m['aperture']):g}" if isinstance(m.get("aperture"), (int, float)) else "—",
            "focal": f"{float(m['focal_mm']):g} mm" if isinstance(m.get("focal_mm"), (int, float)) else "—",
            "size": f"{w} × {h} px" if w else "—",
            "date": str(m.get("datetime") or "—").replace("T", " "),
            "rating": library.stars(int(m.get("rating") or 0)) if isinstance(m.get("rating"), int) else library.stars(0),
            "label": dict(library.LABELS).get(lab, "—") if lab else "—",
        }
        for k, v in vals.items():
            self.meta_labels[k].setText(v)
        parts = [f"ISO {m['iso']}" if m.get("iso") else "", vals["shutter"] if vals["shutter"] != "—" else "",
                 vals["aperture"] if vals["aperture"] != "—" else "", vals["focal"] if vals["focal"] != "—" else ""]
        self.shot_label.setText("   ".join(p for p in parts if p))

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
            return (proc, params, Engine(proc).preview(params, PREVIEW_SIDE))

        self.run_task(work, lambda r: self._set_document(f, *r))

    # ============================================================ library
    def set_library_folder(self, folder: Path) -> None:
        self.library_dir = folder
        self.folder_label.setText(folder.name or str(folder))
        self.folder_label.setToolTip(str(folder))
        self.settings.setValue("last_folder", str(folder))
        if self._thumb_loader is not None:
            self._thumb_loader.cancel()
            self._thumb_loader.wait()
        self.library.clear()
        known = set(FRAW_EXTS) | set(_camera_exts())
        self._sniffed = 0
        files = []
        fraw_kind: dict[Path, bool] = {}
        try:
            entries = sorted(p for p in folder.iterdir() if p.is_file())
        except OSError:
            entries = []
        for p in entries:
            ext = p.suffix.lower()
            if ext in _SKIP_LIST_EXTS:
                continue
            if ext in FRAW_EXTS:
                fraw_kind[p] = True
                files.append(p)
            elif ext in known:
                fraw_kind[p] = False
                files.append(p)
            else:  # extensión rara o renombrada (.bin…): se mira el contenido
                sniffed = getattr(self, "_sniffed", 0)
                if sniffed >= MAX_SNIFF:
                    continue
                self._sniffed = sniffed + 1
                d = _sniff(p)
                if d.is_fabriraw or d.may_be_raw:
                    fraw_kind[p] = d.is_fabriraw
                    files.append(p)
        self._library_items = {}
        self._library_fraw = {str(p): fraw_kind[p] for p in files}
        for p in files:
            raw = not fraw_kind[p]
            it = QListWidgetItem(p.name + ("\nRAW · importar" if raw else ""))
            it.setData(Qt.ItemDataRole.UserRole, str(p))
            it.setToolTip(str(p))
            self.library.addItem(it)
            self._library_items[str(p)] = it
        if not files:
            it = QListWidgetItem("(sin archivos .fraw ni RAW)")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.library.addItem(it)
        loader = ThumbLoader([str(p) for p in files])
        loader.loaded.connect(self._on_thumb)
        self._thumb_loader = loader
        loader.start()
        self._library_selection()

    def _on_thumb(self, path: str, img: QImage | None, info: dict) -> None:
        it = self._library_items.get(path)
        if it is None:
            return
        if img is not None and not img.isNull():
            it.setIcon(QIcon(QPixmap.fromImage(img)))
        meta = (info or {}).get("meta") or {}
        it.setData(RATING_ROLE, int(meta.get("rating") or 0) if isinstance(meta.get("rating"), int) else 0)
        it.setData(LABEL_ROLE, str(meta.get("label") or ""))
        extra = []
        if (info or {}).get("frames", 1) > 1:
            extra.append(f"ráfaga · {info['frames']}")
        if (info or {}).get("error"):
            extra.append("⚠ " + str(info["error"])[:40])
        cam = " ".join(str(meta.get(k, "")) for k in ("make", "model")).strip()
        it.setToolTip(path + (f"\n{cam}" if cam else ""))
        if extra:
            it.setText(Path(path).name + "\n" + " · ".join(extra))

    def _select_library(self, path: Path) -> None:
        it = self._library_items.get(str(path))
        if it is not None:
            self.library.blockSignals(True)
            self.library.setCurrentItem(it)
            self.library.blockSignals(False)
            self.library.scrollToItem(it)

    def _library_clicked(self, it: QListWidgetItem) -> None:
        mods = QApplication.keyboardModifiers()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            return
        self._library_activated(it)

    def _library_activated(self, it: QListWidgetItem) -> None:
        p = it.data(Qt.ItemDataRole.UserRole)
        if p and (self.path is None or Path(p) != self.path) and p != self._loading:
            self.open_path(p)

    def _library_selection(self) -> None:
        n = len(self.selected_paths())
        self.sel_label.setText(f"{n} seleccionadas" if n > 1 else "")

    def _library_dock_moved(self, area) -> None:
        vertical = area in (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea)
        self.library.setFlow(QListView.Flow.TopToBottom if vertical else QListView.Flow.LeftToRight)

    def selected_paths(self) -> list[Path]:
        out = []
        for it in self.library.selectedItems():
            p = it.data(Qt.ItemDataRole.UserRole)
            if p:
                out.append(Path(p))
        return out

    def _rating_targets(self) -> list[Path]:
        sel = [p for p in self.selected_paths() if self._library_fraw.get(str(p), True)]
        if not sel and self.path is not None and self.kind == "fraw":
            sel = [self.path]
        return [p for p in sel if p.suffix.lower() != ".fset" and self._is_fraw(p)]

    @staticmethod
    def _is_fraw(p: Path) -> bool:
        try:
            with open(p, "rb") as fh:
                return fh.read(8) == MAGIC
        except OSError:
            return False

    def rate_selection(self, n: int) -> None:
        self._write_meta_selection({"rating": n}, f"Calificación {library.stars(n)}")

    def label_selection(self, label: str) -> None:
        self._write_meta_selection({"label": label}, f"Marca: {dict(library.LABELS).get(label, label)}")

    def _write_meta_selection(self, upd: dict, msg: str) -> None:
        targets = self._rating_targets()
        if not targets:
            self.statusBar().showMessage("Seleccioná fotos .fraw de la biblioteca")
            return
        done = 0
        for p in targets:
            try:
                if "rating" in upd:
                    meta = library.set_rating(p, upd["rating"])
                else:
                    meta = library.set_label(p, upd["label"])
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, APP_NAME, f"No se pudo escribir META en {p.name}:\n{e}")
                continue
            done += 1
            it = self._library_items.get(str(p))
            if it is not None:
                it.setData(RATING_ROLE, int(meta.get("rating") or 0))
                it.setData(LABEL_ROLE, str(meta.get("label") or ""))
            if self.path is not None and p == self.path and self.ffile is not None:
                self.ffile.meta = meta
                self._fill_meta()
        self.library.viewport().update()
        self.statusBar().showMessage(f"{msg} · {done} foto(s)")

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
        fns, _ = QFileDialog.getOpenFileNames(self, "Importar RAW de cámara", self._last_dir(), f"RAW de cámara ({cams});;Todos los archivos (*.*)")
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

        skipped: list[str] = []

        def work(report, cancelled):
            from ..ingest import ingest_to_file  # noqa: PLC0415
            from ..sniff import NotRawError  # noqa: PLC0415

            done = []
            for i, s in enumerate(srcs):
                if cancelled():
                    break
                report(i, f"Convirtiendo {Path(s).name} ({i + 1}/{len(srcs)})…")
                d = dst_for(s)
                try:
                    ingest_to_file(s, d)
                except NotRawError as e:
                    if len(srcs) == 1:
                        raise
                    skipped.append(str(e))
                    continue
                done.append(d)
            report(len(srcs), "Listo")
            return done

        prog = None if sync else self._busy("Importando…", len(srcs))

        def finished(done: list[str]) -> None:
            self.last_import_skipped = list(skipped)
            if skipped and not sync:
                shown = "\n\n".join(skipped[:8]) + ("\n\n…" if len(skipped) > 8 else "")
                QMessageBox.warning(self, APP_NAME,
                                    f"Se omitieron {len(skipped)} archivo(s) que no son RAW con datos de sensor:\n\n{shown}")
            if not done:
                return
            self.statusBar().showMessage(f"Importados {len(done)} archivo(s)")
            first = Path(done[0])
            self.set_library_folder(first.parent)
            self.open_path(first, sync=sync)

        self.run_task(work, finished, sync=sync, progress=prog)

    # -------------------------------------------------------------- export
    def export_dialog(self) -> None:
        if self.proc is None and not self.selected_paths():
            return
        sel = [p for p in self.selected_paths() if self._library_fraw.get(str(p), False)]
        dlg = ExportDialog(self.params.demosaic, self, selection_count=len(sel), last=self.last_export,
                           sample_name=self.path.name if self.path else "foto.fraw",
                           sample_meta=(self.ffile.meta if self.ffile else {}))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        o = dlg.options()
        self.last_export = {k: getattr(o, k) for k in ("fmt", "bits", "quality", "long_edge", "output_sharpen",
                                                       "metadata", "embed_icc", "template", "folder", "open_folder")}
        if dlg.scope_value() == "selection":
            self.export_selection(sel, o)
            return
        if not o.folder:
            base = self.path.stem if self.path else "imagen"
            if self.kind == "fset":
                base += f"_frame{self.frame_index + 1}"
            from .batch import render_name  # noqa: PLC0415

            suggested = render_name(o.template, Path(base), 1, self.ffile.meta if self.ffile else {}) + o.ext
            filters = {".jpg": "JPEG (*.jpg)", ".png": "PNG (*.png)", ".tif": "TIFF (*.tif)"}
            fn, _ = QFileDialog.getSaveFileName(self, "Exportar", str(Path(self._last_dir()) / suggested), filters[o.ext])
            if not fn:
                return
            if Path(fn).suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                fn += o.ext
            self.export_to(fn, bits=o.bits if o.fmt != "JPEG" else 8, quality=o.quality, demosaic=o.demosaic,
                           long_edge=o.long_edge, output_sharpen=o.output_sharpen, metadata=o.metadata,
                           embed_icc=o.embed_icc, open_folder=o.open_folder)
        else:
            self.export_selection([self.path] if self.path else [], o, current=True)

    def export_again(self) -> None:
        if self.proc is None:
            return
        if not self.last_export:
            self.export_dialog()
            return
        o = ExportOptions(**{k: v for k, v in self.last_export.items() if k in ExportOptions.__dataclass_fields__})
        if not o.folder and self.path is not None:
            o.folder = str(self.path.parent)
        self.export_selection([self.path] if self.path else [], o, current=True)

    def export_to(self, fn: str, bits: int = 8, quality: int = 95, demosaic: str | None = None,
                  half_size: bool = False, *, sync: bool = False, long_edge: int | None = None,
                  output_sharpen: str = "none", metadata: bool = True, embed_icc: bool = True,
                  open_folder: bool = False) -> None:
        eng, params = self.engine, DevParams.coerce(self.params)
        meta = dict(self.ffile.meta) if self.ffile else {}
        assert eng is not None

        def work(report, _c):
            t0 = time.perf_counter()
            report(0, f"Revelando {Path(fn).name}…")
            eng.export(fn, params, bits=bits, quality=quality, long_edge=long_edge, output_sharpen=output_sharpen,
                       embed_icc=embed_icc, metadata=metadata, demosaic=demosaic or params.demosaic, meta=meta)
            return time.perf_counter() - t0

        def done(dt: float) -> None:
            self.statusBar().showMessage(f"Exportado {Path(fn).name} en {dt:.1f} s")
            if open_folder:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(fn).parent)))

        prog = None if sync else self._busy(f"Exportando {Path(fn).name}…")
        if prog is not None:
            prog.setCancelButton(None)
        self.run_task(work, done, sync=sync, progress=prog)

    def export_selection(self, paths: list[Path], opts: ExportOptions, *, current: bool = False, sync: bool = False) -> None:
        jobs = []
        for p in paths:
            if self.path is not None and p == self.path and self.ffile is not None:
                jobs.append(ExportJob(p, DevParams.coerce(self.params), self.ffile))
            else:
                jobs.append(ExportJob(p))
        if not jobs:
            return
        prog = None if sync else self._busy("Exportando…", len(jobs))
        self.last_batch: list[Path] = []

        def work(report, cancelled):
            return export_batch(jobs, opts, report, cancelled)

        def done(out: list[Path]) -> None:
            self.last_batch = out
            self.statusBar().showMessage(f"Exportadas {len(out)} de {len(jobs)} foto(s)"
                                         + (f" en {out[0].parent}" if out else ""))
            if opts.open_folder and out:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out[0].parent)))

        self.run_task(work, done, sync=sync, progress=prog)

    # ============================================================== params
    def _sync_controls(self) -> None:
        self.panel.sync(self.params, self.bypass)
        self._update_history_buttons()

    def _set_checked(self, btn, on: bool) -> None:
        btn.blockSignals(True)
        btn.setChecked(on)
        btn.blockSignals(False)

    def apply_updates(self, updates: dict, label: str, merge_key: str | None, final: bool) -> None:
        """Punto único de edición: aplica ``updates`` (valores guardados) y re-renderiza."""
        if self.proc is None:
            return
        full = self.params.to_dict()
        geom_before = tuple(full[k] for k in GEOMETRY_KEYS)
        for k, v in updates.items():
            if k in SPECS:
                full[k] = v
        if "crop_aspect" in updates and not self.crop_mode:
            self._fit_crop_to_aspect(full)
        self.params = DevParams.from_dict(full)
        if self.crop_mode and "crop_aspect" in updates:
            self._fit_crop_to_aspect_live()
        if self.crop_mode and any(k.startswith("crop_") for k in updates):
            self._push_crop_to_canvas()
        if not self.crop_mode:
            self.history.record(label, self.params.to_dict(), merge_key, final)
            self._refresh_history()
        if final or len(updates) > 1 or "wb_mode" in updates or "gamma_mode" in updates:
            self._sync_controls()
        else:
            for g, sec in self.panel.sections.items():
                sec.set_changed(not self.params.group_is_default(g))
            self.panel.params = self.params
        self._after_param_change(draft=not final, geometry=tuple(self.params.to_dict()[k] for k in GEOMETRY_KEYS) != geom_before)

    def _fit_crop_to_aspect(self, full: dict) -> None:
        p = DevParams.from_dict(full)
        w, h = self.image_size(apply_crop=False) if self.engine else (1, 1)
        a = croptool.aspect_value(p.crop_aspect, w, h)
        if a is None:
            return
        r = croptool.fit_aspect((p.crop_left, p.crop_top, p.crop_right, p.crop_bottom), a, w, h)
        full.update(crop_left=r[0], crop_top=r[1], crop_right=r[2], crop_bottom=r[3])

    def _fit_crop_to_aspect_live(self) -> None:
        full = self.params.to_dict()
        self._fit_crop_to_aspect(full)
        self.params = DevParams.from_dict(full)
        self._push_crop_to_canvas()

    def _after_param_change(self, *, draft: bool = False, geometry: bool = False) -> None:
        self._gen += 1
        self._set_dirty(DevParams.coerce(self.params).to_dict() != self.saved_params)
        self.canvas.clear_tile()
        self.request_render(draft)
        if not draft:
            self._schedule_detail()

    def set_param(self, key: str, value: Any) -> None:
        """Cambio programático (tests / scripting): mueve el control como el usuario."""
        if key in self.sliders:
            self.sliders[key].setValue(value)
        else:
            self.apply_updates({key: value}, history_label(SPECS[key], value), key, True)

    def reset_all(self) -> None:
        if self.proc is None:
            return
        upd = DevParams().to_dict()
        upd.pop("version", None)
        self.apply_updates(upd, "Restablecer todo", None, True)

    def reset_group(self, group: str) -> None:
        p = DevParams.coerce(self.params)
        p.reset_group(group)
        keys = [s.key for s in SPECS.values() if s.group == group]
        label = next(g.label for g in GROUPS if g.key == group)
        self.apply_updates(p.subset(keys), f"Restablecer {label}", None, True)

    def set_bypass(self, group: str, on: bool) -> None:
        if on:
            self.bypass.add(group)
        else:
            self.bypass.discard(group)
        self.panel.sections[group].set_bypassed(on)
        self._gen += 1
        self.canvas.clear_tile()
        self.request_render(False)
        self._schedule_detail()

    # ---------------------------------------------------------------- history
    def _refresh_history(self) -> None:
        lst = self.history_list
        lst.blockSignals(True)
        labels = self.history.labels()
        if lst.count() != len(labels):
            lst.clear()
            lst.addItems(labels)
        else:
            for i, t in enumerate(labels):
                if lst.item(i).text() != t:
                    lst.item(i).setText(t)
        for i in range(lst.count()):
            it = lst.item(i)
            f = it.font()
            f.setBold(i == self.history.index)
            it.setFont(f)
            it.setForeground(QColor("#EDEBF0") if i <= self.history.index else QColor("#5E5B66"))
        if 0 <= self.history.index < lst.count():
            lst.setCurrentRow(self.history.index)
            lst.scrollToItem(lst.item(self.history.index))
        lst.blockSignals(False)
        self.snap_list.clear()
        self.snap_list.addItems([n for n, _ in self.history.snapshots])
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        if hasattr(self, "act_undo"):
            self.act_undo.setEnabled(self.history.can_undo())
            self.act_redo.setEnabled(self.history.can_redo())
        self.btn_undo.setEnabled(self.history.can_undo())
        self.btn_redo.setEnabled(self.history.can_redo())

    def _restore_state(self, state: dict | None) -> None:
        if state is None:
            return
        self.params = DevParams.from_dict(state)
        self._sync_controls()
        self._refresh_history()
        self._after_param_change()

    def undo(self) -> None:
        if self.crop_mode:
            self.cancel_crop()
            return
        self._restore_state(self.history.undo())

    def redo(self) -> None:
        self._restore_state(self.history.redo())

    def goto_history(self, i: int) -> None:
        self._restore_state(self.history.goto(i))

    def new_snapshot(self, name: str | None = None) -> None:
        if self.proc is None:
            return
        if name is None:
            name, ok = QInputDialog.getText(self, "Nueva instantánea", "Nombre:",
                                            text=f"Instantánea {len(self.history.snapshots) + 1}")
            if not ok:
                return
        self.history.add_snapshot(name or "Instantánea", self.params.to_dict())
        self._refresh_history()

    def restore_snapshot(self, i: int) -> None:
        if not 0 <= i < len(self.history.snapshots):
            return
        name, state = self.history.snapshots[i]
        upd = dict(state)
        upd.pop("version", None)
        self.apply_updates(upd, f"Instantánea «{name}»", None, True)

    def _delete_snapshot(self) -> None:
        self.history.remove_snapshot(self.snap_list.currentRow())
        self._refresh_history()

    # ---------------------------------------------------------------- presets
    def user_preset_dir(self) -> Path:
        return self.data_dir / "presets"

    def reload_user_presets(self) -> None:
        self.user_presets = presets.load_dir(self.user_preset_dir())
        self._fill_presets()

    def _fill_presets(self) -> None:
        t = self.preset_tree
        t.clear()
        folders: dict[str, QTreeWidgetItem] = {}

        def folder(name: str, builtin: bool) -> QTreeWidgetItem:
            key = ("★ " if builtin else "") + name
            if key not in folders:
                it = QTreeWidgetItem([("Incorporados · " if builtin else "Usuario · ") + name])
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)
                t.addTopLevelItem(it)
                folders[key] = it
            return folders[key]

        for p in self.builtin_presets + self.user_presets:
            it = QTreeWidgetItem([p.name])
            it.setData(0, PRESET_ROLE, p)
            groups = ", ".join(next(g.label for g in GROUPS if g.key == k) for k in p.groups())
            it.setToolTip(0, f"{p.name}\nToca: {groups or '—'} ({len(p.settings)} ajustes)")
            folder(p.group, p.builtin).addChild(it)
        t.expandAll()

    def all_presets(self) -> list[presets.Preset]:
        return self.builtin_presets + self.user_presets

    def apply_preset(self, preset: presets.Preset) -> None:
        if self.proc is None:
            return
        self.apply_updates(dict(preset.settings), f"Preset «{preset.name}»", None, True)

    def _preset_clicked(self, it: QTreeWidgetItem, _col: int = 0) -> None:
        p = it.data(0, PRESET_ROLE)
        if isinstance(p, presets.Preset):
            self.apply_preset(p)

    def _preset_hovered(self, it: QTreeWidgetItem, _col: int = 0) -> None:
        p = it.data(0, PRESET_ROLE)
        if not isinstance(p, presets.Preset) or self.engine is None:
            return
        params = effective_params(presets.apply_settings(self.params, p.settings), self.bypass)
        self.preset_worker.submit((self.engine, params, p.name))

    def _preset_preview_job(self, job):
        eng, params, name = job
        if eng is not self.engine:
            return None
        img = eng.preview(params, PREVIEW_SIDE, draft=True)
        h, w = img.shape[:2]
        f = max(1, max(w, h) // 360)
        small = np.ascontiguousarray(img[::f, ::f])
        return name, small

    def _on_preset_preview(self, r) -> None:
        name, small = r
        h, w = small.shape[:2]
        qi = QImage(small.data, w, h, int(small.strides[0]), QImage.Format.Format_RGB888).copy()
        pm = QPixmap.fromImage(qi).scaled(self.preset_preview.width() - 8, 170, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
        self.preset_preview.setPixmap(pm)
        self.preset_preview.setToolTip(name)

    def _selected_preset(self) -> presets.Preset | None:
        it = self.preset_tree.currentItem()
        p = it.data(0, PRESET_ROLE) if it is not None else None
        return p if isinstance(p, presets.Preset) else None

    def new_preset_dialog(self) -> None:
        groups = sorted({p.group for p in self.user_presets} | {"Usuario"})
        dlg = NewPresetDialog(self.params, groups, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name, folder, gs = dlg.values()
        self.create_preset(name, gs, folder)

    def create_preset(self, name: str, groups: list[str], folder: str = "Usuario") -> presets.Preset:
        pr = presets.preset_from_params(name, self.params, groups, folder)
        presets.save_preset(pr, self.user_preset_dir())
        self.reload_user_presets()
        self.statusBar().showMessage(f"Preset «{name}» guardado ({len(pr.settings)} ajustes)")
        return pr

    def import_preset_dialog(self) -> None:
        fns, _ = QFileDialog.getOpenFileNames(self, "Importar presets", self._last_dir(), "Presets (*.fpreset *.json)")
        for fn in fns:
            self.import_preset(fn)

    def import_preset(self, fn: str | os.PathLike) -> presets.Preset | None:
        try:
            pr = presets.load_preset(fn)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"No se pudo importar {Path(fn).name}:\n{e}")
            return None
        pr.builtin = False
        if not pr.group or pr.group == "Incorporados":
            pr.group = "Usuario"
        presets.save_preset(pr, self.user_preset_dir())
        self.reload_user_presets()
        return pr

    def export_preset_dialog(self) -> None:
        p = self._selected_preset()
        if p is None:
            QMessageBox.information(self, APP_NAME, "Seleccioná un preset de la lista.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Exportar preset", str(Path(self._last_dir()) / (presets.safe_filename(p.name) + ".fpreset")),
                                            "Preset (*.fpreset)")
        if fn:
            presets.save_preset(presets.Preset(p.name, p.group, dict(p.settings)), path=fn)

    def delete_selected_preset(self) -> None:
        p = self._selected_preset()
        if p is None or p.builtin or p.path is None:
            return
        if QMessageBox.question(self, APP_NAME, f"¿Eliminar el preset «{p.name}»?") == QMessageBox.StandardButton.Yes:
            try:
                p.path.unlink()
            except OSError:
                pass
            self.reload_user_presets()

    # ---------------------------------------------------- copy / paste / sync
    def copy_settings_dialog(self) -> None:
        if self.proc is None:
            return
        default = {g.key for g in GROUPS if not self.params.group_is_default(g.key)} - {"geometry"}
        dlg = GroupsDialog("Copiar ajustes", "Elegí qué grupos copiar:", self.params, "Copiar", default, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.copy_settings(dlg.selected())

    def copy_settings(self, groups: list[str]) -> dict:
        self.clipboard = presets.settings_for_groups(self.params, groups)
        try:
            QApplication.clipboard().setText(json.dumps({"fraw_settings": 1, "settings": self.clipboard}))
        except Exception:  # noqa: BLE001
            pass
        self.statusBar().showMessage(f"Copiados {len(self.clipboard)} ajustes de {len(groups)} grupo(s)")
        return self.clipboard

    def paste_settings(self) -> None:
        s = self.clipboard
        if s is None:
            try:
                d = json.loads(QApplication.clipboard().text())
                s = d.get("settings") if isinstance(d, dict) and d.get("fraw_settings") else None
            except (TypeError, ValueError):
                s = None
        if not s or self.proc is None:
            self.statusBar().showMessage("No hay ajustes copiados")
            return
        self.apply_updates(dict(s), "Pegar ajustes", None, True)

    def sync_dialog(self) -> None:
        targets = [p for p in self._rating_targets() if self.path is None or p != self.path]
        if self.proc is None or not targets:
            QMessageBox.information(self, APP_NAME, "Seleccioná en la biblioteca (Ctrl/Shift+click) las fotos .fraw a sincronizar.")
            return
        default = {g.key for g in GROUPS if not self.params.group_is_default(g.key)} - {"geometry"}
        dlg = GroupsDialog("Sincronizar ajustes", f"Aplicar los ajustes de esta foto a {len(targets)} foto(s) seleccionada(s).\n"
                           "Grupos a sincronizar:", self.params, "Sincronizar", default, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.sync_to(targets, dlg.selected())

    def sync_to(self, targets: list[Path], groups: list[str], *, sync: bool = False) -> None:
        settings = presets.settings_for_groups(self.params, groups)
        prog = None if sync else self._busy("Sincronizando…", len(targets))

        def work(report, cancelled):
            done = []
            for i, p in enumerate(targets):
                if cancelled():
                    break
                report(i, f"{p.name} ({i + 1}/{len(targets)})…")
                f = FrawFile.read(p)
                new = presets.apply_settings(DevParams.from_dict(f.dev), settings)
                update_dev(p, new.to_dict(full=False))
                done.append(p)
            return done

        self.run_task(work, lambda d: self.statusBar().showMessage(f"Ajustes sincronizados en {len(d)} foto(s)"),
                      sync=sync, progress=prog)

    # ------------------------------------------------------------ tools / WB
    def _panel_action(self, a: str) -> None:
        if self.proc is None:
            return
        if a == "auto_tone":
            self.auto_tone()
        elif a == "auto_wb":
            self.auto_wb()
        elif a in ("rotate_left", "rotate_right"):
            order = ["0", "90", "180", "270"]
            cw = a == "rotate_right"
            nxt = order[(order.index(self.params.rotate_90) + (1 if cw else -1)) % 4]
            r = croptool.rotate_rect_90((self.params.crop_left, self.params.crop_top, self.params.crop_right,
                                         self.params.crop_bottom), cw)
            self.apply_updates({"rotate_90": nxt, "crop_left": r[0], "crop_top": r[1], "crop_right": r[2],
                                "crop_bottom": r[3]}, "Girar 90° " + ("a la derecha" if cw else "a la izquierda"), None, True)
        elif a in ("flip_h", "flip_v"):
            h = a == "flip_h"
            r = croptool.flip_rect((self.params.crop_left, self.params.crop_top, self.params.crop_right,
                                    self.params.crop_bottom), h)
            key = "flip_h" if h else "flip_v"
            self.apply_updates({key: not getattr(self.params, key), "crop_left": r[0], "crop_top": r[1],
                                "crop_right": r[2], "crop_bottom": r[3]},
                               "Voltear " + ("horizontal" if h else "vertical"), None, True)

    def auto_tone(self) -> None:
        if self.engine is None:
            return
        try:
            upd = self.engine.auto_tone(effective_params(self.params, self.bypass))
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"Auto tono falló: {e}")
            return
        upd = {k: v for k, v in upd.items() if k in SPECS}
        if upd:
            self.apply_updates(upd, "Auto tono", None, True)
            self.panel.reveal("basic")

    def auto_wb(self) -> None:
        if self.engine is None:
            return
        try:
            upd = self.engine.auto_wb()
        except Exception as e:  # noqa: BLE001
            self.statusBar().showMessage(f"Auto balance falló: {e}")
            return
        upd = {k: v for k, v in upd.items() if k in SPECS}
        if upd:
            self.apply_updates(upd, "Auto balance de blancos", None, True)
        else:
            self.statusBar().showMessage("Auto balance: no se encontró una referencia neutra")

    def _toggle_picker(self, on: bool) -> None:
        if on:
            self._leave_tools(except_="picker")
            self.canvas.set_tool("picker")
        elif self.canvas.tool == "picker":
            self.canvas.set_tool("none")

    def _toggle_target(self, on: bool) -> None:
        if on:
            self._leave_tools(except_="target")
            self.canvas.set_tool("target")
            self.panel.reveal("hsl")
        elif self.canvas.tool == "target":
            self.canvas.set_tool("none")

    def _leave_tools(self, except_: str) -> None:
        if except_ != "picker":
            self._set_checked(self.btn_picker, False)
        if except_ != "target":
            self._set_checked(self.panel.btn_target, False)
        if except_ != "crop" and self.crop_mode:
            self.accept_crop()

    def _on_pick(self, x: float, y: float) -> None:
        if self.engine is None:
            return
        w, h = self.canvas.image_size
        upd = self.engine.sample_wb(x / max(1, w), y / max(1, h), 5, effective_params(self.params, self.bypass))
        if not upd:
            self.statusBar().showMessage("Cuentagotas: zona demasiado oscura o saturada, probá otra")
            return
        upd = {k: v for k, v in upd.items() if k in SPECS}
        self._set_checked(self.btn_picker, False)
        self.canvas.set_tool("none")
        self.apply_updates(upd, "Balance de blancos (cuentagotas)", None, True)
        if self.params.wb_mode == "custom":
            r, _g, b = self.params.wb_custom
            self.statusBar().showMessage(f"Balance personalizado: R {r:.3f} · G 1 · B {b:.3f}")

    def _pixel_at(self, x: float, y: float) -> tuple[int, int, int] | None:
        img = self._current_img
        w, h = self.canvas.image_size
        if img is None or not w or x < 0 or y < 0:
            return None
        px = min(img.shape[1] - 1, int(x * img.shape[1] / w))
        py = min(img.shape[0] - 1, int(y * img.shape[0] / h))
        r, g, b = (int(v) for v in img[py, px])
        return r, g, b

    def _on_hover(self, x: float, y: float) -> None:
        rgb = self._pixel_at(x, y)
        if rgb is None:
            self.rgb_label.setText("")
            return
        r, g, b = rgb
        self.rgb_label.setText(f"R {r:3d}  G {g:3d}  B {b:3d}   ·   {r / 2.55:.0f}% {g / 2.55:.0f}% {b / 2.55:.0f}%   ·   x {x:.0f}  y {y:.0f}")

    def _on_target_drag(self, x: float, y: float, dy: float, final: bool) -> None:
        if self.proc is None:
            return
        comp = self.panel.hsl_component()
        if self._target_base is None:
            rgb = self._pixel_at(x, y)
            if rgb is None:
                return
            hh, ss, _v = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
            hue = hh * 360.0
            weights = {}
            bands = sorted(HSL_BAND_HUES.items(), key=lambda kv: kv[1])
            for i, (band, c) in enumerate(bands):
                nb, nc = bands[(i + 1) % len(bands)]
                span = (nc - c) % 360 or 360
                d = (hue - c) % 360
                if d <= span:
                    t = d / span
                    weights = {band: 1 - t, nb: t}
                    break
            weights = {b: w for b, w in weights.items() if w > 0.05}
            self._target_base = {"weights": weights, "comp": comp,
                                 "base": {f"hsl_{b}_{comp}": getattr(self.params, f"hsl_{b}_{comp}") for b in weights},
                                 "sat": ss}
        tb = self._target_base
        if not tb["weights"]:
            self._target_base = None if final else tb
            return
        upd = {k: max(-1.0, min(1.0, v + dy * 0.004 * tb["weights"][k.split("_")[1]])) for k, v in tb["base"].items()}
        main = max(tb["weights"], key=tb["weights"].get)
        comp_lbl = {"h": "tono", "s": "saturación", "l": "luminancia"}[tb["comp"]]
        self.apply_updates(upd, f"Ajuste dirigido · {HSL_BAND_LABELS[main]} {comp_lbl}", "target", final)
        if final:
            self._target_base = None

    # ---------------------------------------------------------------- crop
    def set_crop_tool(self, on: bool) -> None:
        if self.proc is None:
            self._set_checked(self.panel.btn_crop, False)
            return
        if on == self.crop_mode:
            return
        if on:
            self._leave_tools(except_="crop")
            self._crop_backup = self.params.to_dict()
            self.crop_mode = True
            self.canvas.set_tool("crop")
            self._push_crop_to_canvas()
            self.panel.reveal("geometry")
            self.canvas.clear_tile()
            self.request_render(False)
        else:
            self.accept_crop()

    def _push_crop_to_canvas(self) -> None:
        w, h = self.image_size(apply_crop=False)
        p = self.params
        self.canvas.set_crop((p.crop_left, p.crop_top, p.crop_right, p.crop_bottom),
                             croptool.aspect_value(p.crop_aspect, w, h), p.rotate_deg)

    def _on_crop_rect(self, rect) -> None:
        l, t, r, b = rect
        full = self.params.to_dict()
        full.update(crop_left=l, crop_top=t, crop_right=r, crop_bottom=b)
        self.params = DevParams.from_dict(full)
        for k in ("crop_left", "crop_top", "crop_right", "crop_bottom"):
            self.panel.rows[k].set_silently(getattr(self.params, k))
        self.panel.sections["geometry"].set_changed(not self.params.group_is_default("geometry"))
        self._set_dirty(self.params.to_dict() != self.saved_params)

    def _on_straighten(self, delta: float, final: bool) -> None:
        if self._straighten_base is None:
            self._straighten_base = self.params.rotate_deg
        ang = max(-45.0, min(45.0, round(self._straighten_base + delta, 2)))
        self.apply_updates({"rotate_deg": ang}, f"Enderezar {ang:+.2f}°".replace(".", ","), "rotate_deg", final)
        self.canvas.crop_angle = ang
        if final:
            self._straighten_base = None

    def accept_crop(self) -> None:
        if not self.crop_mode:
            return
        self.crop_mode = False
        self.canvas.set_tool("none")
        self._set_checked(self.panel.btn_crop, False)
        if self._crop_backup is not None and self._crop_backup != self.params.to_dict():
            self.history.record("Recorte", self.params.to_dict(), None, True)
        self._crop_backup = None
        self._refresh_history()
        self._sync_controls()
        self.canvas.clear_tile()
        self._after_param_change()

    def cancel_crop(self) -> None:
        if not self.crop_mode:
            return
        self.crop_mode = False
        self.canvas.set_tool("none")
        self._set_checked(self.panel.btn_crop, False)
        if self._crop_backup is not None:
            self.params = DevParams.from_dict(self._crop_backup)
        self._crop_backup = None
        self._sync_controls()
        self._after_param_change()

    # ---------------------------------------------------------------- dirty
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
        self.btn_export.setEnabled(has)
        if hasattr(self, "act_export"):
            self.act_export.setEnabled(has)
            self.act_save.setEnabled(can_save)
        self.btn_save.setEnabled(can_save and self.dirty)
        self.btn_save.setToolTip(
            "Guardar el revelado no destructivo en el archivo (Ctrl+S)" if self.kind != "fset"
            else "En ráfagas .fset el revelado no se guarda en el archivo (exportá el frame)"
        )
        for w in (self.btn_before, self.btn_split, self.btn_side, self.btn_clip, self.btn_fit, self.btn_100,
                  self.btn_200, self.panel):
            w.setEnabled(has)

    def save_dev(self) -> bool:
        if self.proc is None or self.path is None:
            return False
        if self.kind != "fraw":
            QMessageBox.information(self, APP_NAME, "Las ráfagas .fset no guardan revelado; exportá el frame.")
            return False
        if self.crop_mode:
            self.accept_crop()
        p = DevParams.coerce(self.params)
        d = p.to_dict(full=False)
        try:
            update_dev(self.path, d)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"No se pudo guardar el revelado:\n{e}")
            return False
        self.saved_params = p.to_dict()
        if self.ffile is not None:
            self.ffile.dev = d
        self._set_dirty(False)
        self.statusBar().showMessage(f"Revelado guardado en {self.path.name} (sólo DEV; DATA intacto)")
        return True

    # ============================================================ rendering
    def request_render(self, draft: bool = False) -> None:
        """Limita a una petición cada 16 ms (flanco anterior + temporizador para el posterior)."""
        if self.proc is None:
            return
        now = time.perf_counter()
        if self._req_t0 is None:
            self._req_t0 = now
        self._req_draft = draft
        if self._coalesce.isActive():
            return
        since = (now - self._last_submit) * 1000.0
        if since >= COALESCE_MS:
            self._submit_render()
        else:
            self._coalesce.start(max(1, int(COALESCE_MS - since)))

    def _before_wanted(self) -> bool:
        return self.compare != "off" or self.showing_before

    def _job(self, draft: bool, t0: float | None) -> dict:
        eff = effective_params(self.params, self.bypass)
        apply_crop = not self.crop_mode
        bkey = (tuple(eff.to_dict()[k] for k in GEOMETRY_KEYS), apply_crop, eff.demosaic)
        return {"gen": self._gen, "engine": self.engine, "params": eff, "draft": draft, "apply_crop": apply_crop,
                "clip": (self.clip_sh, self.clip_hl), "before_key": bkey,
                "need_before": self._before_wanted() and bkey != self._before_key,
                "size": self.engine.output_size(eff, apply_crop) if self.engine else (0, 0), "t0": t0}

    def _submit_render(self) -> None:
        if self.engine is None:
            return
        self._last_submit = time.perf_counter()
        self.render_worker.submit(self._job(self._req_draft, self._req_t0))
        self._req_t0 = None

    def _render_job(self, job: dict) -> dict:
        eng: Engine = job["engine"]
        ts = time.perf_counter()
        img = eng.preview(job["params"], PREVIEW_SIDE, apply_crop=job["apply_crop"], draft=job["draft"])
        ms = (time.perf_counter() - ts) * 1000.0
        hist = eng.histogram(img)
        sh, hl = job["clip"]
        overlay = None
        if sh or hl:
            msh, mhl = eng.clipping_mask(img)
            overlay = clip_overlay(msh, mhl, sh, hl)
        before = None
        if job["need_before"]:
            before = eng.preview(before_params(job["params"]), PREVIEW_SIDE, apply_crop=job["apply_crop"], draft=False)
        return dict(job, img=img, hist=hist, overlay=overlay, before=before, ms=ms)

    def _on_render(self, r: dict) -> None:
        if r["engine"] is not self.engine:
            return
        self.render_count += 1
        if r["before"] is not None:
            self._before_img = r["before"]
            self._before_key = r["before_key"]
            self.canvas.set_before(self._before_img)
        self._show_array(r["img"], r["size"])
        if not self.showing_before:
            self.canvas.set_overlay(r["overlay"])
        self.histogram.set_data(r["hist"])
        self.panel.set_histogram(r["hist"])
        kind = "draft" if r["draft"] else "final"
        self.render_ms[kind].append(r["ms"])
        if r["t0"] is not None:
            self.last_render_ms = (time.perf_counter() - r["t0"]) * 1000.0
            self.latency[kind].append(self.last_render_ms)
        self.render_label.setText(f"{'borrador' if r['draft'] else 'vista previa'} {r['ms']:.0f} ms")

    def render_now(self, draft: bool = False) -> np.ndarray | None:
        """Render síncrono en el hilo que llama (tests / scripting)."""
        if self.engine is None:
            return None
        self._coalesce.stop()
        r = self._render_job(self._job(draft, time.perf_counter()))
        self._on_render(r)
        return r["img"]

    def _on_worker_failed(self, msg: str) -> None:
        self.statusBar().showMessage("Error al procesar: " + msg.strip().splitlines()[-1])

    def _on_zoom(self, z: float) -> None:
        self.zoom_label.setText(f"{z * 100:.0f}%")

    def needs_detail(self) -> bool:
        if self.proc is None or self._current_img is None or self.showing_before or self.crop_mode:
            return False
        w, _ = self.canvas.image_size
        magnification = self.canvas.zoom * w / max(1, self._current_img.shape[1])
        return magnification > 1.15 and self.canvas.zoom >= 0.5

    def _schedule_detail(self) -> None:
        if self.needs_detail():
            self._detail_timer.start()
        else:
            self._detail_timer.stop()

    def _submit_detail(self) -> None:
        if not self.needs_detail() or self.ffile is None or self.engine is None:
            return
        r = self.canvas.visible_image_rect()
        if r.isEmpty():
            return
        if self.canvas.has_tile() and self.canvas._tile_rect.contains(r):  # noqa: SLF001 - ya cubierto
            return
        w, h = self.canvas.image_size
        m = 0.02
        rr = QRectF(r.x() - r.width() * m, r.y() - r.height() * m, r.width() * (1 + 2 * m), r.height() * (1 + 2 * m)) \
            .intersected(QRectF(0, 0, w, h))
        z = min(1.0, self.canvas.zoom)
        out_w, out_h = max(2, int(rr.width() * z)), max(2, int(rr.height() * z))
        if out_w * out_h > DETAIL_MAX_PIXELS:
            return
        self.render_label.setText("detalle…")
        self.detail_worker.submit((self._gen, self.engine, self.ffile, effective_params(self.params, self.bypass),
                                   (rr.left(), rr.top(), rr.right(), rr.bottom()), (w, h), (out_w, out_h)))

    def _detail_job(self, job):
        gen, eng, f, params, rect, (w, h), (ow, oh) = job
        if gen != self._gen:
            return None
        t0 = time.perf_counter()
        arr = None
        if eng.has("render_region"):
            rn = (rect[0] / w, rect[1] / h, rect[2] / w, rect[3] / h)
            try:
                arr = eng.render_region(params, rn, ow, oh)
                out = rect
            except Exception as e:  # noqa: BLE001 - se informa y se usa el método de respaldo
                self.detail_error = f"{type(e).__name__}: {e}"
                arr = None
        if arr is None:
            p = DevParams.coerce(params)
            if not all(p.is_default(k) for k in GEOMETRY_KEYS):
                return None  # el fallback sólo sabe recortar sin geometría
            arr, out = develop_region(f, params, rect)
        return gen, eng, arr, out, (time.perf_counter() - t0) * 1000.0

    def _on_detail(self, r) -> None:
        gen, eng, arr, out, ms = r
        if eng is not self.engine or gen != self._gen or self.showing_before or arr is None:
            return
        x0, y0, x1, y1 = out
        self.canvas.set_tile(arr, QRectF(x0, y0, x1 - x0, y1 - y0))
        self.render_label.setText(f"detalle {arr.shape[1]}×{arr.shape[0]} en {ms:.0f} ms")

    # ------------------------------------------------------- before / after
    def _ensure_before(self) -> None:
        if self.engine is None:
            return
        job = self._job(False, None)
        if self._before_img is None or job["before_key"] != self._before_key:
            self._before_img = self.engine.preview(before_params(job["params"]), PREVIEW_SIDE, apply_crop=job["apply_crop"])
            self._before_key = job["before_key"]
            self.canvas.set_before(self._before_img)

    def show_before(self, on: bool) -> None:
        if self.proc is None:
            return
        self._set_checked(self.btn_before, on)
        self.showing_before = on
        if on:
            self._ensure_before()
            self.canvas.clear_tile()
            self.canvas.set_overlay(None)
            self.canvas.set_image(self._before_img, self.image_size())
            self.canvas.show_before_label = "ANTES"
            self.render_label.setText("ANTES (sin ajustes)")
        else:
            self.canvas.show_before_label = ""
            if self._current_img is not None:
                self.canvas.set_image(self._current_img, self.image_size())
            self.render_label.setText("")
            self.request_render(False)
            self._schedule_detail()
        self.canvas.update()

    def set_compare(self, mode: str) -> None:
        self.compare = mode
        self._set_checked(self.btn_split, mode == "split")
        self._set_checked(self.btn_side, mode == "side")
        if mode != "off":
            if self.showing_before:
                self.show_before(False)
            self._ensure_before()
        self.canvas.set_compare(mode)

    def set_clipping(self, sh: bool, hl: bool) -> None:
        self.clip_sh, self.clip_hl = sh, hl
        self._set_checked(self.btn_clip, sh or hl)
        self.histogram.set_clip_state(sh, hl)
        if not (sh or hl):
            self.canvas.set_overlay(None)
        self.request_render(False)

    def _hist_clip(self, which: str, on: bool) -> None:
        if which == "shadows":
            self.set_clipping(on, self.clip_hl)
        else:
            self.set_clipping(self.clip_sh, on)

    def _focus_search(self) -> None:
        self.dock_develop.show()
        self.panel.search.setFocus()
        self.panel.search.selectAll()

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

    # ============================================================= help
    def show_shortcuts(self) -> None:
        ShortcutsDialog(self.shortcut_rows, self).exec()

    def show_about(self) -> None:
        AboutDialog(APP_NAME, __version__, SITE_URL, REPO_URL, app_icon(), self.show_licenses, self).exec()

    def show_licenses(self) -> None:
        texts = []
        for name in ("THIRD_PARTY_NOTICES.md", "LICENSE"):
            p = resource_path(name)
            if p:
                texts.append(p.read_text(encoding="utf-8", errors="replace"))
        LicensesDialog(texts, REPO_URL, self).exec()

    # ============================================================= events
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        pres = [f for f in files if f.lower().endswith(".fpreset")]
        for f in pres:
            self.import_preset(f)
        kinds = {f: _sniff(f) for f in files if os.path.isfile(f) and f not in pres}
        raws = [f for f, d in kinds.items() if d.may_be_raw]
        fr = [f for f, d in kinds.items() if d.is_fabriraw]
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
            self._thumb_loader.wait()
        for w in (self.render_worker, self.detail_worker, self.preset_worker):
            w.stop()
        for t in list(self._tasks):
            t.cancel()
            t.wait()


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
    pal = QPalette()
    for role, col in ((QPalette.ColorRole.Window, "#121117"), (QPalette.ColorRole.WindowText, "#EDEBF0"),
                      (QPalette.ColorRole.Base, "#0E0D12"), (QPalette.ColorRole.AlternateBase, "#121117"),
                      (QPalette.ColorRole.Text, "#EDEBF0"), (QPalette.ColorRole.Button, "#1A1920"),
                      (QPalette.ColorRole.ButtonText, "#EDEBF0"), (QPalette.ColorRole.Highlight, "#F0B232"),
                      (QPalette.ColorRole.HighlightedText, "#09080C"), (QPalette.ColorRole.ToolTipBase, "#1A1920"),
                      (QPalette.ColorRole.ToolTipText, "#EDEBF0"), (QPalette.ColorRole.PlaceholderText, "#6F6C78")):
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
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
