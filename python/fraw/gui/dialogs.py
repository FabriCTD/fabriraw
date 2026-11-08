"""Diálogos: exportar, copiar/pegar/sincronizar por grupos, nuevo preset, atajos, acerca de, licencias."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..devparams import GROUPS, DevParams
from .batch import ExportOptions, render_name

__all__ = ["ExportDialog", "GroupChooser", "GroupsDialog", "NewPresetDialog", "ShortcutsDialog",
           "AboutDialog", "LicensesDialog"]

SHARPEN_CHOICES = (("none", "Ninguno"), ("screen", "Pantalla"), ("matte", "Papel mate"), ("glossy", "Papel brillante"))


def _buttons(dlg: QDialog, ok_text: str) -> QDialogButtonBox:
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    ok = bb.button(QDialogButtonBox.StandardButton.Ok)
    ok.setText(ok_text)
    ok.setObjectName("primary")
    bb.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    return bb


# ------------------------------------------------------------------ export
class ExportDialog(QDialog):
    def __init__(self, demosaic: str = "ahd", parent: QWidget | None = None, *, selection_count: int = 0,
                 last: dict | None = None, sample_name: str = "foto", sample_meta: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Exportar")
        self.setMinimumWidth(520)
        self.sample_name, self.sample_meta = sample_name, sample_meta or {}
        last = last or {}
        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.scope = QComboBox()
        self.scope.addItem("Foto actual", "current")
        if selection_count > 1:
            self.scope.addItem(f"Selección de la biblioteca ({selection_count} fotos)", "selection")
        form.addRow("Exportar", self.scope)
        self.fmt = QComboBox()
        self.fmt.addItems(["JPEG", "PNG", "TIFF"])
        self.bits = QComboBox()
        self.bits.addItems(["8 bits", "16 bits"])
        qrow = QHBoxLayout()
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality_slider.valueChanged.connect(self.quality.setValue)
        self.quality.valueChanged.connect(self.quality_slider.setValue)
        qrow.addWidget(self.quality_slider, 1)
        qrow.addWidget(self.quality)
        form.addRow("Formato", self.fmt)
        form.addRow("Profundidad", self.bits)
        form.addRow("Calidad JPEG", qrow)
        rrow = QHBoxLayout()
        self.resize_chk = QCheckBox("Lado largo")
        self.long_edge = QSpinBox()
        self.long_edge.setRange(64, 30000)
        self.long_edge.setSuffix(" px")
        self.long_edge.setValue(2048)
        rrow.addWidget(self.resize_chk)
        rrow.addWidget(self.long_edge)
        rrow.addStretch(1)
        form.addRow("Redimensionar", rrow)
        self.sharpen = QComboBox()
        for k, lbl in SHARPEN_CHOICES:
            self.sharpen.addItem(lbl, k)
        form.addRow("Enfoque de salida", self.sharpen)
        self.demosaic = QComboBox()
        self.demosaic.addItem("AHD (calidad)", "ahd")
        self.demosaic.addItem("Bilineal (rápido)", "bilinear")
        self.demosaic.setCurrentIndex(max(0, self.demosaic.findData(demosaic)))
        form.addRow("Demosaico", self.demosaic)
        self.metadata = QCheckBox("Incluir metadatos (cámara, fecha)")
        self.icc = QCheckBox("Incrustar perfil sRGB")
        form.addRow("", self.metadata)
        form.addRow("", self.icc)
        trow = QHBoxLayout()
        self.template = QLineEdit()
        self.template.setPlaceholderText("{nombre}")
        self.template.setToolTip("Marcadores: {nombre} {fecha} {n} {camara} {iso}")
        trow.addWidget(self.template, 1)
        for tok in ("{nombre}", "{fecha}", "{n}"):
            b = QPushButton(tok)
            b.setObjectName("chip")
            b.clicked.connect(lambda _c=False, t=tok: self.template.insert(t))
            trow.addWidget(b)
        form.addRow("Nombre", trow)
        self.name_preview = QLabel("")
        self.name_preview.setObjectName("hint")
        form.addRow("", self.name_preview)
        frow = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Junto al archivo original")
        pick = QPushButton("Elegir…")
        pick.clicked.connect(self._pick_folder)
        frow.addWidget(self.folder, 1)
        frow.addWidget(pick)
        form.addRow("Carpeta", frow)
        self.open_folder = QCheckBox("Abrir la carpeta al terminar")
        form.addRow("", self.open_folder)
        lay.addLayout(form)
        lay.addWidget(_buttons(self, "Exportar"))
        # valores previos
        self.fmt.setCurrentText(str(last.get("fmt", "JPEG")))
        self.bits.setCurrentIndex(1 if int(last.get("bits", 8)) == 16 else 0)
        self.quality.setValue(int(last.get("quality", 92)))
        le = last.get("long_edge")
        self.resize_chk.setChecked(bool(le))
        if le:
            self.long_edge.setValue(int(le))
        self.sharpen.setCurrentIndex(max(0, self.sharpen.findData(last.get("output_sharpen", "none"))))
        self.metadata.setChecked(bool(last.get("metadata", True)))
        self.icc.setChecked(bool(last.get("embed_icc", True)))
        self.template.setText(str(last.get("template", "{nombre}")))
        self.folder.setText(str(last.get("folder", "")))
        self.open_folder.setChecked(bool(last.get("open_folder", False)))
        for sig in (self.fmt.currentIndexChanged, self.resize_chk.toggled, self.template.textChanged):
            sig.connect(self._sync)
        self._sync()

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Carpeta destino", self.folder.text() or str(Path.home()))
        if d:
            self.folder.setText(d)

    def _sync(self, *_a) -> None:
        jpeg = self.fmt.currentText() == "JPEG"
        if jpeg:
            self.bits.setCurrentIndex(0)
        self.bits.setEnabled(not jpeg)
        self.quality.setEnabled(jpeg)
        self.quality_slider.setEnabled(jpeg)
        self.long_edge.setEnabled(self.resize_chk.isChecked())
        o = self.options()
        self.name_preview.setText("Ejemplo: " + render_name(o.template, Path(self.sample_name), 1, self.sample_meta) + o.ext)

    def options(self) -> ExportOptions:
        return ExportOptions(
            fmt=self.fmt.currentText(), bits=16 if self.bits.currentIndex() == 1 else 8,
            quality=self.quality.value(), long_edge=self.long_edge.value() if self.resize_chk.isChecked() else None,
            output_sharpen=self.sharpen.currentData(), metadata=self.metadata.isChecked(),
            embed_icc=self.icc.isChecked(), template=self.template.text().strip() or "{nombre}",
            folder=self.folder.text().strip(), demosaic=self.demosaic.currentData(),
            open_folder=self.open_folder.isChecked(),
        )

    def scope_value(self) -> str:
        return self.scope.currentData()


# ------------------------------------------------------------- group chooser
class GroupChooser(QWidget):
    def __init__(self, params: DevParams | None = None, checked: set[str] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        self.boxes: dict[str, QCheckBox] = {}
        for i, g in enumerate(GROUPS):
            changed = params is not None and not params.group_is_default(g.key)
            cb = QCheckBox(g.label + ("  ●" if changed else ""))
            cb.setChecked(g.key in checked if checked is not None else changed)
            if changed:
                cb.setToolTip("Tiene ajustes modificados")
            self.boxes[g.key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        row = QHBoxLayout()
        b_all, b_none, b_changed = QPushButton("Todo"), QPushButton("Nada"), QPushButton("Sólo modificados")
        b_all.clicked.connect(lambda: self.set_all(True))
        b_none.clicked.connect(lambda: self.set_all(False))
        b_changed.clicked.connect(lambda: [cb.setChecked(params is not None and not params.group_is_default(k))
                                           for k, cb in self.boxes.items()])
        for b in (b_all, b_none, b_changed):
            row.addWidget(b)
        row.addStretch(1)
        grid.addLayout(row, (len(GROUPS) + 1) // 2, 0, 1, 2)

    def set_all(self, on: bool) -> None:
        for cb in self.boxes.values():
            cb.setChecked(on)

    def selected(self) -> list[str]:
        return [k for k, cb in self.boxes.items() if cb.isChecked()]


class GroupsDialog(QDialog):
    def __init__(self, title: str, text: str, params: DevParams | None, ok_text: str,
                 checked: set[str] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        self.chooser = GroupChooser(params, checked)
        lay.addWidget(self.chooser)
        lay.addWidget(_buttons(self, ok_text))

    def selected(self) -> list[str]:
        return self.chooser.selected()


class NewPresetDialog(QDialog):
    def __init__(self, params: DevParams, groups_existing: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Nuevo ajuste preestablecido")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("Mi preset")
        self.group = QComboBox()
        self.group.setEditable(True)
        for g in groups_existing or ["Usuario"]:
            self.group.addItem(g)
        form.addRow("Nombre", self.name)
        form.addRow("Carpeta", self.group)
        lay.addLayout(form)
        lay.addWidget(QLabel("Grupos de ajustes que incluye:"))
        self.chooser = GroupChooser(params, {g.key for g in GROUPS if not params.group_is_default(g.key)} - {"geometry"})
        lay.addWidget(self.chooser)
        lay.addWidget(_buttons(self, "Crear"))

    def values(self) -> tuple[str, str, list[str]]:
        return self.name.text().strip() or "Mi preset", self.group.currentText().strip() or "Usuario", self.chooser.selected()


# ------------------------------------------------------------------ shortcuts
class ShortcutsDialog(QDialog):
    def __init__(self, rows: list[tuple[str, str, str]], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Atajos de teclado")
        self.resize(600, 640)
        lay = QVBoxLayout(self)
        flt = QLineEdit()
        flt.setPlaceholderText("Filtrar…")
        lay.addWidget(flt)
        t = QTableWidget(len(rows), 3)
        t.setHorizontalHeaderLabels(["Categoría", "Acción", "Atajo"])
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for i, (cat, action, keys) in enumerate(rows):
            for j, v in enumerate((cat, action, keys)):
                t.setItem(i, j, QTableWidgetItem(v))
        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(t, 1)
        self.table = t

        def apply(text: str) -> None:
            q = text.lower()
            for i in range(t.rowCount()):
                t.setRowHidden(i, bool(q) and not any(q in (t.item(i, j).text().lower()) for j in range(3)))

        flt.textChanged.connect(apply)
        b = QPushButton("Cerrar")
        b.clicked.connect(self.accept)
        lay.addWidget(b, 0, Qt.AlignmentFlag.AlignRight)


# ---------------------------------------------------------------- about/licenses
class AboutDialog(QDialog):
    def __init__(self, app_name: str, version: str, site: str, repo: str, icon, licenses_cb: Callable[[], None],
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Acerca de {app_name}")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(icon.pixmap(72, 72))
        top.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)
        text = QLabel(
            f"<h2 style='margin:0'>{app_name}</h2>"
            f"<p style='color:#A6A3AD'>Versión {version} · formato FabriRAW 1.0 (.fraw / .frw / .fset)</p>"
            "<p><b>Hecho 100% con Claude (Anthropic) a partir de un prompt de FabriCTD.</b></p>"
            f"<p>Sitio: <a style='color:#F0B232' href='{site}'>{site}</a><br>"
            f"Código: <a style='color:#F0B232' href='{repo}'>{repo}</a></p>"
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
        lic.clicked.connect(licenses_cb)
        ok = QPushButton("Cerrar")
        ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        row.addWidget(lic)
        row.addStretch(1)
        row.addWidget(ok)
        lay.addLayout(row)


class LicensesDialog(QDialog):
    def __init__(self, texts: list[str], repo: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Licencias de terceros")
        self.resize(760, 620)
        lay = QVBoxLayout(self)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        if texts:
            md = texts[0]
            if len(texts) > 1:
                md += "\n\n---\n\n## Licencia de CTD Photo Studio / FabriRAW\n\n```\n" + texts[1] + "\n```"
            tb.setMarkdown(md)
        else:
            tb.setPlainText(f"No se encontró THIRD_PARTY_NOTICES.md. Ver {repo}")
        lay.addWidget(tb)
        b = QPushButton("Cerrar")
        b.clicked.connect(self.accept)
        lay.addWidget(b, 0, Qt.AlignmentFlag.AlignRight)
