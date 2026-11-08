# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — CTD Photo Studio portable (Windows, ONEDIR).

Two executables share one ``_internal`` folder (single COLLECT):
  * CTDPhotoStudio.exe — GUI, windowed (no console)
  * ctd-raw.exe        — CLI, console

ONEDIR (not onefile) keeps the Qt / LibRaw DLLs as separate, user-replaceable
files, as required by the LGPL (see THIRD_PARTY_NOTICES.md).

Build:  python packaging/build_portable.py   (or: pyinstaller packaging/ctd_studio.spec)
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 (SPECPATH injected by PyInstaller)
PY = ROOT / "python"
PKG = ROOT / "packaging"
ICON = str(PKG / "icon.ico")

import re

VERSION = re.search(r'__version__ = "([^"]+)"', (PY / "fraw" / "__init__.py").read_text(encoding="utf-8")).group(1)
_vt = tuple(int(x) for x in (VERSION.split(".") + ["0", "0", "0"])[:4] if x.isdigit())
VERSION_FILE = ROOT / "build" / "version_info.txt"
VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
VERSION_FILE.write_text(
    f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={_vt}, prodvers={_vt}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1,
                    subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040a04b0', [
      StringStruct('CompanyName', 'FabriCTD'),
      StringStruct('FileDescription', 'CTD Photo Studio - revelador FabriRAW'),
      StringStruct('FileVersion', '{VERSION}'),
      StringStruct('InternalName', 'CTDPhotoStudio'),
      StringStruct('LegalCopyright', 'Copyright (c) 2026 FabriCTD - Licencia MIT'),
      StringStruct('ProductName', 'CTD Photo Studio'),
      StringStruct('ProductVersion', '{VERSION}'),
      StringStruct('Comments', 'Hecho 100% con Claude (Anthropic) a partir de un prompt de FabriCTD')])]),
    VarFileInfo([VarStruct('Translation', [0x040a, 1200])])
  ]
)
""",
    encoding="utf-8",
)

# Qt modules / Python packages the app never uses (keeps the bundle small).
QT_UNUSED = [
    "QtWebEngine", "QtWebEngineCore", "QtWebEngineWidgets", "QtWebEngineQuick", "QtWebChannel", "QtWebSockets",
    "QtWebView", "Qt3DCore", "Qt3DRender", "Qt3DInput", "Qt3DLogic", "Qt3DAnimation", "Qt3DExtras",
    "QtQuick", "QtQuick3D", "QtQuickControls2", "QtQuickWidgets", "QtQml", "QtQuickTest",
    "QtMultimedia", "QtMultimediaWidgets", "QtSpatialAudio", "QtTextToSpeech",
    "QtCharts", "QtDataVisualization", "QtGraphs", "QtGraphsWidgets", "QtPdf", "QtPdfWidgets",
    "QtBluetooth", "QtNfc", "QtSensors", "QtSerialPort", "QtSerialBus", "QtPositioning", "QtLocation",
    "QtRemoteObjects", "QtScxml", "QtStateMachine", "QtSql", "QtTest", "QtDesigner", "QtHelp", "QtUiTools",
    "QtNetwork", "QtNetworkAuth", "QtHttpServer", "QtOpenGL", "QtOpenGLWidgets", "QtSvgWidgets",
    "QtXml", "QtConcurrent", "QtDBus", "QtAxContainer", "QtCanvasPainter", "QtVirtualKeyboard",
]
EXCLUDES = [f"PySide6.{m}" for m in QT_UNUSED] + [
    "tkinter", "_tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "_pytest",
    "PyQt5", "PyQt6", "PySide2", "pandas", "sympy", "numba", "cv2", "imageio", "skimage",
    "scipy.stats", "scipy.optimize", "scipy.integrate", "scipy.interpolate", "scipy.signal", "scipy.io",
    "scipy.spatial", "scipy.cluster", "scipy.odr", "scipy.datasets", "scipy.misc",
    "scipy.differentiate", "scipy.fft", "scipy.fftpack",  # scipy.sparse: lo usa fraw.pipeline (remuestreo)
    "PIL.ImageTk", "PIL.ImageQt",
    # pulled in only by optional code paths of dependencies (network, YAML, CFFI…)
    "ssl", "_ssl", "yaml", "cffi", "charset_normalizer", "requests", "urllib3", "certifi",
    "setuptools", "pkg_resources", "distutils", "lib2to3", "pydoc_data", "xmlrpc",
    "psutil", "win32api", "win32con", "win32com", "pywintypes", "pythoncom", "win32ui",
]

# DLLs / data pulled in by hooks that the app does not need.
DROP_SUBSTR = (
    "opengl32sw.dll",  # Mesa software OpenGL (~20 MB); the canvas uses QPainter, no OpenGL
    "Qt6Quick", "Qt6Qml", "Qt6Pdf", "Qt6Network", "Qt6OpenGL", "Qt6VirtualKeyboard", "Qt6WebEngine",
    "Qt6Multimedia", "Qt6Charts", "Qt63D", "Qt6Designer", "Qt6ShaderTools",
    "qpdf.dll", "qtvirtualkeyboardplugin", "networkinformation", "qtuiotouchplugin", "tls",
    "qdirect2d.dll",  # alternative platform plugin; qwindows.dll is the one used
    "pyside6/translations", "pyside6/qml",  # dest paths are normalized to "/" and lowercase
)


def _keep(entry) -> bool:
    dest = entry[0].replace("\\", "/").lower()
    return not any(s.lower() in dest for s in DROP_SUBSTR)


HIDDEN = collect_submodules("fraw") + ["rawpy", "rawpy._rawpy", "PIL.Image", "PIL.JpegImagePlugin",
                                       "PIL.PngImagePlugin", "PIL.TiffImagePlugin", "PIL.ImageCms"]
DATAS = [
    (str(PY / "fraw" / "gui" / "theme.qss"), "fraw/gui"),
    (str(PY / "fraw" / "gui" / "icon.png"), "fraw/gui"),
    *[(str(f), "fraw/gui/presets") for f in sorted((PY / "fraw" / "gui" / "presets").glob("*.json"))],
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
    (str(ROOT / "LICENSE"), "."),
]

a_gui = Analysis(  # noqa: F821
    [str(PKG / "entry_studio.py")],
    pathex=[str(PY)],
    datas=DATAS,
    hiddenimports=HIDDEN,
    excludes=EXCLUDES,
    noarchive=False,
    optimize=1,
)
a_gui.binaries = [b for b in a_gui.binaries if _keep(b)]
a_gui.datas = [d for d in a_gui.datas if _keep(d)]

a_cli = Analysis(  # noqa: F821
    [str(PKG / "entry_cli.py")],
    pathex=[str(PY)],
    datas=[],
    hiddenimports=[h for h in HIDDEN if not h.startswith("fraw.gui")],
    excludes=EXCLUDES + ["PySide6", "shiboken6", "fraw.gui"],
    noarchive=False,
    optimize=1,
)

pyz_gui = PYZ(a_gui.pure)  # noqa: F821
pyz_cli = PYZ(a_cli.pure)  # noqa: F821

exe_gui = EXE(  # noqa: F821
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name="CTDPhotoStudio",
    console=False,
    icon=ICON,
    upx=False,
    version=str(VERSION_FILE),
)
exe_cli = EXE(  # noqa: F821
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="ctd-raw",
    console=True,
    icon=ICON,
    upx=False,
    version=str(VERSION_FILE),
)

coll = COLLECT(  # noqa: F821
    exe_gui,
    a_gui.binaries,
    a_gui.datas,
    exe_cli,
    a_cli.binaries,
    a_cli.datas,
    upx=False,
    name="CTD Photo Studio",
)
