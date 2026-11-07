"""Builds the portable Windows package of CTD Photo Studio.

    python packaging/build_portable.py [--skip-icon]

Steps: clean build/ and dist/ → (re)generate icons → PyInstaller (packaging/ctd_studio.spec,
ONEDIR) → add LEEME.txt, LICENSE, THIRD_PARTY_NOTICES.md and the license files of the
bundled wheels → dist/CTD-Photo-Studio-Portable-win64.zip with a single root folder
"CTD Photo Studio/". Prints the size of the folder and the zip.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packaging"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_DIR_NAME = "CTD Photo Studio"
ZIP_NAME = "CTD-Photo-Studio-Portable-win64.zip"
LICENSE_DISTS = ("PySide6", "PySide6_Essentials", "shiboken6", "rawpy", "numpy", "scipy", "pillow")


def version() -> str:
    text = (ROOT / "python" / "fraw" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__ = "([^"]+)"', text).group(1)


LEEME = """CTD Photo Studio {ver} — versión portable para Windows (64 bits)
=====================================================================

Revelador y conversor del formato RAW abierto FabriRAW (.fraw / .frw / .fset).
Hecho 100% con Claude (Anthropic) a partir de un prompt de FabriCTD.
Sitio: https://fraw.fabrictd.com · Código: https://github.com/FabriCTD/fabriraw

CÓMO USAR
---------
1. Descomprimí el ZIP donde quieras (Escritorio, un pendrive, C:\\Herramientas…).
   No requiere instalación ni permisos de administrador.
2. Abrí la carpeta "CTD Photo Studio" y ejecutá CTDPhotoStudio.exe.
3. "Importar RAW" convierte fotos de cámara (CR3, CR2, NEF, ARW, DNG, RAF*, ORF, RW2,
   PEF…) a .fraw; "Abrir" abre .fraw / .frw / .fset. Ajustá con los controles de la
   derecha (doble click en un control = restablecer) y usá "Guardar revelado" (Ctrl+S)
   para guardar los ajustes dentro del .fraw sin tocar los datos del sensor.
   "Exportar" genera JPEG, PNG o TIFF (8/16 bits).
   (*) Sólo sensores Bayer: los X-Trans de Fujifilm no están soportados.

Mantené juntos CTDPhotoStudio.exe, ctd-raw.exe y la carpeta _internal: los .exe no
funcionan si los separás.

LÍNEA DE COMANDOS
-----------------
ctd-raw.exe (en la misma carpeta) es la herramienta de consola. Ejemplos, desde una
terminal abierta en esta carpeta:

    ctd-raw.exe --version
    ctd-raw.exe ingest C:\\Fotos\\*.CR3 --outdir C:\\Fotos\\fraw
    ctd-raw.exe info foto.fraw
    ctd-raw.exe develop foto.fraw -o foto.jpg --ev 0.5 --wb daylight
    ctd-raw.exe dev foto.fraw --set exposure_ev=0.7 contrast=0.2
    ctd-raw.exe verify *.fraw

Ayuda completa: ctd-raw.exe --help  ·  ctd-raw.exe develop --help

DATOS DE USUARIO
----------------
* El programa no escribe nada dentro de su propia carpeta.
* Preferencias (última carpeta, tamaño de la ventana) en el registro de Windows,
  sólo de tu usuario: HKEY_CURRENT_USER\\Software\\FabriCTD\\CTD Photo Studio.
* Asociar .fraw/.frw/.fset con el programa es opcional (menú Herramientas) y se guarda
  en HKEY_CURRENT_USER\\Software\\Classes; "Quitar asociación" lo deshace.
* Para desinstalar: "Quitar asociación" (si la hiciste) y borrá la carpeta. Opcional:
  borrar la clave HKCU\\Software\\FabriCTD.

LICENCIAS
---------
CTD Photo Studio / FabriRAW: MIT (LICENSE). Componentes de terceros (Qt/PySide6 LGPL-3.0,
LibRaw LGPL-2.1/CDDL-1.0, rawpy, NumPy, SciPy, Pillow, Python…): ver
THIRD_PARTY_NOTICES.md y la carpeta "licenses". Las DLL de Qt están en
_internal\\PySide6 y se pueden reemplazar (LGPL).
"""


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def copy_wheel_licenses(dst: Path) -> None:
    for name in LICENSE_DISTS:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        out = dst / dist.metadata["Name"]
        for f in dist.files or []:
            parts = [p.lower() for p in f.parts]
            if not parts[0].endswith(".dist-info"):
                continue
            if "licen" in parts[-1] or "copying" in parts[-1] or (len(parts) > 2 and parts[1] == "licenses"):
                src = Path(dist.locate_file(f))
                if src.is_file():
                    target = out / Path(*f.parts[1:])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-icon", action="store_true", help="no regenerar icon.png / icon.ico")
    a = ap.parse_args()
    if sys.platform != "win32":
        print("aviso: el paquete portable está pensado para Windows", file=sys.stderr)

    ver = version()
    print(f"== CTD Photo Studio {ver}: limpiando build/ y dist/")
    for d in (BUILD, DIST):
        if d.exists():
            shutil.rmtree(d)
    if not a.skip_icon:
        subprocess.run([sys.executable, str(PKG / "make_icon.py")], check=True)

    print("== PyInstaller (onedir)")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(DIST), "--workpath", str(BUILD / "pyinstaller"), str(PKG / "ctd_studio.spec")],
        check=True,
        cwd=str(ROOT),
    )
    app = DIST / APP_DIR_NAME
    for exe in ("CTDPhotoStudio.exe", "ctd-raw.exe"):
        if not (app / exe).is_file():
            raise SystemExit(f"falta {exe} en {app}")

    print("== Archivos extra")
    (app / "LEEME.txt").write_text(LEEME.format(ver=ver).replace("\n", "\r\n"), encoding="utf-8")
    shutil.copy2(ROOT / "LICENSE", app / "LICENSE.txt")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", app / "THIRD_PARTY_NOTICES.md")
    copy_wheel_licenses(app / "licenses")
    # Full LGPL-3.0 / GPL-3.0 texts for Qt/PySide6 (the PySide6 wheels only ship the commercial notice).
    qt_lic = app / "licenses" / "Qt-PySide6"
    qt_lic.mkdir(parents=True, exist_ok=True)
    for f in (PKG / "licenses").glob("*.txt"):
        shutil.copy2(f, qt_lic / f.name)

    print("== ZIP")
    zpath = DIST / ZIP_NAME
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(app.rglob("*")):
            arc = Path(APP_DIR_NAME) / f.relative_to(app)
            if f.is_dir():
                continue
            z.write(f, arc.as_posix())
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    (DIST / "SHA256SUMS.txt").write_text(f"{sha}  {ZIP_NAME}\n", encoding="utf-8")
    print(f"Carpeta: {app}  ({human(dir_size(app))}, {sum(1 for _ in app.rglob('*') if _.is_file())} archivos)")
    print(f"ZIP:     {zpath}  ({human(zpath.stat().st_size)}, {zpath.stat().st_size} bytes)")
    print(f"SHA-256: {sha}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
