# Avisos de terceros — CTD Photo Studio / FabriRAW

CTD Photo Studio y la implementación de referencia FabriRAW (`fabriraw`, módulo `fraw`)
son © 2026 FabriCTD, licencia **MIT** (ver `LICENSE`). Hechos 100% con Claude (Anthropic)
a partir de un prompt de FabriCTD.

La versión portable para Windows (`CTD-Photo-Studio-Portable-win64.zip`) incluye los
componentes de terceros listados abajo, **sin modificar**, como bibliotecas dinámicas
(`.dll` / `.pyd`) dentro de la carpeta `_internal/`. Las versiones son las usadas para
construir el paquete (verificadas con `pip show` y los archivos de licencia de cada wheel).

| Componente | Versión | Licencia | Uso |
|---|---|---|---|
| Qt (vía PySide6 / Shiboken6) | 6.11.1 | **LGPL-3.0-only** (Qt también se ofrece bajo GPL-2.0/3.0 y licencia comercial) | interfaz gráfica |
| PySide6 / Shiboken6 | 6.11.1 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only — usado bajo **LGPL-3.0** | bindings Python de Qt |
| LibRaw (dentro de rawpy, `raw_r.dll`) | 0.22.2 | **LGPL-2.1** o **CDDL-1.0** (a elección) — el wheel de rawpy la distribuye bajo LGPL-2.1 | lectura de RAW de cámara |
| rawpy | 0.28.0 | MIT (© 2014 Maik Riechert) | bindings Python de LibRaw |
| NumPy | 2.5.2 | BSD-3-Clause (incluye partes 0BSD, MIT, Zlib, CC0-1.0; OpenBLAS BSD-3-Clause, LAPACK BSD-3-Clause-Open-MPI, runtime GCC GPL-3.0-or-later WITH GCC-exception-3.1) | cálculo numérico |
| SciPy (sólo `scipy.ndimage` y dependencias) | 1.18.0 | BSD-3-Clause (incluye OpenBLAS/LAPACK BSD y runtime GCC con GCC Runtime Library Exception) | demosaico |
| Pillow | 11.3.0 | MIT-CMU (licencia HPND histórica de PIL) | JPEG/PNG/TIFF y miniaturas |
| Bibliotecas incluidas en Pillow | — | libjpeg-turbo (IJG + BSD-3-Clause + Zlib), libpng (libpng-2.0), libtiff (libtiff), zlib-ng (Zlib), libwebp (BSD-3-Clause), openjpeg (BSD-2-Clause), freetype (FTL), harfbuzz (MIT), lcms2 (MIT), brotli (MIT), xz (0BSD/PD), libavif (BSD-2-Clause) | códecs de imagen |
| zlib / zlib-ng (en Python y en Pillow) | 1.3.1 (zlib-ng) | Zlib | compresión de `DATA` |
| Python (intérprete CPython embebido por PyInstaller) | 3.13 / 3.14 | PSF-2.0 (Python Software Foundation License) | runtime |
| Microsoft Visual C++ Runtime (`vcruntime140*.dll`, `vcomp140.dll`) | — | redistribuible de Microsoft (términos de Visual Studio) | runtime de C/C++ |
| PyInstaller (bootloader) | 6.22 | GPL-2.0-or-later **con excepción del bootloader**, que permite distribuir el ejecutable resultante con cualquier licencia | empaquetado |

Los textos completos de cada licencia están en la carpeta `licenses/` del paquete portable
(copiados de los `*.dist-info` de cada wheel; los textos LGPL-3.0 y GPL-3.0 de Qt/PySide6 en
`licenses/Qt-PySide6/`, porque los wheels de PySide6 sólo traen el aviso comercial) y en los
enlaces de abajo.

---

## Qt / PySide6 — GNU LGPL v3

Qt 6 y PySide6 se usan bajo la **GNU Lesser General Public License v3**
(<https://www.gnu.org/licenses/lgpl-3.0.html>, que incorpora la GPL v3:
<https://www.gnu.org/licenses/gpl-3.0.html>). Copyright © The Qt Company Ltd. y otros contribuyentes.

* **Código fuente:** Qt <https://download.qt.io/official_releases/qt/6.11/> ·
  <https://code.qt.io/cgit/qt/qtbase.git/> · PySide6/Shiboken6
  <https://code.qt.io/cgit/pyside/pyside-setup.git/> (tag `v6.11.1`) y el sdist en
  <https://pypi.org/project/PySide6/6.11.1/>. No modificamos Qt ni PySide6.
* **Cómo reemplazar las bibliotecas Qt (derecho que otorga la LGPL):** la versión portable se
  construye en modo *onedir* precisamente para esto. Las DLL de Qt y los módulos de PySide6
  están como archivos sueltos en `CTD Photo Studio\_internal\PySide6\` (`Qt6Core.dll`,
  `Qt6Gui.dll`, `Qt6Widgets.dll`, `plugins\…`) y `_internal\shiboken6\`. Para usar otra
  compilación (por ejemplo, una modificada por vos), instalá la misma versión mayor/menor
  de PySide6 con `pip install PySide6==6.11.*` (o compilala desde el código fuente) y
  copiá sus archivos encima de los de `_internal\PySide6\` y `_internal\shiboken6\`.
  También podés reconstruir todo el paquete desde el código fuente de este proyecto con
  `python packaging/build_portable.py` (ver README).
* Se excluyeron módulos de Qt que la aplicación no usa (WebEngine, Quick/QML, 3D,
  Multimedia, etc.) para reducir el tamaño; esto no modifica Qt.

## LibRaw — LGPL v2.1 / CDDL v1.0

LibRaw © 2008-2024 LibRaw LLC (<https://www.libraw.org>), basada en dcraw de Dave Coffin.
Doble licencia **LGPL-2.1** (<https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>)
o **CDDL-1.0** (<https://opensource.org/licenses/CDDL-1.0>). El wheel de rawpy incluye el
texto LGPL-2.1 (`rawpy-0.28.0.dist-info/licenses/LICENSE.LibRaw`).
Código fuente: <https://github.com/LibRaw/LibRaw> (0.22.x) y
<https://github.com/letmaik/rawpy> (rawpy 0.28.0, que compila LibRaw sin modificaciones).
LibRaw está enlazada dinámicamente (`_internal\rawpy\raw_r.dll`) y puede reemplazarse por
otra compilación compatible con la misma ABI.

## rawpy — MIT

Copyright © 2014 Maik Riechert. Permission is hereby granted, free of charge, to any person
obtaining a copy of this software… (texto MIT completo: <https://github.com/letmaik/rawpy/blob/main/LICENSE>).

## NumPy — BSD 3-Clause

Copyright © 2005-2025, NumPy Developers. All rights reserved.
<https://github.com/numpy/numpy/blob/main/LICENSE.txt>. El wheel incluye OpenBLAS
(BSD-3-Clause), LAPACK (BSD-3-Clause-Open-MPI) y el runtime de GCC/gfortran
(GPL-3.0-or-later WITH GCC-exception-3.1, que permite la redistribución).

## SciPy — BSD 3-Clause

Copyright © 2001-2002 Enthought, Inc. 2003, SciPy Developers. All rights reserved.
<https://github.com/scipy/scipy/blob/main/LICENSE.txt>. Incluye OpenBLAS/LAPACK (BSD) y el
runtime de GCC con la GCC Runtime Library Exception.

## Pillow — MIT-CMU (HPND)

The Python Imaging Library (PIL) is Copyright © 1997-2011 by Secret Labs AB and
Copyright © 1995-2011 by Fredrik Lundh and contributors. Pillow is Copyright © 2010 by
Jeffrey A. Clark and contributors. Licencia MIT-CMU:
<https://github.com/python-pillow/Pillow/blob/main/LICENSE> (incluye las licencias de
libjpeg-turbo, libpng, libtiff, zlib-ng, libwebp, openjpeg, freetype, harfbuzz, lcms2,
brotli, xz y libavif).

## zlib — Zlib License

© 1995-2024 Jean-loup Gailly y Mark Adler; zlib-ng © Hans Kristian Rosbach y contribuyentes.
<https://zlib.net/zlib_license.html>.

## Python — PSF License Agreement

Copyright © 2001-2026 Python Software Foundation. All rights reserved.
<https://docs.python.org/3/license.html>.

## PyInstaller — GPL-2.0-or-later con excepción del bootloader

El bootloader de PyInstaller incluido en `CTDPhotoStudio.exe` y `ctd-raw.exe` se distribuye
bajo la excepción que permite usarlo en programas con cualquier licencia.
<https://github.com/pyinstaller/pyinstaller/blob/develop/COPYING.txt>.
