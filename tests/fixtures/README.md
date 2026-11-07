# Fixtures dorados FabriRAW

Archivos deterministas usados por los tests de C++ (`core/tests/test_golden.cpp`) y de
Python (`python/tests/test_format.py`) para verificar compatibilidad cruzada.

| Archivo | Compresión | Tamaño | SHA-256 |
|---------|-----------|-------:|---------|
| `golden.fraw`   | ninguna   | 6710 B | `5e22137da1b821e500541c1ef1abddc6a1157a7da3f5106c632b939f838dda3e` |
| `golden_z.fraw` | zlib nivel 6 | 6241 B | `ddf9e80e73f7abfb0b9a43e28d506625e4d760e3f04ebf4601e33b6808f531a8` |

## Regenerar

```sh
# Python (referencia de los archivos commiteados)
cd python
python -m fraw.format --make-fixture ../tests/fixtures/golden.fraw
python -m fraw.format --make-fixture ../tests/fixtures/golden_z.fraw --zlib

# C++ (debe producir exactamente el mismo golden.fraw)
core/build/fraw-info --make-fixture golden.fraw
core/build/fraw-info --make-fixture golden_z.fraw --zlib
```

## Contenido

* HEAD: 64×48, CFA RGGB (`0,1,1,2`), black 512 ×4, white 16383, effective_bits 14, orientation 1, sin crop.
* `raw[y][x] = 512 + (x*97 + y*131) % 15000` (u16 LE, row-major).
* META: `{"make":"FabriCTD","model":"Golden Test","iso":100}` (JSON compacto, UTF-8, claves en ese orden).
* COLR: as_shot `[2.0,1.0,1.5,1.0]`, daylight `[2.1,1.0,1.4,1.0]`,
  rgb_cam `[[1.6,-0.4,-0.2],[-0.2,1.5,-0.3],[0.0,-0.5,1.5]]` (f32), cam_xyz ceros, illuminant 21, flags 1.
* THMB: RGB8 8×6, color sólido (255,128,0).
* DEV: `{"version":1,"exposure_ev":0.5}`.

## Reglas de igualdad

* `golden.fraw` (sin comprimir) **DEBE** ser idéntico byte a byte entre C++ y Python
  (los tests comparan bytes y SHA-256).
* `golden_z.fraw`: los bytes del stream zlib pueden diferir entre builds de zlib
  (p. ej. CPython 3.14 en Windows usa zlib-ng; el core C++ usa madler/zlib 1.3.1 o la zlib
  del sistema). Por eso sólo se exige igualdad del **contenido decodificado**: re-serializar
  sin compresión debe dar exactamente `golden.fraw`. En la máquina de desarrollo actual
  (zlib-ng 1.3.1 en Python, madler/zlib 1.3.1 en C++) los bytes comprimidos coincidieron,
  pero no es un requisito.
