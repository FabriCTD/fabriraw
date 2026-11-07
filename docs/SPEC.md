# FabriRAW (.fraw) — Especificación del formato v1.0

> Documento normativo. Autor del formato: **FabriCTD** · CTD Photo Studio · https://fraw.fabrictd.com
> Las palabras DEBE / NO DEBE / PUEDE se usan en sentido RFC 2119.

## 1. Extensiones

| Extensión | Contenido |
|-----------|-----------|
| `.fraw`   | Archivo FabriRAW principal (una imagen). |
| `.frw`    | Idéntico byte a byte a `.fraw`; extensión corta (8.3). |
| `.fset`   | Contenedor de ráfaga: N archivos `.fraw` embebidos (ver §6). |

## 2. Convenciones

* **Endianness:** todos los enteros y floats son **little-endian**.
* `u8/u16/u32/u64` = enteros sin signo; `f32` = IEEE-754 binary32.
* Todos los structs binarios se empaquetan sin padding (`#pragma pack(push, 1)` en C/C++, `struct` con prefijo `<` en Python).
* Los campos `reserved` **DEBEN** escribirse en cero y **DEBEN** ignorarse al leer.
* Texto (META, DEV ) = **UTF-8 JSON**, sin BOM, objeto en la raíz.

## 3. Estructura del archivo

```
+------------------------------+
| Magic (8 bytes)              |  46 52 41 57 0D 0A 1A 0A   "FRAW\r\n\x1a\n"
+------------------------------+
| Chunk HEAD                   |  obligatorio, SIEMPRE primero
| Chunk META                   |  opcional
| Chunk COLR                   |  obligatorio
| Chunk THMB                   |  opcional
| Chunk DATA                   |  obligatorio
| Chunk DEV                    |  opcional (después de DATA)
| Chunk END                    |  obligatorio, SIEMPRE último
+------------------------------+
```

El magic sigue la idea de PNG: `\r\n` detecta conversiones de fin de línea, `\x1a` detiene `type` en DOS, `\n` final detecta la conversión inversa.

Orden: los writers **DEBEN** emitir los chunks en el orden de la tabla. Los readers **DEBEN** exigir `HEAD` primero y `END ` último, y **DEBEN** aceptar META/COLR/THMB en cualquier orden entre HEAD y DATA. `DEV ` va después de `DATA` para que una edición no destructiva sólo reescriba la cola del archivo.

### 3.1 Framing de chunk (20 bytes + payload)

```c
#pragma pack(push, 1)
struct ChunkHeader {      // 16 bytes
    char     id[4];       // ASCII, p.ej. "HEAD", "DEV " (con espacio)
    uint32_t flags;       // bit0 = CRITICAL; resto reservado (0)
    uint64_t length;      // bytes de payload
};
#pragma pack(pop)
// payload[length]
// uint32_t crc32;        // CRC-32 IEEE 802.3 (el de zlib) sobre id[4] + payload
```

* El CRC cubre los 4 bytes del id **y** el payload (no flags ni length).
* Chunk con CRC incorrecto ⇒ error `CorruptChunk`.
* Chunk desconocido con `CRITICAL` ⇒ error `UnsupportedCriticalChunk`. Sin `CRITICAL` ⇒ se ignora (se PUEDE preservar).
* Un chunk conocido duplicado ⇒ error.

| ID     | Flags CRITICAL | Contenido |
|--------|:---:|-----------|
| `HEAD` | 1 | Geometría y niveles del sensor (§4.1) |
| `META` | 0 | Metadatos EXIF-like en JSON (§4.2) |
| `COLR` | 1 | Balance de blancos y matrices de color (§4.3) |
| `THMB` | 0 | Miniatura (§4.4) |
| `DATA` | 1 | Mosaico Bayer sin interpolar (§4.5) |
| `DEV ` | 0 | Parámetros de revelado no destructivo (§4.6) |
| `END ` | 1 | Fin de archivo, `length = 0` |

Nota: el ID de color es **`COLR`** (los IDs son exactamente 4 bytes; `COLOR` tiene 5).

## 4. Payloads

### 4.1 HEAD — 64 bytes

| Off | Tipo | Campo | Notas |
|----:|------|-------|-------|
| 0  | u16 | version_major | `1`. Un reader DEBE rechazar otro major. |
| 2  | u16 | version_minor | `0`. Minors mayores se aceptan. |
| 4  | u32 | width | píxeles del mosaico completo |
| 8  | u32 | height | |
| 12 | u16 | bits_per_sample | contenedor, SIEMPRE `16` en v1 |
| 14 | u16 | effective_bits | bits reales del ADC (10–16) |
| 16 | u8[4] | cfa_pattern | 2×2 row-major desde (0,0): `0=R 1=G 2=B` |
| 20 | u8 | compression | `0` = ninguna, `1` = zlib (RFC 1950) |
| 21 | u8 | orientation | EXIF 1–8 (1 = normal) |
| 22 | u8 | sample_layout | `0` = Bayer 2×2. Otros valores reservados. |
| 23 | u8 | reserved0 | |
| 24 | u32[4] | black_level | por posición CFA (mismo orden que cfa_pattern) |
| 40 | u32 | white_level | valor de saturación |
| 44 | u32 | crop_left | área activa; los 4 en 0 = imagen completa |
| 48 | u32 | crop_top | |
| 52 | u32 | crop_width | |
| 56 | u32 | crop_height | |
| 60 | u32 | reserved1 | |

Validaciones: `width,height ≥ 2` y pares; `bits_per_sample == 16`; `effective_bits ∈ [8,16]`; cada `cfa_pattern[i] ∈ {0,1,2}` y contiene al menos un 0, un 1 y un 2; `max(black_level) < white_level ≤ 65535`; si hay crop, debe caber en la imagen.

### 4.2 META — JSON

Claves estándar (todas opcionales; claves extra permitidas):

```json
{
  "make": "Canon", "model": "EOS R5", "lens": "RF24-70mm F2.8 L IS USM",
  "iso": 100, "shutter_s": 0.004, "aperture": 2.8, "focal_mm": 50.0,
  "datetime": "2026-11-07T19:09:00", "serial": "…",
  "source_file": "IMG_0001.CR3", "source_format": "CR3",
  "software": "CTD Photo Studio 0.1.0", "artist": "…", "copyright": "…"
}
```

### 4.3 COLR — 112 bytes

| Off | Tipo | Campo | Notas |
|----:|------|-------|-------|
| 0   | f32[4] | as_shot_wb | multiplicadores por **color** `[R, G, B, G2]`, normalizados G = 1 |
| 16  | f32[4] | daylight_wb | ídem, para luz día (pre_mul) |
| 32  | f32[9] | rgb_cam | cámara (post-WB) → sRGB lineal D65, row-major; filas suman ≈ 1 |
| 68  | f32[9] | cam_xyz | XYZ → cámara (estilo DNG ColorMatrix), puede ser 0 |
| 104 | u32 | illuminant | código EXIF LightSource (21 = D65) |
| 108 | u32 | flags | bit0 = rgb_cam válida, bit1 = cam_xyz válida |

Si `flags.bit0 == 0` el procesador usa identidad.

### 4.4 THMB — 16 bytes + datos

| Off | Tipo | Campo |
|----:|------|-------|
| 0  | u8 | format: `0` = JPEG, `1` = RGB8 empaquetado (w·h·3) |
| 1  | u8[3] | reserved |
| 4  | u32 | width |
| 8  | u32 | height |
| 12 | u32 | data_length |
| 16 | u8[data_length] | datos |

`16 + data_length` DEBE ser igual al `length` del chunk.

### 4.5 DATA

* `compression == 0`: `width × height` muestras `u16` LE, row-major, **sin** restar negro, **sin** interpolar.
* `compression == 1`: stream zlib (RFC 1950) que descomprime exactamente a lo anterior.
* Tamaño descomprimido distinto de `width·height·2` ⇒ error.

### 4.6 DEV — JSON (revelado no destructivo)

```json
{
  "version": 1,
  "exposure_ev": 0.0,          // [-4, +4]
  "wb_mode": "as_shot",         // "as_shot" | "daylight" | "custom"
  "wb_custom": [2.0, 1.0, 1.5], // [R, G, B] sólo si wb_mode == "custom"
  "temperature_shift": 0.0,     // [-1, +1] ajuste cálido/frío sobre el WB elegido
  "tint_shift": 0.0,            // [-1, +1] verde/magenta
  "contrast": 0.0,              // [-1, +1]
  "saturation": 1.0,            // [0, 2]
  "highlights": 0.0,            // [-1, +1]
  "shadows": 0.0,               // [-1, +1]
  "gamma_mode": "srgb",         // "srgb" | "power" | "linear"
  "gamma": 2.2,                 // sólo para "power"
  "demosaic": "ahd"             // "bilinear" | "ahd"
}
```

Valores fuera de rango se recortan al rango. Claves ausentes = valor por defecto de arriba.

## 5. Pipeline de revelado de referencia

1. `x = (raw − black[cfa]) / (white − black[cfa])`, recorte [0,1].
2. WB: multiplicar cada píxel por el multiplicador de su color.
3. Demosaico (bilineal o AHD) → RGB cámara.
4. `rgb = rgb_cam · rgb`.
5. Exposición: `rgb · 2^exposure_ev`; luego highlights/shadows/contraste/saturación.
6. Curva de transferencia (sRGB / power / linear), recorte, cuantización a 8 o 16 bits.

## 6. Contenedor de ráfaga `.fset`

```
Magic: 46 53 45 54 0D 0A 1A 0A   "FSET\r\n\x1a\n"
SETH  (CRITICAL) JSON {"version":1,"frame_count":N,"burst_id":"…","created":"…"}
FRAM  (CRITICAL) × N — payload = un archivo .fraw completo (con su magic)
END   (CRITICAL) length 0
```

Mismo framing y CRC que `.fraw`. `frame_count` DEBE coincidir con la cantidad de `FRAM`.

## 7. Errores normativos

`BadMagic`, `Truncated`, `CorruptChunk` (CRC), `UnsupportedVersion`, `UnsupportedCriticalChunk`, `MissingChunk`, `DuplicateChunk`, `BadOrder`, `InvalidHeader`, `SizeMismatch`.
