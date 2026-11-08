# Parámetros de revelado — DEV v2

> Generado desde `python/fraw/devparams.py` (`SCHEMA`), que es la fuente de verdad. Valores tal como se guardan en el JSON del chunk `DEV ` (la columna *UI ×* indica el factor con que la interfaz los muestra).

Reglas: claves ausentes = defecto; fuera de rango = se recorta; tipo inválido = defecto; claves desconocidas = se ignoran. Un archivo DEV v1 es un DEV v2 válido. Con todos los valores en su defecto, el revelado es idéntico al de DEV v1.

## Básico (`basic`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `wb_mode` | Balance de blancos | enum | `as_shot`, `daylight`, `kelvin`, `custom` | `as_shot` | 1 | Balance de blancos |
| `wb_kelvin` | Temperatura | float | 2000 … 50000 K | `5500.0` | 1 | Balance de blancos |
| `wb_kelvin_tint` | Matiz | float | -150 … 150 | `0.0` | 1 | Balance de blancos |
| `wb_custom` | Multiplicadores WB | wb3 | 3 × 0.001 … 64 | `(2.0, 1.0, 1.5)` | 1 | Balance de blancos |
| `temperature_shift` | Temperatura (ajuste fino) | float | -1 … 1 | `0.0` | 100 | Balance de blancos |
| `tint_shift` | Matiz (ajuste fino) | float | -1 … 1 | `0.0` | 100 | Balance de blancos |
| `exposure_ev` | Exposición | float | -4 … 4 EV | `0.0` | 1 | Tono |
| `contrast` | Contraste | float | -1 … 1 | `0.0` | 100 | Tono |
| `highlights` | Altas luces | float | -1 … 1 | `0.0` | 100 | Tono |
| `shadows` | Sombras | float | -1 … 1 | `0.0` | 100 | Tono |
| `whites` | Blancos | float | -1 … 1 | `0.0` | 100 | Tono |
| `blacks` | Negros | float | -1 … 1 | `0.0` | 100 | Tono |
| `texture` | Textura | float | -1 … 1 | `0.0` | 100 | Presencia |
| `clarity` | Claridad | float | -1 … 1 | `0.0` | 100 | Presencia |
| `dehaze` | Neblina | float | -1 … 1 | `0.0` | 100 | Presencia |
| `vibrance` | Intensidad | float | -1 … 1 | `0.0` | 100 | Presencia |
| `saturation` | Saturación | float | 0 … 2 | `1.0` | 100 | Presencia |

## Curva de tonos (`tone_curve`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `curve_p_highlights` | Iluminaciones | float | -1 … 1 | `0.0` | 100 | Curva paramétrica |
| `curve_p_lights` | Claros | float | -1 … 1 | `0.0` | 100 | Curva paramétrica |
| `curve_p_darks` | Oscuros | float | -1 … 1 | `0.0` | 100 | Curva paramétrica |
| `curve_p_shadows` | Sombras | float | -1 … 1 | `0.0` | 100 | Curva paramétrica |
| `curve_master` | Curva RGB | curve | 2–16 puntos `[x,y]` en [0,1] | `[[0,0],[1,1]]` | 1 | Curva de puntos |
| `curve_red` | Curva rojo | curve | 2–16 puntos `[x,y]` en [0,1] | `[[0,0],[1,1]]` | 1 | Curva de puntos |
| `curve_green` | Curva verde | curve | 2–16 puntos `[x,y]` en [0,1] | `[[0,0],[1,1]]` | 1 | Curva de puntos |
| `curve_blue` | Curva azul | curve | 2–16 puntos `[x,y]` en [0,1] | `[[0,0],[1,1]]` | 1 | Curva de puntos |

## Mezclador de color (HSL) (`hsl`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `hsl_red_h` | Rojo | float | -1 … 1 | `0.0` | 100 | Tono |
| `hsl_orange_h` | Naranja | float | -1 … 1 | `0.0` | 100 | Tono |
| `hsl_yellow_h` | Amarillo | float | -1 … 1 | `0.0` | 100 | Tono |
| `hsl_green_h` | Verde | float | -1 … 1 | `0.0` | 100 | Tono |
| `hsl_aqua_h` | Aguamarina | float | -1 … 1 | `0.0` | 100 | Tono |
| `hsl_blue_h` | Azul | float | -1 … 1 | `0.0` | 100 | Tono |
| `hsl_purple_h` | Púrpura | float | -1 … 1 | `0.0` | 100 | Tono |
| `hsl_magenta_h` | Magenta | float | -1 … 1 | `0.0` | 100 | Tono |
| `hsl_red_s` | Rojo | float | -1 … 1 | `0.0` | 100 | Saturación |
| `hsl_orange_s` | Naranja | float | -1 … 1 | `0.0` | 100 | Saturación |
| `hsl_yellow_s` | Amarillo | float | -1 … 1 | `0.0` | 100 | Saturación |
| `hsl_green_s` | Verde | float | -1 … 1 | `0.0` | 100 | Saturación |
| `hsl_aqua_s` | Aguamarina | float | -1 … 1 | `0.0` | 100 | Saturación |
| `hsl_blue_s` | Azul | float | -1 … 1 | `0.0` | 100 | Saturación |
| `hsl_purple_s` | Púrpura | float | -1 … 1 | `0.0` | 100 | Saturación |
| `hsl_magenta_s` | Magenta | float | -1 … 1 | `0.0` | 100 | Saturación |
| `hsl_red_l` | Rojo | float | -1 … 1 | `0.0` | 100 | Luminancia |
| `hsl_orange_l` | Naranja | float | -1 … 1 | `0.0` | 100 | Luminancia |
| `hsl_yellow_l` | Amarillo | float | -1 … 1 | `0.0` | 100 | Luminancia |
| `hsl_green_l` | Verde | float | -1 … 1 | `0.0` | 100 | Luminancia |
| `hsl_aqua_l` | Aguamarina | float | -1 … 1 | `0.0` | 100 | Luminancia |
| `hsl_blue_l` | Azul | float | -1 … 1 | `0.0` | 100 | Luminancia |
| `hsl_purple_l` | Púrpura | float | -1 … 1 | `0.0` | 100 | Luminancia |
| `hsl_magenta_l` | Magenta | float | -1 … 1 | `0.0` | 100 | Luminancia |

## Gradación de color (`grading`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `grade_shadows_hue` | Tono | float | 0 … 360 ° | `0.0` | 1 | Sombras |
| `grade_shadows_sat` | Saturación | float | 0 … 1 | `0.0` | 100 | Sombras |
| `grade_shadows_lum` | Luminancia | float | -1 … 1 | `0.0` | 100 | Sombras |
| `grade_midtones_hue` | Tono | float | 0 … 360 ° | `0.0` | 1 | Medios tonos |
| `grade_midtones_sat` | Saturación | float | 0 … 1 | `0.0` | 100 | Medios tonos |
| `grade_midtones_lum` | Luminancia | float | -1 … 1 | `0.0` | 100 | Medios tonos |
| `grade_highlights_hue` | Tono | float | 0 … 360 ° | `0.0` | 1 | Iluminaciones |
| `grade_highlights_sat` | Saturación | float | 0 … 1 | `0.0` | 100 | Iluminaciones |
| `grade_highlights_lum` | Luminancia | float | -1 … 1 | `0.0` | 100 | Iluminaciones |
| `grade_global_hue` | Tono | float | 0 … 360 ° | `0.0` | 1 | Global |
| `grade_global_sat` | Saturación | float | 0 … 1 | `0.0` | 100 | Global |
| `grade_global_lum` | Luminancia | float | -1 … 1 | `0.0` | 100 | Global |
| `grade_blending` | Fusión | float | 0 … 1 | `0.5` | 100 | Ajustes |
| `grade_balance` | Equilibrio | float | -1 … 1 | `0.0` | 100 | Ajustes |

## Detalle (`detail`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `sharpen_amount` | Cantidad | float | 0 … 1.5 | `0.0` | 100 | Enfoque |
| `sharpen_radius` | Radio | float | 0.5 … 3 px | `1.0` | 1 | Enfoque |
| `sharpen_detail` | Detalle | float | 0 … 1 | `0.25` | 100 | Enfoque |
| `sharpen_masking` | Máscara | float | 0 … 1 | `0.0` | 100 | Enfoque |
| `nr_luminance` | Luminancia | float | 0 … 1 | `0.0` | 100 | Reducción de ruido |
| `nr_luminance_detail` | Detalle de luminancia | float | 0 … 1 | `0.5` | 100 | Reducción de ruido |
| `nr_color` | Color | float | 0 … 1 | `0.0` | 100 | Reducción de ruido |
| `nr_color_detail` | Detalle de color | float | 0 … 1 | `0.5` | 100 | Reducción de ruido |

## Óptica (`optics`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `lens_distortion` | Distorsión | float | -1 … 1 | `0.0` | 100 | Corrección de lente |
| `lens_vignette` | Viñeteo de lente | float | -1 … 1 | `0.0` | 100 | Corrección de lente |
| `ca_red_cyan` | Aberración rojo/cian | float | -1 … 1 | `0.0` | 100 | Aberración cromática |
| `ca_blue_yellow` | Aberración azul/amarillo | float | -1 … 1 | `0.0` | 100 | Aberración cromática |
| `defringe` | Eliminar halos | float | 0 … 1 | `0.0` | 100 | Aberración cromática |

## Geometría y recorte (`geometry`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `rotate_90` | Girar | enum | `0`, `90`, `180`, `270` | `0` | 1 | Orientación |
| `flip_h` | Voltear horizontal | bool | true / false | `False` | 1 | Orientación |
| `flip_v` | Voltear vertical | bool | true / false | `False` | 1 | Orientación |
| `rotate_deg` | Enderezar | float | -45 … 45 ° | `0.0` | 1 | Enderezar |
| `perspective_vertical` | Perspectiva vertical | float | -1 … 1 | `0.0` | 100 | Transformar |
| `perspective_horizontal` | Perspectiva horizontal | float | -1 … 1 | `0.0` | 100 | Transformar |
| `perspective_scale` | Escala | float | 0.5 … 1.5 | `1.0` | 100 | Transformar |
| `crop_aspect` | Proporción | enum | `free`, `original`, `1:1`, `4:5`, `5:4`, `3:2`, `2:3`, `16:9`, `9:16`, `4:3`, `3:4` | `free` | 1 | Recorte |
| `crop_left` | Izquierda | float | 0 … 1 | `0.0` | 100 | Recorte |
| `crop_top` | Arriba | float | 0 … 1 | `0.0` | 100 | Recorte |
| `crop_right` | Derecha | float | 0 … 1 | `1.0` | 100 | Recorte |
| `crop_bottom` | Abajo | float | 0 … 1 | `1.0` | 100 | Recorte |

## Efectos (`effects`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `vignette_amount` | Cantidad | float | -1 … 1 | `0.0` | 100 | Viñeta tras recorte |
| `vignette_midpoint` | Punto medio | float | 0 … 1 | `0.5` | 100 | Viñeta tras recorte |
| `vignette_roundness` | Redondez | float | -1 … 1 | `0.0` | 100 | Viñeta tras recorte |
| `vignette_feather` | Difuminado | float | 0 … 1 | `0.5` | 100 | Viñeta tras recorte |
| `vignette_highlights` | Altas luces | float | 0 … 1 | `0.0` | 100 | Viñeta tras recorte |
| `grain_amount` | Cantidad | float | 0 … 1 | `0.0` | 100 | Grano |
| `grain_size` | Tamaño | float | 0 … 1 | `0.25` | 100 | Grano |
| `grain_roughness` | Aspereza | float | 0 … 1 | `0.5` | 100 | Grano |

## Calibración (`calibration`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `cal_shadows_tint` | Matiz de sombras | float | -1 … 1 | `0.0` | 100 | Sombras |
| `cal_red_hue` | Tono | float | -1 … 1 | `0.0` | 100 | Primario rojo |
| `cal_red_sat` | Saturación | float | -1 … 1 | `0.0` | 100 | Primario rojo |
| `cal_green_hue` | Tono | float | -1 … 1 | `0.0` | 100 | Primario verde |
| `cal_green_sat` | Saturación | float | -1 … 1 | `0.0` | 100 | Primario verde |
| `cal_blue_hue` | Tono | float | -1 … 1 | `0.0` | 100 | Primario azul |
| `cal_blue_sat` | Saturación | float | -1 … 1 | `0.0` | 100 | Primario azul |

## Proceso (`process`)

| Clave | Nombre | Tipo | Rango | Defecto | UI × | Sección |
|---|---|---|---|---|---|---|
| `demosaic` | Demosaico | enum | `ahd`, `bilinear` | `ahd` | 1 | Proceso |
| `highlight_mode` | Altas luces quemadas | enum | `clip`, `blend`, `reconstruct` | `clip` | 1 | Proceso |
| `gamma_mode` | Curva de salida | enum | `srgb`, `power`, `linear` | `srgb` | 1 | Proceso |
| `gamma` | Gamma | float | 1 … 4 | `2.2` | 1 | Proceso |
