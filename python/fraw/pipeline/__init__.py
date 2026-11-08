"""Stages of the FabriRAW development pipeline (see :mod:`fraw.processor` for the order).

Modules: :mod:`.common` (filters, resampling, threading), :mod:`.wb` (Kelvin,
auto WB, highlight recovery), :mod:`.color` (calibration, saturation,
defringe), :mod:`.geometry` (lens optics + geometry), :mod:`.tone` (curves,
local tone, dehaze), :mod:`.hsl`, :mod:`.grading`, :mod:`.detail` (NR,
sharpening), :mod:`.effects` (vignette, grain), :mod:`.autoadjust`,
:mod:`.icc` (output profile + EXIF).
"""
