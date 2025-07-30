;+
; :Author: Gian-Mateo (GM) Tifone
; :Project: MISHA RIT
; :Copyright: MIT
; :Date: July 29 2025
;
; PURPOSE:
;   Implements the Band Generation Process (BGP) from gOSP:
;   generates new bands using autocorrelation, cross-correlation,
;   and nonlinear transforms.
;
; INPUTS:
;   :raster - ENVI raster object (multispectral image)
;
; KEYWORDS:
;   VERBOSE - Print debug messages
;
; :RETURNS:
;   ENVI raster object containing the original and generated bands
;-

function BGP, raster
  compile_opt idl2

  ; -------------------------------------------------------------------
  ; Load rasters
  ; -------------------------------------------------------------------
  nBands = raster.nBands
  rasters = list()
  for i = 0, nBands - 1 do begin
    rasters.add, raster.subset(bands = i)
  endfor

  ; -------------------------------------------------------------------
  ; Container for all output rasters
  ; -------------------------------------------------------------------
  output_rasters = list()

  ; -------------------------------------------------------------------
  ; (RELATIONSHIPS) Generate cross-product bands: (b=band) b_i * b_j
  ; -------------------------------------------------------------------
  for i = 0, nBands - 1 do begin
    bi = rasters[i]
    for j = i, nBands - 1 do begin
      bj = rasters[j]

      ; Stack bi and bj into a 2-band raster
      layerstack_task = ENVITask('BuildBandStack')
      layerstack_task.input_rasters = [bi, bj]
      layerstack_task.execute
      temp_stack = layerstack_task.output_raster

      ; Now perform band math on stacked raster
      bandmath_task = ENVITask('PixelwiseBandMathRaster')
      bandmath_task.input_raster = temp_stack
      bandmath_task.expression = 'b1 * b2'

      bandmath_task.execute

      ; Add result raster to output list
      output_rasters.add, bandmath_task.output_raster
    endfor
  endfor

  ; -------------------------------------------------------------------
  ; (CONTRAST) Generate all (b=band) b_i ^ 2 combinations
  ; -------------------------------------------------------------------
  ; for i = 0, nBands - 1 do begin
  ; bi = rasters[i]

  ; ; ---------------------------------------------------------------
  ; ; Generate expression and execute expression using Band Math
  ; ; Inputs must be ordered and aliased correctly as b1
  ; ; ---------------------------------------------------------------
  ; bandmath_task = ENVITask('PixelwiseBandMathRaster')
  ; bandmath_task.input_raster = [bi]
  ; bandmath_task.expression = 'b1 * b1'
  ; bandmath_task.output_raster_uri = 'b' + i + '_square'
  ; bandmath_task.execute

  ; output_rasters.add, bandmath_task['OUTPUT_RASTER']
  ; endfor

  ; -------------------------------------------------------------------
  ; (DARK AREAS) Generate all (b=band) log b_i combinations
  ; -------------------------------------------------------------------
  ; for i = 0, nBands - 1 do begin
  ; bi = rasters[i]

  ; ; ---------------------------------------------------------------
  ; ; Generate expression and execute expression using Band Math
  ; ; Inputs must be ordered and aliased correctly as b1
  ; ; ---------------------------------------------------------------
  ; bandmath_task = ENVITask('PixelwiseBandMathRaster')
  ; bandmath_task.input_raster = [bi]
  ; bandmath_task.expression = 'alog10(b1)'
  ; bandmath_task.output_raster_uri = 'b' + i + '_log'
  ; bandmath_task.execute

  ; output_rasters.add, bandmath_task['OUTPUT_RASTER']
  ; endfor

  ; -------------------------------------------------------------------
  ; (DARK AREAS) Generate all (b=band) exponent b_i combinations
  ; -------------------------------------------------------------------
  ; for i = 0, nBands - 1 do begin
  ; bi = rasters[i]

  ; ; ---------------------------------------------------------------
  ; ; Generate expression and execute expression using Band Math
  ; ; Inputs must be ordered and aliased correctly as b1
  ; ; ---------------------------------------------------------------
  ; bandmath_task = ENVITask('PixelwiseBandMathRaster')
  ; bandmath_task.input_raster = [bi]
  ; bandmath_task.expression = 'exp(b1)'
  ; bandmath_task.output_raster_uri = 'b' + i + '_exp'
  ; bandmath_task.execute

  ; output_rasters.add, bandmath_task['OUTPUT_RASTER']
  ; endfor

  ; -------------------------------------------------------------------
  ; Stack all output rasters into a single multi-band raster
  ; -------------------------------------------------------------------
  bandstack_task = ENVITask('BuildBandStack')
  bandstack_task.input_rasters = output_rasters
  bandstack_task.execute

  ; -------------------------------------------------------------------
  ; Return stacked raster
  ; -------------------------------------------------------------------
  RETURN, bandstack_task.output_raster
end

; End PRO
