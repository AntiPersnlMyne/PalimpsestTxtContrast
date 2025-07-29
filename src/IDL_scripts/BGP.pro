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
  ; Initialize ENVI session
  ; -------------------------------------------------------------------
  e = envi()
  if ~isa(e, 'ENVI') then begin
    print, 'ENVI must be initialized to run this program.'
    RETURN, ''
  endif

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

      ; ---------------------------------------------------------------
      ; Generate expression and execute expression using Band Math
      ; Inputs must be ordered and aliased correctly as b1, b2
      ; ---------------------------------------------------------------
      task_result = e.doTask('Band Math', $
        input_raster = [bi, bj], $
        expression = 'b1 * b2', $
        output_name = 'b' + strtrim(i + 1, 2) + '_mul_b' + strtrim(j + 1, 2))

      output_rasters.add, task_result['OUTPUT_RASTER']
    endfor
  endfor

  ; -------------------------------------------------------------------
  ; (CONTRAST) Generate all (b=band) b_i ^ 2 combinations
  ; -------------------------------------------------------------------
  ; ; Loop over all bands and create bi^2 (b1 squared) combinations
  ; FOR i = 0, nBands - 1 DO BEGIN
  ; bi = rasters[i]
  ; ; Apply pixel-wise multiplication
  ; mult_result = e.doTask('Band Math', $
  ; INPUT_RASTER=[bi], $
  ; EXPRESSION='b1 * b1', $
  ; OUTPUT_NAME='b' + i + '_square', $
  ; OUTPUT_RASTER=raster_out)
  ;
  ; output_rasters.Add, mult_result['OUTPUT_RASTER']
  ; ENDFOR

  ; -------------------------------------------------------------------
  ; (DARK AREAS) Generate all (b=band) log b_i combinations
  ; -------------------------------------------------------------------
  ; ; Loop over all bands and create alog10(bi) (log base 10 on bi) combinations
  ; FOR i = 0, nBands - 1 DO BEGIN
  ; bi = rasters[i]
  ; ; Apply pixel-wise multiplication
  ; mult_result = e.doTask('Band Math', $
  ; INPUT_RASTER=[bi], $
  ; EXPRESSION='alog10(b1)', $
  ; OUTPUT_NAME='b' + i + '_log', $
  ; OUTPUT_RASTER=raster_out)
  ;
  ; output_rasters.Add, mult_result['OUTPUT_RASTER']
  ; ENDFOR

  ; -------------------------------------------------------------------
  ; (DARK AREAS) Generate all (b=band) exponent b_i combinations
  ; -------------------------------------------------------------------
  ; ; Loop over all bands and create exp(bi) (e to the power of bi) combinations
  ; FOR i = 0, nBands - 1 DO BEGIN
  ; bi = rasters[i]
  ; ; Apply pixel-wise multiplication
  ; mult_result = e.doTask('Band Math', $
  ; INPUT_RASTER=[bi], $
  ; EXPRESSION='exp(b1)', $
  ; OUTPUT_NAME='b' + i + '_exp', $
  ; OUTPUT_RASTER=raster_out)
  ;
  ; output_rasters.Add, mult_result['OUTPUT_RASTER']
  ; ENDFOR

  ; -------------------------------------------------------------------
  ; Stack all output rasters into a single multi-band raster
  ; -------------------------------------------------------------------
  stacked_result = e.doTask('Layer Stack', $
    input_rasters = output_rasters.toArray(), $
    output_name = 'nonlinear_generated_bands')

  ; -------------------------------------------------------------------
  ; Return stacked raster
  ; -------------------------------------------------------------------
  RETURN, stacked_result['OUTPUT_RASTER']
end

; End PRO
