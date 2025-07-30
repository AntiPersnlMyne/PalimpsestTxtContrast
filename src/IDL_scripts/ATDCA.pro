;+
; :Author: Gian-Mateo (GM) Tifone
; :Project: MISHA RIT
; :Copyright: MIT
; :Date: July 29 2025
;
; PURPOSE:
;   Main driver for Generalized Orthogonal Subspace Projection (gOSP)
;   using the Automatic Target Detection and Classification Algorithm (ATDCA).
;
; USAGE:
;   .run ATDCA.run
;
; DESCRIPTION:
;   This script orchestrates the gOSP process using:
;     1.   BGP (Band Generation Process)
;     2.1. TGP (Target Generation Process)
;     2.2. TCP (Target Classification Process)
;
; REQUIRED FILES:
;   - BGP.pro
;   - TGP.pro
;   - TCP.pro
;-

pro ATDCA
  compile_opt idl2

  ; -------------------------------------------------------------------
  ; Initialize ENVI session
  ; -------------------------------------------------------------------
  e = envi(/headless)
  if ~isa(e, 'ENVI') then begin
    print, 'ENVI must be initialized.'
    RETURN
  endif

  ; -------------------------------------------------------------------
  ; Prompt user to select multiple raster files (one-band each)
  ; -------------------------------------------------------------------
  start_dir = 'C:\Users\General Motors\Desktop\Projects\Palimpsest_OSP_Added\PalimpsestTxtContrast\data\input' ; GETENV('USERPROFILE') + '/Desktop' ; Suggested starting point
  filepaths = dialog_pickfile(/multiple_files, $
    filter = '*.dat;*.tif;*.tiff', $
    path = start_dir, $
    title = 'Select single-band raster files')
  if filepaths[0] eq '' then RETURN

  ; Check if 2+ rasters to create band combinations
  nBands = n_elements(filepaths)
  if nBands lt 2 then begin
    print, 'At least two rasters are required.'
    RETURN
  endif

  ; -------------------------------------------------------------------
  ; Open and stack input rasters
  ; -------------------------------------------------------------------
  rasters = list()
  for i = 0, nBands - 1 do begin
    raster = e.openRaster(filepaths[i])
    if raster.nBands ne 1 then begin
      print, 'All inputs must be single-band rasters.'
      RETURN
    endif
    rasters.add, raster
  endfor

  ; Matrix operations more efficient than raster-by-raster
  ; Build bandstack
  buildstack_task = ENVITask('BuildBandStack')
  buildstack_task.input_rasters = rasters
  buildstack_task.execute

  ; -------------------------------------------------------------------
  ; Automatic Target Detection Classification Algorithm (ATDCA)
  ; -------------------------------------------------------------------
  ; Band Generation Process (BGP)
  generated_raster = BGP(buildstack_task.output_raster)

  ; Target Generation Process (TGP)
  target_matrix = TGP(generated_raster, opci_threshold = 0.01, max_targets = 15)

  ; Target Classification Process (TCP)
  TCP, generated_raster, target_matrix, class_images = class_outputs

  ; -------------------------------------------------------------------
  ; Save results as multi-band raster (.dat)
  ; -------------------------------------------------------------------
  output_file = dialog_pickfile(/write, filter = '*.dat', $
    title = 'Save ATDCA classification results')

  if output_file ne '' then begin
    ; Determine number of bands
    if size(class_outputs, /n_dimensions) eq 3 then begin
      dims = size(class_outputs, /dimensions)
      nBands = dims[2]
      nRows = dims[1]
      nColumns = dims[0]
    endif else begin
      dims = size(class_outputs, /dimensions)
      nBands = 1
      nRows = dims[1]
      nColumns = dims[0]
    endelse

    ; Convert classification array into an ENVI raster using ENVI::CreateRaster
    result_raster = e.createRaster(output_file, class_outputs, $
      spatialref = generated_raster.spatialRef, $
      data_type = 4, $ ; 4 = float
      interleave = 'bsq', $
      nrows = nRows, $
      ncolumns = nColumns, $
      nbands = nBands)

    ; Save result
    result_raster.save

    ; Print exit success or exit failure
    print, 'Classification results saved to: ', output_file
  endif else begin
    print, 'No output file selected. Classification results not saved.'
  endelse
end

; End PRO
