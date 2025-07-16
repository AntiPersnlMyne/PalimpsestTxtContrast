; Runs Principle Component Analysis (PCA) Task on raster in ENVI
;
; Author: Gian-Mateo Tifone
; Copyright: RIT MISHA
; License: MIT
; Version: 1.0

pro min_noise_fraction
  compile_opt idl2

  ; Initialize ENVI in headless mode
  e = envi(/headless)

  ; Get command line arguments
  cmd_args = command_line_args()

  ; Check if minimum required arguments are provided
  if n_elements(cmd_args) lt 2 then begin
    print, 'Usage: minimum_noise_fraction <src_directory> <dst_directory> [suffix] [num_components] '
    return
  endif

  ; Assign arguments
  src_dir = cmd_args[0] ; Source directory
  dst_dir = cmd_args[1] ; Destination directory
  suffix = n_elements(cmd_args) gt 2 ? cmd_args[2] : '_mnf' ; Optional suffix. Default '_mnf'
  num_components = n_elements(cmd_args) gt 3 ? fix(cmd_args[3]) : 3 ; Optional MNF param. Default 3

  ; Ensure directories exist
  if ~file_test(src_dir, /directory) then begin
    print, 'Source directory does not exist: ' + src_dir
    return
  endif

  ; Create destination directory if it doesn't exist
  if ~file_test(dst_dir, /directory) then file_mkdir, dst_dir

  ; Get list of .tif files recursively
  tif_files = file_search(src_dir, '*.tif', count = n_tif)
  tiff_files = file_search(src_dir, '*.tiff', count = n_tiff)
  tif_files = append(tif_files, tiff_files) ; Capture both file extension syntax

  ; Process each .tif file
  for i = 0, n_tif + n_tiff - 1 do begin
    ; Open the input file
    input_raster = e.openRaster(tif_files[i])

    ; Create MNF task
    task = ENVITask('ForwardMNFTransform')
    task.input_raster = input_raster
    ; Set num_components if user provided. Otherwise, use ENVI default.
    if num_components gt 3 then begin
      task.output_nbands = num_components
    endif

    ; Execute MNF
    task.execute

    ; Get the output raster
    output_raster = task.output_raster

    ; Generate output filename
    input_filename = file_basename(tif_files[i], '.tif')
    output_filename = input_filename + suffix + '.tif'
    output_filepath = filepath(output_filename, root = dst_dir)

    ; Export processed raster to TIFF
    output_raster.export, output_filepath, 'TIFF'

    ; Close rasters to free memory
    input_raster.close
    output_raster.close
  endfor

  ; Close ENVI
  e.close

  print, 'MNF processing complete. Processed ' + strtrim(n_tif + n_tiff, 2) + ' files.'
end
