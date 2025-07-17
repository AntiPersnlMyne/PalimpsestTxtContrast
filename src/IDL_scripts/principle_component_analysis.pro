; Runs Principle Component Analysis (PCA) Task on raster in ENVI
;
; Author: Gian-Mateo Tifone
; Copyright: RIT MISHA
; License: MIT
; Version: 1.0

pro principle_component_analysis
  compile_opt idl3

  print, 'Found file at: '
  message, '', /informational

  ; Initialize ENVI in headless mode
  e = envi(/headless)

  ; Get command line arguments
  cmd_args = command_line_args()

  ; Check if minimum required arguments are provided
  if n_elements(cmd_args) lt 2 then begin
    print, 'Usage: principle_component_analysis <src_directory> <dst_directory> [suffix]'
    return
  endif

  ; Assign arguments
  src_dir = cmd_args[0] ; Source directory
  dst_dir = cmd_args[1] ; Destination directory
  suffix = n_elements(cmd_args) gt 2 ? cmd_args[2] : '_pca' ; Optional suffix. Default is '_pca'

  ; Ensure source directory exists
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

    ; Create PCA task
    task = ENVITask('ForwardPCATransform')
    task.input_raster = input_raster

    ; Execute PCA
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

  print, 'PCA processing complete. Processed ' + strtrim(n_tif + n_tiff, 2) + ' files.'
end
