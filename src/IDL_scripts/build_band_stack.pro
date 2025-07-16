; Runs Principal Component Analysis (PCA) Task on raster in ENVI
;
; Author: Gian-Mateo
; Description: Processes TIFFs with MNF and outputs to ENVI format
; Version: 1.1

pro build_band_stack
  compile_opt idl2

  ; Initialize ENVI in headless mode
  e = envi(/headless)

  ; Get command line arguments
  cmd_args = command_line_args()

  ; Check if minimum required arguments are provided
  if n_elements(cmd_args) lt 2 then begin
    print, 'Usage: build_band_stack <src_directory> <dst_directory> [suffix] [num_components]'
    return
  endif

  ; Assign arguments
  src_dir = cmd_args[0]
  dst_dir = cmd_args[1]
  suffix = n_elements(cmd_args) gt 2 ? cmd_args[2] : ''
  num_components = n_elements(cmd_args) gt 3 ? fix(cmd_args[3]) : 3

  ; Ensure source directory exists
  if ~file_test(src_dir, /directory) then begin
    print, 'Source directory does not exist: ' + src_dir
    return
  endif

  ; Create destination directory if it doesn't exist
  if ~file_test(dst_dir, /directory) then file_mkdir, dst_dir

  ; Find TIFF files
  tif_files = file_search(src_dir, '*.tif')
  tiff_files = file_search(src_dir, '*.tiff')
  all_files = append(tif_files, tiff_files)

  ; Loop over input files
  for i = 0, n_elements(all_files) - 1 do begin
    ; Open input raster
    input_raster = e.openRaster(all_files[i])

    ; Create and configure MNF task
    task = ENVITask('ForwardMNFTransform')
    task.input_raster = input_raster
    if num_components gt 3 then task.output_nbands = num_components

    ; Execute task
    task.execute

    ; Get MNF output raster
    output_raster = task.output_raster

    ; Construct ENVI output filename
    input_basename = file_basename(all_files[i], '.tif')
    input_basename = strmid(input_basename, 0, strpos(input_basename, '.tiff') eq -1 ? strlen(input_basename) : strpos(input_basename, '.tiff')) ; Normalize name

    envi_output_filename = input_basename + suffix + '.dat'
    envi_output_path = filepath(envi_output_filename, root = dst_dir)

    ; Save as ENVI raster (default format is .dat with .hdr)
    output_raster.save, envi_output_path

    ; Close rasters
    input_raster.close
    output_raster.close
  endfor

  ; Close ENVI
  e.close

  print, 'Band stack complete. Processed ' + strtrim(n_elements(all_files), 2) + ' files.'
end
