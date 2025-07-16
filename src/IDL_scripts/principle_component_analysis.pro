; Runs Principle Component Analysis (PCA) Task on raster in ENVI
;
; Author: Gian-Mateo Tifone
; Copyright: RIT MISHA
; License: MIT
; Version: 1.0

pro principle_component_analysis
  compile_opt idl2
  e = envi(/headless)

  ; src and dst file directories
  cmd_args = command_line_args()

  src_dir = cmd_args[0] ; Source file directory
  dst_dir = cmd_args[1] ; Destination file directory

  file = filepath('qb_boulder_msi', root_dir = e.root_dir, subdirectory = ['data'])
  print, 'Found file at: ', file
  message, '', /informational

  src = read_tiff(filepath('data/input.tif', $
    subdirectory = ['examples', 'data']))
end
