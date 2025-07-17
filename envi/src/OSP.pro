pro perform_OSP
  compile_opt idl3
  e = envi()

  print, 'DOES THIS WORK?'

  file = 'C:/palimsest cube/cube.dat'
  raster = e.openRaster(file)
  ; WORKS
  if ~obj_valid(raster) then begin
    print, 'Failed to open raster.'
    RETURN
  endif

  dims = raster.dimensions
  print, 'Raster dimensions: ', dims

  view = e.getView()
  if ~obj_valid(view) then begin
    print, 'Failed to get ENVI view.'
    RETURN
  endif

  print, 'DOES THIS WORK?'
  message, 'This shows even if PRINT does not'
end
