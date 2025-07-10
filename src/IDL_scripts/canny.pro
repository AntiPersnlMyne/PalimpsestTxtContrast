; ! UNFINISHED CODE !

pro canny
    compile_opt idl3
    ; Read a greyscale image
    nyny = READ_TIFF(FILEPATH('data/input.tif', $
    SUBDIRECTORY=['examples', 'data']))
    ; Resize the image
    nyny = REBIN(nyny, 384, 256)
    ; Perform edge detection using defaults
    filtered = CANNY(nyny)
    ; Create a window
    WINDOW, XSIZE=384, YSIZE=512
    ; Display the original and filtered images
    TVSCL, nyny, 0
    TVSCL, filtered, 1
  

end
