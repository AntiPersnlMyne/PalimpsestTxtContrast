PRO pc
  COMPILE_OPT IDL3, HIDDEN ; Using IDL3 
                           ; Not intended to be called by user, only by program

  ENVI,/RESTORE_BASE_SAVE_FILES ; Load base ENVI routines
  ENVI_BATCH_INIT               ; Headless mode
  e = ENVI()                    ; Engine variable

  ; Open file to process
  file = 'envi/data/output/processed.tif'
  raster = e.OpenRaster(file)
  
  ; Get the task from the catalog of ENVITasks
  task = ENVITask('ForwardPCATransform')
  
  ; Define inputs
  task.INPUT_RASTER = raster
  
  ; Run the task
  task.Execute
  
  e.SaveRaster, task.output_raster, 'pc_output.dat'
END

