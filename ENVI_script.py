import numpy as np        # Number processing
import os                 # File i/o
from pprint import pprint # Pretty print, debug messages
import datetime           # File names

import envipyengine             # Interface with ENVI
import envipyengine.config      # Set envipyengine engine path
from envipyengine import Engine # envipyengine class

# ---------------------------------------------------------------------------------------- 
# Section 0: Configure Engine
# ----------------------------------------------------------------------------------------
# Hardcoded filepath to taskengine in 
task_engine_path = "c:\\Program Files\\NV5\\ENVI61\\IDL91\\bin\\bin.x86_64\\taskengine.exe"
envipyengine.config.set('engine', task_engine_path)

# Returns an ENVI Py Engine object, allowing interface with ENVI
envi_engine = Engine("ENVI") 

# Print all available tasks
# pprint(envi_engine.tasks())


# ---------------------------------------------------------------------------------------- 
# Section 1: Perform a task (toolbox function)
# ----------------------------------------------------------------------------------------
task = envi_engine.task("ForwardPCATransform")
# print(task.description, type(task.description)) # Debug: Task information

# Set input and output paths 
task.input_raster = "data/input/120r-121v_Alex02r_Sinar_LED365_01_corr.tif"
current_time = datetime.datetime.now()
task.output_raster = str(current_time.day) + '_' + str(current_time.hour) + ':' + str(current_time.minute) + '_' + "processed_img.tif"
task.num_components = 3  # Optional: only if the task supports this

# Print task parameters
print(task.parameters)

# Execute task
# envi_engine.run_task(task)


