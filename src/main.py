#!/usr/bin/env python3

"""main.py: Main logic file for image processing on manuscript data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger", "Julie Decker"]
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production". 


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from python_scripts.gosp import gosp 
from python_scripts import improc 
from time import time

import cv2 as cv
import os



# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    # CLAHE -> GOSP
    start = time()    
    improc.process_images( # CLAHE
        src_dir="data/input/arch177_lum",
        dst_dir="data/output/arch177_clahe",
        file_prefix="clahe_",
        transform_fn=improc.clahe,
        transform_kwargs={"tile_grid_size": (1024,1024)}
    )
    gosp(
        # Input information
        input_dir="data/output/arch177_clahe",   
        output_dir="data/output",         
        input_image_types="tif",       
        # BGP and TCP parameters    
        use_sqrt=False,                   
        use_log=False,                     
        max_targets=20,                     
        opci_threshold=0.01,              
        # Parallelism fine-tuning
        window_shape=(1024,1024),          
        max_workers=None,                   
        chunk_size=10,                      
        inflight=2,                         
        # Debug
        verbose=True,                      
    )
    print(f"\n[main/365cor-clahe] - Execution finished -\nRuntime = {(time() - start):.2f}")
    


# --------------------------------------------------------------------------------------------
# Executing Main
# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






