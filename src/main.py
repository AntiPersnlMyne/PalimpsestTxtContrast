#!/usr/bin/env python3

"""main.py: Main logic file for image processing on manuscript data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger", "Julie Decker"]
__license__ = "MIT"
__version__ = "3.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Production" # "Development", or "Production". 


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from python_scripts.gosp import gosp 
from python_scripts import improc 
from python_scripts.utils.fileio import discover_image_files
from time import time


# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    # CLAHE
    start = time()
    gosp(
        # Input information
        input_dir="data/input/arch177_rgb_365cor_lum",   
        output_dir="data/output",         
        input_image_types="tif",       
        # BGP and TCP parameters    
        full_synthetic=False,                   
        skip_bgp=True,                 
        max_targets=10,                     
        opci_threshold=0.01,              
        # Parallelism fine-tuning
        window_shape=(512,512),          
        max_workers=None,                   
        chunk_size=16,                      
        inflight=3,                         
        # Debug
        verbose=True,                      
    )
    print(f"\n[main/rgb_365cor_lum] - Execution finished -\nRuntime = {(time() - start):.2f}")
    


# --------------------------------------------------------------------------------------------
# Executing Main
# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






