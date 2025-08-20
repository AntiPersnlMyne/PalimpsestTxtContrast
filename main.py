#!/usr/bin/env python3

"""main.py: Main logic file for image processing on manuscript data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger", "Julie Decker"]
__license__ = "MIT"
__version__ = "3.1.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production". 

# ---------------
# Useful commands
# ---------------
"""
// (Run) Sets number of avaliable threads from 1 to 8 for multithreading
OMP_NUM_THREADS=8 python src/main.py

// (Development) Compile Cython files 
python setup.py build_ext --inplace

// (Install) Build & Compile Cython files
pip install -e . && rm -r build || del build && rm -r gosp.egg-info || del gosp.egg-info
"""


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from gosp import gosp
# from python_scripts import improc 

from time import time
import multiprocessing as mp
from sys import platform


# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    start = time()

    gosp(
        input_dir="data/input/test",
        output_dir="data/output",
        verbose=True
    )
    
    # gosp(
    #     # Input information
    #     input_dir="data/input/arch177_rgb_365cor_lum",   
    #     output_dir="/media/g-m/FixedDisk/",         
    #     input_image_types="tif",       
    #     # BGP and TCP parameters    
    #     full_synthetic=True,                   
    #     skip_bgp=False,                 
    #     max_targets=40,                     
    #     opci_threshold=0.01,              
    #     # Parallelism fine-tuning
    #     window_shape=(128,128),          
    #     max_workers=None,                   
    #     chunk_size=4,                      
    #     inflight=2,                         
    #     # Debug
    #     verbose=True,                      
    # )
    print(f"\n[main/test] - Execution finished -\nRuntime = {(time() - start):.2f}")
    


# ==============
# Executing Main
# ==============
if __name__ == "__main__":
    # Multiprocessing logic
    # fork is faster  - Linux exclusive
    # spawn is slower - Windows & Linux
    if platform == "win32":
        mp.set_start_method("spawn", force=True)
    else:
        try: mp.set_start_method("fork", force=True)
        except: mp.set_start_method("spawn", force=True)
    
    main()






