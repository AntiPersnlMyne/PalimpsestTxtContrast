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
OMP_NUM_THREADS=8 python main.py

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
import pstats, cProfile


# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    start = time()
    gosp(
        # Input information
        input_dir="data/input/arch177_rgb_365cor",   
        output_dir="data/output",         
        input_image_types="tif",       
        # BGP and TCP parameters    
        full_synthetic=True,                   
        skip_bgp=False,                 
        max_targets=40,                     
        opci_threshold=0.01,              
        # Throughput
        window_shape=(1024,1024),                 
        # Debug
        verbose=True,                      
    )
    print(f"\n[main/arch165_] - Execution finished -\nRuntime = {(time() - start):.2f}")
    


# =========
# Executing
# =========
if __name__ == "__main__":
    # # Multiprocessing logic
    # # fork is faster  - Linux exclusive
    # # spawn is slower - Windows & Linux
    # if platform == "win32":
    #     mp.set_start_method("spawn", force=True)
    # else:
    #     try: mp.set_start_method("fork", force=True)
    #     except: mp.set_start_method("spawn", force=True)
    
    
    # Memory/Performance profiler
    cProfile.runctx("main()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()





