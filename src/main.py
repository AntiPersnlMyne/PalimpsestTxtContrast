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
from python_scripts.gosp import gosp # Automatic Target Detection Classification Algorithm
from time import time



# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    
    gosp(
        # Input information
        input_dir="data/input/test",  # Directory of input images                              <--The only two
        output_dir="data/output",           # Directory for output                                   <--required parameters
        input_image_types="tif",            # Image type of source data
        # BGP and TCP parameters    
        use_sqrt=False,                     # Generate synthetic bands with sqrt  
        use_log=False,                      # Generate synthetic bands with log
        max_targets=10,                     # Number of targets for algorithm to find
        opci_threshold=0.001,               # Target purity threshold
        # Parallelism fine-tuning
        window_shape=(512,512),             # Breaks image into tiles for memory-safe processing 
        max_workers=None,                   # Max number of processes during parallel
        chunk_size=8,                       # How many windows to process at once
        inflight=2,                         # RAM Throughput
        # Debug
        verbose=True,                       # Enable/Disable progress bar
    )


# --------------------------------------------------------------------------------------------
# Executing Main
# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    start = time()
    main()
    print(f"\n[main] - Execution finished -\nRuntime = {(time() - start):.2f}")

# Timing results
# -- BGP --
# block=512, band=3: 215
# block=512, band=3, @njit: 221
# block=512, band=3, @njit, par-pass1, par-pass2: # 17 (12 on laptop)

# BGP, block=256, band=11: 1629
# BGP, block=256, band=11, @njit, par-pass1, par-pass2: 144

# -- BGP->TGP -- 
# block=512, band=3, serial, verbose: 55
# block=512, band=3 parallel, verbose: 20
# 




