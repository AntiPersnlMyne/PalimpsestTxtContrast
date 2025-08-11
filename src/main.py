#!/usr/bin/env python3

"""main.py: Main logic file for image processing on manuscript data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger", "Julie Decker"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production". 


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from python_scripts.atdca import ATDCA # Automatic Target Detection Classification Algorithm
import warnings
from time import time

# Suppress GeoTIFF warning - MISHA data isn't georeferenced, warning safely ignored
warnings.filterwarnings("ignore", category=UserWarning, message="Dataset has no geotransform, gcps, or rpcs.*")


# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    # ATDCA Parameters
    INPUT_DIR:str = "data/input/test"           # <--The only two
    OUTPUT_DIR:str = "data/output"              # <--required parameters
    INPUT_IMG_TYPE:str|tuple[str,...] = "tif"
    WINDOW_SHAPE:tuple = (512,512)
    # BGP and TCP parameters
    USE_SQRT:bool = True
    USE_LOG:bool = False    
    MAX_TARGETS:int = 10
    OCPI_THRESHOLD:float = 0.01
    # Parallelism fine-tuning
    USE_PARALLEL:bool = True
    MAX_WORKERS:int|None = None
    CHUNK_SIZE:int = 8
    INFLIGHT:int = 2
    # Progress bar enable/disable
    VERBOSE:bool = True
    
    
    ATDCA(
        input_dir=INPUT_DIR,              # Directory of input images
        output_dir=OUTPUT_DIR,            # Directory for output
        input_image_type=INPUT_IMG_TYPE,  # Image type of source data
        window_shape=WINDOW_SHAPE,        # Breaks image into tiles for memory-safe processing 
        
        use_sqrt=USE_SQRT,                # Generate synthetic bands with sqrt  
        use_log=USE_LOG,                  # Generate synthetic bands with log
        max_targets=MAX_TARGETS,          # Number of targets for algorithm to find
        ocpi_threshold=OCPI_THRESHOLD,    # Target purity threshold
        
        use_parallel=USE_PARALLEL,        # Use parallel processing
        max_workers=MAX_WORKERS,          # Max number of processes during parallel
        chunk_size=CHUNK_SIZE,            # How many windows to process at once
        inflight=INFLIGHT,                # RAM Throughput
        
        verbose=VERBOSE,
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




