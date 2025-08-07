#!/usr/bin/env python3

"""main.py: Main logic file for image processing on manuscript data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger", "Julie Decker"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production". 


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from python_scripts.atdca import ATDCA # Automatic Target Detection Classification Algorithm
# import python_scripts.improc as improc
import warnings
from time import time

# Suppress GeoTIFF warning - MISHA data isn't georeferenced, warning safely ignored
warnings.filterwarnings("ignore", category=UserWarning, message="Dataset has no geotransform, gcps, or rpcs.*")


# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    # Automatic Target Detection w/ OSP input parameters
    INPUT_DIR:str = r"data/input/test"
    OUTPUT_DIR:str = r"data/output"
    ONE_FILE:bool = False
    BLOCK_SHAPE:tuple = (512,512)
    MAX_TARGETS:int = 10
    USE_SQRT:bool = True
    USE_LOG:bool = False    
    OCPI_THRESHOLD:float = 0.01
    INPUT_IMG_TYPE:str|tuple[str,...] = "tif"
    
    ATDCA(input_dir=INPUT_DIR,              # Directory of input images
          output_dir=OUTPUT_DIR,            # Directory for output
          one_file=ONE_FILE,                # Output as one file or individual bands
          window_shape=BLOCK_SHAPE,         # Breaks image into tiles for memory-safe processing
          max_targets=MAX_TARGETS,          # Number of targets for algorithm to find
          use_sqrt=USE_SQRT,                # Generate synthetic bands with sqrt  
          use_log=USE_LOG,                  # Generate synthetic bands with log
          ocpi_threshold=OCPI_THRESHOLD,    # Target purity threshold
          input_image_type=INPUT_IMG_TYPE   # Image type of source data
    )




# --------------------------------------------------------------------------------------------
# Executing Main
# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    start = time()
    main()
    print(f"\n-- Execution finished --\nNJIT, 11 bands. Runtime = {(time() - start):.2f}")

# TEST 3-band Timing results
# BGP, 512: 215
#




