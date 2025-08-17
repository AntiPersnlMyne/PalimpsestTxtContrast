#!/usr/bin/env python3

"""main.py: Main logic file for image processing on manuscript data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger", "Julie Decker"]
__license__ = "MIT"
__version__ = "3.0.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Production" # "Development", or "Production". 


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from python_scripts.gosp import gosp 
from time import time

# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    # CLAHE
    start = time()
    gosp(
        # Input information
        input_dir="data/input/test",   
        output_dir="/media/g-m/FixedDisk/",         
        input_image_types="tif",       
        # BGP and TCP parameters    
        full_synthetic=True,                   
        skip_bgp=False,                 
        max_targets=30,                     
        opci_threshold=0.01,              
        # Parallelism fine-tuning
        window_shape=(128,128),          
        max_workers=None,                   
        chunk_size=8,                      
        inflight=2,                         
        # Debug
        verbose=True,                      
    )
    print(f"\n[main/rgb_365cor_lum] - Execution finished -\nRuntime = {(time() - start):.2f}")
    


# --------------------------------------------------------------------------------------------
# Executing Main
# --------------------------------------------------------------------------------------------
if __name__ == "__main__":    
    # Option 1: In terminal, suppress warnings (everything?)
    # python src/main.py 2>/dev/null
        
    # Option 2: Just live with the warnings
    main()






