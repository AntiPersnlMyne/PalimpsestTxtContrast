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



# --------------------------------------------------------------------------------------------
# Driver Code
# --------------------------------------------------------------------------------------------
def main():
    
    start = time()
    gosp(
        # Input information
        input_dir="data/input/arch177_rgb_365cor_lum",   
        output_dir="data/output",         
        input_image_types="tif",       
        # BGP and TCP parameters    
        use_sqrt=True,                   
        use_log=True,                     
        max_targets=20,                     
        opci_threshold=0.01,              
        # Parallelism fine-tuning
        window_shape=(512,512),          
        max_workers=None,                   
        chunk_size=8,                      
        inflight=2,                         
        # Debug
        verbose=True,                      
    )
    print(f"\n[main] - Execution finished -\nRuntime = {(time() - start):.2f}")
    
    # Grayscale -> CLAHE -> GOSP
    improc.process_images( # Grayscale
        src_dir="data/input/arch177_rgb",
        dst_dir="data/output/arch177_gray",
        file_prefix="gray_",
        transform_fn=improc.bgr2gray,
    )
    improc.process_images( # CLAHE
        src_dir="data/output/arch177_gray",
        dst_dir="data/output/arch177_clahe",
        file_prefix="clahe_",
        transform_fn=improc.clahe,
        transform_kwargs={"tile_grid_size": (512,512)}
    )
    gosp(
        # Input information
        input_dir="data/output/arch177_clahe",   
        output_dir="data/output",         
        input_image_types="tif",       
        # BGP and TCP parameters    
        use_sqrt=True,                   
        use_log=True,                     
        max_targets=20,                     
        opci_threshold=0.01,              
        # Parallelism fine-tuning
        window_shape=(512,512),          
        max_workers=None,                   
        chunk_size=8,                      
        inflight=2,                         
        # Debug
        verbose=True,                      
    )
    
    


# --------------------------------------------------------------------------------------------
# Executing Main
# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






