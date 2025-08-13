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
    # start = time()
    # gosp(
    #     # Input information
    #     input_dir="data/input/arch165_rgbcor",   
    #     output_dir="data/output/rgb_cor",         
    #     input_image_types="tif",       
    #     # BGP and TCP parameters    
    #     use_sqrt=False,                   
    #     use_log=False,                     
    #     max_targets=10,                     
    #     opci_threshold=0.01,              
    #     # Parallelism fine-tuning
    #     window_shape=(512,512),          
    #     max_workers=None,                   
    #     chunk_size=8,                      
    #     inflight=2,                         
    #     # Debug
    #     verbose=True,                      
    # )
    # print(f"\n[main/rgb-cor] - Execution finished -\nRuntime = {(time() - start):.2f}")
    
    # Create (2)
    improc.process_images(
        src_dir="data/input/arch177_sb_365cor",
        dst_dir="data/output/arch177_sb_365cor_lum",
        file_prefix="lum_",
        transform_fn=improc.extract_luminous,
    )
    


# --------------------------------------------------------------------------------------------
# Executing Main
# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






