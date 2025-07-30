#!/usr/bin/env python3

"""main.py: Main driver file for image processing on manuscript"""

__author__ = "Gian-Mateo (GM) Tifone and Douglas Tavolette"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger"]
__license__ = "MIT"
__version__ = "1.3"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production". 


# Image processing library and helpers
from python_scripts.utils import *          # Helper functions
from python_scripts.improc import *         # Image processing
from python_scripts.utils.fileio import *   # File I/O

def main():
    # Directories for "process_images"
    src_dir = "data/input/"
    dst_dir = "data/output/"
    
    # Single image analysis
    single_img_path = "data/input/Arch_165r_370nm.tif"
    input_img = imread(single_img_path) 

    if input_img is None: # Imread check
        raise FileNotFoundError("Error: Single input image was not read-in")
    else:
        # Logarithmic stretch
        process_images(src_dir, dst_dir, "", log_stretch)
        
        # Bilateral filter
        process_images(src_dir, dst_dir, "", bilateral_filter, 
                       {"diameter": 3,
                        "sigma_color": 50,
                        "sigma_space": 100})
        

        # Close windows and exit
        cv.destroyAllWindows()



    
    
    
    
    


if __name__ == "__main__":
    print("Hello main!")
    main()
    print("Goodbye main!")
