#!/usr/bin/env python3

"""main.py: Main driver file for image processing on manuscript"""

__author__ = "Gian-Mateo (GM) Tifone and Douglas Tavolette"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger"]
__license__ = "MIT"
__version__ = "1.2"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production". 


from python_scripts.improc import utils                # Custom scripts
from python_scripts.improc.utils import process_images # Too lazy to say utils.
import cv2 as cv                                       # Image import 
import time                                            # Execution timing


def main():
    # Directories for "process_images"
    src_dir = "data/input/"
    dst_dir = "data/output/"
    
    # Single image analysis
    single_img_path = "data/input/Arch_165r_370nm.tif"
    input_img = cv.imread(single_img_path, cv.IMREAD_UNCHANGED) 

    if input_img is None: # Imread check
        raise FileNotFoundError("Error: Single input image was not read-in")
    else:
        # Logarithmic stretch
        process_images(src_dir, dst_dir, "", utils.log_stretch)
        
        # Bilateral filter
        process_images(src_dir, dst_dir, "", utils.bilateral_filter, 
                       {"diameter": 3,
                        "sigma_color": 50,
                        "sigma_space": 100})
        

        # Close windows and exit
        cv.destroyAllWindows()



    
    
    
    
    


if __name__ == "__main__":
    print("Hello main!")
    main()
    print("Goodbye main!")
