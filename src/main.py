#!/usr/bin/env python3

"""main.py: Main file for running manuscript processing routine"""

__author__ = "Gian-Mateo (GM) Tifone and Douglas Tavolette"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger"]
__license__ = "MIT"
__version__ = "1.2"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production". 

# Python implementations
from python_scripts import run_idl, modify_hdr, utils 

import numpy as np  
import cv2 as cv
import time         # Time execution

def main():
    
    src_dir = "data/input/"
    dst_dir = "data/output/"
    
    single_img_path = "data/input/Arch_165r_370nm.tif"
    input_img = cv.imread(single_img_path, cv.IMREAD_UNCHANGED) 
    
    if input_img is None:
        raise FileNotFoundError("Error: Single input image was not read-in")
    else:
        stretch_img = utils.log_stretch(input_img)
        cv.imshow("Stretched Image", stretch_img)
    
    
    
    
    
    
    
    pass
    
    
    
    
    


if __name__ == "__main__":
    print("Hello main!")
    main()
    print("Goodbye main!")
