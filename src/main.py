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
from python_scripts import utils, run_idl, modify_hdr #type:ignore -> only bcs of venv

# Function time comparison
import time

def main():
    
    # ALPR Method
    
    src_dir = "data/input"
    dst_dir = "data/output/otsu"
    utils.bilateral_filter(src_dir, dst_dir, 9, 200, 200)


    # Connected Component Analysis Method
    
    
    
    # 
    
    
    
    
    


if __name__ == "__main__":
    print("Hello main!")
    main()
    print("Goodbye main!")
