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
import time

def main():

    # Array size
    SIZE = 10_000_000

    # Test float32
    start_32 = time.time()
    arr32 = np.random.rand(SIZE).astype(np.float32)
    for _ in range(100):
        result32 = arr32 * 2 + 5
    end_32 = time.time()

    # Test float64
    start_64 = time.time()
    arr64 = np.random.rand(SIZE).astype(np.float64)
    for _ in range(100):
        result64 = arr64 * 2 + 5
    end_64 = time.time()

    # Print results
    print(f"float32 time: {end_32 - start_32:.4f} seconds")
    print(f"float64 time: {end_64 - start_64:.4f} seconds")
    print(f"Memory usage float32: {arr32.nbytes / 1024 / 1024:.2f} MB")
    print(f"Memory usage float64: {arr64.nbytes / 1024 / 1024:.2f} MB")

    
    
    
    
    


if __name__ == "__main__":
    print("Hello main!")
    main()
    print("Goodbye main!")
