#!/usr/bin/env python3

"""main.py: Main file for running manuscript processing routine"""

__author__ = "Gian-Mateo (GM) Tifone and Douglas Tavolette"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger"]
__license__ = "MIT"
__version__ = "1.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production". 

# IDL subprocesses
import subprocess
import os
import argparse

# Python subprocesses
from python_scripts import utils, run_idl, modify_hdr


def main():
    
    
    run_idl.run_idl_script(
        idl_script="build_band_stack",
        src_dir="data/input",
        dst_dir="data/output/test_cube",
        args=["_mnf", "5"]
    )
        




if __name__ == "__main__":
    print("Hello main!")
    main()
    print("Goodbye, main!")