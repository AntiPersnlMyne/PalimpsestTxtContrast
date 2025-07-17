#!/usr/bin/env python3

"""main.py: Main file for running manuscript processing routine"""

__author__ = "Gian-Mateo (GM) Tifone and Douglas Tavolette"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger"]
__license__ = "MIT"
__version__ = "2.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production". 

# Python implementations
from python_scripts import utils, run_idl, modify_hdr

# Run IDL scripts
import sys
sys.path.append(r'C:/Program Files/NV5/ENVI61/IDL91/lib/bridges')
from idlpy import *

def main():
    
    src_dir = "/src/IDL_scripts/"
    dst_dir = "/data/output/"
    idl_exe = r"C:\Program Files\NV5\ENVI61\IDL91\bin\bin.x86_64\idl.exe"
    
    # Build ENVI cube
    run_idl.run_idl_script(
        idl_script="test",
        src_dir=src_dir,
        dst_dir=dst_dir,
        args=[""]
    )





if __name__ == "__main__":
    print("Hello, main!")
    main()
    print("Goodbye, main!")