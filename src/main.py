#!/usr/bin/env python3

"""main.py: Main file for running manuscript processing routine"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production". 

# IDL subprocesses
import subprocess
import os
import argparse

# Python subprocesses
from python_scripts import utils


def main():
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process IDL scripts')
    
    parser.add_argument('src_dir', help='Source directory')
    parser.add_argument('dst_dir', help='Destination directory')
    
    # Optional arguments
    parser.add_argument('-s', '--suffix', 
                        help='Optional output file suffix')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Prepare arguments for IDL script
    idl_args = [args.src_dir, args.dst_dir]
    
    
    if args.suffix:
        idl_args.append(args.suffix)
        




if __name__ == "__main__":
    print("Hello main!")
    main()
    print("Goodbye, main!")