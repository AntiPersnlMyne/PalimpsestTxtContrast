#!/usr/bin/env python3

"""main.py: Main file for running manuscript processing routine"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production". 

# IDL subprocesses
import subprocess
import os

# Python subprocesses
from python_scripts import utils


def main():
    print("Hello, world!")
    
    # Set dir for source files to be called
    working_directory = "./src"
    if not os.path.isdir(working_directory):
        print("Invalid working directory:", working_directory)
    else:
        print("Working directory exists")
        
    try:
        """CHANGE OSP.pro TO YOUR FILE'S NAME"""
        idl_script = os.path.join(working_directory, "OSP.pro")
        """THIS MIGHT ALSO NEED TO BE CHANGED"""
        idl_exe = r"C:\Program Files\NV5\ENVI61\IDL91\bin\bin.x86_64\idl.exe"
        
        command = f'"{idl_exe}" -e ".run \'{idl_script}\' ; open_file"'

        result = subprocess.run(command, cwd=working_directory, shell=True, capture_output=True, text=True)
        # subprocess.run[]
        # subprocess.run(["javac", "makeAFile.java"], check=True)
        # subprocess.run(["java", "makeAFile"], check=True)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    except subprocess.CalledProcessError as e:
        print("-- Error in main.py: --\n", e)
        




if __name__ == "__main__":
    main()