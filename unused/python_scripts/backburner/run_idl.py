"""run_idl.py: Run IDL scripts through Python. Saves output to data/output as ENVI .dat raster."""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Prototype", "Development", "Production"

import subprocess
from typing import Any

# Magic string
idl_exe = r"C:\Program Files\NV5\ENVI61\IDL91\bin\bin.x86_64\idl.exe"

arg_1 = "TEST123"
pipes = subprocess.Popen('bash -c idl -e \'PCA_bandstack.pro,' \
                         + arg_1 + ', config_file="path/to/config.ini"\'',
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
std_out, std_err = pipes.communicate()
pipes.wait()

def run_idl_script(idl_script: str, src_dir: str, dst_dir: str, args: list[Any] = []):
    pass
    
