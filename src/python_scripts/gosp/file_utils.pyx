#!/usr/bin/env python3
# distutils: language=c

"""fileio.py: File discovery and deletion"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from typing import List, Tuple
from glob import glob
import os
from warnings import warn


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.0.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Production" # "Prototype", "Development", "Production"



# ===============================
# Delete Temporary File from BGP
# ===============================
def rm(filepath:str) -> None:
    """Deletes file from directory"""
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        warn("[fileio] Filepath does not exist")

# =========================================
# Return Path to Input Bands from Directory
# =========================================
def discover_image_files(
    input_dir: str,
    input_image_type: str|Tuple[str, ...] = "tif"
    ) -> List[str]:
    """
    Discovers and returns a list of image files in a directory matching the given type(s).

    Args:
        input_dir (str): Directory to search for input images.
        input_image_type (str | tuple[str, ...]): File extension(s) to include (e.g. "tif" or ("tif", "png"))

    Returns:
        List[str]: Sorted list of full paths to input image files.
    """
    if isinstance(input_image_type, str):
        input_image_type = (input_image_type,)

    input_files = []
    for file_extension in input_image_type:
        input_files.extend(glob(os.path.join(input_dir, f"*.{file_extension}")))

    input_files.sort()
    return input_files


