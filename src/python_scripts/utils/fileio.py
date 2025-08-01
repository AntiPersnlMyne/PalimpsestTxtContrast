"""fileio.py: File read and write"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import cv2 as cv
import numpy as np
import rasterio
import warnings
from typing import List
import os



# --------------------------------------------------------------------------------------------
# Helper function
# --------------------------------------------------------------------------------------------
def _glob_import(filepath: str, extension: str | List[str] | None = None) -> dict:
    """
    Helper to import images from a directory with optional extension filtering.
    """
    images = {}
    supported_exts = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.tiff', '.tif']
    if extension is None:
        exts = supported_exts # set to default list, will import any support types
    elif isinstance(extension, str):
        exts = ['.' + extension.lower()] # convert to extension - jpg -> .jpg
    else:
        exts = ['.' + e.lower() for e in extension] # imports user specificed types, that match supported

    for filename in os.listdir(filepath):
        _, ext = os.path.splitext(filename)
        if ext.lower() in exts:
            path = os.path.join(filepath, filename)
            image = cv.imread(path, cv.IMREAD_UNCHANGED)
            if image is not None:
                images[filename] = image
    return images


# --------------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------------
def imread(filepath:str) -> np.ndarray:
    """
    Reads in image. Protects against returning None.

    Args:
        filepath (str): Directory to image, include image name and extension e.g. "data/image.png"

    Raises:
        ValueError: Filepath cannot be empty (None)
        TypeError: filepath must be a string (str)
        FileNotFoundError: file was not found in filepath directory

    Returns:
        np.ndarray: Image in BGR format (unless 4+ channels)
    """
    
    # Check filepath validity
    if filepath is None:
        raise ValueError("[FILEIO] Filepath cannot be None")
    if not isinstance(filepath, str):
        raise TypeError("[FILEIO] Provide absolute (e.g. C:Users/.../image.png) or relative (e.g. ../data/image.png) \
                        path to image location on your drive. \
                        Ensure you include the image name and its extension (e.g. /<MyPathWithoutBracket>/image.png")
    image = cv.imread(filepath, cv.IMREAD_UNCHANGED)
    
    # Check image validity
    if image is not None:
        return image
    else:
        raise FileNotFoundError("[FILEIO] Imread file not found")
     
    
def imread_folder(filepath:str, extension:str|List[str]|None = None) -> dict:
    """Imports all images from a folder. Optionally, only imports specific type extension(s).

    Args:
        filepath (str): Filepath to folder, including the folder-name, to import images from.
        extension (str or List[str], Optional): Suffix added to the name of a computer file, e.g. ".jpg". If None, imports all supported image types.

    """
    # Filepath check
    if not os.path.exists(filepath):
        raise Exception(f"[FILEIO] Directory not found: {filepath}")

    # Import images from directory
    images = _glob_import(filepath, extension)
    return images



# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------
def imwrite(filepath:str, image:np.ndarray) -> None:
        # Check filepath validity
    if filepath is None:
        raise ValueError("[FILEIO] Filepath cannot be None")
    if not isinstance(filepath, str):
        raise TypeError("[FILEIO] Provide absolute (e.g. C:Users/.../) or relative (e.g. ../data/) \
                        path to image location on your drive.")
    
    cv.imwrite(filepath, image)


