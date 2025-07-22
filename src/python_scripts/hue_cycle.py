"""hue_switch.py: Automatically visualizes different band combinations and hue shifts of a band combination"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production". 



# --------------------------------------------------------------------------------------------
# User Variables
# --------------------------------------------------------------------------------------------
HUE_INCREMENT:int = 1          # Shifts hue angle by amount, every iteration. Must be between [1-179]
DISPLAY_INTERVAL_MS:int = 0    # Time in miliseconds between image transition
                               #  If 0, continues only with user input



# --------------------------------------------------------------------------------------------
# Import Modules
# --------------------------------------------------------------------------------------------
import os               # filename i/o
import cv2 as cv        # imshow, imread
import numpy as np      # array
from typing import List # List of Ideal band combinations
from math import atan2, cos, sin, sqrt, pi # PCA

# --------------------------------------------------------------------------------------------
# Global Variables
# --------------------------------------------------------------------------------------------
keypress = ""
display_image = np.empty((0))



# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
def _read_files(src_dir: str) -> dict[str, np.ndarray]:
    """Read all .tif or .tiff files in the directory into a dictionary of {filename: image array}.

    Args:
        src_dir (str): Directory of source images

    Returns:
        list[cv.Mat]: Array of all imported images
    """
    
    if not os.path.exists(src_dir):
        raise Exception(f"Source directory not found: {src_dir}")
    
    # Dictionary and filename of imported images
    images = {}
    
    # Search src directory for all .tif, reads in all .tif files
    for filename in os.listdir(src_dir):
        if filename.endswith(('.tif', '.tiff')):
            path = os.path.join(src_dir, filename)
            image = cv.imread(path, cv.IMREAD_UNCHANGED)
            
            images[filename] = image
            
    return images

def _write_files(dst_dir: str, dict_of_images: dict[str, np.ndarray], suffix: str = "") -> None:
    """Export processed files into destination directory

    Args:
        dst_dir (str): Directory for processed images.
        array_of_images (list[cv.Mat]): Array of images to be saved to file.
        suffix (str, Optional): Suffix to be appended to file names. Default is "".
    """
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for filename, image in dict_of_images.items():
        base, _ = os.path.splitext(filename)
        output_path = os.path.join(dst_dir, f"{base}{suffix}.tif")
        success = cv.imwrite(output_path, image)
        if not success:
            raise IOError(f"Failed to write image: {output_path}")



# --------------------------------------------------------------------------------------------
# Image Creation and Manipulation
# --------------------------------------------------------------------------------------------
def build_bgr(images:List[np.ndarray]) -> np.ndarray:
    raise NotImplementedError

def increment_hue(hsv_img:np.ndarray) -> np.ndarray:
    raise NotImplementedError

def bgr2hsv(bgr_img:np.ndarray) -> np.ndarray:
    raise NotImplementedError
    
def hsv2bgr(hsv_img:np.ndarray) -> np.ndarray:
    raise NotImplementedError

def display_img(bgr_img:np.ndarray, wait_time_ms:int) -> None:
    pass



# --------------------------------------------------------------------------------------------
# User Input
# --------------------------------------------------------------------------------------------
def handle_user_input() -> str:
    raise NotImplementedError

def save_settings(band_combination:List[str], hue_angle:int) -> None:
    raise NotImplementedError
    


# --------------------------------------------------------------------------------------------
# Image Processing
# --------------------------------------------------------------------------------------------
def pca(image:List[np.ndarray], max_components:int|None = None) -> List[np.ndarray]:
    """Performs OpenCV's Principle Component Analysis (PCA) and retruns processed image.

    Args:
        image (List[np.ndarray]): Input image.
        max_components (int, optional): How many components PCA should retain; by default (`None`) all the components are retained. Defaults to None.

    Returns:
        (List[np.ndarray]): PCA image with # of channels determined by max_components.
    """
    
    mean = np.empty((0)) # or None
    mean, eigenvectors, _ = cv.PCACompute2(image, mean=mean, max_components=max_components)
    
    raise NotImplementedError
    



# --------------------------------------------------------------------------------------------
# (Optional) Execution
# --------------------------------------------------------------------------------------------

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        