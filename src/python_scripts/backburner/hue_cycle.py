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
def pca(
    multispectral_image:np.ndarray, 
    num_components:int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs OpenCV's Principle Component Analysis (PCA), creates pseudo-color output array.

    Args:
        multispectral_image (np.ndarray): Input image with 3+ bands (image channels).
        num_components   (int, optional): How many components PCA should retain. Defaults to 3.

    Returns:
        pca_bgr_image (np.ndarray): Output image of shape (rows, cols, num_components) and datatype of source.
    """
    
    # Validate input dimensions
    if multispectral_image.ndim != 3:
        raise ValueError("Input image must have shape (rows, cols, channels).")

    # Unpack multispectral data 
    rows, cols, channels = multispectral_image.shape
    total_pixels = rows * cols
    src_dtype = multispectral_image.dtype
    dtype_max = np.iinfo(src_dtype).max

    # Flatten image
    data_matrix = multispectral_image.reshape(total_pixels, channels).astype(np.float32)

    # Compute PCA basis 
    mean_vector, eigenvectors, _ = cv.PCACompute2(
        data=data_matrix,
        mean=np.empty(0),
        maxComponents=num_components
    )

    # Project the data onto principal components
    projected_data = cv.PCAProject(data_matrix, mean_vector, eigenvectors)

    # Reshape projected data back into image form: (rows, cols, num_components)
    pca_image_float = projected_data.reshape(rows, cols, num_components)

    # Allocate output RGB image and normalize each component to [0, 255]
    pca_bgr_image = np.empty_like(pca_image_float, dtype=src_dtype)
    for component_index in range(num_components):
        component = pca_image_float[:, :, component_index]
        min_val = float(component.min())
        max_val = float(component.max())

        if max_val > min_val:
            # Linearly stretch to full 16-bit range
            scaled_component = np.empty_like(component)
            cv.normalize(component, scaled_component, 0, dtype_max, cv.NORM_MINMAX)
        else:
            # Avoid division by zero if the component is constant
            scaled_plane = np.zeros_like(component, dtype=np.float32)

        pca_bgr_image[:, :, component_index] = np.round(scaled_plane).astype(src_dtype)

    return pca_bgr_image
    



# --------------------------------------------------------------------------------------------
# (Optional) Execution
# --------------------------------------------------------------------------------------------

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        