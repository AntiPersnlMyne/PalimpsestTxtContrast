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
        multispec_image (List[np.ndarray]): Input image with 3+ bands (image channels).
        max_components     (int, optional): How many components PCA should retain; by default (`None`) all the components are retained. Defaults to None.

    Returns:
        pca_bgr_image (np.ndarray): Output image of shape (rows, cols, num_components) and datatype of source.
        mean_vector   (np.ndarray): Computed or passed-in mean vector of shape (1, channels).
        eigenvectors  (np.ndarray): Principal component axes, shape (num_components, channels).
        eigenvalues   (np.ndarray): Variances along each principal component, shape (num_components, 1).
    """
    
    # Validate input dimensions
    if multispectral_image.ndim != 3:
        raise ValueError("Input image must have shape (rows, cols, channels).")

    # Unpack dimensions 
    rows, cols, channels = multispectral_image.shape
    total_pixels = rows * cols

    # Flatten image
    data_matrix = multispectral_image.reshape(total_pixels, channels).astype(np.float32)

    # Compute PCA basis 
    mean_vector, eigenvectors, eigenvalues = cv.PCACompute2(
        data_matrix,
        mean=None,
        maxComponents=num_components
    )
    # mean_vector:      shape (1, channels)
    # eigenvectors:     shape (num_components, channels)
    # eigenvalues:      shape (num_components, 1)

    # Project the data onto the top principal components
    # Resulting shape: (total_pixels, num_components)
    projected_data = cv.PCAProject(data_matrix, mean_vector, eigenvectors)

    # Reshape projected data back into image form: (rows, cols, num_components)
    pca_image_float = projected_data.reshape(rows, cols, num_components)

    # Allocate output RGB image and normalize each component to [0, 255]
    pca_rgb_image = np.empty_like(pca_image_float, dtype=np.uint8)
    for component_index in range(num_components):
        component_plane = pca_image_float[:, :, component_index]
        min_val = float(component_plane.min())
        max_val = float(component_plane.max())

        if max_val > min_val:
            # Linearly stretch to full 8-bit range
            scaled_plane = (component_plane - min_val) * (255.0 / (max_val - min_val))
        else:
            # Avoid division by zero if the component is constant
            scaled_plane = np.zeros_like(component_plane, dtype=np.float32)

        pca_rgb_image[:, :, component_index] = np.round(scaled_plane).astype(np.uint8)

    return pca_rgb_image, mean_vector, eigenvectors, eigenvalues
    



# --------------------------------------------------------------------------------------------
# (Optional) Execution
# --------------------------------------------------------------------------------------------

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        