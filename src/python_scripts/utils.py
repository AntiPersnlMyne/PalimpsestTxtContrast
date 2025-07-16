"""utils.py: Image processing routines"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production". 

# Image processing
import numpy as np
import scipy
import skimage
import cv2 as cv
import matplotlib.pyplot as plt

# File management
import os
import tifffile

# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------

def _read_files(src_dir: str) -> dict:
    """Import TIFF files from source directory.

    Args:
        src_dir (str): Directory of source images

    Returns:
        list[cv.Mat]: Array of all imported images
    """
    
    if not os.path.exists(src_dir):
        raise Exception("Source directory not found")
    
    # Dictionary and filename of imported images
    images = {}
    
    # Search src directory for all .tif, reads in all .tif files
    for filename in os.listdir(src_dir):
        if filename.endswith('.tif') or filename.endswith(".tiff"):
            with open(os.path.join(src_dir, filename), 'r') as file:
                # dict( filename: image )
                images.update({file.name: cv.imread(file.name, cv.IMREAD_UNCHANGED)}) 
                
    return images

def _write_files(dst_dir: str, dict_of_images: dict, suffix: str = "") -> None:
    """Export processed files into destination directory

    Args:
        dst_dir (str): Directory for processed images.
        array_of_images (list[cv.Mat]): Array of images to be saved to file.
        suffix (str, Optional): Suffix to be appended to file names. Default is "".
    """
    
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    f_type = ".tif"
    
    for img in dict_of_images:
        # Adds suffix and saves as TIFF, for each image
        cv.imwrite(img[:img.find(".")] + suffix + f_type, 
                   dict_of_images[img])


# --------------------------------------------------------------------------------------------
# Image Processing
# --------------------------------------------------------------------------------------------

def gaussian_blur(src_dir: str, dst_dir: str, kernel_shape = (3,3), sigma_x = 0, file_suffix="") -> None:
    """Applies a Gaussian filter (blur) using a weighted mean

    Args:
        src_dir (str): Directory to input image file(s), accepts `.tif` or `.tiff` images.
        dst_dir (str): Directory to output image file(s).
        kernel_size (tuple, optional): Shape - (row, col) - of blur kernel. Defaults to (3,3).
        sigma_x (int, optional): The Gaussian kernel standard deviation. Defaults to 0.
        file_suffix(str, optional): Suffix to be appended to processed files. Default is "".
    """
    src_images = _read_files(src_dir=src_dir)
    dst_images = src_images.copy() # Shallow copy ; readability purposes only
    
    for img in src_images:
        # Run operation and update dst_dict with processed image
        dst_images.update( img = cv.GaussianBlur(src_images[img], kernel_shape, sigma_x) )
    
    _write_files(dst_dir, dst_images, file_suffix)

def clahe(src_dir: str, dst_dir: str, tile_grid_size: tuple, clip_limit: int = 3, file_suffix: str = "") -> None:
    """
    Contrast Limited Adaptive Historgram Equalization, region-based histogram equalization for under/over-exposed images. Tilewise operation.

    Args:
        src_dir (str): Directory to input image file(s). Accepts `.tif` or `.tiff` images.
        dst_dir (str, optional): Directory to output image file(s).
        tile_grid_size (tuple, optional): Breaks up image into M x N tiles, to process each tile individually.
        clip_limit (int, optional): Threshold for contrast limiting. Typically leave this value in the range of 2-5. If you set the value too large, process may maximize local contrast, which will, in turn, maximize noise. Defaults to 3.
        file_suffix (str, optional): Suffix to be appended to processed files. Default is "".
    """
    
    src_images = _read_files(src_dir=src_dir)
    dst_images = src_images.copy() # Shallow copy ; readability purposes only
    
    # Create CLAHE operator 
    #   note: Only works on grayscale (single-band) images
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    for img in src_images:
        # Run operation and update dst_dict with processed image
        dst_images.update( img = clahe.apply(src_images[img]) )
    
    _write_files(dst_dir, dst_images, file_suffix)

def otsu(src_dir: str, dst_dir: str = "", file_suffix: str = "") -> None:
    """Applies OTSU threshold, maximally separating foreground from background, as binary image.

    Args:
        src_dir (str): Directory to input image file(s). Accepts `.tif` or `.tiff` images.
        dst_dir (str, optional): Directory to output image file(s). 
        file_suffix (str, optional): Suffix to be appended to processed files. Default is "".
    """
    
    # NOTE: Consider maxVal to be lower, considering most data is in the lower DC (for 165r at least)
    
    src_images = _read_files(src_dir=src_dir)
    dst_images = src_images.copy() # Shallow copy ; readability purposes only
    
    for img in src_images:
        # note: iinfo is unnecessary if assuming each .tif is the same bit depth
        _, otsu_image = cv.threshold(src_images[img], 0, float( np.iinfo(src_images[img].dtype).max ), cv.THRESH_BINARY + cv.THRESH_OTSU)
        dst_images.update(img = otsu_image)
        
    _write_files(dst_dir, dst_images, file_suffix)

def gaussian_threshold(src_dir: str, dst_dir: str, block_size: int, mean_subtract = 0, file_suffix="") -> None:
    """Gaussian thresholding dtermines the threshold for a pixel based on a small region around it. Tilewise operation.

    Args:
        src_dir (str): Directory to input image file(s). Accepts `.tif` or `.tiff` images.
        dst_dir (str, optional): Directory to output image file(s).
        block_size (int): Determines the size of the neighbourhood area.
        mean_subtract (int, optional): Constant subtracted from average or weighted mean of neighborhood. Defaults to 0.
        file_suffix (str, optional): Suffix to be appended to processed files. Default is "".
    """
    
    src_images = _read_files(src_dir)
    dst_images = src_images.copy() # Shallow copy ; readability purposes only
    
    for img in src_images:
        # note: iinfo is unnecessary if assuming each .tif is the same bit depth
        dst_images.update(img = cv.adaptiveThreshold(img, float( np.iinfo(src_images[img].dtype).max ), cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                            cv.THRESH_BINARY, block_size, mean_subtract)
                          )
        
    _write_files(dst_dir, dst_images, file_suffix)
    
def dilation(src_dir:str, dst_dir:str, kernel_shape:tuple = (3,3), iterations:int = 1, file_suffix="") -> None:
    """Dilation thickens stroke lines by adding pixels to the boundaries of objects in an image.

    Args:
        src_dir (str): Directory to input image file(s). Accepts `.tif` or `.tiff` images.
        dst_dir (str, optional): Directory to output image file(s). 
        kernel_shape (tuple, optional): Larger kernel leads to broader strokes. Defaults to (3,3).
        iterations (int, optional): Amount of iterations by the kernel, increases strokes added. Defaults to 1.
        file_suffix (str, optional): Suffix to be appended to processed files. Default is "".
    """
    
    src_images = _read_files(src_dir)
    dst_images = src_images.copy() # Shallow copy ; readability purposes only
    
    # Defining kernel
    kernel = np.ones(kernel_shape,dtype=np.uint8)
    
    for img in src_images:
        # Applying dilation operation
        dst_images.update(img = cv.dilate(src_images[img], kernel, iterations=iterations))

    _write_files(dst_dir, dst_images, file_suffix)
   
def erosion(src_dir:str, dst_dir:str, kernel_shape:tuple = (3,3), iterations:int = 1, file_suffix="") -> None:
    """Erosion thins stroke lines by subtracting pixels to the boundaries of objects in an image.

    Args:
        src_dir (str): Directory to input image file(s). Accepts `.tif` or `.tiff` images.
        dst_dir (str, optional): Directory to output image file(s). 
        kernel_shape (tuple, optional): Larger kernel leads to broader strokes. Defaults to (3,3).
        iterations (int, optional): Amount of iterations by the kernel, increases strokes added. Defaults to 1.
        file_suffix (str, optional): Suffix to be appended to processed files. Default is "".
    """
    
    src_images = _read_files(src_dir)
    dst_images = src_images.copy() # Shallow copy ; readability purposes only
    
    # Defining kernel
    kernel = np.ones(kernel_shape,dtype=np.uint8)
    
    for img in src_images:
        # Applying dilation operation
        dst_images.update(img = cv.erode(src_images[img], kernel, iterations=iterations))

    _write_files(dst_dir, dst_images, file_suffix)
   
   
    
    
    
    
    
    