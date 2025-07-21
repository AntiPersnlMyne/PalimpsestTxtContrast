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

# Typing
from typing import Any
Mat = np.typing.NDArray[Any]
from enum import Enum

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
# Filters
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
        dst_images[img] = cv.GaussianBlur(src_images[img], kernel_shape, sigma_x)
    
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
        dst_images[img] = clahe.apply(src_images[img])
    
    _write_files(dst_dir, dst_images, file_suffix)

def bilateral_filter(src_dir: str, dst_dir: str, diameter:int, sigma_color:int, sigma_space:int, file_suffix: str = "") -> None:
    """Filter for smoothening images and reducing noise, while preserving edges.

    Args:
        src_dir (str): Directory to input image file(s). Accepts `.tif` or `.tiff` images.
        dst_dir (str, optional): Directory to output image file(s).
        diameter (int): Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace. 
        sigma_color (int): Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color. 
        sigma_space (int): 	Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace. 
        file_suffix (str, optional): Suffix to be appended to processed files. Default is "".
    """
    
    src_images = _read_files(src_dir=src_dir)
    dst_images = src_images.copy() # Shallow copy ; readability purposes only
    
    for img in src_images:
        # Run operation and update dst_dict with processed image
        dst_images[img] = cv.bilateralFilter(src_images[img], diameter, sigma_color, sigma_space)
    
    _write_files(dst_dir, dst_images, file_suffix)
    pass

# --------------------------------------------------------------------------------------------
# Bitwise Operations
# --------------------------------------------------------------------------------------------
# Enumeration for readabiltiy
class BitwiseOperation(Enum):
    AND = 0
    NOT = 1
    OR = 2
    XOR = 3

# Constant for user input
AND:int = 0
NOT:int = 1
OR:int = 2 
XOR:int = 3

def bitwise(source1:Mat, source2:Mat, operation:int, dst_dir:str|None = None, mask:Mat|None = None) -> Mat|None:
    """Performs bitwise (AND, NOT, OR, XOR) operation on two images.

    Args:
        source1 (str): First image array
        source2 (str): Second image array
        operation (str): Operation type. Use utils.AND,.OR,.NOT,.XOR.
        dst_dir (str | None, optional): Directory for processed image. If None, returns image. Defaults to None.
        mask (Mat | None, optional): Image mask. Defaults to None.
    """
    
    # bitwise only works on uint8 or float
    source1 = source1.astype(np.float16)
    source2 = source2.astype(np.float16)
    
    match operation:
        case BitwiseOperation.AND:
            bit_image = cv.bitwise_and(source1, source2, mask=mask)
        case BitwiseOperation.OR: 
            bit_image = cv.bitwise_or(source1, source2, mask=mask)
        case BitwiseOperation.XOR:
            bit_image = cv.bitwise_xor(source1, source2, mask=mask)
        case BitwiseOperation.NOT:
            bit_image = cv.bitwise_not(source1, mask=mask)
        case _:
            raise Exception("Please pass valid bitwise operation type: NOR, OR, XOR, AND")
        
    # Write img to dst if path is provided, else return iamge
    if dst_dir is not None: cv.imwrite(dst_dir, bit_image.astype(np.uint16))
    else: return bit_image.astype(np.uint16)



# --------------------------------------------------------------------------------------------
# Thresholding
# --------------------------------------------------------------------------------------------
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
        dst_images[img] = otsu_image
        
    _write_files(dst_dir, dst_images, file_suffix)

def gaussian_threshold(src_dir:str, dst_dir:str, block_size:int, mean_subtract:int = 0, file_suffix="") -> None:
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

    # -- OpenCV only accepts 8-bit, create mask and apply --
    
    for img in src_images:
                
        # Convert down 16-bit to 8-bit
        img_8bit = cv.normalize(src_images[img], None, 0, 255, cv.NORM_MINMAX) #type:ignore
        img_8bit = img_8bit.astype(np.uint8)

        # Binary mask
        mask = cv.adaptiveThreshold(
            img_8bit,            
            255,                 
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            block_size,
            mean_subtract
        )

        # Apply mask to 16-bit image
        result = np.where(mask == 255, src_images[img], 0)
        
        # note: iinfo is unnecessary if assuming each .tif is the same bit depth
        dst_images[img] = result
        
    _write_files(dst_dir, dst_images, file_suffix)

def binary_threshold(src_dir: str, dst_dir: str, thresh: int = -1, invert:bool = False, file_suffix="") -> None:
    """Simple binary threshold, everything below thresh goes to 0 (black) and above thresh to maxval (white). Flag to invert thresh logic.

    Args:
        src_dir (str): Directory to input image file(s). Accepts `.tif` or `.tiff` images.
        dst_dir (str, optional): Directory to output image file(s).
        thresh (int): Pixel threshold value. Defaults to Triangle algorithm to choose the optimal threshold value .
        invert (bool, optional): Invert threshold logic; below thresh to maxval (white) and above to 0 (black). Defaults to False.
        file_suffix (str, optional): Suffix to be appended to processed files. Default is "".
    """
    
    src_images = _read_files(src_dir)
    dst_images = src_images.copy() # Shallow copy ; readability purposes only
    
    # Translate func param values into OpenCV parsable params
    inversion_state = cv.THRESH_BINARY_INV if invert else cv.THRESH_BINARY # Invert logic, else regular logic
    for img in src_images:
        maxval = float( np.iinfo(src_images[img].dtype).max ) # img datatype's max value
        break
    thresh = cv.THRESH_TRIANGLE if (thresh<0) else thresh # triangle thresh algorithm, else user input
    
    for img in src_images:
        _, thresh_img = cv.threshold(src_images[img], thresh, maxval, inversion_state, None)
        dst_images[img] = thresh_img
        pass
    
    _write_files(dst_dir, dst_images, file_suffix)
    
def to_zero_threshold(src_dir: str, dst_dir: str, thresh: int = -1, invert:bool = False, file_suffix="") -> None:
    """
     Min/max threshold. Values above thresh stay the same, below thresh to zero. Flag to invert logic.
    
    Args:
        src_dir (str): Directory to input image file(s). Accepts `.tif` or `.tiff` images.
        dst_dir (str, optional): Directory to output image file(s).
        thresh (int): Pixel threshold value. Defaults to Triangle algorithm to choose the optimal threshold value .
        invert (bool, optional): Invert threshold logic; below thresh to maxval (white) and above to 0 (black). Defaults to False.
        file_suffix (str, optional): Suffix to be appended to processed files. Default is "".
    """
    
    src_images = _read_files(src_dir)
    dst_images = src_images.copy() # Shallow copy ; readability purposes only
    
    # Translate func param values into OpenCV parsable params
    inversion_state = cv.THRESH_TOZERO_INV if invert else cv.THRESH_TOZERO # Invert logic, else regular logic
    for img in src_images:
        maxval = float( np.iinfo(src_images[img].dtype).max ) # img datatype's max value
        break
    thresh = cv.THRESH_TRIANGLE if (thresh<0) else thresh # triangle thresh algorithm, else user input
    
    for img in src_images:
        _, thresh_img = cv.threshold(src_images[img], thresh, maxval, inversion_state, None)
        dst_images[img] = thresh_img
        pass
    
    _write_files(dst_dir, dst_images, file_suffix)
    
    
    
# --------------------------------------------------------------------------------------------
# Morphological Operations
# --------------------------------------------------------------------------------------------
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
        dst_images[img] = cv.dilate(src_images[img], kernel, iterations=iterations)

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
        dst_images[img] = cv.erode(src_images[img], kernel, iterations=iterations)

    _write_files(dst_dir, dst_images, file_suffix)
   
   
    
    
    
    
    
    