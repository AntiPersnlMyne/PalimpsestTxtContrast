"""utils.py: Image processing helpers"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production", or "Prototype". 



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
# Image processing
import numpy as np
import cv2 as cv

# Management
import os
import warnings

# Typing
from typing import Any, Optional, Callable, List, Tuple, Sequence
from enum import IntEnum


# --------------------------------------------------------------------------------------------
# Enumerations (custom datatype flags)
# --------------------------------------------------------------------------------------------
# Sharpen constant aliases
class SharpenMethod(IntEnum):
    KERNEL2D = 0
    UNSHARP_MASKING = 1
    HIGH_BOOST = 2
    
# Sharpen constant aliases
class BitwiseOperation(IntEnum):
    AND = 0
    NOT = 1
    OR = 2
    XOR = 3


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


def _write_files(
    dst_dir: str, 
    dict_of_images: dict[str, np.ndarray], 
    suffix: str = ""
    ) -> None:
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


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        image (np.ndarray): Input image.
        dtype (np.dtype): Output image desired datatype.

    Returns:
        np.ndarray: Normalized image, with cv.NORM_MINMAX, as type dtype.
    """
    
    max_val = np.iinfo(img.dtype).max                         # Normalize dtype maxvalue
    norm_img = np.empty_like(img)                             # Empty array for output
    cv.normalize(img, norm_img, 0, max_val, cv.NORM_MINMAX)   # Normalized between 0 and maxval
    
    return norm_img.astype(img.dtype) # Returned normalized as dtype


def _normalize_image_range(
    img: np.ndarray, 
    min_val: int|None,
    max_val: int|None
    ) -> np.ndarray:
    """
    Normalize an image to the full range of its dtype, optionally using a specified min/max range.
    
    Args:
        img (np.ndarray): Input image.
        min_val (int, optional): Minimum input intensity. If None, uses image min.
        max_val (int, optional): Maximum input intensity. If None, uses image max.

    Returns:
        np.ndarray: Linearly stretched image, same dtype as input.
    """
    dtype_info = np.iinfo(img.dtype)
    dtype_min, dtype_max = dtype_info.min, dtype_info.max

    img_min = min_val if min_val is not None else int(img.min())
    img_max = max_val if max_val is not None else int(img.max())

    if img_min == img_max:
        raise Exception("-- Error: invalid range. min_val and max_val must be different. --")

    # Clip to the range to avoid values outside the stretch window
    clipped = np.clip(img, img_min, img_max)

    # Normalize using OpenCV (note: output is float32 unless dtype is specified)
    normalize_image = np.empty_like(clipped)
    cv.normalize(
        src=clipped,
        dst=normalize_image,
        alpha=dtype_min,
        beta=dtype_max,
        norm_type=cv.NORM_MINMAX,
        dtype=_return_cv_dtype(img=img)
    )
    
    return normalize_image


def _clip_or_norm(
    img: np.ndarray, 
    dtype: np.dtype, 
    normalize: bool
    ) -> np.ndarray:
    """Sharpen helper function. Clips output range to dtype, or normalizes output range to dtype's min and max.

    Args:
        image (np.ndarray): Input image.
        dtype (np.dtype): Output image desired datatype.
        normalize (bool): If true, cv.MINMAX normalizes output to dtype range. Otherwise, only clips to dtype range.

    Returns:
        np.ndarray(np.ndarray): Clipped or normalized output image as type dtype.
    """
    if normalize:
        return _normalize_image(img) # Returns as dtype
    else:
        return np.clip(img, 0, np.iinfo(dtype).max).astype(dtype)


def _return_cv_dtype(img:np.ndarray|None = None, np_dtype:np.dtype|None = None) -> int:
    """Converts np.dtype to cv.dtype. Pass in one parameter to return the dtype. If both parameters are passed (not None), np_dtype is ignored.
    Args:
        img (np.ndarray | None, optional): Returns the dtype of image as cv.dtype. If None, assumes np.dtype parameter was passed instead. Defaults to None.
        np_dtype (np.dtype | None, optional): Converts the np.dtype to cv.dtype. If None, assumes img parameter was passed instead. Defaults to None.
    """
    # Return img dtype
    if img is not None:
        # Warn second parameter is being ignored
        if np_dtype is not None: warnings.warn("np_dtype parameter is ignored")
        
        # Return cv dtype
        img_dtype = img.dtype
        match img_dtype:
            case np.uint8:
                return cv.CV_8U
            case np.uint16:
                return cv.CV_16U
            case np.float32:
                return cv.CV_32F
            case np.float64:
                return cv.CV_64F
            case _:
                raise Exception(f"Warning: image dtype - {img_dtype} - is not supported")
            
    # Return np_dtype 
    if np_dtype is not None:
        match np_dtype:
            case np.uint8:
                return cv.CV_8U
            case np.uint16:
                return cv.CV_16U
            case np.float32:
                return cv.CV_32F
            case np.float64:
                return cv.CV_64F
            case _:
                raise Exception(f"Warning: image dtype - {np_dtype} - is not supported")

    warnings.warn("Error: No valid parameters given", RuntimeWarning)
    return -1 # Exit falure, will crash OpenCV


def process_images(
    src_dir: str,
    dst_dir: str,
    file_suffix: str,
    transform_fn: Callable[..., np.ndarray],
    transform_kwargs: Optional[dict[str, Any]] = None
) -> None:
    """
    Apply a transform function to all images in a directory and write outputs.

    Args:
        src_dir (str): Directory of input images.
        dst_dir (str): Directory to write transformed images.
        file_suffix (str): Suffix to append to output filenames.
        transform_fn (Callable): A function that accepts an image and returns a transformed image.
        transform_kwargs (dict, optional): Optional keyword arguments passed to the transform function. e.g., to pass keyword arguments (kwargs) for utils.bitwise: {"source2": image, "dst_dir": "data/output"}.
    """
    
    if transform_kwargs is None:
        transform_kwargs = {}

    # Read input and Allocate output
    src_images = _read_files(src_dir)
    dst_images = {}

    # Process all images from source directory
    for name, img in src_images.items():
        try:
            dst_images[name] = transform_fn(img, **transform_kwargs)
        except Exception as e:
            warnings.warn(f"Failed to process image '{name}': {e}")

    _write_files(dst_dir, dst_images, file_suffix)



    
    
    
    