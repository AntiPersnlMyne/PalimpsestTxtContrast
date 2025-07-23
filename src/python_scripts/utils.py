"""utils.py: Image processing routines"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "2.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production", or "Prototype". 

# Image processing
import numpy as np
import scipy
import skimage
import cv2 as cv
import matplotlib.pyplot as plt

# Management
import os
import warnings

# Typing
from typing import Any, Optional, Callable
from enum import Enum

# Sharpen constant aliases
class SharpenMethod(Enum):
    KERNEL2D = 0
    UNSHARP_MASKING = 1
    HIGH_BOOST = 2
    
# Sharpen constant aliases
class BitwiseOperation(Enum):
    AND = 0
    NOT = 1
    OR = 2
    XOR = 3

# Bitwise constants
AND:int = 0
NOT:int = 1
OR:int = 2 
XOR:int = 3

# Sharpen constants
KERNEL2D:int = 0
UNSHARP_MASKING:int = 1
HIGH_BOOST:int = 2 



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
        dtype=img.dtype
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
        transform_kwargs (dict, optional): Optional keyword arguments passed to the transform function.
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



# --------------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------------
def gaussian_blur(
    img:np.ndarray,
    kernel_shape:tuple = (3,3), 
    sigma_x:int = 0, 
    ) -> np.ndarray:
    """
    Applies a Gaussian filter (blur) using a weighted mean

    Args:
        img (np.ndarray): Input image.
        kernel_size (tuple, optional): Shape - (row, col) - of blur kernel. Defaults to (3,3).
        sigma_x (int, optional): The Gaussian kernel standard deviation. Defaults to 0.
    """
    
    return cv.GaussianBlur(img, kernel_shape, sigma_x)
    

def clahe(
    img:np.ndarray,
    tile_grid_size:tuple, 
    clip_limit: int = 3, 
    ) -> np.ndarray:
    """
    Contrast Limited Adaptive Historgram Equalization, region-based histogram equalization for under/over-exposed images. Tilewise operation.

    Args:
        img (np.ndarray): Input image.
        tile_grid_size (tuple): Breaks up image into tuple(r,c) size tiles, to process each tile individually.
        clip_limit (int, optional): Threshold for contrast limiting. Typically leave this value in the range of 2-5. If you set the value too large, process may maximize local contrast, which will, in turn, maximize noise. Defaults to 3.
    """
    
    # Assert grayscale
    if img.ndim > 1:
        raise Exception("Error: CLAHE input image must be grayscale.")

    # Create CLAHE operator 
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    return clahe.apply(img)
    

def bilateral_filter(
    img:np.ndarray,
    diameter:int, 
    sigma_color:float, 
    sigma_space:float, 
    ) -> np.ndarray:
    """Filter for smoothening images and reducing noise, while preserving edges.

    Args:
        img (np.ndarray): Input image.
        diameter (int): Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace. 
        sigma_color (float): Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color. 
        sigma_space (float): Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace. 
    """
    
    return cv.bilateralFilter(src=img, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    

def sharpen(
    img:np.ndarray,
    s:int = 1, 
    radius:int = 3, 
    sharp_method:int = KERNEL2D, 
    normalize:bool = False, 
    ) -> np.ndarray:
    """
    Sharpens image by enhancing contrast at edges.

    Args:
        img (np.ndarray): Input image.
        s (int, optional): This controls how much edges are amplified. Must be >= 1. Defaults to 1.
        radius (int, optional): Defines the radius (in pixels) of the Gaussian blur kernel for UNSHARP and HIGH_BOOST. Must be an odd integer (e.g., 3, 5, ...). Defaults to 3.
        sharp_method(int, optional): Sharpening methods in utils.SharpenMethod. Defaults to KERNEL2D.
        normalize(bool, optional): If true, stretch output to the full dynamic range of the original datatype. Defaults to False.
    """
    
    # Parameter validity
    if radius < 3 or radius % 2 == 0:
        raise ValueError("-- Error: radius in sharpen must be an odd integer >= 3 --")
    if s < 1:
        raise ValueError("-- Error: sharpening strength 's' must be >= 1 --")
    if s < 4 and sharp_method == KERNEL2D:
        warnings.warn(" -- Warning: s < 4 may result inexpected behavior (inverted or dim imagery) --", UserWarning)

    # Gaussian blur standard deviation; magic-number!
    sigma_x = 0
    
    sharpen_kernel = np.array(
        [[ 0, -1,  0],
         [-1,  s, -1],
         [ 0, -1,  0]],
        dtype=np.float32)
    
    src_dtype = img.dtype
    float_img = img.astype(np.float32)
    
    match sharp_method:
        case SharpenMethod.KERNEL2D: 
            sharp_img = cv.filter2D(float_img, cv.CV_32F, sharpen_kernel)

            return _clip_or_norm(sharp_img, src_dtype, normalize)
            
        case SharpenMethod.UNSHARP_MASKING: 
            # Low-frequency details
            lowpass_img = cv.GaussianBlur(float_img, (radius, radius), sigma_x)
            # High-frequency details
            highpass_img = float_img - lowpass_img

            # Combine high-frequency details to enhance edges
            sharp_img = (float_img + s * highpass_img).astype(np.float32)

            return _clip_or_norm(sharp_img, src_dtype, normalize)

        case SharpenMethod.HIGH_BOOST:
            # Low-frequency details
            lowpass_img = cv.GaussianBlur(float_img, (radius, radius), sigma_x)
            
            # High-boost apply 
            sharp_img = (s * float_img - lowpass_img).astype(np.float32)

            return _clip_or_norm(sharp_img, src_dtype, normalize)
            
        case _:
            raise Exception(f"-- Error: Provide valid sharpen method. Use utils.SharpenMethod. --")



# --------------------------------------------------------------------------------------------
# Bitwise Operations
# --------------------------------------------------------------------------------------------
def bitwise(
    source1:np.ndarray, 
    source2:..., # Nothing can possibly go wrong
    operation:int, 
    dst_dir:str|None = None, 
    mask:np.ndarray|None = None
    ) -> np.ndarray|None:
    """Performs bitwise (AND, NOT, OR, XOR) operation on two images.

    Args:
        source1 (np.ndarray): First image array
        source2 (np.ndarray): Second image array
        operation (int): Operation type. Use utils.BitwiseOperation.
        dst_dir (str|None, optional): Directory for processed image. If None, returns image. Defaults to None.
        mask (Mat|None, optional): Image mask. Defaults to None.
        
    Returns:
        np.ndarray|None: If dst_dir is None, returns bitwise image. Otherwise, saves image to dst_dir as source1 dtype.
    """
    
    # bitwise only works on uint8 or float
    source1 = source1.astype(np.float16)
    source2 = source2.astype(np.float16)
    
    if source2 is not None and source2 != BitwiseOperation.NOT:
        match operation:
            case BitwiseOperation.AND:
                img = cv.bitwise_and(source1, source2, mask=mask)
            case BitwiseOperation.OR: 
                img = cv.bitwise_or(source1, source2, mask=mask)
            case BitwiseOperation.XOR:
                img = cv.bitwise_xor(source1, source2, mask=mask)
            
    elif source2 is not None and source2 == BitwiseOperation.NOT:
            img = cv.bitwise_not(source1, mask=mask)        
            
    else: 
        raise Exception("Please pass valid bitwise operation type: NOR, OR, XOR, AND")
        
    # Write img to dst if path is provided, else return iamge as source1 dtype
    if dst_dir is not None: cv.imwrite(dst_dir, img.astype(np.iinfo(source1).dtype))
    else: return img.astype(np.iinfo(source1).dtype)



# --------------------------------------------------------------------------------------------
# Thresholding
# --------------------------------------------------------------------------------------------
def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """
    Apply OTSU's binarization to a single image, maximizing foreground/background separation.

    Args:
        img (np.ndarray): Input 2D image with integer dtype.

    Returns:
        np.ndarray: Binary image after OTSU thresholding.
    """
    
    # Datatype and dims check
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("otsu_threshold requires an image with integer dtype.")
    if img.ndim != 2:
        raise ValueError("Input image must be 2D.")

    # Normalize to 8-bit for OTSU's histogram-based thresholding
    img_8bit = cv.normalize(img, np.empty(0), 0, 255, norm_type=cv.NORM_MINMAX).astype(np.uint8)

    # Apply OTSU thresholding
    _, binary = cv.threshold(img_8bit, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Use mask on original image, or return binary mask depending on use-case
    output = np.where(binary == 255, img, 0).astype(img.dtype)
    return output



def gaussian_threshold(
    img:np.ndarray,
    block_size:int = 3, 
    c_subtract:int = 0, 
    ) -> np.ndarray:
    """
    Apply Gaussian adaptive thresholding to a single image.

    Args:
        img (np.ndarray): Input image.
        block_size (int, optional): Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on. Defaults to 3.
        mean_subtract (int, optional): Constant subtracted from average or weighted mean of neighborhood. Defaults to 0.
    """
    
    # Datatype and dims check
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("gaussian_threshold requires an integer dtype image.")
    if img.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError("block_size must be an odd integer ≥ 3.")
            
    # Convert down to 8-bit
    img_8bit = cv.normalize(img, np.empty(0), 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    # Generate binary mask
    mask = cv.adaptiveThreshold(
        img_8bit,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY,
        blockSize=block_size,
        C=c_subtract
    )

    # Apply binary mask to original image
    output = np.where(mask == 255, img, 0).astype(img.dtype)
    return output


def binary_threshold(
    img: np.ndarray,
    thresh: int|None = None,
    invert: bool = False
    ) -> np.ndarray:
    """
    Apply binary thresholding to a single image, with optional auto-threshold and inversion.

    Args:
        img (np.ndarray): Input image (integer dtype, 2D).
        thresh (int, optional): Pixel threshold value. If None, use Triangle algorithm.
        invert (bool): If True, use inverse binary logic (below=white, above=black).

    Returns:
        np.ndarray: Binary thresholded image.
    """
    
    # Datatype and dims check
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("binary_threshold requires an integer dtype image.")
    if img.ndim != 2:
        raise ValueError("Input image must be 2D.")

    # Thresh type to OpenCV readable
    thresh_type = cv.THRESH_BINARY_INV if invert else cv.THRESH_BINARY
    maxval = np.iinfo(img.dtype).max

    if thresh is None:
        # Normalize image to 0–255 for OpenCV Triangle algorithm
        img_uint8 = cv.normalize(img, np.empty(0), 0, 255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        
        # Calculate and scale thresh value
        thresh_val, _ = cv.threshold(img_uint8, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
        thresh = int(thresh_val * maxval / 255)

    _, img_thresh = cv.threshold(img, thresh, maxval, thresh_type)
    return img_thresh

    
def to_zero_threshold(
    img:np.ndarray,
    thresh:int|None = None,
    invert:bool = False
    ) -> np.ndarray:
    """
    Apply TOZERO thresholding to a single image, with optional inversion and auto-thresholding.

    Args:
        img (np.ndarray): Input single-channel image with integer dtype.
        thresh (int, optional): Threshold value. If None, use Triangle method to estimate it.
        invert (bool): If True, use inverted logic (below threshold retained, others zeroed).

    Returns:
        np.ndarray: Thresholded image.
    """
    
    # Datatype check 
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("to_zero_threshold requires an integer input image.")

    # Thresh type to OpenCV readable
    thresh_type = cv.THRESH_TOZERO_INV if invert else cv.THRESH_TOZERO
    
    maxval = np.iinfo(img.dtype).max

    if thresh is None:
        # Normalize image to 0–255 for OpenCV Triangle algorithm
        img_uint8 = cv.normalize(img, np.empty(0), 0, 255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        thresh_val, _ = cv.threshold(img_uint8, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
        thresh = int(thresh_val * maxval / 255)  # Scale thresh back to original dtype range

    # Use thresh value to threshold original image
    _, img_thresh = cv.threshold(img, thresh, maxval, thresh_type)
    return img_thresh

    
    
# --------------------------------------------------------------------------------------------
# Morphological Operations
# --------------------------------------------------------------------------------------------
def dilation(
    img:np.ndarray,
    kernel_shape:tuple = (3,3), 
    iterations:int = 1, 
    ) -> np.ndarray:
    """
    Dilation thickens stroke lines by adding pixels to the boundaries of objects in an image.

    Args:
        img (np.ndarray): Input image.
        kernel_shape (tuple, optional): Larger kernel leads to broader strokes. Defaults to (3,3).
        iterations (int, optional): Amount of iterations by the kernel, increases strokes added. Defaults to 1.
        
    Returns:
        np.ndarray: Dilated image.
    """
    
    # Dilation kernel
    kernel = np.ones(kernel_shape,dtype=np.uint8)
    
    return cv.dilate(img, kernel, iterations=iterations)


def erosion(
    img:np.ndarray,
    kernel_shape:tuple = (3,3), 
    iterations:int = 1, 
    ) -> np.ndarray:
    """
    Erosion thins stroke lines by subtracting the pixels on the boundaries of objects.

    Args:
        img (np.ndarray): Input image.
        kernel_shape (tuple, optional): Larger kernel leads to broader strokes. Defaults to (3,3).
        iterations (int, optional): Amount of iterations by the kernel, increases strokes added. Defaults to 1.
        
    Returns:
        np.ndarray: Eroded image.
    """
    
    # Erosion kernel
    kernel = np.ones(kernel_shape,dtype=np.uint8)
    
    return cv.erode(src=img, kernel=kernel, iterations=iterations)
   


# --------------------------------------------------------------------------------------------
# Contrast-related Operations
# --------------------------------------------------------------------------------------------
def fft_magnitude_spectrum_float(img: np.ndarray, use_log:bool = False) -> np.ndarray:
    """
    Compute the float32 magnitude spectrum of a 2D image's FFT. Oprtionally, return log-magnitude.

    Args:
        img (np.ndarray): Input image (2D, integer dtype).
        use_log (bool): If true, compute log-magnitude spectrum. Defauls to False.

    Returns:
        np.ndarray: Float32 magnitude spectrum.
    """
    
    # Datatype and dims check
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("fft_magnitude_spectrum expects integer input.")
    if img.ndim != 2:
        raise ValueError("Input image must be 2D grayscale.")

    # Flaot for precision
    img_float = img.astype(np.float32)

    # Compute 2D-FFT
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)

    # Magnitude spectrum
    magnitude = np.abs(fft_shifted)
    
    # (optional) Magnitude-log spectrum
    if use_log: magnitude = np.log1p(magnitude)

    return magnitude.astype(np.float32)
    
def scale_brightness(
    img:np.ndarray,
    alpha:int = 1, 
    beta:int = 0
    ) -> np.ndarray:
    """
    Modifies the linear contrast (alpha), and additive brightness (beta).

    Args:
        img (np.ndarray): Input image.
        alpha (int, optional): Contrast control. E.g., 1.5 is a 50% increase in contrast. Defaults to 1.
        beta (int, optional): Brightness control. Adds a uniform value beta to all pixel values. Defaults to 0.
    """
    
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)

def linear_stretch(
    img: np.ndarray,
    min_val: int|None = None,
    max_val: int|None = None
    ) -> np.ndarray: 
    """
    Apply a linear histogram stretch to an image using optional custom min/max range.

    Args:
        img (np.ndarray): Input image (integer dtype).
        min_val (Optional[int]): Lower bound of pixel range. If None, uses image min.
        max_val (Optional[int]): Upper bound of pixel range. If None, uses image max.

    Returns:
        np.ndarray: Linearly stretched image of the same dtype.
    """
    
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("linear_stretch requires an image with integer dtype.")

    # Datatype information
    dtype_info = np.iinfo(img.dtype)
    out_min, out_max = dtype_info.min, dtype_info.max

    # Image min and max
    img_min = int(img.min()) if min_val is None else min_val
    img_max = int(img.max()) if max_val is None else max_val

    # Image min and max bounds checking
    if img_min == img_max:
        return np.full_like(img, out_min)  # Avoid division by zero
    if img_min > img_max:
        raise ValueError(f"Invalid stretch range: min_val ({img_min}) > max_val ({img_max}).")
    if not (dtype_info.min <= img_min <= dtype_info.max and dtype_info.min <= img_max <= dtype_info.max):
        raise ValueError(
            f"Stretch range ({img_min}, {img_max}) out of bounds for dtype {img.dtype}: "
            f"{dtype_info.min} to {dtype_info.max}"
        )

    # Clip input to avoid out-of-range effects
    clipped = np.clip(img, img_min, img_max)

    # Normalize to full dtype range
    img_stretched = cv.normalize(
        src=clipped,
        dst=np.empty(0),
        alpha=out_min,
        beta=out_max,
        norm_type=cv.NORM_MINMAX,
        dtype=img.dtype
    )

    return img_stretched
    
def log_stretch(img: np.ndarray) -> np.ndarray:
    """
    Apply a logarithmic stretch to enhance darker regions of the image.

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Stretched image with same dtype.
    """
    
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("logarithmic_stretch requires integer dtype image.")

    # Datatype information
    dtype_info = np.iinfo(img.dtype)
    out_min, out_max = dtype_info.min, dtype_info.max

    # Float conversion for log operation
    img_float = img.astype(np.float32)
    img_normalized = img_float / dtype_info.max  # Normalize to [0, 1]

    # Log transformation
    norm_const = 1.0 / np.log(1 + 1.0)  # Scale constant for normalization
    img_log = norm_const * np.log1p(img_normalized) 

    # Scale result to dtype range
    img_stretched = cv.normalize(
        img_log,
        dst=np.empty(0),
        alpha=out_min,
        beta=out_max,
        norm_type=cv.NORM_MINMAX,
        dtype=img.dtype
    )

    return img_stretched

def percentile_stretch(
    img: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0
    ) -> np.ndarray:
    """
    Apply a percentile-based linear stretch to an image to reduce the influence of outliers.

    Args:
        img (np.ndarray): Input image with integer dtype.
        lower_percentile (float): Lower percentile threshold (0.0 - 100.0). Defauls to 2.0.
        upper_percentile (float): Upper percentile threshold (0.0 - 100.0). Defaults to 98.0.

    Returns:
        np.ndarray: Percentile-stretched image with same dtype.
    """
    
    # Input and percentile parameter check
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("percentile_stretch requires an image with integer dtype.")
    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError("Percentiles must satisfy: 0 <= lower < upper <= 100.")

    # Datatype information
    dtype_info = np.iinfo(img.dtype)
    out_min, out_max = dtype_info.min, dtype_info.max

    # Column-wise flatten image
    img_flat = img.flatten()
    min_val = int(np.percentile(img_flat, lower_percentile))
    max_val = int(np.percentile(img_flat, upper_percentile))

    # Stretch value error handling
    if min_val == max_val:
        return np.full_like(img, out_min)  
    if min_val > max_val:
        raise ValueError("Computed min_val is > max_val; invalid stretch.")

    # Clip and stretch using OpenCV
    clipped = np.clip(img, min_val, max_val)
    
    stretched = cv.normalize(
        src=clipped,
        dst=np.empty(0),
        alpha=out_min,
        beta=out_max,
        norm_type=cv.NORM_MINMAX,
        dtype=img.dtype
    )

    return stretched
    
    
    
    
    
    