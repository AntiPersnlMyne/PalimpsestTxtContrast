"""utils.py: Image processing routines"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Development", or "Production", or "Prototype". 


# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
# Python imports
import numpy as np
import cv2 as cv
import warnings
from typing import Sequence, List

# Helper functions
from ..utils.improc_utils import _read_files, _write_files, _normalize_image, _normalize_image_range, _clip_or_norm, _return_cv_img_dtype
# ENUM
from ..utils.improc_utils import *


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
    diameter:int = 5, 
    sigma_color:float = 50, 
    sigma_space:float = 50, 
    ) -> np.ndarray:
    """
    Filter for smoothening images and reducing noise, while preserving edges.

    Args:
        img (np.ndarray): Input image.
        diameter (int): Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace. 
        sigma_color (float): Filter standard deviation (sigma) in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigma_space) will be mixed together, resulting in larger areas of semi-equal color. 
        sigma_space (float): Filter standard deviation (sigma) in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When diameter>0, it specifies the neighborhood size regardless of sigma_space. Otherwise, diameter is proportional to sigma_space. 
    """
    
    # Input typchecking
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("Image must have integer dtype.")

    # Convert to float32
    dtype_info = np.iinfo(img.dtype)
    float_img = img.astype(np.float32)
    
    filtered_img = cv.bilateralFilter(src=float_img, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    # Clip values and convert to original dtype
    filtered_clipped = np.clip(filtered_img, dtype_info.min, dtype_info.max)
    result = filtered_clipped.astype(img.dtype)
    
    return result
    

def sharpen(
    img:np.ndarray,
    s:int = 1, 
    radius:int = 3, 
    sharp_method:int = SharpenMethod.KERNEL2D, 
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
    if s < 4 and sharp_method == SharpenMethod.KERNEL2D:
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
    source2:np.ndarray|None, # Nothing can possibly go wrong
    operation:BitwiseOperation, 
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
    
    # Input matching check
    if source2 is not None and source1.dtype != source2.dtype:
        raise TypeError("source1 and source2 must be same type")
    if source2 is not None and source1.shape != source2.shape:
        raise ValueError("source1 and source2 must have the same shape.")
    
    # bitwise only works on uint8 or float
    source1 = source1.astype(np.float32)
    source2 = source2.astype(np.float32) if source2 is not None else None
    
    # source1 and source2 provided
    if source2 is not None and operation != BitwiseOperation.NOT:
        match operation:
            case BitwiseOperation.AND:
                img = cv.bitwise_and(source1, source2, mask=mask)
            case BitwiseOperation.OR: 
                img = cv.bitwise_or(source1, source2, mask=mask)
            case BitwiseOperation.XOR:
                img = cv.bitwise_xor(source1, source2, mask=mask)
                
    # source1 provided only
    elif operation == BitwiseOperation.NOT:
        if source2 is not None: warnings.warn("Operation bitwise NOT, source 2 ignored.")
        img = cv.bitwise_not(source1, mask=mask)
        
    # Invalid bitwise operaiton
    else: 
        raise Exception("Please pass valid bitwise operation type: NOR, OR, XOR, AND")
        
    # Write img to dst if path is provided
    if dst_dir is not None: cv.imwrite(filename=dst_dir, img=img)
    
    return img.astype(np.iinfo(source1).dtype)



# --------------------------------------------------------------------------------------------
# Thresholding
# --------------------------------------------------------------------------------------------
def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """
    Apply OTSU's thresholding maximally separating background from foreground.

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Binary image - same dtype as img - where pixels >= threshold are 255 (white), else 0 (black).
    """
    
    # Datatype and dims check
    if not np.issubdtype(img.dtype, np.integer):
        raise TypeError("otsu_threshold_manual requires an integer dtype image.")
    if img.ndim != 2:
        raise ValueError("Input image must be 2D.")

    # Flatten image
    flat = img.ravel()
    
    # Preserve bit depth of source with bin number
    dtype_info = np.iinfo(img.dtype)
    bins = dtype_info.max + 1  # e.g., 65536 for uint16

    # Compute histogram
    hist, _ = np.histogram(flat, bins=bins, range=(0, dtype_info.max), density=False)
    total = flat.size

    # Compute OTSU's threshold
    prob = hist / total
    omega = np.cumsum(prob)                 # cumulative probability
    mu = np.cumsum(prob * np.arange(bins))  # cumulative mean
    mu_t = mu[-1]                           # total mean

    sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-10)  # Between-class variance

    otsu_thresh = np.argmax(sigma_b_squared)
    
    # Apply threshold to original image
    img_otsu = np.where(img >= otsu_thresh, 255, 0).astype(img.dtype)
    return img_otsu


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
def morph_dilation(
    img:np.ndarray,
    kernel_size:int= 3, 
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
    kernel = np.ones((kernel_size,kernel_size),dtype=np.uint8)
    
    return cv.dilate(img, kernel, iterations=iterations)


def morph_erosion(
    img:np.ndarray,
    kernel_size:int = 3, 
    iterations:int = 1, 
    ) -> np.ndarray:
    """
    Erosion thins stroke lines by subtracting the pixels on the boundaries of objects.

    Args:
        img (np.ndarray): Input image.
        kernel_shape (tuple, optional): Larger kernel leads to broader strokes. Requires odd integer, e.g. 3 = (3,3), 5 = (5,5), etc. Defaults to 3.
        iterations (int, optional): Amount of iterations by the kernel, increases strokes added. Defaults to 1.
        
    Returns:
        np.ndarray: Eroded image.
    """
    
    # Kernel value check
    if kernel_size % 2 == 0 or kernel_size <= 1: 
        raise ValueError("Kernel size must be an odd integer, greater than 1.")
    
    # Erosion kernel
    kernel = np.ones((kernel_size,kernel_size),dtype=np.uint8)
    
    return cv.erode(src=img, kernel=kernel, iterations=iterations)


def morph_open(
    img:np.ndarray,
    kernel_size:int = 3,
    iterations:int = 1
    ) -> np.ndarray:
    """
    Erosion followed by Dilation. It is useful in removing small holes outside the foreground object, or small white points outside the object. 

    Args:
        img (np.ndarray): Input image.
        kernel_shape (tuple, optional): Larger kernel leads to more agressive noise removal. Requires odd integer, e.g. 3 = (3,3), 5 = (5,5), etc. Defaults to 3.
        iterations (int, optional): Amount of iterations by the kernel, increases strokes added. Defaults to 1.

    Returns:
        np.ndarray: Processed image.
    """
    if kernel_size % 2 == 0 or kernel_size <= 1: 
        raise ValueError("Kernel size must be an odd integer, greater than 1.")
    
    # close  kernel
    kernel = np.ones((kernel_size,kernel_size),dtype=np.uint8)
    
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=kernel, iterations=iterations)


def morph_close(
    img:np.ndarray,
    kernel_size:np.ndarray,
    iterations:int = 1
    ) -> np.ndarray:
    """
    Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object. 

    Args:
        img (np.ndarray): Input image.
        kernel_shape (tuple, optional): Larger kernel leads to more agressive noise removal. Requires odd integer, e.g. 3 = (3,3), 5 = (5,5), etc. Defaults to 3.
        iterations (int, optional): Amount of iterations by the kernel, increases strokes added. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    if kernel_size % 2 == 0 or kernel_size <= 1: 
        raise ValueError("Kernel size must be an odd integer, greater than 1.")
    
    # close  kernel
    kernel = np.ones((kernel_size,kernel_size),dtype=np.uint8)
    
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=kernel, iterations=iterations)


def morph_tophat(
    img:np.ndarray, 
    kernel_size:np.ndarray,
    iterations:int = 1
    ) -> np.ndarray:
    """
    The top-hat filter is used to enhance bright objects of interest in a dark background.
      The very small details are enhanced and taken out using the Top-Hat operation. Hence, 
      it is useful in observing the minor details of the inputs when are present as 
      light pixels on a dark background. 

    Args:
        img (np.ndarray): Input image.
        kernel_shape (tuple, optional): Larger kernel leads to more agressive noise removal. Requires odd integer, e.g. 3 = (3,3), 5 = (5,5), etc. Defaults to 3.
        iterations (int, optional): Amount of iterations by the kernel, increases strokes added. Defaults to 1.

    Returns:
        np.ndarray: Processed image.
    """
    
    # Kernel value check
    if kernel_size % 2 == 0 or kernel_size <= 1: 
        raise ValueError("Kernel size must be an odd integer, greater than 1.")
    
    # tophat  kernel
    kernel = np.ones((kernel_size,kernel_size),dtype=np.uint8)
    
    return cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel=kernel, iterations=iterations)


def morph_blackhat(
    img:np.ndarray, 
    kernel_size:np.ndarray,
    iterations:int = 1
    ) -> np.ndarray:
    """
    The top-hat filter is used to enhance bright objects of interest in a dark background.
      The very small details are enhanced and taken out using the Top-Hat operation. Hence, 
      it is useful in observing the minor details of the inputs when are present as 
      light pixels on a dark background. 

    Args:
        img (np.ndarray): Input image.
        kernel_shape (tuple, optional): Larger kernel leads to more agressive noise removal. Requires odd integer, e.g. 3 = (3,3), 5 = (5,5), etc. Defaults to 3.
        iterations (int, optional): Amount of iterations by the kernel, increases strokes added. Defaults to 1.

    Returns:
        np.ndarray: Processed image.
    """
    
    # Kernel value check
    if kernel_size % 2 == 0 or kernel_size <= 1: 
        raise ValueError("Kernel size must be an odd integer, greater than 1.")
    
    # tophat  kernel
    kernel = np.ones((kernel_size,kernel_size),dtype=np.uint8)
    
    return cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel=kernel, iterations=iterations)




# --------------------------------------------------------------------------------------------
# Contrast-related Operations
# --------------------------------------------------------------------------------------------
def fft_magnitude(
    img: np.ndarray, 
    use_log:bool = False
    ) -> np.ndarray:
    """
    Compute the float32 magnitude spectrum of a 2D image's FFT. Oprtionally, return log-magnitude.

    Args:
        img (np.ndarray): Input image (2D, integer dtype).
        use_log (bool): If true, compute log-magnitude spectrum. Defauls to False.

    Returns:
        np.ndarray (float32): Magnitude spectrum.
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
        dtype=_return_cv_img_dtype(img=img)
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
        dtype=_return_cv_img_dtype(img=img)
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
        dtype=_return_cv_img_dtype(img=img)
    )

    return stretched
    
    
    
# --------------------------------------------------------------------------------------------
# Character Recognition Operations
# --------------------------------------------------------------------------------------------
def find_contours(img:np.ndarray) -> tuple[Sequence[np.ndarray], np.ndarray]:
    raise NotImplementedError
    
def draw_contours(img:np.ndarray, contours:List[np.ndarray], color:tuple, thickness:int):
    raise NotImplementedError
    
    
    
# --------------------------------------------------------------------------------------------
# Color Space
# --------------------------------------------------------------------------------------------
def extract_luminous(img:np.ndarray) -> np.ndarray:
    """
    Reads in BGR image, converts to CIELAB space, returns the L channel.

    Args:
        img (np.ndarray): Input BGR image.
        
    Returns:
        np.ndarray: Luminous (L) channel of `img`.
    """
    return cv.cvtColor(img.astype(np.float32), cv.COLOR_BGR2LAB, None)[:,:,0]


