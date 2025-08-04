"""bgp.py: Band Generation Process, crated new non-linear bondinations of existing bands"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
from itertools import combinations
from tqdm import tqdm
from typing import Callable, Union, Tuple
from numba import njit
from ..utils.math_utils import normalize_data 


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]
ImageBlock = np.ndarray
ImageShape = Tuple[int, int]

ImageReader = Callable[[Union[str, WindowType]], Union[ImageBlock, ImageShape, None]]
ImageWriter = Callable[[WindowType, ImageBlock], None]


# --------------------------------------------------------------------------------------------
# Band Generation Process (BGP)
# --------------------------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def compute_correlated_bands(image_block:np.ndarray) -> np.ndarray:
    """
    Creates new, non-linear bands from existing bands for the ATDCA algorithm.
    This process is based on the GOSP paper (Cheng and Ren).

    Args:
        image_block (np.ndarray): A 3D numpy array representing a block of the image,
                                  with shape (bands, height, width).

    Returns:
        np.ndarray: A 3D numpy array containing the newly generated correlated bands,
                    with shape (new_bands, height, width).
    """
    
    # channels-first, band-centric approach
    num_bands, height, width = image_block.shape
        # Calculate all unique cross-correlations
    new_bands = []
    # Use combinations to avoid duplicating pairs (e.g., band_1, band_2 and band_2, band_1)
    for band_a_idx, band_b_idx in combinations(range(num_bands), 2):
        band_a = image_block[band_a_idx]
        band_b = image_block[band_b_idx]
        
        # Element-wise multiplication to create the new band (cross-correlation)
        new_band = band_a * band_b
        new_bands.append(new_band)
        
    # Stack the new bands into a single numpy array
    new_bands = np.stack(new_bands, axis=0)

    # Apply min-max normalization to the new bands to scale their values
    # to the range [0, 1]. This is crucial for keeping values in a manageable
    # range for subsequent steps of the algorithm.
    new_bands = normalize_data(new_bands)
    
    return new_bands


def band_generation_process_to_block(
    block, 
    use_sqrt=True,
    use_log=False
) -> np.ndarray:
    """
    Applies the Band Generation Process (BGP) to raster block. Optional flags align with Cheng and Ren (2000)'s implementation.

    Args:
        block (np.ndarray|Tuple[int,int]): Input block of shape (H, W, B) with float32 pixel values.
        use_sqrt (bool, optional): If True, includes square-root bands.. Defaults to True.
        use_log (bool, optional): If True, includes logarithmic bands ( log(1 + B) ). Defaults to False.

    Returns:
        np.ndarray:  Augmented (added bands) block of shape (H, W, B').
    """
    
    output_bands = [block]

    # Auto-correlated bands (bi * bi)
    output_bands.append(block ** 2)
    
    # Cross-correlated bands (i < j) (b1 * bj)
    cross_bands = compute_cross_bands(block)
    output_bands.append(cross_bands)

    # Square-root bands
    if use_sqrt:
        sqrt_bands = np.sqrt(np.clip(block, 0, None))
        output_bands.append(sqrt_bands)

    # Logarithmic bands
    if use_log:
        log_bands = np.log1p(np.clip(block, 0, None))  # log(1 + x)
        output_bands.append(log_bands)

    # Concat (combine) new bands into block
    return np.concatenate(output_bands, axis=2).astype(np.float32)


def band_generation_process(
    image_reader: ImageReader,
    image_writer: ImageWriter,
    block_shape: Tuple[int, int] = (512, 512),
    use_sqrt: bool = True,
    use_log: bool = False
    ) -> None:
    """
    Applies band generation across the full image block by block.

    Args:
        image_reader (Callable): Function to read a block or get shape. Accepts:
                                - "shape": returns (height, width)
                                - window: returns block (H, W, B) or None on failure
        image_writer (Callable): Function to write a processed block to output image.
        block_shape (Tuple[int, int], optional): Processing block shape. Defaults to (512, 512).
        use_sqrt (bool, optional): Include square-root bands. Defaults to True.
        use_log (bool, optional): Include log bands. Defaults to False.

    Returns:
        None
    """
    
    # Get image reader and block data
    image_shape = image_reader("shape")
    if image_shape is None:
        raise ValueError("[BGP] image_reader returned None, cannot determine image dimensions.")
    
    # Get block size
    image_height, image_width = image_shape
    block_height, block_width = block_shape

    # Progress bar
    for row_off in tqdm(range(0, image_height, block_height), desc="Processing BGP", colour='CYAN'):
        for col_off in range(0, image_width, block_width):
            # Catch/prevent out-of-bounds
            actual_height = min(block_height, image_height - row_off)
            actual_width = min(block_width, image_width - col_off)
            
            # Extract window and input as block
            window = ((row_off, col_off), (actual_height, actual_width))
            input_block = image_reader(window)

            # Ignore empty/bad blocks
            if input_block is None:
                continue

            output_block = band_generation_process_to_block(
                block=input_block,
                use_sqrt=use_sqrt,
                use_log=use_log
            )

            image_writer(window, output_block)











