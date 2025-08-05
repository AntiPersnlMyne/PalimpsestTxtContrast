"""bgp.py: Band Generation Process, crated new non-linear bondinations of existing bands"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
from itertools import combinations
from tqdm import tqdm
from typing import Callable, Union, Tuple, List
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
# @njit(fastmath=True, cache=True)
def create_bands_from_block(
    image_block: ImageBlock,
    use_sqrt: bool,
    use_log: bool
) -> List[np.ndarray]:
    """
    Creates new, non-linear bands from existing bands for the ATDCA algorithm.
    This process is based on the GOSP paper (Cheng and Ren).

    Args:
        image_block (np.ndarray): A 3D numpy array representing a block of the image,
                                  with shape (bands, height, width).
        use_sqrt (bool): Flag to indicate if sqrt bands should be generated.
        use_log (bool): Flag to indicate if log bands should be generated.

    Returns:
        np.ndarray: A 3D numpy array containing the newly generated correlated bands,
                    with shape (new_bands, height, width).
    """
    num_bands, height, width = image_block.shape
    new_bands = []

    # Get the original bands
    original_bands = [image_block[i] for i in range(num_bands)]
    new_bands.extend(original_bands)

    # Cross-correlation bands
    if num_bands >= 2:
        for band_a_idx, band_b_idx in combinations(range(num_bands), 2):
            cross_correlation_band = image_block[band_a_idx] * image_block[band_b_idx]
            new_bands.append(cross_correlation_band)

    # Sqrt-transformed bands
    if use_sqrt:
        for band in original_bands:
            sqrt_band = np.sqrt(band)
            new_bands.append(sqrt_band)

    return new_bands


def get_global_min_max(
    image_reader: ImageReader,
    image_shape: ImageShape,
    block_shape: ImageShape,
    use_sqrt: bool,
    use_log: bool
) -> Tuple[float, float]:
    """
    Performs a first pass over the image to find the global minimum and maximum values
    for the newly generated bands.

    Args:
        image_reader (ImageReader): Reader function for the input image.
        image_shape (ImageShape): The dimensions of the entire image.
        block_shape (Tuple[int, int]): The size of the processing blocks.
        use_sqrt (bool): Flag to indicate if sqrt bands were generated.
        use_log (bool): Flag to indicate if log bands were generated.

    Returns:
        Tuple[float, float]: A tuple containing the global minimum and maximum values.
    """
    global_min = np.inf
    global_max = -np.inf
    image_height, image_width = image_shape
    block_height, block_width = block_shape

    print("[BGP] First pass: Gathering global statistics...")
    for row_off in tqdm(range(0, image_height, block_height), desc="[BGP] First pass", colour='CYAN'):
        for col_off in range(0, image_width, block_width):
            actual_height = min(block_height, image_height - row_off)
            actual_width = min(block_width, image_width - col_off)
            window = ((row_off, col_off), (actual_height, actual_width))
            input_block = image_reader(window)
            
            # Check if the block is a valid numpy array before processing
            if isinstance(input_block, np.ndarray):
                new_bands = create_bands_from_block(input_block, use_sqrt, use_log)
                
                # Update global min/max
                global_min = min(global_min, np.min(new_bands))
                global_max = max(global_max, np.max(new_bands))
            
    return global_min, global_max



def band_generation_process(
    image_reader: ImageReader,
    image_writer: ImageWriter,
    block_shape: Tuple[int, int] = (512, 512),
    use_sqrt: bool = True,
    use_log: bool = False
):
    """
    The main Band Generation Process function, now implementing a two-pass approach
    for global normalization and supporting additional non-linear bands.
    """
    # Get image reader and block data
    image_shape = image_reader("window_shape")
    if image_shape is None:
        raise ValueError("[BGP] image_reader returned None, cannot determine image dimensions.")
    
    image_height, image_width = image_shape
    block_height, block_width = block_shape

    # First Pass: Find global min/max
    if isinstance(image_shape, tuple): # Check if image_shape is type ImageShape
        global_min, global_max = get_global_min_max(image_reader, image_shape, block_shape, use_sqrt, use_log)
    
    # Second Pass: Normalize and write
    print("[BGP] Second pass: Normalizing and writing blocks...")
    for row_off in tqdm(range(0, image_height, block_height), desc="[BGP] Second pass", colour='CYAN'):
        for col_off in range(0, image_width, block_width):
            actual_height = min(block_height, image_height - row_off)
            actual_width = min(block_width, image_width - col_off)
            
            window = ((row_off, col_off), (actual_height, actual_width))
            input_block = image_reader(window)

            # Check if the block is a valid numpy array before processing
            if isinstance(input_block, np.ndarray):
                # Create the bands from the block
                output_block = create_bands_from_block(input_block, use_sqrt, use_log)

                # Apply global normalization
                output_block = normalize_data(output_block, min_val=global_min, max_val=global_max)

                # Write the normalized block
                image_writer(window, output_block)







