"""bgp.py: Band Generation Process, crated new non-linear bondinations of existing bands"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


import numpy as np
from itertools import combinations
from tqdm import tqdm
from typing import Callable, Union, Tuple

# Window, Reader, and Writer datatypes
# Keeps function signatures clean
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]
ImageBlock = np.ndarray
ImageShape = Tuple[int, int]

ImageReader = Callable[[Union[str, WindowType]], Union[ImageBlock, ImageShape, None]]
ImageWriter = Callable[[WindowType, ImageBlock], None]


def _band_generation_process_to_block(
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
        np.ndarray:  Augmented block of shape (H, W, B').
        Includes:
        - Original bands
        - Auto-correlated bands
        - Cross-correlated bands
        - Optional: square-root and/or log bands
    """
    
    _, _, block_bands = block.shape
    output_bands = []

    # Original bands
    output_bands.append(block)

    # Auto-correlated bands (bi * bi)
    auto_bands = block ** 2
    output_bands.append(auto_bands)

    # Cross-correlated bands (i < j) (b1 * bj)
    cross_band_list = []
    for i, j in combinations(range(block_bands), 2):
        cross = block[:, :, i] * block[:, :, j]
        cross_band_list.append(cross[:, :, np.newaxis])
    cross_bands = np.concatenate(cross_band_list, axis=2)
    output_bands.append(cross_bands)

    # Square-root bands
    if use_sqrt:
        sqrt_bands = np.sqrt(np.clip(block, 0, None))
        output_bands.append(sqrt_bands)

    # Logarithmic bands (optional)
    if use_log:
        log_bands = np.log1p(np.clip(block, 0, None))  # log(1 + x)
        output_bands.append(log_bands)

    # Concat (combine) new bands into block
    bgp_block = np.concatenate(output_bands, axis=2).astype(np.float32)
    return bgp_block


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
    reader = image_reader("shape")
    if reader is None:
        raise ValueError("image_reader returned None, cannot determine image dimensions.")
    
    # Get block size
    image_height, image_width = reader
    block_height, block_width = block_shape

    # Progress bar
    for row_off in tqdm(range(0, image_height, block_height), desc="Processing BGP", colour='00ff80'):
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

            output_block = _band_generation_process_to_block(
                block=input_block,
                use_sqrt=use_sqrt,
                use_log=use_log
            )

            image_writer(window, output_block)











