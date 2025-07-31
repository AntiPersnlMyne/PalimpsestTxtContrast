"""tgp.py: Target Generation Process. Automatically creates N most significant targets in target detection for pixel classification"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# src/python_scripts/tgp.py

import numpy as np
from tqdm import tqdm 
from typing import Callable, Tuple

# Define window tuple
ImageWindow = Tuple[Tuple[int, int], Tuple[int, int]]

def target_generation_process(
    
    # image_reader:Callable[ [str|ImageWindow], np.ndarray|Tuple[int,int] ], 
    image_reader,
    block_shape=(512, 512)
    ):
    """
    Generates up to `max_targets` target vectors using OSP and OPCI stopping rule.

    Args:
        image_reader (Callable): A function serves as an interface to read image data. It supports two modes:
            - image_reader("shape"): Returns (height, width) of the full image.
            - image_reader(window): Returns a block of shape (H, W, B) where `window` is a tuple of the form ((row_offset, col_offset), (height, width)).
        
        block_shape (tuple, optional): Shape of blocks to read (height, width). Defaults to (512, 512).

    Returns:
        (np.ndarray, tuple): (t0_vector, t0_coords).\n 
            t0-vector: Spectral vector of the initial target (shape: [bands]).
            t0_coords: (row, col) coordinates of the pixel with the maximum spectral norm.
    """

    max_norm = -np.inf
    t0_vector = None
    t0_coords = None

    image_height, image_width = image_reader("shape")  # special mode to get metadata
    block_height, block_width = block_shape

    for row_off in tqdm(range(0, image_height, block_height), desc="Scanning rows"):
        for col_off in range(0, image_width, block_width):
            window = ((row_off, col_off), (block_height, block_width))
            block = image_reader(window)  # shape: (H, W, B)

            if block is None:
                continue

            # Flatten to 2D: (H*W, B)
            h, w, b = block.shape
            reshaped = block.reshape(-1, b)
            norms = np.linalg.norm(reshaped, axis=1)

            local_max_idx = np.argmax(norms)
            local_max_val = norms[local_max_idx]

            if local_max_val > max_norm:
                max_norm = local_max_val
                t0_vector = reshaped[local_max_idx]
                local_row = local_max_idx // w
                local_col = local_max_idx % w
                t0_coords = (row_off + local_row, col_off + local_col)

    return t0_vector, t0_coords
