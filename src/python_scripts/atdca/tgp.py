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

def target_generation_process(image_reader, block_shape=(512, 512)):
    """
    Stage 1 of the Target Generation Process (TGP).
    Finds initial target T0: the pixel vector with the maximum L2 norm.
    
    Parameters:
    -----------
    image_reader : Callable[[tuple], np.ndarray]
        A function that accepts a window tuple ((row_off, col_off), (height, width))
        and returns a 3D NumPy array of shape (height, width, bands).
    block_shape : tuple
        Shape of blocks to read (height, width).

    Returns:
    --------
    t0_vector : np.ndarray
        The spectral vector (shape: [bands]) of the initial target.
    t0_coords : tuple
        The (row, col) coordinates of the target pixel in the full image.
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
