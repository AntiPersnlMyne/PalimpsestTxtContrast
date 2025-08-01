"""tgp.py: Target Generation Process. Automatically creates N most significant targets in target detection for pixel classification"""

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
from typing import Callable, List, Tuple, Union
import numpy as np
from tqdm import tqdm
from ..utils.math_utils import (
    compute_orthogonal_projection_matrix,
    project_block_onto_subspace,
    compute_opci
)


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
ImageReader = Callable[[Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]], Union[np.ndarray, Tuple[int, int], None]]
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]
SpectralVectors = Tuple[List[np.ndarray], List[Tuple[int, int]]]


# --------------------------------------------------------------------------------------------
# TGP Function
# --------------------------------------------------------------------------------------------
def target_generation_process(
    image_reader: ImageReader,
    max_targets: int = 10,
    opci_threshold: float = 0.01,
    block_shape: Tuple[int, int] = (512, 512)
) -> SpectralVectors:
    """
    Target Generation Process (TGP).

    Iteratively projects image into orthogonal subspace and extracts new target vectors
    until OPCI falls below a threshold or a max target count is reached.

    Args:
        image_reader (ImageReader): Function to stream image data by window or query shape.
        max_targets (int, optional): Max number of targets to extract. Defaults to 10.
        opci_threshold (float, optional): Stop if OPCI falls below this. Defaults to 0.01.
                                          Higher threshold (e.g. 0.1) creates less pure targets.
                                          Lower threshold (e.g. 0.001) creates more pure targets.
        block_shape (Tuple[int, int], optional): Size of image blocks to process. Defaults to (512, 512).
                                                 More blocks is easier on PC's memory, but slower overall.

    Returns:
        SpectralVectors:
            List of target spectral vectors, and their (row, col) coordinates.
    """
    # Get image reader and block data
    image_shape = image_reader("shape")
    if image_shape is None:
        raise ValueError("image_reader returned None, cannot determine image dimensions.")
    
    # Get block size
    image_height, image_width = image_shape
    block_height, block_width = block_shape

    targets = []
    coords = []

    # Get a small valid sample block to infer band count
    sample_block = image_reader(((0, 0), (1, 1)))
    if not isinstance(sample_block, np.ndarray):
        raise ValueError("[TGP] image_reader did not return a valid data block")

    # At iteration = 0, this means no projection; identity matrix = no filtering.
    num_bands = sample_block.shape[2]
    projection_matrix = np.eye(num_bands, dtype=np.float32)
    
    for iteration in tqdm(range(max_targets), desc="Processing TGP", colour="MAGENTA"):
        # Initalize local-best variables
        max_norm = -np.inf
        best_vector = None
        best_coords = None

        # Iterate through entire image
        for row_off in range(0, image_height, block_height):
            for col_off in range(0, image_width, block_width):
                # Check: block boundary being out-of-bounds
                actual_height = min(block_height, image_height - row_off)
                actual_width = min(block_width, image_width - col_off)

                # Get next block to process
                window = ((row_off, col_off), (actual_height, actual_width))
                block = image_reader(window)
                
                # Check: Block isn't empty or returning window shape
                if not isinstance(block, np.ndarray): 
                    continue
                
                # Project block 
                projected = project_block_onto_subspace(block, projection_matrix)
                
                # Check: projected data is 3D image
                if projected.ndim != 3: raise ValueError(f"Expected 3D block, got shape {projected.shape}")
                
                # Reshpe data and normalize
                reshaped = projected.reshape(-1, projected.shape[2])
                norms = np.linalg.norm(reshaped, axis=1)

                # Find local maximum per-block (i.e. best target per block)
                local_max_idx = np.argmax(norms)
                local_max_val = norms[local_max_idx]

                # Replace target If local target is best found so far
                if local_max_val > max_norm:
                    max_norm = local_max_val
                    best_vector = reshaped[local_max_idx]
                    row_in_block = local_max_idx // actual_width
                    col_in_block = local_max_idx % actual_width
                    best_coords = (row_off + row_in_block, col_off + col_in_block)

        if best_vector is None:
            print(f"[TGP] No more valid blocks to evaluate.")
            break

        if iteration > 0:
            opci_val = compute_opci(projection_matrix, best_vector)
            if opci_val < opci_threshold:
                print(f"[TGP] Stopping: OPCI ({opci_val:.5f}) < threshold ({opci_threshold})")
                break

        targets.append(best_vector)
        coords.append(best_coords)

        # Update projection space
        projection_matrix = compute_orthogonal_projection_matrix(targets)

    return targets, coords

