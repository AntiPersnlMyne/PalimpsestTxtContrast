"""tcp.py: Target Classification Process. Automatically classified pixels into one of N classes found by tgp.py."""

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
from typing import Callable, List, Union, Tuple
import numpy as np
from tqdm import tqdm
from ...python_scripts.utils.math_utils import compute_orthogonal_projection_matrix


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
ImageReader = Callable[[Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]], Union[np.ndarray, Tuple[int, int], None]]
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]
ImageWriter = Callable[[int], Callable[[WindowType, np.ndarray], None]]


# --------------------------------------------------------------------------------------------
# TCP Function
# --------------------------------------------------------------------------------------------
def run_tcp_classification(
    image_reader: ImageReader,
    targets: List[np.ndarray],
    image_writer_factory: ImageWriter,
    block_shape: Tuple[int, int] = (512, 512)
    ) -> None:
    """
    Runs the Target Classification Process (TCP) for each target.

    Args:
        image_reader (ImageReader): Function to stream image blocks.
        targets (List[np.ndarray]): List of target vectors (each shape: [bands]).
        image_writer_factory (Callable): Returns a writer for target index (i.e., per-target writer).
        block_shape (Tuple[int, int], optional): Processing block size. Defaults to (512, 512).

    Returns:
        None
    """
    # Get image reader and block data
    reader = image_reader("shape")
    if reader is None:
        raise ValueError("[TCP] image_reader returned None, cannot determine image dimensions.")
    
    # Get block size
    image_height, image_width = reader
    block_height, block_width = block_shape

    for target_idx, target_vector in enumerate(targets):
        print(f"[TCP] Classifying target {target_idx}...")

        # Create P that nulls all *other* targets
        other_targets = [t for i, t in enumerate(targets) if i != target_idx]
        if other_targets:
            P = compute_orthogonal_projection_matrix(other_targets)
        else:
            P = np.eye(target_vector.shape[0], dtype=np.float32)

        writer = image_writer_factory(target_idx)

        for row_off in tqdm(range(0, image_height, block_height), desc=f"[TCP] Target {target_idx}"):
            for col_off in range(0, image_width, block_width):
                actual_height = min(block_height, image_height - row_off)
                actual_width = min(block_width, image_width - col_off)

                window = ((row_off, col_off), (actual_height, actual_width))
                block = image_reader(window)
                if not isinstance(block, np.ndarray):
                    continue

                projected = block @ P.T  # shape: (H, W, B)
                response = np.tensordot(projected, target_vector, axes=([2], [0]))  # shape: (H, W)

                writer(window, response.astype(np.float32))
