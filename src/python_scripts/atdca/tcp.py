"""tcp.py: Target Classification Process. Automatically classified pixels into one of N classes found by tgp.py."""

from __future__ import annotations

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
import rasterio
from rasterio.windows import Window
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

# Project modules (adjust import paths to your tree)
from .rastio import MultibandBlockReader, MultibandBlockWriter
from .tgp import _make_windows, WindowType  # reuse window enumerator
from ..utils.math_utils import compute_orthogonal_projection_matrix, project_block_onto_subspace
from ..utils.parallel import submit_streaming


# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
def _compute_pk(targets: List[np.ndarray]) -> List[np.ndarray]:
    """
    For each target k, build projection matrix P_k that projects out all other (i.e. [0,k-1]) targets .

    targets
    """
    K = len(targets)
    Pk: List[np.ndarray] = []
    for k in range(K):
        others = [targets[j] for j in range(K) if j != k]
        if not others:
            # Only one target → identity: P x = x
            P = np.eye(targets[0].shape[0], dtype=np.float32)
        else:
            P = compute_orthogonal_projection_matrix(others).astype(np.float32)
        Pk.append(P)
    return Pk


# --------------------------------------------------------------------------------------------
# TCP Function
# --------------------------------------------------------------------------------------------
def target_classification_process(
    image_reader: ImageReader,
    targets: List[np.ndarray],
    image_writer_factory: ImageWriter,
    block_shape: Tuple[int, int] = (512, 512)
    ) -> None:
    """
    Runs the Target Classification Process (TCP).

    Args:
        image_reader (ImageReader): Function to stream image blocks.
        targets (List[np.ndarray]): List of target vectors (each shape: [bands]).
        image_writer_factory (Callable): Returns an image writer for a target index (i.e., per-target writer).
        block_shape (Tuple[int, int], optional): Processing block size. Defaults to (512, 512).

    Returns:
        None
    """
    # Get image reader and block data
    reader = image_reader("window_shape")
    if reader is None:
        raise ValueError("[TCP] image_reader returned None, cannot determine image dimensions.")
    
    # Get block size
    image_height, image_width = reader
    block_height, block_width = block_shape

    for target_idx, target_vector in enumerate(targets):
        # Create P that projects out ("nulls") all other targets
        other_targets = [t for i, t in enumerate(targets) if i != target_idx]
        if other_targets:
            P_target_mat = compute_orthogonal_projection_matrix(other_targets)
        else:
            P_target_mat = np.eye(target_vector.shape[0], dtype=np.float32)

        # Get image writer
        writer = image_writer_factory(target_idx)

        for row_off in tqdm(range(0, image_height, block_height), desc=f"[TCP] Target {target_idx}", colour="YELLOW"):
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

                # Project block onto P targets
                projected = block @ P_target_mat.T  # shape: (H, W, B)
                response = np.tensordot(projected, target_vector, axes=([2], [0]))  # shape: (H, W)

                writer(window, response.astype(np.float32))
