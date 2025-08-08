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
import numpy as np

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Sequence
from numba import njit

from ..utils.math_utils import (
    compute_orthogonal_projection_matrix,
    project_block_onto_subspace,
    compute_opci,
    block_l2_norms
)
from ..atdca.rastio import MultibandBlockReader
from ..utils.parallel import scan_for_max_parallel



# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]
ImageBlock = np.ndarray
ImageShape = Tuple[int, int]

@dataclass
class Target:
    value: float
    row: int
    col: int
    band_spectrum: np.ndarray  # shape (bands,)



# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
def _best_target(
    *,
    paths: List[str],
    windows: Iterable[WindowType],
    p_matrix:np.ndarray|None
) -> Target:
    """
    Find global argmax of ||x|| or ||P x|| across all pixels.

    Args:
        paths (list[str]): One multiband file or many single-band files for dataset.
        windows (Iterable[WindowType]): List of windows to iterate over.
        p_matrix (np.ndarray | None): Projection matrix. If not provided (None),
            assumes first target i.e. no P matrix created yet.
    Returns:
        Target (dataclass): Best target found in dataset. Object's data: value, row, col, band_spectrum.
    """
    
    # Initalize best_target output
    best_target:Target = Target(0,0,0,np.empty(0))

    with MultibandBlockReader(list(paths)) as reader:        
        for window in windows:
            # project block - can be optimized by not checking every iteration?
            block = reader.read_multiband_block(window)  # (bands, h, w)
            if p_matrix is not None: block = project_block_onto_subspace(block, p_matrix)
            
            # Compute L2 norm and returns tile
            norms = block_l2_norms(block)  # shape: (height,width)
            
            # Find pixel within tile with largest norm
            max_px_idx = int(np.argmax(norms))
            max_px_val = float(norms.flat[max_px_idx])

            # Convert the flat index back into row/col coordinates
            (row_off, col_off), (win_height, win_width) = window
            block_row, block_col = divmod(max_px_idx, win_width)
            
            # Convert tile-local coordinates to full image coordinates 
            img_row, img_col = row_off + block_row, col_off + block_col
            
             # Extract all bands (bands,:,:) from the best pixel
            bands = block[:, block_row, block_col].astype(np.float32)
           
            # Update best target
            target = Target(max_px_val, img_row, img_col, bands)
            if target.value > best_target.value:
                best_target = target

    # Check some target was found and return
    assert best_target, "No pixels scanned"
    return best_target 


def _make_windows(image_shape: Tuple[int, int], window_shape: Tuple[int, int]):
    """
    Generates all possible windows over an image. Used in _best_target.
    
    Args: 
        image_shape (Tuple[int,int]): (height,width) of entire source image.
        window_shape (Tuple[int,int]): (height,width) of window.
    """
    # Get image and window dimensions
    img_height, img_width = image_shape
    win_height, win_width = window_shape
    windows:List[WindowType] = []
    
    for row_off in range(0, img_height, win_height):
        for col_off in range(0, img_width, win_width):
            
            # Prevent window from out-of-bounds
            actual_height = min(win_height, img_height - row_off)
            actual_width = min(win_width, img_width - col_off)
            
            # Create window and append to list
            windows.append( ((row_off, col_off), (actual_height, actual_width)) )
    
    return windows



# --------------------------------------------------------------------------------------------
# TGP Function
# --------------------------------------------------------------------------------------------
def target_generation_process(
    *,
    generated_bands:List[str],
    window_shape:Tuple[int,int],
    max_targets:int = 10,
    ocpi_threshold:float = 0.01,       
    use_parallel:bool = False,  # vvv Parallelization parameters vvv
    max_workers:int|None = None,
    inflight:int,
    show_progress:bool
    ) -> List[np.ndarray]:
    """
    Target Generation Process (TGP).

    Iteratively projects image into orthogonal subspace and extracts new target vectors
    until OPCI falls below a threshold or a max target count is reached.

    Args:
        generated_bands (Sequence[str]): Either </path/to/gen_band_norm.tif> (multiband) or a list of single-band files.
        window_shape (Tuple[int,int]): Size of tile ("block") of data to process.
        max_targets (int, optional): Max number of targets to extract. Defaults to 10.
        opci_threshold (float, optional): Stop if OCPI of target falls below this. Defaults to 0.01.
            Higher threshold (e.g. 0.1) creates less pure targets.
            Lower threshold (e.g. 0.001) creates more pure targets.
        use_parallel (bool): Process data serially (slow) or in parallel (fast). 
            If True, processes faster but uses more RAM.
            Defaults to False.

    Returns:
        List[np.ndarray]: List of targets (t0, t1, t2, ...); 
    """
    
    # Get dims for windows
    # Validate band-major layout
    with MultibandBlockReader(generated_bands) as reader:
        # Determine final dimensions from small test block (10, 10)
        im_height, im_width = reader.image_shape()  
        dummy_block = reader.read_multiband_block(  
            ((0, 0), (10,10))
        )
        num_bands = int(dummy_block.shape[0])
        if num_bands < 1: raise ValueError("Input image must have at least 1 band")

    # Get all possible windows 
    windows = _make_windows((im_height, im_width), window_shape)

    # Calculate initial target
    if use_parallel:
        t0 = scan_for_max_parallel(
            paths=generated_bands, windows=windows, p_matrix=None,
            max_workers=max_workers, inflight=inflight, show_progress=show_progress
        )
    else:
        t0 = _best_target(paths=generated_bands, windows=windows, p_matrix=None)    
    
    # Check target has same num_bands as input data
    if t0.band_spectrum.shape[0] != num_bands: raise ValueError(f"Band mismatch: discovered {num_bands} bands, candidate has {t0.band_spectrum.shape[0]}")
    
    targets:List[np.ndarray] = [t0.band_spectrum]

    # Compute new p_marix with first target
    p_matrix = compute_orthogonal_projection_matrix(targets).astype(np.float32)  # (bands,bands)

    for _ in range(1, max_targets):
        # Find next candidate in orthogonal space
        if use_parallel:
            best_target = scan_for_max_parallel(
                paths=generated_bands, windows=windows, p_matrix=p_matrix,
                max_workers=max_workers, inflight=inflight, show_progress=show_progress,
            )
        else:
            best_target = _best_target(paths=generated_bands, windows=windows, p_matrix=p_matrix)

        # Evaluate OPCI metric - determines stopping criteria
        opci = compute_opci(p_matrix, best_target.band_spectrum)
        if opci < ocpi_threshold:
            break

        # Accept best_target and update projection
        targets.append(best_target.band_spectrum)
        p_matrix = compute_orthogonal_projection_matrix(targets).astype(np.float32)

    return targets # generated targets; size: [<= max_targets]

