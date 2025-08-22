#!/usr/bin/env python3
# distutils: language=c


"""tgp.pyx: Target Generation Process. Automatically creates N most significant targets in target detection for pixel classification"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np

from typing import List, Tuple
from logging import warn

from ..build.math_utils import compute_orthogonal_projection_matrix, compute_opci
from ..build.rastio import MultibandBlockReader
from ..build.parallel import best_target_parallel


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.2"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
WindowType = np.ndarray


# --------------------------------------------------------------------------------------------
# C Helper Function
# --------------------------------------------------------------------------------------------
cdef int[:,:] _generate_windows_cy(
    int img_height, 
    int img_width, 
    int win_height, 
    int win_width
):
    """
    Generate window offsets and sizes for an image.
    
    Returns:
        windows: int[:, :] memoryview of shape (total_windows, 4)
                 Each row: (row_off, col_off, actual_height, actual_width)
    """
    cdef:
        int n_rows = (img_height + win_height - 1) // win_height
        int n_cols = (img_width + win_width - 1) // win_width
        int total_windows = n_rows * n_cols
        int[:, :] win_mv
        np.ndarray[int, ndim=2] windows = np.empty((total_windows, 4), dtype=np.int32)
    
    win_mv = windows

    cdef int row_idx, col_idx, win_idx
    cdef int row_off, col_off, actual_height, actual_width

    win_idx = 0
    for row_idx in range(n_rows):
        row_off = row_idx * win_height
        actual_height = win_height if row_off + win_height <= img_height else img_height - row_off

        for col_idx in range(n_cols):
            col_off = col_idx * win_width
            actual_width = win_width if col_off + win_width <= img_width else img_width - col_off

            # Fill window valuess
            win_mv[win_idx, 0] = row_off
            win_mv[win_idx, 1] = col_off
            win_mv[win_idx, 2] = actual_height
            win_mv[win_idx, 3] = actual_width

            win_idx += 1

    return win_mv


# --------------------------------------------------------------------------------------------
# TGP Function
# --------------------------------------------------------------------------------------------
def target_generation_process(
    *,
    generated_bands:List[str],
    window_shape:Tuple[int,int],
    max_targets:int,
    opci_threshold:float,        
    max_workers:int|None = None,  # currently unused
    inflight:int,                 # currently unused
    verbose:bool
) -> List[np.ndarray]:
    """
    Target Generation Process (TGP).

    Iteratively projects image into orthogonal subspace and extracts new target vectors
    until OPCI falls below a threshold or a max target count is reached.

    Args:
        generated_bands (List[str]): 
            Path to gen_band_norm.tif i.e. </path/to/gen_band_norm.tif>
        window_shape (Tuple[int,int]):
            Size of tile ("block") of data to process.
        max_targets (int, optional): 
            Max number of targets to extract. Defaults to 10.
        opci_threshold (float, optional): 
            Stop if OPCI of target falls below this. Bigger number = less targets.
            Higher threshold (e.g. 0.1) creates less pure targets.
            Lower threshold (e.g. 0.001) creates more pure targets.

    Returns:
        List[np.ndarray]: List of targets (T0, T1, T2, ...); 
    """

    targets:List[np.ndarray] = []
    
    # Get image shape for window creation
    with MultibandBlockReader(generated_bands) as reader:
        num_bands:int = reader.total_bands
        image_shape:tuple = reader.image_shape

    # ==============================
    # Image size & window dimensions
    # ==============================
    if info: info("[BGP] Getting image dimensions ...")
    with MultibandBlockReader(input_image_paths) as reader:
        img_height, img_width = reader.image_shape
        # Small 1x1 test block to calc number of output bands 
        test_win = np.array([0, 0, 1, 1], dtype=np.int32)
        test_block = np.array(reader.read_multiband_block(test_win), copy=True)

    # ============================================================
    # Generate windows
    # ============================================================
    if verb: info("Generating windows ...")
    # Generate array of window dimensions (num_windows, 4) 
    cdef int[:,:] win_mv = _generate_windows_cy(img_height, img_width, win_height, win_width)
    total_windows = win_mv.shape[0]


    # =================================
    # Find the first target T0 and OPCI
    # =================================
    T0 = best_target_parallel(
        paths=generated_bands, 
        windows=windows, 
        p_matrix=None,
        max_workers=max_workers, 
        inflight=inflight, 
        show_progress=show_progress
    )
    targets.append(T0.band_spectrum)

    # # Check target has same num_bands as input data
    # if T0.band_spectrum.shape[0] != num_bands: raise ValueError(f"Band mismatch: discovered {num_bands} bands, candidate has {T0.band_spectrum.shape[0]}")
    
    
    # =========================================
    # Iterate for subsequence targets (t1...tk)
    # =========================================
    # Iterate until max_targets or OPCI falls below threshold 
    while len(targets) < max_targets: 
        # a. Compute the orthogonal projection matrix for the current set of targets
        p_matrix = compute_orthogonal_projection_matrix(targets)

        # b. Find the best target in the projected subspace
        best_target = best_target_parallel(
            paths=generated_bands, windows=windows, p_matrix=p_matrix,
            max_workers=max_workers, inflight=inflight, show_progress=show_progress,
        )

        # Evaluate OPCI - checks for early stopping
        opci = compute_opci(p_matrix, best_target.band_spectrum)
        if not np.isfinite(opci): 
            warn("[tcp] OCPI reached a value of infinity, now exiting tgp.")
            opci = 0.0 # prevent NaN, fallback to 1.0 
        
        if opci < opci_threshold:
            if show_progress: print("[TGP] opci fell below threshold, no more targets generated..")
            break

        # Accept best_target 
        targets.append(best_target.band_spectrum)
        
    return targets # generated targets; size: [<= max_targets]

