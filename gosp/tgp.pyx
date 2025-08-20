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

from .math_utils import compute_orthogonal_projection_matrix, compute_opci
from .rastio import MultibandBlockReader
from .parallel import best_target_parallel, Target


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
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]


# --------------------------------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------------------------------
def _make_windows(image_shape: Tuple[int, int], window_shape: Tuple[int, int]):
    """
    Generates all possible windows over an image. Used in _best_target.
    
    Args: 
        image_shape (Tuple[int,int]): (height,width) of entire source image.
        window_shape (Tuple[int,int]): (height,width) of window.
    """
   # Get image and window dimensions
    cdef:
        Py_ssize_t img_height, img_width
        Py_ssize_t win_height, win_width
        Py_ssize_t row_off, col_off
        Py_ssize_t actual_height, actual_width

    img_height = image_shape[0]
    img_width  = image_shape[1]
    win_height = window_shape[0]
    win_width  = window_shape[1]

    windows: List[WindowType] = []

    # Create windows
    for row_off in range(0, img_height, win_height):
        for col_off in range(0, img_width, win_width):
            # Prevent window from out-of-bounds
            actual_height = win_height if win_height < (img_height - row_off) else (img_height - row_off)
            actual_width = win_width if win_width < (img_width - col_off) else (img_width - col_off)
            
            # Create window and append to list
            windows.append( ((<int>row_off, <int>col_off), (<int>actual_height, <int>actual_width)) )
    
    return windows


# --------------------------------------------------------------------------------------------
# TGP Function
# --------------------------------------------------------------------------------------------
def target_generation_process(
    *,
    generated_bands:List[str],
    window_shape:Tuple[int,int],
    max_targets:int,
    opci_threshold:float,        
    max_workers:int|None = None,  # vvv Parallelization parameters vvv
    inflight:int,
    show_progress:bool
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

    # Generate all windows for image 
    windows:list = _make_windows(image_shape, window_shape)


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

    # Check target has same num_bands as input data
    if T0.band_spectrum.shape[0] != num_bands: raise ValueError(f"Band mismatch: discovered {num_bands} bands, candidate has {T0.band_spectrum.shape[0]}")
    
    
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
            opci = 0.0 # prevent NaN fallback to 1.0 
        
        if opci < opci_threshold:
            if show_progress: print("[TGP] opci fell below threshold, no more targets generated..")
            break

        # Accept best_target 
        targets.append(best_target.band_spectrum)
        
    return targets # generated targets; size: [<= max_targets]

