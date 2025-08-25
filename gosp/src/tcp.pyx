#!/usr/bin/env python3
# distutils: language=c

"""tcp.pyx: Target Classification Process. Automatically classified pixels into one of N classes found by tgp.py."""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
cimport numpy as np
from cython.parallel import prange

from typing import List, Sequence, Tuple
from libc.math cimport fmaf
from logging import info
import tqdm as tqdm

from gosp.build.rastio import (
    MultibandBlockReader, 
    MultibandBlockWriter  
)
from gosp.build.math_utils import(
    compute_orthogonal_complement_matrix,
    project_block_onto_subspace
)


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.2.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
ctypedef np.float32_t float32_t


# --------------------------------------------------------------------------------------------
# C Functions
# --------------------------------------------------------------------------------------------
cdef inline int _compute_scores_inner(
    float32_t[:, :, :] proj_mv, # (bands, h, w)
    const float32_t[:] targ_mv, # (bands,)
    float32_t[:, :] out_mv      # (h, w) 
) nogil:
    """
    Compute dot(targets[k], proj[:, r, c]) for each pixel (r,c) and store into out_mv.
    
    Args: 
        proj_mv (float64):
            Memory view of projection matrix
        targ_mv (const float64):
            Memory view of targets (to be classified)
        out_mv (float64):
            Memory view of output slice (height,width)
    """
    cdef:
        Py_ssize_t bands = proj_mv.shape[0]
        Py_ssize_t height = proj_mv.shape[1]
        Py_ssize_t width = proj_mv.shape[2]
        Py_ssize_t b, row, col
        double sum
    
    for row in range(height):
        for col in range(width):
            sum = 0.0
            # accumulate dot product over bands
            for b in range(bands):
                sum = fmaf(targ_mv[b], proj_mv[b, row, col], sum)
            out_mv[row, col] = sum


cdef int[:,:] _generate_windows(
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
        np.ndarray[np.int32_t, ndim=2] windows = np.empty((total_windows, 4), dtype=np.int32)
        int[:, :] win_mv = windows
        int row_idx, col_idx, win_idx = 0
        int row_off, col_off, actual_h, actual_w

    for row_idx in range(n_rows):
        row_off = row_idx * win_height
        actual_h = win_height if row_off + win_height <= img_height else img_height - row_off
        for col_idx in range(n_cols):
            col_off = col_idx * win_width
            actual_w = win_width if col_off + win_width <= img_width else img_width - col_off

            win_mv[win_idx, 0] = row_off
            win_mv[win_idx, 1] = col_off
            win_mv[win_idx, 2] = actual_h
            win_mv[win_idx, 3] = actual_w
            win_idx += 1

    return win_mv



def target_classification_process(
    *, # requirement of keyword args
    generated_bands: Sequence[str],
    window_shape: Tuple[int, int],
    targets: List[np.ndarray],
    output_dir: str,
    scores_filename: str = "targets_classified.tif",
    verbose:bool = True,
) -> None:
    """
    Compute per-target OSP scores across the image without parallel processing.

    Args:
        generated_bands (Sequence[str]): Paths to the generated bands.
        window_shape (Tuple[int,int]): Tile (block) size.
        targets (List[np.ndarray]): List of target spectra (from TGP).
        output_dir (str): Directory to write output TIFF.
        scores_filename (str): Output filename.
        verbose (bool): Enable progress/info messages.

    Returns:
        None: Writes output to disk.
    """
    cdef:
        # Window dimensions and memoryview
        int win_height = <int> window_shape[0]
        int win_width  = <int> window_shape[1]
        int img_height, img_width, img_bands
        int[:,:] win_mv
        # Iterators
        int i, total_windows, k, k_target
        Py_ssize_t k_targets
        # memoryviews used inside nogil
        float32_t[:, :] out_mv
        float32_t[:] targ_mv
        float32_t[:, :, :] proj_mv

    if len(targets) == 0:
        raise ValueError("[TCP] No targets provided (TGP output is empty).")

    # ==============================
    # Image size & window dimensions
    # ==============================
    if verbose: info("[TCP] Reading image dimensions ...")
    with MultibandBlockReader(generated_bands) as reader:
        img_height, img_width = reader.image_shape
        img_bands = reader.total_bands


    # ==============================================
    # Prepare targets and projection matrices
    # ==============================================
    if verbose: info("[TCP] Preparing targets ...")
    k_targets = len(targets)
    targets_arr = np.stack([np.asarray(t, dtype=np.float32) for t in targets], axis=0)
    if not targets_arr.flags['C_CONTIGUOUS']:
        targets_arr = np.ascontiguousarray(targets_arr, dtype=np.float32)


    # ==============================================
    # Build Pk matrices (K, B, B)
    # ==============================================
    Pk_list = []
    # Iterate through 'k' targets
    for k in range(k_targets):
        if k_targets > 1:
            other_targets = [targets[j] for j in range(k_targets) if j != k]
            Pk_list.append(compute_orthogonal_complement_matrix(other_targets).astype(np.float32))
        else:
            Pk_list.append(np.eye(img_bands, dtype=np.float32))
    Pk_arr = np.stack(Pk_list, axis=0)
    # Check C Contiguous
    if not Pk_arr.flags['C_CONTIGUOUS']:
        Pk_arr = np.ascontiguousarray(Pk_arr, dtype=np.float32)
        

    # ==============================================
    # Generate windows
    # ==============================================
    if verbose: info("[TCP] Generating windows ...")
    win_mv = _generate_windows(img_height, img_width, win_height, win_width)
    total_windows = win_mv.shape[0]


    # ==============================================
    # Initialize writer
    # ==============================================
    # Create memoryviews
    cdef float32_t[:,:,:,:] proj_blocks_mv # (K, B, h, w)
    cdef float32_t[:,:,:] scores_mv  # (K, h, w)
    cdef float32_t[:,:] targets_mv = targets_arr  # (K, B)
    # Open dataset reader/writer
    with MultibandBlockWriter(
        output_dir=output_dir,
        output_image_shape=(img_height, img_width),
        output_image_name=scores_filename,
        window_shape=window_shape,
        num_bands=k_targets,
        output_datatype=np.float32,
    ) as writer:
        with MultibandBlockReader(generated_bands) as reader:
            # Loop over windows sequentially
            for i in tqdm(range(total_windows), desc="[TCP] Classifying pixels", unit="win", colour="YELLOW"):
                row_off = win_mv[i, 0]
                col_off = win_mv[i, 1]
                win_h   = win_mv[i, 2]
                win_w   = win_mv[i, 3]


                # Read block
                win = np.asarray([row_off, col_off, win_h, win_w], dtype=np.intc)
                block = reader.read_multiband_block(win)
                # Check C contigouous
                if not block.flags['C_CONTIGUOUS']:
                    block = np.ascontiguousarray(block, dtype=np.float32)


                # Pre-allocate proj_blocks: (K, bands, h, w)
                proj_blocks = np.empty((k_targets, img_bands, win_h, win_w), dtype=np.float32)

                # Compute projected blocks for each target
                if k_targets == 1:
                    # If only one target, projection is identity -> copy block
                    proj_blocks[0] = block
                else:
                    for k in range(k_targets):
                        # project_block_onto_subspace expects (bands,h,w) and a P matrix (B,B)
                        tmp_proj = project_block_onto_subspace(block, Pk_arr[k])
                        if tmp_proj.dtype != np.float32:
                            tmp_proj = tmp_proj.astype(np.float32, copy=False)
                        if not tmp_proj.flags['C_CONTIGUOUS']:
                            tmp_proj = np.ascontiguousarray(tmp_proj, dtype=np.float32)
                        # Add block to block list
                        proj_blocks[k] = tmp_proj


                # prepare output score array (K, h, w)
                scores = np.empty((k_targets, win_h, win_w), dtype=np.float32)


                # Create memoryviews (must be done before nogil)
                proj_blocks_mv = proj_blocks
                scores_mv = scores
                targets_mv = targets_mv


                # Parallel over targets; each target computes its own (h,w) plane
                for k_target in prange(k_targets, nogil=True, schedule='dynamic'):
                    # proj_mv: (bands, h, w) view for this target
                    _compute_scores_inner(
                        proj_blocks_mv[k_target, :, :, :],   # (B,h,w)
                        targets_mv[k_target, :],       # (B,)
                        scores_mv[k_target, :, :]            # (h,w)
                    )



                # Write scores onto output band for this window
                writer.write_block(window=win, block=scores)


    return None
