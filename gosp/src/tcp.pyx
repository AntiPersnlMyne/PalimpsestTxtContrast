#!/usr/bin/env python3
# distutils: language=c

"""tcp.pyx: Target Classification Process. Automatically classified pixels into one of N classes found by tgp.py."""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
cimport numpy as np

from typing import List, Sequence, Tuple
from libc.math cimport fmaf
from logging import info
from tqdm import tqdm

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
ctypedef np.float32_t float_t


# --------------------------------------------------------------------------------------------
# C Helper Function
# --------------------------------------------------------------------------------------------
cdef inline int _compute_scores_inner(
    float_t[:, :, :] proj_mv,           # (bands, h, w)
    const float_t[:] targ_mv,           # (bands,)
    float_t[:, :] out_mv                # (h, w) 
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
        double acc
    
    for row in range(height):
        for col in range(width):
            acc = 0.0
            # accumulate dot product over bands
            for b in range(bands):
                acc = fmaf(targ_mv[b], proj_mv[b, row, col], acc)
            out_mv[row, col] = acc


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




def target_classification_process(
    *, # requirement of keyword args
    generated_bands: Sequence[str],
    window_shape: Tuple[int, int],
    targets: List[np.ndarray],
    output_dir: str,
    scores_filename: str = "targets_classified.tif",
    max_workers: int|None = None,
    inflight: int = 2,
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
        int win_height = <int> window_shape[0]
        int win_width  = <int> window_shape[1]
        int img_height, img_width, img_bands
        int[:,:] win_mv
        int i, total_windows, k, row_off, col_off, win_h, win_w
        Py_ssize_t k_targets

        float_t[:, :, :] proj_mv
        float_t[:]       targ_mv
        float_t[:, :]     out_mv
        float_t[:, :, :] scores_mv
        float_t[:, :] targets_mv

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
    targets_arr = np.stack([t.astype(np.float32) for t in targets], axis=0)  # (K,B)
    Pk_list = []

    for k in range(k_targets):
        if k_targets > 1:
            # Exclude k-th target
            other_targets = [targets[j] for j in range(k_targets) if j != k]
            Pk_list.append(compute_orthogonal_complement_matrix(other_targets).astype(np.float32))
        else:
            # Single target -> identity
            Pk_list.append(np.eye(img_bands, dtype=np.float32))
    Pk_arr = np.stack(Pk_list, axis=0)  # (K,B,B)

    # Ensure contiguous memory
    if not targets_arr.flags['C_CONTIGUOUS']:
        targets_arr = np.ascontiguousarray(targets_arr, dtype=np.float32)
    if not Pk_arr.flags['C_CONTIGUOUS']:
        Pk_arr = np.ascontiguousarray(Pk_arr, dtype=np.float32)

    # Targets memoryview
    targets_mv = targets_arr  # (K,B)


    # ==============================================
    # Generate windows
    # ==============================================
    if verbose: info("[TCP] Generating windows ...")
    win_mv = _generate_windows_cy(img_height, img_width, win_height, win_width)
    total_windows = win_mv.shape[0]


    # ==============================================
    # Initialize writer
    # ==============================================
    with MultibandBlockWriter(
        output_dir=output_dir,
        output_image_shape=(img_height, img_width),
        output_image_name=scores_filename,
        window_shape=window_shape,
        num_bands=k_targets,
        output_datatype=np.float32,
    ) as writer:

        # Loop over windows sequentially
        for i in tqdm(range(total_windows), desc="[TCP] Classifying pixels", unit="win", colour="WHITE"):
            row_off = win_mv[i, 0]
            col_off = win_mv[i, 1]
            win_h   = win_mv[i, 2]
            win_w   = win_mv[i, 3]

            # Read block
            win = np.asarray([row_off, col_off, win_h, win_h])
            block = reader.read_multiband_block(win)#.astype(np.float32, copy=False)
            # Check C condigouous
            if not block.flags['C_CONTIGUOUS']:
                block = np.ascontiguousarray(block, dtype=np.float32)

            # Prepare score array: (K, h, w)
            scores = np.empty((k_targets, win_h, win_w), dtype=np.float32)
            scores_mv = scores

            # Compute per-target projected dot-products
            for k in range(k_targets):
                if k_targets == 1:
                    proj_block = block
                else:
                    proj_block = project_block_onto_subspace(block, Pk_arr[k])
                    if not proj_block.flags['C_CONTIGUOUS']:
                        proj_block = np.ascontiguousarray(proj_block, dtype=np.float32)

                proj_mv = proj_block
                targ_mv = targets_mv[k]
                out_mv  = scores_mv[k]

                with nogil:
                    _compute_scores_inner(proj_mv, targ_mv, out_mv)

            # Write scores for this window
            writer.write_block(window=win, block=scores)



