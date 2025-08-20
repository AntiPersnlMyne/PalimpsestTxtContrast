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
from libc.math cimport fmaf # fused multiply-add-function

# Project modules
from .rastio import MultibandBlockReader, MultibandBlockWriter
from .tgp import _make_windows, WindowType  
from .math_utils import compute_orthogonal_projection_matrix, project_block_onto_subspace
from .parallel import submit_streaming


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
ctypedef np.float32_t float_t


# --------------------------------------------------------------------------------------------
# Parallel processing
# --------------------------------------------------------------------------------------------
# Worker state reuses a reader
_worker_state: dict[str, object] = {
    "reader": None,     # MultibandBlockReader
    "targets": None,    # (targets, bands)
    "Pk": None,         # (targets, bands, bands)
}


def _init_tcp_worker(paths:Sequence[str], targets:np.ndarray, Pk:np.ndarray) -> None:
    """Initializer: store immutable arrays in each worker to avoid pickling per task."""
    _worker_state["reader"]  = MultibandBlockReader(list(paths))
    _worker_state["targets"] = targets.astype(np.float32, copy=False)
    _worker_state["Pk"]      = Pk.astype(np.float32, copy=False)



cdef inline int _compute_scores_inner(
    float_t[:, :, :] proj_mv,           # (bands, h, w)
    const float_t[:] targ_mv,           # (bands,)
    float_t[:, :] out_mv                # (h, w) 
) nogil:
    """
    Compute dot(targets[k], proj[:, r, c]) for each pixel (r,c) and store into out_mv.
    
    Parameters
    ----------
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



def _tcp_window(window: WindowType) -> Tuple[WindowType, np.ndarray]:
    """
    Compute per-target OSP responses for a single window.

    Returns:
        Tuple[WindowType, np.ndarray]: (window, scores), where scores are (K-targets, height, width); one band per target.
    """
    # Grab worker-shared objects (reader, targets, Pk)
    reader: MultibandBlockReader = _worker_state["reader"] 
    targets: np.ndarray = _worker_state["targets"]  # (k-targets, bands)
    Pk: np.ndarray = _worker_state["Pk"]            # (k-targets, bands, bands)

    # Read block: (bands, height, width)
    block = reader.read_multiband_block(window).astype(np.float32, copy=False)
    (_, _), (win_height, win_width) = window

    # Preallocate scores: shape (K, h, w)
    k_targets = targets.shape[0]
    scores = np.empty((k_targets, win_height, win_width), dtype=np.float32)

    # Ensure targets and block are contiguous ; memoryview for fast access in nogil loops
    if not targets.flags['C_CONTIGUOUS']:
        targets = np.ascontiguousarray(targets, dtype=np.float32)

    cdef float_t[:, :] targets_mv = targets  # (K, B)
    if not block.flags['C_CONTIGUOUS']:
        block = np.ascontiguousarray(block, dtype=np.float32)

    # Create memoryviews
    # targets_mv[k] is a 1D view; cast to contiguous float[:] for nogil
    # 2D view for the output slice (h, w)
    cdef float_t[:, :, :] proj_mv 
    cdef float_t[:] targ_mv
    cdef float_t[:, :] out_mv

    # For each target, compute projected block and then per-pixel dot product
    cdef Py_ssize_t k
    for k in range(k_targets):
        # If only one target, projecting out nothing saves a call; keep same semantics
        if k_targets == 1:
            proj_block = block
        else:
            # Project block into subspace, excluding target k
            proj_block = project_block_onto_subspace(block, Pk[k])

        # Ensure proj_block is float32 contiguous ndarray
        if not proj_block.flags['C_CONTIGUOUS'] or proj_block.dtype != np.float32:
            proj_block = np.ascontiguousarray(proj_block, dtype=np.float32)

        proj_mv = proj_block
        targ_mv = targets_mv[k]
        out_mv = scores[k]

        # Call the inner C kernel without the GIL
        with nogil:
            _compute_scores_inner(proj_mv, targ_mv, out_mv)

    return window, scores



def target_classification_process(
    *, # requirement of keyword args
    generated_bands: Sequence[str],
    window_shape: Tuple[int, int],
    targets: List[np.ndarray],
    output_dir: str,
    scores_filename: str = "targets_classified.tif",
    max_workers: int|None = None,
    inflight: int = 2,
    show_progress: bool = True,
) -> None:
    """
    Compute per-target OSP scores across the image and optionally a label map.


    Parallel safety:
      - Workers read-only; parent writes. Worker functions are top-level.
      - Tasks are (window,) 1-tuples to avoid pickling issues on Windows.
    """
    if len(targets) == 0:
        raise ValueError("[TCP] No targets provided (TGP output is empty).")

    # Discover geometry and confirm target dimensionality matches
    with MultibandBlockReader(list(generated_bands)) as reader:
        img_height, img_width = reader.image_shape
        sample = reader.read_multiband_block(((0,0), (2,2))) # sample 2x2 block
        band_count = int(sample.shape[0])

    for i, target in enumerate(targets):
        if target.shape != (band_count,):
            raise ValueError(f"[TCP] Target {i} shape {target.shape}, expected ({band_count},)")

    # Precompute per-target projectors and pack arrays for per-task access
    Pk_list = [compute_orthogonal_projection_matrix([targets[j] for j in range(len(targets)) if j != k]).astype(np.float32)
               if len(targets) > 1 else np.eye(band_count, dtype=np.float32)
               for k in range(len(targets))]
    
    targets_arr = np.stack([target.astype(np.float32) for target in targets], axis=0)  # (K,B)
    Pk_arr = np.stack(Pk_list, axis=0).astype(np.float32)                              # (K,B,B)
    k_targets = targets_arr.shape[0]

    # Score and label writer
    scores_writer = MultibandBlockWriter(
        output_dir=output_dir,
        output_image_shape=(img_height, img_width),
        output_image_name=scores_filename,
        window_shape=window_shape,
        num_bands=k_targets,
        output_datatype=np.float32,
    )

    # Build all windows
    windows = _make_windows((img_height, img_width), window_shape)

    # Score ("label") all pixels in image
    with scores_writer as writer:
        def _consume(result: Tuple[WindowType, np.ndarray]) -> None:
            # Consumer executes in parent; safe to write here
            window, scores = result                           # (K,h,w)
            writer.write_block(window=window, block=scores)

        tasks = [(window,) for window in windows]  # one-arg tasks: (window,)
        submit_streaming(
            worker=_tcp_window,
            initializer=_init_tcp_worker,
            initargs=(list(generated_bands), targets_arr, Pk_arr),
            tasks=tasks,
            consumer=_consume,
            max_workers=max_workers,
            inflight=inflight,
            prog_bar_label="[TCP] Labeling targets",
            prog_bar_color="WHITE",
            show_progress=show_progress,
        )
        






