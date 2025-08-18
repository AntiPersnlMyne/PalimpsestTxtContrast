#!/usr/bin/env python3

"""tcp.py: Target Classification Process. Automatically classified pixels into one of N classes found by tgp.py."""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
from typing import List, Sequence, Tuple

# Project modules
from .rastio import MultibandBlockReader, MultibandBlockWriter, window_imread
from .tgp import _make_windows, WindowType  
from ..utils.math_utils import compute_orthogonal_projection_matrix, project_block_onto_subspace
from .parallel import submit_streaming


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Parallel processing
# --------------------------------------------------------------------------------------------
# Worker state reuses a reader
_worker_state: dict[str, object] = {
    "reader": None,     # MultibandBlockReader
    "targets": None,    # (K, B)
    "Pk": None,         # (K, B, B)
}


def _init_tcp_worker(paths:Sequence[str], targets:np.ndarray, Pk:np.ndarray) -> None:
    """Initializer: store immutable arrays in each worker to avoid pickling per task."""
    _worker_state["reader"] = MultibandBlockReader(list(paths))
    _worker_state["targets"] = targets.astype(np.float32, copy=False)
    _worker_state["Pk"] = Pk.astype(np.float32, copy=False)


def _tcp_window(window: WindowType) -> Tuple[WindowType, np.ndarray]:
    """
    Compute per-target OSP responses for a single window.

    Returns:
        Tuple[WindowType, np.ndarray]: (window, scores), where scores are (K-targets, height, width); one band per target.
    """
    reader:MultibandBlockReader = _worker_state["reader"] # type:ignore ;
    targets:np.ndarray = _worker_state["targets"]  # type:ignore ; (K, B)
    Pk:np.ndarray = _worker_state["Pk"]            # type:ignore ; (K, B, B)

    block = reader.read_multiband_block(window).astype(np.float32, copy=False)  # (bands, height, width)
    (_, _), (win_height, win_width) = window
    
    k_targets, _ = targets.shape
    scores = np.empty((k_targets, win_height, win_width), dtype=np.float32)

    # Calculate OSP score per target k
    for k_target in range(k_targets):
        # Nothing to project out if k_targets == 1
        proj = block if k_targets == 1 else project_block_onto_subspace(block, Pk[k_target])  # (bands, height, width)
        scores[k_target] = np.tensordot(targets[k_target], proj, axes=([0], [0])).astype(np.float32, copy=False)

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
        img_height, img_width = reader.image_shape()
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
        






