"""tcp.py: Target Classification Process. Automatically classified pixels into one of N classes found by tgp.py."""

from __future__ import annotations

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
from typing import List, Sequence, Tuple
from contextlib import nullcontext

# Project modules
from .rastio import MultibandBlockReader, MultibandBlockWriter, window_imread
from .tgp import _make_windows, WindowType  
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
    k_targets = len(targets)
    Pk: List[np.ndarray] = []
    
    # Iterate targets and create P_matrix(s)
    for k_target in range(k_targets):
        targets = [targets[k] for k in range(k_targets) if k != k_target]
        if not targets:
            # No targets to project 
            P_matrix = np.eye(targets[0].shape[0], dtype=np.float32)
        else:
            # Project targets
            P_matrix = compute_orthogonal_projection_matrix(targets).astype(np.float32)
        Pk.append(P_matrix)
        
    return Pk



# ============================================================================================
# Parallel processing
# ============================================================================================
_worker_state: dict[str, object] = {
    "paths": None,       # Sequence[str]
    "targets": None,     # np.ndarray, (K-targets, bands)
    "Pk": None,          # np.ndarray, (K-targets, bands, bands)
}


def _init_tcp_worker(paths:Sequence[str], targets:np.ndarray, Pk:np.ndarray) -> None:
    """Initializer: store immutable arrays in each worker to avoid pickling per task."""
    _worker_state["paths"] = list(paths)
    _worker_state["targets"] = targets.astype(np.float32)
    _worker_state["Pk"] = Pk.astype(np.float32)


def _tcp_window(window: WindowType) -> Tuple[WindowType, np.ndarray]:
    """Compute per-target OSP responses for a single window.

    Returns:
        Tuple[WindowType, np.ndarray]: (window, scores), where scores are (K-targets, height, width); one band per target.
    """
    paths:Sequence[str] = _worker_state["paths"]   # type:ignore ;
    targets:np.ndarray = _worker_state["targets"]  # type:ignore ; (K, B)
    Pk:np.ndarray = _worker_state["Pk"]            # type:ignore ; (K, B, B)

    (row_off, col_off), (height, width) = window
    block = window_imread(paths, window).astype(np.float32)  # (bands, height, width)

    k_targets, bands = targets.shape
    scores = np.empty((k_targets, height, width), dtype=np.float32)

    # OSP score per target k: d_k(x) = t_k^T (P_k x)
    # For each target, project using P_k then compute t_k^T (P_k x)
    for k_target in range(k_targets):
        # Nothing to project out
        if k_targets == 1:
            proj = block
        # Project out previous k targets
        else:
            proj = project_block_onto_subspace(block, Pk[k_target])  # (bands, height, width)
        # Fast, multidimensional dot product over band axis
        # tensordot: (B,) * (B,height,width) over band axis -> (height,width) 
        scores[k_target] = np.tensordot(targets[k_target], proj, axes=([0], [0])).astype(np.float32)

    return window, scores



# --------------------------------------------------------------------------------------------
# Target Classification Process Function
# --------------------------------------------------------------------------------------------
def target_classification_process(
    *, # requirement of keyword args
    generated_bands: Sequence[str],
    window_shape: Tuple[int, int],
    targets: List[np.ndarray],
    output_dir: str,
    scores_filename: str = "tcp_scores.tif",
    write_labels: bool = True,
    labels_filename: str = "tcp_labels.tif",
    use_parallel: bool = True,
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
        sample = reader.read_multiband_block(((0,0), (5,5))) # sample 5x5 block
        band_count = int(sample.shape[0])

    for i, target in enumerate(targets):
        if target.shape != (band_count,):
            raise ValueError(f"[TCP] Target {i} shape {target.shape}, expected ({band_count},)")

    # Precompute per-target projectors and pack arrays for per-task access
    Pk_list = _compute_pk(targets)
    targets_arr = np.stack([target.astype(np.float32) for target in targets], axis=0)  # (K,B)
    Pk_arr = np.stack(Pk_list, axis=0).astype(np.float32)                              # (K,B,B)
    k_targets = targets_arr.shape[0]

    # Score and label writer
    scores_writer = MultibandBlockWriter(
        output_path=output_dir,
        output_image_shape=(img_height, img_width),
        output_image_name=scores_filename,
        num_bands=k_targets,
        output_datatype=np.float32,
    )
    labels_writer = (
        MultibandBlockWriter(
            output_path=output_dir,
            output_image_shape=(img_height, img_width),
            output_image_name=labels_filename,
            num_bands=1,
            output_datatype=np.uint16,
        ) if write_labels 
        else None
    )

    # Build all windows
    windows = _make_windows((img_height, img_width), window_shape)

    if use_parallel:
        # Parallel streaming: workers compute; parent writes immediately.
        with scores_writer as s_writer, (labels_writer if labels_writer else nullcontext()) as l_writer:
            def _consume(result: Tuple[WindowType, np.ndarray]) -> None:
                # Consumer executes in parent; safe to write here
                window, scores = result                           # (K,h,w)
                s_writer.write_block(window=window, block=scores)
                if l_writer is not None:
                    # Argmax across K scores → 1-band label map
                    labels = np.argmax(scores, axis=0, keepdims=True).astype(np.uint16)
                    l_writer.write_block(window=window, block=labels)

            tasks = [(window,) for window in windows]  # one-arg tasks: (window,)
            submit_streaming(
                worker=_tcp_window,
                initializer=_init_tcp_worker,
                initargs=(list(generated_bands), targets_arr, Pk_arr),
                tasks=tasks,
                consumer=_consume,
                max_workers=max_workers,
                inflight=inflight,
                prog_bar_label="[TCP] Classify",
                prog_bar_color="BLACK",
                show_progress=show_progress,
            )
            
    # Serial path (versus parallel) 
    else:
        with scores_writer as s_writer, (labels_writer if labels_writer else nullcontext()) as l_writer:
            for window in windows:
                # Read block of data
                block = window_imread(generated_bands, window).astype(np.float32)  # (B,h,w)
                (_, _), (win_height, win_width) = window
                
                # Compute scores
                scores = np.empty((k_targets, win_height, win_width), dtype=np.float32)
                
                # Iterate targets and project
                for k_target in range(k_targets):
                    proj_matrix = block if  k_targets == 1 else project_block_onto_subspace(block, Pk_arr[k_target])
                    scores[k_target] = np.tensordot(targets_arr[k_target], proj_matrix, axes=([0], [0]))
                s_writer.write_block(window=window, block=scores)
                if l_writer is not None:
                    labels = np.argmax(scores, axis=0, keepdims=True).astype(np.uint16)
                    l_writer.write_block(window=window, block=labels)








