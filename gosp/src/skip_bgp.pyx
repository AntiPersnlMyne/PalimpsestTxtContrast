#!/usr/bin/env python3
# distutils: language=c

"""
skip_bgp.pyx: 
Allows user to skip the BGP to only process input files (Cythonized).
This module preserves the original API but uses typed variables and memoryviews
where appropriate for efficiency.
"""

from __future__ import annotations

import numpy as np
cimport numpy as cnp

from typing import List, Tuple, Iterable, Any, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from tqdm import tqdm
import rasterio

from .build.rastio import MultibandBlockReader, MultibandBlockWriter

# NumPy typedefs for memoryviews
cnp.import_array()
ctypedef cnp.float32_t float32_t

# Re-exported type alias
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]

# Worker state (module-level; each process will set its own via initializer)
_pt_state: dict = {
    "reader": None,
    "dtype": np.float32,
}


def _init_pt_worker(paths: List[str], dtype_name: str) -> None:
    """
    Initializer for worker processes: open a MultibandBlockReader and store dtype.
    Kept as a Python function for picklability by ProcessPoolExecutor.
    """
    # store a reader instance in module-level state for worker
    _pt_state["reader"] = MultibandBlockReader(list(paths))
    _pt_state["dtype"] = np.dtype(dtype_name)


def _read_original_chunk(windows_chunk: List[WindowType]) -> List[Tuple[WindowType, np.ndarray]]:
    """
    Worker: read the original multiband blocks for a chunk of windows.
    Returns list of (window, block) where block is a numpy.ndarray dtype `out_dtype`.
    """
    # Access worker-global variables (set in _init_pt_worker)
    reader = _pt_state["reader"]  # type: ignore
    out_dtype = _pt_state["dtype"]  # type: ignore

    results: List[Tuple[WindowType, np.ndarray]] = []
    # Local references to accelerate attribute lookup
    read_multiband_block = reader.read_multiband_block

    for window in windows_chunk:
        # Read directly as float32 contiguous (reader already returns float32 contiguous arrays)
        block = read_multiband_block(window)
        # Ensure dtype matches requested output dtype without unnecessary copy
        if block.dtype != out_dtype:
            block = block.astype(out_dtype, copy=False)
        else:
            # Guarantee contiguous (should be, but be defensive)
            if not block.flags['C_CONTIGUOUS']:
                block = np.ascontiguousarray(block)
        results.append((window, block))
    return results


def _chunked(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """
    Yield lists of up to `size` items from `iterable`.
    Kept Pythonic for clarity; this is not a hot path relative to numeric work.
    """
    it = iter(iterable)
    while True:
        chunk: List[Any] = []
        for _ in range(size):
            try:
                chunk.append(next(it))
            except StopIteration:
                break
        if not chunk:
            return
        yield chunk


def _build_windows(src_h: int, src_w: int, win_h: int, win_w: int) -> List[WindowType]:
    """
    Build window list from source image dims and window dims.
    Implemented with typed local ints for speed.
    """
    cdef int r, c
    windows: List[WindowType] = []
    for r in range(0, src_h, win_h):
        for c in range(0, src_w, win_w):
            h = win_h if win_h <= (src_h - r) else (src_h - r)
            w = win_w if win_w <= (src_w - c) else (src_w - c)
            windows.append(((r, c), (h, w)))
    return windows


def _total_band_count(paths: Sequence[str]) -> int:
    """
    Count total number of bands across a list of raster files.
    """
    total = 0
    for p in paths:
        with rasterio.open(p, "r") as s:
            total += int(s.count)
    if total <= 0:
        raise ValueError("[passthrough] No bands discovered in inputs")
    return total


def write_original_multiband(
    *,
    input_image_paths: List[str],
    output_dir: str,
    window_shape: Tuple[int, int] = (512, 512),
    output_dtype: Any = np.float32,
    max_workers: int | None = None,
    inflight: int = 2,
    chunk_size: int = 4,
    show_progress: bool = True,
) -> None:
    """
    Create a single multiband GeoTIFF that contains ONLY the original input bands,
    stacked in input order, windowed and streamed (no synthetic bands, no normalization).

    - Handles mixed inputs (single- and multi-band).
    - Uses MultibandBlockReader/Writer to keep memory bounded.
    - Band order: for each file in input order, include all its bands in band order.

    Parameters
    ----------
    input_image_paths (List[str]):
        Paths to input rasters.
    output_dir (str):
        Directory for the output TIFF.
    output_name (str):
        Output filename.
    window_shape Tuple[int, int]:
        Block size (h, w).
    output_dtype (np.dtype):
        If None, auto-select a safe common dtype (prefers float32).
    max_workers (int | None):
        Process workers, None = all cores.
    inflight (int):
        At most inflight * max_workers chunks in flight.
    chunk_size (int):
        Windows per task; higher reduces overhead, increases RAM.
    show_progress (bool):
        Display progress bars.
    """
    if not input_image_paths:
        raise ValueError("[skipBGP] Empty input list")

    # Setup reader to discover shape
    reader = MultibandBlockReader(input_image_paths)
    try:
        src_height, src_width = reader.image_shape()
    finally:
        # we don't close reader here because worker processes re-open their own readers;
        # keep local reader live only for geometry extraction
        pass

    num_bands = _total_band_count(input_image_paths)

    # Output dtype as numpy dtype (stable name for passing to workers)
    out_dtype = np.dtype(output_dtype)

    # Prepare windows and chunks
    win_height, win_width = window_shape
    windows = _build_windows(src_height, src_width, win_height, win_width)
    chunks = list(_chunked(windows, chunk_size))

    # Create writer and stream writes
    output_name = "gen_band_norm.tif"  # preserved name for compatibility
    with MultibandBlockWriter(
        output_dir=output_dir,
        output_image_shape=(src_height, src_width),
        output_image_name=output_name,
        window_shape=window_shape,
        output_datatype=out_dtype,
        num_bands=num_bands,
    ) as writer:

        # Launch worker pool: each worker will run _init_pt_worker to create its own reader
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_pt_worker,
            initargs=(list(input_image_paths), out_dtype.name),
        ) as ex:
            pending: set[Future] = set()
            target_inflight = max(1, (max_workers or 1) * max(1, inflight))

            chunk_iter = iter(chunks)
            # Submit initially up to target_inflight tasks
            for _ in range(target_inflight):
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                pending.add(ex.submit(_read_original_chunk, chunk))

            prog_bar = tqdm(total=len(windows), desc="[skipBGP] Passthrough - original bands", unit="win", colour="CYAN") if show_progress else None

            try:
                while pending:
                    done = next(as_completed(pending))
                    pending.remove(done)
                    results = done.result()  # list[(window, block)]

                    for window, block in results:
                        # Safety: ensure band axis matches writer count
                        if block.shape[0] != num_bands:
                            raise RuntimeError(f"[passthrough] Band count drift: got {block.shape[0]}, expected {num_bands}")

                        # Ensure contiguous float32 before writing
                        if block.dtype != np.float32 or not block.flags['C_CONTIGUOUS']:
                            # Convert in parent process to avoid sending non-contiguous arrays to rasterio
                            block = np.ascontiguousarray(block, dtype=np.float32)

                        writer.write_block(window=window, block=block)

                        if prog_bar is not None:
                            prog_bar.update(1)

                    # Submit next chunk if available
                    try:
                        chunk = next(chunk_iter)
                    except StopIteration:
                        chunk = None

                    if chunk is not None:
                        pending.add(ex.submit(_read_original_chunk, chunk))

            finally:
                if prog_bar is not None:
                    prog_bar.close()


def copy_single_multiband_fast(src_path: str, dst_path: str) -> None:
    """
    Fast copy for the trivial case: one multiband file to new multiband file.
    Skips windowing; preserves profile exactly (assumed file can fit in memory).
    """
    with rasterio.open(src_path, "r") as src:
        profile = src.profile.copy()
        with rasterio.open(dst_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                data = src.read(window=window)
                dst.write(data, window=window)
