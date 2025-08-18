#!/usr/bin/env python3

"""skip_bgp.py: Allows user to skip the BGP (i.e. skipping the g in gosp, to the ("ATDCA") ) to only process input files"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from __future__ import annotations
import numpy as np
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import List, Tuple, Iterable, Any, Sequence
from tqdm import tqdm

from .rastio import MultibandBlockReader, MultibandBlockWriter

WindowType = Tuple[Tuple[int, int], Tuple[int, int]]


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



# Worker state
_pt_state: dict[str, Any] = {
    "reader": None,
    "dtype": np.float32,
}


def _init_pt_worker(paths: List[str], dtype: str) -> None:
    _pt_state["reader"] = MultibandBlockReader(list(paths))
    _pt_state["dtype"] = np.dtype(dtype)


def _read_original_chunk(windows_chunk: List[WindowType]) -> List[Tuple[WindowType, np.ndarray]]:
    reader:MultibandBlockReader = _pt_state["reader"]  # type: ignore
    out_dtype:np.dtype = _pt_state["dtype"]            # type: ignore
    band_stack:List[Tuple[WindowType, np.ndarray]] = []
    for window in windows_chunk:
        block = reader.read_multiband_block(window).astype(out_dtype, copy=False)
        band_stack.append((window, block))
    return band_stack


def _chunked(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
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
    windows: List[WindowType] = []
    for r in range(0, src_h, win_h):
        for c in range(0, src_w, win_w):
            h = min(win_h, src_h - r)
            w = min(win_w, src_w - c)
            windows.append(((r, c), (h, w)))
    return windows


def _total_band_count(paths: Sequence[str]) -> int:
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
    output_dtype:... = np.float32,
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
    if not input_image_paths: raise ValueError("[skipBGP] Empty input list")

    # Shape & band count
    reader = MultibandBlockReader(input_image_paths)
    src_height, src_width = reader.image_shape()
    num_bands = _total_band_count(input_image_paths)

    # Dtype
    out_dtype = np.dtype(output_dtype) 

    # Windows
    win_height, win_width = window_shape
    windows = _build_windows(src_height, src_width, win_height, win_width)
    chunks = list(_chunked(windows, chunk_size))

    # Writer
    with MultibandBlockWriter(
        output_dir=output_dir,
        output_image_shape=(src_height, src_width),
        output_image_name="gen_band_norm.tif", # Keeps it consistent with BGP; yes this is bad practice
        window_shape=window_shape,
        output_datatype=out_dtype,
        num_bands=num_bands,
    ) as writer:

        # If exactly one multiband input, optional fast path via rasterio copy is possible,
        # but currently keeping streaming for consistent tiling/profile from MultibandBlockWriter.
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_pt_worker,
            initargs=(list(input_image_paths), out_dtype.name),
        ) as ex:
            pending: set[Future] = set()
            target_inflight = max(1, (max_workers or 1) * max(1, inflight))

            chunk_iter = iter(chunks)
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
                        # safety: ensure band axis matches writer count
                        if block.shape[0] != num_bands:
                            raise RuntimeError(f"[passthrough] Band count drift: got {block.shape[0]}, expected {num_bands}")
                        writer.write_block(window=window, block=block)
                        
                        if prog_bar is not None: prog_bar.update(1)
                    
                    try: chunk = next(chunk_iter)
                    except StopIteration: chunk = None
                    
                    if chunk is not None: pending.add(ex.submit(_read_original_chunk, chunk))
                    
                    
            finally:
                if prog_bar is not None: prog_bar.close()


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
