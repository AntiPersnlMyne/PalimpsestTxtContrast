"""parallel.py: Parallelization ("multiprocessing") wrapper API. Compatible with Numba."""

from __future__ import annotations # has to be up here for some reason

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Protocol, Any, Callable
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm



# --------------------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------------------
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]  # ((row_off, col_off), (height, width))


class SupportsWriteBlock(Protocol):
    """Protocol for writer objects that expose a `write_block` method.

    Your writer (e.g., MultibandBlockWriter) should implement:
        write_block(window: WindowType, block: np.ndarray) -> None
    """

    def write_block(self, window: WindowType, block: np.ndarray) -> None:
        ...


# --------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------
# Global (per-process) state initialized via initializer to avoid pickling large arrays per task
_worker_state: dict[str, Any] = {
    "path": None,          # str
    "band_mins": None,     # np.ndarray, shape (bands,)
    "band_maxs": None,     # np.ndarray, shape (bands,)
    "clip01": True,        # bool
}


def _init_normalize_worker(unorm_path: str, band_mins: np.ndarray, band_maxs: np.ndarray, clip01: bool) -> None:
    """Initializer for worker processes.

    Stores immutable references to inputs in a process-local dict so each task
    doesn't have to receive large arrays via pickling.
    """
    _worker_state["path"] = unorm_path
    # ensure contiguous float32 for predictable bandwidth
    _worker_state["band_mins"] = np.asarray(band_mins, dtype=np.float32)
    _worker_state["band_maxs"] = np.asarray(band_maxs, dtype=np.float32)
    _worker_state["clip01"] = bool(clip01)


def _normalize_window(window: WindowType) -> Tuple[WindowType, np.ndarray]:
    """
    Worker function: read one window from the multiband TIFF, normalize per band, return the block.

    Returns:
        (window, norm_block): Tuple[WindowType, np.ndarray]: norm_block has shape (bands, height, width), dtype float32.
    """
    (row_off, col_off), (h, w) = window
    path:str = _worker_state["path"]
    mins:np.ndarray = _worker_state["band_mins"]
    maxs:np.ndarray = _worker_state["band_maxs"]
    clip01: bool = _worker_state["clip01"]

    # Re-open the dataset inside the worker process. No shared handles.
    with rasterio.open(path, "r") as src:
        block = src.read(window=Window(col_off, row_off, w, h))  #type:ignore

    # Vectorized per-band normalization: (x - min) / (max - min)
    denom = np.maximum(maxs - mins, 1e-8).astype(np.float32)
    norm = (block.astype(np.float32) - mins[:, None, None]) / denom[:, None, None]
    if clip01:
        np.clip(norm, 0.0, 1.0, out=norm)

    return window, norm


# --------------------------------------------------------------------------------------
# Parallelization Functions
# --------------------------------------------------------------------------------------

def parallel_normalize(
    *,
    unorm_path: str,
    windows: Iterable[WindowType],
    band_mins: np.ndarray,
    band_maxs: np.ndarray,
    writer: SupportsWriteBlock,
    max_workers: Optional[int] = None,
    inflight: int = 2,
    show_progress: bool = True,
    clip01: bool = True,
) -> None:
    """Normalize windows in parallel with a bounded, streaming pattern.

    Parameters
    ----------
    unorm_path : str
        Path to the unnormalized multiband GeoTIFF written in Pass 1.
    windows : Iterable[WindowType]
        Sequence/generator of windows to process. Order is not required.
    band_mins, band_maxs : np.ndarray
        Global per-band statistics, shape (bands,). Must match the output band count.
    writer : SupportsWriteBlock
        Open writer object that will receive each normalized block.
    max_workers : int, optional
        # of worker processes. Defaults to `os.cpu_count()` if None.
    inflight : int, default 2
        At most `inflight * max_workers` tasks will be in flight at once.
        Lower to reduce RAM; raise to improve throughput.
    show_progress : bool, default True
        If True, displays a progress bar via tqdm (if available).
    clip01 : bool, default True
        Clip normalized values into [0,1].
    """
    windows_iter = iter(windows)

    # Wrap iterator in a list to know total count for progress. If `windows` is a generator
    # and you don't want to materialize it, set show_progress=False for unknown total.
    windows_list: Optional[List[WindowType]] = None
    total = None
    if show_progress:
        try:
            windows_list = list(windows_iter)
            total = len(windows_list)
            windows_iter = iter(windows_list)
        except TypeError:
            # Not materializable; proceed without total
            pass

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_normalize_worker,
        initargs=(unorm_path, band_mins, band_maxs, clip01),
    ) as ex:
        pending: set[Future] = set()

        # Prime the pump: submit up to inflight * workers tasks
        target_inflight = max(1, (max_workers or 1) * max(1, inflight))
        for _ in range(target_inflight):
            try:
                w = next(windows_iter)
            except StopIteration:
                break
            pending.add(ex.submit(_normalize_window, w))

        # Progress bar setup
        pbar = None
        if show_progress and tqdm is not None:
            pbar = tqdm(total=total, desc="Pass 2: normalize", unit="win")

        # Drain as tasks complete; submit new ones to keep inflight bounded
        try:
            while pending:
                done = next(as_completed(pending))
                pending.remove(done)

                window, norm_block = done.result()  # may raise from worker
                writer.write_block(window=window, block=norm_block)
                if pbar is not None:
                    pbar.update(1)

                try:
                    w = next(windows_iter)
                except StopIteration:
                    w = None
                if w is not None:
                    pending.add(ex.submit(_normalize_window, w))
        except Exception:
            # Cancel all outstanding tasks on error for faster teardown
            for fut in pending:
                fut.cancel()
            raise
        finally:
            if pbar is not None:
                pbar.close()


