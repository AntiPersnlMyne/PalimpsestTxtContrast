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
import importlib
from os import getpid



# --------------------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------------------
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]  # ((row_off, col_off), (height, width))


class SupportsWriteBlock(Protocol):
    """
    Protocol for writer objects that expose a `write_block` method.

    Your writer (e.g., MultibandBlockWriter) should implement:
        write_block(window: WindowType, block: np.ndarray) -> None
    """

    def write_block(self, window: WindowType, block: np.ndarray) -> None:
        ...



# --------------------------------------------------------------------------------------
# Pass 1: read -> create synthetic bands
# --------------------------------------------------------------------------------------

# Per-process state for generation
_gen_state: dict[str, Any] = {
    "paths": None,       # List[str]
    "use_sqrt": False,
    "use_log": False,
    "bands_fn": None,   # callable(image_block, use_sqrt, use_log) -> np.ndarray (bands, h, w)
}


def _init_generate_worker(input_paths: List[str], func_module: str, func_name: str, use_sqrt: bool, use_log: bool) -> None:
    """Initializer for Pass 1 workers.

    Args:
        input_paths (List[str]): Paths to input rasters. May be one multiband file or many single-band files.
        func_module (str): Relative path to the module (file) that the function is from. Formatted same as Python's relative imports e.g., python_scripts.atdca.bgp.
        func_name (str): Function name for the band-generation function; avoids pickling callables.
        use_sqrt (bool): Flag forwarded to the band_generation function.
        use_log (bool): Flag forwarded to the band_generation function.
    """
    _gen_state["paths"] = list(input_paths)
    _gen_state["use_sqrt"] = bool(use_sqrt)
    _gen_state["use_log"] = bool(use_log)
    mod = importlib.import_module(func_module)
    _gen_state["bands_fn"] = getattr(mod, func_name)


def _read_input_window(paths: Sequence[str], window: WindowType) -> np.ndarray:
    """Read a window from input paths into shape (bands, h, w) without relying on project-specific readers."""
    (row_off, col_off), (h, w) = window
    if len(paths) == 1:
        with rasterio.open(paths[0], "r") as src:
            return src.read(window=Window(col_off, row_off, w, h))#type:ignore
    else:
        # stack band 1 from each single-band file
        out = None
        for i, p in enumerate(paths):
            with rasterio.open(p, "r") as src:
                band = src.read(1, window=Window(col_off, row_off, w, h))#type:ignore
                if out is None:
                    out = np.empty((len(paths), h, w), dtype=band.dtype)
                out[i] = band
        assert out is not None
        return out


def _generate_window(window: WindowType) -> Tuple[WindowType, np.ndarray, np.ndarray, np.ndarray]:
    """Worker: read inputs, create synthetic bands, return block and per-band stats.

    Returns
    -------
    (window, new_bands, mins, maxs)
        new_bands: (bands, h, w) synthetic output
        mins/maxs: (bands,) local stats for aggregation in parent
    """
    paths = _gen_state["paths"]
    use_sqrt = _gen_state["use_sqrt"]
    use_log = _gen_state["use_log"]
    bands_fn = _gen_state["bands_fn"]

    block = _read_input_window(paths, window)
    new_bands = bands_fn(block, use_sqrt, use_log)
    mins = new_bands.min(axis=(1, 2)).astype(np.float32)
    maxs = new_bands.max(axis=(1, 2)).astype(np.float32)
    return window, new_bands.astype(np.float32), mins, maxs


def parallel_generate_streaming(
    *,
    input_paths:Sequence[str],
    windows:Iterable[WindowType],
    writer:SupportsWriteBlock,
    func_module:str,
    func_name:str,
    use_sqrt:bool,
    use_log:bool,
    max_workers:int|None = None,
    inflight:int = 2
) -> np.ndarray:
    """
    Pass 1 streaming generation with bounded memory and per-band stats aggregation.

    Args:
        input_paths (List[str]): Path to input rasters. Accepts one? multiband many single-band files.
        windows (List[WindowType]): Total list of windows to process.
        writer (SupportsWriteBlock): Open writer where synthetic blocks will be written. Uses `MultibandBlockWriter`.
        func_module (str): Import path for the band-generation function; avoids pickling callables.
        func_name (str): Function name for the band-generation function; avoids pickling callables.
        use_sqrt (bool): Flag forwarded to the band_generation function.
        use_log (bool): Flag forwarded to the band_generation function.
        max_workers (int, optional): Number of processes to run in parallel. If None, lets program decide. Defaults to None.
        inflight (int): 
            At most `inflight * max_workers` tasks will be in flight ("worked on") at once.
            Lower to reduce RAM; raise to improve throughput.
            Defaults to 2.

    Returns:
        np.ndarray: Array of band statistics.
            Array of shape (2, bands) where [0] = global mins, [1] = global maxs. Used in bgp for Pass 2.
    """
    # Discover band count from the first completed task, and grow stats arrays accordingly
    global_mins: np.ndarray|None = None
    global_maxs: np.ndarray|None = None

    windows_iter = iter(windows)
    windows_list: List[WindowType]|None = None
    total = None

    try:
        windows_list = list(windows_iter)
        total = len(windows_list)
        windows_iter = iter(windows_list)
    except TypeError:
        pass

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_generate_worker,
        initargs=(list(input_paths), func_module, func_name, use_sqrt, use_log),
    ) as ex:
        pending: set[Future] = set()
        target_inflight = max(1, (max_workers or 1) * max(1, inflight))
        for _ in range(target_inflight):
            try:
                w = next(windows_iter)
            except StopIteration:
                break
            pending.add(ex.submit(_generate_window, w))


        prog_bar = tqdm(total=total, desc="[BGP] First pass - create", unit="win", colour="CYAN")

        try:
            while pending:
                done = next(as_completed(pending))
                pending.remove(done)
                window, new_bands, mins, maxs = done.result()

                # Initialize global stats once we know band count
                if global_mins is None or global_maxs is None:
                    global_mins = mins.copy()
                    global_maxs = maxs.copy()
                else:
                    # Aggregate per band
                    global_mins = np.minimum(global_mins, mins)
                    global_maxs = np.maximum(global_maxs, maxs)

                writer.write_block(window=window, block=new_bands)
                prog_bar.update(1)

                try:
                    w = next(windows_iter)
                except StopIteration:
                    w = None
                if w is not None:
                    pending.add(ex.submit(_generate_window, w))
        except Exception as e:
            # Cancel all future tasks
            for fututre_task in pending:
                fututre_task.cancel()
            raise Exception(f"[parallel] Error during tasks execution:\n{e}")
        finally:
            prog_bar.close()

    # Return global min/max per band alone row-major array
    assert global_mins is not None and global_maxs is not None, "No windows processed"
    return np.stack([global_mins, global_maxs], axis=0)




# --------------------------------------------------------------------------------------
# Pass 2: Normalize data
# --------------------------------------------------------------------------------------
# Global (per-process) state initialized via initializer to avoid pickling large arrays per task
_worker_state: dict[str, Any] = {
    "path": None,          # str
    "band_mins": None,     # np.ndarray, shape (bands,)
    "band_maxs": None,     # np.ndarray, shape (bands,)
}


def _init_normalize_worker(unorm_path: str, band_mins: np.ndarray, band_maxs: np.ndarray) -> None:
    """
    Initializer specifically for normalizer function.

    Stores immutable references to inputs in a process-local dict so each task
    doesn't have to receive large arrays via pickling.

    Args:
        unorm_path (str): Path to unnormalized dataset.
        band_mins (np.ndarray): Array, each index is minimum value per band.
        band_maxs (np.ndarray): Array, each index is maximum value per band.
    """
    _worker_state["path"] = unorm_path
    _worker_state["band_mins"] = np.asarray(band_mins, dtype=np.float32)
    _worker_state["band_maxs"] = np.asarray(band_maxs, dtype=np.float32)


def _normalize_window(window: WindowType) -> Tuple[WindowType, np.ndarray]:
    """
    Worker function: read one window from the multiband TIFF, normalize per band, return the block.

    Returns:
        Tuple[WindowType, np.ndarray]: norm_block - (window, norm_block) - has shape (bands, height, width), dtype float32.
    """

        
    (row_off, col_off), (h, w) = window
    path:str = _worker_state["path"]
    mins:np.ndarray = _worker_state["band_mins"]
    maxs:np.ndarray = _worker_state["band_maxs"]

    # Re-open the dataset inside the worker process. No shared handles.
    with rasterio.open(path, "r") as src:
        block = src.read(window=Window(col_off, row_off, w, h))  #type:ignore

    # Vectorized per-band normalization: (x - min) / (max - min)
    denom = np.maximum(maxs - mins, 1e-8).astype(np.float32)
    norm = (block.astype(np.float32) - mins[:, None, None]) / denom[:, None, None]
    np.clip(norm, 0.0, 1.0, out=norm)

    return window, norm


def parallel_normalize(
    *,
    unorm_path:str,
    windows:Iterable[WindowType],
    band_mins:np.ndarray,
    band_maxs:np.ndarray,
    writer:SupportsWriteBlock,
    max_workers:Optional[int] = None,
    inflight:int = 2,
) -> None:
    """Normalize windows in parallel with a bounded, streaming pattern.

    Parameters
    ----------
    unorm_path (str): Path to the unnormalized multiband GeoTIFF written in Pass 2.
    windows (Iterable[WindowType]): Sequence of windows to process. Order is not required.
    band_mins (np.ndarray): Array of band maximums in band-major order.
    band_maxs (np.ndarray): Array of band maximums in band-major order.
    writer (SupportsWriteBlock): Writer object that will receive each normalized block.
    max_workers (int, optional): Number of worker processes. If None, defaults to `os.cpu_count()` (i.e. all of them). Defaults to None.
    inflight (int): 
        At most `inflight * max_workers` tasks will be in flight ("worked on") at once.
        Lower to reduce RAM; raise to improve throughput.
        Defaults to 2.
    """
    windows_iter = iter(windows)

    # Wrap iterator in a list to know total count for progress.
    windows_list:List[WindowType]|None = None
    total = None
    
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
        initargs=(unorm_path, band_mins, band_maxs),
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
        prog_bar = tqdm(total=total, desc="[BGP] Second pass - normalize", unit="win", colour="MAGENTA")

        # Drain as tasks complete; submit new ones to keep inflight bounded
        try:
            while pending:
                done = next(as_completed(pending))
                pending.remove(done)

                window, norm_block = done.result()  # may raise from worker
                writer.write_block(window=window, block=norm_block)
                prog_bar.update(1)

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
            prog_bar.close()

