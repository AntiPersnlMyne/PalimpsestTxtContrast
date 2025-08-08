"""parallel.py: Parallelization ("multiprocessing") wrapper API. 

Usage:
- Pass 1 (create synthetic bands): `parallel_generate_streaming(...)`
- Pass 2 (normalize bands):        `parallel_normalize_streaming(...)`
- Generic advanced API:            `submit_streaming(...)`

Notes
-----
- This module uses **streaming, chunked** submission to keep memory bounded.
- Workers re-open rasters locally (safe across processes).
- Parent process writes results immediately to avoid concurrent writes.
- Avoids sending large numpy arrays over IPC; sends only small window tuples.
"""

from __future__ import annotations  # must be first

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "2.0.1" 
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development"  # "Prototype", "Development", "Production"

# --------------------------------------------------------------------------------------------
# Imports & thread oversubscription guards (safe defaults)
# --------------------------------------------------------------------------------------------
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Protocol, Any, Callable
from tqdm import tqdm
from rasterio.windows import Window

import numpy as np
import rasterio
import importlib



# --------------------------------------------------------------------------------------
# Window Data Structure-like-things
# --------------------------------------------------------------------------------------
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]  # ((row_off, col_off), (height, width))


class SupportsWriteBlock(Protocol):
    """Protocol for writer objects that expose a `write_block` method.

    Your writer (e.g., MultibandBlockWriter) should implement:
        write_block(window: WindowType, block: np.ndarray) -> None
    """

    def write_block(self, window: WindowType, block: np.ndarray) -> None:  # pragma: no cover (protocol)
        ...


def _chunked(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Yield lists of up to `size` items from `iterable`.
    Coarsens submission to reduce scheduling/pickling overhead.
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


# ======================================================================================
# Pass 1: read -> create synthetic bands (generation)
# ======================================================================================
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
        input_paths: Paths to input rasters. One multiband or many single-band files.
        func_module: Absolute module path, e.g. "python_scripts.atdca.bgp".
        func_name: Top-level function name to call, e.g. "_create_bands_from_block".
        use_sqrt, use_log: Flags forwarded to the band-generation function.
    """
    _gen_state["paths"] = list(input_paths)
    _gen_state["use_sqrt"] = bool(use_sqrt)
    _gen_state["use_log"] = bool(use_log)
    mod = importlib.import_module(func_module)
    _gen_state["bands_fn"] = getattr(mod, func_name)


def _read_input_window(paths: Sequence[str], window: WindowType) -> np.ndarray:
    """Read a window from input paths into shape (bands, h, w) without project-specific readers."""
    (row_off, col_off), (h, w) = window
    if len(paths) == 1:
        with rasterio.open(paths[0], "r") as src:
            return src.read(window=Window(col_off, row_off, w, h)) #type:ignore
    else:
        out = None
        for i, p in enumerate(paths):
            with rasterio.open(p, "r") as src:
                band = src.read(1, window=Window(col_off, row_off, w, h)) #type:ignore
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

    block = _read_input_window(paths, window).astype(np.float32)
    new_bands = bands_fn(block, use_sqrt, use_log).astype(np.float32)
    mins = new_bands.min(axis=(1, 2)).astype(np.float32)
    maxs = new_bands.max(axis=(1, 2)).astype(np.float32)
    return window, new_bands, mins, maxs


def _generate_windows_chunk(windows_chunk: List[WindowType]) -> List[Tuple[WindowType, np.ndarray, np.ndarray, np.ndarray]]:
    """Worker: read inputs, create synthetic bands for a chunk of windows."""
    paths = _gen_state["paths"]
    use_sqrt = _gen_state["use_sqrt"]
    use_log = _gen_state["use_log"]
    bands_fn = _gen_state["bands_fn"]

    out: List[Tuple[WindowType, np.ndarray, np.ndarray, np.ndarray]] = []
    for window in windows_chunk:
        block = _read_input_window(paths, window).astype(np.float32)
        new_bands = bands_fn(block, use_sqrt, use_log).astype(np.float32)
        mins = new_bands.min(axis=(1, 2)).astype(np.float32)
        maxs = new_bands.max(axis=(1, 2)).astype(np.float32)
        out.append((window, new_bands, mins, maxs))
    return out


def parallel_generate_streaming(
    *,
    input_paths: Sequence[str],
    windows: Iterable[WindowType],
    writer: SupportsWriteBlock,
    func_module: str,
    func_name: str,
    use_sqrt: bool,
    use_log: bool,
    max_workers: int | None = None,
    inflight: int = 2,
    chunk_size: int = 4,
    show_progress: bool = True,
    desc: str = "[BGP] First pass - create",
) -> np.ndarray:
    """Pass 1 streaming generation with bounded memory and per-band stats aggregation.

    Returns
    -------
    band_stats : np.ndarray
        Shape (2, bands) where [0] = global mins, [1] = global maxs.
    """
    windows_list = list(windows)
    chunks = list(_chunked(windows_list, chunk_size))

    global_mins: np.ndarray | None = None
    global_maxs: np.ndarray | None = None

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_generate_worker,
        initargs=(list(input_paths), func_module, func_name, use_sqrt, use_log),
    ) as ex:
        pending: set[Future] = set()
        target_inflight = max(1, (max_workers or 1) * max(1, inflight))

        chunk_iter = iter(chunks)
        for _ in range(target_inflight):
            try:
                c = next(chunk_iter)
            except StopIteration:
                break
            pending.add(ex.submit(_generate_windows_chunk, c))

        pbar = tqdm(total=len(windows_list), desc=desc, unit="win", colour="CYAN") if show_progress else None

        try:
            while pending:
                done = next(as_completed(pending))
                pending.remove(done)
                results = done.result()  # list of (window, new_bands, mins, maxs)

                for window, new_bands, mins, maxs in results:
                    writer.write_block(window=window, block=new_bands)
                    global_mins = np.minimum(global_mins, mins) if global_mins is not None else mins.copy()
                    global_maxs = np.maximum(global_maxs, maxs) if global_maxs is not None else maxs.copy()
                    if pbar is not None:
                        pbar.update(1)

                try:
                    c = next(chunk_iter)
                except StopIteration:
                    c = None
                if c is not None:
                    pending.add(ex.submit(_generate_windows_chunk, c))
        except Exception as e:
            for f in pending: f.cancel()
            raise Exception(f"[parallel] Error during Pass 1:\n{e}")
        finally:
            if pbar is not None:
                pbar.close()

    assert global_mins is not None and global_maxs is not None, "No windows processed"
    return np.stack([global_mins, global_maxs], axis=0)


# ======================================================================================
# Pass 2: normalization (uses unnormalized TIFF produced in Pass 1)
# ======================================================================================
_worker_state: dict[str, Any] = {
    "path": None,          # str
    "band_mins": None,     # np.ndarray, shape (bands,)
    "band_maxs": None,     # np.ndarray, shape (bands,)
    "clip01": True,        # bool
}


def _init_normalize_worker(unorm_path: str, band_mins: np.ndarray, band_maxs: np.ndarray, clip01: bool) -> None:
    """Initializer for normalization workers (Pass 2)."""
    _worker_state["path"] = unorm_path
    _worker_state["band_mins"] = np.asarray(band_mins, dtype=np.float32)
    _worker_state["band_maxs"] = np.asarray(band_maxs, dtype=np.float32)
    _worker_state["clip01"] = bool(clip01)

    # Light warmup (no disk I/O) to pre-allocate and ensure BLAS pools are quiet
    b = _worker_state["band_mins"].shape[0]
    dummy = np.zeros((b, 4, 4), dtype=np.float32)
    denom = np.maximum(_worker_state["band_maxs"] - _worker_state["band_mins"], 1e-8)
    _ = (dummy - _worker_state["band_mins"][..., None, None]) / denom[..., None, None]


def _normalize_windows_chunk(windows_chunk: List[WindowType]) -> List[Tuple[WindowType, np.ndarray]]:
    """Worker: read a chunk of windows and return normalized blocks for each.

    Returns list[(window, norm_block)] where norm_block has shape (bands, h, w), float32.
    """
    # print(f"[worker] pid={getpid()} processing {len(windows_chunk)} windows")
    path: str = _worker_state["path"]
    mins: np.ndarray = _worker_state["band_mins"]
    maxs: np.ndarray = _worker_state["band_maxs"]
    clip01: bool = _worker_state["clip01"]

    denom = np.maximum(maxs - mins, 1e-8).astype(np.float32)
    out: List[Tuple[WindowType, np.ndarray]] = []

    with rasterio.open(path, "r") as src:
        for window in windows_chunk:
            (row_off, col_off), (h, w) = window
            block = src.read(window=Window(col_off, row_off, w, h)).astype(np.float32) #type:ignore
            norm = (block - mins[:, None, None]) / denom[:, None, None]
            if clip01:
                np.clip(norm, 0.0, 1.0, out=norm)
            out.append((window, norm))

    return out


def parallel_normalize_streaming(
    *,
    unorm_path: str,
    windows: Iterable[WindowType],
    band_mins: np.ndarray,
    band_maxs: np.ndarray,
    writer: SupportsWriteBlock,
    max_workers: int | None = None,
    inflight: int = 2,
    chunk_size: int = 4,
    show_progress: bool = True,
    clip01: bool = True,
) -> None:
    """Normalize windows in parallel with a bounded, streaming, **chunked** pattern.

    This function does **not** compute mins/maxs (they come from Pass 1).
    It simply reads, normalizes, and streams blocks to the parent writer.
    """
    windows_list = list(windows)
    chunks = list(_chunked(windows_list, chunk_size))

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_normalize_worker,
        initargs=(unorm_path, band_mins, band_maxs, clip01),
    ) as ex:
        pending: set[Future] = set()
        target_inflight = max(1, (max_workers or 1) * max(1, inflight))

        chunk_iter = iter(chunks)
        for _ in range(target_inflight):
            try:
                chunk = next(chunk_iter)
            except StopIteration:
                break
            pending.add(ex.submit(_normalize_windows_chunk, chunk))

        prog_bar = tqdm(total=len(windows_list), desc="[BGP] Second pass - normalize", unit="win", colour="MAGENTA")

        try:
            while pending:
                done = next(as_completed(pending))
                pending.remove(done)
                results = done.result()  # list[(window, norm_block)]

                for window, norm_block in results:
                    writer.write_block(window=window, block=norm_block)
                    prog_bar.update(1)

                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    chunk = None
                if chunk is not None:
                    pending.add(ex.submit(_normalize_windows_chunk, chunk))
        except Exception as e:
            for f in pending: f.cancel()
            raise Exception(f"[parallel] Error during Pass 2:\n{e}")
        finally:
            prog_bar.close()


# ======================================================================================
# Generic streaming submission
# ======================================================================================

def submit_streaming(
    *,
    worker:Callable[..., Any],
    initializer:Callable[..., Any]|None = None,
    initargs:Tuple[()] = (),
    tasks:Iterable[Tuple[Any, ...]],
    consumer:Callable[[Any], None],
    max_workers:int|None = None,
    inflight:int = 2,
    prog_bar_label:str = "stream",
    prog_bar_color:str = "WHITE"
    ) -> None:
    """
    Generic helper function designed to parallelize tasks in a streaming fashion. 
    It avoids loading all data into memory at once by submitting tasks to a 
    process pool and consuming the results as they become available. 
    
    Args:
        worker (Callable[..., Any]): The function to be executed in parallel. 
            It should accept arguments (passed as a tuple).
        initializer (Callable[..., Any] | None, optional): A function to initialize 
            the worker process (used to set up any global state).
            Defaults to None.
        initargs (Tuple[()]): A tuple of arguments to pass to the initializer function.
            Likely, matches the function signature of the function that the initalizer is initalizing.
        tasks (Iterable[Tuple[Any, ...]]): An iterable of tasks to be executed. 
            Each task is represented as a tuple of arguments for the function.
        consumer (Callable[[Any], None]): A function that consumes the results returned by the worker. 
            This is where you write the results to disk or aggregate them.
        max_workers (int | None, optional): The maximum number of worker processes to use.
            If None, uses all CPU cores. Defaults to None.
        inflight (int, optional): The maximum number of tasks that can be in progress at any time. 
            This controls memory usage. Larger number processes more tasks, increases RAM usage.
            Defaults to 2.
        prog_bar_label (str, optional): Label for the progress bar.
        prog_bar_color (str, optional): Color for the progress bar.
    """
    tasks_list = list(tasks)
    total = len(tasks_list)

    # manage the worker processes
    with ProcessPoolExecutor(
        max_workers=max_workers, 
        initializer=initializer, 
        initargs=initargs
        ) as ex:

        pending:set[Future] = set()
        target_inflight = max(1, (max_workers or 1) * max(1, inflight))

        task_iterator = iter(tasks_list)
        for _ in range(target_inflight):
            try: 
                # grab next task from stack
                task = next(task_iterator)
            except StopIteration: 
                # no more tasks left
                break
            
            # Submit tasks to the process pool
            pending.add(ex.submit(worker, *task))

        prog_bar = tqdm(total=total, desc=prog_bar_label, unit="task", colour=prog_bar_color) 

        try:
            while pending:
                # Removes completed task from the "todo list"
                done = next(as_completed(pending)) 
                pending.remove(done) 
                result = done.result() 
                
                # Give results to consumer to write to disc
                consumer(result)
                
                # Update progress bar with +1 task completed
                prog_bar.update(1)
                
                try:
                    # Get next task to submit
                    task = next(task_iterator)
                except StopIteration: 
                    # signal no more tasks left to run
                    task = None
                if task is not None:
                    # Submit tasks to the process pool
                    pending.add(ex.submit(worker, *task))
        
        except Exception as e:
            # Cancels all future tasks from executing
            for future_task in pending:
                future_task.cancel()
            raise e
        
        finally:
            # Shut down progress bar - fail or success
            prog_bar.close()
