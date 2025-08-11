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
__version__ = "2.1.0" 
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development"  # "Prototype", "Development", "Production"

# --------------------------------------------------------------------------------------------
# Imports & thread oversubscription guards (safe defaults)
# --------------------------------------------------------------------------------------------
import numpy as np
import rasterio
import importlib

from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import Iterable, Iterator, List, Sequence, Tuple, Protocol, Any, Callable
from tqdm import tqdm
from rasterio.windows import Window
from dataclasses import dataclass

from ..atdca.rastio import MultibandBlockReader
from ..utils.math_utils import (
    project_block_onto_subspace,
    block_l2_norms
)



# --------------------------------------------------------------------------------------
# Data Structure-like-things
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


@dataclass
class Target:
    value: float
    row: int
    col: int
    band_spectrum: np.ndarray  # shape (bands,)



# --------------------------------------------------------------------------------------
# Pass 1: read -> create synthetic bands (generation)
# --------------------------------------------------------------------------------------

_gen_state: dict[str, Any] = {
    "paths": None,       # List[str]
    "use_sqrt": False,
    "use_log": False,
    "bands_fn": None,   # callable(image_block, use_sqrt, use_log) -> np.ndarray (bands, h, w)
}


def _init_generate_worker(
    input_paths: List[str], 
    func_module: str, 
    func_name: str, 
    use_sqrt: bool, 
    use_log: bool
    ) -> None:
    """
    Initializer for Pass 1 workers.

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
    """
    Streaming generation with bounded memory and per-band stats aggregation.

    Returns:
        band_stats (np.ndarray): Shape=(2, bands) where [0] = global mins, [1] = global maxs.
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

        prog_bar = tqdm(total=len(windows_list), desc=desc, unit="win", colour="CYAN") if show_progress else None

        try:
            while pending:
                done = next(as_completed(pending))
                pending.remove(done)
                results = done.result()  # list of (window, new_bands, mins, maxs)

                for window, new_bands, mins, maxs in results:
                    writer.write_block(window=window, block=new_bands)
                    global_mins = np.minimum(global_mins, mins) if global_mins is not None else mins.copy()
                    global_maxs = np.maximum(global_maxs, maxs) if global_maxs is not None else maxs.copy()
                    if prog_bar is not None:
                        prog_bar.update(1)

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
            if prog_bar is not None: prog_bar.close()

    assert global_mins is not None and global_maxs is not None, "No windows processed"
    return np.stack([global_mins, global_maxs], axis=0)


# --------------------------------------------------------------------------------------
# Pass 2: bands -> normalize -> write (normalization)
# --------------------------------------------------------------------------------------

_worker_state: dict[str, Any] = {
    """Stores global information accessible to the worker"""
    
    "path": None,          # str
    "band_mins": None,     # np.ndarray; shape: (bands,)
    "band_maxs": None,     # np.ndarray; shape: (bands,)
}


def _init_normalize_worker(
    unorm_path: str, 
    band_mins: np.ndarray, 
    band_maxs: np.ndarray
    ) -> None:
    """
    Initalizes worker process by
    1. Setup global information accessable to the worker
    2. Ensure any BLAS pools are initalized and resources ready

    Args:
        unorm_path (str): Path to unnormalized dataset.
        band_mins (np.ndarray): Array of min-value per band.
        band_maxs (np.ndarray): Array of max-value per band.
    """
    # Instantiate global worker variables
    _worker_state["path"] = unorm_path
    _worker_state["band_mins"] = np.asarray(band_mins, dtype=np.float32)
    _worker_state["band_maxs"] = np.asarray(band_maxs, dtype=np.float32)

    # Light warmup (no disk I/O) to pre-allocate and ensure BLAS pools are quiet
    b_dim_len = _worker_state["band_mins"].shape[0]
    dummy_block = np.zeros((b_dim_len, 4, 4), dtype=np.float32)
    denom = np.maximum(_worker_state["band_maxs"] - _worker_state["band_mins"], 1e-8) # prevent div 0
    _ = (dummy_block - _worker_state["band_mins"][..., None, None]) / denom[..., None, None]


def _normalize_windows_chunk(
    windows_chunk: List[WindowType]
    ) -> List[Tuple[WindowType, np.ndarray]]:
    """
    (per-worker) Read a chunk of windows and return normalized blocks for each.

    Args:
        windows_chunk (List[WindowType]): List of windows - not chunks - to normalize.

    Returns:
        List[Tuple[WindowType, np.ndarray]]: list[(window, norm_block)] where norm_block has shape (bands, h, w), float32.
    """
    
    # Get global worker variables for workers to access
    path: str = _worker_state["path"]
    mins: np.ndarray = _worker_state["band_mins"]
    maxs: np.ndarray = _worker_state["band_maxs"]

    # bandwise calculate denominator; const varaible
    denom = np.maximum(maxs - mins, 1e-8).astype(np.float32) # prevent div 0
    out:List[Tuple[WindowType, np.ndarray]] = []

    # Calculate norm of each window and add results to list
    with rasterio.open(path, "r") as src:
        for window in windows_chunk:
            
            # Get data block
            (row_off, col_off), (h, w) = window
            block = src.read(window=Window(col_off, row_off, w, h)).astype(np.float32) #type:ignore
            
            # np.newaxis for block dimension compatibility - doesn't add new data
            norm = (block - mins[:, np.newaxis, np.newaxis]) / denom[:, np.newaxis, np.newaxis]
            np.clip(norm, 0.0, 1.0, out=norm)
            
            # append normalized tile to end of output
            out.append((window, norm))

    return out # length: len(windows_chunk) 


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
    show_progress:bool = False
    ) -> None:
    """
    Orchestrates the parallel processing using submit_streaming. 
    It sets up the parameters for the streaming pipeline, initializes the worker 
    processes, and submits the normalization tasks to the pool of workers.
    
    `NOTE`: keyword arguments required; This means that you cannot pass these arguments positionally; you must use their names.

    Args:
        unorm_path (str): Path to the unnormalized dataset (multiband TIFF) written in Pass 1.
        windows (Iterable[WindowType]): Sequence of windows to process. Order is not required.
        band_mins (np.ndarray): 1-D array of band maximums.
        band_maxs (np.ndarray): 1-D array of band maximums.
        writer (SupportsWriteBlock): Writer object that will receive each normalized block.
        max_workers (int, optional): Number of worker processes. If None, defaults to os.cpu_count() (i.e. all of them). Defaults to None.
        chunk_size (int, optional): Number of windows processed per task. Increase to reduce overhead. Defaults to 4.
        inflight (int): 
            At most inflight * max_workers tasks will be in flight ("worked on") at once.
            Lower to reduce RAM; raise to improve throughput.
            Defaults to 2.
    """
    windows_list = list(windows)
    chunks = list(_chunked(windows_list, chunk_size))

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_normalize_worker,
        initargs=(unorm_path, band_mins, band_maxs),
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

        if show_progress: prog_bar = tqdm(total=len(windows_list), desc="[BGP] Second pass - normalize", unit="win", colour="MAGENTA")

        try:
            while pending:
                done = next(as_completed(pending))
                pending.remove(done)
                results = done.result()  # list[(window, norm_block)]

                for window, norm_block in results:
                    writer.write_block(window=window, block=norm_block)
                    if show_progress: prog_bar.update(1)

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
            if show_progress: prog_bar.close()



# --------------------------------------------------------------------------------------
# Parallel scan helpers (submit_streaming): initializer + worker + driver
# --------------------------------------------------------------------------------------
_scan_state: dict[str, Any|None] = {
    "paths": None,              # Sequence[str]
    "projection_matrix": None,  # np.ndarray | None
    "reader": None,             # MultibandBlockReader
}


def _init_scan_worker(paths: Sequence[str], projection: np.ndarray|None) -> None:
    """
    Initializer for parallel scan workers.

    Opens a reader in each process and stores the projection matrix (if any).
    """
    _scan_state["paths"] = list(paths)
    _scan_state["projection"] = projection.astype(np.float32) if projection is not None else None
    _scan_state["reader"] = MultibandBlockReader(list(paths))


def _scan_window(window: WindowType) -> Tuple[float, int, int, np.ndarray]:
    """
    (per worker) Scan a single window and return its best candidate.

    Args:
        windows (Iterable[WindowType]): List of windows to iterate over.

    Returns:
        Tuple[float,int,int,np.ndarray]: Values of a Target dataclass: (value, row, col, band_spectrum)
    """
    # 
    assert isinstance(window, tuple) and len(window) == 2 \
    and isinstance(window[0], tuple) and isinstance(window[1], tuple), \
        f"Expected WindowType ((row_off,col_off),(h,w)), got: {window!r}"
        
    # Get worker varaibles
    reader:MultibandBlockReader = _scan_state["reader"] #type:ignore 
    p_matrix = _scan_state["projection_matrix"]
    (row_off, col_off), (win_height, win_width) = window
    
    # Project block onto P matrix
    block = reader.read_multiband_block(window)  # (bands, h, w)
    if p_matrix is not None: block = project_block_onto_subspace(block=block, projection_matrix=p_matrix)  # (bands, h, w)

        # Compute L2 norm and returns tile
    norms = block_l2_norms(block)  # shape: (h, w)
    
    # Find pixel within tile with largest norm
    max_px_idx = int(np.argmax(norms))
    max_px_val = float(norms.flat[max_px_idx])

    block_row, block_col = divmod(max_px_idx, win_width)
    
    # Convert tile-local coordinates to full image coordinates
    im_row, im_col = row_off + block_row, col_off + block_col
    
    # Extract all bands (bands,:,:) from the best pixel
    bands = block[:, block_row, block_col].astype(np.float32)
    
    # Return Target dataclass values
    return max_px_val, im_row, im_col, bands


def scan_for_max_parallel(
    *,
    paths: Sequence[str],
    windows: Iterable[WindowType],
    p_matrix: np.ndarray|None,
    max_workers: int|None = None,
    inflight: int = 2,
    show_progress: bool = True,
) -> Target:
    """
    Parallel variant of global argmax scan using `submit_streaming`.

    The parent process reduces local candidates into a single global best.
    """
    # Initalize best_target output
    best_target:Target = Target(0,0,0,np.empty(0))

    def _consume(result: Tuple[float, int, int, np.ndarray]) -> None:
        nonlocal best_target
        value, row, col, band_spec = result
        target = Target(value, row, col, band_spec)
        if target.value > best_target.value: best_target = target

    tasks = [(window,) for window in list(windows)] # wrap args (window) in tuple for streaming

    submit_streaming(
        worker= _scan_window, 
        initializer=_init_scan_worker,
        initargs=(list(paths), p_matrix),
        tasks=tasks,
        consumer=_consume,
        max_workers=max_workers,
        inflight=inflight,
        prog_bar_label="[TGP] Target generation",
        prog_bar_color="YELLOW",
        show_progress=show_progress
    )

    assert best_target is not None, "No pixels scanned; empty window list purghaps?"
    return best_target




# --------------------------------------------------------------------------------------
# Generic Streaming Submission
# --------------------------------------------------------------------------------------

def submit_streaming(
    *,
    worker:Callable[..., Any],
    initializer:Callable[..., Any]|None = None,
    initargs:Tuple = (),
    tasks:Iterable[Tuple[Any, ...]],
    consumer:Callable[[Any], None],
    max_workers:int|None = None,
    inflight:int = 2,
    prog_bar_label:str = "stream",
    prog_bar_color:str = "WHITE",
    show_progress:bool = False
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

        if show_progress: prog_bar = tqdm(total=total, desc=prog_bar_label, unit="task", colour=prog_bar_color) 

        try:
            while pending:
                # Removes completed task from the "todo list"
                done = next(as_completed(pending)) 
                pending.remove(done) 
                result = done.result() 
                
                # Give results to consumer to write to disc
                consumer(result)
                
                # Update progress bar with +1 task completed
                if show_progress: prog_bar.update(1)
                
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
            if show_progress: prog_bar.close()
