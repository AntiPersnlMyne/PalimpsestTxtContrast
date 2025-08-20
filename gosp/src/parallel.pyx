#!/usr/bin/env python3
# distutils: language=c

"""parallel.pyx: Parallelization ("multiprocessing") wrapper API. 

Notes
-----
- This module uses **streaming, chunked** submission to keep memory bounded.
- Workers re-open rasters locally (safe across processes).
- Parent process writes results immediately to avoid concurrent writes.
- Avoids sending large numpy arrays over IPC; sends only small window tuples.

Cython notes 
------------ 
- Heavy numeric work is implemented with nogil on float32 memoryviews. 
- IO (rasterio) stays under the GIL; seemed like something bad would happen.
"""

# --------------------------------------------------------------------------------------------
# Imports & thread oversubscription guards (safe defaults)
# --------------------------------------------------------------------------------------------
from __future__ import annotations  

import os
import importlib
from typing import Iterable, Iterator, List, Sequence, Tuple, Any, Callable
from dataclasses import dataclass
from tqdm import tqdm

cimport numpy as np
import numpy as np

from libc.math cimport fmaxf, fminf
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
import multiprocessing as mp

import rasterio
from rasterio.windows import Window

from ..build.rastio import MultibandBlockReader
from ..build.math_utils import project_block_onto_subspace


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
__status__ = "Development"  # "Prototype", "Development", "Production"


# Guard thread oversubscription inside forked workers
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ctypedef np.float32_t float_t
WindowType = Tuple[Tuple, Tuple] # ((row_off, col_off), (height, width))

@dataclass
class Target:
    value: float
    row: int
    col: int
    band_spectrum: np.ndarray  # shape=(bands,)

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------
def _chunked(
    iterable: Iterable[Any], 
    size: int
) -> Iterator[List[Any]]: 
    """Yield lists of up to size items from iterable. Coarsens submission to reduce scheduling/pickling overhead. """ 
    it = iter(iterable) 
    while True: 
        chunk:List[Any] = [] 
        for _ in range(size): 
            try: chunk.append(next(it)) 
            except StopIteration: break 
        
        # No more windows to put into chunk
        if not chunk: 
            return 
            
        yield chunk


cdef inline float _clampf(const float x) nogil: 
    # Clamp to [0,1] 
    return fminf(1.0, fmaxf(0.0, x)) 


cdef int _bandwise_minmax( 
    float_t[:, :, :] band_data, 
    float_t[:] out_mins, 
    float_t[:] out_maxs, 
) nogil: 
    cdef:
        Py_ssize_t bands  = band_data.shape[0] 
        Py_ssize_t height = band_data.shape[1] 
        Py_ssize_t width  = band_data.shape[2] 
        Py_ssize_t b, row, col 
        float_t px_val 
    
    for b in range(bands): 
        out_mins[b] = band_data[b, 0, 0] 
        out_maxs[b] = band_data[b, 0, 0] 
        
        for row in range(height): 
            for col in range(width): 
                px_val = band_data[b, row, col] 
                # Set new min
                if px_val < out_mins[b]: 
                    out_mins[b] = px_val 
                # Set new max
                elif px_val > out_maxs[b]: 
                    out_maxs[b] = px_val 


cdef int _normalize_inplace(
    float_t[:, :, :] block, 
    const float_t[:] mins, 
    const float_t[:] denom
) nogil:
    cdef:
        Py_ssize_t bands = block.shape[0] 
        Py_ssize_t height = block.shape[1] 
        Py_ssize_t width = block.shape[2] 
        Py_ssize_t b, row, col 
        float_t min_val, dnom, px_val 
    
    for b in range(bands): 
        min_val = mins[b] 
        dnom = denom[b] 
        for row in range(height): 
            for col in range(width): 
                px_val = (block[b, row, col] - min_val) / dnom
                block[b, row, col] = _clampf(px_val) 
                

cdef int _argmax_l2_norms(
    float_t[:, :, :] block,
    float_t* out_max_val, 
    Py_ssize_t* out_flat_idx, 
) nogil:
    cdef:
        Py_ssize_t bands = block.shape[0] 
        Py_ssize_t height = block.shape[1] 
        Py_ssize_t width = block.shape[2] 
        Py_ssize_t b, row, col 
        float_t acc, best = -3.3e38 # cannot -np.inf without GIL
        Py_ssize_t best_idx = 0, idx = 0 
    
    for row in range(height): 
        for col in range(width): 
            acc = 0.0 
            for b in range(bands): 
                acc += block[b, row, col] * block[b, row, col] 
            if acc > best: 
                best = acc 
                best_idx = idx 
            idx += 1 
    out_max_val[0] = best 
    out_flat_idx[0] = best_idx


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
    "full_synthetic": False,    # Optional log and sqrt in bgp
    "bands_fn": None,           # callable(image_block, full_synthetic) -> np.ndarray (bands, h, w)
    "reader": None,             # MultibandBlockReader per worker
}


def _init_generate_worker(
    input_paths: List[str], 
    func_module: str, 
    func_name: str, 
    full_synthetic: bool, 
) -> None:
    """
    Initializer for Pass 1 workers.

    Parameters
    ----------
        input_paths (list[str]): 
            Paths to input rasters. One multiband or many single-band files.
        func_module (str): 
            Absolute module path, e.g. "python_scripts.gosp.bgp".
        func_name (str): 
            Top-level function name to call, e.g. "_create_bands_from_block".
        full_synthetic (bool): 
            Flag forwarded to the band-generation function.
    """
    _gen_state["full_synthetic"] = full_synthetic
    _gen_state["bands_fn"] = getattr(importlib.import_module(func_module), func_name)
    _gen_state["reader"] = MultibandBlockReader(list(input_paths))


def _generate_windows_chunk(
    windows_chunk: List[tuple]
) -> List[Tuple[WindowType, np.ndarray, np.ndarray, np.ndarray]]:
    """Worker: read inputs, create synthetic bands for a chunk of windows."""
    
    # Get worker varaibles
    reader = _gen_state["reader"]
    full_synthetic = _gen_state["full_synthetic"]
    bands_fn = _gen_state["bands_fn"]

    # Instantiate output and mv
    band_stack:List[Tuple[WindowType, np.ndarray, np.ndarray, np.ndarray]] = []
    cdef:
        float_t[:, :, :] nb_mv
        np.ndarray[float_t] mins_np 
        np.ndarray[float_t] maxs_np
        float_t[:] mins_mv
        float_t[:] maxs_mv

    # Generate bandstack
    for window in windows_chunk:
        
        # Generate new bands
        block = reader.read_multiband_block(window)
        new_bands = bands_fn(block, full_synthetic)
        
        # Compute per-band mins/max
        nb_mv = new_bands 
        mins_np = np.empty(nb_mv.shape[0], dtype=np.float32) 
        maxs_np = np.empty(nb_mv.shape[0], dtype=np.float32) 
        # Pass numpy (np) to C-level before entering noGIL
        mins_mv = mins_np
        maxs_mv = maxs_np

        # Compute min/max in noGIL, keep it on otherwise
        with nogil:  
            _bandwise_minmax(nb_mv, mins_mv, maxs_mv) 

        band_stack.append((window, new_bands, mins_np, maxs_np))
    
    return band_stack


def parallel_generate_streaming(
    *,
    input_paths:Sequence[str],
    windows:Iterable[WindowType],
    writer:object,
    func_module:str,
    func_name:str,
    full_synthetic:bool,
    max_workers:int|None = None,
    inflight:int,
    chunk_size:int,
    show_progress:bool,
    desc: str = "[BGP] First pass - create",
) -> np.ndarray:
    """
    Streaming generation with bounded memory and per-band stats aggregation.

    Returns:
        band_stats (np.ndarray): Shape=(2, bands) where [0] = global mins, [1] = global maxs.
    """
    # Get multiprocessing context depending on user's OS
    # Build worker function around context
    context = mp.get_context()
    module = __import__(func_module, fromlist=[func_name])
    func = getattr(module, func_name)
    
    # Preallocate varaibles
    windows_list = list(windows)
    chunks = list(_chunked(windows_list, chunk_size))
    cdef np.ndarray global_mins = None
    cdef np.ndarray global_maxs = None

    # Multiprocess to run tasks
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_generate_worker,
        initargs=(list(input_paths), func_module, func_name, full_synthetic),
        mp_context=context
    ) as exec:
        # Instantiate vars ; ensure inflight >= 1
        pending: set[Future] = set()
        target_inflight = max(1, (max_workers or 1) * max(1, inflight))

        # Iterate through chunks and give to workers
        chunk_iter = iter(chunks)
        for _ in range(target_inflight):
            try:
                chunk = next(chunk_iter)
            except StopIteration:
                break
            pending.add(exec.submit(_generate_windows_chunk, chunk))

        # Show progress in terminal
        prog_bar = tqdm(total=len(windows_list), desc=desc, unit="win", colour="CYAN") if show_progress else None
        
        try:
            # Process all tasks in 'todo' list
            while pending:
                done = next(as_completed(pending))
                pending.remove(done)
                results = done.result()  # list of (window, new_bands, mins, maxs)

                # Doing the task: write blocks and update progress
                for window, new_bands, mins, maxs in results:
                    writer.write_block(window=window, block=new_bands)
                    global_mins = np.minimum(global_mins, mins) if global_mins is not None else mins.copy()
                    global_maxs = np.maximum(global_maxs, maxs) if global_maxs is not None else maxs.copy()
                    if prog_bar is not None: prog_bar.update(1)

                # Get next chunk to process
                try: chunk = next(chunk_iter)
                except StopIteration: chunk = None
                    
                # Send next chunk to process
                if chunk is not None:
                    pending.add(exec.submit(_generate_windows_chunk, chunk))
        
        
        # Catch any errors/exceptions during processing the 'todo' list
        except Exception as e:
            for f in pending: f.cancel()
            raise Exception(f"[parallel] Error during Pass 1:\n{e}")
        
        
        # End execution; close prog-bar
        finally:
            if prog_bar is not None: 
                prog_bar.close()

    assert global_mins is not None and global_maxs is not None, "No windows processed"
    return np.stack([global_mins, global_maxs], axis=0)


# --------------------------------------------------------------------------------------
# Pass 2: bands -> normalize -> write (normalization)
# --------------------------------------------------------------------------------------
_worker_state: dict[str, Any] = {
    """Stores global information accessible to the worker"""
    
    "src": None,           # rasterio.DatasetReader
    "band_mins": None,     # np.ndarray; shape: (bands,)
    "band_maxs": None,     # np.ndarray; shape: (bands,)
}


def _init_normalize_worker(
    unorm_path: str, 
    band_mins: np.ndarray, 
    band_maxs: np.ndarray
    ) -> None:
    """
    Initalizes worker process by setting global information accessable to the worker

    Args:
        unorm_path (str): Path to unnormalized dataset (filename not included).
        band_mins (np.ndarray): Array of min-value per band.
        band_maxs (np.ndarray): Array of max-value per band.
    """
    # Instantiate global worker variables
    _worker_state["src"] = rasterio.open(unorm_path, "r")
    _worker_state["band_mins"] = np.asarray(band_mins, dtype=np.float32)
    _worker_state["band_maxs"] = np.asarray(band_maxs, dtype=np.float32)

    # Warmup: allocate tiny array and compute once to prime threads/allocator
    b_mins = _worker_state["band_mins"].shape[0]
    dummy_block = np.zeros((b_mins, 2, 2), dtype=np.float32)
    denom = np.maximum(_worker_state["band_maxs"] - _worker_state["band_mins"], 1e-8) # prevent div 0
    np.divide(dummy_block, denom[:, None, None], out=dummy_block)


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
    src:rasterio.DatasetReader = _worker_state["src"] # DatasetReader
    mins:np.ndarray = _worker_state["band_mins"]      # (bands,)
    maxs:np.ndarray = _worker_state["band_maxs"]      # (bands,)

    # bandwise calculate denominator; const varaible
    denom = np.maximum(maxs - mins, 1e-8).astype(np.float32) # 1e-8 prevent div 0
    output:List[Tuple[WindowType, np.ndarray]] = []

    # Memory views
    cdef:
        float_t[:, :, :] block_mv 
        float_t[:]       mins_mv
        float_t[:]       denom_mv 

    for window in windows_chunk:
        # Read block from window
        (row_off, col_off), (win_height, win_width) = window
        block = src.read(window=Window(col_off, row_off, win_width, win_height)).astype(np.float32)
        # Cythonized in-place normalization + clamp [0,1]
        block_mv = block
        mins_mv = mins
        denom_mv = denom
        with nogil:
            _normalize_inplace(block_mv, mins_mv, denom_mv)

        output.append((window, block))

    return output


def parallel_normalize_streaming(
    *,
    unorm_path: str,
    windows: Iterable[WindowType],
    band_mins: np.ndarray,
    band_maxs: np.ndarray,
    writer,
    max_workers: int | None = None,
    inflight:int,
    chunk_size:int,
    show_progress:bool
    ) -> None:
    """
    Orchestrates the parallel processing using submit_streaming. 
    It sets up the parameters for the streaming pipeline, initializes the worker 
    processes, and submits the normalization tasks to the pool of workers.
    
    `NOTE`: keyword arguments required; This means that you cannot pass these arguments positionally; you must use their names.

    Args:
        unorm_path (str): 
            Path to the unnormalized dataset (multiband TIFF) written in Pass 1.
        windows (list(tuple)): 
            Sequence of windows to process. Order is not required.
        band_mins (np.ndarray): 
            1-D array of band maximums.
        band_maxs (np.ndarray): 
            1-D array of band maximums.
        writer (MultibandBlockWriter): 
            Writer object that will receive each normalized block.
        max_workers (int): 
            Number of worker processes. If None, defaults to os.cpu_count() (i.e. all of them). Defaults to None.
        chunk_size (int): 
            Number of windows processed per task. Increase to reduce overhead.
        inflight (int): 
            At most inflight * max_workers tasks will be in flight ("worked on") at once.
            Lower to reduce RAM; raise to improve throughput.
        show_progress (bool, optional):
            If true, display progress bar
    """
    windows_list = list(windows)
    chunks = list(_chunked(windows_list, chunk_size))

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_normalize_worker,
        initargs=(unorm_path, band_mins, band_maxs),
    ) as exec:
        # "todo" list of all future tasks
        pending: set[Future] = set()
        # Prevent inflight from being negative
        target_inflight = max(1, (max_workers or 1) * max(1, inflight))

        chunk_iter = iter(chunks)
        for _ in range(target_inflight):
            # Submit as many chunks to pool as inflight permits
            try: chunk = next(chunk_iter) # grabs next task
            except StopIteration: break   # no more tasks left
            
            # Submit tasks to the process pool
            pending.add(exec.submit(_normalize_windows_chunk, chunk))

        # Instantiate terminal progress bar if user-specified
        prog_bar = tqdm(total=len(windows_list), desc="[BGP] Second pass - normalize", unit="win", colour="MAGENTA") if show_progress else None

        try:
            # Process all tasks in "todo" list
            while pending:
                # Removes completed task from the "todo list"
                done = next(as_completed(pending))
                pending.remove(done)
                results = done.result()  # list[(window, norm_block)]

                # Doing the task: write blocks and update progress
                for window, norm_block in results:
                    writer.write_block(window=window, block=norm_block)
                    if prog_bar is not None: prog_bar.update(1)

                
                try: chunk = next(chunk_iter)       # Get next chunk to process
                except StopIteration: chunk = None  # signal no more chunks
                
                # Send next chunk to process pool
                if chunk is not None:
                    pending.add(exec.submit(_normalize_windows_chunk, chunk))
        
        
        # Catch any errors/exceptions during processing the 'todo' list
        except Exception as e:
            for task in pending: task.cancel()
            raise Exception(f"[parallel] Error during Pass 2:\n{e}")
        
        
        # Always shut down progress bar fail or succeed
        finally:
            if prog_bar is not None: prog_bar.close()



# --------------------------------------------------------------------------------------
# Parallel scan helpers (submit_streaming): initializer + worker + driver
# --------------------------------------------------------------------------------------
_scan_state: dict[str, Any|None] = {
    "paths": None,              # List[str]
    "projection_matrix": None,  # np.ndarray | None
    "reader": None,             # MultibandBlockReader
}


def _init_scan_worker(paths: Sequence[str], projection: np.ndarray|None) -> None:
    """
    Initializer for parallel scan workers.

    Opens a reader in each process and stores the projection matrix (if any).
    """
    _scan_state["paths"] = list(paths)
    _scan_state["projection_matrix"] = projection.astype(np.float32) if projection is not None else None
    _scan_state["reader"] = MultibandBlockReader(list(paths))


def _scan_window(window: WindowType) -> Tuple[float_t, int, int, np.ndarray]:
    """
    (per worker) Scan a single window and return its best candidate.

    Args:
        windows (Iterable[WindowType]): List of windows to iterate over.

    Returns:
        Tuple[float,int,int,np.ndarray]: Values of a Target dataclass: (value, row, col, band_spectrum)
    """
    assert (
        isinstance(window, tuple)
        and len(window) == 2
        and isinstance(window[0], tuple)
        and isinstance(window[1], tuple)
    ), f"Band WindowType: {window!r}"

    reader:MultibandBlockReader = _scan_state["reader"]  # type:ignore 
    p_matrix = _scan_state["projection_matrix"]
    (row_off, col_off), (_, win_width) = window

    orig_block = reader.read_multiband_block(window)  # (bands, h, w)
    proj_block = project_block_onto_subspace(block=orig_block, projection_matrix=p_matrix) if p_matrix is not None else orig_block

    # Fast argmax of L2 norms
    cdef float_t[:, :, :] mv = proj_block
    cdef float_t max_px_val
    cdef Py_ssize_t max_px_idx
    with nogil:
        _argmax_l2_norms(mv, &max_px_val, &max_px_idx)

    block_row, block_col = divmod(int(max_px_idx), int(win_width))

    bands_orig = orig_block[:, block_row, block_col]

    return float(max_px_val), row_off + block_row, col_off + block_col, bands_orig


def best_target_parallel(
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

    The parent process reduces local candidate Targets into a single global best.
    """
    # Initalize output
    best_target:Target = Target(value=-np.inf, row=-1, col=-1, band_spectrum=np.empty(0, dtype=np.float32))
    seen = 0

    # Determine best target
    def _consume(result: Tuple[float, int, int, np.ndarray]) -> None:
        nonlocal best_target, seen
        seen += 1
        value, row, col, band_spec = result
        # Compare size ("value") of targets' L2 norms
        if value > best_target.value: 
            best_target = Target(value, row, col, band_spec)


    tasks = [(window,) for window in list(windows)] # wrap args (window) in tuple for streaming

    # Submit tasks to parallel processing
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

    assert seen > 0 and best_target.band_spectrum.size > 0, "No pixels scanned; empty window list purghaps?"
    return best_target


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
    Function parallelizes tasks in a streaming fashion between cores with python's multiprocessing lbirary.
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

    with ProcessPoolExecutor(
        max_workers=max_workers, 
        initializer=initializer, 
        initargs=initargs
    ) as exec:
        # "todo" list of all future tasks
        pending = set()
        # Prevent inflight from being negative
        target_inflight = max(1, (max_workers or 1) * max(1, inflight))

        task_iterator = iter(tasks_list)
        for _ in range(target_inflight):
            # Submit as many chunks to pool as inflight permits
            try: task = next(task_iterator) # grabs next task
            except StopIteration: break     # no more tasks left
            
            # Submit tasks to the process pool
            pending.add(exec.submit(worker, *task))

        prog_bar = tqdm(total=total, desc=prog_bar_label, unit="task", colour=prog_bar_color)  if show_progress else None

        try:
            # Process all tasks in "todo" list
            while pending:
                # Removes completed task from the "todo list"
                done = next(as_completed(pending)) 
                pending.remove(done) 
                result = done.result() 
                
                # Give results to consumer to write to disc
                # Update progress bar with +1 task completed
                consumer(result)
                if prog_bar is not None: 
                    prog_bar.update(1)
                
                # Get next task to process
                try: task = next(task_iterator)
                except StopIteration: task = None # signal no more tasks left to run
                
                # Submit next task to the process pool
                if task is not None:
                    pending.add(exec.submit(worker, *task))
        
        
        # Catch any errors/exceptions during processing the 'todo' list
        except Exception as e:
            # Cancels all future tasks from executing
            for future_task in pending:
                future_task.cancel()
            raise e
        
        # Always shut down progress bar fail or succeed
        finally:
            if prog_bar is not None: prog_bar.close()
