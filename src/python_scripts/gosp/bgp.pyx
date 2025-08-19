# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False


"""bgp.py: Band Generation Process, creates new non-linear bondinations of existing bands"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from libc.math cimport sqrtf, log1pf
from typing import Tuple, List
from os.path import join

from .rastio import MultibandBlockReader, MultibandBlockWriter
from .parallel import parallel_normalize_streaming, parallel_generate_streaming
from .file_utils import rm


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.3"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
np.import_array()

# Typed aliases for readability
ctypedef np.float32_t float_t
ctypedef np.uint16_t uint16_t
ctypedef Py_ssize_t psize_t


# ------------------
# C helper functions
# ------------------
cdef inline psize_t _expected_total_bands_cy(
    psize_t nbands,
    bint full_synthetic
) nogil:
    """
    Expected total bands from input band count `nbands` and `full_synthetic` flag.

    Formula:
      total = n + n*(n-1)/2   # original + pairwise
      if full_synthetic: add 2*n (sqrt + log1p)
    """
    cdef psize_t total = nbands + (nbands * (nbands - 1)) // 2
    if full_synthetic:
        total += 2 * nbands
    return total


cdef void _create_bands_from_block_kernel(
    float_t[:, :, :] src_mv,        # (bands, height, width)
    float_t[:, :, :] bandstack_mv,        # (total_bands, h, w) preallocated
    bint full_synthetic
) nogil:
    """
    Fill bandstack_mv with generated bands.
    + auto- and cross-correlation
    + (optional) sqrt and log1p
    """
    cdef:
        psize_t bands  = src_mv.shape[0]
        psize_t height = src_mv.shape[1]
        psize_t width  = src_mv.shape[2]

        psize_t dst_idx = 0
        psize_t b, row, col, i, j

    # ==============
    # Original Bands
    # ==============
    for b in range(bands):
        for row in range(height):
            for col in range(width):
                bandstack_mv[dst_idx+b, row, col] = src_mv[b, row, col]
    dst_idx += bands

    # ========================================
    # Correlations (no redundant combinations)
    # ========================================
    # Ensure no redundant combinations
    for i in range(bands - 1):
        for j in range(i + 1, bands):
            # Correlation multiplication
            for row in range(height):
                for col in range(width):
                    bandstack_mv[dst_idx, row, col] = src_mv[i, row, col] * src_mv[j, row, col]
            dst_idx += 1

    # ===================================
    # "full_synthetic" log and sqrt bands
    # ===================================
    if full_synthetic:
        # sqrt bands
        for b in range(bands):
            for row in range(height):
                for col in range(width):
                    # clamp negative to 0
                    bandstack_mv[dst_idx+b, row, col] = sqrtf(src_mv[b, row, col]) if src_mv[b, row, col] >= 0.0 else 0.0
        dst_idx += bands

        # log bands
        for b in range(bands):
            for row in range(height):
                for col in range(width):
                    val = src_mv[b, row, col]
                    # guard: log1p undefined for val <= -1; 
                    if val <= -1:
                        bandstack_mv[dst_idx + b, row, col] = log1pf(-0.99) # -1 is undefined
                    else:
                        bandstack_mv[dst_idx + b, row, col] = log1pf(val)
        dst_idx += bands


# ----------------------------------------
# Python-callable band creation function
# (must be importable by workers)
# ----------------------------------------
def _create_bands_from_block(image_block:np.ndarray, full_synthetic:bool):
    """
    Python-callable wrapper used by worker processes.

    Args:
      image_block (ndarray):
        (bands, height, width). Data will be cast to float32).
      full_synthetic (bool):
        If true, generates additional sqrt and log bands.

    Returns:
      ndarray: band_stack=(total_bands, height, width) ; dtype=float32
    """
    if image_block.ndim != 3:
        raise ValueError(f"[BGP] image_block must be 3D (bands,h,w); got shape {image_block.shape}")

    # Ensure float32 and contiguous layout (no copies if already correct)
    if image_block.dtype != np.float32 or not image_block.flags['C_CONTIGUOUS']:
        src = np.ascontiguousarray(image_block, dtype=np.float32)
    else:
        src = image_block

    # Source dims
    bands  = src.shape[0]
    height = src.shape[1]
    width  = src.shape[2]

    # Calculat output (#bands) size
    tot_num_bands = _expected_total_bands_cy(bands, full_synthetic)

    # Preallocate output array (float32)
    band_stack = np.empty((tot_num_bands, height, width), dtype=np.float32)

    # Get fast C memoryviews 
    cdef float_t[:, :, :] src_mv = src
    cdef float_t[:, :, :] bs_mv = band_stack

    # noGIL kernel to fill band_stack
    with nogil:
        _create_bands_from_block_kernel(src_mv, bs_mv, full_synthetic)

    return band_stack


# --------------------------------------------------------------------------------------------
# Band Generation Process (BGP)
# --------------------------------------------------------------------------------------------
cpdef None band_generation_process(
    List[str] input_image_paths,
    str output_dir,
    Tuple[uint16_t,uint16_t] window_shape,
    bint full_synthetic,
    uint16_t max_workers,
    uint16_t chunk_size,
    uint16_t inflight,
    bint show_progress
    ):
    """
    The Band Generation Process. Generates synthetic, non-linear bands as combinations of existing bands. 
    Output is normalized range [0,1] per band.

    Args:
        input_image_paths (List[str]): 
            List of paths to input images / multispectral data.
        dst_dir (str): 
            Output directory of generated band image. 
        window_shape (Tuple[int,int]): 
            Shape of each block to process. 
            Larger blocks proceess faster and use more memory. 
            Smaller block process slower with a smaller memory footprint. 
        full_synthetic (bint): 
            If True, generate ln and sqrt bands.
        max_workers (int): 
            Number of cores for paralellization. If None, defaults to number of processors on the machine.
            i.e. more workers = more fast.
        chunk_size (int): 
            How many windows of data the program can parallelize at once. 
            i.e. more chunks = more fast. Try 8 or 16 if RAM allows.
        inflight (int): 
            Controls memory footprint. At most `inflight * max_workers` blocks in RAM. Defaults to 2.
        verbose (bint): 
            If true, shows progress bars.
    """
    # ============================================================
    # Scan the input to obtain image size & window dimensions
    # ============================================================
    cdef:
        MultibandBlockReader input_dataset = MultibandBlockReader(input_image_paths)
        tuple[uint16_t, uint16_t] input_shape = input_dataset.image_shape()
        # Extract dimensions
        const uint16_t src_height = input_shape[0]
        const uint16_t src_width = input_shape[1]
        const uint16_t win_height = window_shape[0]
        const uint16_t win_width = window_shape[1]  
    
    # ============================================================
    # Peek one-pixel block to calculate tot num output bands
    # ============================================================
    cdef np.ndarray[float_t, ndim=3] tiny_block
    cdef np.ndarray[float_t, ndim=2] sample_bands
    # Read in (bands,1,1) block 
    tiny_block = input_dataset.read_multiband_block(((0, 0), (1, 1)))
    # Returns (num_output_bands,...)
    sample_bands = _create_bands_from_block(tiny_block, full_synthetic)
    cdef uint16_t num_output_bands = sample_bands.shape[0] #type:ignore[CythonreportGeneralTypeIssues]
    # free the temporary data
    del tiny_block, sample_bands
    
    # ============================================================
    # Arrays to hold global min/max for later normalization
    # ============================================================
    cdef:
        np.ndarray[float_t, ndim=1] np_band_mins = np.full(num_output_bands, np.inf, dtype=float_t)
        np.ndarray[float_t, ndim=1] np_band_maxs = np.full(num_output_bands, -np.inf, dtype=float_t)
        # Memory view for faster access
        float_t[:] band_mins = np_band_mins #type:ignore[CythonreportGeneralTypeIssues]
        float_t[:] band_maxs = np_band_maxs #type:ignore[CythonreportGeneralTypeIssues]
    
    # ============================================================
    # Preallocate list of windows
    # ============================================================
    cdef:
        # Calc how many windows fit inside the source
        uint16_t n_rows = (src_height + win_height - 1) // win_height
        uint16_t n_cols = (src_width  + win_width  - 1) // win_width
        uint16_t total_windows = n_rows * n_cols
        # List of window coordinate-tuples
        list windows = [None] * total_windows
        int win_idx = 0
        
        int row_off, col_off, actual_height, actual_width

    # Calc windows indices
    for row_off in range(0, src_height, win_height):
        for col_off in range(0, src_width, win_width):
            # Ensure window does not index out-of-bounds
            actual_height = win_height if row_off + win_height <= src_height else src_height - row_off
            actual_width  = win_width  if col_off + win_width  <= src_width  else src_width  - col_off
            windows[win_idx] = ((row_off, col_off), (actual_height, actual_width)) #type:ignore[CythonreportGeneralTypeIssues]
            win_idx += 1


    # --------------------------------------------------------------------------------------------
    # Pass 1: Generate unnormalized output
    # --------------------------------------------------------------------------------------------
    # (Hardcoded) output file names
    cdef: 
        str output_unorm_filename = "gen_band_unorm.tif" # un-normalized bands
        str output_norm_filename = "gen_band_norm.tif"   # normalized bands
        str unorm_path = join(output_dir, output_unorm_filename)
    
    # Write unnormalized output, collect global stats
    with MultibandBlockWriter(
        output_dir          = output_dir,
        output_image_shape  = input_shape,
        output_image_name   = output_unorm_filename,
        window_shape        = window_shape,
        num_bands           = num_output_bands,
        output_datatype     = np.float32
    ) as writer:
        
        cdef tuple band_stats = parallel_generate_streaming(
            input_paths     = input_image_paths,
            windows         = windows,
            writer          = writer,
            func_module     = "python_scripts.gosp.bgp", 
            func_name       = "_create_bands_from_block",   
            full_synthetic  = full_synthetic,
            max_workers     = max_workers,                       
            chunk_size      = chunk_size,
            inflight        = inflight,                            
            show_progress   = show_progress
        )

    # Extract stats from band_stats
    band_mins[:] = band_stats[0] # type:ignore[PythonreportGeneralTypeIssues]
    band_maxs[:] = band_stats[1] # type:ignore[PythonreportGeneralTypeIssues]

    
    # --------------------------------------------------------------------------------------------
    # Pass 2: Normalize output
    # --------------------------------------------------------------------------------------------
    with MultibandBlockWriter(
        output_dir          = output_dir,
        output_image_shape  = input_shape,
        output_image_name   = output_norm_filename,
        window_shape        = window_shape,
        num_bands           = num_output_bands,
        output_datatype     = float_t
    ) as writer:
        # Stream in parallel: workers read+normalize, parent writes
        parallel_normalize_streaming(
            unorm_path      = unorm_path,
            windows         = windows,
            band_mins       = band_mins,
            band_maxs       = band_maxs,
            writer          = writer,
            max_workers     = max_workers, 
            inflight        = inflight,
            chunk_size      = chunk_size,
            show_progress   = show_progress
        )

    # ============================================================================================
    # Cleanup Temporary File
    # ============================================================================================
    rm(unorm_path) 
    
            