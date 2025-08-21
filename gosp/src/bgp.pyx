#!/usr/bin/env python3
# distutils: language = c
# cython: profile=True

"""bgp.pyx: Band Generation Process, creates new non-linear bondinations of existing bands"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np

from typing import Tuple, List
from os.path import join

from cython.parallel import prange
from libc.math cimport sqrtf, log1pf

from gosp.build.rastio import MultibandBlockReader, MultibandBlockWriter
# from gosp.build.parallel import parallel_normalize_streaming, parallel_generate_streaming
from gosp.build.file_utils import rm


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.5"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
np.import_array()

# Typed aliases for readability
ctypedef np.float32_t float_t
ctypedef Py_ssize_t psize_t


# ------------------
# C helper functions
# ------------------
cdef inline psize_t _total_bands_cy(
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


cdef int _create_bands_from_block_cy(
    float_t[:, :, :] src_mv,        # (bands, height, width)
    float_t[:, :, :] bandstack_mv,  # (total_bands, h, w) preallocated
    bint full_synthetic
) noexcept nogil:
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
        psize_t idx, total_px=height * width

        float_t val

    # ==============
    # Original Bands
    # ==============
    for b in range(bands):
        # Parallel by rows - kinda like pointers
        for row in prange(height, nogil=True, schedule="static"):  
            for col in range(width):
                bandstack_mv[dst_idx+b, row, col] = src_mv[b, row, col]
                
    dst_idx += bands

    # ========================================
    # Correlations (no redundant combinations)
    # ========================================
    # Ensure no redundant combinations
    for i in range(bands - 1):
        for j in range(i + 1, bands):
            for row in prange(height, nogil=True, schedule="static"):
                    for col in range(width):
                        bandstack_mv[dst_idx, row, col] = src_mv[i, row, col] * src_mv[j, row, col]
            dst_idx += 1

    # ===================================
    # "full_synthetic" log and sqrt bands
    # ===================================
    if full_synthetic:
        # sqrt bands
        for b in range(bands):
            for row in prange(height, nogil=True, schedule="static"):
                for col in range(width):
                    # clamp negative to 0
                    val = src_mv[b, row, col]
                    bandstack_mv[dst_idx+b, row, col] = sqrtf(val) if val >= 0.0 else 0.0
        dst_idx += bands

        # log bands
        for b in range(bands):
            for row in prange(height, nogil=True, schedule="static"):
                for col in range(width):
                    val = src_mv[b, row, col]
                    # Check: log1p undefined for val <= -1; 
                    bandstack_mv[dst_idx + b, row, col] = log1pf(-0.99) if val <= -1.0 else log1pf(val)
        dst_idx += bands


cdef int _normalize_block_cy(
    float_t[:, :, :] block,
    float_t[:]       band_mins,
    float_t[:]       band_maxs
) nogil:
    """
    Normalize each band in a block: (x - min) / (max - min).
    Safe against division by zero.
    """
    cdef:
        size_t b, row, col, height, width, bands
        float_t denom, block_val

    bands = block.shape[0]
    height = block.shape[1]
    width = block.shape[2]

    for b in prange(bands, nogil=True, schedule="static"):
        denom = band_maxs[b] - band_mins[b]
        if denom == 0:
            for row in range(height):
                for col in range(width):
                    block[b, row, col] = 0.0
        else:
            for row in range(height):
                for col in range(width):
                    block_val = block[b, row, col]
                    block[b, row, col] = (block_val - band_mins[b]) / denom


cdef int _block_minmax_cy(
    float_t[:, :, :] block,      # shape (bands, h, w)
    float_t[:] band_mins,        # global running mins
    float_t[:] band_maxs         # global running maxs
) nogil:
    """
    Update band_mins and band_maxs with values from this block.
    Scans each band in one pass.
    """
    cdef:
        psize_t b, row, col
        psize_t bands = block.shape[0]
        psize_t height = block.shape[1]
        psize_t width = block.shape[2]
        float_t v, local_min, local_max

    for b in prange(bands, nogil=True, schedule="static"):  # parallel over bands
        local_min = band_mins[b]
        local_max = band_maxs[b]
        for row in range(height):
            for col in range(width):
                v = block[b, row, col]
                if v < local_min:
                    local_min = v
                elif v > local_max:
                    local_max = v

        band_mins[b] = local_min
        band_maxs[b] = local_max


cdef int[:,:] _generate_windows_cy(
    int img_height, 
    int img_width, 
    int win_height, 
    int win_width
    ):
    """
    Generate window offsets and sizes for an image.
    
    Returns:
        windows: int[:, :] memoryview of shape (total_windows, 4)
                 Each row: (row_off, col_off, actual_height, actual_width)
    """
    cdef:
        int n_rows = (img_height + win_height - 1) // win_height
        int n_cols = (img_width + win_width - 1) // win_width
        int total_windows = n_rows * n_cols
        int[:, :] win_mv
        np.ndarray[int, ndim=2] windows = np.empty((total_windows, 4), dtype=int)
    
    win_mv = windows

    cdef int row_idx, col_idx, win_idx
    cdef int row_off, col_off, actual_height, actual_width

    win_idx = 0
    for row_idx in range(n_rows):
        row_off = row_idx * win_height
        actual_height = win_height if row_off + win_height <= img_height else img_height - row_off

        for col_idx in range(n_cols):
            col_off = col_idx * win_width
            actual_width = win_width if col_off + win_width <= img_width else img_width - col_off

            # Fill window valuess
            win_mv[win_idx, 0] = row_off
            win_mv[win_idx, 1] = col_off
            win_mv[win_idx, 2] = actual_height
            win_mv[win_idx, 3] = actual_width

            win_idx += 1

    return win_mv


# ----------------------------------------
# Python-callable Band Creation Function
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
        raise ValueError(f"[BGP] image_block must be 3D (bands,h,w); got shape {(image_block.shape[0], image_block.shape[1])}")

    # Ensure float32 and contiguous layout (no copies if already correct)
    if image_block.dtype != np.float32 or not image_block.flags['C_CONTIGUOUS']:
        src = np.ascontiguousarray(image_block, dtype=np.float32)
    else:
        src = image_block

    # Source dims
    cdef: 
        psize_t bands  = <psize_t> src.shape[0]
        psize_t height = <psize_t> src.shape[1]
        psize_t width  = <psize_t> src.shape[2]

    # Convert bool to bint
    cdef bint full_syn = full_synthetic

    # Calculat output (#bands) size
    tot_num_bands = _total_bands_cy(bands, full_syn)

    # Preallocate output array (float32)
    band_stack = np.empty((int(tot_num_bands), height, width), dtype=np.float32)

    # Get fast C memoryviews 
    cdef float_t[:, :, :] src_mv = src
    cdef float_t[:, :, :] bs_mv = band_stack

    # noGIL kernel to create band_stack
    with nogil:
        _create_bands_from_block_cy(src_mv, bs_mv, full_syn)

    return band_stack



# --------------------------------------------------------------------------------------------
# Band Generation Process (BGP, Python callable)
# --------------------------------------------------------------------------------------------
def band_generation_process(
    input_image_paths:List[str],
    output_dir:str,
    window_shape:Tuple[int, int],
    full_synthetic:bint,
    max_workers:int|None,   # currently unused
    chunk_size:int,         # currently unused
    inflight:int,           # currently unused
    show_progress:bint,
):
    """
    The Band Generation Process. Generates synthetic, non-linear bands as combinations of existing bands. 
    Output is normalized range [0,1] per band.

    Overview:
      - Scans input to determine image shape / number of output bands.
      - Runs Pass 1 (generate un-normalized bands) in parallel via parallel_generate_streaming.
      - Runs Pass 2 (normalize) via parallel_normalize_streaming.
      - Removes temporary unnormalized file.

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
    cdef:
        bint full_syn = full_synthetic
        int img_height, img_width
        int win_height = <int> window_shape[0] 
        int win_width  = <int> window_shape[1]

    # (Hardcoded) output file names
    output_unorm_filename:str = "gen_band_unorm.tif" # un-normalized bands
    output_norm_filename:str = "gen_band_norm.tif"   # normalized bands
    unorm_path:str = join(output_dir, output_unorm_filename)


    # ==============================
    # Image size & window dimensions
    # ==============================
    with MultibandBlockReader(input_image_paths) as reader:
        img_height, img_width = reader.image_shape
        # Small test block to calc number of output bands 
        test_block = np.array(reader.read_multiband_block(((0, 0), (1, 1))), copy=True)
    
    # Ensure float32 and contigouous
    if test_block.dtype != np.float32 or not test_block.flags['C_CONTIGUOUS']:
        test_block = np.ascontiguousarray(test_block, dtype=np.float32)

    # Create bands to test output size
    sample_output = _create_bands_from_block(test_block, full_syn)
    num_output_bands = int(sample_output.shape[0])
    
    # free temporaries
    del test_block, sample_output
    

    # ============================================================
    # Generate windows
    # ============================================================
    # Generate array of window dimensions (num_windows, 4) 
    cdef int[:,:] win_mv = _generate_windows_cy(img_height, img_width, win_height, win_width)
    total_windows = win_mv.shape[0]


    # ============================================================
    # Initialize arrays for band stack and global min/max
    # ============================================================
    # Use small dummy block shape for initialization
    cdef:
        np.ndarray[np.float32_t, ndim=3] band_stack = np.empty((num_output_bands, win_height, win_width), dtype=np.float32)
        float_t[:, :, :] bstack_mv = band_stack

        np.ndarray[np.float32_t, ndim=1] band_mins = np.full(num_output_bands, np.inf, dtype=np.float32)
        np.ndarray[np.float32_t, ndim=1] band_maxs = np.full(num_output_bands, -np.inf, dtype=np.float32)
        float_t[:] mins_mv = band_mins
        float_t[:] maxs_mv = band_maxs



    # --------------------------------------------------------------------------------------------
    # Pass 1: Generate unnormalized output + global min/max for pass 2
    # --------------------------------------------------------------------------------------------   
    # Reader - original data blocks
    # Writer - unorm data to disk
    with MultibandBlockReader(input_image_paths) as reader, \
         MultibandBlockWriter(
            output_dir          = output_dir,
            output_image_shape  = (img_height, img_width),
            output_image_name   = output_unorm_filename,
            window_shape        = window_shape,
            num_bands           = num_output_bands,
            output_datatype     = np.float32
        ) as writer:                

        # Windows store along 0 dimension
        # Data stored alone 1 dimension
        for i in range(total_windows):
            # Build window
            row_off = win_mv[i,0] 
            col_off = win_mv[i,1]
            height  = win_mv[i,2]
            width   = win_mv[i,3]
            win = (row_off, col_off, height, width)
            # Read block from window
            block = reader.read_multiband_block(win)
            # Create synthetic bands
            new_block = _create_bands_from_block(block, full_syn)
            band_stack[:new_block.shape[0], :new_block.shape[1], :new_block.shape[2]] = new_block
            # Update min/max per band
            with nogil:
                _block_minmax_cy(bstack_mv, mins_mv, maxs_mv)
            # Write unnormalized block
            writer.write_block(band_stack, win)


    
    # --------------------------------------------------------------------------------------------
    # Pass 2: Normalize output
    # --------------------------------------------------------------------------------------------
    # Instantiate block memory view
    cdef float_t[:, :, :] block_mv
    
    # Reader - unorm block
    # Writer - norm data to disk
    with MultibandBlockReader([unorm_path]) as reader, \
         MultibandBlockWriter(
            output_dir          = output_dir,
            output_image_shape  = (img_height, img_width),
            output_image_name   = output_norm_filename,
            window_shape        = window_shape,
            num_bands           = num_output_bands,
            output_datatype     = np.float32
        ) as writer:

        for i in range(win_mv.shape[0]):
            # Build window
            row_off, col_off, height, width = win_mv[i,0], win_mv[i,1], win_mv[i,2], win_mv[i,3]
            win = (row_off, col_off, height, width)
            # Read block from window
            block = reader.read_multiband_block(win)
            # Efficient memory views to data
            block_mv = block

            # Normalize blocks with C-kernels
            with nogil:
                _normalize_block_cy(block_mv, mins_mv, maxs_mv)
            # Writes block to disk
            writer.write_block(block, win)
                


    # ======================
    # Cleanup Temporary File
    # ======================
    rm(unorm_path) 
    