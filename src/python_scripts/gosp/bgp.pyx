# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False


"""bgp.py: Band Generation Process, creates new non-linear bondinations of existing bands"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from typing import Tuple, List

# External helpers (already compiled as cdef/cpdef in the same module)
cdef extern from "your_helpers_module.h":
    # MultibandBlockReader/Writer are already cdef classes with the
    # context‑manager interface you used in Python.
    pass  # (your actual declarations go here)

from .rastio import MultibandBlockReader, MultibandBlockWriter
from ..utils.fileio import rm
from .parallel import parallel_normalize_streaming, parallel_generate_streaming


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
np.import_array()

ctypedef tuple[tuple[int,int], tuple[int,int]] window_t # [(col_off, row_off), (width, height)]
ctypedef np.float32 float_t
ctypedef np.uint16 uint16_t


# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
cdef uint16_t _expected_total_bands(uint16_t n, bint full_synthetic) noexcept nogil:
    """Returns expected output size (i.e. number of bands) from band generation process"""
    cdef uint16_t total = n + (n * (n - 1)) // 2
    if full_synthetic: total += 2*n
    return total



# --------------------------------------------------------------------------------------------
# Band Generation Process (BGP)
# --------------------------------------------------------------------------------------------
cdef np.ndarray[float_t, cast=False] _create_bands_from_block (
    np.ndarray[float_t, cast=False] image_block,
    bint full_synthetic
    ) noexcept nogil:
    """
    Creates new, non-linear bands from existing bands for the GOSP algorithm.

    Args:
        image_block (np.ndarray, float32): 
            A 3D numpy array representing a block of the image,
            with shape (bands, height, width).
        use_sqrt (bint): 
            Flag to indicate if sqrt and log bands should be generated.

    Returns:
        np.ndarray: 
            A 3D numpy array containing old and the newly generated correlated bands,
            with shape (new_bands, height, width) where (height, width) are
            determined by the original image_block.
    """
    cdef: 
        uint16_t src_bands = <uint16_t>image_block.shape[0]     # type: ignore[reportGeneralTypeIssues]
        uint16_t src_height = <uint16_t>image_block.shape[1]    # type: ignore[reportGeneralTypeIssues]
        uint16_t src_width  = <uint16_t>image_block.shape[2]    # type: ignore[reportGeneralTypeIssues]
        uint16_t total = _expected_total_bands(src_bands, full_synthetic)

    cdef np.ndarray[float_t, ndim=3] band_stack = \
        np.empty((total, src_height, src_width), dtype=float_t)

    cdef float_t[:, :, :] src_view = image_block  # type:ignore[reportGeneralTypeIssues]
    cdef float_t[:, :, :] stack_view = band_stack # type:ignore[reportGeneralTypeIssues]
    # Ensure bands dont overlap when written to array
    cdef uint16_t idx = <uint16_t> 0 
    
    # ============================================================================================
    # Original Bands
    # ============================================================================================
    stack_view[idx:idx+src_bands, :, :] = src_view # type:ignore[reportGeneralTypeIssues]
    idx += src_bands

    # ============================================================================================
    # Pairwise Correlations
    # ============================================================================================\
    cdef: 
        int band  # int compatible with range
        uint16_t count

    for band in range(src_bands - 1):
        count = src_bands - 1 - band
        stack_view[idx:idx+count, :, :] = src_view[band, :, :] * src_view[band+1:, :, :] # type:ignore[reportGeneralTypeIssues]
        idx += count

    # ============================================================================================
    # "full_synthetic" -- ln and sqrt
    # ============================================================================================\
    if full_synthetic:
        # sqrt
        np.sqrt(src_view, out=stack_view[idx:idx+src_bands, :, :]) # type:ignore[reportGeneralTypeIssues]
        idx += src_bands
        # log1p
        np.log1p(src_view, out=stack_view[idx:idx+src_bands, :, :]) # type:ignore[reportGeneralTypeIssues]
        idx += src_bands

    if idx != total:
        raise AssertionError(f"[BGP] Created bands #={idx} does not match expected={total}")
    
    return band_stack


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
    cdef MultibandBlockReader input_dataset = MultibandBlockReader(input_image_paths)
    cdef tuple[np.uint16_t,np.uint16_t] input_shape = input_dataset.image_shape()
    # Extract dimensions
    cdef np.uint16_t src_height = input_shape[0]
    cdef np.uint16_t src_width = input_shape[1]
    cdef np.uint16_t win_height = window_shape[0]
    cdef np.uint16_t win_width = window_shape[1]     
    
    # ============================================================
    # Peek one-pixel block to calculate tot num output bands
    # ============================================================
    cdef np.ndarray[cfloat, ndim=3] tiny_block
    tiny_block = input_dataset.read_multiband_block(((0, 0), (1, 1)))
    cdef np.ndarray[cfloat, ndim=2] sample_bands
    sample_bands = _create_bands_from_block(tiny_block, full_synthetic)
    cdef int num_output_bands = sample_bands.shape[0] #type:ignore[CythonreportGeneralTypeIssues]
    # free the temporary data
    del tiny_block, sample_bands
    
    # ============================================================
    # Arrays to hold global min/max for later normalization
    # ============================================================
    cdef np.ndarray[cfloat, ndim=1] np_band_mins = np.full(num_output_bands, np.inf, dtype=np.float32)
    cdef np.ndarray[cfloat, ndim=1] np_band_maxs = np.full(num_output_bands, -np.inf, dtype=np.float32)
    # Memory view for faster access
    cdef float[:] band_mins = np_band_mins #type:ignore[CythonreportGeneralTypeIssues]
    cdef float[:] band_maxs = np_band_maxs #type:ignore[CythonreportGeneralTypeIssues]
    
    # ============================================================
    # Preallocate list of windows
    # ============================================================
    cdef int n_rows = (src_height + win_height - 1) // win_height
    cdef int n_cols = (src_width  + win_width  - 1) // win_width
    cdef int total_windows = n_rows * n_cols

    cdef list windows = [None] * total_windows # empty array of Nones
    cdef int win_idx = 0

    cdef int row_off, col_off, actual_height, actual_width
    for row_off in range(0, src_height, win_height):
        for col_off in range(0, src_width, win_width):
            actual_height = win_height if row_off + win_height <= src_height else src_height - row_off
            actual_width  = win_width  if col_off + win_width  <= src_width  else src_width  - col_off
            windows[win_idx] = ((row_off, col_off), (actual_height, actual_width))
            win_idx += 1


    output_unorm_filename = "gen_band_unorm.tif"
    output_norm_filename = "gen_band_norm.tif"
    
    # --------------------------------------------------------------------------------------------
    # Pass 1: Generate unnormalized output
    # --------------------------------------------------------------------------------------------

    # Write unnormalized output, collect global stats
    with MultibandBlockWriter(
        output_dir=output_dir,
        output_image_shape=input_shape,
        output_image_name=output_unorm_filename,
        window_shape=window_shape,
        num_bands=num_output_bands,
        output_datatype=np.float32,
    ) as writer:
        band_stats = parallel_generate_streaming(
            input_paths=input_image_paths,
            windows=windows,
            writer=writer,
            func_module="python_scripts.gosp.bgp", 
            func_name="_create_bands_from_block",   
            full_synthetic=full_synthetic,
            max_workers=max_workers,                       
            chunk_size=chunk_size,
            inflight=inflight,                            
            show_progress=show_progress
        )

    # Set bands' min/max
    band_mins, band_maxs = band_stats[0], band_stats[1]

    
    # --------------------------------------------------------------------------------------------
    # Pass 2: Normalize output
    # --------------------------------------------------------------------------------------------
    unorm_path = f"{output_dir}/{output_unorm_filename}"
    
    # Get all possible windows 
    windows = []
    for row_off in range(0, src_height, win_height):
        for col_off in range(0, src_width, win_width):
            actual_height = min(win_height, src_height - row_off)
            actual_width = min(win_width, src_width - col_off)
            windows.append(((row_off, col_off), (actual_height, actual_width)))

    # Open the writer
    with MultibandBlockWriter(
        output_dir=output_dir,
        output_image_shape=input_shape,
        output_image_name=output_norm_filename,
        window_shape=window_shape,
        num_bands=num_output_bands,
        output_datatype=np.float32
    ) as writer:
        # Stream in parallel: workers read+normalize, parent writes
        parallel_normalize_streaming(
            unorm_path=unorm_path,
            windows=windows,
            band_mins=band_mins,
            band_maxs=band_maxs,
            writer=writer,
            max_workers=max_workers, 
            inflight=inflight,
            chunk_size=chunk_size,
            show_progress=show_progress
        )
                    
    rm(unorm_path) # delete unnorm data
    
            