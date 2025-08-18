# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False


"""bgp.py: Band Generation Process, creates new non-linear bondinations of existing bands"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from typing import Tuple, List

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
__version__ = "3.1.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
# Rasterio data chunk
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]


# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
def _expected_total_bands(int n, bint full_synthetic) -> int:
    """Returns expected output size (i.e. number of bands) from band generation process"""
    cdef int total = n + (n * (n - 1)) // 2
    if full_synthetic: total += 2*n
    return total



# --------------------------------------------------------------------------------------------
# Band Generation Process (BGP)
# --------------------------------------------------------------------------------------------
def _create_bands_from_block(
    image_block:np.ndarray,
    full_synthetic:bint,
) -> np.ndarray:
    """
    Creates new, non-linear bands from existing bands for the GOSP algorithm.

    Args:
        image_block (np.ndarray): 
            A 3D numpy array representing a block of the image,
            with shape (bands, height, width).
        use_sqrt (bint): 
            Flag to indicate if sqrt bands should be generated.
        use_log (bint): 
            Flag to indicate if log bands should be generated.

    Returns:
        np.ndarray: 
            A 3D numpy array containing old and the newly generated correlated bands,
            with shape (new_bands, height, width) where (height, width) are
            determined by the original image_block.
    """
    cpdef np.ndarray[float32_t, ndim=3] _create_bands_from_block(
        np.ndarray[float32_t, ndim=3] image_block,
        bint full_synthetic):

    cdef Py_ssize_t src_bands = image_block.shape[0]
    cdef Py_ssize_t src_height = image_block.shape[1]
    cdef Py_ssize_t src_width  = image_block.shape[2]
    cdef int total = _expected_total_bands(<int>src_bands, full_synthetic)

    cdef np.ndarray[float32_t, ndim=3] band_stack = np.empty(
        (total, src_height, src_width), dtype=np.float32
    )

    cdef float32_t[:, :, :] src_view = image_block
    cdef float32_t[:, :, :] stack_view = band_stack

    cdef Py_ssize_t idx = 0
    # Copy original bands
    stack_view[idx:idx+src_bands, :, :] = src_view
    idx += src_bands

    # Pairwise correlations
    cdef int band, count
    for band in range(src_bands - 1):
        count = src_bands - 1 - band
        stack_view[idx:idx+count, :, :] = src_view[band, :, :] * src_view[band+1:, :, :]
        idx += count

    if full_synthetic:
        # sqrt
        np.sqrt(src_view, out=stack_view[idx:idx+src_bands, :, :])
        idx += src_bands
        # log1p
        np.log1p(src_view, out=stack_view[idx:idx+src_bands, :, :])
        idx += src_bands

    if idx != total:
        raise AssertionError(f"[BGP] Created bands #={idx} does not match expected={total}")
    
    return band_stack


def band_generation_process(
    input_image_paths:List[str],
    output_dir:str,
    window_shape:Tuple[int,int],
    full_synthetic:bint,
    max_workers:int|None,
    chunk_size:int,
    inflight:int,
    show_progress:bint,
    ) -> None:
    """
    The Band Generation Process. Generates synthetic, non-linear bands as combinations of existing bands. 
    Output is normalized range [0,1] per band.

    Args:
        input_image_paths (List[str]): List of paths to input images / multispectral data.
        dst_dir (str): Output directory of generated band image. 
        window_shape (Tuple[int,int]): Shape of each block to process. 
            Larger blocks proceess faster and use more memory. 
            Smaller block process slower with a smaller memory footprint. 
        use_sqrt (bool): If True, generate bands using square root.
        use_log (bool): If True, generate bands using log base 10.
        max_workers (int|None, optional): Number of cores for paralellization. If None, defaults to number of processors on the machine.
            i.e. more workers = more fast. Defaults to None.
        chunk_size (int, optional): How many windows of data the program can parallelize at once. 
            i.e. more chunks = more fast. Try 8 or 16 if RAM allows. Defaults to 4.
        inflight (int, optional): Controls memory footprint. At most `inflight * max_workers` blocks in RAM. Defaults to 2.
        verbose (bool): If true, shows progress bars.
    """
        
    # Initial scan to determine band count and output shape
    input_dataset = MultibandBlockReader(input_image_paths)
    input_shape = input_dataset.image_shape()
    src_height, src_width = input_shape
    win_height, win_width = window_shape    
    
    # Determine number of bands up front by using a preview block
    dummy_block = input_dataset.read_multiband_block(((0, 0), (1,1)))
    sample_bands = _create_bands_from_block(dummy_block, full_synthetic)
    num_output_bands = sample_bands.shape[0]
    del dummy_block, sample_bands # free memory
    
    # initalize band-wise norm variables
    band_mins = np.full(num_output_bands, np.inf, dtype=np.float32)
    band_maxs = np.full(num_output_bands, -np.inf, dtype=np.float32)
    
    output_unorm_filename = "gen_band_unorm.tif"
    output_norm_filename = "gen_band_norm.tif"
    
    # --------------------------------------------------------------------------------------------
    # Pass 1: Generate unnormalized output
    # --------------------------------------------------------------------------------------------
    # Build all possible windows
    windows = []
    for row_off in range(0, src_height, win_height):
        for col_off in range(0, src_width, win_width):
            actual_height = min(win_height, src_height - row_off)
            actual_width = min(win_width, src_width - col_off)
            windows.append(((row_off, col_off), (actual_height, actual_width)))

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
    

# Optional quick self-check (not used by pipeline)
if __name__ == "__main__":
    # Tiny sanity test ensures counts & ordering fill
    rng = np.random.default_rng(0)
    block = rng.random((4, 3, 2), dtype=np.float32)  # (bands, H, W)
    for full_synthetic in (False, True):
        out = _create_bands_from_block(block, full_synthetic)
        expected = _expected_total_bands(4, full_synthetic)
        assert out.shape == (expected, 3, 2)
            