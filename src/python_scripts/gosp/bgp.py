"""bgp.py: Band Generation Process, crated new non-linear bondinations of existing bands"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "2.2.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
from typing import Tuple, List
from .rastio import *
from ..utils.fileio import rm
from ..utils.parallel import parallel_normalize_streaming, parallel_generate_streaming



# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]
ImageBlock = np.ndarray


# --------------------------------------------------------------------------------------------
# Band Generation Process (BGP)
# --------------------------------------------------------------------------------------------
# @njit(parallel=True, fastmath=True, cache=True)
# @njit
def _create_bands_from_block(
    image_block: ImageBlock,
    use_sqrt: bool,
    use_log: bool
) -> np.ndarray:
    """
    Creates new, non-linear bands from existing bands for the ATDCA algorithm.

    Args:
        image_block (np.ndarray): A 3D numpy array representing a block of the image,
                                  with shape (bands, height, width).
        use_sqrt (bool): Flag to indicate if sqrt bands should be generated.
        use_log (bool): Flag to indicate if log bands should be generated.

    Returns:
        np.ndarray: A 3D numpy array containing old and the newly generated correlated bands,
                    with shape (new_bands, height, width) where (height, width) are
                    determined by the original image_block.
    """
        
    # Pre-allocate output size
    num_bands, height, width = image_block.shape 
    
    tot_num_bands = num_bands                           # original
    tot_num_bands += (num_bands * (num_bands - 1)) // 2 # correlation
    if use_sqrt: tot_num_bands += num_bands             # sqrt
    if use_log: tot_num_bands += num_bands              # log
    
    new_bands = np.empty((tot_num_bands, height, width), dtype=image_block.dtype) 

    idx = 0 # correctly stack new bands to output
    
    # original
    for band in range(num_bands):
        new_bands[idx,:,:] = image_block[band,:,:]
        idx+=1

    # correlation
    for band_a_idx in range(num_bands):
        for band_b_idx in range(band_a_idx + 1, num_bands): # avoids duplicate combinations
            new_bands[idx,:,:] = image_block[band_a_idx,:,:] * image_block[band_b_idx,:,:]
            idx+=1

    # sqrt 
    if use_sqrt:
        for band in range(num_bands):
            new_bands[idx,:,:] = np.sqrt(image_block[band,:,:])
            idx+=1
            
    # log
    if use_log:
        for band in range(num_bands):
            new_bands[idx,:,:] = np.log1p(image_block[band,:,:])
            idx+=1

    return new_bands


def band_generation_process(
    input_image_paths:List[str],
    output_dir:str,
    window_shape:Tuple[int,int],
    use_sqrt:bool,
    use_log:bool,
    max_workers:int|None = None,
    chunk_size:int = 4,
    inflight:int = 2,
    show_progress:bool = True
    
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
    dummy_block = input_dataset.read_multiband_block(((0, 0), (5,5)))
    sample_bands = _create_bands_from_block(dummy_block, use_sqrt, use_log)
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
            h = min(win_height, src_height - row_off)
            w = min(win_width, src_width - col_off)
            windows.append(((row_off, col_off), (h, w)))

    # Write unnormalized output, collect global stats
    with MultibandBlockWriter(
        output_path=output_dir,
        output_image_shape=input_shape,
        output_image_name=output_unorm_filename,
        num_bands=num_output_bands,
        output_datatype=np.float32,
    ) as writer:
        band_stats = parallel_generate_streaming(
            input_paths=input_image_paths,          # list[str]; one multiband or many single-band
            windows=windows,
            writer=writer,
            func_module="python_scripts.atdca.bgp", # module path where _create_bands_from_block lives
            func_name="_create_bands_from_block",   # function name, without parentheses
            use_sqrt=use_sqrt,
            use_log=use_log,
            max_workers=max_workers,                       
            chunk_size=chunk_size,
            inflight=2,                             # tune for RAM vs throughput
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
            h = min(win_height, src_height - row_off)
            w = min(win_width, src_width - col_off)
            windows.append(((row_off, col_off), (h, w)))

    # Open the writer
    with MultibandBlockWriter(
        output_path=output_dir,
        output_image_shape=input_shape,
        output_image_name=output_norm_filename,
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
            max_workers=max_workers,  # use all cores
            inflight=inflight,               # cap RAM: ~workers*inflight blocks in memory
            chunk_size=chunk_size,
            show_progress=show_progress
        )
                    
    rm(unorm_path) # delete unnorm data
    
    




