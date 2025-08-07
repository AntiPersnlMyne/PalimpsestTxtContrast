"""bgp.py: Band Generation Process, crated new non-linear bondinations of existing bands"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
from itertools import combinations
from tqdm import tqdm
from typing import Callable, Union, Tuple, List
from numba import njit
from ..utils.math_utils import normalize_data 
from .rastio import *
from ..utils.fileio import rm
from os import system


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]
ImageBlock = np.ndarray


# --------------------------------------------------------------------------------------------
# Band Generation Process (BGP)
# --------------------------------------------------------------------------------------------
# @njit(fastmath=True, cache=True)
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
    
    # extract bands into list
    num_bands, _, _ = image_block.shape 
    original_bands = [image_block[band_idx,:,:] for band_idx in range(num_bands)]
    new_bands = original_bands.copy() # new_bands also stores the original 

    # cross-correlation bands
    for band_a_idx, band_b_idx in combinations(range(num_bands), 2):
        new_bands.append(image_block[band_a_idx,:,:] * image_block[band_b_idx,:,:])

    # optional, add sqrt and log bands
    if use_sqrt:
        new_bands.extend([np.sqrt(band) for band in original_bands])
    if use_log:
        new_bands.extend([np.log1p(band) for band in original_bands])

    return np.stack(new_bands, axis=0) # shape: (new_bands, height, width)


def band_generation_process(
    input_image_paths:List[str],
    output_dir:str,
    window_shape:Tuple[int,int],
    use_sqrt:bool,
    use_log:bool
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
    """
        
    # Initial scan to determine band count and output shape
    input_dataset = MultibandBlockReader(input_image_paths)
    input_shape = input_dataset.image_shape()
    src_height, src_width = input_shape
    win_height, win_width = window_shape
    
    # Determine number of bands up front by using a preview block
    preview_block = input_dataset.read_multiband_block(((0, 0), window_shape))
    sample_bands = _create_bands_from_block(preview_block, use_sqrt, use_log)
    num_output_bands = sample_bands.shape[0]
    del preview_block, sample_bands # free memory
    
    # # Reload dataset after reading preview (some readers may be errored)
    # input_dataset = MultibandBlockReader(input_image_paths, window_shape)
    
    # initalize band-wise norm variables
    band_mins = np.full(num_output_bands, np.inf, dtype=np.float32)
    band_maxs = np.full(num_output_bands, -np.inf, dtype=np.float32)
    
    output_unorm_filename = "gen_band_unorm.tif"
    output_norm_filename = "gen_band_norm.tif"
    
    # --------------------------------------------------------------------------------------------
    # Pass 1: Generate unnormalized output
    # --------------------------------------------------------------------------------------------
    with MultibandBlockReader(input_image_paths) as reader:
        with MultibandBlockWriter(
            output_path =        output_dir, 
            output_image_shape = input_shape, 
            output_image_name =  output_unorm_filename, 
            num_bands =          num_output_bands, 
            output_datatype =    np.float32
        ) as writer:
            print("[BGP] Generating new bands (pass 1) ...")
            for row_off in tqdm(range(0, src_height, win_height), desc="[BGP] First pass", colour="CYAN"):
                for col_off in range(0, src_width, win_width):
                    
                    # prevent block from accessing out-of-bounds
                    actual_height = min(win_height, src_height - row_off)
                    actual_width = min(win_width, src_width - col_off)
                    window = ((row_off, col_off), (actual_height, actual_width))
            
                    # create new bands from data block
                    block = reader.read_multiband_block(window=window)
                    new_bands = _create_bands_from_block(image_block=block, use_sqrt=use_sqrt, use_log=use_log)

                    # update global min/max per-band
                    # maximum - element-wise | max - along (height,width) per band
                    band_mins = np.minimum(band_mins, np.min(new_bands, axis=(1, 2)))
                    band_maxs = np.maximum(band_maxs, np.max(new_bands, axis=(1, 2)))
                    
                    # write block
                    writer.write_block(window=window, block=new_bands)
                    del block, new_bands  # free memory

    
    # --------------------------------------------------------------------------------------------
    # Pass 2: Normalize output
    # --------------------------------------------------------------------------------------------
    unorm_path = f"{output_dir}/{output_unorm_filename}"
    with MultibandBlockReader([unorm_path]) as reader:
        with MultibandBlockWriter(
            output_path =        output_dir, 
            output_image_shape = input_shape, 
            output_image_name =  output_norm_filename, 
            num_bands =          num_output_bands, 
            output_datatype =    np.float32
        ) as writer:
            print("[BGP] Normalizing bands (pass 2) ...") 
            for row_off in tqdm(range(0, src_height, win_height), desc="[BGP] Second pass", colour="CYAN"):
                for col_off in range(0, src_width, win_width):
                        
                    # prevent block from accessing out-of-bounds
                    actual_height = min(win_height, src_height - row_off)
                    actual_width = min(win_width, src_width - col_off)
                    window = ((row_off, col_off), (actual_height, actual_width))
            
                    # get data block and normalize
                    block = reader.read_multiband_block(window=window)
                    norm_block = np.empty_like(block)
                    for i in range(block.shape[0]):
                        norm_block[i] = normalize_data(block[i], min_val=band_mins[i], max_val=band_maxs[i])

                    # write block
                    writer.write_block(window=window, block=norm_block)
                    del block, norm_block # free memory    
                    
    rm(unorm_path) # free unnorm after norm written
    
    




