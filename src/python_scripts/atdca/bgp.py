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
from typing import Callable, Union, Tuple, List, Any
from numba import njit
from ..utils.math_utils import normalize_data 
import rastio
from ..utils.fileio import rm


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
WindowType = Tuple[Tuple[int, int], Tuple[int, int]]
ImageBlock = np.ndarray
ImageShape = Tuple[int,int]
BlockShape = Tuple[int,int]

ImageReader = Callable[[Union[str, WindowType]], Union[ImageBlock, ImageShape, None]]
ImageWriter = Callable[[WindowType, ImageBlock], None]


# --------------------------------------------------------------------------------------------
# Band Generation Process (BGP)
# --------------------------------------------------------------------------------------------
# @njit(fastmath=True, cache=True)
def create_bands_from_block(
    image_block: ImageBlock,
    use_sqrt: bool,
    use_log: bool
) -> List[np.ndarray]:
    """
    Creates new, non-linear bands from existing bands for the ATDCA algorithm.

    Args:
        image_block (np.ndarray): A 3D numpy array representing a block of the image,
                                  with shape (bands, height, width).
        use_sqrt (bool): Flag to indicate if sqrt bands should be generated.
        use_log (bool): Flag to indicate if log bands should be generated.

    Returns:
        np.ndarray: A 3D numpy array containing the newly generated correlated bands,
                    with shape (new_bands, height, width) where (height, width) are
                    determined by the original image_block.
    """
    
    assert image_block.shape[0] == image_block.shape[1], ("[BGP] Data into create_bands_from_block is expected to be band-major")
    
    # Band-major has bands as first dimension
    num_bands, _, _ = image_block.shape

    # Get the original bands
    original_bands = [image_block[i] for i in range(num_bands)]
    new_bands = original_bands

    # Cross-correlation bands
    # Uses `combinations` to prevent redundant cross-correlated bands
    for band_a_idx, band_b_idx in combinations(range(num_bands), 2):
        cross_correlation_band = image_block[band_a_idx,:,:] * image_block[band_b_idx,:,:]
        new_bands.append(cross_correlation_band)

    # Sqrt-transformed bands
    if use_sqrt:
        for band in original_bands:
            sqrt_band = np.sqrt(band)
            new_bands.append(sqrt_band)
            
    # Log-transformed bands
    if use_log:
        for band in original_bands:
            sqrt_band = np.log1p(band)
            new_bands.append(sqrt_band)

    return new_bands


def band_generation_process(
    input_image_paths:List[str],
    dst_dir:str,
    window_shape:Tuple[int,int] = (512,512),
    use_sqrt:bool = False,
    use_log:bool = False
    ) -> None:
    """
    The Band Generation Process. 
    Generates synthetic, non-linear bands as combinations of existing bands. 
    Creates bands by auto and cross-correlation, optionally add
    sqrt and log bands. 
    Operates in two passes. Pass one creates the data. Pass two normalizes.

    Args:
        input_image_paths (List[str]): List of paths to input images / multispectral data.
        dst_dir (str): Output directory of generated band image. 
        window_shape (Tuple[int,int], optional): Shape of each block to process. 
            Larger blocks proceess faster and use more memory. 
            Smaller block process slower with a smaller memory footprint. 
            Defaults to (512,512).
        use_sqrt (bool, optional): If True, generate bands using square root. Defaults to False.
        use_log (bool, optional): If True, generate bands using log base 10. Defaults to False.
    """
    # Create dataset from input data
    # NOTE: Assumes few enough bands to store on RAM
    # NOTE: Conversion of data to band-major is handles by read_window_data
    input_dataset = rastio.create_dataset_from_bands(input_image_paths)
    
    # Get image and window dimensions
    input_shape = input_dataset[0].shape
    input_img_height, input_img_width = input_shape
    window_height, window_width = window_shape
    
    # Initalize normalization variables
    global_min = np.inf
    global_max = -np.inf
    

    # Create (un normalized) output dataset
    output_unorm_filename = "gen_band_unorm.tif"
    output_unorm_dataset = rastio.BlockWriterDataset(output_path=dst_dir, 
                                               output_image_shape=input_shape, 
                                               output_image_name=output_unorm_filename,
                                               output_datatype=np.float32)
    
    with output_unorm_dataset as dst:
        print("[BGP] Generating new bands and calculating global min/max ...")
        for row_off in tqdm(range(0, input_img_height, window_height), desc="[BGP] First pass", colour="CYAN"):
            for col_off in range(0, input_img_width, window_width):
                
                # Prevent block from accessing out-of-bounds
                actual_height = min(window_height, input_img_height - row_off)
                actual_width = min(window_width, input_img_width - col_off)
        
                # Set new window coordinates
                window = ((row_off, col_off), (actual_height, actual_width))
        
                # Set block of input data to process
                input_block = rastio.read_window_data(
                    dataset=input_dataset, 
                    window=window)
                
                # Reorder data to band-major i.e. bands as first dim: (height, width, bands) -> (bands, height, width)
                np.transpose(input_block, (2, 0, 1))
                
                # Create new bands
                new_bands = create_bands_from_block(
                    image_block=input_block, 
                    use_sqrt=use_sqrt, 
                    use_log=use_log)

                # Update global min/max 
                local_min, local_max = np.min(new_bands), np.max(new_bands)
                global_min = min(local_min, global_min)
                global_max = max(local_max, global_max)
                
                # Reorder data to row-major i.e. heigh as first dim: (bands, height, width) -> (height, width, bands)
                np.transpose(new_bands, (1, 2, 0))
                np.transpose(input_block, (1, 2, 0))
                
                # Concatenate old and new data to output
                dst.write(np.concatenate((input_block, new_bands), axis=2))
        
        
        # Create output dataset
        output_norm_filename = "gen_band_norm.tif"
        output_norm_dataset = rastio.BlockWriterDataset(output_path=dst_dir, 
                                                output_image_shape=input_shape, 
                                                output_image_name=output_norm_filename,
                                                output_datatype=np.float32)
    
    # Free memory
    del input_dataset
    
    # Normalize data 
    with output_norm_dataset as dst:
        print("[BGP] Normalizing the data ...") 
        for row_off in tqdm(range(0, input_img_height, window_height), desc="[BGP] Second pass", colour="WHITE"):
            for col_off in range(0, input_img_width, window_width):
                
                # Prevent block from accessing out-of-bounds
                actual_height = min(window_height, input_img_height - row_off)
                actual_width = min(window_width, input_img_width - col_off)
        
                # Set new window coordinates
                window = ((row_off, col_off), (actual_height, actual_width))
        
                # Set block of input data to process
                unnorm_block = rastio.read_window_data(
                    dataset=output_unorm_dataset, 
                    window=window)
                
                norm_block = normalize_data(unnorm_block,
                                            min_val=global_min,
                                            max_val=global_max)
                
                dst.write(norm_block)
                
                
    # Cleanup temporary data
    rm(dst_dir + '/' + output_unorm_filename)
        
        
        












