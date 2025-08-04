"""rastio.py: Handles I/O of raster data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from numpy import transpose, newaxis, ndarray, float32, concatenate, dtype
from warnings import warn
from rasterio import open
from rasterio.windows import Window
from typing import List, Tuple, Callable
from os.path import exists, dirname
from os import makedirs


# --------------------------------------------------------------------------------------------
# Reader (Input)
# --------------------------------------------------------------------------------------------
def get_virtual_multiband_reader(band_paths: List[str]) -> Callable:
    """
    Returns a reader that stacks multiple single-band images into a virtual multiband image.

    Args:
        band_paths (List[str]): List of paths to single-band TIFF files.

    Returns:
        Callable: Callable image data reader function. Reader returns -> Tuple|Array|None
    """
    datasets = [open(p) for p in band_paths]

    def _multiband_reader_func(window:str|List[tuple]) -> Tuple|ndarray|None:
        """
        Returns one of two outputs, dependent on input.
        Returns the shape of the reader if passed "window_shape" i.e. (height, width)
        Returns a block of data in a given window if passed 

        Args:
            window (str | List[tuple]): "window_shape" or current window of processing image.

        Returns:
            tuple|np.ndarray|None: Image shape: (height, width) 
                                   Block of data: (bands, height, width) 
                                   Error: None
        """
        if window == "window_shape":
            return datasets[0].shape

        (row_off, col_off), (height, width) = window
        full_window = Window(col_off, row_off, width, height) #type:ignore
        
        blocks = []
        for dataset in datasets:
            try:
                blocks.append(dataset.read(1, window=full_window))
            except Exception:
                warn("Failed to read window, returning None.")
                return None

        # Check for invalid blocks and return None
        if not all(isinstance(block, ndarray) and block.size > 0 for block in blocks):
            return None

        # Stack the bands and transpose to (bands, height, width)
        return transpose(
                    concatenate([block[newaxis, :, :] for block in blocks], axis=0), 
                    (0, 1, 2)
                ).astype(float32)
        
    return _multiband_reader_func


# --------------------------------------------------------------------------------------------
# Writer (Output)
# --------------------------------------------------------------------------------------------
def get_block_writer(
    output_path: str,
    image_shape: Tuple[int, int],
    dtype:type = float32,
    profile_template: dict|None = None
):
    """
    Returns a writer that handles writing blocks to an output raster file.
    This version dynamically determines the number of output bands from the first block
    it receives, creating a more robust and flexible pipeline.

    Args:
        output_path (str): The path to the output raster file.
        image_shape (Tuple[int, int]): The dimensions of the entire image.
        dtype (type, optional): The data type of the output raster. Defaults to float32.
        profile_template (dict, optional): A rasterio profile to use for metadata. Defaults to None.

    Returns:
        Callable: A function that writes a block of data to the output raster.
    """
    # Ensure output path is valid, creating the directory if it doesn't exist
    if not exists(output_path):
        makedirs(dirname(output_path), exist_ok=True)
    
    # Close this file after the writer is done
    dataset = None

    def _block_writer_with_setup(window, block):
        nonlocal dataset
        
        # This will only run on the very first call.
        if dataset is None:
            # Determine the number of bands from the block shape
            num_bands, _, _ = block.shape
            
            # Image data
            image_height, image_width = image_shape
        
            # TIFF profile ("structure") 
            profile = {
                "driver": "GTiff", # GeoTIFF supports 4+ GB TIFF files
                "height": image_height,
                "width": image_width,
                "count": num_bands,  # Determined from the block
                "dtype": dtype,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "interleave": "band",
                "BIGTIFF": "YES"  # Enables 4+ GB filesize
            }
        
            # Update metadata if provided
            if profile_template:
                profile.update({key: value for key, value in profile_template.items() if key in profile})
        
            # Import raster with optional profile data
            dataset = open(output_path, "w", **profile)

        (row_off, col_off), (height, width) = window
        
        # We assume the incoming block is of shape (bands, height, width)
        dataset.write(block, window=Window(col_off, row_off, width, height)) #type:ignore

    return _block_writer_with_setup




