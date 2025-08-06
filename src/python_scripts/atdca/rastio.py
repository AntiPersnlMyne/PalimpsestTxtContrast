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
import numpy as np
from warnings import warn
import rasterio
from typing import List, Tuple
from os.path import exists
from os import makedirs
from rasterio.windows import Window

WindowType = Tuple[Tuple[int, int], Tuple[int, int]]



# --------------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------------
def create_dataset_from_bands(input_bands:List[str]) -> List[np.ndarray]:
    """
    Creates dataset from list of input bands.
    A dataset is just a multi-channel image of all the data. 

    Args:
        input_bands (List[str]): List of paths to input band directories.

    Returns:
        List: Dataset.
    """
    # Optional efficiency:
    # chunk_size: 500 * 1024 * 1024
    # rasterio.open(path, chunked=True, block_size=chunk_size)
    
    # Keywords for rasterio.open() are supplied by the incomming data
    return [rasterio.open(path) for path in input_bands]


def read_window_data(
    dataset, 
    window:rasterio.windows.Window, #type:ignore
    bands:List[int]|None = None, 
    dtype:np.dtype|None = None
    ) -> np.ndarray:
    """
    Reads a block of data, from a given window, of a dataset. Returned data is row-major order.

    Parameters:
        dataset (rasterio.Dataset): The input dataset to read from.
        window (rasterio.Window): The window ("slice") of the dataset to read.
        bands (list, optional): Which bands to read. Indexing begins at 1. 
            E.x., with a 4 band image, `[1,3]` reads bands 1, 2 and omits 3, 4. 
            If None, reads all bands. 
            Defaults to None.
        dtype (numpy.dtype, optional): The numpy data type for the read window. 
            If None, defaults to float32. 
            Defaults to None.

    Returns:
        numpy.ndarray: The read window of data in row-major order (height, width, bands).

    Raises:
        Exception: If there is an error reading the raster data.
    """

    try:
        # Read the data from the dataset within the specified window and band indices
        return dataset.read(indexes=bands, window=window, out_dtype=dtype)

    except rasterio.RasterioIOError as e:
        raise Exception(f"[rastio] Error reading raster data:\n{e}")
    except Exception as e:
        raise Exception(f"[rastio] Error in read_window_data:\n{e}")



# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------
def _block_writer(window, block, dataset):
    """Writes a block of data to the raster dataset, at the specified window."""
    (row_off, col_off), (height, width) = window
    dataset.write(block, window=Window(col_off, row_off, width, height), indexes=0) #type:ignore


# Example Usage:
'''
with BlockWriterDataset('/example.tif', block, np.float32) as dst:
    data = np.ones((150, 250), dtype=np.float32)
    dst.write(data)
'''
class BlockWriterDataset:
    """
    A class that handles writing blocks of data to an output raster file.

    Args:
        output_path (str): The path to the output raster file.
        output_image_shape (Tuple[int, int]): The dimensions (rows, cols) of the output image.
        output_dtype (type, optional): The data type of the output raster. Defaults to float32.
    """
    
    def __init__(self, output_path, output_image_shape, output_image_name, output_datatype=np.float32):
        self.output_path = output_path
        self.output_shape = output_image_shape
        self.output_name = output_image_name
        self.output_dtype = output_datatype
        self.dataset = None 

    def __enter__(self):
        """Initializes the output file ("dataset") and returns writeable object"""
        
        num_bands = 1           # Lazy initalization finds the actual band number after block 1 is written
        blockxsize = 256        # Size of each processed chunk ("Window")
        blockysize = blockxsize # blocks must be square
        
        # Determine num_bands and create dataset on first write
        self.profile = {
            "driver": "GTiff", # Supports 4+ GB TIFF files
            "height": self.output_shape[0],
            "width": self.output_shape[1],
            "count": num_bands, 
            "dtype": self.output_dtype,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": blockxsize, 
            "blockysize": blockysize, # xsize must equal ysize 
            "interleave": "band",
            "BIGTIFF": "YES" # Enables 4+ GB files
        }
        
        # Check for valid output path - otherwise create it
        f_len = len(self.output_name)
        makedirs(self.output_path[-f_len], exist_ok=True)
        
        self.dataset = rasterio.open(self.output_path + '/' + self.output_name, "w", **self.profile)
        return self.dataset 

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes dataset after finished writing to it"""
        if self.dataset:
            self.dataset.close() 

    def write(self, block):
        """Writes a block of data to the output raster ("dataset")"""
        # (0,0) - Starting coordinate in output raster to begin writing
        # block.shape - size of data blocks that will be written per call of `write`
        window = (0, 0), block.shape
        # win - Origin and size of window where block will be written
        # bloc - block of data that will be written
        _block_writer(window, block, self.dataset)  # Call block writer



