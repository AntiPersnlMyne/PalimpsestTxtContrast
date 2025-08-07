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
# Reader (Input)
# --------------------------------------------------------------------------------------------
class MultibandBlockReader:
    """
    A class for reading multi-band raster datasets in blocks (windows). Each band
    is expected to be its own file.

    Attributes:
        filenames (List[str]): A list of paths to the raster file.
        tile_size (Tuple[int, int]): The size of each block to read.  Defaults to (256, 256).
        srcs (List[rasterio.DatasetReader]): Dataset objects (images), one for each input file. 
    """
    
    def __init__(self, filenames: List[str], window_size: Tuple[int, int] = (256, 256)):
        """
        Initializes the MultiBandBlockReader object.

        Args:
            filenames (List[str]): A list of paths to the raster files.
            tile_size (Tuple[int, int]): The size of each block to read.
        """
        self.filenames = filenames
        self.tile_size = window_size
        self.srcs = []
        for filename in self.filenames:
            try:
                src = rasterio.open(filename, 'r')
                self.srcs.append(src)
            except rasterio.RasterioIOError as e:
                raise Exception(f"Error opening {filename}: {e}") # Re-raise the exception to halt execution
        
    def __enter__(self):
        """
        Allows use of the MultiBandBlockReader as a context manager (using 'with').
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes all open raster files.
        """
        for src in self.srcs:
            src.close()
            

    def __del__(self):
        """
        Closes all open raster files when object is no longer referenced
        """
        for src in self.srcs:
            src.close()
            
    
    def image_shape(self) -> Tuple[int, int]:
        return self.srcs[0].shape  # (height, width)
            
            
    def read_multiband_block(
        self,
        window:WindowType
        ) -> np.ndarray:
        """
        Reads a block of data from each raster file and combines them into a single multi-band array.

        Args:
            window (WindowType): Region of input raster to pull data.

        Returns:
            np.ndarray: A multiband Numpy array representing the block of data 
            with shape (height, width, bands), where (height, width) defined by window.
        """
        
        # Block dimensions
        num_bands = len(self.srcs)
        (row_off, col_off), (block_height, block_width) = window

        multi_band_block = np.empty((num_bands, block_height, block_width), dtype=np.float32)

        for band_idx, src in enumerate(self.srcs):
            try:
                block = src.read(1, window=Window(col_off, row_off, block_width, block_height)) #type:ignore
                multi_band_block[band_idx,:,:] = block # assumes band-major
            except Exception as e:
                raise Exception(f"[rastio] Error reading band {band_idx}:\n{e}")

        return multi_band_block



# --------------------------------------------------------------------------------------------
# Writer (Output)
# --------------------------------------------------------------------------------------------
class MultibandBlockWriter:
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
        makedirs(self.output_path, exist_ok=True)
        
        self.dataset = rasterio.open(self.output_path + '/' + self.output_name, "w", **self.profile)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes dataset after finished writing to it"""
        if self.dataset:
            self.dataset.close() 

    def write_block(self, window: WindowType, block: np.ndarray) -> None:
        (row_off, col_off), (height, width) = window
        expected_shape = (block.shape[0], height, width)
        if block.shape != expected_shape:
            raise ValueError(f"Block shape {block.shape} does not match expected shape {expected_shape} for window {window}")
        
        if self.dataset:
            # Check if expected band-count matches block's band-count
            if self.dataset.count != block.shape[0]:
                # Update band count in dataset metadata and reopen with correct count
                self.dataset.close()
                self.profile["count"] = block.shape[0] # ( [0] bands, [1] height, [2] width)
                self.dataset = rasterio.open(self.output_path + '/' + self.output_name, "w", **self.profile)

            win = Window(col_off, row_off, width, height) #type:ignore

            self.dataset.write(block, window=win)



