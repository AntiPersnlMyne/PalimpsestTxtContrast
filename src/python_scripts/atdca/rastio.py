"""rastio.py: Handles I/O of raster data"""

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
    A class for reading multi-band raster datasets in blocks (windows).
    Supports both single-band files (multiple files) and true multiband files.

    Attributes:
        filenames (List[str]): A list of paths to the raster file.
        tile_size (Tuple[int, int]): The size of each block to read.  Defaults to (256, 256).
        srcs (List[rasterio.DatasetReader]): Dataset objects (images), one for each input file. 
    """
    
    def __init__(self, filepaths: List[str]):
        """
        Initializes the MultiBandBlockReader object. Reads blocks of data from specifed window ("mask") of raster.

        Args:
            filepaths (List[str]): A list of path(s) to the raster files.
        """
        self.filepaths = filepaths
        self.srcs = []
        for filepath in self.filepaths:
            try:
                src = rasterio.open(filepath, 'r')
                self.srcs.append(src)
            except rasterio.RasterioIOError as e:
                raise Exception(f"[rastio] Error in MultibandBlockReader when opening {filepath}: {e}") # Re-raise the exception to halt execution
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes all open raster files."""
        for src in self.srcs:
            src.close()
            
    def __del__(self):
        """Closes all open raster files when object is no longer referenced"""
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
            window (WindowType): Region of raster to pull data: ( (row_off, col_off), (height, width) )

        Returns:
            np.ndarray: A multiband Numpy array representing the block of data 
                with shape (height, width, bands), where (height, width) defined by window.
        """
        
        # get window dimensions
        (row_off, col_off), (block_height, block_width) = window
        
        if len(self.srcs) == 1 and self.srcs[0].count > 1:
            # Single multi-band file
            try:
                block = self.srcs[0].read(
                    window=Window(col_off, row_off, block_width, block_height) #type:ignore
                )
                return block  # shape: (bands, height, width)
            except Exception as e:
                raise Exception(f"[rastio] Error reading multi-band block:\n{e}")

        else:
            # Multiple single-band files (assumes each src is 1-band)
            multi_band_block = np.empty((len(self.srcs), block_height, block_width), dtype=np.float32)
            for i, src in enumerate(self.srcs):
                try:
                    band = src.read(1, window=Window(col_off, row_off, block_width, block_height)) #type:ignore
                    multi_band_block[i] = band
                except Exception as e:
                    raise Exception(f"[rastio] Error reading band {i} from {src.name}:\n{e}")
            
            return multi_band_block # shape: (bands, height, width)


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
    
    def __init__(self, output_path, output_image_shape, output_image_name, window_shape:Tuple[int,int] = (512,512), num_bands: int|None = None, output_datatype=np.float32):
        self.output_path = output_path
        self.output_shape = output_image_shape
        self.output_name = output_image_name
        self.output_dtype = output_datatype
        self.window_shape = window_shape
        self.dataset = None 
        self.num_bands = num_bands or 1

    def __enter__(self):      
        self.profile = {
            "driver": "GTiff", # Supports 4+ GB TIFF files
            "height": self.output_shape[0],
            "width": self.output_shape[1],
            "count": self.num_bands, 
            "dtype": self.output_dtype,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": self.window_shape[1], 
            "blockysize": self.window_shape[0],
            "interleave": "band",
            "BIGTIFF": "YES" # Enables 4+ GB files
        }

        makedirs(self.output_path, exist_ok=True) # Check for valid output path - otherwise create it
        self.dataset = rasterio.open(self.output_path + '/' + self.output_name, "w", **self.profile)
        return self # return reference to dataset

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes dataset after finished writing to it"""
        if self.dataset:
            self.dataset.close() 

    def write_block(self, window:WindowType, block:np.ndarray):
        """
        Write block of data to dataset

        Args:
            window (WindowType): Section of output dataset to write block to. Size: ( (row_off, col_off), (win_height, win_width) )
            block (np.ndarray): Block of data to be written. Size: (bands, win_height, win_width).
        """
        # Set write parameters
        (row_off, col_off), (height, width) = window
        expected_shape = (self.profile["count"], height, width)
        self.profile["blockxsize"] = width
        self.profile["blockxsize"] = height
        
        assert block.shape == expected_shape, f"[rastio] Shape mismatch: {block.shape} vs expected: {expected_shape}"

        win = Window(col_off, row_off, width, height) #type:ignore
        indexes = list(range(1, block.shape[0] + 1)) # rasterio 1-indexes its bands

        if self.dataset:
            self.dataset.write(block, window=win, indexes=indexes)
        else:
            print(f"[rastio] Attempted to write but dataset is not initialized")



