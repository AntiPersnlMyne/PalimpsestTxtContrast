"""rastio.py: Handles I/O of raster data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Production" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
import rasterio
from typing import List, Tuple, Sequence
from os import makedirs
from rasterio.windows import Window

WindowType = Tuple[Tuple[int, int], Tuple[int, int]]


# --------------------------------------------------------------------------------------------
# Imread by window (tcp.py)
# --------------------------------------------------------------------------------------------
def window_imread(filepaths: Sequence[str], window: WindowType) -> np.ndarray:
    """
    Reads a tile from a single window (bands, height, width).
    Handles both one multiband file (one path) or many single-band files (many paths).

    Args:
        filepaths (Sequence[str]): Path to a file, including filename and extension. 
            If given one path, assumes one single-band or one multiband.
            If given multiple paths, assumes many single-bands.
        window (WindowType): Dimensions and locaiton of window to read data from raster. Format: (row_off, col_off, width, height).

    Returns:
        np.ndarray: block (bands, height, width)
    """    
    # Assert input is not empty
    assert len(filepaths) != 0, "[rastio] window_imread: empty input file list"
    
    (row_off, col_off), (win_height, win_width) = window
        
    # One file optimized return
    if len(filepaths) == 1:
        with rasterio.open(filepaths[0], 'r') as src:
            return src.read(window=Window(col_off, row_off, win_width, win_height)) #type:ignore
     
    # Variables verify file shapes match across files
    with rasterio.open(filepaths[0], 'r') as src0:
        height0, width0 = src0.shape    
    
    # Preallocate output band-major order; size: (#-bands, height, width)
    total_bands:int = 0
    for file in filepaths:
        with rasterio.open(file) as src:
            total_bands += int(src.count)
    
    band_stack:np.ndarray = np.empty((total_bands, win_height, win_width), dtype=np.float32)
    
    idx:int = 0 # Ensure bands dont overlap
    for path in enumerate(filepaths):
        # Accepts single-band or multi-band
        with rasterio.open(path, 'r') as src:
            # Assert consistent image shape across inputs
            assert src.shape != (height0, width0), f"[rastio] All inputs must share shape. Got {src.shape} vs expected {(height0, width0)} for {src}"
            
            # Read in data as block
            block = src.read(window=Window(col_off, row_off, win_width, win_height)) #type:ignore
            b = block.shape[0] # b = number of bands
            
            # Add block to output, cvt float32 for downstream math
            band_stack[idx:idx+b] = block.astype(np.float32, copy=False) 
            
            idx += b
            
    return band_stack



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
        # Assert filepaths isn't empty
        assert filepaths is not None, f"[rastio] MultibandBlockReader: empty file list"
        
        # Read in all files from filepaths
        self.filepaths = filepaths
        self.srcs = []
        for filepath in self.filepaths:
            try:
                self.srcs.append(rasterio.open(filepath, 'r'))
            except rasterio.RasterioIOError as e:
                raise Exception(f"[rastio] MultibandBlockReader: Error opening {filepath}: {e}") 
        
        # Validate consistent data shape and compute total bands
        height0, width0 = self.srcs[0].shape
        for src in self.srcs[1:]:
            if src.shape != (height0, width0):
                raise ValueError(f"[rastio] Input shapes must match. Got {src.shape} vs {(height0, width0)} ({src.name})")
        self._shape = (height0, width0)
        self.total_bands = int(sum(src.count for src in self.srcs))
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes all open raster files"""
        for src in self.srcs:
            src.close()
            
    def __del__(self):
        """Closes all open raster files"""
        for src in self.srcs:
            try: src.close()
            except: pass # quietly exit
    
    def image_shape(self) -> Tuple[int, int]:
        return self._shape  # (height, width)
            
    def read_multiband_block(
        self,
        window:WindowType
        ) -> np.ndarray:
        """
        Reads a block of data from each raster file and combines them into a single multi-band array.

        Args:
            window (WindowType): Region of raster to pull data: ( (row_off, col_off), (win_height, win_width) )

        Returns:
            np.ndarray: A multiband Numpy array representing the block of data 
                with shape (bands, height, width), where (height, width) defined by window.
        """
        
        (row_off, col_off), (win_height, win_width) = window
        
        # Preallocate output
        band_stack = np.empty((self.total_bands, win_height, win_width), dtype=np.float32)
        
        # Add blocks to bandstack
        idx = 0
        for s in self.srcs:
            block = s.read(window=Window(col_off, row_off, win_width, win_height)) # type:ignore
            b = block.shape[0]
            band_stack[idx:idx+b] = block.astype(np.float32, copy=False)
            idx+=b

        # Check expected output size
        assert idx == self.total_bands, f"[rastio] Filled {idx} of expected {self.total_bands} bands; input changed?"
        
        return band_stack


# --------------------------------------------------------------------------------------------
# Writer (Output)
# --------------------------------------------------------------------------------------------
class MultibandBlockWriter:
    """
    A class that handles writing blocks of data to an output raster file.

    Args:
        output_path (str): The path to the output raster file.
        output_image_shape (Tuple[int, int]): The dimensions (rows, cols) of the output image.
        output_dtype (np.type, optional): The data type of the output raster.
    """
    
    def __init__(
        self, 
        output_dir:str, 
        output_image_shape:Tuple[int,int], 
        output_image_name:str, 
        window_shape:Tuple[int,int], 
        output_datatype,
        num_bands: int|None = None
    ):
        self.output_dir = output_dir
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
            "compress": None, # zstd if file-size becomes issue 
            "tiled": True,
            "blockxsize": self.window_shape[1], 
            "blockysize": self.window_shape[0],
            "interleave": "band",
            "BIGTIFF": "YES" # Enables 4+ GB files
        }

        # Check for valid output path for intermediate dataset file,
        # otherwise create it
        makedirs(self.output_dir, exist_ok=True) 
        self.dataset = rasterio.open(f"{self.output_dir}/{self.output_name}", "w", **self.profile)
        return self 

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes dataset after finished writing to it"""
        if self.dataset: self.dataset.close() 

    def write_block(self, window:WindowType, block:np.ndarray):
        """
        Write block of data to dataset

        Args:
            window (WindowType): Section of output dataset to write block to. Size: ( (row_off, col_off), (win_height, win_width) )
            block (np.ndarray): Block of data to be written. Size: (bands, win_height, win_width).
        """
        (row_off, col_off), (win_height, win_width) = window
        
        # Check expected write shape
        # size: (#-bands, height, width)
        expected_shape = (self.profile["count"], win_height, win_width)
        assert block.shape == expected_shape, f"[rastio] Shape mismatch: {block.shape} vs expected: {expected_shape}"

        win = Window(col_off, row_off, win_width, win_height) #type:ignore

        assert self.dataset, "[rastio] Attempted to write but dataset is not initialized"
        
        self.dataset.write(block, window=win, indexes=list(range(1, block.shape[0] + 1)))


