#!/usr/bin/env python3

"""rastio.py: Handles I/O of raster data"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from rasterio.vrt import buildvrt
from os import makedirs

import rasterio
from rasterio.windows import Window
cdef extern from "Python.h":
    pass  # silence “Python.h” warning when using only typed calls


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.3"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
np.import_array()

WindowType = tuple[tuple[int, int], tuple[int, int]]

ctypedef np.float32_t float_t
ctypedef np.uint16_t uint16_t


# --------------------------------------------------------------------------------------------
# Reader (Input)
# --------------------------------------------------------------------------------------------
cdef class MultibandBlockReader:
    """
    A class for reading multi-band raster datasets in blocks (windows).
    Supports both single-band files and multiband files.

    Attributes:
        filepaths (list): 
            A list of paths to the raster file.
        total_bands (int):
            Aggregate number of bands from all files
        win_shape (tuple):
            Window (win_height, win_width)
    """
    cdef:
        object vrt # VRT dataset
        int total_bands
        tuple img_shape
        list filepaths
    
    def __cinit__(self, list filepaths):
        """
        Initializes the reader.

        Parameters
        ----------
            filepaths (List[str]): A list of path(s) to the raster files.
        """
        # Check filepaths isn't empty
        assert filepaths is not None, "[rastio] MultibandBlockReader: empty file list"
        
        self.filepaths = filepaths
        # Comverge rasters into "one" dataset 
        try:
            self.vrt = buildvrt([rasterio.open(p) for p in self.filepaths])
            self.total_bands = self.vrt.count 
            self.img_shape = self.vrt.shape
            # Test if opening vrt errors
            self.vrt.read(1, window=Window(0, 0, 1, 1))
        except: 
            raise Exception (f"[rastio] MultibandBlockReader: Error opening files during __init__")

        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
            
    def __del__(self):
        self.close()

    cdef void close(self) noexcept:
        try:
            if self.vrt:self.vrt.close()
        except: 
            pass # exit quietly
    
    cdef tuple image_shape(self):
        """Returns (rows, cols)"""
        return self.img_shape
            
    cdef np.ndarray[float_t, ndim=3] tupleread_multiband_block(
        self,
        tuple window
        ):
        """
        Reads a block of data and returns (bands, rows, cols)
        
        Parameters
        ----------
            window (WindowType): Region of raster to pull data: ( (row_off, col_off), (win_height, win_width) )

        Returns
        ----------
            np.ndarray: A multiband Numpy array representing the block of data 
                with shape (bands, height, width), where (height, width) defined by window.
        """
        cdef:
            tuple offsets   = window[0]
            tuple win_dims  = window[1]
            int row_off     = offsets[0]
            int col_off     = offsets[1]
            int win_h       = win_dims[0]
            int win_w       = win_dims[1]

            # Block: (bands, rows, cols)
            np.ndarray[float_t, ndim=3] block 
            
        # Preallocate block
        block = np.empty((self.total_bands, win_h, win_w), dtype=np.float32, order="C")
        
       

        # ============================================================================================
        # Read & Return Multiband Block
        # ============================================================================================
        # Read directly into preallocated block
        self.vrt.read(window=Window(col_off, row_off, win_w, win_h), out=block)
        
        if block.shape[0] != self.total_bands:
            raise RuntimeError("[rastio] Band count mismatch after reading window")

        # Create a typed memoryview
        cdef float_t[:, :, :] block_mv = block
        
        return block


# --------------------------------------------------------------------------------------------
# Writer (Output)
# --------------------------------------------------------------------------------------------
cdef class MultibandBlockWriter:
    """
    A class that handles writing blocks of data to an output raster file.

    Attributes
    ----------
        output_dir (str): 
            The path to the output raster directory (filename not included).
        output_image_shape (tuple): 
            The dimensions (rows, cols) of the output image.
        output_image_name (str):
            filename.ext of output file. E.g., `raster.tif`.
        window_shape (tuple):
            Window dimensions (height, width).
        output_dtype (np.type, optional): 
            The data type of the output raster. Defaults to np.float32.
        num_bands (int):
            Number of output bands.
        compress_zstd (bint):
            Compresses output file with ZSTD compression. Smaller file = slower IO speed.
    """
    cdef:
        str out_dir
        tuple out_image_shape
        str out_image_name
        tuple win_shape
        object out_datatype
        int num_bands
        object dataset
        dict profile
    
    def __cinit__(
        self, 
        str out_dir, 
        tuple out_image_shape, 
        str out_image_name, 
        tuple win_shape, 
        object out_datatype,
        int num_bands,
    ):
        self.out_dir         = out_dir
        self.out_image_shape = out_image_shape
        self.out_image_name  = out_image_name
        self.win_shape       = win_shape
        self.out_datatype    = out_datatype
        self.num_bands       = num_bands
        self.dataset         = None
        self.profile         = {}


    def __enter__(self):      
        self.profile = {
            # GTFF (BIGTIFF) supports 4+ GB TIFF files
            "driver": "GTiff", 
            "height": self.out_shape[0],
            "width": self.out_shape[1],
            "count": self.num_bands, 
            "dtype": self.out_dtype,
            "tiled": True,
            "blockxsize": self.win_shape[1], 
            "blockysize": self.win_shape[0],
            "interleave": "band",
            "BIGTIFF": "YES",
            "compress": None, # Optional: replace with "ZSTD"
        }

        # Check for valid output path for intermediate dataset file,
        # otherwise create it
        makedirs(self.output_dir, exist_ok=True) 
        self.dataset = rasterio.open(f"{self.output_dir}/{self.output_name}", "w", **self.profile)
        return self 

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 

    cdef void close(self) noexcept:
        try:
            if self.dataset: self.dataset.close()
        except: 
            pass

    def write_block(
        self, 
        tuple window, 
        np.ndarray[output_datatype, ndim=3] block
    ):
        """
        Write block of data to dataset

        Parameters
        ----------
            window (tuple[tuple, tuple]): 
                Section of output dataset to write block to. shape=( (row_off, col_off), (win_height, win_width) )
            block (np.ndarray): 
                Block of data to be written. Size: (bands, win_height, win_width).
        """
        cdef:
            tuple offs   = window[0]
            tuple dims   = window[1]
            int row_off  = offs[0]
            int col_off  = offs[1]
            int win_h    = dims[0]
            int win_w    = dims[1]
        
        # ==============
        # Shape Checking
        # ==============
        if block.dtype != np.float32 or not block.flags['C_CONTIGUOUS']:
            # Ensure correct dtype and contiguous
            block = np.ascontiguousarray(block, dtype=np.float32)

        # Create memoryview
        cdef float_t[:, :, :] block_mv = block

        # Check dims and dataset
        if block_mv.shape[0] != self.num_bands or block_mv.shape[1] != win_h or block_mv.shape[2] != win_w:
            raise ValueError(f"[rastio] Shape mismatch: got {block.shape} vs expected ({self.num_bands}, {win_h}, {win_w})")
        if not self.dataset:
            raise RuntimeError("[rastio] Attempted to write but dataset is not initialized")
        
        # ==============================
        # Write & Return Multiband Block
        # ==============================
        win = Window(col_off, row_off, win_w, win_h) 
        self.dataset.write(block, window=win, indexes=range(1, self.num_bands + 1))


