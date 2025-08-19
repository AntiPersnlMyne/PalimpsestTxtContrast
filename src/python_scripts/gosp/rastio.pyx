#!/usr/bin/env python3
# distutils: language=c

"""rastio.py: Handles I/O of raster data"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from osgeo import gdal
from os import makedirs

import rasterio
from rasterio.windows import Window
cdef extern from "Python.h": pass  # silence “Python.h” warning when using only typed calls

from .file_utils import rm


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.5"
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
        object dataset
        str vrt_path
        
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
        self.vrt_path = "output.vrt"

        # Comverge rasters into "one" dataset = VRT
        try:
            self.vrt = gdal.BuildVRT(self.vrt_path, filepaths)
            self.dataset = gdal.Open(self.vrt)

            self.total_bands = self.dataset.RasterCount
            self.img_shape = (self.dataset.RasterYSize, self.dataset.RasterXSize)
        except: 
            raise Exception (f"[rastio] MultibandBlockReader: Error opening files during __init__")

        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dataset = None # close dataset
        self.close()
            
    def __del__(self):
        self.dataset = None # close dataset
        rm(self.vrt_path)
        self.close()

    cdef void close(self) noexcept:
        self.dataset = None
        self.close()
    
    def image_shape(self) -> tuple:
        """Returns (rows, cols)"""
        return self.img_shape

    # cpdef maybe?       
    def read_multiband_block(self, tuple window):
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
            np.ndarray[float_t, ndim=3] block 
            
        # Preallocate block
        block = np.empty((self.total_bands, win_h, win_w), dtype=np.float32, order="C")
        
        # Create a typed memoryview
        cdef float_t[:, :, :] block_mv = block

        # Fill block
        for i in range(1, self.total_bands + 1):
            band_data = np.asarray(self.dataset.GetRasterBand(i).ReadAsArray(col_off, row_off, win_w, win_h), np.float32)
            block_mv[i-1,:,:] = band_data[:,:]
    
        
        if block.shape[0] != self.total_bands:
            raise RuntimeError("[rastio] Band count mismatch after reading window")
        return block


# --------------------------------------------------------------------------------------------
# Writer (Output)
# --------------------------------------------------------------------------------------------
cdef class MultibandBlockWriter:
    """
    A class that handles writing blocks of data to an output raster file.

    Attributes
    ----------
        out_dir (str): 
            The path to the output raster directory (filename not included).
        out_image_shape (tuple): 
            The dimensions (rows, cols) of the output image.
        out_image_name (str):
            filename.ext of output file. E.g., `raster.tif`.
        win_shape (tuple):
            Window dimensions (height, width).
        out_dtype (np.type, optional): 
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
            "height": self.out_image_shape[0],
            "width": self.out_image_shape[1],
            "count": self.num_bands, 
            "dtype": self.out_datatype,
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
        self.dataset = rasterio.open(f"{self.output_dir}/{self.out_image_name}", "w", **self.profile)
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
        np.ndarray[float_t, ndim=3] block
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
            raise ValueError(f"[rastio] Shape mismatch: got {(block_mv.shape[0],block_mv.shape[1],block_mv.shape[2])} vs expected ({self.num_bands}, {win_h}, {win_w})")
        if not self.dataset:
            raise RuntimeError("[rastio] Attempted to write but dataset is not initialized")
        
        # ==============================
        # Write & Return Multiband Block
        # ==============================
        win = Window(col_off, row_off, win_w, win_h) 
        self.dataset.write(block, window=win, indexes=range(1, self.num_bands + 1))


