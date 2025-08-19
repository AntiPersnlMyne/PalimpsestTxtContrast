#!/usr/bin/env python3
# distutils: language=c

"""rastio.py: Handles I/O of raster data"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from os import makedirs
from os.path import join
import tempfile
import os

from osgeo import gdal
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
__version__ = "3.1.6"
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
# Helper Functions
# --------------------------------------------------------------------------------------------
def _build_vrt(vrt_path:str, filepaths:list[str], separate=True, allow_projection_difference=True) -> object:
    """
    Create a virtual raster (VRT) from a set of input raster files.
    
    args:
        out_dir (str): 
            Path to the output VRT file (filename not included).
        input_pattern (str): 
            Glob pattern for input raster files.
        separate (bool): 
            If True, stack bands separately.
        allow_projection_difference (bool): 
            If True, allow rasters with different projections.
    
    Returns:
        object: VRT dataset object.
    """
    if not filepaths:
        raise FileNotFoundError(f"[rastio] No files in filepaths")

    # Build VRT options
    vrt_options = gdal.BuildVRTOptions(
        separate=separate,
        allowProjectionDifference=allow_projection_difference
    )

    # Create the VRT
    vrt = gdal.BuildVRT(vrt_path, filepaths, options=vrt_options)
    if vrt is None:
        raise RuntimeError("Failed to build VRT")

    return vrt


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
        
        # Define temporary parh for VRT object
        fd, vrt_path = tempfile.mkstemp(suffix=".vrt")
        os.close(fd)
        self.vrt_path = vrt_path

        # Create VRT
        self.vrt = _build_vrt(
            vrt_path=self.vrt_path,
            filepaths=filepaths
        )

        try:
            self.dataset = gdal.Open(self.vrt_path)

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
        try:
            self.close()
        except Exception:
            pass
        # Remove the vrt file if exists
        try:
            rm(self.vrt_path)
        except Exception:
            pass

    def close(self):
        """Safely close dataset and free VRT path if present."""
        try:
            if self.dataset is not None: self.dataset = None
        except Exception:
            pass
    
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
            Py_ssize_t i
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
        str output_dir, 
        tuple output_image_shape, 
        str output_image_name, 
        tuple window_shape, 
        object output_datatype,
        int num_bands,
    ):
        self.out_dir         = output_dir
        self.out_image_shape = output_image_shape
        self.out_image_name  = output_image_name
        self.win_shape       = window_shape
        self.out_datatype    = output_datatype
        self.num_bands       = num_bands
        self.dataset         = None
        self.profile         = {}


    def __enter__(self):      
        # Build rasterio profile using the attributes set above
        self.profile = {
            "driver": "GTiff",
            "height": int(self.out_image_shape[0]),
            "width": int(self.out_image_shape[1]),
            "count": int(self.num_bands),
            # rasterio expects np.dtype or string
            "dtype": self.out_datatype if isinstance(self.out_datatype, np.dtype) else np.dtype(self.out_datatype).name,
            "tiled": True,
            "blockxsize": int(self.win_shape[1]),
            "blockysize": int(self.win_shape[0]),
            "interleave": "band",
            "BIGTIFF": "YES",
            "compress": None,
        }

        # Check for valid output path for intermediate dataset file,
        # otherwise create it
        makedirs(self.out_dir, exist_ok=True) 
        out_path = join(self.out_dir, self.out_image_name)
        self.dataset = rasterio.open(out_path, "w", **self.profile)
        return self 

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 

    def close(self):
        try:
            if self.dataset: self.dataset.close()
        except: 
            if self.dataset is not None: self.dataset = None
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


