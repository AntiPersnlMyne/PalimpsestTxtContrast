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
from numpy import transpose, float32, newaxis, ndarray, float32, concatenate
from warnings import warn
from rasterio import open
from rasterio.windows import Window
from typing import List, Union, Tuple
from os.path import exists
from os import makedirs


# --------------------------------------------------------------------------------------------
# Reader (Input)
# --------------------------------------------------------------------------------------------
def get_virtual_multiband_reader(band_paths: List[str]):
    """
    Returns a reader that stacks multiple single-band images into a virtual multiband image.

    Args:
        band_paths (List[str]): List of paths to single-band TIFF files.

    Returns:
        Callable: image_reader(window) -> Tuple|Array|None
    """
    datasets = [open(p) for p in band_paths]

    # Check all bands must match shape
    ref_shape = (datasets[0].height, datasets[0].width)
    for dataset in datasets:
        if (dataset.height, dataset.width) != ref_shape:
            raise ValueError("-- Error in get_virtual_multiband_reader:\nAll bands must have the same dimensions --")

    def _image_reader(window: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]) -> Union[ndarray, Tuple[int, int], None]:
        if window == "shape":
            return ref_shape

        # Window dimensions
        (row_off, col_off), (height, width) = window
        
        try:
            block_list = []
            for ds in datasets:
                data = ds.read(1, window=Window(col_off, row_off, width, height))  # type:ignore
                block_list.append(data[:, :, newaxis])  # Expand to (H, W, 1)

            stacked = concatenate(block_list, axis=2)  # Shape: (H, W, B)
            return stacked.astype(float32)

        except Exception as e:
            warn(f"[Warning] Failed to read window {window}: {e}")
            return None

    return _image_reader



# --------------------------------------------------------------------------------------------
# Writer (Output)
# --------------------------------------------------------------------------------------------
def get_block_writer(output_path, image_shape, num_output_bands, dtype=float32, profile_template=None):
    """
    Returns a function to write blocks to a raster image.

    Args:
        output_path (str): Path to output GeoTIFF.
        image_shape (tuple): (height, width) of full image.
        num_output_bands (int): Number of bands in the output image.
        dtype (np.dtype): Data type of output image.
        profile_template (dict, optional): Optional rasterio profile to inherit metadata.

    Returns:
        Callable: writer(window: tuple, block: np.ndarray) -> None
    """
    
    # Make output path if it doesn't exist
    if not exists(output_path):
        makedirs(output_path)
    
    # Image data
    image_height, image_width = image_shape

    # TIFF profile ("structure") 
    profile = {
        "driver": "GTiff", # GeoTIFF supports 4+ GB TIFF files
        "height": image_height,
        "width": image_width,
        "count": num_output_bands,
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
        profile.update({k: v for k, v in profile_template.items() if k in profile})

    # Import raster
    dataset = open(output_path, "w", **profile)

    def _block_writer(window, block):
        """
        Writes a block to the output raster.

        Args:
            window (tuple): ((row_off, col_off), (height, width))
            block (np.ndarray): Block data of shape (height, width, bands)
        
        Returns: 
            Callable: Funtion to write blocks onto the raster image
        """
        
        # Window data
        (row_off, col_off), (height, width) = window
        
        # Rasterio expects shape: (bands, height, width)
        data = transpose(block, (2, 0, 1))
        dataset.write(data, window=Window(col_off, row_off, width, height)) #type:ignore

    return _block_writer




