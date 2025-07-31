"""rastio.py: Handles I/O of raster data"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



from numpy import transpose
from warnings import warn
from rasterio import open
from rasterio.windows import Window


def get_block_reader(image_path:str):
    """
    Returns a callable that reads blocks from a raster image.

    The returned function supports:
    - `window=( (row_off, col_off), (height, width) )`: returns a block (rows, cols, bands)
    - `window='shape'`: returns full image shape (height, width)

    Args:
        image_path (str): Path to the input TIFF image.

    Returns:
        image_reader (Callable): Image reader object that reads blocks from raster image
    """

    dataset = open(image_path)

    def image_reader(window):
        if window == "shape":
            return dataset.height, dataset.width

        (row_off, col_off), (width, height) = window

        try:
            # Read all bands in the window; shape: (bands, height, width)
            data = dataset.read(
                window = Window(col_off, row_off, width, height) #type:ignore
            )
            # Transpose to shape: (height, width, bands)
            return transpose(data, (1, 2, 0))

        except Exception as e:
            warn(f"- Warning: Skipping block at ({row_off}, {col_off}):\n{e}")
            return None

    return image_reader