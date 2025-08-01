"""atdca_pipeline.py: Wraps BGP + TGP + TCP workflow into a pipeline.
                      ATDCA: Automatic Target Detection Classification Algorithm
                      Does: Automatically finds N (integer > 1) likely targets in image and 
                            classififes all pixels
"""

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
# Import pipeline modules
from python_scripts.atdca import *

# Pyhton Modules
import numpy as np


# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
def _make_tcp_writer(output_template:str, image_shape:tuple[int, int]):
    def _factory(target_index:int):
        output_path = output_template.format(target_index)
        return get_block_writer(
            output_path=output_path,
            image_shape=image_shape,
            num_output_bands=1,
            dtype=np.float32
        )
    return _factory


def make_tcp_writer_factory(output_template: str, image_shape: Tuple[int, int]):
    """
    Creates a factory that returns a block writer for each target index.

    Args:
        output_template (str): Path template like "output/class_target_{:02}.tif"
        image_shape (Tuple[int, int]): Full image dimensions (height, width)

    Returns:
        Callable: writer_factory(target_idx) -> writer function
    """
    def factory(target_idx: int):
        output_path = output_template.format(target_idx)
        return get_block_writer(
            output_path=output_path,
            image_shape=image_shape,
            num_output_bands=1,
            dtype=np.float32
        )
    return factory


# --------------------------------------------------------------------------------------------
# ATDCA Pipeline Execution
# --------------------------------------------------------------------------------------------

# IO Paths
input_dir = r"data\input\test"
output_path = r"data\output\image_bgp.tif"







