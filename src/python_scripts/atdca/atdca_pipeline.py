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
from .bgp import *
from .tgp import *
from .tcp import *
from .rastio import *
from ..utils.fileio import *

# Pyhton Modules
import numpy as np
import os


# --------------------------------------------------------------------------------------------
# ATDCA Pipeline
# --------------------------------------------------------------------------------------------
def ATDCA(
    input_dir:str, 
    output_dir:str, 
    output_filename:str = "targets", 
    one_file:bool = False,
    block_shape:tuple = (512,512),
    max_targets:int = 10,
    use_sqrt:bool = True,
    use_log:bool = False,
    ocpi_threshold:float = 0.01,
    input_image_type:str|tuple[str, ...] = "tif"
    ) -> None:
    """
    Runs the Automatic Target Detection Classification Algorithm (ATDCA)

    Args:
        input_dir (str): Directory containing multiple **single-band** (sb) images, OR one **multi-band** (mb) image.
        output_dir (str): Output directory for processed image(s).
        output_filename (str, optional): Filename for processed image. If `one_image` is set to True, sets name to that file. If `one_file` is set to False, creates sequenced outputs with `output_filename` as base, appending a number corresponding to which target each image belongs to. E.g., if `output_filename` is "processed", then the resultings names would be: "processed_0", "processed_1". '_1' refers to Target 1, '_2' to Target 2 etc.. Defaults to "targets".
        one_file (bool, optional): If True, concatenates outputs images into one file. If False, outputs separate files for each target. Defaults to False.
        block_shape (tuple, optional): Size of block to process each image, in a tile-wise operation. The larger the tile, the *faster* the operation, but the heavier the load on your PC's memory. May cause program to crash -- via unsufficient memory -- if set too large. Defaults to (512,512).
        max_targets (int, optional): Maximum number of targets for the algorithm to find. Program may end prematurely (i.e. before number of max_targets is reached) if ocpi is set too low. Defaults to 10.
        ocpi_threshold (float, optional): Target purity score. The lower the value (e.g., 0.001), the more pure the target categories. The larger the value (e.g., 0.1), the less pure the target categories. Larger values capture more noise, but are more forgiving to slight target variations. Defaults to 0.01.
        input_image_type (str | tuple[str, ...], optional): File extension of image type without the `.` (e.g. tif, png, jpg). If set to tuple (i.e. list of types), will read all images of those types. Defaults to "tif".
    """
    # Locate all image files of user-specified type in input directory
    input_files = discover_image_files(input_dir, input_image_type)
    
    if not input_files:
        raise FileNotFoundError(f"No input images found in {input_dir} with extensions: {input_image_type}")

    print(f"[ATDCA] Found {len(input_files)} input band(s). Initializing reader...")
    
    # Get reader which stacks multiple single-band images into a virtual multiband image
    reader = get_virtual_multiband_reader(input_files)

    # Get shape of window
    shape = reader("shape")
    if shape is None:
        raise ValueError("[ATDCA] Reader could not determine shape of window")
    
    # Run small test-block before potentially failing midway through heavy computation
    sample_block = reader(((0, 0), (min(256, shape[0]), min(256, shape[1]))))
    if not isinstance(sample_block, np.ndarray):
        raise ValueError("Expected sample block as np.ndarray, got invalid type.")
    
    bgp_block = band_generation_process_to_block(sample_block)
    num_bgp_bands = bgp_block.shape[2]

    bgp_output_path = os.path.join(output_dir, f"{output_filename}_bgp.tif")
    writer = get_block_writer(
        output_path=bgp_output_path,
        image_shape=shape,
        num_output_bands=num_bgp_bands,
        dtype=np.float32
    )

    print("[ATDCA] Running Band Generation Process (BGP)...")
    band_generation_process(
        image_reader=reader,
        image_writer=writer,
        block_shape=block_shape,
        use_sqrt=use_sqrt,
        use_log=use_log
    )

    print("[ATDCA] Running Target Generation Process (TGP)...")
    bgp_reader = get_virtual_multiband_reader([bgp_output_path])
    targets, _ = target_generation_process(
        image_reader=bgp_reader,
        max_targets=max_targets,
        opci_threshold=ocpi_threshold,
        block_shape=block_shape
    )
    print(f"[ATDCA] TGP detected {len(targets)} target(s).")

    print("[ATDCA] Running Target Classification Process (TCP)...")
    if isinstance(shape, tuple): # Check shape returns window size
        writer_factory = make_tcp_writer_factory(
            output_dir=output_dir,
            output_filename=output_filename,
            image_shape=shape,
            one_file=one_file
        )
    target_classification_process(
        image_reader=bgp_reader,
        targets=targets,
        image_writer_factory=writer_factory,
        block_shape=block_shape
    )

    print(f"[ATDCA] Complete. Results written to: {output_dir}")



