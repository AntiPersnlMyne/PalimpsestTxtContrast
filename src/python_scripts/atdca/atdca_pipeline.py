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
# Imports Pipeline Modules
# --------------------------------------------------------------------------------------------
from .bgp import band_generation_process
from .tgp import target_generation_process
# from .tcp import *
from ..utils.fileio import discover_image_files



# --------------------------------------------------------------------------------------------
# ATDCA Pipeline
# --------------------------------------------------------------------------------------------
def ATDCA(
    input_dir:str, 
    output_dir:str, 
    verbose:bool = False,
    one_file:bool = False,
    window_shape:tuple = (512,512),
    max_targets:int = 10,
    use_sqrt:bool = False,
    use_log:bool = False,
    ocpi_threshold:float = 0.01,
    input_image_type:str|tuple[str, ...] = "tif",
    chunk_size:int = 4,
    inflight:int = 2
    ) -> None:
    """
    Runs the Automatic Target Detection Classification Algorithm (ATDCA)

    Args:
        input_dir (str): Directory containing multiple **single-band** (sb) images, OR one **multi-band** (mb) image.
        output_dir (str): Output directory for processed image(s).
        one_file (bool, optional): If True, concatenates outputs images into one file. If False, outputs separate files for each target. Defaults to False.
        window_shape (tuple, optional): Size of block to process each image, in a tile-wise operation. The larger the tile, the *faster* the operation, but the heavier the load on your PC's memory. May cause program to crash -- via unsufficient memory -- if set too large. Defaults to (512,512).
        max_targets (int, optional): Maximum number of targets for the algorithm to find. Program may end prematurely (i.e. before number of max_targets is reached) if ocpi is set too low. Defaults to 10.
        use_sqrt (bool, optional): Use square root when creating sythetic bands. May give algorithm better results if True. Defaults to False.
        use_log (bool, optional): Use logarithm when compute synthetic bands. May give algorithm better results if True. Defaults to False.
        ocpi_threshold (float, optional): Target purity score. The lower the value (e.g., 0.001), the more pure the target categories. The larger the value (e.g., 0.1), the less pure the target categories. Larger values capture more noise, but are more forgiving to slight target variations. Defaults to 0.01.
        input_image_type (str | tuple[str, ...], optional): File extension of image type without the `.` (e.g. tif, png, jpg). If set to tuple (i.e. list of types), will read all images of those types. Defaults to "tif".
    """
    
    # --------------------------------------------------------------------------------------------
    # Get input data
    # --------------------------------------------------------------------------------------------
    # Locate all image files of user-specified type in input directory
    input_files = discover_image_files(input_dir, input_image_type)
    
    if not input_files:
        raise FileNotFoundError(f"No input images found in {input_dir} with extension(s): {input_image_type}")

    print(f"[ATDCA] Found {len(input_files)} input band(s).")
    
    print("[ATDCA] Running Band Generation Process (BGP)...")
    band_generation_process(
        input_image_paths=input_files,
        output_dir=output_dir,
        window_shape=window_shape,
        use_sqrt=use_sqrt,
        use_log=use_log,
        max_workers=None,
        chunk_size=chunk_size,
        inflight=inflight,
        show_progress=verbose
    )


    print("[ATDCA] Running Target Generation Process (TGP)...")
    target_generation_process(
        generated_bands=[f"{output_dir}/gen_band_norm.tif"],
        window_shape=window_shape,
        max_targets=max_targets,
        ocpi_threshold=ocpi_threshold,
        use_parallel=False,
        max_workers=None,
        inflight=inflight,
        show_progress=verbose
    )

    # print(f"[ATDCA] TGP detected {len(targets)} target(s).")




    # print("[ATDCA] Running Target Classification Process (TCP)...")
    # if isinstance(shape, tuple): # Check shape returns window size
    #     writer_factory = make_tcp_writer_factory(
    #         output_dir=output_dir,
    #         output_filename=output_filename,
    #         image_shape=shape,
    #         one_file=one_file
    #     )
        
    # target_classification_process(
    #     image_reader=bgp_reader,
    #     targets=targets,
    #     image_writer_factory=writer_factory,
    #     block_shape=block_shape
    # )

    # print(f"[ATDCA] Complete. Results written to: {output_dir}")



