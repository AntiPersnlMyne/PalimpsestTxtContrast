#!/usr/bin/env python3

"""gosp_pipeline.py: GOSP (Generalized Orthogonal Subspace Projection)

Implements: The OSP-based, automatic target detection workflow laid out by Chang and Ren.
[Ref] Hsuan Ren, Student Member, IEEE, and Chein-I Chang, Senior Member, IEEE 2000

Does: Automatically finds K (an integer > 1) likely targets in image and classififes all pixels to a target likelihood.
                            
Stages:
    0. Compile Cython (.pyx) code
    1. Band Generation Process (BGP)        - Create synthetic bands from raw imagery
    2. Target Generation Process (TGP)      - Iteratively discover target spectra using OSP
    3. Target Classification Process (TCP)  - Classify image using discovered targets
"""

# --------------------------------------------------------------------------------------------
# Imports Pipeline Modules
# --------------------------------------------------------------------------------------------
import logging
from warnings import filterwarnings
from os import remove
from numpy import ndarray

from ..build.bgp import band_generation_process
from ..build.tgp import target_generation_process
from ..build.tcp import target_classification_process
from ..build.skip_bgp import write_original_multiband
from ..build.file_utils import discover_image_files


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.2.5"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


# GeoTIFF warning suppression
filterwarnings("ignore", category=UserWarning, message="Dataset has no geotransform, gcps, or rpcs.*")
logging.disable(logging.WARNING)


# --------------------------------------------------------------------------------------------
# GOSP Pipeline
# --------------------------------------------------------------------------------------------
def gosp(
    input_dir:str, 
    output_dir:str, 
    input_image_types:str|tuple[str, ...] = "tif",
    window_shape:tuple = (512,512),
    full_synthetic:bool = False,
    skip_bgp:bool = False,
    max_targets:int = 10,
    opci_threshold:float = 0.01,
    verbose:bool = False,
) -> None:
    """
    Runs the Generalized Orthogonal Subspace Projection pipeline (GOSP)

    Args:
        input_dir (str): 
            Directory containing raw imagery. Accepts multiple single-band or multi-band image.
        output_dir (str): 
            Output directory for processed image(s).
        input_image_type (str | tuple[str, ...], optional): 
            File extension of image type without the `.` (e.g. tif, png, jpg). If set to tuple (i.e. list of types), will read all images of those types. Defaults to "tif".
        window_shape (tuple, optional): 
            Size of block to process each image, in a tile-wise operation. The larger the tile, the *faster* the operation, but the heavier the load on your PC's memory. May cause program to crash -- via unsufficient memory -- if set too large. Defaults to (512,512).
        
        full_synthetic (bool, optional): 
            Adds optional square root and log when creating sythetic bands. May give algorithm better results if True. Defaults to False.
        skip_bgp (bool, optional): 
            Skips generating synthetic bands. Set to true if synthetic data exceeds your PC disc storage. Defaults to False.
        
        max_targets (int, optional): 
            Maximum number of targets for the algorithm to find. Program may end prematurely (i.e. before number of max_targets is reached) if ocpi is set low. Defaults to 10.
        ocpi_threshold (float, optional): 
            Target purity score. The higher the value (e.g., 0.001), the more pure the target categories. The larger the value (e.g., 0.1), the less pure the target categories. Larger values capture more noise, but are more forgiving to slight target variations. Defaults to 0.01.
    
        verbose (bool, optional): 
            Enable/Disable loading bars in terminal.
    """
    # IO variables
    input_files = discover_image_files(input_dir, input_image_types)
    generated_bands_dir = f"{output_dir}/gen_band_norm.tif"
    targets_classified_dir = f"{output_dir}/target_classified"
    
    # Check input data exists
    if not input_files: raise FileNotFoundError(f"No input images found in {input_dir} with extension(s): {input_image_types}")

    # Verbose enables debug messages, else prints warnings/errors only
    if verbose: logging.basicConfig(level=logging.INFO)
    else: logging.basicConfig(level=logging.WARNING)


    logging.info("[GOSP] Running Band Generation Process (BGP)...")
    
    if not skip_bgp:
        band_generation_process(
            input_image_paths=input_files,
            output_dir=output_dir,
            window_shape=window_shape,
            full_synthetic=full_synthetic,
            verbose=verbose
        )
    else:
        # Change TGP to read input files instead of synthetic bands
        write_original_multiband(
            input_image_paths=input_files,
            output_dir=output_dir,
            window_shape=window_shape,
            verbose=verbose
        )


    logging.info("[GOSP] Running Target Generation Process (TGP)...")
        
    targets:list[ndarray] = target_generation_process(
        generated_bands=[generated_bands_dir],
        window_shape=window_shape,
        max_targets=max_targets,
        opci_threshold=opci_threshold,
        verbose=verbose
    )

    logging.info(f"[GOSP] TGP detected {len(targets)} target(s).")
    logging.info("[GOSP] Running Target Classification Process (TCP)...")
    
    target_classification_process(
        generated_bands=[generated_bands_dir],
        window_shape=window_shape,
        targets=targets,
        output_dir=targets_classified_dir,
        verbose=verbose
    )

    remove(generated_bands_dir) # Cleanup temp file

    logging.info(f"[GOSP] Complete. Results written to: {targets_classified_dir}")

