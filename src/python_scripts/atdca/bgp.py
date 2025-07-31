"""bgp.py: Band Generation Process, crated new non-linear bondinations of existing bands"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"


import numpy as np
import itertools

def band_generation_process(block: np.ndarray, use_sqrt=True, use_log=False) -> np.ndarray:
    """
    Applies the Band Generation Process (BGP) to raster block. Optional flags align with Cheng and Ren (2000)'s implementation.

    Args:
        block (np.ndarray): Input block of shape (H, W, B) with float32 pixel values.
        use_sqrt (bool, optional): If True, includes square-root bands.. Defaults to True.
        use_log (bool, optional): If True, includes logarithmic bands ( log(1 + B) ). Defaults to False.

    Returns:
        np.ndarray:  Augmented block of shape (H, W, B').
        Includes:
        - Original bands
        - Auto-correlated bands
        - Cross-correlated bands
        - Optional: square-root and/or log bands
    """
    
    _, _, block_bands = block.shape
    output_bands = []

    # Original bands
    output_bands.append(block)

    # Auto-correlated bands (bi * bi)
    auto_bands = block ** 2
    output_bands.append(auto_bands)

    # Cross-correlated bands (i < j) (b1 * bj)
    cross_band_list = []
    for i, j in itertools.combinations(range(block_bands), 2):
        cross = block[:, :, i] * block[:, :, j]
        cross_band_list.append(cross[:, :, np.newaxis])
    cross_bands = np.concatenate(cross_band_list, axis=2)
    output_bands.append(cross_bands)

    # Square-root bands
    if use_sqrt:
        sqrt_bands = np.sqrt(np.clip(block, 0, None))
        output_bands.append(sqrt_bands)

    # Logarithmic bands (optional)
    if use_log:
        log_bands = np.log1p(np.clip(block, 0, None))  # log(1 + x)
        output_bands.append(log_bands)

    # Concat (combine) new bands into block
    bgp_block = np.concatenate(output_bands, axis=2).astype(np.float32)
    return bgp_block

