# Assuming all your provided scripts are in the same directory or accessible via Python path
from .tgp import target_generation_process, _make_windows, best_target_parallel
from ..utils.math_utils import compute_orthogonal_projection_matrix, compute_opci
from .rastio import MultibandBlockReader
import numpy as np
import sys

# Replace this with the actual path to your generated bands
GENERATED_BANDS_PATH = "data/output/gen_band_norm.tif"

def debug_target_generation_process(
    generated_bands: list[str],
    window_shape: tuple[int, int],
    max_targets: int = 10,
    opci_threshold: float = 0.05
) -> list[np.ndarray]:
    
    # 1. Find the first target (T0)
    # The code for this part is the same as in your original function.
    # It will find T0 and append it to the targets list.
    targets = target_generation_process(
        generated_bands=generated_bands,
        window_shape=window_shape,
        max_targets=1,  # Stop after the first target
        opci_threshold=1.0, # Set a very high threshold to ensure it runs at least once
        show_progress=True,
        inflight=2,
    )
    
    if not targets:
        print("Error: No initial target found. Exiting.")
        return []

    print("\n" + "="*50)
    print("DEBUGGING TGP EXECUTION")
    print("="*50 + "\n")

    # Print the first target
    print("First Target (T0) Spectrum:")
    print(targets[0])
    print("\n" + "-"*50 + "\n")
    
    # 2. Re-run the main loop just once to get the second candidate and OPCI
    
    # a. Compute the orthogonal projection matrix for the first target
    p_matrix = compute_orthogonal_projection_matrix(targets)
    print("Calculated Projection Matrix (based on T0):")
    print(p_matrix)
    print("\n" + "-"*50 + "\n")
    
    # b. Find the best target in the projected subspace
    # We will use the original function call to find this candidate

    
    with MultibandBlockReader(generated_bands) as reader:
        image_shape = reader.image_shape()
    windows = _make_windows(image_shape, window_shape)
    
    new_target_candidate = best_target_parallel(
        paths=generated_bands,
        windows=windows,
        p_matrix=p_matrix
    )
    
    print("New Target Candidate (T1) Spectrum:")
    print(new_target_candidate.band_spectrum)
    print("\n" + "-"*50 + "\n")
    
    # c. Calculate and print the OPCI value
    opci_squared = compute_opci(p_matrix, new_target_candidate.band_spectrum)
    opci = np.sqrt(opci_squared)
    
    print(f"Calculated OPCI value: {opci}")
    print(f"OPCI Threshold: {opci_threshold}")
    
    return targets

if not GENERATED_BANDS_PATH:
    print("Please specify the path to your generated bands file in the script.")
    sys.exit(1)
    