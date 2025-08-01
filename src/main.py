# #!/usr/bin/env python3

# """main.py: Main driver file for image processing on manuscript"""

# __author__ = "Gian-Mateo (GM) Tifone and Douglas Tavolette"
# __copyright__ = "2025, RIT MISHA"
# __credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger"]
# __license__ = "MIT"
# __version__ = "1.3"
# __maintainer__ = "MISHA Team"
# __email__ = "mt9485@rit.edu"
# __status__ = "Prototype" # "Development", or "Production". 


# # Image processing library and helpers
# from python_scripts.utils import *          # Helper functions
# from python_scripts.improc import *         # Image processing

# def main():
#     # Directories for "process_images"
#     src_dir = "data/input/"
#     dst_dir = "data/output/"

#     # Logarithmic stretch
#     process_images(src_dir, dst_dir, "", log_stretch)
    
#     # Bilateral filter
#     process_images(src_dir, dst_dir, "", bilateral_filter, 
#                     {"diameter": 3,
#                     "sigma_color": 50,
#                     "sigma_space": 100})
    

#     # Close windows and exit
#     close_windows()



    
    
    
    
    


# if __name__ == "__main__":
#     print("Hello main!")
#     main()
#     print("Goodbye main!")



"""atdca_pipeline.py: Wraps BGP + TGP + TCP workflow into a pipeline.
                      ATDCA: Automatic Target Detection Classification Algorithm
                      Does: Automatically finds N likely targets in image and 
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



# Import pipeline modules
from python_scripts.utils import *
from python_scripts.atdca import *
from typing import Union, Tuple


# Pyhton Modules
import numpy as np


# IO Paths
input_dir = r"data\input\test"
output_path = r"data\output\image_bgp.tif"

ImageReader = Callable[[Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]], Union[np.ndarray, Tuple[int, int], None]]

def test_tgp_synthetic():
    height, width, bands = 64, 64, 3
    synthetic_image = np.zeros((height, width, bands), dtype=np.float32)

    # Inject synthetic targets
    synthetic_image[5, 5] = np.array([10, 0, 0])     # T0
    synthetic_image[10, 10] = np.array([0, 10, 0])   # T1
    synthetic_image[20, 20] = np.array([0, 0, 10])   # T2

    # Mock image_reader
    def mock_reader(window: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]):
        if window == "shape":
            return (height, width)
        (row_off, col_off), (h, w) = window
        if isinstance(row_off, int) and isinstance(col_off, int) and isinstance(h, int) and isinstance(w, int):
            return synthetic_image[row_off:row_off+h, col_off:col_off+w]

    # Run TGP
    targets, coords = target_generation_process(
        image_reader=mock_reader,
        max_targets=3,
        opci_threshold=0.001,
        block_shape=(32, 32)
    )

    print("[TEST] Detected coordinates:", coords)
    print("[TEST] Target vectors:")
    for vec in targets:
        print(vec)

    # Assert known target coordinates
    expected_coords = [(5, 5), (10, 10), (20, 20)]
    for actual, expected in zip(coords, expected_coords):
        assert actual == expected, f"Expected target at {expected}, got {actual}"

    # Assert target values are roughly correct
    expected_vectors = [
        np.array([10, 0, 0], dtype=np.float32),
        np.array([0, 10, 0], dtype=np.float32),
        np.array([0, 0, 10], dtype=np.float32),
    ]
    for vec, expected in zip(targets, expected_vectors):
        error = np.linalg.norm(vec - expected)
        assert error < 1e-3, f"Target mismatch: expected {expected}, got {vec}"

    print("✅ TGP synthetic test passed.")


if __name__ == "__main__":
    test_tgp_synthetic()

