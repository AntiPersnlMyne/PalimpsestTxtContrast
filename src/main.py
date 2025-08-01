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
from python_scripts import utils

# Pyhton Modules
import numpy as np


# IO Paths
input_dir = r"data\input\test"
output_path = r"data\output\image_bgp.tif"



test_block = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [1, 1, 1]]
], dtype=np.float32)  # shape (2, 2, 3)

T0 = np.array([1, 0, 0], dtype=np.float32)
P = utils.compute_orthogonal_projection_matrix([T0])

projected = utils.project_block_onto_subspace(test_block, P)

            # Extract the first component from all projected pixels
first_components = projected[:, :, 0]

print("First components after projection:\n", first_components)

# Assert that they're (almost) zero
tolerance = 1e-6
if np.all(np.abs(first_components) < tolerance):
    print("✅ All first components successfully removed by projection.")
else:
    print("❌ Some components were not projected out.")





