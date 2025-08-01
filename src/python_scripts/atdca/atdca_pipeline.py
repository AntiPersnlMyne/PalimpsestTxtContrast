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
import bgp, tgp, tcp, rastio, config
from python_scripts import utils

# Pyhton Modules
import numpy as np


# IO Paths
input_dir = r"data\input\test"
output_path = r"data\output\image_bgp.tif"



# Target vector along x-axis
t0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

# Input vector
v = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Expected result: v projected orthogonally to x-axis → [0, 2, 3]
expected = np.array([0.0, 2.0, 3.0], dtype=np.float32)

# Compute projection matrix
P = utils.compute_orthogonal_projection_matrix([t0])

# Apply projection: P @ v
projected = P @ v

print("Original vector     :", v)
print("Expected projection :", expected)
print("Actual projection   :", projected)

# Check error
error = np.linalg.norm(projected - expected)
print("Projection error    :", error)
            





