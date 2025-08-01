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

# Other imports
import os
import numpy as np

# Assumes single-band TIFF
input_dir = r"data\input"
output_path = r"data\output\image_bgp.tif"

# Import all band paths from folder
band_paths = []
for filename in os.listdir(input_dir):
    if filename.endswith('.tif'):
        band_paths.append("data/input/" + filename)
            

print(f"[INFO] Using {len(band_paths)} input bands...")
reader = rastio.get_virtual_multiband_reader(band_paths)

# Step 2: Find initial target T0
t0_vector, t0_coords = tgp.target_generation_process(reader)
print(f"T0 found at {t0_coords} with vector: {t0_vector}")

# Step 3: Prepare to run BGP
input_shape = reader("shape")
sample_block = reader(((0, 0), (256, 256)))
bgp_block = bgp._band_generation_process_to_block(sample_block, use_sqrt=True, use_log=False)
num_output_bands = bgp_block.shape[2]

# Step 4: Create writer and run BGP
writer = rastio.get_block_writer(
    output_path=output_path,
    image_shape=input_shape,
    num_output_bands=num_output_bands,
    dtype=np.float32
)

bgp.band_generation_process(
    image_reader=reader,
    image_writer=writer,
    block_shape=(512, 512),
    use_sqrt=True,
    use_log=False
)

print(f"[INFO] Band generation completed: {output_path}")






