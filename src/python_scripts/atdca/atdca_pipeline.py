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


image_path = r"data\input\Arch_165r_370nm.tif"
reader = rastio.get_block_reader(image_path)

t0_vector, t0_coords = tgp.target_generation_process(reader)
print(f"T0 found at {t0_coords} with vector: {t0_vector}")


def run_atdca_pipeline():
    raise NotImplementedError




