#!/usr/bin/env python3

"""main.py: Main driver file for image processing on manuscript"""

__author__ = "Gian-Mateo (GM) Tifone and Douglas Tavolette"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone", "Douglas Tavolette", "Roger Easton Jr.", "David Messinger"]
__license__ = "MIT"
__version__ = "1.3"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production". 



# Import pipeline modules
from python_scripts.atdca import *
from typing import Union, Tuple


# Pyhton Modules
import numpy as np



# Simulate a 3-target synthetic image
ImageReader = Callable[[Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]], Union[np.ndarray, Tuple[int, int], None]]

def test_tcp_synthetic():
    height, width, bands = 32, 32, 3
    synthetic_image = np.zeros((height, width, bands), dtype=np.float32)

    # Inject target matches
    synthetic_image[5, 5] = np.array([10, 0, 0])   # Target 0
    synthetic_image[10, 10] = np.array([0, 10, 0]) # Target 1
    synthetic_image[15, 15] = np.array([0, 0, 10]) # Target 2

    targets = [
        np.array([10, 0, 0], dtype=np.float32),
        np.array([0, 10, 0], dtype=np.float32),
        np.array([0, 0, 10], dtype=np.float32)
    ]

    image_shape = (height, width)

    # Mock image_reader
    def reader(window: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]):
        if window == "shape":
            return image_shape
        (row_off, col_off), (h, w) = window
        if isinstance(row_off, int) and isinstance(col_off, int) and isinstance(h, int) and isinstance(w, int):
            return synthetic_image[row_off:row_off+h, col_off:col_off+w]

    # Writer factory — stores output in memory
    result_images = {}

    def make_writer_factory():
        def writer_factory(target_idx: int):
            result_images[target_idx] = np.zeros((height, width), dtype=np.float32)

            def writer(window, block):
                (row_off, col_off), (h, w) = window
                result_images[target_idx][row_off:row_off+h, col_off:col_off+w] = block
            return writer
        return writer_factory

    target_classification_process(
        image_reader=reader,
        targets=targets,
        image_writer_factory=make_writer_factory(),
        block_shape=(16, 16)
    )

    # Check that classification maps highlight correct pixels
    for idx, expected_pos in enumerate([(5, 5), (10, 10), (15, 15)]):
        classified_map = result_images[idx]
        response = classified_map[expected_pos]
        print(f"[TEST] Target {idx} classified value at {expected_pos}: {response:.3f}")
        assert response > 5.0, f"Expected strong response at {expected_pos}, got {response}"

    print("✅ TCP synthetic test passed.")

if __name__ == "__main__":
    test_tcp_synthetic()


