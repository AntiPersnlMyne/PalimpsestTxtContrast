import numpy as np
import cv2

# Typing
from typing import Callable

def tile_process(image: np.ndarray, processing_function: Callable[[np.ndarray], np.ndarray], tile_size: int, verbose: bool = False, **func_params) -> np.ndarray | None:
    """
    Applies a given image processing function to an image tile-wise.

    Args:
        image (np.ndarray): The input image (NumPy array).
        processing_function (callable): The function to apply to each tile.
                                        It must accept a NumPy array (the tile)
                                        as its first argument and return a NumPy array
                                        of the same shape.
                                        [] specified the type of the arguements
        tile_size (int): The size of the square tiles (e.g., 64 for 64x64 tiles).
        verbose (bool, optional): If True, print detailed messages. Defaults to True.
        **func_params: Arbitrary keyword arguments to pass to the processing_function.

    Returns:
        np.ndarray: The processed image, stitched back together from the processed tiles.
                    Returns None if there's an error.
    """
    # Initialize verboseprint based on the 'verbose' argument
    v_print = print if verbose else lambda *a, **k: None

    if not isinstance(image, np.ndarray):
        print("Error: Input 'image' must be a NumPy array.")
        return None
    if not callable(processing_function):
        print("Error: 'processing_function' must be a callable (a function).")
        return None
    if not isinstance(tile_size, int) or tile_size <= 0:
        print("Error: 'tile_size' must be a positive integer.")
        return None

    height, width = image.shape[:2] # Get height and width, ignoring channels for now
    processed_image = np.zeros_like(image, dtype=image.dtype) # Initialize output image

    v_print(f"Image dimensions: {width}x{height}")
    v_print(f"Tile size: {tile_size}x{tile_size}")

    # Determine the number of tiles along height and width
    num_tiles_h = (height + tile_size - 1) // tile_size  # Ceiling division
    num_tiles_w = (width + tile_size - 1) // tile_size

    v_print(f"Number of tiles (rows x columns): {num_tiles_h} x {num_tiles_w}")

    for y in range(num_tiles_h):
        for x in range(num_tiles_w):
            # Calculate tile coordinates
            start_y = y * tile_size
            end_y = min((y + 1) * tile_size, height) # Ensure end_y doesn't exceed image height
            start_x = x * tile_size
            end_x = min((x + 1) * tile_size, width) # Ensure end_x doesn't exceed image width

            # Extract the current tile
            tile = image[start_y:end_y, start_x:end_x]
            v_print(f"Processing tile at ({x}, {y}): slice [{start_y}:{end_y}, {start_x}:{end_x}] "
                         f"with shape {tile.shape}")

            if tile.size == 0: # Skip empty tiles, though min() should prevent this with proper ranges
                v_print(f"Warning: Skipping empty tile at ({x}, {y}).")
                continue

            try:
                # Apply the processing function to the tile, passing additional parameters
                processed_tile = processing_function(tile, **func_params)

                # Ensure the processed tile has the same shape as the original tile
                # This is crucial for stitching. If the function changes shape,
                # you'd need padding/cropping or a different stitching strategy.
                if processed_tile.shape != tile.shape:
                    print(f"Error: Processing function changed tile shape at ({x}, {y}). "
                          f"Expected {tile.shape}, got {processed_tile.shape}. "
                          f"Cannot stitch. Returning None.")
                    return None

                # Place the processed tile back into the output image
                processed_image[start_y:end_y, start_x:end_x] = processed_tile

            except Exception as e:
                print(f"Error processing tile at ({x}, {y}): {e}")
                return None

    v_print("All tiles processed and stitched.")
    return processed_image