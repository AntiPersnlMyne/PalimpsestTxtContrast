# bgp_cython.pyx
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, log1p

@cython.wraparound(False)
def create_bands_from_block_cy(np.ndarray[cnp.float32_t, ndim=3] image_block,
                               bint full_synthetic):
    """
    Fully Cython-optimized version of _create_bands_from_block.
    Generates original bands, pairwise products, and optional sqrt/log bands.
    """
    cdef:
        int src_bands = image_block.shape[0]
        int src_height = image_block.shape[1]
        int src_width = image_block.shape[2]
        int total, idx = 0
        int band, i, y, x

    # Compute expected total bands
    total = src_bands + (src_bands * (src_bands - 1)) // 2
    if full_synthetic: 
        total += 2 * src_bands

    # Allocate output array
    cdef np.ndarray[cnp.float32_t, ndim=3] band_stack = np.empty((total, src_height, src_width), dtype=np.float32)

    # Original bands
    for band in range(src_bands):
        for row in range(src_height):
            for col in range(src_width):
                band_stack[idx, row, col] = image_block[band, row, col]
        idx += 1

    # Correlations
    for band in range(src_bands - 1):
        for i in range(band + 1, src_bands):
            for row in range(src_height):
                for col in range(src_width):
                    band_stack[idx, row, col] = image_block[band, row, col] * image_block[i, row, col]
            idx += 1

    # Optional sqrt and log
    if full_synthetic:
        # sqrt bands
        for band in range(src_bands):
            for y in range(src_height):
                for x in range(src_width):
                    band_stack[idx, y, x] = sqrt(image_block[band, y, x])
            idx += 1

        # natural log bands
        for band in range(src_bands):
            for y in range(src_height):
                for x in range(src_width):
                    band_stack[idx, y, x] = log1p(image_block[band, y, x])
            idx += 1

    # Check bands created matches expected amount
    if idx != total: raise ValueError(f"[BGP] Created bands #={idx} does not match expected={total}")

    return band_stack
