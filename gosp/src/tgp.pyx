#!/usr/bin/env python3
# distutils: language=c


"""tgp.pyx: Target Generation Process. Automatically creates N most significant targets in target detection for pixel classification"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np
cimport numpy as np
from cython.parallel import prange

from ..build.rastio import MultibandBlockReader


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.3.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# C Functions
# --------------------------------------------------------------------------------------------
cdef void _extract_window(
    int row_start,
    int row_end,
    int col_start,
    int col_end,
    np.float32_t[:,:,:] vrt,
    np.float32_t[:,:] out_window
) noexcept nogil:
    """
    Copy a sub-image from vrt into out_window.

    Parameters
    ----------
    row_start, row_end : int
        Pixel indices of the vertical window (half-open: [row_start, row_end))
    col_start, col_end : int
        Pixel indices of the horizontal window (half-open)
    vrt : float[:,:,:]
        The synthetic image with shape (bands, height, width)
    out_window : ndarray[float32, ndim=2]
        Pre-allocated buffer array with shape ((row_end-row_start)*(col_end-col_start), bands)
    """
    cdef:
        Py_ssize_t band
        Py_ssize_t row, col
        Py_ssize_t out_row = 0
        Py_ssize_t rows = row_end - row_start
        Py_ssize_t cols = col_end - col_start
        Py_ssize_t bands = vrt.shape[0]

    for row in range(rows):
        for col in range(cols):
            for band in range(bands):
                out_window[out_row, band] = vrt[band, row_start + row, col_start + col]
            out_row += 1


cdef void _best_target(
    np.float32_t[:,:] pixel_window,
    np.float32_t[:,:] orthog_complement,
    np.float32_t[:]   best_target,
    np.float32_t[:,:] projected
) noexcept nogil:
    """
    For every pixel in pixel_window compute its projection onto the
    orthogonal complement defined by *orthog_complement* and keep the
    pixel with the largest projection norm.

    Parameters
    ----------
    pixel_window : ndarray[float32, ndim=2]
        Pixels flattened into rows; shape (num_pixels, bands)
    orthog_complement : ndarray[float32, ndim=2]
        Projection matrix (bands, bands-orthog_dim)
    best_target : ndarray[float32, ndim=1]
        Buffer that receives the pixel with the largest norm
    """
    cdef:
        Py_ssize_t band, j
        Py_ssize_t pixel
        Py_ssize_t proj_dim = orthog_complement.shape[1]
        Py_ssize_t num_pixels = pixel_window.shape[0]
        Py_ssize_t bands = pixel_window.shape[1]

    # Project every pixel
    for pixel in range(num_pixels):
        for j in range(proj_dim):
            projected[pixel, j] = 0.0
            for band in range(bands):
                projected[pixel, j] += pixel_window[pixel, band] * orthog_complement[band, j]

    # Find the pixel with the largest projection norm
    cdef float max_norm = -1e36
    cdef float norm
    cdef int best_idx = -1
    for pixel in range(num_pixels):
        norm = 0.0
        for band in range(bands):
            norm += projected[pixel, band] * projected[pixel, band]
        if norm > max_norm:
            max_norm = norm
            best_idx = pixel

    # Copy the best pixel back into best_target
    for band in range(bands):
        best_target[band] = pixel_window[best_idx, band]



# --------------------------------------------------------------------------------------------
# TGP Function
# --------------------------------------------------------------------------------------------
def target_generation_process(
    np.float32_t[:,:,::1] vrt,
    int window_height,
    int window_width,
    np.float32_t[:,:] orthog_complement,
    bint verbose
) -> np.ndarray[np.float32_t]:
    """
    Build the target matrix by scanning the synthetic image in a grid of
    overlapping windows.  The function is fully typed, releases the GIL
    where possible, and uses OpenMP to split the work across cores.

    Parameters
    ----------
    synthetic_vrt : ndarray[float32, ndim=3]
        Synthetic image with shape (bands, height, width)
    window_height, window_width : int
        Size of the scanning window in pixels
    orthog_complement : ndarray[float32, ndim=2]
        Projection matrix that defines the orthogonal complement
        (bands, bands-orthog_dim)

    Returns
    -------
    ndarray[float32, ndim=2]
        Target matrix with shape (num_windows, bands)
   """
    cdef:
        Py_ssize_t height = vrt.shape[1]
        Py_ssize_t width = vrt.shape[2]
        Py_ssize_t bands = vrt.shape[0]


        Py_ssize_t rows_per_window = (height + window_height - 1) // window_height
        Py_ssize_t cols_per_window = (width + window_width - 1) // window_width
        Py_ssize_t total_windows = rows_per_window * cols_per_window


        # Allocate the output buffer once – we will fill it in place.
        np.ndarray[np.float32_t, ndim=2] target_matrix = np.empty(
        (total_windows, bands), dtype=np.float32
        )


        # A buffer that holds a single window (flattened) -- allocated in Python space
        np.ndarray[np.float32_t, ndim=2] flattened_window = np.empty(
        (window_height * window_width, bands), dtype=np.float32
        )


        # Projected buffer: max pixels equals flattened_window rows, proj_dim from orthog_complement
        np.ndarray[np.float32_t, ndim=2] projected = np.empty(
        (window_height * window_width, orthog_complement.shape[1]), dtype=np.float32
        )


        # Array to contain the best pixel of the current window
        np.ndarray[np.float32_t, ndim=1] best_pixel = np.empty(bands, dtype=np.float32)


    # Create memoryviews 
    cdef np.float32_t[:, :, ::1] vrt_mv = vrt
    cdef np.float32_t[:, ::1] target_mv = target_matrix
    cdef np.float32_t[:, ::1] flat_mv = flattened_window
    cdef np.float32_t[:, ::1] proj_mv = projected
    cdef np.float32_t[::1] best_mv = best_pixel


    cdef:
        Py_ssize_t row_offset, col_offset, window_index
        int row_start, row_end, col_start, col_end
        Py_ssize_t band

    # Create progress bar
    print("[TGP] Beginning TGP ...")
    # Parallel loop through windows
    for window_index in prange(total_windows, nogil=True, schedule='dynamic'):
        # Compute offsets
        row_offset = (window_index // cols_per_window) * window_height
        col_offset = (window_index % cols_per_window) * window_width

        # Window indices
        row_start = <int> row_offset
        col_start = <int> col_offset
        # Bounds check
        row_end = row_offset + window_height 
        if row_end > height: <int> height
        col_end = col_offset + window_width 
        if col_end > width:  <int> width

        # Extract window into the pre-allocated flat buffer 'flat_mv'
        _extract_window(row_start, row_end, col_start, col_end, vrt_mv, flat_mv)

        # Compute best target using the projected buffer
        _best_target(flat_mv, orthog_complement, best_mv, proj_mv)


        # Copy best pixel into the final target matrix (use memoryview indexing)
        for band in range(bands):
            target_mv[window_index, band] = best_mv[band]
    
    
    return target_matrix

