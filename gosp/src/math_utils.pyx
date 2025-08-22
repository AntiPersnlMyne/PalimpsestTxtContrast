#!/usr/bin/env python3
# distutils: language=c
# cython: profile=True

"""math_utils.pyx: Linear algebra, matrix, and calculus helper functions"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
cimport numpy as np

from libc.math cimport sqrt as csqrt


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.2"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Production" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Constants (to prevent DIV0)
# --------------------------------------------------------------------------------------------
OPCI_EPS = 1e-12   # denom floor
OPCI_TOL = 1e-9    # clamp tolerance


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
ctypedef np.float32_t float_t
ctypedef Py_ssize_t psize_t


# --------------------------------------------------------------------------------------------
# C Helper Functions
# --------------------------------------------------------------------------------------------
cdef int _block_l2_cy(
    float_t[:, :, :] block_mv, 
    float_t[:, :] out_mv
) nogil:
    """
    Compute L2 norm at each pixel from a block (bands, h, w).
    out_mv[row, col] = sqrt(sum_b block[b, row, col]^2).
    """
    cdef:
        psize_t bands = block_mv.shape[0]
        psize_t height = block_mv.shape[1]
        psize_t width = block_mv.shape[2]
        psize_t b, row, col
        float_t sum
    
    for row in range(height):
        for col in range(width):
            sum = 0.0
            for b in range(bands):
                sum += block_mv[b, row, col] * block_mv[b, row, col]
            out_mv[row, col] = <float_t> csqrt(sum)


cdef int _matvec_cy(
    float_t[:, :] pmat_mv, 
    float_t[:]    x_mv,
    float_t[:]    y_mv
) nogil:
    """
    Compute y = M @ x where M is (n,n), x is (n,), y is (n,).
    All arrays are float32 memoryview.
    """
    cdef:
        psize_t n = pmat_mv.shape[0]
        psize_t i, j
        float_t sum
    
    for i in range(n):
        sum = 0.0
        for j in range(n):
            sum += pmat_mv[i, j] * x_mv[j]
        y_mv[i] = sum


# --------------------------------------------------------------------------------------------
# Matrix Operand Functions
# --------------------------------------------------------------------------------------------
def compute_orthogonal_complement_matrix(
    target_vectors:list[np.ndarray]
) -> np.ndarray:
    """
    Construct an orthogonal projection matrix onto the complement of the
    subspace spanned by given target vectors.


    P_perp = I - U U^+ , where U stacks target_vectors columnwise.


    Args:
    target_vectors (list[np.ndarray]): list of 1D arrays, each (bands,).


    Returns:
    np.ndarray: (bands, bands) float32 symmetric projector.
    """
    if len(target_vectors) == 0:
        raise ValueError("Must provide at least one target vector")

    # Stack in double precision for stable pinv
    U = np.stack(target_vectors, axis=1).astype(np.float64, copy=False)
    P = U @ np.linalg.pinv(U)
    B = U.shape[0]
    I = np.eye(B, dtype=np.float64)
    P_perp = I - P
    P_perp = 0.5 * (P_perp + P_perp.T)
    # Force float32 and return
    return P_perp.astype(np.float32, copy=False)


def project_block_onto_complement(
    block: np.ndarray,
    proj_matrix: np.ndarray|None
) -> np.ndarray:
    """
    Projects every pixel in a block into the orthogonal subspace defined by the projection matrix.

    Args:
        block (np.ndarray):
            Input block of shape (bands, height, width)
        projection_matrix (np.ndarray): 
            Projection matrix of shape (bands, bands)

    Returns:
        np.ndarray: Projected block of same shape as block (bands, height, width)
    """
    # 1 target ("None") = Identity matrix ("block") 
    if proj_matrix is None: return block

    if block.ndim != 3: raise ValueError("block must be 3D (bands, h, w)")

    cdef:
        int bands  = block.shape[0]
        int height = block.shape[1]
        int width  = block.shape[2]

    # Enforce block float32 contiguous 
    block = np.ascontiguousarray(block, dtype=np.float32)
    # Reshape to (bands, pixels)
    reshaped = block.reshape(bands, -1)

    P_mat = np.ascontiguousarray(proj_matrix, dtype=np.float32)
    if P_mat.shape != (bands, bands):
        raise ValueError(f"Bad proj_matrix shape {P_mat.shape}, expected {(bands, bands)}")
    P_mat = 0.5 * (P_mat + P_mat.T)

    projected = P_mat.dot(reshaped)
    return projected.reshape(bands, height, width).astype(np.float32, copy=False)


def compute_opci(
    p_matrix: np.ndarray,
    spectrum: np.ndarray
) -> float:
    """
    Compute Orthogonal Projection Contrast Index (OPCI).


    OPCI = sqrt( (x^T P x) / (x^T x) ).


    - Values near 1.0 mean the vector lies mostly in the orthogonal
    complement (novel target).
    - Values near 0.0 mean the vector lies mostly within the existing
    subspace (redundant target).


    Args:
    p_matrix (np.ndarray): (bands, bands) orthogonal complement matrix.
    spectrum (np.ndarray): 1D spectral vector (bands,).


    Returns:
    float: OPCI value in [0,1].
    """
    # Convert to a 1D contiguous vector 
    x_vec = np.asarray(spectrum, dtype=np.float32).reshape(-1)

    # Replace NaNs/Infs with zero
    if not np.isfinite(x_vec).all():
        x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute denominator: ||x||^2
    denom_energy = np.dot(x_vec, x_vec).item()
    # Very small or zero energy vector — nothing to project.
    if denom_energy <= OPCI_EPS:
        return 0.0

    P = np.ascontiguousarray(p_matrix, dtype=np.float32)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] != x_vec.shape[0]:
        raise ValueError(f"[compute_opci] Bad shapes: P={P.shape}, x={x_vec.shape}")
    P = 0.5 * (P + P.T)


    num_bands = x_vec.shape[0]
    y_vec = np.empty(num_bands, dtype=np.float32)


    cdef float_t[:, :] P_mv = P
    cdef float_t[:] x_mv = x_vec
    cdef float_t[:] y_mv = y_vec


    with nogil:
        _matvec_cy(P_mv, x_mv, y_mv)


    numerator_energy = np.dot(x_vec, y_vec).item()
    opci = numerator_energy / denom_energy


    if not np.isfinite(opci):
        return 0.0
    if opci < -OPCI_TOL:
        return 0.0
    if opci > 1.0 + OPCI_TOL:
        return 1.0


    cdef float_t opci_clamped = <float> min(max(opci, 0.0), 1.0)
    return <float_t> csqrt(opci_clamped)
