#!/usr/bin/env python3
# distutils: language=c

"""math_utils.py: Linear algebra, matrix, and calculus helper functions"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from __future__ import annotations
import numpy as np
cimport numpy as np
from libc.math cimport sqrt as csqrt
from typing import List, Tuple


# --------------------------------------------------------------------------------------------
# Authorship Information
# --------------------------------------------------------------------------------------------
__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "3.1.1"
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
# Typedefs for memoryviews
ctypedef np.float32_t float_t
ctypedef np.float64_t double_t

# Typing
cdef extern from *:
    pass



# --------------------------------------------------------------------------------------------
# Helper C kernels
# --------------------------------------------------------------------------------------------
cdef int _block_l2_kernel(
    float_t[:, :, :] block_mv, 
    float_t[:, :] out_mv
) nogil:
    """
    block_mv: (bands, h, w)
    out_mv: (h, w) where we store sqrt(sum_i block[i,r,c]^2)
    """
    cdef:
        Py_ssize_t bands = block_mv.shape[0]
        Py_ssize_t height = block_mv.shape[1]
        Py_ssize_t width = block_mv.shape[2]
        Py_ssize_t b, row, col
        float_t acc
    
    for row in range(height):
        for col in range(width):
            acc = 0.0
            for b in range(bands):
                acc += block_mv[b, row, col] * block_mv[b, row, col]
            out_mv[row, col] = <float_t> csqrt(acc)


cdef int _matvec_quad_form_double(
    double_t[:, :] pmat_mv, 
    double_t[:] x_mv,
    double_t[:] y_mv
) nogil:
    """
    y = P @ x, where P is (n,n) and x is (n,). Output y is (n,).
    """
    cdef:
        Py_ssize_t n = pmat_mv.shape[0]
        Py_ssize_t i, j
        double_t acc
    
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += pmat_mv[i, j] * x_mv[j]
        y_mv[i] = acc


# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
def block_l2_norms(block:np.ndarray) -> np.ndarray:
    """
    Compute L2 (Euclidian) norms from (num_bands, height, width) block

    Args:
        block (np.ndarray): Data block from dataset. 

    Returns:
        np.ndarray: 1D array of norms, each index representing a band.
    """
    if block.ndim != 3:
        raise ValueError("block_l2_norms expects a 3D array (bands, h, w)")

    # Ensure float32 contiguous to take memoryview
    if block.dtype != np.float32 or not block.flags['C_CONTIGUOUS']:
        block = np.ascontiguousarray(block, dtype=np.float32)

    cdef height = block.shape[1]
    cdef width  = block.shape[2]
    
    norms = np.empty((height, width), dtype=np.float32)

    # Obtain typed memoryviews and call nogil kernel
    cdef float_t[:, :, :] block_mv = block
    cdef float_t[:, :] out_mv = norms

    # Use Cython to compute norms with noGIL
    with nogil:
        _block_l2_kernel(block_mv, out_mv)

    # out_mv is reference to norms
    return norms


# --------------------------------------------------------------------------------------------
# Matrix Operand Functions
# --------------------------------------------------------------------------------------------
def compute_orthogonal_projection_matrix(
    target_vectors:list[np.ndarray]
    ) -> np.ndarray:
    """
    Computes orthogonal projection matrix P_orth = I - U @ pinv(U)
    where U stacks target_vectors columnwise (B, K).
    Returns float32 symmetric orthogonal projector of shape (B,B).

    Args:
        target_vectors (list[np.ndarray]): List of 1D target spectral vectors (each shape: [bands])

    Returns:
        np.ndarray[float32]: Orthogonal projection matrix.
    """
    if len(target_vectors) == 0:
        raise ValueError("Must provide at least one target vector")

    # Stack into double precision for numeric stability in pinv
    U = np.stack(target_vectors, axis=1).astype(np.float64, copy=False)  # (bands, k-targets)
    # NumPy pinv relies on BLAS/LAPACK = (fast, multithreaded)
    P = U @ np.linalg.pinv(U)
    B = U.shape[0]
    I = np.eye(B, dtype=np.float64)
    P_orth = I - P
    # symmetrize and cast once
    P_orth = 0.5 * (P_orth + P_orth.T)
    # Enforce float32 and return
    return P_orth.astype(np.float32, copy=False)


def project_block_onto_subspace(
    block: np.ndarray,
    projection_matrix: np.ndarray|None
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
    if projection_matrix is None: return block

    if block.ndim != 3: raise ValueError("block must be 3D (bands, h, w)")

    cdef int bands  = block.shape[0]
    cdef int height = block[1]
    cdef int width  = block[2]

    # Enforce block float32 contiguous 
    if block.dtype != np.float32 or not block.flags['C_CONTIGUOUS']:
        block = np.ascontiguousarray(block, dtype=np.float32)

    # Reshape to (bands, pixels)
    reshaped = block.reshape(bands, -1).astype(np.float32, copy=False)

    # Force projection matrix to float32 and symmetrize
    pmat = np.asarray(projection_matrix, dtype=np.float32)
    if pmat.shape != (bands, bands):
        raise ValueError(f"[project_block_onto_subspace] Bad shapes: P{pmat.shape}, block{(block.shape[0], block.shape[1])}")

    pmat = 0.5 * (pmat + pmat.T)

    # Compute (bands, pixels)
    projected = pmat.dot(reshaped)

    # Reshape back and ensure float32
    return projected.reshape(bands, height, width).astype(np.float32, copy=False)


def compute_opci(
    projection_matrix: np.ndarray,
    spectrum: np.ndarray
) -> float:
    """
    Computes OPCI = sqrt( (x^T * P * x) / (x^T * x) ).

    This returns a scalar in [0,1] measuring how much of `spectrum` (x)
    lies *in the subspace defined by* projection_matrix P. Small values
    (near 0) mean x is mostly orthogonal to the projected subspace; large
    values (near 1) mean x is mostly inside the subspace.

    Args:
        projection_matrix (np.ndarray):
            Orthogonal projection matrix P (shape: (B,B)) produced by
            compute_orthogonal_projection_matrix(...). It should be symmetric;
            we will symmetrize it defensively.
        spectrum (np.ndarray):
            1D spectral vector (shape: (B,) or (B,1)). This is the pixel
            spectrum being evaluated.

    Returns:
        float: OPCI value in [0,1] (sqrt of ratio of projected energy to total energy).
    """
    # Convert to a 1D contiguous vector in double precision for numerical stability.
    x_vec = np.asarray(spectrum, dtype=np.float64).reshape(-1)

    # Replace NaNs/Infs with zero
    if not np.isfinite(x_vec).all():
        x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)

    # ============================
    # Compute denominator: ||x||^2
    # ============================
    denom_energy = float(np.dot(x_vec, x_vec))
    # Very small or zero energy vector — nothing to project.
    if denom_energy <= OPCI_EPS:
        return 0.0

    # =================
    # Projection Matrix
    # =================
    # Load the projection matrix as double precision. 
    pmat = np.asarray(projection_matrix, dtype=np.float64)
    if pmat.ndim != 2 or pmat.shape[0] != pmat.shape[1] or pmat.shape[0] != x_vec.shape[0]:
        raise ValueError(f"[compute_opci] Bad shapes: P={pmat.shape}, x={x_vec.shape}")

    # Numerical ops on P can introduce tiny asymmetries. Symmetrize to keep
    # the quadratic form x^T P x real and numerically stable.
    pmat = 0.5 * (pmat + pmat.T)

    # =================
    # Compute y = P @ x  
    # =================
    # Allocate y as a double vector and compute the matrix-vector product
    num_bands = x_vec.shape[0]
    y_vec = np.empty(num_bands, dtype=np.float64)

    # Create typed memoryviews to pass into the nogil kernel.
    cdef double_t[:, :] pmat_mv = pmat
    cdef double_t[:] x_mv = x_vec
    cdef double_t[:] y_mv = y_vec

    # Compute y_mv[:] = pmat_mv @ x_mv ; in noGIL
    with nogil:
        _matvec_quad_form_double(pmat_mv, x_mv, y_mv)

    # ==========
    # OPCI ratio
    # ==========
    # numerator_energy = x^T * (P * x) = x^T y
    numerator_energy = float(np.dot(x_vec, y_vec))

    # ratio is the fractional energy captured by projection: numerator / denom
    opci = numerator_energy / denom_energy

    # ==================
    # Clamp values [0,1]
    # ==================
    # Non-finite result 
    if not np.isfinite(opci):
        return 0.0
    # Negative beyond tolerance.
    if opci < -OPCI_TOL:
        opci_clamped = 0.0
    # Positive beyond tolerance
    elif opci > 1.0 + OPCI_TOL:
        opci_clamped = 1.0
    # Final [0,1] clamp
    else:
        if opci < 0.0:
            opci_clamped = 0.0
        elif opci > 1.0:
            opci_clamped = 1.0
        else:
            opci_clamped = float(opci)

    # =============
    # Correct Norms
    # =============
    # Used formula deviates slightly from original paper -> computes a ratio of squared norms; 
    # Fix: take the square root
    return float(np.sqrt(opci_clamped))
