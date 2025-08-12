"""math_utils.py: Linear algebra, matrix, and calculus helper functions"""

from __future__ import annotations

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.2.1"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from numpy import linalg as LA
from numpy.typing import NDArray
from typing import List, Tuple

import numpy as np


# Constants to prevent div0
OPCI_EPS = 1e-12   # denom floor
OPCI_TOL = 1e-9    # clamp tolerance


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
SpectralVector = NDArray[np.float32]
SpectralVectors = Tuple[List[SpectralVector], List[Tuple[int, int]]]



# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
def block_l2_norms(block:np.ndarray) -> np.ndarray:
    """
    Compute L2 (Euclidian) norms from (num_bands, height, width) block

    Args:
        data (np.ndarray): _description_
        min_val (float): _description_
        max_val (float): _description_

    Returns:
        np.ndarray: _description_
    """
    # return normalize(block, block, norm_type=NORM_L2) # opencv
    norms = np.sum(block.astype(np.float32) ** 2, axis=0, dtype=np.float32)
    return np.sqrt(norms, dtype=np.float32)



# --------------------------------------------------------------------------------------------
# Matrix Operand Functions
# --------------------------------------------------------------------------------------------
def compute_orthogonal_projection_matrix(
    target_vectors:list[SpectralVector]
    ) -> np.ndarray:
    """
    Computes the orthogonal projection matrix for a given list of spectral vectors.
    This is used to project pixel vectors orthogonally away from known targets.

    Args:
        target_vectors (list[np.ndarray]): List of 1D target spectral vectors (each shape: [bands])

    Returns:
        np.ndarray[float32]: Orthogonal projection matrix.
    """
    if len(target_vectors) == 0:
        raise ValueError("Must provide at least one target vector")

    # Create the matrix with vectors as columns
    U = np.stack(target_vectors, axis=1)

    # Compute the projection matrix P_U = U(U^T U)^-1 U^T
    UtU = U.T @ U
    
    # Handle the case where UtU is singular (not invertible)
    # This check is crucial for stability, especially when a target vector is a zero vector
    # In a professional context, you would handle this more gracefully.
    if LA.det(UtU) == 0:
        num_bands = U.shape[0]
        I = np.eye(num_bands, dtype=np.float32)
        return I

    UtU_inv = LA.inv(UtU)
    P_U = U @ UtU_inv @ U.T

    # The orthogonal projection matrix is P_orth = I - P_U
    num_bands = U.shape[0]
    I = np.eye(num_bands, dtype=np.float32)
    
    return I - P_U


def project_block_onto_subspace(
    block: np.ndarray,
    projection_matrix: np.ndarray|None
) -> np.ndarray:
    """
    Projects every pixel in a block into the orthogonal subspace defined by the projection matrix.

    Args:
        block (np.ndarray): Input block of shape (bands, height, width)
        projection_matrix (np.ndarray): Projection matrix of shape (bands, bands)

    Returns:
        np.ndarray: Projected block of same shape as block (bands, height, width)
    """
    if projection_matrix is None: return block
    
    bands, height, width = block.shape

    # Reshape block from (bands, height, width) to (pixels, bands)
    # The transpose is needed to correctly align the dimensions
    reshaped = block.reshape(bands, -1).astype(np.float32, copy=False)
    p_matrix =  np.asarray(projection_matrix, dtype=np.float32)
    if p_matrix.shape != (bands, bands):
        raise ValueError(f"[project_block_onto_subspace] Bad shapes: P{p_matrix.shape}, block{block.shape}")
    
    # Apply the projection matrix
    p_matrix = 0.5 * (p_matrix + p_matrix.T)
    
    # Reshape the projected block back to the original shape
    return (p_matrix @ reshaped).reshape(bands,height,width) \
            .astype(np.float32, copy=False)


def compute_opci(
    projection_matrix: np.ndarray,
    spectrum: np.ndarray
) -> float:
    """
    Computes the Orthogonal Projection Correlation Index (OPCI) for a candidate target vector.
    If OPCI is small (e.g., < 0.01), then T is almost already spanned by the previous targets, 
    and should be discarded or used to stop iteration.
    
    Notes:
        This implementation is a modification on the original by Cheng and Ren 2000. They make 
        multiple IO calls to the original image data to compute: ||P_Tx|| / ||x||
        This implementation calls: numerator (x^T * P_T * x) and denominator (x^T * x).
        This is an alternative method to calculate the same values without repeated IO calls.
        The only caveat is that it requires a sqrt, as ||x||^2 = x^T * x, hence the denominator 
        is a "power of magnitude" (is that a word)? off; easily fixed with np.sqrt.

    Args:
        projection_matrix (np.ndarray): Orthogonal projection matrix.
        spectrum (np.ndarray): Original pixel spectrum. 

    Returns:
        float: OPCI value, representing the residual norm after projection
    """

    # Pixel x from [Ren and Cheng 2000]
    x = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    # Clean up potential NaNs/Inf
    if not np.isfinite(spectrum).all():
        x = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)

    # Early out on zero vector
    denom = float(np.dot(x, x))
    if denom <= OPCI_EPS: return 0.0

    # Symmetrize P (guards against tiny asymmetry so x^T P x stays ~real)
    p_matrix = np.asarray(projection_matrix, dtype=np.float64)
    if p_matrix.ndim != 2 or p_matrix.shape[0] != p_matrix.shape[1] \
        or p_matrix.shape[0] != x.shape[0]:
            raise ValueError(f"[compute_opci] Bad shapes: P{p_matrix.shape}, x{x.shape}")
    p_matrix = 0.5 * (p_matrix + p_matrix.T)

    # Quadratic form via y = P x
    y = p_matrix @ x
    numerator = float(np.dot(x, y))

    # Clamp ratio to [0,1] within tolerance
    ratio = numerator / denom
    if not np.isfinite(ratio): return 0.0
    # Negative due to numeric issues; project to 0
    if ratio < -OPCI_TOL: ratio = 0.0
    # Slightly above 1 because of rounding; clamp
    elif ratio > 1.0 + OPCI_TOL: ratio = 1.0
    # Clean tiny negatives/overs by clipping
    else: ratio = float(np.clip(ratio, 0.0, 1.0))

    return float(np.sqrt(ratio))

