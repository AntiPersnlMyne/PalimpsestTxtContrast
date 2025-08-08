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
from numba import njit
from numpy import linalg as LA
from numpy.typing import NDArray
from typing import List, Tuple, Iterable, Sequence
from dataclasses import dataclass
from cv2 import normalize, NORM_L2
from ..atdca.rastio import WindowType, MultibandBlockReader

import numpy as np



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
    projection_matrix: np.ndarray
) -> np.ndarray:
    """
    Projects every pixel in a block into the orthogonal subspace defined by the projection matrix.

    Args:
        block (np.ndarray): Input block of shape (height, width, bands)
        projection_matrix (np.ndarray): Projection matrix of shape (bands, bands)

    Returns:
        np.ndarray: Projected block of same shape as block (height, width, bands)
    """
    num_bands, height, width = block.shape

    # Reshape block from (bands, height, width) to (pixels, bands)
    # The transpose is needed to correctly align the dimensions
    reshaped = block.reshape(num_bands, height * width).T

    # Apply the projection matrix
    projected = reshaped @ projection_matrix
    
    # Reshape the projected block back to the original shape
    return projected.T.reshape(num_bands, height, width)


def compute_opci(
    projection_matrix: np.ndarray,
    target_vector: SpectralVector
) -> float:
    """
    Computes the Orthogonal Projection Correlation Index (OPCI) for a candidate target vector.
    If OPCI is small (e.g., < 0.01), then T is almost already spanned by the previous targets, 
    and should be discarded or used to stop iteration

    Args:
        projection_matrix (np.ndarray): Orthogonal projection matrix.
        candidate_target (np.ndarray): Target candidate vector.

    Returns:
        float: OPCI value, representing the residual norm after projection
    """

    numerator = target_vector.T @ projection_matrix @ target_vector
    denominator = target_vector.T @ target_vector
    
    # We use .item() to extract the single float value from the resulting NumPy arrays.
    return float( (numerator / denominator).item() ) # float ( [#]-># )

