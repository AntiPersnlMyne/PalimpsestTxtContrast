"""math_utils.py: Linear algebra, matrix, and calculus helper functions"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np


# --------------------------------------------------------------------------------------------
# Matrix Operand Functions
# --------------------------------------------------------------------------------------------
def compute_orthogonal_projection_matrix(target_vectors:list[np.ndarray]) -> np.ndarray:
    """
    Computes the orthogonal projection matrix for the subspace spanned by target vectors.
    This is used to project pixel vectors orthogonally away from known targets.

    Args:
        target_vectors (list[np.ndarray]): List of 1D target vectors (each shape: [bands])

    Returns:
        np.ndarray: Type=Float32. Orthogonal projection matrix of shape (bands, bands)
    """
    if len(target_vectors) == 0:
        raise ValueError("Must provide at least one target vector")

    # Stack into matrix M (shape: [bands, k])
    matrix_M = np.column_stack(target_vectors)  # shape: (bands, k)
    
    # Compute projection: P = I - M(MᵀM)^-1 Mᵀ
    # @ is Python shorthand for __matmul__ ("matrix multuiplication")
    identity = np.eye(matrix_M.shape[0], dtype=np.float32)
    pinv_term = np.linalg.pinv(matrix_M.T @ matrix_M) @ matrix_M.T
    projection_matrix = identity - matrix_M @ pinv_term

    return projection_matrix.astype(np.float32)


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
    height, width, num_bands = block.shape
    reshaped = block.reshape(-1, num_bands)  # shape: (num_pixels, bands)

    projected = reshaped @ projection_matrix.T  # apply projection
    projected_block = projected.reshape(height, width, num_bands)

    return projected_block.astype(np.float32)


def compute_opci(
    projection_matrix: np.ndarray,
    candidate_target: np.ndarray
) -> float:
    """
    Computes the Orthogonal Projection Correlation Index (OPCI) for a candidate target vector.
    If OPCI is small (e.g., < 0.01), then T is almost already spanned by the previous targets, 
    and should be discarded or used to stop iteration

    Args:
        projection_matrix (np.ndarray): Orthogonal projection matrix (shape: [bands, bands])
        candidate_target (np.ndarray): Target candidate vector (shape: [bands])

    Returns:
        float: OPCI value, representing the residual norm after projection
    """
    projected = projection_matrix @ candidate_target
    opci_value = np.linalg.norm(projected)

    return float(opci_value)


