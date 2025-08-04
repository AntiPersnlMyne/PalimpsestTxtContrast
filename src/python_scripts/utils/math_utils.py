"""math_utils.py: Linear algebra, matrix, and calculus helper functions"""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.1.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Development" # "Prototype", "Development", "Production"



# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
import numpy as np


# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
def normalize_data(data: np.ndarray, min_val:float|None = None, max_val:float|None= None) -> np.ndarray:
    """
    Normalizes a numpy array to the range [0, 1] using min-max scaling.
    This function scales the data to a common range, which is essential
    for algorithms sensitive to the magnitude of input features.

    Args:
        data (np.ndarray): The input array to be normalized.
        min_val (float, optional): The global minimum value to use for scaling.
                                   If not provided, the local min of 'data' is used.
        max_val (float, optional): The global maximum value to use for scaling.
                                   If not provided, the local max of 'data' is used.

    Returns:
        np.ndarray: The normalized array, with values in the range [0, 1].
    """
    # If global min/max are not provided, use the local min/max from the data
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)

    # Check for division by zero, which would happen if min_val equals max_val
    if max_val and min_val is not None:
        data_range = max_val - min_val
        if  data_range == 0:
            # Return zeros if all values are the same, indicating no variation
            return np.zeros_like(data)

    # Apply the min-max normalization formula using the provided or local min/max
    normalized_data = (data - min_val) / (data_range)
    
    # Ensure all normalized values are within the [0, 1] range due to floating point inaccuracies
    return np.clip(normalized_data, 0, 1)


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
    num_bands, height, width  = block.shape
    reshaped = block.reshape(-1, num_bands)  # shape: (num_pixels, bands)

    projected = reshaped @ projection_matrix.T  # apply projection
    projected_block = projected.reshape(height, width, num_bands)

    return projected_block.astype(np.float32)


def compute_opci(
    projection_matrix: np.ndarray,
    target_vector: np.ndarray
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
    # projected = projection_matrix @ target_vector
    # opci_value = np.linalg.norm(projected)

    # return float(opci_value)
    
    projected = projection_matrix @ target_vector
    numerator = np.linalg.norm(projected) ** 2
    denominator = np.linalg.norm(target_vector) ** 2
    return float(numerator / denominator)



