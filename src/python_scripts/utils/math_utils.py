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
from numba import njit
from numpy import linalg as LA
from numpy.typing import NDArray
from typing import List, Tuple
from warnings import warn


# --------------------------------------------------------------------------------------------
# Custom Datatypes
# --------------------------------------------------------------------------------------------
SpectralVector = NDArray[np.float32]
SpectralVectors = Tuple[List[SpectralVector], List[Tuple[int, int]]]


# --------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------
# @njit(fastmath=True, cache=True)
def normalize_data(
    data: np.ndarray,
    min_val:float,
    max_val:float
    ) -> np.ndarray:
    """
    Normalizes a numpy array to the range [0, 1] using min-max scaling.

    Args:
        data (np.ndarray): The input array to be normalized.

    Returns:
        np.ndarray: The normalized array, with values in the range [0, 1].
    """
    
    print(f"[math_utils] Data shape is {data.shape}")
    
    # Check datatype
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
        
    # Flatten data
    orig_shape = data.shape
    data = data.flatten()
    
    # Normalize data to range [0,1]
    data = (data - min_val) / (max_val - min_val)

    # Restore data in original shape
    return data.reshape(orig_shape)
    


# --------------------------------------------------------------------------------------------
# Matrix Operand Functions
# --------------------------------------------------------------------------------------------
# @njit(fastmath=True, cache=True)
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



# @njit(fastmath=True, cache=True)
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
    
    # --- ADD THESE PRINT STATEMENTS ---
    print(f"Shape of reshaped matrix (A): {reshaped.shape}")
    print(f"Shape of projection_matrix (B): {projection_matrix.shape}")
    # ----------------------------------

    # Apply the projection matrix
    projected = reshaped @ projection_matrix
    
    # Reshape the projected block back to the original shape
    return projected.T.reshape(num_bands, height, width)


# @njit(fastmath=True, cache=True)
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
