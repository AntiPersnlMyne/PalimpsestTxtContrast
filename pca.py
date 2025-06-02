import numpy as np
import tifffile
from sklearn.decomposition import PCA
import cv2 as cv
from typing import Optional

# pip install numpy tifffile scikit-learn opencv-python

def pca(image_path: str, band_order: str = "bgr", verbose: bool = False) -> Optional[np.ndarray]:
    """
    Performs Principal Component Analysis (PCA) on a multi-spectral TIFF image
    
    Args:
        image_path (str): The path to the input TIFF image file.
        band_order (str): Assignment of PC components to output channels (BGR by default).
                          The assignment will always proceed PC1, then PC2, then PC3
                          in correspondence with the band_order string.
                          E.g., "bgr" will correspond to: PC1->Blue, PC2->Green, PC3->Red.
                          Can include overlapping channels (e.g., "bgb"), where unassigned
                          channels are zero-padded (black).
                          
    Returns:
        NDArray[np.uint8] or None: The principal components as a 3-channel BGR image 
                                   (np.uint8 format), or None if an error occurs.
    """
    
# ---------------------------------------------------------------------------------------------
# Section 0: Initialization
# ---------------------------------------------------------------------------------------------
    # Initialize verbose debug statements; prints only is verbose is true
    v_print = print if verbose else lambda *a, **k: None
    
    # Check valid band order
    allowed_band_chars = 'rgb' # Allowed characters (i.e., channels of a BGR image)
    if len(band_order) == 3 and all(char in allowed_band_chars for char in band_order):
        # Convert user input to lowercase if necessary
        band_order = band_order.lower()
    else:
        print(f"Error: Invalid pca band order '{band_order}'. Please provide a 3-character band order using only 'r', 'g', or 'b'")
        print("Example: 'rgb' or 'bgb'")
        return None
    
    num_pca_components = 3 # Number of principal components out from PC analysis
    
    
# ---------------------------------------------------------------------------------------------
# Section 1: Load in image
# ---------------------------------------------------------------------------------------------
    try:
        v_print(f"Loading TIFF image from: {image_path}")
        img = tifffile.imread(image_path)

        # Check the image is multi-dimensional (bands)
        if img.ndim < 3:
            print("Error: Input image must have at least 3 spectral bands (channels).")
            return None

        #Get dimensions of image (rows, columns, bands)
        rows, cols, bands = img.shape
        
        
# ---------------------------------------------------------------------------------------------
# Section 2: Prepare the image data for PCA
# ---------------------------------------------------------------------------------------------
        v_print("Reshaping image data for PCA")
       
        # Flatten/vectorize each band
        # PCA input is (samples, features), where pixels are samples and bands are features.
        img_reshaped = img.reshape(rows * cols, bands)
        
        v_print(f"Reshaped image shape for PCA: {img_reshaped.shape}")
        
        
# ---------------------------------------------------------------------------------------------
# Section 3: Perform PCA
# ---------------------------------------------------------------------------------------------
        # Perform PCA using imported library
        v_print(f"Performing PCA with {num_pca_components} components")
        pca = PCA(num_pca_components)
        principal_components = pca.fit_transform(img_reshaped)
        v_print(f"Shape of principal components: {principal_components.shape}")


# ---------------------------------------------------------------------------------------------
# Section 4: Restructure in displayable image format
# ---------------------------------------------------------------------------------------------
        # Construct into standard image format (row x col x channel)
        v_print("Reshaping principal components back to image format")
        # pca_image holds raw unnormalized PC values=.
        pca_image = principal_components.reshape(rows, cols, num_pca_components)
        v_print(f"Shape of PCA image: {pca_image.shape}")


# ---------------------------------------------------------------------------------------------
# Section 5: Apply Band Order
# ---------------------------------------------------------------------------------------------
        v_print("Applying band order and preparing final image.")
        
        # Array for final image
        band_ordered_image = np.zeros((rows, cols, 3), dtype=np.float32) 

        # Define the mapping from 'rgb' characters to BGR output channel indices
        # 'b' -> 0 (Blue channel), 'g' -> 1 (Green channel), 'r' -> 2 (Red channel)
        channel_map = {'b': 0, 'g': 1, 'r': 2}
         
        # Assign principal components to output channels based on band_order
        # PC1 corresponds to principal_components[:, 0]
        # PC2 corresponds to principal_components[:, 1]
        # PC3 corresponds to principal_components[:, 2]
        
        for i in range(num_pca_components):
            # Get the character for the i-th component from band_order (e.g., 'r' for PC1 if band_order is 'rgb')
            band_char = band_order[i]
            # Get the output channel index for this character
            dst_channel_index = channel_map[band_char]
            # Assign the i-th principal component to the output channel
            band_ordered_image[:, :, dst_channel_index] += pca_image[:, :, i]

        v_print("Image bands re-ordered")


# ---------------------------------------------------------------------------------------------
# Section 6: Normalize for Display and Return Result
# ---------------------------------------------------------------------------------------------
 # Normalize each principal component independently to 0-255 range
        v_print("Normalizing principal components for display")
        
        # Initialize the final output image with uint8 type
        final_image = np.zeros_like(band_ordered_image, dtype=np.uint8)
        
        # Iterate through output channels (BGR)
        for i in range(num_pca_components):
            # Channel-wise normalization
            channel_min_val = band_ordered_image[:, :, i].min()
            channel_max_val = band_ordered_image[:, :, i].max()
            
            if channel_max_val - channel_min_val > 0:
                final_image[:, :, i] = ((band_ordered_image[:, :, i] - channel_min_val) 
                                        / (channel_max_val - channel_min_val) * 255).astype(np.uint8)
            else:
                final_image[:, :, i] = np.zeros_like(band_ordered_image[:, :, i], dtype=np.uint8)
    
        v_print("PCA processing complete")
        return final_image
    
# ---------------------------------------------------------------------------------------------
# Section 7: Exception handling
# ---------------------------------------------------------------------------------------------
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None