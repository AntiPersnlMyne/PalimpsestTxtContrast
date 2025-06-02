"""Performs image processing techniques on source (src) image. Saves result to destination (dst) image.
   - {Section 0} contains all user parameters ('hyperparameters') 
   - Order of image processing as follows:
   -- PCA
   -- CLAHE
"""

import img_processors  # image processing functions
import process_by_tile # Tile-wise operations on image

import cv2 as cv   # imread and imwrite
import tifffile    # Reading in tiff files
import os          # Import the os module for path operations
import numpy as np # Array operations and OpenCV compatability

# Silence Pylance being overly picky
# pyright: reportCallIssue=false

if __name__ == "__main__":
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!  [ ONLY SECTION YOU SHOULD MODIFY] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ---------------------------------------------------------------------------------------------
# Section 0: Runtime Flags 
# ---------------------------------------------------------------------------------------------
    # Path to image file 
    img_path = "../data/120r-121v_Alex02r_Sinar_LED365_01_corr.tif"
    # img_path = "Data/dst_images/ENVI_PCA.png"
    
    # Path to save image
    dst_path = "../data/dst_images"
    
    # Destination filename
    dst_filename = "processed_palimpsest"
    
    # Destination file-format
    dst_file_extension = ".png"
    
    # Band order for PCA assignment
    band_order = "bgr"
    
    # Tile size (in px) for PNG compression like processing--tilewise--across image
    tile_size = 64
    
    
# ---------------------------------------------------------------------------------------------
# Section 1: Load in image(s)
# ---------------------------------------------------------------------------------------------
    # Debug message
    print(f"Loading image from: {img_path}")
    
    # Determine file extension to choose the correct loader
    _, file_extension = os.path.splitext(img_path)
    # Convert to lowercase for consistent checking
    file_extension = file_extension.lower() 
    
    # Temporary variable for imported image (to silence Pylance)
    img = np.zeros((1, 1, 1), dtype=np.float64) 
    
    # Import image
    try: 
        # If tiff file extension:
        if file_extension == '.tif' or file_extension == '.tiff': 
            img = tifffile.imread(img_path) 
        # All other file extensions:
        else: 
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    
    # Exception Handling
    except (FileNotFoundError):
        print("Source image not found")
    except Exception as e:
        print("Error: Could not read in source image\n\nCheck your path, filename, and compatible file extension.")

    # Check the image is multi-dimensional (bands)
    if img.ndim < 3:
        print("Error: Input image must have at least 3 spectral bands (channels).")
    
    
# ---------------------------------------------------------------------------------------------
# Section 2: PCA
# ---------------------------------------------------------------------------------------------
    # Custom flag. Lets user/ENVI do the PCA, and feed result to algorithm 
    if (file_extension[:3] != "envi"):        
        # Perform PCA tile-wise                  
        img = process_by_tile.tile_process(img, img_processors.pca, tile_size, band_order, False, band_order, False)
        
        # Breakdown of the function call:
        # tile_process(image, processor/function, tile size, band order, verbose of tile_process, **arguments for processor)
        # **arguments for processor = band_order, False
    
    
# ---------------------------------------------------------------------------------------------
# Section 3: CLAHE (Contrast Limited Adaptive Histogram Enhancement)
# ---------------------------------------------------------------------------------------------
    
    
    
# ---------------------------------------------------------------------------------------------
# Section 4: Save Image
# ---------------------------------------------------------------------------------------------
    # Initialize which image is getting saved
    dst_image = img

    # Full imwrite destination
    destination = dst_path+dst_filename+dst_file_extension
    
    # Write image to destination
    cv.imwrite(destination, dst_image)