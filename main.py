import pca
import tile_processor

import cv2 as cv

if __name__ == "__main__":
    # Path to image file
    image_path = "Data\\120r-121v_Alex02r_Sinar_LED365_01_corr.tif"

    # Which color bands get assigned to each PC componen
    band_order = "rgb"

    # Perform PCA
    pca_image = pca.pca(image_path, band_order)

    # Display the image
    print("Displaying the resulting PCA image. Press any key to close...")
    
    # imshow parameters
    cv.namedWindow("PCA Result", cv.WINDOW_KEEPRATIO | cv.WINDOW_NORMAL)
    # cv.resizeWindow("PCA Result", 800, 600)
    
    # Display image
    cv.imshow("PCA Result", pca_image)
    cv.waitKey(0)
    cv.destroyAllWindows()