import os
import cv2 as cv
import tifffile

def blur(kernel_size: int = 3) -> None:
    """Perform gaussian blur. Outputs to data/output/blur folder.

    Args:
        kernel_size (int, optional): Defined gaussian kernel size, more is more blur. Defaults to 3.
    """
    # Define input data location
    input_path  = "data/input"

    # Create output folders to contain outputs
    output_dir = "data/output/blur"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Process each tif file in input folder
    for filename in os.listdir(input_path):
        if filename.endswith('.tif'):  # Change the extension as needed
            with open(os.path.join(input_path, filename), 'r') as file:
                # Read in tif
                image = cv.imread(file.name)
                # Blur tif
                image_blur = cv.GaussianBlur(image, (kernel_size,kernel_size), cv.BORDER_DEFAULT)
                # Write tif file to output
                cv.imwrite(output_dir + '/' + filename[:-4] + "_blur.tif", image_blur)

        
        