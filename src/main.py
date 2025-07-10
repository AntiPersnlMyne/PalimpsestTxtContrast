# import numpy as np
# import scipy
# import skimage
# import cv2 as cv
# import matplotlib.pyplot as plt
# import os

from python_scripts import gaussian_blur

if __name__ == "__main__":
    # Create blurred version of the tif files
    gaussian_blur.blur(3)
    print("-- Finished blur --")

