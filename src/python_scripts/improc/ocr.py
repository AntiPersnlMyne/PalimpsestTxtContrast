"""ocr.py: Optical Character Recognition (OCR) workflow. Uses OpenCV and pytesseract."""

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production", or "Prototype". 


import cv2 as cv
import pytesseract

# Pyteseract expects RGB
img = cv.imread(r'data/input/Arch_165r_370nm.tif', cv.IMREAD_COLOR_RGB)

# Print pytesseract's guess at reading text
print( pytesseract.image_to_string(img) )


