""" gui.py: A devleopment playground for display and manipulation techniques with GUI input/output """

__author__ = "Gian-Mateo (GM) Tifone"
__copyright__ = "2025, RIT MISHA"
__credits__ = ["Gian-Mateo Tifone"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "MISHA Team"
__email__ = "mt9485@rit.edu"
__status__ = "Prototype" # "Development", or "Production", or "Prototype". 

import cv2 as cv
import numpy as np

def get_click_coord(event:int,x:int,y:int, flags, param):
    """
    Gets mouseX and mouseY (click coordinates)

    Args:
        event: Keyboard/Mouse (aka. user) input
        x (int): x/horizontal coordinate of mouse
        y (int): y/vertical coordinate of mouse
    """
    # Globally accessable mouseX and mouseY coordinate after click
    global mouseX,mouseY
    
    # Listen for mouse event
    if event == cv.EVENT_LBUTTONUP:
        mouseX,mouseY = x,y
        
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', get_click_coord)

while(1):
    cv.imshow('image',img)
    key = cv.waitKey(20) & 0xFF
    if key == 27: # I assume 'esc'?
        break
    elif key == ord('a'): # Prints the clicked coord
        print (mouseX, mouseY)







