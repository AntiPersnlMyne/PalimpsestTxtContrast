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
from screeninfo import get_monitors

def display(image:np.ndarray, win_name:str, wait_time_ms:int = 0) -> None:
    """
    Displays image. Displays as uint8, so does not faithfully display original image bit depth.

    Args:
        image (np.ndarray): Image to display.
        win_name (str): Title of shown image window.
        wait_time_ms (int, optional): Image window destruction duration determined by wait_time_ms; 0 means show indefinitely. Defaults to 0.
        
    Returns:
        None
    """
    
    # if image.dtype != np.uint8:
    #     dtype_info = np.iinfo(np.uint8)
    #     # Auto-normalize for display purposes only
    #     image = cv.normalize(image, np.empty(0), dtype_info.min, dtype_info.max, norm_type=cv.NORM_MINMAX).astype(np.uint8)
    
    for m in get_monitors():
        # NOTE: Assumes you only have one monitor. May break if using multi-monitor system.
        global window_width, window_height
        
        # Adjust window to be 60% 
        window_width = m.width * 0.6
        window_height = m.height * 0.6
        break
        
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, int(window_width), int(window_height))
    cv.imshow(win_name, image)
    cv.waitKey(wait_time_ms)
    
    return None
    
    
def close_windows() -> None:
    """Destroy (close) all OpenCV windows"""
    cv.destroyAllWindows()
    return None


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

# while(1):
#     cv.imshow('image',img)
#     key = cv.waitKey(20) & 0xFF
#     if key == 27: # I assume 'esc'?
#         break
#     elif key == ord('a'): # Prints the clicked coord
#         print (mouseX, mouseY)







