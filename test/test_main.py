import cv2
import numpy as np
import pytest
import sys # Import the sys module
import os # Import the os module
# Add the parent directory (where main.py is located) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import start_webcam, convert_to_hsv, detect_color

def test_start_webcam():
     cap = start_webcam()
     assert cap.isOpened, "Webcam failed to start."
     
def test_convert_to_hsv():
   # Create a dummy BGR image (a simple 2x2 image with blue color)
    bgr_image = np.array([[[255, 0, 0], [255, 0, 0]], [[255, 0, 0], [255, 0, 0]]], dtype=np.uint8)

    # Convert the BGR image to HSV
    hsv_image = convert_to_hsv(bgr_image)

    # Expected HSV value for pure blue
    expected_hsv = np.array([[[120, 255, 255], [120, 255, 255]], [[120, 255, 255], [120, 255, 255]]], dtype=np.uint8)

    assert np.array_equal(hsv_image, expected_hsv), "HSV conversion failed."
    
def test_detect_color():
    # Create a dummy BGR image (a simple 2x2 image with blue color)
    bgr_image = np.array([[[255, 0, 0], [255, 0, 0]], [[255, 0, 0], [255, 0, 0]]], dtype=np.uint8)
    hsv_image = convert_to_hsv(bgr_image)
    
    # Define the HSV range for blue
    lower_blue = np.array([100, 150, 150])
    upper_blue = np.array([140, 255, 255])

    # Detect blue in the dummy image
    result = detect_color(bgr_image, hsv_image, lower_blue, upper_blue)
    
    # Since the entire image is blue, the result should be the same as the input image
    assert np.array_equal(result, bgr_image), "Color detection failed for blue."

# If running directly, execute pytest
if __name__ == "__main__":
    pytest.main()