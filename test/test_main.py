import sys # Import the sys module
import os # Import the os module

import cv2
import numpy as np
from unittest.mock import patch
import pytest

# Add the parent directory (where main.py is located) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import start_webcam, convert_to_hsv, detect_color, draw_bounding_box

@patch('cv2.VideoCapture')
def test_start_webcam(mock_video_capture):
    # Test when the webcam starts successfully
    mock_video_capture.return_value.isOpened.return_value = True
    cap = start_webcam()
    assert cap.isOpened(), "Webcam failed to start."
    
    # Test wehen the webcam fails to start (mock failure)
    mock_video_capture.return_value.isOpened.return_value = False
    with pytest.raises(SystemExit):
        start_webcam()

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
    _, mask = detect_color(bgr_image, hsv_image, lower_blue, upper_blue)
    
    # Expected mask should be white (255) for the detected blue regions
    expected_mask = np.array([[255, 255], [255, 255]], dtype=np.uint8)
    
    # Check if the mask matches the expected mask
    assert np.array_equal(mask, expected_mask), "Mask creation failed for blue detection."
    
def test_draw_bounding_box():
    # Create a dummy BGR image (a simple 2x2 image with blue color)
    bgr_image = np.array([[[255, 0, 0], [255, 0, 0]], [[255, 0, 0], [255, 0, 0]]], dtype=np.uint8)
    hsv_image = convert_to_hsv(bgr_image)
    
    # Define the HSV range for blue
    lower_blue = np.array([100, 150, 150])
    upper_blue = np.array([140, 255, 255])

    # Detect blue in the dummy image
    _, mask = detect_color(bgr_image, hsv_image, lower_blue, upper_blue)
    
    # Draw the bounding box
    draw_bounding_box(bgr_image, mask)
    
    # The function does not return anything, so we can only verify that it runs without error
    assert bgr_image is not None, "Bounding box drawing failed."

    
    # Draw the bounding box
# If running directly, execute pytest
if __name__ == "__main__":
    pytest.main()