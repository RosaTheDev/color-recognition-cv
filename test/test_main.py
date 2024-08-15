import sys # Import the sys module
import os # Import the os module
import cv2
import numpy as np
from unittest.mock import patch
import pytest
import queue
import threading

# Add the parent directory (where main.py is located) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import start_webcam, convert_to_hsv, detect_color, draw_bounding_box

@patch('cv2.VideoCapture')
def test_start_webcam(mock_video_capture):
    # Setup the mocks
    mock_video_capture.return_value.isOpened.return_value = True
    mock_video_capture.return_value.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))  # Mock frame
    
    # Create the necessary arguments
    frame_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    
    # Call the function with the necessary arguments
    capture_thread = threading.Thread(target=start_webcam, args=(frame_queue, stop_event))
    capture_thread.start()
    
    # Start the thread and give it a slight delay to ensure it begins execution
    capture_thread.join(timeout=0.1)
    
    # Let the thread run for a short time and then stop it
    stop_event.set()
    capture_thread.join(timeout=1)  # Ensure the thread joins within 1 second
    
    # Ensure the webcam started successfully
    assert not frame_queue.empty(), "Frame queue should not be empty"

    # Ensure the first frame is a valid numpy array
    frame = frame_queue.get()
    assert isinstance(frame, np.ndarray), "Captured frame should be a numpy array"
    assert frame.shape == (480, 640, 3), "Captured frame should have the correct dimensions"
    
    # Test wehen the webcam fails to start (mock failure)
    mock_video_capture.return_value.isOpened.return_value = False
    with pytest.raises(SystemExit):
        start_webcam(frame_queue, stop_event)

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

# If running directly, execute pytest
if __name__ == "__main__":
    pytest.main()