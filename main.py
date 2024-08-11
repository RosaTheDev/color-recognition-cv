# Import the necessary libraries
import cv2 # OpenCV library for computer vision tasks
import numpy as np # For handling arrays
# Start the webcam
cap = cv2.VideoCapture(0) # '0' is the default webcam



# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
    
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break
    
    # Convert the frame from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the HSV reange for the color blue
    lower_blue = np.array([100, 150, 150]) # Lower bound for blue
    upper_blue = np.array([140, 255, 255]) # Upper bound for blue
    
    # Create a mask for the blue color
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
    # Apply the mask to get the blue parts of the frame
    blue_only = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Display the original frame
    cv2.imshow('Webcam Feed', frame)
    
    # Display the HSV frame || not really needed but cool to see what Hue, Saturation and Value the camera picks up
    cv2.imshow('HSV Feed', hsv_frame)
    
    # Display the blue-detected frame
    cv2.imshow('Blue Detection', blue_only)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()