# Import the necessary libraries
import cv2 # OpenCV library for computer vision tasks

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
    
    # Display the original frame
    cv2.imshow('Webcam Feed', frame)
    
    # Display the HSV frame || not really needed but cool to see what Hue, Saturation and Value the camera picks up
    # cv2.imshow('HSV Feed', hsv_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()