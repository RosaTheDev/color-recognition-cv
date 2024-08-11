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
    
    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()