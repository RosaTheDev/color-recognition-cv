# Import the necessary libraries
import cv2 # OpenCV library for computer vision tasks
import numpy as np # For handling arrays

# Initialize the webcam and return the VideoCapture object.
def start_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    return cap

 # Convert a BGR frame to HSV color space.
def convert_to_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask and detect a specific color in the frame 
def detect_color(frame, hsv_frame, lower_bound, upper_bound):
   mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
   return cv2.bitwise_and(frame, frame, mask=mask)

def main():
    # Start the webcam
    cap = start_webcam()
    
    # Define the HSV range for the color blue
    lower_blue = np.array([100, 150, 150])
    upper_blue = np.array([140, 255, 255])
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting...")
            break
        
        # Convert the frame from BGR to HSV
        hsv_frame = convert_to_hsv(frame)
        
        # Detect the blue color
        blue_only = detect_color(frame, hsv_frame, lower_blue, upper_blue)
        
        
        # Display the original frame
        cv2.imshow('Webcam Feed', frame)
        
        # Display the blue-detected frame
        cv2.imshow('Blue Detection', blue_only)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()