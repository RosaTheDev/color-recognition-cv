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
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask 

# Display the original and processed frames
def display_frame_with_bbox(original_frame):
    cv2.imshow('Webcam Feed', original_frame)
    
# Resize the frame to a smaller scale.
def resize_frame(frame, scale = 0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Draw a red bounding box around the detected blue regions
def draw_bounding_box(frame, mask):
    """
    Draws a red bounding box around the detected color area.
    :param frame: The original frame to draw the box on.
    :param mask: Binary mask where the detected color is white.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red color in BGR
    else: 
        print("No Contours found.")
        
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
        
        # Resize the frame to half its original size
        frame = resize_frame(frame, scale=0.5)
        
        # Convert the frame from BGR to HSV
        hsv_frame = convert_to_hsv(frame)
        
        # Detect the blue color and get the mask
        blue_only, mask = detect_color(frame, hsv_frame, lower_blue, upper_blue)
        
        # Draw red bounding box around the detected blue areas
        draw_bounding_box(frame, mask)
        
        # Display the frame with the bounding box and the blue-detected frame
        display_frame_with_bbox(frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()