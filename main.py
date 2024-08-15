import cv2
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Function to capture frames from the webcam
def start_webcam(frame_queue, stop_event):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame from webcam.")
            continue
        frame_queue.put(frame)
    cap.release()

# Function to process frames (convert to HSV, detect color, draw bounding box)
def process_frame(frame_queue, processed_queue, stop_event):
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            hsv_frame = convert_to_hsv(frame)
            lower_blue = np.array([100, 150, 150])
            upper_blue = np.array([140, 255, 255])
            _, mask = detect_color(frame, hsv_frame, lower_blue, upper_blue)
            draw_bounding_box(frame, mask)
            processed_queue.put(frame)  # Put the original frame with bounding box in the queue
        else:
            continue

# Helper function to convert BGR to HSV
def convert_to_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Helper function to detect color in the frame
def detect_color(frame, hsv_frame, lower_bound, upper_bound):
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask 

# Helper function to draw bounding box around detected color
def draw_bounding_box(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red bounding box

# Main function to handle threading and display frames
def main():
    frame_queue = queue.Queue(maxsize=10)
    processed_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Start capture and processing threads
        executor.submit(start_webcam, frame_queue, stop_event)
        executor.submit(process_frame, frame_queue, processed_queue, stop_event)

        try:
            while True:
                if not processed_queue.empty():
                    frame = processed_queue.get()
                    cv2.imshow('Webcam Feed', frame)  # Display the original frame with bounding box

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

        finally:
            stop_event.set()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
