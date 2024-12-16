import cv2
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Define the HSV ranges for different colors (global definition)
colors = {
    'red1': (np.array([0, 150, 150]), np.array([10, 255, 255])),  # More precise red
    'red2': (np.array([170, 150, 150]), np.array([180, 255, 255])),
    'blue': (np.array([90, 60, 80]), np.array([130, 255, 255])),  # Adjusted to include darker blue
    'green': (np.array([45, 55, 55]), np.array([90, 255, 255]))  # Adjusted green range
}

# Initialize a dictionary to store the positions of detected objects
tracker = {'red': None, 'blue': None, 'green': None}


def start_webcam(frame_queue, stop_event):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        raise SystemExit(1)
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
    finally:
        cap.release()  # Ensure the camera is released when the loop exits

def process_frame(frame_queue, processed_queue, stop_event):
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            hsv_frame = convert_to_hsv(frame)

            # Red detection and tracking
            _, mask1 = detect_color(frame, hsv_frame, colors['red1'][0], colors['red1'][1])
            _, mask2 = detect_color(frame, hsv_frame, colors['red2'][0], colors['red2'][1])
            red_mask = cv2.bitwise_or(mask1, mask2)
            red_mask = cv2.erode(red_mask, None, iterations=2)
            red_mask = cv2.dilate(red_mask, None, iterations=2)
            update_tracker('red', red_mask, frame)

            # Blue detection and tracking
            _, blue_mask = detect_color(frame, hsv_frame, colors['blue'][0], colors['blue'][1])
            blue_mask = cv2.erode(blue_mask, None, iterations=2)
            blue_mask = cv2.dilate(blue_mask, None, iterations=2)
            update_tracker('blue', blue_mask, frame)

            # Green detection and tracking
            _, green_mask = detect_color(frame, hsv_frame, colors['green'][0], colors['green'][1])
            green_mask = cv2.erode(green_mask, None, iterations=2)
            green_mask = cv2.dilate(green_mask, None, iterations=2)
            update_tracker('green', green_mask, frame)
            
            processed_queue.put(frame)
        else:
            continue

def convert_to_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def detect_color(frame, hsv_frame, lower_bound, upper_bound):
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask

def update_tracker(color_name, mask, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        tracker[color_name] = (x, y, w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, f'{color_name} tracking', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        tracker[color_name] = None  # No object detected

def display_frame_with_bbox(original_frame):
    cv2.imshow('Webcam Feed', original_frame)

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
                    display_frame_with_bbox(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
        finally:
            stop_event.set()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
