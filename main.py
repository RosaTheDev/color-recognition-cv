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
    'green': (np.array([45, 55, 55]), np.array([130, 255, 255]))  # Adjusted green range
}


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

            # Red detection
            _, mask1 = detect_color(frame, hsv_frame, colors['red1'][0], colors['red1'][1])
            _, mask2 = detect_color(frame, hsv_frame, colors['red2'][0], colors['red2'][1])
            red_mask = cv2.bitwise_or(mask1, mask2)
            red_mask = cv2.erode(red_mask, None, iterations=2)
            red_mask = cv2.dilate(red_mask, None, iterations=2)
            draw_bounding_box(frame, red_mask)
            cv2.putText(frame, 'red', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Blue detection
            _, blue_mask = detect_color(frame, hsv_frame, colors['blue'][0], colors['blue'][1])
            blue_mask = cv2.erode(blue_mask, None, iterations=2)
            blue_mask = cv2.dilate(blue_mask, None, iterations=2)
            draw_bounding_box(frame, blue_mask)
            cv2.putText(frame, 'blue', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Green detection
            _, green_mask = detect_color(frame, hsv_frame, colors['green'][0], colors['green'][1])
            green_mask = cv2.erode(green_mask, None, iterations=2)
            green_mask = cv2.dilate(green_mask, None, iterations=2)
            draw_bounding_box(frame, green_mask)
            cv2.putText(frame, 'green', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            processed_queue.put(frame)
        else:
            continue



def convert_to_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def detect_color(frame, hsv_frame, lower_bound, upper_bound):
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result, mask

def draw_bounding_box(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

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
