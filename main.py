import cv2
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Define the HSV ranges for different colors (global definition)
colors = {
    'blue': (np.array([100, 150, 150]), np.array([130, 255, 255])),
    'light_blue': (np.array([90, 50, 70]), np.array([130, 255, 255])),  # Covers lighter to regular blue
    'deep_blue': (np.array([110, 150, 50]), np.array([130, 255, 255])),  # Covers deeper shades of blue
    'red1': (np.array([0, 160, 160]), np.array([10, 255, 255])),    # Lower boundary for red
    'red2': (np.array([170, 160, 160]), np.array([180, 255, 255])),  # Upper boundary for red
    'light_green': (np.array([35, 50, 50]), np.array([85, 255, 255])),  # Covers lighter to regular green
    'deep_green': (np.array([35, 100, 50]), np.array([85, 255, 150])),  # Covers deeper shades of green
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

            # Detect red
            _, mask1 = detect_color(frame, hsv_frame, colors['red1'][0], colors['red1'][1])
            _, mask2 = detect_color(frame, hsv_frame, colors['red2'][0], colors['red2'][1])
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply morphological operations
            red_mask = cv2.erode(red_mask, None, iterations=2)
            red_mask = cv2.dilate(red_mask, None, iterations=2)

            draw_bounding_box(frame, red_mask)
            cv2.putText(frame, 'red', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Detect light blue
            _, light_blue_mask = detect_color(frame, hsv_frame, colors['light_blue'][0], colors['light_blue'][1])
            light_blue_mask = cv2.erode(light_blue_mask, None, iterations=2)
            light_blue_mask = cv2.dilate(light_blue_mask, None, iterations=2)
            
            # Detect deep blue
            _, deep_blue_mask = detect_color(frame, hsv_frame, colors['deep_blue'][0], colors['deep_blue'][1])
            deep_blue_mask = cv2.erode(deep_blue_mask, None, iterations=2)
            deep_blue_mask = cv2.dilate(deep_blue_mask, None, iterations=2)

            # Combine masks for all shades of blue
            blue_mask = cv2.bitwise_or(light_blue_mask, deep_blue_mask)
            draw_bounding_box(frame, blue_mask)
            cv2.putText(frame, 'blue', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            
            # Detect light green
            _, light_green_mask = detect_color(frame, hsv_frame, colors['light_green'][0], colors['light_green'][1])
            light_green_mask = cv2.erode(light_green_mask, None, iterations=2)
            light_green_mask = cv2.dilate(light_green_mask, None, iterations=2)
            
            # Detect deep green
            _, deep_green_mask = detect_color(frame, hsv_frame, colors['deep_green'][0], colors['deep_green'][1])
            deep_green_mask = cv2.erode(deep_green_mask, None, iterations=2)
            deep_green_mask = cv2.dilate(deep_green_mask, None, iterations=2)

            # Combine masks for all shades of green
            green_mask = cv2.bitwise_or(light_green_mask, deep_green_mask)
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
