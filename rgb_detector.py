import numpy as np
import cv2

# Capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, image_frame = webcam.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)

    # Red color range
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

    # Green color range
    green_lower = np.array([40, 70, 70], np.uint8)
    green_upper = np.array([80, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    # Blue color range
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)