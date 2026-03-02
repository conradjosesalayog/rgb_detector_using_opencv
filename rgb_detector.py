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
    kernel = np.ones((5, 5), np.uint8)

    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # Detect Red
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image_frame, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Detect Green
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_frame, "Green Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Detect Blue
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image_frame, "Blue Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    cv2.imshow("Multiple Color Detection in Real-Time", image_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()