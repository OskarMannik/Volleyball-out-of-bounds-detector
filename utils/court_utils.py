import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent


# Global variables for click events
points = []
window_name = 'Select Court Corners'

window_closed = False

def click_event(event, x, y, flags, params):
    global points, img, window_closed
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # Red circle for marking
        points.append((x, y))
        cv2.imshow(window_name, img)
        if len(points) >= 4:
            window_closed = True  # Set the flag to close the window

def select_court_corners(frame):
    global img, points, window_closed
    points = []  # Reset points to avoid retaining any previous data
    window_closed = False  # Reset the flag
    img = frame.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    cv2.imshow(window_name, img)

    # Wait until the window should be closed
    while not window_closed:
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
            break

    cv2.destroyAllWindows()
    return points


def draw_court_outline(frame, court_corners):
    if len(court_corners) != 4:
        raise ValueError("Four corners must be provided to draw the court outline")
    # Ensure the points are connected in a rectangular pattern
    pts = np.array([court_corners[0], court_corners[1], court_corners[3], court_corners[2]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=5)
    return frame
