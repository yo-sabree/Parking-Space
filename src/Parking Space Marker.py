import cv2
import pickle
import numpy as np

image_path = "../Data/Parking_Sample.png"
img = cv2.imread(image_path)
img = cv2.resize(img, (1000, 700))

points = []

def draw_rectangle(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.imshow("Mark Parking Spaces", img)
        if len(points) == 4:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("Mark Parking Spaces", img)
            points.clear()

cv2.imshow("Mark Parking Spaces", img)
cv2.setMouseCallback("Mark Parking Spaces", draw_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

with open("Model/parking_spaces.pkl", "wb") as f:
    pickle.dump(points, f)
