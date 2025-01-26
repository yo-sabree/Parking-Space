import cv2
import pickle
import numpy as np


IMAGE_PATH = "../Data/Parking_Sample.png"
OUTPUT_PATH = "../Model/parking_spaces_poly.rb"

all_points = []
current_points = []


def draw_polylines(event, x, y, flags, param):
    global all_points, current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Mark Parking Spaces", img)

        if len(current_points) == 4:
            pts = np.array(current_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow("Mark Parking Spaces", img)
            all_points.append(current_points)
            current_points = []


img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (1000, 700))

cv2.imshow("Mark Parking Spaces", img)
cv2.setMouseCallback("Mark Parking Spaces", draw_polylines)
cv2.waitKey(0)
cv2.destroyAllWindows()

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(all_points, f)

print(f"Saved {len(all_points)} parking spaces.")
