import cv2
import pickle

with open("Model/parking_spaces.pkl", "rb") as f:
    rectangles = pickle.load(f)

image_path = "../Data/Parking_Sample.mp4"
img = cv2.imread(image_path)

occupied_count = 0
empty_count = 0

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for rect in rectangles:
    roi = gray[rect[1]:rect[3], rect[0]:rect[2]]
    _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(thresh) > 1000:
        color = (0, 0, 255)
        occupied_count += 1
    else:
        color = (0, 255, 0)
        empty_count += 1
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)

text = f"Occupied: {occupied_count}  Empty: {empty_count}"
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow("Parking Lot Monitoring", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
