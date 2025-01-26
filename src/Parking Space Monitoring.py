import cv2
import pickle
import numpy as np

with open("../Model/parking_spaces_poly.rb", "rb") as f:
    parking_spaces = pickle.load(f)

video = cv2.VideoCapture("../Data/Parking_Sample.mp4")

empty_count = 0
occupied_count = 0

def parking(img, frame):
    global empty_count, occupied_count
    empty_count = 0
    occupied_count = 0

    for polyline in parking_spaces:
        if len(polyline) > 2:
            pts = np.array(polyline, np.int32).reshape((-1, 1, 2))
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

            mask_area = cv2.bitwise_and(img, img, mask=mask)
            count = cv2.countNonZero(mask_area)
            total_area = cv2.countNonZero(mask)

            if total_area == 0:
                continue

            status = "Empty" if count < (total_area * 0.5) else "Occupied"
            color = (0, 255, 0) if status == "Empty" else (0, 0, 255)

            if status == "Empty":
                empty_count += 1
            else:
                occupied_count += 1

            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Empty: {empty_count}", (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Occupied: {occupied_count}", (10, 60), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

while True:
    status, frame = video.read()
    if status:
        frame = cv2.resize(frame, (1000, 700))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        thres = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        median = cv2.medianBlur(thres, 3)
        kernal = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(median, kernal, iterations=2)

        parking(dil, frame)

        cv2.imshow("Parking Space Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
