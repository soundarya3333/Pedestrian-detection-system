import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)


def Detector(frame):
    frame = imutils.resize(frame, width=700)
    # Using Sliding window concept
    rects, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    c = 1

    for x, y, w, h in pick:
        # Draw the pedestrian bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Estimate distance based on bounding box size
        distance = 5000 / (w * h) ** 0.5  # Simple distance estimation
        distance_text = f"{distance:.2f} cm" if distance < 100 else f"{distance / 100:.2f} m"

        # Put the distance text
        cv2.putText(frame, distance_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Label the pedestrian
        cv2.putText(frame, f'P{c}', (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        c += 1

    # Display the total number of pedestrians detected
    cv2.putText(frame, f'Total Persons : {c - 1}', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = Detector(frame)
    cv2.imshow('output', output)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
