import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Read the image
image = cv2.imread('image1.jpg')
image = imutils.resize(image, width=min(400, image.shape[1]))

# Detect people in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

# Apply non-maxima suppression to the bounding boxes
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# Draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

# Add direction detection feature
directions = []
for i in range(len(pick)):
    if i > 0:
        # Compare the current position with the previous position
        if pick[i][0] > pick[i-1][0]:
            directions.append('Right')
        elif pick[i][0] < pick[i-1][0]:
            directions.append('Left')
        else:
            directions.append('Stationary')
    else:
        directions.append('N/A')  # No previous data for the first detection

# Annotate directions on the image
for (rect, direction) in zip(pick, directions):
    (xA, yA, xB, yB) = rect
    cv2.putText(image, direction, (xA, yB + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output image
cv2.imshow("Pedestrian Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
