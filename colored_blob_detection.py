import cv2
import numpy as np

cam = cv2.VideoCapture(0)

#yellowLower = (50, 50, 50)
#yellowUpper = (70, 255, 255)

yellowLower = (15, 100, 100)
yellowUpper = (29, 255, 255)

kernel = np.ones((3,3),np.uint8)

while True:
    (grabbed, frame) = cam.read()

    if not grabbed:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    
    #remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #find contours
    (_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 1000:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("masked", mask)
    cv2.imshow("img", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
