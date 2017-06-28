import cv2
import numpy as np

cam = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setHistory(1000)

while True:
    (grabbed, frame) = cam.read()

    if grabbed:
        fgmask = fgbg.apply(frame)

        cv2.imshow('img', fgmask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
