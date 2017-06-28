#Basic motion detection
import cv2
import numpy as np

def resize(image, width):
    #maintain the aspect ratio even after resizing
    ratio = width/image.shape[1]
    height = int(image.shape[0]*ratio)
    dim = (int(width), height)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

cam = cv2.VideoCapture(0)

kernel = np.ones((5,5), np.uint8)

firstFrame = None
min_area = 1000
min_thresh = 75

for i in range(50):
    (grabbed, frame) = cam.read()

while(1):
    (grabbed, frame) = cam.read()
    
    width = 500.0
    #frame = resize(frame, width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #blur the image to remove high intensity points due to camera issues
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    #take absolute difference of intial frame and current frame
    diffFrame = cv2.absdiff(firstFrame, gray)
    cv2.imshow('difference', diffFrame)
    _, thresh = cv2.threshold(diffFrame, min_thresh, 255, cv2.THRESH_BINARY)
    #remove noise, play around with parameters like kernel etc to get better output
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow('img', thresh)
    #find contours
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("im", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
