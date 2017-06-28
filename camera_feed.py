#Camera feed
import cv2

cam = cv2.VideoCapture(0)

while(1):
    (grabbed, frame) = cam.read()
    frame = cv2.GaussianBlur(frame, (21,21), 0)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #print type(frame)
    cv2.imshow('img', frame)

    #cv2.waitKey()

    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
