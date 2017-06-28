import cv2

cam = cv2.VideoCapture(0)

faceCascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(faceCascadePath)

while(1):
    (grabbed, frame) = cam.read()
    
    if not grabbed:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faceRects = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    faceRects = faceCascade.detectMultiScale(gray)

    if len(faceRects):
        for(x,y,w,h) in faceRects:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('face', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
